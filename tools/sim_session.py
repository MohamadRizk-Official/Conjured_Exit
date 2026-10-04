"""Scripted end-to-end session against a running mission server (normally ``mission.py --sim``).

Drives the same HTTP API the dashboard uses and samples telemetry at 10 Hz, then prints a report:

    A  hover test        arm -> hover; time to reach height, held height, landing, back to idle
    B  teach a route     record_start (the sim hand carries the drone) -> record_stop -> save_as
    C  cast the route    arm -> cast; time in each state, end position vs the saved path's end
    D  alarm + stop      arm -> alarm; STOP mid-flight; estop latency; clear_alarm back to idle

Tracking loss (blind descent) is covered by tests/test_flight.py; the sim tracker never loses the drone.

    python tools/sim_session.py --base http://127.0.0.1:8765 --out results/sim_session.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))

    def state(self) -> dict:
        return self._get("/api/state")

    def telemetry(self) -> dict:
        return self._get("/api/telemetry")

    def cmd(self, cmd_name: str, **args) -> dict:
        body = json.dumps({"name": cmd_name, "args": args}).encode("utf-8")
        req = urllib.request.Request(self.base + "/api/command", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))


class Session:
    def __init__(self, api: Api, hz: float = 10.0) -> None:
        self.api = api
        self.dt = 1.0 / hz
        self.t0 = time.perf_counter()
        self.samples: list[dict] = []
        self.events: list[tuple[float, str]] = []

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, text: str) -> None:
        self.events.append((self.now(), text))
        print(f"[{self.now():7.2f}] {text}", flush=True)

    def cmd(self, cmd_name: str, **args) -> None:
        self.mark(f"cmd {cmd_name} {args if args else ''}")
        self.api.cmd(cmd_name, **args)

    def sample(self) -> dict:
        tel = self.api.telemetry()
        tel["t"] = self.now()
        self.samples.append(tel)
        return tel

    def wait_for(self, pred, timeout: float, what: str) -> bool:
        t_end = self.now() + timeout
        while self.now() < t_end:
            tel = self.sample()
            if pred(tel):
                self.mark(f"reached: {what}")
                return True
            time.sleep(self.dt)
        self.mark(f"TIMEOUT waiting for {what} ({timeout:.0f} s)")
        return False

    def wait_states(self, sequence: list[str], timeout: float) -> dict[str, float]:
        """Wait for the flight state to pass through ``sequence`` in order; returns when each was first seen."""
        seen: dict[str, float] = {}
        idx = 0
        t_end = self.now() + timeout
        while self.now() < t_end and idx < len(sequence):
            tel = self.sample()
            st = tel.get("flight_state")
            if st == sequence[idx]:
                seen[st] = tel["t"]
                z = tel["position"][2] if tel.get("position") else float("nan")
                self.mark(f"flight state -> {st}  z={z:.2f}")
                idx += 1
            time.sleep(self.dt)
        if idx < len(sequence):
            self.mark(f"TIMEOUT: never saw {sequence[idx]} (saw {list(seen)})")
        return seen

    def z_between(self, t_a: float, t_b: float) -> list[float]:
        return [s["position"][2] for s in self.samples if t_a <= s["t"] <= t_b and s.get("position")]

    def xy_at_end(self):
        p = self.samples[-1].get("position") if self.samples else None
        return (p[0], p[1]) if p else None


def scenario_hover(s: Session, hold_s: float) -> dict:
    s.mark("=== A. hover test ===")
    s.cmd("clear_alarm")
    s.cmd("arm", on=True)
    time.sleep(0.3)
    s.cmd("hover")
    seen = s.wait_states(["takeoff", "hover", "landing", "idle"], timeout=hold_s + 30)
    res: dict = {"states": seen}
    if "hover" in seen and "landing" in seen:
        zs = s.z_between(seen["hover"] + 0.5, seen["landing"] - 0.2)
        res["held_z_mean"] = sum(zs) / len(zs) if zs else None
        res["held_z_min"] = min(zs) if zs else None
        res["held_z_max"] = max(zs) if zs else None
        res["hold_s"] = seen["landing"] - seen["hover"]
    if "takeoff" in seen and "hover" in seen:
        res["climb_s"] = seen["hover"] - seen["takeoff"]
    if "landing" in seen and "idle" in seen:
        res["landing_s"] = seen["idle"] - seen["landing"]
    res["final_z"] = s.samples[-1]["position"][2] if s.samples else None
    return res


def scenario_teach(s: Session, name: str, carry_name: str = "exit_a") -> dict:
    s.mark("=== B. teach a route (sim hand carries the drone) ===")
    # The sim hand walks the demo path `carry_name` once; record for its duration plus a margin.
    dur = None
    for p in s.api.state().get("paths", []):
        if p.get("name") == carry_name:
            dur = p.get("duration") or p.get("duration_s") or p.get("seconds")
    record_s = float(dur) + 1.5 if dur else 20.0
    s.mark(f"recording for {record_s:.1f} s (carry path {carry_name}, duration {dur})")
    s.cmd("record_start")
    t_end = s.now() + record_s
    while s.now() < t_end:
        s.sample()
        time.sleep(0.1)
    st = s.api.state()
    s.mark(f"recording samples so far: {(st.get('recording') or {}).get('n_samples')}")
    s.cmd("record_stop")
    time.sleep(0.5)
    s.cmd("save_as", name=name)
    time.sleep(0.8)
    st = s.api.state()
    names = [p.get("name") for p in st.get("paths", [])]
    rec = st.get("recording") or {}
    s.mark(f"saved paths: {names}; recording samples {rec.get('n_samples')}")
    return {"n_samples": rec.get("n_samples"), "saved": name in names, "paths": names}


def scenario_cast(s: Session, name: str, paths_dir: str) -> dict:
    s.mark("=== C. cast the saved route ===")
    s.cmd("clear_alarm")
    s.cmd("arm", on=True)
    time.sleep(0.3)
    s.cmd("cast", name=name)
    seen = s.wait_states(["takeoff", "hover", "flying", "landing", "idle"], timeout=120)
    res: dict = {"states": seen}
    end_xy = s.xy_at_end()
    try:
        data = json.loads(Path(paths_dir, f"{name}.json").read_text(encoding="utf-8"))
        pts = data.get("points") or data.get("path") or []
        last = pts[-1] if pts else None
        if last and end_xy:
            dx, dy = end_xy[0] - float(last[0]), end_xy[1] - float(last[1])
            res["end_error_m"] = (dx * dx + dy * dy) ** 0.5
            res["path_points"] = len(pts)
    except Exception as exc:  # noqa: BLE001
        res["path_file_note"] = f"could not read saved path: {exc!r}"
    if "flying" in seen and "landing" in seen:
        res["flying_s"] = seen["landing"] - seen["flying"]
    return res


def scenario_alarm_stop(s: Session) -> dict:
    s.mark("=== D. alarm, then STOP mid-flight ===")
    s.cmd("clear_alarm")
    s.cmd("arm", on=True)
    time.sleep(0.3)
    s.cmd("alarm")
    seen = s.wait_states(["takeoff", "hover", "flying"], timeout=40)
    res: dict = {"states_before_stop": seen}
    if "flying" in seen:
        time.sleep(1.5)
        z_before = s.sample()["position"][2]
        t_stop = s.now()
        s.cmd("stop")
        ok = s.wait_for(lambda t: t.get("flight_state") == "estop", 5, "flight state estop")
        res["estop_latency_s"] = (s.now() - t_stop) if ok else None
        res["z_at_stop"] = z_before
        res["armed_after_stop"] = s.sample().get("armed")
        s.cmd("clear_alarm")
        ok2 = s.wait_for(lambda t: t.get("flight_state") == "idle", 5, "flight state idle after clear")
        res["cleared_to_idle"] = ok2
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="results/sim_session.json")
    ap.add_argument("--path-name", default="sim_exit_a")
    ap.add_argument("--hold-s", type=float, default=8.0)
    ap.add_argument("--paths-dir", default=None)
    ap.add_argument("--scenarios", default="A,B,C,D")
    args = ap.parse_args()

    import config
    paths_dir = args.paths_dir or config.PATHS_DIR
    api = Api(args.base)
    st = api.state()
    print(f"server: link connected={st.get('link', {}).get('connected')}  ui={st.get('ui')}  "
          f"paths={[p.get('name') for p in st.get('paths', [])]}")
    s = Session(api)
    report: dict = {"base": args.base, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    want = set(x.strip().upper() for x in args.scenarios.split(","))
    if "A" in want:
        report["A_hover"] = scenario_hover(s, args.hold_s)
    if "B" in want:
        report["B_teach"] = scenario_teach(s, args.path_name)
    if "C" in want:
        report["C_cast"] = scenario_cast(s, args.path_name, paths_dir)
    if "D" in want:
        report["D_alarm_stop"] = scenario_alarm_stop(s)
    report["events"] = s.events
    report["log_tail"] = api.state().get("log", [])[-40:]
    report["n_samples"] = len(s.samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report, "samples": s.samples}, indent=1), encoding="utf-8")

    print()
    print("REPORT")
    for key in ("A_hover", "B_teach", "C_cast", "D_alarm_stop"):
        if key in report:
            print(f"  {key}:")
            for k, v in report[key].items():
                if isinstance(v, float):
                    v = round(v, 3)
                if isinstance(v, dict):
                    v = {kk: round(vv, 2) for kk, vv in v.items()}
                print(f"      {k}: {v}")
    print(f"  samples: {len(s.samples)}  saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
