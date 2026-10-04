"""Hand-carry axis check against a running mission server (motors off).

Records /api/telemetry at 10 Hz while the drone is carried by hand, then reports per axis what the
camera saw and whether the drone's own estimate followed. Expected sequence (about 0.5 m each,
back to the start between moves):

    1. +x : away from the camera, along the floor sheet's top edge
    2. +y : to your LEFT while you face away from the camera
    3. +z : straight up, hold, back down

    python tools/carry_check.py record --seconds 75 --out results/carry_trace.csv
    python tools/carry_check.py analyze --in results/carry_trace.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

FIELDS = ["t", "tracking_ok", "cam_x", "cam_y", "cam_z", "cam_yaw_deg", "est_x", "est_y", "est_z", "cam_vs_est_m", "bat_v"]


def record(base: str, seconds: float, out: Path, hz: float = 10.0) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = 0
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        while time.perf_counter() - t0 < seconds:
            try:
                with urllib.request.urlopen(base.rstrip("/") + "/api/telemetry", timeout=2) as r:
                    d = json.loads(r.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                print("telemetry read failed:", exc, file=sys.stderr)
                time.sleep(1.0 / hz)
                continue
            cam = d.get("camera_xyz") or [None, None, None]
            est = d.get("position") or [None, None, None]
            w.writerow([f"{time.perf_counter() - t0:.2f}", 1 if d.get("tracking_ok") else 0,
                        *[("" if v is None else f"{v:.3f}") for v in cam],
                        "" if d.get("camera_yaw_deg") is None else f"{d['camera_yaw_deg']:.1f}",
                        *[("" if v is None else f"{v:.3f}") for v in est],
                        "" if d.get("camera_vs_estimate_m") is None else f"{d['camera_vs_estimate_m']:.3f}",
                        "" if d.get("battery_v") is None else f"{d['battery_v']:.2f}"])
            fh.flush()
            n += 1
            time.sleep(1.0 / hz)
    print(f"recorded {n} samples over {seconds:.0f} s to {out}")
    return 0


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze_rows(rows: list[dict]) -> dict:
    """Per-axis excursion from the first second's mean, estimate error, tracking gaps, axis order."""
    good = [r for r in rows if r.get("tracking_ok") in ("1", 1, True) and _f(r.get("cam_x")) is not None]
    res: dict = {"samples": len(rows), "tracked": len(good)}
    if len(good) < 5:
        res["verdict"] = "too few tracked samples: the camera did not see the drone marker"
        return res
    t = [_f(r["t"]) for r in good]
    base_n = max(3, sum(1 for x in t if x - t[0] <= 1.0))
    axes = {}
    for ax in ("x", "y", "z"):
        vals = [_f(r[f"cam_{ax}"]) for r in good]
        base = sum(vals[:base_n]) / base_n
        dev = [v - base for v in vals]
        i_max = max(range(len(dev)), key=lambda i: dev[i])
        i_min = min(range(len(dev)), key=lambda i: dev[i])
        axes[ax] = {"start": round(base, 3), "max_plus": round(dev[i_max], 3), "t_plus": round(t[i_max], 1),
                    "max_minus": round(dev[i_min], 3), "t_minus": round(t[i_min], 1)}
    res["axes"] = axes
    errs = [_f(r.get("cam_vs_est_m")) for r in good]
    errs = [e for e in errs if e is not None]
    res["estimate_error_m"] = {"median": round(sorted(errs)[len(errs) // 2], 3) if errs else None,
                               "max": round(max(errs), 3) if errs else None}
    # tracking gaps longer than 0.3 s while samples kept coming
    gaps = []
    tt = [_f(r["t"]) for r in rows]
    oks = [r.get("tracking_ok") in ("1", 1, True) for r in rows]
    start = None
    for i, ok in enumerate(oks):
        if not ok and start is None:
            start = tt[i]
        if ok and start is not None:
            if tt[i] - start >= 0.3:
                gaps.append((round(start, 1), round(tt[i] - start, 1)))
            start = None
    res["tracking_gaps"] = gaps
    order = sorted(("x", "y", "z"), key=lambda a: axes[a]["t_plus"])
    res["axis_order_seen"] = order
    notes = []
    for ax, label in (("x", "+x (away from camera)"), ("y", "+y (your left)"), ("z", "+z (up)")):
        a = axes[ax]
        if a["max_plus"] >= 0.25:
            notes.append(f"{label}: camera saw +{a['max_plus']:.2f} m at t={a['t_plus']:.0f} s (good)")
        elif a["max_minus"] <= -0.25:
            notes.append(f"{label}: camera saw {a['max_minus']:.2f} m at t={a['t_minus']:.0f} s: the axis is REVERSED")
        else:
            notes.append(f"{label}: no clear move seen (max +{a['max_plus']:.2f} / {a['max_minus']:.2f} m)")
    if res["estimate_error_m"]["max"] is not None:
        if res["estimate_error_m"]["max"] > 0.3:
            notes.append(f"drone estimate lagged the camera by up to {res['estimate_error_m']['max']:.2f} m: "
                         f"extpos not reaching the drone or estimator not reset")
        else:
            notes.append(f"drone estimate followed the camera (median error {res['estimate_error_m']['median']:.2f} m, "
                         f"max {res['estimate_error_m']['max']:.2f} m)")
    if gaps:
        notes.append(f"tracking dropped {len(gaps)} time(s) for >= 0.3 s at t={[g[0] for g in gaps]} s")
    if order != ["x", "y", "z"]:
        notes.append(f"axes peaked in the order {order}, expected x, y, z: check which way you moved")
    res["notes"] = notes
    return res


def analyze(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    res = analyze_rows(rows)
    print(json.dumps({k: v for k, v in res.items() if k != "notes"}, indent=1))
    print()
    for n in res.get("notes", [res.get("verdict", "")]):
        print("-", n)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--base", default="http://127.0.0.1:8765")
    r.add_argument("--seconds", type=float, default=75.0)
    r.add_argument("--out", default="results/carry_trace.csv")
    a = sub.add_parser("analyze")
    a.add_argument("--in", dest="inp", default="results/carry_trace.csv")
    args = ap.parse_args()
    if args.cmd == "record":
        return record(args.base, args.seconds, Path(args.out))
    return analyze(Path(args.inp))


if __name__ == "__main__":
    sys.exit(main())
