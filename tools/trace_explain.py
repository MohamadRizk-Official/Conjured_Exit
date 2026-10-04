"""Read results/flight_trace.csv (written by mission.FlightTrace) and explain the LAST flight in plain words.

    python tools/trace_explain.py                      # last flight in results/flight_trace.csv
    python tools/trace_explain.py --all                # every flight in the file
    python tools/trace_explain.py --in other.csv

Each flight = a run of rows whose state is not 'idle' (plus the tail the trace writes afterwards).
The verdict is rule-based and deliberately blunt; the numbers are printed next to it.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def split_flights(rows: list[dict], gap_s: float = 3.0) -> list[list[dict]]:
    """Group rows into flights: a new flight starts when the time jumps by more than gap_s, or when a
    'takeoff' row follows an idle/estop row (back-to-back flights in one trace)."""
    flights: list[list[dict]] = []
    cur: list[dict] = []
    last_t = None
    last_state = None
    for r in rows:
        t = _f(r.get("t"))
        if t is None:
            continue
        state = r.get("state", "")
        new_takeoff = state == "takeoff" and last_state in ("idle", "estop")
        if cur and ((last_t is not None and t - last_t > gap_s) or new_takeoff):
            flights.append(cur)
            cur = []
        cur.append(r)
        last_t, last_state = t, state
    if cur:
        flights.append(cur)
    return flights


def explain(flight: list[dict]) -> dict:
    t0 = _f(flight[0]["t"])
    states = [r.get("state", "") for r in flight]
    seq: list[tuple[float, str]] = []
    for r in flight:
        st = r.get("state", "")
        if not seq or seq[-1][1] != st:
            seq.append((round(_f(r["t"]) - t0, 2), st))
    cam_rows = [r for r in flight if r.get("cam_ok") in ("1", 1) and _f(r.get("cam_z")) is not None]
    takeoff_rows = [r for r in flight if r.get("state") == "takeoff"]
    res: dict = {"duration_s": round(_f(flight[-1]["t"]) - t0, 2), "states": seq,
                 "rows": len(flight), "camera_rows": len(cam_rows)}
    if not cam_rows:
        res["verdict"] = "the camera never had a fix during this flight: the marker was not visible"
        return res
    start = cam_rows[0]
    sx, sy, sz = _f(start["cam_x"]), _f(start["cam_y"]), _f(start["cam_z"])
    zs = [_f(r["cam_z"]) for r in cam_rows]
    xy_drift = [math.hypot(_f(r["cam_x"]) - sx, _f(r["cam_y"]) - sy) for r in cam_rows]
    i_zmax = max(range(len(zs)), key=lambda i: zs[i])
    res["max_height_gain_m"] = round(zs[i_zmax] - sz, 2)
    res["t_max_height_s"] = round(_f(cam_rows[i_zmax]["t"]) - t0, 2)
    res["max_xy_drift_m"] = round(max(xy_drift), 2)
    # first loss of camera fix during takeoff/hover/flying
    lost_t = None
    for r in flight:
        if r.get("state") in ("takeoff", "hover", "flying") and r.get("cam_ok") not in ("1", 1):
            lost_t = round(_f(r["t"]) - t0, 2)
            break
    res["camera_lost_at_s"] = lost_t
    bats = [_f(r.get("bat_v")) for r in flight]
    bats = [b for b in bats if b is not None]
    res["battery_v"] = {"start": bats[0], "min": min(bats)} if bats else None
    est_err = []
    for r in cam_rows:
        ex, ey, ez = _f(r.get("est_x")), _f(r.get("est_y")), _f(r.get("est_z"))
        if None not in (ex, ey, ez):
            est_err.append(math.hypot(_f(r["cam_x"]) - ex, _f(r["cam_y"]) - ey, _f(r["cam_z"]) - ez))
    res["estimate_vs_camera_max_m"] = round(max(est_err), 2) if est_err else None
    yaws = [_f(r.get("cam_yaw_deg")) for r in cam_rows]
    yaws = [y for y in yaws if y is not None]
    if len(yaws) >= 2:
        d = (yaws[-1] - yaws[0] + 180) % 360 - 180
        res["marker_yaw_change_deg"] = round(d, 0)

    # ---- verdict rules, most specific first
    gain, drift = res["max_height_gain_m"], res["max_xy_drift_m"]
    ended = states[-1] if states else ""
    reached_hover = any(s == "hover" for s in states)
    sag = (res["battery_v"]["start"] - res["battery_v"]["min"]) if res["battery_v"] else None
    stopped = "estop" in states and lost_t is None
    if stopped and gain >= 0.4:
        v = (f"it flew to {gain:.2f} m, then the motors were cut by STOP or a safety rule at "
             f"{seq[[s for _, s in seq].index('estop')][0]:.1f} s (drift {drift:.2f} m)")
    elif res["estimate_vs_camera_max_m"] is not None and res["estimate_vs_camera_max_m"] > 0.5 and gain < 0.3:
        v = (f"the drone's own position estimate disagreed with the camera by up to "
             f"{res['estimate_vs_camera_max_m']:.1f} m, so it was steering toward the wrong place: "
             f"camera positions were not reaching it, or the estimator was not reset")
    elif reached_hover and ended in ("idle",) and gain >= 0.4:
        v = f"it flew: climbed {gain:.2f} m, drifted at most {drift:.2f} m sideways, landed normally"
    elif gain < 0.15 and sag is not None and sag > 0.6:
        v = (f"it never lifted (max {gain:.2f} m) while the battery sagged {sag:.2f} V under load: "
             f"not enough thrust, change the battery")
    elif drift >= 0.3 and gain < 0.3:
        v = (f"it slid {drift:.2f} m sideways while only {gain:.2f} m up: the drone's heading did not match "
             f"the camera frame, so its corrections pushed sideways")
    elif lost_t is not None and gain < 0.25:
        v = (f"the camera lost the marker {lost_t:.1f} s in at only {gain:.2f} m of height: the marker tilted, "
             f"blurred or left the view (check it is flat and the drone is not yanked by a cable); motors were cut")
    elif lost_t is not None:
        v = f"the camera lost the marker {lost_t:.1f} s in at {gain:.2f} m up; the safety routine took over"
    elif gain < 0.15:
        v = f"it barely lifted (max {gain:.2f} m) and the guard cut the motors; thrust or weight problem"
    else:
        v = f"climbed {gain:.2f} m, drifted {drift:.2f} m, ended in state '{ended}'"
    res["verdict"] = v
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="results/flight_trace.csv")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    path = Path(args.inp)
    if not path.exists():
        print(f"no trace at {path}")
        return 1
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    flights = split_flights(rows)
    if not flights:
        print("trace is empty")
        return 1
    for fl in (flights if args.all else flights[-1:]):
        r = explain(fl)
        print(f"flight of {r['duration_s']} s, states {r['states']}")
        for k in ("max_height_gain_m", "t_max_height_s", "max_xy_drift_m", "camera_lost_at_s", "battery_v",
                  "estimate_vs_camera_max_m", "marker_yaw_change_deg"):
            if k in r:
                print(f"  {k}: {r[k]}")
        print(f"  VERDICT: {r['verdict']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
