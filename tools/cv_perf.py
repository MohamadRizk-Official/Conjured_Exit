#!/usr/bin/env python
"""cv_perf.py -- OpenCV throughput benchmark for the pathcaster tracker hot path.

Synthetic 640x480 BGR frame -> per camera: cvtColor(BGR2HSV), then per marker
colour: inRange -> findContours -> moments (centroid).  Then undistortPoints on
the centroids and triangulatePoints across the two cameras.

Reports ms per op and an estimated ms per 2-camera tracking cycle.
Budget (README.md "Tracker", 30 Hz): < 15 ms per cycle.

Run from the project root inside the cf64 venv (x64 Python; on the ARM64 laptop
this runs under Prism emulation, which is exactly what we are measuring):

    .\\cf64\\Scripts\\python.exe tools\\cv_perf.py
    .\\cf64\\Scripts\\python.exe tools\\cv_perf.py --iters 1000 --json results\\cv_perf.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import struct
import sys
import time

import cv2
import numpy as np

BUDGET_MS = 15.0
N_CAMS = 2

# HSV (lower, upper) per sticky-note colour.  The synthetic frame paints pure
# BGR blobs, so these are centred on the pure hues (red=0, green=60, blue=120).
COLOR_RANGES = {
    "red":   ((0, 120, 120), (10, 255, 255)),     # nose
    "blue":  ((110, 120, 120), (130, 255, 255)),  # tail
    "green": ((50, 120, 120), (70, 255, 255)),    # wand tip
}


def env_report() -> dict:
    x64_build = "AMD64" in sys.version
    # x64 process on an ARM64 host: PROCESSOR_ARCHITEW6432=ARM64 (same trick as WOW64).
    # platform.machine() also says ARM64 for an emulated process on Python 3.12 (WMI query).
    host = os.environ.get("PROCESSOR_ARCHITEW6432") or platform.machine()
    return {
        "python": sys.version.split()[0],
        "python_build": "x64" if x64_build else platform.machine(),
        "pointer_bits": struct.calcsize("P") * 8,
        "host_arch": host,
        "emulated": bool(x64_build and host.upper().startswith("ARM")),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cv_threads": cv2.getNumThreads(),
        "cv_optimized": cv2.useOptimized(),
        "cpu_count": os.cpu_count(),
    }


def synth_frame(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Dim noisy room + 3 saturated marker blobs + a sprinkle of saturated noise
    pixels so each mask yields a few distractor contours like a real frame does."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(40, 110, size=(h, w, 3), dtype=np.uint8)
    noise = rng.random((h, w)) < 0.0015
    frame[noise] = rng.integers(0, 256, size=(int(noise.sum()), 3), dtype=np.uint8)
    cv2.circle(frame, (int(w * 0.30), int(h * 0.50)), 12, (0, 0, 255), -1)   # red  (nose)
    cv2.circle(frame, (int(w * 0.62), int(h * 0.42)), 12, (255, 0, 0), -1)   # blue (tail)
    cv2.circle(frame, (int(w * 0.80), int(h * 0.70)), 10, (0, 255, 0), -1)   # green (wand)
    return frame


def centroid(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def camera_pipeline(frame, ranges, kernel=None):
    """What the tracker does per camera per frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = {}
    for name, (lo, hi) in ranges.items():
        mask = cv2.inRange(hsv, lo, hi)
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        out[name] = centroid(mask)
    return out


def make_geometry(w, h, n, seed=1):
    """Two synthetic cameras ~90 deg apart plus n random 3-D points visible to both."""
    rng = np.random.default_rng(seed)
    K = np.array([[600.0, 0.0, w / 2.0], [0.0, 600.0, h / 2.0], [0.0, 0.0, 1.0]])
    dist = np.array([-0.12, 0.04, 0.0005, -0.0003, 0.0])
    R1, t1 = np.eye(3), np.zeros((3, 1))
    R2 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])   # cam2 looks along -x
    t2 = np.array([[-2.5], [0.0], [2.5]])                                # cam2 centre at (2.5, 0, 2.5)
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    X = np.column_stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n), rng.uniform(2.0, 3.0, n)])
    pix1, _ = cv2.projectPoints(X, cv2.Rodrigues(R1)[0], t1, K, dist)
    pix2, _ = cv2.projectPoints(X, cv2.Rodrigues(R2)[0], t2, K, dist)
    return K, dist, P1, P2, X, pix1.reshape(-1, 1, 2), pix2.reshape(-1, 1, 2)


def triangulate(P1, P2, u1, u2):
    a = np.ascontiguousarray(u1.reshape(-1, 2).T)
    b = np.ascontiguousarray(u2.reshape(-1, 2).T)
    Xh = cv2.triangulatePoints(P1, P2, a, b)
    return (Xh[:3] / Xh[3]).T


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"mean": statistics.fmean(ts), "median": ts[len(ts) // 2],
            "p95": ts[int(0.95 * (len(ts) - 1))], "max": ts[-1], "n": len(ts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--colors", type=int, default=3, choices=[1, 2, 3],
                    help="marker colours per camera (nose, tail, wand) = inRange passes per frame")
    ap.add_argument("--points", type=int, default=10, help="points for undistort/triangulate timing")
    ap.add_argument("--threads", type=int, help="cv2.setNumThreads() before measuring")
    ap.add_argument("--json", help="also write results to this JSON file")
    args = ap.parse_args()
    if args.threads is not None:
        cv2.setNumThreads(args.threads)

    env = env_report()
    print("== environment")
    for k, v in env.items():
        print(f"  {k:14s} {v}")

    w, h = args.width, args.height
    frame = synth_frame(w, h)
    ranges = {k: (np.array(lo, np.uint8), np.array(hi, np.uint8))
              for k, (lo, hi) in list(COLOR_RANGES.items())[: args.colors]}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = next(iter(ranges.values()))
    mask = cv2.inRange(hsv, lo, hi)

    # sanity: the pipeline must actually find the blobs
    found = camera_pipeline(frame, ranges)
    missing = [k for k, v in found.items() if v is None]
    if missing:
        print(f"!! synthetic blobs not detected for {missing}; timing still valid but check ranges")

    K, dist, P1, P2, X, pix1, pix2 = make_geometry(w, h, args.points)
    u1 = cv2.undistortPoints(pix1, K, dist, P=K)
    u2 = cv2.undistortPoints(pix2, K, dist, P=K)
    err_m = float(np.linalg.norm(triangulate(P1, P2, u1, u2) - X, axis=1).max())

    pipe_key = f"camera pipeline ({args.colors} colours)"
    pipe_m_key = f"camera pipeline +3x3 open ({args.colors} colours)"
    und_key = f"undistortPoints ({args.points} pts)"
    tri_key = f"triangulatePoints ({args.points} pairs)"

    ops = {}
    ops["cvtColor BGR2HSV"] = bench(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), args.iters)
    ops["inRange (1 colour)"] = bench(lambda: cv2.inRange(hsv, lo, hi), args.iters)
    ops["findContours+moments (1 colour)"] = bench(lambda: centroid(mask), args.iters)
    ops["morphologyEx OPEN 3x3 (optional)"] = bench(lambda: cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), args.iters)
    ops[pipe_key] = bench(lambda: camera_pipeline(frame, ranges), args.iters)
    ops[pipe_m_key] = bench(lambda: camera_pipeline(frame, ranges, kernel), args.iters)
    ops[und_key] = bench(lambda: cv2.undistortPoints(pix1, K, dist, P=K), args.iters)
    ops[tri_key] = bench(lambda: triangulate(P1, P2, u1, u2), args.iters)

    print(f"\n== per-op timing, {w}x{h}, {args.iters} iters (ms)")
    print(f"  {'op':44s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'max':>8s}")
    for name, r in ops.items():
        print(f"  {name:44s} {r['mean']:8.3f} {r['median']:8.3f} {r['p95']:8.3f} {r['max']:8.3f}")
    print(f"  triangulation max error vs ground truth: {err_m * 1000:.4f} mm")

    pipe, pipe_m, und, tri = ops[pipe_key], ops[pipe_m_key], ops[und_key], ops[tri_key]

    def cycle(p, key):
        return N_CAMS * p[key] + N_CAMS * und[key] + tri[key]

    cyc = {"mean": cycle(pipe, "mean"), "p95": cycle(pipe, "p95"),
           "mean_morph": cycle(pipe_m, "mean"), "p95_morph": cycle(pipe_m, "p95")}
    verdict = "PASS" if cyc["p95"] < BUDGET_MS else ("MARGINAL" if cyc["mean"] < BUDGET_MS else "FAIL")

    print(f"\n== estimated 2-camera tracking cycle ({args.colors} colours/camera; camera read excluded)")
    print(f"  as specified      mean {cyc['mean']:7.3f} ms   p95 {cyc['p95']:7.3f} ms   -> ~{1000 / cyc['mean']:.0f} Hz max")
    print(f"  with 3x3 open     mean {cyc['mean_morph']:7.3f} ms   p95 {cyc['p95_morph']:7.3f} ms   -> ~{1000 / cyc['mean_morph']:.0f} Hz max")
    print(f"  budget {BUDGET_MS:.0f} ms for 30 Hz  ->  {verdict}")

    # Emulation note: Prism multithreading can behave differently; show single-thread cost too.
    saved = cv2.getNumThreads()
    cv2.setNumThreads(1)
    st = bench(lambda: camera_pipeline(frame, ranges), args.iters)
    cv2.setNumThreads(saved)
    print(f"  (camera pipeline with cv2.setNumThreads(1): mean {st['mean']:.3f} ms "
          f"vs {pipe['mean']:.3f} ms with {saved} threads)")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"env": env, "args": vars(args), "ops": ops, "cycle_ms": cyc, "verdict": verdict,
                       "triangulation_max_err_mm": err_m * 1000, "single_thread_pipeline": st}, f, indent=2)
        print(f"  wrote {args.json}")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
