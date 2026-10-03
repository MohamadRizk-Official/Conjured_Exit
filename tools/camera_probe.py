#!/usr/bin/env python
"""camera_probe.py -- headless webcam probe for the pathcaster tracker.

For each OpenCV backend (MSMF, DSHOW) and camera index 0..4: open, read the
reported width/height/fps, time the first frame, read N frames to measure the
real frame rate and per-read latency, then release.  No GUI windows are opened.

Every (backend, index) probe runs in a child process with a timeout, so a
backend that hangs on open/read shows up as TIMEOUT instead of freezing the
whole probe.  Camera friendly names are listed via WinRT (installed with
bleak) to tell the built-in webcam from the USB one; MSMF index order normally
follows that enumeration order.

Usage (project root, cf64 venv):
    .\\cf64\\Scripts\\python.exe tools\\camera_probe.py
    .\\cf64\\Scripts\\python.exe tools\\camera_probe.py --indices 0-1 --frames 90 --width 640 --height 480 --fps 30
    .\\cf64\\Scripts\\python.exe tools\\camera_probe.py --backends MSMF --json results\\camera_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import cv2

BACKENDS = {"MSMF": cv2.CAP_MSMF, "DSHOW": cv2.CAP_DSHOW}
BUILTIN_RE = re.compile(r"integrated|built.?in|internal|front|rear|facing", re.I)


# ----------------------------------------------------------------------------- worker
def worker(backend: str, index: int, frames: int, width, height, fps) -> dict:
    """Runs in the child process. Prints nothing itself; returns a dict for JSON."""
    r: dict = {"backend": backend, "index": index}
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    r["open_ms"] = (time.perf_counter() - t0) * 1000.0
    r["opened"] = bool(cap.isOpened())
    if not r["opened"]:
        cap.release()
        return r
    try:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        r["backend_name"] = cap.getBackendName()
        r["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        r["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        r["fps_reported"] = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF
        r["fourcc"] = fourcc.to_bytes(4, "little").decode("ascii", "replace").strip("\x00") if fourcc else ""

        t1 = time.perf_counter()
        ok, frame = cap.read()
        r["first_frame_ms"] = (time.perf_counter() - t1) * 1000.0
        r["first_ok"] = bool(ok)
        if ok and frame is not None:
            r["frame_shape"] = list(frame.shape)

        lat, fails = [], 0
        t_start = time.perf_counter()
        for _ in range(frames):
            ta = time.perf_counter()
            ok, frame = cap.read()
            tb = time.perf_counter()
            if ok and frame is not None:
                lat.append((tb - ta) * 1000.0)
            else:
                fails += 1
        elapsed = time.perf_counter() - t_start
        r["frames_ok"] = len(lat)
        r["fails"] = fails
        r["real_fps"] = len(lat) / elapsed if elapsed > 0 else 0.0
        if lat:
            lat.sort()
            r["mean_read_ms"] = statistics.fmean(lat)
            r["p95_read_ms"] = lat[int(0.95 * (len(lat) - 1))]
            r["max_read_ms"] = lat[-1]
    finally:
        cap.release()
    return r


# ----------------------------------------------------------------------------- parent
def run_probe(backend: str, index: int, args) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", backend, str(index), "--frames", str(args.frames)]
    for flag in ("width", "height", "fps"):
        v = getattr(args, flag)
        if v:
            cmd += [f"--{flag}", str(v)]
    env = dict(os.environ, OPENCV_LOG_LEVEL="ERROR", PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"backend": backend, "index": index, "opened": None,
                "note": f"TIMEOUT >{args.timeout:.0f}s (backend hung on open/read)"}
    wall = time.perf_counter() - t0
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("{")]
    if cp.returncode != 0 or not lines:
        tail = cp.stderr.strip().splitlines()[-1] if cp.stderr.strip() else ""
        return {"backend": backend, "index": index, "opened": None,
                "note": f"CRASH rc={cp.returncode} {tail}"[:140]}
    r = json.loads(lines[-1])
    r["wall_s"] = wall
    if args.verbose and cp.stderr.strip():
        r["stderr"] = cp.stderr.strip()
    return r


def list_video_devices():
    """[(name, device_id, likely_builtin)] for every video-capture device.

    Tries WinRT DeviceInformation (the winrt-* packages ship with bleak), then
    falls back to PowerShell Get-PnpDevice.  Built-in heuristic: a name like
    "Integrated Camera" or a device id that is not on the USB bus.
    """
    try:
        import asyncio
        from winrt.windows.devices.enumeration import DeviceClass, DeviceInformation

        async def go():
            # pywinrt (winrt-* 3.x) names the DeviceClass overload explicitly; older
            # projections expose it through plain find_all_async.
            try:
                return await DeviceInformation.find_all_async_device_class(DeviceClass.VIDEO_CAPTURE)
            except AttributeError:
                return await DeviceInformation.find_all_async(DeviceClass.VIDEO_CAPTURE)

        coll = asyncio.run(go())
        try:
            items = list(coll)
        except TypeError:
            items = [coll.get_at(i) for i in range(coll.size)]
        out = []
        for d in items:
            name, did = d.name, d.id
            out.append((name, did, bool(BUILTIN_RE.search(name)) or "USB#" not in did.upper()))
        return out, "WinRT DeviceInformation"
    except Exception as e:
        try:
            ps = ("Get-PnpDevice -Class Camera,Image -Status OK | "
                  "ForEach-Object { $_.FriendlyName + '|' + $_.InstanceId }")
            cp = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                capture_output=True, text=True, timeout=30)
            out = []
            for line in cp.stdout.splitlines():
                if "|" in line:
                    name, did = line.split("|", 1)
                    out.append((name.strip(), did.strip(),
                                bool(BUILTIN_RE.search(name)) or "USB\\" not in did.upper()))
            return out, f"Get-PnpDevice (WinRT failed: {type(e).__name__})"
        except Exception as e2:
            return [], f"no device listing ({type(e).__name__}; {type(e2).__name__})"


def fmt_row(r: dict) -> str:
    be, idx = r["backend"], r["index"]
    if r.get("opened") is None:
        return f"  {be:5s} {idx:3d}  {'??':5s} {'':>7s} {'':>7s} {'':>16s} {'':>5s} {'':>7s} {'':>7s} {'':>7s} {'':>4s}  {r.get('note', '')}"
    if not r["opened"]:
        return f"  {be:5s} {idx:3d}  {'no':5s} {r['open_ms']:7.0f} {'':>7s} {'':>16s} {'':>5s} {'':>7s} {'':>7s} {'':>7s} {'':>4s}  not opened"
    rep = f"{r.get('width', 0)}x{r.get('height', 0)}@{r.get('fps_reported', 0):.0f}"
    shape = r.get("frame_shape")
    note = ""
    if shape and (shape[1] != r.get("width") or shape[0] != r.get("height")):
        note = f"actual frame {shape[1]}x{shape[0]}"
    if not r.get("first_ok"):
        note = (note + "; " if note else "") + "first read FAILED (in use by another app?)"
    if r.get("fails"):
        note = (note + "; " if note else "") + f"{r['fails']} failed reads"
    return (f"  {be:5s} {idx:3d}  {'yes':5s} {r['open_ms']:7.0f} {r.get('first_frame_ms', 0):7.0f} {rep:>16s} "
            f"{r.get('fourcc', ''):>5s} {r.get('real_fps', 0):7.1f} {r.get('mean_read_ms', 0):7.1f} "
            f"{r.get('p95_read_ms', 0):7.1f} {r.get('fails', 0):4d}  {note}")


HEADER = (f"  {'bknd':5s} {'idx':3s}  {'open':5s} {'open_ms':>7s} {'1st_ms':>7s} {'reported WxH@fps':>16s} "
          f"{'4cc':>5s} {'realfps':>7s} {'read_ms':>7s} {'p95_ms':>7s} {'fail':>4s}  note")


def summarize(results: list[dict], devices, source: str):
    print("\n== video devices (" + source + "; MSMF index order usually matches this list)")
    if not devices:
        print("  (none listed)")
    for i, (name, did, builtin) in enumerate(devices):
        bus = "USB" if "USB" in did.upper() else "non-USB"
        print(f"  [{i}] {name}  ({bus}; {'likely BUILT-IN' if builtin else 'likely external USB webcam'})")

    print("\n== summary")
    opened = [r for r in results if r.get("opened")]
    if not opened:
        print("  no camera opened on any backend")
        return None
    for be in BACKENDS:
        rs = [r for r in results if r["backend"] == be]
        ok = [r for r in rs if r.get("opened")]
        bad = [r for r in rs if r.get("opened") is None]
        if ok:
            print(f"  {be:5s}: {len(ok)} camera(s) opened  idx={[r['index'] for r in ok]}  "
                  f"mean open {statistics.fmean(r['open_ms'] for r in ok):.0f} ms, "
                  f"mean first frame {statistics.fmean(r.get('first_frame_ms', 0) for r in ok):.0f} ms, "
                  f"{len(bad)} timeout/crash")
        else:
            print(f"  {be:5s}: nothing opened, {len(bad)} timeout/crash")

    def score(be):
        rs = [r for r in results if r["backend"] == be]
        ok = [r for r in rs if r.get("opened") and r.get("frames_ok")]
        hang = sum(1 for r in rs if r.get("opened") is None)
        t_open = statistics.fmean(r["open_ms"] + r.get("first_frame_ms", 0) for r in ok) if ok else 1e9
        return (-len(ok), hang, t_open)

    best = min(BACKENDS, key=score)
    print(f"  recommended backend: cv2.CAP_{best}  (most cameras, no hangs, fastest open+first frame)")
    for r in sorted(opened, key=lambda r: (r["backend"] != best, r["index"])):
        if r["backend"] == best:
            name = devices[r["index"]][0] if r["index"] < len(devices) else "?"
            print(f"    cv2.VideoCapture({r['index']}, cv2.CAP_{best})  {r.get('width')}x{r.get('height')} "
                  f"real {r.get('real_fps', 0):.1f} fps  read {r.get('mean_read_ms', 0):.1f} ms  -> {name}")
    return best


def parse_indices(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", nargs=2, metavar=("BACKEND", "INDEX"), help=argparse.SUPPRESS)
    ap.add_argument("--backends", default="MSMF,DSHOW", help="comma list from MSMF,DSHOW")
    ap.add_argument("--indices", default="0-4", help="e.g. 0-4 or 0,1,3")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, help="request this frame width before grabbing")
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--timeout", type=float, default=25.0,
                    help="seconds per (backend,index) probe before declaring a hang")
    ap.add_argument("--json", help="write all results to this JSON file")
    ap.add_argument("--verbose", action="store_true", help="include child stderr (OpenCV warnings)")
    args = ap.parse_args()

    if args.worker:
        print(json.dumps(worker(args.worker[0], int(args.worker[1]), args.frames, args.width, args.height, args.fps)))
        return 0

    backends = [b.strip().upper() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            ap.error(f"unknown backend {b}; choose from {list(BACKENDS)}")
    indices = parse_indices(args.indices)

    print(f"== camera probe  python {sys.version.split()[0]} ({'x64' if 'AMD64' in sys.version else 'native'})  "
          f"opencv {cv2.__version__}  frames={args.frames}  timeout={args.timeout:.0f}s/probe")
    print("   (each probe is a child process; a camera held by another app opens but fails to read)")
    print(HEADER)
    results: list[dict] = []
    for be in backends:
        for idx in indices:
            r = run_probe(be, idx, args)
            results.append(r)
            print(fmt_row(r), flush=True)

    devices, source = list_video_devices()
    best = summarize(results, devices, source)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results, "recommended_backend": best,
                       "devices": [{"name": n, "id": i, "likely_builtin": b} for n, i, b in devices],
                       "device_source": source}, f, indent=2)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
