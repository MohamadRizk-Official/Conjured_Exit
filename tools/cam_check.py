#!/usr/bin/env python
"""cam_check.py -- live placement check for the ArUco rig (one camera).

Shows the camera with detections drawn and a thirds grid, and prints one status line whenever what
it sees changes (and every --every seconds regardless): for each marker id its position in the
picture (left/centre/right x top/middle/bottom), its width in pixels and the approximate distance
(from the camera focal length and the marker's real size). Also saves an annotated snapshot to
--snapshot every second so a remote helper can look at the picture.

    cf64\\Scripts\\python.exe tools\\cam_check.py                 # window + report, q quits
    cf64\\Scripts\\python.exe tools\\cam_check.py --once          # one frame, report, snapshot, exit
    cf64\\Scripts\\python.exe tools\\cam_check.py --headless --log results\\cam_check.log

Camera index/resolution/backend come from calib/tracker.json; marker sizes too (id 0 floor,
id 1 drone, id 2 wand). Intrinsics (calib/camN_intrinsics.npz) are used only for the distance
estimate; the check works without them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tracker  # noqa: E402

BACKENDS = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}


def region(cx: float, cy: float, w: int, h: int) -> str:
    xs = ("left", "centre", "right")[min(2, int(3 * cx / w))]
    ys = ("top", "middle", "bottom")[min(2, int(3 * cy / h))]
    return f"{ys}-{xs}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until q)")
    ap.add_argument("--every", type=float, default=10.0, help="print a line at least this often")
    ap.add_argument("--snapshot", default="results/cam_check.jpg")
    ap.add_argument("--log", default=None, help="also append status lines to this file")
    args = ap.parse_args(argv)

    cfg = tracker.TrackerConfig.load()
    cam_idx = cfg.aruco_camera if args.camera is None else args.camera
    sizes = {cfg.floor_marker_id if hasattr(cfg, "floor_marker_id") else 0: getattr(cfg, "marker_size_m", 0.15),
             cfg.drone_marker_id: cfg.drone_marker_size_m, cfg.wand_marker_id: cfg.wand_marker_size_m}
    fx = None
    try:
        d = np.load(os.path.join(cfg.calib_dir, f"cam{cam_idx}_intrinsics.npz"))
        fx = float(d["K"][0, 0])
    except Exception:  # noqa: BLE001
        pass

    cap = cv2.VideoCapture(cam_idx, BACKENDS.get(cfg.backend.upper(), cv2.CAP_ANY))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    ok, frame = cap.read()
    if not ok:
        print(f"ERROR: camera {cam_idx} gives no frames", file=sys.stderr)
        return 2
    h, w = frame.shape[:2]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    logf = open(args.log, "a", encoding="ascii", errors="replace") if args.log else None
    os.makedirs(os.path.dirname(args.snapshot) or ".", exist_ok=True)

    def emit(line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp}  {line}", flush=True)
        if logf:
            logf.write(f"{stamp}  {line}\n")
            logf.flush()

    emit(f"camera {cam_idx} {w}x{h} {cfg.backend}; fx {fx:.0f}px" if fx else f"camera {cam_idx} {w}x{h} {cfg.backend}; no intrinsics (no distances)")
    t0 = time.perf_counter()
    last_key = None
    last_print = 0.0
    last_snap = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                emit("no frame")
                time.sleep(0.2)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = det.detectMarkers(gray)
            parts = []
            key = []
            if ids is not None:
                for c, i in zip(corners, ids.ravel()):
                    pts = c.reshape(4, 2)
                    cx, cy = pts.mean(axis=0)
                    side = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
                    reg = region(cx, cy, w, h)
                    dist = f" ~{fx * sizes.get(int(i), 0.05) / max(side, 1e-6):.1f}m" if fx and int(i) in sizes else ""
                    parts.append(f"id{int(i)}: {reg} {side:.0f}px{dist}")
                    key.append((int(i), reg, int(side // 8)))
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            now = time.perf_counter()
            key_t = tuple(sorted(key))
            if key_t != last_key or now - last_print >= args.every:
                emit(" | ".join(parts) if parts else "no markers")
                last_key, last_print = key_t, now
            # thirds grid + text
            for k in (1, 2):
                cv2.line(frame, (k * w // 3, 0), (k * w // 3, h), (90, 90, 90), 1)
                cv2.line(frame, (0, k * h // 3), (w, k * h // 3), (90, 90, 90), 1)
            cv2.putText(frame, " | ".join(parts) if parts else "no markers", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if now - last_snap >= 1.0 or args.once:
                cv2.imwrite(args.snapshot, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                last_snap = now
            if args.once:
                break
            if not args.headless:
                cv2.imshow("cam check (q quits)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.05)
            if args.seconds and now - t0 >= args.seconds:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if logf:
            logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
