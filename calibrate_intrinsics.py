#!/usr/bin/env python
"""calibrate_intrinsics.py -- checkerboard intrinsic calibration for ONE webcam.

Step 1 of the tracker calibration (see tracker.py):
    intrinsics per camera (this)  ->  tape the cameras down  ->  extrinsics per
    camera (calibrate_extrinsics.py)  ->  colours (tools/hsv_tune.py).

Show the checkerboard on a laptop/phone screen (or print it) and move it slowly
in front of the camera: tilt it, bring it close, cover the image corners.  A
frame is accepted when all inner corners are found and the board has moved at
least --min-move px since the last accepted frame.  After --frames frames
cv2.calibrateCamera runs and calib/cam{index}_intrinsics.npz is written with
K, dist, image_size, rms (plus per-view errors).

    cf64\\Scripts\\python.exe calibrate_intrinsics.py --camera 0 --board 9x6 --square 0.024
    cf64\\Scripts\\python.exe calibrate_intrinsics.py --camera 1 --board 9x6 --square 0.024 --frames 25 --no-show

--board COLSxROWS counts INNER corners (the classic OpenCV board is 9x6 = 10x7
squares); --square is the square side in metres as measured ON THE SCREEN.
The square size only scales the (unused) board poses, so a rough value is fine
for intrinsics.  Quality: rms < 0.5 px excellent, < 1 px fine, > 1.5 px redo.
Keys in the window: q quit, c calibrate now (needs >= 6 frames).
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from tracker import BACKENDS, PYTHON_HINT, Camera, CameraThread, TrackerConfig

log = logging.getLogger("calibrate_intrinsics")
MIN_FRAMES = 6


@dataclass
class CalibResult:
    rms: float
    K: np.ndarray
    dist: np.ndarray
    per_view_rms: np.ndarray


def board_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """(cols*rows, 3) float32 board coordinates (z = 0), same order as the detectors."""
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_m)
    return objp


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """All inner corners as (N,1,2) float32, or None.  SB detector first, legacy fallback."""
    n = pattern[0] * pattern[1]
    try:
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok and corners is not None and len(corners.reshape(-1, 2)) == n:
            return corners.reshape(-1, 1, 2).astype(np.float32)
    except cv2.error:
        pass
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if not ok or corners is None:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return corners.reshape(-1, 1, 2).astype(np.float32)


def calibrate_from_points(objpoints: list[np.ndarray], imgpoints: list[np.ndarray], image_size: tuple[int, int],
                          full_model: bool = False) -> CalibResult:
    """cv2.calibrateCamera wrapper.  Default model: k1 k2 p1 p2 (k3 fixed to 0, robust
    with few frames); --full-model frees k3.  image_size = (width, height)."""
    flags = 0 if full_model else cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, tuple(image_size), None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = proj.reshape(-1, 2) - np.asarray(i).reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    return CalibResult(float(rms), np.asarray(K, np.float64), np.asarray(dist, np.float64).reshape(-1),
                       np.asarray(per_view))


def coverage_fraction(corner_sets: list[np.ndarray], image_size: tuple[int, int], grid: tuple[int, int] = (8, 6)) -> float:
    """Fraction of an 8x6 grid of image cells touched by at least one corner."""
    if not corner_sets:
        return 0.0
    pts = np.concatenate([c.reshape(-1, 2) for c in corner_sets])
    gx = np.clip((pts[:, 0] / image_size[0] * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip((pts[:, 1] / image_size[1] * grid[1]).astype(int), 0, grid[1] - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(grid[0] * grid[1])


def save_intrinsics(path: pathlib.Path, res: CalibResult, image_size: tuple[int, int], n_frames: int,
                    board: str, square_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, K=res.K, dist=res.dist, image_size=np.array(image_size, dtype=np.int32), rms=res.rms,
             per_view_rms=res.per_view_rms, n_frames=n_frames, board=board, square_m=square_m,
             created=time.strftime("%Y-%m-%d %H:%M:%S"))


def quality_hint(rms: float) -> str:
    if rms < 0.5:
        return "excellent"
    if rms < 1.0:
        return "fine"
    if rms < 1.5:
        return "usable; more tilt / corner coverage would help"
    return "POOR - redo: hold still, tilt the board, cover the image corners, avoid screen glare"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--board", default=None, help="inner corners COLSxROWS (default from tracker.json: 9x6)")
    ap.add_argument("--square", type=float, default=None, help="square side in metres (default from tracker.json)")
    ap.add_argument("--frames", type=int, default=20, help="frames to collect")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--min-move", type=float, default=20.0, help="mean corner motion (px) required between frames")
    ap.add_argument("--min-interval", type=float, default=0.4, help="seconds between accepted frames")
    ap.add_argument("--timeout", type=float, default=240.0, help="give up after this many seconds")
    ap.add_argument("--full-model", action="store_true", help="also fit k3 (default fixes k3 = 0)")
    ap.add_argument("--no-show", action="store_true", help="headless: print progress instead of a window")
    ap.add_argument("--config", default="calib/tracker.json")
    ap.add_argument("--calib-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = TrackerConfig.load(args.config)
    board = args.board or cfg.board
    square = args.square if args.square is not None else cfg.square_m
    cols, rows = (int(x) for x in board.lower().split("x"))
    pattern = (cols, rows)
    backend = BACKENDS[args.backend or cfg.backend]
    width, height = args.width or cfg.width, args.height or cfg.height
    out = Camera.intrinsics_path(args.camera, args.calib_dir or cfg.calib_dir)
    show = not args.no_show

    cam = CameraThread(args.camera, backend, width, height, cfg.fps)
    cam.start()
    if not cam.wait_open(10.0):
        print(f"ERROR: {cam.error or 'camera did not open in time'}", file=sys.stderr)
        return 2
    print(f"== intrinsics cam{args.camera}: board {cols}x{rows} inner corners, square {square*1000:.1f} mm, "
          f"collecting {args.frames} frames, output {out}")
    print("   move the board slowly: tilt it, come close, cover the image corners.  q = quit, c = calibrate now")

    objp = board_object_points(cols, rows, square)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    last_corners: np.ndarray | None = None
    last_accept = -1e9
    last_seq = -1
    flash_until = 0.0
    image_size: tuple[int, int] | None = None
    deadline = time.perf_counter() + args.timeout
    rc = 0
    try:
        while len(imgpoints) < args.frames:
            now = time.perf_counter()
            if now > deadline:
                print(f"timeout after {args.timeout:.0f}s with {len(imgpoints)} frames")
                break
            frame, t = cam.latest()
            seq = cam.seq
            if frame is None or seq == last_seq:
                if show and (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                time.sleep(0.003)
                continue
            last_seq = seq
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern)
            accepted = False
            if corners is not None:
                moved = (last_corners is None
                         or float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - last_corners.reshape(-1, 2), axis=1)))
                         >= args.min_move)
                if moved and (t or now) - last_accept >= args.min_interval:
                    objpoints.append(objp.copy())
                    imgpoints.append(corners)
                    last_corners, last_accept, accepted = corners, (t or now), True
                    flash_until = now + 0.25
                    if not show:
                        print(f"  accepted {len(imgpoints)}/{args.frames}  coverage {coverage_fraction(imgpoints, image_size):.0%}")
            if show:
                vis = frame.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(vis, pattern, corners, True)
                if now < flash_until:
                    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 0), 8)
                status = (f"cam{args.camera} {cam.fps:.0f} fps  accepted {len(imgpoints)}/{args.frames}  "
                          f"coverage {coverage_fraction(imgpoints, image_size):.0%}  "
                          f"{'board FOUND - move it' if corners is not None else 'no board'}")
                cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(f"intrinsics cam{args.camera}", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c") and len(imgpoints) >= MIN_FRAMES:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if show:
            cv2.destroyAllWindows()

    if len(imgpoints) < MIN_FRAMES or image_size is None:
        print(f"not enough frames ({len(imgpoints)} < {MIN_FRAMES}); nothing saved")
        return 1

    res = calibrate_from_points(objpoints, imgpoints, image_size, full_model=args.full_model)
    fx, fy, cx, cy = res.K[0, 0], res.K[1, 1], res.K[0, 2], res.K[1, 2]
    print(f"== result cam{args.camera}: {len(imgpoints)} frames, {image_size[0]}x{image_size[1]}")
    print(f"   RMS reprojection error {res.rms:.3f} px ({quality_hint(res.rms)}); per view "
          f"min {res.per_view_rms.min():.2f} max {res.per_view_rms.max():.2f} px; "
          f"coverage {coverage_fraction(imgpoints, image_size):.0%}")
    print(f"   fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}  (expect cx~{image_size[0]/2:.0f}, cy~{image_size[1]/2:.0f}, "
          f"fx~fy; hfov {2*np.degrees(np.arctan(image_size[0]/(2*fx))):.0f} deg)")
    print(f"   dist {np.array2string(res.dist, precision=4, suppress_small=True)}")
    worst = res.per_view_rms.max()
    if worst > 3 * max(res.rms, 0.2):
        print(f"   note: one view has {worst:.2f} px error (motion blur?); rerun if the RMS is poor")
    save_intrinsics(out, res, image_size, len(imgpoints), board, square)
    print(f"   saved {out}")
    print(f"   next: tape the camera down, then {PYTHON_HINT} calibrate_extrinsics.py --camera {args.camera} "
          f"--marker-size {cfg.marker_size_m}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
