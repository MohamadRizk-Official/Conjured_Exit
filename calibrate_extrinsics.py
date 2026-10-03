#!/usr/bin/env python
"""calibrate_extrinsics.py -- camera pose (R, t) from the floor ArUco marker.

Step 3 of the tracker calibration (see tracker.py): run it for EACH camera
after the cameras are mounted and taped down and the intrinsics exist.

    cf64\\Scripts\\python.exe calibrate_extrinsics.py --print-marker calib\\marker0.png   # print this once
    cf64\\Scripts\\python.exe calibrate_extrinsics.py --camera 0 --marker-size 0.15
    cf64\\Scripts\\python.exe calibrate_extrinsics.py --camera 1 --marker-size 0.15

World frame = marker frame (metres, right-handed, z up out of the floor):
    origin  = marker centre
    +x      = forward = the direction the marker's TOP edge points
              (the top of the image as generated/printed)
    +y      = left (when standing behind the marker looking along +x)
    +z      = up
Place the printed marker flat on the floor with its top edge pointing in the
direction you want the drone to call "forward"; the drone's nose must point
along +x before takeoff (README.md).

Corner ordering: cv2.aruco returns the 4 corners clockwise from the marker's
top-left as printed: [TL, TR, BR, BL].  With h = marker_size / 2 their world
coordinates are
    TL (+h, +h, 0)   forward-left        TR (+h, -h, 0)   forward-right
    BR (-h, -h, 0)   back-right          BL (-h, +h, 0)   back-left
solvePnP (IPPE for planar targets + LM refinement) on the median of --frames
detections gives rvec/tvec with X_cam = R X_world + t.  The script saves
calib/cam{index}_extrinsics.npz (R, t, rvec, tvec, reproj_px, cam_pos, ...) and
prints the camera position as a sanity check: it must be ABOVE the floor
(z > 0) and roughly where you mounted it (~2 m away).

--marker-size is the side of the BLACK square in metres; measure the printout.
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

log = logging.getLogger("calibrate_extrinsics")

ARUCO_DICTS: dict[str, int] = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}


@dataclass
class ExtrinsicsResult:
    R: np.ndarray           # (3,3) world -> camera rotation
    t: np.ndarray           # (3,1)
    rvec: np.ndarray        # (3,)
    tvec: np.ndarray        # (3,)
    reproj_px: float        # RMS over the 4 corners

    @property
    def camera_position(self) -> np.ndarray:
        return (-self.R.T @ self.t).reshape(3)

    @property
    def camera_forward(self) -> np.ndarray:
        return (self.R.T @ np.array([0.0, 0.0, 1.0])).reshape(3)


def marker_object_points(marker_size_m: float) -> np.ndarray:
    """World coordinates of the marker corners in cv2.aruco order [TL, TR, BR, BL]."""
    h = float(marker_size_m) / 2.0
    return np.array([[h, h, 0.0],      # TL: forward-left
                     [h, -h, 0.0],     # TR: forward-right
                     [-h, -h, 0.0],    # BR: back-right
                     [-h, h, 0.0]],    # BL: back-left
                    dtype=np.float64)


def solve_extrinsics(corners_px: np.ndarray, K: np.ndarray, dist: np.ndarray | None,
                     marker_size_m: float) -> ExtrinsicsResult:
    """Camera pose from the 4 detected marker corners (raw pixels, aruco order).

    Uses solvePnPGeneric(IPPE) to get both planar solutions, keeps the ones with
    the camera above the floor, picks the lowest reprojection error, refines with LM.
    """
    obj = marker_object_points(marker_size_m)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if n < 1:
        raise RuntimeError("solvePnP found no solution")
    cands = []
    for r, t, e in zip(rvecs, tvecs, np.asarray(errs).reshape(-1)):
        R, _ = cv2.Rodrigues(r)
        z = float((-R.T @ t.reshape(3, 1))[2, 0])
        cands.append((z <= 0.0, float(e), r, t))   # above-floor solutions sort first
    cands.sort(key=lambda c: (c[0], c[1]))
    _, _, rvec, tvec = cands[0]
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img.reshape(-1, 2)
    reproj = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    return ExtrinsicsResult(R=np.asarray(R, np.float64), t=np.asarray(tvec, np.float64).reshape(3, 1),
                            rvec=np.asarray(rvec, np.float64).reshape(3), tvec=np.asarray(tvec, np.float64).reshape(3),
                            reproj_px=reproj)


def save_extrinsics(path: pathlib.Path, res: ExtrinsicsResult, corners_px: np.ndarray, marker_size_m: float,
                    marker_id: int, n_frames: int, spread_mm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, R=res.R, t=res.t.reshape(3), rvec=res.rvec, tvec=res.tvec, reproj_px=res.reproj_px,
             cam_pos=res.camera_position, cam_forward=res.camera_forward, corners_px=np.asarray(corners_px),
             marker_size_m=marker_size_m, marker_id=marker_id, n_frames=n_frames, spread_mm=spread_mm,
             created=time.strftime("%Y-%m-%d %H:%M:%S"))


def write_marker_png(path: pathlib.Path, dictionary: int, marker_id: int, px: int = 1000) -> None:
    """Printable marker with a white quiet zone and a 'TOP = +x forward' label."""
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    img = cv2.aruco.generateImageMarker(d, marker_id, px)
    m = px // 5
    canvas = np.full((px + 2 * m, px + 2 * m), 255, np.uint8)
    canvas[m:m + px, m:m + px] = img
    cv2.putText(canvas, f"TOP EDGE -> world +x (forward)   id {marker_id}", (m, m // 2),
                cv2.FONT_HERSHEY_SIMPLEX, px / 1000.0, 0, 2, cv2.LINE_AA)
    cv2.putText(canvas, "measure the black square -> --marker-size", (m, px + 2 * m - m // 3),
                cv2.FONT_HERSHEY_SIMPLEX, px / 1200.0, 0, 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--marker-size", type=float, default=None, help="black square side, metres (default tracker.json)")
    ap.add_argument("--id", type=int, default=None, help="marker id (default tracker.json: 0)")
    ap.add_argument("--dict", default=None, choices=sorted(ARUCO_DICTS), help="ArUco dictionary (default DICT_4X4_50)")
    ap.add_argument("--frames", type=int, default=15, help="detections to average")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--config", default="calib/tracker.json")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--print-marker", metavar="PNG", help="write a printable marker image and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = TrackerConfig.load(args.config)
    dict_name = args.dict or cfg.aruco_dict
    marker_id = args.id if args.id is not None else cfg.aruco_id
    size = args.marker_size if args.marker_size is not None else cfg.marker_size_m
    calib_dir = args.calib_dir or cfg.calib_dir

    if args.print_marker:
        p = pathlib.Path(args.print_marker)
        write_marker_png(p, ARUCO_DICTS[dict_name], marker_id)
        print(f"wrote {p}: print it as large as possible, measure the black square and pass --marker-size")
        return 0

    cam_calib = Camera.load(args.camera, calib_dir)
    if not cam_calib.has_intrinsics:
        print(f"ERROR: no intrinsics for camera {args.camera}; run {PYTHON_HINT} calibrate_intrinsics.py "
              f"--camera {args.camera} first", file=sys.stderr)
        return 2
    out = Camera.extrinsics_path(args.camera, calib_dir)
    show = not args.no_show

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name]),
                                       cv2.aruco.DetectorParameters())
    cam = CameraThread(args.camera, BACKENDS[args.backend or cfg.backend], cfg.width, cfg.height, cfg.fps)
    cam.start()
    if not cam.wait_open(10.0):
        print(f"ERROR: {cam.error or 'camera did not open in time'}", file=sys.stderr)
        return 2
    print(f"== extrinsics cam{args.camera}: {dict_name} id {marker_id}, size {size*1000:.0f} mm, "
          f"averaging {args.frames} detections -> {out}")
    print("   marker flat on the floor, top edge pointing forward (+x); do not move the camera.  q = quit")

    corner_sets: list[np.ndarray] = []
    per_frame_pos: list[np.ndarray] = []
    last_seq = -1
    deadline = time.perf_counter() + args.timeout
    try:
        while len(corner_sets) < args.frames:
            if time.perf_counter() > deadline:
                print(f"timeout after {args.timeout:.0f}s with {len(corner_sets)} detections")
                break
            frame, _t = cam.latest()
            seq = cam.seq
            if frame is None or seq == last_seq:
                if show and (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                time.sleep(0.003)
                continue
            last_seq = seq
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rej = detector.detectMarkers(gray)
            found = None
            if ids is not None:
                for c, i in zip(corners, ids.reshape(-1)):
                    if int(i) == marker_id:
                        found = np.asarray(c, np.float64).reshape(4, 2)
                        break
            if found is not None:
                corner_sets.append(found)
                try:
                    per_frame_pos.append(solve_extrinsics(found, cam_calib.K, cam_calib.dist, size).camera_position)
                except Exception as exc:  # pragma: no cover
                    log.warning("per-frame solve failed: %s", exc)
                if not show:
                    print(f"  detection {len(corner_sets)}/{args.frames}")
            if show:
                vis = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                if found is not None and per_frame_pos:
                    r = solve_extrinsics(found, cam_calib.K, cam_calib.dist, size)
                    cv2.drawFrameAxes(vis, cam_calib.K, cam_calib.dist, r.rvec, r.tvec, size)
                status = (f"cam{args.camera} {cam.fps:.0f} fps  detections {len(corner_sets)}/{args.frames}  "
                          f"{'marker ' + str(marker_id) + ' FOUND' if found is not None else 'looking for id ' + str(marker_id)}")
                cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(f"extrinsics cam{args.camera}", vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if show:
            cv2.destroyAllWindows()

    if len(corner_sets) < 3:
        print(f"not enough detections ({len(corner_sets)}); nothing saved")
        return 1

    med = np.median(np.stack(corner_sets), axis=0)
    res = solve_extrinsics(med, cam_calib.K, cam_calib.dist, size)
    pos, fwd = res.camera_position, res.camera_forward
    spread_mm = float(np.std(np.stack(per_frame_pos), axis=0).max() * 1000.0) if per_frame_pos else float("nan")
    dist_m = float(np.linalg.norm(pos))
    tilt = float(np.degrees(np.arcsin(np.clip(-fwd[2], -1.0, 1.0))))

    print(f"== result cam{args.camera}: {len(corner_sets)} detections, reprojection RMS {res.reproj_px:.2f} px, "
          f"per-frame position spread {spread_mm:.0f} mm")
    print(f"   camera position (world): x {pos[0]:+.3f}  y {pos[1]:+.3f}  z {pos[2]:+.3f} m  "
          f"-> {dist_m:.2f} m from the marker, {pos[2]:.2f} m above the floor, looking down {tilt:.0f} deg")
    ok = True
    if pos[2] <= 0.2:
        ok = False
        print("   WARNING: camera at or below the floor -> wrong solution; is the marker mirrored/printed "
              "from a photo, or the wrong id?")
    if fwd[2] >= 0:
        ok = False
        print("   WARNING: camera is looking UP; the pose is wrong")
    if not 0.5 <= dist_m <= 6.0:
        ok = False
        print(f"   WARNING: {dist_m:.1f} m from the marker is implausible; check --marker-size ({size} m)")
    if res.reproj_px > 2.0:
        print("   WARNING: reprojection error > 2 px; redo the intrinsics or print a bigger marker")
    if spread_mm > 50:
        print("   WARNING: unstable pose between frames (> 50 mm); the marker is small in the image - move "
              "the camera closer or print a bigger marker")
    save_extrinsics(out, res, med, size, marker_id, len(corner_sets), spread_mm)
    print(f"   saved {out}{'' if ok else '  (with warnings)'}")
    print(f"   next: {PYTHON_HINT} tools\\hsv_tune.py --camera {args.camera} --name nose   (then tail, wand)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
