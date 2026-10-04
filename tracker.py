#!/usr/bin/env python
"""tracker.py -- two-webcam colour-marker tracker for the Pathcaster drone.

Pipeline (README.md "Tracker"):
    one capture thread per camera (newest frame wins)
    -> HSV threshold per sticky-note colour -> largest-blob centroid per camera
    -> undistort -> cv2.triangulatePoints with P = K[R|t]
    -> nose/tail fusion (position + yaw) -> outlier gating
    -> TrackerState at the camera rate (~30 Hz), read with Tracker.get_state().

Frames and units
    World frame: metres, x forward, y left, z up.  Origin = centre of the floor
    ArUco marker, +x = the direction the marker's "top" edge points (see
    calibrate_extrinsics.py).  Camera extrinsics follow OpenCV: X_cam = R X_world + t.
    yaw (radians) = heading of the tail->nose vector (nose minus tail) projected
    on the ground plane: yaw 0 = nose along +x, +pi/2 = nose along +y (left).
    Timestamps are time.perf_counter() seconds (same clock as feed.py in-process).

Undistortion convention
    Camera.undistort_point() returns *pixel* coordinates in the ideal, distortion-
    free camera with the same K (cv2.undistortPoints with P=K).  triangulate()
    therefore takes RAW detected pixels, undistorts them itself and uses
    Camera.P = K @ [R|t].  reprojection_error() projects a world point WITH
    distortion and compares against the raw pixels (RMS over cameras, px).

Calibration procedure (robotics teammate, run from the project root in cf64)
    1. Intrinsics, once per camera, anywhere:
         cf64\\Scripts\\python.exe calibrate_intrinsics.py --camera 0 --board 9x6 --square 0.024
         cf64\\Scripts\\python.exe calibrate_intrinsics.py --camera 1 --board 9x6 --square 0.024
    2. Mount both cameras high, ~90 degrees apart, tape them down.  Put the printed
       ArUco marker (DICT_4X4_50 id 0) flat on the floor where both cameras see it,
       top edge pointing in the direction you want to call forward (+x).
    3. Extrinsics, per camera (do not touch the cameras afterwards):
         cf64\\Scripts\\python.exe calibrate_extrinsics.py --camera 0 --marker-size 0.15
         cf64\\Scripts\\python.exe calibrate_extrinsics.py --camera 1 --marker-size 0.15
    4. Colours, per marker, under the demo lighting:
         cf64\\Scripts\\python.exe tools\\hsv_tune.py --camera 0 --name nose
         cf64\\Scripts\\python.exe tools\\hsv_tune.py --camera 0 --name tail
         cf64\\Scripts\\python.exe tools\\hsv_tune.py --camera 0 --name wand
    5. Check:  cf64\\Scripts\\python.exe tracker.py --show

Files: calib/cam{i}_intrinsics.npz (K, dist, image_size, rms),
calib/cam{i}_extrinsics.npz (R, t, rvec, reproj_px, ...), calib/tracker.json
(TrackerConfig; written by hsv_tune.py).  Missing files are reported and the
tracker keeps running in "2d-only" mode (detections only, tracking_ok False).

Single-camera ArUco mode (ArucoTracker; no triangulation, no colour tuning)
    A printed DICT_4X4_50 marker id 1 (7 cm, calib/marker1_drone_7cm.png) is taped
    flat on top of the drone with its TOP EDGE pointing at the nose.  ONE calibrated
    camera sees the flight volume from ~1.5-2.5 m, ~45 degrees down.  Every frame:
    cv2.aruco.ArucoDetector -> cv2.solvePnP(SOLVEPNP_IPPE_SQUARE) on the RAW corners
    with the camera's distortion model (same model Camera.project uses; no separate
    undistortion step) -> marker centre / axes mapped into the world frame with the
    camera extrinsics (X_world = R^T (X_cam - t)).  position = marker centre; yaw =
    heading of the marker's top-edge direction (the marker's +y axis) in world xy:
    yaw 0 = nose along world +x, +pi/2 = nose along +y (left).  IPPE returns two
    planar solutions: the one whose marker normal points most towards world +z
    (marker faces up) wins, ties by reprojection error.  Marker id 2 (5 cm) is the
    optional wand tip (lowest-reprojection solution, position only).  Gating,
    tracking_ok, fps and latency are the same as the two-camera tracker.
    Setup (print calib/marker0_floor_15cm.png and calib/marker1_drone_7cm.png at 100 %):
      1. cf64\\Scripts\\python.exe calibrate_intrinsics.py --camera 0 --board 9x6 --square 0.024
      2. mount/tape the camera; tape the floor marker (id 0) down, TOP edge pointing forward (+x)
      3. cf64\\Scripts\\python.exe calibrate_extrinsics.py --camera 0 --marker-size 0.15
      4. tape marker id 1 flat on top of the drone, TOP edge at the nose
      5. cf64\\Scripts\\python.exe tracker.py --aruco --camera 0 --show
         (headless check: tracker.py --aruco --camera 0 --headless --seconds 10)
    Accuracy (tests/test_tracker_aruco.py, synthetic 640x480 f=600 px camera 2 m from
    the origin, 45 deg down, 7 cm marker at (0,0,0.5) = 1.7 m away and 26 px wide,
    0.5 px corner noise): position error mean 2.9 cm / p95 7 cm, almost entirely
    along the viewing ray (lateral ~1 mm); yaw error mean 0.9 deg / p95 2.5 deg.
    At 2.3 m (21 px) expect ~5 cm mean.  The depth error scales with distance^2 /
    marker size: a 10 cm marker or a camera at 1.5 m roughly halves it; 1280x720
    doubles the pixels per marker.  Corner refinement SUBPIX is on by default.

CLI
    python tracker.py --show               windows per camera + state at 5 Hz
    python tracker.py --headless --seconds 10
    python tracker.py --aruco --camera 0 [--show | --headless --seconds N]
    python tracker.py --sim [--path exit_a] [--seconds 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

__all__ = [
    "BACKENDS",
    "MARKERS",
    "DEFAULT_CONFIG_PATH",
    "HsvRange",
    "TrackerConfig",
    "MarkerDetector",
    "Camera",
    "camera_from_look_at",
    "triangulate",
    "reprojection_error",
    "ARUCO_DICTS",
    "square_marker_object_points",
    "MarkerPose",
    "solve_marker_pose",
    "ArucoTracker",
    "FrameSource",
    "CameraThread",
    "ImageSource",
    "CameraView",
    "TrackerState",
    "JumpGate",
    "PoseFuser",
    "Tracker",
    "SimTracker",
]

log = logging.getLogger("tracker")

BACKENDS: dict[str, int] = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
ARUCO_DICTS: dict[str, int] = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}
MARKERS: tuple[str, ...] = ("nose", "tail", "wand")
DEFAULT_CONFIG_PATH = "calib/tracker.json"
PYTHON_HINT = "cf64\\Scripts\\python.exe"


# ============================================================================ config


@dataclass(frozen=True)
class HsvRange:
    """Inclusive HSV range in OpenCV units (H 0..179, S/V 0..255).

    If ``lo[0] > hi[0]`` the hue range wraps around 180 (red): the detector then
    ORs the ranges [lo_h..179] and [0..hi_h].
    """

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    @property
    def wraps(self) -> bool:
        return self.lo[0] > self.hi[0]

    def to_dict(self) -> dict[str, list[int]]:
        return {"lo": [int(x) for x in self.lo], "hi": [int(x) for x in self.hi]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HsvRange":
        lo, hi = d["lo"], d["hi"]
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError(f"HSV range needs 3 values each, got lo={lo} hi={hi}")
        return cls(tuple(int(x) for x in lo), tuple(int(x) for x in hi))  # type: ignore[arg-type]


# Placeholders only: pure red / blue / green so synthetic tests pass.  Real
# sticky-note colours come from tools/hsv_tune.py -> calib/tracker.json.
DEFAULT_COLORS: dict[str, HsvRange] = {
    "nose": HsvRange((170, 100, 80), (10, 255, 255)),    # red (wraps around 180)
    "tail": HsvRange((100, 100, 80), (130, 255, 255)),   # blue
    "wand": HsvRange((45, 80, 80), (80, 255, 255)),      # green
}


@dataclass
class TrackerConfig:
    """Every tunable of the tracker.  JSON file: calib/tracker.json (partial files OK)."""

    camera_indices: tuple[int, int] = (0, 1)
    backend: str = "DSHOW"              # key of BACKENDS
    width: int = 640
    height: int = 480
    fps: int = 30
    colors: dict[str, HsvRange] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    min_area: int = 30                  # px^2, smaller blobs are ignored
    open_kernel: int = 3                # morphological open kernel; 0 disables
    aruco_dict: str = "DICT_4X4_50"     # all ArUco markers (floor origin, drone, wand)
    aruco_id: int = 0                   # floor marker = world origin
    marker_size_m: float = 0.15         # floor marker black square side, metres
    drone_marker_id: int = 1            # ArucoTracker: marker taped on top of the drone, top edge at the nose
    drone_marker_size_m: float = 0.07
    wand_marker_id: int = 2             # ArucoTracker: optional wand-tip marker
    wand_marker_size_m: float = 0.05
    aruco_camera: int = 0               # ArucoTracker: the single camera to use
    aruco_corner_refine: str = "SUBPIX" # ArucoDetector cornerRefinementMethod: NONE | SUBPIX | CONTOUR | APRILTAG
    board: str = "9x6"                  # checkerboard INNER corners cols x rows
    square_m: float = 0.024             # checkerboard square side, metres
    max_jump_m: float = 0.5             # gating: max jump from the previous accepted fix
    max_reproj_px: float = 8.0          # gating: max RMS reprojection error of a marker
    reacquire_after: int = 10           # consecutive jump rejections before accepting anyway
    lost_after_s: float = 0.3           # tracking_ok False when the last fix is older (safety rule)
    max_sync_s: float = 0.06            # max age difference between the two frames of a pair
    calib_dir: str = "calib"

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "colors":
                d[k] = {name: r.to_dict() for name, r in v.items()}
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackerConfig":
        cfg = cls()
        for k, v in data.items():
            if k == "colors":
                for name, r in v.items():
                    cfg.colors[name] = HsvRange.from_dict(r)
            elif k == "camera_indices":
                cfg.camera_indices = tuple(int(x) for x in v)  # type: ignore[assignment]
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                log.warning("tracker config: unknown key %r ignored", k)
        if cfg.backend.upper() not in BACKENDS:
            raise ValueError(f"backend must be one of {list(BACKENDS)}, got {cfg.backend!r}")
        cfg.backend = cfg.backend.upper()
        return cfg

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "TrackerConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | pathlib.Path = DEFAULT_CONFIG_PATH) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | pathlib.Path | None = None) -> "TrackerConfig":
        """Config from ``path`` (default calib/tracker.json) or defaults if it is missing."""
        p = pathlib.Path(path or DEFAULT_CONFIG_PATH)
        if p.exists():
            cfg = cls.from_json(p)
            log.info("tracker config loaded from %s", p)
            return cfg
        log.warning("%s not found -> default TrackerConfig (placeholder colours; run tools\\hsv_tune.py)", p)
        return cls()

    @staticmethod
    def save_color(name: str, lo: Sequence[int], hi: Sequence[int],
                   path: str | pathlib.Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
        """Merge one HSV range into the JSON config file (creates it if needed)."""
        p = pathlib.Path(path)
        data: dict[str, Any] = {}
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8") or "{}")
        data.setdefault("colors", {})[name] = HsvRange(tuple(lo), tuple(hi)).to_dict()  # type: ignore[arg-type]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    @property
    def backend_id(self) -> int:
        return BACKENDS[self.backend.upper()]

    @property
    def board_size(self) -> tuple[int, int]:
        cols, rows = self.board.lower().split("x")
        return int(cols), int(rows)


# ========================================================================== detector


class MarkerDetector:
    """HSV inRange -> optional 3x3 open -> largest contour centroid.

    ``detect(hsv)`` takes an HSV image (convert once per frame, share across
    colours) and returns ``(u, v, area)`` in pixels or ``None``.
    """

    def __init__(self, hsv_lo: Sequence[int], hsv_hi: Sequence[int], min_area: float = 30,
                 open_kernel: int = 3) -> None:
        self.lo = np.array(hsv_lo, dtype=np.uint8).reshape(3)
        self.hi = np.array(hsv_hi, dtype=np.uint8).reshape(3)
        self.min_area = float(min_area)
        self.wraps = bool(self.lo[0] > self.hi[0])
        self.kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
                       if open_kernel and open_kernel >= 2 else None)

    @classmethod
    def from_range(cls, r: HsvRange, cfg: TrackerConfig) -> "MarkerDetector":
        return cls(r.lo, r.hi, cfg.min_area, cfg.open_kernel)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        if self.wraps:
            lo1 = self.lo
            hi1 = np.array([179, self.hi[1], self.hi[2]], np.uint8)
            lo2 = np.array([0, self.lo[1], self.lo[2]], np.uint8)
            hi2 = self.hi
            m = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
        else:
            m = cv2.inRange(hsv, self.lo, self.hi)
        if self.kernel is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        return m

    def detect(self, hsv: np.ndarray) -> tuple[float, float, float] | None:
        cnts, _ = cv2.findContours(self.mask(hsv), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"] < self.min_area or m["m00"] <= 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"])

    def detect_bgr(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        return self.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))


# ============================================================================ camera


@dataclass
class Camera:
    """Calibration of one camera.  X_cam = R @ X_world + t (OpenCV convention)."""

    index: int
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    image_size: tuple[int, int] | None = None   # (width, height)
    rms: float | None = None                     # intrinsic calibration RMS, px

    def __post_init__(self) -> None:
        if self.K is not None:
            self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if self.dist is not None:
            self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        elif self.K is not None:
            self.dist = np.zeros(5)
        if self.R is not None:
            self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        if self.t is not None:
            self.t = np.asarray(self.t, dtype=np.float64).reshape(3, 1)
        if self.image_size is not None:
            self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    # ---- files -------------------------------------------------------------
    @staticmethod
    def intrinsics_path(index: int, calib_dir: str | pathlib.Path = "calib") -> pathlib.Path:
        return pathlib.Path(calib_dir) / f"cam{index}_intrinsics.npz"

    @staticmethod
    def extrinsics_path(index: int, calib_dir: str | pathlib.Path = "calib") -> pathlib.Path:
        return pathlib.Path(calib_dir) / f"cam{index}_extrinsics.npz"

    @classmethod
    def load(cls, index: int, calib_dir: str | pathlib.Path = "calib") -> "Camera":
        """Load whatever calibration exists; warn (with the command to run) for what is missing."""
        cam = cls(index)
        ip, ep = cls.intrinsics_path(index, calib_dir), cls.extrinsics_path(index, calib_dir)
        if ip.exists():
            d = np.load(ip)
            cam.K = np.asarray(d["K"], np.float64).reshape(3, 3)
            cam.dist = np.asarray(d["dist"], np.float64).reshape(-1)
            if "image_size" in d:
                cam.image_size = (int(d["image_size"][0]), int(d["image_size"][1]))
            if "rms" in d:
                cam.rms = float(d["rms"])
        else:
            log.warning("%s missing -> camera %d has no intrinsics (2D-only). Run: %s calibrate_intrinsics.py "
                        "--camera %d --board 9x6 --square 0.024", ip, index, PYTHON_HINT, index)
        if ep.exists():
            d = np.load(ep)
            cam.R = np.asarray(d["R"], np.float64).reshape(3, 3)
            cam.t = np.asarray(d["t"], np.float64).reshape(3, 1)
        else:
            log.warning("%s missing -> camera %d has no extrinsics (2D-only). Run: %s calibrate_extrinsics.py "
                        "--camera %d --marker-size 0.15", ep, index, PYTHON_HINT, index)
        return cam

    # ---- properties --------------------------------------------------------
    @property
    def has_intrinsics(self) -> bool:
        return self.K is not None

    @property
    def has_extrinsics(self) -> bool:
        return self.R is not None and self.t is not None

    @property
    def calibrated(self) -> bool:
        return self.has_intrinsics and self.has_extrinsics

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K @ [R|t] (pixel units, distortion-free)."""
        if not self.calibrated:
            raise ValueError(f"camera {self.index} is not fully calibrated (K={self.has_intrinsics}, "
                             f"R/t={self.has_extrinsics})")
        return self.K @ np.hstack([self.R, self.t])  # type: ignore[operator]

    @property
    def rvec(self) -> np.ndarray:
        return cv2.Rodrigues(self.R)[0]

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates, -R^T t, shape (3,)."""
        return (-self.R.T @ self.t).reshape(3)  # type: ignore[union-attr]

    @property
    def forward(self) -> np.ndarray:
        """Optical axis direction in world coordinates (R^T [0,0,1])."""
        return (self.R.T @ np.array([0.0, 0.0, 1.0])).reshape(3)  # type: ignore[union-attr]

    # ---- geometry ----------------------------------------------------------
    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """Raw pixels (N,2) -> undistorted pixels (N,2) in the ideal camera K (P=K)."""
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.K is None:
            raise ValueError(f"camera {self.index} has no intrinsics")
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def undistort_point(self, u: float, v: float) -> tuple[float, float]:
        p = self.undistort_points(np.array([[u, v]]))[0]
        return float(p[0]), float(p[1])

    def project_points(self, xyz: np.ndarray) -> np.ndarray:
        """World points (N,3) -> raw pixel coordinates (N,2) including distortion."""
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 1, 3)
        px, _ = cv2.projectPoints(pts, self.rvec, self.t, self.K, self.dist)
        return px.reshape(-1, 2)

    def project(self, xyz: Sequence[float]) -> tuple[float, float]:
        p = self.project_points(np.asarray(xyz, dtype=np.float64).reshape(1, 3))[0]
        return float(p[0]), float(p[1])


def camera_from_look_at(index: int, position: Sequence[float], target: Sequence[float],
                        K: np.ndarray, dist: np.ndarray | None = None,
                        up: Sequence[float] = (0.0, 0.0, 1.0)) -> Camera:
    """Synthetic camera at ``position`` looking at ``target`` (world z up); for tests/sims."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    f = np.asarray(target, dtype=np.float64).reshape(3) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=np.float64))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)                              # camera y = down
    R_cw = np.column_stack([r, d, f])               # camera axes in world coordinates
    R = R_cw.T
    t = -R @ pos.reshape(3, 1)
    return Camera(index, K=K, dist=dist, R=R, t=t)


def triangulate(camA: Camera, camB: Camera, uvA: Sequence[float], uvB: Sequence[float]) -> np.ndarray:
    """World point (3,) from RAW pixel detections in two calibrated cameras."""
    a = camA.undistort_point(float(uvA[0]), float(uvA[1]))
    b = camB.undistort_point(float(uvB[0]), float(uvB[1]))
    X = cv2.triangulatePoints(camA.P, camB.P,
                              np.array([[a[0]], [a[1]]], dtype=np.float64),
                              np.array([[b[0]], [b[1]]], dtype=np.float64))
    return (X[:3, 0] / X[3, 0]).reshape(3)


def reprojection_error(cams: Sequence[Camera], uvs: Sequence[Sequence[float]], xyz: Sequence[float]) -> float:
    """RMS pixel distance between the raw detections and ``xyz`` projected (with distortion)."""
    errs = []
    for cam, uv in zip(cams, uvs):
        pu, pv = cam.project(xyz)
        errs.append((pu - float(uv[0])) ** 2 + (pv - float(uv[1])) ** 2)
    return float(math.sqrt(sum(errs) / len(errs))) if errs else 0.0


# ======================================================================= ArUco pose


def square_marker_object_points(size_m: float) -> np.ndarray:
    """Marker-frame corners in cv2.SOLVEPNP_IPPE_SQUARE order = cv2.aruco corner order
    [TL, TR, BR, BL]: x right, y towards the TOP edge, z out of the marker face."""
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


@dataclass
class MarkerPose:
    """Pose of a square marker in the WORLD frame (via the camera extrinsics)."""

    xyz: np.ndarray          # marker centre, world, (3,)
    yaw: float               # heading of the marker's top-edge direction (+y axis) in world xy, rad
    normal: np.ndarray       # marker +z (out of the face) in world, (3,)
    rvec: np.ndarray         # marker -> camera (for cv2.drawFrameAxes), (3,)
    tvec: np.ndarray         # (3,)
    reproj_px: float         # RMS over the 4 corners
    up: bool                 # normal[2] > 0 (marker faces up)
    n_solutions: int


_IPPE_FAIL_PX = 2.0     # both IPPE candidates worse than this = the solver failed, not a bad detection
_IPPE_NUDGE = np.array([[[1e-3, -1e-3]], [[-1e-3, 1e-3]], [[1e-3, 1e-3]], [[-1e-3, -1e-3]]])


def _rms_px(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray,
            img: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return float(math.sqrt(float(np.mean(np.sum(d * d, axis=1)))))


def _ippe_square_candidates(obj: np.ndarray, img: np.ndarray, K: np.ndarray,
                            dist: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n < 1:
        return []
    return [(r, t, float(e)) for r, t, e in zip(rvecs, tvecs, np.asarray(errs, dtype=np.float64).reshape(-1))]


def solve_marker_pose(cam: Camera, corners_px: np.ndarray, size_m: float, prefer_up: bool = True,
                      refine: bool = True) -> MarkerPose:
    """World pose of a square marker from its 4 RAW pixel corners (cv2.aruco order).

    solvePnPGeneric(IPPE_SQUARE) with the camera's distortion model returns the two
    planar solutions; with ``prefer_up`` the one whose world normal has the largest
    +z component wins (the drone marker faces up), otherwise the lower reprojection
    error.  The pick is LM-refined (kept only if it does not get worse), then the
    centre/axes are mapped to the world frame.

    OpenCV's IPPE closed form breaks down when corners share EXACTLY equal image
    coordinates (marker edges exactly parallel to the image axes, e.g. yaw 0 seen
    from an azimuth-0 camera): the candidates come back with a wrong rotation.  The
    corners are therefore always nudged by 1e-3 px (sub-0.1 mm effect) before IPPE,
    and if both candidates still fit worse than 2 px SQPnP provides a single solution.
    """
    if not cam.calibrated:
        raise ValueError(f"camera {cam.index} is not fully calibrated")
    obj = square_marker_object_points(size_m)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 1, 2)
    K, dist = cam.K, cam.dist
    cands = [(r, t, _rms_px(obj, r, t, K, dist, img))
             for r, t, _ in _ippe_square_candidates(obj, img + _IPPE_NUDGE, K, dist)]
    if not cands or min(c[2] for c in cands) > _IPPE_FAIL_PX:
        ok, r, t = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise RuntimeError("solvePnP found no solution for the marker")
        cands = [(r, t, _rms_px(obj, r, t, K, dist, img))]
    best: tuple[tuple[float, float], np.ndarray, np.ndarray, float] | None = None
    for r, t, e in cands:
        nz = float((cam.R.T @ cv2.Rodrigues(r)[0][:, 2])[2])  # type: ignore[union-attr]
        key = (-nz if prefer_up else 0.0, e)
        if best is None or key < best[0]:
            best = (key, r, t, e)
    _, rvec, tvec, err = best  # type: ignore[misc]
    if refine:
        r2, t2 = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        e2 = _rms_px(obj, r2, t2, K, dist, img)
        if e2 <= err + 1e-9:                 # LM can diverge from a poor start; never accept a worse fit
            rvec, tvec, err = r2, t2, e2
    R_cm = cv2.Rodrigues(rvec)[0]
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    xyz = (cam.R.T @ (tvec - cam.t)).reshape(3)  # type: ignore[union-attr]
    top = (cam.R.T @ R_cm[:, 1]).reshape(3)  # type: ignore[union-attr]
    normal = (cam.R.T @ R_cm[:, 2]).reshape(3)  # type: ignore[union-attr]
    return MarkerPose(xyz=xyz, yaw=math.atan2(top[1], top[0]), normal=normal,
                      rvec=np.asarray(rvec, dtype=np.float64).reshape(3), tvec=tvec.reshape(3),
                      reproj_px=err, up=bool(normal[2] > 0), n_solutions=len(cands))


# ===================================================================== frame sources


def _rate(stamps: Sequence[float], now: float, stale_s: float = 1.0) -> float:
    """Frames per second over a window of timestamps; 0 when fewer than 2 or stalled."""
    if len(stamps) < 2 or now - stamps[-1] > stale_s:
        return 0.0
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 0 else 0.0


class FrameSource(threading.Thread):
    """Base for newest-frame-wins producers (CameraThread, ImageSource).

    ``latest()`` -> (frame, t) or (None, None); ``frame_event`` is set on every new
    frame; ``fps`` estimates the delivery rate; ``wait_open()`` blocks until the
    source is running or has failed (``error``).
    """

    def __init__(self, index: int, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self.index = index
        self.frame_event = threading.Event()
        self.error: str | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t: float | None = None
        self._seq = 0
        self._stamps: deque[float] = deque(maxlen=60)
        self._stop_evt = threading.Event()
        self._opened = threading.Event()
        self._ready = threading.Event()     # opened OR failed
        self._done = threading.Event()

    def _publish(self, frame: np.ndarray, t: float) -> None:
        with self._lock:
            self._frame, self._t = frame, t
            self._seq += 1
            self._stamps.append(t)
        self.frame_event.set()

    def _mark_open(self) -> None:
        self._opened.set()
        self._ready.set()

    def _finish(self) -> None:
        self._done.set()
        self._ready.set()

    def latest(self) -> tuple[np.ndarray | None, float | None]:
        with self._lock:
            return self._frame, self._t

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def fps(self) -> float:
        with self._lock:
            stamps = list(self._stamps)
        return _rate(stamps, time.perf_counter())

    @property
    def is_open(self) -> bool:
        return self._opened.is_set() and not self._done.is_set()

    def wait_open(self, timeout: float = 10.0) -> bool:
        self._ready.wait(timeout)
        return self.is_open

    def stop(self, join: bool = True, timeout: float = 3.0) -> None:
        self._stop_evt.set()
        if join and self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)


class CameraThread(FrameSource):
    """Daemon capture thread for one webcam; keeps only the newest frame.

    Survives a webcam that drops off USB (three times on 2026-10-04, laptop on battery): when the
    device does not open, or stops delivering frames for ``fail_limit`` reads, the capture is
    released and reopened every ``reopen_s``. ``is_open`` is False in between (the tracker then
    reports 2d-only and tracking_ok stays False) and True again as soon as frames resume.
    """

    def __init__(self, index: int, backend: int = cv2.CAP_DSHOW, width: int = 640, height: int = 480,
                 fps: int = 30, fail_limit: int = 90, reopen_s: float = 2.0) -> None:
        super().__init__(index, name=f"cam{index}")
        self.backend = backend
        self.width, self.height, self.req_fps = width, height, fps
        self.fail_limit = int(fail_limit)
        self.reopen_s = float(reopen_s)
        self.reopens = 0
        self.actual_size: tuple[int, int] | None = None
        self.backend_name = next((k for k, v in BACKENDS.items() if v == backend), str(backend))

    def _open(self):
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.req_fps:
            cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ignored by DSHOW, honoured by some backends
        self.actual_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cap

    def run(self) -> None:  # noqa: C901 - one loop with a reopen path
        cap = None
        try:
            while not self._stop_evt.is_set():
                cap = self._open()
                if cap is None:
                    self.error = f"camera {self.index} did not open (backend {self.backend_name})"
                    self._opened.clear()
                    self._ready.set()                 # wait_open() returns False at once
                    if self._stop_evt.wait(self.reopen_s):
                        break
                    continue
                if self._opened.is_set() or self.reopens or self.error:
                    log.info("camera %d reopened (%dx%d)", self.index, *self.actual_size)
                self.error = None
                self._mark_open()
                fails = 0
                while not self._stop_evt.is_set():
                    ok, frame = cap.read()
                    t = time.perf_counter()
                    if not ok or frame is None:
                        fails += 1
                        if fails >= self.fail_limit:
                            self.error = f"camera {self.index} stopped delivering frames; reopening"
                            break
                        time.sleep(0.01)
                        continue
                    fails = 0
                    self._publish(frame, t)
                cap.release()
                cap = None
                if self._stop_evt.is_set():
                    break
                self._opened.clear()                  # is_open False until the reopen succeeds
                self.reopens += 1
                log.warning("camera %d lost (%s); reopening every %.1f s", self.index, self.error, self.reopen_s)
                if self._stop_evt.wait(self.reopen_s):
                    break
        except Exception as exc:  # pragma: no cover - hardware path
            self.error = f"camera {self.index}: {exc!r}"
        finally:
            if cap is not None:
                cap.release()
            self._finish()


class ImageSource(FrameSource):
    """Replays a fixed image (or ``fn(t) -> image``) at ``rate_hz``; for tests and sims."""

    def __init__(self, index: int, frame: np.ndarray | Callable[[float], np.ndarray], rate_hz: float = 30.0) -> None:
        super().__init__(index, name=f"img{index}")
        self._src = frame
        self.period = 1.0 / rate_hz

    def run(self) -> None:
        self._mark_open()
        t0 = time.perf_counter()
        next_t = t0
        try:
            while not self._stop_evt.is_set():
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(next_t - now, 0.005))
                    continue
                next_t += self.period
                frame = self._src(now - t0) if callable(self._src) else self._src
                self._publish(frame, time.perf_counter())
        finally:
            self._finish()


@dataclass
class CameraView:
    """Latest per-camera result (for drawing / 2D diagnostics)."""

    index: int
    frame: np.ndarray
    t: float
    detections: dict[str, tuple[float, float, float] | None]   # colour name or "id<n>" -> (u, v, area)
    fps: float
    markers: dict[int, np.ndarray] = field(default_factory=dict)                     # ArUco id -> (4,2) raw corners
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)    # id -> (rvec, tvec), camera frame


# ============================================================================= state


@dataclass(frozen=True)
class TrackerState:
    """Thread-safe snapshot returned by ``get_state()``.

    t            perf_counter() time of the last ACCEPTED drone fix (0.0 if none yet)
    xyz          drone position (x forward, y left, z up, metres), None until the first fix;
                 midpoint of nose and tail (or nose/tail only: see yaw_ok) - held when lost
    yaw          radians, 0 = nose along +x, +pi/2 = nose along +y (left); last good value held
    tracking_ok  True only when an accepted 3D fix (= both cameras saw the drone) is
                 younger than lost_after_s (0.3 s)
    wand_xyz / wand_ok   same for the wand tip
    fps          tracker loop rate (cycles/s), ~camera rate
    latency_ms   frame arrival -> state publish, ms (exposure adds ~1 frame on top)
    yaw_ok       tail seen in the last fix (False -> position from one marker, yaw held)
    reproj_px    worst RMS reprojection error of the markers in the last accepted fix
    rejects      consecutive drone fixes rejected by the jump gate
    mode         "3d" | "2d-only" (missing camera or calibration) | "sim"
    age_s        now - t at evaluation time, None if no fix yet
    sync_ms      age difference of the two camera frames used
    t_wall       time.time() of the last accepted fix (for other processes / the UI)
    """

    t: float
    xyz: tuple[float, float, float] | None
    yaw: float | None
    tracking_ok: bool
    wand_xyz: tuple[float, float, float] | None
    wand_ok: bool
    fps: float
    latency_ms: float
    yaw_ok: bool = False
    reproj_px: float = 0.0
    rejects: int = 0
    mode: str = "3d"
    age_s: float | None = None
    sync_ms: float = 0.0
    t_wall: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (tuples -> lists) for the WebSocket UI."""
        d = {}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return d


class JumpGate:
    """Rejects a position that jumps more than ``max_jump_m`` from the previous
    accepted one; after ``reacquire_after`` consecutive rejections the next
    candidate is accepted (re-acquire)."""

    def __init__(self, max_jump_m: float, reacquire_after: int) -> None:
        self.max_jump_m = float(max_jump_m)
        self.reacquire_after = int(reacquire_after)
        self.last: np.ndarray | None = None
        self.rejects = 0

    def check(self, xyz: np.ndarray) -> bool:
        if (self.last is not None and self.rejects < self.reacquire_after
                and float(np.linalg.norm(xyz - self.last)) > self.max_jump_m):
            self.rejects += 1
            return False
        self.last = np.asarray(xyz, dtype=np.float64).reshape(3)
        self.rejects = 0
        return True

    def reset(self) -> None:
        self.last, self.rejects = None, 0


class PoseFuser:
    """Pure (thread-safe, camera-free) fusion of per-marker 3D fixes into a TrackerState.

    ``update(t, nose, tail, wand, reproj_px)`` applies the reprojection gate per
    marker, derives the drone centre + yaw, applies the jump gate (with
    re-acquire) and stores the result.  ``state(now)`` evaluates tracking_ok /
    wand_ok against ``lost_after_s``.  The reprojection gate has no re-acquire:
    a geometrically inconsistent fix is never a good fix.
    """

    def __init__(self, cfg: TrackerConfig, mode: str = "3d") -> None:
        self.cfg = cfg
        self.mode = mode
        self._lock = threading.Lock()
        self._drone_gate = JumpGate(cfg.max_jump_m, cfg.reacquire_after)
        self._wand_gate = JumpGate(cfg.max_jump_m, cfg.reacquire_after)
        self._t_fix = 0.0
        self._t_wall = 0.0
        self._xyz: np.ndarray | None = None
        self._yaw: float | None = None
        self._yaw_ok = False
        self._reproj = 0.0
        self._sep: float | None = None          # EMA of the nose-tail distance, metres
        self._t_wand = 0.0
        self._wand: np.ndarray | None = None
        self._fps = 0.0
        self._latency_ms = 0.0
        self._sync_ms = 0.0
        self.rejects_total = 0

    @staticmethod
    def _xyz_or_none(p: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
        if p is None:
            return None
        a = np.asarray(p, dtype=np.float64).reshape(3)
        return None if not np.all(np.isfinite(a)) else a

    def update(self, t: float, nose: Sequence[float] | None = None, tail: Sequence[float] | None = None,
               wand: Sequence[float] | None = None, reproj_px: Mapping[str, float] | None = None, *,
               fps: float = 0.0, latency_ms: float = 0.0, sync_ms: float = 0.0) -> TrackerState:
        rp = dict(reproj_px or {})
        lim = self.cfg.max_reproj_px

        def gated(name: str, p: Sequence[float] | None) -> np.ndarray | None:
            a = self._xyz_or_none(p)
            if a is None:
                return None
            if rp.get(name, 0.0) > lim:
                self.rejects_total += 1
                return None
            return a

        n, tl, w = gated("nose", nose), gated("tail", tail), gated("wand", wand)
        with self._lock:
            center: np.ndarray | None = None
            yaw: float | None = None
            yaw_ok = False
            worst = 0.0
            if n is not None and tl is not None:
                d = n - tl
                if np.hypot(d[0], d[1]) > 1e-6:
                    yaw, yaw_ok = math.atan2(d[1], d[0]), True
                center = 0.5 * (n + tl)
                sep = float(np.linalg.norm(d))
                self._sep = sep if self._sep is None else 0.9 * self._sep + 0.1 * sep
                worst = max(rp.get("nose", 0.0), rp.get("tail", 0.0))
            elif n is not None or tl is not None:
                p = n if n is not None else tl
                sign = -0.5 if n is not None else 0.5        # centre is behind the nose / ahead of the tail
                if self._yaw is not None and self._sep is not None:
                    center = p + sign * self._sep * np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
                else:
                    center = p
                worst = rp.get("nose" if n is not None else "tail", 0.0)
            return self._commit_locked(t, center, yaw, yaw_ok, worst, w, fps, latency_ms, sync_ms)

    def update_pose(self, t: float, xyz: Sequence[float] | None = None, yaw: float | None = None,
                    wand: Sequence[float] | None = None, reproj_px: Mapping[str, float] | None = None, *,
                    fps: float = 0.0, latency_ms: float = 0.0, sync_ms: float = 0.0) -> TrackerState:
        """Direct drone pose input (ArucoTracker): centre + yaw, optional wand position.

        ``reproj_px`` keys: "drone", "wand".  Same gates as :meth:`update`."""
        rp = dict(reproj_px or {})
        lim = self.cfg.max_reproj_px
        center = self._xyz_or_none(xyz)
        if center is not None and rp.get("drone", 0.0) > lim:
            self.rejects_total += 1
            center = None
        w = self._xyz_or_none(wand)
        if w is not None and rp.get("wand", 0.0) > lim:
            self.rejects_total += 1
            w = None
        yaw_ok = center is not None and yaw is not None and math.isfinite(yaw)
        with self._lock:
            return self._commit_locked(t, center, yaw if yaw_ok else None, yaw_ok, rp.get("drone", 0.0), w,
                                       fps, latency_ms, sync_ms)

    def _commit_locked(self, t: float, center: np.ndarray | None, yaw: float | None, yaw_ok: bool, worst: float,
                       wand: np.ndarray | None, fps: float, latency_ms: float, sync_ms: float) -> TrackerState:
        if center is not None:
            if self._drone_gate.check(center):
                self._t_fix, self._t_wall = t, time.time()
                self._xyz, self._reproj, self._yaw_ok = center, worst, yaw_ok
                if yaw_ok:
                    self._yaw = yaw
            else:
                self.rejects_total += 1
        if wand is not None and self._wand_gate.check(wand):
            self._t_wand, self._wand = t, wand
        self._fps, self._latency_ms, self._sync_ms = fps, latency_ms, sync_ms
        return self._state_locked(t)

    def reset(self) -> None:
        with self._lock:
            self._drone_gate.reset()
            self._wand_gate.reset()
            self._xyz = self._wand = None
            self._yaw, self._yaw_ok, self._sep = None, False, None
            self._t_fix = self._t_wand = 0.0

    def state(self, now: float | None = None) -> TrackerState:
        with self._lock:
            return self._state_locked(time.perf_counter() if now is None else now)

    def _state_locked(self, now: float) -> TrackerState:
        have = self._xyz is not None
        age = (now - self._t_fix) if have else None
        ok = bool(have and age is not None and age <= self.cfg.lost_after_s and self.mode != "2d-only")
        wand_ok = bool(self._wand is not None and (now - self._t_wand) <= self.cfg.lost_after_s
                       and self.mode != "2d-only")
        return TrackerState(
            t=self._t_fix,
            xyz=tuple(float(v) for v in self._xyz) if have else None,  # type: ignore[arg-type]
            yaw=self._yaw,
            tracking_ok=ok,
            wand_xyz=tuple(float(v) for v in self._wand) if self._wand is not None else None,  # type: ignore[arg-type]
            wand_ok=wand_ok,
            fps=self._fps,
            latency_ms=self._latency_ms,
            yaw_ok=self._yaw_ok,
            reproj_px=self._reproj,
            rejects=self._drone_gate.rejects,
            mode=self.mode,
            age_s=age,
            sync_ms=self._sync_ms,
            t_wall=self._t_wall,
        )


# =========================================================================== tracker


class _BaseTracker:
    """Shared lifecycle of Tracker (two cameras, colours) and ArucoTracker (one camera).

    Subclasses provide ``_make_sources()``, ``_update_mode()`` and ``_cycle()``.
    """

    GOOD_MODE = "3d"

    def __init__(self, cfg: TrackerConfig, cameras: Sequence[Camera], sources: Sequence[FrameSource] | None) -> None:
        self.cfg = cfg
        self.cameras: list[Camera] = list(cameras)
        self.sources: list[FrameSource] = list(sources) if sources is not None else []
        self.fuser = PoseFuser(self.cfg, mode="2d-only")
        self.mode_reason = "not started"
        self._views: list[CameraView | None] = [None] * len(self.cameras)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycle_stamps: deque[float] = deque(maxlen=60)
        self.stats: dict[str, Any] = {"cycles": 0, "fixes": 0}

    # ---- subclass hooks ----------------------------------------------------
    def _make_sources(self) -> list[FrameSource]:
        raise NotImplementedError

    def _update_mode(self, force_log: bool = False) -> None:
        raise NotImplementedError

    def _cycle(self) -> None:
        raise NotImplementedError

    # ---- lifecycle ---------------------------------------------------------
    def start(self, open_timeout: float = 10.0) -> "_BaseTracker":
        if not self.sources:
            self.sources = self._make_sources()
        for src in self.sources:
            if not src.is_alive() and not src._done.is_set():
                src.start()
        for src in self.sources:
            if src.wait_open(open_timeout):
                size = getattr(src, "actual_size", None)
                log.info("camera %d open%s", src.index, f" ({size[0]}x{size[1]})" if size else "")
            else:
                log.error("camera %d unavailable: %s", src.index, src.error or "timeout while opening")
        self._update_mode(force_log=True)
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="tracker", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(3.0)
        for src in self.sources:
            src.stop()

    def __enter__(self) -> "_BaseTracker":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---- public state ------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.fuser.mode

    def get_state(self, now: float | None = None) -> TrackerState:
        return self.fuser.state(now)

    def get_views(self) -> list[CameraView | None]:
        with self._lock:
            return list(self._views)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.stats))

    @property
    def fps(self) -> float:
        with self._lock:
            stamps = list(self._cycle_stamps)
        return _rate(stamps, time.perf_counter())

    # ---- internals ---------------------------------------------------------
    def _set_mode(self, reasons: Sequence[str], force_log: bool = False) -> None:
        mode = self.GOOD_MODE if not reasons else "2d-only"
        reason = "; ".join(reasons)
        if mode != self.fuser.mode or force_log or reason != self.mode_reason:
            if mode == self.GOOD_MODE:
                log.info("tracker mode %s: camera(s) open and calibrated", mode)
            else:
                log.warning("tracker mode 2d-only (no 3D pose, tracking_ok stays False): %s", reason)
        self.fuser.mode, self.mode_reason = mode, reason

    def _camera_reasons(self, cameras: Sequence[Camera], sources: Sequence[FrameSource]) -> list[str]:
        reasons = []
        for cam in cameras:
            if not cam.has_intrinsics:
                reasons.append(f"cam{cam.index}: no intrinsics")
            elif not cam.has_extrinsics:
                reasons.append(f"cam{cam.index}: no extrinsics")
        for src in sources:
            if not src.is_open:
                reasons.append(f"cam{src.index}: not open ({src.error or 'no frames'})")
        return reasons

    def _record_cycle(self, now: float) -> float:
        with self._lock:
            self._cycle_stamps.append(now)
            return _rate(list(self._cycle_stamps), now)

    def _master(self) -> FrameSource | None:
        for src in self.sources:
            if src.is_open:
                return src
        return None

    def _loop(self) -> None:
        last_mode_check = 0.0
        while not self._stop_evt.is_set():
            master = self._master()
            if master is None:
                time.sleep(0.1)
                now = time.perf_counter()
                if now - last_mode_check > 1.0:
                    self._update_mode()
                    last_mode_check = now
                continue
            if not master.frame_event.wait(0.1):
                self._update_mode()
                continue
            master.frame_event.clear()
            now = time.perf_counter()
            if now - last_mode_check > 1.0:
                self._update_mode()
                last_mode_check = now
            self._cycle()


class Tracker(_BaseTracker):
    """Two-camera colour-marker tracker: camera threads + detection + triangulation loop.

    ``Tracker(cfg).start()`` opens the cameras from ``cfg.camera_indices``;
    pass ``cameras=`` / ``sources=`` to inject calibration and frame producers
    (tests, synthetic two-camera setups).  Works with ONE camera or no
    calibration in "2d-only" mode: detections are still produced per camera and
    ``tracking_ok`` stays False.
    """

    GOOD_MODE = "3d"

    def __init__(self, config: TrackerConfig | None = None, *, cameras: Sequence[Camera] | None = None,
                 sources: Sequence[FrameSource] | None = None) -> None:
        cfg = config if config is not None else TrackerConfig.load()
        cams = list(cameras) if cameras is not None else [Camera.load(i, cfg.calib_dir) for i in cfg.camera_indices]
        super().__init__(cfg, cams, sources)
        self.detectors: dict[str, MarkerDetector] = {
            name: MarkerDetector.from_range(r, self.cfg) for name, r in self.cfg.colors.items()}
        self.stats.update({"pairs": 0, "detections": {c.index: {n: 0 for n in self.detectors} for c in self.cameras}})

    def _make_sources(self) -> list[FrameSource]:
        return [CameraThread(i, self.cfg.backend_id, self.cfg.width, self.cfg.height, self.cfg.fps)
                for i in self.cfg.camera_indices]

    def _update_mode(self, force_log: bool = False) -> None:
        reasons = []
        if len(self.cameras) < 2 or len(self.sources) < 2:
            reasons.append("fewer than two cameras configured")
        reasons += self._camera_reasons(self.cameras[:2], self.sources[:2])
        self._set_mode(reasons, force_log)

    def _cycle(self) -> None:
        views: list[CameraView | None] = []
        for cam, src in zip(self.cameras, self.sources):
            frame, t = src.latest()
            if frame is None or t is None:
                views.append(None)
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dets = {name: det.detect(hsv) for name, det in self.detectors.items()}
            views.append(CameraView(cam.index, frame, t, dets, src.fps))

        fixes: dict[str, np.ndarray] = {}
        reproj: dict[str, float] = {}
        sync_ms = 0.0
        paired = False
        if self.fuser.mode == "3d" and len(views) >= 2 and views[0] is not None and views[1] is not None:
            vA, vB = views[0], views[1]
            sync_ms = abs(vA.t - vB.t) * 1000.0
            if sync_ms <= self.cfg.max_sync_s * 1000.0:
                paired = True
                camA, camB = self.cameras[0], self.cameras[1]
                for name in self.detectors:
                    dA, dB = vA.detections.get(name), vB.detections.get(name)
                    if dA is None or dB is None:
                        continue
                    uvA, uvB = dA[:2], dB[:2]
                    xyz = triangulate(camA, camB, uvA, uvB)
                    if not np.all(np.isfinite(xyz)):
                        continue
                    fixes[name] = xyz
                    reproj[name] = reprojection_error((camA, camB), (uvA, uvB), xyz)

        stamps = [v.t for v in views if v is not None]
        now = time.perf_counter()
        t_fix = max(stamps) if stamps else now
        latency_ms = (now - min(stamps)) * 1000.0 if stamps else 0.0
        fps = self._record_cycle(now)
        state = self.fuser.update(t_fix, fixes.get("nose"), fixes.get("tail"), fixes.get("wand"), reproj,
                                  fps=fps, latency_ms=latency_ms, sync_ms=sync_ms)
        with self._lock:
            self._views = views
            st = self.stats
            st["cycles"] += 1
            st["pairs"] += int(paired)
            st["fixes"] += int(state.t == t_fix and state.xyz is not None and ("nose" in fixes or "tail" in fixes))
            for v in views:
                if v is not None:
                    for name, d in v.detections.items():
                        if d is not None:
                            st["detections"][v.index][name] += 1


class ArucoTracker(_BaseTracker):
    """Single-camera pose tracker: ArUco marker on top of the drone (see module docstring).

    Same public API as Tracker (``start/stop/get_state/get_views/get_stats``,
    ``mode`` == "aruco" when the camera is open and calibrated, else "2d-only"
    with the reason in ``mode_reason``; in 2d-only mode the detected marker ids
    still appear in ``get_views()[0].detections`` as ``"id<n>"``).
    """

    GOOD_MODE = "aruco"

    def __init__(self, config: TrackerConfig | None = None, *, camera: Camera | None = None,
                 source: FrameSource | None = None) -> None:
        cfg = config if config is not None else TrackerConfig.load()
        cam = camera if camera is not None else Camera.load(cfg.aruco_camera, cfg.calib_dir)
        super().__init__(cfg, [cam], [source] if source is not None else None)
        if cfg.aruco_dict not in ARUCO_DICTS:
            raise ValueError(f"unknown ArUco dictionary {cfg.aruco_dict!r}")
        params = cv2.aruco.DetectorParameters()
        refine = f"CORNER_REFINE_{cfg.aruco_corner_refine.upper()}"
        if not hasattr(cv2.aruco, refine):
            raise ValueError(f"aruco_corner_refine must be NONE, SUBPIX, CONTOUR or APRILTAG, got {cfg.aruco_corner_refine!r}")
        params.cornerRefinementMethod = getattr(cv2.aruco, refine)   # SUBPIX halves the corner error on small markers
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[cfg.aruco_dict]), params)
        self.stats.update({"seen": {}, "wand_fixes": 0})

    @property
    def camera(self) -> Camera:
        return self.cameras[0]

    def _make_sources(self) -> list[FrameSource]:
        return [CameraThread(self.cfg.aruco_camera, self.cfg.backend_id, self.cfg.width, self.cfg.height, self.cfg.fps)]

    def _update_mode(self, force_log: bool = False) -> None:
        self._set_mode(self._camera_reasons(self.cameras[:1], self.sources[:1]), force_log)

    def _cycle(self) -> None:
        src = self.sources[0]
        frame, t = src.latest()
        if frame is None or t is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rej = self.detector.detectMarkers(gray)
        markers: dict[int, np.ndarray] = {}
        dets: dict[str, tuple[float, float, float] | None] = {}
        if ids is not None:
            for c, i in zip(corners, ids.reshape(-1)):
                pts = np.asarray(c, dtype=np.float64).reshape(4, 2)
                markers[int(i)] = pts
                ctr = pts.mean(axis=0)
                dets[f"id{int(i)}"] = (float(ctr[0]), float(ctr[1]), float(cv2.contourArea(pts.astype(np.float32))))
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        drone: MarkerPose | None = None
        wand_xyz: np.ndarray | None = None
        reproj: dict[str, float] = {}
        if self.fuser.mode == "aruco":
            cfg, cam = self.cfg, self.camera
            if cfg.drone_marker_id in markers:
                try:
                    drone = solve_marker_pose(cam, markers[cfg.drone_marker_id], cfg.drone_marker_size_m, prefer_up=True)
                    poses[cfg.drone_marker_id] = (drone.rvec, drone.tvec)
                    reproj["drone"] = drone.reproj_px
                except (cv2.error, RuntimeError) as exc:
                    log.debug("drone marker pose failed: %s", exc)
            if cfg.wand_marker_id in markers and cfg.wand_marker_id != cfg.drone_marker_id:
                try:
                    wp = solve_marker_pose(cam, markers[cfg.wand_marker_id], cfg.wand_marker_size_m, prefer_up=False)
                    poses[cfg.wand_marker_id] = (wp.rvec, wp.tvec)
                    wand_xyz, reproj["wand"] = wp.xyz, wp.reproj_px
                except (cv2.error, RuntimeError) as exc:
                    log.debug("wand marker pose failed: %s", exc)
        now = time.perf_counter()
        fps = self._record_cycle(now)
        state = self.fuser.update_pose(t, drone.xyz if drone else None, drone.yaw if drone else None, wand_xyz, reproj,
                                       fps=fps, latency_ms=(now - t) * 1000.0)
        view = CameraView(self.camera.index, frame, t, dets, src.fps, markers=markers, poses=poses)
        with self._lock:
            self._views = [view]
            st = self.stats
            st["cycles"] += 1
            st["fixes"] += int(drone is not None and state.t == t and state.xyz is not None)
            st["wand_fixes"] += int(wand_xyz is not None and state.wand_ok)
            for mid in markers:
                st["seen"][str(mid)] = st["seen"].get(str(mid), 0) + 1


# ============================================================================== sim


class SimTracker:
    """Replays a ``paths.demo_paths()`` path (or any ``paths.Path``) with noise through
    the same PoseFuser, so feed/UI code can use ``get_state()`` without cameras.

    ``dropout_every_s`` > 0 simulates tracking loss for ``dropout_s`` every so often
    (tracking_ok goes False after lost_after_s), to exercise the land-on-lost rule.
    ``yaw=None`` follows the path tangent; default 0 (nose along +x, as flown).
    """

    def __init__(self, path: Any = "exit_a", speed: float | None = None, *, wand_path: Any = "spiral",
                 noise_m: float = 0.005, yaw: float | None = 0.0, yaw_noise_rad: float = 0.02,
                 rate_hz: float = 30.0, dropout_every_s: float = 0.0, dropout_s: float = 0.5,
                 loop: bool = True, seed: int = 0, config: TrackerConfig | None = None) -> None:
        import paths as P  # local import: scipy-backed, not needed by the real tracker

        demos = P.demo_paths()
        self.path = demos[path] if isinstance(path, str) else path
        self.wand_path = demos[wand_path] if isinstance(wand_path, str) else wand_path
        self.speed_factor = 1.0 if speed is None else float(speed) / max(self.path.speed, 1e-9)
        self.cfg = config if config is not None else TrackerConfig()
        self.fuser = PoseFuser(self.cfg, mode="sim")
        self.noise_m, self.yaw, self.yaw_noise = noise_m, yaw, yaw_noise_rad
        self.period = 1.0 / rate_hz
        self.dropout_every_s, self.dropout_s, self.loop = dropout_every_s, dropout_s, loop
        self.rng = np.random.default_rng(seed)
        self.cameras: list[Camera] = []
        self.sources: list[FrameSource] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self.half_sep = 0.05   # nose/tail half distance, metres

    @property
    def mode(self) -> str:
        return "sim"

    def start(self, open_timeout: float = 0.0) -> "SimTracker":
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="simtracker", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def __enter__(self) -> "SimTracker":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def get_state(self, now: float | None = None) -> TrackerState:
        return self.fuser.state(now)

    def get_views(self) -> list[CameraView | None]:
        return []

    def get_stats(self) -> dict[str, Any]:
        return {}

    def _path_time(self, path: Any, elapsed: float) -> float:
        dur = max(path.duration, 1e-9)
        t = elapsed * self.speed_factor
        return (t % dur) if self.loop else min(t, dur)

    def _loop(self) -> None:
        t0 = time.perf_counter()
        next_t = t0
        while not self._stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(next_t - now, 0.005))
                continue
            next_t += self.period
            el = now - t0
            dropout = self.dropout_every_s > 0 and (el % self.dropout_every_s) < self.dropout_s
            if dropout:
                self.fuser.update(now, None, None, None, fps=1.0 / self.period, latency_ms=0.0)
                continue
            tp = self._path_time(self.path, el)
            xyz = self.path.position_at(tp) + self.rng.normal(0.0, self.noise_m, 3)
            if self.yaw is None:
                ahead = self.path.position_at(min(tp + 0.2, self.path.duration))
                d = ahead - xyz
                yaw = math.atan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 1e-4 else 0.0
            else:
                yaw = self.yaw
            yaw += self.rng.normal(0.0, self.yaw_noise)
            half = self.half_sep * np.array([math.cos(yaw), math.sin(yaw), 0.0])
            wand = None
            if self.wand_path is not None:
                tw = self._path_time(self.wand_path, el)
                wand = self.wand_path.position_at(tw) + self.rng.normal(0.0, self.noise_m, 3)
            self.fuser.update(now, xyz + half, xyz - half, wand, fps=1.0 / self.period,
                              latency_ms=float(self.rng.uniform(5.0, 15.0)))


# ============================================================================== CLI

_DRAW_COLORS = {"nose": (0, 0, 255), "tail": (255, 0, 0), "wand": (0, 255, 0)}


def _fmt_xyz(p: Sequence[float] | None) -> str:
    return "--" if p is None else f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"


def format_state(state: TrackerState, views: Sequence[CameraView | None], elapsed: float,
                 indices: Sequence[int] | None = None) -> str:
    yaw = "--" if state.yaw is None else f"{math.degrees(state.yaw):+.1f}deg{'' if state.yaw_ok else '(held)'}"
    s = (f"[{elapsed:6.1f}s] {state.mode:7s} ok={str(state.tracking_ok):5s} xyz={_fmt_xyz(state.xyz)} yaw={yaw} "
         f"rep={state.reproj_px:.1f}px rej={state.rejects} wand={_fmt_xyz(state.wand_xyz)}{' ok' if state.wand_ok else ''} "
         f"| loop {state.fps:4.1f}fps lat {state.latency_ms:4.0f}ms sync {state.sync_ms:3.0f}ms")
    for i, v in enumerate(views):
        if v is None:
            idx = indices[i] if indices is not None and i < len(indices) else "?"
            s += f" | cam{idx}: no frame"
            continue
        dets = " ".join(f"{n}={'--' if d is None else f'({d[0]:.0f},{d[1]:.0f},a{d[2]:.0f})'}"
                        for n, d in v.detections.items())
        s += f" | cam{v.index} {v.fps:4.1f}fps {dets}"
    return s


def draw_view(view: CameraView, state: TrackerState, camera: Camera | None = None, axis_len: float = 0.05) -> np.ndarray:
    """Frame with colour detections, ArUco outlines (+ axes when ``camera`` is calibrated) and the state."""
    img = view.frame.copy()
    for mid, pts in view.markers.items():
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, (0, 255, 255), 2)
        tl = tuple(int(round(x)) for x in pts[0])
        cv2.circle(img, tl, 5, (0, 0, 255), -1)                    # TL corner = start of the TOP edge
        cv2.line(img, tl, tuple(int(round(x)) for x in pts[1]), (0, 0, 255), 3)   # TOP edge in red (nose side)
        cv2.putText(img, f"id{mid}", (int(pts[:, 0].max()) + 4, int(pts[:, 1].min())), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        if camera is not None and camera.has_intrinsics and mid in view.poses:
            rvec, tvec = view.poses[mid]
            cv2.drawFrameAxes(img, camera.K, camera.dist, rvec, tvec, axis_len, 2)
    for name, d in view.detections.items():
        if d is None or name.startswith("id"):
            continue
        c = _DRAW_COLORS.get(name, (255, 255, 255))
        u, v = int(round(d[0])), int(round(d[1]))
        r = max(6, int(math.sqrt(max(d[2], 1.0)) / 2))
        cv2.circle(img, (u, v), r, c, 2)
        cv2.drawMarker(img, (u, v), c, cv2.MARKER_CROSS, 10, 1)
        cv2.putText(img, name, (u + r + 2, v - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    txt1 = f"cam{view.index} {view.fps:.1f} fps   mode {state.mode}   tracking {'OK' if state.tracking_ok else 'LOST'}"
    txt2 = (f"xyz {_fmt_xyz(state.xyz)}  yaw {'--' if state.yaw is None else f'{math.degrees(state.yaw):+.0f}deg'}"
            f"  reproj {state.reproj_px:.1f}px  lat {state.latency_ms:.0f}ms")
    for i, txt in enumerate((txt1, txt2)):
        cv2.putText(img, txt, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if state.tracking_ok else (0, 200, 255), 1, cv2.LINE_AA)
    return img


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--show", action="store_true", help="window per camera with detections drawn")
    mode.add_argument("--headless", action="store_true", help="no windows; print the state at --rate Hz")
    mode.add_argument("--sim", action="store_true", help="replay a demo path with noise instead of cameras")
    ap.add_argument("--aruco", action="store_true",
                    help="single-camera ArUco pose mode (marker id drone_marker_id on top of the drone)")
    ap.add_argument("--camera", type=int, default=None, help="--aruco: camera index (default config aruco_camera)")
    ap.add_argument("--seconds", type=float, default=None, help="run time (default: until q / Ctrl-C; 10 for --headless)")
    ap.add_argument("--rate", type=float, default=5.0, help="state print rate, Hz")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="TrackerConfig JSON")
    ap.add_argument("--cameras", type=int, nargs="+", metavar="IDX", help="two-camera mode: override camera indices")
    ap.add_argument("--backend", choices=sorted(BACKENDS), help="override capture backend")
    ap.add_argument("--path", default="exit_a", help="--sim: demo path name (spiral, square, exit_a, exit_b)")
    ap.add_argument("--dropout", type=float, default=0.0, help="--sim: simulate 0.5 s tracking loss every N s")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = TrackerConfig.load(args.config)
    if args.cameras:
        cfg.camera_indices = tuple(args.cameras)  # type: ignore[assignment]
    if args.backend:
        cfg.backend = args.backend

    if args.sim and args.aruco:
        ap.error("--sim and --aruco are exclusive")
    seconds = args.seconds if args.seconds is not None else (10.0 if args.headless else None)
    tracker: _BaseTracker | SimTracker
    if args.sim:
        tracker = SimTracker(args.path, config=cfg, dropout_every_s=args.dropout)
        print(f"== sim tracker: path {tracker.path!r}")
    elif args.aruco:
        if args.camera is not None:
            cfg.aruco_camera = args.camera
        tracker = ArucoTracker(cfg)
        print(f"== aruco tracker: camera {cfg.aruco_camera} backend {cfg.backend} {cfg.width}x{cfg.height}  "
              f"{cfg.aruco_dict}: drone id {cfg.drone_marker_id} ({cfg.drone_marker_size_m * 100:.0f} cm, top edge = nose), "
              f"wand id {cfg.wand_marker_id} ({cfg.wand_marker_size_m * 100:.0f} cm)  (config {args.config})")
    else:
        tracker = Tracker(cfg)
        print(f"== tracker: cameras {cfg.camera_indices} backend {cfg.backend} {cfg.width}x{cfg.height} "
              f"colours {sorted(cfg.colors)}  (config {args.config})")
    show = args.show and not args.sim
    cams_by_index = {c.index: c for c in tracker.cameras}
    axis_len = cfg.drone_marker_size_m * 0.75

    tracker.start()
    indices = [c.index for c in tracker.cameras]
    t0 = time.perf_counter()
    next_print = t0
    rc = 0
    try:
        while True:
            now = time.perf_counter()
            if seconds is not None and now - t0 >= seconds:
                break
            state = tracker.get_state()
            views = tracker.get_views()
            if now >= next_print:
                print(format_state(state, views, now - t0, indices), flush=True)
                next_print += 1.0 / args.rate
            if show:
                for v in views:
                    if v is not None:
                        cv2.imshow(f"cam{v.index}", draw_view(v, state, cams_by_index.get(v.index), axis_len))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        if show:
            cv2.destroyAllWindows()
        if isinstance(tracker, _BaseTracker):
            st = tracker.get_stats()
            el = max(time.perf_counter() - t0, 1e-9)
            extra = f"{st['pairs']} synced pairs, " if "pairs" in st else ""
            print(f"== done: mode {tracker.mode} ({tracker.mode_reason or 'ok'}); {st['cycles']} cycles in {el:.1f}s "
                  f"= {st['cycles'] / el:.1f} Hz, {extra}{st['fixes']} accepted drone fixes")
            for idx, dets in st.get("detections", {}).items():
                print(f"   cam{idx} detections: " + ", ".join(f"{n} {c}/{st['cycles']}" for n, c in dets.items()))
            if "seen" in st:
                seen = ", ".join(f"id{k} {v}/{st['cycles']}" for k, v in sorted(st["seen"].items(), key=lambda kv: int(kv[0])))
                print(f"   markers seen: {seen or 'none'}; wand fixes {st.get('wand_fixes', 0)}")
            if tracker.mode == "2d-only":
                print("   (2D-only: fix the reasons above to get 3D fixes and tracking_ok)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
