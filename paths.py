"""Path engine for Pathcaster: record a 3D point stream -> clean it -> waypoints -> JSON.

One engine, two modes:

* **Guide mode** records the drone's own tracked position while a human carries it
  along an exit route.  The route is flown at TRUE scale, so it is only *clamped*
  into the geofence (never rescaled).
* **Spell mode** records a wand tip drawing a shape in the air.  The shape is
  uniformly scaled (no distortion) and translated so it fits inside the geofence.

Both modes feed the same pipeline::

    smooth (Savitzky-Golay) -> resample by arc length -> fit/clamp into the box
    -> re-time at constant speed -> save as JSON -> sparse go_to waypoints

Coordinate conventions (same as ``tracker.py``): world frame, metres,
``x`` forward, ``y`` left, ``z`` up.  Times are seconds.

Usage::

    from paths import PathRecorder, clean_path, save_path, load_path, to_waypoints

    rec = PathRecorder()                       # ~30 Hz tracker stream
    for t, (x, y, z) in tracker_stream():
        rec.add(t, x, y, z)                    # drops jitter / NaN automatically

    path = clean_path(rec.points, mode="guide", name="Exit A")
    save_path(path)                            # -> paths/exit_a.json

    for wp in load_path("exit a").waypoints(speed=0.3, dt=0.4):
        cf.go_to(wp.x, wp.y, wp.z, yaw=wp.yaw, duration_s=wp.duration)

Pure numpy/scipy; no hardware imports.  Run the tests with
``python -m unittest tests.test_paths -v`` from the project root.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np

import config
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

__all__ = [
    "Box",
    "GEOFENCE",
    "PathRecorder",
    "Waypoint",
    "Path",
    "DEFAULT_SPEED",
    "DEFAULT_SPACING",
    "DEFAULT_DT",
    "MIN_GO_TO_DURATION",
    "DEFAULT_DIRECTORY",
    "FILE_FORMAT",
    "MODES",
    "smooth",
    "resample_by_distance",
    "path_length",
    "fit_to_box",
    "clamp_to_box",
    "retime",
    "to_waypoints",
    "clean_path",
    "slugify",
    "save_path",
    "load_path",
    "list_paths",
    "demo_paths",
]

FloatArray = NDArray[np.float64]

DEFAULT_SPEED: float = 0.3          # m/s, slow and steady for a 15 g payload drone
DEFAULT_SPACING: float = 0.05       # m, resampling step of a cleaned path
DEFAULT_DT: float = 0.4             # s, time between go_to waypoints (0.3-0.5 s per README.md)
MIN_GO_TO_DURATION: float = 0.1     # s, never ask the commander for a shorter move
DEFAULT_DIRECTORY: str = "paths"    # where named paths are saved
FILE_FORMAT: str = "pathcaster.path.v1"
MODES: tuple[str, ...] = ("guide", "spell")


# --------------------------------------------------------------------------- box


@dataclass(frozen=True)
class Box:
    """Axis-aligned flight volume in world coordinates (metres)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self) -> None:
        if not (self.xmin < self.xmax and self.ymin < self.ymax and self.zmin < self.zmax):
            raise ValueError(f"Box must have min < max on every axis, got {self}")

    @property
    def lower(self) -> FloatArray:
        return np.array([self.xmin, self.ymin, self.zmin], dtype=float)

    @property
    def upper(self) -> FloatArray:
        return np.array([self.xmax, self.ymax, self.zmax], dtype=float)

    @property
    def size(self) -> FloatArray:
        return self.upper - self.lower

    @property
    def center(self) -> FloatArray:
        return (self.upper + self.lower) / 2.0

    def shrink(self, margin: float) -> "Box":
        """Return the box shrunk by ``margin`` on every face."""
        lo = self.lower + margin
        hi = self.upper - margin
        if np.any(hi <= lo):
            raise ValueError(f"margin {margin} m leaves no room inside {self}")
        return Box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    def clamp(self, point: ArrayLike) -> FloatArray:
        """Clamp one point ``(3,)`` or many points ``(N, 3)`` into the box."""
        return np.clip(np.asarray(point, dtype=float), self.lower, self.upper)

    def inside(self, points: ArrayLike, tol: float = 1e-9) -> NDArray[np.bool_]:
        """Per-point boolean mask: True where the point lies inside the box."""
        pts = _as_points(points)
        return np.all((pts >= self.lower - tol) & (pts <= self.upper + tol), axis=1)

    def contains(self, points: ArrayLike, tol: float = 1e-9) -> bool:
        """True if every point of ``(3,)`` or ``(N, 3)`` lies inside the box."""
        return bool(np.all(self.inside(points, tol)))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Box":
        return cls(**{k: float(data[k]) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")})


#: Volume both cameras see: 1.5 m square footprint, 0.2 m to 1.2 m up (README.md, Flight).
GEOFENCE: Box = Box(xmin=config.GEOFENCE_X[0], xmax=config.GEOFENCE_X[1],
                    ymin=config.GEOFENCE_Y[0], ymax=config.GEOFENCE_Y[1],
                    zmin=config.GEOFENCE_Z[0], zmax=config.GEOFENCE_Z[1])   # single source: config.py


# ---------------------------------------------------------------------- helpers


def _as_points(points: ArrayLike) -> FloatArray:
    """Coerce to a float ``(N, 3)`` array; a single ``(3,)`` point becomes ``(1, 3)``."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise ValueError(f"expected (N, 3) points, got shape {pts.shape}")
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got shape {pts.shape}")
    return pts


def _segment_lengths(pts: FloatArray) -> FloatArray:
    if len(pts) < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def _cumulative_length(pts: FloatArray) -> FloatArray:
    """Arc length from the first point to each point, ``(N,)``; empty for N == 0."""
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    return np.concatenate([[0.0], np.cumsum(_segment_lengths(pts))])


def _dedupe_consecutive(pts: FloatArray, eps: float = 1e-12) -> FloatArray:
    """Drop points that coincide with their predecessor (zero-length segments)."""
    if len(pts) < 2:
        return pts.copy()
    keep = np.concatenate([[True], _segment_lengths(pts) > eps])
    return pts[keep]


def _moving_average(pts: FloatArray, window: int) -> FloatArray:
    """Centered moving average along axis 0 with edge padding (window is odd)."""
    half = window // 2
    padded = np.pad(pts, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)])


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, Boxes and paths to plain JSON types."""
    if isinstance(obj, Box):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    return obj


def path_length(points: ArrayLike) -> float:
    """Total polyline length in metres (0.0 for fewer than two points)."""
    pts = _as_points(points)
    return float(_segment_lengths(pts).sum()) if len(pts) >= 2 else 0.0


# --------------------------------------------------------------------- recorder


class PathRecorder:
    """Collects a ~30 Hz stream of tracked positions into a raw path.

    Samples closer than ``min_step`` metres to the previously *kept* sample are
    dropped (stationary jitter), as are samples with any non-finite value
    (tracking lost / NaN from the triangulator).
    """

    def __init__(self, min_step: float = 0.01) -> None:
        if min_step < 0:
            raise ValueError("min_step must be >= 0")
        self.min_step: float = float(min_step)
        self._times: list[float] = []
        self._points: list[tuple[float, float, float]] = []
        self.dropped: int = 0

    def add(self, t: float, x: float, y: float, z: float) -> bool:
        """Append one sample. Returns True if it was kept."""
        try:
            tt, xx, yy, zz = float(t), float(x), float(y), float(z)
        except (TypeError, ValueError):
            self.dropped += 1
            return False
        if not (math.isfinite(tt) and math.isfinite(xx) and math.isfinite(yy) and math.isfinite(zz)):
            self.dropped += 1
            return False
        if self._points and math.dist((xx, yy, zz), self._points[-1]) < self.min_step:
            self.dropped += 1
            return False
        self._times.append(tt)
        self._points.append((xx, yy, zz))
        return True

    def extend(self, samples: Iterable[Sequence[float]]) -> int:
        """Add many ``(t, x, y, z)`` rows. Returns how many were kept."""
        return sum(1 for s in samples if self.add(*s))

    @property
    def points(self) -> FloatArray:
        """Kept positions as an ``(N, 3)`` array (``(0, 3)`` when empty)."""
        if not self._points:
            return np.zeros((0, 3), dtype=float)
        return np.array(self._points, dtype=float)

    @property
    def times(self) -> FloatArray:
        """Timestamps of the kept positions, ``(N,)``."""
        return np.array(self._times, dtype=float)

    @property
    def last(self) -> tuple[float, float, float] | None:
        """Most recent kept position, or None."""
        return self._points[-1] if self._points else None

    def clear(self) -> None:
        self._times.clear()
        self._points.clear()
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"PathRecorder(n={len(self)}, dropped={self.dropped}, min_step={self.min_step})"


# -------------------------------------------------------------------- geometry


def smooth(points: ArrayLike, window: int = 11, polyorder: int = 3) -> FloatArray:
    """Savitzky-Golay smoothing of each axis along the path.

    The window is shrunk to an odd number <= len(points).  When that leaves too
    few points for the polynomial order, a centered moving average is used; for
    fewer than three points the input is returned unchanged.  Never raises on
    short inputs.
    """
    pts = _as_points(points)
    n = len(pts)
    if n < 3:
        return pts.copy()
    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return pts.copy()
    if 0 <= polyorder < w:
        return np.asarray(savgol_filter(pts, w, polyorder, axis=0, mode="interp"), dtype=float)
    return _moving_average(pts, w)


def resample_by_distance(points: ArrayLike, spacing: float = DEFAULT_SPACING) -> FloatArray:
    """Resample a polyline to (nearly) uniform arc-length spacing.

    Linear interpolation along the polyline.  The first and last points are
    always kept exactly; the number of segments is ``round(length / spacing)``
    (at least 1), so the actual spacing is within ``spacing / (2 * segments)``
    of the request.  Consecutive duplicate points are removed first.  Inputs
    with fewer than two distinct points are returned as-is.
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    pts = _dedupe_consecutive(_as_points(points))
    if len(pts) < 2:
        return pts
    s = _cumulative_length(pts)
    total = float(s[-1])
    n_seg = max(1, int(round(total / spacing)))
    s_new = np.linspace(0.0, total, n_seg + 1)
    out = np.column_stack([np.interp(s_new, s, pts[:, k]) for k in range(3)])
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def fit_to_box(
    points: ArrayLike,
    box: Box = GEOFENCE,
    margin: float = 0.05,
    preserve_aspect: bool = True,
) -> tuple[FloatArray, float | FloatArray, FloatArray]:
    """Scale down (never up) and translate a path so it lies inside ``box`` shrunk by ``margin``.

    With ``preserve_aspect`` one scale factor is used for x, y and z, so the
    shape is not distorted; otherwise each axis gets its own factor.  Scaling is
    about the path's bounding-box centre.  A path that already fits is only
    translated, and only as far as needed (small paths are NOT inflated).

    Returns ``(fitted, scale, offset)`` with ``fitted == scale * points + offset``;
    ``scale`` is a float (or a ``(3,)`` array when ``preserve_aspect=False``).
    """
    pts = _as_points(points)
    inner = box.shrink(margin)
    if len(pts) == 0:
        return pts.copy(), (1.0 if preserve_aspect else np.ones(3)), np.zeros(3)

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = hi - lo
    room = inner.size
    with np.errstate(divide="ignore", invalid="ignore"):
        per_axis = np.where(extent > 0, room / np.where(extent > 0, extent, 1.0), np.inf)
    per_axis = np.minimum(per_axis, 1.0)  # never inflate

    scale: float | FloatArray
    if preserve_aspect:
        scale = float(per_axis.min())
    else:
        scale = per_axis

    center = (lo + hi) / 2.0
    scaled = center + scale * (pts - center)

    s_lo, s_hi = scaled.min(axis=0), scaled.max(axis=0)
    shift = np.zeros(3)
    below = s_lo < inner.lower
    above = s_hi > inner.upper
    shift[below] = inner.lower[below] - s_lo[below]
    shift[above & ~below] = inner.upper[above & ~below] - s_hi[above & ~below]

    offset = center * (1.0 - scale) + shift
    fitted = scale * pts + offset
    return fitted, scale, offset


def clamp_to_box(points: ArrayLike, box: Box = GEOFENCE) -> FloatArray:
    """Clamp every point into ``box`` (Guide mode: true scale, no rescaling)."""
    return box.clamp(_as_points(points))


def retime(points: ArrayLike, speed: float = DEFAULT_SPEED) -> FloatArray:
    """Constant-speed timestamps from cumulative arc length; ``t[0] == 0``."""
    if speed <= 0:
        raise ValueError("speed must be > 0")
    return _cumulative_length(_as_points(points)) / float(speed)


# ------------------------------------------------------------------- waypoints


@dataclass(frozen=True)
class Waypoint:
    """One sparse ``go_to`` target for the flight layer.

    ``duration`` is the time to move here from the previous waypoint (or from the
    start, for the first one); ``t`` is the cumulative replay time at which the
    drone should arrive.  ``yaw`` is kept at 0 (nose along world +x) so the
    nose/tail markers stay visible to both cameras.
    """

    x: float
    y: float
    z: float
    t: float
    duration: float
    yaw: float = 0.0

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def to_waypoints(
    points: ArrayLike,
    speed: float = DEFAULT_SPEED,
    dt: float = DEFAULT_DT,
    *,
    start: ArrayLike | None = None,
    min_duration: float = MIN_GO_TO_DURATION,
    yaw: float = 0.0,
) -> list[Waypoint]:
    """Turn a cleaned path into sparse waypoints about ``speed * dt`` metres apart.

    Each waypoint's ``duration`` is its segment length divided by ``speed``, but
    never below ``min_duration``.  If ``start`` (the drone's hover position) is
    given, the first duration covers the distance from ``start`` to the first
    point; otherwise it is ``min_duration``.  The flight layer should still
    clamp every target into the geofence before calling ``go_to``.
    """
    if speed <= 0 or dt <= 0:
        raise ValueError("speed and dt must be > 0")
    pts = resample_by_distance(points, spacing=speed * dt)
    if len(pts) == 0:
        return []
    times = retime(pts, speed)
    durations = np.diff(times, prepend=0.0)
    if start is not None:
        start_pt = np.asarray(start, dtype=float).reshape(3)
        durations[0] = float(np.linalg.norm(pts[0] - start_pt)) / speed
    durations = np.maximum(durations, min_duration)
    arrival = np.cumsum(durations)
    return [
        Waypoint(float(p[0]), float(p[1]), float(p[2]), float(t), float(d), float(yaw))
        for p, t, d in zip(pts, arrival, durations)
    ]


# ------------------------------------------------------------------------ path


@dataclass(eq=False)
class Path:
    """A cleaned, re-timed path ready to save or fly."""

    name: str
    mode: str
    points: FloatArray
    times: FloatArray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = _as_points(self.points)
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        if len(self.times) != len(self.points):
            raise ValueError(
                f"times ({len(self.times)}) and points ({len(self.points)}) must have the same length"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def length(self) -> float:
        """Total path length in metres."""
        return path_length(self.points)

    @property
    def duration(self) -> float:
        """Replay time of the last point in seconds (0 for an empty path)."""
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def speed(self) -> float:
        """Design speed in m/s (from meta, else the default)."""
        return float(self.meta.get("speed_mps", DEFAULT_SPEED))

    def waypoints(
        self,
        speed: float | None = None,
        dt: float = DEFAULT_DT,
        *,
        start: ArrayLike | None = None,
        min_duration: float = MIN_GO_TO_DURATION,
    ) -> list[Waypoint]:
        """Sparse ``go_to`` waypoints; see :func:`to_waypoints`."""
        return to_waypoints(
            self.points,
            speed=self.speed if speed is None else speed,
            dt=dt,
            start=start,
            min_duration=min_duration,
        )

    def position_at(self, t: float) -> FloatArray:
        """Interpolated position at replay time ``t`` (clamped to the ends); for UI preview."""
        if len(self.points) == 0:
            raise ValueError("empty path")
        if len(self.points) == 1:
            return self.points[0].copy()
        return np.array([np.interp(t, self.times, self.points[:, k]) for k in range(3)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FILE_FORMAT,
            "name": self.name,
            "mode": self.mode,
            "points": self.points.tolist(),
            "times": self.times.tolist(),
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str | None = None) -> "Path":
        try:
            points = data["points"]
            times = data["times"]
        except KeyError as exc:
            raise ValueError(f"path JSON is missing field {exc}") from exc
        return cls(
            name=str(data.get("name") or name or "path"),
            mode=str(data.get("mode", "guide")),
            points=np.asarray(points, dtype=float),
            times=np.asarray(times, dtype=float),
            meta=dict(data.get("meta") or {}),
        )

    def __repr__(self) -> str:
        return (
            f"Path(name={self.name!r}, mode={self.mode!r}, n={len(self.points)}, "
            f"length={self.length:.2f} m, duration={self.duration:.1f} s)"
        )


def clean_path(
    raw_points: ArrayLike,
    mode: str = "guide",
    box: Box = GEOFENCE,
    speed: float = DEFAULT_SPEED,
    spacing: float = DEFAULT_SPACING,
    *,
    name: str = "",
    window: int = 11,
    polyorder: int = 3,
    margin: float = 0.05,
    fit: bool | None = None,
) -> Path:
    """One-call pipeline: smooth -> resample -> fit (spell) / clamp (guide) -> retime.

    ``mode="spell"`` scales the shape uniformly into ``box`` shrunk by ``margin``;
    ``mode="guide"`` keeps true scale and only clamps points into ``box``.
    ``fit`` overrides that default (``fit=False`` on a spell keeps its real size).
    Non-finite rows are dropped.  Raises ``ValueError`` for fewer than two
    distinct finite points.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    raw = _as_points(raw_points)
    source_samples = len(raw)
    finite = np.all(np.isfinite(raw), axis=1)
    raw = raw[finite]
    if len(_dedupe_consecutive(raw)) < 2:
        raise ValueError("clean_path needs at least two distinct finite points")

    do_fit = (mode == "spell") if fit is None else bool(fit)

    smoothed = smooth(raw, window=window, polyorder=polyorder)
    resampled = resample_by_distance(smoothed, spacing=spacing)

    meta: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_samples": int(source_samples),
        "dropped_nonfinite": int(source_samples - int(finite.sum())),
        "smooth_window": int(window),
        "polyorder": int(polyorder),
        "spacing_m": float(spacing),
        "speed_mps": float(speed),
        "box": box.to_dict(),
        "fitted": bool(do_fit),
    }

    if do_fit:
        placed, scale, offset = fit_to_box(resampled, box=box, margin=margin)
        meta["margin_m"] = float(margin)
        meta["scale"] = _jsonable(scale)
        meta["offset"] = offset.tolist()
    else:
        placed = clamp_to_box(resampled, box=box)
        moved = np.any(np.abs(placed - resampled) > 1e-12, axis=1)
        meta["clamped_points"] = int(moved.sum())

    if not np.array_equal(placed, resampled):
        # scaling/clamping changed the geometry: restore uniform spacing (also
        # removes points that collapsed onto a wall when clamped).
        placed = resample_by_distance(placed, spacing=spacing)

    times = retime(placed, speed=speed)
    meta["length_m"] = path_length(placed)
    meta["duration_s"] = float(times[-1]) if len(times) else 0.0
    meta["n_points"] = int(len(placed))
    return Path(name=name, mode=mode, points=placed, times=times, meta=meta)


# --------------------------------------------------------------------- storage


def slugify(name: str) -> str:
    """Lower-case ``[a-z0-9_]`` filename stem (``"Exit A" -> "exit_a"``)."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "path"


def _path_file(name: str, directory: str | pathlib.Path) -> pathlib.Path:
    stem = str(name)
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    return pathlib.Path(directory) / f"{slugify(stem)}.json"


def save_path(path: Path, directory: str | pathlib.Path = DEFAULT_DIRECTORY) -> str:
    """Write ``path`` as ``<directory>/<slug>.json`` and return that file path.

    The slug comes from ``path.name`` (``"path"`` if the name is empty).
    """
    target = _path_file(path.name, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(path.to_dict(), fh, indent=1)
        fh.write("\n")
    return str(target)


def load_path(name: str, directory: str | pathlib.Path = DEFAULT_DIRECTORY) -> Path:
    """Load a saved path by name (slugified) or by its ``.json`` filename."""
    target = _path_file(name, directory)
    if not target.is_file():
        available = ", ".join(list_paths(directory)) or "none"
        raise FileNotFoundError(f"no saved path {target} (available: {available})")
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Path.from_dict(data, name=target.stem)


def list_paths(directory: str | pathlib.Path = DEFAULT_DIRECTORY) -> list[str]:
    """Sorted names (file stems) of the saved paths in ``directory``."""
    folder = pathlib.Path(directory)
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json") if p.is_file())


# ------------------------------------------------------------------------ demo


def _polyline(vertices: ArrayLike, step: float = 0.02) -> FloatArray:
    """Densely sample a polyline through ``vertices`` (used to fake a recording)."""
    return resample_by_distance(np.asarray(vertices, dtype=float), spacing=step)


def demo_paths(speed: float = DEFAULT_SPEED, box: Box = GEOFENCE) -> dict[str, Path]:
    """Synthetic, already-cleaned paths for UI/replay testing without hardware.

    * ``spiral``: 2 turns of radius 0.4 m rising from z=0.4 to 1.0 m (spell)
    * ``square``: 1 m sides at z=0.7 m, starting at the (-0.5, -0.5) corner (spell)
    * ``exit_a``: L-route 1.2 m forward (+x) then 0.8 m left (+y) at z=0.6 m (guide)
    * ``exit_b``: mirror of exit_a, turning right (-y) (guide)
    """
    n = 240
    ang = np.linspace(0.0, 2.0 * 2.0 * np.pi, n)
    spiral = np.column_stack([0.4 * np.cos(ang), 0.4 * np.sin(ang), np.linspace(0.4, 1.0, n)])

    square = _polyline(
        [[-0.5, -0.5, 0.7], [0.5, -0.5, 0.7], [0.5, 0.5, 0.7], [-0.5, 0.5, 0.7], [-0.5, -0.5, 0.7]]
    )
    exit_a = _polyline([[-0.6, -0.4, 0.6], [0.6, -0.4, 0.6], [0.6, 0.4, 0.6]])
    exit_b = _polyline([[-0.6, 0.4, 0.6], [0.6, 0.4, 0.6], [0.6, -0.4, 0.6]])

    specs: list[tuple[str, str, FloatArray]] = [
        ("spiral", "spell", spiral),
        ("square", "spell", square),
        ("exit_a", "guide", exit_a),
        ("exit_b", "guide", exit_b),
    ]
    out: dict[str, Path] = {}
    for name, mode, pts in specs:
        path = clean_path(pts, mode=mode, box=box, speed=speed, name=name, window=7)
        path.meta["demo"] = True
        out[name] = path
    return out


if __name__ == "__main__":  # pragma: no cover - quick manual check
    for _name, _path in demo_paths().items():
        _wps = _path.waypoints()
        print(f"{_path!r}: {len(_wps)} waypoints, inside geofence={GEOFENCE.contains(_path.points)}")
