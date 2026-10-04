"""flight.py -- Pathcaster flight layer over cflib (link-agnostic; URI comes from config.LINK_URI).

Decisions (README.md, 2026-10-03):
  * Position setpoints streamed at 20 Hz (`cf.commander.send_position_setpoint`, 18-byte packet),
    so the firmware's setpoint watchdog stops the drone by itself if the link drops.
    No high-level-commander go_to. If someone uses the high-level commander, call
    `resume_low_level()` (-> send_notify_setpoint_stop) before streaming again.
  * Takeoff = ramp z up at the current xy, then hover. Landing = ramp z down, then stop setpoint.
  * Emergency stop = fire-and-forget, sent 3 times (loc.send_emergency_stop + send_stop_setpoint).
  * Downlink budget = ONE log block at 10 Hz, 7 FP16 variables = 14 bytes, one BLE notification.
  * Connect once per session (`connect()`), TOC cache on, keep the link alive.

Safety rules (non-negotiable): every target is clamped into paths.GEOFENCE (takeoff/landing
ramps may go below the geofence floor); tracking lost > 0.3 s -> blind descent (thrust only, no position), motors off; spacebar or the "stop"
command -> emergency stop; takeoff is refused until the estimator converged and tracking is ok.

Desk check over BLE, motors never spin:
    python flight.py --check --seconds 20

The Flight class takes any object with the cflib Crazyflie attribute shape (commander, loc,
extpos, platform, param, log) so it is unit-tested against fakes with a fake clock.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import config
import paths
from toc_names import resolve as resolve_name

log = logging.getLogger("flight")

# --------------------------------------------------------------------------- link budget
# A log data packet is 1 CRTP header + 1 block id + 3 timestamp bytes + data. To fit one BLE
# notification (19 bytes of CRTP) the data must be <= 14 bytes -> 7 x FP16.
# kalman.stateX/Y/Z instead of stateEstimate.x/y/z: identical values when the Kalman estimator
# runs, and their TOC entries survive the nRF 2024.10 name corruption (see toc_names.py).
FLIGHT_LOG_VARIABLES: list[tuple[str, str]] = [
    ("pm.vbat", "FP16"),
    ("kalman.stateX", "FP16"),
    ("kalman.stateY", "FP16"),
    ("kalman.stateZ", "FP16"),
    ("kalman.varPX", "FP16"),
    ("kalman.varPY", "FP16"),
    ("kalman.varPZ", "FP16"),
]
LOG_TYPE_SIZES = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4,
                  "float": 4, "FP16": 2}
LOG_PERIOD_MS = 100          # 10 Hz; the only log traffic during flight
LOG_BLOCK_NAME = "pathcaster"


MAX_LOG_VARS_PER_UPLINK_PACKET = 5   # cmd + block id + 5 x (type + id16) = 17 payload -> 18-byte packet


def make_log_config(name: str, period_in_ms: int):
    """A cflib LogConfig whose create/append packets never exceed 20 bytes.

    nRF firmware 2024.10 corrupts every fragmented (> 20-byte) BLE packet in BOTH directions; a
    7-variable block sent in one 24-byte CREATE packet arrived at the drone with one variable
    missing (measured with tools/log_probe.py). cflib already continues a block with APPEND
    packets when a packet is full, so we only lower the per-packet variable count.
    """
    from cflib.crazyflie.log import LogConfig

    class SmallPacketLogConfig(LogConfig):
        def _setup_log_elements(self, pk, next_to_add):
            end = min(len(self.variables), next_to_add + MAX_LOG_VARS_PER_UPLINK_PACKET)
            for i in range(next_to_add, end):
                var = self.variables[i]
                if not var.is_toc_variable():
                    raise ValueError("raw-memory log variables are not supported over BLE")
                element_id = self.cf.log.toc.get_element_id(var.name)
                pk.data.append(var.get_storage_and_fetch_byte())
                if self.useV2:
                    pk.data.append(element_id & 0xFF)
                    pk.data.append((element_id >> 8) & 0xFF)
                else:
                    pk.data.append(element_id)
            return end >= len(self.variables), end

    return SmallPacketLogConfig(name=name, period_in_ms=period_in_ms)


class FlightRefused(Exception):
    """A pre-condition for the requested action is not met; nothing was sent."""


class FlightAborted(Exception):
    """The current manoeuvre was interrupted by a safety action (land or emergency stop)."""


def ramp(z0: float, z1: float, duration_s: float, hz: float) -> list[float]:
    """Linear z profile from z0 (exclusive) to z1 (inclusive) with round(duration*hz) steps."""
    n = max(1, int(round(duration_s * hz)))
    return [z0 + (z1 - z0) * (i + 1) / n for i in range(n)]


@dataclass
class FlightConfig:
    link_uri: str = config.LINK_URI
    setpoint_hz: float = 20.0
    takeoff_height: float = config.TAKEOFF_HEIGHT_M
    takeoff_time_s: float = 2.0
    hover_time_s: float = 1.5
    land_time_s: float = 2.5
    land_cutoff_z: float = 0.06
    speed: float = config.REPLAY_SPEED_MPS            # transit speed to the path start
    tracking_lost_land_s: float = config.TRACKING_LOST_LAND_S
    tracker_stale_s: float = 0.5                      # tracker state older than this counts as lost
    converge_window: int = 10
    converge_var_threshold: float = 0.001
    converge_timeout_s: float = 20.0
    log_period_ms: int = LOG_PERIOD_MS
    geofence: paths.Box = paths.GEOFENCE
    yaw_deg: float = 0.0                              # nose along world +x
    arm: bool = True                                  # send an arming request before takeoff
    thrust_unlock: bool = True                        # 3 zero-thrust packets before the climb (hop.py did this)
    estop_repeats: int = 3
    blind_descent_thrust: tuple = (40000, 34000)  # tracking lost in flight: thrust-only steps (16-bit, hover ~40000)
    blind_descent_step_s: float = 1.0             # seconds per step, then motors off
    blind_descent_min_z: float = 0.4              # below this last target height (or during take-off): motors off instead
    takeoff_min_rise_frac: float = 0.0
    takeoff_max_drift_m: float = 0.0                  # abort the ramp when the camera sees > this much xy drift (0 = off)
    marker_yaw_offset_deg: float = 0.0                # nose heading relative to the marker's top edge (CCW positive)           # >0: abort (motors off) if the measured rise after the ramp
                                                  # is below this fraction of the commanded climb (mission uses 0.4)


class Telemetry:
    """The 10 Hz log block decoded: battery, position estimate and Kalman variance window."""

    _VAR_KEYS = {"kalman.varPX": 0, "kalman.varPY": 1, "kalman.varPZ": 2}

    def __init__(self, window: int = 10, var_threshold: float = 0.001) -> None:
        self.window = window
        self.var_threshold = var_threshold
        self.battery_v: Optional[float] = None
        self.battery_min_v: Optional[float] = None       # lowest reading since reset_battery_min() (sag under load)
        self.x = self.y = self.z = 0.0
        self.var = [float("nan")] * 3
        self._hist = [deque(maxlen=window) for _ in range(3)]
        self.timestamp_ms: Optional[int] = None
        self.n_updates = 0
        self._lock = threading.Lock()

    def update(self, timestamp_ms: int, data: dict, _logconf=None) -> None:
        with self._lock:
            self.timestamp_ms = timestamp_ms
            self.n_updates += 1
            if "pm.vbat" in data:
                self.battery_v = float(data["pm.vbat"])
                if self.battery_min_v is None or self.battery_v < self.battery_min_v:
                    self.battery_min_v = self.battery_v
            for key in ("kalman.stateX", "stateEstimate.x"):
                if key in data:
                    self.x = float(data[key])
            for key in ("kalman.stateY", "stateEstimate.y"):
                if key in data:
                    self.y = float(data[key])
            for key in ("kalman.stateZ", "stateEstimate.z"):
                if key in data:
                    self.z = float(data[key])
            for key, axis in self._VAR_KEYS.items():
                if key in data:
                    v = float(data[key])
                    self.var[axis] = v
                    self._hist[axis].append(v)

    def reset_battery_min(self) -> None:
        """Restart the sag measurement from the current reading (call right before the motors start)."""
        with self._lock:
            self.battery_min_v = self.battery_v

    @property
    def position(self) -> tuple[float, float, float]:
        with self._lock:
            return (self.x, self.y, self.z)

    @property
    def converged(self) -> bool:
        """Same criterion as cflib's motion-capture examples: variance stopped moving on all axes."""
        with self._lock:
            for hist in self._hist:
                if len(hist) < self.window:
                    return False
                if max(hist) - min(hist) >= self.var_threshold:
                    return False
            return True

    def summary(self) -> str:
        v = "nan" if self.battery_v is None else f"{self.battery_v:.2f} V"
        return (f"bat {v}  pos ({self.x:+.2f}, {self.y:+.2f}, {self.z:+.2f})  "
                f"var ({self.var[0]:.4f}, {self.var[1]:.4f}, {self.var[2]:.4f})  "
                f"{'CONVERGED' if self.converged else 'not converged'}")


class Flight:
    """Streams setpoints to a cflib Crazyflie (or a fake with the same attributes)."""

    STATES = ("idle", "takeoff", "hover", "flying", "landing", "estop")

    def __init__(self, cf, cfg: Optional[FlightConfig] = None, tracker=None, telemetry: Optional[Telemetry] = None,
                 commands: Optional[Callable[[], Optional[str]]] = None,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep) -> None:
        self.cf = cf
        self.cfg = cfg or FlightConfig()
        self.tracker = tracker
        self.last_initial_yaw_rad: Optional[float] = None   # what the last estimator reset told the drone
        self._takeoff_origin: Optional[tuple[float, float]] = None
        self.telemetry = telemetry or Telemetry(self.cfg.converge_window, self.cfg.converge_var_threshold)
        self.commands = commands          # callable returning the next command name ("stop", "land") or None
        self.clock = clock
        self.sleep = sleep
        self.state = "idle"
        self.last_abort_reason = ""
        self.on_state: Optional[Callable[[str], None]] = None
        self.logconf = None
        self._stop_requested = threading.Event()
        self._land_requested = threading.Event()
        self._last_target: Optional[tuple[float, float, float]] = None
        self._lost_since: Optional[float] = None

    # ------------------------------------------------------------------ state helpers
    def _set_state(self, state: str) -> None:
        self.state = state
        log.info("flight state -> %s", state)
        if self.on_state:
            try:
                self.on_state(state)
            except Exception:  # noqa: BLE001
                log.exception("on_state callback failed")

    def request_stop(self) -> None:
        """Thread-safe emergency stop request (spacebar, UI, voice). Acted on at the next tick."""
        self._stop_requested.set()

    def request_land(self) -> None:
        self._land_requested.set()

    def reset_estop(self) -> None:
        """Clear an emergency stop so a new takeoff may be attempted (the drone may need a reboot)."""
        self._stop_requested.clear()
        self._land_requested.clear()
        self._set_state("idle")

    def resume_low_level(self) -> None:
        """Call after any high-level-commander use before streaming setpoints again."""
        self.cf.commander.send_notify_setpoint_stop()

    # ------------------------------------------------------------------ hardware setup
    def setup_estimator(self) -> None:
        """Select the Kalman estimator and reset it. Needs a connected cflib Crazyflie."""
        toc = self.cf.param.toc
        self.cf.param.set_value(resolve_name(toc, "stabilizer.estimator"), "2")
        self.sleep(0.1)
        # The gyro only integrates heading CHANGES; the reset takes its zero from kalman.initialYaw.
        # Position-only extpos never corrects yaw, so a wrong zero rotates every controller output
        # (2026-10-04 03:51: the drone slid north-west at 8 cm instead of climbing).
        self.last_initial_yaw_rad = None
        heading = self.marker_heading_rad()
        if heading is not None:
            try:
                self.cf.param.set_value(resolve_name(toc, "kalman.initialYaw"), f"{heading:.4f}")
                self.last_initial_yaw_rad = heading
                log.info("estimator reset: initial yaw %.1f deg from the marker", math.degrees(heading))
            except KeyError as e:
                log.warning("kalman.initialYaw not in this firmware's TOC (%s); heading left at 0", e)
        reset = resolve_name(toc, "kalman.resetEstimation")
        self.cf.param.set_value(reset, "1")
        self.sleep(0.1)
        self.cf.param.set_value(reset, "0")

    def read_param(self, complete_name: str):
        """Read a param by its TRUE name through the BLE-corruption-aware resolver."""
        return self.cf.param.get_value(resolve_name(self.cf.param.toc, complete_name))

    def start_log(self) -> list[str]:
        """Start the single 10 Hz log block. Returns the variables that were actually available."""
        conf = make_log_config(LOG_BLOCK_NAME, self.cfg.log_period_ms)
        used = []
        alias: dict[str, str] = {}      # name as cflib knows it -> true name
        toc = getattr(self.cf.log, "toc", None)
        for name, fetch_as in FLIGHT_LOG_VARIABLES:
            seen = name
            if toc is not None:
                try:
                    seen = resolve_name(toc, name)
                except KeyError as e:
                    log.warning("log variable %s skipped: %s", name, e)
                    continue
            conf.add_variable(seen, fetch_as)
            alias[seen] = name
            used.append(name)

        def _on_data(timestamp, data, logconf):
            self.telemetry.update(timestamp, {alias.get(k, k): v for k, v in data.items()}, logconf)

        conf.data_received_cb.add_callback(_on_data)
        conf.error_cb.add_callback(lambda _c, msg: log.error("log block error: %s", msg))
        self.cf.log.add_config(conf)
        conf.start()
        self.logconf = conf
        log.info("log block started: %s @ %d ms", used, self.cfg.log_period_ms)
        return used

    def wait_for_estimator(self, timeout_s: Optional[float] = None) -> bool:
        deadline = self.clock() + (self.cfg.converge_timeout_s if timeout_s is None else timeout_s)
        while self.clock() < deadline:
            if self.telemetry.converged:
                return True
            self.sleep(0.1)
        return self.telemetry.converged

    # ------------------------------------------------------------------ safety
    def _tracking_ok(self) -> bool:
        if self.tracker is None:
            return True
        st = self.tracker.get_state()
        if st is None or not st.tracking_ok:
            return False
        return (self.clock() - st.t) <= self.cfg.tracker_stale_s

    def _refuse_if_unsafe(self) -> None:
        if self._stop_requested.is_set():
            self.emergency_stop()
            raise FlightRefused("emergency stop requested")
        if self.state == "estop":
            raise FlightRefused("emergency stop active; reset_estop() first")
        if self.state != "idle":
            raise FlightRefused(f"cannot take off from state '{self.state}'")
        if not self.telemetry.converged:
            raise FlightRefused("estimator not converged")
        if not self._tracking_ok():
            raise FlightRefused("tracking not ok")

    def _check_safety(self, check_tracking: bool) -> None:
        """Runs before every setpoint. Takes the safety action, then raises FlightAborted."""
        if self._stop_requested.is_set():
            self.emergency_stop()
            raise FlightAborted("emergency stop requested")
        cmd = self.commands() if self.commands else None
        if cmd:
            name = str(cmd).strip().lower()
            if name in ("stop", "estop", "emergency_stop"):
                self.emergency_stop()
                raise FlightAborted("stop command")
            if name == "land":
                self._land_requested.set()
        if self._land_requested.is_set() and self.state != "landing":
            self._land_requested.clear()
            self.land()
            raise FlightAborted("land command")
        if check_tracking and self.tracker is not None:
            now = self.clock()
            if self._tracking_ok():
                self._lost_since = None
            else:
                if self._lost_since is None:
                    self._lost_since = now
                elif now - self._lost_since >= self.cfg.tracking_lost_land_s:
                    self._lost_since = None
                    z_last = self._last_target[2] if self._last_target is not None else None
                    low = z_last is None or z_last < self.cfg.blind_descent_min_z
                    in_takeoff = self.state == "takeoff"
                    if in_takeoff or low:
                        # 04:27: a blind descent fired 2 s into the ramp and drove the drone into the floor
                        # at thrust. Near the floor the drop is harmless; a thrust burst in an unknown
                        # attitude is not.
                        self.soft_stop()
                        where = "during take-off" if in_takeoff or z_last is None else f"at {z_last:.2f} m"
                        raise FlightAborted(f"tracking lost > {self.cfg.tracking_lost_land_s} s {where}: motors off")
                    self.blind_descent()
                    raise FlightAborted(f"tracking lost > {self.cfg.tracking_lost_land_s} s: blind descent, motors off")
        if self.state == "takeoff" and self.cfg.takeoff_max_drift_m > 0 and self._takeoff_origin is not None:
            xyz = self._tracker_xyz()
            if xyz is not None:
                drift = math.hypot(xyz[0] - self._takeoff_origin[0], xyz[1] - self._takeoff_origin[1])
                if drift > self.cfg.takeoff_max_drift_m:
                    self.blind_descent()
                    raise FlightAborted(f"drifted {drift:.2f} m sideways during take-off (heading mismatch?): "
                                        f"blind descent, motors off")

    def _clamp(self, x: float, y: float, z: float, floor_ok: bool) -> tuple[float, float, float]:
        box = self.cfg.geofence
        zmin = 0.0 if floor_ok else box.zmin
        return (float(min(max(x, box.xmin), box.xmax)),
                float(min(max(y, box.ymin), box.ymax)),
                float(min(max(z, zmin), box.zmax)))

    # ------------------------------------------------------------------ streaming core
    def _stream(self, target_at: Callable[[float], tuple], duration_s: float, *, floor_ok: bool = False,
                check_tracking: bool = True) -> None:
        hz = self.cfg.setpoint_hz
        dt = 1.0 / hz
        n = max(1, int(round(duration_s * hz)))
        t0 = self.clock()
        for i in range(n):
            self._check_safety(check_tracking)
            x, y, z = target_at(i * dt)
            x, y, z = self._clamp(float(x), float(y), float(z), floor_ok)
            self.cf.commander.send_position_setpoint(x, y, z, self.yaw_setpoint_deg())
            self._last_target = (x, y, z)
            nxt = t0 + (i + 1) * dt
            now = self.clock()
            if nxt > now:
                self.sleep(nxt - now)

    def _tracker_xyz(self) -> Optional[tuple[float, float, float]]:
        if self.tracker is None:
            return None
        st = self.tracker.get_state()
        if st is None or not st.tracking_ok or st.xyz is None:
            return None
        return (float(st.xyz[0]), float(st.xyz[1]), float(st.xyz[2]))

    def yaw_setpoint_deg(self) -> float:
        """Absolute yaw for position setpoints: the heading the estimator was given (so the drone holds
        its nose where it is), else cfg.yaw_deg. Asking for 0 while the drone knows it faces 173 deg
        commands a half-turn spin at lift-off (04:27)."""
        if self.last_initial_yaw_rad is not None:
            return float(math.degrees(self.last_initial_yaw_rad))
        return float(self.cfg.yaw_deg)

    def marker_heading_rad(self) -> Optional[float]:
        """The drone's nose heading in the world frame: the marker's top-edge yaw plus
        cfg.marker_yaw_offset_deg, wrapped to (-pi, pi]. None when the tracker has no yaw."""
        if self.tracker is None:
            return None
        st = self.tracker.get_state()
        yaw = getattr(st, "yaw", None) if st is not None else None
        if yaw is None or not math.isfinite(float(yaw)):
            return None
        h = float(yaw) + math.radians(self.cfg.marker_yaw_offset_deg)
        return math.atan2(math.sin(h), math.cos(h))

    def _measured_height(self) -> Optional[float]:
        """Height from the tracker when it has a fix, else from the drone's own estimate."""
        if self.tracker is not None:
            st = self.tracker.get_state()
            if st is not None and st.tracking_ok and st.xyz is not None:
                return float(st.xyz[2])
        return float(self.telemetry.position[2])

    def _start_position(self) -> tuple[float, float, float]:
        if self.tracker is not None:
            st = self.tracker.get_state()
            if st is not None and st.tracking_ok:
                return tuple(float(v) for v in st.xyz)
        return self.telemetry.position

    # ------------------------------------------------------------------ public manoeuvres
    def takeoff(self) -> bool:
        """Arm, ramp up at the current xy, hover. Returns True when hovering, False if aborted."""
        self._refuse_if_unsafe()
        x, y, z0 = self._start_position()
        self._takeoff_origin = (float(x), float(y))
        self.last_abort_reason = ""
        if self.cfg.arm:
            self.cf.platform.send_arming_request(True)
        if self.cfg.thrust_unlock:
            for _ in range(3):
                self.cf.commander.send_setpoint(0, 0, 0, 0)
        reset_min = getattr(self.telemetry, "reset_battery_min", None)
        if callable(reset_min):
            reset_min()
        v_start = getattr(self.telemetry, "battery_v", None)
        self._set_state("takeoff")
        zs = ramp(max(0.0, z0), self.cfg.takeoff_height, self.cfg.takeoff_time_s, self.cfg.setpoint_hz)
        hz = self.cfg.setpoint_hz
        try:
            self._stream(lambda t: (x, y, zs[min(int(round(t * hz)), len(zs) - 1)]), self.cfg.takeoff_time_s,
                         floor_ok=True)
            if self.cfg.takeoff_min_rise_frac > 0:
                z_now = self._measured_height()
                climb = self.cfg.takeoff_height - max(0.0, z0)
                if z_now is not None and (z_now - max(0.0, z0)) < self.cfg.takeoff_min_rise_frac * climb:
                    self.soft_stop()
                    v_min = getattr(self.telemetry, "battery_min_v", None)
                    sag = (f"; battery sagged {v_start:.2f} -> {v_min:.2f} V"
                           if v_start is not None and v_min is not None else "")
                    raise FlightAborted(f"takeoff did not lift (measured z {z_now:.2f} m after the ramp to "
                                        f"{self.cfg.takeoff_height:.2f} m{sag}): motors off")
            self._set_state("hover")
            self._stream(lambda t: (x, y, self.cfg.takeoff_height), self.cfg.hover_time_s, floor_ok=True)
            return True
        except FlightAborted as e:
            self.last_abort_reason = str(e)
            return False

    def hold(self, seconds: float) -> bool:
        """Keep streaming the last target (state 'hover') for `seconds`. False if aborted (stop/land/tracking)."""
        if self.state != "hover":
            raise FlightRefused(f"hold needs state 'hover', not '{self.state}' (take off first)")
        self.last_abort_reason = ""
        x, y, z = self._last_target
        try:
            self._stream(lambda t: (x, y, z), max(0.0, float(seconds)))
            return True
        except FlightAborted as e:
            self.last_abort_reason = str(e)
            return False

    def fly_path(self, path: paths.Path, land: bool = True) -> bool:
        """Transit to the path start at cfg.speed, follow it at its own timing, hover, (land)."""
        if self.state != "hover":
            raise FlightRefused(f"fly_path needs state 'hover', not '{self.state}' (take off first)")
        self.last_abort_reason = ""
        try:
            self._set_state("flying")
            start = np.array(self._last_target, dtype=float)
            p0 = np.asarray(path.points[0], dtype=float)
            dist = float(np.linalg.norm(p0 - start))
            transit = dist / self.cfg.speed if self.cfg.speed > 0 else 0.0
            if transit > 0:
                self._stream(lambda t: start + (p0 - start) * min(1.0, t / transit), transit)
            duration = float(path.duration)
            self._stream(lambda t: path.position_at(min(t, duration)), duration)
            self._set_state("hover")
            end = np.asarray(path.points[-1], dtype=float)
            self._stream(lambda t: end, self.cfg.hover_time_s)
            if land:
                self.land()
            return True
        except FlightAborted as e:
            self.last_abort_reason = str(e)
            return False

    def land(self) -> None:
        """Ramp z down from the last target to the cutoff height, then send the stop setpoint."""
        if self.state in ("idle", "estop"):
            return
        self._set_state("landing")
        x, y, z0 = self._last_target or self.telemetry.position
        zs = ramp(z0, self.cfg.land_cutoff_z, self.cfg.land_time_s, self.cfg.setpoint_hz)
        hz = self.cfg.setpoint_hz
        self._stream(lambda t: (x, y, zs[min(int(round(t * hz)), len(zs) - 1)]), self.cfg.land_time_s,
                     floor_ok=True, check_tracking=False)
        self.cf.commander.send_stop_setpoint()
        self._last_target = None
        self._set_state("idle")

    def blind_descent(self) -> None:
        """Tracking is gone, so the position estimate cannot be trusted: come down on attitude + thrust only.

        Level attitude, thrust a little below hover for cfg.blind_descent_step_s per step, then motors off.
        Ends in state 'estop' (the drone may not be where we think it is; a clear is required before the
        next flight). A pending stop request cuts the motors immediately.
        """
        self._set_state("landing")
        cmd = self.cf.commander
        hz = self.cfg.setpoint_hz
        dt = 1.0 / hz
        try:
            cmd.send_setpoint(0.0, 0.0, 0.0, 0)          # thrust unlock (one zero-thrust packet)
            for thrust in self.cfg.blind_descent_thrust:
                for _ in range(max(1, int(round(self.cfg.blind_descent_step_s * hz)))):
                    if self._stop_requested.is_set():
                        self.emergency_stop()
                        return
                    cmd.send_setpoint(0.0, 0.0, 0.0, int(thrust))
                    self.sleep(dt)
            cmd.send_setpoint(0.0, 0.0, 0.0, 0)
        except Exception:  # noqa: BLE001
            log.exception("blind descent failed; emergency stop")
            self.emergency_stop()
            return
        cmd.send_stop_setpoint()
        self._last_target = None
        self._set_state("estop")

    def soft_stop(self) -> None:
        """Motors off with stop setpoints only (no supervisor lock): the drone stays armable without a reboot.

        Used when the drone is on the ground anyway (take-off did not lift). Ends in 'estop' so a clear is
        required before the next attempt.
        """
        for _ in range(self.cfg.estop_repeats):
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:  # noqa: BLE001
                log.exception("send_stop_setpoint failed")
        self._last_target = None
        self._set_state("estop")

    def emergency_stop(self) -> None:
        """Motors off NOW. Fire-and-forget, repeated cfg.estop_repeats times."""
        for _ in range(self.cfg.estop_repeats):
            try:
                self.cf.loc.send_emergency_stop()
            except Exception:  # noqa: BLE001
                log.exception("send_emergency_stop failed")
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:  # noqa: BLE001
                log.exception("send_stop_setpoint failed")
        self._last_target = None
        self._set_state("estop")

    # ------------------------------------------------------------------ keyboard E-stop
    def start_keyboard_estop(self) -> Optional[threading.Thread]:
        """Spacebar -> emergency stop, 'l' -> land. Windows console only; returns the thread."""
        if sys.platform != "win32" or not sys.stdin or not sys.stdin.isatty():
            return None
        import msvcrt

        def worker() -> None:
            while True:
                ch = msvcrt.getwch()
                if ch == " ":
                    print("\n[SPACE] EMERGENCY STOP")
                    self.request_stop()
                elif ch in ("l", "L"):
                    print("\n[L] land requested")
                    self.request_land()

        th = threading.Thread(target=worker, name="kbd-estop", daemon=True)
        th.start()
        return th


# --------------------------------------------------------------------------- fakes
class FakeCrazyflie:
    """Stand-in for a cflib Crazyflie with no link: counts setpoints, swallows everything else.

    Used by ``hover.py --dry-run``, ``mission.py --sim`` (through ``mission_sim.SimDrone``) and tests.
    ``param.toc`` / ``log.toc`` hold the names the flight code resolves with ``toc_names.resolve``.
    """

    TOC = {
        "stabilizer": {"estimator": 143, "roll": 1, "pitch": 2},
        "kalman": {"resetEstimation": 116, "stateX": 3, "stateY": 4, "stateZ": 5, "varPX": 6, "varPY": 7, "varPZ": 8},
        "locSrv": {"extPosStdDev": 107, "extQuatStdDev": 108},
        "pm": {"vbat": 9},
        "supervisor": {"info": 10},
    }

    class _Commander:
        def __init__(self) -> None:
            self.n = 0
            self.stops = 0
            self.notify_stops = 0
            self.last = None

        def send_position_setpoint(self, x, y, z, yaw):
            self.n += 1
            self.last = (x, y, z, yaw)

        def send_stop_setpoint(self):
            self.stops += 1

        def send_notify_setpoint_stop(self, remain_valid_milliseconds=0):
            self.notify_stops += 1

        def send_setpoint(self, roll, pitch, yawrate, thrust):
            self.rpyt_n = getattr(self, "rpyt_n", 0) + 1

    class _Any:
        """Any attribute not set explicitly is a no-op method."""

        def __init__(self, **attrs) -> None:
            self.__dict__.update(attrs)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    def __init__(self) -> None:
        self.commander = self._Commander()
        self.loc = self._Any()
        self.extpos = self._Any()
        self.platform = self._Any()
        self.supervisor = self._Any()
        self.param = self._Any(toc={k: dict(v) for k, v in self.TOC.items()})
        self.log = self._Any(toc={k: dict(v) for k, v in self.TOC.items()})

    def close_link(self) -> None:
        pass


# --------------------------------------------------------------------------- link lifecycle
def disable_link_pinger(cf) -> None:
    """cflib >= 0.1.34 pings the link-echo channel at 10 Hz after connecting (LinkStatistics).

    Over BLE each ping is a serialized write plus a downlink slot, which starves the parameter
    download and competes with our 10 Hz log block, so it is stopped as soon as cflib starts it.
    """
    stats = getattr(cf, "link_statistics", None)
    if stats is None:
        return

    def _stop(_uri) -> None:
        try:
            stats.stop()
            log.info("cflib link-statistics pinger disabled (BLE budget)")
        except Exception:  # noqa: BLE001
            log.exception("could not stop cflib link-statistics pinger")

    cf.connected.add_callback(_stop)


def connect(uri: Optional[str] = None, rw_cache: str = "cache", timeout_s: float = 120.0, *,
            setup_timeout_s: float = 30.0, attempts: int = 3, retry_delay_s: float = 4.0):
    """Connect ONCE per session (TOC cache on) and return the cflib Crazyflie. Raises on failure.

    Over BLE one lost notification during cflib's connection setup (log/mem/param TOC handshake) stalls
    cflib forever: it never retries those requests. Good setups finish in 12-16 s (measured 2026-10-04),
    so if ``connected`` has not fired after ``setup_timeout_s`` the link is closed and opened again, up
    to ``attempts`` times. A reported connection failure (no drone found, link dropped) is raised at once.
    """
    import cflib.crtp
    from cflib.crazyflie import Crazyflie

    try:
        import ble_link
        ble_link.register()
        ble_link.configure(pump_hz=config.BLE_PUMP_HZ, stream_ports=tuple(config.BLE_STREAM_PORTS))
        ble_link.BleDriver.fast_interval = bool(config.BLE_FAST_INTERVAL)
    except ImportError:
        log.warning("ble_link not available; only built-in cflib drivers")
    cflib.crtp.init_drivers()
    uri = uri or config.LINK_URI
    t0 = time.perf_counter()
    for attempt in range(1, max(1, attempts) + 1):
        cf = Crazyflie(rw_cache=rw_cache)
        disable_link_pinger(cf)
        setup_done = threading.Event()
        done = threading.Event()
        failure: list[str] = []
        cf.connected.add_callback(lambda _u: setup_done.set())
        cf.fully_connected.add_callback(lambda _u: done.set())
        cf.connection_failed.add_callback(lambda _u, msg: (failure.append(str(msg)), done.set()))
        cf.connection_lost.add_callback(lambda _u, msg: log.error("connection lost: %s", msg))
        log.info("connecting to %s (attempt %d/%d; TOCs from cache, ~50 s over BLE)", uri, attempt, attempts)
        cf.open_link(uri)
        if not setup_done.wait(setup_timeout_s) and not done.is_set():
            log.warning("connection setup stalled for %.0f s (lost handshake reply over BLE); reconnecting", setup_timeout_s)
            cf.close_link()
            if attempt < attempts:
                time.sleep(retry_delay_s)
                continue
            raise TimeoutError(f"connection setup to {uri} stalled {attempts} times")
        if not done.wait(timeout_s):
            cf.close_link()
            raise TimeoutError(f"no connection to {uri} after {timeout_s:.0f} s")
        if failure:
            raise ConnectionError(failure[0])
        link = getattr(cf, "link", None)
        if link is not None and hasattr(link, "pump_hz"):
            # TOC download wants a fast pump (one downlink packet per uplink packet); in flight every null is
            # an acknowledged write that delays setpoints and extpos (2026-10-04: 50 writes/s -> 1.3 s lag).
            link.pump_hz = float(config.BLE_PUMP_FLIGHT_HZ)
        log.info("connected to %s in %.1f s", uri, time.perf_counter() - t0)
        return cf
    raise TimeoutError(f"no connection to {uri}")


def _check(args: argparse.Namespace) -> int:
    """Motors-off desk check: connect, estimator setup, 10 Hz log, convergence report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cf = connect(args.uri)
    fl = Flight(cf, FlightConfig(link_uri=args.uri or config.LINK_URI))
    try:
        fl.setup_estimator()
        used = fl.start_log()
        print(f"log block variables: {used}")
        try:
            est = fl.read_param("stabilizer.estimator")
            print(f"stabilizer.estimator = {est!r}  ({'Kalman' if str(est) == '2' else 'NOT Kalman'})")
        except Exception as e:  # noqa: BLE001
            print(f"stabilizer.estimator read failed: {e!r}")
        t0 = time.perf_counter()
        last_n = 0
        while time.perf_counter() - t0 < args.seconds:
            time.sleep(0.5)
            print(f"{time.perf_counter() - t0:5.1f} s  updates {fl.telemetry.n_updates:4d}  {fl.telemetry.summary()}")
            last_n = fl.telemetry.n_updates
        rate = last_n / max(1e-9, args.seconds)
        ok = fl.telemetry.converged and rate > 5.0
        print(f"\nlog rate {rate:.1f} Hz (target 10)  estimator {'CONVERGED' if fl.telemetry.converged else 'NOT converged'}"
              f"  -> takeoff would be {'ALLOWED' if ok else 'REFUSED'} (tracking not checked here)")
        return 0 if ok else 1
    finally:
        if fl.logconf is not None:
            try:
                fl.logconf.stop()
            except Exception:  # noqa: BLE001
                pass
        cf.commander.send_stop_setpoint()
        cf.close_link()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="motors-off desk check: connect, log, convergence")
    ap.add_argument("--uri", default=None, help=f"override config.LINK_URI ({config.LINK_URI})")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    if args.check:
        sys.exit(_check(args))
    ap.print_help()


if __name__ == "__main__":
    main()
