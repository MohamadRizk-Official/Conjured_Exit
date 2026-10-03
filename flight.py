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
ramps may go below the geofence floor); tracking lost > 0.3 s -> land; spacebar or the "stop"
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
    estop_repeats: int = 3


class Telemetry:
    """The 10 Hz log block decoded: battery, position estimate and Kalman variance window."""

    _VAR_KEYS = {"kalman.varPX": 0, "kalman.varPY": 1, "kalman.varPZ": 2}

    def __init__(self, window: int = 10, var_threshold: float = 0.001) -> None:
        self.window = window
        self.var_threshold = var_threshold
        self.battery_v: Optional[float] = None
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
                    self.land()
                    raise FlightAborted(f"tracking lost > {self.cfg.tracking_lost_land_s} s")

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
            self.cf.commander.send_position_setpoint(x, y, z, self.cfg.yaw_deg)
            self._last_target = (x, y, z)
            nxt = t0 + (i + 1) * dt
            now = self.clock()
            if nxt > now:
                self.sleep(nxt - now)

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
        self.last_abort_reason = ""
        if self.cfg.arm:
            self.cf.platform.send_arming_request(True)
        self._set_state("takeoff")
        zs = ramp(max(0.0, z0), self.cfg.takeoff_height, self.cfg.takeoff_time_s, self.cfg.setpoint_hz)
        hz = self.cfg.setpoint_hz
        try:
            self._stream(lambda t: (x, y, zs[min(int(round(t * hz)), len(zs) - 1)]), self.cfg.takeoff_time_s,
                         floor_ok=True)
            self._set_state("hover")
            self._stream(lambda t: (x, y, self.cfg.takeoff_height), self.cfg.hover_time_s, floor_ok=True)
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


def connect(uri: Optional[str] = None, rw_cache: str = "cache", timeout_s: float = 120.0):
    """Connect ONCE per session (TOC cache on) and return the cflib Crazyflie. Raises on failure."""
    import cflib.crtp
    from cflib.crazyflie import Crazyflie

    try:
        import ble_link
        ble_link.register()
        ble_link.configure(pump_hz=config.BLE_PUMP_HZ)
        ble_link.BleDriver.fast_interval = bool(config.BLE_FAST_INTERVAL)
    except ImportError:
        log.warning("ble_link not available; only built-in cflib drivers")
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache=rw_cache)
    disable_link_pinger(cf)
    done = threading.Event()
    failure: list[str] = []
    cf.fully_connected.add_callback(lambda _u: done.set())
    cf.connection_failed.add_callback(lambda _u, msg: (failure.append(str(msg)), done.set()))
    cf.connection_lost.add_callback(lambda _u, msg: log.error("connection lost: %s", msg))
    uri = uri or config.LINK_URI
    t0 = time.perf_counter()
    log.info("connecting to %s (first connect downloads TOCs, ~20-25 s over BLE)", uri)
    cf.open_link(uri)
    if not done.wait(timeout_s):
        cf.close_link()
        raise TimeoutError(f"no connection to {uri} after {timeout_s:.0f} s")
    if failure:
        raise ConnectionError(failure[0])
    log.info("connected to %s in %.1f s", uri, time.perf_counter() - t0)
    return cf


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
