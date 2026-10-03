"""LIVE bridge: publishes the REAL drone link onto the UI bus (counterpart of ``ui/sim.py``).

Owns one cflib connection for the whole process and a 10 Hz publishing thread::

    bridge = LiveBridge(BUS, COMMANDS)      # uri defaults to config.LINK_URI
    bridge.connect()                        # blocks: flight.connect() once, estimator setup, 10 Hz log
    bridge.start()                          # publish link/drone/flight at 10 Hz, drain commands
    bridge.pause()                          # stop publishing, link stays up (switch back to SIM)
    bridge.close()                          # shutdown only: stop log blocks + cf.close_link()

What it publishes (``ui.state.AppState``): ``link{uri, connected, connecting, battery_v, error}``,
``drone{x, y, z}`` from ``stateEstimate`` (yaw stays 0: the onboard yaw is not in the log budget),
``flight{state, estimator_converged}`` from ``flight.Flight`` and ``hwcheck{...}``.  ``tracking`` is
left to the tracker.

Hardware check: a SECOND log block ``"hwcheck"`` (``stabilizer.roll``/``pitch`` as FP16, 5 Hz,
4 bytes -> one BLE notification) runs ONLY while ``fl.state == "idle"`` and the bridge is
publishing.  It is stopped the moment the flight state leaves idle (``Flight.on_state`` hook plus
a 10 Hz poll as a fallback) and restarted when it returns.  Together with the flight block
(10 Hz) that is 15 downlink packets/s, inside the ~30/s BLE budget.

Commands in LIVE mode: ``stop`` -> ``fl.request_stop()`` immediately (from the pusher's thread via
``CommandQueue.on_push``) AND ``fl.emergency_stop()`` from the bridge thread (fire-and-forget,
harmless when idle); ``land`` -> ``fl.request_land()``.  Everything else is logged
"not wired in LIVE mode yet" and left alone until the flight integration lands.

Assumptions about ``flight.py`` (read 2026-10-03):
  * ``flight.connect(uri)`` blocks until cflib ``fully_connected`` (<= 120 s), raises
    ``TimeoutError``/``ConnectionError`` on failure, registers ``ble_link`` itself.
  * ``flight.Flight(cf, FlightConfig(link_uri=uri), telemetry=flight.Telemetry())``; ``fl.state``
    is one of ``Flight.STATES``; ``fl.on_state`` is an optional callback we chain into.
  * ``fl.setup_estimator()`` then ``fl.start_log()`` creates the single 10 Hz block and stores it
    in ``fl.logconf`` (has ``.stop()``); ``fl.telemetry`` exposes ``battery_v`` (None until the
    first sample), ``position`` and ``converged``.
  * ``fl.request_stop()`` / ``fl.request_land()`` are thread-safe; ``fl.emergency_stop()`` sends
    ``loc.send_emergency_stop`` + ``send_stop_setpoint`` three times and sets state "estop".
  * ``cflib.crazyflie.log.LogConfig`` blocks can be ``stop()``ped and ``start()``ed again.

No hardware module is imported at module level: ``flight``/``cflib`` are imported lazily inside
the default factories, so tests inject fakes and never touch bleak.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import config
from ui.state import Command, CommandQueue, StateBus

try:  # nRF 2024.10 mangles long TOC names over BLE (README.md "Verified BLE facts"); always resolve
    from toc_names import resolve as resolve_name
except ImportError:  # pragma: no cover - module missing: assume an intact TOC

    def resolve_name(toc: Any, complete_name: str) -> str:
        return complete_name


__all__ = ["LiveBridge", "HWCHECK_LOG_NAME", "HWCHECK_PERIOD_MS", "HWCHECK_VARIABLES"]

HWCHECK_LOG_NAME = "hwcheck"
HWCHECK_PERIOD_MS = 200  # 5 Hz
HWCHECK_VARIABLES: list[tuple[str, str]] = [("stabilizer.roll", "FP16"), ("stabilizer.pitch", "FP16")]


def _default_connect(uri: str) -> Any:
    import flight

    return flight.connect(uri)


def _default_flight(cf: Any, uri: str) -> Any:
    import flight

    return flight.Flight(cf, flight.FlightConfig(link_uri=uri), telemetry=flight.Telemetry())


def _default_logconfig(name: str, period_in_ms: int) -> Any:
    from cflib.crazyflie.log import LogConfig

    return LogConfig(name=name, period_in_ms=period_in_ms)


class LiveBridge:
    def __init__(
        self,
        bus: StateBus,
        commands: CommandQueue,
        uri: str | None = None,
        *,
        connect_fn: Callable[[str], Any] = _default_connect,
        flight_factory: Callable[[Any, str], Any] = _default_flight,
        logconfig_factory: Callable[[str, int], Any] = _default_logconfig,
        rate_hz: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.bus = bus
        self.commands = commands
        self.uri = uri or config.LINK_URI
        self._connect_fn = connect_fn
        self._flight_factory = flight_factory
        self._logconfig_factory = logconfig_factory
        self.period = 1.0 / rate_hz
        self.clock = clock

        self.cf: Any = None
        self.fl: Any = None
        self.log_variables: list[str] = []
        self.hwcheck_conf: Any = None
        self.hwcheck_samples = 0
        self._hw_keys: dict[str, str] = {}  # name cflib stored (maybe mangled) -> true name

        self._connect_lock = threading.Lock()
        self._hw_lock = threading.RLock()
        self._hw_running = False
        self._tick_lock = threading.Lock()
        self._active = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        commands.on_push(self._on_push)

    # ----------------------------------------------------------------- status

    @property
    def connected(self) -> bool:
        return self.cf is not None

    @property
    def active(self) -> bool:
        """True while the bridge is the one publishing to the bus."""
        return self._active.is_set()

    @property
    def hwcheck_running(self) -> bool:
        return self._hw_running

    # ------------------------------------------------------------- lifecycle

    def connect(self) -> Any:
        """Connect ONCE (blocking, raises on failure) and return the ``Flight``.  Idempotent."""
        with self._connect_lock:
            if self.cf is not None:
                return self.fl
            self.bus.update(link={"uri": self.uri, "connected": False, "connecting": True, "error": ""})
            self.bus.log(f"LIVE: connecting to {self.uri} ... (first connect ~30-60 s over BLE)")
            t0 = time.perf_counter()
            try:
                cf = self._connect_fn(self.uri)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                self.bus.update(link={"connected": False, "connecting": False, "error": err})
                self.bus.log(f"LIVE: connect FAILED after {time.perf_counter() - t0:.0f} s: {err}")
                raise
            self.bus.log(f"LIVE: connected to {self.uri} in {time.perf_counter() - t0:.1f} s; estimator + 10 Hz log")
            try:
                fl = self._flight_factory(cf, self.uri)
                fl.setup_estimator()
                used = fl.start_log()
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                self.bus.update(link={"connected": False, "connecting": False, "error": err})
                self.bus.log(f"LIVE: setup FAILED: {err}")
                try:
                    cf.close_link()
                except Exception:  # noqa: BLE001
                    pass
                raise
            self.log_variables = list(used or [])
            prev = getattr(fl, "on_state", None)

            def on_state(state: str, _prev: Any = prev) -> None:
                if _prev is not None:
                    _prev(state)
                self._sync_hwcheck(state)

            fl.on_state = on_state
            self.cf, self.fl = cf, fl
            lost = getattr(cf, "connection_lost", None)  # cflib Caller; absent on fakes
            if lost is not None and hasattr(lost, "add_callback"):
                lost.add_callback(self._on_connection_lost)
            self.bus.update(
                link={"uri": self.uri, "connected": True, "connecting": False, "error": ""},
                flight={"state": fl.state, "estimator_converged": bool(fl.telemetry.converged)},
            )
            self.bus.log(f"LIVE: log block running ({', '.join(self.log_variables) or 'no variables'})")
            return fl

    def start(self, spawn_thread: bool = True) -> None:
        """Begin publishing (``connect()`` first).  ``spawn_thread=False`` lets tests call ``tick()``."""
        if self.cf is None:
            raise RuntimeError("LiveBridge.connect() before start()")
        self._active.set()
        if spawn_thread and (self._thread is None or not self._thread.is_alive()):
            self._shutdown.clear()
            self._thread = threading.Thread(target=self._run, name="pathcaster-live", daemon=True)
            self._thread.start()
        self._sync_hwcheck(self.fl.state)
        self.bus.log("LIVE: publishing drone telemetry at 10 Hz")

    def pause(self) -> None:
        """Stop publishing and the hwcheck block; the link and the flight log stay up."""
        if not self._active.is_set():
            return
        self._active.clear()
        with self._tick_lock:  # wait for a tick in progress
            pass
        self._stop_hwcheck()
        self.bus.log("LIVE: paused (link stays connected)")

    def close(self) -> None:
        """Shutdown only: stop both log blocks and close the link."""
        self.pause()
        self._shutdown.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(2.0)
        fl, cf = self.fl, self.cf
        if fl is not None and getattr(fl, "logconf", None) is not None:
            try:
                fl.logconf.stop()
            except Exception:  # noqa: BLE001
                pass
        if cf is not None:
            try:
                cf.close_link()
            except Exception:  # noqa: BLE001
                pass
            self.bus.log("LIVE: link closed")
        self.cf = self.fl = None
        self.hwcheck_conf = None
        self.bus.update(link={"connected": False, "connecting": False})

    def _on_connection_lost(self, _uri: str, msg: str) -> None:
        """cflib lost the link (BLE dropped): say so on screen; the operator restarts the server."""
        self._stop_hwcheck()
        self.bus.update(link={"connected": False, "connecting": False, "error": f"connection lost: {msg}"})
        self.bus.log(f"LIVE: CONNECTION LOST ({msg}) - power-cycle the drone and restart the server")

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:  # pragma: no cover - timing loop; tick() is tested
        while not self._shutdown.is_set():
            if self._active.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - never let the publisher die
                    self.bus.log(f"LIVE: tick error {type(exc).__name__}: {exc}")
            self._shutdown.wait(self.period)

    def tick(self) -> None:
        """One publishing step: drain commands, publish telemetry, sync the hwcheck block."""
        with self._tick_lock:
            if not self._active.is_set() or self.fl is None:
                return
            for cmd in self.commands.drain():
                self.handle(cmd)
            fl = self.fl
            tel = fl.telemetry
            x, y, z = tel.position
            bat = tel.battery_v
            lost = self.bus.snapshot().link.error.startswith("connection lost")
            self.bus.update(
                link={"uri": self.uri, "connected": not lost, "connecting": False, "battery_v": float(bat) if bat is not None else 0.0},
                drone={"x": float(x), "y": float(y), "z": float(z), "yaw": 0.0},
                flight={"state": str(fl.state), "estimator_converged": bool(tel.converged)},
            )
            self._sync_hwcheck(fl.state)

    # -------------------------------------------------------------- commands

    def handle(self, cmd: Command) -> None:
        fl = self.fl
        if fl is None:
            return
        tag = f"[{cmd.source}] " if cmd.source != "ui" else ""
        if cmd.name == "stop":
            fl.request_stop()
            fl.emergency_stop()
            self.bus.log(f"{tag}LIVE: EMERGENCY STOP sent (motors off)")
        elif cmd.name == "land":
            fl.request_land()
            self.bus.log(f"{tag}LIVE: land requested")
        elif cmd.name == "set_source":
            pass  # the server's SourceManager handles it
        else:
            self.bus.log(f"{tag}LIVE: '{cmd.name}' not wired in LIVE mode yet")

    def _on_push(self, cmd: Command) -> None:
        """Fast path from the pusher's thread: a stop request must not wait for the next tick."""
        if cmd.name == "stop" and self._active.is_set() and self.fl is not None:
            try:
                self.fl.request_stop()
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- hwcheck

    def _sync_hwcheck(self, state: str) -> None:
        with self._hw_lock:
            want = state == "idle" and self._active.is_set() and self.cf is not None
            if want and not self._hw_running:
                self._start_hwcheck()
            elif not want and self._hw_running:
                self._stop_hwcheck()

    def _start_hwcheck(self) -> None:
        with self._hw_lock:
            if self._hw_running or self.cf is None:
                return
            conf = self.hwcheck_conf
            if conf is None:
                conf = self._logconfig_factory(HWCHECK_LOG_NAME, HWCHECK_PERIOD_MS)
                toc = getattr(self.cf.log, "toc", None)
                self._hw_keys = {}
                for true_name, fetch_as in HWCHECK_VARIABLES:
                    stored = true_name
                    if toc is not None:  # cflib keys its TOC by the (possibly mangled) BLE names
                        try:
                            stored = resolve_name(toc, true_name)
                        except KeyError as exc:
                            self.bus.log(f"LIVE: hwcheck variable {true_name} unresolvable in this TOC ({exc}); skipped")
                            continue
                    conf.add_variable(stored, fetch_as)
                    self._hw_keys[stored] = true_name
                if not self._hw_keys:
                    self.bus.log("LIVE: hardware check disabled: no roll/pitch variables in the TOC")
                    return
                conf.data_received_cb.add_callback(self._on_hwcheck)
                conf.error_cb.add_callback(lambda _c, msg: self.bus.log(f"LIVE: hwcheck log error: {msg}"))
                self.cf.log.add_config(conf)
                self.hwcheck_conf = conf
            try:
                conf.start()
            except Exception as exc:  # noqa: BLE001
                self.bus.log(f"LIVE: hwcheck start failed: {exc}")
                return
            self._hw_running = True
            self.bus.update(hwcheck={"active": True})
            self.bus.log("LIVE: hardware check ON (roll/pitch @ 5 Hz while idle)")

    def _stop_hwcheck(self) -> None:
        with self._hw_lock:
            if not self._hw_running:
                return
            self._hw_running = False
            try:
                self.hwcheck_conf.stop()
            except Exception as exc:  # noqa: BLE001
                self.bus.log(f"LIVE: hwcheck stop failed: {exc}")
            self.bus.update(hwcheck={"active": False})
            self.bus.log("LIVE: hardware check off")

    def _on_hwcheck(self, timestamp: int, data: dict[str, Any], _logconf: Any = None) -> None:
        # cflib reports values under the names it stored; map them back to the true names
        vals = {true: data[stored] for stored, true in self._hw_keys.items() if stored in data}
        roll = float(vals.get("stabilizer.roll", data.get("stabilizer.roll", math.nan)))
        pitch = float(vals.get("stabilizer.pitch", data.get("stabilizer.pitch", math.nan)))
        self.hwcheck_samples += 1
        self.bus.update(hwcheck={"roll_deg": roll, "pitch_deg": pitch, "ts": float(self.clock()), "active": True})
