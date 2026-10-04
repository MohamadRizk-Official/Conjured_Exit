"""Mission app: the one process that owns the drone and joins tracker, feed, paths, flight and the UI.

Run (x64 venv, native Windows, from the project root)::

    cf64\\Scripts\\python.exe mission.py --sim                  # fake drone that follows its setpoints, demo paths
    cf64\\Scripts\\python.exe mission.py --tracker sim          # REAL link; SimTracker positions (motors-off link test)
    cf64\\Scripts\\python.exe mission.py --tracker aruco --camera 1

then open http://127.0.0.1:8765/ , click ARM, then ALARM or Cast.

Design: docs/superpowers/specs/2026-10-03-mission-app-design.md (local file). Rules it keeps: one process
on the BLE link; every packet <= 20 bytes (everything goes through flight.Flight / feed.PositionFeed);
names through toc_names.resolve; one 10 Hz log block in flight (the supervisor block is stopped for the
flight); geofence clamp in Flight; position only into the drone, so flights start nose along +x.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import config
import feed
import flight
import hop
import paths
from toc_names import resolve as resolve_name
from ui.pathstore import PathStore
from ui.state import BUS, COMMANDS, MODES, Command, CommandQueue, StateBus, as_bool

log = logging.getLogger("mission")

EXIT_PATHS = {"A": "exit_a", "B": "exit_b"}


@dataclass
class MissionConfig:
    publish_hz: float = 20.0
    record_hz: float = 30.0
    relaunch_delay_s: float = 2.0          # touchdown -> relaunch on the other exit
    hover_height_m: float = 0.6            # 'hover' command defaults (page button: 0.6 m, 8 s)
    hover_hold_s: float = 8.0
    agree_tol_m: float = 0.2               # estimator vs camera before takeoff
    min_battery_v: float = 3.7
    ext_std: Optional[float] = 0.05        # locSrv.extPosStdDev; None = leave the drone's value
    converge_retry_s: float = 5.0          # one estimator reset + wait when the estimate disagrees
    feed_status_every_s: float = 10.0      # log feed.status() this often
    allow_flights: bool = True             # False: real link + fake tracker -> arming never flies


def exit_letter(path_name: str) -> Optional[str]:
    for letter, name in EXIT_PATHS.items():
        if path_name == name:
            return letter
    return None


@dataclass
class HoverPlan:
    """A hover test: straight up to height_m, hold for seconds, land. Flown by Mission._worker like a path."""

    height_m: float
    seconds: float


class RouteRecorder:
    """Samples the tracker into a paths.PathRecorder while recording; publishes live points and the count.

    Guide mode records the drone (``xyz`` while ``tracking_ok``); spell mode records the wand
    (``wand_xyz`` while ``wand_ok``). ``tick()`` is one sample (tests); ``start()`` runs it in a thread.
    """

    def __init__(self, tracker, bus: StateBus, *, rate_hz: float = 30.0,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep) -> None:
        self.tracker = tracker
        self.bus = bus
        self.period = 1.0 / rate_hz
        self.clock = clock
        self.sleep = sleep
        self.rec = paths.PathRecorder(min_step=0.01)
        self.mode = "guide"
        self.active = False
        self._lost_logged = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def n(self) -> int:
        return len(self.rec)

    @property
    def points(self) -> np.ndarray:
        return self.rec.points

    def start(self, mode: str, spawn_thread: bool = True) -> None:
        self.rec.clear()
        self.mode = mode
        self.active = True
        self._lost_logged = False
        self.bus.update(recording={"active": True, "mode": mode, "n_samples": 0, "live_points": []})
        if spawn_thread:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="mission-recorder", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.active = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None
        self.bus.update(recording={"active": False, "n_samples": self.n})

    def tick(self) -> bool:
        st = self.tracker.get_state()
        if self.mode == "spell":
            ok = st is not None and bool(getattr(st, "wand_ok", False)) and getattr(st, "wand_xyz", None) is not None
            xyz = st.wand_xyz if ok else None
        else:
            ok = st is not None and bool(st.tracking_ok) and st.xyz is not None
            xyz = st.xyz if ok else None
        if not ok:
            if not self._lost_logged:
                self.bus.log("recording: tracking lost, waiting")
                self._lost_logged = True
            return False
        self._lost_logged = False
        x, y, z = (float(v) for v in xyz)
        kept = self.rec.add(self.clock(), x, y, z)
        if kept:
            self.bus.append_live_point(x, y, z)
            self.bus.update(recording={"n_samples": self.n})
        return kept

    def _run(self) -> None:  # pragma: no cover - timing loop; tick() is tested
        nxt = self.clock()
        while not self._stop.is_set() and self.active:
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                self.bus.log(f"recording error: {exc!r}")
            nxt += self.period
            now = self.clock()
            if nxt > now:
                self.sleep(nxt - now)
            else:
                nxt = now


def _disagreement_m(fl: flight.Flight, tracker) -> float:
    st = tracker.get_state()
    if st is None or st.xyz is None:
        return float("inf")
    est = fl.telemetry.position
    return max(abs(float(est[i]) - float(st.xyz[i])) for i in range(3))


def preflight(m) -> Optional[str]:
    """Ordered pre-takeoff checks. Returns the first refusal reason, or None when the drone may fly.

    Reads m.armed, m.link_lost, m.tracker, m.fl, m.supervisor (None in sim), m.cf, m.cfg, m.sleep.
    """
    if not m.armed:
        return "disarmed"
    if not m.cfg.allow_flights:
        return "flights disabled (fake tracker on the real link)"
    if m.link_lost:
        return "link lost"
    st = m.tracker.get_state()
    if st is None or not st.tracking_ok or st.xyz is None:
        return "tracker not locked"
    fl = m.fl
    if not fl.telemetry.converged:
        fl.setup_estimator()
        if not fl.wait_for_estimator(m.cfg.converge_retry_s):
            return "estimator not converged"
    err = _disagreement_m(fl, m.tracker)
    if err > m.cfg.agree_tol_m:
        fl.setup_estimator()                      # one reset with camera input, then re-check
        fl.wait_for_estimator(m.cfg.converge_retry_s)
        err = _disagreement_m(fl, m.tracker)
        if err > m.cfg.agree_tol_m or not fl.telemetry.converged:
            return f"estimate disagrees with camera by {err:.2f} m"
    sup = m.supervisor
    if sup is not None:
        info = sup.info
        if info is not None and info & (hop.BIT_CRASHED | hop.BIT_IS_LOCKED):
            hop.crash_recovery_request(m.cf)
            m.sleep(1.0)
            info = sup.info
        if info is not None and info & (hop.BIT_CRASHED | hop.BIT_IS_LOCKED | hop.BIT_IS_TUMBLED):
            return f"supervisor: {hop.decode_info(info)}"
        vbat = sup.vbat if sup.vbat is not None else fl.telemetry.battery_v
    else:
        vbat = fl.telemetry.battery_v
    if vbat is not None and vbat < m.cfg.min_battery_v:
        return f"battery {vbat:.2f} V"
    return None


class Mission:
    """Owns the drone for the session: commands in, flights out, state published. See the module docstring."""

    def __init__(self, cf, tracker, store: PathStore, *, bus: StateBus | None = None,
                 commands: CommandQueue | None = None, cfg: MissionConfig | None = None,
                 flight_cfg: flight.FlightConfig | None = None, supervisor=None, sim: bool = False,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 inline_flights: bool = False) -> None:
        self.cf, self.tracker, self.store = cf, tracker, store
        self.bus = bus if bus is not None else BUS
        self.commands = commands if commands is not None else COMMANDS
        self.cfg = cfg or MissionConfig()
        self.sim = sim
        self.supervisor = supervisor
        self.clock, self.sleep = clock, sleep
        self.inline_flights = inline_flights
        self.fl = flight.Flight(cf, flight_cfg or flight.FlightConfig(), tracker=tracker, clock=clock, sleep=sleep)
        self.feed = feed.PositionFeed(cf, tracker, clock=clock, sleep=sleep)
        self.rec = RouteRecorder(tracker, self.bus, rate_hz=self.cfg.record_hz, clock=clock, sleep=sleep)

        self.mode = "guide"
        self.armed = False
        self.blocked: set[str] = set()
        self.alarm_active = False
        self.alarm_exit = "A"
        self.active_path: Optional[str] = None
        self.link_lost = False
        self.in_flight = False                       # between preflight pass and touchdown/stop
        self._pending_relaunch: Optional[paths.Path] = None
        self._current_path: Optional[paths.Path] = None
        self._replay: Optional[tuple[paths.Path, float]] = None
        self._flight_thread: threading.Thread | None = None
        self._handled_stops: set[tuple[float, str]] = set()
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_feed_log = self.clock()

        self.fl.on_state = self._on_flight_state
        self.commands.on_push(self._on_push)
        names = self.store.names()
        if names:
            self.active_path = "exit_a" if "exit_a" in names else names[0]
        uri = "sim://fake-drone" if sim else getattr(self.fl.cfg, "link_uri", config.LINK_URI)
        self.bus.update(mode=self.mode, paths=self.store.summaries(), active_path=self.active_path,
                        flight={"state": self.fl.state, "armed": False}, link={"uri": uri})

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.tracker.start()
        self.feed.start()
        if not self.sim:
            self.fl.start_log()
            if self.cfg.ext_std is not None:
                try:
                    self.cf.param.set_value(resolve_name(self.cf.param.toc, "locSrv.extPosStdDev"), str(self.cfg.ext_std))
                except Exception as exc:  # noqa: BLE001
                    self.bus.log(f"could not set locSrv.extPosStdDev: {exc!r}")
            lost = getattr(self.cf, "connection_lost", None)
            if lost is not None and hasattr(lost, "add_callback"):
                lost.add_callback(self._on_connection_lost)
        if self.supervisor is not None:
            self.supervisor.start()
        self.bus.update(link={"connected": True, "connecting": False, "error": ""}, source="sim" if self.sim else "live")
        self._spawn(self._estimator_manager, "mission-estimator")
        self._spawn(self._publish_loop, "mission-publish")
        self._spawn(self._command_loop, "mission-commands")
        self.bus.log("mission ready: ARM on the page, then ALARM or Cast")

    def stop(self) -> None:
        self._shutdown.set()
        for th in self._threads:
            th.join(1.0)
        if self.rec.active:
            self.rec.stop()
        self.feed.stop()
        try:
            self.tracker.stop()
        except Exception:  # noqa: BLE001
            pass
        if self.supervisor is not None:
            self.supervisor.stop()
        if self.fl.logconf is not None:
            try:
                self.fl.logconf.stop()
            except Exception:  # noqa: BLE001
                pass
        close = getattr(self.cf, "close_link", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        self.bus.update(link={"connected": False})

    def _spawn(self, target, name: str) -> None:
        th = threading.Thread(target=target, name=name, daemon=True)
        th.start()
        self._threads.append(th)

    def _estimator_manager(self) -> None:  # pragma: no cover - thread; the reset itself is Flight's
        while not self._shutdown.is_set():
            st = self.tracker.get_state()
            if st is not None and st.tracking_ok and st.xyz is not None:
                self.sleep(0.5)                         # let a few extpos packets in first
                try:
                    self.fl.setup_estimator()
                    self.bus.log("tracker locked: estimator reset with camera position")
                except Exception as exc:  # noqa: BLE001
                    self.bus.log(f"estimator setup failed: {exc!r}")
                return
            self._shutdown.wait(0.2)

    def _publish_loop(self) -> None:  # pragma: no cover - timing loop; publish() is tested
        period = 1.0 / self.cfg.publish_hz
        while not self._shutdown.is_set():
            try:
                self.publish()
            except Exception as exc:  # noqa: BLE001
                self.bus.log(f"publish error: {exc!r}")
            self._shutdown.wait(period)

    def _command_loop(self) -> None:  # pragma: no cover - thread; handle() is tested
        while not self._shutdown.is_set():
            cmd = self.commands.pop(timeout=0.1)
            if cmd is None:
                continue
            try:
                self.handle(cmd)
            except Exception as exc:  # noqa: BLE001
                self.bus.log(f"ERROR handling {cmd.name}: {exc!r}")

    # ------------------------------------------------------------------ callbacks

    def _on_flight_state(self, state: str) -> None:
        with self._lock:
            if state == "flying" and self._current_path is not None:
                self._replay = (self._current_path, self.clock())
            elif state != "flying":
                self._replay = None
            armed = self.armed
        self.bus.update(flight={"state": state, "estimator_converged": bool(self.fl.telemetry.converged), "armed": armed})

    def _on_connection_lost(self, _uri: str, msg: str) -> None:
        with self._lock:
            self.link_lost = True
            self.armed = False
            self._pending_relaunch = None
        self.bus.update(link={"connected": False, "connecting": False, "error": f"connection lost: {msg}"},
                        flight={"armed": False})
        self.bus.log(f"CONNECTION LOST ({msg}): disarmed; power-cycle the drone and restart mission.py")

    def _on_push(self, cmd: Command) -> None:
        """Fast path on the pusher's thread: a stop must never wait for the command loop."""
        if cmd.name == "stop":
            try:
                self._do_stop(cmd)
            except Exception:  # noqa: BLE001
                log.exception("stop fast path failed")

    # ------------------------------------------------------------------ commands

    @staticmethod
    def _tag(cmd: Command) -> str:
        return f"[{cmd.source}] " if cmd.source != "ui" else ""

    def handle(self, cmd: Command) -> None:
        name, args, tag = cmd.name, cmd.args or {}, self._tag(cmd)
        with self._lock:
            if name == "arm":
                self._cmd_arm(args, tag)
            elif name == "set_mode":
                mode = str(args.get("mode", "")).lower()
                if mode not in MODES:
                    self.bus.log(f"{tag}set_mode: unknown mode {mode!r}")
                elif self.rec.active:
                    self.bus.log(f"{tag}set_mode refused: recording in {self.rec.mode} mode")
                else:
                    self.mode = mode
                    self.bus.update(mode=mode)
                    self.bus.log(f"{tag}mode -> {mode}")
            elif name == "select_path":
                p = self.store.get(str(args.get("name", "")))
                if p is None:
                    self.bus.log(f"{tag}select_path: no path {args.get('name')!r}")
                else:
                    self.active_path = p.name
                    self.bus.update(active_path=p.name)
                    self.bus.log(f"{tag}selected {p.name}")
            elif name == "record_start":
                self._cmd_record_start(tag)
            elif name == "record_stop":
                if not self.rec.active:
                    self.bus.log(f"{tag}record_stop: not recording")
                else:
                    self.rec.stop()
                    self._recording_hook(False)
                    self.bus.log(f"{tag}recording stopped: {self.rec.n} samples")
            elif name == "save_as":
                self._cmd_save_as(args, tag)
            elif name == "hover":
                self._cmd_hover(args, tag)
            elif name == "cast":
                p = self.store.get(str(args.get("name", "")))
                if p is None:
                    self.bus.log(f"{tag}cast: no path {args.get('name')!r}")
                elif not self.armed:
                    self.bus.log(f"{tag}cast refused: disarmed (arm first)")
                else:
                    self._start_mission(p, f"cast {p.name}", tag)
            elif name == "alarm":
                self._cmd_alarm(tag)
            elif name == "exit_blocked":
                self._cmd_exit_blocked(args, tag)
            elif name == "land":
                if self.in_flight:
                    self._pending_relaunch = None
                    self.fl.request_land()
                    self.bus.log(f"{tag}landing")
                else:
                    self.bus.log(f"{tag}land: not flying")
            elif name == "stop":
                self._do_stop(cmd)
            elif name == "clear_alarm":
                if self.fl.state == "estop":
                    self.fl.reset_estop()
                self.blocked.clear()
                self.alarm_active = False
                self.bus.update(alarm={"active": False, "blocked_exits": []}, flight={"state": self.fl.state})
                self.bus.log(f"{tag}alarm cleared")
            elif name == "set_source":
                pass  # server infrastructure, never ours
            else:
                self.bus.log(f"{tag}unhandled command {name}")

    def _cmd_arm(self, args: dict, tag: str) -> None:
        on = as_bool(args.get("on", True))
        self.armed = on
        if not on:
            self._pending_relaunch = None
        self.bus.update(flight={"armed": on})
        self.bus.log(f"{tag}{'ARMED - alarm/cast will fly' if on else 'disarmed'}")

    def _recording_hook(self, active: bool) -> None:
        hook = getattr(self.tracker, "on_recording", None)
        if callable(hook):
            hook(active)

    def _cmd_record_start(self, tag: str) -> None:
        if self.rec.active:
            self.bus.log(f"{tag}already recording")
            return
        if self.in_flight or self.fl.state != "idle":
            self.bus.log(f"{tag}record_start refused: flight state {self.fl.state}")
            return
        self.rec.start(self.mode, spawn_thread=not self.inline_flights)
        self._recording_hook(True)
        what = "drone (hand-carried)" if self.mode == "guide" else "wand tip"
        self.bus.log(f"{tag}recording {what} ...")

    def _cmd_save_as(self, args: dict, tag: str) -> None:
        pname = str(args.get("name", "")).strip()
        if not pname:
            self.bus.log(f"{tag}save_as: empty name")
            return
        if self.rec.active:
            self.rec.stop()
            self._recording_hook(False)
        if self.rec.n < 2:
            self.bus.log(f"{tag}save_as: nothing to save (need at least 2 samples, have {self.rec.n})")
            return
        try:
            p = paths.clean_path(self.rec.points, mode=self.rec.mode, name=pname)
        except ValueError as exc:
            self.bus.log(f"{tag}save_as: nothing to save ({exc})")
            return
        file = self.store.save(p)
        self.active_path = p.name
        self.bus.update(paths=self.store.summaries(), active_path=p.name)
        self.bus.log(f"{tag}saved {p.name}: {p.length:.2f} m, {p.duration:.1f} s -> {file}")

    def _open_exit(self) -> str:
        for ex in ("A", "B"):
            if ex not in self.blocked:
                return ex
        return "A"

    def _cmd_hover(self, args: dict, tag: str) -> None:
        if not self.armed:
            self.bus.log(f"{tag}hover refused: disarmed (arm first)")
            return
        try:
            height = float(args.get("height_m", self.cfg.hover_height_m))
            seconds = float(args.get("seconds", self.cfg.hover_hold_s))
        except (TypeError, ValueError):
            self.bus.log(f"{tag}hover: bad arguments {args!r}")
            return
        box = self.fl.cfg.geofence
        height = min(max(height, box.zmin), box.zmax)
        seconds = min(max(seconds, 1.0), 30.0)
        self._start_mission(HoverPlan(height, seconds), f"hover test: {height:.2f} m for {seconds:.0f} s", tag)

    def _cmd_alarm(self, tag: str) -> None:
        if not self.armed:
            self.bus.log(f"{tag}alarm refused: disarmed (arm first)")
            return
        ex = self._open_exit()
        path = self.store.exit_path(ex)
        if path is None:
            path = next((p for p in self.store.all().values() if p.mode == "guide"), None)
            if path is not None:
                self.bus.log(f"{tag}no exit_{ex.lower()} stored; using {path.name}")
        if path is None:
            self.bus.log(f"{tag}ALARM but no exit route stored!")
            return
        self.mode = "guide"
        self.alarm_active = True
        self.alarm_exit = ex
        self.bus.update(mode="guide", alarm={"active": True, "exit": ex, "blocked_exits": sorted(self.blocked)})
        self._start_mission(path, f"ALARM: leading out via exit {ex}", tag)

    def _cmd_exit_blocked(self, args: dict, tag: str) -> None:
        ex = str(args.get("exit", "A")).strip().upper()[:1] or "A"
        if ex not in ("A", "B"):
            self.bus.log(f"{tag}exit_blocked: unknown exit {ex!r}")
            return
        self.blocked.add(ex)
        other = "B" if ex == "A" else "A"
        self.bus.update(alarm={"blocked_exits": sorted(self.blocked)})
        self.bus.log(f"{tag}exit {ex} blocked")
        if not (self.alarm_active and self.alarm_exit == ex):
            return
        path = self.store.exit_path(other)
        if path is None:
            self.bus.log(f"no exit_{other.lower()} path stored; cannot reroute")
            return
        if self.in_flight:
            self._pending_relaunch = path
            self.fl.request_land()
            self.bus.log(f"REROUTE -> exit {other} ({path.name}): landing first, relaunch in {self.cfg.relaunch_delay_s:.0f} s")
        else:
            self.alarm_exit = other
            self.bus.update(alarm={"exit": other})
            self.bus.log(f"REROUTE -> exit {other} ({path.name})")
            self._start_mission(path, f"reroute to exit {other}", tag)

    def _do_stop(self, cmd: Command) -> None:
        tag = self._tag(cmd)
        with self._lock:
            key = (cmd.ts, cmd.source)
            if key in self._handled_stops:
                return                                   # the queued copy of a stop already handled on push
            self._handled_stops.add(key)
            if len(self._handled_stops) > 200:
                self._handled_stops.clear()
                self._handled_stops.add(key)
            self.armed = False
            self._pending_relaunch = None
            in_flight = self.in_flight
        self.fl.request_stop()
        if not in_flight:
            self.fl.emergency_stop()                     # nobody is streaming: send the 3 stops ourselves
        self.bus.update(flight={"armed": False, "state": self.fl.state})
        self.bus.log(f"{tag}EMERGENCY STOP sent (motors off)")

    # ------------------------------------------------------------------ flight worker

    def _start_mission(self, path, why: str, tag: str = "") -> bool:
        """path: a paths.Path to fly, or a HoverPlan."""
        hover = isinstance(path, HoverPlan)
        with self._lock:
            busy = self.in_flight or (self._flight_thread is not None and self._flight_thread.is_alive())
            if busy or self.fl.state != "idle":
                self.bus.log(f"{tag}{why} refused: flight in progress ({self.fl.state})")
                return False
            if not hover:
                self._current_path = path
                self.active_path = path.name
        if hover:
            self.bus.log(f"{tag}{why}")
        else:
            self.bus.update(active_path=path.name)
            self.bus.log(f"{tag}{why}: {path.name} ({path.length:.2f} m, {path.duration:.1f} s)")
        if self.inline_flights:
            self._worker(path)
            return True
        self._flight_thread = threading.Thread(target=self._worker, args=(path,), name="mission-flight", daemon=True)
        self._flight_thread.start()
        return True

    def _worker(self, path: paths.Path) -> None:
        flew = False
        try:
            while path is not None:
                reason = preflight(self)
                if reason:
                    self.bus.log(f"refused: {reason}")
                    break
                hover = isinstance(path, HoverPlan)
                with self._lock:
                    if not self.armed:                   # a stop arrived during preflight
                        self.bus.log("mission cancelled")
                        break
                    self._current_path = None if hover else path
                    if not hover:
                        self.active_path = path.name
                    self.in_flight = True
                flew = True
                if not hover:
                    self.bus.update(active_path=path.name)
                if self.supervisor is not None:
                    self.supervisor.stop()               # one log block in flight
                saved_height = self.fl.cfg.takeoff_height
                try:
                    if hover:
                        self.fl.cfg.takeoff_height = path.height_m
                    if not self.fl.takeoff():
                        self.bus.log(f"takeoff aborted: {self.fl.last_abort_reason}")
                    elif hover:
                        self.bus.log(f"hovering at {path.height_m:.2f} m for {path.seconds:.0f} s")
                        if not self.fl.hold(path.seconds):
                            self.bus.log(f"hover aborted: {self.fl.last_abort_reason}")
                        else:
                            self.fl.land()
                            self.bus.log("landed after hover test")
                    elif not self.fl.fly_path(path, land=True):
                        self.bus.log(f"flight aborted: {self.fl.last_abort_reason}")
                    else:
                        self.bus.log(f"landed after {path.name}")
                finally:
                    self.fl.cfg.takeoff_height = saved_height
                    with self._lock:
                        self.in_flight = False
                    if self.supervisor is not None:
                        self.supervisor.start()
                with self._lock:
                    nxt, self._pending_relaunch = self._pending_relaunch, None
                    relaunch = nxt is not None and self.fl.state == "idle" and self.armed and not hover
                    if relaunch:
                        ex = exit_letter(nxt.name)
                        if ex:
                            self.alarm_exit = ex
                if relaunch:
                    if exit_letter(nxt.name):
                        self.bus.update(alarm={"exit": self.alarm_exit})
                    self.bus.log(f"relaunch -> {nxt.name} in {self.cfg.relaunch_delay_s:.0f} s")
                    self.sleep(self.cfg.relaunch_delay_s)
                    path = nxt
                else:
                    path = None
        except Exception as exc:  # noqa: BLE001
            try:
                self.fl.emergency_stop()
            except Exception:  # noqa: BLE001
                log.exception("emergency stop after worker failure failed")
            self.bus.log(f"!! {exc!r} -> EMERGENCY STOP")
            flew = True
        finally:
            with self._lock:
                if flew:
                    self.armed = False
                self.in_flight = False
                self._current_path = None
                self._replay = None
                armed = self.armed
            self.bus.update(flight={"armed": armed, "state": self.fl.state})

    # ------------------------------------------------------------------ publishing

    def publish(self) -> None:
        st = self.tracker.get_state()
        fl, tel = self.fl, self.fl.telemetry
        ok = st is not None and bool(st.tracking_ok) and st.xyz is not None
        if ok:
            x, y, z = (float(v) for v in st.xyz)
            yaw = float(st.yaw) if getattr(st, "yaw", None) is not None else 0.0
        else:
            x, y, z = tel.position
            yaw = 0.0
        wand_ok = st is not None and bool(getattr(st, "wand_ok", False)) and getattr(st, "wand_xyz", None) is not None
        wand = {"x": float(st.wand_xyz[0]), "y": float(st.wand_xyz[1]), "z": float(st.wand_xyz[2]), "ok": True} if wand_ok else {"ok": False}
        with self._lock:
            rp, armed, mode, active = self._replay, self.armed, self.mode, self.active_path
            alarm = {"active": self.alarm_active, "exit": self.alarm_exit, "blocked_exits": sorted(self.blocked)}
            lost = self.link_lost
        if rp is not None and fl.state == "flying":
            path, t0 = rp
            dur = float(path.duration)
            t = min(max(self.clock() - t0, 0.0), dur) if dur > 0 else 0.0
            replay = {"active": True, "t": t, "duration": dur, "progress": (t / dur) if dur > 0 else 1.0}
        else:
            replay = {"active": False, "t": 0.0, "duration": 0.0, "progress": 0.0}
        bat = tel.battery_v
        self.bus.update(
            mode=mode, active_path=active,
            drone={"x": x, "y": y, "z": z, "yaw": yaw},
            tracking={"ok": ok, "fps": float(getattr(st, "fps", 0.0) or 0.0) if st is not None else 0.0,
                      "latency_ms": float(getattr(st, "latency_ms", 0.0) or 0.0) if st is not None else 0.0},
            wand=wand,
            link={"connected": not lost, "battery_v": float(bat) if bat is not None else 0.0},
            flight={"state": fl.state, "estimator_converged": bool(tel.converged), "armed": armed},
            replay=replay, alarm=alarm,
            recording={"active": self.rec.active, "mode": self.rec.mode},
        )
        now = self.clock()
        if now - self._last_feed_log >= self.cfg.feed_status_every_s:
            self._last_feed_log = now
            self.bus.log(self.feed.status())


def _make_tracker(kind: str, camera: Optional[int]):
    import hover  # cv2 lives behind this import; keep mission importable without it

    return hover.make_tracker(kind, 0 if camera is None else camera)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="fake drone that follows its setpoints; demo paths; no link")
    ap.add_argument("--tracker", choices=["aruco", "sim"], default="aruco", help="real link: camera tracker or SimTracker")
    ap.add_argument("--camera", type=int, default=None, help="--tracker aruco: camera index (default: calib/tracker.json)")
    ap.add_argument("--uri", default=None, help=f"override config.LINK_URI ({config.LINK_URI})")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the page from a phone on the same Wi-Fi")
    ap.add_argument("--paths-dir", default=config.PATHS_DIR)
    ap.add_argument("--ext-std", type=float, default=0.05, help="locSrv.extPosStdDev (m)")
    ap.add_argument("--no-ui", action="store_true", help="do not serve the page from this process")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = MissionConfig(ext_std=args.ext_std, allow_flights=args.sim or args.tracker != "sim")
    feeder = None
    if args.sim:
        import mission_sim

        store = PathStore(args.paths_dir, demo=True)
        cf = mission_sim.SimDrone()
        tracker = mission_sim.SimDroneTracker(cf)
        mission = Mission(cf, tracker, store, cfg=cfg, sim=True)
        feeder = mission_sim.SimTelemetry(mission.fl, cf).start()
        print("[mission] SIM: fake drone, demo paths (exit_a, exit_b, spiral, square)")
    else:
        import hover

        store = PathStore(args.paths_dir, demo=False)
        tracker = _make_tracker(args.tracker, args.camera)
        uri = args.uri or config.LINK_URI
        print(f"[mission] connecting to {uri} (about 50 s over BLE) ...")
        cf = flight.connect(uri)
        mission = Mission(cf, tracker, store, cfg=cfg, flight_cfg=flight.FlightConfig(link_uri=uri),
                          supervisor=hover.SupervisorWatch(cf))
        print(f"[mission] connected; tracker={args.tracker}; paths: {', '.join(store.names()) or 'none yet'}")

    if not args.no_ui:
        from ui.server import SourceManager, create_app, run_server

        app = create_app(mission.bus, mission.commands, store, manager=SourceManager(mission.bus, mission.commands))
        threading.Thread(target=run_server, args=(app,), kwargs={"host": args.host, "port": args.port, "log_level": "warning"},
                         name="mission-ui", daemon=True).start()
        print(f"[mission] UI: http://{args.host}:{args.port}/")

    mission.start()
    print("[mission] ready: ARM on the page, then ALARM or Cast. Ctrl-C to quit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[mission] shutting down")
    finally:
        if feeder is not None:
            feeder.stop()
        mission.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
