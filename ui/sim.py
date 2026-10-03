"""Hardware-free simulator behind ``python -m ui.server --sim``.

A 30 Hz thread that plays the role of tracker + feed + flight + path engine so
the operator UI can be exercised end to end:

* drains the ``CommandQueue`` and reacts to every command in the vocabulary
* ``alarm`` / ``cast`` fly the drone dot along a stored path (takeoff -> fly ->
  land) using the path's own times; ``exit_blocked A`` mid-flight reroutes to
  ``exit_b`` (and vice versa)
* ``record_start`` fakes a hand-carried drone (guide) or a wand drawing (spell)
  by following a demo shape with noise and appending live points;
  ``record_stop`` + ``save_as`` run the REAL ``paths.clean_path`` and
  ``save_path`` and refresh the path list
* drops the tracking flag for a moment every few seconds, drains the battery,
  and writes a log line for everything it does

Nothing here touches hardware; it is the reference for what the real
flight/tracker code should publish on the bus.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

import config
from paths import Path, clean_path
from ui.pathstore import PathStore
from ui.state import LIVE_POINTS_MAX, MODES, CommandQueue, StateBus

__all__ = ["Simulator"]

GROUND_Z = 0.02
HOVER_SPEED = config.REPLAY_SPEED_MPS  # m/s, used for takeoff / transit legs


def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    u = min(1.0, max(0.0, u))
    return a + (b - a) * u


class Simulator(threading.Thread):
    def __init__(
        self,
        bus: StateBus,
        commands: CommandQueue,
        store: PathStore,
        rate_hz: float = 30.0,
        time_scale: float = 1.0,
        glitch_period_s: float = 9.0,
    ) -> None:
        super().__init__(name="pathcaster-sim", daemon=True)
        self.bus = bus
        self.commands = commands
        self.store = store
        self.dt = 1.0 / rate_hz
        self.time_scale = time_scale
        self.glitch_period = glitch_period_s
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._step_lock = threading.Lock()
        self._hw_last = -1.0
        self.rng = np.random.default_rng(7)

        # physical state
        self.t = 0.0
        self.pos = np.array([-0.6, -0.4, GROUND_Z])
        self.yaw = 0.0
        self.wand = np.array([0.0, 0.0, 0.8])
        self.battery = 4.10

        # flight state machine
        self.phase: str = "idle"  # idle | takeoff | transit | flying | landing | estop
        self.path: Path | None = None
        self.replay_t = 0.0
        self.leg_from = self.pos.copy()
        self.leg_to = self.pos.copy()
        self.leg_dur = 1.0
        self.leg_t = 0.0
        self.after_leg = "flying"

        # recording
        self.rec_active = False
        self.rec_mode = "guide"
        self.rec_src: Path | None = None
        self.rec_t = 0.0
        self.rec_origin = self.pos.copy()
        self.rec_raw: list[tuple[float, float, float, float]] = []

        self.alarm_active = False
        self.alarm_exit = "A"
        self.blocked: list[str] = []
        self.converged_at = 2.0
        self.mode = "guide"
        self.active_path: str | None = None

    # ----------------------------------------------------------------- control

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout)

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        """Park the simulator: no publishing, no command draining (LIVE mode takes over).

        Returns only after a step in progress has finished, so nothing from the
        simulator reaches the bus after this call.
        """
        self._paused.set()
        with self._step_lock:
            pass

    def resume(self) -> None:
        """Resume publishing; starts the thread on first use."""
        self._paused.clear()
        if not self.is_alive() and not self._stop.is_set():
            self.start()

    def run(self) -> None:  # pragma: no cover - timing loop; logic is in step()
        self.setup()
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                next_tick = time.perf_counter()
                continue
            with self._step_lock:
                if not self._paused.is_set():
                    self.step(self.dt * self.time_scale)
            next_tick += self.dt
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    def setup(self) -> None:
        names = self.store.names()
        self.active_path = "exit_a" if "exit_a" in names else (names[0] if names else None)
        self.bus.update(
            mode=self.mode,
            link={"uri": config.LINK_URI, "connected": True, "battery_v": self.battery, "rssi": -58},
            tracking={"ok": True, "fps": 30.0, "latency_ms": 33.0},
            drone={"x": self.pos[0], "y": self.pos[1], "z": self.pos[2], "yaw": 0.0},
            paths=self.store.summaries(),
            active_path=self.active_path,
            flight={"state": "idle", "estimator_converged": False},
        )
        self.bus.log(f"SIM started: {len(names)} paths ({', '.join(names)})")
        self.bus.log(f"link {config.LINK_URI} connected (simulated)")

    # ------------------------------------------------------------------- tick

    def step(self, dt: float) -> None:
        self.t += dt
        for cmd in self.commands.drain():
            try:
                self.handle(cmd.name, cmd.args, cmd.source)
            except Exception as exc:  # noqa: BLE001 - never let a bad command kill the sim
                self.bus.log(f"ERROR handling {cmd.name}: {exc}")
        self.advance(dt)
        self.publish()

    def handle(self, name: str, args: dict[str, Any], source: str) -> None:
        tag = f"[{source}] " if source != "ui" else ""
        if name == "set_mode":
            mode = str(args.get("mode", "")).lower()
            if mode not in MODES:
                self.bus.log(f"{tag}set_mode: unknown mode {mode!r}")
                return
            self.mode = mode
            self.bus.log(f"{tag}mode -> {mode}")
        elif name == "select_path":
            pname = str(args.get("name", ""))
            if self.store.get(pname) is None:
                self.bus.log(f"{tag}select_path: no path {pname!r}")
                return
            self.active_path = self.store.get(pname).name
            self.bus.log(f"{tag}selected {self.active_path}")
        elif name == "alarm":
            self.mode = "guide"
            self.alarm_active = True
            self.alarm_exit = self._open_exit()
            path = self.store.exit_path(self.alarm_exit) or self._fallback_path("guide")
            if path is None:
                self.bus.log(f"{tag}ALARM but no exit route stored!")
                return
            self.bus.log(f"{tag}ALARM: leading out via exit {self.alarm_exit} ({path.name})")
            self._start_flight(path)
        elif name == "cast":
            pname = str(args.get("name", ""))
            path = self.store.get(pname)
            if path is None:
                self.bus.log(f"{tag}cast: no path {pname!r}")
                return
            self.bus.log(f"{tag}cast {path.name} ({path.length:.2f} m, {path.duration:.1f} s)")
            self._start_flight(path)
        elif name == "exit_blocked":
            ex = str(args.get("exit", "A")).strip().upper()[:1] or "A"
            if ex not in self.blocked:
                self.blocked.append(ex)
            other = "B" if ex == "A" else "A"
            self.bus.log(f"{tag}exit {ex} blocked")
            if self.alarm_active and self.alarm_exit == ex:
                path = self.store.exit_path(other)
                if path is None:
                    self.bus.log(f"no exit_{other.lower()} path stored; cannot reroute")
                    return
                self.alarm_exit = other
                self.bus.log(f"REROUTE -> exit {other} ({path.name})")
                if self.phase in ("takeoff", "transit", "flying"):
                    self._start_leg(path.position_at(0.0), "flying", path=path, phase="transit")
                else:
                    self._start_flight(path)
        elif name == "land":
            if self.phase in ("takeoff", "transit", "flying"):
                self.bus.log(f"{tag}landing")
                self._start_landing()
            else:
                self.bus.log(f"{tag}land: not flying")
        elif name == "stop":
            self.phase = "estop"
            self.bus.log(f"{tag}EMERGENCY STOP (motors off)")
        elif name == "clear_alarm":
            self.alarm_active = False
            self.blocked = []
            if self.phase == "estop":
                self.phase = "idle"
                self.pos[2] = GROUND_Z
            self.bus.log(f"{tag}alarm cleared")
        elif name == "record_start":
            if self.rec_active:
                self.bus.log("already recording")
                return
            self.rec_mode = self.mode
            self.rec_src = self._record_source(self.rec_mode)
            if self.rec_src is None:
                self.bus.log("record_start: no demo shape to follow")
                return
            self.rec_active = True
            self.rec_t = 0.0
            self.rec_raw = []
            self.rec_origin = (self.pos if self.rec_mode == "guide" else self.wand).copy()
            self.bus.update(recording={"active": True, "mode": self.rec_mode, "n_samples": 0, "live_points": []})
            what = "drone (hand-carried)" if self.rec_mode == "guide" else "wand tip"
            self.bus.log(f"{tag}recording {what} ...")
        elif name == "record_stop":
            if not self.rec_active:
                self.bus.log("record_stop: not recording")
                return
            self.rec_active = False
            self.bus.log(f"{tag}recording stopped: {len(self.rec_raw)} samples")
        elif name == "save_as":
            pname = str(args.get("name", "")).strip()
            if not pname:
                self.bus.log("save_as: empty name")
                return
            if len(self.rec_raw) < 2:
                self.bus.log("save_as: nothing recorded yet")
                return
            raw = np.array([(x, y, z) for (_t, x, y, z) in self.rec_raw])
            path = clean_path(raw, mode=self.rec_mode, name=pname)
            file = self.store.save(path)
            self.active_path = path.name
            self.bus.update(paths=self.store.summaries())
            self.bus.log(f"{tag}saved {path.name}: {path.length:.2f} m, {path.duration:.1f} s -> {file}")
        elif name == "set_source":
            pass  # handled by ui.server.SourceManager
        else:
            self.bus.log(f"{tag}unhandled command {name}")

    # ------------------------------------------------------------- flight sim

    def _open_exit(self) -> str:
        for ex in ("A", "B"):
            if ex not in self.blocked:
                return ex
        return "A"

    def _fallback_path(self, mode: str) -> Path | None:
        for p in self.store.all().values():
            if p.mode == mode:
                return p
        return None

    def _record_source(self, mode: str) -> Path | None:
        active = self.store.get(self.active_path) if self.active_path else None
        if active is not None and active.mode == mode:
            return active
        pref = "exit_a" if mode == "guide" else "spiral"
        return self.store.get(pref) or self._fallback_path(mode)

    def _start_flight(self, path: Path) -> None:
        self.path = path
        self.active_path = path.name
        self.rec_active = False
        start = path.position_at(0.0)
        if self.phase in ("takeoff", "transit", "flying"):
            self._start_leg(start, "flying", path=path, phase="transit")
        else:
            self.pos[2] = GROUND_Z
            self._start_leg(start, "flying", path=path, phase="takeoff")

    def _start_leg(self, target: np.ndarray, after: str, *, path: Path | None, phase: str) -> None:
        self.path = path if path is not None else self.path
        if path is not None:
            self.active_path = path.name
        self.leg_from = self.pos.copy()
        self.leg_to = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(self.leg_to - self.leg_from))
        self.leg_dur = max(1.0, dist / HOVER_SPEED)
        self.leg_t = 0.0
        self.after_leg = after
        self.phase = phase
        self.replay_t = 0.0

    def _start_landing(self) -> None:
        target = self.pos.copy()
        target[2] = GROUND_Z
        self._start_leg(target, "idle", path=None, phase="landing")

    def advance(self, dt: float) -> None:
        prev = self.pos.copy()
        if self.phase in ("takeoff", "transit", "landing"):
            self.leg_t += dt
            self.pos = _lerp(self.leg_from, self.leg_to, self.leg_t / self.leg_dur)
            if self.leg_t >= self.leg_dur:
                if self.after_leg == "flying" and self.path is not None:
                    self.phase = "flying"
                    self.replay_t = 0.0
                    self.bus.log(f"flying {self.path.name}")
                else:
                    self.phase = "idle"
                    self.bus.log("landed")
        elif self.phase == "flying" and self.path is not None:
            self.replay_t += dt
            self.pos = self.path.position_at(self.replay_t)
            if self.replay_t >= self.path.duration:
                self.bus.log(f"{self.path.name} complete")
                self._start_landing()
        elif self.phase == "estop":
            self.pos[2] = max(GROUND_Z, self.pos[2] - 2.5 * dt)  # falls
        elif self.phase == "idle" and self.rec_active and self.rec_mode == "guide":
            self.pos = self._recorded_point(dt)

        # yaw follows the velocity while airborne, else relaxes to 0 (nose along +x)
        vel = (self.pos - prev) / max(dt, 1e-6)
        if self.phase in ("flying", "transit") and np.hypot(vel[0], vel[1]) > 0.05:
            self.yaw = math.atan2(vel[1], vel[0])
        else:
            self.yaw *= 0.9

        # wand: visible in spell mode (drawing when recording, idling otherwise)
        if self.mode == "spell":
            if self.rec_active and self.rec_mode == "spell":
                self.wand = self._recorded_point(dt)
            else:
                self.wand = np.array(
                    [0.3 * math.sin(0.7 * self.t), 0.3 * math.sin(0.9 * self.t + 1.0), 0.8 + 0.1 * math.cos(0.5 * self.t)]
                )

        # battery: ~7 min of flight
        drain = (0.6 / 420.0) if self.phase in ("takeoff", "transit", "flying") else (0.6 / 7200.0)
        self.battery = max(3.3, self.battery - drain * dt)

    def _recorded_point(self, dt: float) -> np.ndarray:
        """Next hand-carried / wand sample along the source shape, with 5 mm noise."""
        assert self.rec_src is not None
        self.rec_t += dt
        if self.rec_t > self.rec_src.duration:
            # shape finished: hold still, stop appending
            return self.rec_src.position_at(self.rec_src.duration)
        target = self.rec_src.position_at(self.rec_t)
        blend = min(1.0, self.rec_t / 1.0)  # first second: hand moves to the start
        p = _lerp(self.rec_origin, target, blend) + self.rng.normal(0.0, 0.005, 3)
        self.rec_raw.append((self.t, float(p[0]), float(p[1]), float(p[2])))
        self.bus.append_live_point(*p)
        return p

    # ---------------------------------------------------------------- publish

    def publish(self) -> None:
        glitch = (self.t % self.glitch_period) < 0.3 and self.t > 5.0
        replay_active = self.phase == "flying" and self.path is not None
        duration = float(self.path.duration) if self.path is not None else 0.0
        rt = min(self.replay_t, duration) if replay_active else 0.0
        flight_state = {
            "idle": "idle", "takeoff": "takeoff", "transit": "flying", "flying": "flying",
            "landing": "landing", "estop": "estop",
        }[self.phase]
        # fake desk hardware check (what ui.live publishes from stabilizer.roll/pitch): 5 Hz while idle
        hw: dict[str, Any]
        if self.phase == "idle":
            if self.t - self._hw_last >= 0.2:
                self._hw_last = self.t
                hw = {
                    "active": True,
                    "roll_deg": round(8.0 * math.sin(0.8 * self.t), 2),
                    "pitch_deg": round(5.0 * math.sin(0.55 * self.t + 1.0), 2),
                    "ts": time.time(),
                }
            else:
                hw = {"active": True}
        else:
            self._hw_last = -1.0
            hw = {"active": False}
        self.bus.update(
            mode=self.mode,
            hwcheck=hw,
            # link.error is left alone: a failed LIVE connect must stay visible after the fallback
            link={"uri": config.LINK_URI, "connected": True, "connecting": False,
                  "battery_v": round(self.battery, 3), "rssi": int(-58 + 4 * math.sin(self.t))},
            tracking={
                "ok": not glitch,
                "fps": round(29.5 + 0.5 * math.sin(self.t * 2.0), 1),
                "latency_ms": round(33.0 + 4.0 * math.sin(self.t * 1.3) + (40.0 if glitch else 0.0), 1),
            },
            drone={"x": float(self.pos[0]), "y": float(self.pos[1]), "z": float(self.pos[2]), "yaw": float(self.yaw)},
            wand={"x": float(self.wand[0]), "y": float(self.wand[1]), "z": float(self.wand[2]), "ok": self.mode == "spell" and not glitch},
            recording={"active": self.rec_active, "mode": self.rec_mode},
            active_path=self.active_path,
            replay={
                "active": replay_active,
                "t": round(rt, 3),
                "duration": round(duration, 3),
                "progress": round(rt / duration, 4) if duration > 0 else 0.0,
            },
            alarm={"active": self.alarm_active, "exit": self.alarm_exit, "blocked_exits": list(self.blocked)},
            flight={"state": flight_state, "estimator_converged": self.t > self.converged_at and self.phase != "estop"},
        )
