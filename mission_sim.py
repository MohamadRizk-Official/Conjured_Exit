"""Hardware-free stand-ins for ``mission.py --sim``: a drone that follows its setpoints, a tracker that
sees it (and a "hand" that carries it while recording), and a telemetry feeder for ``flight.Telemetry``.
Nothing here touches cflib, bleak or OpenCV."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

import flight
import paths


@dataclass
class SimState:
    """The subset of ``tracker.TrackerState`` that ``mission.Mission`` and ``flight.Flight`` read."""

    t: float
    xyz: Optional[tuple[float, float, float]]
    yaw: Optional[float]
    tracking_ok: bool
    wand_xyz: Optional[tuple[float, float, float]]
    wand_ok: bool
    fps: float = 30.0
    latency_ms: float = 5.0
    mode: str = "sim"


class SimDrone(flight.FakeCrazyflie):
    """A FakeCrazyflie whose position jumps to every position setpoint; a stop setpoint puts it on the floor."""

    def __init__(self, start: tuple[float, float, float] = (-0.6, -0.4, 0.0)) -> None:
        super().__init__()
        self.pos = np.array(start, dtype=float)
        self.battery_v = 4.0
        drone = self

        class _Cmd(flight.FakeCrazyflie._Commander):
            def send_position_setpoint(self, x, y, z, yaw):
                super().send_position_setpoint(x, y, z, yaw)
                drone.pos[:] = (x, y, z)

            def send_stop_setpoint(self):
                super().send_stop_setpoint()
                drone.pos[2] = 0.0

        self.commander = _Cmd()


class SimDroneTracker:
    """Reports the SimDrone's position (plus noise) as a tracker would, with a wand circling a demo path.

    ``on_recording(True)`` on a grounded drone carries it along ``carry_path`` (the hand-carried teach
    walk); ``on_recording(False)`` stops the hand and puts the drone down."""

    def __init__(self, drone: SimDrone, *, wand_path: Any = "spiral", carry_path: Any = "exit_a",
                 noise_m: float = 0.005, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep, seed: int = 0) -> None:
        demos = paths.demo_paths()
        self.drone = drone
        self.wand_path = demos[wand_path] if isinstance(wand_path, str) else wand_path
        self.carry_path = demos[carry_path] if isinstance(carry_path, str) else carry_path
        self.noise_m = float(noise_m)
        self.clock = clock
        self.sleep = sleep
        self.rng = np.random.default_rng(seed)
        self.t0 = clock()
        self.carrying = False
        self._carry_stop = threading.Event()
        self._carry_thread: threading.Thread | None = None

    @property
    def mode(self) -> str:
        return "sim"

    def start(self, open_timeout: float = 0.0) -> "SimDroneTracker":
        self.t0 = self.clock()
        return self

    def stop(self) -> None:
        self.on_recording(False)

    def get_state(self, now: float | None = None) -> SimState:
        now = self.clock() if now is None else now
        noise = self.rng.normal(0.0, self.noise_m, 3) if self.noise_m > 0 else np.zeros(3)
        xyz = tuple(float(v) for v in self.drone.pos + noise)
        dur = max(float(self.wand_path.duration), 1e-9)
        wand = self.wand_path.position_at((now - self.t0) % dur)
        return SimState(t=now, xyz=xyz, yaw=0.0, tracking_ok=True,
                        wand_xyz=tuple(float(v) for v in wand), wand_ok=True)

    def carry_step(self, t: float) -> None:
        """Move the drone to the carry path's position at time ``t`` (used by the hand thread and tests)."""
        self.drone.pos[:] = self.carry_path.position_at(min(float(t), float(self.carry_path.duration)))

    def on_recording(self, active: bool) -> None:
        if active:
            if self.drone.pos[2] > 0.05 or (self._carry_thread is not None and self._carry_thread.is_alive()):
                return
            self._carry_stop.clear()
            self._carry_thread = threading.Thread(target=self._carry, name="sim-hand", daemon=True)
            self._carry_thread.start()
        else:
            self._carry_stop.set()
            if self._carry_thread is not None:
                self._carry_thread.join(1.0)
                self._carry_thread = None

    def _carry(self) -> None:
        self.carrying = True
        t0 = self.clock()
        dur = float(self.carry_path.duration)
        try:
            while not self._carry_stop.is_set():
                t = self.clock() - t0
                self.carry_step(t)
                if t >= dur:
                    break
                self.sleep(1.0 / 30.0)
        finally:
            self.drone.pos[2] = 0.0  # put it down
            self.carrying = False


class SimTelemetry:
    """Feeds ``fl.telemetry`` from the SimDrone at 10 Hz with a tiny constant variance (so convergence passes)."""

    def __init__(self, fl: flight.Flight, drone: SimDrone, *, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.fl = fl
        self.drone = drone
        self.clock = clock
        self.sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> None:
        x, y, z = (float(v) for v in self.drone.pos)
        self.fl.telemetry.update(int(self.clock() * 1000), {
            "kalman.stateX": x, "kalman.stateY": y, "kalman.stateZ": z,
            "kalman.varPX": 1e-4, "kalman.varPY": 1e-4, "kalman.varPZ": 1e-4,
            "pm.vbat": float(self.drone.battery_v),
        })

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self.sleep(0.1)

    def start(self) -> "SimTelemetry":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None
