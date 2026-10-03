"""feed.py -- stream the camera tracker's pose into the drone's onboard Kalman filter.

Uplink only, fire-and-forget, ~30 Hz (config.EXTPOS_RATE_HZ):
  * default (use_yaw=False) -> cf.extpos.send_extpos(x, y, z)             (13-byte packet)
  * use_yaw=True and yaw known -> cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)  (29 bytes)
  * tracking not ok or state older than max_age_s -> nothing is sent (the EKF must not be fed
    stale or guessed positions).

Why position-only by default: nRF firmware 2024.10 corrupts every BLE packet longer than
20 bytes (README.md, "Verified BLE facts"), and the 29-byte pose packet would reach the drone
with a byte missing. With position-only input the heading is unobservable while hovering and
drifts with the gyro bias; keep flights short (the demo route is < 1 min) and start with the
nose along +x. Switch use_yaw=True once the nRF firmware is updated.

Usage:
    feed = PositionFeed(cf, tracker)      # tracker.get_state() -> .t .xyz .yaw .tracking_ok
    feed.start()                          # background thread
    ...
    feed.stop()
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

import config

log = logging.getLogger("feed")


def yaw_to_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    """Rotation about world z only: (qx, qy, qz, qw)."""
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


class PositionFeed:
    def __init__(self, cf, tracker, rate_hz: float = config.EXTPOS_RATE_HZ, use_yaw: bool = False,
                 max_age_s: float = 0.2, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cf = cf
        self.tracker = tracker
        self.rate_hz = rate_hz
        self.use_yaw = use_yaw
        self.max_age_s = max_age_s
        self.clock = clock
        self.sleep = sleep
        self.sent = 0
        self.skipped = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> bool:
        """Send one pose if the tracker has a fresh, valid fix. Returns True when something was sent."""
        st = self.tracker.get_state()
        if st is None or not st.tracking_ok or (self.clock() - st.t) > self.max_age_s:
            self.skipped += 1
            return False
        x, y, z = (float(v) for v in st.xyz)
        try:
            if self.use_yaw and st.yaw is not None:
                qx, qy, qz, qw = yaw_to_quaternion(float(st.yaw))
                self.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            else:
                self.cf.extpos.send_extpos(x, y, z)
        except Exception:  # noqa: BLE001
            self.errors += 1
            log.exception("extpos send failed")
            return False
        self.sent += 1
        return True

    def _run(self) -> None:
        dt = 1.0 / self.rate_hz
        nxt = self.clock()
        while not self._stop.is_set():
            self.tick()
            nxt += dt
            now = self.clock()
            if nxt > now:
                self.sleep(nxt - now)
            else:
                nxt = now  # fell behind: do not burst to catch up

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="position-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def status(self) -> str:
        return f"extpos sent {self.sent} skipped {self.skipped} errors {self.errors}"
