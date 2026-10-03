"""hover.py -- autonomous hover: camera position -> onboard Kalman -> position-setpoint takeoff,
hold, land. The first "on its own" flight.

Needs a working tracker (`python tracker.py --aruco --show` must say mode aruco / tracking ok).
Sequence:
  connect -> Kalman + 10 Hz log -> tracker running and locked -> extpos feed at 30 Hz
  -> estimator reset + convergence (variance window AND estimate within 0.2 m of the camera)
  -> crash recovery / supervisor check -> type FLY -> Flight.takeoff (ramp to height at the
     current xy, position setpoints at 20 Hz) -> hold -> Flight.land -> stop.
Safety (flight.py): geofence clamp, tracking lost > 0.3 s -> land, spacebar -> emergency stop
x3, 'l' -> land now, any exception -> emergency stop.

    python hover.py --tracker aruco --camera 0 --height 0.5 --hold 8
    python hover.py --tracker sim --dry-run          # no hardware: fake drone + simulated tracker

Stop ui.server first (one BLE link). Launch from a real console (cmd /c start ...).
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

import config
import feed
import flight
import hop
from toc_names import resolve

log = logging.getLogger("hover")


class SupervisorWatch:
    """Tiny pre-flight log block (supervisor bits + battery); stopped before takeoff."""

    def __init__(self, cf) -> None:
        self.cf = cf
        self.info = None
        self.vbat = None
        self.conf = None

    def start(self) -> None:
        conf = flight.make_log_config("pre", 100)
        alias = {}
        for name, fetch_as in (("supervisor.info", "uint16_t"), ("pm.vbat", "FP16")):
            seen = resolve(self.cf.log.toc, name)
            conf.add_variable(seen, fetch_as)
            alias[seen] = name

        def on_data(_ts, data, _lc):
            d = {alias.get(k, k): v for k, v in data.items()}
            if "supervisor.info" in d:
                self.info = int(d["supervisor.info"])
            if "pm.vbat" in d:
                self.vbat = float(d["pm.vbat"])

        conf.data_received_cb.add_callback(on_data)
        self.cf.log.add_config(conf)
        conf.start()
        self.conf = conf

    def stop(self) -> None:
        if self.conf is not None:
            try:
                self.conf.stop()
            except Exception:  # noqa: BLE001
                pass
            self.conf = None

    def line(self) -> str:
        return f"bat {self.vbat if self.vbat is None else f'{self.vbat:.2f} V'}  supervisor[" \
               f"{'?' if self.info is None else hop.decode_info(self.info)}]"


def make_tracker(kind: str, camera: int):
    import tracker as tr
    cfg = tr.TrackerConfig.load() if hasattr(tr.TrackerConfig, "load") else tr.TrackerConfig()
    if kind == "sim":
        return tr.SimTracker("exit_a")
    if kind == "aruco":
        if not hasattr(tr, "ArucoTracker"):
            raise SystemExit("tracker.ArucoTracker not available yet (single-camera mode is being added)")
        if hasattr(cfg, "aruco_camera"):
            cfg.aruco_camera = camera
        return tr.ArucoTracker(cfg)
    return tr.Tracker(cfg)


def wait_for(pred, timeout_s: float, poll: float = 0.1, clock=time.perf_counter, sleep=time.sleep) -> bool:
    t0 = clock()
    while clock() - t0 < timeout_s:
        if pred():
            return True
        sleep(poll)
    return bool(pred())


def run_sequence(cf, tracker, fl: flight.Flight, pf: feed.PositionFeed, *, hold_s: float, confirm, clock=time.perf_counter,
                 sleep=time.sleep, supervisor=None, ext_std: float | None = 0.03) -> int:
    """The hover sequence against injectable objects (unit-tested with fakes)."""
    fl.cfg.hover_time_s = hold_s
    if ext_std is not None:
        try:
            cf.param.set_value(resolve(cf.param.toc, "locSrv.extPosStdDev"), str(ext_std))
        except Exception as e:  # noqa: BLE001
            log.warning("could not set locSrv.extPosStdDev: %r", e)
    tracker.start()
    if not wait_for(lambda: tracker.get_state() is not None and tracker.get_state().tracking_ok, 20.0, clock=clock, sleep=sleep):
        print("ABORT: tracker never locked on the drone (is the marker visible? calibration files present?)")
        return 1
    st = tracker.get_state()
    print(f"tracker locked: xyz=({st.xyz[0]:+.2f}, {st.xyz[1]:+.2f}, {st.xyz[2]:+.2f}) yaw={st.yaw}")
    pf.start()
    sleep(0.5)
    fl.setup_estimator()                                    # Kalman + reset, now with position input
    converged = fl.wait_for_estimator(20.0)
    est = fl.telemetry.position
    st = tracker.get_state()
    err = max(abs(est[i] - st.xyz[i]) for i in range(3))
    print(f"estimator {'converged' if converged else 'NOT converged'}: estimate ({est[0]:+.2f}, {est[1]:+.2f}, {est[2]:+.2f}) "
          f"vs camera ({st.xyz[0]:+.2f}, {st.xyz[1]:+.2f}, {st.xyz[2]:+.2f})  max err {err:.2f} m  {pf.status()}")
    if not converged or err > 0.2:
        print("ABORT: estimator not trustworthy (feed rate? marker jitter? wrong extrinsics?)")
        return 1
    if supervisor is not None:
        if supervisor.info is not None and supervisor.info & (hop.BIT_CRASHED | hop.BIT_IS_LOCKED):
            print("supervisor CRASHED/LOCKED -> crash recovery request")
            hop.crash_recovery_request(cf)
            sleep(1.0)
        print("pre-flight:", supervisor.line())
        if supervisor.info is not None and supervisor.info & (hop.BIT_CRASHED | hop.BIT_IS_LOCKED | hop.BIT_IS_TUMBLED):
            print("ABORT: supervisor not ready; power-cycle the drone flat and rerun")
            return 1
        if supervisor.vbat is not None and supervisor.vbat < 3.7:
            print(f"ABORT: battery {supervisor.vbat:.2f} V")
            return 1
    if not confirm():
        print("cancelled")
        return 1
    if supervisor is not None:
        supervisor.stop()                                   # one log block in flight
    fl.start_keyboard_estop()
    print(f"TAKEOFF to {fl.cfg.takeoff_height} m, hold {hold_s} s (spacebar = emergency stop, l = land)")
    if not fl.takeoff():
        print(f"takeoff aborted: {fl.last_abort_reason}")
        return 2
    fl.land()
    print(f"landed. {pf.status()}")
    return 0


class _FakeCF:
    """Stand-in for --dry-run: records setpoints, no link."""

    class _Commander:
        def __init__(self):
            self.n = 0
            self.stops = 0

        def send_position_setpoint(self, x, y, z, yaw):
            self.n += 1

        def send_stop_setpoint(self):
            self.stops += 1

        def send_notify_setpoint_stop(self):
            pass

    class _Any:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    def __init__(self):
        self.commander = self._Commander()
        self.loc = self._Any()
        self.extpos = self._Any()
        self.platform = self._Any()
        self.param = self._Any()
        self.param.toc = {"stabilizer": {"estimator": 143}, "kalman": {"resetEstimation": 116},
                          "locSrv": {"extPosStdDev": 107}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", choices=["aruco", "color", "sim"], default="aruco")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--height", type=float, default=config.TAKEOFF_HEIGHT_M)
    ap.add_argument("--hold", type=float, default=8.0)
    ap.add_argument("--ext-std", type=float, default=0.05,
                    help="locSrv.extPosStdDev (m); single-camera marker depth noise is ~3-7 cm, so 0.05 by default")
    ap.add_argument("--uri", default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="fake drone; exercises the sequence")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    tracker = make_tracker(args.tracker, args.camera)
    cfg = flight.FlightConfig(takeoff_height=args.height)
    if args.dry_run:
        cf = _FakeCF()
        tel = flight.Telemetry()
        fl = flight.Flight(cf, cfg, tracker=tracker, telemetry=tel)
        # feed the telemetry from the tracker so the convergence check can pass without a drone
        def fake_log():
            while True:
                st = tracker.get_state()
                if st is not None and st.xyz is not None:
                    tel.update(0, {"kalman.stateX": st.xyz[0], "kalman.stateY": st.xyz[1], "kalman.stateZ": st.xyz[2],
                                   "kalman.varPX": 0.0001, "kalman.varPY": 0.0001, "kalman.varPZ": 0.0001, "pm.vbat": 4.0})
                time.sleep(0.1)
        threading.Thread(target=fake_log, daemon=True).start()
        pf = feed.PositionFeed(cf, tracker)
        rc = run_sequence(cf, tracker, fl, pf, hold_s=args.hold, confirm=lambda: True, ext_std=None)
        print(f"dry run finished rc={rc}: {cf.commander.n} setpoints, {cf.commander.stops} stop setpoints")
        tracker.stop()
        pf.stop()
        sys.exit(rc)

    cf = flight.connect(args.uri)
    fl = flight.Flight(cf, cfg, tracker=tracker, telemetry=flight.Telemetry())
    pf = feed.PositionFeed(cf, tracker)
    sup = SupervisorWatch(cf)
    try:
        fl.start_log()
        sup.start()
        confirm = (lambda: True) if args.yes else (
            lambda: hop.console_readline("\nClear area, marker on top, nose along +x. Type FLY to take off: ") == "FLY")
        rc = run_sequence(cf, tracker, fl, pf, hold_s=args.hold, confirm=confirm, supervisor=sup, ext_std=args.ext_std)
    except BaseException as e:  # noqa: BLE001
        print(f"\n!! {e!r} -> EMERGENCY STOP")
        fl.emergency_stop()
        rc = 2
    finally:
        pf.stop()
        tracker.stop()
        sup.stop()
        if fl.logconf is not None:
            try:
                fl.logconf.stop()
            except Exception:  # noqa: BLE001
                pass
        cf.close_link()
    sys.exit(rc)


if __name__ == "__main__":
    main()
