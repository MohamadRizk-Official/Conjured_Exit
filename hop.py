"""hop.py -- first flight: a barometer-held hop with NO positioning system.

Attitude is held level, height comes from the Kalman estimator's z (barometer + IMU, reset to 0 at
the floor just before arming; the complementary estimator reports sea-level altitude on this
firmware, ~100 m, so it cannot be used for a z setpoint). The drone WILL drift sideways by tens
of centimetres. A CRASHED/LOCKED supervisor state is cleared with a crash-recovery request. Use a clear 2 x 2 m area, props on,
battery > 3.7 V, someone on the spacebar.

Sequence (all setpoints 18-byte packets at 20 Hz, link-safe):
  connect -> Kalman estimator reset -> small 10 Hz log (battery, supervisor bits, kalman z)
  -> pre-flight checks -> type FLY -> arm -> zero setpoint (thrust unlock)
  -> ramp z 0 -> height over t_up -> hold -> ramp down to 5 cm -> stop setpoint x3 -> disarm.
Spacebar = emergency stop (3x), 'l' = land now. Any exception = emergency stop.

    python hop.py --dry-run                 # prints the profile, no link
    python hop.py                           # 0.4 m, hold 3 s
    python hop.py --height 0.3 --hold 2 --yes

Stop `ui.server` first: only one process can hold the BLE link.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

import config
import flight
from toc_names import resolve

log = logging.getLogger("hop")

# supervisor.info bits (crazyflie-firmware supervisor.c)
BIT_CAN_BE_ARMED, BIT_IS_ARMED, BIT_AUTO_ARM, BIT_CAN_FLY = 1 << 0, 1 << 1, 1 << 2, 1 << 3
BIT_IS_FLYING, BIT_IS_TUMBLED, BIT_IS_LOCKED, BIT_CRASHED = 1 << 4, 1 << 5, 1 << 6, 1 << 7


def console_readline(prompt: str) -> str:
    """Read a line from the real console (CONIN$) even if stdin was inherited as NUL."""
    print(prompt, end="", flush=True)
    stream = sys.stdin
    if sys.platform == "win32":
        try:
            stream = open("CONIN$", "r", encoding="utf-8", errors="replace")
        except OSError:
            stream = sys.stdin
    return stream.readline().strip()


def arming_request(cf, do_arm: bool) -> None:
    sup = getattr(cf, "supervisor", None)
    if sup is not None and hasattr(sup, "send_arming_request"):
        sup.send_arming_request(do_arm)
    else:
        cf.platform.send_arming_request(do_arm)


def crash_recovery_request(cf) -> None:
    sup = getattr(cf, "supervisor", None)
    if sup is not None and hasattr(sup, "send_crash_recovery_request"):
        sup.send_crash_recovery_request()
    else:
        cf.platform.send_crash_recovery_request()


def decode_info(bits: int) -> str:
    names = [("canBeArmed", BIT_CAN_BE_ARMED), ("armed", BIT_IS_ARMED), ("autoArm", BIT_AUTO_ARM),
             ("canFly", BIT_CAN_FLY), ("flying", BIT_IS_FLYING), ("TUMBLED", BIT_IS_TUMBLED),
             ("LOCKED", BIT_IS_LOCKED), ("CRASHED", BIT_CRASHED)]
    return " ".join(n for n, b in names if bits & b) or "none"


def hop_profile(height: float, t_up: float, t_hold: float, t_down: float, hz: float,
                floor: float = 0.05) -> list[float]:
    """z targets at hz: linear ramp up (exclusive of 0), hold, linear ramp down to `floor`."""
    up = flight.ramp(0.0, height, t_up, hz)
    hold = [height] * max(1, int(round(t_hold * hz)))
    down = flight.ramp(height, floor, t_down, hz)
    return up + hold + down


class HopTelemetry:
    def __init__(self) -> None:
        self.vbat = None
        self.info = None
        self.z = None
        self.n = 0
        self.lock = threading.Lock()

    def update(self, _ts, data, _lc) -> None:
        with self.lock:
            self.n += 1
            if "pm.vbat" in data:
                self.vbat = float(data["pm.vbat"])
            if "supervisor.info" in data:
                self.info = int(data["supervisor.info"])
            if "kalman.stateZ" in data:
                self.z = float(data["kalman.stateZ"])

    def line(self) -> str:
        v = "?" if self.vbat is None else f"{self.vbat:.2f} V"
        z = "?" if self.z is None else f"{self.z:+.2f} m"
        i = "?" if self.info is None else decode_info(self.info)
        return f"bat {v}  z {z}  supervisor[{i}]"


def start_hop_log(cf, tel: HopTelemetry):
    conf = flight.make_log_config("hop", 100)
    wanted = [("pm.vbat", "FP16"), ("supervisor.info", "uint16_t"), ("kalman.stateZ", "FP16")]
    alias = {}
    for name, fetch_as in wanted:
        seen = resolve(cf.log.toc, name)
        conf.add_variable(seen, fetch_as)
        alias[seen] = name
    conf.data_received_cb.add_callback(lambda ts, d, lc: tel.update(ts, {alias.get(k, k): v for k, v in d.items()}, lc))
    conf.error_cb.add_callback(lambda lc, msg: log.error("hop log error: %s", msg))
    cf.log.add_config(conf)
    conf.start()
    return conf


def fly(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    profile = hop_profile(args.height, args.t_up, args.hold, args.t_down, args.hz)
    print(f"profile: {len(profile)} setpoints at {args.hz:g} Hz = {len(profile) / args.hz:.1f} s, "
          f"max z {max(profile):.2f} m")
    if args.dry_run:
        print("dry run: " + " ".join(f"{z:.2f}" for z in profile[:: max(1, int(args.hz))]) + " ...")
        return 0

    cf = flight.connect(args.uri)
    fl = flight.Flight(cf)
    tel = HopTelemetry()
    logconf = None
    try:
        ptoc = cf.param.toc
        cf.param.set_value(resolve(ptoc, "stabilizer.estimator"), "2")   # Kalman: z relative to start
        time.sleep(0.3)
        reset = resolve(ptoc, "kalman.resetEstimation")
        cf.param.set_value(reset, "1")
        time.sleep(0.2)
        cf.param.set_value(reset, "0")
        logconf = start_hop_log(cf, tel)
        t0 = time.time()
        while tel.n < 15 and time.time() - t0 < 5:
            time.sleep(0.1)
        print("pre-flight:", tel.line())
        if tel.vbat is None or tel.vbat < args.min_vbat:
            print(f"ABORT: battery {tel.vbat} V < {args.min_vbat} V")
            return 1
        if tel.info is not None and tel.info & (BIT_CRASHED | BIT_IS_LOCKED):
            print("supervisor is CRASHED/LOCKED -> sending crash recovery request")
            crash_recovery_request(cf)
            time.sleep(1.0)
            print("after recovery:", tel.line())
            if tel.info & (BIT_CRASHED | BIT_IS_LOCKED):
                print("ABORT: still CRASHED/LOCKED. Power-cycle the drone (flat, props up) and rerun.")
                return 1
        if tel.info is not None and tel.info & BIT_IS_TUMBLED:
            print("ABORT: drone reports TUMBLED (put it flat, props up)")
            return 1
        if tel.z is None or abs(tel.z) > 0.3:
            print(f"ABORT: altitude estimate {tel.z} m is not near 0 after reset; keep the drone still and rerun")
            return 1
        if not args.yes:
            answer = console_readline("\nClear 2 x 2 m? Props on? Someone on the spacebar? Type FLY to take off: ")
            if answer != "FLY":
                print(f"cancelled (got {answer!r})")
                return 1
        fl.start_keyboard_estop()

        arming_request(cf, True)
        time.sleep(0.6)
        if tel.info is not None and not tel.info & BIT_IS_ARMED:
            log.warning("not armed after the arming request; trying param system.arm=1")
            try:
                cf.param.set_value(resolve(ptoc, "system.arm"), "1")
            except KeyError:
                pass
            time.sleep(0.6)
        print("armed check:", tel.line())
        if tel.info is not None and not tel.info & BIT_IS_ARMED:
            print("ABORT: drone would not arm")
            return 1

        for _ in range(3):                           # thrust unlock
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.05)

        dt = 1.0 / args.hz
        t_start = time.perf_counter()
        print("TAKEOFF (spacebar = emergency stop, l = land now)")
        land_now = False
        for i, z in enumerate(profile):
            if fl._stop_requested.is_set():
                raise RuntimeError("spacebar emergency stop")
            if fl._land_requested.is_set() and not land_now:
                land_now = True
                remaining = flight.ramp(z, 0.05, args.t_down, args.hz)
                profile = profile[: i] + remaining
                print("landing now")
            if tel.z is not None and tel.z > args.height + 1.0:
                raise RuntimeError(f"altitude runaway {tel.z:.2f} m")
            cf.commander.send_zdistance_setpoint(0.0, 0.0, 0.0, float(z))
            if i % int(args.hz) == 0:
                print(f"  t={i * dt:4.1f}s target z {z:.2f}  {tel.line()}")
            nxt = t_start + (i + 1) * dt
            now = time.perf_counter()
            if nxt > now:
                time.sleep(nxt - now)
        for _ in range(3):
            cf.commander.send_stop_setpoint()
            time.sleep(0.05)
        print("landed:", tel.line())
        return 0
    except BaseException as e:  # noqa: BLE001 - any failure -> motors off
        print(f"\n!! {e!r} -> EMERGENCY STOP")
        fl.emergency_stop()
        return 2
    finally:
        try:
            arming_request(cf, False)
        except Exception:  # noqa: BLE001
            pass
        if logconf is not None:
            try:
                logconf.stop()
            except Exception:  # noqa: BLE001
                pass
        cf.close_link()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=float, default=0.4)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--t-up", type=float, default=1.5)
    ap.add_argument("--t-down", type=float, default=1.5)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--min-vbat", type=float, default=3.7)
    ap.add_argument("--uri", default=None, help=f"default {config.LINK_URI}")
    ap.add_argument("--yes", action="store_true", help="skip the FLY confirmation")
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(fly(ap.parse_args()))


if __name__ == "__main__":
    main()
