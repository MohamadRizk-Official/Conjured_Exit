"""Tests for flight.py / feed.py: trajectory generation, safety rules and link budget.

Everything runs against fakes with a fake clock, so no drone, no sleeping.
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

import feed
import flight
import paths
from flight import Flight, FlightConfig, FlightRefused, Telemetry


# ----------------------------------------------------------------- fakes
class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, s)


class FakeCommander:
    def __init__(self) -> None:
        self.setpoints: list[tuple[float, float, float, float]] = []
        self.stops = 0
        self.notify_stops = 0

    def send_position_setpoint(self, x, y, z, yaw):
        self.setpoints.append((x, y, z, yaw))

    def send_stop_setpoint(self):
        self.stops += 1

    def send_notify_setpoint_stop(self, remain_valid_milliseconds=0):
        self.notify_stops += 1


class FakeLoc:
    def __init__(self) -> None:
        self.emergency_stops = 0

    def send_emergency_stop(self):
        self.emergency_stops += 1


class FakeExtpos:
    def __init__(self) -> None:
        self.poses: list[tuple] = []
        self.positions: list[tuple] = []

    def send_extpose(self, x, y, z, qx, qy, qz, qw):
        self.poses.append((x, y, z, qx, qy, qz, qw))

    def send_extpos(self, x, y, z):
        self.positions.append((x, y, z))


class FakePlatform:
    def __init__(self) -> None:
        self.arming: list[bool] = []

    def send_arming_request(self, do_arm: bool):
        self.arming.append(do_arm)


class FakeParam:
    # a minimal TOC so toc_names.resolve() finds the names flight.py sets
    toc = {"stabilizer": {"estimator": 143}, "kalman": {"resetEstimation": 116},
           "locSrv": {"extPosStdDev": 107, "extQuatStdDev": 108}}

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_value(self, name, value):
        self.values[name] = str(value)


class FakeCF:
    def __init__(self) -> None:
        self.commander = FakeCommander()
        self.loc = FakeLoc()
        self.extpos = FakeExtpos()
        self.platform = FakePlatform()
        self.param = FakeParam()


@dataclass
class FakeState:
    t: float
    xyz: tuple
    yaw: float | None
    tracking_ok: bool


class FakeTracker:
    """tracking_ok flips to False once `lose_after_calls` get_state() calls have happened."""

    def __init__(self, clock: FakeClock, xyz=(0.0, 0.0, 0.0), yaw=0.0, lose_after_calls=None) -> None:
        self.clock = clock
        self.xyz = xyz
        self.yaw = yaw
        self.calls = 0
        self.lose_after_calls = lose_after_calls

    def get_state(self):
        self.calls += 1
        ok = self.lose_after_calls is None or self.calls <= self.lose_after_calls
        return FakeState(t=self.clock(), xyz=self.xyz, yaw=self.yaw, tracking_ok=ok)


def converged_telemetry(x=0.0, y=0.0, z=0.0) -> Telemetry:
    tel = Telemetry()
    for i in range(12):
        tel.update(i * 100, {"pm.vbat": 3.9, "kalman.stateX": x, "kalman.stateY": y, "kalman.stateZ": z,
                             "kalman.varPX": 0.0002, "kalman.varPY": 0.0002, "kalman.varPZ": 0.0002})
    return tel


def make_flight(clock, tracker=None, telemetry=None, commands=None, **cfg_overrides):
    cfg = FlightConfig(**cfg_overrides)
    cf = FakeCF()
    fl = Flight(cf, cfg, tracker=tracker, telemetry=telemetry, commands=commands, clock=clock, sleep=clock.sleep)
    return fl, cf


# ----------------------------------------------------------------- tests
class RampTests(unittest.TestCase):
    def test_ramp_is_monotonic_has_hz_times_duration_points_and_ends_at_target(self):
        zs = flight.ramp(0.0, 0.5, duration_s=2.0, hz=20.0)
        self.assertEqual(len(zs), 40)
        self.assertAlmostEqual(zs[-1], 0.5)
        self.assertTrue(all(b >= a for a, b in zip(zs, zs[1:])))
        self.assertGreater(zs[0], 0.0)

    def test_ramp_down_is_monotonic_decreasing(self):
        zs = flight.ramp(0.5, 0.05, duration_s=1.0, hz=10.0)
        self.assertEqual(len(zs), 10)
        self.assertTrue(all(b <= a for a, b in zip(zs, zs[1:])))
        self.assertAlmostEqual(zs[-1], 0.05)


class LinkBudgetTests(unittest.TestCase):
    def test_log_block_fits_one_ble_notification(self):
        # 1 header + 1 block id + 3 timestamp + data must be <= 19 bytes -> data <= 14
        size = sum(flight.LOG_TYPE_SIZES[t] for _, t in flight.FLIGHT_LOG_VARIABLES)
        self.assertLessEqual(size, 14)
        self.assertEqual({t for _, t in flight.FLIGHT_LOG_VARIABLES}, {"FP16"})
        self.assertGreaterEqual(flight.LOG_PERIOD_MS, 100)
        names = [n for n, _ in flight.FLIGHT_LOG_VARIABLES]
        for needed in ("pm.vbat", "kalman.stateX", "kalman.stateY", "kalman.stateZ",
                       "kalman.varPX", "kalman.varPY", "kalman.varPZ"):
            self.assertIn(needed, names)


class TelemetryTests(unittest.TestCase):
    def test_not_converged_before_window_is_full(self):
        tel = Telemetry(window=10, var_threshold=0.001)
        for i in range(5):
            tel.update(i * 100, {"kalman.varPX": 0.0001, "kalman.varPY": 0.0001, "kalman.varPZ": 0.0001})
        self.assertFalse(tel.converged)

    def test_converged_when_variance_spread_is_small_on_all_axes(self):
        self.assertTrue(converged_telemetry().converged)

    def test_not_converged_when_one_axis_still_moving(self):
        tel = Telemetry(window=10, var_threshold=0.001)
        for i in range(12):
            tel.update(i * 100, {"kalman.varPX": 0.0001, "kalman.varPY": 0.0001, "kalman.varPZ": 0.05 * (i % 2)})
        self.assertFalse(tel.converged)

    def test_position_and_battery_are_exposed(self):
        tel = converged_telemetry(0.1, 0.2, 0.3)
        self.assertEqual(tel.position, (0.1, 0.2, 0.3))
        self.assertAlmostEqual(tel.battery_v, 3.9)


class TakeoffTests(unittest.TestCase):
    def test_takeoff_refused_when_estimator_not_converged(self):
        clock = FakeClock()
        fl, cf = make_flight(clock, tracker=FakeTracker(clock), telemetry=Telemetry())
        with self.assertRaises(FlightRefused):
            fl.takeoff()
        self.assertEqual(cf.commander.setpoints, [])
        self.assertEqual(cf.platform.arming, [])

    def test_takeoff_refused_when_tracking_not_ok(self):
        clock = FakeClock()
        fl, cf = make_flight(clock, tracker=FakeTracker(clock, lose_after_calls=0), telemetry=converged_telemetry())
        with self.assertRaises(FlightRefused):
            fl.takeoff()
        self.assertEqual(cf.commander.setpoints, [])

    def test_takeoff_arms_then_ramps_z_to_height_at_current_xy_and_hovers(self):
        clock = FakeClock()
        tracker = FakeTracker(clock, xyz=(0.3, -0.2, 0.0))
        fl, cf = make_flight(clock, tracker=tracker, telemetry=converged_telemetry(),
                             takeoff_height=0.5, takeoff_time_s=2.0, hover_time_s=1.0, setpoint_hz=20.0)
        fl.takeoff()
        sp = cf.commander.setpoints
        self.assertEqual(cf.platform.arming, [True])
        self.assertAlmostEqual(len(sp), 60, delta=2)              # (2 + 1) s * 20 Hz
        self.assertAlmostEqual(sp[-1][2], 0.5)
        zs = [p[2] for p in sp]
        self.assertTrue(all(0.0 < z <= 0.5 + 1e-9 for z in zs))
        self.assertTrue(all(abs(p[0] - 0.3) < 1e-9 and abs(p[1] + 0.2) < 1e-9 for p in sp))
        self.assertTrue(all(p[3] == 0.0 for p in sp))              # yaw 0 = nose along +x
        self.assertEqual(fl.state, "hover")


class FlyPathTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.tracker = FakeTracker(self.clock, xyz=(0.0, 0.0, 0.0))
        self.fl, self.cf = make_flight(self.clock, tracker=self.tracker, telemetry=converged_telemetry(),
                                       takeoff_height=0.5, takeoff_time_s=1.0, hover_time_s=0.5, setpoint_hz=20.0)
        self.path = paths.demo_paths()["square"]

    def test_fly_path_streams_targets_along_path_inside_geofence(self):
        self.fl.takeoff()
        n_before = len(self.cf.commander.setpoints)
        ok = self.fl.fly_path(self.path, land=False)
        self.assertTrue(ok)
        sp = np.array([p[:3] for p in self.cf.commander.setpoints[n_before:]])
        self.assertTrue(paths.GEOFENCE.contains(sp))
        # the first targets move from the hover point towards the path start, then follow the path
        d_start = np.linalg.norm(sp - self.path.points[0], axis=1)
        self.assertLess(d_start.min(), 0.03)
        d_end = np.linalg.norm(sp - self.path.points[-1], axis=1)
        self.assertLess(d_end.min(), 0.03)
        self.assertEqual(self.fl.state, "hover")

    def test_fly_path_with_land_ends_on_the_ground_with_a_stop_setpoint(self):
        self.fl.takeoff()
        ok = self.fl.fly_path(self.path)
        self.assertTrue(ok)
        self.assertLessEqual(self.cf.commander.setpoints[-1][2], self.fl.cfg.land_cutoff_z + 1e-9)
        self.assertGreaterEqual(self.cf.commander.stops, 1)
        self.assertEqual(self.fl.state, "idle")

    def test_flight_targets_outside_geofence_are_clamped(self):
        wild = paths.Path(name="wild", mode="spell",
                          points=np.array([[0.0, 0.0, 0.7], [3.0, -3.0, 2.5]]), times=np.array([0.0, 1.0]), meta={})
        self.fl.takeoff()
        self.fl.fly_path(wild, land=False)
        sp = np.array([p[:3] for p in self.cf.commander.setpoints])
        self.assertTrue(paths.GEOFENCE.contains(sp[sp[:, 2] >= paths.GEOFENCE.zmin]))
        self.assertAlmostEqual(sp[:, 0].max(), paths.GEOFENCE.xmax)
        self.assertAlmostEqual(sp[:, 2].max(), paths.GEOFENCE.zmax)

    def test_tracking_lost_longer_than_limit_lands_and_aborts(self):
        self.fl.takeoff()
        # lose tracking right after takeoff; the limit is 0.3 s = 6 ticks at 20 Hz
        self.tracker.lose_after_calls = self.tracker.calls
        ok = self.fl.fly_path(self.path)
        self.assertFalse(ok)
        self.assertIn("tracking", self.fl.last_abort_reason)
        self.assertLessEqual(self.cf.commander.setpoints[-1][2], self.fl.cfg.land_cutoff_z + 1e-9)
        self.assertGreaterEqual(self.cf.commander.stops, 1)
        self.assertEqual(self.fl.state, "idle")
        # it did not fly the whole route
        self.assertLess(len(self.cf.commander.setpoints), 0.5 * self.path.duration * 20)

    def test_brief_tracking_glitch_does_not_land(self):
        self.fl.takeoff()
        calls = {"n": 0}
        real_get_state = self.tracker.get_state

        def glitchy():
            calls["n"] += 1
            st = real_get_state()
            if 5 <= calls["n"] <= 7:        # 3 ticks = 0.15 s < 0.3 s
                st.tracking_ok = False
            return st

        self.tracker.get_state = glitchy
        ok = self.fl.fly_path(self.path)
        self.assertTrue(ok)

    def test_stop_command_during_flight_triggers_emergency_stop(self):
        self.fl.commands = lambda: "stop" if self.fl.state == "flying" else None
        self.assertTrue(self.fl.takeoff())
        ok = self.fl.fly_path(self.path)
        self.assertFalse(ok)
        self.assertEqual(self.cf.loc.emergency_stops, 3)
        self.assertEqual(self.cf.commander.stops, 3)
        self.assertEqual(self.fl.state, "estop")
        with self.assertRaises(FlightRefused):
            self.fl.fly_path(self.path)

    def test_land_command_during_flight_lands_normally(self):
        self.fl.commands = lambda: "land" if self.fl.state == "flying" else None
        self.assertTrue(self.fl.takeoff())
        ok = self.fl.fly_path(self.path)
        self.assertFalse(ok)
        self.assertIn("land", self.fl.last_abort_reason)
        self.assertEqual(self.cf.loc.emergency_stops, 0)
        self.assertEqual(self.fl.state, "idle")


class EmergencyStopTests(unittest.TestCase):
    def test_emergency_stop_sends_three_times_and_blocks_takeoff(self):
        clock = FakeClock()
        fl, cf = make_flight(clock, tracker=FakeTracker(clock), telemetry=converged_telemetry())
        fl.emergency_stop()
        self.assertEqual(cf.loc.emergency_stops, 3)
        self.assertEqual(cf.commander.stops, 3)
        self.assertEqual(fl.state, "estop")
        with self.assertRaises(FlightRefused):
            fl.takeoff()

    def test_request_stop_from_another_thread_is_honoured_on_next_tick(self):
        clock = FakeClock()
        fl, cf = make_flight(clock, tracker=FakeTracker(clock), telemetry=converged_telemetry(),
                             takeoff_time_s=1.0, hover_time_s=0.5)
        fl.request_stop()
        with self.assertRaises(FlightRefused):
            fl.takeoff()
        self.assertEqual(cf.loc.emergency_stops, 3)


class LandTests(unittest.TestCase):
    def test_land_ramps_down_from_last_target_then_sends_stop(self):
        clock = FakeClock()
        fl, cf = make_flight(clock, tracker=FakeTracker(clock), telemetry=converged_telemetry(),
                             takeoff_height=0.6, takeoff_time_s=1.0, hover_time_s=0.0, land_time_s=1.0,
                             setpoint_hz=10.0)
        fl.takeoff()
        n = len(cf.commander.setpoints)
        fl.land()
        zs = [p[2] for p in cf.commander.setpoints[n:]]
        self.assertEqual(len(zs), 10)
        self.assertTrue(all(b <= a for a, b in zip(zs, zs[1:])))
        self.assertAlmostEqual(zs[-1], fl.cfg.land_cutoff_z)
        self.assertEqual(cf.commander.stops, 1)
        self.assertEqual(fl.state, "idle")


class FeedTests(unittest.TestCase):
    def test_yaw_only_quaternion(self):
        qx, qy, qz, qw = feed.yaw_to_quaternion(math.pi / 2)
        self.assertEqual((qx, qy), (0.0, 0.0))
        self.assertAlmostEqual(qz, math.sin(math.pi / 4))
        self.assertAlmostEqual(qw, math.cos(math.pi / 4))

    def test_tick_sends_extpose_when_tracking_ok_and_skips_when_not(self):
        clock = FakeClock()
        cf = FakeCF()
        tracker = FakeTracker(clock, xyz=(0.1, 0.2, 0.3), yaw=0.0, lose_after_calls=1)
        pf = feed.PositionFeed(cf, tracker, rate_hz=30.0, use_yaw=True, clock=clock)
        self.assertTrue(pf.tick())
        self.assertFalse(pf.tick())
        self.assertEqual(cf.extpos.poses, [(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)])
        self.assertEqual(pf.sent, 1)
        self.assertEqual(pf.skipped, 1)

    def test_default_is_position_only_even_when_yaw_is_known(self):
        # the 29-byte pose packet is corrupted by nRF firmware 2024.10 (fragmented uplink)
        clock = FakeClock()
        cf = FakeCF()
        tracker = FakeTracker(clock, xyz=(0.1, 0.2, 0.3), yaw=0.5)
        pf = feed.PositionFeed(cf, tracker, clock=clock)
        self.assertTrue(pf.tick())
        self.assertEqual(cf.extpos.positions, [(0.1, 0.2, 0.3)])
        self.assertEqual(cf.extpos.poses, [])

    def test_log_block_creation_never_exceeds_20_byte_packets(self):
        self.assertLessEqual(2 + 3 * flight.MAX_LOG_VARS_PER_UPLINK_PACKET + 1, 20)

    def test_tick_falls_back_to_position_only_when_yaw_unknown(self):
        clock = FakeClock()
        cf = FakeCF()
        tracker = FakeTracker(clock, xyz=(0.1, 0.2, 0.3), yaw=None)
        pf = feed.PositionFeed(cf, tracker, clock=clock)
        self.assertTrue(pf.tick())
        self.assertEqual(cf.extpos.positions, [(0.1, 0.2, 0.3)])
        self.assertEqual(cf.extpos.poses, [])

    def test_tick_skips_stale_state(self):
        clock = FakeClock()
        cf = FakeCF()
        tracker = FakeTracker(clock, xyz=(0.1, 0.2, 0.3))
        pf = feed.PositionFeed(cf, tracker, clock=clock, max_age_s=0.2)
        st = tracker.get_state()
        tracker.get_state = lambda: st          # frozen, old state
        clock.sleep(1.0)
        self.assertFalse(pf.tick())
        self.assertEqual(pf.skipped, 1)


if __name__ == "__main__":
    unittest.main()
