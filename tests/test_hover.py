"""hover.run_sequence against fakes: locks tracker, converges, takes off, holds, lands."""
import unittest

import feed
import flight
import hover
from tests.test_flight import FakeCF, FakeClock, FakeTracker, converged_telemetry


class StartableTracker(FakeTracker):
    def __init__(self, clock, **kw):
        super().__init__(clock, **kw)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True


class HoverSequenceTests(unittest.TestCase):
    def make(self, tracker=None, telemetry=None):
        clock = FakeClock()
        tracker = tracker or StartableTracker(clock, xyz=(0.2, -0.1, 0.0))
        cf = FakeCF()
        cfg = flight.FlightConfig(takeoff_height=0.5, takeoff_time_s=1.0, land_time_s=1.0, setpoint_hz=20.0)
        fl = flight.Flight(cf, cfg, tracker=tracker, telemetry=telemetry or converged_telemetry(0.2, -0.1, 0.0),
                           clock=clock, sleep=clock.sleep)
        pf = feed.PositionFeed(cf, tracker, clock=clock, sleep=clock.sleep)
        pf.start = lambda: None       # no thread in tests
        pf.stop = lambda: None
        return clock, tracker, cf, fl, pf

    def test_full_sequence_takes_off_holds_and_lands(self):
        clock, tracker, cf, fl, pf = self.make()
        rc = hover.run_sequence(cf, tracker, fl, pf, hold_s=2.0, confirm=lambda: True,
                                clock=clock, sleep=clock.sleep, ext_std=None)
        self.assertEqual(rc, 0)
        self.assertTrue(tracker.started)
        sp = cf.commander.setpoints
        self.assertAlmostEqual(len(sp), 20 + 40 + 20, delta=2)         # up 1 s, hold 2 s, down 1 s
        self.assertTrue(all(abs(p[0] - 0.2) < 1e-9 and abs(p[1] + 0.1) < 1e-9 for p in sp))
        self.assertAlmostEqual(max(p[2] for p in sp), 0.5)
        self.assertGreaterEqual(cf.commander.stops, 1)
        self.assertEqual(fl.state, "idle")

    def test_aborts_when_tracker_never_locks(self):
        clock = FakeClock()
        tracker = StartableTracker(clock, lose_after_calls=0)
        clock, tracker, cf, fl, pf = self.make(tracker=tracker)
        rc = hover.run_sequence(cf, tracker, fl, pf, hold_s=2.0, confirm=lambda: True,
                                clock=clock, sleep=clock.sleep, ext_std=None)
        self.assertEqual(rc, 1)
        self.assertEqual(cf.commander.setpoints, [])

    def test_aborts_when_estimate_disagrees_with_camera(self):
        clock, tracker, cf, fl, pf = self.make(telemetry=converged_telemetry(1.5, 0.0, 0.0))
        rc = hover.run_sequence(cf, tracker, fl, pf, hold_s=2.0, confirm=lambda: True,
                                clock=clock, sleep=clock.sleep, ext_std=None)
        self.assertEqual(rc, 1)
        self.assertEqual(cf.commander.setpoints, [])

    def test_cancelled_confirmation_sends_nothing(self):
        clock, tracker, cf, fl, pf = self.make()
        rc = hover.run_sequence(cf, tracker, fl, pf, hold_s=2.0, confirm=lambda: False,
                                clock=clock, sleep=clock.sleep, ext_std=None)
        self.assertEqual(rc, 1)
        self.assertEqual(cf.commander.setpoints, [])


if __name__ == "__main__":
    unittest.main()
