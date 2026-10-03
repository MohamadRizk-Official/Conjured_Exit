"""mission_sim: a fake drone that follows its setpoints and a tracker that sees it."""
import unittest

import numpy as np

import flight
import paths
from mission_sim import SimDrone, SimDroneTracker, SimTelemetry
from tests.test_flight import FakeClock


class SimDroneTests(unittest.TestCase):
    def test_drone_follows_setpoints_and_stop_grounds_it(self):
        d = SimDrone()
        d.commander.send_position_setpoint(0.1, 0.2, 0.5, 0.0)
        np.testing.assert_allclose(d.pos, [0.1, 0.2, 0.5])
        d.commander.send_stop_setpoint()
        self.assertEqual(d.pos[2], 0.0)
        self.assertEqual(d.commander.n, 1)
        self.assertEqual(d.commander.stops, 1)
        self.assertIsInstance(d.param.toc, dict)

    def test_tracker_reports_drone_and_wand(self):
        clock = FakeClock()
        d = SimDrone()
        tr = SimDroneTracker(d, noise_m=0.0, clock=clock, sleep=clock.sleep).start()
        d.pos[:] = (0.3, -0.2, 0.4)
        st = tr.get_state()
        self.assertTrue(st.tracking_ok)
        self.assertEqual(st.xyz, (0.3, -0.2, 0.4))
        self.assertEqual(st.yaw, 0.0)
        self.assertTrue(st.wand_ok)
        self.assertEqual(len(st.wand_xyz), 3)
        self.assertEqual(st.t, clock())
        self.assertEqual(st.mode, "sim")

    def test_carry_moves_drone_along_path(self):
        clock = FakeClock()
        d = SimDrone()
        tr = SimDroneTracker(d, noise_m=0.0, clock=clock, sleep=clock.sleep)
        tr.carry_step(0.0)
        start = d.pos.copy()
        tr.carry_step(float(tr.carry_path.duration) / 2)
        self.assertGreater(float(np.linalg.norm(d.pos - start)), 0.1)

    def test_on_recording_only_carries_a_grounded_drone(self):
        clock = FakeClock()
        d = SimDrone()
        tr = SimDroneTracker(d, noise_m=0.0, clock=clock, sleep=clock.sleep)
        d.pos[2] = 0.5                      # airborne: the hand must not grab it
        tr.on_recording(True)
        self.assertFalse(tr.carrying)
        tr.on_recording(False)

    def test_telemetry_converges_on_drone_position(self):
        clock = FakeClock()
        d = SimDrone()
        d.pos[:] = (0.1, 0.1, 0.0)
        fl = flight.Flight(d, flight.FlightConfig(), clock=clock, sleep=clock.sleep)
        tel = SimTelemetry(fl, d, clock=clock, sleep=clock.sleep)
        for _ in range(12):
            tel.tick()
        self.assertTrue(fl.telemetry.converged)
        self.assertEqual(fl.telemetry.position, (0.1, 0.1, 0.0))
        self.assertEqual(fl.telemetry.battery_v, 4.0)

    def test_flight_flies_demo_path_on_sim_drone(self):
        clock = FakeClock()
        d = SimDrone()
        tr = SimDroneTracker(d, noise_m=0.0, clock=clock, sleep=clock.sleep).start()
        fl = flight.Flight(d, flight.FlightConfig(takeoff_time_s=1.0, hover_time_s=0.5, land_time_s=1.0),
                           tracker=tr, clock=clock, sleep=clock.sleep)
        tel = SimTelemetry(fl, d, clock=clock, sleep=clock.sleep)
        for _ in range(12):
            tel.tick()
        fl.setup_estimator()
        self.assertTrue(fl.takeoff())
        self.assertTrue(fl.fly_path(paths.demo_paths()["exit_a"], land=True))
        self.assertEqual(fl.state, "idle")
        self.assertLess(d.pos[2], 0.01)
        self.assertGreater(d.commander.n, 100)


class MissionOnSimDroneTests(unittest.TestCase):
    def test_alarm_flies_exit_a_end_to_end(self):
        import tempfile
        from mission import Mission, MissionConfig
        from ui.pathstore import PathStore
        from ui.state import Command, CommandQueue, StateBus

        clock = FakeClock()
        d = SimDrone()
        tr = SimDroneTracker(d, noise_m=0.0, clock=clock, sleep=clock.sleep)
        with tempfile.TemporaryDirectory() as tmp:
            bus, q = StateBus(), CommandQueue()
            m = Mission(d, tr, PathStore(tmp, demo=True), bus=bus, commands=q, cfg=MissionConfig(relaunch_delay_s=0.1),
                        flight_cfg=flight.FlightConfig(takeoff_time_s=1.0, hover_time_s=0.5, land_time_s=1.0),
                        sim=True, clock=clock, sleep=clock.sleep, inline_flights=True)
            tel = SimTelemetry(m.fl, d, clock=clock, sleep=clock.sleep)
            for _ in range(12):
                tel.tick()
            m.handle(Command("arm", {"on": True}))
            m.handle(Command("alarm"))
            s = bus.snapshot_dict()
            self.assertEqual(s["flight"]["state"], "idle")
            self.assertTrue(s["alarm"]["active"])
            self.assertEqual(s["active_path"], "exit_a")
            self.assertLess(d.pos[2], 0.01)
            self.assertGreater(d.commander.n, 100)
            self.assertFalse(s["flight"]["armed"])


if __name__ == "__main__":
    unittest.main()
