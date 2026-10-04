"""mission.Mission against fakes: recording, arming, flights, reroute, stop, preflight, publish."""
from __future__ import annotations

import pathlib
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

import flight
import hop
import paths
from mission import Mission, MissionConfig, RouteRecorder, preflight
from tests.test_flight import FakeCF, FakeClock, converged_telemetry
from toc_names import resolve
from ui.pathstore import PathStore
from ui.state import Command, CommandQueue, StateBus


@dataclass
class State:
    t: float
    xyz: Optional[tuple]
    yaw: Optional[float]
    tracking_ok: bool
    wand_xyz: Optional[tuple] = None
    wand_ok: bool = False
    fps: float = 30.0
    latency_ms: float = 5.0


class MovingTracker:
    """Static until on_recording(True); while recording, xyz advances `step` metres along +x per
    get_state() call so PathRecorder keeps every sample. The wand hovers 0.6 m above the drone."""

    def __init__(self, clock, start=(0.0, 0.0, 0.3), step=0.02, tracking_ok=True, wand_ok=False):
        self.clock = clock
        self.pos = np.array(start, dtype=float)
        self.step = step
        self.tracking_ok = tracking_ok
        self.wand_ok = wand_ok
        self.moving = False
        self.recording_calls: list[bool] = []
        self.started = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        pass

    def on_recording(self, active: bool):
        self.recording_calls.append(active)
        self.moving = active

    def get_state(self):
        if self.moving:
            self.pos[0] += self.step
        p = tuple(float(v) for v in self.pos)
        return State(t=self.clock(), xyz=p, yaw=getattr(self, 'yaw_value', 0.0), tracking_ok=self.tracking_ok,
                     wand_xyz=(p[0], p[1], p[2] + 0.6), wand_ok=self.wand_ok)


class RouteRecorderTests(unittest.TestCase):
    def test_guide_mode_records_drone_and_publishes_live_points(self):
        clock = FakeClock()
        bus = StateBus()
        tr = MovingTracker(clock)
        tr.on_recording(True)
        rec = RouteRecorder(tr, bus, clock=clock, sleep=clock.sleep)
        rec.start("guide", spawn_thread=False)
        kept = sum(rec.tick() for _ in range(25))
        self.assertEqual(kept, 25)
        self.assertEqual(rec.n, 25)
        self.assertEqual(rec.points.shape, (25, 3))
        s = bus.snapshot_dict()
        self.assertTrue(s["recording"]["active"])
        self.assertEqual(s["recording"]["mode"], "guide")
        self.assertEqual(s["recording"]["n_samples"], 25)
        self.assertEqual(len(s["recording"]["live_points"]), 25)
        rec.stop()
        self.assertFalse(bus.snapshot_dict()["recording"]["active"])
        self.assertEqual(rec.n, 25)                     # samples survive stop()

    def test_spell_mode_records_the_wand(self):
        clock = FakeClock()
        bus = StateBus()
        tr = MovingTracker(clock, wand_ok=True)
        tr.on_recording(True)
        rec = RouteRecorder(tr, bus, clock=clock, sleep=clock.sleep)
        rec.start("spell", spawn_thread=False)
        for _ in range(10):
            rec.tick()
        self.assertEqual(rec.n, 10)
        self.assertTrue(np.allclose(rec.points[:, 2], 0.9))

    def test_lost_tracking_records_nothing_and_logs_once(self):
        clock = FakeClock()
        bus = StateBus()
        tr = MovingTracker(clock, tracking_ok=False)
        rec = RouteRecorder(tr, bus, clock=clock, sleep=clock.sleep)
        rec.start("guide", spawn_thread=False)
        for _ in range(10):
            self.assertFalse(rec.tick())
        self.assertEqual(rec.n, 0)
        lines = [l for l in bus.snapshot_dict()["log"] if "tracking lost" in l]
        self.assertEqual(len(lines), 1)

    def test_start_clears_previous_samples(self):
        clock = FakeClock()
        bus = StateBus()
        tr = MovingTracker(clock)
        tr.on_recording(True)
        rec = RouteRecorder(tr, bus, clock=clock, sleep=clock.sleep)
        rec.start("guide", spawn_thread=False)
        rec.tick()
        rec.stop()
        rec.start("guide", spawn_thread=False)
        self.assertEqual(rec.n, 0)
        self.assertEqual(bus.snapshot_dict()["recording"]["live_points"], [])


class FakeSupervisor:
    def __init__(self, info=hop.BIT_CAN_BE_ARMED | hop.BIT_CAN_FLY, vbat=4.0):
        self.info = info
        self.vbat = vbat
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def line(self):
        return "fake supervisor"


class RecoveringCF(FakeCF):
    """platform.send_crash_recovery_request clears CRASHED on the attached fake supervisor (if clears)."""

    def __init__(self, sup, clears=True):
        super().__init__()
        self.sup = sup
        self.recoveries = 0

        def recover():
            self.recoveries += 1
            if clears:
                self.sup.info &= ~hop.BIT_CRASHED

        self.platform.send_crash_recovery_request = recover


class _Pre:
    """Just the attributes preflight() reads."""

    def __init__(self, cf, tracker, fl, supervisor, armed=True, link_lost=False, cfg=None, sleep=lambda s: None):
        self.cf, self.tracker, self.fl, self.supervisor = cf, tracker, fl, supervisor
        self.armed, self.link_lost, self.cfg, self.sleep = armed, link_lost, cfg or MissionConfig(), sleep


def make_pre(*, armed=True, link_lost=False, tracking_ok=True, telemetry=None, sup=None, cf=None):
    clock = FakeClock()
    tr = MovingTracker(clock, tracking_ok=tracking_ok)
    cf = cf or FakeCF()
    fl = flight.Flight(cf, flight.FlightConfig(), tracker=tr, telemetry=telemetry or converged_telemetry(0.0, 0.0, 0.3),
                       clock=clock, sleep=clock.sleep)
    return _Pre(cf, tr, fl, sup, armed=armed, link_lost=link_lost, sleep=clock.sleep), cf, fl


class PreflightTests(unittest.TestCase):
    def test_passes_when_everything_is_fine(self):
        m, cf, fl = make_pre(sup=FakeSupervisor())
        self.assertIsNone(preflight(m))

    def test_disarmed(self):
        m, _, _ = make_pre(armed=False)
        self.assertEqual(preflight(m), "disarmed")

    def test_link_lost(self):
        m, _, _ = make_pre(link_lost=True)
        self.assertEqual(preflight(m), "link lost")

    def test_tracker_not_locked(self):
        m, _, _ = make_pre(tracking_ok=False)
        self.assertEqual(preflight(m), "tracker not locked")

    def test_estimate_far_from_camera_refuses_after_one_reset(self):
        m, cf, fl = make_pre(telemetry=converged_telemetry(1.5, 0.0, 0.3))
        reason = preflight(m)
        self.assertIn("disagrees", reason)
        self.assertEqual(cf.param.values[resolve(cf.param.toc, "kalman.resetEstimation")], "0")  # reset issued

    def test_crashed_supervisor_is_recovered_once(self):
        sup = FakeSupervisor(info=hop.BIT_CRASHED | hop.BIT_CAN_BE_ARMED)
        cf = RecoveringCF(sup)
        m, _, _ = make_pre(sup=sup, cf=cf)
        self.assertIsNone(preflight(m))
        self.assertEqual(cf.recoveries, 1)

    def test_still_crashed_refuses(self):
        sup = FakeSupervisor(info=hop.BIT_CRASHED)
        cf = RecoveringCF(sup, clears=False)
        m, _, _ = make_pre(sup=sup, cf=cf)
        self.assertTrue(preflight(m).startswith("supervisor"))
        self.assertEqual(cf.recoveries, 1)

    def test_low_battery_refuses(self):
        m, _, _ = make_pre(sup=FakeSupervisor(vbat=3.6))
        self.assertTrue(preflight(m).startswith("battery"))

    def test_preflight_logs_the_supervisor_line(self):
        m, _, _ = make_pre(sup=FakeSupervisor(info=hop.BIT_CAN_BE_ARMED | hop.BIT_CAN_FLY, vbat=4.0))
        m.bus = StateBus()
        self.assertIsNone(preflight(m))
        lines = [l for l in m.bus.snapshot_dict()["log"] if "pre-flight" in l]
        self.assertEqual(len(lines), 1)
        self.assertIn("canBeArmed", lines[0])
        self.assertIn("4.00 V", lines[0])

    def test_flights_disabled_by_config(self):
        m, _, _ = make_pre(sup=FakeSupervisor())
        m.cfg = MissionConfig(allow_flights=False)
        self.assertIn("flights disabled", preflight(m))
        self.assertIsNone(preflight(make_pre(sup=FakeSupervisor())[0]))


def make_mission(tmpdir, *, armed=False, cf=None, tracker_kwargs=None, supervisor=None, inline=True, cfg=None):
    clock = FakeClock()
    tr = MovingTracker(clock, **(tracker_kwargs or {}))
    cf = cf or FakeCF()
    bus, q = StateBus(), CommandQueue()
    store = PathStore(tmpdir, demo=True)
    fcfg = flight.FlightConfig(takeoff_time_s=1.0, hover_time_s=0.5, land_time_s=1.0, setpoint_hz=20.0)
    m = Mission(cf, tr, store, bus=bus, commands=q, cfg=cfg or MissionConfig(relaunch_delay_s=0.1),
                flight_cfg=fcfg, supervisor=supervisor, sim=True, clock=clock, sleep=clock.sleep,
                inline_flights=inline)
    m.fl.telemetry = converged_telemetry(0.0, 0.0, 0.3)
    if armed:
        m.handle(Command("arm", {"on": True}))
    return m, cf, bus, q, store, clock, tr


def setpoint_hook(cf, at_n, fn):
    """Call fn() from inside the fake commander when the at_n-th setpoint is sent (mid-flight injection)."""
    orig = cf.commander.send_position_setpoint
    fired = []

    def hook(x, y, z, yaw):
        orig(x, y, z, yaw)
        if len(cf.commander.setpoints) == at_n and not fired:
            fired.append(True)
            fn()

    cf.commander.send_position_setpoint = hook
    return fired


class RecordingCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_save_lists_and_selects(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        m.handle(Command("record_start"))
        self.assertEqual(tr.recording_calls, [True])
        for _ in range(40):
            m.rec.tick()
        m.handle(Command("record_stop"))
        self.assertEqual(tr.recording_calls, [True, False])
        s = bus.snapshot_dict()
        self.assertEqual(s["recording"]["n_samples"], 40)
        self.assertEqual(len(s["recording"]["live_points"]), 40)
        self.assertFalse(s["recording"]["active"])
        m.handle(Command("save_as", {"name": "Demo"}))
        s = bus.snapshot_dict()
        self.assertIn("demo", {p["name"] for p in s["paths"]})
        self.assertEqual(s["active_path"], "demo")
        self.assertTrue((pathlib.Path(self.tmp.name) / "demo.json").is_file())
        self.assertEqual(store.get("demo").mode, "guide")

    def test_record_while_tracking_lost_records_nothing(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, tracker_kwargs={"tracking_ok": False})
        m.handle(Command("record_start"))
        for _ in range(10):
            m.rec.tick()
        m.handle(Command("record_stop"))
        m.handle(Command("save_as", {"name": "nothing"}))
        s = bus.snapshot_dict()
        self.assertEqual(s["recording"]["n_samples"], 0)
        self.assertTrue(any("tracking lost" in l for l in s["log"]))
        self.assertTrue(any("nothing to save" in l for l in s["log"]))
        self.assertFalse((pathlib.Path(self.tmp.name) / "nothing.json").exists())

    def test_spell_mode_records_wand(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, tracker_kwargs={"wand_ok": True})
        m.handle(Command("set_mode", {"mode": "spell"}))
        m.handle(Command("record_start"))
        for _ in range(20):
            m.rec.tick()
        m.handle(Command("record_stop"))
        m.handle(Command("save_as", {"name": "loop"}))
        self.assertEqual(store.get("loop").mode, "spell")
        self.assertEqual(bus.snapshot_dict()["recording"]["mode"], "spell")

    def test_record_start_refused_while_flying(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        setpoint_hook(cf, 30, lambda: m.handle(Command("record_start")))
        m.handle(Command("cast", {"name": "exit_a"}))
        s = bus.snapshot_dict()
        self.assertTrue(any("record_start refused" in l for l in s["log"]))
        self.assertEqual(s["recording"]["n_samples"], 0)
        self.assertEqual(tr.recording_calls, [])


class FlightCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_cast_refused_while_disarmed(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        m.handle(Command("cast", {"name": "exit_a"}))
        self.assertEqual(cf.commander.setpoints, [])
        self.assertEqual(m.fl.state, "idle")
        self.assertTrue(any("disarmed" in l for l in bus.snapshot_dict()["log"]))

    def test_alarm_refused_while_disarmed(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        m.handle(Command("alarm"))
        self.assertEqual(cf.commander.setpoints, [])
        self.assertFalse(bus.snapshot_dict()["alarm"]["active"])

    def test_cast_flies_inside_geofence_publishes_replay_and_disarms(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        seen = {}

        def peek():
            m.publish()
            seen.update(bus.snapshot_dict()["replay"])

        setpoint_hook(cf, 45, peek)
        m.handle(Command("cast", {"name": "exit_a"}))
        self.assertEqual(m.fl.state, "idle")
        self.assertGreater(len(cf.commander.setpoints), 50)
        box = paths.GEOFENCE
        for x, y, z, _yaw in cf.commander.setpoints:
            self.assertTrue(box.xmin - 1e-9 <= x <= box.xmax + 1e-9)
            self.assertTrue(box.ymin - 1e-9 <= y <= box.ymax + 1e-9)
            self.assertLessEqual(z, box.zmax + 1e-9)
        self.assertTrue(seen["active"])
        self.assertAlmostEqual(seen["duration"], float(store.get("exit_a").duration), places=6)
        self.assertGreaterEqual(cf.commander.stops, 1)
        self.assertFalse(m.armed)
        s = bus.snapshot_dict()
        self.assertFalse(s["flight"]["armed"])
        self.assertEqual(s["flight"]["state"], "idle")
        self.assertEqual(s["active_path"], "exit_a")
        self.assertTrue(any("landed" in l for l in s["log"]))

    def test_alarm_with_a_blocked_flies_exit_b(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        m.handle(Command("exit_blocked", {"exit": "A"}))
        m.handle(Command("alarm"))
        s = bus.snapshot_dict()
        self.assertEqual(s["alarm"]["exit"], "B")
        self.assertTrue(s["alarm"]["active"])
        self.assertEqual(s["alarm"]["blocked_exits"], ["A"])
        self.assertEqual(s["active_path"], "exit_b")
        self.assertEqual(m.fl.state, "idle")
        self.assertGreater(len(cf.commander.setpoints), 50)

    def test_exit_blocked_mid_flight_lands_then_relaunches_other_exit(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        setpoint_hook(cf, 40, lambda: m.handle(Command("exit_blocked", {"exit": "A"})))
        m.handle(Command("alarm"))
        s = bus.snapshot_dict()
        text = "\n".join(s["log"])
        self.assertIn("REROUTE", text)
        self.assertIn("relaunch", text)
        self.assertEqual(s["alarm"]["exit"], "B")
        self.assertEqual(s["active_path"], "exit_b")
        self.assertEqual(m.fl.state, "idle")
        self.assertFalse(m.armed)
        self.assertGreaterEqual(cf.commander.stops, 2)          # two landings
        self.assertEqual(cf.platform.arming, [True, True])      # two takeoffs

    def test_exit_blocked_without_other_route_logs_and_keeps_flying(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        store.directory.joinpath("exit_b.json").unlink(missing_ok=True)
        store.demo = False                                       # demo set gone -> only files remain
        p = store.get("exit_a") or paths.demo_paths()["exit_a"]
        store.save(p)                                            # exit_a on disk, no exit_b
        setpoint_hook(cf, 40, lambda: m.handle(Command("exit_blocked", {"exit": "A"})))
        m.handle(Command("alarm"))
        s = bus.snapshot_dict()
        self.assertTrue(any("cannot reroute" in l for l in s["log"]))
        self.assertEqual(m.fl.state, "idle")
        self.assertEqual(cf.platform.arming, [True])             # one takeoff only

    def test_land_command_lands_and_cancels_relaunch(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)

        def block_then_land():
            m.handle(Command("exit_blocked", {"exit": "A"}))
            m.handle(Command("land"))

        setpoint_hook(cf, 40, block_then_land)
        m.handle(Command("alarm"))
        self.assertEqual(m.fl.state, "idle")
        self.assertEqual(cf.platform.arming, [True])             # no relaunch after an operator landing

    def test_stop_mid_flight_estops_three_times_then_clear(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        setpoint_hook(cf, 30, lambda: q.push("stop"))            # fast path via CommandQueue.on_push
        m.handle(Command("cast", {"name": "exit_a"}))
        self.assertEqual(m.fl.state, "estop")
        self.assertEqual(cf.loc.emergency_stops, 3)
        self.assertFalse(m.armed)
        m.handle(q.pop_nowait())                                 # the queued copy: log only
        self.assertEqual(cf.loc.emergency_stops, 3)
        m.handle(Command("clear_alarm"))
        self.assertEqual(m.fl.state, "idle")
        s = bus.snapshot_dict()
        self.assertEqual(s["alarm"]["blocked_exits"], [])
        self.assertFalse(s["alarm"]["active"])
        self.assertFalse(s["flight"]["armed"])

    def test_stop_while_idle_fires_emergency_stop_immediately(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        q.push("stop")
        self.assertEqual(cf.loc.emergency_stops, 3)
        self.assertEqual(m.fl.state, "estop")
        self.assertFalse(m.armed)
        m.handle(q.pop_nowait())
        self.assertEqual(cf.loc.emergency_stops, 3)

    def test_stop_during_preflight_cancels_the_mission(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        m.fl.telemetry = converged_telemetry(1.5, 0.0, 0.3)      # forces the reset-and-wait path
        orig = m.fl.setup_estimator

        def reset_then_stop():
            orig()
            q.push("stop")

        m.fl.setup_estimator = reset_then_stop
        m.handle(Command("cast", {"name": "exit_a"}))
        self.assertEqual(cf.platform.arming, [])                 # never armed the drone
        self.assertEqual(m.fl.state, "estop")

    def test_connection_lost_disarms_and_refuses(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        m._on_connection_lost("ble://x", "timeout")
        self.assertFalse(m.armed)
        s = bus.snapshot_dict()
        self.assertFalse(s["link"]["connected"])
        m.handle(Command("arm", {"on": True}))
        m.handle(Command("cast", {"name": "exit_a"}))
        self.assertEqual(cf.commander.setpoints, [])
        self.assertTrue(any("link lost" in l for l in bus.snapshot_dict()["log"]))

    def test_threaded_flight_does_not_block_handle(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True, inline=False)
        m.handle(Command("cast", {"name": "exit_a"}))            # returns at once; the flight runs in a thread
        self.assertIsNotNone(m._flight_thread)
        m._flight_thread.join(5.0)
        self.assertFalse(m._flight_thread.is_alive())
        self.assertEqual(m.fl.state, "idle")
        self.assertFalse(m.armed)

    def test_set_mode_select_path_and_unknowns(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        m.handle(Command("set_mode", {"mode": "spell"}))
        m.handle(Command("select_path", {"name": "square"}))
        m.handle(Command("select_path", {"name": "nope"}))
        m.handle(Command("set_mode", {"mode": "weird"}))
        s = bus.snapshot_dict()
        self.assertEqual(s["mode"], "spell")
        self.assertEqual(s["active_path"], "square")
        self.assertTrue(any("no path" in l for l in s["log"]))
        self.assertTrue(any("unknown mode" in l for l in s["log"]))


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_reports_tracker_link_flight(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        m.publish()
        s = bus.snapshot_dict()
        self.assertTrue(s["tracking"]["ok"])
        self.assertAlmostEqual(s["drone"]["z"], 0.3)
        self.assertAlmostEqual(s["link"]["battery_v"], 3.9)
        self.assertTrue(s["link"]["connected"])
        self.assertTrue(s["flight"]["armed"])
        self.assertTrue(s["flight"]["estimator_converged"])
        self.assertEqual(s["flight"]["state"], "idle")
        self.assertFalse(s["replay"]["active"])
        self.assertEqual(s["mode"], "guide")
        self.assertEqual(s["active_path"], "exit_a")

    def test_publish_falls_back_to_estimate_when_tracking_lost(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, tracker_kwargs={"tracking_ok": False})
        m.publish()
        s = bus.snapshot_dict()
        self.assertFalse(s["tracking"]["ok"])
        self.assertAlmostEqual(s["drone"]["z"], 0.3)               # from converged_telemetry


class HoverCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_hover_refused_while_disarmed(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        m.handle(Command("hover"))
        self.assertEqual(cf.commander.setpoints, [])
        self.assertTrue(any("hover refused" in l and "disarmed" in l for l in bus.snapshot_dict()["log"]))

    def test_hover_climbs_holds_lands_and_disarms(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        m.handle(Command("hover", {"height_m": 0.6, "seconds": 2}))
        sp = cf.commander.setpoints
        self.assertAlmostEqual(len(sp), 20 + 10 + 40 + 20, delta=3)      # up 1 s, settle 0.5 s, hold 2 s, down 1 s
        self.assertAlmostEqual(max(p[2] for p in sp), 0.6, places=6)
        self.assertTrue(all(abs(p[0]) < 1e-9 and abs(p[1]) < 1e-9 for p in sp))   # straight up from (0,0)
        self.assertEqual(m.fl.state, "idle")
        self.assertFalse(m.armed)
        s = bus.snapshot_dict()
        self.assertEqual(s["active_path"], "exit_a")                     # a hover does not touch the selection
        self.assertFalse(s["replay"]["active"])
        self.assertTrue(any("hover test" in l for l in s["log"]))
        self.assertTrue(any("landed" in l for l in s["log"]))
        self.assertEqual(m.fl.cfg.takeoff_height, 0.5)                   # restored afterwards

    def test_hover_height_and_time_are_clamped(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True)
        m.handle(Command("hover", {"height_m": 5.0, "seconds": 0.0}))
        sp = cf.commander.setpoints
        self.assertLessEqual(max(p[2] for p in sp), 1.2 + 1e-9)
        self.assertGreaterEqual(len(sp), 20 + 10 + 20 + 20 - 3)          # hold clamped up to >= 1 s

    def test_hover_uses_config_defaults(self):
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name, armed=True,
                                                      cfg=MissionConfig(relaunch_delay_s=0.1, hover_hold_s=1.0, hover_height_m=0.8))
        m.handle(Command("hover"))
        sp = cf.commander.setpoints
        self.assertAlmostEqual(max(p[2] for p in sp), 0.8, places=6)
        self.assertAlmostEqual(len(sp), 20 + 10 + 20 + 20, delta=3)


class _View:
    def __init__(self, frame, markers=None):
        self.index = 1
        self.frame = frame
        self.t = 0.0
        self.detections = {}
        self.fps = 30.0
        self.markers = markers or {}
        self.poses = {}


class _ViewTracker:
    def __init__(self, view):
        self._view = view

    def get_views(self):
        return [self._view]

    def get_state(self):
        return None


class CameraSnapshotTests(unittest.TestCase):
    def test_snapshot_is_a_jpeg_with_markers_drawn(self):
        from mission import camera_snapshot_jpeg
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        corners = np.array([[40, 30], [80, 30], [80, 70], [40, 70]], dtype=np.float32)
        data = camera_snapshot_jpeg(_ViewTracker(_View(frame, {0: corners})), label="id0 seen")
        self.assertIsInstance(data, bytes)
        self.assertEqual(data[:2], b"\xff\xd8")                      # JPEG magic
        self.assertTrue(np.all(frame == 0))                            # the tracker's frame is not modified

    def test_snapshot_without_a_frame_is_none(self):
        from mission import camera_snapshot_jpeg
        self.assertIsNone(camera_snapshot_jpeg(_ViewTracker(None)))
        self.assertIsNone(camera_snapshot_jpeg(object()))              # tracker without get_views (SimTracker)


class NanTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_nan_position_counts_as_not_tracked_and_nan_yaw_becomes_zero(self):
        import json
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        tr.pos[:] = (float("nan"), 0.0, 0.3)
        m.publish()
        s = bus.snapshot_dict()
        json.dumps(s, allow_nan=False)
        self.assertFalse(s["tracking"]["ok"])
        self.assertAlmostEqual(s["drone"]["z"], 0.3)             # falls back to the estimate
        tr.pos[:] = (0.1, 0.0, 0.3)
        tr.yaw_value = float("nan")
        m.publish()
        s = bus.snapshot_dict()
        json.dumps(s, allow_nan=False)
        self.assertTrue(s["tracking"]["ok"])
        self.assertEqual(s["drone"]["yaw"], 0.0)


class TelemetryDictTests(unittest.TestCase):
    def test_telemetry_dict_reports_estimator_numbers(self):
        import json
        from mission import telemetry_dict
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        d = telemetry_dict(m)
        json.dumps(d, allow_nan=False)
        self.assertTrue(d["converged"])
        self.assertEqual(d["position"], [0.0, 0.0, 0.3])
        self.assertEqual(len(d["var"]), 3)
        self.assertAlmostEqual(d["battery_v"], 3.9)
        self.assertEqual(d["n_updates"], 12)
        self.assertEqual(d["flight_state"], "idle")
        self.assertIn("feed", d)
        self.assertIn("camera_xyz", d)
        self.assertIn("battery_min_v", d)                 # lowest voltage seen under load
        self.assertIsNone(d["supervisor"])                 # no supervisor watch in sim
        m.supervisor = FakeSupervisor(info=hop.BIT_CRASHED | hop.BIT_CAN_BE_ARMED, vbat=3.95)
        d = telemetry_dict(m)
        self.assertIn("CRASHED", d["supervisor"])
        self.assertAlmostEqual(d["supervisor_vbat"], 3.95)

    def test_telemetry_dict_with_no_samples(self):
        import json
        from mission import telemetry_dict
        m, cf, bus, q, store, clock, tr = make_mission(self.tmp.name)
        m.fl.telemetry = flight.Telemetry()
        d = telemetry_dict(m)
        json.dumps(d, allow_nan=False)
        self.assertFalse(d["converged"])
        self.assertIsNone(d["battery_v"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
