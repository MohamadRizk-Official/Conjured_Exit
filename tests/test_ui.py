"""Tests for the operator UI backend (``ui.state``, ``ui.server``, ``ui.sim``).

No hardware, no browser.  The server tests run a real uvicorn instance on a
free localhost port in a background thread (httpx / fastapi.testclient is not
installed in cf64, and the brief allows only fastapi, uvicorn and websockets).

    cf64\\Scripts\\python.exe -m unittest tests.test_ui -v
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flight  # noqa: E402  (pure python at import time: Telemetry is used by the live-bridge fakes)
from ui.live import HWCHECK_LOG_NAME, HWCHECK_PERIOD_MS, HWCHECK_VARIABLES, LiveBridge  # noqa: E402
from ui.pathstore import PathStore  # noqa: E402
from ui.server import SourceManager, create_app  # noqa: E402
from ui.sim import Simulator  # noqa: E402
from ui.state import (  # noqa: E402
    COMMAND_NAMES,
    COMMAND_SPECS,
    LIVE_POINTS_MAX,
    LOG_MAX,
    SOURCES,
    AppState,
    Command,
    CommandQueue,
    StateBus,
)

STATE_KEYS = {
    "source", "mode", "link", "tracking", "drone", "wand", "recording", "paths", "active_path",
    "replay", "alarm", "flight", "hwcheck", "log", "ts",
}
DEMO_NAMES = {"spiral", "square", "exit_a", "exit_b"}


def _sample_args(name: str) -> dict:
    required, _ = COMMAND_SPECS[name]
    samples = {"mode": "spell", "name": "spiral", "exit": "A", "source": "sim", "on": True}
    return {a: samples[a] for a in required}


def _wait_until(pred, timeout: float = 3.0, step: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


# ------------------------------------------------------ fakes for the live bridge
# Attribute shape of cflib's Crazyflie / LogConfig and of flight.Flight as used by ui.live.
# No cflib, no bleak, no hardware.


class _Caller:
    def __init__(self) -> None:
        self.callbacks: list = []

    def add_callback(self, fn) -> None:
        self.callbacks.append(fn)

    def call(self, *args) -> None:
        for fn in list(self.callbacks):
            fn(*args)


class FakeLogConfig:
    def __init__(self, name: str, period_in_ms: int) -> None:
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables: list[tuple[str, str | None]] = []
        self.data_received_cb = _Caller()
        self.error_cb = _Caller()
        self.starts = 0
        self.stops = 0
        self.running = False

    def add_variable(self, name: str, fetch_as: str | None = None) -> None:
        self.variables.append((name, fetch_as))

    def start(self) -> None:
        self.starts += 1
        self.running = True

    def stop(self) -> None:
        self.stops += 1
        self.running = False

    def emit(self, timestamp_ms: int, data: dict) -> None:
        """What cflib does when a log packet arrives."""
        self.data_received_cb.call(timestamp_ms, data, self)


class FakeLog:
    def __init__(self) -> None:
        self.configs: list = []
        self.toc = None  # ui.live skips the TOC check when there is none

    def add_config(self, conf) -> None:
        self.configs.append(conf)


class FakeLiveCF:
    def __init__(self) -> None:
        self.log = FakeLog()
        self.connection_lost = _Caller()
        self.closed = 0

    def close_link(self) -> None:
        self.closed += 1


class FakeFlight:
    """Mirrors the flight.Flight surface ui.live uses (state, telemetry, on_state, requests)."""

    def __init__(self, cf) -> None:
        self.cf = cf
        self.state = "idle"
        self.telemetry = flight.Telemetry()
        self.on_state = None
        self.logconf = None
        self.estimator_setups = 0
        self.stop_requests = 0
        self.land_requests = 0
        self.emergency_stops = 0

    def setup_estimator(self) -> None:
        self.estimator_setups += 1

    def start_log(self) -> list[str]:
        self.logconf = FakeLogConfig("pathcaster", 100)
        self.logconf.start()
        return ["pm.vbat", "stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]

    def request_stop(self) -> None:
        self.stop_requests += 1

    def request_land(self) -> None:
        self.land_requests += 1

    def emergency_stop(self) -> None:
        self.emergency_stops += 1
        self.set_state("estop")

    def set_state(self, state: str) -> None:  # what Flight._set_state does
        self.state = state
        if self.on_state:
            self.on_state(state)


class FakeSim:
    """pause/resume/stop surface of ui.sim.Simulator."""

    def __init__(self) -> None:
        self.paused = True
        self.resumes = 0
        self.stopped = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self.resumes += 1

    def stop(self, timeout: float = 0.0) -> None:
        self.stopped = True


class FakeBridge:
    """connect/start/pause/close surface of ui.live.LiveBridge, without hardware."""

    def __init__(self, fail: bool = False, delay: float = 0.0) -> None:
        self.uri = "fake://0"
        self.fail = fail
        self.delay = delay
        self.connected = False
        self.active = False
        self.connects = 0
        self.closed = False

    def connect(self):
        self.connects += 1
        time.sleep(self.delay)
        if self.fail:
            raise ConnectionError("Too many packets lost")
        self.connected = True

    def start(self) -> None:
        self.active = True

    def pause(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False
        self.closed = True


# ----------------------------------------------------------------- state bus


class StateBusTests(unittest.TestCase):
    def test_default_state_is_plain_json(self):
        bus = StateBus()
        snap = bus.snapshot_dict()
        self.assertEqual(set(snap), STATE_KEYS)
        json.dumps(snap)  # must not raise
        self.assertEqual(snap["flight"]["state"], "idle")
        self.assertIsNone(snap["active_path"])

    def test_source_and_hwcheck_fields(self):
        snap = StateBus().snapshot_dict()
        self.assertEqual(snap["source"], "none")
        self.assertIn(snap["source"], SOURCES)
        self.assertEqual(snap["hwcheck"], {"active": False, "roll_deg": 0.0, "pitch_deg": 0.0, "ts": 0.0})
        self.assertEqual(set(snap["link"]), {"uri", "connected", "connecting", "battery_v", "rssi", "error"})
        self.assertIn("set_source", COMMAND_NAMES)
        self.assertEqual(COMMAND_SPECS["set_source"][0], ("source",))

    def test_update_merges_nested_dicts(self):
        bus = StateBus()
        v0 = bus.version
        bus.update(drone={"x": 1.25, "yaw": 0.5}, mode="spell", flight={"state": "flying"})
        s = bus.snapshot()
        self.assertIsInstance(s, AppState)
        self.assertEqual((s.drone.x, s.drone.y, s.drone.yaw), (1.25, 0.0, 0.5))
        self.assertEqual(s.mode, "spell")
        self.assertEqual(s.flight.state, "flying")
        self.assertTrue(s.flight.estimator_converged is False)  # untouched sibling field
        self.assertGreater(bus.version, v0)
        with self.assertRaises(KeyError):
            bus.update(nope=1)
        with self.assertRaises(KeyError):
            bus.update(drone={"pitch": 1})

    def test_patch_dotted_path(self):
        bus = StateBus()
        bus.patch("link.connected", True)
        bus.patch("tracking.fps", 29.5)
        bus.patch("alarm.blocked_exits", ["A"])
        s = bus.snapshot_dict()
        self.assertTrue(s["link"]["connected"])
        self.assertEqual(s["tracking"]["fps"], 29.5)
        self.assertEqual(s["alarm"]["blocked_exits"], ["A"])
        with self.assertRaises(KeyError):
            bus.patch("drone.nope", 1)
        with self.assertRaises(KeyError):
            bus.patch("nothing.x", 1)

    def test_numpy_values_become_plain(self):
        import numpy as np

        bus = StateBus()
        bus.update(drone={"x": np.float64(0.5), "z": np.float32(1.0)})
        bus.update(paths=[{"name": "p", "mode": "guide", "length_m": np.float64(1.0), "duration_s": 3.0, "n_points": np.int64(4)}])
        snap = bus.snapshot_dict()
        json.dumps(snap)
        self.assertIsInstance(snap["drone"]["x"], float)
        self.assertIsInstance(snap["paths"][0]["n_points"], int)

    def test_two_threads_updating_concurrently(self):
        bus = StateBus()
        n = 3000
        errors: list[BaseException] = []

        def worker_a():
            try:
                for i in range(n):
                    bus.update(drone={"x": float(i)}, tracking={"fps": 30.0})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def worker_b():
            try:
                for i in range(n):
                    bus.patch("wand.y", float(i))
                    if i % 100 == 0:
                        bus.log(f"line {i}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        snaps: list[dict] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                snaps.append(bus.snapshot_dict())

        ta, tb, tr = threading.Thread(target=worker_a), threading.Thread(target=worker_b), threading.Thread(target=reader)
        tr.start()
        ta.start()
        tb.start()
        ta.join(30)
        tb.join(30)
        stop.set()
        tr.join(5)
        self.assertEqual(errors, [])
        s = bus.snapshot_dict()
        self.assertEqual(s["drone"]["x"], float(n - 1))
        self.assertEqual(s["wand"]["y"], float(n - 1))
        self.assertGreaterEqual(bus.version, 2 * n)
        self.assertLessEqual(len(s["log"]), LOG_MAX)
        self.assertGreater(len(snaps), 0)
        for snap in snaps[:50]:
            self.assertEqual(set(snap), STATE_KEYS)

    def test_live_points_and_log_are_capped(self):
        bus = StateBus()
        for i in range(LIVE_POINTS_MAX + 120):
            bus.append_live_point(i, 0.0, 0.5)
        for i in range(LOG_MAX + 25):
            bus.log(f"msg {i}")
        s = bus.snapshot_dict()
        self.assertEqual(len(s["recording"]["live_points"]), LIVE_POINTS_MAX)
        self.assertEqual(s["recording"]["n_samples"], LIVE_POINTS_MAX + 120)
        self.assertEqual(s["recording"]["live_points"][-1][0], float(LIVE_POINTS_MAX + 119))
        self.assertEqual(len(s["log"]), LOG_MAX)
        self.assertTrue(s["log"][-1].endswith(f"msg {LOG_MAX + 24}"))

    def test_subscription_from_thread(self):
        bus = StateBus()
        sub = bus.subscribe()
        self.assertTrue(sub.changed())  # never taken yet
        sub.take()
        self.assertFalse(sub.changed())
        self.assertFalse(sub.wait(timeout=0.05))
        threading.Timer(0.05, lambda: bus.update(mode="spell")).start()
        self.assertTrue(sub.wait(timeout=2.0))
        self.assertEqual(sub.take()["mode"], "spell")
        sub.close()

    def test_subscription_from_asyncio(self):
        bus = StateBus()

        async def main():
            sub = bus.subscribe()  # binds to this loop
            sub.take()
            self.assertFalse(await sub.wait_async(timeout=0.05))
            threading.Timer(0.05, lambda: bus.patch("drone.z", 0.9)).start()
            self.assertTrue(await sub.wait_async(timeout=2.0))
            snap = sub.take()
            sub.close()
            return snap

        snap = asyncio.run(main())
        self.assertEqual(snap["drone"]["z"], 0.9)


# -------------------------------------------------------------- command queue


class CommandQueueTests(unittest.TestCase):
    def test_round_trip(self):
        q = CommandQueue()
        before = time.time()
        cmd = q.push("cast", {"name": "spiral"})
        self.assertIsInstance(cmd, Command)
        self.assertEqual(len(q), 1)
        got = q.pop(timeout=1.0)
        self.assertEqual((got.name, got.args, got.source), ("cast", {"name": "spiral"}, "ui"))
        self.assertGreaterEqual(got.ts, before)
        self.assertIsNone(q.pop(timeout=0.01))
        self.assertTrue(q.empty())

    def test_vocabulary(self):
        q = CommandQueue()
        for name in COMMAND_NAMES:
            q.push(name, _sample_args(name), source="voice")
        names = [c.name for c in q.drain()]
        self.assertEqual(names, list(COMMAND_NAMES))
        self.assertIn("alarm", names)
        self.assertIn("exit_blocked", names)
        with self.assertRaises(ValueError):
            q.push("launch_missiles")
        with self.assertRaises(ValueError):
            q.push("cast")  # needs name
        with self.assertRaises(ValueError):
            q.push("cast", "spiral")  # args must be a dict

    def test_threads_push_and_pop(self):
        q = CommandQueue()
        seen: list[str] = []

        def producer(tag):
            for i in range(200):
                q.push("select_path", {"name": f"{tag}{i}"})

        def consumer():
            while len(seen) < 400:
                c = q.pop(timeout=2.0)
                if c is None:
                    return
                seen.append(c.args["name"])

        threads = [threading.Thread(target=producer, args=("a",)), threading.Thread(target=producer, args=("b",)), threading.Thread(target=consumer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(len(seen), 400)
        self.assertEqual(q.pushed, 400)

    def test_on_push_listener(self):
        q = CommandQueue()
        got = []
        q.on_push(lambda c: got.append(c.name))
        q.push("land")
        self.assertEqual(got, ["land"])


# ------------------------------------------------------------------ simulator


class AsBoolTests(unittest.TestCase):
    def test_strings_and_values(self):
        from ui.state import as_bool
        for v in (True, 1, "1", "true", "True", " on ", "yes", "armed"):
            self.assertTrue(as_bool(v), v)
        for v in (False, 0, "0", "false", "off", "no", "", None):
            self.assertFalse(as_bool(v), v)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bus = StateBus()
        self.q = CommandQueue()
        self.store = PathStore(self.tmp.name, demo=True)
        self.sim = Simulator(self.bus, self.q, self.store)
        self.sim.setup()

    def tearDown(self):
        self.tmp.cleanup()

    def run_for(self, seconds: float, dt: float = 1 / 30):
        for _ in range(int(seconds / dt)):
            self.sim.step(dt)

    def test_setup_publishes_demo_paths(self):
        s = self.bus.snapshot_dict()
        self.assertEqual({p["name"] for p in s["paths"]}, DEMO_NAMES)
        self.assertTrue(s["link"]["connected"])
        self.assertEqual(s["active_path"], "exit_a")

    def test_alarm_flies_exit_a_then_reroutes_to_b(self):
        self.q.push("arm", {"on": True})
        self.q.push("alarm")
        self.run_for(0.2)
        s = self.bus.snapshot_dict()
        self.assertTrue(s["alarm"]["active"])
        self.assertEqual(s["alarm"]["exit"], "A")
        self.assertEqual(s["flight"]["state"], "takeoff")
        self.run_for(3.0)
        s = self.bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "flying")
        self.assertTrue(s["replay"]["active"])
        self.assertEqual(s["active_path"], "exit_a")
        self.q.push("exit_blocked", {"exit": "A"})
        self.run_for(0.2)
        s = self.bus.snapshot_dict()
        self.assertEqual(s["alarm"]["exit"], "B")
        self.assertEqual(s["active_path"], "exit_b")
        self.assertEqual(s["alarm"]["blocked_exits"], ["A"])
        self.run_for(20.0)  # transit + full exit_b + landing
        s = self.bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "idle")
        self.assertFalse(s["replay"]["active"])
        self.assertLess(s["drone"]["z"], 0.05)
        self.assertTrue(any("REROUTE" in line for line in s["log"]))

    def test_record_and_save_as_uses_real_path_engine(self):
        self.q.push("set_mode", {"mode": "spell"})
        self.q.push("record_start")
        self.run_for(3.0)
        s = self.bus.snapshot_dict()
        self.assertTrue(s["recording"]["active"])
        self.assertEqual(s["recording"]["mode"], "spell")
        self.assertGreater(s["recording"]["n_samples"], 50)
        self.assertEqual(len(s["recording"]["live_points"]), s["recording"]["n_samples"])
        self.assertTrue(s["wand"]["ok"])
        self.q.push("record_stop")
        self.q.push("save_as", {"name": "Test Loop"})
        self.run_for(0.1)
        s = self.bus.snapshot_dict()
        self.assertIn("test_loop", {p["name"] for p in s["paths"]})
        self.assertEqual(s["active_path"], "test_loop")
        self.assertTrue((pathlib.Path(self.tmp.name) / "test_loop.json").is_file())
        saved = self.store.get("test_loop")
        self.assertEqual(saved.mode, "spell")
        self.assertGreater(len(saved.points), 2)

    def test_stop_and_clear(self):
        self.q.push("arm", {"on": True})
        self.q.push("cast", {"name": "spiral"})
        self.run_for(2.0)
        self.q.push("stop")
        self.run_for(0.1)
        self.assertEqual(self.bus.snapshot_dict()["flight"]["state"], "estop")
        self.q.push("clear_alarm")
        self.run_for(0.1)
        s = self.bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "idle")
        self.assertFalse(s["alarm"]["active"])

    def test_alarm_refused_while_disarmed(self):
        self.q.push("alarm")
        self.run_for(0.2)
        s = self.bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "idle")
        self.assertFalse(s["alarm"]["active"])
        self.assertFalse(s["flight"]["armed"])
        self.assertTrue(any("disarmed" in line for line in s["log"]))

    def test_cast_refused_while_disarmed(self):
        self.q.push("cast", {"name": "spiral"})
        self.run_for(0.2)
        self.assertEqual(self.bus.snapshot_dict()["flight"]["state"], "idle")

    def test_arm_then_alarm_disarms_after_landing(self):
        self.q.push("arm", {"on": True})
        self.run_for(0.1)
        self.assertTrue(self.bus.snapshot_dict()["flight"]["armed"])
        self.q.push("alarm")
        self.run_for(0.2)
        self.assertEqual(self.bus.snapshot_dict()["flight"]["state"], "takeoff")
        self.run_for(40.0)  # takeoff + exit_a + landing
        s = self.bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "idle")
        self.assertFalse(s["flight"]["armed"])

    def test_stop_disarms(self):
        self.q.push("arm", {"on": True})
        self.q.push("cast", {"name": "spiral"})
        self.run_for(1.0)
        self.q.push("stop")
        self.run_for(0.1)
        s = self.bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "estop")
        self.assertFalse(s["flight"]["armed"])

    def test_arm_accepts_string_false(self):
        self.q.push("arm", {"on": True})
        self.q.push("arm", {"on": "false"})
        self.run_for(0.1)
        self.assertFalse(self.bus.snapshot_dict()["flight"]["armed"])


# ---------------------------------------------------------------- live bridge


class LiveBridgeTests(unittest.TestCase):
    def make(self, fail: bool = False):
        bus, q = StateBus(), CommandQueue()
        cf = FakeLiveCF()
        fl = FakeFlight(cf)
        calls = {"connect": 0, "uris": []}

        def connect_fn(uri: str):
            calls["connect"] += 1
            calls["uris"].append(uri)
            if fail:
                raise TimeoutError(f"no connection to {uri} after 120 s")
            return cf

        bridge = LiveBridge(
            bus, q, uri="fake://0",
            connect_fn=connect_fn,
            flight_factory=lambda cf_, uri: fl,
            logconfig_factory=FakeLogConfig,
        )
        return bus, q, cf, fl, bridge, calls

    def test_connect_once_and_publish_link(self):
        bus, q, cf, fl, bridge, calls = self.make()
        self.assertFalse(bridge.connected)
        out = bridge.connect()
        self.assertIs(out, fl)
        self.assertEqual(calls, {"connect": 1, "uris": ["fake://0"]})
        self.assertEqual(fl.estimator_setups, 1)
        self.assertTrue(fl.logconf.running)  # the single 10 Hz flight block
        s = bus.snapshot_dict()
        self.assertEqual((s["link"]["uri"], s["link"]["connected"], s["link"]["connecting"]), ("fake://0", True, False))
        self.assertTrue(any("connecting to fake://0" in line for line in s["log"]))
        self.assertIsNone(bridge.hwcheck_conf)  # not publishing yet -> no hwcheck block
        bridge.connect()  # idempotent: never a second connection
        self.assertEqual(calls["connect"], 1)

    def test_connect_failure_is_reported(self):
        bus, q, cf, fl, bridge, calls = self.make(fail=True)
        with self.assertRaises(TimeoutError):
            bridge.connect()
        self.assertFalse(bridge.connected)
        s = bus.snapshot_dict()
        self.assertFalse(s["link"]["connected"])
        self.assertFalse(s["link"]["connecting"])
        self.assertIn("TimeoutError", s["link"]["error"])
        self.assertTrue(any("FAILED" in line for line in s["log"]))

    def test_hwcheck_block_only_while_idle(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        conf = bridge.hwcheck_conf
        self.assertIsNotNone(conf)
        self.assertEqual((conf.name, conf.period_in_ms), (HWCHECK_LOG_NAME, HWCHECK_PERIOD_MS))
        self.assertEqual(conf.period_in_ms, 200)
        self.assertEqual(conf.variables, [("stabilizer.roll", "FP16"), ("stabilizer.pitch", "FP16")])
        self.assertEqual(conf.variables, HWCHECK_VARIABLES)
        self.assertEqual(cf.log.configs, [conf])
        self.assertEqual((conf.starts, conf.stops), (1, 0))
        self.assertTrue(bus.snapshot_dict()["hwcheck"]["active"])

        fl.set_state("takeoff")  # the moment the state leaves idle
        self.assertEqual((conf.starts, conf.stops), (1, 1))
        self.assertFalse(conf.running)
        self.assertFalse(bus.snapshot_dict()["hwcheck"]["active"])
        fl.set_state("hover")
        fl.set_state("flying")
        self.assertEqual(conf.stops, 1)  # no double stop
        fl.set_state("idle")  # back on the ground -> restarted, same block
        self.assertEqual((conf.starts, conf.stops), (2, 1))
        self.assertTrue(bus.snapshot_dict()["hwcheck"]["active"])
        self.assertEqual(cf.log.configs, [conf])  # never re-added

        fl.state = "landing"  # state changed without the callback: the 10 Hz poll catches it
        bridge.tick()
        self.assertEqual(conf.stops, 2)
        self.assertFalse(bus.snapshot_dict()["hwcheck"]["active"])

    def test_hwcheck_resolves_ble_mangled_toc_names(self):
        bus, q, cf, fl, bridge, calls = self.make()
        # what cflib stores after the nRF 2024.10 corruption (tests/test_toc_names.py LOG_TOC)
        cf.log.toc = {"stabilizer": {"rol\x00": 216, "pith\x00": 217, "yaw": 218}, "pm": {"vbat": 89}}
        bridge.connect()
        bridge.start(spawn_thread=False)
        conf = bridge.hwcheck_conf
        self.assertEqual(conf.variables, [("stabilizer.rol\x00", "FP16"), ("stabilizer.pith\x00", "FP16")])
        conf.emit(5000, {"stabilizer.rol\x00": 2.0, "stabilizer.pith\x00": -3.0})  # keys = stored names
        s = bus.snapshot_dict()
        self.assertEqual((s["hwcheck"]["roll_deg"], s["hwcheck"]["pitch_deg"]), (2.0, -3.0))

    def test_hwcheck_disabled_when_toc_lacks_roll_pitch(self):
        bus, q, cf, fl, bridge, calls = self.make()
        cf.log.toc = {"stabilizer": {"yaw": 218}}
        bridge.connect()
        bridge.start(spawn_thread=False)
        self.assertFalse(bridge.hwcheck_running)
        self.assertFalse(bus.snapshot_dict()["hwcheck"]["active"])
        self.assertEqual(cf.log.configs, [])
        log_text = "\n".join(bus.snapshot_dict()["log"])
        self.assertIn("stabilizer.roll unresolvable", log_text)
        self.assertIn("hardware check disabled", log_text)
        bridge.tick()  # must not retry-spam
        self.assertEqual(cf.log.configs, [])

    def test_hwcheck_sample_reaches_bus(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        v0 = bus.version
        bridge.hwcheck_conf.emit(123456, {"stabilizer.roll": 3.5, "stabilizer.pitch": -1.25})
        s = bus.snapshot_dict()
        self.assertEqual(s["hwcheck"]["roll_deg"], 3.5)
        self.assertEqual(s["hwcheck"]["pitch_deg"], -1.25)
        self.assertTrue(s["hwcheck"]["active"])
        self.assertGreater(s["hwcheck"]["ts"], 0.0)
        self.assertGreater(bus.version, v0)
        self.assertEqual(bridge.hwcheck_samples, 1)

    def test_tick_publishes_telemetry(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        for i in range(12):
            fl.telemetry.update(i * 100, {"pm.vbat": 3.87, "stateEstimate.x": 0.1, "stateEstimate.y": -0.2,
                                          "stateEstimate.z": 0.3, "kalman.varPX": 0.0002, "kalman.varPY": 0.0002,
                                          "kalman.varPZ": 0.0002})
        bridge.tick()
        s = bus.snapshot_dict()
        self.assertEqual(s["drone"], {"x": 0.1, "y": -0.2, "z": 0.3, "yaw": 0.0})
        self.assertAlmostEqual(s["link"]["battery_v"], 3.87)
        self.assertTrue(s["link"]["connected"])
        self.assertEqual(s["flight"], {"state": "idle", "estimator_converged": True, "armed": False})

    def test_stop_command_calls_emergency_stop(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        q.push("stop")
        self.assertEqual(fl.stop_requests, 1)  # fast path, before any tick
        self.assertEqual(fl.emergency_stops, 0)
        bridge.tick()
        self.assertEqual(fl.emergency_stops, 1)
        self.assertGreaterEqual(fl.stop_requests, 2)
        self.assertEqual(len(q), 0)
        s = bus.snapshot_dict()
        self.assertEqual(s["flight"]["state"], "estop")
        self.assertFalse(s["hwcheck"]["active"])  # not idle any more
        self.assertTrue(any("EMERGENCY STOP" in line for line in s["log"]))

    def test_stop_fast_path_ignored_when_not_publishing(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()  # connected but SIM is the active source
        q.push("stop")
        self.assertEqual(fl.stop_requests, 0)
        self.assertEqual(len(q), 1)  # left for whoever is draining

    def test_land_and_unwired_commands(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        q.push("land")
        q.push("cast", {"name": "spiral"})
        q.push("alarm")
        bridge.tick()
        self.assertEqual(fl.land_requests, 1)
        self.assertEqual(fl.emergency_stops, 0)
        log_text = "\n".join(bus.snapshot_dict()["log"])
        self.assertIn("'cast' not wired in LIVE mode yet", log_text)
        self.assertIn("'alarm' not wired in LIVE mode yet", log_text)

    def test_connection_lost_is_shown(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        self.assertTrue(bridge.hwcheck_conf.running)
        cf.connection_lost.call("fake://0", "Too many packets lost")
        s = bus.snapshot_dict()
        self.assertFalse(s["link"]["connected"])
        self.assertIn("connection lost", s["link"]["error"])
        self.assertFalse(s["hwcheck"]["active"])
        self.assertTrue(any("CONNECTION LOST" in line for line in s["log"]))
        bridge.tick()  # the 10 Hz publish must not flip it back to connected
        self.assertFalse(bus.snapshot_dict()["link"]["connected"])

    def test_pause_and_close(self):
        bus, q, cf, fl, bridge, calls = self.make()
        bridge.connect()
        bridge.start(spawn_thread=False)
        conf = bridge.hwcheck_conf
        self.assertTrue(conf.running)
        bridge.pause()
        self.assertFalse(bridge.active)
        self.assertFalse(conf.running)
        self.assertTrue(bridge.connected)  # link stays up
        self.assertEqual(cf.closed, 0)
        bridge.tick()  # no-op while paused
        bridge.close()
        self.assertEqual(cf.closed, 1)
        self.assertEqual(fl.logconf.stops, 1)
        self.assertFalse(bridge.connected)
        self.assertFalse(bus.snapshot_dict()["link"]["connected"])


# ------------------------------------------------------------- source manager


class SourceManagerTests(unittest.TestCase):
    def test_sim_to_live_and_back(self):
        bus, q = StateBus(), CommandQueue()
        sim, live = FakeSim(), FakeBridge(delay=0.05)
        m = SourceManager(bus, q, sim=sim, live=live)
        self.assertEqual(m.available, ["sim", "live", "none"])
        self.assertEqual(m.set_source("sim")["source"], "sim")
        self.assertFalse(sim.paused)
        self.assertEqual(bus.snapshot_dict()["source"], "sim")

        st = m.set_source("live")
        self.assertEqual(st["source"], "live")
        self.assertTrue(st["connecting"])
        self.assertTrue(sim.paused)
        self.assertTrue(bus.snapshot_dict()["link"]["connecting"])
        with self.assertRaises(ValueError):
            m.set_source("sim")  # refused while the connect is in progress
        self.assertTrue(_wait_until(lambda: not m.connecting))
        self.assertEqual(m.source, "live")
        self.assertEqual(live.connects, 1)
        self.assertTrue(live.active)
        self.assertEqual(bus.snapshot_dict()["source"], "live")

        self.assertEqual(m.set_source("sim")["source"], "sim")
        self.assertFalse(live.active)
        self.assertTrue(live.connected)  # link kept
        self.assertFalse(sim.paused)
        self.assertEqual(sim.resumes, 2)
        m.set_source("live")  # already connected: no worker, immediate
        self.assertFalse(m.connecting)
        self.assertEqual(live.connects, 1)
        self.assertTrue(live.active)
        m.close()
        self.assertTrue(live.closed)
        self.assertTrue(sim.stopped)

    def test_live_connect_failure_falls_back_to_sim(self):
        bus, q = StateBus(), CommandQueue()
        sim, live = FakeSim(), FakeBridge(fail=True)
        m = SourceManager(bus, q, sim=sim, live=live)
        m.set_source("sim")
        m.set_source("live")
        self.assertTrue(_wait_until(lambda: not m.connecting))
        self.assertEqual(m.source, "sim")
        self.assertFalse(sim.paused)
        self.assertFalse(live.active)
        s = bus.snapshot_dict()
        self.assertEqual(s["source"], "sim")
        self.assertIn("ConnectionError: Too many packets lost", s["link"]["error"])
        self.assertTrue(any("LIVE unavailable" in line and "back to SIM" in line for line in s["log"]))
        m.set_source("none")  # an explicit switch clears the error badge
        self.assertEqual(bus.snapshot_dict()["link"]["error"], "")

    def test_live_connect_failure_without_sim_goes_to_none(self):
        bus, q = StateBus(), CommandQueue()
        m = SourceManager(bus, q, live=FakeBridge(fail=True))
        m.set_source("live")
        self.assertTrue(_wait_until(lambda: not m.connecting))
        self.assertEqual(m.source, "none")
        self.assertEqual(bus.snapshot_dict()["source"], "none")

    def test_unavailable_sources_raise(self):
        bus, q = StateBus(), CommandQueue()
        m = SourceManager(bus, q)
        self.assertEqual(m.available, ["none"])
        with self.assertRaises(ValueError):
            m.set_source("live")
        with self.assertRaises(ValueError):
            m.set_source("sim")
        with self.assertRaises(ValueError):
            m.set_source("bogus")
        self.assertEqual(m.set_source("none")["source"], "none")


# --------------------------------------------------------------------- server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("content-type", ""), r.read()


def _post_json(url: str, body: dict, timeout: float = 5.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class ServerTests(unittest.TestCase):
    """Real uvicorn on a free port, fresh bus/queue, demo paths (what --sim uses)."""

    @classmethod
    def setUpClass(cls):
        import uvicorn

        cls.tmp = tempfile.TemporaryDirectory()
        cls.bus = StateBus()
        cls.q = CommandQueue()
        cls.store = PathStore(cls.tmp.name, demo=True)
        cls.fake_sim = FakeSim()
        cls.fake_live = FakeBridge()
        cls.manager = SourceManager(cls.bus, cls.q, sim=cls.fake_sim, live=cls.fake_live)
        cls.app = create_app(cls.bus, cls.q, cls.store, manager=cls.manager)
        cls.manager.set_source("sim")
        cls.port = _free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cfg = uvicorn.Config(cls.app, host="127.0.0.1", port=cls.port, log_level="warning", lifespan="off")
        cls.server = uvicorn.Server(cfg)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 15
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.02)
        if not cls.server.started:
            raise RuntimeError("uvicorn did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(10)
        cls.tmp.cleanup()

    def setUp(self):
        self.q.drain()

    def test_index_serves_html(self):
        status, ctype, body = _get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        text = body.decode("utf-8", "replace")
        self.assertIn("<!doctype html>", text.lower())
        self.assertIn("Pathcaster", text)

    def test_favicon(self):
        status, ctype, body = _get(self.base + "/favicon.ico")
        self.assertEqual(status, 200)
        self.assertIn("image/svg+xml", ctype)
        self.assertIn(b"<svg", body)

    def test_state_has_documented_keys(self):
        status, ctype, body = _get(self.base + "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        snap = json.loads(body)
        self.assertEqual(set(snap), STATE_KEYS)
        self.assertEqual(set(snap["link"]), {"uri", "connected", "connecting", "battery_v", "rssi", "error"})
        self.assertEqual(set(snap["hwcheck"]), {"active", "roll_deg", "pitch_deg", "ts"})
        self.assertIn(snap["source"], SOURCES)
        self.assertEqual(set(snap["tracking"]), {"ok", "fps", "latency_ms"})
        self.assertEqual(set(snap["drone"]), {"x", "y", "z", "yaw"})
        self.assertEqual(set(snap["wand"]), {"x", "y", "z", "ok"})
        self.assertEqual(set(snap["recording"]), {"active", "mode", "n_samples", "live_points"})
        self.assertEqual(set(snap["replay"]), {"active", "t", "duration", "progress"})
        self.assertEqual(set(snap["alarm"]), {"active", "exit", "blocked_exits"})
        self.assertEqual(set(snap["flight"]), {"state", "estimator_converged", "armed"})
        self.assertEqual({p["name"] for p in snap["paths"]}, DEMO_NAMES)

    def test_config(self):
        _, _, body = _get(self.base + "/api/config")
        cfg = json.loads(body)
        self.assertEqual(set(cfg["geofence"]), {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
        self.assertEqual(set(cfg["commands"]), set(COMMAND_NAMES))
        self.assertTrue(cfg["demo"])
        self.assertEqual(cfg["sources"], ["sim", "live", "none"])

    def test_set_source_round_trip_over_http(self):
        self.assertEqual(self.manager.source, "sim")
        status, resp = _post_json(self.base + "/api/command", {"name": "set_source", "args": {"source": "live"}})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["status"]["source"], "live")
        self.assertEqual(len(self.q), 0)  # handled by the server, never queued
        self.assertTrue(_wait_until(lambda: not self.manager.connecting))
        self.assertTrue(self.fake_live.connected)
        self.assertTrue(self.fake_live.active)
        self.assertTrue(self.fake_sim.paused)
        _, _, body = _get(self.base + "/api/source")
        src = json.loads(body)
        self.assertEqual((src["source"], src["connecting"], src["live_connected"]), ("live", False, True))
        self.assertEqual(json.loads(_get(self.base + "/api/state")[2])["source"], "live")

        status, resp = _post_json(self.base + "/api/command", {"name": "set_source", "args": {"source": "sim"}})
        self.assertEqual(status, 200)
        self.assertEqual(resp["status"]["source"], "sim")
        self.assertFalse(self.fake_live.active)
        self.assertTrue(self.fake_live.connected)
        self.assertFalse(self.fake_sim.paused)
        self.assertEqual(json.loads(_get(self.base + "/api/state")[2])["source"], "sim")

        status, resp = _post_json(self.base + "/api/command", {"name": "set_source", "args": {"source": "bogus"}})
        self.assertEqual(status, 400)
        self.assertEqual(self.manager.source, "sim")

    def test_paths_list_and_detail(self):
        _, _, body = _get(self.base + "/api/paths")
        paths = json.loads(body)
        self.assertEqual({p["name"] for p in paths}, DEMO_NAMES)
        for p in paths:
            self.assertEqual(set(p), {"name", "mode", "length_m", "duration_s", "n_points"})
            self.assertGreater(p["length_m"], 0)
        _, _, body = _get(self.base + "/api/paths/spiral")
        d = json.loads(body)
        self.assertEqual(d["mode"], "spell")
        self.assertEqual(len(d["points"]), len(d["times"]))
        self.assertEqual(len(d["points"][0]), 3)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _get(self.base + "/api/paths/does_not_exist")
        self.assertEqual(ctx.exception.code, 404)

    def test_post_command_enqueues(self):
        status, resp = _post_json(self.base + "/api/command", {"name": "cast", "args": {"name": "square"}})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        cmd = self.q.pop(timeout=1.0)
        self.assertEqual((cmd.name, cmd.args), ("cast", {"name": "square"}))
        status, resp = _post_json(self.base + "/api/command", {"name": "bogus"})
        self.assertEqual(status, 400)
        self.assertIsNone(self.q.pop(timeout=0.05))

    def test_select_path_and_set_mode_apply_immediately(self):
        _post_json(self.base + "/api/command", {"name": "select_path", "args": {"name": "exit_b"}})
        _post_json(self.base + "/api/command", {"name": "set_mode", "args": {"mode": "spell"}})
        s = self.bus.snapshot_dict()
        self.assertEqual(s["active_path"], "exit_b")
        self.assertEqual(s["mode"], "spell")
        self.assertEqual([c.name for c in self.q.drain()], ["select_path", "set_mode"])

    def test_websocket_streams_snapshots_and_accepts_commands(self):
        from websockets.sync.client import connect

        with connect(f"ws://127.0.0.1:{self.port}/ws", open_timeout=5) as ws:
            first = json.loads(ws.recv(timeout=5))
            self.assertEqual(first["type"], "state")
            self.assertEqual(set(first["state"]), STATE_KEYS)

            ws.send(json.dumps({"type": "command", "name": "alarm", "args": {}}))
            ack = None
            for _ in range(10):
                msg = json.loads(ws.recv(timeout=5))
                if msg["type"] == "ack":
                    ack = msg
                    break
            self.assertIsNotNone(ack)
            self.assertEqual(ack["command"]["name"], "alarm")
            cmd = self.q.pop(timeout=1.0)
            self.assertEqual(cmd.name, "alarm")

            # a producer thread changes the state -> a new snapshot arrives
            self.bus.update(drone={"x": 0.4242}, flight={"state": "flying"})
            got = None
            for _ in range(20):
                msg = json.loads(ws.recv(timeout=5))
                if msg["type"] == "state" and msg["state"]["drone"]["x"] == 0.4242:
                    got = msg["state"]
                    break
            self.assertIsNotNone(got)
            self.assertEqual(got["flight"]["state"], "flying")

            ws.send(json.dumps({"type": "command", "name": "nope"}))
            err = None
            for _ in range(10):
                msg = json.loads(ws.recv(timeout=5))
                if msg["type"] == "error":
                    err = msg
                    break
            self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
