"""Shared UI contract: the ``AppState`` that is streamed to the browser, the
thread-safe ``StateBus`` that holds it, and the ``CommandQueue`` through which
the browser (and voice) ask the flight/engine code to do things.

Nothing here touches hardware.  Import this from tracker/feed/flight/voice::

    from ui.state import BUS, COMMANDS, Command

    # producers (any thread, ~30 Hz is fine):
    BUS.update(drone={"x": x, "y": y, "z": z, "yaw": yaw},
               tracking={"ok": True, "fps": 29.7, "latency_ms": 33.0})
    BUS.patch("flight.state", "flying")
    BUS.log("takeoff ok")

    # consumers (the flight loop):
    cmd = COMMANDS.pop(timeout=0.1)          # Command | None
    if cmd and cmd.name == "cast": fly(cmd.args["name"])

Command vocabulary (``Command.name`` -> ``Command.args``).  The UI buttons and
the voice module push exactly these; the flight/engine loop pops them.  Voice
and AI never touch motors directly: a command only *selects* a path or an
action, and everything flies through the geofence + safety layer.

    alarm          {}                      Guide mode "fire" / "lead me out": take off and
                                           fly the active exit route (Exit A unless blocked).
    set_mode       {"mode": "guide"|"spell"}
    record_start   {}                      start recording (drone in guide, wand tip in spell)
    record_stop    {}                      stop recording; keep the raw samples for save_as
    save_as        {"name": str}           clean the last recording and save it as <name>
    cast           {"name": str}           fly the stored path <name> (spell "cast <name>")
    exit_blocked   {"exit": "A"|"B"}       voice "Exit A is blocked": reroute to the other exit
    land           {}                      controlled landing
    stop           {}                      EMERGENCY STOP (motors off)
    clear_alarm    {}                      reset alarm + blocked exits; back to idle
    select_path    {"name": str}           highlight/preview a stored path (no flight)
    set_source     {"source": "sim"|"live"|"none"}   which producer feeds the bus: the simulator
                                           (ui.sim) or the real drone link (ui.live).  Handled by
                                           the server itself (ui.server.SourceManager), not queued.

``AppState`` fields (all plain JSON types after ``to_json()``):

    source          "sim" | "live" | "none"     who is publishing drone/link/flight right now
    mode            "guide" | "spell"
    link            {uri, connected, connecting, battery_v, rssi, error}
    tracking        {ok, fps, latency_ms}
    drone           {x, y, z, yaw}              metres / radians, world frame (x fwd, y left, z up)
    wand            {x, y, z, ok}
    recording       {active, mode, n_samples, live_points}   live_points = last ~500 [x, y, z]
    paths           [{name, mode, length_m, duration_s, n_points}, ...]
    active_path     name | None
    replay          {active, t, duration, progress}
    alarm           {active, exit ("A"|"B"), blocked_exits [..]}
    flight          {state: "idle"|"takeoff"|"hover"|"flying"|"landing"|"estop", estimator_converged, armed}
                    (flight.Flight.STATES; "hover" = the hold after take-off and the hover test; armed = the ARMED switch)
    hwcheck         {active, roll_deg, pitch_deg, ts}   5 Hz stabilizer.roll/pitch while the real
                    drone is idle on the desk (ui.live), so tilting it by hand shows on screen
    log             last 50 status lines ("HH:MM:SS  text")
    ts              unix time of the last change
"""

from __future__ import annotations

import asyncio
import math
import copy
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Iterator

__all__ = [
    "COMMAND_SPECS",
    "COMMAND_NAMES",
    "MODES",
    "SOURCES",
    "FLIGHT_STATES",
    "LIVE_POINTS_MAX",
    "LOG_MAX",
    "Link",
    "Tracking",
    "Pose",
    "Wand",
    "Recording",
    "PathInfo",
    "Replay",
    "Alarm",
    "Flight",
    "HwCheck",
    "AppState",
    "Subscription",
    "StateBus",
    "Command",
    "CommandQueue",
    "BUS",
    "COMMANDS",
]

MODES: tuple[str, ...] = ("guide", "spell")
SOURCES: tuple[str, ...] = ("sim", "live", "none")
FLIGHT_STATES: tuple[str, ...] = ("idle", "takeoff", "hover", "flying", "landing", "estop")
LIVE_POINTS_MAX: int = 500
LOG_MAX: int = 50

#: name -> (argument names, one-line description).  Single source of truth for the vocabulary.
COMMAND_SPECS: dict[str, tuple[tuple[str, ...], str]] = {
    "alarm": ((), "take off and fly the active exit route (voice: 'fire', 'lead me out')"),
    "set_mode": (("mode",), "switch the engine between 'guide' and 'spell'"),
    "record_start": ((), "start recording the drone (guide) or the wand tip (spell)"),
    "record_stop": ((), "stop recording and keep the raw samples"),
    "save_as": (("name",), "clean the last recording and save it under <name>"),
    "cast": (("name",), "fly the stored path <name>"),
    "exit_blocked": (("exit",), "mark exit 'A' or 'B' blocked and reroute to the other one"),
    "land": ((), "controlled landing"),
    "stop": ((), "EMERGENCY STOP: motors off immediately"),
    "clear_alarm": ((), "reset alarm, blocked exits and e-stop; back to idle"),
    "arm": (("on",), "arm (true) or disarm (false) the drone; alarm/cast fly only while armed"),
    "hover": ((), "hover test: take off to ~0.6 m, hold 15 s, land (optional args height_m, seconds)"),
    "select_path": (("name",), "highlight / preview a stored path without flying"),
    "set_source": (("source",), "feed the UI from 'sim' (fake drone) or 'live' (real link); 'none' = nobody"),
}
COMMAND_NAMES: tuple[str, ...] = tuple(COMMAND_SPECS)


def as_bool(value: Any) -> bool:
    """Coerce a command argument to bool: bools/numbers as usual; strings 'true/1/yes/on/armed' are True."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "armed")
    return bool(value)


# ------------------------------------------------------------------- state model


@dataclass
class Link:
    uri: str = ""
    connected: bool = False
    connecting: bool = False
    battery_v: float = 0.0
    rssi: int = 0
    error: str = ""


@dataclass
class Tracking:
    ok: bool = False
    fps: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


@dataclass
class Wand:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    ok: bool = False


@dataclass
class Recording:
    active: bool = False
    mode: str = "guide"
    n_samples: int = 0
    live_points: list[list[float]] = field(default_factory=list)


@dataclass
class PathInfo:
    name: str
    mode: str
    length_m: float
    duration_s: float
    n_points: int = 0


@dataclass
class Replay:
    active: bool = False
    t: float = 0.0
    duration: float = 0.0
    progress: float = 0.0


@dataclass
class Alarm:
    active: bool = False
    exit: str = "A"
    blocked_exits: list[str] = field(default_factory=list)


@dataclass
class Flight:
    state: str = "idle"
    estimator_converged: bool = False
    armed: bool = False


@dataclass
class HwCheck:
    """Desk hardware check: live attitude from the real drone while it is idle."""

    active: bool = False
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    ts: float = 0.0


@dataclass
class AppState:
    """Everything the operator screen shows.  Plain data; see the module docstring."""

    source: str = "none"
    mode: str = "guide"
    link: Link = field(default_factory=Link)
    tracking: Tracking = field(default_factory=Tracking)
    drone: Pose = field(default_factory=Pose)
    wand: Wand = field(default_factory=Wand)
    recording: Recording = field(default_factory=Recording)
    paths: list[PathInfo] = field(default_factory=list)
    active_path: str | None = None
    replay: Replay = field(default_factory=Replay)
    alarm: Alarm = field(default_factory=Alarm)
    flight: Flight = field(default_factory=Flight)
    hwcheck: HwCheck = field(default_factory=HwCheck)
    log: list[str] = field(default_factory=list)
    ts: float = 0.0

    def to_json(self) -> dict[str, Any]:
        """Plain dict of plain types (JSON-ready; no numpy, no dataclasses)."""
        return _plain(asdict(self))


def _plain(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays and tuples to JSON types; NaN/inf floats become 0.0
    (JSON has no NaN, and one NaN from a tracker would otherwise break every state request)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy array or scalar
        return _plain(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _plain(obj.item())
        except (TypeError, ValueError):
            return obj
    return obj


def _merge_into(target: Any, value: Any) -> Any:
    """Return the new value for a field: dicts merge into dataclasses, otherwise replace."""
    if is_dataclass(target) and not isinstance(target, type):
        if isinstance(value, dict):
            names = {f.name for f in fields(target)}
            for k, v in value.items():
                if k not in names:
                    raise KeyError(f"{type(target).__name__} has no field {k!r}")
                setattr(target, k, _merge_into(getattr(target, k), v))
            return target
        if is_dataclass(value) and type(value) is type(target):
            return value
        raise TypeError(f"cannot assign {type(value).__name__} to {type(target).__name__}")
    return value


# ---------------------------------------------------------------------- the bus


class Subscription:
    """Change notifier for one consumer of the bus (a websocket, a logger, ...).

    Works from plain threads (``wait``) and from asyncio (``wait_async``).  Only
    the *fact* of a change is signalled; the consumer then calls ``take()`` to
    get the latest snapshot, so a slow consumer coalesces bursts instead of
    queueing them.
    """

    def __init__(self, bus: "StateBus", loop: asyncio.AbstractEventLoop | None) -> None:
        self._bus = bus
        self._loop = loop
        self._tevent = threading.Event()
        self._aevent: asyncio.Event | None = asyncio.Event() if loop is not None else None
        self.seen: int = -1
        self.closed = False

    # called by the bus (under its lock) from ANY thread
    def _notify(self) -> None:
        self._tevent.set()
        if self._loop is not None and self._aevent is not None:
            try:
                self._loop.call_soon_threadsafe(self._aevent.set)
            except RuntimeError:  # loop closed
                pass

    def changed(self) -> bool:
        return self._bus.version != self.seen

    def take(self) -> dict[str, Any]:
        """Mark the current version as seen and return its JSON snapshot."""
        self.seen, snap = self._bus.snapshot_versioned()
        return snap

    def wait(self, timeout: float | None = None) -> bool:
        """Block (thread) until the state changes. Returns True if it did."""
        if self.changed():
            return True
        got = self._tevent.wait(timeout)
        self._tevent.clear()
        return got or self.changed()

    async def wait_async(self, timeout: float | None = None) -> bool:
        """Await a change from asyncio code. Returns True if the state changed."""
        if self.changed():
            return True
        if self._aevent is None:
            # subscribed outside a loop: fall back to a thread wait
            return await asyncio.get_running_loop().run_in_executor(None, self.wait, timeout)
        try:
            await asyncio.wait_for(self._aevent.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        self._aevent.clear()
        return self.changed()

    def close(self) -> None:
        self.closed = True
        self._bus._unsubscribe(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class StateBus:
    """Thread-safe holder of one ``AppState`` with a version counter.

    ``update``/``patch``/``log`` may be called from any thread at 30 Hz; every
    call bumps ``version`` and wakes subscribers.  ``snapshot()`` returns a deep
    copy of the state, ``snapshot_dict()`` the JSON-ready dict.
    """

    def __init__(self, state: AppState | None = None) -> None:
        self._state = state if state is not None else AppState()
        self._lock = threading.RLock()
        self._version = 0
        self._subs: set[Subscription] = set()
        self._state.ts = time.time()

    # -- producers ----------------------------------------------------------

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def update(self, **patch: Any) -> int:
        """Merge top-level fields.  Nested dataclass fields accept partial dicts::

            bus.update(drone={"x": 1.0}, mode="spell", flight={"state": "flying"})

        Returns the new version.  Unknown field names raise ``KeyError``.
        """
        names = {f.name for f in fields(AppState)}
        with self._lock:
            for k, v in patch.items():
                if k not in names:
                    raise KeyError(f"AppState has no field {k!r}")
                setattr(self._state, k, _merge_into(getattr(self._state, k), v))
            return self._bump()

    def patch(self, path: str, value: Any) -> int:
        """Set one nested field by dotted path, e.g. ``patch("drone.x", 0.5)``."""
        parts = path.split(".")
        with self._lock:
            obj: Any = self._state
            for p in parts[:-1]:
                if not (is_dataclass(obj) and hasattr(obj, p)):
                    raise KeyError(f"no such state path {path!r}")
                obj = getattr(obj, p)
            leaf = parts[-1]
            if not (is_dataclass(obj) and leaf in {f.name for f in fields(obj)}):
                raise KeyError(f"no such state path {path!r}")
            setattr(obj, leaf, _merge_into(getattr(obj, leaf), value))
            return self._bump()

    def log(self, text: str) -> int:
        """Append one timestamped status line (keeps the last ``LOG_MAX``)."""
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        with self._lock:
            self._state.log.append(line)
            del self._state.log[:-LOG_MAX]
            return self._bump()

    def append_live_point(self, x: float, y: float, z: float) -> int:
        """Add a recording sample to ``recording.live_points`` (keeps the last 500)."""
        with self._lock:
            rec = self._state.recording
            rec.live_points.append([float(x), float(y), float(z)])
            del rec.live_points[:-LIVE_POINTS_MAX]
            rec.n_samples += 1
            return self._bump()

    def modify(self, fn: Any) -> int:
        """Run ``fn(state)`` under the lock for compound edits; returns the new version."""
        with self._lock:
            fn(self._state)
            return self._bump()

    def _bump(self) -> int:
        # caller holds the lock
        self._version += 1
        self._state.ts = time.time()
        for sub in list(self._subs):
            sub._notify()
        return self._version

    # -- consumers ----------------------------------------------------------

    def snapshot(self) -> AppState:
        with self._lock:
            return copy.deepcopy(self._state)

    def snapshot_dict(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_json()

    def snapshot_versioned(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            return self._version, self._state.to_json()

    def subscribe(self) -> Subscription:
        """Register a change listener.  Call from the thread/loop that will wait on it."""
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        sub = Subscription(self, loop)
        with self._lock:
            self._subs.add(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.discard(sub)


# ------------------------------------------------------------------- commands


@dataclass(frozen=True)
class Command:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    source: str = "ui"

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.args), "ts": self.ts, "source": self.source}


class CommandQueue:
    """Thread-safe FIFO of ``Command``s from the UI / voice to the flight loop.

    ``push`` validates the name against ``COMMAND_SPECS`` (``ValueError`` for an
    unknown command) and normalises ``args`` to a dict.  ``pop`` blocks with an
    optional timeout; ``drain`` returns everything queued right now.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._q: queue.Queue[Command] = queue.Queue(maxsize)
        self._lock = threading.Lock()
        self._listeners: list[Any] = []
        self.pushed: int = 0

    def push(self, name: str, args: dict[str, Any] | None = None, source: str = "ui") -> Command:
        if name not in COMMAND_SPECS:
            raise ValueError(f"unknown command {name!r}; valid: {', '.join(COMMAND_NAMES)}")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("command args must be an object")
        required, _ = COMMAND_SPECS[name]
        missing = [a for a in required if a not in args]
        if missing:
            raise ValueError(f"command {name!r} needs args {missing}")
        cmd = Command(name=name, args=dict(args), ts=time.time(), source=source)
        self._q.put(cmd)
        with self._lock:
            self.pushed += 1
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(cmd)
            except Exception:  # noqa: BLE001 - listeners must never break the queue
                pass
        return cmd

    def pop(self, timeout: float | None = None) -> Command | None:
        """Next command, or None after ``timeout`` seconds (None = block forever)."""
        try:
            return self._q.get(timeout=timeout) if timeout is not None else self._q.get()
        except queue.Empty:
            return None

    def pop_nowait(self) -> Command | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[Command]:
        out: list[Command] = []
        while True:
            cmd = self.pop_nowait()
            if cmd is None:
                return out
            out.append(cmd)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.drain())

    def __len__(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def on_push(self, fn: Any) -> None:
        """Register ``fn(cmd)`` to be called synchronously on every push (logging, metrics)."""
        with self._lock:
            self._listeners.append(fn)


#: Process-wide defaults.  Tracker/flight/voice import these; the server serves them.
BUS = StateBus()
COMMANDS = CommandQueue()
