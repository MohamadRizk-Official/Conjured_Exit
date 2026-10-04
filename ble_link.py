"""
ble_link.py -- cflib CRTP link driver for the Crazyflie 2.x over Bluetooth LE (bleak 3).

Usage (the URI is the ONLY thing that differs from the USB link):

    import ble_link, cflib.crtp
    ble_link.register()            # idempotent, does NOT touch Bluetooth
    cflib.crtp.init_drivers()
    ... SyncCrazyflie("ble://DB:04:E8:22:6F:CC") / Crazyflie().open_link("ble://") ...

URI forms
    ble://                      first device whose advertised name starts with "Crazyflie"
    ble://DB:04:E8:22:6F:CC     by BLE address
    ble://Crazyflie-226FCC      by exact advertised name
    ble://DB:04:E8:22:6F:CC?pump_hz=30&write_with_response=0&stream_ports=3,6,7,8
                                any option below may be given per connection as a query string

Send policy (measured on our drone, nRF firmware 2024.10, 2026-10-03: connection interval
~30 ms; the nRF delivers about ONE downlink notification per connection event and silently
drops the rest; bursts of K back-to-back writes with a downlink pending lose replies for
K >= 2; all-with-response serialized writes -> zero loss; uplink itself is never lost):
    * One asyncio.Lock serializes every GATT write (WWR returns in ~1.5 ms, with-response ~30 ms).
    * Packets on the streaming ports (STREAM_PORTS = {3 commander, 6 localization/extpos,
      7 generic setpoint, 8 high-level commander}) are fire-and-forget: write-without-response
      on the CRTP characteristic 0202 (or two CRTPUP 0203 fragments when longer than 20 bytes).
    * Every other packet (param, log, mem, platform, console, link ...) is written WITH
      response on 0202, so at most one such write is in flight per connection event. That is
      what makes TOC downloads and param acks lossless. A reliable packet longer than 20
      bytes has to go as two write-without-response CRTPUP fragments; the sender then waits
      reliable_gap_s (one connection interval) before the next write.
    * Downlink is ACK-driven (the STM32 releases ONE queued downlink packet per uplink packet,
      radiolink.c), so when nothing has been sent for 1/pump_hz the sender writes a 0xFF null
      packet WITH response. The lock guarantees the pump never overlaps a with-response write.

Options (class defaults shown; change them from config.py with
``ble_link.configure(pump_hz=..., stream_ports=..., ...)``, per instance with
``BleDriver(pump_hz=...)``, or per connection in the URI query string):
    pump_hz             (30)      null-packet rate while idle; 0 disables the pump (do not).
    stream_ports        (3,6,7,8) CRTP ports written without response (fire and forget).
    write_with_response (False)   True = streaming ports are written with response as well
                                  (slow but lossless; debugging aid).
    reliable_gap_s      (0.03)    pause after a reliable packet sent as CRTPUP fragments.
    max_inflight_writes (0)       0 = unlimited. Otherwise at most this many GATT writes are
                                  issued per inflight_window_s (extra pacing knob, unused by default).
    inflight_window_s   (0.06)    window for max_inflight_writes.
    scan_timeout        (10)      seconds to look for the device before connecting.
    connect_timeout     (20)      bleak connect timeout in seconds.

Design
    * One asyncio event loop in a daemon thread, started lazily and shared by every
      driver instance (bleak is asyncio-only; cflib is thread/callback based).
    * connect()/close()/scan_interface() run coroutines on that loop via
      asyncio.run_coroutine_threadsafe and wait with a timeout.
    * send_packet() hands header+payload bytes to a single sender task through an
      asyncio.Queue (loop.call_soon_threadsafe), so order is preserved. It never blocks and
      never raises (cflib holds a lock around it with no try/finally).
    * receive_packet() reads a thread-safe queue.Queue of CRTPPacket objects filled by the
      CRTPDOWN notification callback through the tolerant CrtpDownReassembler. Semantics
      match cflib's usbdriver: time=0 non-blocking, time=None or <0 block, time>0 timeout.
    * On an unexpected BLE disconnect the link is marked down and link_error_callback is
      called exactly once -- from a helper thread, because cflib's handler calls
      link.close() synchronously and close() must not run on the asyncio loop thread.
    * close() flushes packets already queued (cflib's zero setpoint on close_link) for up to
      one second, then stops notifications and disconnects.
    * needs_resending = False: the BLE link layer is reliable and reliable-class packets are
      written with response, so cflib's radio-style resend timers are not used.

Do NOT pair the drone in Windows settings; connect directly. Any Crazyradio traffic
disables BLE until the drone is power-cycled.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import logging
import queue
import re
import threading

from bleak import BleakClient, BleakScanner

from cflib.crtp.crtpdriver import CRTPDriver
from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.exceptions import WrongUriType

from ble_framing import (
    CRTP_MAX_PACKET,
    CRTP_UUID,
    CRTPDOWN_UUID,
    CRTPUP_UUID,
    NAME_PREFIX,
    NULL_PACKET,
    CrtpDownReassembler,
    UnroutablePacket,
    crtp_port,
    uplink_writes,
)

__all__ = ["BleDriver", "register", "configure", "get_event_loop", "OPTIONS", "STREAM_PORTS"]

logger = logging.getLogger(__name__)

URI_RE = re.compile(r"^ble://([^?]*)(?:\?(.*))?$", re.IGNORECASE)
BLE_ADDRESS_RE = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")

# CRTP ports whose packets are fire-and-forget (write-without-response).
STREAM_PORTS = frozenset({3, 6, 7, 8})   # commander, localization, generic setpoint, high-level commander

DEFAULT_CLOSE_TIMEOUT = 10.0
CLOSE_FLUSH_TIMEOUT = 1.0         # how long close() lets the sender flush queued packets
TX_QUEUE_MAX = 64                 # outgoing packets buffered before the oldest is dropped
MAX_CONSECUTIVE_TX_ERRORS = 5     # GATT write failures in a row before the link is declared dead


def _parse_bool(text) -> bool:
    if isinstance(text, bool):
        return text
    t = str(text).strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def _parse_ports(value) -> frozenset:
    """'3,6,7,8' or any iterable of ints -> frozenset of CRTP ports."""
    if isinstance(value, str):
        items = [p for p in re.split(r"[,\s;]+", value.strip()) if p]
    else:
        items = list(value)
    ports = frozenset(int(p) for p in items)
    if any(not 0 <= p <= 15 for p in ports):
        raise ValueError("CRTP ports must be in 0..15")
    return ports


# name -> (parser, validator) ; validator returns an error string or None
OPTIONS = {
    "pump_hz": (float, lambda v: None if v >= 0 else "must be >= 0"),
    "stream_ports": (_parse_ports, lambda v: None),
    "write_with_response": (_parse_bool, lambda v: None),
    "reliable_gap_s": (float, lambda v: None if v >= 0 else "must be >= 0"),
    "max_inflight_writes": (int, lambda v: None if v >= 0 else "must be >= 0"),
    "inflight_window_s": (float, lambda v: None if v > 0 else "must be > 0"),
    "scan_timeout": (float, lambda v: None if v > 0 else "must be > 0"),
    "connect_timeout": (float, lambda v: None if v > 0 else "must be > 0"),
    # nRF 2024.10 corrupts fragmented (> 20-byte) packets in both directions: refuse them by
    # default instead of silently delivering damaged data. 0 = no limit (fragment as per the doc).
    "max_uplink_packet": (int, lambda v: None if v >= 0 else "must be >= 0"),
}


def _coerce_options(options: dict) -> dict:
    """Validate/convert an options mapping (strings allowed, e.g. from a URI query)."""
    out = {}
    for key, raw in options.items():
        if key not in OPTIONS:
            raise ValueError(f"unknown BLE link option {key!r} (known: {', '.join(OPTIONS)})")
        parser, validator = OPTIONS[key]
        try:
            value = parser(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad value for BLE link option {key}={raw!r}: {e}") from None
        err = validator(value)
        if err:
            raise ValueError(f"bad value for BLE link option {key}={raw!r}: {err}")
        out[key] = value
    return out


# --------------------------------------------------------------------------------------
# Shared asyncio loop in a daemon thread
# --------------------------------------------------------------------------------------
_loop_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background asyncio loop, starting its daemon thread on first use."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop_thread is not None and _loop_thread.is_alive():
            return _loop
        loop = asyncio.new_event_loop()
        started = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(started.set)
            try:
                loop.run_forever()
            finally:
                loop.close()

        thread = threading.Thread(target=run, name="ble-link-asyncio", daemon=True)
        thread.start()
        started.wait(5.0)
        _loop, _loop_thread = loop, thread
        return loop


def _in_loop_thread() -> bool:
    return _loop_thread is not None and threading.current_thread() is _loop_thread


def _run(coro, timeout: float):
    """Run ``coro`` on the shared loop from a foreign thread and wait for its result."""
    if _in_loop_thread():
        coro.close()
        raise RuntimeError("blocking BLE call made from the asyncio loop thread")
    fut = asyncio.run_coroutine_threadsafe(coro, get_event_loop())
    try:
        return fut.result(timeout)
    except concurrent.futures.TimeoutError:
        fut.cancel()
        raise TimeoutError(f"BLE operation timed out after {timeout:g} s") from None


def _adv_name(device, adv) -> str:
    """Advertised name of a scan result (AdvertisementData first, BLEDevice as fallback)."""
    return getattr(adv, "local_name", None) or getattr(device, "name", None) or ""


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
class BleDriver(CRTPDriver):
    """cflib CRTP link driver for the Crazyflie BLE service (``ble://`` URIs)."""

    # Defaults; see the module docstring. Override with configure(), BleDriver(**opts) or the URI.
    pump_hz = 30.0
    stream_ports = STREAM_PORTS
    write_with_response = False
    reliable_gap_s = 0.03
    max_inflight_writes = 0
    inflight_window_s = 0.06
    scan_timeout = 10.0
    connect_timeout = 20.0
    max_uplink_packet = 20          # refuse longer packets (nRF 2024.10 corrupts fragments); 0 = fragment

    def __init__(self, **options) -> None:
        CRTPDriver.__init__(self)
        self.needs_resending = False
        for key, value in _coerce_options(options).items():
            setattr(self, key, value)
        self.uri = ""
        self.link_error_callback = None
        self.link_quality_callback = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._device = None
        self._device_name = ""
        self._device_address = ""
        self._has_crtp = False
        self._has_crtpup = False

        self._state_lock = threading.Lock()
        self._connected = False        # link fully up (after start_notify)
        self._closing = False          # close() in progress or done: ignore disconnect callbacks
        self._closed = False
        self._error_reported = False   # link_error_callback fired (at most once); sending stops

        self._in_queue: queue.Queue = queue.Queue()      # CRTPPacket or None sentinel
        self._tx_queue: asyncio.Queue | None = None      # bytes (header + payload) or None to stop
        self._write_lock: asyncio.Lock | None = None
        self._sender_task: asyncio.Task | None = None
        self._reasm = CrtpDownReassembler()
        self._pid = 0
        self._recent_writes: collections.deque = collections.deque()

        self._tx_packets = 0           # real CRTP packets written
        self._tx_null = 0              # 0xFF pump packets written
        self._tx_writes_wwr = 0        # GATT writes without response
        self._tx_writes_wr = 0         # GATT writes with response
        self._tx_errors = 0
        self._tx_dropped = 0
        self._tx_consecutive_errors = 0
        self._rx_packets = 0
        self._rx_fragments = 0
        self._tx_by_port: dict[str, int] = {}   # "p<port>c<ch>" -> count of real packets written
        self._rx_by_port: dict[str, int] = {}   # "p<port>c<ch>" -> count of packets received
        self.fast_interval = True               # Windows 11: ask for the 15 ms connection interval
        self._conn_param_request = None         # WinRT request object; must live as long as the connection
        self._conn_interval_ms: float | None = None
        self._interval_task = None

    # ------------------------------------------------------------------ URI handling
    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str, dict]:
        """Return (kind, value, options) for a ble:// URI.

        kind is 'any' (value ''), 'address' (upper-cased 'DB:04:...') or 'name'. options are
        the validated query-string options. Raises WrongUriType for anything that is not a
        ``ble://`` URI and ValueError for a bad option.
        """
        m = URI_RE.match(uri or "")
        if not m:
            raise WrongUriType("Not a BLE URI")
        target = (m.group(1) or "").strip()
        raw_opts = {}
        for part in filter(None, (m.group(2) or "").split("&")):
            key, _, value = part.partition("=")
            raw_opts[key.strip()] = value.strip()
        options = _coerce_options(raw_opts)
        if not target:
            return "any", "", options
        if BLE_ADDRESS_RE.match(target):
            return "address", target.upper(), options
        return "name", target, options

    # ------------------------------------------------------------------ CRTPDriver API
    def connect(self, uri, link_quality_callback, link_error_callback):
        """Scan for the device, connect, verify the CRTP characteristics, subscribe, start the pump.

        Raises WrongUriType for non-BLE URIs (cflib then tries the next driver) and a
        plain Exception when the device cannot be found or connected.
        """
        kind, value, options = self.parse_uri(uri)   # WrongUriType / ValueError propagate
        if self._client is not None:
            raise Exception("BleDriver: link already open")
        for key, val in options.items():
            setattr(self, key, val)

        self.uri = uri
        self.link_quality_callback = link_quality_callback
        self.link_error_callback = link_error_callback
        with self._state_lock:
            self._closing = False
            self._closed = False
            self._error_reported = False
        self._loop = get_event_loop()

        timeout = float(self.scan_timeout) + float(self.connect_timeout) + 15.0
        try:
            _run(self._async_connect(kind, value), timeout)
        except Exception as e:
            self._client = None
            self._connected = False
            raise Exception(f"BleDriver: could not connect to {uri}: {e}") from e
        logger.info("BleDriver: connected to %s (%s); pump %g Hz, stream ports %s, write_with_response=%s",
                    self._device_name, self._device_address, self.pump_hz, sorted(self.stream_ports),
                    self.write_with_response)

    def send_packet(self, pk):
        """Queue a CRTP packet for the sender task. Never blocks, never raises."""
        loop, tx_queue = self._loop, self._tx_queue
        if loop is None or tx_queue is None or self._error_reported:
            self._tx_dropped += 1
            return
        try:
            data = bytes([pk.header & 0xFF]) + bytes(pk.data)
        except Exception as e:  # noqa: BLE001 - malformed packet object
            logger.error("BleDriver: cannot serialise packet %r: %r", pk, e)
            self._tx_dropped += 1
            return
        if len(data) > CRTP_MAX_PACKET:
            logger.error("BleDriver: dropping oversized CRTP packet (%d bytes)", len(data))
            self._tx_dropped += 1
            return
        try:
            loop.call_soon_threadsafe(self._enqueue_tx, data)
        except RuntimeError:  # loop closed
            self._tx_dropped += 1

    def receive_packet(self, time=0):
        """Return the next CRTPPacket, or None.

        time == 0: non-blocking. time is None or < 0: block until a packet arrives or the
        link goes down. time > 0: wait at most ``time`` seconds. Never raises after a
        disconnect or close.
        """
        if time == 0:
            try:
                return self._in_queue.get(False)
            except queue.Empty:
                return None
        if time is None or time < 0:
            # Do not block forever on a dead link; a None sentinel wakes blocked callers
            # when the link goes down while we wait.
            if not self._connected and self._in_queue.empty():
                return None
            try:
                return self._in_queue.get(True)
            except queue.Empty:
                return None
        try:
            return self._in_queue.get(True, time)
        except queue.Empty:
            return None

    def get_status(self):
        if self._connected:
            state = f"connected to {self._device_name} ({self._device_address})"
        elif self._error_reported:
            state = f"link lost ({self._device_name} {self._device_address})"
        elif self._closed:
            state = "closed"
        else:
            state = "not connected"
        return (f"ble: {state}; pump {self.pump_hz:g} Hz, nulls {self._tx_null}; "
                f"tx {self._tx_packets} pkt ({self._tx_writes_wwr} wwr / {self._tx_writes_wr} wr writes) / "
                f"{self._tx_errors} err / {self._tx_dropped} dropped; "
                f"rx {self._rx_packets} pkt / {self._rx_fragments} frag / {self._reasm.orphans} orphan / "
                f"{self._reasm.incomplete} incomplete (lenfield={self._reasm.convention or '?'}); "
                f"stream_ports={','.join(str(p) for p in sorted(self.stream_ports))}, "
                f"write_with_response={self.write_with_response}, max_inflight_writes={self.max_inflight_writes}")

    def stats(self) -> dict:
        """Counters and settings for diagnostics (used by the hardware test script)."""
        return {
            "connected": self._connected,
            "device_name": self._device_name,
            "device_address": self._device_address,
            "pump_hz": float(self.pump_hz),
            "stream_ports": sorted(self.stream_ports),
            "write_with_response": bool(self.write_with_response),
            "reliable_gap_s": float(self.reliable_gap_s),
            "max_inflight_writes": int(self.max_inflight_writes),
            "inflight_window_s": float(self.inflight_window_s),
            "tx_packets": self._tx_packets,
            "tx_null": self._tx_null,
            "tx_writes_wwr": self._tx_writes_wwr,
            "tx_writes_wr": self._tx_writes_wr,
            "tx_errors": self._tx_errors,
            "tx_dropped": self._tx_dropped,
            "rx_packets": self._rx_packets,
            "rx_fragments": self._rx_fragments,
            "rx_orphans": self._reasm.orphans,
            "rx_incomplete": self._reasm.incomplete,
            "rx_convention": self._reasm.convention,
            "conn_interval_ms": self._conn_interval_ms,
            "tx_by_port": dict(sorted(self._tx_by_port.items())),
            "rx_by_port": dict(sorted(self._rx_by_port.items())),
            "has_crtp_char": self._has_crtp,
            "has_crtpup_char": self._has_crtpup,
            "link_error_reported": self._error_reported,
        }

    def get_name(self):
        return "ble"

    def scan_interface(self, address=None):
        """BLE scan (scan_timeout seconds); returns [[uri, name], ...] for Crazyflies found.

        ``address`` is ignored unless it is a BLE address string, in which case only that
        device is reported.
        """
        if self._client is not None:
            raise Exception("Cannot scan for links while the link is open!")
        only = address.upper() if isinstance(address, str) and BLE_ADDRESS_RE.match(address) else None
        try:
            return _run(self._async_scan(only), float(self.scan_timeout) + 10.0)
        except Exception as e:  # noqa: BLE001 - no adapter, Bluetooth off, ...
            logger.warning("BleDriver: BLE scan failed: %r", e)
            return []

    def enum(self):
        return [uri for uri, _ in self.scan_interface()]

    def get_help(self):
        return ("ble://                   first device named Crazyflie-*\n"
                "ble://DB:04:E8:22:6F:CC  by BLE address\n"
                "ble://Crazyflie-226FCC   by advertised name\n"
                "ble://<target>?pump_hz=30&stream_ports=3,6,7,8&write_with_response=0  options")

    def close(self):
        """Flush queued packets, stop the pump/sender, stop notifications, disconnect. Idempotent."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._closing = True
            self._connected = False
        loop, client = self._loop, self._client
        if loop is not None and client is not None:
            if _in_loop_thread():
                loop.create_task(self._async_close())
            else:
                try:
                    _run(self._async_close(), DEFAULT_CLOSE_TIMEOUT)
                except Exception as e:  # noqa: BLE001
                    logger.warning("BleDriver: error while closing: %r", e)
        self._client = None
        self._tx_queue = None
        self._drain_in_queue()
        self._in_queue.put(None)  # wake any blocked receive_packet()
        logger.info("BleDriver: closed %s", self.uri)

    # ------------------------------------------------------------------ asyncio side
    async def _find_device(self, kind: str, value: str):
        timeout = float(self.scan_timeout)
        if kind == "address":
            return await BleakScanner.find_device_by_address(value, timeout=timeout)
        if kind == "name":
            return await BleakScanner.find_device_by_filter(
                lambda d, adv: _adv_name(d, adv) == value, timeout=timeout)
        return await BleakScanner.find_device_by_filter(
            lambda d, adv: _adv_name(d, adv).startswith(NAME_PREFIX), timeout=timeout)

    # ------------------------------------------------------------------ connection interval (Windows 11)
    def _winrt_device(self, client):
        return getattr(getattr(client, "_backend", None), "_requester", None)

    def _read_interval_ms(self, dev) -> float | None:
        try:
            return float(dev.get_connection_parameters().connection_interval) * 1.25
        except Exception:  # noqa: BLE001
            return None

    def _request_fast_interval(self, dev) -> bool:
        """Ask Windows for BluetoothLEPreferredConnectionParameters.throughput_optimized (15 ms).

        Measured on our drone: Windows connects at 15 ms, the nRF's own parameter update (~2 s
        after connect) moves it to 45 ms (write-with-response 80 ms); this request brings it back
        to 15 ms (~33 ms) and it stays. The request object must be kept alive.
        """
        try:
            from winrt.windows.devices.bluetooth import BluetoothLEPreferredConnectionParameters as Pref
            req = dev.request_preferred_connection_parameters(Pref.throughput_optimized)
            self._conn_param_request = req
            logger.info("BleDriver: requested throughput-optimized connection parameters (status %s)", req.status)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("BleDriver: could not request a fast connection interval: %r", e)
            return False

    async def _settle_interval(self, client, wait_s: float = 4.0) -> None:
        """Wait for the nRF's own connection-parameter update (15 -> 45 ms, ~2 s after connect) and
        restore the fast interval BEFORE cflib starts its TOC/param download.

        Measured 2026-10-04: when the 45 -> 15 ms switch landed during the parameter TOC request, the
        reply was lost, cflib never retries it, and the connect stalled until the 120 s timeout (3 of
        5 attempts). Settling the interval first costs ~2.5 s and removes the race.
        """
        import sys
        if sys.platform != "win32":
            return
        dev = self._winrt_device(client)
        if dev is None:
            return
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        while loop.time() - t0 < wait_s:
            interval = self._read_interval_ms(dev)
            self._conn_interval_ms = interval
            if interval is not None and interval > 20.0:
                logger.info("BleDriver: connection interval is %.2f ms before any CRTP traffic; requesting 15 ms", interval)
                if self._request_fast_interval(dev):
                    await asyncio.sleep(0.6)
                    self._conn_interval_ms = self._read_interval_ms(dev)
                    logger.info("BleDriver: connection interval now %s ms (settled before cflib)", self._conn_interval_ms)
                return
            await asyncio.sleep(0.25)
        logger.info("BleDriver: connection interval %s ms unchanged for %.0f s; continuing", self._conn_interval_ms, wait_s)

    async def _interval_guard(self, client) -> None:
        """Keep the connection interval short: re-request whenever Windows reports > 20 ms.

        First check only after cflib's connection setup (TOC fetches) is normally done: a switch during
        a reliable request can lose its reply (see _settle_interval)."""
        import sys
        if sys.platform != "win32":
            return
        dev = self._winrt_device(client)
        if dev is None:
            return
        checks = [8.0, 12.0, 16.0] + [20.0 + 10.0 * i for i in range(10_000)]
        t0 = asyncio.get_running_loop().time()
        for t in checks:
            await asyncio.sleep(max(0.0, t - (asyncio.get_running_loop().time() - t0)))
            if not self._connected or self._closing:
                return
            interval = self._read_interval_ms(dev)
            self._conn_interval_ms = interval
            if interval is None:
                return
            if interval > 20.0:
                logger.info("BleDriver: connection interval is %.2f ms; requesting 15 ms", interval)
                if not self._request_fast_interval(dev):
                    return
                await asyncio.sleep(0.5)
                self._conn_interval_ms = self._read_interval_ms(dev)
                logger.info("BleDriver: connection interval now %s ms", self._conn_interval_ms)

    async def _async_connect(self, kind: str, value: str) -> None:
        device = await self._find_device(kind, value)
        if device is None:
            raise Exception(
                f"no Crazyflie found for {self.uri!r} within {self.scan_timeout:g} s. Is it powered on, "
                f"unpaired in Windows, not talked to by a Crazyradio since boot, and not already connected?")
        self._device = device
        self._device_address = getattr(device, "address", "") or value
        self._device_name = getattr(device, "name", None) or (value if kind == "name" else "Crazyflie")

        client = BleakClient(device, disconnected_callback=self._on_disconnected, timeout=float(self.connect_timeout))
        try:
            await client.connect()
            services = client.services
            has_crtp = services.get_characteristic(CRTP_UUID) is not None
            has_crtpup = services.get_characteristic(CRTPUP_UUID) is not None
            has_down = services.get_characteristic(CRTPDOWN_UUID) is not None
            if not has_down or not (has_crtp or has_crtpup):
                raise Exception("Crazyflie CRTP characteristics (0202/0203 + 0204) not found: not a Crazyflie, "
                                "or the Crazyflie service is not exposed")
            if not has_crtp:
                logger.warning("BleDriver: CRTP characteristic 0202 missing; all packets go through CRTPUP "
                               "(single-fragment CRTPUP writes are known to be dropped by nRF firmware 2024.10)")
            if not has_crtpup:
                logger.warning("BleDriver: CRTPUP characteristic 0203 missing; packets longer than 20 bytes "
                               "will be dropped")
            self._has_crtp, self._has_crtpup = has_crtp, has_crtpup
            self._tx_queue = asyncio.Queue(maxsize=TX_QUEUE_MAX)
            self._write_lock = asyncio.Lock()
            self._reasm = CrtpDownReassembler()
            self._pid = 0
            self._recent_writes.clear()
            await client.start_notify(CRTPDOWN_UUID, self._on_notify)
            if self.fast_interval:
                await self._settle_interval(client)   # before cflib sends anything (see the docstring)
            self._client = client
            self._connected = True
            self._sender_task = asyncio.get_running_loop().create_task(self._sender(client, self._tx_queue))
            if self.fast_interval:
                self._interval_task = asyncio.get_running_loop().create_task(self._interval_guard(client))
        except BaseException:
            self._client = None
            self._tx_queue = None
            self._connected = False
            try:
                if client.is_connected:
                    await client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            raise

    async def _async_scan(self, only_address: str | None) -> list:
        found = await BleakScanner.discover(timeout=float(self.scan_timeout), return_adv=True)
        result = []
        for device, adv in found.values():
            name = _adv_name(device, adv)
            if not name.startswith(NAME_PREFIX):
                continue
            if only_address and device.address.upper() != only_address:
                continue
            result.append([f"ble://{device.address}", name])
        result.sort(key=lambda e: (e[1], e[0]))
        return result

    async def _async_close(self) -> None:
        task, self._sender_task = self._sender_task, None
        tx_queue = self._tx_queue
        if task is not None and task is not asyncio.current_task():
            if tx_queue is not None and not task.done():
                # let already-queued packets (e.g. cflib's zero setpoint) go out, then stop
                if tx_queue.full():
                    tx_queue.get_nowait()
                tx_queue.put_nowait(None)
                try:
                    await asyncio.wait_for(task, CLOSE_FLUSH_TIMEOUT)   # cancels the task on timeout
                except BaseException:  # noqa: BLE001 - timeout, CancelledError or a late write error
                    pass
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001
                    pass
        self._tx_queue = None
        client = self._client
        if client is not None and client.is_connected:
            try:
                await asyncio.wait_for(client.stop_notify(CRTPDOWN_UUID), 3.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                await client.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.warning("BleDriver: disconnect failed: %r", e)

    async def _sender(self, client, tx_queue: asyncio.Queue) -> None:
        """Single writer: real packets in order; a 0xFF null packet whenever idle for 1/pump_hz.

        ``self.pump_hz`` is re-read for every packet so a client can connect with a fast pump (TOC
        download is one downlink packet per uplink packet) and slow it down for flight, where every
        null is an acknowledged write that delays the real setpoints (measured 2026-10-04)."""
        while True:
            pump_hz = float(self.pump_hz or 0.0)
            pump_interval = 1.0 / pump_hz if pump_hz > 0 else None
            is_null = False
            try:
                if pump_interval is None:
                    data = await tx_queue.get()
                else:
                    data = await asyncio.wait_for(tx_queue.get(), pump_interval)
            except asyncio.TimeoutError:
                data, is_null = NULL_PACKET, True
            if data is None or self._error_reported:
                return
            try:
                await self._write_packet(client, data, is_null)
            except asyncio.CancelledError:
                raise
            except UnroutablePacket as e:
                self._tx_dropped += 1
                logger.error("BleDriver: %s", e)
                continue
            except Exception as e:  # noqa: BLE001
                self._tx_errors += 1
                self._tx_consecutive_errors += 1
                logger.warning("BleDriver: GATT write failed (%d in a row): %r", self._tx_consecutive_errors, e)
                if self._tx_consecutive_errors >= MAX_CONSECUTIVE_TX_ERRORS:
                    self._fail_link(f"BleDriver: {self._tx_consecutive_errors} consecutive BLE write errors "
                                    f"to {self._device_name}; last: {e!r}")
                    return
                continue
            if is_null:
                self._tx_null += 1
            else:
                self._tx_packets += 1
                key = f"p{data[0] >> 4}c{data[0] & 3}"
                self._tx_by_port[key] = self._tx_by_port.get(key, 0) + 1
            self._tx_consecutive_errors = 0

    def _is_streaming(self, data: bytes, is_null: bool) -> bool:
        """Fire-and-forget (write-without-response) or reliable (write-with-response)?"""
        if is_null or self.write_with_response:
            return False
        return crtp_port(data[0]) in self.stream_ports

    async def _write_packet(self, client, data: bytes, is_null: bool = False) -> None:
        """Write one whole CRTP packet (or the null packet) under the single write lock."""
        limit = int(getattr(self, "max_uplink_packet", 0) or 0)
        if not is_null and limit and len(data) > limit:
            raise UnroutablePacket(
                f"{len(data)}-byte packet on port {crtp_port(data[0])} refused: nRF firmware 2024.10 corrupts "
                f"fragmented BLE packets; keep every packet <= {limit} bytes (max_uplink_packet=0 to allow)")
        writes = uplink_writes(data, self._pid, self._has_crtp, self._has_crtpup)   # UnroutablePacket
        fragmented = writes[0][0] == CRTPUP_UUID
        if fragmented:
            self._pid = (self._pid + 1) & 3
        streaming = self._is_streaming(data, is_null)
        async with self._write_lock:
            for uuid, chunk in writes:
                await self._throttle()
                # 0202 carries whole packets: reliable ones WITH response. CRTPUP is write-without-response.
                response = (uuid == CRTP_UUID) and not streaming
                await client.write_gatt_char(uuid, chunk, response=response)
                if response:
                    self._tx_writes_wr += 1
                else:
                    self._tx_writes_wwr += 1
                self._note_write()
            if fragmented and not streaming and self.reliable_gap_s > 0:
                # a reliable packet had to go as WWR fragments: keep the next write out of this connection event
                await asyncio.sleep(float(self.reliable_gap_s))

    async def _throttle(self) -> None:
        """Enforce max_inflight_writes per inflight_window_s (0 = unlimited)."""
        limit = int(self.max_inflight_writes or 0)
        if limit <= 0:
            return
        window = float(self.inflight_window_s)
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            while self._recent_writes and now - self._recent_writes[0] >= window:
                self._recent_writes.popleft()
            if len(self._recent_writes) < limit:
                return
            await asyncio.sleep(max(0.0005, window - (now - self._recent_writes[0])))

    def _note_write(self) -> None:
        if int(self.max_inflight_writes or 0) > 0:
            self._recent_writes.append(asyncio.get_running_loop().time())

    # ------------------------------------------------------------------ loop-thread callbacks
    def _enqueue_tx(self, data: bytes) -> None:
        tx_queue = self._tx_queue
        if tx_queue is None or self._error_reported:
            self._tx_dropped += 1
            return
        if tx_queue.full():
            try:
                tx_queue.get_nowait()   # drop the oldest; stale setpoints are worse than lost ones
                self._tx_dropped += 1
            except asyncio.QueueEmpty:
                pass
        tx_queue.put_nowait(data)

    def _on_notify(self, _characteristic, data) -> None:
        self._rx_fragments += 1
        pkt = self._reasm.feed(bytes(data))
        if pkt is None:
            return
        self._rx_packets += 1
        key = f"p{pkt[0] >> 4}c{pkt[0] & 3}"
        self._rx_by_port[key] = self._rx_by_port.get(key, 0) + 1
        self._in_queue.put(CRTPPacket(pkt[0], bytearray(pkt[1:])))

    def _on_disconnected(self, _client) -> None:
        if self._closing or not self._connected:
            return
        self._fail_link(f"BleDriver: BLE connection to {self._device_name} ({self._device_address}) lost")

    def _fail_link(self, msg: str) -> None:
        """Mark the link down and report once via link_error_callback (on a helper thread)."""
        with self._state_lock:
            if self._error_reported or self._closing:
                return
            self._error_reported = True
            self._connected = False
        logger.warning(msg)
        self._in_queue.put(None)  # wake blocked receive_packet() callers
        cb = self.link_error_callback
        if cb is not None:
            # cflib's handler calls link.close() synchronously; that must not happen on the
            # asyncio thread (close() waits on the loop), so hand it to a helper thread.
            threading.Thread(target=self._call_error_cb, args=(cb, msg),
                             name="ble-link-error-cb", daemon=True).start()

    @staticmethod
    def _call_error_cb(cb, msg: str) -> None:
        try:
            cb(msg)
        except Exception:  # noqa: BLE001
            logger.exception("BleDriver: link_error_callback raised")

    def _drain_in_queue(self) -> None:
        while True:
            try:
                self._in_queue.get_nowait()
            except queue.Empty:
                return


# --------------------------------------------------------------------------------------
# Configuration and cflib registration
# --------------------------------------------------------------------------------------
def configure(**options) -> dict:
    """Set class-wide defaults, e.g. from config.py:
    ``ble_link.configure(pump_hz=30, stream_ports={3, 6, 7, 8}, write_with_response=False)``.

    Returns the current defaults. Unknown names or bad values raise ValueError.
    """
    for key, value in _coerce_options(options).items():
        setattr(BleDriver, key, value)
    return {key: getattr(BleDriver, key) for key in OPTIONS}


def register() -> type:
    """Make cflib's URI dispatch aware of BleDriver. Idempotent; never touches Bluetooth.

    cflib 0.1.34 keeps its driver classes in the module-level list ``cflib.crtp.CLASSES``;
    ``init_drivers()`` extends that list and ``get_link_driver()`` / ``scan_interfaces()``
    iterate over it, so appending once is all that is needed. Works whether it is called
    before or after ``cflib.crtp.init_drivers()``.
    """
    import cflib.crtp
    if BleDriver not in cflib.crtp.CLASSES:
        cflib.crtp.CLASSES.append(BleDriver)
    return BleDriver
