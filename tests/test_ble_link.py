"""
Unit tests for ble_framing.py and ble_link.py (no Bluetooth hardware, no bleak I/O).

Run from the project root with the venv interpreter:
    cf\\Scripts\\python.exe -m unittest discover -s tests -v

bleak's BleakClient / BleakScanner are replaced by fakes via monkeypatching the names in
the ble_link module, so the asyncio-loop thread, queues, pump and callback plumbing run for
real while nothing touches the radio.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ble_link  # noqa: E402
from ble_framing import (  # noqa: E402
    CRTP_UUID,
    CRTPDOWN_UUID,
    CRTPUP_UUID,
    NULL_PACKET,
    CrtpDownReassembler,
    UnroutablePacket,
    crtp_channel,
    crtp_header,
    crtp_port,
    crtpup_fragments,
    uplink_writes,
)

import cflib.crtp  # noqa: E402
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort  # noqa: E402
from cflib.crtp.exceptions import WrongUriType  # noqa: E402

logging.getLogger("ble_link").setLevel(logging.CRITICAL)
logging.getLogger("cflib").setLevel(logging.CRITICAL)


def wait_until(pred, timeout=3.0, step=0.005) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


# ======================================================================================
# Independent model of the nRF51 CRTPUP receive side, official convention (written from
# the protocol rules, not copied from the driver) so the two implementations check each other.
# ======================================================================================
class FakeNrfUplinkReceiver:
    def __init__(self) -> None:
        self.pid = None
        self.length = 0
        self.buf = b""
        self.packets: list[bytes] = []
        self.errors = 0

    def write(self, frag: bytes) -> None:
        assert 1 <= len(frag) <= 20, f"fragment of {len(frag)} bytes does not fit the characteristic"
        ctrl = frag[0]
        pid = (ctrl >> 5) & 0x03
        if ctrl & 0x80:
            self.pid = pid
            self.length = (ctrl & 0x1F) + 1
            self.buf = bytes(frag[1:])
        else:
            if self.pid is None or pid != self.pid:
                self.errors += 1
                return
            self.buf += bytes(frag[1:])
        if len(self.buf) == self.length:
            self.packets.append(self.buf)
            self.pid = None
        elif len(self.buf) > self.length:
            self.errors += 1
            self.pid = None


def decode_uplink(writes, include_nulls=False) -> list[bytes]:
    """CRTP packets the drone would receive from a list of fake GATT writes, in order."""
    nrf = FakeNrfUplinkReceiver()
    pkts = []
    for uuid, data, _response in writes:
        if uuid == CRTP_UUID:
            assert 1 <= len(data) <= 20
            if data == NULL_PACKET and not include_nulls:
                continue
            pkts.append(data)
        elif uuid == CRTPUP_UUID:
            before = len(nrf.packets)
            nrf.write(data)
            if len(nrf.packets) > before:
                pkts.append(nrf.packets[-1])
        else:
            raise AssertionError(f"write to unexpected characteristic {uuid}")
    assert nrf.errors == 0, "fake nRF saw malformed CRTPUP fragments"
    return pkts


def make_packet(port: int, channel: int, payload: bytes) -> bytes:
    return bytes([crtp_header(port, channel)]) + payload


def cf_packet(port: int, channel: int, payload: bytes) -> CRTPPacket:
    pk = CRTPPacket()
    pk.set_header(port, channel)
    pk.data = payload
    return pk


# Fragments captured from our drone (nRF 2024.10) on CRTPDOWN
CAPTURED_LONG_1 = bytes.fromhex("9f 00 43 46 47 42 4c 4b 3a 20 76 31 2c 20 76 65 72 69 66 69")
CAPTURED_LONG_2 = bytes.fromhex("9f 61 74 69 6f 6e 20 5b 4f 4b 5d 0a 00")
CAPTURED_LONG_PACKET = bytes.fromhex("00 43 46 47 42 4c 4b 3a 20 76 31 2c 20 76 65 72 69 66 69"
                                     "61 74 69 6f 6e 20 5b 4f 4b 5d 0a 00")
CAPTURED_SHORT = bytes.fromhex("85 00 4b 5d 2e 0a")
CAPTURED_SHORT_PACKET = bytes.fromhex("00 4b 5d 2e 0a")


class TestFraming(unittest.TestCase):
    def test_header(self):
        self.assertEqual(crtp_header(15, 0), 0xFC)
        self.assertEqual(crtp_header(6, 0), 0x6C)
        self.assertEqual(crtp_header(2, 3), 0x2F)
        self.assertEqual(crtp_header(0, 0), 0x0C)
        for port in range(16):
            for ch in range(4):
                h = crtp_header(port, ch)
                self.assertEqual(crtp_port(h), port)
                self.assertEqual(crtp_channel(h), ch)
                self.assertEqual(h & 0x0C, 0x0C)
        pk = CRTPPacket()
        pk.set_header(CRTPPort.LOCALIZATION, 0)
        self.assertEqual(pk.header, crtp_header(6, 0))
        self.assertEqual(crtp_port(NULL_PACKET[0]), 15)
        self.assertEqual(len(CAPTURED_LONG_PACKET), 31)
        self.assertEqual(crtp_port(CAPTURED_LONG_PACKET[0]), CRTPPort.CONSOLE)

    def test_crtpup_fragments_official_convention(self):
        for length in (1, 13, 19, 20, 21, 24, 25, 31):
            for pid in range(4):
                with self.subTest(length=length, pid=pid):
                    pkt = make_packet(6, 1, bytes((i * 7 + 3) & 0xFF for i in range(length - 1)))
                    frags = crtpup_fragments(pkt, pid)
                    self.assertEqual(len(frags), 1 if length <= 19 else 2)
                    self.assertEqual(frags[0][0], 0x80 | (pid << 5) | (length - 1))
                    self.assertEqual(frags[0][1:], pkt[:19])
                    if len(frags) == 2:
                        self.assertEqual(frags[1][0], pid << 5)   # continuation: PID bits only
                        self.assertEqual(frags[1][1:], pkt[19:])
                        self.assertEqual(len(frags[0]), 20)
                        self.assertEqual(len(frags[1]), 1 + length - 19)
                    nrf = FakeNrfUplinkReceiver()
                    for f in frags:
                        nrf.write(f)
                    self.assertEqual(nrf.packets, [pkt])
                    self.assertEqual(nrf.errors, 0)

    def test_bad_lengths(self):
        for fn in (lambda p: crtpup_fragments(p, 0), lambda p: uplink_writes(p, 0)):
            with self.assertRaises(ValueError):
                fn(b"")
            with self.assertRaises(ValueError):
                fn(bytes(32))

    def test_pid_is_masked(self):
        pkt = make_packet(15, 2, bytes(25))
        self.assertEqual(crtpup_fragments(pkt, 4), crtpup_fragments(pkt, 0))
        self.assertEqual(crtpup_fragments(pkt, 5), crtpup_fragments(pkt, 1))
        self.assertEqual(crtpup_fragments(pkt, 7), crtpup_fragments(pkt, 3))

    def test_uplink_routing(self):
        self.assertEqual(uplink_writes(NULL_PACKET, 0), [(CRTP_UUID, b"\xff")])
        for length in (1, 13, 20):
            pkt = make_packet(6, 0, bytes(length - 1))
            self.assertEqual(uplink_writes(pkt, 3), [(CRTP_UUID, pkt)])         # whole, never fragmented
        for length in (21, 24, 25, 31):
            pkt = make_packet(8, 0, bytes(range(length - 1)))
            writes = uplink_writes(pkt, 2)
            self.assertEqual([u for u, _ in writes], [CRTPUP_UUID, CRTPUP_UUID])
            self.assertEqual([d for _, d in writes], crtpup_fragments(pkt, 2))
            self.assertEqual(decode_uplink([(u, d, False) for u, d in writes]), [pkt])
        # fallbacks
        short = make_packet(6, 0, bytes(12))
        self.assertEqual(uplink_writes(short, 1, has_crtp=False), [(CRTPUP_UUID, crtpup_fragments(short, 1)[0])])
        long_ = make_packet(8, 0, bytes(23))
        with self.assertRaises(UnroutablePacket):
            uplink_writes(long_, 1, has_crtp=True, has_crtpup=False)
        with self.assertRaises(UnroutablePacket):
            uplink_writes(short, 1, has_crtp=False, has_crtpup=False)


class TestDownReassembler(unittest.TestCase):
    def test_captured_firmware_2024_10_frames(self):
        down = CrtpDownReassembler()
        self.assertIsNone(down.feed(CAPTURED_LONG_1))
        self.assertTrue(down.pending)
        pkt = down.feed(CAPTURED_LONG_2)
        self.assertEqual(pkt, CAPTURED_LONG_PACKET)
        self.assertEqual(len(pkt), 31)
        self.assertEqual(pkt[0], 0x00)
        # verbatim capture (30 payload bytes; the bytes are delivered exactly as received)
        self.assertEqual(pkt[1:], b"CFGBLK: v1, verifiation [OK]\n\x00")
        self.assertEqual(down.convention, "len")
        self.assertEqual(down.feed(CAPTURED_SHORT), CAPTURED_SHORT_PACKET)
        self.assertEqual(len(CAPTURED_SHORT_PACKET), 5)
        self.assertEqual((down.packets, down.orphans, down.incomplete), (2, 0, 0))

    def test_captured_short_frame_first(self):
        down = CrtpDownReassembler()
        self.assertEqual(down.feed(CAPTURED_SHORT), CAPTURED_SHORT_PACKET)
        self.assertIsNone(down.convention)
        self.assertIsNone(down.feed(CAPTURED_LONG_1))
        self.assertEqual(down.feed(CAPTURED_LONG_2), CAPTURED_LONG_PACKET)
        self.assertEqual(down.convention, "len")

    def test_firmware_2024_10_20_byte_packet(self):
        # lenfield = 20 (full length), 19 data bytes, then the start byte repeated with the last byte
        pkt = make_packet(2, 1, bytes(range(19)))
        down = CrtpDownReassembler()
        self.assertIsNone(down.feed(bytes([0x80 | 20]) + pkt[:19]))
        self.assertEqual(down.feed(bytes([0x80 | 20]) + pkt[19:]), pkt)
        self.assertEqual(down.convention, "len")
        # and a 19-byte packet (lenfield 19, 19 data bytes) is complete at once
        pkt19 = make_packet(2, 1, bytes(18))
        self.assertEqual(down.feed(bytes([0x80 | 19]) + pkt19), pkt19)
        self.assertFalse(down.pending)

    def test_doc_style_frames(self):
        down = CrtpDownReassembler()
        pkt = bytes([0x00]) + bytes(range(30))        # 31 bytes -> lenfield 30 under len-1
        first = bytes([0x9E]) + pkt[:19]
        cont = bytes([0x00]) + pkt[19:]
        self.assertEqual(len(cont), 13)
        self.assertIsNone(down.feed(first))
        self.assertEqual(down.feed(cont), pkt)
        self.assertEqual(len(pkt), 31)
        self.assertEqual(down.convention, "len-1")
        # single fragment: lenfield 4 -> 5 bytes
        self.assertEqual(down.feed(bytes([0x84]) + CAPTURED_SHORT_PACKET), CAPTURED_SHORT_PACKET)
        # now a 20-byte packet (lenfield 19) correctly waits for its 1-byte continuation
        pkt20 = make_packet(5, 2, bytes(range(19)))
        self.assertIsNone(down.feed(bytes([0x80 | 19]) + pkt20[:19]))
        self.assertEqual(down.feed(bytes([0x00]) + pkt20[19:]), pkt20)
        # PIDs are honoured on doc firmware
        for pid in (1, 2, 3, 0):
            f1, f2 = crtpup_fragments(pkt, pid)
            self.assertIsNone(down.feed(f1))
            self.assertEqual(down.feed(f2), pkt)
        self.assertEqual((down.orphans, down.incomplete), (0, 0))

    def test_doc_style_20_byte_packet_before_convention_is_known(self):
        down = CrtpDownReassembler()
        pkt20 = make_packet(5, 2, bytes(range(19)))
        # ambiguous: lenfield 19 with 19 data bytes is complete under 'len' -> released early
        self.assertEqual(down.feed(bytes([0x80 | 19]) + pkt20[:19]), pkt20[:19])
        # the tail arrives as an orphan and teaches us the drone speaks len-1
        self.assertIsNone(down.feed(bytes([0x00]) + pkt20[19:]))
        self.assertEqual(down.orphans, 1)
        self.assertEqual(down.convention, "len-1")
        # from now on 20-byte packets are complete
        self.assertIsNone(down.feed(bytes([0x80 | 19]) + pkt20[:19]))
        self.assertEqual(down.feed(bytes([0x00]) + pkt20[19:]), pkt20)

    def test_orphans_and_abandoned_partials(self):
        down = CrtpDownReassembler()
        down.convention = "len-1"
        self.assertIsNone(down.feed(bytes([0x20]) + b"xyz"))       # continuation with nothing pending
        self.assertEqual(down.orphans, 1)
        big = make_packet(5, 0, bytes(range(25)))
        f1, f2 = crtpup_fragments(big, 1)
        self.assertIsNone(down.feed(f1))
        wrong_pid = bytes([2 << 5]) + f2[1:]
        self.assertIsNone(down.feed(wrong_pid))                      # pid mismatch: partial abandoned + orphan
        self.assertEqual((down.orphans, down.incomplete), (2, 1))
        self.assertFalse(down.pending)
        self.assertIsNone(down.feed(f2))                             # the real tail is now an orphan too
        self.assertEqual(down.orphans, 3)
        self.assertIsNone(down.feed(f1))
        self.assertEqual(down.feed(f2), big)                         # recovery
        # a new start mid-packet replaces the partial one
        self.assertIsNone(down.feed(f1))
        small = make_packet(0, 0, b"ok")
        self.assertEqual(down.feed(crtpup_fragments(small, 3)[0]), small)
        self.assertEqual(down.incomplete, 2)
        self.assertIsNone(down.feed(b""))                            # empty notification ignored
        self.assertEqual(down.orphans, 3)

    def test_repeated_start_byte_that_cannot_be_a_continuation(self):
        # firmware 2024.10 repeats the start byte on the 2nd fragment; if the 2nd fragment of
        # one packet is lost and the next packet has the same control byte, the 19-byte
        # fragment must be treated as a new start (19 + 19 > 31), not glued on.
        down = CrtpDownReassembler()
        a = bytes([0x9F]) + b"A" * 19
        b = bytes([0x9F]) + b"B" * 19
        tail = bytes([0x9F]) + b"b" * 12
        self.assertIsNone(down.feed(a))
        self.assertIsNone(down.feed(b))
        self.assertEqual(down.incomplete, 1)
        self.assertEqual(down.feed(tail), b"B" * 19 + b"b" * 12)
        self.assertEqual(down.convention, "len")

    def test_never_truncates(self):
        down = CrtpDownReassembler()
        down.convention = "len"
        data = make_packet(0, 0, b"short")
        self.assertEqual(down.feed(bytes([0x80 | 2]) + data), data)   # lenfield lies, bytes win


# ======================================================================================
# Fakes for bleak
# ======================================================================================
class FakeDevice:
    def __init__(self, name, address):
        self.name = name
        self.address = address

    def __repr__(self):
        return f"FakeDevice({self.address}, {self.name})"


class FakeAdv(SimpleNamespace):
    pass


class FakeServices:
    def __init__(self, uuids):
        self._uuids = {u.lower() for u in uuids}

    def get_characteristic(self, specifier):
        u = str(specifier).lower()
        if u in self._uuids:
            return SimpleNamespace(uuid=u, properties=["write", "write-without-response", "notify"])
        return None


class FakeClient:
    """Mimics the bleak 3 BleakClient surface the driver uses; records the response flag per write."""
    instances: list["FakeClient"] = []
    char_uuids = [CRTP_UUID, CRTPUP_UUID, CRTPDOWN_UUID]
    connect_error: Exception | None = None

    def __init__(self, device, disconnected_callback=None, timeout=None, **kwargs):
        self.device = device
        self.disconnected_callback = disconnected_callback
        self.timeout = timeout
        self.is_connected = False
        self.writes: list[tuple[str, bytes, object]] = []      # (uuid, data, response flag)
        self.write_times: list[float] = []
        self.notify_cb = {}
        self.stop_notify_calls: list[str] = []
        self.disconnect_calls = 0
        self.fail_writes = False
        FakeClient.instances.append(self)

    async def connect(self):
        if FakeClient.connect_error is not None:
            raise FakeClient.connect_error
        self.is_connected = True

    @property
    def services(self):
        return FakeServices(FakeClient.char_uuids)

    async def start_notify(self, uuid, cb, **kwargs):
        assert self.is_connected
        self.notify_cb[str(uuid).lower()] = cb

    async def stop_notify(self, uuid):
        self.stop_notify_calls.append(str(uuid).lower())
        self.notify_cb.pop(str(uuid).lower(), None)

    async def write_gatt_char(self, uuid, data, response=None):
        if not self.is_connected:
            raise Exception("simulated: not connected")
        if self.fail_writes:
            raise Exception("simulated GATT write failure")
        self.writes.append((str(uuid).lower(), bytes(data), response))
        self.write_times.append(time.perf_counter())

    async def disconnect(self):
        self.disconnect_calls += 1
        was = self.is_connected
        self.is_connected = False
        if was and self.disconnected_callback is not None:
            self.disconnected_callback(self)   # bleak fires it for intentional disconnects too

    # --- test helpers: deliver events on the loop thread, as bleak does ---
    def notify(self, frag: bytes) -> None:
        cb = self.notify_cb[CRTPDOWN_UUID]
        ble_link.get_event_loop().call_soon_threadsafe(cb, None, bytearray(frag))

    def drop_connection(self) -> None:
        self.is_connected = False
        ble_link.get_event_loop().call_soon_threadsafe(self.disconnected_callback, self)

    def real_writes(self):
        return [w for w in self.writes if not (w[0] == CRTP_UUID and w[1] == NULL_PACKET)]

    def null_writes(self):
        return [w for w in self.writes if w[0] == CRTP_UUID and w[1] == NULL_PACKET]


class FakeScanner:
    devices: list[FakeDevice] = []
    calls = 0

    @classmethod
    async def find_device_by_filter(cls, filterfunc, timeout=10.0, **kwargs):
        cls.calls += 1
        for d in cls.devices:
            if filterfunc(d, FakeAdv(local_name=d.name)):
                return d
        return None

    @classmethod
    async def find_device_by_address(cls, address, timeout=10.0, **kwargs):
        cls.calls += 1
        for d in cls.devices:
            if d.address.lower() == address.lower():
                return d
        return None

    @classmethod
    async def discover(cls, timeout=5.0, return_adv=False, **kwargs):
        cls.calls += 1
        if return_adv:
            return {d.address: (d, FakeAdv(local_name=d.name)) for d in cls.devices}
        return list(cls.devices)


OUR_NAME, OUR_ADDR = "Crazyflie-226FCC", "DB:04:E8:22:6F:CC"
OTHER_NAME, OTHER_ADDR = "Crazyflie-AABBCC", "AA:BB:CC:DD:EE:FF"
SCAN_RESULT = [[f"ble://{OUR_ADDR}", OUR_NAME], [f"ble://{OTHER_ADDR}", OTHER_NAME]]   # sorted by name


def default_devices():
    return [
        FakeDevice("SomeHeadphones", "11:22:33:44:55:66"),
        FakeDevice(None, "77:88:99:AA:BB:CC"),
        FakeDevice(OTHER_NAME, OTHER_ADDR),
        FakeDevice(OUR_NAME, OUR_ADDR),
    ]


class DriverTestBase(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []
        FakeClient.connect_error = None
        FakeClient.char_uuids = [CRTP_UUID, CRTPUP_UUID, CRTPDOWN_UUID]
        FakeScanner.devices = default_devices()
        FakeScanner.calls = 0
        patches = [
            mock.patch.object(ble_link, "BleakClient", FakeClient),
            mock.patch.object(ble_link, "BleakScanner", FakeScanner),
            mock.patch.object(ble_link.BleDriver, "scan_timeout", 0.5),
            mock.patch.object(ble_link.BleDriver, "connect_timeout", 0.5),
            mock.patch.object(ble_link.BleDriver, "pump_hz", 0.0),            # pump off unless a test turns it on
            mock.patch.object(ble_link.BleDriver, "stream_ports", ble_link.STREAM_PORTS),
            mock.patch.object(ble_link.BleDriver, "write_with_response", False),
            mock.patch.object(ble_link.BleDriver, "reliable_gap_s", 0.0),     # no pacing pause unless tested
            mock.patch.object(ble_link.BleDriver, "max_inflight_writes", 0),
            mock.patch.object(ble_link.BleDriver, "inflight_window_s", 0.06),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.drivers: list[ble_link.BleDriver] = []

    def tearDown(self):
        for d in self.drivers:
            d.close()

    def make_driver(self, **options) -> ble_link.BleDriver:
        d = ble_link.BleDriver(**options)
        self.drivers.append(d)
        return d

    def connected_driver(self, uri=f"ble://{OUR_ADDR}", err_cb=None, **options):
        drv = self.make_driver(**options)
        drv.connect(uri, None, err_cb if err_cb is not None else mock.Mock())
        return drv, FakeClient.instances[-1]


class TestDriverConnect(DriverTestBase):
    def test_wrong_uri_type_does_not_scan(self):
        drv = self.make_driver()
        for uri in ("usb://0", "radio://0/80/2M/E7E7E7E7E7", "", "bl://x", "ble:/x", None):
            with self.subTest(uri=uri):
                with self.assertRaises(WrongUriType):
                    drv.connect(uri, None, mock.Mock())
        self.assertEqual(FakeScanner.calls, 0)
        self.assertEqual(FakeClient.instances, [])
        self.assertIn("not connected", drv.get_status())

    def test_parse_uri(self):
        p = ble_link.BleDriver.parse_uri
        self.assertEqual(p("ble://"), ("any", "", {}))
        self.assertEqual(p("ble://db:04:e8:22:6f:cc"), ("address", OUR_ADDR, {}))
        self.assertEqual(p("ble://Crazyflie-226FCC"), ("name", OUR_NAME, {}))
        self.assertEqual(p("BLE://Crazyflie-226FCC"), ("name", OUR_NAME, {}))
        kind, value, opts = p(f"ble://{OUR_ADDR}?pump_hz=100&write_with_response=1&max_inflight_writes=2"
                              "&inflight_window_s=0.03&scan_timeout=3&connect_timeout=7&stream_ports=3,6"
                              "&reliable_gap_s=0.05")
        self.assertEqual((kind, value), ("address", OUR_ADDR))
        self.assertEqual(opts, {"pump_hz": 100.0, "write_with_response": True, "max_inflight_writes": 2,
                                "inflight_window_s": 0.03, "scan_timeout": 3.0, "connect_timeout": 7.0,
                                "stream_ports": frozenset({3, 6}), "reliable_gap_s": 0.05})
        self.assertEqual(p("ble://?write_with_response=false")[2], {"write_with_response": False})
        for bad in ("ble://x?nope=1", "ble://x?pump_hz=fast", "ble://x?pump_hz=-1", "ble://x?max_inflight_writes=1.5",
                    "ble://x?write_with_response=maybe", "ble://x?stream_ports=3,99", "ble://x?stream_ports=a"):
            with self.subTest(uri=bad):
                with self.assertRaises(ValueError):
                    p(bad)

    def test_bad_uri_option_raises_plain_exception_not_wrong_uri_type(self):
        drv = self.make_driver()
        with self.assertRaises(ValueError):
            drv.connect(f"ble://{OUR_ADDR}?bogus=1", None, mock.Mock())
        self.assertEqual(FakeScanner.calls, 0)

    def test_uri_options_override_defaults_for_this_connection(self):
        drv, client = self.connected_driver(f"ble://{OUR_ADDR}?pump_hz=25&write_with_response=1&max_inflight_writes=3")
        self.assertEqual((drv.pump_hz, drv.write_with_response, drv.max_inflight_writes), (25.0, True, 3))
        s = drv.stats()
        self.assertEqual((s["pump_hz"], s["write_with_response"], s["max_inflight_writes"]), (25.0, True, 3))
        self.assertEqual(ble_link.BleDriver.pump_hz, 0.0)   # class default untouched
        self.assertIn("pump 25 Hz", drv.get_status())
        self.assertIn("write_with_response=True", drv.get_status())

    def test_constructor_options_and_configure(self):
        drv = self.make_driver(pump_hz=75, max_inflight_writes=1, stream_ports=[6])
        self.assertEqual((drv.pump_hz, drv.max_inflight_writes, drv.stream_ports), (75.0, 1, frozenset({6})))
        with self.assertRaises(ValueError):
            ble_link.BleDriver(nonsense=1)
        with self.assertRaises(ValueError):
            ble_link.BleDriver(pump_hz="fast")
        defaults = ble_link.configure(pump_hz=100, write_with_response=True, max_inflight_writes=4, stream_ports="3,7")
        self.assertEqual((defaults["pump_hz"], defaults["write_with_response"], defaults["max_inflight_writes"],
                          defaults["stream_ports"]), (100.0, True, 4, frozenset({3, 7})))
        fresh = self.make_driver()
        self.assertEqual((fresh.pump_hz, fresh.write_with_response, fresh.max_inflight_writes, fresh.stream_ports),
                         (100.0, True, 4, frozenset({3, 7})))
        self.assertEqual((drv.pump_hz, drv.max_inflight_writes), (75.0, 1))   # instance values win
        with self.assertRaises(ValueError):
            ble_link.configure(pump_hz=-5)
        with self.assertRaises(ValueError):
            ble_link.configure(unknown_option=1)
        self.assertEqual(set(ble_link.OPTIONS), {"pump_hz", "stream_ports", "write_with_response", "reliable_gap_s",
                                                 "max_inflight_writes", "inflight_window_s", "scan_timeout",
                                                 "connect_timeout", "max_uplink_packet"})
        self.assertEqual(ble_link.STREAM_PORTS, frozenset({3, 6, 7, 8}))

    def test_connect_by_address(self):
        drv, client = self.connected_driver(f"ble://{OUR_ADDR}")
        self.assertIs(client.device, FakeScanner.devices[3])
        self.assertTrue(client.is_connected)
        self.assertIn(CRTPDOWN_UUID, client.notify_cb)
        self.assertIn(OUR_NAME, drv.get_status())
        self.assertIn("connected", drv.get_status())
        self.assertEqual(drv.get_name(), "ble")
        self.assertFalse(drv.needs_resending)
        s = drv.stats()
        self.assertTrue(s["connected"])
        self.assertTrue(s["has_crtp_char"])
        self.assertTrue(s["has_crtpup_char"])
        self.assertEqual(s["stream_ports"], [3, 6, 7, 8])

    def test_connect_by_exact_name(self):
        drv, client = self.connected_driver(f"ble://{OTHER_NAME}")
        self.assertEqual(client.device.address, OTHER_ADDR)
        self.assertIn(OTHER_ADDR, drv.get_status())

    def test_connect_first_crazyflie_by_default(self):
        drv, client = self.connected_driver("ble://")
        self.assertEqual(client.device.name, OTHER_NAME)  # first name-prefix match, headphones skipped

    def test_connect_fails_when_no_device(self):
        drv = self.make_driver()
        with self.assertRaises(Exception) as cm:
            drv.connect("ble://Crazyflie-000000", None, mock.Mock())
        self.assertNotIsInstance(cm.exception, WrongUriType)
        self.assertIn("no Crazyflie found", str(cm.exception))
        self.assertEqual(FakeClient.instances, [])
        self.assertIn("not connected", drv.get_status())
        self.assertIsNone(drv.receive_packet(0))

    def test_connect_fails_when_bleak_connect_raises(self):
        FakeClient.connect_error = Exception("simulated connect failure")
        drv = self.make_driver()
        with self.assertRaises(Exception) as cm:
            drv.connect(f"ble://{OUR_ADDR}", None, mock.Mock())
        self.assertIn("simulated connect failure", str(cm.exception))
        self.assertFalse(drv.stats()["connected"])

    def test_connect_fails_when_characteristics_missing(self):
        for uuids in ([CRTP_UUID, CRTPUP_UUID], [CRTPDOWN_UUID], []):
            with self.subTest(uuids=uuids):
                FakeClient.char_uuids = uuids
                err = mock.Mock()
                drv = self.make_driver()
                with self.assertRaises(Exception) as cm:
                    drv.connect(f"ble://{OUR_ADDR}", None, err)
                self.assertIn("characteristics", str(cm.exception))
                client = FakeClient.instances[-1]
                self.assertFalse(client.is_connected)
                self.assertEqual(client.disconnect_calls, 1)
                time.sleep(0.02)
                err.assert_not_called()   # a failed connect is reported by raising, not via the callback

    def test_scan_interface(self):
        drv = self.make_driver()
        self.assertEqual(drv.scan_interface(), SCAN_RESULT)
        self.assertEqual(drv.scan_interface(OUR_ADDR.lower()), [[f"ble://{OUR_ADDR}", OUR_NAME]])
        self.assertEqual(drv.enum(), [f"ble://{OUR_ADDR}", f"ble://{OTHER_ADDR}"])
        self.assertIn("ble://", drv.get_help())

    def test_scan_interface_swallows_errors(self):
        async def boom(**kwargs):
            raise RuntimeError("no adapter")
        with mock.patch.object(FakeScanner, "discover", staticmethod(boom)):
            self.assertEqual(self.make_driver().scan_interface(), [])


class TestSendPolicy(DriverTestBase):
    def test_response_flag_per_port(self):
        drv, client = self.connected_driver()
        ext = cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12))      # port 6, 13 bytes: streaming
        par = cf_packet(CRTPPort.PARAM, 1, bytes([1, 2, 3]))      # port 2: reliable
        drv.send_packet(ext)
        drv.send_packet(par)
        self.assertTrue(wait_until(lambda: len(client.writes) == 2))
        self.assertEqual(client.writes[0], (CRTP_UUID, bytes([ext.header]) + bytes(12), False))
        self.assertEqual(client.writes[1], (CRTP_UUID, bytes([par.header]) + bytes([1, 2, 3]), True))
        # every other non-streaming port is written with response
        reliable_ports = [CRTPPort.CONSOLE, CRTPPort.MEM, CRTPPort.LOGGING, CRTPPort.SUPERVISOR,
                          CRTPPort.PLATFORM, CRTPPort.LINKCTRL, 1, 10]
        for port in reliable_ports:
            drv.send_packet(cf_packet(port, 0, bytes([port])))
        for port in (CRTPPort.COMMANDER, CRTPPort.COMMANDER_GENERIC, CRTPPort.SETPOINT_HL):
            drv.send_packet(cf_packet(port, 0, bytes([port]) * 14))
        self.assertTrue(wait_until(lambda: len(client.writes) == 2 + len(reliable_ports) + 3))
        flags = [(crtp_port(data[0]), resp) for _, data, resp in client.writes[2:]]
        self.assertEqual(flags, [(p, True) for p in reliable_ports] + [(3, False), (7, False), (8, False)])
        s = drv.stats()
        self.assertEqual((s["tx_writes_wwr"], s["tx_writes_wr"]), (4, 1 + len(reliable_ports)))
        self.assertIn("4 wwr / 9 wr writes", drv.get_status())

    def test_long_packets_are_refused_by_default(self):
        # nRF 2024.10 corrupts fragmented packets: a 25-byte packet is dropped, a 13-byte one goes out
        drv, client = self.connected_driver()
        drv.send_packet(cf_packet(CRTPPort.SETPOINT_HL, 0, bytes(24)))
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))
        self.assertTrue(wait_until(lambda: len(client.writes) == 1))
        self.assertNotEqual(client.writes[0][0], CRTPUP_UUID)
        self.assertTrue(wait_until(lambda: drv.stats()["tx_dropped"] == 1))

    def test_25_byte_port_8_packet_is_exactly_two_crtpup_fragments(self):
        drv, client = self.connected_driver(max_uplink_packet=0)
        payload = bytes(range(24))
        drv.send_packet(cf_packet(CRTPPort.SETPOINT_HL, 0, payload))    # 25-byte packet
        self.assertTrue(wait_until(lambda: len(client.writes) == 2))
        pkt = bytes([crtp_header(8, 0)]) + payload
        (u1, f1, r1), (u2, f2, r2) = client.writes
        self.assertEqual((u1, u2), (CRTPUP_UUID, CRTPUP_UUID))
        self.assertEqual((r1, r2), (False, False))
        self.assertEqual(f1[0], 0x80 | (0 << 5) | 24)
        self.assertEqual(f2[0], 0 << 5)
        self.assertEqual(f1[1:] + f2[1:], pkt)
        self.assertEqual(decode_uplink(client.writes), [pkt])
        # second fragmented packet uses pid 1
        drv.send_packet(cf_packet(CRTPPort.SETPOINT_HL, 0, payload))
        self.assertTrue(wait_until(lambda: len(client.writes) == 4))
        self.assertEqual(client.writes[2][1][0], 0x80 | (1 << 5) | 24)
        self.assertEqual(client.writes[3][1][0], 1 << 5)
        time.sleep(0.02)
        self.assertEqual(len(client.writes), 4, "no extra writes for a fragmented packet")

    def test_routing_order_and_pid_over_mixed_sizes(self):
        drv, client = self.connected_driver(max_uplink_packet=0)   # allow fragmentation for this test
        sent = []
        for i, payload_len in enumerate((12, 30, 19, 20, 23, 30, 1, 30, 29)):   # mix of <=20 and 21..31 bytes
            port = CRTPPort.LOCALIZATION if i % 2 == 0 else CRTPPort.SETPOINT_HL
            pk = cf_packet(port, i % 4, bytes([i]) * payload_len)
            sent.append(bytes([pk.header]) + bytes(pk.data))
            drv.send_packet(pk)
        n_writes = sum(1 if len(p) <= 20 else 2 for p in sent)
        self.assertTrue(wait_until(lambda: len(client.writes) == n_writes), client.writes)
        for uuid, chunk, response in client.writes:
            self.assertIs(response, False)             # streaming ports: fire and forget
            self.assertLessEqual(len(chunk), 20)
            if uuid == CRTPUP_UUID and chunk[0] & 0x80:
                self.assertGreater((chunk[0] & 0x1F) + 1, 20, "short packet must never go via CRTPUP")
        self.assertEqual([u for u, _, _ in client.writes],
                         sum(([CRTP_UUID] if len(p) <= 20 else [CRTPUP_UUID] * 2 for p in sent), []))
        self.assertEqual(decode_uplink(client.writes), sent)        # order preserved end to end
        starts = [c for u, c, _ in client.writes if u == CRTPUP_UUID and c[0] & 0x80]
        # 6 packets are longer than 20 bytes (31, 21, 24, 31, 31, 30): PID counts over those only
        self.assertEqual([(c[0] >> 5) & 3 for c in starts], [0, 1, 2, 3, 0, 1])
        s = drv.stats()
        self.assertEqual((s["tx_packets"], s["tx_null"], s["tx_errors"], s["tx_dropped"]), (9, 0, 0, 0))

    def test_reliable_fragmented_packet_pauses_one_connection_interval(self):
        drv, client = self.connected_driver(reliable_gap_s=0.05, max_uplink_packet=0)
        drv.send_packet(cf_packet(CRTPPort.MEM, 0, bytes(30)))            # reliable, 31 bytes -> 2 WWR fragments
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))   # streaming, right behind it
        drv.send_packet(cf_packet(CRTPPort.PARAM, 0, bytes(5)))
        self.assertTrue(wait_until(lambda: len(client.writes) == 4))
        t = client.write_times
        self.assertLess(t[1] - t[0], 0.03)                 # the two fragments go back to back
        self.assertGreaterEqual(t[2] - t[1], 0.045)        # then one connection interval of silence
        self.assertLess(t[3] - t[2], 0.03)                 # short packets are not paced
        self.assertEqual([r for _, _, r in client.writes], [False, False, False, True])

    def test_null_packet_pump_writes_with_response(self):
        drv, client = self.connected_driver(pump_hz=200, max_uplink_packet=0)
        time.sleep(0.15)
        nulls = client.null_writes()
        self.assertGreaterEqual(len(nulls), 5, "pump did not run")
        for uuid, chunk, response in nulls:
            self.assertEqual((uuid, chunk, response), (CRTP_UUID, NULL_PACKET, True))
        self.assertEqual(drv.stats()["tx_null"], len(client.null_writes()))
        self.assertIn("nulls ", drv.get_status())
        # real packets interleave with nulls but keep their order
        sent = []
        for i in range(8):
            pk = cf_packet(CRTPPort.LOCALIZATION, 0, bytes([i]) * (12 if i % 2 else 30))
            sent.append(bytes([pk.header]) + bytes(pk.data))
            drv.send_packet(pk)
        self.assertTrue(wait_until(lambda: drv.stats()["tx_packets"] == 8))
        self.assertEqual(decode_uplink(client.writes), sent)
        before = drv.stats()["tx_null"]
        time.sleep(0.1)
        self.assertGreater(drv.stats()["tx_null"], before)   # keeps pumping afterwards
        drv.close()
        after = drv.stats()["tx_null"]
        time.sleep(0.05)
        self.assertEqual(drv.stats()["tx_null"], after)        # pump stops with the link

    def test_pump_off(self):
        drv, client = self.connected_driver()   # pump_hz patched to 0
        time.sleep(0.1)
        self.assertEqual(client.writes, [])
        self.assertEqual(drv.stats()["tx_null"], 0)

    def test_write_with_response_option_makes_streaming_ports_reliable_too(self):
        drv, client = self.connected_driver(f"ble://{OUR_ADDR}?write_with_response=1&pump_hz=200&max_uplink_packet=0")
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))
        drv.send_packet(cf_packet(CRTPPort.SETPOINT_HL, 0, bytes(25)))
        self.assertTrue(wait_until(lambda: drv.stats()["tx_packets"] == 2 and drv.stats()["tx_null"] >= 1))
        for uuid, chunk, response in client.writes:
            if uuid == CRTP_UUID:
                self.assertIs(response, True, chunk)      # whole packets and nulls await the response
            else:
                self.assertIs(response, False, chunk)     # CRTPUP fragments are write-without-response

    def test_stream_ports_configurable(self):
        drv, client = self.connected_driver(f"ble://{OUR_ADDR}?stream_ports=3,6")
        drv.send_packet(cf_packet(CRTPPort.SETPOINT_HL, 0, bytes(5)))    # port 8 no longer streaming
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(5)))
        self.assertTrue(wait_until(lambda: len(client.writes) == 2))
        self.assertEqual([r for _, _, r in client.writes], [True, False])

    def test_max_inflight_writes_throttles(self):
        drv, client = self.connected_driver(max_inflight_writes=2, inflight_window_s=0.05)
        t0 = time.perf_counter()
        for i in range(6):
            drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes([i]) * 12))
        self.assertTrue(wait_until(lambda: len(client.writes) == 6))
        elapsed = time.perf_counter() - t0
        self.assertGreaterEqual(elapsed, 0.09, "6 writes with 2 per 50 ms must take at least ~100 ms")
        times = client.write_times
        for i in range(len(times) - 2):
            self.assertGreaterEqual(times[i + 2] - times[i], 0.045, "more than 2 writes inside one window")
        self.assertEqual(decode_uplink(client.writes), [bytes([crtp_header(6, 0)]) + bytes([i]) * 12 for i in range(6)])
        drv2, client2 = self.connected_driver(f"ble://{OTHER_ADDR}")    # unlimited: fast
        t0 = time.perf_counter()
        for i in range(6):
            drv2.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes([i]) * 12))
        self.assertTrue(wait_until(lambda: len(client2.writes) == 6))
        self.assertLess(time.perf_counter() - t0, 0.09)

    def test_fallback_without_0202_uses_crtpup(self):
        FakeClient.char_uuids = [CRTPUP_UUID, CRTPDOWN_UUID]
        drv, client = self.connected_driver(pump_hz=200)
        self.assertFalse(drv.stats()["has_crtp_char"])
        pk = cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12))
        drv.send_packet(pk)
        self.assertTrue(wait_until(lambda: drv.stats()["tx_packets"] == 1 and drv.stats()["tx_null"] >= 1))
        self.assertTrue(all(u == CRTPUP_UUID and r is False for u, _, r in client.writes))
        nrf = FakeNrfUplinkReceiver()
        for _, chunk, _ in client.writes:
            nrf.write(chunk)
        self.assertIn(bytes([pk.header]) + bytes(12), nrf.packets)
        self.assertIn(NULL_PACKET, nrf.packets)
        self.assertEqual(nrf.errors, 0)

    def test_without_0203_long_packets_are_dropped_short_ones_sent(self):
        FakeClient.char_uuids = [CRTP_UUID, CRTPDOWN_UUID]
        drv, client = self.connected_driver()
        self.assertFalse(drv.stats()["has_crtpup_char"])
        short = cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12))
        drv.send_packet(cf_packet(CRTPPort.SETPOINT_HL, 0, bytes(25)))
        drv.send_packet(short)
        self.assertTrue(wait_until(lambda: drv.stats()["tx_packets"] == 1))
        self.assertEqual(client.writes, [(CRTP_UUID, bytes([short.header]) + bytes(12), False)])
        self.assertEqual(drv.stats()["tx_dropped"], 1)
        self.assertEqual(drv.stats()["tx_errors"], 0)
        self.assertTrue(drv.stats()["connected"])

    def test_send_before_connect_and_after_close_is_silent(self):
        drv = self.make_driver()
        pk = cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12))
        drv.send_packet(pk)   # must not raise
        self.assertEqual(drv.stats()["tx_dropped"], 1)
        drv.connect(f"ble://{OUR_ADDR}", None, mock.Mock())
        drv.close()
        drv.send_packet(pk)
        self.assertEqual(drv.stats()["tx_dropped"], 2)

    def test_oversized_packet_is_dropped_not_raised(self):
        drv, client = self.connected_driver()
        drv.send_packet(CRTPPacket(crtp_header(6, 0), bytes(31)))
        time.sleep(0.05)
        self.assertEqual(client.writes, [])
        self.assertEqual(drv.stats()["tx_dropped"], 1)

    def test_close_flushes_queued_packets(self):
        # cflib's close_link() sends a zero setpoint and closes the link right away
        drv, client = self.connected_driver(pump_hz=100)
        stop = cf_packet(CRTPPort.COMMANDER, 0, bytes(14))
        drv.send_packet(stop)
        drv.send_packet(cf_packet(CRTPPort.PARAM, 0, bytes(3)))
        drv.close()
        pkts = decode_uplink(client.writes)
        self.assertEqual(pkts[-2:], [bytes([stop.header]) + bytes(14), bytes([crtp_header(2, 0)]) + bytes(3)])
        self.assertFalse(client.is_connected)
        self.assertEqual(client.disconnect_calls, 1)


class TestReceiveAndLifecycle(DriverTestBase):
    def test_receive_packet_timing_semantics(self):
        drv, client = self.connected_driver()
        t0 = time.perf_counter()
        self.assertIsNone(drv.receive_packet(0))                      # time=0: non-blocking
        self.assertLess(time.perf_counter() - t0, 0.05)
        t0 = time.perf_counter()
        self.assertIsNone(drv.receive_packet(0.1))                    # time>0: times out
        dt = time.perf_counter() - t0
        self.assertGreaterEqual(dt, 0.08)
        self.assertLess(dt, 1.0)
        # a notification (firmware 2024.10 framing: lenfield = full length) becomes a CRTPPacket
        pkt = make_packet(CRTPPort.CONSOLE, 0, b"hello")
        client.notify(bytes([0x80 | len(pkt)]) + pkt)
        got = drv.receive_packet(1.0)
        self.assertIsInstance(got, CRTPPacket)
        self.assertEqual((got.port, got.channel, bytes(got.data), got.header), (CRTPPort.CONSOLE, 0, b"hello", pkt[0]))
        self.assertIsNone(drv.receive_packet(0))
        # time=None: blocks until something arrives (two-fragment packet, repeated start byte)
        pkt2 = make_packet(CRTPPort.PARAM, 1, bytes(range(30)))
        ctrl = bytes([0x80 | len(pkt2)])
        threading.Timer(0.15, lambda: (client.notify(ctrl + pkt2[:19]), client.notify(ctrl + pkt2[19:]))).start()
        t0 = time.perf_counter()
        got = drv.receive_packet(None)
        self.assertGreaterEqual(time.perf_counter() - t0, 0.1)
        self.assertEqual((got.port, got.channel, bytes(got.data)), (CRTPPort.PARAM, 1, bytes(range(30))))
        # time<0 behaves like None
        threading.Timer(0.05, lambda: client.notify(bytes([0x80 | len(pkt)]) + pkt)).start()
        self.assertEqual(bytes(drv.receive_packet(-1).data), b"hello")
        s = drv.stats()
        self.assertEqual((s["rx_packets"], s["rx_fragments"], s["rx_orphans"], s["rx_convention"]), (3, 4, 0, "len"))

    def test_receive_works_while_pump_runs(self):
        drv, client = self.connected_driver(pump_hz=200)
        pkts = [make_packet(CRTPPort.LOGGING, 2, bytes([i]) * 20) for i in range(20)]
        for p in pkts:
            client.notify(bytes([0x80 | len(p)]) + p[:19])
            client.notify(bytes([0x80 | len(p)]) + p[19:])
        got = [drv.receive_packet(1.0) for _ in pkts]
        self.assertEqual([bytes([g.header]) + bytes(g.data) for g in got], pkts)
        self.assertGreater(drv.stats()["tx_null"], 0)
        self.assertEqual(drv.stats()["rx_orphans"], 0)

    def test_captured_frames_and_console_dump_pass_through(self):
        drv, client = self.connected_driver()
        client.notify(CAPTURED_LONG_1)
        client.notify(CAPTURED_LONG_2)
        client.notify(CAPTURED_SHORT)
        got = drv.receive_packet(1.0)
        self.assertEqual((got.port, got.channel), (CRTPPort.CONSOLE, 0))
        self.assertEqual(bytes(got.data), CAPTURED_LONG_PACKET[1:])
        self.assertEqual(len(got.data), 30)
        got = drv.receive_packet(1.0)
        self.assertEqual(bytes(got.data), CAPTURED_SHORT_PACKET[1:])
        # a burst of ~35 console packets on connect is just queued in order
        for i in range(35):
            p = make_packet(CRTPPort.CONSOLE, 0, f"line {i:02d}\n".encode())
            client.notify(bytes([0x80 | len(p)]) + p)
        lines = [bytes(drv.receive_packet(1.0).data) for _ in range(35)]
        self.assertEqual(lines, [f"line {i:02d}\n".encode() for i in range(35)])
        self.assertIsNone(drv.receive_packet(0))
        self.assertEqual(drv.stats()["rx_packets"], 37)

    def test_other_ports_pass_through_doc_framing(self):
        drv, client = self.connected_driver()
        pkts = [
            make_packet(CRTPPort.LINKCTRL, 0, bytes(12)),
            make_packet(CRTPPort.LOGGING, 2, bytes(25)),
            make_packet(CRTPPort.PLATFORM, 1, b""),   # header only
            make_packet(CRTPPort.PARAM, 0, bytes(29)),
        ]
        for i, p in enumerate(pkts):
            for f in crtpup_fragments(p, i):          # doc framing: len-1 + pid-only continuation
                client.notify(f)
        got = [drv.receive_packet(1.0) for _ in pkts]
        self.assertEqual([bytes([g.header]) + bytes(g.data) for g in got], pkts)
        self.assertIsNone(drv.receive_packet(0))
        self.assertEqual(drv.stats()["rx_convention"], "len-1")

    def test_unexpected_disconnect_reports_once_and_stays_safe(self):
        err = mock.Mock()
        drv, client = self.connected_driver(err_cb=err, pump_hz=100)
        waiter_result = []
        waiter = threading.Thread(target=lambda: waiter_result.append(drv.receive_packet(None)))
        waiter.start()
        time.sleep(0.05)
        client.drop_connection()
        self.assertTrue(wait_until(lambda: err.call_count == 1))
        msg = err.call_args[0][0]
        self.assertIn("BleDriver", msg)
        self.assertIn("lost", msg)
        self.assertIn(OUR_NAME, msg)
        waiter.join(2.0)
        self.assertFalse(waiter.is_alive(), "blocked receive_packet(None) was not woken by the disconnect")
        self.assertEqual(waiter_result, [None])
        self.assertIsNone(drv.receive_packet(0))
        self.assertIsNone(drv.receive_packet(None))   # link down + empty: returns at once
        self.assertIsNone(drv.receive_packet(0.05))
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))   # no raise
        self.assertIn("link lost", drv.get_status())
        self.assertFalse(drv.stats()["connected"])
        nulls = drv.stats()["tx_null"]
        time.sleep(0.05)
        self.assertEqual(drv.stats()["tx_null"], nulls)   # pump stopped
        client.drop_connection()                          # a second event or close() must not report again
        drv.close()
        drv.close()
        time.sleep(0.1)
        self.assertEqual(err.call_count, 1)

    def test_intentional_close_does_not_report_error(self):
        err = mock.Mock()
        drv, client = self.connected_driver(err_cb=err, pump_hz=100)
        time.sleep(0.03)
        drv.close()
        self.assertFalse(client.is_connected)
        self.assertEqual(client.disconnect_calls, 1)
        self.assertEqual(client.stop_notify_calls, [CRTPDOWN_UUID])
        time.sleep(0.05)
        err.assert_not_called()
        self.assertIn("closed", drv.get_status())

    def test_close_is_idempotent_and_drains(self):
        drv, client = self.connected_driver()
        p = make_packet(0, 0, b"x")
        client.notify(bytes([0x80 | len(p)]) + p)
        self.assertTrue(wait_until(lambda: drv.stats()["rx_packets"] == 1))
        drv.close()
        drv.close()
        drv.close()
        self.assertEqual(client.disconnect_calls, 1)
        self.assertIsNone(drv.receive_packet(0))
        self.assertIsNone(drv.receive_packet(None))
        self.assertIsNone(drv.receive_packet(0.01))
        self.make_driver().close()   # never connected: still safe

    def test_persistent_write_errors_report_link_error(self):
        err = mock.Mock()
        drv, client = self.connected_driver(err_cb=err)
        client.fail_writes = True
        for _ in range(ble_link.MAX_CONSECUTIVE_TX_ERRORS + 3):
            drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))
        self.assertTrue(wait_until(lambda: err.call_count == 1))
        self.assertIn("write errors", err.call_args[0][0])
        self.assertEqual(drv.stats()["tx_errors"], ble_link.MAX_CONSECUTIVE_TX_ERRORS)
        self.assertIn(f"{ble_link.MAX_CONSECUTIVE_TX_ERRORS} err", drv.get_status())
        self.assertFalse(drv.stats()["connected"])
        time.sleep(0.1)
        self.assertEqual(err.call_count, 1)

    def test_single_write_error_is_tolerated(self):
        err = mock.Mock()
        drv, client = self.connected_driver(err_cb=err)
        client.fail_writes = True
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))
        self.assertTrue(wait_until(lambda: drv.stats()["tx_errors"] == 1))
        client.fail_writes = False
        drv.send_packet(cf_packet(CRTPPort.LOCALIZATION, 0, bytes(12)))
        self.assertTrue(wait_until(lambda: drv.stats()["tx_packets"] == 1))
        time.sleep(0.05)
        err.assert_not_called()
        self.assertTrue(drv.stats()["connected"])


class TestCflibIntegration(DriverTestBase):
    """Goes through cflib's own dispatch and Crazyflie class, with the fake BLE stack."""

    def setUp(self):
        super().setUp()
        self._saved_classes = list(cflib.crtp.CLASSES)
        self.addCleanup(self._restore_classes)

    def _restore_classes(self):
        cflib.crtp.CLASSES[:] = self._saved_classes

    def test_register_is_idempotent(self):
        cflib.crtp.CLASSES[:] = []
        self.assertIs(ble_link.register(), ble_link.BleDriver)
        ble_link.register()
        ble_link.register()
        self.assertEqual(cflib.crtp.CLASSES.count(ble_link.BleDriver), 1)
        self.assertEqual(FakeScanner.calls, 0)   # registering never scans
        self.assertEqual(FakeClient.instances, [])

    def test_get_link_driver_and_scan_interfaces_through_cflib(self):
        cflib.crtp.CLASSES[:] = []
        ble_link.register()
        self.assertEqual(cflib.crtp.scan_interfaces(), SCAN_RESULT)
        self.assertEqual(cflib.crtp.get_interfaces_status(), {"ble": ble_link.BleDriver().get_status()})
        err = mock.Mock()
        link = cflib.crtp.get_link_driver(f"ble://{OUR_NAME}", None, err)
        self.assertIsInstance(link, ble_link.BleDriver)
        self.drivers.append(link)
        self.assertTrue(FakeClient.instances[-1].is_connected)
        self.assertIsNone(cflib.crtp.get_link_driver("usb://0", None, err))  # only BleDriver registered here

    def test_crazyflie_open_link_rx_and_close_link(self):
        from cflib.crazyflie import Crazyflie
        cflib.crtp.CLASSES[:] = []
        ble_link.register()
        cf = Crazyflie()
        established = threading.Event()
        disconnected = threading.Event()
        failed = []
        cf.link_established.add_callback(lambda uri: established.set())
        cf.disconnected.add_callback(lambda uri: disconnected.set())
        cf.connection_failed.add_callback(lambda uri, msg: failed.append(msg))
        cf.open_link(f"ble://{OUR_ADDR}?pump_hz=100")
        self.assertEqual(failed, [])
        self.assertIsInstance(cf.link, ble_link.BleDriver)
        client = FakeClient.instances[-1]
        # cflib starts the connection setup right away (reliable, with response) and the pump runs
        self.assertTrue(wait_until(lambda: len(client.real_writes()) >= 1 and len(client.null_writes()) >= 2))
        first = client.real_writes()[0]
        self.assertEqual(first[0], CRTP_UUID)
        self.assertIs(first[2], True)
        self.assertNotIn(crtp_port(first[1][0]), ble_link.STREAM_PORTS)
        self.assertGreaterEqual(len(decode_uplink(client.writes)), 1)
        # downlink through cflib's incoming thread (console dump style packet)
        console = make_packet(CRTPPort.CONSOLE, 0, b"SYS: hi\n")
        client.notify(bytes([0x80 | len(console)]) + console)
        self.assertTrue(established.wait(2.0))
        cf.close_link()
        self.assertTrue(disconnected.is_set())
        self.assertIsNone(cf.link)
        self.assertFalse(client.is_connected)
        self.assertEqual(client.disconnect_calls, 1)
        # cflib's zero setpoint sent by close_link() was flushed before the disconnect
        last = decode_uplink(client.writes)[-1]
        self.assertEqual((crtp_port(last[0]), len(last)), (CRTPPort.COMMANDER, 15))
        n = len(client.writes)
        time.sleep(0.05)
        self.assertEqual(len(client.writes), n)   # pump stopped with the link

    def test_crazyflie_handles_ble_drop_via_link_error_callback(self):
        from cflib.crazyflie import Crazyflie
        cflib.crtp.CLASSES[:] = []
        ble_link.register()
        cf = Crazyflie()
        failed = threading.Event()
        msgs = []
        cf.connection_failed.add_callback(lambda uri, msg: (msgs.append(msg), failed.set()))
        cf.open_link(f"ble://{OUR_ADDR}")
        link = cf.link
        client = FakeClient.instances[-1]
        client.drop_connection()
        # cflib's _link_error_cb runs on our helper thread, calls link.close() and reports
        self.assertTrue(failed.wait(3.0), "connection_failed was not called after the BLE drop")
        self.assertIn("lost", msgs[0])
        self.assertIsNone(cf.link)
        self.assertFalse(link.stats()["connected"])
        self.assertIn("link lost", link.get_status())
        cf.close_link()   # harmless with link already gone


if __name__ == "__main__":
    unittest.main()
