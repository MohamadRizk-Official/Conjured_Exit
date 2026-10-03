"""
ble_framing.py -- CRTP-over-Bluetooth-LE framing for the Crazyflie 2.x.

Shared by the cflib link driver (ble_link.py) and its tests. Dependency-free so it can be
unit-tested without bleak, cflib or a drone.

Protocol facts (verified on our drone, nRF firmware 2024.10, on 2026-10-03; the firmware
sources and the official doc are at
https://www.bitcraze.io/documentation/repository/crazyflie2-nrf-firmware/master/protocols/ble/):

* A CRTP packet is 1 header byte + up to 30 payload bytes (31 bytes max).
  header = (port << 4) | 0x0C | channel  (cflib sets bits 2-3; the drone sends them as 0).
* GATT service 00000201-1c7f-4f9e-947b-43b7c00a9a08 with three characteristics:
    CRTP     ...0202  one whole packet, <= 20 bytes. write-without-response works
                      (verified at 100 Hz). THE route for every packet of <= 20 bytes.
    CRTPUP   ...0203  uplink fragments: 1 control byte + <= 19 data bytes, write-without-
                      response. Used ONLY for packets of 21..31 bytes, always as exactly two
                      fragments in the official convention (below). A single-fragment CRTPUP
                      write is NOT delivered by firmware 2024.10, so short packets never go here.
    CRTPDOWN ...0204  downlink notifications, same fragment layout. The ONLY downlink.
* Official (doc) control byte, used for UPLINK fragments:
      bit 7 = start of packet, bits 5-6 = PID (incremented once per fragmented packet, mod 4),
      bits 0-4 = packet length - 1. The continuation fragment's control byte is the PID bits only.
* DOWNLINK framing on firmware 2024.10 deviates from the doc: bits 0-4 = FULL packet length,
  the second fragment repeats the first control byte verbatim (start bit set), PID is always 0.
  Master firmware (2026-09-03 "Fix malformed BLE CRTP downlink packets") follows the doc.
  CrtpDownReassembler accepts both and learns which one the drone speaks.
* Downlink is ACK-driven: the STM32 (radiolink.c) releases exactly ONE queued downlink packet
  per uplink packet received. Idle links must therefore be pumped with null packets (one byte
  0xFF, like cflib's radiodriver does); see ble_link.BleDriver.pump_hz.
* The advertisement contains the name "Crazyflie-XXXXXX" and the Device Information service
  only, NOT the Crazyflie service UUID: scan by name prefix.
"""
from __future__ import annotations

__all__ = [
    "SERVICE_UUID", "CRTP_UUID", "CRTPUP_UUID", "CRTPDOWN_UUID", "NAME_PREFIX",
    "MAX_FRAGMENT_PAYLOAD", "FRAGMENT_MAX", "CRTP_MAX_PACKET", "CRTP_MAX_PAYLOAD",
    "CRTP_CHAR_MAX_PACKET", "NULL_PACKET", "UnroutablePacket",
    "crtp_header", "crtp_port", "crtp_channel", "crtpup_fragments", "uplink_writes",
    "CrtpDownReassembler",
]

SERVICE_UUID = "00000201-1c7f-4f9e-947b-43b7c00a9a08"
CRTP_UUID = "00000202-1c7f-4f9e-947b-43b7c00a9a08"      # whole packet <= 20 B (write / write-without-response)
CRTPUP_UUID = "00000203-1c7f-4f9e-947b-43b7c00a9a08"    # control byte + data, write-without-response
CRTPDOWN_UUID = "00000204-1c7f-4f9e-947b-43b7c00a9a08"  # control byte + data, notify (the ONLY downlink)
NAME_PREFIX = "Crazyflie"                               # advertised local name is "Crazyflie-XXXXXX"

MAX_FRAGMENT_PAYLOAD = 19     # 20-byte characteristic minus 1 control byte
FRAGMENT_MAX = 20             # largest CRTPUP write / CRTPDOWN notification
CRTP_MAX_PAYLOAD = 30
CRTP_MAX_PACKET = 31          # 1 header + 30 payload
CRTP_CHAR_MAX_PACKET = 20     # packets up to this size are written whole to the CRTP characteristic
NULL_PACKET = b"\xff"         # link-layer null packet: lets the drone release one queued downlink packet

_START = 0x80
_PID_SHIFT = 5
_PID_MASK = 0x03
_LEN_MASK = 0x1F


class UnroutablePacket(ValueError):
    """The connected device lacks the characteristic needed to carry this packet."""


def crtp_header(port: int, channel: int) -> int:
    """Build a CRTP header byte: (port << 4) | 0x0C | channel."""
    return ((port & 0x0F) << 4) | 0x0C | (channel & 0x03)


def crtp_port(header: int) -> int:
    """Port number (0-15) of a CRTP header byte."""
    return (header >> 4) & 0x0F


def crtp_channel(header: int) -> int:
    """Channel number (0-3) of a CRTP header byte."""
    return header & 0x03


def crtpup_fragments(packet: bytes, pid: int) -> list[bytes]:
    """Split one whole CRTP packet (header + payload) into CRTPUP writes, official convention.

    First fragment: control = 0x80 | (pid & 3) << 5 | (len(packet) - 1), then up to 19 bytes.
    Continuation (only for packets longer than 19 bytes): control = (pid & 3) << 5, then the rest.
    This is the only variant nRF firmware 2024.10 accepted on the uplink. The caller
    increments ``pid`` once per fragmented packet.
    """
    if not 1 <= len(packet) <= CRTP_MAX_PACKET:
        raise ValueError(f"bad CRTP packet length {len(packet)} (must be 1..{CRTP_MAX_PACKET})")
    packet = bytes(packet)
    pid_bits = (pid & _PID_MASK) << _PID_SHIFT
    first = bytes([_START | pid_bits | ((len(packet) - 1) & _LEN_MASK)]) + packet[:MAX_FRAGMENT_PAYLOAD]
    if len(packet) <= MAX_FRAGMENT_PAYLOAD:
        return [first]
    return [first, bytes([pid_bits]) + packet[MAX_FRAGMENT_PAYLOAD:]]


def uplink_writes(packet: bytes, pid: int, has_crtp: bool = True, has_crtpup: bool = True) -> list[tuple[str, bytes]]:
    """Route one whole CRTP packet to GATT writes: a list of (characteristic uuid, bytes).

    <= 20 bytes -> one write of the whole packet on CRTP (0202).
    21..31 bytes -> two CRTPUP (0203) fragments, official convention.
    If 0202 is missing, short packets fall back to a single CRTPUP fragment (doc behaviour;
    known NOT to be delivered by firmware 2024.10, which does have 0202). If 0203 is missing
    a long packet cannot be sent: UnroutablePacket.
    """
    packet = bytes(packet)
    if not 1 <= len(packet) <= CRTP_MAX_PACKET:
        raise ValueError(f"bad CRTP packet length {len(packet)} (must be 1..{CRTP_MAX_PACKET})")
    if len(packet) <= CRTP_CHAR_MAX_PACKET and has_crtp:
        return [(CRTP_UUID, packet)]
    if has_crtpup:
        return [(CRTPUP_UUID, frag) for frag in crtpup_fragments(packet, pid)]
    raise UnroutablePacket(
        f"cannot send a {len(packet)}-byte CRTP packet: device has "
        f"{'no CRTPUP (0203) characteristic' if has_crtp else 'neither CRTP (0202) nor CRTPUP (0203)'}")


class CrtpDownReassembler:
    """Rebuilds whole CRTP packets from CRTPDOWN notification fragments; tolerant of both
    framing variants seen in the wild.

    Feed every notification payload to :meth:`feed`; it returns the complete packet
    (header + payload, ``bytes``) when one is finished, otherwise ``None``.

    Rules:
      * A fragment continues the pending packet if its start bit is clear and its PID matches
        (doc firmware), OR its control byte equals the pending one verbatim (firmware
        2024.10 repeats the start byte). A continuation that would exceed 31 bytes is not one.
      * A start fragment with fewer than 19 data bytes is a complete packet. One with 19 data
        bytes waits for a continuation only if the expected total exceeds 19, where the
        expected total is the length field itself ("len" convention) until a two-fragment
        packet reveals the convention (field == total -> "len", field + 1 == total -> "len-1").
        A continuation that arrives while nothing is pending is an orphan; while the
        convention is still unknown it proves the drone speaks "len-1".
      * Received bytes are delivered as-is, never truncated to the length field.

    Counters: :attr:`packets`, :attr:`orphans` (continuations with no pending packet),
    :attr:`incomplete` (pending packets abandoned when a new start arrived) and
    :attr:`convention` (None until learned).
    """

    def __init__(self) -> None:
        self.convention: str | None = None
        self.packets = 0
        self.orphans = 0
        self.incomplete = 0
        self._ctrl: int | None = None   # control byte of the pending packet, None when idle
        self._pid = 0
        self._lenfield = 0
        self._buf = bytearray()

    @property
    def pending(self) -> bool:
        """True while the first fragment of a long packet is waiting for its continuation."""
        return self._ctrl is not None

    def _expected_total(self, lenfield: int) -> int:
        return lenfield + 1 if self.convention == "len-1" else lenfield

    def _learn(self, lenfield: int, total: int) -> None:
        if self.convention is not None:
            return
        if lenfield == total:
            self.convention = "len"
        elif lenfield + 1 == total:
            self.convention = "len-1"

    def feed(self, frag: bytes) -> bytes | None:
        """Consume one notification payload; return a whole CRTP packet when complete, else None."""
        if not frag:
            return None
        frag = bytes(frag)
        ctrl, data = frag[0], frag[1:]
        start = bool(ctrl & _START)
        pid = (ctrl >> _PID_SHIFT) & _PID_MASK
        lenfield = ctrl & _LEN_MASK

        if self._ctrl is not None:
            looks_like_continuation = (not start and pid == self._pid) or ctrl == self._ctrl
            if looks_like_continuation and len(self._buf) + len(data) <= CRTP_MAX_PACKET:
                self._buf += data
                pkt = bytes(self._buf)
                self._learn(self._lenfield, len(pkt))
                self._ctrl = None
                self.packets += 1
                return pkt
            self.incomplete += 1   # a different packet started: the pending one is lost
            self._ctrl = None

        if not start:
            self.orphans += 1
            if self.convention is None:
                self.convention = "len-1"   # we must have released a 19-byte packet too early
            return None

        if len(data) >= MAX_FRAGMENT_PAYLOAD and self._expected_total(lenfield) > MAX_FRAGMENT_PAYLOAD:
            self._ctrl, self._pid, self._lenfield = ctrl, pid, lenfield
            self._buf = bytearray(data)
            return None

        self.packets += 1
        return data
