#!/usr/bin/env python
"""
ble_bench.py -- Pathcaster task 1: standalone Bluetooth LE benchmark for the Crazyflie 2.x.

Pure bleak, no cflib. Sends ONLY CRTP link-layer packets (port 15: echo ch 0, sink ch 2)
and null packets (0xFF). It never touches the commander, so the motors cannot start.

Hardware facts this script is built on (verified live on our drone, nRF firmware 2024.10):
  * The STM32 releases exactly ONE queued downlink packet per uplink packet it receives
    (radiolink.c). Nothing comes down unless something goes up, so when idle we "pump"
    the link with 0xFF null packets, like cflib's radio driver does.
  * Uplink: the CRTP characteristic 0202 takes a whole packet <= 20 bytes with
    write-without-response. The fragmented CRTPUP 0203 route is probed separately.
  * Downlink: notifications on CRTPDOWN 0204, control byte + data. On 2024.10 the length
    field is the FULL packet length and the 2nd fragment repeats the start control byte;
    master firmware (fixed 2026-09) uses length-1 and a pid-only continuation byte.
    The reassembler tolerates both.

Phases
  0. Drain : pump nulls until the drone's console backlog (and any stale replies) is out.
  1. CRTPUP: probe the fragmented uplink with each length convention, short + long packets.
  2. Latency: N echo packets (13 bytes) -> RTT median / p95 / max, loss.
  3. Throughput: 13-byte sink packets at 30 / 60 / 100 Hz for 10 s each -> achieved rate,
     write errors, late sends, disconnects; every Nth packet is an echo probe (RTT + loss
     under load).
  PASS = 30 Hz holds with < 5 % loss AND p95 RTT < 100 ms AND no disconnect.

Usage (PowerShell, inside the cf venv):
    python ble_bench.py                               # scan for "Crazyflie-*", run everything
    python ble_bench.py --address DB:04:E8:22:6F:CC
    python ble_bench.py --rates 30 60 --duration 5 --echo-count 100
    python ble_bench.py --echo-mode pingpong          # one-at-a-time RTT instead of paced

Before running: power-cycle the drone if a Crazyradio has talked to it, do NOT pair it in
Windows settings, and unplug the USB cable.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import statistics
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic

# --- Protocol constants (verified against crazyflie-firmware / crazyflie2-nrf-firmware) ---
SERVICE_UUID = "00000201-1c7f-4f9e-947b-43b7c00a9a08"
CRTP_UUID = "00000202-1c7f-4f9e-947b-43b7c00a9a08"      # whole packet <= 20 B, write / write-without-response
CRTPUP_UUID = "00000203-1c7f-4f9e-947b-43b7c00a9a08"    # control byte + data, write-without-response
CRTPDOWN_UUID = "00000204-1c7f-4f9e-947b-43b7c00a9a08"  # control byte + data, notify (the ONLY downlink)
NAME_PREFIX = "Crazyflie"

CRTP_PORT_CONSOLE = 0x00
CRTP_PORT_LINK = 0x0F
LINK_ECHO, LINK_SOURCE, LINK_SINK = 0, 1, 2
MAX_FRAGMENT_PAYLOAD = 19          # 20-byte characteristic minus 1 control byte
CRTP_MAX_PACKET = 31               # 1 header + 30 payload
WHOLE_PACKET_MAX = 20              # biggest packet the CRTP characteristic accepts
NULL_PACKET = bytes([0xFF])        # cflib's "nothing to say, give me downlink" packet

ECHO_FMT = "<Id"                   # seq (u32) + send time (f64) = 12 B -> 13 B packet with header
ECHO_LEN = 1 + struct.calcsize(ECHO_FMT)


def crtp_header(port: int, channel: int) -> int:
    return ((port & 0x0F) << 4) | 0x0C | (channel & 0x03)


def crtpup_fragments(packet: bytes, pid: int, len_mode: str = "len-1", cont_mode: str = "pid") -> list[bytes]:
    """Split one whole CRTP packet (header + payload) into CRTPUP writes.

    len_mode : 'len-1' (official doc) or 'len' (low 5 bits = full packet length).
    cont_mode: 'pid' (continuation control byte = pid bits only, official doc) or
               'repeat' (continuation repeats the start control byte, 2024.10 downlink style).
    """
    if not 1 <= len(packet) <= CRTP_MAX_PACKET:
        raise ValueError(f"bad CRTP packet length {len(packet)}")
    lenfield = len(packet) - 1 if len_mode == "len-1" else len(packet)
    ctrl = 0x80 | ((pid & 3) << 5) | (lenfield & 0x1F)
    first = bytes([ctrl]) + packet[:MAX_FRAGMENT_PAYLOAD]
    if len(packet) <= MAX_FRAGMENT_PAYLOAD:
        return [first]
    cont = ctrl if cont_mode == "repeat" else ((pid & 3) << 5)
    return [first, bytes([cont]) + packet[MAX_FRAGMENT_PAYLOAD:]]


def crtpup_fragments_forced_two(packet: bytes, pid: int) -> list[bytes]:
    """Doc convention, but a short packet is followed by an empty pid-only continuation."""
    frags = crtpup_fragments(packet, pid, "len-1", "pid")
    if len(frags) == 1:
        frags.append(bytes([(pid & 3) << 5]))
    return frags


class CrtpDownReassembler:
    """Rebuilds whole CRTP packets from CRTPDOWN fragments; tolerant of both nRF framings."""

    def __init__(self) -> None:
        self.ctrl: int | None = None       # control byte of the pending (incomplete) packet
        self.lenfield = 0
        self.buf = bytearray()
        self.orphans = 0                   # continuation with nothing pending
        self.dropped = 0                   # pending packet abandoned by a new start fragment
        self.convention: str | None = None  # 'len' (2024.10) or 'len-1' (doc), learned at runtime

    def _expected(self) -> int:
        return self.lenfield + 1 if self.convention == "len-1" else self.lenfield

    def feed(self, frag: bytes) -> bytes | None:
        if not frag:
            return None
        ctrl = frag[0]
        start = bool(ctrl & 0x80)
        pid = (ctrl >> 5) & 3
        if self.ctrl is not None and ((not start and pid == ((self.ctrl >> 5) & 3)) or ctrl == self.ctrl):
            self.buf += frag[1:]
            pkt = bytes(self.buf)
            if self.lenfield == len(pkt):
                self.convention = "len"
            elif self.lenfield + 1 == len(pkt):
                self.convention = "len-1"
            self.ctrl = None
            self.buf = bytearray()
            return pkt
        if not start:
            self.orphans += 1
            if self.convention is None:
                self.convention = "len-1"
            return None
        if self.ctrl is not None:
            self.dropped += 1
        data = frag[1:]
        self.lenfield = ctrl & 0x1F
        if len(data) >= MAX_FRAGMENT_PAYLOAD and self._expected() > len(data):
            self.ctrl = ctrl
            self.buf = bytearray(data)
            return None
        self.ctrl = None
        return bytes(data)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


def ms(x: float) -> str:
    return "nan" if x != x else f"{x * 1000:.1f} ms"


class Bench:
    def __init__(self, client: BleakClient, args: argparse.Namespace) -> None:
        self.client = client
        self.args = args
        self.verbose = args.verbose
        self.pid = 0
        self.down = CrtpDownReassembler()
        self.send_lock = asyncio.Lock()
        self.t_last_send = 0.0
        self.t_last_rx = 0.0
        self.nulls_sent = 0
        self.pump_errors = 0
        self.pump_enabled = False
        self.pump_hz = args.pump_hz
        self.rx_by_port: dict[int, int] = {}
        self.rx_frags = 0
        self.pending: dict[int, float] = {}
        self.rtts: list[float] = []
        self.stale_echo: list[bytes] = []   # echo replies we did not ask for in this phase
        self.console = bytearray()
        self.console_lines: list[str] = []
        self.disconnected = asyncio.Event()
        self.waiters: dict[int, asyncio.Future] = {}

    # ---------------- uplink ----------------
    async def send(self, packet: bytes, route: str | None = None,
                   len_mode: str = "len-1", cont_mode: str = "pid") -> None:
        if route is None:
            route = "crtp" if len(packet) <= WHOLE_PACKET_MAX else "crtpup"
        async with self.send_lock:
            if route == "crtp":
                if len(packet) > WHOLE_PACKET_MAX:
                    raise ValueError("packet too long for the CRTP characteristic")
                await self.client.write_gatt_char(CRTP_UUID, packet, response=self.args.with_response)
            else:
                frags = (crtpup_fragments_forced_two(packet, self.pid) if cont_mode == "empty"
                         else crtpup_fragments(packet, self.pid, len_mode, cont_mode))
                for frag in frags:
                    await self.client.write_gatt_char(CRTPUP_UUID, frag, response=False)
                self.pid = (self.pid + 1) & 3
            self.t_last_send = time.perf_counter()

    async def pump_task(self) -> None:
        """Keep the downlink flowing: send a null packet whenever the link has been idle."""
        while not self.disconnected.is_set():
            if self.pump_hz <= 0:
                await asyncio.sleep(0.05)
                continue
            interval = 1.0 / self.pump_hz
            await asyncio.sleep(interval / 2)
            if self.pump_enabled and time.perf_counter() - self.t_last_send >= interval:
                try:
                    await self.send(NULL_PACKET, route="crtp")
                    self.nulls_sent += 1
                except Exception:  # noqa: BLE001
                    self.pump_errors += 1

    @staticmethod
    def echo_packet(seq: int, t: float, pad: int = 0) -> bytes:
        return bytes([crtp_header(CRTP_PORT_LINK, LINK_ECHO)]) + struct.pack(ECHO_FMT, seq, t) + bytes(pad)

    @staticmethod
    def sink_packet(seq: int, t: float) -> bytes:
        return bytes([crtp_header(CRTP_PORT_LINK, LINK_SINK)]) + struct.pack(ECHO_FMT, seq, t)

    # ---------------- downlink ----------------
    def on_crtpdown(self, _c: BleakGATTCharacteristic, data: bytearray) -> None:
        self.rx_frags += 1
        self.t_last_rx = time.perf_counter()
        pkt = self.down.feed(bytes(data))
        if pkt:
            self.handle_packet(pkt)

    def handle_packet(self, pkt: bytes) -> None:
        port, ch = pkt[0] >> 4, pkt[0] & 3
        self.rx_by_port[port] = self.rx_by_port.get(port, 0) + 1
        if port == CRTP_PORT_LINK and ch == LINK_ECHO:
            seq = struct.unpack_from("<I", pkt, 1)[0] if len(pkt) >= 5 else None
            t0 = self.pending.pop(seq, None) if seq is not None else None
            if t0 is None:
                self.stale_echo.append(pkt)
                return
            rtt = time.perf_counter() - t0
            self.rtts.append(rtt)
            fut = self.waiters.pop(seq, None)
            if fut and not fut.done():
                fut.set_result(rtt)
        elif port == CRTP_PORT_CONSOLE:
            self.console += pkt[1:].rstrip(b"\x00")
            while b"\n" in self.console:
                line, _, self.console = self.console.partition(b"\n")
                text = line.decode(errors="replace").rstrip()
                self.console_lines.append(text)
                if self.verbose:
                    print(f"    [console] {text}")
        elif self.verbose:
            print(f"    [rx port {port} ch {ch}] {pkt.hex()}")

    def reset_probe_stats(self) -> None:
        self.pending.clear()
        self.rtts.clear()
        self.stale_echo.clear()
        self.waiters.clear()

    # ---------------- phases ----------------
    async def drain(self, max_s: float, quiet_s: float) -> dict:
        print(f"\n== Drain: pumping nulls at {self.args.pump_hz:g} Hz until {quiet_s:g} s of silence (max {max_s:g} s)")
        self.reset_probe_stats()
        n_ports_before = dict(self.rx_by_port)
        t0 = time.perf_counter()
        self.t_last_rx = t0
        self.pump_hz = max(self.args.pump_hz, 20.0)
        self.pump_enabled = True
        while time.perf_counter() - t0 < max_s and not self.disconnected.is_set():
            await asyncio.sleep(0.05)
            if time.perf_counter() - self.t_last_rx > quiet_s and time.perf_counter() - t0 > 1.0:
                break
        elapsed = time.perf_counter() - t0
        self.pump_hz = self.args.pump_hz
        got = {p: n - n_ports_before.get(p, 0) for p, n in self.rx_by_port.items()}
        res = {"elapsed_s": elapsed, "packets_by_port": {f"port_{p}": n for p, n in sorted(got.items()) if n},
               "stale_echo_replies": len(self.stale_echo), "console_lines": len(self.console_lines)}
        print(f"   {elapsed:.1f} s, nulls sent {self.nulls_sent}, packets by port {res['packets_by_port']}, "
              f"stale echo replies {len(self.stale_echo)}")
        for line in self.console_lines[-6:]:
            print(f"   console: {line}")
        return res

    async def crtpup_probe(self, settle_s: float) -> dict:
        """Find out whether / how the fragmented CRTPUP route reaches the STM32."""
        print("\n== CRTPUP probe: 1 short (13 B) + 1 long (25 B, two fragments) echo per variant, pump on")
        variants = [("len-1 + pid-only continuation (doc)", "len-1", "pid"),
                    ("len   + pid-only continuation", "len", "pid"),
                    ("len-1 + repeated start byte", "len-1", "repeat"),
                    ("len   + repeated start byte", "len", "repeat"),
                    ("len-1 + EMPTY continuation after a short pkt", "len-1", "empty")]
        results = {}
        for i, (name, len_mode, cont_mode) in enumerate(variants):
            self.reset_probe_stats()
            base = 100000 + i * 10
            errors = []
            for j, pad in ((0, 0), (1, 12)):
                seq = base + j
                t = time.perf_counter()
                self.pending[seq] = t
                try:
                    await self.send(self.echo_packet(seq, t, pad), route="crtpup", len_mode=len_mode, cont_mode=cont_mode)
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))
                    self.pending.pop(seq, None)
                await asyncio.sleep(settle_s)
            short_ok = base not in self.pending and not errors
            long_ok = (base + 1) not in self.pending and not errors
            results[name] = {"len_mode": len_mode, "cont_mode": cont_mode, "short_ok": short_ok, "long_ok": long_ok,
                             "errors": errors}
            print(f"   {name:40s} short {'OK ' if short_ok else 'no '}  long {'OK ' if long_ok else 'no '}"
                  + (f"  errors {errors}" if errors else ""))
        working = [v for v in results.values() if v["long_ok"]]
        print("   -> fragmented uplink works with: " + (", ".join(f"{v['len_mode']}/{v['cont_mode']}" for v in working)
                                                        if working else "NOTHING (packets > 20 B cannot be sent)"))
        return results

    async def latency_test(self, count: int, mode: str, rate_hz: float, reply_timeout: float) -> dict:
        self.reset_probe_stats()
        print(f"\n== Latency: {count} echo packets via CRTP char, mode={mode}"
              + (f" @ {rate_hz:g} Hz" if mode == "paced" else "") + f", pump {self.args.pump_hz:g} Hz")
        errors = 0
        last_err = ""
        t_start = time.perf_counter()
        if mode == "pingpong":
            for seq in range(count):
                if self.disconnected.is_set():
                    break
                fut: asyncio.Future = asyncio.get_running_loop().create_future()
                self.waiters[seq] = fut
                t = time.perf_counter()
                self.pending[seq] = t
                try:
                    await self.send(self.echo_packet(seq, t))
                    await asyncio.wait_for(fut, reply_timeout)
                except asyncio.TimeoutError:
                    self.pending.pop(seq, None)
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    last_err = repr(e)
                    self.pending.pop(seq, None)
                finally:
                    self.waiters.pop(seq, None)
        else:
            interval = 1.0 / rate_hz
            next_t = time.perf_counter()
            for seq in range(count):
                if self.disconnected.is_set():
                    break
                now = time.perf_counter()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                t = time.perf_counter()
                self.pending[seq] = t
                try:
                    await self.send(self.echo_packet(seq, t))
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    last_err = repr(e)
                    self.pending.pop(seq, None)
                next_t += interval
            await asyncio.sleep(reply_timeout)  # grace period for the last replies
        elapsed = time.perf_counter() - t_start
        sent = count - errors
        lost = len(self.pending)
        res = {
            "mode": mode, "count": count, "sent": sent, "replies": len(self.rtts), "lost": lost,
            "loss_pct": 100.0 * lost / max(1, sent), "write_errors": errors, "last_error": last_err,
            "rtt_min_s": min(self.rtts) if self.rtts else float("nan"),
            "rtt_median_s": statistics.median(self.rtts) if self.rtts else float("nan"),
            "rtt_p95_s": pct(self.rtts, 95),
            "rtt_max_s": max(self.rtts) if self.rtts else float("nan"),
            "elapsed_s": elapsed, "disconnected": self.disconnected.is_set(),
            "lost_seqs": sorted(self.pending)[:60],
            "write_with_response": self.args.with_response, "pump_hz": self.args.pump_hz,
        }
        print(f"   sent {sent}  replies {len(self.rtts)}  lost {lost} ({res['loss_pct']:.1f}%)  write errors {errors}")
        if lost:
            print(f"   lost seqs (first 40): {sorted(self.pending)[:40]}")
        print(f"   RTT min {ms(res['rtt_min_s'])}  median {ms(res['rtt_median_s'])}  p95 {ms(res['rtt_p95_s'])}  max {ms(res['rtt_max_s'])}")
        if last_err:
            print(f"   last error: {last_err}")
        return res

    async def rate_test(self, hz: float, duration: float, probe_every: int, reply_timeout: float) -> dict:
        self.reset_probe_stats()
        interval = 1.0 / hz
        n_target = int(round(hz * duration))
        print(f"\n== Throughput: {n_target} x 13-byte sink packets @ {hz:g} Hz for {duration:g} s via CRTP char"
              + (f" (every {probe_every}th is an echo probe)" if probe_every else ""))
        sent = errors = late = probes = 0
        last_err = ""
        write_times: list[float] = []
        t_start = time.perf_counter()
        next_t = t_start
        for i in range(n_target):
            if self.disconnected.is_set():
                break
            now = time.perf_counter()
            if now < next_t:
                await asyncio.sleep(next_t - now)
            elif now - next_t > interval:
                late += 1
            t = time.perf_counter()
            if probe_every and i % probe_every == 0:
                pkt = self.echo_packet(i, t)
                self.pending[i] = t
                probes += 1
            else:
                pkt = self.sink_packet(i, t)
            tw = time.perf_counter()
            try:
                await self.send(pkt)
                sent += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                last_err = repr(e)
                self.pending.pop(i, None)
            write_times.append(time.perf_counter() - tw)
            next_t += interval
        elapsed = time.perf_counter() - t_start
        await asyncio.sleep(reply_timeout)
        probe_lost = len(self.pending)
        achieved = sent / elapsed if elapsed > 0 else 0.0
        res = {
            "target_hz": hz, "duration_s": duration, "n_target": n_target, "sent": sent,
            "write_errors": errors, "late_sends": late, "achieved_hz": achieved,
            "achieved_pct": 100.0 * achieved / hz, "elapsed_s": elapsed,
            "write_call_median_s": statistics.median(write_times) if write_times else float("nan"),
            "write_call_p95_s": pct(write_times, 95),
            "write_call_max_s": max(write_times) if write_times else float("nan"),
            "probes": probes, "probe_replies": len(self.rtts), "probe_lost": probe_lost,
            "probe_loss_pct": 100.0 * probe_lost / max(1, probes) if probes else float("nan"),
            "probe_rtt_median_s": statistics.median(self.rtts) if self.rtts else float("nan"),
            "probe_rtt_p95_s": pct(self.rtts, 95),
            "probe_rtt_max_s": max(self.rtts) if self.rtts else float("nan"),
            "disconnected": self.disconnected.is_set(), "last_error": last_err,
        }
        print(f"   sent {sent}/{n_target} in {elapsed:.2f} s -> {achieved:.1f} Hz ({res['achieved_pct']:.1f}% of target)"
              f"  write errors {errors}  late sends {late}  disconnected {res['disconnected']}")
        print(f"   write() call time median {ms(res['write_call_median_s'])}  p95 {ms(res['write_call_p95_s'])}  max {ms(res['write_call_max_s'])}")
        if probes:
            print(f"   probes {probes}  replies {len(self.rtts)}  lost {probe_lost} ({res['probe_loss_pct']:.1f}%)"
                  f"  RTT under load median {ms(res['probe_rtt_median_s'])}  p95 {ms(res['probe_rtt_p95_s'])}  max {ms(res['probe_rtt_max_s'])}")
        if last_err:
            print(f"   last error: {last_err}")
        return res


async def find_crazyflie(address: str | None, timeout: float):
    if address:
        print(f"Looking for {address} ...")
        return await BleakScanner.find_device_by_address(address, timeout=timeout)
    print(f'Scanning for a device named "{NAME_PREFIX}*" ({timeout:g} s) ...')
    return await BleakScanner.find_device_by_filter(
        lambda d, adv: (adv.local_name or d.name or "").startswith(NAME_PREFIX), timeout=timeout
    )


def set_windows_timer_resolution(enable: bool) -> None:
    """1 ms system timer so asyncio.sleep can pace 100 Hz (default Windows tick is 15.6 ms)."""
    if sys.platform != "win32":
        return
    try:
        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)
    except Exception:  # noqa: BLE001
        pass


async def run(args: argparse.Namespace) -> int:
    device = await find_crazyflie(args.address, args.scan_timeout)
    if device is None:
        print("FAIL: no Crazyflie found. Is it powered on? Did a Crazyradio talk to it (then power-cycle)? "
              "Is it paired in Windows settings (unpair it)?")
        return 2
    print(f"Found {device.name} at {device.address}")

    bench_holder: dict[str, Bench] = {}

    def on_disconnect(_client: BleakClient) -> None:
        b = bench_holder.get("b")
        if b is not None and not b.disconnected.is_set():
            print("!! BLE DISCONNECTED")
            b.disconnected.set()

    client = BleakClient(device, disconnected_callback=on_disconnect, timeout=args.connect_timeout)
    t0 = time.perf_counter()
    await client.connect()
    t_connect = time.perf_counter() - t0
    print(f"Connected in {t_connect:.2f} s, MTU {client.mtu_size}")

    chars = {c.uuid.lower(): c for s in client.services for c in s.characteristics}
    for name, uuid in (("CRTP", CRTP_UUID), ("CRTPUP", CRTPUP_UUID), ("CRTPDOWN", CRTPDOWN_UUID)):
        c = chars.get(uuid)
        print(f"   {name:8s} {uuid}  {'props=' + ','.join(c.properties) if c else 'MISSING'}")
    if CRTP_UUID not in chars or CRTPDOWN_UUID not in chars:
        print("FAIL: Crazyflie service characteristics not found")
        await client.disconnect()
        return 2

    bench = Bench(client, args)
    bench_holder["b"] = bench
    await client.start_notify(CRTPDOWN_UUID, bench.on_crtpdown)
    pump = asyncio.create_task(bench.pump_task())

    results: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": {"name": device.name, "address": device.address},
        "connect_s": t_connect, "mtu": client.mtu_size, "pump_hz": args.pump_hz,
        "python": sys.version.split()[0],
    }
    try:
        results["drain"] = await bench.drain(args.drain_max, args.drain_quiet)
        if not args.skip_crtpup:
            results["crtpup"] = await bench.crtpup_probe(args.reply_timeout)
        if not args.skip_latency:
            results["latency"] = await bench.latency_test(args.echo_count, args.echo_mode, args.echo_rate, args.reply_timeout)
        if not args.skip_throughput:
            results["throughput"] = []
            for hz in args.rates:
                if bench.disconnected.is_set():
                    print("   skipping remaining rates: disconnected")
                    break
                results["throughput"].append(await bench.rate_test(hz, args.duration, args.probe_every, args.reply_timeout))
    finally:
        bench.pump_enabled = False
        pump.cancel()
        results["rx_by_port"] = {f"port_{p}": n for p, n in sorted(bench.rx_by_port.items())}
        results["rx_fragments"] = bench.rx_frags
        results["downlink_length_convention"] = bench.down.convention
        results["orphan_fragments"] = bench.down.orphans
        results["dropped_partial_packets"] = bench.down.dropped
        results["nulls_sent"] = bench.nulls_sent
        results["pump_errors"] = bench.pump_errors
        results["console_lines"] = bench.console_lines
        results["firmware_console_version_lines"] = [l for l in bench.console_lines if "version" in l.lower()]
        if client.is_connected:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # ---------------- verdict ----------------
    lat = results.get("latency")
    lat_ok = bool(lat) and lat["rtt_p95_s"] < 0.100 and lat["loss_pct"] < 5.0
    r30 = next((r for r in results.get("throughput", []) if abs(r["target_hz"] - 30) < 1e-6), None)
    r30_ok = (
        r30 is not None
        and r30["achieved_pct"] >= 95.0
        and 100.0 * r30["write_errors"] / max(1, r30["n_target"]) < 5.0
        and not r30["disconnected"]
        and (r30["probes"] == 0 or r30["probe_loss_pct"] < 5.0)
    )
    verdict = "PASS" if (lat_ok and r30_ok) else "FAIL"
    results["verdict"] = verdict
    print("\n================ SUMMARY ================")
    print(f"downlink packets by CRTP port: {results['rx_by_port']}  (fragments {bench.rx_frags}, "
          f"orphans {bench.down.orphans}, dropped partials {bench.down.dropped}, "
          f"length convention {bench.down.convention}, nulls sent {bench.nulls_sent})")
    for l in results["firmware_console_version_lines"]:
        print(f"firmware: {l}")
    if "crtpup" in results:
        ok = [k for k, v in results["crtpup"].items() if v["long_ok"]]
        print("CRTPUP  : fragmented uplink " + (f"works with {ok}" if ok else "DOES NOT WORK -> packets > 20 B impossible"))
    if lat:
        print(f"latency : p95 {ms(lat['rtt_p95_s'])} (< 100 ms needed), loss {lat['loss_pct']:.1f}% (< 5% needed) -> {'ok' if lat_ok else 'NOT ok'}")
    else:
        print("latency : skipped")
    if r30:
        print(f"30 Hz   : achieved {r30['achieved_hz']:.1f} Hz ({r30['achieved_pct']:.1f}%), errors {r30['write_errors']}, "
              f"probe loss {r30['probe_loss_pct']:.1f}%, RTT p95 {ms(r30['probe_rtt_p95_s'])}, disconnected {r30['disconnected']} -> {'ok' if r30_ok else 'NOT ok'}")
    else:
        print("30 Hz   : not run")
    for r in results.get("throughput", []):
        if r is not r30:
            print(f"{r['target_hz']:g} Hz  : achieved {r['achieved_hz']:.1f} Hz ({r['achieved_pct']:.1f}%), errors {r['write_errors']}, "
                  f"late {r['late_sends']}, probe loss {r['probe_loss_pct']:.1f}%, RTT p95 {ms(r['probe_rtt_p95_s'])} (informational)")
    print(f"\n>>> {verdict} <<<")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"ble_bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    return 0 if verdict == "PASS" else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", help="BLE address; default: scan for a name starting with 'Crazyflie'")
    ap.add_argument("--scan-timeout", type=float, default=15.0)
    ap.add_argument("--connect-timeout", type=float, default=20.0)
    ap.add_argument("--pump-hz", type=float, default=50.0, help="null-packet rate that keeps the downlink flowing when idle; 0 = off")
    ap.add_argument("--with-response", action="store_true", help="use write-WITH-response on the CRTP characteristic (paces to one write per connection event)")
    ap.add_argument("--drain-max", type=float, default=20.0, help="max seconds to drain the drone's downlink backlog")
    ap.add_argument("--drain-quiet", type=float, default=1.0, help="seconds of downlink silence that ends the drain")
    ap.add_argument("--echo-count", type=int, default=200)
    ap.add_argument("--echo-mode", choices=["paced", "pingpong"], default="paced")
    ap.add_argument("--echo-rate", type=float, default=20.0, help="echo send rate in paced mode (Hz)")
    ap.add_argument("--rates", type=float, nargs="+", default=[30, 60, 100])
    ap.add_argument("--duration", type=float, default=10.0, help="seconds per throughput rate")
    ap.add_argument("--probe-every", type=int, default=10, help="every Nth throughput packet is an echo probe; 0 = none")
    ap.add_argument("--reply-timeout", type=float, default=1.0, help="grace period to wait for late echo replies (s)")
    ap.add_argument("--skip-crtpup", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print console text and unknown packets")
    args = ap.parse_args()

    set_windows_timer_resolution(True)
    try:
        code = asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        code = 130
    finally:
        set_windows_timer_resolution(False)
    sys.exit(code)


if __name__ == "__main__":
    main()
