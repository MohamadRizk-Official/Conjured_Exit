"""Loopback integrity test for the BLE link: do uplink packets arrive intact?

Sends numbered packets to the Crazyflie's echo channel (CRTP port 15, channel 0) and checks that
every echo carries exactly the payload that was sent. No motors are involved. One connection,
several phases:

    stream             port 15 written WITHOUT response (how setpoints on ports 3/7 went until 2026-10-04)
    reliable           port 15 written WITH response
    reliable+extpos    as above, plus extpos on port 6 at --extpos-hz (the camera feed's load)
    stream+extpos      fire-and-forget echo plus the extpos load

Each phase may carry its own rates: ``reliable@15+extpos@10`` = echo with response at 15 Hz while
extpos runs at 10 Hz. ``--pump-hz`` sets the driver's idle null-packet rate after connect.

Measured 2026-10-04 (nRF 2024.10, Windows 11, 15 ms interval, pump 100 Hz):
    stream 20 Hz alone              4/200 corrupted          (first bytes overwritten with FF 00 FF)
    stream 20 Hz + extpos 30 Hz   183/200 corrupted          <- the flight configuration until then
    reliable 20 Hz alone            0/200, median 83 ms
    reliable 20 Hz + extpos 30 Hz   0/200 but median 1.3 s   (50 acknowledged writes/s is too many)

A CORRUPTED echo (payload differs from what was sent) proves the uplink mangles packets in that
mode. Missing echoes can be uplink or downlink loss (the nRF drops notifications on bursts), so
only corruption is conclusive by itself.

    python tools/ble_echo_test.py --hz 20 --n 200 --extpos-hz 30
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flight  # noqa: E402

ECHO_PORT = 15
ECHO_CHANNEL = 0
PAYLOAD_LEN = 16   # 1 header + 16 = 17 bytes, under the 20-byte limit


def payload_for(seq: int) -> bytes:
    """Deterministic 16-byte payload: seq (2 bytes) + a pattern that depends on seq."""
    body = bytes(((seq * 7 + i * 13) & 0xFF) for i in range(PAYLOAD_LEN - 2))
    return struct.pack("<H", seq & 0xFFFF) + body


class EchoPhase:
    def __init__(self, cf, name: str, hz: float, n: int, extpos_hz: float = 0.0, settle: float = 2.0):
        self.cf, self.name, self.hz, self.n, self.extpos_hz, self.settle = cf, name, hz, n, extpos_hz, settle
        self.lock = threading.Lock()
        self.sent_at: dict[int, float] = {}
        self.echoed: dict[int, float] = {}
        self.corrupted: list[tuple[int, bytes]] = []
        self.unknown = 0
        self.latencies: list[float] = []
        self.extpos_sent = 0

    def on_echo(self, pk) -> None:
        if pk.channel != ECHO_CHANNEL:
            return
        data = bytes(pk.data)
        now = time.perf_counter()
        with self.lock:
            if len(data) < 2:
                self.corrupted.append((-1, data))
                return
            seq = struct.unpack("<H", data[:2])[0]
            if seq not in self.sent_at:
                self.unknown += 1
                return
            if data != payload_for(seq):
                self.corrupted.append((seq, data))
            elif seq not in self.echoed:
                self.echoed[seq] = now
                self.latencies.append(now - self.sent_at[seq])

    def run(self) -> dict:
        from cflib.crtp.crtpstack import CRTPPacket

        stop = threading.Event()

        def extpos_load() -> None:
            dt = 1.0 / self.extpos_hz
            nxt = time.perf_counter()
            while not stop.is_set():
                try:
                    self.cf.extpos.send_extpos(0.0, 0.0, 0.0)
                    self.extpos_sent += 1
                except Exception as exc:  # noqa: BLE001
                    print("extpos send failed:", exc)
                nxt += dt
                time.sleep(max(0.0, nxt - time.perf_counter()))

        self.cf.add_port_callback(ECHO_PORT, self.on_echo)
        if self.extpos_hz > 0:
            threading.Thread(target=extpos_load, daemon=True).start()
            time.sleep(1.0)
        dt = 1.0 / self.hz
        t0 = time.perf_counter()
        nxt = t0
        for seq in range(self.n):
            pk = CRTPPacket()
            pk.set_header(ECHO_PORT, ECHO_CHANNEL)
            pk.data = payload_for(seq)
            with self.lock:
                self.sent_at[seq] = time.perf_counter()
            self.cf.send_packet(pk)
            nxt += dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
        elapsed = time.perf_counter() - t0
        time.sleep(self.settle)
        stop.set()
        self.cf.remove_port_callback(ECHO_PORT, self.on_echo)
        with self.lock:
            bad = {c[0] for c in self.corrupted}
            missing = [s for s in range(self.n) if s not in self.echoed and s not in bad]
            lat = sorted(self.latencies)
        res = {
            "phase": self.name, "sent": self.n, "actual_hz": self.n / elapsed, "intact": len(self.echoed),
            "corrupted": len(self.corrupted), "missing": len(missing), "unknown": self.unknown,
            "extpos_sent": self.extpos_sent,
            "lat_med_ms": 1000 * lat[len(lat) // 2] if lat else float("nan"),
            "lat_p90_ms": 1000 * lat[int(len(lat) * 0.9)] if lat else float("nan"),
            "lat_max_ms": 1000 * lat[-1] if lat else float("nan"),
            "examples": [(s, d.hex(), payload_for(s).hex() if s >= 0 else "?") for s, d in self.corrupted[:5]],
            "missing_seqs": missing[:15],
        }
        return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--hz", type=float, default=20.0, help="echo packet rate (setpoints go at 20 Hz)")
    ap.add_argument("--n", type=int, default=200, help="echo packets per phase")
    ap.add_argument("--extpos-hz", type=float, default=30.0, help="extpos load rate for the +extpos phases")
    ap.add_argument("--phases", default="stream,reliable,reliable+extpos,stream+extpos")
    ap.add_argument("--pump-hz", type=float, default=None,
                    help="driver idle null-packet rate after connect (default: config.BLE_PUMP_HZ)")
    args = ap.parse_args()

    cf = flight.connect(args.uri)
    link = cf.link
    # cflib's link-quality pinger also listens on port 15 and chokes on our payloads: unhook it.
    try:
        cf.incoming.cb = [c for c in cf.incoming.cb
                          if getattr(getattr(c, "callback", None), "__name__", "") != "_ping_response"]
    except Exception:  # noqa: BLE001
        pass
    if args.pump_hz is not None:
        link.pump_hz = float(args.pump_hz)      # the sender re-reads it every packet
    base_ports = set(getattr(link, "stream_ports", ()))
    print(f"connected; driver stream ports = {sorted(base_ports)}; pump {getattr(link, 'pump_hz', '?')} Hz")
    time.sleep(1.0)
    results = []
    for name in [p.strip() for p in args.phases.split(",") if p.strip()]:
        main_part, _, extpos_part = name.partition("+")
        mode, _, hz_txt = main_part.partition("@")
        streaming = mode.startswith("stream")
        hz = float(hz_txt) if hz_txt else args.hz
        extpos_hz = 0.0
        if extpos_part:
            _, _, ehz = extpos_part.partition("@")
            extpos_hz = float(ehz) if ehz else args.extpos_hz
        link.stream_ports = (base_ports | {ECHO_PORT}) if streaming else (base_ports - {ECHO_PORT})
        print(f"\n--- phase {name}: echo {'WITHOUT' if streaming else 'WITH'} response at {hz:g} Hz x {args.n}"
              f"{f', extpos {extpos_hz:g} Hz' if extpos_hz else ''}")
        r = EchoPhase(cf, name, hz, args.n, extpos_hz).run()
        results.append(r)
        print(f"    intact {r['intact']}/{r['sent']}  CORRUPTED {r['corrupted']}  missing {r['missing']}  "
              f"unknown {r['unknown']}  actual {r['actual_hz']:.1f} Hz  extpos sent {r['extpos_sent']}")
        print(f"    latency ms: median {r['lat_med_ms']:.0f}  p90 {r['lat_p90_ms']:.0f}  max {r['lat_max_ms']:.0f}")
        for s, got, want in r["examples"]:
            print(f"    corrupt seq {s}: got {got}  want {want}")
        if r["missing_seqs"]:
            print(f"    missing seqs: {r['missing_seqs']}")
        time.sleep(1.0)
    status = getattr(link, "get_status", None)
    if callable(status):
        print("\nlink:", status())
    print("\nSUMMARY")
    for r in results:
        print(f"  {r['phase']:17} intact {r['intact']:4}/{r['sent']}  corrupted {r['corrupted']:3}  missing {r['missing']:3}"
              f"  median {r['lat_med_ms']:.0f} ms  max {r['lat_max_ms']:.0f} ms")
    cf.close_link()
    return 2 if any(r["corrupted"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
