"""Measure the BLE connection interval (write-with-response round trip) and how many back-to-back
uplink writes per event the drone can answer without dropping downlink. Link-layer packets only."""
import asyncio
import statistics
import struct
import sys
import time

sys.path.insert(0, r"C:\Users\Smshr\code\pathcaster")
from bleak import BleakClient, BleakScanner
from ble_bench import CRTP_PORT_LINK, CRTP_UUID, CRTPDOWN_UUID, LINK_ECHO, CrtpDownReassembler, crtp_header

ADDR = sys.argv[1] if len(sys.argv) > 1 else "DB:04:E8:22:6F:CC"
NULL = bytes([0xFF])
got: dict[int, float] = {}
down = CrtpDownReassembler()


def on_down(_c, data):
    pkt = down.feed(bytes(data))
    if pkt and (pkt[0] >> 4) == CRTP_PORT_LINK and (pkt[0] & 3) == LINK_ECHO and len(pkt) >= 5:
        got[struct.unpack_from("<I", pkt, 1)[0]] = time.perf_counter()


def echo(seq):
    return bytes([crtp_header(CRTP_PORT_LINK, LINK_ECHO)]) + struct.pack("<I", seq) + bytes(8)


async def main():
    dev = await BleakScanner.find_device_by_address(ADDR, timeout=15)
    if not dev:
        print("not found"); return
    async with BleakClient(dev, timeout=20) as client:
        print("connected, mtu", client.mtu_size)
        await client.start_notify(CRTPDOWN_UUID, on_down)

        print("\n== 1: write-WITH-response round trip (= connection interval estimate), 40 nulls")
        durs = []
        for _ in range(40):
            t = time.perf_counter()
            await client.write_gatt_char(CRTP_UUID, NULL, response=True)
            durs.append(time.perf_counter() - t)
        print(f"   write+response: min {min(durs)*1000:.1f}  median {statistics.median(durs)*1000:.1f}  "
              f"p95 {sorted(durs)[int(len(durs)*0.95)]*1000:.1f}  max {max(durs)*1000:.1f} ms")

        print("\n== 2: write-WITHOUT-response call time, 40 nulls back-to-back (Windows queue acceptance)")
        durs = []
        for _ in range(40):
            t = time.perf_counter()
            await client.write_gatt_char(CRTP_UUID, NULL, response=False)
            durs.append(time.perf_counter() - t)
        print(f"   wwr call: min {min(durs)*1000:.1f}  median {statistics.median(durs)*1000:.1f}  max {max(durs)*1000:.1f} ms")
        await asyncio.sleep(0.5)

        print("\n== 3: RTT with everything write-with-response: echo, then null (releases the reply)")
        rtts = []
        seq = 1000
        for _ in range(20):
            seq += 1
            t = time.perf_counter()
            await client.write_gatt_char(CRTP_UUID, echo(seq), response=True)
            await client.write_gatt_char(CRTP_UUID, NULL, response=True)
            for _ in range(50):
                if seq in got:
                    break
                await asyncio.sleep(0.005)
            if seq in got:
                rtts.append(got[seq] - t)
        print(f"   {len(rtts)}/20 replies, RTT median {statistics.median(rtts)*1000:.1f} ms, "
              f"max {max(rtts)*1000:.1f} ms" if rtts else "   no replies")

        print("\n== 4: burst test: K echo writes WITHOUT response back-to-back, then with-response nulls to flush")
        print("   (K-1 replies must leave the nRF in one connection event; shows how many it can carry)")
        seq = 5000
        for k in (1, 2, 3, 4, 5, 6, 8, 12):
            delivered = total = 0
            for _rep in range(4):
                seqs = []
                for _ in range(k):
                    seq += 1
                    seqs.append(seq)
                    await client.write_gatt_char(CRTP_UUID, echo(seq), response=False)
                await asyncio.sleep(0.25)
                for _ in range(4):
                    await client.write_gatt_char(CRTP_UUID, NULL, response=True)
                await asyncio.sleep(0.25)
                delivered += sum(1 for s in seqs if s in got)
                total += k
            print(f"   burst K={k:2d}: {delivered}/{total} replies ({100*delivered/total:.0f}%)")

        print("\n== 5: steady write-WITHOUT-response echo at 15 Hz (one per ~connection interval?) 60 packets")
        seqs = []
        for i in range(60):
            seq += 1
            seqs.append(seq)
            await client.write_gatt_char(CRTP_UUID, echo(seq), response=False)
            await asyncio.sleep(1 / 15)
        for _ in range(4):
            await client.write_gatt_char(CRTP_UUID, NULL, response=True)
        await asyncio.sleep(0.3)
        d = sum(1 for s in seqs if s in got)
        print(f"   15 Hz wwr: {d}/60 replies ({100*d/60:.0f}%)")


asyncio.run(main())
