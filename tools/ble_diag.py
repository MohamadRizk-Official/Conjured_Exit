"""Raw BLE diagnostic: subscribe to BOTH notify characteristics, try every uplink route,
print every notification as hex. Link-layer packets only (port 15), motors cannot start."""
import asyncio
import sys
import time

sys.path.insert(0, r"C:\Users\Smshr\code\pathcaster")
from bleak import BleakClient, BleakScanner
from ble_bench import (CRTP_PORT_LINK, CRTP_UUID, CRTPDOWN_UUID, CRTPUP_UUID, LINK_ECHO,
                       LINK_SOURCE, crtp_header, crtpup_fragments)

ADDR = sys.argv[1] if len(sys.argv) > 1 else "DB:04:E8:22:6F:CC"
rx = []


def cb(tag):
    def _cb(_c, data):
        rx.append((time.perf_counter(), tag, bytes(data)))
        print(f"      <- {tag} {bytes(data).hex()}")
    return _cb


async def step(client, name, writes):
    print(f"\n== {name}")
    n0 = len(rx)
    for uuid, data, resp in writes:
        try:
            await client.write_gatt_char(uuid, data, response=resp)
            print(f"      -> wrote {data.hex()} to {uuid[4:8]} response={resp}")
        except Exception as e:  # noqa: BLE001
            print("      write error:", repr(e))
        await asyncio.sleep(0.3)
    await asyncio.sleep(1.2)
    print(f"   {len(rx) - n0} notification(s)")


async def main():
    dev = await BleakScanner.find_device_by_address(ADDR, timeout=15)
    if not dev:
        print("not found")
        return
    async with BleakClient(dev, timeout=20) as client:
        print("connected, mtu", client.mtu_size)
        for s in client.services:
            if "0201-1c7f" in s.uuid:
                for c in s.characteristics:
                    print(f"   char {c.uuid[4:8]} props={c.properties} descriptors={[d.uuid[4:8] for d in c.descriptors]}")
        await client.start_notify(CRTPDOWN_UUID, cb("CRTPDOWN"))
        await client.start_notify(CRTP_UUID, cb("CRTP    "))
        print("notifications enabled on both; idle 2 s")
        await asyncio.sleep(2.0)
        echo = bytes([crtp_header(CRTP_PORT_LINK, LINK_ECHO)]) + b"\x11\x22\x33\x44"
        src = bytes([crtp_header(CRTP_PORT_LINK, LINK_SOURCE)])
        await step(client, "A: echo via CRTPUP write-without-response, pid 0 then pid 1",
                   [(CRTPUP_UUID, crtpup_fragments(echo, 0)[0], False),
                    (CRTPUP_UUID, crtpup_fragments(echo, 1)[0], False)])
        await step(client, "B: echo via CRTP char, write WITH response, x2",
                   [(CRTP_UUID, echo, True), (CRTP_UUID, echo, True)])
        await step(client, "C: echo via CRTP char, write-without-response, x2",
                   [(CRTP_UUID, echo, False), (CRTP_UUID, echo, False)])
        await step(client, "D: source request via CRTP char (with response) then via CRTPUP",
                   [(CRTP_UUID, src, True), (CRTPUP_UUID, crtpup_fragments(src, 2)[0], False)])
        await step(client, "E: CRTPUP with length field = len (not len-1), in case firmware differs",
                   [(CRTPUP_UUID, bytes([0x80 | (3 << 5) | len(echo)]) + echo, False)])
        print("\n== F: direct reads")
        for name, u in (("CRTP", CRTP_UUID), ("CRTPDOWN", CRTPDOWN_UUID)):
            try:
                v = await client.read_gatt_char(u)
                print(f"   read {name}: {bytes(v).hex()!r}")
            except Exception as e:  # noqa: BLE001
                print(f"   read {name} error: {e!r}")
        print(f"\nTOTAL notifications: {len(rx)}")


asyncio.run(main())
