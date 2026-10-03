"""BLE diagnostic 2: firmware version, downlink pumping with null packets, source-channel text test,
CRTPUP length-convention variants. Link-layer packets only (port 15), motors cannot start."""
import asyncio
import sys
import time

sys.path.insert(0, r"C:\Users\Smshr\code\pathcaster")
from bleak import BleakClient, BleakScanner
from ble_bench import CRTP_PORT_LINK, CRTP_UUID, CRTPDOWN_UUID, CRTPUP_UUID, LINK_ECHO, LINK_SOURCE, crtp_header

ADDR = sys.argv[1] if len(sys.argv) > 1 else "DB:04:E8:22:6F:CC"
NULL = bytes([0xFF])
DIS = {"2a29": "manufacturer", "2a24": "model", "2a25": "serial", "2a26": "firmware_rev",
       "2a27": "hardware_rev", "2a28": "software_rev"}
rx = []
t_start = time.perf_counter()


def show(data: bytes) -> str:
    ctrl = data[0]
    txt = "".join(chr(b) if 32 <= b < 127 else "." for b in data[1:])
    return (f"ctrl=0x{ctrl:02x} start={ctrl >> 7} pid={(ctrl >> 5) & 3} lenfield={ctrl & 0x1f} "
            f"data[{len(data) - 1}]={data[1:].hex()} '{txt}'")


def cb(tag):
    def _cb(_c, data):
        rx.append((time.perf_counter(), tag, bytes(data)))
        print(f"      <- {time.perf_counter() - t_start:7.3f}s {tag} {show(bytes(data))}")
    return _cb


async def w(client, uuid, data, resp=False, pause=0.35):
    try:
        await client.write_gatt_char(uuid, data, response=resp)
        print(f"      -> {time.perf_counter() - t_start:7.3f}s wrote {data.hex()} to {uuid[4:8]}")
    except Exception as e:  # noqa: BLE001
        print(f"      -> write {data.hex()} to {uuid[4:8]} ERROR {e!r}")
    await asyncio.sleep(pause)


async def main():
    dev = await BleakScanner.find_device_by_address(ADDR, timeout=15)
    if not dev:
        print("not found")
        return
    async with BleakClient(dev, timeout=20) as client:
        print("connected, mtu", client.mtu_size)
        print("== Device Information Service")
        for s in client.services:
            for c in s.characteristics:
                key = c.uuid[4:8]
                if key in DIS:
                    try:
                        v = await client.read_gatt_char(c)
                        print(f"   {DIS[key]:14s} = {bytes(v)!r}")
                    except Exception as e:  # noqa: BLE001
                        print(f"   {DIS[key]:14s} read error {e!r}")
        await client.start_notify(CRTPDOWN_UUID, cb("DOWN"))
        await client.start_notify(CRTP_UUID, cb("CRTP"))
        await asyncio.sleep(0.5)

        print("\n== 1: flush queued downlink with 4 null packets (0xFF via 0202 write-without-response)")
        for _ in range(4):
            await w(client, CRTP_UUID, NULL)

        print("\n== 2: SOURCE request via 0202 (reply must be the text 'Bitcraze Crazyflie'), then 2 nulls")
        await w(client, CRTP_UUID, bytes([crtp_header(CRTP_PORT_LINK, LINK_SOURCE)]))
        await w(client, CRTP_UUID, NULL)
        await w(client, CRTP_UUID, NULL)

        print("\n== 3: ECHO 'ABCD' via 0202, then 2 nulls")
        await w(client, CRTP_UUID, bytes([crtp_header(CRTP_PORT_LINK, LINK_ECHO)]) + b"ABCD")
        await w(client, CRTP_UUID, NULL)
        await w(client, CRTP_UUID, NULL)

        print("\n== 4: ECHO via CRTPUP variants, each followed by 1 null on 0202 (a 5-byte reply = that variant works)")
        echo = bytes([crtp_header(CRTP_PORT_LINK, LINK_ECHO)]) + b"WXYZ"
        variants = {
            "len-1 (doc)": bytes([0x80 | (1 << 5) | (len(echo) - 1)]) + echo,
            "len": bytes([0x80 | (2 << 5) | len(echo)]) + echo,
            "len+1": bytes([0x80 | (3 << 5) | (len(echo) + 1)]) + echo,
            "no start bit, len-1": bytes([(0 << 5) | (len(echo) - 1)]) + echo,
            "write WITH response, len-1": None,
        }
        for name, frag in variants.items():
            print(f"   -- variant: {name}")
            if frag is None:
                await w(client, CRTPUP_UUID, bytes([0x80 | (1 << 5) | (len(echo) - 1)]) + echo, resp=True)
            else:
                await w(client, CRTPUP_UUID, frag)
            await w(client, CRTP_UUID, NULL)

        print("\n== 5: pump with nulls every 0.25 s for 5 s to see what the drone sends unprompted")
        n0 = len(rx)
        for _ in range(20):
            await w(client, CRTP_UUID, NULL, pause=0.25)
        print(f"   {len(rx) - n0} fragments during pump")
        print(f"\nTOTAL fragments: {len(rx)}")


asyncio.run(main())
