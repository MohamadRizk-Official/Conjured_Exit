"""Read the live BLE connection interval (Windows 11 WinRT) and try to shorten it.

Phase 1: for 12 s print the connection parameters Windows reports and the time one
         write-with-response takes (the drone requests a connection-parameter update ~5 s
         after connecting, so the interval may change here).
Phase 2: request BluetoothLEPreferredConnectionParameters.throughput_optimized (and, if that
         is refused, an explicit 7.5-15 ms range), keep the request object alive, and measure
         again for 12 s.
Link-layer null packets only; the motors cannot start.
"""
import asyncio
import statistics
import sys
import time

sys.path.insert(0, r"C:\Users\Smshr\code\pathcaster")
from bleak import BleakClient, BleakScanner
from ble_bench import CRTP_UUID

ADDR = sys.argv[1] if len(sys.argv) > 1 else "DB:04:E8:22:6F:CC"
NULL = bytes([0xFF])


def describe(dev) -> str:
    try:
        p = dev.get_connection_parameters()
        return (f"interval {p.connection_interval * 1.25:.2f} ms, latency {p.connection_latency}, "
                f"timeout {p.link_timeout * 10} ms")
    except Exception as e:  # noqa: BLE001
        return f"(get_connection_parameters failed: {e!r})"


async def measure(client, dev, seconds: float, label: str) -> None:
    t_end = time.perf_counter() + seconds
    durs = []
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        await client.write_gatt_char(CRTP_UUID, NULL, response=True)
        d = time.perf_counter() - t
        durs.append(d)
        print(f"   [{label}] t={seconds - (t_end - time.perf_counter()):5.1f}s  write+response {d * 1000:6.1f} ms   {describe(dev)}")
        await asyncio.sleep(0.5)
    print(f"   [{label}] write+response median {statistics.median(durs) * 1000:.1f} ms, min {min(durs) * 1000:.1f}, max {max(durs) * 1000:.1f}")


async def main() -> None:
    from winrt.windows.devices.bluetooth import BluetoothLEPreferredConnectionParameters as Pref

    device = await BleakScanner.find_device_by_address(ADDR, timeout=15)
    if device is None:
        print("not found")
        return
    async with BleakClient(device, timeout=20) as client:
        dev = client._backend._requester  # WinRT BluetoothLEDevice behind bleak
        print("connected; initial:", describe(dev))
        await measure(client, dev, 12.0, "before")

        presets = [n for n in ("throughput_optimized", "balanced", "power_optimized") if hasattr(Pref, n)]
        print("available presets:", presets)
        keep = []
        try:
            pref = Pref.throughput_optimized
            print(f"requesting throughput_optimized: interval {pref.min_connection_interval * 1.25:.2f}-"
                  f"{pref.max_connection_interval * 1.25:.2f} ms, latency {pref.connection_latency}, "
                  f"timeout {pref.link_timeout * 10} ms")
            req = dev.request_preferred_connection_parameters(pref)
            keep.append(req)
            print("   request status:", req.status)
        except Exception as e:  # noqa: BLE001
            print("   request failed:", repr(e))
        await asyncio.sleep(2.0)
        print("after request:", describe(dev))
        await measure(client, dev, 12.0, "after")
        print("final:", describe(dev))
        del keep


asyncio.run(main())
