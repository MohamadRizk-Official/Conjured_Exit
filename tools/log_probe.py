"""Measure raw log-packet sizes over BLE for 7- and 6-variable FP16 blocks (motors off)."""
import logging
import sys
import time

sys.path.insert(0, r"C:\Users\Smshr\code\pathcaster")
import flight
from toc_names import resolve
from cflib.crazyflie.log import LogConfig

logging.basicConfig(level=logging.WARNING)
cf = flight.connect()
sizes = []
cf.add_port_callback(5, lambda pk: sizes.append((pk.channel, len(pk.data))))
toc = cf.log.toc


def run_block(names, seconds=2.5):
    conf = LogConfig(name="probe", period_in_ms=200)
    for n in names:
        conf.add_variable(resolve(toc, n), "FP16")
    got = []
    conf.data_received_cb.add_callback(lambda ts, data, lc: got.append(dict(data)))
    conf.error_cb.add_callback(lambda lc, msg: print("   block error:", msg))
    sizes.clear()
    cf.log.add_config(conf)
    conf.start()
    time.sleep(seconds)
    conf.stop()
    time.sleep(0.3)
    data_sizes = sorted({s for ch, s in sizes if ch == 2})
    print(f"{len(names)} vars ({2 * len(names)} data bytes, expected CRTP packet {5 + 2 * len(names)} B): "
          f"raw port-5 data channel payload sizes seen {data_sizes} -> packet sizes {[s + 1 for s in data_sizes]}; "
          f"decoded OK {len(got)} packets")
    if got:
        last = got[-1]
        print("   last decoded:", {k.replace(chr(0), '?'): round(v, 4) for k, v in last.items()})
    return got


try:
    seven = ["pm.vbat", "kalman.stateX", "kalman.stateY", "kalman.stateZ", "kalman.varPX", "kalman.varPY", "kalman.varPZ"]
    run_block(seven)
    run_block(seven[:6])
    run_block(["pm.vbat", "kalman.varPX", "kalman.varPY", "kalman.varPZ"])
    run_block(["stabilizer.roll", "stabilizer.pitch"])
finally:
    cf.close_link()
