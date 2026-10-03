#!/usr/bin/env python
"""
test_ble_link_hw.py -- Pathcaster task 2 HARDWARE test of the cflib BLE link driver.

Needs the real drone and a FREE BLE link (only one connection is possible: stop ble_bench.py
or any other BLE client first). Run in native Windows Python from the project root:

    cf\\Scripts\\python.exe test_ble_link_hw.py
    cf\\Scripts\\python.exe test_ble_link_hw.py --uri ble://Crazyflie-226FCC --skip-scan -v
    cf\\Scripts\\python.exe test_ble_link_hw.py --pump-hz 100 --write-with-response --max-inflight 1

Before running: unplug the USB cable, make sure the drone is NOT paired in Windows Bluetooth
settings, and power-cycle the drone if any Crazyradio has talked to it since boot.

What it does (same cflib code path as `usb://0`; only the URI differs):
  1. ble_link.register() + cflib.crtp.init_drivers(); print cflib.crtp.scan_interfaces().
  2. SyncCrazyflie(uri, cf=Crazyflie(rw_cache='cache')) -> connect + TOC download. Expect the
     FIRST connect to take ~20-25 s (param + log TOC, ~600 packets at ~30/s, one reliable
     write per connection event); TOCs are cached in ./cache so later connects are fast.
     A progress line is printed every 5 s while waiting.
  3. Wait for the parameter values, read and print `stabilizer.estimator`.
  4. Stream cf.extpos.send_extpos(0, 0, 0) at 30 Hz for 10 s, counting exceptions,
     driver-level write errors/drops and disconnects. Print rate, elapsed time, pump rate,
     null packets sent and the link status.
  5. Close cleanly and print PASS or FAIL.

Send policy under test (ble_link.py): extpos/commander ports write-without-response, everything
else (TOC, param, log) write-with-response, 0xFF null pump with response at 30 Hz when idle.

It never sends commander / motor commands. (cflib's own Crazyflie.close_link() always
sends one zero setpoint -- thrust 0 -- as its standard "stop" message; that cannot spin
the motors.)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ble_link  # noqa: E402

import cflib.crtp  # noqa: E402
from cflib.crazyflie import Crazyflie  # noqa: E402
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # noqa: E402

DEFAULT_URI = "ble://DB:04:E8:22:6F:CC"


def now() -> str:
    return time.strftime("%H:%M:%S")


def link_status(cf: Crazyflie, link=None) -> str:
    link = link or cf.link
    if link is None:
        return "link: None (closed / lost)"
    try:
        return link.get_status()
    except Exception as e:  # noqa: BLE001
        return f"link status unavailable: {e!r}"


def link_stats(cf: Crazyflie, link=None) -> dict:
    link = link or cf.link
    if isinstance(link, ble_link.BleDriver):
        return link.stats()
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=DEFAULT_URI,
                    help=f"cflib URI (default {DEFAULT_URI}); also ble:// or ble://Crazyflie-226FCC, "
                         "options may be appended as ?pump_hz=..&write_with_response=..&max_inflight_writes=..")
    ap.add_argument("--rate", type=float, default=30.0, help="extpos rate in Hz (default 30)")
    ap.add_argument("--duration", type=float, default=10.0, help="streaming time in s (default 10)")
    ap.add_argument("--pump-hz", type=float, default=None, help=f"null-packet pump rate (default {ble_link.BleDriver.pump_hz:g})")
    ap.add_argument("--write-with-response", action="store_true", help="write 0202 packets WITH response")
    ap.add_argument("--max-inflight", type=int, default=None,
                    help=f"max GATT writes per window, 0 = unlimited (default {ble_link.BleDriver.max_inflight_writes})")
    ap.add_argument("--inflight-window", type=float, default=None,
                    help=f"window in s for --max-inflight (default {ble_link.BleDriver.inflight_window_s:g})")
    ap.add_argument("--skip-scan", action="store_true", help="do not run cflib.crtp.scan_interfaces() first")
    ap.add_argument("--connect-timeout", type=float, default=180.0,
                    help="watchdog: abort if connect + TOC + param download take longer (s, default 180)")
    ap.add_argument("--cache", default="cache", help="cflib TOC cache dir (default ./cache)")
    ap.add_argument("-v", "--verbose", action="store_true", help="cflib/ble_link debug logging + drone console output")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("cflib").setLevel(logging.WARNING)

    # link options (class-wide defaults; the URI query string can still override per connection)
    opts = {}
    if args.pump_hz is not None:
        opts["pump_hz"] = args.pump_hz
    if args.write_with_response:
        opts["write_with_response"] = True
    if args.max_inflight is not None:
        opts["max_inflight_writes"] = args.max_inflight
    if args.inflight_window is not None:
        opts["inflight_window_s"] = args.inflight_window
    settings = ble_link.configure(**opts)
    print(f"[{now()}] BLE link options: " + ", ".join(f"{k}={v}" for k, v in settings.items()))

    result = {
        "connected": False, "connect_s": 0.0, "params_s": 0.0, "param_ok": False, "param_value": None,
        "sent": 0, "target": 0, "exceptions": 0, "late": 0, "elapsed_s": 0.0, "disconnected": False,
        "tx_errors": 0, "tx_dropped": 0, "nulls": 0, "status": "", "stats": {}, "error": "",
    }

    # 1. registration (no Bluetooth activity) and interface scan
    ble_link.register()
    cflib.crtp.init_drivers()
    print(f"[{now()}] cflib drivers: {[c.__name__ for c in cflib.crtp.CLASSES]}")
    if not args.skip_scan:
        print(f"[{now()}] cflib.crtp.scan_interfaces() (BLE scan ~{ble_link.BleDriver.scan_timeout:g} s; "
              f"radio/USB scans print errors if no dongle/libusb, that is expected) ...")
        try:
            found = cflib.crtp.scan_interfaces()
        except Exception as e:  # noqa: BLE001
            found = []
            print(f"   scan_interfaces raised: {e!r}")
        for entry in found:
            print(f"   {entry}")
        if not found:
            print("   (no interfaces found)")

    # 2. connect with the SAME cflib API we use on usb://0; only the URI differs
    cf = Crazyflie(rw_cache=args.cache)
    # cflib 0.1.34 pings the link-echo channel at 10 Hz after connect; over BLE that starves the
    # param download (measured: 1 param/s) and eats downlink slots. Stop it as soon as it starts.
    cf.connected.add_callback(lambda uri: cf.link_statistics.stop())
    lost = threading.Event()
    link_holder = {}

    def on_connection_lost(uri, msg):
        print(f"[{now()}] !! connection_lost: {msg}")
        result["disconnected"] = True
        lost.set()

    def on_link_established(uri):
        link_holder["link"] = cf.link
        print(f"[{now()}] link established (first packet from the drone)")

    cf.connection_lost.add_callback(on_connection_lost)
    cf.disconnected.add_callback(lambda uri: print(f"[{now()}] disconnected from {uri}"))
    cf.connection_failed.add_callback(lambda uri, msg: print(f"[{now()}] connection_failed: {msg}"))
    cf.link_established.add_callback(on_link_established)
    cf.connected.add_callback(lambda uri: print(f"[{now()}] connected: TOCs downloaded"))
    if args.verbose:
        cf.console.receivedChar.add_callback(lambda text: sys.stdout.write(f"   [console] {text}"))

    # watchdog: SyncCrazyflie.open_link() blocks without a timeout
    setup_done = threading.Event()

    def watchdog():
        if not setup_done.wait(args.connect_timeout):
            print(f"\n[{now()}] FAIL: connect/TOC/param download did not finish within {args.connect_timeout:g} s")
            print(f"   {link_status(cf, link_holder.get('link'))}")
            print("   hints: drone powered? paired in Windows (unpair)? other BLE client still connected? "
                  "Crazyradio talked to it (power-cycle)? If rx packets stay at 0 the pump is not releasing "
                  "downlink packets; if tx errors grow, try --write-with-response or --max-inflight 1.")
            print(">>> FAIL <<<")
            os._exit(2)

    threading.Thread(target=watchdog, name="connect-watchdog", daemon=True).start()

    t_connect0 = time.perf_counter()

    def progress():
        # one line every 5 s while connecting / downloading TOCs and params
        while not setup_done.wait(5.0):
            s = link_stats(cf, link_holder.get("link"))
            if s:
                print(f"[{now()}]   ... {time.perf_counter() - t_connect0:5.1f} s: rx {s.get('rx_packets', 0)} pkt, "
                      f"tx {s.get('tx_packets', 0)} pkt ({s.get('tx_writes_wr', 0)} with-response), "
                      f"nulls {s.get('tx_null', 0)}, write errors {s.get('tx_errors', 0)}, "
                      f"orphans {s.get('rx_orphans', 0)}, lenfield {s.get('rx_convention') or '?'}")
                print(f"              tx by port {s.get('tx_by_port')}  rx by port {s.get('rx_by_port')}")
            else:
                print(f"[{now()}]   ... {time.perf_counter() - t_connect0:5.1f} s: scanning / connecting")

    threading.Thread(target=progress, name="connect-progress", daemon=True).start()
    print(f"[{now()}] connecting to {args.uri} ...")
    try:
        with SyncCrazyflie(args.uri, cf=cf) as scf:
            link_holder["link"] = cf.link
            result["connected"] = True
            result["connect_s"] = time.perf_counter() - t_connect0
            print(f"[{now()}] SyncCrazyflie open after {result['connect_s']:.1f} s")
            print(f"   {link_status(cf)}")

            # 3. param values (downloaded after the TOCs) -> stabilizer.estimator
            print(f"[{now()}] waiting for parameter values ...")
            t0 = time.perf_counter()
            while not scf.is_params_updated():
                if lost.is_set():
                    raise RuntimeError("link lost while downloading parameters")
                if time.perf_counter() - t0 > args.connect_timeout:
                    raise RuntimeError("timeout waiting for parameter values")
                time.sleep(0.05)
            result["params_s"] = time.perf_counter() - t0
            print(f"[{now()}] all parameters updated after {result['params_s']:.1f} s")
            setup_done.set()

            from toc_names import resolve as _resolve   # nRF 2024.10 mangles long TOC names
            value = scf.cf.param.get_value(_resolve(scf.cf.param.toc, "stabilizer.estimator"))
            result["param_value"] = value
            result["param_ok"] = value is not None and str(value) != ""
            names = {1: "complementary", 2: "Kalman"}
            try:
                desc = names.get(int(value), "?")
            except (TypeError, ValueError):
                desc = "?"
            print(f"[{now()}] stabilizer.estimator = {value!r} ({desc})")

            # 4. stream external position at `rate` Hz for `duration` s (port 6, no commander traffic)
            interval = 1.0 / args.rate
            n_target = int(round(args.rate * args.duration))
            result["target"] = n_target
            print(f"[{now()}] streaming extpos(0,0,0) at {args.rate:g} Hz for {args.duration:g} s "
                  f"({n_target} packets), pump {link_stats(cf).get('pump_hz', '?')} Hz ...")
            stats0 = link_stats(cf)
            sent = exceptions = late = 0
            last_err = ""
            t_start = time.perf_counter()
            next_t = t_start
            for i in range(n_target):
                if lost.is_set():
                    print(f"[{now()}] stopping stream: link lost after {i} packets")
                    break
                t_now = time.perf_counter()
                if t_now < next_t:
                    time.sleep(next_t - t_now)
                elif t_now - next_t > interval:
                    late += 1
                try:
                    scf.cf.extpos.send_extpos(0.0, 0.0, 0.0)
                    sent += 1
                except Exception as e:  # noqa: BLE001
                    exceptions += 1
                    last_err = repr(e)
                next_t += interval
                if (i + 1) % max(1, int(args.rate)) == 0:
                    s = link_stats(cf)
                    print(f"   {i + 1:4d}/{n_target} sent, exceptions {exceptions}, "
                          f"driver tx_err {s.get('tx_errors', '?')} dropped {s.get('tx_dropped', '?')} "
                          f"nulls {s.get('tx_null', '?')}, rx {s.get('rx_packets', '?')} pkt")
            elapsed = time.perf_counter() - t_start
            time.sleep(0.3)  # let the sender task flush the last writes
            stats1 = link_stats(cf)
            result.update(sent=sent, exceptions=exceptions, late=late, elapsed_s=elapsed, stats=stats1)
            result["tx_errors"] = stats1.get("tx_errors", 0) - stats0.get("tx_errors", 0)
            result["tx_dropped"] = stats1.get("tx_dropped", 0) - stats0.get("tx_dropped", 0)
            result["nulls"] = stats1.get("tx_null", 0)
            result["status"] = link_status(cf)
            print(f"[{now()}] streamed {sent}/{n_target} packets in {elapsed:.2f} s "
                  f"-> {sent / elapsed if elapsed > 0 else 0:.1f} Hz; python exceptions {exceptions}; "
                  f"late sends {late}; driver write errors {result['tx_errors']}; dropped {result['tx_dropped']}; "
                  f"nulls sent so far {result['nulls']} (pump {stats1.get('pump_hz', '?')} Hz)")
            if last_err:
                print(f"   last exception: {last_err}")
            print(f"   {result['status']}")
            print(f"[{now()}] closing link ...")
        # 5. SyncCrazyflie.__exit__ -> cf.close_link() -> BleDriver.close()
        print(f"[{now()}] link closed; {link_status(cf, link_holder.get('link'))}")
    except KeyboardInterrupt:
        result["error"] = "interrupted"
        print(f"\n[{now()}] interrupted")
        try:
            cf.close_link()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        print(f"[{now()}] ERROR: {e!r}")
        if not result["status"]:
            result["status"] = link_status(cf, link_holder.get("link"))
        try:
            cf.close_link()
        except Exception:  # noqa: BLE001
            pass
    finally:
        setup_done.set()

    # verdict
    final_stats = result["stats"] or link_stats(cf, link_holder.get("link"))
    streamed_ok = (result["target"] > 0 and result["sent"] == result["target"]
                   and result["elapsed_s"] >= args.duration - 0.5)
    ok = (result["connected"] and result["param_ok"] and streamed_ok and result["exceptions"] == 0
          and result["tx_errors"] == 0 and result["tx_dropped"] == 0 and not result["disconnected"]
          and not result["error"])
    print("\n================ SUMMARY ================")
    print(f"uri            : {args.uri}")
    print(f"link options   : pump_hz={final_stats.get('pump_hz', settings['pump_hz'])}, "
          f"stream_ports={final_stats.get('stream_ports', sorted(settings['stream_ports']))}, "
          f"write_with_response={final_stats.get('write_with_response', settings['write_with_response'])}, "
          f"max_inflight_writes={final_stats.get('max_inflight_writes', settings['max_inflight_writes'])}")
    print(f"writes         : {final_stats.get('tx_writes_wwr', '?')} without response, "
          f"{final_stats.get('tx_writes_wr', '?')} with response")
    print(f"connected      : {result['connected']} (SyncCrazyflie open after {result['connect_s']:.1f} s, "
          f"params after {result['params_s']:.1f} s)")
    print(f"param read     : {result['param_ok']} (stabilizer.estimator = {result['param_value']!r})")
    print(f"streamed       : {result['sent']}/{result['target']} in {result['elapsed_s']:.2f} s "
          f"(late {result['late']})")
    print(f"errors         : python exceptions {result['exceptions']}, driver write errors {result['tx_errors']}, "
          f"dropped {result['tx_dropped']}")
    print(f"null packets   : {final_stats.get('tx_null', result['nulls'])} sent in total "
          f"(pump {final_stats.get('pump_hz', '?')} Hz)")
    print(f"downlink       : {final_stats.get('rx_packets', '?')} pkt / {final_stats.get('rx_fragments', '?')} frag, "
          f"orphans {final_stats.get('rx_orphans', '?')}, incomplete {final_stats.get('rx_incomplete', '?')}, "
          f"lenfield convention {final_stats.get('rx_convention', '?')}")
    print(f"disconnected   : {result['disconnected']}")
    if result["error"]:
        print(f"error          : {result['error']}")
    print(f"link status    : {result['status'] or link_status(cf, link_holder.get('link'))}")
    print(f"\n>>> {'PASS' if ok else 'FAIL'} <<<")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
