"""Pathcaster shared configuration. Every module reads tunables from here.

The cflib link URI lives ONLY here (README.md rule). Switch desk <-> flight by
changing LINK_URI or by setting the PATHCASTER_LINK_URI environment variable:
    usb://0                     Crazyflie on the USB cable (desk development, x64 machines)
    ble://DB:04:E8:22:6F:CC     our drone over Bluetooth LE (flight)
    ble://                      first Crazyflie found by BLE scan
"""
import os

DRONE_NAME = "Crazyflie-226FCC"
DRONE_BLE_ADDRESS = "DB:04:E8:22:6F:CC"

LINK_URI = os.environ.get("PATHCASTER_LINK_URI", f"ble://{DRONE_BLE_ADDRESS}")

# BLE link driver tuning (ble_link.py). Measured 2026-10-03: with a 15 ms connection interval and
# pump 100 Hz the parameter download runs at ~10 params/s (30 s for 310 params).
BLE_PUMP_FLIGHT_HZ = 20.0    # pump after connect: every null is an acknowledged write that delays real packets
BLE_PUMP_HZ = 100.0          # null packets per second when idle (keeps the downlink flowing)
BLE_FAST_INTERVAL = True     # ask Windows 11 for the 15 ms connection interval
# CRTP ports written WITHOUT response (fire and forget). EMPTY on purpose. Loopback test 2026-10-04
# (tools/ble_echo_test.py, nRF 2024.10): fire-and-forget packets arrive mangled (first bytes overwritten
# with FF 00 FF), 4/200 when alone and 183/200 while the camera feed runs. That was why every take-off
# lurched, skidded or flipped: the setpoints the drone received were garbage. Every port now goes
# write-with-response; the budget is ~30 acknowledged writes/s, so keep extpos + setpoint rates inside it.
BLE_STREAM_PORTS = ()

# Position feed (feed.py): external position rate into the onboard Kalman filter, and the setpoint
# stream rate (flight.py via mission.py). Both are acknowledged BLE writes; loopback 2026-10-04:
# 15 + 15 per second stays at ~0.1 s latency, 40/s lags 0.75 s, 50/s lags 1.3 s (tools/ble_echo_test.py).
EXTPOS_RATE_HZ = 15.0
SETPOINT_RATE_HZ = 15.0

# Safety (flight.py). Non-negotiable values from README.md.
TRACKING_LOST_LAND_S = 0.3      # tracking lost longer than this -> land
REPLAY_SPEED_MPS = 0.3          # fixed slow replay speed
WAYPOINT_DT_S = 0.4             # one go_to roughly every 0.3-0.5 s
TAKEOFF_HEIGHT_M = 0.5

# Geofence: the volume both cameras see. paths.py exposes the same box as GEOFENCE.
GEOFENCE_X = (-0.75, 0.75)
GEOFENCE_Y = (-0.75, 0.75)
GEOFENCE_Z = (0.2, 1.2)

# Data directories
PATHS_DIR = "paths"
CALIB_DIR = "calib"
