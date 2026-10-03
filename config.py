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
BLE_PUMP_HZ = 100.0          # null packets per second when idle (keeps the downlink flowing)
BLE_FAST_INTERVAL = True     # ask Windows 11 for the 15 ms connection interval

# Position feed (feed.py): external position rate into the onboard Kalman filter.
EXTPOS_RATE_HZ = 30.0

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
