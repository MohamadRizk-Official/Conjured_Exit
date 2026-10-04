# Conjured Exit (Pathcaster)

Team Slitherpuff, Hack Dearborn 2026. Theme "Conjure Reality", track Shaping Society (crisis response).

> Show it the way once. It'll lead you out forever.

Exit signs can't move, and in a fire they can point you the wrong way. Conjured Exit teaches a
palm-sized Crazyflie 2.1+ drone an escape route by carrying it along that route once. On an alarm
button or a spoken "fire", the drone takes off and flies the route in front of you. Teach two exits
and say "exit A is blocked": it leads you to exit B. The same engine powers a "spell" mode: draw a
shape in the air with a wand and say "cast" to fly it.

There is no radio dongle. The drone is flown over the laptop's own Bluetooth LE through a cflib
link driver written during the hackathon (`ble_link.py`), after reverse engineering the drone's BLE
behaviour and working around a firmware bug that corrupts every packet longer than 20 bytes.

## How it works

```
webcam ---> tracker.py (ArUco marker on the drone, 30 Hz, metres)
              |                |
              |                +--> paths.PathRecorder --> paths.clean_path --> paths/<name>.json
              v
           feed.py (external position, 30 Hz) --> drone's onboard Kalman filter  (BLE, ble_link.py)
                                                        ^
flight.py (Flight) --- position setpoints at 20 Hz -----+     <-- cast / alarm / land / stop
ui/  FastAPI + WebSocket backend, React + three.js page   (buttons today, voice next)
```

* **Record:** carry the drone (motors off) along a route; the tracker logs its position.
* **Clean:** smooth, resample, clamp into the geofence (guide mode) or scale to fit (spell mode),
  re-time to a slow constant speed, save as JSON.
* **Fly:** take off, transit to the path start, follow it with streamed position setpoints, land.
  The drone's own Kalman filter fuses the camera position with its IMU, so the camera only has to
  be right on average.

## Hardware

* Crazyflie 2.1+ (stock: IMU + barometer, no positioning deck, no magnetometer). It cannot hold
  position on its own; camera tracking is required for every flight.
* No Crazyradio. Links: USB cable (`usb://0`, desk work on x64 machines) or the laptop's
  Bluetooth LE (`ble://<address>`, flight).
* One USB webcam (or the built-in one) and two printed ArUco markers:
  `calib/marker0_floor_15cm.png` is the world origin on the floor, `calib/marker1_drone_7cm.png`
  goes on top of the drone with its top edge towards the nose. Print both at 100 % scale.
* Windows 11 laptop. Developed on a Snapdragon X (Windows on ARM64), see the venv notes below.

## Setup (Windows, native Python 3.12; not WSL)

Anything that touches Bluetooth, USB or cameras must run in native Windows Python.

```powershell
# if scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1        # venv cf   : bleak + cflib (native; ARM64 or x64). Link tests, benchmarks.
.\setup_cv_env.ps1     # venv cf64 : OpenCV + cflib + bleak + fastapi (x64). Tracker, flight, UI.
```

Two venvs because PyPI has no OpenCV wheel for Windows ARM64: `cf64` is an x64 Python running
under Windows' x64 emulation, which is fast enough (a full tracking cycle costs about 4 ms). On an
ordinary x64 machine both scripts simply use the normal interpreter; `cf64` is the one to use.

The link URI lives in one place, `config.py` (`LINK_URI`), or in the environment:

```powershell
$env:PATHCASTER_LINK_URI = "ble://DB:04:E8:22:6F:CC"   # our drone; "ble://" scans for the first one
```

Frontend: a built copy is committed in `ui/web/dist`, so Node is optional. To change the page:
`cd ui\web; npm install; npm run dev` (hot reload on http://localhost:5173, proxies to the backend)
and `npm run build` when done.

## Running

Only one process may hold the drone's Bluetooth link at a time. Stop every other script before
connecting. The first connect takes about 50 s over BLE (parameter table download).

```powershell
# the whole system without hardware: fake drone that follows its setpoints, demo paths, the page on
# http://127.0.0.1:8765/ . Click ARMED, then ALARM or Cast; "Exit A blocked" mid-flight lands and relaunches exit B.
.\cf64\Scripts\python.exe mission.py --sim

# the same engine on the real drone (serves the page itself; nobody else may hold the link)
.\cf64\Scripts\python.exe mission.py --tracker sim                 # motors-off link test, fake positions
.\cf64\Scripts\python.exe mission.py --tracker aruco --camera N    # camera tracking, flights when ARMED

# operator UI alone, hardware-free simulator (demo paths, scripted fake drone)
.\cf64\Scripts\python.exe -m ui.server --sim

# UI with the real drone: SIM/LIVE toggle on the page; LIVE shows battery, position estimate,
# estimator convergence and a roll/pitch hardware-check card (tilt the drone, watch it move)
.\cf64\Scripts\python.exe -m ui.server --sim --live

# motors-off desk check over the real link: connect, 10 Hz log block, estimator convergence
.\cf64\Scripts\python.exe flight.py --check --seconds 20

# BLE link acceptance test (connect, params, extpos at 30 Hz, counters)
.\cf\Scripts\python.exe test_ble_link_hw.py --skip-scan --pump-hz 100

# raw BLE benchmark (no cflib): latency, throughput, notification loss
.\cf\Scripts\python.exe ble_bench.py
```

Camera rig (once per camera position; `N` is the camera index printed by `tools\camera_probe.py`):

```powershell
.\cf64\Scripts\python.exe calibrate_intrinsics.py --camera N --board 9x6 --square 0.024 --frames 20
.\cf64\Scripts\python.exe calibrate_extrinsics.py --camera N --marker-size 0.15     # floor marker in view
.\cf64\Scripts\python.exe tracker.py --aruco --camera N --show                       # live check
.\cf64\Scripts\python.exe tracker.py --sim --headless --seconds 5                    # no cameras
```

First autonomous hover (clear floor, drone on the floor near the floor marker, nose along +x).
Start it from a real console window so the FLY prompt can read the keyboard:

```powershell
cmd /c start powershell -NoExit -Command "Set-Location $PWD; .\cf64\Scripts\python.exe hover.py --tracker aruco --camera N --height 0.5 --hold 8"
```

It locks the tracker, streams positions, resets the estimator and waits for convergence, clears a
"crashed" supervisor state, asks you to type `FLY`, climbs to 0.5 m, holds, lands. Spacebar is the
emergency stop, `l` lands now, and it lands by itself if the camera loses the drone for 0.3 s.
`--tracker sim --dry-run` exercises the whole sequence without hardware.

Tests (267, no hardware needed):

```powershell
.\cf64\Scripts\python.exe -m unittest discover -s tests
```

## Repository layout

| path | what |
|---|---|
| `ble_link.py`, `ble_framing.py` | cflib `ble://` link driver (bleak), CRTP framing, null-packet pump, 20-byte packet limit |
| `tools/ble_echo_test.py` | loopback integrity test: numbered packets to the echo channel, per-mode corruption and latency |
| `tools/sim_session.py` | scripted hover / teach / cast / alarm-and-stop session against a running server, with a report |
| `ble_bench.py`, `test_ble_link_hw.py` | raw BLE benchmark and the driver's hardware acceptance test |
| `toc_names.py` | resolves parameter/log names that the drone's radio firmware mangles over BLE |
| `tracker.py` | ArUco single-camera tracker (`ArucoTracker`), colour two-camera tracker, `SimTracker` |
| `calibrate_intrinsics.py`, `calibrate_extrinsics.py` | checkerboard intrinsics, floor-marker extrinsics, printable markers |
| `feed.py` | streams the tracked position into the drone's Kalman filter at 30 Hz |
| `flight.py` | `Flight`: takeoff/land ramps, path following, geofence, tracking-lost landing, emergency stop, one log block |
| `paths.py` | `PathRecorder`, `clean_path`, geofence `Box`, JSON save/load, demo paths |
| `hop.py`, `hover.py` | first flight (barometer hop) and the autonomous hover sequence |
| `mission.py`, `mission_sim.py` | the mission app: ARMED gate, record/save/replay, alarm and reroute, pre-flight gate, serves the page; sim stand-ins |
| `ui/` | FastAPI backend (`ui.server`), state bus and command queue (`ui.state`), simulator, live bridge, React page (`ui/web`) |
| `tools/` | BLE diagnostics, connection-interval probe, camera probe, OpenCV speed check, HSV tuner |
| `shims/` | `libusb_package` stand-in so cflib imports on Windows ARM64 |
| `tests/` | unit tests with fakes for the drone, clock and tracker |
| `config.py` | the single place for the link URI, rates, geofence and safety constants |

## Verified BLE facts

Measured on the Crazyflie 2.1+ with nRF firmware 2024.10 and a Windows 11 laptop (bleak 3,
WinRT). Everything in the driver follows from these.

* GATT: service `0201`, CRTP characteristic `0202` for whole packets up to 20 bytes, CRTPUP `0203`
  for fragmented uplink, CRTPDOWN `0204` notifications for downlink.
* **Write-without-response corrupts packets.** Loopback test (`tools/ble_echo_test.py`, numbered
  packets to the echo channel, 4 October): fire-and-forget writes came back with their first bytes
  overwritten (`FF 00 FF`) in 4 of 200 packets when alone and in 183 of 200 while acknowledged
  extpos traffic ran alongside. That was the flight configuration until then, so the drone was
  flying on garbage setpoints: every take-off lurched, skidded or flipped. Every port is now written
  with response (`BLE_STREAM_PORTS = ()`), which delivered 200 of 200 intact.
* Acknowledged writes have a budget of about 30 per second at the 15 ms interval: 15 Hz extpos plus
  15 Hz setpoints runs at about 0.1 s round-trip latency, 40 per second lags 0.75 s and 50 per
  second lags 1.3 s. The idle null pump drops from 100 Hz to 20 Hz once connected, because each
  null is an acknowledged write too.
* The downlink is driven by the uplink: the drone releases one downlink packet per uplink packet,
  so an idle link gets nothing. The driver sends null packets ("pump") at 100 Hz when idle.
* The nRF firmware drops one notification per connection event when more than one is queued, so
  all traffic is serialized through one acknowledged write at a time.
* Every packet longer than 20 bytes is corrupted in both directions (byte 19 lost, a garbage byte
  appended). Consequences: parameter/log names arrive mangled (`toc_names.resolve` maps the true
  names), log blocks are built with at most 5 variables per packet, the 29-byte pose packet and
  the 25-byte `go_to` are never used, and the driver refuses to send anything over 20 bytes.
* Windows negotiates a 15 ms connection interval, which the drone then raises to 45 or 60 ms.
  Requesting the throughput-optimised parameters restores 15 ms; the driver re-requests it when
  it slips. Parameter download then runs at about 10 parameters per second.
* cflib's link-statistics pinger (10 Hz) is disabled after connecting; it would eat the budget.
* Acceptance run: connected in 21 s, all parameters in 31 s, `stabilizer.estimator` read back as 2
  (Kalman), 300 of 300 external positions delivered at 30 Hz, zero write errors, zero drops.

## Flight

Rules that keep the one battery and the few spare propellers alive:

* Only one process holds the link. Connect once at start, keep it alive, TOC cache on.
* Position setpoints streamed at 15 Hz (acknowledged writes, see the BLE budget above); if the link
  drops, the firmware's setpoint watchdog stops the drone. Take-off and landing are ramps of the same setpoints; landing ends with a stop
  setpoint from 6 cm.
* One 10 Hz log block in flight: battery, Kalman x/y/z and position variance (14 bytes, one BLE
  notification). Nothing else is logged while flying.
* Every target is clamped into the geofence (1.5 m square, 0.2 to 1.2 m up). Tracking lost for
  longer than 0.3 s triggers a blind descent (level attitude, thrust-only steps, motors off) because
  the position estimate can no longer be trusted. A take-off that has not gained 40 % of the climb
  by the end of the ramp is cut with stop setpoints, and the abort message records how far the
  battery sagged. The emergency stop (spacebar, STOP button) locks the drone until a reboot.
* The supervisor often boots in a "crashed" state; a crash-recovery request clears it before
  arming. The complementary estimator reports altitude above sea level, so flights use the Kalman
  estimator with a reset and external position.
* Position only goes into the drone; its heading comes from the gyro, which only knows heading
  changes. Every estimator reset therefore sets `kalman.initialYaw` from the camera-measured marker
  heading (plus `DRONE_MARKER_YAW_OFFSET_DEG` when the marker is not taped top-edge-to-nose). Without
  it the position controller pushes in a rotated direction and the drone slides sideways instead of
  climbing (4 October, 03:51). A take-off that drifts more than 0.4 m sideways is cut with a blind
  descent. Flights stay under a minute.

## Tracker

* One camera, OpenCV ArUco (DICT_4X4_50). Marker id 0 (15 cm) flat on the floor defines the world
  frame: x forward along its top edge, y left, z up. Marker id 1 (7 cm) on top of the drone, top
  edge towards the nose. Pose from `solvePnP` (IPPE square), facing-up disambiguation, jump gate.
* Synthetic accuracy at 1.7 m: about 3 cm mean and 7 cm at the 95th percentile along the camera
  ray, yaw within 3 degrees. Position noise given to the drone as 5 cm (`locSrv.extPosStdDev`).
* A two-camera colour-marker tracker (nose/tail/wand colours, triangulation) is in the same file
  and was the original plan; the single-camera ArUco path needs one calibration and no colour tuning.

## Status (3 October 2026)

Done: BLE benchmark and driver (hardware acceptance passed), tracker code and calibration tools,
flight layer, path engine, UI with simulator and live telemetry, first motorised flight (barometer
hop, caught by hand as expected without position input), autonomous hover sequence tested against
fakes and in dry run, the mission app (`mission.py`: ARMED gate, record / save / replay, alarm,
land-then-relaunch reroute, pre-flight checks) verified end to end in simulation.

Open: camera calibration on the rig, first autonomous hover on hardware, the mission app on the real
link and camera, voice commands in the browser, the demo.

## License

MIT, see `LICENSE`.
