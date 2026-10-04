#!/usr/bin/env python
"""cam_server.py -- standalone camera page for rig setup, no drone needed.

Serves the ArUco tracker's picture in the browser with a verdict written on it:

* relative mode (no calibration needed): when the floor marker (id 0) and the drone marker (id 1)
  are both in the picture, the angle between their top edges is measured directly in the image.
  Lay the floor-marker sheet next to the drone with its TOP EDGE pointing the way the drone's nose
  points, then turn the paper on the drone until the page says "aligned".
* world mode (needs calib/camN_extrinsics.npz for the camera's CURRENT position): the drone
  marker's heading relative to world +x, and its position.

    cf64\\Scripts\\python.exe tools\\cam_server.py [--camera N] [--port 8766]
    open http://127.0.0.1:8766/camera          (picture, 2 frames/s)
         http://127.0.0.1:8766/api/pose        (json: tracking_ok, xyz, heading_deg, relative_deg, verdict)

Only one program can hold the camera: stop mission.py / tracker.py / cam_check.py first.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tracker  # noqa: E402
from mission import _CAMERA_HTML, camera_snapshot_jpeg  # noqa: E402


def turn_advice(deg: float, tol: float = 10.0) -> str:
    """deg > 0: the drone marker's top edge is rotated clockwise (as seen in the picture) from the reference."""
    if deg > tol:
        return "turn the paper COUNTER-clockwise"
    if deg < -tol:
        return "turn the paper CLOCKWISE"
    return "YES, aligned"


def relative_angle_deg(markers: dict) -> float | None:
    """Angle (degrees, -180..180) of marker 1's top edge relative to marker 0's top edge, in the image.

    ArUco corners are ordered clockwise from the marker's top-left corner, so corner1 - corner0 is the
    top edge. Image y points down, so a positive result means 'clockwise on screen'."""
    if 0 not in markers or 1 not in markers:
        return None

    def top_edge_angle(c):
        c = np.asarray(c, dtype=float).reshape(4, 2)
        v = c[1] - c[0]
        return math.degrees(math.atan2(v[1], v[0]))

    d = top_edge_angle(markers[1]) - top_edge_angle(markers[0])
    return (d + 180.0) % 360.0 - 180.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    import uvicorn
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, JSONResponse

    cfg = tracker.TrackerConfig.load()
    if args.camera is not None:
        cfg.aruco_camera = args.camera
    tr = tracker.ArucoTracker(cfg).start()

    def pose() -> dict:
        st = tr.get_state()
        ok = bool(st is not None and st.tracking_ok and st.xyz is not None)
        deg = None
        if ok and st.yaw is not None and math.isfinite(float(st.yaw)):
            deg = math.degrees(float(st.yaw))
        views = tr.get_views()
        view = views[0] if views else None
        markers = dict(getattr(view, "markers", None) or {})
        rel = relative_angle_deg(markers)
        if rel is not None:
            verdict = f"relative to the floor marker {rel:+.0f} deg: {turn_advice(rel)}"
        elif 1 in markers:
            verdict = "drone marker seen; put the floor marker next to it (top edge = nose direction) for the angle"
        else:
            verdict = "drone marker not seen"
        return {
            "tracking_ok": ok,
            "xyz": [float(v) for v in st.xyz] if ok else None,
            "heading_deg": deg,
            "relative_deg": rel,
            "markers_seen": sorted(int(k) for k in markers),
            "verdict": verdict,
            "mode": getattr(tr, "mode", "?"),
            "fps": float(getattr(st, "fps", 0.0)) if st is not None else 0.0,
        }

    app = FastAPI(title="Pathcaster camera setup")

    @app.get("/api/camera.jpg", include_in_schema=False)
    def camera_jpg():
        p = pose()
        label = p["verdict"]
        if p["relative_deg"] is None and p["heading_deg"] is not None:
            x, y, z = p["xyz"]
            label += f"   | world heading {p['heading_deg']:+.0f} deg, at x {x:+.2f} y {y:+.2f} z {z:+.2f} m (needs current extrinsics)"
        data = camera_snapshot_jpeg(tr, label)
        if data is None:
            return Response(status_code=503, content=b"no camera frame", media_type="text/plain")
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/camera", include_in_schema=False)
    @app.get("/", include_in_schema=False)
    def camera_page():
        return HTMLResponse(_CAMERA_HTML)

    @app.get("/api/pose")
    def api_pose():
        return JSONResponse(pose())

    print(f"[cam_server] camera {cfg.aruco_camera} -> http://{args.host}:{args.port}/camera  (mode {getattr(tr, 'mode', '?')})", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        tr.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
