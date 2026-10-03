#!/usr/bin/env python
"""hsv_tune.py -- pick the HSV range of one sticky-note colour and save it to calib/tracker.json.

Step 4 of the tracker calibration (see tracker.py), once per marker colour,
under the lighting of the demo spot:

    cf64\\Scripts\\python.exe tools\\hsv_tune.py --camera 0 --name nose
    cf64\\Scripts\\python.exe tools\\hsv_tune.py --camera 0 --name tail
    cf64\\Scripts\\python.exe tools\\hsv_tune.py --camera 0 --name wand

Window: live frame (left) with the detected blob, mask (right), six trackbars.
    left-click on the sticky note   seed the range from that pixel (H +-10, S/V lower bounds)
    trackbars                       fine-tune; H lo > H hi means a wrap-around (red) range
    s                               save the range into calib/tracker.json (merged, other colours kept)
    r                               reset to the range currently in the config
    q / Esc                         quit
Aim for a mask that shows ONLY the note as a solid blob from every distance you
will use; hold the drone where the cameras will see it.  --image FILE tunes on
a still image instead of a camera.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tracker import BACKENDS, DEFAULT_COLORS, CameraThread, HsvRange, MarkerDetector, TrackerConfig  # noqa: E402

BARS = (("H lo", 179), ("H hi", 179), ("S lo", 255), ("S hi", 255), ("V lo", 255), ("V hi", 255))


def _nop(_v: int) -> None:
    pass


class Tuner:
    def __init__(self, win: str, initial: HsvRange) -> None:
        self.win = win
        self.hsv: np.ndarray | None = None
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        for (name, mx) in BARS:
            cv2.createTrackbar(name, win, 0, mx, _nop)
        self.set_range(initial)
        cv2.setMouseCallback(win, self.on_mouse)

    def set_range(self, r: HsvRange) -> None:
        vals = [r.lo[0], r.hi[0], r.lo[1], r.hi[1], r.lo[2], r.hi[2]]
        for (name, _mx), v in zip(BARS, vals):
            cv2.setTrackbarPos(name, self.win, int(v))

    def get_range(self) -> HsvRange:
        v = [cv2.getTrackbarPos(name, self.win) for name, _mx in BARS]
        return HsvRange((v[0], v[2], v[4]), (v[1], v[3], v[5]))

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        h, w = self.hsv.shape[:2]
        if x >= w:           # click on the mask half: map back onto the frame
            x -= w
        if not (0 <= x < w and 0 <= y < h):
            return
        patch = self.hsv[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
        hh, ss, vv = (int(np.median(patch[:, i])) for i in range(3))
        lo_h, hi_h = (hh - 10) % 180, (hh + 10) % 180
        self.set_range(HsvRange((lo_h, max(40, ss - 70), max(40, vv - 70)), (hi_h, 255, 255)))
        print(f"sampled HSV ({hh},{ss},{vv}) -> H {lo_h}..{hi_h}{' (wraps)' if lo_h > hi_h else ''}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--name", required=True, help="marker name: nose | tail | wand (any name is accepted)")
    ap.add_argument("--config", default="calib/tracker.json")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    ap.add_argument("--image", help="tune on a still image instead of the camera")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = TrackerConfig.load(args.config)
    initial = cfg.colors.get(args.name) or DEFAULT_COLORS.get(args.name) or HsvRange((0, 100, 80), (179, 255, 255))

    still = None
    cam = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            print(f"ERROR: cannot read {args.image}", file=sys.stderr)
            return 2
    else:
        cam = CameraThread(args.camera, BACKENDS[args.backend or cfg.backend], cfg.width, cfg.height, cfg.fps)
        cam.start()
        if not cam.wait_open(10.0):
            print(f"ERROR: {cam.error or 'camera did not open in time'}", file=sys.stderr)
            return 2

    win = f"hsv_tune: {args.name}  (click note | s save | r reset | q quit)"
    tuner = Tuner(win, initial)
    print(f"== tuning '{args.name}' from {initial.lo}..{initial.hi}; config {args.config}")
    saved_msg_until = 0.0
    try:
        while True:
            frame = still if still is not None else cam.latest()[0]  # type: ignore[union-attr]
            if frame is None:
                if (cv2.waitKey(10) & 0xFF) in (27, ord("q")):
                    break
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            tuner.hsv = hsv
            r = tuner.get_range()
            det = MarkerDetector(r.lo, r.hi, cfg.min_area, cfg.open_kernel)
            mask = det.mask(hsv)
            found = det.detect(hsv)
            vis = frame.copy()
            if found is not None:
                u, v, a = found
                cv2.circle(vis, (int(u), int(v)), max(6, int(np.sqrt(a) / 2)), (0, 255, 0), 2)
                cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
            txt = (f"{args.name}: H {r.lo[0]}-{r.hi[0]}{' wrap' if r.wraps else ''} S {r.lo[1]}-{r.hi[1]} "
                   f"V {r.lo[2]}-{r.hi[2]}   blob "
                   + (f"({found[0]:.0f},{found[1]:.0f}) area {found[2]:.0f}" if found else "none"))
            if time.perf_counter() < saved_msg_until:
                txt += "   SAVED"
            both = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            cv2.putText(both, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(both, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, both)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                tuner.set_range(initial)
            if key == ord("s"):
                TrackerConfig.save_color(args.name, r.lo, r.hi, args.config)
                print(f"saved {args.name}: lo {r.lo} hi {r.hi} -> {args.config}")
                saved_msg_until = time.perf_counter() + 1.5
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
