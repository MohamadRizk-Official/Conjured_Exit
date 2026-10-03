"""Unit tests for the tracker math (no cameras needed).  Run from the project root:

    cf64\\Scripts\\python.exe -m unittest tests.test_tracker_math -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest

import cv2
import numpy as np

import calibrate_extrinsics as CE
import calibrate_intrinsics as CI
from tracker import (
    Camera,
    FrameSource,
    HsvRange,
    ImageSource,
    JumpGate,
    MarkerDetector,
    PoseFuser,
    SimTracker,
    Tracker,
    TrackerConfig,
    TrackerState,
    camera_from_look_at,
    draw_view,
    format_state,
    reprojection_error,
    triangulate,
)

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.12, 0.04, 0.0005, -0.0003, 0.01])
RADIUS, HEIGHT = 2.0 * math.cos(math.radians(30)), 2.0 * math.sin(math.radians(30))   # 2 m away, 30 deg down


def synth_rig(dist: np.ndarray | None = DIST) -> tuple[Camera, Camera]:
    """Two cameras 2 m from the origin, 90 deg apart in azimuth, looking down 30 deg."""
    cam_a = camera_from_look_at(0, (RADIUS, 0.0, HEIGHT), (0.0, 0.0, 0.0), K, dist)
    cam_b = camera_from_look_at(1, (0.0, RADIUS, HEIGHT), (0.0, 0.0, 0.0), K, dist)
    return cam_a, cam_b


# ------------------------------------------------------------------ geometry


class TestGeometry(unittest.TestCase):
    def test_look_at_camera_convention(self) -> None:
        cam_a, _ = synth_rig()
        np.testing.assert_allclose(cam_a.position, [RADIUS, 0.0, HEIGHT], atol=1e-9)
        np.testing.assert_allclose(cam_a.R @ cam_a.R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(cam_a.R)), 1.0, places=9)
        fwd = cam_a.forward
        np.testing.assert_allclose(fwd, -cam_a.position / np.linalg.norm(cam_a.position), atol=1e-9)
        self.assertLess(fwd[2], 0.0)                               # looks down
        self.assertAlmostEqual(math.degrees(math.asin(-fwd[2])), 30.0, places=6)
        u, v = cam_a.project((0.0, 0.0, 0.0))
        self.assertAlmostEqual(u, 320.0, places=6)                 # origin on the optical axis
        self.assertAlmostEqual(v, 240.0, places=6)
        # a point to the LEFT in the world (+y) appears to the RIGHT in cam A (it faces -x)
        self.assertGreater(cam_a.project((0.0, 0.3, 0.0))[0], 320.0)
        # a higher point appears higher in the image (smaller v)
        self.assertLess(cam_a.project((0.0, 0.0, 0.3))[1], 240.0)

    def test_P_and_undistort_convention(self) -> None:
        cam_a, _ = synth_rig()
        np.testing.assert_allclose(cam_a.P, cam_a.K @ np.hstack([cam_a.R, cam_a.t]))
        p = np.array([0.3, -0.2, 0.7])
        raw = cam_a.project(p)                                     # with distortion
        pin = cam_a.P @ np.append(p, 1.0)
        pin = pin[:2] / pin[2]                                     # ideal pinhole pixel
        self.assertGreater(np.linalg.norm(np.array(raw) - pin), 0.5)    # distortion is significant here
        und = cam_a.undistort_point(*raw)
        np.testing.assert_allclose(und, pin, atol=1e-3)            # undistorted == pinhole (P=K convention)

    def test_triangulate_roundtrip_under_5mm(self) -> None:
        cam_a, cam_b = synth_rig()
        worst = 0.0
        for x in (-0.5, 0.0, 0.5):
            for y in (-0.5, 0.0, 0.5):
                for z in (0.2, 0.7, 1.2):
                    p = np.array([x, y, z])
                    uv_a, uv_b = cam_a.project(p), cam_b.project(p)
                    xyz = triangulate(cam_a, cam_b, uv_a, uv_b)
                    err = float(np.linalg.norm(xyz - p))
                    worst = max(worst, err)
                    self.assertLess(err, 0.005, f"point {p}: {err*1000:.2f} mm")
                    self.assertLess(reprojection_error((cam_a, cam_b), (uv_a, uv_b), xyz), 0.05)
        self.assertLess(worst, 0.001)                              # noise-free: well under a millimetre

    def test_triangulate_with_pixel_noise_is_centimetric(self) -> None:
        cam_a, cam_b = synth_rig()
        rng = np.random.default_rng(1)
        p = np.array([0.2, -0.1, 0.6])
        errs = []
        for _ in range(50):
            uv_a = np.array(cam_a.project(p)) + rng.normal(0, 0.5, 2)
            uv_b = np.array(cam_b.project(p)) + rng.normal(0, 0.5, 2)
            errs.append(np.linalg.norm(triangulate(cam_a, cam_b, uv_a, uv_b) - p))
        self.assertLess(float(np.mean(errs)), 0.01)                # 0.5 px noise -> ~mm, not cm

    def test_reprojection_error_detects_bad_correspondence(self) -> None:
        cam_a, cam_b = synth_rig()
        p, q = np.array([0.1, 0.0, 0.5]), np.array([0.1, 0.4, 0.5])
        xyz = triangulate(cam_a, cam_b, cam_a.project(p), cam_b.project(q))   # mismatched markers
        self.assertGreater(reprojection_error((cam_a, cam_b), (cam_a.project(p), cam_b.project(q)), xyz), 8.0)

    def test_uncalibrated_camera_errors(self) -> None:
        with self.assertRaises(ValueError):
            _ = Camera(0, K=K).P
        with self.assertRaises(ValueError):
            Camera(0).undistort_point(1.0, 2.0)


# -------------------------------------------------------------------- fusion


class TestPoseFuser(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TrackerConfig()
        self.f = PoseFuser(self.cfg, mode="3d")

    def test_yaw_zero_nose_forward(self) -> None:
        s = self.f.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5))
        self.assertAlmostEqual(s.yaw, 0.0, places=9)
        self.assertTrue(s.yaw_ok)
        np.testing.assert_allclose(s.xyz, [0.0, 0.0, 0.5], atol=1e-12)
        self.assertEqual(s.t, 1.0)
        self.assertTrue(s.tracking_ok)
        self.assertEqual(s.mode, "3d")

    def test_yaw_left_is_plus_half_pi(self) -> None:
        s = self.f.update(1.0, nose=(0.0, 0.1, 0.5), tail=(0.0, -0.1, 0.5))
        self.assertAlmostEqual(s.yaw, math.pi / 2, places=9)

    def test_yaw_backwards_is_pi(self) -> None:
        s = self.f.update(1.0, nose=(-0.1, 0.0, 0.5), tail=(0.1, 0.0, 0.5))
        self.assertAlmostEqual(abs(s.yaw), math.pi, places=9)

    def test_yaw_ignores_height_difference(self) -> None:
        s = self.f.update(1.0, nose=(0.1, 0.1, 0.55), tail=(-0.1, -0.1, 0.45))
        self.assertAlmostEqual(s.yaw, math.pi / 4, places=9)

    def test_nose_only_uses_held_yaw_and_separation(self) -> None:
        self.f.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5))
        s = self.f.update(1.033, nose=(0.1, 0.0, 0.5), tail=None)
        self.assertFalse(s.yaw_ok)
        self.assertAlmostEqual(s.yaw, 0.0)                         # held
        np.testing.assert_allclose(s.xyz, [0.0, 0.0, 0.5], atol=1e-9)   # centre, not the nose
        self.assertTrue(s.tracking_ok)
        s = self.f.update(1.066, nose=None, tail=(-0.1, 0.0, 0.5))
        np.testing.assert_allclose(s.xyz, [0.0, 0.0, 0.5], atol=1e-9)
        self.assertEqual(s.t, 1.066)

    def test_nose_only_without_history_is_nose(self) -> None:
        s = self.f.update(1.0, nose=(0.1, 0.0, 0.5))
        np.testing.assert_allclose(s.xyz, [0.1, 0.0, 0.5])
        self.assertIsNone(s.yaw)
        self.assertFalse(s.yaw_ok)

    def test_two_metre_jump_rejected_and_tracking_lost_after_0_3s(self) -> None:
        self.f.update(0.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5))
        s = self.f.update(0.033, nose=(2.1, 0.0, 0.5), tail=(1.9, 0.0, 0.5))
        np.testing.assert_allclose(s.xyz, [0.0, 0.0, 0.5])         # kept the previous fix
        self.assertEqual(s.t, 0.0)
        self.assertEqual(s.rejects, 1)
        self.assertTrue(self.f.state(0.1).tracking_ok)
        self.assertTrue(self.f.state(0.3).tracking_ok)             # exactly lost_after_s is still ok
        lost = self.f.state(0.35)
        self.assertFalse(lost.tracking_ok)
        self.assertAlmostEqual(lost.age_s, 0.35)
        np.testing.assert_allclose(lost.xyz, [0.0, 0.0, 0.5])      # last position still reported

    def test_small_motion_accepted(self) -> None:
        self.f.update(0.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5))
        s = self.f.update(0.033, nose=(0.11, 0.01, 0.5), tail=(-0.09, 0.01, 0.5))
        np.testing.assert_allclose(s.xyz, [0.01, 0.01, 0.5], atol=1e-9)
        self.assertEqual(s.rejects, 0)

    def test_reacquire_after_ten_consecutive_rejections(self) -> None:
        self.f.update(0.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5))
        t = 0.0
        for i in range(10):
            t += 0.033
            s = self.f.update(t, nose=(2.1, 0.0, 0.5), tail=(1.9, 0.0, 0.5))
            self.assertEqual(s.rejects, i + 1)
            np.testing.assert_allclose(s.xyz, [0.0, 0.0, 0.5])
        t += 0.033
        s = self.f.update(t, nose=(2.1, 0.0, 0.5), tail=(1.9, 0.0, 0.5))
        np.testing.assert_allclose(s.xyz, [2.0, 0.0, 0.5])         # 11th accepted
        self.assertEqual(s.rejects, 0)
        self.assertTrue(s.tracking_ok)

    def test_reprojection_gate(self) -> None:
        s = self.f.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5), reproj_px={"nose": 9.0, "tail": 0.5})
        np.testing.assert_allclose(s.xyz, [-0.1, 0.0, 0.5])        # nose dropped -> tail only, no history
        self.assertFalse(s.yaw_ok)
        self.assertAlmostEqual(s.reproj_px, 0.5)
        f2 = PoseFuser(self.cfg)
        s = f2.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5), reproj_px={"nose": 9.0, "tail": 8.5})
        self.assertIsNone(s.xyz)
        self.assertFalse(s.tracking_ok)
        s = f2.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5), reproj_px={"nose": 8.0, "tail": 7.9})
        self.assertIsNotNone(s.xyz)                                # <= threshold passes
        self.assertAlmostEqual(s.reproj_px, 8.0)

    def test_reprojection_gate_has_no_reacquire(self) -> None:
        for i in range(15):
            s = self.f.update(float(i), nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5), reproj_px={"nose": 20.0, "tail": 20.0})
        self.assertIsNone(s.xyz)

    def test_wand(self) -> None:
        s = self.f.update(1.0, wand=(0.3, 0.2, 0.9))
        np.testing.assert_allclose(s.wand_xyz, [0.3, 0.2, 0.9])
        self.assertTrue(s.wand_ok)
        self.assertIsNone(s.xyz)
        self.assertFalse(s.tracking_ok)                            # the drone was not seen
        self.assertFalse(self.f.state(1.5).wand_ok)
        s = self.f.update(1.033, wand=(2.3, 0.2, 0.9))             # wand jump rejected too
        np.testing.assert_allclose(s.wand_xyz, [0.3, 0.2, 0.9])

    def test_2d_only_mode_is_never_ok(self) -> None:
        f = PoseFuser(self.cfg, mode="2d-only")
        s = f.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5), wand=(0.0, 0.0, 1.0))
        self.assertFalse(s.tracking_ok)
        self.assertFalse(s.wand_ok)
        self.assertEqual(s.mode, "2d-only")

    def test_nan_fix_ignored(self) -> None:
        s = self.f.update(1.0, nose=(float("nan"), 0.0, 0.5), tail=(-0.1, 0.0, 0.5))
        np.testing.assert_allclose(s.xyz, [-0.1, 0.0, 0.5])

    def test_state_is_json_friendly(self) -> None:
        s = self.f.update(1.0, nose=(0.1, 0.0, 0.5), tail=(-0.1, 0.0, 0.5), wand=(0.3, 0.2, 0.9), fps=30.0, latency_ms=12.0)
        d = json.loads(json.dumps(s.as_dict()))
        self.assertEqual(d["xyz"], [0.0, 0.0, 0.5])
        self.assertEqual(d["fps"], 30.0)
        self.assertIsInstance(s, TrackerState)
        self.assertIn("ok=", format_state(s, [], 1.0))

    def test_jump_gate_unit(self) -> None:
        g = JumpGate(0.5, 3)
        self.assertTrue(g.check(np.zeros(3)))
        self.assertFalse(g.check(np.array([1.0, 0, 0])))
        self.assertTrue(g.check(np.array([0.4, 0, 0])))
        self.assertEqual(g.rejects, 0)
        g.reset()
        self.assertTrue(g.check(np.array([5.0, 5, 5])))


# ------------------------------------------------------------------ detector


def blank(h: int = 480, w: int = 640, grey: int = 60) -> np.ndarray:
    return np.full((h, w, 3), grey, np.uint8)


class TestMarkerDetector(unittest.TestCase):
    def test_green_square_centroid_and_area(self) -> None:
        img = blank()
        img[300:320, 400:420] = (0, 255, 0)
        d = MarkerDetector((45, 80, 80), (80, 255, 255)).detect_bgr(img)
        self.assertIsNotNone(d)
        u, v, area = d
        self.assertAlmostEqual(u, 409.5, delta=0.6)
        self.assertAlmostEqual(v, 309.5, delta=0.6)
        self.assertTrue(300 <= area <= 400, area)                  # contour area of a 20x20 block ~ 361

    def test_wrong_colour_not_detected(self) -> None:
        img = blank()
        img[300:320, 400:420] = (0, 255, 0)
        self.assertIsNone(MarkerDetector((100, 100, 80), (130, 255, 255)).detect_bgr(img))

    def test_red_wraparound_range(self) -> None:
        img = blank()
        img[100:130, 100:130] = (40, 0, 255)                       # hue ~175 (magenta-red), large
        img[400:410, 500:510] = (0, 0, 255)                        # hue 0 (pure red), small
        wrap = MarkerDetector((170, 100, 80), (10, 255, 255))
        self.assertTrue(wrap.wraps)
        d = wrap.detect_bgr(img)
        self.assertAlmostEqual(d[0], 114.5, delta=0.6)             # largest blob wins
        self.assertAlmostEqual(d[1], 114.5, delta=0.6)
        # both hue bands are in the mask (the 3x3 open clips only the block corners)
        self.assertTrue(np.all(wrap.mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[402:408, 502:508] == 255))
        self.assertTrue(np.all(wrap.mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[105:125, 105:125] == 255))
        low_only = MarkerDetector((0, 100, 80), (10, 255, 255), min_area=10)
        d = low_only.detect_bgr(img)
        self.assertAlmostEqual(d[0], 504.5, delta=0.6)             # the hue-175 blob is excluded
        high_only = MarkerDetector((170, 100, 80), (179, 255, 255))
        self.assertAlmostEqual(high_only.detect_bgr(img)[0], 114.5, delta=0.6)

    def test_min_area(self) -> None:
        img = blank()
        img[200:203, 200:203] = (0, 255, 0)
        self.assertIsNone(MarkerDetector((45, 80, 80), (80, 255, 255), min_area=30, open_kernel=0).detect_bgr(img))
        self.assertIsNotNone(MarkerDetector((45, 80, 80), (80, 255, 255), min_area=1, open_kernel=0).detect_bgr(img))

    def test_open_removes_speckle(self) -> None:
        img = blank()
        img[300:320, 400:420] = (0, 255, 0)
        rng = np.random.default_rng(0)
        for _ in range(200):                                       # single-pixel green speckle
            y, x = rng.integers(0, 480), rng.integers(0, 640)
            img[y, x] = (0, 255, 0)
        d = MarkerDetector((45, 80, 80), (80, 255, 255), open_kernel=3).detect_bgr(img)
        self.assertAlmostEqual(d[0], 409.5, delta=1.0)
        self.assertAlmostEqual(d[1], 309.5, delta=1.0)

    def test_default_ranges_match_pure_colours(self) -> None:
        cfg = TrackerConfig()
        img = blank()
        img[100:120, 100:120] = (0, 0, 255)
        img[200:220, 300:320] = (255, 0, 0)
        img[300:320, 500:520] = (0, 255, 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dets = {n: MarkerDetector.from_range(r, cfg).detect(hsv) for n, r in cfg.colors.items()}
        self.assertAlmostEqual(dets["nose"][0], 109.5, delta=0.6)
        self.assertAlmostEqual(dets["tail"][0], 309.5, delta=0.6)
        self.assertAlmostEqual(dets["wand"][0], 509.5, delta=0.6)


# -------------------------------------------------------------------- config


class TestConfig(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        cfg = TrackerConfig(camera_indices=(1, 2), backend="MSMF", max_jump_m=0.7)
        cfg.colors["nose"] = HsvRange((5, 6, 7), (8, 9, 10))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "tracker.json")
            cfg.to_json(p)
            cfg2 = TrackerConfig.from_json(p)
        self.assertEqual(cfg2.camera_indices, (1, 2))
        self.assertEqual(cfg2.backend, "MSMF")
        self.assertEqual(cfg2.max_jump_m, 0.7)
        self.assertEqual(cfg2.colors["nose"], HsvRange((5, 6, 7), (8, 9, 10)))
        self.assertEqual(cfg2.colors["tail"], TrackerConfig().colors["tail"])
        self.assertEqual(cfg2.board_size, (9, 6))
        self.assertEqual(cfg2.backend_id, cv2.CAP_MSMF)

    def test_save_color_merges_into_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tracker.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"max_reproj_px": 6.0, "colors": {"nose": {"lo": [1, 2, 3], "hi": [4, 5, 6]}}}, f)
            TrackerConfig.save_color("tail", (100, 90, 80), (130, 255, 255), p)
            TrackerConfig.save_color("nose", (170, 90, 80), (10, 255, 255), p)
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["max_reproj_px"], 6.0)
            self.assertEqual(set(raw["colors"]), {"nose", "tail"})
            cfg = TrackerConfig.from_json(p)
        self.assertEqual(cfg.max_reproj_px, 6.0)
        self.assertEqual(cfg.colors["tail"], HsvRange((100, 90, 80), (130, 255, 255)))
        self.assertTrue(cfg.colors["nose"].wraps)
        self.assertEqual(cfg.colors["wand"], TrackerConfig().colors["wand"])   # default kept

    def test_save_color_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "new", "tracker.json")
            TrackerConfig.save_color("wand", (40, 50, 60), (80, 255, 255), p)
            self.assertEqual(TrackerConfig.from_json(p).colors["wand"], HsvRange((40, 50, 60), (80, 255, 255)))

    def test_bad_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrackerConfig.from_dict({"backend": "V4L2"})

    def test_load_missing_file_gives_defaults(self) -> None:
        cfg = TrackerConfig.load(os.path.join(tempfile.gettempdir(), "definitely_missing_tracker.json"))
        self.assertEqual(cfg, TrackerConfig())


# ------------------------------------------------------- calibration helpers


class TestCalibrationHelpers(unittest.TestCase):
    def test_marker_corner_model(self) -> None:
        obj = CE.marker_object_points(0.15)
        np.testing.assert_allclose(obj[0], [0.075, 0.075, 0.0])    # TL = forward-left
        np.testing.assert_allclose(obj[1], [0.075, -0.075, 0.0])   # TR = forward-right
        np.testing.assert_allclose(obj[2], [-0.075, -0.075, 0.0])  # BR = back-right
        np.testing.assert_allclose(obj[3], [-0.075, 0.075, 0.0])   # BL = back-left
        self.assertTrue(np.all(obj[:, 2] == 0))

    def test_solve_extrinsics_recovers_synthetic_pose(self) -> None:
        cam_a, _ = synth_rig()
        corners = cam_a.project_points(CE.marker_object_points(0.15))
        res = CE.solve_extrinsics(corners, K, DIST, 0.15)
        np.testing.assert_allclose(res.camera_position, cam_a.position, atol=2e-3)
        np.testing.assert_allclose(res.R, cam_a.R, atol=1e-3)
        self.assertLess(res.reproj_px, 0.01)
        cam = Camera(0, K=K, dist=DIST, R=res.R, t=res.t)
        self.assertGreater(cam.position[2], 0.0)
        self.assertLess(cam.forward[2], 0.0)

    def test_solve_extrinsics_with_pixel_noise(self) -> None:
        cam_a, _ = synth_rig()
        rng = np.random.default_rng(3)
        obj = CE.marker_object_points(0.15)
        for _ in range(10):
            corners = cam_a.project_points(obj) + rng.normal(0.0, 0.2, (4, 2))
            res = CE.solve_extrinsics(corners, K, DIST, 0.15)
            self.assertGreater(res.camera_position[2], 0.0)          # never the below-floor solution
            self.assertLess(np.linalg.norm(res.camera_position - cam_a.position), 0.15)

    def test_detect_generated_marker_corner_order(self) -> None:
        """cv2.aruco corner order is TL,TR,BR,BL of the marker as generated (top edge up)."""
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = np.full((480, 640), 255, np.uint8)
        img[90:390, 170:470] = cv2.aruco.generateImageMarker(d, 0, 300)
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(img)
        self.assertEqual(ids.reshape(-1).tolist(), [0])
        c = corners[0].reshape(4, 2)
        np.testing.assert_allclose(c[0], [170, 90], atol=1.5)        # TL
        np.testing.assert_allclose(c[1], [469, 90], atol=1.5)        # TR
        np.testing.assert_allclose(c[2], [469, 389], atol=1.5)       # BR
        np.testing.assert_allclose(c[3], [170, 389], atol=1.5)       # BL

    def test_intrinsics_roundtrip(self) -> None:
        cols, rows, sq = 9, 6, 0.024
        dist_true = np.array([-0.12, 0.04, 0.0005, -0.0003, 0.0])
        objp = CI.board_object_points(cols, rows, sq)
        self.assertEqual(objp.shape, (54, 3))
        np.testing.assert_allclose(objp[1], [sq, 0, 0])
        np.testing.assert_allclose(objp[cols], [0, sq, 0])
        centre = objp.mean(axis=0)
        rng = np.random.default_rng(7)
        objpoints, imgpoints = [], []
        targets = [(u, v) for u in (120, 320, 520) for v in (100, 240, 380)] + [(320, 240)] * 6
        for (u, v) in targets:
            for _ in range(50):
                rvec = rng.uniform(-0.5, 0.5, 3) * np.array([1.0, 1.0, 0.6])
                R, _ = cv2.Rodrigues(rvec)
                z = rng.uniform(0.35, 0.6)
                tvec = z * np.array([(u - 320) / 600.0, (v - 240) / 600.0, 1.0]) - R @ centre
                px, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_true)
                px = px.reshape(-1, 2)
                if np.all(px[:, 0] > 5) and np.all(px[:, 0] < 635) and np.all(px[:, 1] > 5) and np.all(px[:, 1] < 475):
                    objpoints.append(objp)
                    imgpoints.append(px.reshape(-1, 1, 2).astype(np.float32))
                    break
        self.assertGreaterEqual(len(imgpoints), 12)
        res = CI.calibrate_from_points(objpoints, imgpoints, (640, 480))
        self.assertLess(res.rms, 0.05)
        self.assertAlmostEqual(res.K[0, 0], 600.0, delta=3.0)
        self.assertAlmostEqual(res.K[1, 1], 600.0, delta=3.0)
        self.assertAlmostEqual(res.K[0, 2], 320.0, delta=2.0)
        self.assertAlmostEqual(res.K[1, 2], 240.0, delta=2.0)
        self.assertAlmostEqual(res.dist[0], -0.12, delta=0.01)
        self.assertEqual(res.dist[4], 0.0)                          # k3 fixed
        self.assertEqual(len(res.per_view_rms), len(imgpoints))
        self.assertGreater(CI.coverage_fraction(imgpoints, (640, 480)), 0.5)

    def test_intrinsics_npz_roundtrip_through_camera_load(self) -> None:
        res = CI.CalibResult(0.3, K, DIST, np.array([0.3]))
        with tempfile.TemporaryDirectory() as d:
            CI.save_intrinsics(Camera.intrinsics_path(3, d), res, (640, 480), 1, "9x6", 0.024)
            cam_a, _ = synth_rig()
            ext = CE.ExtrinsicsResult(cam_a.R, cam_a.t, cam_a.rvec.reshape(3), cam_a.t.reshape(3), 0.1)
            CE.save_extrinsics(Camera.extrinsics_path(3, d), ext, np.zeros((4, 2)), 0.15, 0, 15, 2.0)
            cam = Camera.load(3, d)
        self.assertTrue(cam.calibrated)
        np.testing.assert_allclose(cam.K, K)
        np.testing.assert_allclose(cam.dist, DIST)
        np.testing.assert_allclose(cam.position, cam_a.position, atol=1e-9)
        self.assertEqual(cam.image_size, (640, 480))
        self.assertAlmostEqual(cam.rms, 0.3)

    def test_camera_load_missing_files_is_uncalibrated(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertLogs("tracker", level="WARNING") as cm:
                cam = Camera.load(9, d)
        self.assertFalse(cam.has_intrinsics)
        self.assertFalse(cam.calibrated)
        self.assertTrue(any("calibrate_intrinsics.py --camera 9" in m for m in cm.output))


# ------------------------------------------------------- tracker end-to-end


class _FailingSource(FrameSource):
    def run(self) -> None:
        self.error = "no such camera"
        self._finish()


def render_markers(cam: Camera, markers: dict[str, tuple[float, float, float]]) -> np.ndarray:
    img = blank()
    colours = {"nose": (0, 0, 255), "tail": (255, 0, 0), "wand": (0, 255, 0)}
    for name, p in markers.items():
        u, v = cam.project(p)
        cv2.circle(img, (int(round(u)), int(round(v))), 8, colours[name], -1)
    return img


def wait_for(pred, timeout: float = 3.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


class TestTrackerSynthetic(unittest.TestCase):
    def test_two_synthetic_cameras_give_3d_fix(self) -> None:
        cam_a, cam_b = synth_rig()
        markers = {"nose": (0.1, 0.05, 0.5), "tail": (-0.1, 0.05, 0.5), "wand": (0.2, -0.15, 0.3)}
        for cam in (cam_a, cam_b):                                  # every marker inside both images
            for p in markers.values():
                u, v = cam.project(p)
                self.assertTrue(10 < u < 630 and 10 < v < 470, (cam.index, p, u, v))
        src_a = ImageSource(0, render_markers(cam_a, markers), rate_hz=60)
        src_b = ImageSource(1, render_markers(cam_b, markers), rate_hz=60)
        tr = Tracker(TrackerConfig(), cameras=[cam_a, cam_b], sources=[src_a, src_b])
        tr.start()
        try:
            def ready() -> bool:
                st = tr.get_state()
                return st.tracking_ok and st.wand_ok and st.fps > 10.0     # fps needs a few cycles
            self.assertTrue(wait_for(ready))
            s = tr.get_state()
            self.assertEqual(s.mode, "3d")
            np.testing.assert_allclose(s.xyz, [0.0, 0.05, 0.5], atol=0.01)
            self.assertAlmostEqual(s.yaw, 0.0, delta=0.05)
            self.assertTrue(s.yaw_ok)
            np.testing.assert_allclose(s.wand_xyz, markers["wand"], atol=0.01)
            self.assertLess(s.reproj_px, 2.0)
            self.assertGreater(s.fps, 10.0)
            self.assertLess(s.latency_ms, 200.0)
            views = tr.get_views()
            self.assertEqual(len(views), 2)
            for v in views:
                self.assertIsNotNone(v)
                self.assertTrue(all(v.detections[n] is not None for n in markers))
            st = tr.get_stats()
            self.assertGreater(st["fixes"], 0)
            drawn = draw_view(views[0], s)                          # --show drawing path, no window
            self.assertEqual(drawn.shape, views[0].frame.shape)
            self.assertFalse(np.array_equal(drawn, views[0].frame))
            self.assertIn("cam0", format_state(s, views, 1.0))
        finally:
            tr.stop()
        self.assertFalse(src_a.is_alive())
        self.assertFalse(tr.get_state(time.perf_counter() + 1.0).tracking_ok)    # stale after stop

    def test_one_camera_is_2d_only_but_detects(self) -> None:
        cam_a, _ = synth_rig()
        markers = {"nose": (0.1, 0.0, 0.5), "tail": (-0.1, 0.0, 0.5)}
        src_a = ImageSource(0, render_markers(cam_a, markers), rate_hz=60)
        tr = Tracker(TrackerConfig(), cameras=[cam_a, Camera(1)], sources=[src_a, _FailingSource(1, "cam1")])
        with self.assertLogs("tracker", level="WARNING") as cm:
            tr.start()
            try:
                self.assertTrue(wait_for(lambda: tr.get_views()[0] is not None))
                time.sleep(0.1)
                s = tr.get_state()
                self.assertEqual(s.mode, "2d-only")
                self.assertFalse(s.tracking_ok)
                self.assertIsNone(s.xyz)
                v = tr.get_views()[0]
                self.assertIsNotNone(v.detections["nose"])
                self.assertIsNotNone(v.detections["tail"])
                self.assertIsNone(v.detections["wand"])
                self.assertIsNone(tr.get_views()[1])
            finally:
                tr.stop()
        self.assertTrue(any("2d-only" in m for m in cm.output))
        self.assertIn("cam1: not open", tr.mode_reason)

    def test_uncalibrated_cameras_are_2d_only(self) -> None:
        src = ImageSource(0, blank(), rate_hz=60)
        tr = Tracker(TrackerConfig(), cameras=[Camera(0), Camera(1)], sources=[src, ImageSource(1, blank(), 60)])
        with self.assertLogs("tracker", level="WARNING"):
            tr.start()
        try:
            self.assertTrue(wait_for(lambda: tr.get_views()[0] is not None))
            self.assertEqual(tr.mode, "2d-only")
            self.assertIn("no intrinsics", tr.mode_reason)
        finally:
            tr.stop()


class TestSimTracker(unittest.TestCase):
    def test_sim_replays_demo_path(self) -> None:
        import paths as P

        sim = SimTracker("square", rate_hz=60, seed=1)
        sim.start()
        try:
            self.assertTrue(wait_for(lambda: sim.get_state().tracking_ok, 2.0))
            s = sim.get_state()
        finally:
            sim.stop()
        self.assertEqual(s.mode, "sim")
        self.assertTrue(s.tracking_ok)
        self.assertTrue(s.wand_ok)
        self.assertTrue(P.GEOFENCE.contains(np.array(s.xyz), tol=0.05))
        self.assertAlmostEqual(s.yaw, 0.0, delta=0.15)
        self.assertTrue(s.yaw_ok)
        self.assertGreater(s.fps, 30.0)

    def test_sim_dropout_loses_tracking(self) -> None:
        sim = SimTracker("exit_a", rate_hz=60, dropout_every_s=10.0, dropout_s=10.0)   # always dropped out
        sim.start()
        time.sleep(0.2)
        sim.stop()
        s = sim.get_state()
        self.assertIsNone(s.xyz)
        self.assertFalse(s.tracking_ok)


if __name__ == "__main__":
    unittest.main()
