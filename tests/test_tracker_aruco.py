"""Tests for the single-camera ArUco pose mode (ArucoTracker).  Run from the project root:

    cf64\\Scripts\\python.exe -m unittest tests.test_tracker_aruco -v

The accuracy test prints the expected error of a 7 cm marker seen from ~2 m by a
640x480 camera with 0.5 px corner noise (numbers quoted in the tracker docs).
"""

from __future__ import annotations

import math
import time
import unittest

import cv2
import numpy as np

from tracker import (
    ArucoTracker,
    Camera,
    FrameSource,
    ImageSource,
    MarkerPose,
    PoseFuser,
    TrackerConfig,
    camera_from_look_at,
    draw_view,
    format_state,
    solve_marker_pose,
    square_marker_object_points,
)

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.12, 0.04, 0.0005, -0.0003, 0.01])
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def cam45(distance: float = 2.0, azimuth_deg: float = 0.0, dist: np.ndarray | None = DIST, index: int = 0) -> Camera:
    """Camera ``distance`` metres from the origin, 45 degrees down, at the given azimuth."""
    r, h = distance * math.cos(math.radians(45)), distance * math.sin(math.radians(45))
    az = math.radians(azimuth_deg)
    return camera_from_look_at(index, (r * math.cos(az), r * math.sin(az), h), (0.0, 0.0, 0.0), K, dist)


def marker_world_corners(centre, yaw: float, size: float, tilt=(0.0, 0.0, 0.0)) -> np.ndarray:
    """World corners [TL,TR,BR,BL] of a square marker lying level (then tilted by the
    axis-angle ``tilt``), top edge pointing along ``yaw`` (0 = +x), face up."""
    obj = square_marker_object_points(size)
    top = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    r_wm = np.column_stack([right, top, [0.0, 0.0, 1.0]])
    tilt = np.asarray(tilt, dtype=np.float64)
    if np.any(tilt):
        r_wm = cv2.Rodrigues(tilt)[0] @ r_wm
    return np.asarray(centre, dtype=np.float64) + obj @ r_wm.T


def yaw_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def render_marker(frame: np.ndarray, marker_id: int, corners_px: np.ndarray, marker_px: int = 120, border: int = 20) -> None:
    """Warp a generated marker (with white quiet zone) onto ``frame`` at the given pixel corners (TL,TR,BR,BL)."""
    img = cv2.aruco.generateImageMarker(ARUCO, marker_id, marker_px)
    canvas = np.full((marker_px + 2 * border, marker_px + 2 * border), 255, np.uint8)
    canvas[border:border + marker_px, border:border + marker_px] = img
    s = float(marker_px + border)
    src = np.float32([[border, border], [s, border], [s, s], [border, s]])
    h_mat = cv2.getPerspectiveTransform(src, np.float32(corners_px))
    warped = cv2.warpPerspective(canvas, h_mat, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(np.full(canvas.shape, 255, np.uint8), h_mat, (frame.shape[1], frame.shape[0]),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]


def wait_for(pred, timeout: float = 3.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


# ---------------------------------------------------------------- pose solver


class TestMarkerPoseSolver(unittest.TestCase):
    def test_object_points_are_ippe_square_order(self) -> None:
        obj = square_marker_object_points(0.07)
        np.testing.assert_allclose(obj, [[-0.035, 0.035, 0], [0.035, 0.035, 0], [0.035, -0.035, 0], [-0.035, -0.035, 0]])

    def test_world_corner_helper_matches_floor_marker_convention(self) -> None:
        c = marker_world_corners((0, 0, 0), 0.0, 0.15)
        np.testing.assert_allclose(c[0], [0.075, 0.075, 0])        # TL = forward-left (same as calibrate_extrinsics)
        np.testing.assert_allclose(c[1], [0.075, -0.075, 0])       # TR = forward-right
        c = marker_world_corners((0, 0, 0), math.pi / 2, 0.15)
        np.testing.assert_allclose(c[0], [-0.075, 0.075, 0], atol=1e-12)   # top edge now points +y

    def test_pose_recovery_noise_free(self) -> None:
        cam = cam45(2.0)
        centres = [(0.0, 0.0, 0.5), (0.4, 0.3, 0.3), (-0.3, -0.4, 0.9), (0.2, -0.5, 1.1), (-0.5, 0.4, 0.25)]
        yaws = [0.0, math.pi / 2, math.pi, -math.pi / 2, math.radians(30)]
        tilts = [(0.0, 0.0, 0.0), (math.radians(8), math.radians(5), 0.0)]
        n = 0
        for centre in centres:
            for yaw in yaws:
                for tilt in tilts:
                    corners = cam.project_points(marker_world_corners(centre, yaw, 0.07, tilt))
                    if not (np.all(corners > 2) and np.all(corners[:, 0] < 638) and np.all(corners[:, 1] < 478)):
                        continue
                    mp = solve_marker_pose(cam, corners, 0.07)
                    n += 1
                    pos_err = float(np.linalg.norm(mp.xyz - centre))
                    self.assertLess(pos_err, 0.02, (centre, yaw, tilt, pos_err))
                    self.assertLess(math.degrees(yaw_diff(mp.yaw, yaw)), 3.0, (centre, yaw, tilt, mp.yaw))
                    self.assertTrue(mp.up)
                    self.assertGreater(mp.normal[2], 0.95)
                    self.assertLess(mp.reproj_px, 0.05)
                    self.assertEqual(mp.n_solutions, 2)
        self.assertGreaterEqual(n, 30)

    def test_yaw_definition_top_edge_is_nose(self) -> None:
        cam = cam45(2.0)
        for yaw_deg in (0, 90, 180, -90, 45):
            corners = cam.project_points(marker_world_corners((0, 0, 0.5), math.radians(yaw_deg), 0.07))
            mp = solve_marker_pose(cam, corners, 0.07)
            self.assertLess(math.degrees(yaw_diff(mp.yaw, math.radians(yaw_deg))), 0.5, yaw_deg)
        # rotating the corner list (as if the marker were taped with the wrong edge forward) shifts yaw by 90 deg
        corners = cam.project_points(marker_world_corners((0, 0, 0.5), 0.0, 0.07))
        mp = solve_marker_pose(cam, np.roll(corners, -1, axis=0), 0.07)
        self.assertLess(math.degrees(yaw_diff(mp.yaw, -math.pi / 2)), 0.5)

    def test_facing_up_disambiguation_under_noise(self) -> None:
        cam = cam45(2.0)
        rng = np.random.default_rng(5)
        obj_c = (0.0, 0.0, 0.5)
        flips_without_preference = 0
        for _ in range(150):
            yaw = rng.uniform(-math.pi, math.pi)
            tilt = rng.normal(0.0, math.radians(4), 3)
            true = marker_world_corners(obj_c, yaw, 0.07, tilt)
            corners = cam.project_points(true) + rng.normal(0.0, 0.5, (4, 2))
            mp = solve_marker_pose(cam, corners, 0.07, prefer_up=True)
            self.assertTrue(mp.up)
            self.assertGreater(mp.normal[2], 0.7, mp.normal)              # never the mirrored (near-horizontal) normal
            self.assertLess(float(np.linalg.norm(mp.xyz - obj_c)), 0.25)
            alt = solve_marker_pose(cam, corners, 0.07, prefer_up=False)
            if alt.normal[2] < 0.7:
                flips_without_preference += 1
        print(f"\n  [aruco] lowest-reprojection pick would have taken the mirrored pose in "
              f"{flips_without_preference}/150 noisy frames; facing-up pick: 0/150")

    def test_accuracy_7cm_marker_at_2m_half_pixel_noise(self) -> None:
        """Expected error of the one-camera mode (numbers for the report)."""
        cam = cam45(2.0)
        rng = np.random.default_rng(11)
        centre = np.array([0.0, 0.0, 0.5])
        ray = (centre - cam.position)
        ray /= np.linalg.norm(ray)
        width_px = float(np.ptp(cam.project_points(marker_world_corners(centre, 0.0, 0.07))[:, 0]))
        pos, along, lateral, yaw_e = [], [], [], []
        for _ in range(400):
            yaw = rng.uniform(-math.pi, math.pi)
            corners = cam.project_points(marker_world_corners(centre, yaw, 0.07)) + rng.normal(0.0, 0.5, (4, 2))
            mp = solve_marker_pose(cam, corners, 0.07)
            d = mp.xyz - centre
            pos.append(np.linalg.norm(d))
            along.append(abs(float(d @ ray)))
            lateral.append(float(np.linalg.norm(d - (d @ ray) * ray)))
            yaw_e.append(math.degrees(yaw_diff(mp.yaw, yaw)))
        p = {k: (float(np.mean(v)), float(np.percentile(v, 95))) for k, v in
             (("total", pos), ("along-ray", along), ("lateral", lateral), ("yaw deg", yaw_e))}
        print(f"\n  [aruco] 7 cm marker, {np.linalg.norm(centre - cam.position):.2f} m from the camera "
              f"({width_px:.0f} px wide), 0.5 px noise, 400 trials:  mean / p95")
        for k, (m, p95) in p.items():
            unit = "" if "deg" in k else " m"
            print(f"      {k:10s} {m:7.3f} / {p95:7.3f}{unit}")
        self.assertLess(p["lateral"][0], 0.01)                        # lateral is millimetric
        self.assertLess(p["total"][0], 0.10)                          # depth-dominated, decimetre-class at worst
        self.assertLess(p["yaw deg"][0], 10.0)

    def test_closer_camera_is_more_accurate(self) -> None:
        rng = np.random.default_rng(2)
        centre = np.array([0.0, 0.0, 0.5])

        def mean_err(cam: Camera) -> float:
            errs = []
            for _ in range(150):
                corners = cam.project_points(marker_world_corners(centre, 0.3, 0.07)) + rng.normal(0.0, 0.5, (4, 2))
                errs.append(np.linalg.norm(solve_marker_pose(cam, corners, 0.07).xyz - centre))
            return float(np.mean(errs))

        self.assertLess(mean_err(cam45(1.5)), mean_err(cam45(2.5)))

    def test_uncalibrated_camera_raises(self) -> None:
        with self.assertRaises(ValueError):
            solve_marker_pose(Camera(0, K=K), np.zeros((4, 2)), 0.07)


class TestPoseFuserUpdatePose(unittest.TestCase):
    def test_update_pose_and_gates(self) -> None:
        f = PoseFuser(TrackerConfig(), mode="aruco")
        s = f.update_pose(1.0, (0.1, 0.2, 0.5), 0.3, (0.5, 0.0, 0.9), {"drone": 0.4, "wand": 0.2}, fps=30.0, latency_ms=8.0)
        np.testing.assert_allclose(s.xyz, [0.1, 0.2, 0.5])
        self.assertAlmostEqual(s.yaw, 0.3)
        self.assertTrue(s.yaw_ok and s.tracking_ok and s.wand_ok)
        self.assertEqual(s.mode, "aruco")
        self.assertAlmostEqual(s.reproj_px, 0.4)
        s = f.update_pose(1.033, (2.1, 0.2, 0.5), 0.3, None, {"drone": 0.4})      # 2 m jump
        np.testing.assert_allclose(s.xyz, [0.1, 0.2, 0.5])
        self.assertEqual(s.rejects, 1)
        s = f.update_pose(1.066, (0.1, 0.2, 0.5), 1.0, None, {"drone": 9.0})      # bad corners
        self.assertAlmostEqual(s.yaw, 0.3)
        self.assertEqual(s.t, 1.0)
        s = f.update_pose(1.1, None, None, None, {})
        self.assertEqual(s.t, 1.0)
        self.assertFalse(f.state(1.4).tracking_ok)
        self.assertTrue(f.state(1.29).tracking_ok)


# ------------------------------------------------------------- end to end


class _FailingSource(FrameSource):
    def run(self) -> None:
        self.error = "no such camera"
        self._finish()


class TestArucoTrackerSynthetic(unittest.TestCase):
    def make_scene(self, cam: Camera, cfg: TrackerConfig):
        drone_c, drone_yaw = np.array([0.0, 0.0, 0.4]), math.radians(30)
        wand_c = np.array([0.2, -0.3, 0.3])
        frame = np.full((480, 640, 3), 128, np.uint8)
        drone_px = cam.project_points(marker_world_corners(drone_c, drone_yaw, cfg.drone_marker_size_m))
        wand_px = cam.project_points(marker_world_corners(wand_c, math.radians(-60), cfg.wand_marker_size_m,
                                                           (math.radians(10), 0.0, 0.0)))
        for px in (drone_px, wand_px):                               # well inside the frame, quiet zone included
            self.assertTrue(np.all(px > 25) and np.all(px[:, 0] < 615) and np.all(px[:, 1] < 455), px)
        render_marker(frame, cfg.drone_marker_id, drone_px)
        render_marker(frame, cfg.wand_marker_id, wand_px)
        return frame, drone_c, drone_yaw, wand_c, drone_px

    def test_rendered_marker_is_detected_with_canonical_corner_order(self) -> None:
        cfg = TrackerConfig()
        cam = cam45(1.2)
        frame, _, _, _, drone_px = self.make_scene(cam, cfg)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX      # as ArucoTracker (default config)
        det = cv2.aruco.ArucoDetector(ARUCO, params)
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.reshape(-1))}
        self.assertEqual(set(found), {cfg.drone_marker_id, cfg.wand_marker_id})
        np.testing.assert_allclose(found[cfg.drone_marker_id], drone_px, atol=1.5)   # same TL,TR,BR,BL order

    def test_end_to_end_aruco_tracker(self) -> None:
        cfg = TrackerConfig()
        cam = cam45(1.2)
        frame, drone_c, drone_yaw, wand_c, _ = self.make_scene(cam, cfg)
        src = ImageSource(0, frame, rate_hz=60)
        tr = ArucoTracker(cfg, camera=cam, source=src)
        tr.start()
        try:
            def ready() -> bool:
                st = tr.get_state()
                return st.tracking_ok and st.wand_ok and st.fps > 10.0
            self.assertTrue(wait_for(ready), tr.get_state())
            s = tr.get_state()
            self.assertEqual(s.mode, "aruco")
            self.assertEqual(tr.mode, "aruco")
            np.testing.assert_allclose(s.xyz, drone_c, atol=0.03)
            self.assertLess(math.degrees(yaw_diff(s.yaw, drone_yaw)), 3.0)
            self.assertTrue(s.yaw_ok)
            np.testing.assert_allclose(s.wand_xyz, wand_c, atol=0.03)
            self.assertLess(s.reproj_px, 1.5)
            self.assertLess(s.latency_ms, 200.0)
            v = tr.get_views()[0]
            self.assertIsNotNone(v)
            self.assertEqual(set(v.markers), {cfg.drone_marker_id, cfg.wand_marker_id})
            self.assertEqual(set(v.poses), {cfg.drone_marker_id, cfg.wand_marker_id})
            self.assertIn(f"id{cfg.drone_marker_id}", v.detections)
            st = tr.get_stats()
            self.assertGreater(st["fixes"], 0)
            self.assertGreater(st["wand_fixes"], 0)
            self.assertIn(str(cfg.drone_marker_id), st["seen"])
            drawn = draw_view(v, s, cam, 0.05)                        # --show path incl. drawFrameAxes
            self.assertEqual(drawn.shape, frame.shape)
            self.assertFalse(np.array_equal(drawn, frame))
            self.assertIn("id1", format_state(s, [v], 1.0, [0]))
        finally:
            tr.stop()
        self.assertFalse(tr.get_state(time.perf_counter() + 1.0).tracking_ok)

    def test_uncalibrated_camera_is_2d_only_but_reports_ids(self) -> None:
        cfg = TrackerConfig()
        frame, *_ = self.make_scene(cam45(1.2), cfg)
        tr = ArucoTracker(cfg, camera=Camera(0), source=ImageSource(0, frame, rate_hz=60))
        with self.assertLogs("tracker", level="WARNING") as cm:
            tr.start()
            try:
                self.assertTrue(wait_for(lambda: tr.get_views()[0] is not None))
                s = tr.get_state()
                v = tr.get_views()[0]
            finally:
                tr.stop()
        self.assertEqual(s.mode, "2d-only")
        self.assertFalse(s.tracking_ok)
        self.assertIsNone(s.xyz)
        self.assertIn("id1", v.detections)
        self.assertIn("id2", v.detections)
        self.assertEqual(v.poses, {})
        self.assertIn("cam0: no intrinsics", tr.mode_reason)
        self.assertTrue(any("2d-only" in m for m in cm.output))

    def test_missing_camera_is_2d_only(self) -> None:
        tr = ArucoTracker(TrackerConfig(), camera=cam45(1.2), source=_FailingSource(0, "cam0"))
        with self.assertLogs("tracker", level="WARNING"):
            tr.start()
        try:
            time.sleep(0.15)
            self.assertEqual(tr.mode, "2d-only")
            self.assertIn("cam0: not open", tr.mode_reason)
            self.assertFalse(tr.get_state().tracking_ok)
        finally:
            tr.stop()

    def test_marker_with_other_id_is_ignored_for_pose(self) -> None:
        cam = cam45(1.2)
        frame, *_ = self.make_scene(cam, TrackerConfig())             # scene has ids 1 (drone) and 2 (wand)
        cfg = TrackerConfig(drone_marker_id=7)                        # tracker expects id 7 -> no drone pose
        tr = ArucoTracker(cfg, camera=cam, source=ImageSource(0, frame, rate_hz=60))
        tr.start()
        try:
            self.assertTrue(wait_for(lambda: tr.get_state().wand_ok))
            s = tr.get_state()
            self.assertIsNone(s.xyz)
            self.assertFalse(s.tracking_ok)
            self.assertTrue(s.wand_ok)
        finally:
            tr.stop()


if __name__ == "__main__":
    unittest.main()
