"""Unit tests for the path engine.  Run from the project root:

    cf\\Scripts\\python.exe -m unittest tests.test_paths -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np

import paths as P
from paths import (
    GEOFENCE,
    Box,
    Path,
    PathRecorder,
    Waypoint,
    clamp_to_box,
    clean_path,
    demo_paths,
    fit_to_box,
    list_paths,
    load_path,
    path_length,
    resample_by_distance,
    retime,
    save_path,
    slugify,
    smooth,
    to_waypoints,
)


def spiral(n: int = 300, turns: float = 2.0, r: float = 0.4, z0: float = 0.4, z1: float = 1.0) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi * turns, n)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang), np.linspace(z0, z1, n)])


def seg_lengths(pts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


class TestBox(unittest.TestCase):
    def test_geofence_dimensions(self) -> None:
        np.testing.assert_allclose(GEOFENCE.size, [1.5, 1.5, 1.0])
        self.assertEqual(GEOFENCE.zmin, 0.2)
        self.assertEqual(GEOFENCE.zmax, 1.2)

    def test_contains_and_clamp(self) -> None:
        inside = np.array([[0.0, 0.0, 0.5], [0.75, -0.75, 1.2]])
        outside = np.array([[0.0, 0.0, 0.5], [0.8, 0.0, 0.5]])
        self.assertTrue(GEOFENCE.contains(inside))
        self.assertFalse(GEOFENCE.contains(outside))
        self.assertTrue(GEOFENCE.contains([0.1, 0.1, 0.3]))
        np.testing.assert_array_equal(GEOFENCE.inside(outside), [True, False])
        np.testing.assert_allclose(GEOFENCE.clamp([2.0, -2.0, 0.0]), [0.75, -0.75, 0.2])
        clamped = GEOFENCE.clamp(outside)
        self.assertEqual(clamped.shape, (2, 3))
        self.assertTrue(GEOFENCE.contains(clamped))

    def test_shrink_and_validation(self) -> None:
        inner = GEOFENCE.shrink(0.05)
        self.assertAlmostEqual(inner.xmin, -0.7)
        self.assertAlmostEqual(inner.zmax, 1.15)
        with self.assertRaises(ValueError):
            GEOFENCE.shrink(0.6)
        with self.assertRaises(ValueError):
            Box(0, 0, 0, 1, 0, 1)
        self.assertEqual(Box.from_dict(GEOFENCE.to_dict()), GEOFENCE)


class TestRecorder(unittest.TestCase):
    def test_drops_near_duplicates_and_nonfinite(self) -> None:
        rec = PathRecorder(min_step=0.01)
        self.assertTrue(rec.add(0.00, 0.0, 0.0, 0.5))
        self.assertFalse(rec.add(0.03, 0.005, 0.0, 0.5))          # closer than min_step
        self.assertFalse(rec.add(0.06, float("nan"), 0.0, 0.5))   # NaN
        self.assertFalse(rec.add(0.09, 0.1, float("inf"), 0.5))   # inf
        self.assertFalse(rec.add(float("nan"), 0.2, 0.0, 0.5))    # bad timestamp
        self.assertTrue(rec.add(0.12, 0.02, 0.0, 0.5))
        self.assertTrue(rec.add(0.15, 0.04, 0.0, 0.5))
        self.assertEqual(len(rec), 3)
        self.assertEqual(rec.dropped, 4)
        self.assertEqual(rec.points.shape, (3, 3))
        np.testing.assert_allclose(rec.times, [0.00, 0.12, 0.15])
        self.assertEqual(rec.last, (0.04, 0.0, 0.5))

    def test_empty_and_clear(self) -> None:
        rec = PathRecorder()
        self.assertEqual(rec.points.shape, (0, 3))
        self.assertEqual(rec.times.shape, (0,))
        self.assertIsNone(rec.last)
        rec.extend([(0.0, 0.0, 0.0, 0.5), (0.1, 0.1, 0.0, 0.5)])
        self.assertEqual(len(rec), 2)
        rec.clear()
        self.assertEqual(len(rec), 0)
        self.assertEqual(rec.dropped, 0)


class TestSmooth(unittest.TestCase):
    def test_reduces_noise_on_spiral(self) -> None:
        rng = np.random.default_rng(42)
        truth = spiral(n=300)
        noisy = truth + rng.normal(0.0, 0.02, truth.shape)
        smoothed = smooth(noisy, window=11, polyorder=3)
        self.assertEqual(smoothed.shape, truth.shape)
        self.assertLess(rms(smoothed, truth), 0.6 * rms(noisy, truth))

    def test_never_raises_on_short_paths(self) -> None:
        rng = np.random.default_rng(0)
        for n in (0, 1, 2, 3, 5, 10):
            pts = rng.normal(size=(n, 3))
            out = smooth(pts, window=11, polyorder=3)
            self.assertEqual(out.shape, (n, 3))
            self.assertTrue(np.all(np.isfinite(out)))
            if n < 3:
                np.testing.assert_array_equal(out, pts)
        # even window / polyorder >= window also degrade gracefully
        pts = rng.normal(size=(7, 3))
        self.assertEqual(smooth(pts, window=4, polyorder=3).shape, (7, 3))
        self.assertEqual(smooth(pts, window=3, polyorder=5).shape, (7, 3))

    def test_straight_line_is_preserved(self) -> None:
        line = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.full(50, 0.7)])
        np.testing.assert_allclose(smooth(line), line, atol=1e-9)


class TestResample(unittest.TestCase):
    def test_spacing_and_endpoints_on_2m_path(self) -> None:
        rng = np.random.default_rng(1)
        # irregularly sampled straight 2 m line
        s = np.sort(np.concatenate([[0.0, 2.0], rng.uniform(0.0, 2.0, 60)]))
        line = np.column_stack([s, np.zeros_like(s), np.full_like(s, 0.6)])
        out = resample_by_distance(line, spacing=0.05)
        d = seg_lengths(out)
        self.assertTrue(np.all(np.abs(d - 0.05) <= 0.005), msg=f"spacing range {d.min()}..{d.max()}")
        np.testing.assert_array_equal(out[0], line[0])
        np.testing.assert_array_equal(out[-1], line[-1])
        self.assertAlmostEqual(path_length(out), 2.0, places=9)

        # gently curved 2 m path (fine input sampling)
        u = np.linspace(0.0, 1.0, 2000)
        curve = np.column_stack([1.9 * u, 0.15 * np.sin(2 * np.pi * u), 0.7 + 0.05 * u])
        curve *= 2.0 / path_length(curve)
        out = resample_by_distance(curve, spacing=0.05)
        d = seg_lengths(out)
        self.assertTrue(np.all(np.abs(d - 0.05) <= 0.005))
        np.testing.assert_array_equal(out[0], curve[0])
        np.testing.assert_array_equal(out[-1], curve[-1])

    def test_duplicates_and_degenerate_inputs(self) -> None:
        pts = np.array([[0, 0, 0.5], [0, 0, 0.5], [1, 0, 0.5], [1, 0, 0.5], [1, 0, 0.5], [2, 0, 0.5]], float)
        out = resample_by_distance(pts, spacing=0.5)
        self.assertEqual(len(out), 5)
        self.assertTrue(np.all(seg_lengths(out) > 0))
        self.assertEqual(resample_by_distance(np.zeros((0, 3))).shape, (0, 3))
        self.assertEqual(resample_by_distance([[0.1, 0.2, 0.3]]).shape, (1, 3))
        same = resample_by_distance([[0.1, 0.2, 0.3]] * 5)
        self.assertEqual(same.shape, (1, 3))
        # spacing longer than the path -> just the endpoints
        out = resample_by_distance([[0, 0, 0.5], [0.3, 0, 0.5]], spacing=10.0)
        self.assertEqual(out.shape, (2, 3))
        with self.assertRaises(ValueError):
            resample_by_distance(pts, spacing=0.0)


class TestFitToBox(unittest.TestCase):
    def test_wide_path_is_scaled_uniformly_into_shrunk_box(self) -> None:
        u = np.linspace(0.0, 1.0, 200)
        wide = np.column_stack([-1.5 + 3.0 * u, 0.5 * np.sin(2 * np.pi * u), 0.7 + 0.2 * np.cos(4 * np.pi * u)])
        fitted, scale, offset = fit_to_box(wide, box=GEOFENCE, margin=0.05)
        inner = GEOFENCE.shrink(0.05)
        self.assertTrue(inner.contains(fitted))
        self.assertIsInstance(scale, float)
        self.assertLess(scale, 1.0)
        ext_in = wide.max(axis=0) - wide.min(axis=0)
        ext_out = fitted.max(axis=0) - fitted.min(axis=0)
        np.testing.assert_allclose(ext_out / ext_out[0], ext_in / ext_in[0], atol=1e-6)
        np.testing.assert_allclose(fitted, scale * wide + offset, atol=1e-9)
        # the limiting axis is used to its full extent
        self.assertAlmostEqual(ext_out[0], inner.size[0], places=9)

    def test_small_path_is_not_inflated(self) -> None:
        small = np.column_stack([np.linspace(0.0, 0.3, 30), np.zeros(30), np.full(30, 0.7)])
        fitted, scale, offset = fit_to_box(small)
        self.assertEqual(scale, 1.0)
        np.testing.assert_allclose(offset, 0.0)
        np.testing.assert_allclose(fitted, small)

    def test_small_path_outside_is_only_translated(self) -> None:
        small = np.column_stack([np.linspace(1.0, 1.3, 30), np.zeros(30), np.full(30, 1.6)])
        fitted, scale, offset = fit_to_box(small, margin=0.05)
        self.assertEqual(scale, 1.0)
        self.assertTrue(GEOFENCE.shrink(0.05).contains(fitted))
        np.testing.assert_allclose(fitted - small, np.tile(offset, (30, 1)), atol=1e-12)
        self.assertAlmostEqual(fitted[:, 0].max(), 0.7)      # pushed just inside +x
        self.assertAlmostEqual(fitted[:, 2].max(), 1.15)     # pushed just below the ceiling
        self.assertEqual(offset[1], 0.0)                     # y already fit: untouched

    def test_non_uniform_scaling_option(self) -> None:
        wide = np.column_stack([np.linspace(-2, 2, 50), np.linspace(-0.1, 0.1, 50), np.full(50, 0.7)])
        fitted, scale, _ = fit_to_box(wide, preserve_aspect=False)
        self.assertEqual(np.shape(scale), (3,))
        self.assertTrue(GEOFENCE.shrink(0.05).contains(fitted))
        self.assertLess(scale[0], 1.0)
        self.assertEqual(scale[1], 1.0)

    def test_clamp_to_box(self) -> None:
        pts = np.array([[5.0, -5.0, 9.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.5]])
        out = clamp_to_box(pts)
        np.testing.assert_allclose(out[0], [0.75, -0.75, 1.2])
        np.testing.assert_allclose(out[1], [0.0, 0.0, 0.2])
        np.testing.assert_allclose(out[2], pts[2])


class TestRetime(unittest.TestCase):
    def test_monotonic_constant_speed(self) -> None:
        pts = resample_by_distance(spiral(), spacing=0.05)
        t = retime(pts, speed=0.3)
        self.assertEqual(t.shape, (len(pts),))
        self.assertEqual(t[0], 0.0)
        self.assertTrue(np.all(np.diff(t) > 0))
        mean_speed = path_length(pts) / t[-1]
        self.assertLess(abs(mean_speed - 0.3) / 0.3, 0.01)
        seg_speed = seg_lengths(pts) / np.diff(t)
        np.testing.assert_allclose(seg_speed, 0.3, rtol=1e-9)
        with self.assertRaises(ValueError):
            retime(pts, speed=0.0)
        self.assertEqual(retime(np.zeros((0, 3))).shape, (0,))


class TestWaypoints(unittest.TestCase):
    def test_spacing_durations_and_total_time(self) -> None:
        pts = resample_by_distance(spiral(), spacing=0.02)
        speed, dt = 0.3, 0.4
        wps = to_waypoints(pts, speed=speed, dt=dt)
        self.assertTrue(all(isinstance(w, Waypoint) for w in wps))
        xyz = np.array([w.xyz for w in wps])
        d = seg_lengths(xyz)
        self.assertTrue(np.all(np.abs(d - speed * dt) <= 0.1 * speed * dt), msg=f"{d.min()}..{d.max()}")
        durations = np.array([w.duration for w in wps])
        self.assertTrue(np.all(durations > 0))
        self.assertTrue(np.all(durations >= P.MIN_GO_TO_DURATION))
        self.assertEqual(wps[0].duration, P.MIN_GO_TO_DURATION)
        np.testing.assert_allclose(np.diff([w.t for w in wps]), durations[1:])
        self.assertAlmostEqual(wps[0].t, wps[0].duration)
        expected_total = path_length(xyz) / speed + wps[0].duration
        self.assertAlmostEqual(wps[-1].t, expected_total, delta=1e-6)
        # the waypoint polyline is a chord approximation of the path
        self.assertLess(abs(path_length(xyz) - path_length(pts)) / path_length(pts), 0.02)
        np.testing.assert_allclose(xyz[0], pts[0])
        np.testing.assert_allclose(xyz[-1], pts[-1])

    def test_start_position_sets_first_duration(self) -> None:
        pts = np.array([[0.0, 0.0, 0.5], [0.6, 0.0, 0.5]])
        wps = to_waypoints(pts, speed=0.3, dt=0.4, start=[0.0, 0.0, 0.2])
        self.assertAlmostEqual(wps[0].duration, 1.0)
        self.assertEqual(to_waypoints(np.zeros((0, 3))), [])
        with self.assertRaises(ValueError):
            to_waypoints(pts, speed=0.0)


class TestCleanPath(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)

    def test_spell_mode_fits_large_shape(self) -> None:
        raw = spiral(n=400, r=1.5, z0=0.0, z1=2.5) + self.rng.normal(0, 0.01, (400, 3))
        path = clean_path(raw, mode="spell", name="Big Spiral")
        self.assertIsInstance(path, Path)
        self.assertTrue(GEOFENCE.contains(path.points))
        self.assertTrue(GEOFENCE.shrink(0.05).contains(path.points, tol=1e-6))
        self.assertEqual(path.mode, "spell")
        self.assertEqual(len(path.points), len(path.times))
        self.assertTrue(np.all(np.diff(path.times) > 0))
        self.assertTrue(path.meta["fitted"])
        self.assertLess(path.meta["scale"], 1.0)
        self.assertEqual(len(path.meta["offset"]), 3)
        self.assertEqual(path.meta["source_samples"], 400)
        self.assertIn("created", path.meta)
        self.assertAlmostEqual(path.meta["length_m"], path.length)
        self.assertAlmostEqual(path.meta["duration_s"], path.duration)
        self.assertAlmostEqual(path.length / path.duration, 0.3, places=9)

    def test_guide_mode_clamps_at_true_scale(self) -> None:
        # real-scale L route with noise, partly poking out of the volume
        route = resample_by_distance([[-0.6, -0.4, 0.6], [0.6, -0.4, 0.6], [0.6, 0.5, 1.4]], spacing=0.02)
        raw = route + self.rng.normal(0, 0.01, route.shape)
        raw[10] = [np.nan, 0.0, 0.5]  # a tracking-lost sample
        path = clean_path(raw, mode="guide", name="exit a")
        self.assertTrue(GEOFENCE.contains(path.points))
        self.assertFalse(path.meta["fitted"])
        self.assertNotIn("scale", path.meta)
        self.assertGreater(path.meta["clamped_points"], 0)
        self.assertEqual(path.meta["dropped_nonfinite"], 1)
        # the in-volume part is flown at true scale (not shrunk)
        self.assertAlmostEqual(path.points[:, 0].min(), -0.6, delta=0.03)
        self.assertAlmostEqual(path.points[:, 0].max(), 0.6, delta=0.03)
        self.assertAlmostEqual(path.points[:, 2].max(), 1.2, delta=1e-9)
        # fit override: spell at true scale, guide with fitting
        self.assertFalse(clean_path(raw, mode="spell", fit=False).meta["fitted"])
        self.assertTrue(clean_path(raw, mode="guide", fit=True).meta["fitted"])

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            clean_path(np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            clean_path([[0.1, 0.2, 0.3]] * 10)
        with self.assertRaises(ValueError):
            clean_path(spiral(), mode="teleport")

    def test_save_load_round_trip_and_listing(self) -> None:
        path = clean_path(spiral(), mode="spell", name="My Spiral!")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_paths(tmp), [])
            file = save_path(path, directory=tmp)
            self.assertEqual(os.path.basename(file), "my_spiral.json")
            self.assertTrue(os.path.isfile(file))
            with open(file, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["format"], P.FILE_FORMAT)
            self.assertEqual(set(data), {"format", "name", "mode", "points", "times", "meta"})
            self.assertIsInstance(data["points"][0][0], float)

            loaded = load_path("My Spiral!", directory=tmp)
            self.assertEqual(loaded.name, path.name)
            self.assertEqual(loaded.mode, path.mode)
            np.testing.assert_array_equal(loaded.points, path.points)
            np.testing.assert_array_equal(loaded.times, path.times)
            self.assertEqual(loaded.meta, json.loads(json.dumps(P._jsonable(path.meta))))
            # by slug and by filename too
            self.assertEqual(load_path("my_spiral", directory=tmp).name, path.name)
            self.assertEqual(load_path("my_spiral.json", directory=tmp).name, path.name)

            other = clean_path(spiral(r=0.2), mode="guide", name="Exit A")
            save_path(other, directory=tmp)
            self.assertEqual(list_paths(tmp), ["exit_a", "my_spiral"])
            with self.assertRaises(FileNotFoundError):
                load_path("nope", directory=tmp)
        self.assertEqual(list_paths(os.path.join(tmp, "missing")), [])

    def test_slugify(self) -> None:
        self.assertEqual(slugify("Exit A"), "exit_a")
        self.assertEqual(slugify("  Big  Spiral!! "), "big_spiral")
        self.assertEqual(slugify("exit-b"), "exit_b")
        self.assertEqual(slugify(""), "path")
        self.assertEqual(slugify("***"), "path")

    def test_path_helpers(self) -> None:
        path = clean_path(spiral(), mode="spell", name="s")
        np.testing.assert_allclose(path.position_at(-1.0), path.points[0])
        np.testing.assert_allclose(path.position_at(1e9), path.points[-1])
        mid = path.position_at(path.duration / 2)
        self.assertTrue(GEOFENCE.contains(mid))
        wps = path.waypoints()
        self.assertGreater(len(wps), 5)
        # waypoints are chords of the curve, so flight time is slightly below length/speed
        flight_time = wps[-1].t - wps[0].duration
        self.assertLess(abs(flight_time - path.length / 0.3) / (path.length / 0.3), 0.01)
        with self.assertRaises(ValueError):
            Path("x", "guide", np.zeros((3, 3)), np.zeros(2))
        with self.assertRaises(ValueError):
            Path("x", "nope", np.zeros((2, 3)), np.zeros(2))


class TestDemoPaths(unittest.TestCase):
    def test_demo_paths_are_clean_and_inside_geofence(self) -> None:
        demos = demo_paths()
        self.assertEqual(set(demos), {"spiral", "square", "exit_a", "exit_b"})
        for name, path in demos.items():
            with self.subTest(name=name):
                self.assertEqual(path.name, name)
                self.assertTrue(GEOFENCE.contains(path.points))
                self.assertEqual(len(path.points), len(path.times))
                self.assertGreater(len(path.points), 10)
                self.assertTrue(np.all(np.diff(path.times) > 0))
                self.assertTrue(path.meta["demo"])
                wps = path.waypoints()
                self.assertTrue(GEOFENCE.contains(np.array([w.xyz for w in wps])))
        self.assertEqual(demos["spiral"].mode, "spell")
        self.assertEqual(demos["exit_a"].mode, "guide")
        self.assertAlmostEqual(demos["spiral"].points[0, 2], 0.4, delta=0.02)
        self.assertAlmostEqual(demos["spiral"].points[-1, 2], 1.0, delta=0.02)
        self.assertAlmostEqual(demos["square"].length, 4.0, delta=0.1)
        self.assertAlmostEqual(demos["exit_a"].length, 2.0, delta=0.05)
        # exit_b mirrors exit_a across the x axis
        np.testing.assert_allclose(demos["exit_b"].points[:, 1], -demos["exit_a"].points[:, 1], atol=1e-9)
        np.testing.assert_allclose(demos["exit_b"].points[:, 0], demos["exit_a"].points[:, 0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
