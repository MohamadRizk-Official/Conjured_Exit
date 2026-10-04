"""tools/trace_explain.py and tools/carry_check.py: the plain-English read-outs must say the right thing
for the failure shapes seen on 2026-10-04."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import carry_check  # noqa: E402
import trace_explain  # noqa: E402


def row(t, state, cam=(0.3, 0.0, 0.05), ok=True, est=None, bat=4.1, yaw=170.0):
    est = est or cam
    return {"t": f"{t:.3f}", "state": state, "cam_ok": "1" if ok else "0",
            "cam_x": f"{cam[0]:.3f}" if ok else "", "cam_y": f"{cam[1]:.3f}" if ok else "",
            "cam_z": f"{cam[2]:.3f}" if ok else "", "cam_yaw_deg": f"{yaw:.1f}" if ok else "",
            "est_x": f"{est[0]:.3f}", "est_y": f"{est[1]:.3f}", "est_z": f"{est[2]:.3f}", "bat_v": f"{bat:.2f}"}


class TraceExplainTests(unittest.TestCase):
    def test_sideways_slide_is_blamed_on_heading(self):
        rows = [row(100 + i * 0.05, "takeoff", cam=(0.3 + 0.03 * i, 0.0 + 0.02 * i, 0.05 + 0.005 * i)) for i in range(40)]
        rows += [row(102 + i * 0.05, "estop", cam=(1.5, 0.8, 0.1)) for i in range(5)]
        r = trace_explain.explain(rows)
        self.assertIn("sideways", r["verdict"])
        self.assertIn("heading", r["verdict"])
        self.assertGreater(r["max_xy_drift_m"], 0.3)

    def test_marker_lost_low_is_blamed_on_the_marker(self):
        rows = [row(100 + i * 0.05, "takeoff", cam=(0.3, 0.0, 0.05 + 0.01 * i)) for i in range(20)]
        rows += [row(101 + i * 0.05, "takeoff", ok=False) for i in range(10)]
        rows += [row(101.5 + i * 0.05, "estop", ok=False) for i in range(5)]
        r = trace_explain.explain(rows)
        self.assertAlmostEqual(r["camera_lost_at_s"], 1.0, places=1)
        self.assertIn("lost the marker", r["verdict"])
        self.assertIn("motors were cut", r["verdict"])

    def test_no_lift_with_big_sag_is_blamed_on_the_battery(self):
        rows = [row(100 + i * 0.05, "takeoff", cam=(0.3, 0.0, 0.05 + 0.001 * i), bat=4.0 - 0.02 * i) for i in range(60)]
        rows += [row(103 + i * 0.05, "estop", bat=3.3) for i in range(5)]
        r = trace_explain.explain(rows)
        self.assertIn("battery", r["verdict"])
        self.assertLess(r["max_height_gain_m"], 0.15)

    def test_a_good_hover_is_reported_as_flown(self):
        rows = [row(100 + i * 0.05, "takeoff", cam=(0.3, 0.0, 0.05 + 0.014 * i)) for i in range(40)]
        rows += [row(102 + i * 0.05, "hover", cam=(0.31, 0.01, 0.61)) for i in range(100)]
        rows += [row(107 + i * 0.05, "landing", cam=(0.31, 0.01, 0.6 - 0.012 * i)) for i in range(50)]
        rows += [row(109.5 + i * 0.05, "idle", cam=(0.31, 0.01, 0.05)) for i in range(10)]
        r = trace_explain.explain(rows)
        self.assertTrue(r["verdict"].startswith("it flew"), r["verdict"])
        self.assertGreaterEqual(r["max_height_gain_m"], 0.5)

    def test_stop_in_flight_is_reported_as_a_cut_not_a_landing(self):
        rows = [row(100 + i * 0.05, "takeoff", cam=(0.3, 0.0, 0.05 + 0.014 * i)) for i in range(40)]
        rows += [row(102 + i * 0.05, "flying", cam=(0.3 + 0.01 * i, 0.0, 0.6)) for i in range(30)]
        rows += [row(103.5 + i * 0.05, "estop", cam=(0.6, 0.0, 0.3)) for i in range(5)]
        rows += [row(103.75 + i * 0.05, "idle", cam=(0.6, 0.0, 0.05)) for i in range(5)]
        r = trace_explain.explain(rows)
        self.assertIn("motors were cut", r["verdict"])
        self.assertNotIn("landed normally", r["verdict"])

    def test_estimate_far_from_camera_is_blamed_on_the_feed(self):
        rows = [row(100 + i * 0.05, "takeoff", cam=(0.3, 0.0, 0.05), est=(1.5, 1.0, 0.0)) for i in range(30)]
        rows += [row(101.5 + i * 0.05, "estop", cam=(0.3, 0.0, 0.05), est=(1.5, 1.0, 0.0)) for i in range(5)]
        r = trace_explain.explain(rows)
        self.assertIn("disagreed with the camera", r["verdict"])

    def test_flights_are_split_on_time_gaps(self):
        rows = [row(100 + i * 0.05, "takeoff") for i in range(10)] + [row(500 + i * 0.05, "takeoff") for i in range(10)]
        self.assertEqual(len(trace_explain.split_flights(rows)), 2)

    def test_back_to_back_flights_are_split_at_the_next_takeoff(self):
        rows = [row(100 + i * 0.05, "takeoff") for i in range(10)]
        rows += [row(100.5 + i * 0.05, "idle") for i in range(10)]            # the 2 s tail after the first flight
        rows += [row(101 + i * 0.05, "takeoff") for i in range(10)]           # second flight 0.5 s later
        rows += [row(101.5 + i * 0.05, "estop") for i in range(5)]
        flights = trace_explain.split_flights(rows)
        self.assertEqual(len(flights), 2)
        self.assertEqual(flights[1][0]["state"], "takeoff")
        self.assertEqual(flights[1][-1]["state"], "estop")


class CarryCheckTests(unittest.TestCase):
    def _rows(self, moves):
        """moves: list of (dx, dy, dz) per sample at 10 Hz, starting from (0.3, 0, 0.06)."""
        rows = []
        for i, (dx, dy, dz) in enumerate(moves):
            x, y, z = 0.3 + dx, 0.0 + dy, 0.06 + dz
            rows.append({"t": f"{i * 0.1:.2f}", "tracking_ok": "1", "cam_x": f"{x:.3f}", "cam_y": f"{y:.3f}",
                         "cam_z": f"{z:.3f}", "cam_yaw_deg": "170.0", "est_x": f"{x:.3f}", "est_y": f"{y:.3f}",
                         "est_z": f"{z:.3f}", "cam_vs_est_m": "0.010", "bat_v": "4.10"})
        return rows

    def test_correct_axes_in_order_are_reported_good(self):
        moves = [(0, 0, 0)] * 20
        moves += [(0.5, 0, 0)] * 20 + [(0, 0, 0)] * 10
        moves += [(0, 0.5, 0)] * 20 + [(0, 0, 0)] * 10
        moves += [(0, 0, 0.5)] * 20 + [(0, 0, 0)] * 10
        r = carry_check.analyze_rows(self._rows(moves))
        self.assertEqual(r["axis_order_seen"], ["x", "y", "z"])
        self.assertTrue(all("(good)" in n for n in r["notes"][:3]), r["notes"])
        self.assertIn("followed the camera", r["notes"][3])

    def test_reversed_axis_is_called_out(self):
        moves = [(0, 0, 0)] * 20 + [(0, -0.5, 0)] * 20 + [(0, 0, 0)] * 10
        r = carry_check.analyze_rows(self._rows(moves))
        self.assertTrue(any("REVERSED" in n and "+y" in n for n in r["notes"]), r["notes"])

    def test_untracked_recording_has_a_verdict(self):
        rows = self._rows([(0, 0, 0)] * 3)
        for rr in rows:
            rr["tracking_ok"] = "0"
        self.assertIn("did not see", carry_check.analyze_rows(rows)["verdict"])


if __name__ == "__main__":
    unittest.main()
