"""CameraThread must survive a webcam that drops off USB and comes back (three times on 2026-10-04).

The thread reopens the device by itself: ``is_open`` goes False while the camera is gone (so the
tracker reports 2d-only and tracking_ok stays False) and True again when frames resume. No server
restart needed.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tracker  # noqa: E402


class _ScriptedCapture:
    """Fake cv2.VideoCapture. Each instance follows the next entry of ``script``:
    ("dead",)                -> isOpened() False
    ("frames", n)            -> n good frames, then read() fails forever
    ("forever",)             -> good frames until released
    """

    script: list = []
    instances: list = []

    def __init__(self, index, backend=None):
        self.index, self.backend = index, backend
        self.behaviour = list(self.script.pop(0)) if self.script else ["forever"]
        self.reads = 0
        self.released = False
        _ScriptedCapture.instances.append(self)

    def isOpened(self):
        return self.behaviour[0] != "dead"

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 1280.0 if prop == tracker.cv2.CAP_PROP_FRAME_WIDTH else 720.0

    def read(self):
        self.reads += 1
        if self.behaviour[0] == "frames" and self.reads > self.behaviour[1]:
            return False, None
        time.sleep(0.002)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def wait_until(pred, timeout=5.0):
    t_end = time.perf_counter() + timeout
    while time.perf_counter() < t_end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class CameraThreadReopenTests(unittest.TestCase):
    def setUp(self):
        _ScriptedCapture.instances = []

    def _start(self, script, **kw):
        _ScriptedCapture.script = [list(s) for s in script]
        patcher = mock.patch.object(tracker.cv2, "VideoCapture", _ScriptedCapture)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = tracker.CameraThread(1, backend=tracker.cv2.CAP_MSMF, width=1280, height=720,
                                   fail_limit=5, reopen_s=0.05, **kw)
        src.start()
        self.addCleanup(lambda: src.stop(join=True, timeout=2.0))
        return src

    def test_camera_that_stops_delivering_is_reopened(self):
        src = self._start([("frames", 3), ("forever",)])
        self.assertTrue(src.wait_open(2.0))
        self.assertTrue(wait_until(lambda: not src.is_open, 3.0), "is_open never dropped")
        self.assertIn("stopped delivering", src.error or "")
        self.assertTrue(wait_until(lambda: src.is_open, 3.0), "camera was not reopened")
        self.assertEqual(src.reopens, 1)
        self.assertIsNone(src.error)
        self.assertEqual(len(_ScriptedCapture.instances), 2)
        self.assertTrue(_ScriptedCapture.instances[0].released)
        self.assertTrue(wait_until(lambda: src.seq > 3, 2.0), "no frames after the reopen")

    def test_camera_missing_at_start_is_retried_until_it_appears(self):
        src = self._start([("dead",), ("dead",), ("forever",)])
        self.assertTrue(wait_until(lambda: src.is_open, 3.0), "never opened")
        self.assertEqual(len(_ScriptedCapture.instances), 3)
        self.assertIsNone(src.error)

    def test_stop_ends_the_thread_even_while_the_camera_is_missing(self):
        src = self._start([("dead",)] * 50)
        time.sleep(0.15)
        src.stop(join=True, timeout=2.0)
        self.assertFalse(src.is_alive())
        self.assertIn("did not open", src.error or "")

    def test_wait_open_reports_failure_quickly_when_the_camera_is_missing(self):
        src = self._start([("dead",)] * 50)
        t0 = time.perf_counter()
        self.assertFalse(src.wait_open(2.0))
        self.assertLess(time.perf_counter() - t0, 1.5)          # ready-or-failed fires on the first failure


if __name__ == "__main__":
    unittest.main()
