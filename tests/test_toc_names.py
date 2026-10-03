"""toc_names: resolving true variable names against the nRF-2024.10-mangled TOC."""
import unittest

from toc_names import as_seen, is_intact, resolve

# Taken from the real cached TOCs of our drone (cache/47303730.json, cache/E74449A3.json)
PARAM_TOC = {
    "stabilizer": {"estmator": 143, "conroller": 144, "sto": 145},
    "kalman": {"resetEsimation": 116, "initial": 1, "pNAcc_x": 2, "mNGyro_ollpitch": 3},
    "locSrv": {"extPosSdDev": 107, "extQuattdDev": 108},
    "pm": {"criticalLowoltage": 10, "lowVoltage": 11},
    "system": {"arm": 20},
}
LOG_TOC = {
    "pm": {"vbat": 89, "batteryLeve\x00": 96},
    "kalman": {"varPX": 1, "varPY": 2, "varPZ": 3, "stateX": 4, "stateY": 5, "stateZ": 6, "rtUpdat\x00": 7},
    "stabilizer": {"rol\x00": 216, "pith\x00": 217, "yaw": 218, "thrst\x00": 219},
    "stateEstimate": {"\x00": 224, "x\x00": 234, "y\x00": 235, "z\x00": 236, "oll\x00": 231},
}


class CorruptionModelTests(unittest.TestCase):
    def test_short_entries_are_intact(self):
        self.assertTrue(is_intact("pm", "vbat"))
        self.assertTrue(is_intact("kalman", "stateX"))
        self.assertFalse(is_intact("stabilizer", "roll"))

    def test_predicts_the_observed_mangling(self):
        self.assertEqual(as_seen("stabilizer", "estimator"), ("stabilizer", "estmator"))
        self.assertEqual(as_seen("kalman", "resetEstimation"), ("kalman", "resetEsimation"))
        self.assertEqual(as_seen("kalman", "mNGyro_rollpitch"), ("kalman", "mNGyro_ollpitch"))
        self.assertEqual(as_seen("locSrv", "extPosStdDev"), ("locSrv", "extPosSdDev"))
        self.assertEqual(as_seen("stabilizer", "roll"), ("stabilizer", "rol"))
        self.assertEqual(as_seen("pm", "vbat"), ("pm", "vbat"))


class ResolveTests(unittest.TestCase):
    def test_intact_name_resolves_to_itself(self):
        self.assertEqual(resolve(LOG_TOC, "pm.vbat"), "pm.vbat")
        self.assertEqual(resolve(LOG_TOC, "kalman.stateX"), "kalman.stateX")
        self.assertEqual(resolve(PARAM_TOC, "system.arm"), "system.arm")

    def test_mangled_param_names_resolve_to_what_cflib_stored(self):
        self.assertEqual(resolve(PARAM_TOC, "stabilizer.estimator"), "stabilizer.estmator")
        self.assertEqual(resolve(PARAM_TOC, "stabilizer.controller"), "stabilizer.conroller")
        self.assertEqual(resolve(PARAM_TOC, "kalman.resetEstimation"), "kalman.resetEsimation")
        self.assertEqual(resolve(PARAM_TOC, "locSrv.extPosStdDev"), "locSrv.extPosSdDev")
        self.assertEqual(resolve(PARAM_TOC, "locSrv.extQuatStdDev"), "locSrv.extQuattdDev")
        self.assertEqual(resolve(PARAM_TOC, "pm.criticalLowVoltage"), "pm.criticalLowoltage")

    def test_mangled_log_names_keep_cflibs_trailing_nul(self):
        self.assertEqual(resolve(LOG_TOC, "stabilizer.roll"), "stabilizer.rol\x00")
        self.assertEqual(resolve(LOG_TOC, "stabilizer.pitch"), "stabilizer.pith\x00")
        self.assertEqual(resolve(LOG_TOC, "stabilizer.yaw"), "stabilizer.yaw")
        self.assertEqual(resolve(LOG_TOC, "pm.batteryLevel"), "pm.batteryLeve\x00")

    def test_colliding_names_raise_instead_of_guessing(self):
        with self.assertRaises(KeyError):
            resolve(LOG_TOC, "stateEstimate.x")     # could be x, vx, ax or qx
        # initialX / initialY / initialZ all shrink to 'initial': undetectable from the TOC alone,
        # so the resolver returns the surviving entry. Callers must avoid such names.
        self.assertEqual(resolve(PARAM_TOC, "kalman.initialX"), "kalman.initial")

    def test_unknown_names_raise(self):
        with self.assertRaises(KeyError):
            resolve(PARAM_TOC, "stabilizer.nonsense")
        with self.assertRaises(KeyError):
            resolve(PARAM_TOC, "nogroup.estimator")
        with self.assertRaises(KeyError):
            resolve(PARAM_TOC, "noDotHere")

    def test_accepts_a_cflib_toc_object(self):
        class FakeToc:
            toc = PARAM_TOC
        self.assertEqual(resolve(FakeToc(), "stabilizer.estimator"), "stabilizer.estmator")


if __name__ == "__main__":
    unittest.main()
