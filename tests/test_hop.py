import unittest

import hop


class HopProfileTests(unittest.TestCase):
    def test_profile_ramps_up_holds_and_ramps_down_to_floor(self):
        p = hop.hop_profile(0.4, t_up=1.5, t_hold=3.0, t_down=1.5, hz=20.0)
        self.assertEqual(len(p), 30 + 60 + 30)
        self.assertAlmostEqual(max(p), 0.4)
        self.assertAlmostEqual(p[-1], 0.05)
        self.assertTrue(all(0.0 < z <= 0.4 + 1e-9 for z in p))
        up, hold, down = p[:30], p[30:90], p[90:]
        self.assertTrue(all(b >= a for a, b in zip(up, up[1:])))
        self.assertTrue(all(abs(z - 0.4) < 1e-9 for z in hold))
        self.assertTrue(all(b <= a for a, b in zip(down, down[1:])))

    def test_supervisor_bits_decode(self):
        self.assertEqual(hop.decode_info(hop.BIT_IS_ARMED | hop.BIT_CAN_FLY), "armed canFly")
        self.assertIn("TUMBLED", hop.decode_info(hop.BIT_IS_TUMBLED))
        self.assertEqual(hop.decode_info(0), "none")

    def test_telemetry_decodes_true_names(self):
        tel = hop.HopTelemetry()
        tel.update(0, {"pm.vbat": 3.9, "supervisor.info": hop.BIT_IS_ARMED, "kalman.stateZ": 0.12}, None)
        self.assertEqual((tel.vbat, tel.info, tel.z), (3.9, hop.BIT_IS_ARMED, 0.12))
        self.assertIn("armed", tel.line())


if __name__ == "__main__":
    unittest.main()
