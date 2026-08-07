import unittest
from pathlib import Path

import numpy as np

from striping_process_hef import HSVFPFilter


class HSVFPFilterCPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).with_name("hsv_fp_filter_strip.pt")
        cls.fp_filter = HSVFPFilter(str(model_path))

    def test_dummy_bgr_prediction_on_cpu(self):
        self.assertTrue(self.fp_filter.enabled)
        self.assertEqual(
            next(self.fp_filter.model.parameters()).device.type,
            "cpu",
        )
        accepted, probability = self.fp_filter.predict(
            np.zeros((64, 64, 3), dtype=np.uint8)
        )
        self.assertIsInstance(accepted, bool)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        print(
            f"[strip-fp-test] TP probability={probability:.2f} "
            f"accepted={accepted}"
        )


if __name__ == "__main__":
    unittest.main()
