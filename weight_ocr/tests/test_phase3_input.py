import unittest

import numpy as np

from workers.inference_worker import prepare_phase3_input


class Phase3InputTests(unittest.TestCase):
    def test_phase3_input_is_uint8_rgb_512_square(self):
        bgr = np.zeros((200, 400, 3), dtype=np.uint8)
        bgr[:, :] = (10, 20, 30)
        model_input = prepare_phase3_input(bgr, 512, 512)
        self.assertEqual(model_input.shape, (512, 512, 3))
        self.assertEqual(model_input.dtype, np.uint8)
        self.assertTrue(model_input.flags.c_contiguous)
        np.testing.assert_array_equal(model_input[256, 256], (30, 20, 10))
        np.testing.assert_array_equal(model_input[20, 20], (114, 114, 114))


if __name__ == "__main__":
    unittest.main()
