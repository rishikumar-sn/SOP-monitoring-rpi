import unittest

import numpy as np

from vision.letterbox import letterbox
from vision.obb_mapping import (
    draw_polygon_copy,
    map_model_corners,
    model_corners_to_roi,
)


class OBBMappingTests(unittest.TestCase):
    def setUp(self):
        roi = np.zeros((765, 1147, 3), dtype=np.uint8)
        _model_image, self.transform = letterbox(roi, 512, 512)

    def test_model_corners_map_back_to_roi_and_full_frame(self):
        expected_roi = np.array(
            ((100, 200), (500, 200), (500, 320), (100, 320)),
            dtype=np.float32,
        )
        model_corners = np.array(
            [self.transform.roi_to_model_point(x, y) for x, y in expected_roi],
            dtype=np.float32,
        )
        roi_corners, full_corners = map_model_corners(
            model_corners,
            self.transform,
            (816, 252, 1963, 1017),
            (1440, 2560, 3),
        )
        np.testing.assert_allclose(roi_corners, expected_roi, atol=1e-4)
        np.testing.assert_allclose(
            full_corners,
            expected_roi + np.array((816, 252), dtype=np.float32),
            atol=1e-4,
        )

    def test_model_corners_are_clipped_to_roi_pixels(self):
        model_corners = np.array(
            ((-20, -30), (600, -30), (600, 700), (-20, 700)),
            dtype=np.float32,
        )
        roi_corners = model_corners_to_roi(model_corners, self.transform)
        np.testing.assert_array_equal(
            roi_corners,
            ((0, 0), (1146, 0), (1146, 764), (0, 764)),
        )

    def test_debug_drawing_does_not_modify_clean_image(self):
        clean = np.full((100, 200, 3), 127, dtype=np.uint8)
        before = clean.copy()
        output = draw_polygon_copy(
            clean,
            np.array(((20, 20), (180, 20), (180, 80), (20, 80))),
            "LCD",
        )
        np.testing.assert_array_equal(clean, before)
        self.assertFalse(np.array_equal(output, clean))


if __name__ == "__main__":
    unittest.main()
