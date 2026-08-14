import itertools
import unittest

import cv2
import numpy as np

from vision.perspective import expand_quad_points, order_quad_points, rectify_lcd


class PerspectiveTests(unittest.TestCase):
    def test_quad_expansion_adds_source_area(self):
        corners = np.array(((10, 10), (110, 10), (110, 60), (10, 60)), np.float32)
        expanded = expand_quad_points(corners, horizontal=0.10, vertical=0.20)
        np.testing.assert_allclose(
            expanded,
            np.array(((0, 0), (120, 0), (120, 70), (0, 70)), np.float32),
        )

    def test_corner_order_is_stable_for_every_input_permutation(self):
        expected = np.array(
            ((100, 80), (340, 105), (320, 205), (75, 180)),
            dtype=np.float32,
        )
        for permutation in itertools.permutations(expected):
            with self.subTest(permutation=permutation):
                ordered = order_quad_points(permutation)
                np.testing.assert_array_equal(ordered, expected)

    def test_perspective_warp_restores_horizontal_orientation(self):
        canonical = np.zeros((90, 240, 3), dtype=np.uint8)
        canonical[:12] = (0, 255, 0)
        canonical[-12:] = (255, 0, 0)
        canonical[:, :12] = (0, 0, 255)
        canonical[:, -12:] = (0, 255, 255)
        cv2.putText(
            canonical,
            "138",
            (72, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        source = np.array(((0, 0), (239, 0), (239, 89), (0, 89)), np.float32)
        destination = np.array(
            ((90, 70), (350, 110), (315, 225), (65, 175)),
            np.float32,
        )
        projection = cv2.getPerspectiveTransform(source, destination)
        distorted = cv2.warpPerspective(canonical, projection, (420, 280))

        raw, rectified = rectify_lcd(distorted, destination, inner_margin=0.05)
        self.assertGreater(raw.shape[1], raw.shape[0])
        self.assertGreater(rectified.shape[1], rectified.shape[0])
        self.assertGreater(float(raw[:10, :, 1].mean()), 150.0)
        self.assertGreater(float(raw[-10:, :, 0].mean()), 150.0)
        self.assertGreater(float(raw[:, :10, 2].mean()), 150.0)
        self.assertGreater(float(raw[:, -10:, 1].mean()), 150.0)
        self.assertLess(rectified.shape[0], raw.shape[0])
        self.assertLess(rectified.shape[1], raw.shape[1])

    def test_tall_warp_is_rotated_to_horizontal(self):
        image = np.full((240, 100, 3), 127, dtype=np.uint8)
        corners = np.array(((10, 10), (90, 10), (90, 230), (10, 230)))
        raw, _rectified = rectify_lcd(image, corners, inner_margin=0)
        self.assertGreater(raw.shape[1], raw.shape[0])


if __name__ == "__main__":
    unittest.main()
