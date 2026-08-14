import unittest

import numpy as np

from vision.letterbox import letterbox


class LetterboxTests(unittest.TestCase):
    CASES = (
        ((1920, 1080), (512, 288), (0, 112, 0, 112)),
        ((1400, 1000), (512, 366), (0, 73, 0, 73)),
        ((1000, 1000), (512, 512), (0, 0, 0, 0)),
        ((500, 1200), (213, 512), (149, 0, 150, 0)),
    )

    def test_required_source_sizes_keep_aspect_ratio(self):
        for source_size, resized_size, padding in self.CASES:
            with self.subTest(source_size=source_size):
                source_width, source_height = source_size
                image = np.zeros((source_height, source_width, 3), dtype=np.uint8)
                model_image, transform = letterbox(image)
                self.assertEqual(model_image.shape, (512, 512, 3))
                self.assertEqual(
                    (transform.original_width, transform.original_height),
                    source_size,
                )
                self.assertEqual(
                    (transform.resized_width, transform.resized_height),
                    resized_size,
                )
                self.assertEqual(
                    (
                        transform.pad_left,
                        transform.pad_top,
                        transform.pad_right,
                        transform.pad_bottom,
                    ),
                    padding,
                )
                self.assertAlmostEqual(
                    transform.scale,
                    min(512 / source_width, 512 / source_height),
                )

    def test_point_roundtrip_error_is_below_one_pixel(self):
        for source_size, _resized_size, _padding in self.CASES:
            with self.subTest(source_size=source_size):
                source_width, source_height = source_size
                image = np.zeros((source_height, source_width, 3), dtype=np.uint8)
                _model_image, transform = letterbox(image)
                points = (
                    (0.0, 0.0),
                    (source_width - 1.0, 0.0),
                    (0.0, source_height - 1.0),
                    (source_width - 1.0, source_height - 1.0),
                    (source_width * 0.5, source_height * 0.5),
                    (source_width * 0.237, source_height * 0.811),
                )
                for roi_x, roi_y in points:
                    model_x, model_y = transform.roi_to_model_point(roi_x, roi_y)
                    mapped_x, mapped_y = transform.model_to_roi_point(
                        model_x,
                        model_y,
                    )
                    error = np.hypot(mapped_x - roi_x, mapped_y - roi_y)
                    self.assertLess(error, 1.0)

    def test_padding_and_image_dtype_are_preserved(self):
        image = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)
        model_image, transform = letterbox(image)
        self.assertEqual(model_image.dtype, image.dtype)
        np.testing.assert_array_equal(model_image[0, 0], (114, 114, 114))
        center_x, center_y = transform.roi_to_model_point(100, 50)
        np.testing.assert_array_equal(
            model_image[int(round(center_y)), int(round(center_x))],
            (10, 20, 30),
        )


if __name__ == "__main__":
    unittest.main()
