import math
import unittest

import numpy as np

from inference.obb_decoder import (
    RotatedDetection,
    decode_angle,
    decode_dfl,
    decode_onnx_output,
    decode_scale,
    rotated_nms,
    sigmoid,
)


class OBBDecoderTests(unittest.TestCase):
    def test_sigmoid_is_stable_for_large_logits(self):
        values = sigmoid(np.array([-1000.0, 0.0, 1000.0], dtype=np.float32))
        np.testing.assert_allclose(values, (0.0, 0.5, 1.0), atol=1e-7)

    def test_standard_yolov8_angle_formula(self):
        self.assertAlmostEqual(float(decode_angle(np.array([0.0]))[0]), math.pi / 4)
        low, high = decode_angle(np.array([-1000.0, 1000.0]))
        self.assertAlmostEqual(float(low), -math.pi / 4, places=6)
        self.assertAlmostEqual(float(high), 3 * math.pi / 4, places=6)

    def test_dfl_expectation_recovers_selected_bins(self):
        logits = np.full((1, 64), -20.0, dtype=np.float32)
        for side, selected_bin in enumerate((1, 2, 3, 4)):
            logits[0, side * 16 + selected_bin] = 20.0
        distances = decode_dfl(logits)
        np.testing.assert_allclose(distances[0], (1, 2, 3, 4), atol=1e-5)

    def test_dist2rbox_uses_grid_center_stride_and_rotation(self):
        regression = np.full((1, 1, 64), -20.0, dtype=np.float32)
        for side, selected_bin in enumerate((1, 2, 3, 4)):
            regression[0, 0, side * 16 + selected_bin] = 20.0
        classification = np.array([[[10.0]]], dtype=np.float32)
        angle = np.array([[[0.0]]], dtype=np.float32)
        detections = decode_scale(regression, classification, angle, 8, 0.25)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertAlmostEqual(detection.center_x, 4.0, places=4)
        self.assertAlmostEqual(
            detection.center_y,
            (0.5 + math.sqrt(2.0)) * 8,
            places=4,
        )
        self.assertAlmostEqual(detection.width, 32.0, places=4)
        self.assertAlmostEqual(detection.height, 48.0, places=4)
        self.assertAlmostEqual(detection.angle_degrees, 45.0, places=4)

    def test_rotated_nms_suppresses_overlapping_box(self):
        detections = [
            RotatedDetection(100, 100, 80, 30, math.radians(20), 0.95),
            RotatedDetection(101, 101, 80, 30, math.radians(20), 0.80),
            RotatedDetection(300, 300, 80, 30, math.radians(20), 0.70),
        ]
        kept = rotated_nms(detections, 0.25, 0.30)
        self.assertEqual([d.confidence for d in kept], [0.95, 0.70])

    def test_decoded_onnx_output_applies_confidence_threshold(self):
        output = np.zeros((1, 6, 3), dtype=np.float32)
        output[0, :, 0] = (100, 120, 80, 30, 0.9, 0.2)
        output[0, :, 1] = (200, 220, 60, 20, 0.2, -0.1)
        output[0, :, 2] = (300, 320, 0, 20, 0.8, 0.3)
        detections = decode_onnx_output(output, 0.25)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)

    def test_polygon_long_edge_rotates_with_angle(self):
        detection = RotatedDetection(100, 100, 120, 20, math.radians(30), 0.9)
        corners = detection.corners()
        edges = np.roll(corners, -1, axis=0) - corners
        longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        orientation = math.degrees(math.atan2(longest[1], longest[0])) % 180
        self.assertAlmostEqual(orientation, 30.0, places=4)


if __name__ == "__main__":
    unittest.main()
