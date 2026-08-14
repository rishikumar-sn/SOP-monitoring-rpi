from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from validation.sample_saver import normalize_reading, save_validation_sample


class ValidationSampleSaverTests(unittest.TestCase):
    def test_reading_normalization_preserves_label_digits(self):
        self.assertEqual(normalize_reading("1.17"), ("1.17", "117"))
        self.assertEqual(normalize_reading("5717"), ("5717", "5717"))
        with self.assertRaises(ValueError):
            normalize_reading("1.2.3")

    def test_successful_result_saves_images_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            artifact = root / "source_rectified.png"
            cv2.imwrite(str(artifact), np.full((20, 60, 3), 90, np.uint8))
            result = {
                "detections": [{"confidence": 0.92}],
                "backend": "ONNX Runtime CPU",
                "digit_result": {
                    "success": True,
                    "digits": "117",
                    "confidence": 0.81,
                    "failed_slots": [],
                },
                "lcd_rectified": str(artifact),
            }
            sample_dir, metadata = save_validation_sample(
                root / "samples",
                "1.17",
                "Normal",
                np.zeros((40, 80, 3), np.uint8),
                np.zeros((100, 160, 3), np.uint8),
                (10, 20, 90, 60),
                inference_result=result,
                captured_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
            )
            self.assertTrue((sample_dir / "roi.png").is_file())
            self.assertTrue((sample_dir / "full.png").is_file())
            self.assertTrue((sample_dir / "lcd_rectified.png").is_file())
            stored = json.loads((sample_dir / "metadata.json").read_text())
            self.assertEqual(stored["label_digits"], "117")
            self.assertEqual(stored["inference_status"], "decoded")
            self.assertEqual(metadata["predicted_digits"], "117")

    def test_failed_read_is_saved_without_predicted_digits(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = {
                "detections": [{"confidence": 0.88}],
                "digit_result": {
                    "success": False,
                    "digits": None,
                    "confidence": 0.51,
                    "failed_slots": [3],
                },
            }
            sample_dir, metadata = save_validation_sample(
                Path(temporary_dir) / "samples",
                "8.90",
                "Glare",
                np.zeros((40, 80, 3), np.uint8),
                np.zeros((100, 160, 3), np.uint8),
                (10, 20, 90, 60),
                inference_result=result,
            )
            self.assertTrue((sample_dir / "metadata.json").is_file())
            self.assertEqual(metadata["inference_status"], "read_failed")
            self.assertIsNone(metadata["predicted_digits"])
            self.assertEqual(metadata["failed_slots"], [3])

    def test_lcd_not_found_still_saves_clean_capture(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sample_dir, metadata = save_validation_sample(
                Path(temporary_dir) / "samples",
                "1.558",
                "Jewellery near display",
                np.zeros((40, 80, 3), np.uint8),
                np.zeros((100, 160, 3), np.uint8),
                (10, 20, 90, 60),
                inference_result={"detections": [], "digit_result": None},
            )
            self.assertEqual(metadata["inference_status"], "lcd_not_found")
            self.assertTrue((sample_dir / "roi.png").is_file())
            self.assertTrue((sample_dir / "full.png").is_file())


if __name__ == "__main__":
    unittest.main()
