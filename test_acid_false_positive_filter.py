from __future__ import annotations

import builtins
import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "test_embedded_run_local4",
    BASE_DIR / "run-local4.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load run-local4.py for acid-filter tests.")
RUN_LOCAL = importlib.util.module_from_spec(SPEC)
_original_import = builtins.__import__


def _import_without_runtime_hardware(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in {
        "hailo_platform",
        "librosa",
        "pyqtgraph",
        "PyQt5",
        "sounddevice",
    }:
        raise ImportError(f"{name} disabled for acid-filter unit tests")
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _import_without_runtime_hardware
try:
    SPEC.loader.exec_module(RUN_LOCAL)
finally:
    builtins.__import__ = _original_import


class _AcidModel:
    def predict(self, _frame, **_kwargs):
        return [{"score": 0.90, "bbox": [20, 10, 60, 50]}]


class _AcidFilter:
    def __init__(self, accepted: bool, probability: float):
        self.accepted = accepted
        self.probability = probability
        self.last_crop = None

    def predict(self, crop):
        self.last_crop = crop.copy()
        return self.accepted, self.probability


class AcidFalsePositiveFilterTests(unittest.TestCase):
    def setUp(self):
        self.original_model = RUN_LOCAL.MODEL_ACID
        self.original_filter = RUN_LOCAL.ACID_FP_FILTER
        RUN_LOCAL.MODEL_ACID = _AcidModel()

    def tearDown(self):
        RUN_LOCAL.MODEL_ACID = self.original_model
        RUN_LOCAL.ACID_FP_FILTER = self.original_filter

    def test_rejected_candidate_cannot_count_as_acid(self):
        classifier = _AcidFilter(False, 0.12)
        RUN_LOCAL.ACID_FP_FILTER = classifier
        frame = np.zeros((80, 100, 3), dtype=np.uint8)

        annotated, acid_detected, info = RUN_LOCAL.process_acid_frame(frame)

        self.assertFalse(acid_detected)
        self.assertIsNone(info["acid_bbox"])
        self.assertEqual(info["accepted_count"], 0)
        self.assertEqual(info["rejected_count"], 1)
        self.assertEqual(int(np.count_nonzero(annotated)), 0)
        self.assertIsNotNone(classifier.last_crop)
        self.assertEqual(classifier.last_crop.shape[0], classifier.last_crop.shape[1])

    def test_accepted_candidate_preserves_acid_detection(self):
        RUN_LOCAL.ACID_FP_FILTER = _AcidFilter(True, 0.91)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)

        annotated, acid_detected, info = RUN_LOCAL.process_acid_frame(frame)

        self.assertTrue(acid_detected)
        self.assertEqual(info["acid_bbox"], (20, 10, 60, 50))
        self.assertEqual(info["accepted_count"], 1)
        self.assertEqual(info["rejected_count"], 0)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)

    def test_candidate_checkpoint_classifies_locked_crops(self):
        model_path = (
            BASE_DIR
            / "BeadFalsePositive"
            / "model_projects"
            / "bestnewacid"
            / "MobileNetV3"
            / "bestnewacid_mobilenet_v3_candidate.pt"
        )
        locked_dir = (
            BASE_DIR
            / "BeadFalsePositive"
            / "model_projects"
            / "bestnewacid"
            / "ResNet18"
            / "locked_crop_evaluation"
        )
        classifier = RUN_LOCAL.AcidMobileNetV3Filter(model_path)
        false_crop = cv2.imread(
            str(locked_dir / "false_positive" / "20260812_150232_1e2e29d8_candidate_001.png")
        )
        true_crop = cv2.imread(
            str(locked_dir / "true_detection" / "20260812_150603_87983181_candidate_001.png")
        )

        false_accepted, false_probability = classifier.predict(false_crop)
        true_accepted, true_probability = classifier.predict(true_crop)

        self.assertFalse(false_accepted, false_probability)
        self.assertTrue(true_accepted, true_probability)


if __name__ == "__main__":
    unittest.main()
