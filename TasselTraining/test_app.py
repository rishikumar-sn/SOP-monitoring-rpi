import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import (
    crop_to_test_bed,
    DatasetStore,
    MIN_SAMPLES_PER_CLASS,
    MobileNetPredictor,
    PRODUCTION_CHECKPOINT,
    make_masked_crop,
    normalized_roi_from_rect,
    prepare_tassel_test_bed,
    roi_rect_for_shape,
    training_shortfall,
)
from train_mobilenet import split_samples


class TasselTrainingTests(unittest.TestCase):
    def test_test_bed_preparation_keeps_disconnected_tassel_material(self) -> None:
        image = np.full((300, 300, 3), 245, dtype=np.uint8)
        cv2.ellipse(image, (150, 225), (85, 35), 0, 0, 180, (20, 20, 80), 10)
        cv2.line(image, (75, 210), (65, 80), (20, 130, 190), 6)
        cv2.line(image, (65, 80), (45, 45), (30, 150, 210), 8)
        prepared = prepare_tassel_test_bed(image)
        labels, _, stats, _ = cv2.connectedComponentsWithStats(
            prepared.working_mask,
            8,
        )
        component_areas = sorted(
            (
                int(stats[index, cv2.CC_STAT_AREA])
                for index in range(1, labels)
                if stats[index, cv2.CC_STAT_AREA] >= 40
            ),
            reverse=True,
        )
        self.assertGreaterEqual(len(component_areas), 2)

    def test_normalized_test_bed_roi_crops_outside_background(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:80, 50:150] = (10, 20, 30)
        roi = normalized_roi_from_rect((50, 20, 150, 80), image.shape[:2])
        self.assertEqual(roi_rect_for_shape(roi, image.shape[:2]), (50, 20, 150, 80))
        crop, rect = crop_to_test_bed(image, roi)
        self.assertEqual(rect, (50, 20, 150, 80))
        self.assertEqual(crop.shape, (60, 100, 3))
        self.assertTrue(np.all(crop == (10, 20, 30)))

    def test_masked_crop_excludes_unselected_pixels(self) -> None:
        image = np.full((100, 120, 3), (10, 20, 30), dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[30:70, 45:75] = 1
        crop = make_masked_crop(image, mask, padding_ratio=0.10)
        self.assertTrue(np.any(np.all(crop == (10, 20, 30), axis=2)))
        self.assertTrue(np.any(np.all(crop == 255, axis=2)))

    def test_relabel_removes_stale_class_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = DatasetStore(Path(temporary))
            image = np.full((80, 100, 3), 200, dtype=np.uint8)
            mask = np.zeros((80, 100), dtype=np.uint8)
            mask[20:60, 30:70] = 1
            manifest = store.create_session(image, image, mask, "Necklace")
            old_path = store.save_sample(
                manifest,
                "auto_candidate",
                "tassel",
                image,
                mask,
                "model_prediction",
            )
            new_path = store.save_sample(
                manifest,
                "auto_candidate",
                "false_positive",
                image,
                mask,
                "model_prediction",
            )
            self.assertFalse(old_path.exists())
            self.assertTrue(new_path.exists())
            payload = json.loads(
                (store.session_dir(manifest) / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                payload["samples"]["auto_candidate"]["label"],
                "false_positive",
            )

    def test_session_records_test_bed_roi_and_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = DatasetStore(Path(temporary))
            image = np.full((80, 100, 3), 200, dtype=np.uint8)
            test_bed = image[10:70, 20:80]
            mask = np.ones(test_bed.shape[:2], dtype=np.uint8)
            roi = (0.2, 0.125, 0.8, 0.875)
            manifest = store.create_session(
                image,
                test_bed,
                mask,
                "Necklace",
                test_bed,
                roi,
                (20, 10, 80, 70),
            )
            self.assertEqual(manifest["test_bed_roi_normalized"], list(roi))
            self.assertTrue((store.session_dir(manifest) / "test_bed_crop.png").exists())

    def test_training_preflight_and_split_are_class_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary)
            for class_name in ("tassel", "false_positive"):
                class_dir = labels / class_name
                class_dir.mkdir()
                for index in range(MIN_SAMPLES_PER_CLASS):
                    cv2.imwrite(
                        str(class_dir / f"sample_{index}.png"),
                        np.full((20, 20, 3), index, dtype=np.uint8),
                    )
            counts, missing = training_shortfall(labels)
            self.assertEqual(counts["tassel"], MIN_SAMPLES_PER_CLASS)
            self.assertEqual(missing, {"tassel": 0, "false_positive": 0})
            training, validation = split_samples(
                {
                    class_name: sorted((labels / class_name).glob("*.png"))
                    for class_name in ("tassel", "false_positive")
                },
                seed=42,
            )
            self.assertEqual({target for _, target in training}, {0, 1})
            self.assertEqual({target for _, target in validation}, {0, 1})

    def test_production_checkpoint_loads_and_predicts(self) -> None:
        predictor = MobileNetPredictor(PRODUCTION_CHECKPOINT)
        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        cv2.line(image, (40, 20), (60, 80), (0, 0, 180), 5)
        mask = np.zeros((100, 120), dtype=np.uint8)
        cv2.line(mask, (40, 20), (60, 80), 1, 9)
        probability = predictor.predict(image, mask)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
