from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stone_review_app import (
    COLOR_DRAW_BGR,
    ReviewImage,
    ReviewRegion,
    ReviewStore,
    discover_dataset_image_paths,
    discover_review_images,
    reconstruct_region_mask,
)


class StoneReviewAppTest(unittest.TestCase):
    def test_dataset_discovery_uses_sorted_roi_images_from_pairs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for name in (
                "Necklace_002_full.png",
                "Necklace_002_roi.png",
                "Necklace_001_full.png",
                "Necklace_001_roi.png",
            ):
                (root / name).touch()

            paths = discover_dataset_image_paths(root)

            self.assertEqual(
                [path.name for path in paths],
                ["Necklace_001_roi.png", "Necklace_002_roi.png"],
            )

    def test_reconstructs_rendered_region_at_recorded_area(self):
        color_mask = np.full((30, 40, 3), 20, dtype=np.uint8)
        original_mask = np.zeros((30, 40), dtype=np.uint8)
        cv2.circle(original_mask, (16, 14), 5, 255, cv2.FILLED)
        color_mask[original_mask > 0] = COLOR_DRAW_BGR["Red"]
        contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_mask, contours, -1, (245, 245, 245), 1)
        x, y, width, height = cv2.boundingRect(original_mask)
        region = {
            "bbox": [x, y, width, height],
            "center": [16, 14],
            "color": "Red",
            "area_px": cv2.countNonZero(original_mask),
        }

        reconstructed, exact = reconstruct_region_mask(color_mask, region)

        self.assertTrue(exact)
        self.assertTrue(np.array_equal(reconstructed, original_mask))

    def test_store_requires_color_for_true_stone_and_clears_it_for_false(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image = np.full((80, 80, 3), 180, dtype=np.uint8)
            mask = np.zeros((80, 80), dtype=np.uint8)
            mask[35:45, 35:45] = 255
            region = ReviewRegion(
                data={
                    "region_id": 1,
                    "bbox": [35, 35, 10, 10],
                    "area_px": 100,
                    "color": "Red",
                    "source_methods": ["non_gold_hsv_residual"],
                    "segmentation_method": "opencv",
                },
                mask=mask,
                exact_reconstruction=True,
                sample_id="sample-one",
            )
            item = ReviewImage(
                report_path=root / "runtime" / "s" / "main_stone_detection" / "report.json",
                session_name="s",
                stage_name="main_stone_detection",
                jewel_id=1,
                image_bgr=image,
                bbox_global=[10, 20, 80, 80],
                regions=[region],
            )
            store = ReviewStore(root / "dataset", root / "runtime")

            with self.assertRaises(ValueError):
                store.label_region(item, region, "true_stone", None)
            true_entry = store.label_region(item, region, "true_stone", "Green")
            self.assertEqual(true_entry["color"], "Green")
            self.assertTrue((root / "dataset" / true_entry["crop_path"]).is_file())
            self.assertTrue((root / "dataset" / true_entry["mask_path"]).is_file())

            false_entry = store.label_region(item, region, "false_positive", "Red")
            self.assertIsNone(false_entry["color"])
            self.assertEqual(store.counts(), {"true_stone": 0, "false_positive": 1})

    def test_discovers_report_and_preserves_newest_first_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            runtime_root = Path(temporary_dir) / "runtime_sessions"
            for index, session in enumerate(("old", "new")):
                stage = runtime_root / session / "main_stone_detection"
                stage.mkdir(parents=True)
                masked = np.dstack(
                    (
                        np.full((20, 20, 3), 180, dtype=np.uint8),
                        np.full((20, 20), 255, dtype=np.uint8),
                    )
                )
                color_mask = np.full((20, 20, 3), 20, dtype=np.uint8)
                mask = np.zeros((20, 20), dtype=np.uint8)
                mask[6:12, 7:13] = 255
                color_mask[mask > 0] = COLOR_DRAW_BGR["Green"]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(color_mask, contours, -1, (245, 245, 245), 1)
                masked_path = stage / "jewel_01_masked.png"
                color_path = stage / "jewel_01_color_mask.png"
                cv2.imwrite(str(masked_path), masked)
                cv2.imwrite(str(color_path), color_mask)
                report = {
                    "jewels": [
                        {
                            "jewel_id": 1,
                            "bbox_global": [0, 0, 20, 20],
                            "outputs": {
                                "masked_png": str(masked_path),
                                "color_mask_png": str(color_path),
                            },
                            "regions": [
                                {
                                    "region_id": 1,
                                    "bbox": [7, 6, 6, 6],
                                    "center": [9, 8],
                                    "area_px": 36,
                                    "color": "Green",
                                }
                            ],
                        }
                    ]
                }
                report_path = stage / "preprocessed_gem_report.json"
                report_path.write_text(json.dumps(report), encoding="utf-8")
                timestamp = 1000 + index
                report_path.touch()
                import os

                os.utime(report_path, (timestamp, timestamp))

            items, errors = discover_review_images(runtime_root)

            self.assertEqual(errors, [])
            self.assertEqual([item.session_name for item in items], ["new", "old"])
            self.assertTrue(items[0].regions[0].exact_reconstruction)


if __name__ == "__main__":
    unittest.main()
