import base64
import glob
import json
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from Segmentation import segment_necklace_fastsam as segmentation


def decode_png(value: str, flags: int) -> np.ndarray:
    encoded = np.frombuffer(base64.b64decode(value), dtype=np.uint8)
    return cv2.imdecode(encoded, flags)


class TasselAutonomousTests(unittest.TestCase):
    def test_absorbs_smaller_chain_fragment_touching_detected_tassel(self) -> None:
        necklace_mask = np.zeros((80, 140), dtype=np.uint8)
        necklace_mask[25:55, 10:75] = 1
        necklace_mask[30:50, 75:100] = 1
        necklace_mask[32:48, 104:132] = 1
        necklace_mask[65:70, 115:120] = 1
        parts = {
            "pendant": np.zeros_like(necklace_mask),
            "chain": np.zeros_like(necklace_mask),
            "tassel": np.zeros_like(necklace_mask),
        }
        parts["chain"][25:55, 10:75] = 1
        parts["chain"][32:48, 104:132] = 1
        parts["chain"][65:70, 115:120] = 1
        parts["tassel"][30:50, 75:100] = 1

        absorbed_area = segmentation.absorb_terminal_tassel_fragments(
            parts,
            necklace_mask,
        )

        self.assertEqual(absorbed_area, 16 * 28)
        self.assertTrue(parts["chain"][40, 30])
        self.assertTrue(parts["tassel"][40, 110])
        self.assertTrue(parts["chain"][67, 117])
        self.assertFalse(np.any(parts["chain"] & parts["tassel"]))

    def test_retained_templates_require_geometry_and_mobilenet(self) -> None:
        positives = 0
        negatives = 0

        for feedback_path in sorted(glob.glob("Segmentation/feedback/*.json")):
            with open(feedback_path, "r", encoding="utf-8") as handle:
                feedback = json.load(handle)
            if not all(
                feedback.get(key)
                for key in ("object_template_png", "object_template_mask_png")
            ):
                continue

            image = decode_png(feedback["object_template_png"], cv2.IMREAD_COLOR)
            object_mask = (
                decode_png(
                    feedback["object_template_mask_png"],
                    cv2.IMREAD_GRAYSCALE,
                )
                > 0
            ).astype(np.uint8)
            jewel_type = feedback.get("jewel_type") or "Necklace"

            if feedback.get("tassel_mask_png"):
                tassel_mask = (
                    decode_png(
                        feedback["tassel_mask_png"],
                        cv2.IMREAD_GRAYSCALE,
                    )
                    > 0
                ).astype(np.uint8)
                tassel_mask &= object_mask
                geometry = segmentation.tassel_candidate_evidence(
                    tassel_mask,
                    object_mask,
                    image,
                    jewel_type,
                )
                classifier = segmentation.classify_tassel_candidate(
                    image,
                    tassel_mask,
                )
                self.assertTrue(geometry["accepted"], feedback_path)
                self.assertTrue(classifier["accepted"], feedback_path)
                positives += 1
            elif feedback.get("no_tassel"):
                _, color_maps = segmentation.estimate_background_mask(image)
                seed, _, _, _ = segmentation.detect_tassel_seed(
                    object_mask,
                    image,
                    color_maps["textile_mask"],
                    jewel_type,
                )
                self.assertFalse(seed.any(), feedback_path)
                negatives += 1

        self.assertEqual(positives, 21)
        self.assertEqual(negatives, 7)

    def test_autonomous_run_does_not_load_saved_feedback(self) -> None:
        image = np.full((24, 24, 3), 255, dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[5:19, 5:19] = 1
        parts = {
            "pendant": np.zeros_like(mask),
            "chain": mask.copy(),
            "tassel": np.zeros_like(mask),
        }
        args = Namespace(
            output_dir="/tmp/tassel-autonomous-test",
            feedback_dir="Segmentation/feedback",
            autonomous_mode=True,
        )

        with (
            patch.object(
                segmentation,
                "load_manual_feedback",
                side_effect=AssertionError("feedback must not be loaded"),
            ),
            patch.object(
                segmentation,
                "segment_necklace",
                return_value=(parts, {}),
            ),
            patch.object(segmentation, "save_outputs"),
        ):
            _, debug = segmentation.run_segmentation(
                Path("working_source.png"),
                Path("fast_sam_s.hef"),
                object(),
                args,
                preprocessed_image=image,
                preprocessed_mask=mask,
            )

        self.assertTrue(debug["autonomous_mode"])


if __name__ == "__main__":
    unittest.main()
