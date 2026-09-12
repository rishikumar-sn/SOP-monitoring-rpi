#!/usr/bin/env python3
"""Offline replay for the proposed jewelry-mask minus gold-HSV stone logic.

This script does not participate in the integrated application. It reads saved
runtime sessions and writes comparison artifacts to a separate output folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import jewel_gem_hsv_report as stone_detection
import stone_area_calculator


MIN_COMPONENT_AREA_PX = stone_area_calculator.MIN_STONE_COMPONENT_AREA_PX


def _load_session_state(session_dir: Path) -> dict[str, Any]:
    state_path = session_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing saved state: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _load_saved_inputs(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    image_path = session_dir / "source" / "preprocessed.png"
    mask_path = session_dir / "source" / "preprocessed_mask.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing saved image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Missing saved jewelry mask: {mask_path}")
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return image, np.where(mask > 0, 255, 0).astype(np.uint8)


def _saved_main_result(state: dict[str, Any]) -> dict[str, Any]:
    stones = state.get("stone_detection") or {}
    return stones.get("main") or stones.get("side") or {}


def _calibration_from_state(state: dict[str, Any]) -> tuple[float, float]:
    calibration = _saved_main_result(state).get("area_calibration") or {}
    scale_x = calibration.get("analysis_mm_per_pixel_x")
    scale_y = calibration.get("analysis_mm_per_pixel_y")
    if scale_x is None:
        scale_x = calibration.get("mm_per_pixel_x")
    if scale_y is None:
        scale_y = calibration.get("mm_per_pixel_y")
    if not scale_x or not scale_y:
        raise ValueError("Saved session does not contain metric area calibration.")
    return float(scale_x), float(scale_y)


def _previous_result_summary(state: dict[str, Any]) -> dict[str, Any]:
    previous = _saved_main_result(state)
    previous_weight = previous.get("weight_estimate") or {}
    return {
        "stone_area_px": int(previous.get("stone_area_px", 0) or 0),
        "stone_percentage": float(previous.get("stone_percentage", 0.0) or 0.0),
        "stone_instance_count": int(previous.get("stone_instance_count", 0) or 0),
        "estimated_total_minimum_g": previous_weight.get("estimated_total_minimum_g"),
        "estimated_total_typical_g": previous_weight.get(
            "estimated_total_typical_g",
            previous_weight.get("estimated_total_average_g"),
        ),
        "estimated_total_maximum_g": previous_weight.get("estimated_total_maximum_g"),
    }


def build_gold_subtraction_masks(
    image_bgr: np.ndarray,
    jewelry_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return broad gold, raw non-gold remainder, and noise-cleaned remainder."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gold_mask = stone_detection.build_gold_mask(hsv, jewelry_mask, strict=False)
    raw_stone_mask = cv2.bitwise_and(
        jewelry_mask,
        cv2.bitwise_not(gold_mask),
    )
    clean_stone_mask = stone_area_calculator.remove_small_components(
        raw_stone_mask,
        MIN_COMPONENT_AREA_PX,
    )
    return gold_mask, raw_stone_mask, clean_stone_mask


def _colorized_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = np.full((*mask.shape[:2], 3), 255, dtype=np.uint8)
    output[mask > 0] = color
    return output


def _build_overlay(
    image_bgr: np.ndarray,
    jewelry_mask: np.ndarray,
    gold_mask: np.ndarray,
    stone_mask: np.ndarray,
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[jewelry_mask == 0] = 255
    colors = overlay.copy()
    colors[gold_mask > 0] = (0, 215, 255)
    colors[stone_mask > 0] = (255, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.55, colors, 0.45, 0.0)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (stone_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (180, 0, 180), 2)
    cv2.putText(
        overlay,
        "Gold HSV removed | Magenta = non-gold remainder",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return overlay


def replay_session(session_dir: Path, output_root: Path) -> dict[str, Any]:
    state = _load_session_state(session_dir)
    image, jewelry_mask = _load_saved_inputs(session_dir)
    gold_mask, raw_stone_mask, stone_mask = build_gold_subtraction_masks(
        image,
        jewelry_mask,
    )

    scale_x, scale_y = _calibration_from_state(state)
    measurements = stone_area_calculator.calculate_stone_measurements(
        {"Gold HSV remainder": stone_mask},
        scale_x,
        scale_y,
        min_component_area_pixels=MIN_COMPONENT_AREA_PX,
    )
    visible_stone_area_mm2 = (
        float(cv2.countNonZero(stone_mask)) * scale_x * scale_y
    )
    setting_profile = stone_area_calculator.normalize_stone_setting_profile(
        (state.get("stone_detection") or {}).get("setting_profile")
    )
    jewel_weight = (state.get("weight_details") or {}).get("jewel_weight_g")
    weight_estimate = stone_area_calculator.apply_stone_setting_weight_model(
        measurements,
        setting_profile,
        visible_stone_area_mm2,
        jewel_weight,
    )

    jewel_px = int(cv2.countNonZero(jewelry_mask))
    gold_px = int(cv2.countNonZero(gold_mask))
    stone_px = int(cv2.countNonZero(stone_mask))
    result = {
        "session_id": session_dir.name,
        "logic": "saved jewelry mask - broad gold HSV mask = stone mask",
        "source_image": str(session_dir / "source" / "preprocessed.png"),
        "source_mask": str(session_dir / "source" / "preprocessed_mask.png"),
        "jewelry_area_px": jewel_px,
        "gold_area_px": gold_px,
        "gold_percentage_of_jewelry": round(
            gold_px / float(jewel_px) * 100.0 if jewel_px else 0.0,
            2,
        ),
        "stone_area_px": stone_px,
        "stone_percentage_of_jewelry": round(
            stone_px / float(jewel_px) * 100.0 if jewel_px else 0.0,
            2,
        ),
        "stone_instance_count": int(measurements.get("instance_count", 0) or 0),
        "mm_per_pixel_x": scale_x,
        "mm_per_pixel_y": scale_y,
        "visible_stone_area_mm2": round(visible_stone_area_mm2, 4),
        "jewel_weight_g": jewel_weight,
        "stone_setting_profile": setting_profile,
        "weight_estimate": weight_estimate,
        "previous_production_result": _previous_result_summary(state),
        "warning": (
            "This prototype intentionally treats every non-gold pixel inside the "
            "saved jewelry mask as stone. It does not reject thread, tassel, beads, "
            "shadows, dark metal, or gold missed by the HSV range."
        ),
    }

    output_dir = output_root / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = _build_overlay(image, jewelry_mask, gold_mask, stone_mask)
    comparison = cv2.hconcat(
        [
            image,
            _colorized_mask(gold_mask, (0, 215, 255)),
            _colorized_mask(stone_mask, (255, 0, 255)),
        ]
    )
    cv2.imwrite(str(output_dir / "gold_mask.png"), gold_mask)
    cv2.imwrite(str(output_dir / "stone_mask_raw.png"), raw_stone_mask)
    cv2.imwrite(str(output_dir / "stone_mask_clean.png"), stone_mask)
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "comparison.png"), comparison)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay gold-HSV subtraction stone masking on saved sessions."
    )
    parser.add_argument(
        "session_dirs",
        nargs="+",
        type=Path,
        help="Saved runtime session directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("offline_results/gold_hsv_stone_replay"),
        help="Separate output directory; production session data is not modified.",
    )
    args = parser.parse_args()

    results = [
        replay_session(session_dir.resolve(), args.output.resolve())
        for session_dir in args.session_dirs
    ]
    summary_path = args.output.resolve() / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        estimate = result.get("weight_estimate") or {}
        print(
            f"{result['session_id']}: "
            f"stones={result['stone_percentage_of_jewelry']:.2f}% "
            f"instances={result['stone_instance_count']} "
            f"grams={estimate.get('estimated_total_minimum_g')}.."
            f"{estimate.get('estimated_total_maximum_g')}"
        )
    print(f"Saved offline replay to {summary_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
