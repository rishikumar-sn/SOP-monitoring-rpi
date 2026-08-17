#!/usr/bin/env python3
"""Candidate refinement and post-segmentation analysis for stone analysis V2.

This module deliberately has no Hailo dependency.  A shared FastSAM-compatible
sampler may be supplied by the integrated application; otherwise every
candidate follows the OpenCV/seed fallback path.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Candidate/refinement configuration.  These values are intentionally kept in
# one place because camera resolution and jewel scale vary in production.
FASTSAM_STONE_REFINEMENT_ENABLED = True
STONE_CANDIDATE_MIN_AREA = 6
STONE_CANDIDATE_MAX_COUNT = 48
STONE_FASTSAM_MAX_CANDIDATES = 8
STONE_CROP_PADDING_RATIO = 2.0
STONE_CROP_MIN_SIZE = 64
STONE_CROP_MAX_SIZE = 384
STONE_FASTSAM_MIN_SEED_OVERLAP = 0.45
STONE_FASTSAM_MAX_GROWTH = 18.0
STONE_FASTSAM_MAX_GOLD_OVERLAP = 0.38
STONE_METAL_RING_WIDTH = 4
STONE_BLACK_MAX_LIGHTNESS = 28.0
STONE_BLACK_MAX_CHROMA = 12.0
STONE_LOW_RISK_THRESHOLD = 5.0
STONE_MODERATE_RISK_THRESHOLD = 20.0
HIGH_RISK_STONE_THRESHOLD = 40.0


def _binary(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    result = np.asarray(mask)
    if result.ndim == 3:
        result = result[:, :, 0]
    result = np.where(result > 0, 255, 0).astype(np.uint8)
    if shape is not None and result.shape != shape:
        result = cv2.resize(
            result,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return result


def _bbox(mask: np.ndarray) -> list[int]:
    points = cv2.findNonZero(mask)
    if points is None:
        return [0, 0, 0, 0]
    return [int(value) for value in cv2.boundingRect(points)]


def _centroid(mask: np.ndarray) -> list[int]:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        x, y, width, height = _bbox(mask)
        return [x + width // 2, y + height // 2]
    return [
        int(round(moments["m10"] / moments["m00"])),
        int(round(moments["m01"] / moments["m00"])),
    ]


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _shape_metrics(mask: np.ndarray) -> dict[str, float]:
    area = max(1, int(cv2.countNonZero(mask)))
    contour = _largest_contour(mask)
    x, y, width, height = _bbox(mask)
    if contour is None:
        return {
            "compactness": 0.0,
            "solidity": 0.0,
            "aspect_ratio": 99.0,
            "bbox_area": float(max(1, width * height)),
        }
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = max(0.0, float(cv2.contourArea(contour)))
    hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
    return {
        "compactness": (
            4.0 * math.pi * contour_area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        ),
        "solidity": contour_area / hull_area,
        "aspect_ratio": max(width, height) / float(max(1, min(width, height))),
        "bbox_area": float(max(1, width * height)),
    }


def _ring(mask: np.ndarray, width: int = STONE_METAL_RING_WIDTH) -> np.ndarray:
    size = max(3, int(width) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.bitwise_and(cv2.dilate(mask, kernel), cv2.bitwise_not(mask))


def _metal_surround_ratio(mask: np.ndarray, gold_mask: np.ndarray) -> float:
    ring = _ring(mask)
    ring_area = int(cv2.countNonZero(ring))
    if ring_area <= 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(ring, gold_mask)) / float(ring_area)


def build_color_measurement_image(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    background_calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build a conservative color image without CLAHE or chroma boosting."""
    measured = image_bgr.copy()
    center = (background_calibration or {}).get("bgr_center")
    if isinstance(center, (list, tuple)) and len(center) == 3:
        reference = np.asarray(center, dtype=np.float32)
        reference_hsv = cv2.cvtColor(
            np.uint8([[np.clip(reference, 0, 255)]]),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        # Apply a white-reference gain only when the saved background sample is
        # bright and nearly neutral.  A coloured test bed is not a white card.
        if int(reference_hsv[1]) <= 45 and int(reference_hsv[2]) >= 100:
            target = float(np.mean(reference))
            gains = np.clip(target / np.maximum(reference, 1.0), 0.85, 1.18)
            measured = np.clip(
                measured.astype(np.float32) * gains.reshape(1, 1, 3),
                0,
                255,
            ).astype(np.uint8)
    measured[_binary(jewel_mask, measured.shape[:2]) == 0] = (255, 255, 255)
    return measured


def _structural_components(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    gold_mask: np.ndarray,
) -> list[dict[str, Any]]:
    jewel = _binary(jewel_mask, image_bgr.shape[:2])
    jewel_area = max(1, int(cv2.countNonZero(jewel)))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:3] - 128.0, axis=2)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 4.0)
    local_contrast = cv2.absdiff(gray, local_mean)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    canny = cv2.Canny(gray, 45, 135)
    canny = cv2.bitwise_and(canny, jewel)
    enclosed = cv2.morphologyEx(
        canny,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(enclosed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boundary = cv2.bitwise_and(
        jewel,
        cv2.bitwise_not(cv2.erode(jewel, np.ones((3, 3), np.uint8))),
    )
    components: list[dict[str, Any]] = []
    max_area = max(30, int(jewel_area * 0.12))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < STONE_CANDIDATE_MIN_AREA or area > max_area:
            continue
        mask = np.zeros_like(jewel)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        mask = cv2.bitwise_and(mask, jewel)
        area_px = int(cv2.countNonZero(mask))
        if area_px < STONE_CANDIDATE_MIN_AREA:
            continue
        metrics = _shape_metrics(mask)
        if metrics["aspect_ratio"] > 4.5 or metrics["solidity"] < 0.55:
            continue
        outline = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        outline_px = max(1, int(cv2.countNonZero(outline)))
        edge_enclosure = cv2.countNonZero(cv2.bitwise_and(outline, canny)) / float(outline_px)
        contrast_score = min(1.0, float(np.median(local_contrast[mask > 0])) / 28.0)
        gradient_score = min(1.0, float(np.median(gradient[outline > 0])) / 90.0)
        boundary_share = cv2.countNonZero(cv2.bitwise_and(mask, boundary)) / float(area_px)
        metal_surround = _metal_surround_ratio(mask, gold_mask)
        gold_continuity = cv2.countNonZero(
            cv2.bitwise_and(mask, gold_mask)
        ) / float(area_px)
        median_chroma = float(np.median(chroma[mask > 0]))
        score = (
            0.20 * min(1.0, metrics["compactness"] / 0.75)
            + 0.18 * metrics["solidity"]
            + 0.22 * min(1.0, edge_enclosure / 0.35)
            + 0.16 * contrast_score
            + 0.12 * gradient_score
            + 0.12 * metal_surround
            - 0.30 * boundary_share
            - 0.14 * gold_continuity
        )
        if score < 0.52:
            continue
        source_methods = ["structural", "edge_enclosure", "local_contrast"]
        median_l = float(np.median(lab[:, :, 0][mask > 0])) * 100.0 / 255.0
        median_b = float(np.median(lab[:, :, 2][mask > 0])) - 128.0
        if median_chroma <= 14.0:
            source_methods.append("white_colorless_structure")
        if median_b >= 13.0 and metal_surround >= 0.20:
            source_methods.append("yellow_structure")
        if median_l <= 32.0:
            source_methods.append("dark_structure")
        components.append(
            {
                "mask": mask,
                "confidence": round(float(min(0.95, score)), 3),
                "source_methods": source_methods,
                "structural_score": round(float(score), 3),
                "edge_enclosure": round(float(edge_enclosure), 3),
                "local_contrast_score": round(float(contrast_score), 3),
                "metal_surround_ratio": round(float(metal_surround), 3),
                "gold_continuity": round(float(gold_continuity), 3),
            }
        )
    components.sort(
        key=lambda item: (-float(item["confidence"]), -cv2.countNonZero(item["mask"]))
    )
    return components


def generate_stone_candidates(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    seed_regions: list[dict[str, Any]],
    gold_mask: np.ndarray,
    glare_mask: np.ndarray | None = None,
    max_candidates: int = STONE_CANDIDATE_MAX_COUNT,
) -> list[dict[str, Any]]:
    """Convert color/learned seeds and structural evidence into candidates."""
    shape = image_bgr.shape[:2]
    jewel = _binary(jewel_mask, shape)
    candidates: list[dict[str, Any]] = []
    for region in seed_regions:
        proposal = _binary(region.get("mask"), shape)
        seed = _binary(region.get("seed_mask", proposal), shape)
        seed = cv2.bitwise_and(seed, jewel)
        if cv2.countNonZero(seed) < STONE_CANDIDATE_MIN_AREA:
            continue
        sources = ["hsv_lab"]
        source_label = str(region.get("source") or "").strip()
        if source_label:
            sources.append(source_label)
        if region.get("learned_matches"):
            sources.append("learned_profile")
        expansion = region.get("stone_region_expansion") or {}
        if expansion.get("method"):
            sources.append(str(expansion["method"]))
        confidence = 0.58
        confidence += min(0.16, float(region.get("circularity", 0.0)) * 0.12)
        confidence += min(0.10, len(region.get("learned_matches") or []) * 0.10)
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "seed_mask": seed,
                "proposal_mask": proposal,
                "bbox": _bbox(seed),
                "centroid": _centroid(seed),
                "source_methods": sorted(set(sources)),
                "initial_color_votes": dict(region.get("color_mix_percent") or {
                    str(region.get("color") or "Unknown"): 100.0
                }),
                "seed_area_px": int(cv2.countNonZero(seed)),
                "confidence": round(min(0.95, confidence), 3),
                "source_region": region,
            }
        )

    for structural in _structural_components(image_bgr, jewel, gold_mask):
        mask = structural["mask"]
        area = max(1, int(cv2.countNonZero(mask)))
        duplicate: dict[str, Any] | None = None
        for candidate in candidates:
            overlap = cv2.countNonZero(cv2.bitwise_and(mask, candidate["seed_mask"]))
            if overlap / float(min(area, max(1, candidate["seed_area_px"]))) >= 0.45:
                duplicate = candidate
                break
        if duplicate is not None:
            duplicate["source_methods"] = sorted(
                set(duplicate["source_methods"] + structural["source_methods"])
            )
            duplicate["structural_score"] = structural["structural_score"]
            duplicate["metal_surround_ratio"] = structural["metal_surround_ratio"]
            duplicate["gold_continuity"] = structural["gold_continuity"]
            duplicate["confidence"] = round(
                max(float(duplicate["confidence"]), float(structural["confidence"])),
                3,
            )
            continue
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "seed_mask": mask,
                "proposal_mask": mask.copy(),
                "bbox": _bbox(mask),
                "centroid": _centroid(mask),
                "source_methods": structural["source_methods"],
                "initial_color_votes": {},
                "seed_area_px": area,
                "confidence": structural["confidence"],
                "structural_score": structural["structural_score"],
                "metal_surround_ratio": structural["metal_surround_ratio"],
                "gold_continuity": structural["gold_continuity"],
                "source_region": {},
            }
        )

    if glare_mask is not None:
        glare = cv2.bitwise_and(_binary(glare_mask, shape), jewel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(glare, 8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not 6 <= area <= max(40, int(cv2.countNonZero(jewel) * 0.02)):
                continue
            seed = np.where(labels == label, 255, 0).astype(np.uint8)
            if any(
                cv2.countNonZero(cv2.bitwise_and(seed, item["seed_mask"])) > 0
                for item in candidates
            ):
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "seed_mask": seed,
                    "proposal_mask": seed.copy(),
                    "bbox": _bbox(seed),
                    "centroid": _centroid(seed),
                    "source_methods": ["reflection_evidence"],
                    "initial_color_votes": {},
                    "seed_area_px": area,
                    # Reflection alone is deliberately below the FastSAM
                    # acceptance priority and never confirms a stone by itself.
                    "confidence": 0.42,
                    "source_region": {},
                }
            )

    candidates.sort(
        key=lambda item: (-float(item["confidence"]), -int(item["seed_area_px"]))
    )
    candidates = candidates[: max(1, int(max_candidates))]
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = index
    return candidates


def _candidate_crop_bbox(candidate: dict[str, Any], image_shape: tuple[int, int]) -> list[int]:
    x, y, width, height = candidate["bbox"]
    target = int(round(max(width, height) * (1.0 + STONE_CROP_PADDING_RATIO)))
    target = max(STONE_CROP_MIN_SIZE, min(STONE_CROP_MAX_SIZE, target))
    center_x, center_y = candidate["centroid"]
    x1 = max(0, center_x - target // 2)
    y1 = max(0, center_y - target // 2)
    x2 = min(image_shape[1], x1 + target)
    y2 = min(image_shape[0], y1 + target)
    x1 = max(0, x2 - target)
    y1 = max(0, y2 - target)
    return [int(x1), int(y1), int(x2), int(y2)]


def _refinement_metrics(
    mask: np.ndarray,
    seed: np.ndarray,
    jewel_mask: np.ndarray,
    gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
) -> dict[str, float]:
    area = max(1, int(cv2.countNonZero(mask)))
    seed_area = max(1, int(cv2.countNonZero(seed)))
    jewel_area = max(1, int(cv2.countNonZero(jewel_mask)))
    seed_overlap = cv2.countNonZero(cv2.bitwise_and(mask, seed))
    shape = _shape_metrics(mask)
    jewel_boundary = cv2.bitwise_and(
        jewel_mask,
        cv2.bitwise_not(cv2.erode(jewel_mask, np.ones((3, 3), np.uint8))),
    )
    x, y, width, height = _bbox(mask)
    sx, sy, sw, sh = _bbox(seed)
    return {
        "seed_area_px": float(seed_area),
        "refined_area_px": float(area),
        "growth_ratio": area / float(seed_area),
        "seed_overlap": seed_overlap / float(seed_area),
        "jewel_area_fraction": area / float(jewel_area),
        "gold_overlap": cv2.countNonZero(cv2.bitwise_and(mask, gold_mask)) / float(area),
        "strict_gold_overlap": cv2.countNonZero(cv2.bitwise_and(mask, strict_gold_mask)) / float(area),
        "bbox_growth": (width * height) / float(max(1, sw * sh)),
        "boundary_touch_percentage": cv2.countNonZero(
            cv2.bitwise_and(mask, jewel_boundary)
        ) / float(area),
        "compactness": shape["compactness"],
        "solidity": shape["solidity"],
        "aspect_ratio": shape["aspect_ratio"],
        "metal_surround_ratio": _metal_surround_ratio(mask, gold_mask),
    }


def _accept_refinement(metrics: dict[str, float], structural_score: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    growth_limit = min(
        STONE_FASTSAM_MAX_GROWTH,
        7.0 + max(0.0, structural_score) * 10.0,
    )
    jewel_fraction_limit = 0.12 if metrics["seed_area_px"] < 80 else 0.32
    if metrics["seed_overlap"] < STONE_FASTSAM_MIN_SEED_OVERLAP:
        reasons.append("insufficient_seed_overlap")
    if metrics["growth_ratio"] > growth_limit:
        reasons.append("excessive_area_growth")
    if metrics["jewel_area_fraction"] > jewel_fraction_limit:
        reasons.append("covers_too_much_of_jewel")
    if metrics["strict_gold_overlap"] > STONE_FASTSAM_MAX_GOLD_OVERLAP:
        reasons.append("strict_gold_overlap")
    if metrics["boundary_touch_percentage"] > 0.42:
        reasons.append("jewel_boundary_following")
    if metrics["aspect_ratio"] > 5.0 and metrics["compactness"] < 0.30:
        reasons.append("long_thin_gold_structure")
    if metrics["solidity"] < 0.35:
        reasons.append("irregular_low_solidity")
    return not reasons, reasons


def refine_candidate_with_fastsam(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    candidate: dict[str, Any],
    fastsam_model: Any,
    gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
    inference_lock: Any = None,
) -> dict[str, Any] | None:
    """Run shared FastSAM once on a padded candidate crop and score every mask."""
    if fastsam_model is None or not FASTSAM_STONE_REFINEMENT_ENABLED:
        return None
    x1, y1, x2, y2 = _candidate_crop_bbox(candidate, image_bgr.shape[:2])
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    with (inference_lock if inference_lock is not None else nullcontext()):
        detections = fastsam_model.infer(
            crop,
            conf_thres=0.20,
            iou_thres=0.70,
            mask_thres=0.50,
        )
    best: dict[str, Any] | None = None
    for detection in detections:
        local = _binary(detection.mask, crop.shape[:2])
        proposed = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        proposed[y1:y2, x1:x2] = local
        proposed = cv2.bitwise_and(proposed, jewel_mask)
        metrics = _refinement_metrics(
            proposed,
            candidate["seed_mask"],
            jewel_mask,
            gold_mask,
            strict_gold_mask,
        )
        accepted, rejected_reasons = _accept_refinement(
            metrics,
            float(candidate.get("structural_score", 0.0)),
        )
        if not accepted:
            continue
        contains_centroid = bool(
            proposed[candidate["centroid"][1], candidate["centroid"][0]] > 0
        )
        score = (
            0.24 * metrics["seed_overlap"]
            + 0.16 * min(1.0, float(detection.score))
            + 0.14 * min(1.0, metrics["compactness"] / 0.75)
            + 0.12 * metrics["solidity"]
            + 0.12 * (1.0 - min(1.0, metrics["strict_gold_overlap"]))
            + 0.10 * min(1.0, metrics["metal_surround_ratio"] / 0.55)
            + 0.12 * float(contains_centroid)
        )
        if best is None or score > best["confidence"]:
            best = {
                "mask": proposed,
                "method": "fastsam",
                "confidence": round(float(score), 3),
                "diagnostics": {
                    **{key: round(float(value), 4) for key, value in metrics.items()},
                    "contains_candidate_centroid": contains_centroid,
                    "crop_bbox": [x1, y1, x2, y2],
                    "fastsam_score": round(float(detection.score), 4),
                    "rejected_reasons": rejected_reasons,
                },
            }
    return best


def _components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    support = cv2.dilate(seed, np.ones((3, 3), np.uint8))
    keep: list[int] = []
    for label in range(1, count):
        if np.any((labels == label) & (support > 0)):
            keep.append(label)
    if not keep:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = [largest]
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


def refine_candidate_with_opencv(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    candidate: dict[str, Any],
    gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
) -> dict[str, Any] | None:
    """Use seeded GrabCut, then retain the prior conservative OpenCV proposal."""
    x1, y1, x2, y2 = _candidate_crop_bbox(candidate, image_bgr.shape[:2])
    crop = image_bgr[y1:y2, x1:x2]
    local_jewel = jewel_mask[y1:y2, x1:x2]
    local_seed = candidate["seed_mask"][y1:y2, x1:x2]
    local_proposal = candidate["proposal_mask"][y1:y2, x1:x2]
    local_strict_gold = strict_gold_mask[y1:y2, x1:x2]
    if crop.size and cv2.countNonZero(local_seed) >= STONE_CANDIDATE_MIN_AREA:
        grab_mask = np.full(local_seed.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        grab_mask[local_jewel == 0] = cv2.GC_BGD
        probable = cv2.dilate(
            cv2.bitwise_or(local_seed, local_proposal),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        grab_mask[probable > 0] = cv2.GC_PR_FGD
        definite = cv2.erode(local_seed, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(definite) == 0:
            definite = local_seed
        grab_mask[local_strict_gold > 0] = cv2.GC_BGD
        grab_mask[local_seed > 0] = cv2.GC_PR_FGD
        grab_mask[definite > 0] = cv2.GC_FGD
        try:
            cv2.grabCut(
                crop,
                grab_mask,
                None,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                2,
                cv2.GC_INIT_WITH_MASK,
            )
            local_result = np.where(
                (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            local_result = _components_touching_seed(local_result, local_seed)
            proposed = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            proposed[y1:y2, x1:x2] = local_result
            proposed = cv2.bitwise_and(proposed, jewel_mask)
            metrics = _refinement_metrics(
                proposed,
                candidate["seed_mask"],
                jewel_mask,
                gold_mask,
                strict_gold_mask,
            )
            accepted, rejected_reasons = _accept_refinement(
                metrics,
                float(candidate.get("structural_score", 0.0)),
            )
            if accepted:
                return {
                    "mask": proposed,
                    "method": "opencv",
                    "confidence": round(
                        0.46
                        + 0.22 * metrics["seed_overlap"]
                        + 0.12 * metrics["compactness"],
                        3,
                    ),
                    "diagnostics": {
                        **{key: round(float(value), 4) for key, value in metrics.items()},
                        "opencv_method": "seeded_grabcut",
                        "rejected_reasons": rejected_reasons,
                    },
                }
        except cv2.error:
            pass

    proposal = cv2.bitwise_and(candidate["proposal_mask"], jewel_mask)
    metrics = _refinement_metrics(
        proposal,
        candidate["seed_mask"],
        jewel_mask,
        gold_mask,
        strict_gold_mask,
    )
    accepted, rejected_reasons = _accept_refinement(
        metrics,
        float(candidate.get("structural_score", 0.0)),
    )
    if accepted:
        return {
            "mask": proposal,
            "method": "opencv",
            "confidence": round(0.42 + 0.20 * metrics["seed_overlap"], 3),
            "diagnostics": {
                **{key: round(float(value), 4) for key, value in metrics.items()},
                "opencv_method": "existing_adaptive_refinement",
                "rejected_reasons": rejected_reasons,
            },
        }
    return None


def refine_stone_candidate(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    candidate: dict[str, Any],
    gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
    fastsam_model: Any = None,
    inference_lock: Any = None,
    allow_fastsam: bool = True,
) -> dict[str, Any]:
    """Refine a candidate without ever forcing a rejected FastSAM mask."""
    fastsam_error: str | None = None
    if allow_fastsam and fastsam_model is not None:
        try:
            result = refine_candidate_with_fastsam(
                image_bgr,
                jewel_mask,
                candidate,
                fastsam_model,
                gold_mask,
                strict_gold_mask,
                inference_lock=inference_lock,
            )
            if result is not None:
                return result
        except Exception as exc:  # Hailo failure must not stop stone analysis.
            fastsam_error = str(exc)
    result = refine_candidate_with_opencv(
        image_bgr,
        jewel_mask,
        candidate,
        gold_mask,
        strict_gold_mask,
    )
    if result is not None:
        if fastsam_error:
            result["diagnostics"]["fastsam_error"] = fastsam_error
        return result
    seed = cv2.bitwise_and(candidate["seed_mask"], jewel_mask)
    metrics = _refinement_metrics(
        seed,
        candidate["seed_mask"],
        jewel_mask,
        gold_mask,
        strict_gold_mask,
    )
    source_methods = set(candidate.get("source_methods") or [])
    fallback_accepted = not (
        source_methods == {"reflection_evidence"}
        or (
            metrics["strict_gold_overlap"] > 0.65
            and not (
                "yellow_structure" in source_methods
                and float(candidate.get("structural_score", 0.0)) >= 0.65
                and float(candidate.get("metal_surround_ratio", 0.0)) >= 0.18
            )
            and "learned_profile" not in source_methods
        )
    )
    return {
        "mask": seed,
        "method": "seed_fallback",
        "accepted": fallback_accepted,
        "confidence": round(min(0.55, float(candidate.get("confidence", 0.4))), 3),
        "diagnostics": {
            **{key: round(float(value), 4) for key, value in metrics.items()},
            "fastsam_error": fastsam_error,
            "fallback_reason": (
                "no_acceptable_refined_mask"
                if fallback_accepted
                else "reflection_or_gold_seed_lacks_boundary_support"
            ),
        },
    }


def _watershed_split(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    min_area: int,
) -> list[np.ndarray]:
    area = int(cv2.countNonZero(mask))
    if area < min_area * 3:
        return [mask]
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak < 2.0:
        return [mask]
    peaks = np.where(distance >= peak * 0.56, 255, 0).astype(np.uint8)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, markers = cv2.connectedComponents(peaks)
    if count <= 2 or count > 9:
        return [mask]
    watershed_markers = markers.astype(np.int32) + 1
    watershed_markers[mask == 0] = 1
    watershed_markers[(mask > 0) & (peaks == 0)] = 0
    cv2.watershed(image_bgr, watershed_markers)
    parts: list[np.ndarray] = []
    for label in range(2, count + 1):
        part = np.where((watershed_markers == label) & (mask > 0), 255, 0).astype(np.uint8)
        if cv2.countNonZero(part) >= min_area:
            parts.append(part)
    retained = sum(cv2.countNonZero(part) for part in parts)
    return parts if len(parts) >= 2 and retained >= area * 0.70 else [mask]


def build_final_stone_instances(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
    fastsam_model: Any = None,
    inference_lock: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine, de-duplicate and split candidates into visible stone instances."""
    occupied = np.zeros(jewel_mask.shape, dtype=np.uint8)
    instances: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    fastsam_used = 0
    seed_distances: list[np.ndarray] = []
    for candidate in candidates:
        distance_input = np.full(jewel_mask.shape, 255, dtype=np.uint8)
        distance_input[candidate["seed_mask"] > 0] = 0
        seed_distances.append(cv2.distanceTransform(distance_input, cv2.DIST_L2, 3))
    for candidate_index, candidate in enumerate(candidates):
        allow_fastsam = bool(
            fastsam_used < STONE_FASTSAM_MAX_CANDIDATES
            and float(candidate.get("confidence", 0.0)) >= 0.55
        )
        refined = refine_stone_candidate(
            image_bgr,
            jewel_mask,
            candidate,
            gold_mask,
            strict_gold_mask,
            fastsam_model=fastsam_model,
            inference_lock=inference_lock,
            allow_fastsam=allow_fastsam,
        )
        if allow_fastsam and fastsam_model is not None:
            fastsam_used += 1
        mask = refined["mask"]
        competing_seed_indexes = [
            index
            for index, other in enumerate(candidates)
            if index != candidate_index
            and cv2.countNonZero(
                cv2.bitwise_and(mask, other["seed_mask"])
            ) > 0
        ]
        if competing_seed_indexes:
            nearest_other = np.minimum.reduce(
                [seed_distances[index] for index in competing_seed_indexes]
            )
            ownership = seed_distances[candidate_index] <= nearest_other
            mask = np.where((mask > 0) & ownership, 255, 0).astype(np.uint8)
            refined["diagnostics"]["nearest_seed_instance_separation"] = True
        area = max(1, int(cv2.countNonZero(mask)))
        overlap = cv2.countNonZero(cv2.bitwise_and(mask, occupied)) / float(area)
        reflection_only = set(candidate.get("source_methods") or []) == {
            "reflection_evidence"
        }
        accepted = (
            bool(refined.get("accepted", True))
            and not reflection_only
            and overlap < 0.62
        )
        if accepted and overlap > 0:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(occupied))
            accepted = cv2.countNonZero(mask) >= STONE_CANDIDATE_MIN_AREA
        diagnostic = {
            "candidate_id": int(candidate["candidate_id"]),
            "candidate_method": list(candidate["source_methods"]),
            "segmentation_method": refined["method"],
            "seed_area_px": int(candidate["seed_area_px"]),
            "accepted": bool(accepted),
            "candidate_confidence": float(candidate["confidence"]),
            "boundary_confidence": float(refined["confidence"]),
            "overlap_with_accepted_instances": round(float(overlap), 4),
            **refined["diagnostics"],
        }
        candidate_diagnostics.append(diagnostic)
        if not accepted:
            continue
        parts = _watershed_split(
            image_bgr,
            mask,
            max(STONE_CANDIDATE_MIN_AREA, int(candidate["seed_area_px"] * 0.25)),
        )
        for part in parts:
            source_region = dict(candidate.get("source_region") or {})
            source_region.update(
                {
                    "mask": part,
                    "seed_mask": candidate["seed_mask"],
                    "area_px": int(cv2.countNonZero(part)),
                    "bbox": _bbox(part),
                    "center": _centroid(part),
                    "candidate_id": int(candidate["candidate_id"]),
                    "source_methods": list(candidate["source_methods"]),
                    "segmentation_method": refined["method"],
                    "boundary_confidence": float(refined["confidence"]),
                    "refinement_diagnostics": dict(refined["diagnostics"]),
                    "watershed_split": len(parts) > 1,
                }
            )
            instances.append(source_region)
            occupied = cv2.bitwise_or(occupied, part)
            method_counts[refined["method"]] += 1
    for index, instance in enumerate(instances, start=1):
        instance["region_id"] = index
    return instances, {
        "candidate_count": len(candidates),
        "accepted_candidate_count": sum(bool(item["accepted"]) for item in candidate_diagnostics),
        "stone_instance_count": len(instances),
        "fastsam_candidate_calls": fastsam_used,
        "segmentation_method_counts": dict(method_counts),
        "candidates": candidate_diagnostics,
    }


def _basic_lab_color(l_star: float, a_star: float, b_star: float) -> str:
    chroma = math.hypot(a_star, b_star)
    if l_star <= STONE_BLACK_MAX_LIGHTNESS and chroma <= STONE_BLACK_MAX_CHROMA:
        return "Black"
    # Chromatic evidence takes precedence over darkness.  This is the critical
    # dark-green-versus-black rule.
    if a_star <= -7.0 and chroma >= 10.0:
        return "Green"
    if b_star <= -25.0 and a_star < 45.0:
        return "Blue"
    if a_star >= 14.0 and b_star >= -15.0 and b_star < 14.0 and l_star >= 38.0:
        return "Pink"
    if a_star >= 22.0 and b_star <= -15.0:
        return "Purple/Violet"
    if a_star >= 12.0 and b_star >= 20.0 and b_star > a_star * 1.15:
        return "Orange"
    if b_star >= 15.0 and -9.0 <= a_star <= 18.0:
        return "Yellow/Gold"
    if a_star >= 20.0 and b_star >= 8.0:
        return "Red"
    if chroma <= 12.0 and l_star >= 45.0:
        return "White/Colorless"
    if l_star <= 34.0 and chroma <= 18.0:
        return "Black"
    return "White/Colorless" if chroma < 15.0 else "Multicolor/Color-changing"


def classify_stone_instance_color(
    color_image_bgr: np.ndarray,
    instance_mask: np.ndarray,
    gold_mask: np.ndarray | None = None,
    lab_image: np.ndarray | None = None,
    hsv_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Classify one final instance from robust interior LAB statistics."""
    mask = _binary(instance_mask, color_image_bgr.shape[:2])
    area = int(cv2.countNonZero(mask))
    if area >= 30:
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(eroded) >= max(8, int(area * 0.35)):
            mask = eroded
    lab = (
        lab_image.astype(np.float32, copy=False)
        if lab_image is not None
        else cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    )
    hsv = (
        hsv_image.astype(np.float32, copy=False)
        if hsv_image is not None
        else cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    )
    l_star = lab[:, :, 0] * (100.0 / 255.0)
    a_star = lab[:, :, 1] - 128.0
    b_star = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_star, b_star)
    valid = (mask > 0) & (l_star > 8.0) & (l_star < 96.0)
    if np.count_nonzero(valid) < 6:
        valid = mask > 0
    if not np.any(valid):
        return {
            "color": "White/Colorless",
            "display_color": "White/Colorless",
            "color_confidence": 0.0,
            "lab_median": [0.0, 0.0, 0.0],
            "secondary_colors": [],
        }
    median_l = float(np.median(l_star[valid]))
    median_a = float(np.median(a_star[valid]))
    median_b = float(np.median(b_star[valid]))
    median_chroma = float(math.hypot(median_a, median_b))
    color = _basic_lab_color(median_l, median_a, median_b)

    ys, xs = np.where(valid)
    labels = [
        _basic_lab_color(float(l_star[y, x]), float(a_star[y, x]), float(b_star[y, x]))
        for y, x in zip(ys, xs)
    ]
    counts = Counter(labels)
    label_array = np.asarray(labels, dtype=object)
    min_y = int(ys.min())
    min_x = int(xs.min())
    local_ys = ys - min_y
    local_xs = xs - min_x
    local_shape = (
        int(ys.max()) - min_y + 1,
        int(xs.max()) - min_x + 1,
    )
    meaningful: list[tuple[str, int]] = []
    valid_count = max(1, len(labels))
    for label, count in counts.most_common():
        if label == "Multicolor/Color-changing" or count / float(valid_count) < 0.18:
            continue
        cluster_mask = np.zeros(local_shape, dtype=np.uint8)
        label_points = label_array == label
        cluster_mask[local_ys[label_points], local_xs[label_points]] = 255
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(cluster_mask, 8)
        largest = max(
            (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)),
            default=0,
        )
        if largest >= max(5, int(valid_count * 0.06)):
            meaningful.append((label, count))
    secondary_colors = [label for label, _ in meaningful if label != color]
    if len(meaningful) >= 2 and meaningful[1][1] / float(valid_count) >= 0.18:
        color = "Multicolor/Color-changing"
        secondary_colors = [label for label, _ in meaningful[:3]]

    dominant_share = counts.most_common(1)[0][1] / float(valid_count)
    confidence = min(0.98, 0.48 + 0.50 * dominant_share)
    display_color = (
        "Multicolor / Mixed Appearance"
        if color == "Multicolor/Color-changing"
        else color
    )
    diagnostics: dict[str, Any] = {}
    if color == "Yellow/Gold" and gold_mask is not None:
        metal_surround = _metal_surround_ratio(instance_mask, gold_mask)
        diagnostics["metal_surround_ratio"] = round(metal_surround, 3)
        if metal_surround < 0.18:
            diagnostics["classification_warning"] = "possible_yellow_gold_stone_low_enclosure"
            confidence = min(confidence, 0.55)
    return {
        "color": color,
        "display_color": display_color,
        "color_confidence": round(float(confidence), 3),
        "lab_median": [round(median_l, 2), round(median_a, 2), round(median_b, 2)],
        "lab_chroma": round(median_chroma, 2),
        "hsv_median": [
            round(float(np.median(hsv[:, :, channel][valid])), 2)
            for channel in range(3)
        ],
        "secondary_colors": secondary_colors,
        "valid_color_pixel_count": int(valid_count),
        **diagnostics,
    }


def calculate_stone_surface_risk(
    stone_percentage: float,
    stone_instance_count: int,
    reflection_risk: bool = False,
    low_threshold: float = STONE_LOW_RISK_THRESHOLD,
    moderate_threshold: float = STONE_MODERATE_RISK_THRESHOLD,
    high_threshold: float = HIGH_RISK_STONE_THRESHOLD,
) -> dict[str, Any]:
    """Classify risk from visible mask coverage, never from estimated mass."""
    coverage = max(0.0, float(stone_percentage))
    count = max(0, int(stone_instance_count))
    if count <= 0 or coverage <= 0.0:
        level, status = "NONE", "NO SIGNIFICANT STONES DETECTED"
    elif coverage > float(high_threshold):
        level, status = "HIGH", "HIGH STONE AREA — RISK"
    elif coverage >= float(moderate_threshold):
        level, status = "MODERATE", "MODERATE STONE AREA"
    else:
        level, status = "LOW", "LOW STONE AREA — NORMAL"
    return {
        "level": level,
        "status": status,
        "stone_surface_coverage_percent": round(coverage, 2),
        "stone_instance_count": count,
        "high_risk": level == "HIGH",
        "reflection_risk": bool(reflection_risk),
        "thresholds": {
            "low_percent": float(low_threshold),
            "moderate_percent": float(moderate_threshold),
            "high_percent": float(high_threshold),
        },
        "basis": "final_stone_union_mask_over_segmented_jewel_mask",
    }


def build_stone_weight_feature_vector(
    report: dict[str, Any],
    weight_estimate: dict[str, Any],
    jewel_type: str = "",
    total_jewel_weight_g: float | None = None,
    view: str = "main",
) -> dict[str, Any]:
    """Return serializable image-derived features for future calibration."""
    measurements = report.get("stone_measurements") or {}
    instances = list(measurements.get("instances") or [])
    areas = [float(item.get("area_mm2", 0.0)) for item in instances]
    diameters = [float(item.get("equivalent_diameter_mm", 0.0)) for item in instances]
    shape_counts = Counter(str(item.get("shape") or "unknown") for item in instances)
    color_counts = Counter(str(item.get("color") or "Unknown") for item in instances)

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"total": 0.0, "mean": 0.0, "median": 0.0, "minimum": 0.0, "maximum": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "total": round(float(array.sum()), 4),
            "mean": round(float(array.mean()), 4),
            "median": round(float(np.median(array)), 4),
            "minimum": round(float(array.min()), 4),
            "maximum": round(float(array.max()), 4),
        }

    reflections = [jewel.get("reflection") or {} for jewel in report.get("jewels") or []]
    return {
        "schema_version": 2,
        "jewel_type": str(jewel_type or ""),
        "view": str(view or "main"),
        "total_jewel_weight_g": (
            round(float(total_jewel_weight_g), 4)
            if total_jewel_weight_g is not None
            else None
        ),
        "stone_surface_coverage_percent": float(report.get("stone_percentage", 0.0)),
        "stone_instance_count": int(report.get("stone_instance_count", len(instances))),
        "stone_area_mm2": summary(areas),
        "equivalent_diameter_mm": summary(diameters),
        "shape_counts": dict(shape_counts),
        "color_counts": dict(color_counts),
        "reflection_coverage_percent": round(
            sum(float(item.get("coverage_percent", 0.0)) for item in reflections),
            3,
        ),
        "maximum_reflection_density_percent": round(
            max((float(item.get("local_density_percent", 0.0)) for item in reflections), default=0.0),
            3,
        ),
        "physics_minimum_g": weight_estimate.get("estimated_total_minimum_g"),
        "physics_typical_g": weight_estimate.get(
            "estimated_total_typical_g",
            weight_estimate.get("estimated_total_average_g"),
        ),
        "physics_maximum_g": weight_estimate.get("estimated_total_maximum_g"),
    }


def save_debug_artifacts(
    output_dir: Path,
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    instances: list[dict[str, Any]],
) -> dict[str, str]:
    """Save optional tuning artifacts.  The production caller keeps this off."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_union = np.zeros(jewel_mask.shape, dtype=np.uint8)
    structural_union = np.zeros(jewel_mask.shape, dtype=np.uint8)
    for candidate in candidates:
        candidate_union = cv2.bitwise_or(candidate_union, candidate["seed_mask"])
        if "structural" in candidate.get("source_methods", []):
            structural_union = cv2.bitwise_or(
                structural_union,
                candidate["seed_mask"],
            )
    instance_union = np.zeros(jewel_mask.shape, dtype=np.uint8)
    instance_ids = np.zeros((*jewel_mask.shape, 3), dtype=np.uint8)
    for index, instance in enumerate(instances, start=1):
        mask = instance["mask"]
        instance_union = cv2.bitwise_or(instance_union, mask)
        color = ((index * 83) % 255, (index * 151) % 255, (index * 211) % 255)
        instance_ids[mask > 0] = color
    overlay = image_bgr.copy()
    overlay[instance_union > 0] = cv2.addWeighted(
        overlay[instance_union > 0],
        0.45,
        np.full_like(overlay[instance_union > 0], (0, 0, 255)),
        0.55,
        0,
    )
    color_labels = image_bgr.copy()
    for instance in instances:
        center_x, center_y = _centroid(instance["mask"])
        cv2.putText(
            color_labels,
            str(instance.get("display_color") or instance.get("color") or "Stone"),
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    artifacts = {
        "01_original.jpg": image_bgr,
        "02_jewel_mask.png": jewel_mask,
        "03_candidate_seeds.png": candidate_union,
        "04_structural_candidates.png": structural_union,
        "05_refined_instances.png": instance_union,
        "06_instance_ids.png": instance_ids,
        "07_color_labels.jpg": color_labels,
        "08_final_overlay.jpg": overlay,
    }
    paths: dict[str, str] = {}
    for name, artifact in artifacts.items():
        path = output_dir / name
        cv2.imwrite(str(path), artifact)
        paths[name] = str(path)
    return paths
