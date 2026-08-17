#!/usr/bin/env python3
"""
Clean jewel masking + HSV gemstone color analysis.

Reference:
- jewel extraction follows the Otsu + contour-hierarchy cleanup idea from otsusave.py
- gemstone colors are reported as "possible types" from the user-provided color table

Example:
    python jewel_gem_hsv_report.py captures\roi_20260501_145621.jpg
    python jewel_gem_hsv_report.py captures --output gem_outputs
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

try:
    from sahi.slicing import get_slice_bboxes as sahi_get_slice_bboxes
except ImportError:
    sahi_get_slice_bboxes = None

try:
    import stone_area_calculator as _stone_area_calc
    _STONE_AREA_CALC_AVAILABLE = True
except ImportError:
    _stone_area_calc = None  # type: ignore[assignment]
    _STONE_AREA_CALC_AVAILABLE = False

try:
    import stone_analysis_v2 as _stone_v2
    _STONE_V2_AVAILABLE = True
except ImportError:
    _stone_v2 = None  # type: ignore[assignment]
    _STONE_V2_AVAILABLE = False

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Minimum stone component area used for stone_area_calculator noise removal.
# Exposed here so it can be tuned via the existing configuration system.
MIN_STONE_COMPONENT_AREA_PX: int = 5
# Minimum area (px) for the LARGEST single detected region of a color class.
# Colors where every region is smaller than this threshold are considered
# scattered reflection noise (e.g. gold chain links) and are suppressed.
MIN_SINGLE_STONE_REGION_PX: int = 80
# Also applied as a fraction of jewel area so large jewels have a proportional floor.
MIN_SINGLE_STONE_REGION_JEWEL_FRACTION: float = 0.001
MICRO_STONE_MIN_AREA_PX: int = 4
MICRO_STONE_MAX_AREA_PX: int = 80
MICRO_STONE_MIN_GROUP_COUNT: int = 3
MICRO_STONE_MIN_GROUP_AREA_PX: int = 14
MICRO_STONE_LINK_DISTANCE_PX: float = 24.0
MICRO_STONE_MIN_LOCAL_JEWEL_DENSITY: float = 0.24

GEMSTONE_OPTIONS = {
    "Red": ["Ruby", "Garnet", "Spinel", "Red Coral", "Rhodolite", "Pyrope"],
    "Blue": ["Sapphire", "Aquamarine", "Blue Topaz", "Turquoise", "Lapis Lazuli", "Kyanite", "Iolite"],
    "Green": ["Emerald", "Peridot", "Jade", "Malachite", "Green Tourmaline", "Tsavorite"],
    "Yellow/Gold": ["Yellow Sapphire", "Citrine", "Yellow Topaz", "Heliodor", "Golden Beryl"],
    "Purple/Violet": ["Amethyst", "Purple Sapphire", "Fluorite", "Charoite"],
    "Pink": ["Rose Quartz", "Pink Sapphire", "Morganite", "Pink Tourmaline", "Rhodonite"],
    "Orange": ["Fire Opal", "Carnelian", "Spessartite Garnet", "Orange Sapphire"],
    "Black": ["Onyx", "Black Diamond", "Obsidian", "Hematite", "Black Spinel"],
    "White/Colorless": ["Diamond", "White Sapphire", "Quartz", "Moonstone", "Goshenite"],
    "Multicolor/Color-changing": ["Opal", "Alexandrite", "Tourmaline", "Labradorite", "Fluorite"],
}

COLOR_DRAW_BGR = {
    "Red": (40, 40, 230),
    "Blue": (230, 120, 30),
    "Green": (60, 200, 70),
    "Yellow/Gold": (30, 210, 245),
    "Purple/Violet": (180, 70, 215),
    "Pink": (180, 120, 255),
    "Orange": (0, 150, 255),
    "Black": (30, 30, 30),
    "White/Colorless": (240, 240, 240),
    "Multicolor/Color-changing": (255, 180, 0),
}

ANNOTATION_TEXT_BGR = (45, 45, 45)
ANNOTATION_MUTED_TEXT_BGR = (105, 105, 105)
ANNOTATION_CARD_BGR = (252, 252, 252)
ANNOTATION_BORDER_BGR = (218, 218, 218)

HSV_COLOR_RANGES = {
    "Red": [((0, 150, 45), (8, 255, 255)), ((170, 190, 45), (179, 255, 255))],
    "Orange": [((9, 80, 55), (18, 255, 255))],
    "Yellow/Gold": [((19, 55, 55), (39, 255, 255))],
    "Green": [((40, 40, 40), (84, 255, 255))],
    "Blue": [((85, 45, 40), (135, 255, 255))],
    "Purple/Violet": [((136, 40, 40), (159, 255, 255))],
    "Pink": [((160, 20, 80), (179, 189, 255))],
}


Rect = tuple[int, int, int, int]

DEFAULT_SAHI_ENABLED = True
DEFAULT_SAHI_SLICE_SIZE = 256
DEFAULT_SAHI_OVERLAP = 0.30
DEFAULT_GLARE_REMOVAL_ENABLED = True
DEFAULT_GLARE_THRESHOLD = 225
DEFAULT_GLARE_PATCH_SIZE = 14
DEFAULT_GLARE_SATURATION_MAX = 60
DEFAULT_COLOR_CORRECTION = {
    # Display-only controls. Analysis never uses this saturation multiplier.
    "saturation": 1.0,
    "contrast": 1.05,
    "brightness": 0.0,
}
DEFAULT_ANALYSIS_NORMALIZATION = {
    "white_balance": True,
    "clahe_clip_limit": 1.8,
    "shadow_gamma": 0.92,
    "color_boost": 1.0,
    "green_recovery": 1.15,
    "dark_green_recovery": 1.20,
}
BACKGROUND_BOUNDARY_WIDTH_PX = 4
BACKGROUND_MATCH_REJECT_SHARE = 0.42
BACKGROUND_HOLE_MAX_AREA_FRACTION = 0.20
BACKGROUND_HOLE_MIN_RING_SUPPORT = 0.70
BACKGROUND_HOLE_MAX_LIGHTNESS_STD = 11.0
BACKGROUND_HOLE_MAX_CHROMA_STD = 6.0
REFLECTION_COVERAGE_FLAG_PERCENT = 1.50
REFLECTION_LOCAL_DENSITY_FLAG_PERCENT = 18.0
MIN_GOLD_PIXELS_FOR_STONE_DETECTION = 120
MIN_GOLD_RATIO_FOR_STONE_DETECTION = 0.012

# Thin/small ROI edge protection: when erosion consumes >50% of mask or
# the area is below threshold, white-stone edge filters tighten to avoid
# false positives from background boundary artifacts on delicate gold details.
TINY_ROI_AREA_THRESHOLD = 8000
TINY_ROI_EROSION_RATIO = 0.50
TINY_BORDER_SHARE_MAX = 0.06
TINY_MAX_DEPTH_MIN = 3.5
TINY_DEEP_SHARE_MIN = 0.22
TINY_CORE_SHARE_MIN = 0.995

# White stone edge artifact filtering: prevents false white detection from weak glare removal
# at sparse gold region boundaries where background white leaks through
SPARSE_GOLD_REGION_THRESHOLD = 300  # Below this gold px count = sparse region
SPARSE_GOLD_WHITE_STONE_MIN_SIZE = 50  # Minimum white stone size in sparse regions
SPARSE_GOLD_WHITE_BORDER_SHARE_MAX = 0.02  # Max border touching in sparse regions
SPARSE_GOLD_WHITE_GOLD_NEIGHBOR_MIN = 0.35  # Min direct gold pixel adjacency for white stones in sparse
WEAK_GLARE_THRESHOLD = 200  # Below this = weak glare removal mode active
WEAK_GLARE_WHITE_EDGE_BUFFER = 3  # Extra erosion iterations for edge filtering when glare is weak

# Advanced glare detection: statistical approach for dynamic thresholds
# Use mean - 2σ for more robust saturation gating across varying jewelry
GLARE_SAT_SIGMA_THRESHOLD = 2.0  # Saturation: mean - 2*std cutoff
GLARE_DYNAMIC_SAT_MIN = 20  # Absolute floor for saturation in glare detection
GLARE_VALUE_PERCENTILE = 98  # High value percentile for glare pixels
GLARE_LOCAL_CONTRAST_FACTOR = 1.5  # Multiplier for local contrast in specular detection

# LAB color space for white/colorless stone validation
# White diamonds/gems have distinct A-B signature even under lighting variation
LAB_WHITE_A_MIN = -5  # Min A channel value for white stones
LAB_WHITE_A_MAX = 8  # Max A channel value for white stones  
LAB_WHITE_B_MIN = -8  # Min B channel value for white stones
LAB_WHITE_B_MAX = 10  # Max B channel value for white stones
LAB_WHITE_L_MIN = 115  # Min L (lightness) for white/colorless stones
LAB_SATURATION_MAX = 8  # Max sqrt(A^2 + B^2) for white stones in LAB

# Geometric filtering for false white stone elimination
STONE_CIRCULARITY_MIN = 0.50  # Real stones are compact (> 0.5), edge strips < 0.2
STONE_CIRCULARITY_MAX = 0.98  # Reject near-perfect circles (likely noise)
METAL_PROXIMITY_PIXELS = 8  # Stone must have metal within this distance
SAHI_TILE_BORDER_DEAD_ZONE = 8  # Dead zone pixels around each SAHI tile boundary
SAHI_DUPLICATE_IOU_THRESHOLD = 0.55  # Minimum IoU to consider duplicates across tiles

# Stone masks start as conservative HSV seeds, then grow across the complete
# local stone face. This makes the measured area much less sensitive to a
# highlight changing a black pixel to gray or a colored pixel to low saturation.
STONE_REGION_GROW_ENABLED = True
STONE_REGION_GROW_MIN_RADIUS_PX = 3
STONE_REGION_GROW_MAX_RADIUS_PX = 16
STONE_REGION_GROW_BLACK_MAX_JEWEL_FRACTION = 0.06
STONE_REGION_GROW_COLOR_MAX_JEWEL_FRACTION = 0.06
STONE_REGION_GROW_BLACK_MAX_SEED_MULTIPLIER = 24.0
STONE_REGION_GROW_COLOR_MAX_SEED_MULTIPLIER = 12.0
STONE_REGION_GROW_COLOR_MAX_RADIUS_PX = 18
STONE_REGION_GROW_COLOR_MIN_SEED_AREA_PX = 12
STONE_REGION_GROW_COLOR_MAX_GOLD_OVERLAP_SHARE = 0.25
STONE_REGION_GROW_MIN_AREA_GAIN = 1.08
STONE_REGION_GROW_MAX_GOLD_PIXELS = 0
STONE_REGION_GROW_GRABCUT_ITERATIONS = 1
COLOR_STONE_REGION_GROW_COLORS = {
    "Red",
    "Blue",
    "Green",
    "Purple/Violet",
    "Pink",
    "Multicolor/Color-changing",
}

# Reflective gold can create many tiny warm-white islands that look like
# colorless stones after broad-gold subtraction. Suppress only the clustered
# gold-reflection pattern so a single real white stone is still allowed.
REFLECTIVE_WHITE_ONLY_MAX_PERCENT = 4.0
REFLECTIVE_WHITE_ONLY_MIN_REGIONS = 4
REFLECTIVE_WHITE_ONLY_MAX_LARGEST_REGION_PX = 180
REFLECTIVE_WHITE_ONLY_MAX_LARGEST_JEWEL_FRACTION = 0.008
REFLECTIVE_WHITE_WARM_HUE_MIN = 8.0
REFLECTIVE_WHITE_WARM_HUE_MAX = 45.0
REFLECTIVE_WHITE_WARM_SAT_MIN = 14.0
REFLECTIVE_WHITE_WARM_VALUE_MIN = 185.0
REFLECTIVE_WHITE_WARM_LAB_B_MIN = 5.5
REFLECTIVE_EDGE_MICRO_AREA_MAX_PX = 12

KERNEL_ELLIPSE_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
KERNEL_ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
HSV_COLOR_RANGE_ARRAYS = {
    color_name: [
        (np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        for lower, upper in ranges
    ]
    for color_name, ranges in HSV_COLOR_RANGES.items()
}
GOLD_RANGE_BROAD_A = (np.array([8, 35, 35], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
GOLD_RANGE_BROAD_B = (np.array([26, 20, 25], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
GOLD_RANGE_STRICT = (np.array([14, 80, 40], dtype=np.uint8), np.array([40, 255, 245], dtype=np.uint8))


def expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            parent = path.parent if str(path.parent) not in {"", "."} else Path(".")
            paths.extend(sorted(p for p in parent.glob(path.name) if p.suffix.lower() in IMAGE_EXTENSIONS))
            continue
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
            continue
        raise FileNotFoundError(f"Input not found or not an image: {item}")
    return paths


def normalize_ignore_regions(
    ignore_regions: list[Rect] | None,
    image_shape: tuple[int, ...],
) -> list[Rect]:
    if not ignore_regions:
        return []

    image_h, image_w = image_shape[:2]
    normalized: list[Rect] = []
    for region in ignore_regions:
        if len(region) != 4:
            continue

        x, y, w, h = (int(round(value)) for value in region)
        x0 = max(0, min(image_w, min(x, x + w)))
        y0 = max(0, min(image_h, min(y, y + h)))
        x1 = max(0, min(image_w, max(x, x + w)))
        y1 = max(0, min(image_h, max(y, y + h)))
        if x1 <= x0 or y1 <= y0:
            continue
        normalized.append((x0, y0, x1 - x0, y1 - y0))

    return normalized


def apply_ignore_regions_to_image(
    image_bgr: np.ndarray,
    ignore_regions: list[Rect] | None,
    fill_bgr: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, list[Rect]]:
    normalized = normalize_ignore_regions(ignore_regions, image_bgr.shape)
    if not normalized:
        return image_bgr.copy(), []

    masked_image = image_bgr.copy()
    fill = np.array(fill_bgr, dtype=np.uint8)
    for x, y, w, h in normalized:
        masked_image[y:y + h, x:x + w] = fill
    return masked_image, normalized


def apply_mask_to_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    fill_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    masked_image = np.full_like(image_bgr, fill_bgr)
    masked_image[mask > 0] = image_bgr[mask > 0]
    return masked_image


def normalize_color_correction(settings: dict | None) -> dict[str, float]:
    raw = settings or {}
    return {
        "saturation": max(0.50, min(2.50, float(raw.get("saturation", DEFAULT_COLOR_CORRECTION["saturation"])))),
        "contrast": max(0.70, min(1.60, float(raw.get("contrast", DEFAULT_COLOR_CORRECTION["contrast"])))),
        "brightness": max(-60.0, min(60.0, float(raw.get("brightness", DEFAULT_COLOR_CORRECTION["brightness"])))),
    }


def normalize_analysis_normalization(settings: dict | None) -> dict:
    raw = settings or {}
    return {
        "white_balance": bool(raw.get("white_balance", True)),
        "clahe_clip_limit": max(
            1.0,
            min(
                4.0,
                float(
                    raw.get(
                        "clahe_clip_limit",
                        DEFAULT_ANALYSIS_NORMALIZATION["clahe_clip_limit"],
                    )
                ),
            ),
        ),
        "shadow_gamma": max(
            0.70,
            min(
                1.20,
                float(
                    raw.get(
                        "shadow_gamma",
                        DEFAULT_ANALYSIS_NORMALIZATION["shadow_gamma"],
                    )
                ),
            ),
        ),
        "color_boost": max(
            1.0,
            min(
                2.5,
                float(
                    raw.get(
                        "color_boost",
                        DEFAULT_ANALYSIS_NORMALIZATION["color_boost"],
                    )
                ),
            ),
        ),
        "green_recovery": max(
            1.0,
            min(
                1.5,
                float(
                    raw.get(
                        "green_recovery",
                        DEFAULT_ANALYSIS_NORMALIZATION["green_recovery"],
                    )
                ),
            ),
        ),
        "dark_green_recovery": max(
            1.0,
            min(
                1.6,
                float(
                    raw.get(
                        "dark_green_recovery",
                        DEFAULT_ANALYSIS_NORMALIZATION["dark_green_recovery"],
                    )
                ),
            ),
        ),
    }


def normalize_learned_stone_profiles(profiles: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    for index, raw in enumerate(profiles or []):
        if not isinstance(raw, dict):
            continue
        color = str(raw.get("color") or "").strip()
        hsv_center = raw.get("hsv_center")
        hsv_tolerance = raw.get("hsv_tolerance")
        lab_center = raw.get("lab_center")
        lab_tolerance = raw.get("lab_tolerance")
        if color not in GEMSTONE_OPTIONS:
            continue
        if not (
            isinstance(hsv_center, (list, tuple))
            and len(hsv_center) == 3
            and isinstance(hsv_tolerance, (list, tuple))
            and len(hsv_tolerance) == 3
        ):
            continue
        try:
            hsv_center_values = [
                float(hsv_center[0]) % 180.0,
                max(0.0, min(255.0, float(hsv_center[1]))),
                max(0.0, min(255.0, float(hsv_center[2]))),
            ]
            hsv_tolerance_values = [
                max(1.0, min(40.0, float(hsv_tolerance[0]))),
                max(4.0, min(140.0, float(hsv_tolerance[1]))),
                max(4.0, min(140.0, float(hsv_tolerance[2]))),
            ]
        except (TypeError, ValueError):
            continue
        normalized_lab_center = None
        normalized_lab_tolerance = None
        if (
            isinstance(lab_center, (list, tuple))
            and len(lab_center) == 3
            and isinstance(lab_tolerance, (list, tuple))
            and len(lab_tolerance) == 3
        ):
            try:
                normalized_lab_center = [
                    max(0.0, min(255.0, float(value)))
                    for value in lab_center
                ]
                normalized_lab_tolerance = [
                    max(3.0, min(90.0, float(value)))
                    for value in lab_tolerance
                ]
            except (TypeError, ValueError):
                normalized_lab_center = None
                normalized_lab_tolerance = None
        profile_id = str(raw.get("id") or f"learned-{index + 1}").strip()
        label = str(raw.get("label") or f"Learned {color}").strip()
        try:
            sample_area_px = max(
                0,
                int(raw.get("sample_component_area_px", 0)),
            )
        except (TypeError, ValueError):
            sample_area_px = 0
        normalized.append(
            {
                "id": profile_id[:80],
                "label": label[:80],
                "color": color,
                "hsv_center": [round(value, 2) for value in hsv_center_values],
                "hsv_tolerance": [
                    round(value, 2) for value in hsv_tolerance_values
                ],
                "lab_center": (
                    [round(value, 2) for value in normalized_lab_center]
                    if normalized_lab_center is not None
                    else None
                ),
                "lab_tolerance": (
                    [round(value, 2) for value in normalized_lab_tolerance]
                    if normalized_lab_tolerance is not None
                    else None
                ),
                "sampled_at": raw.get("sampled_at"),
                "sample_component_area_px": sample_area_px,
            }
        )
    return normalized


def build_learned_profile_masks(
    hsv_image: np.ndarray,
    lab_image: np.ndarray,
    jewel_mask: np.ndarray,
    learned_profiles: list[dict] | None,
    color_boost: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    profile_masks: dict[str, np.ndarray] = {}
    profiles_by_id: dict[str, dict] = {}
    boost = max(1.0, min(2.5, float(color_boost)))
    hsv_float = hsv_image.astype(np.float32)
    lab_float = lab_image.astype(np.float32)
    for profile in normalize_learned_stone_profiles(learned_profiles):
        profile_id = profile["id"]
        center = profile["hsv_center"]
        tolerance = profile["hsv_tolerance"]
        neutral_profile = profile["color"] in {"White/Colorless", "Black"}
        hue_tolerance = min(float(tolerance[0]), 16.0 if neutral_profile else 10.0)
        saturation_tolerance = min(float(tolerance[1]), 45.0)
        value_tolerance = min(float(tolerance[2]), 50.0)
        hue_ok = hue_distance(hsv_float[:, :, 0], center[0]) <= (
            hue_tolerance * (0.95 + (0.08 * boost))
        )
        sat_ok = np.abs(hsv_float[:, :, 1] - center[1]) <= (
            saturation_tolerance * (0.95 + (0.10 * boost))
        )
        val_ok = np.abs(hsv_float[:, :, 2] - center[2]) <= (
            value_tolerance * (0.95 + (0.08 * boost))
        )
        matched = sat_ok & val_ok
        if not neutral_profile or float(center[1]) >= 35.0:
            matched &= hue_ok
        lab_center = profile.get("lab_center")
        lab_tolerance = profile.get("lab_tolerance")
        if lab_center is not None and lab_tolerance is not None:
            lab_ok = np.ones(hsv_image.shape[:2], dtype=bool)
            for channel in range(3):
                effective_tolerance = min(float(lab_tolerance[channel]), 32.0)
                lab_ok &= (
                    np.abs(lab_float[:, :, channel] - float(lab_center[channel]))
                    <= effective_tolerance * (0.95 + (0.06 * boost))
                )
            matched &= lab_ok
        mask = (matched.astype(np.uint8) * 255)
        mask = cv2.bitwise_and(mask, jewel_mask)
        if cv2.countNonZero(mask) > 0:
            profile_masks[profile_id] = mask
            profiles_by_id[profile_id] = profile
    return profile_masks, profiles_by_id


def build_normalized_analysis_image(
    image_bgr: np.ndarray,
    settings: dict | None = None,
    background_calibration: dict | None = None,
) -> np.ndarray:
    """
    Build the classical-analysis image without globally multiplying saturation.

    White balance is estimated from the saved neutral test-bed sample. CLAHE
    and gamma are applied only to LAB lightness; LAB chroma is preserved.
    """
    normalized_settings = normalize_analysis_normalization(settings)
    working = image_bgr.astype(np.float32)

    if normalized_settings["white_balance"] and background_calibration:
        center = background_calibration.get("bgr_center")
        if isinstance(center, (list, tuple)) and len(center) == 3:
            center_array = np.asarray(center, dtype=np.float32)
            if np.all(center_array > 1.0):
                neutral = float(np.mean(center_array))
                gains = np.clip(neutral / center_array, 0.70, 1.40)
                working *= gains.reshape(1, 1, 3)

    balanced = np.clip(working, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=normalized_settings["clahe_clip_limit"],
        tileGridSize=(8, 8),
    )
    lightness = clahe.apply(lightness)
    gamma = normalized_settings["shadow_gamma"]
    if abs(gamma - 1.0) > 1e-3:
        lut = np.array(
            [
                np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255)
                for value in range(256)
            ],
            dtype=np.uint8,
        )
        lightness = cv2.LUT(lightness, lut)
    return cv2.cvtColor(
        cv2.merge([lightness, channel_a, channel_b]),
        cv2.COLOR_LAB2BGR,
    )


def apply_color_correction(
    image_bgr: np.ndarray,
    settings: dict | None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    correction = normalize_color_correction(settings)
    working = image_bgr.astype(np.float32)
    working = (working - 127.5) * correction["contrast"] + 127.5 + correction["brightness"]
    working = np.clip(working, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * correction["saturation"], 0, 255)
    corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if mask is None:
        return corrected

    result = image_bgr.copy()
    result[mask > 0] = corrected[mask > 0]
    return result


def hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    direct = np.abs(hue.astype(np.float32) - float(center))
    return np.minimum(direct, 180.0 - direct)


def build_background_match_mask(
    image_bgr: np.ndarray,
    calibration: dict | None,
) -> np.ndarray:
    if not calibration:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    hsv_center = calibration.get("hsv_center")
    hsv_tolerance = calibration.get("hsv_tolerance")
    lab_center = calibration.get("lab_center")
    lab_tolerance = calibration.get("lab_tolerance")
    if not (
        isinstance(hsv_center, (list, tuple))
        and len(hsv_center) == 3
        and isinstance(hsv_tolerance, (list, tuple))
        and len(hsv_tolerance) == 3
    ):
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue_ok = hue_distance(hsv[:, :, 0], float(hsv_center[0])) <= float(hsv_tolerance[0])
    sat_ok = np.abs(hsv[:, :, 1].astype(np.float32) - float(hsv_center[1])) <= float(hsv_tolerance[1])
    val_ok = np.abs(hsv[:, :, 2].astype(np.float32) - float(hsv_center[2])) <= float(hsv_tolerance[2])
    if bool(calibration.get("low_saturation", False)):
        hsv_ok = sat_ok & val_ok
    else:
        hsv_ok = hue_ok & sat_ok & val_ok

    if (
        isinstance(lab_center, (list, tuple))
        and len(lab_center) == 3
        and isinstance(lab_tolerance, (list, tuple))
        and len(lab_tolerance) == 3
    ):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_ok = np.ones(image_bgr.shape[:2], dtype=bool)
        for channel in range(3):
            lab_ok &= np.abs(lab[:, :, channel] - float(lab_center[channel])) <= float(lab_tolerance[channel])
        hsv_ok &= lab_ok

    return (hsv_ok.astype(np.uint8) * 255)


def build_inner_boundary_band(mask: np.ndarray, width_px: int) -> tuple[np.ndarray, np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    band = np.where((binary > 0) & (distance <= max(1, int(width_px))), 255, 0).astype(np.uint8)
    return band, distance


def build_strict_background_match_mask(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    calibration: dict | None = None,
) -> np.ndarray:
    """Find pixels that closely match the actual test-bed background.

    The adaptive signature is sampled from pixels outside the jewel mask in
    the padded crop. When a saved calibration exists, both tests must agree.
    This intentionally uses tighter tolerances than gemstone classification.
    """
    binary = (jewel_mask > 0).astype(np.uint8)
    outside = binary == 0
    if int(np.count_nonzero(outside)) < 12:
        outside = np.zeros(binary.shape, dtype=bool)
        border_width = max(1, min(6, min(binary.shape) // 8))
        outside[:border_width, :] = True
        outside[-border_width:, :] = True
        outside[:, :border_width] = True
        outside[:, -border_width:] = True

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[outside]
    if samples.size == 0:
        return build_background_match_mask(image_bgr, calibration)

    center = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - center), axis=0)
    tolerances = np.array(
        [
            max(4.0, min(14.0, 3.0 * float(mad[0]) + 3.0)),
            max(3.0, min(10.0, 3.0 * float(mad[1]) + 2.0)),
            max(3.0, min(10.0, 3.0 * float(mad[2]) + 2.0)),
        ],
        dtype=np.float32,
    )
    adaptive = np.all(np.abs(lab - center.reshape(1, 1, 3)) <= tolerances, axis=2)
    adaptive_mask = adaptive.astype(np.uint8) * 255

    calibrated_mask = build_background_match_mask(image_bgr, calibration)
    if calibration and cv2.countNonZero(calibrated_mask) > 0:
        adaptive_mask = cv2.bitwise_and(adaptive_mask, calibrated_mask)
    return adaptive_mask


def remove_enclosed_background_from_jewel_mask(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    calibration: dict | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove uniform background islands accidentally filled by Otsu/closing.

    Only enclosed, background-colored components are removed. Bright stones
    are retained when they contain facet/edge variation or do not closely
    match the sampled background.
    """
    binary = np.where(jewel_mask > 0, 255, 0).astype(np.uint8)
    jewel_area = int(cv2.countNonZero(binary))
    stats = {
        "measurement_space": "native_crop_pixels",
        "input_jewel_area_px": jewel_area,
        "background_hole_pixels_removed": 0,
        "background_hole_count": 0,
        "refined_jewel_area_px": jewel_area,
    }
    if jewel_area <= 0:
        return binary, stats

    strict_background = build_strict_background_match_mask(
        image_bgr,
        binary,
        calibration,
    )
    candidates = cv2.bitwise_and(strict_background, binary)
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        (candidates > 0).astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return binary, stats

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    max_component_area = max(
        8,
        int(round(jewel_area * BACKGROUND_HOLE_MAX_AREA_FRACTION)),
    )
    min_component_area = max(3, int(round(jewel_area * 0.00001)))
    refined = binary.copy()
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for label in range(1, count):
        area = int(component_stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area or area > max_component_area:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        ring = cv2.subtract(cv2.dilate(component, ring_kernel, iterations=1), component)
        ring_px = int(cv2.countNonZero(ring))
        if ring_px <= 0:
            continue
        ring_support = int(
            cv2.countNonZero(cv2.bitwise_and(ring, binary))
        ) / float(ring_px)
        if ring_support < BACKGROUND_HOLE_MIN_RING_SUPPORT:
            continue

        pixels = lab[component > 0]
        if pixels.size == 0:
            continue
        lightness_std = float(np.std(pixels[:, 0]))
        chroma_std = max(
            float(np.std(pixels[:, 1])),
            float(np.std(pixels[:, 2])),
        )
        if (
            lightness_std > BACKGROUND_HOLE_MAX_LIGHTNESS_STD
            or chroma_std > BACKGROUND_HOLE_MAX_CHROMA_STD
        ):
            continue

        refined[component > 0] = 0
        stats["background_hole_pixels_removed"] += area
        stats["background_hole_count"] += 1

    if cv2.countNonZero(refined) == 0:
        return binary, stats
    stats["refined_jewel_area_px"] = int(cv2.countNonZero(refined))
    return refined, stats


def fill_component_holes(component_mask: np.ndarray, max_growth_ratio: float = 2.20) -> np.ndarray:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return component_mask
    filled = np.zeros_like(component_mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    original_area = max(1, int(cv2.countNonZero(component_mask)))
    filled_area = int(cv2.countNonZero(filled))
    if filled_area > int(round(original_area * max_growth_ratio)):
        return component_mask
    return filled


def crop_to_nonzero_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, Rect | None]:
    points = cv2.findNonZero(mask)
    if points is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(points)
    return (
        image_bgr[y:y + h, x:x + w].copy(),
        mask[y:y + h, x:x + w].copy(),
        (int(x), int(y), int(w), int(h)),
    )


def clamp_overlap_ratio(overlap_ratio: float) -> float:
    return max(0.0, min(0.49, float(overlap_ratio)))


def build_axis_starts(length: int, window: int, step: int) -> list[int]:
    if window >= length:
        return [0]

    starts: list[int] = []
    position = 0
    while position + window < length:
        starts.append(position)
        position += step

    last_start = max(0, length - window)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def fallback_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> list[list[int]]:
    overlap_h = int(round(slice_height * clamp_overlap_ratio(overlap_height_ratio)))
    overlap_w = int(round(slice_width * clamp_overlap_ratio(overlap_width_ratio)))
    step_y = max(1, slice_height - overlap_h)
    step_x = max(1, slice_width - overlap_w)

    boxes: list[list[int]] = []
    for y0 in build_axis_starts(image_height, slice_height, step_y):
        for x0 in build_axis_starts(image_width, slice_width, step_x):
            x1 = min(image_width, x0 + slice_width)
            y1 = min(image_height, y0 + slice_height)
            boxes.append([int(x0), int(y0), int(x1), int(y1)])
    return boxes


def generate_sahi_slice_bboxes(
    image_shape: tuple[int, ...],
    slice_size: int,
    overlap_ratio: float,
) -> list[Rect]:
    image_h, image_w = image_shape[:2]
    if image_h <= 0 or image_w <= 0:
        return []

    slice_size = max(64, int(slice_size))
    slice_h = min(image_h, slice_size)
    slice_w = min(image_w, slice_size)
    overlap_ratio = clamp_overlap_ratio(overlap_ratio)
    if slice_h >= image_h and slice_w >= image_w:
        return []

    if sahi_get_slice_bboxes is not None:
        raw_boxes = sahi_get_slice_bboxes(
            image_height=image_h,
            image_width=image_w,
            slice_height=slice_h,
            slice_width=slice_w,
            auto_slice_resolution=False,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )
    else:
        raw_boxes = fallback_slice_bboxes(
            image_height=image_h,
            image_width=image_w,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

    unique_boxes: list[Rect] = []
    seen: set[Rect] = set()
    for x0, y0, x1, y1 in raw_boxes:
        normalized = (
            max(0, min(image_w, int(x0))),
            max(0, min(image_h, int(y0))),
            max(0, min(image_w, int(x1))) - max(0, min(image_w, int(x0))),
            max(0, min(image_h, int(y1))) - max(0, min(image_h, int(y0))),
        )
        if normalized[2] <= 0 or normalized[3] <= 0:
            continue
        if normalized == (0, 0, image_w, image_h):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_boxes.append(normalized)
    return unique_boxes


def point_in_bbox(point_x: int, point_y: int, bbox: list[int] | tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    return x <= point_x <= (x + w) and y <= point_y <= (y + h)


def otsu_clean_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Gradual-threshold Otsu for robust large-jewel extraction.

    A small threshold offset (+15) below the Otsu-derived value ensures
    thin gold / reflective edge pixels that fall near the boundary are
    still captured.  No erosion preserves fine detail.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Offset matches dimension_calib pattern:  Otsu + small delta
    adjusted = min(255, thresh_val + 20)
    _, otsu = cv2.threshold(gray, adjusted, 255, cv2.THRESH_BINARY_INV)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, KERNEL_ELLIPSE_3, iterations=3)
    return otsu


def otsu_clean_mask_for_earring(image_bgr: np.ndarray) -> np.ndarray:
    """Earring-specific Otsu tuned to preserve tiny objects and ignore glare holes."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, KERNEL_ELLIPSE_3, iterations=2)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, KERNEL_ELLIPSE_3, iterations=1)
    return otsu


def contour_mask_with_holes(
    shape: tuple[int, int],
    contours: list[np.ndarray],
    hierarchy: np.ndarray,
    contour_index: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contours[contour_index]], -1, 255, -1)

    child = hierarchy[contour_index][2]
    while child != -1:
        cv2.drawContours(mask, [contours[child]], -1, 0, -1)
        child = hierarchy[child][0]

    return mask


def contour_mask_preserve_small_holes(
    shape: tuple[int, int],
    contours: list[np.ndarray],
    hierarchy: np.ndarray,
    contour_index: int,
    small_hole_area: float,
) -> np.ndarray:
    """Mirror otsusave.py behavior: keep small inner glare holes filled."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contours[contour_index]], -1, 255, -1)

    child = hierarchy[contour_index][2]
    while child != -1:
        child_area = cv2.contourArea(contours[child])
        if child_area > small_hole_area:
            cv2.drawContours(mask, [contours[child]], -1, 0, -1)
        child = hierarchy[child][0]

    return mask


def crop_to_mask(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    padding: int = 8,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int, int, int] | None]:
    points = cv2.findNonZero(mask)
    if points is None:
        return None, None, None

    x, y, w, h = cv2.boundingRect(points)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(bgr_image.shape[1], x + w + padding)
    y1 = min(bgr_image.shape[0], y + h + padding)
    return (
        bgr_image[y0:y1, x0:x1].copy(),
        mask[y0:y1, x0:x1].copy(),
        (x0, y0, x1 - x0, y1 - y0),
    )


def zoom_pair(image_bgr: np.ndarray, mask: np.ndarray, zoom_scale: int) -> tuple[np.ndarray, np.ndarray]:
    if zoom_scale <= 1:
        return image_bgr.copy(), mask.copy()

    zoomed_image = cv2.resize(
        image_bgr,
        None,
        fx=zoom_scale,
        fy=zoom_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    zoomed_mask = cv2.resize(
        mask,
        None,
        fx=zoom_scale,
        fy=zoom_scale,
        interpolation=cv2.INTER_NEAREST,
    )
    zoomed_mask = np.where(zoomed_mask > 0, 255, 0).astype(np.uint8)
    return zoomed_image, zoomed_mask


def normalize_kernel_size(size: int, minimum: int = 1) -> int:
    normalized = max(minimum, int(size))
    if normalized % 2 == 0:
        normalized += 1
    return normalized


def is_probable_jewelry(hsv_image: np.ndarray, mask: np.ndarray) -> bool:
    pixels = mask > 0
    if not np.any(pixels):
        return False

    hue = hsv_image[:, :, 0][pixels]
    sat = hsv_image[:, :, 1][pixels]
    val = hsv_image[:, :, 2][pixels]

    mean_val = float(val.mean())
    colored_fraction = float(((sat >= 35) & (val >= 50)).mean())
    dark_fraction = float((val <= 80).mean())
    gold_like_fraction = float(
        ((hue >= 14) & (hue <= 40) & (sat >= 55) & (val >= 40)).mean()
    )
    return (
        mean_val >= 80.0
        and colored_fraction >= 0.18
        and dark_fraction <= 0.55
        and gold_like_fraction >= 0.10
    )


def is_probable_small_jewelry(hsv_image: np.ndarray, mask: np.ndarray) -> bool:
    """Relaxed fallback for tiny earrings/jhumkas that fail the main jewelry gate."""
    pixels = mask > 0
    if not np.any(pixels):
        return False

    hue = hsv_image[:, :, 0][pixels]
    sat = hsv_image[:, :, 1][pixels]
    val = hsv_image[:, :, 2][pixels]

    mean_sat = float(sat.mean())
    mean_val = float(val.mean())
    colored_fraction = float(((sat >= 20) & (val >= 35)).mean())
    dark_fraction = float((val <= 60).mean())
    gold_like_fraction = float(
        ((hue >= 10) & (hue <= 45) & (sat >= 25) & (val >= 25)).mean()
    )
    return (
        gold_like_fraction >= 0.02
        or (colored_fraction >= 0.06 and mean_sat >= 22.0 and mean_val >= 35.0 and dark_fraction <= 0.85)
    )


def append_candidate_from_mask(
    image_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    candidates: list[dict],
) -> bool:
    points = cv2.findNonZero(candidate_mask)
    if points is None:
        return False

    x, y, w, h = cv2.boundingRect(points)
    local_bgr = image_bgr[y:y + h, x:x + w].copy()
    local_mask = candidate_mask[y:y + h, x:x + w].copy()
    cropped_bgr, cropped_mask, cropped_bbox = crop_to_mask(local_bgr, local_mask, padding=8)
    if cropped_bgr is None or cropped_mask is None or cropped_bbox is None:
        return False

    cx, cy, cw, ch = cropped_bbox
    candidates.append(
        {
            "bbox_global": (x + cx, y + cy, cw, ch),
            "crop_bgr": cropped_bgr,
            "crop_mask": cropped_mask,
            "area_px": int(cv2.countNonZero(cropped_mask)),
        }
    )
    return True


def build_candidates_from_component_mask(
    image_bgr: np.ndarray,
    component_mask: np.ndarray,
    min_area: int = 40,
    max_candidates: int | None = None,
    reject_border_touching: bool = True,
    min_area_ratio_to_largest: float = 0.0,
) -> list[dict]:
    candidate_mask = (component_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    candidates: list[dict] = []
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    relative_min_area = int(largest_area * max(0.0, float(min_area_ratio_to_largest)))
    effective_min_area = max(int(min_area), relative_min_area)

    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < effective_min_area:
            continue

        mask = np.where(labels == idx, 255, 0).astype(np.uint8)
        if reject_border_touching and mask_touches_image_border(mask, margin=2):
            continue
        if not reject_border_touching and mask_touches_image_border(mask, margin=2):
            width = int(stats[idx, cv2.CC_STAT_WIDTH])
            height = int(stats[idx, cv2.CC_STAT_HEIGHT])
            aspect_ratio = max(width, height) / max(1, min(width, height))
            if aspect_ratio > 8.0:
                continue
        append_candidate_from_mask(image_bgr, mask, candidates)

    candidates.sort(key=lambda item: (-item["area_px"], item["bbox_global"][1], item["bbox_global"][0]))
    if max_candidates is not None:
        candidates = candidates[:max(1, int(max_candidates))]
    candidates.sort(key=lambda item: (item["bbox_global"][1], item["bbox_global"][0]))
    return candidates


def mask_touches_image_border(mask: np.ndarray, margin: int = 2) -> bool:
    if mask.size == 0:
        return False
    margin = max(1, int(margin))
    top = mask[:margin, :]
    bottom = mask[-margin:, :]
    left = mask[:, :margin]
    right = mask[:, -margin:]
    return any(np.any(region > 0) for region in (top, bottom, left, right))


def extract_jewel_candidates(
    image_bgr: np.ndarray,
    min_area: int = 120,
    extraction_mode: str = "default",
    external_mask: np.ndarray | None = None,
) -> list[dict]:
    is_earring_mode = extraction_mode == "earring"
    if external_mask is not None:
        otsu = external_mask.astype(np.uint8)
        if otsu.ndim == 3:
            otsu = otsu.squeeze()
        if otsu.shape[:2] != image_bgr.shape[:2]:
            otsu = cv2.resize(otsu, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        otsu = (otsu > 0).astype(np.uint8) * 255
    else:
        otsu = otsu_clean_mask_for_earring(image_bgr) if is_earring_mode else otsu_clean_mask(image_bgr)
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return []

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hierarchy = hierarchy[0]
    candidates: list[dict] = []
    small_hole_area = max(24.0, float((image_bgr.shape[0] * image_bgr.shape[1]) // 12000))
    effective_min_area = max(60, int(min_area * 0.6)) if is_earring_mode else min_area

    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue

        area = cv2.contourArea(contour)
        if area < effective_min_area:
            continue

        if is_earring_mode:
            full_mask = contour_mask_preserve_small_holes(
                otsu.shape,
                contours,
                hierarchy,
                index,
                small_hole_area=small_hole_area,
            )
        else:
            full_mask = contour_mask_with_holes(otsu.shape, contours, hierarchy, index)

        is_valid = is_probable_small_jewelry(hsv_image, full_mask) if is_earring_mode else is_probable_jewelry(hsv_image, full_mask)
        if not is_valid:
            continue

        append_candidate_from_mask(image_bgr, full_mask, candidates)

    if not candidates and is_earring_mode:
        fallback_masks: list[tuple[float, np.ndarray]] = []
        for index, contour in enumerate(contours):
            if hierarchy[index][3] != -1:
                continue

            area = cv2.contourArea(contour)
            if area < effective_min_area:
                continue

            full_mask = contour_mask_preserve_small_holes(
                otsu.shape,
                contours,
                hierarchy,
                index,
                small_hole_area=small_hole_area,
            )
            if mask_touches_image_border(full_mask, margin=3):
                continue

            points = cv2.findNonZero(full_mask)
            if points is None:
                continue
            _, _, w, h = cv2.boundingRect(points)
            bbox_area = max(1, w * h)
            fill_ratio = float(cv2.countNonZero(full_mask)) / float(bbox_area)
            if fill_ratio < 0.08:
                continue

            fallback_masks.append((area, full_mask))

        fallback_masks.sort(key=lambda item: item[0], reverse=True)
        for _, full_mask in fallback_masks[:6]:
            append_candidate_from_mask(image_bgr, full_mask, candidates)

    if not candidates and not is_earring_mode:
        relaxed_min_area = max(40, int(min_area * 0.4))
        for index, contour in enumerate(contours):
            if hierarchy[index][3] != -1:
                continue

            area = cv2.contourArea(contour)
            if area < relaxed_min_area:
                continue

            full_mask = contour_mask_with_holes(otsu.shape, contours, hierarchy, index)
            if not is_probable_small_jewelry(hsv_image, full_mask):
                continue

            append_candidate_from_mask(image_bgr, full_mask, candidates)

    candidates.sort(key=lambda item: (item["bbox_global"][1], item["bbox_global"][0]))
    if candidates:
        largest_area = max(item["area_px"] for item in candidates)
        if largest_area >= 4000:
            min_keep_area = max(250, int(round(largest_area * 0.03)))
            candidates = [item for item in candidates if item["area_px"] >= min_keep_area]
    return candidates


def build_gold_mask(hsv_image: np.ndarray, jewel_mask: np.ndarray, strict: bool = False) -> np.ndarray:
    if strict:
        gold_mask = cv2.inRange(hsv_image, GOLD_RANGE_STRICT[0], GOLD_RANGE_STRICT[1])
    else:
        gold_mask = cv2.bitwise_or(
            cv2.inRange(hsv_image, GOLD_RANGE_BROAD_A[0], GOLD_RANGE_BROAD_A[1]),
            cv2.inRange(hsv_image, GOLD_RANGE_BROAD_B[0], GOLD_RANGE_BROAD_B[1]),
        )
    gold_mask = cv2.bitwise_and(gold_mask, jewel_mask)
    gold_mask = cv2.morphologyEx(
        gold_mask,
        cv2.MORPH_CLOSE,
        KERNEL_ELLIPSE_3,
        iterations=1,
    )
    return cv2.bitwise_and(gold_mask, jewel_mask)


def gold_support_stats(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
) -> tuple[int, float]:
    mask_px = int(cv2.countNonZero(jewel_mask))
    if mask_px <= 0:
        return 0, 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gold_mask = build_gold_mask(hsv, jewel_mask, strict=False)
    gold_px = int(cv2.countNonZero(gold_mask))
    return gold_px, gold_px / float(mask_px)


def has_enough_gold_for_stone_detection(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    min_gold_px: int = MIN_GOLD_PIXELS_FOR_STONE_DETECTION,
    min_gold_ratio: float = MIN_GOLD_RATIO_FOR_STONE_DETECTION,
) -> bool:
    gold_px, gold_ratio = gold_support_stats(image_bgr, jewel_mask)
    return gold_px >= int(min_gold_px) and gold_ratio >= float(min_gold_ratio)


def build_specular_glare_mask_advanced(
    roi_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    threshold: int = DEFAULT_GLARE_THRESHOLD,
    saturation_max: int = DEFAULT_GLARE_SATURATION_MAX,
    patch_size: int = DEFAULT_GLARE_PATCH_SIZE,
    gray_image: np.ndarray | None = None,
    hsv_image: np.ndarray | None = None,
) -> np.ndarray:
    """
    Advanced glare detection using:
    1. Dynamic saturation thresholds (mean - 2σ)
    2. Local contrast analysis for specular detection
    3. Conservative masking to preserve genuine stones
    """
    if cv2.countNonZero(jewel_mask) == 0:
        return np.zeros_like(jewel_mask)

    gray = gray_image if gray_image is not None else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = hsv_image if hsv_image is not None else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Step 1: High-value highlight detection
    threshold = max(0, min(255, int(threshold)))
    highlight_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    highlight_mask = cv2.bitwise_and(highlight_mask, jewel_mask)
    
    if cv2.countNonZero(highlight_mask) == 0:
        return np.zeros_like(jewel_mask)

    # Step 2: Dynamic saturation gating using mean - 2σ approach
    # Only high-value pixels with very low saturation are true glare
    sat_values = hsv[highlight_mask > 0, 1]
    if sat_values.size > 10:
        sat_mean = float(np.mean(sat_values))
        sat_std = float(np.std(sat_values))
        dynamic_sat_threshold = max(
            GLARE_DYNAMIC_SAT_MIN,
            int(sat_mean - GLARE_SAT_SIGMA_THRESHOLD * sat_std),
        )
    else:
        dynamic_sat_threshold = saturation_max

    # Two-condition saturation guard: S very low AND V extremely high
    low_sat_mask = np.where(hsv[:, :, 1] <= dynamic_sat_threshold, 255, 0).astype(np.uint8)
    very_high_value_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    
    glare_mask = cv2.bitwise_and(highlight_mask, low_sat_mask)
    glare_mask = cv2.bitwise_and(glare_mask, very_high_value_mask)

    # Step 3: Local contrast enhancement (specular region detection)
    # Glare typically has sharp edges and high local contrast
    if patch_size > 0:
        blur_kernel = normalize_kernel_size(max(3, patch_size // 2), minimum=3)
        blurred = cv2.blur(gray, (blur_kernel, blur_kernel))
        local_contrast = cv2.absdiff(gray, blurred).astype(np.float32)
        contrast_mean = float(np.mean(local_contrast[jewel_mask > 0]))
        contrast_threshold = contrast_mean * GLARE_LOCAL_CONTRAST_FACTOR
        high_contrast_mask = np.where(local_contrast >= contrast_threshold, 255, 0).astype(np.uint8)
        glare_mask = cv2.bitwise_and(glare_mask, high_contrast_mask)

    # Step 4: Gold context filtering (only glare touching gold is relevant)
    gold_context = build_gold_mask(hsv, jewel_mask, strict=False)
    if cv2.countNonZero(gold_context) > 0:
        context_kernel_size = normalize_kernel_size(max(3, patch_size // 3), minimum=3)
        context_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_kernel_size, context_kernel_size),
        )
        gold_context = cv2.dilate(gold_context, context_kernel, iterations=1)
        glare_mask = cv2.bitwise_and(glare_mask, gold_context)

    # Step 5: Conservative morphology (open for small noise, close for continuity)
    glare_mask = cv2.morphologyEx(
        glare_mask,
        cv2.MORPH_OPEN,
        KERNEL_ELLIPSE_3,
        iterations=1,
    )
    glare_mask = cv2.morphologyEx(
        glare_mask,
        cv2.MORPH_CLOSE,
        KERNEL_ELLIPSE_3,  # Use smaller kernel for conservative closing
        iterations=1,
    )
    
    return cv2.bitwise_and(glare_mask, jewel_mask)


def build_specular_glare_mask(
    roi_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    threshold: int = DEFAULT_GLARE_THRESHOLD,
    saturation_max: int = DEFAULT_GLARE_SATURATION_MAX,
    patch_size: int = DEFAULT_GLARE_PATCH_SIZE,
    gray_image: np.ndarray | None = None,
    hsv_image: np.ndarray | None = None,
) -> np.ndarray:
    """
    Wrapper that uses advanced glare detection by default.
    Falls back to simple approach if advanced method fails.
    """
    try:
        return build_specular_glare_mask_advanced(
            roi_bgr, jewel_mask, threshold, saturation_max, patch_size, gray_image, hsv_image
        )
    except Exception:
        # Fallback to original simple method
        if cv2.countNonZero(jewel_mask) == 0:
            return np.zeros_like(jewel_mask)

        gray = gray_image if gray_image is not None else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        hsv = hsv_image if hsv_image is not None else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        threshold = max(0, min(255, int(threshold)))
        saturation_max = max(0, min(255, int(saturation_max)))
        highlight_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        low_sat_mask = np.where(hsv[:, :, 1] <= saturation_max, 255, 0).astype(np.uint8)
        glare_mask = cv2.bitwise_and(highlight_mask, low_sat_mask)
        glare_mask = cv2.bitwise_and(glare_mask, jewel_mask)

        gold_context = build_gold_mask(hsv, jewel_mask, strict=False)
        if cv2.countNonZero(gold_context) > 0:
            context_kernel_size = normalize_kernel_size(max(3, patch_size // 3), minimum=3)
            context_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (context_kernel_size, context_kernel_size),
            )
            gold_context = cv2.dilate(gold_context, context_kernel, iterations=1)
            glare_mask = cv2.bitwise_and(glare_mask, gold_context)

        glare_mask = cv2.morphologyEx(
            glare_mask,
            cv2.MORPH_OPEN,
            KERNEL_ELLIPSE_3,
            iterations=1,
        )
        glare_mask = cv2.morphologyEx(
            glare_mask,
            cv2.MORPH_CLOSE,
            KERNEL_ELLIPSE_5,
            iterations=1,
        )
        return cv2.bitwise_and(glare_mask, jewel_mask)


def patch_based_inpaint(
    image_bgr: np.ndarray,
    glare_mask: np.ndarray,
    jewel_mask: np.ndarray,
    patch_size: int = DEFAULT_GLARE_PATCH_SIZE,
) -> tuple[np.ndarray, int, np.ndarray]:
    if cv2.countNonZero(glare_mask) == 0:
        return image_bgr.copy(), 0, np.zeros_like(glare_mask)

    cleaned = image_bgr.copy()
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned, 0, np.zeros_like(glare_mask)

    patch_size = max(1, int(patch_size))
    dilate_kernel_size = normalize_kernel_size(patch_size, minimum=3)
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilate_kernel_size, dilate_kernel_size),
    )
    min_component_area = max(4, patch_size // 3)
    inpainted_regions = 0
    applied_mask = np.zeros_like(glare_mask)

    for contour in contours:
        contour_mask = np.zeros_like(glare_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        contour_mask = cv2.bitwise_and(contour_mask, jewel_mask)
        if cv2.countNonZero(contour_mask) < min_component_area:
            continue

        inpaint_mask = cv2.dilate(contour_mask, dilate_kernel, iterations=1)
        inpaint_mask = cv2.bitwise_and(inpaint_mask, jewel_mask)
        if cv2.countNonZero(inpaint_mask) == 0:
            continue

        cleaned = cv2.inpaint(
            cleaned,
            inpaint_mask,
            inpaintRadius=patch_size,
            flags=cv2.INPAINT_TELEA,
        )
        inpainted_regions += 1
        applied_mask = cv2.bitwise_or(applied_mask, inpaint_mask)

    return cleaned, inpainted_regions, applied_mask


def reflection_metrics(glare_mask: np.ndarray, jewel_mask: np.ndarray) -> dict:
    if (
        _STONE_AREA_CALC_AVAILABLE
        and hasattr(_stone_area_calc, "calculate_reflection_risk")
    ):
        try:
            return _stone_area_calc.calculate_reflection_risk(
                glare_mask,
                jewel_mask,
                coverage_risk_percent=REFLECTION_COVERAGE_FLAG_PERCENT,
                local_density_risk_percent=REFLECTION_LOCAL_DENSITY_FLAG_PERCENT,
            )
        except Exception:
            pass

    jewel_px = max(1, int(cv2.countNonZero(jewel_mask)))
    glare_px = int(cv2.countNonZero(glare_mask))
    if glare_px <= 0:
        return {
            "coverage_percent": 0.0,
            "local_density_percent": 0.0,
            "region_count": 0,
            "largest_region_px": 0,
            "largest_region_percent": 0.0,
            "flagged": False,
            "risk_status": "NORMAL",
            "possible_transparent_stones": False,
            "level": "normal",
            "message": "No dense reflection signature detected.",
        }

    binary_glare = (glare_mask > 0).astype(np.float32)
    binary_jewel = (jewel_mask > 0).astype(np.float32)
    height, width = glare_mask.shape[:2]
    window = max(9, min(41, int(round(min(height, width) * 0.12))))
    if window % 2 == 0:
        window += 1
    local_glare = cv2.boxFilter(binary_glare, -1, (window, window), normalize=False)
    local_jewel = cv2.boxFilter(binary_jewel, -1, (window, window), normalize=False)
    local_density = np.divide(
        local_glare,
        np.maximum(local_jewel, 1.0),
        out=np.zeros_like(local_glare),
        where=local_jewel > 0,
    )
    local_density_percent = float(local_density.max() * 100.0)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (glare_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 3
    ]
    coverage_percent = glare_px / float(jewel_px) * 100.0
    flagged = bool(
        coverage_percent >= REFLECTION_COVERAGE_FLAG_PERCENT
        or (
            coverage_percent >= 0.55
            and local_density_percent >= REFLECTION_LOCAL_DENSITY_FLAG_PERCENT
        )
    )
    if coverage_percent >= 3.0 or local_density_percent >= 32.0:
        level = "high"
    elif flagged:
        level = "elevated"
    else:
        level = "normal"
    message = (
        "Dense reflection detected; possible additional transparent/colorless gemstones may be present. "
        "Treat this jewel as RISK."
        if flagged
        else "Reflection is present but not dense enough to classify the jewel as RISK."
    )
    return {
        "coverage_percent": round(coverage_percent, 2),
        "local_density_percent": round(local_density_percent, 2),
        "region_count": len(component_areas),
        "largest_region_px": max(component_areas, default=0),
        "largest_region_percent": round(
            max(component_areas, default=0) / float(jewel_px) * 100.0,
            2,
        ),
        "flagged": flagged,
        "risk_status": "RISK" if flagged else "NORMAL",
        "possible_transparent_stones": flagged,
        "level": level,
        "message": message,
    }


def remove_specular_glare(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    enabled: bool = DEFAULT_GLARE_REMOVAL_ENABLED,
    threshold: int = DEFAULT_GLARE_THRESHOLD,
    patch_size: int = DEFAULT_GLARE_PATCH_SIZE,
    saturation_max: int = DEFAULT_GLARE_SATURATION_MAX,
) -> tuple[np.ndarray, np.ndarray, dict]:
    stats = {
        "enabled": bool(enabled),
        "threshold": int(threshold),
        "patch_size": int(max(1, patch_size)),
        "saturation_max": int(saturation_max),
        "glare_mask_px": 0,
        "glare_region_count": 0,
        "glare_inpaint_px": 0,
        "reflection": reflection_metrics(np.zeros_like(roi_mask), roi_mask),
    }
    if not enabled or cv2.countNonZero(roi_mask) == 0:
        return roi_bgr.copy(), np.zeros_like(roi_mask), stats

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = build_specular_glare_mask(
        roi_bgr,
        roi_mask,
        threshold=threshold,
        saturation_max=saturation_max,
        patch_size=patch_size,
        gray_image=gray,
        hsv_image=hsv,
    )
    stats["glare_mask_px"] = int(cv2.countNonZero(glare_mask))
    stats["reflection"] = reflection_metrics(glare_mask, roi_mask)
    if stats["glare_mask_px"] == 0:
        return roi_bgr.copy(), np.zeros_like(roi_mask), stats

    cleaned_bgr, glare_region_count, inpainted_mask = patch_based_inpaint(
        roi_bgr,
        glare_mask,
        roi_mask,
        patch_size=patch_size,
    )
    stats["glare_region_count"] = int(glare_region_count)
    stats["glare_inpaint_px"] = int(cv2.countNonZero(inpainted_mask))
    # Combine the original glare mask (before inpainting) with the inpainted area
    # so that all glare-affected pixels are excluded from white stone detection.
    # This allows milder inpainting (smaller patch_size) while still preventing
    # glare-on-gold from masquerading as false white stones.
    white_exclude = inpainted_mask
    if cv2.countNonZero(glare_mask) > 0:
        if white_exclude is not None:
            white_exclude = cv2.bitwise_or(white_exclude, glare_mask)
        else:
            white_exclude = glare_mask.copy()
    return cleaned_bgr, white_exclude, stats


def mask_from_ranges(hsv_image: np.ndarray, ranges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
    return mask


def build_raw_color_masks(
    hsv_image: np.ndarray,
    jewel_mask: np.ndarray,
    white_exclude_mask: np.ndarray | None = None,
    lab_image: np.ndarray | None = None,
    analysis_normalization: dict | None = None,
) -> dict[str, np.ndarray]:
    sat = hsv_image[:, :, 1]
    val = hsv_image[:, :, 2]
    masks: dict[str, np.ndarray] = {}
    normalization = normalize_analysis_normalization(analysis_normalization)
    color_boost = normalization["color_boost"]
    boosted_hsv = hsv_image.copy()
    boosted_hsv[:, :, 1] = np.clip(
        boosted_hsv[:, :, 1].astype(np.float32) * color_boost,
        0,
        255,
    ).astype(np.uint8)

    for color_name, ranges in HSV_COLOR_RANGE_ARRAYS.items():
        color_mask = mask_from_ranges(boosted_hsv, ranges)
        masks[color_name] = cv2.bitwise_and(color_mask, jewel_mask)

    red_hue_extension = int(round(4.0 * (color_boost - 1.0)))
    red_sat_min = int(round(max(60.0, 150.0 / color_boost)))
    red_extra = np.where(
        (
            (
                hsv_image[:, :, 0] <= min(16, 8 + red_hue_extension)
            )
            | (
                hsv_image[:, :, 0] >= max(162, 170 - red_hue_extension)
            )
        )
        & (hsv_image[:, :, 1] >= red_sat_min)
        & (hsv_image[:, :, 2] >= 35),
        255,
        0,
    ).astype(np.uint8)
    masks["Red"] = cv2.bitwise_or(
        masks["Red"],
        cv2.bitwise_and(red_extra, jewel_mask),
    )

    if lab_image is None:
        lab_image = cv2.cvtColor(
            cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR),
            cv2.COLOR_BGR2LAB,
        )
    lab_l = lab_image[:, :, 0].astype(np.float32)
    lab_a = lab_image[:, :, 1].astype(np.float32) - 128.0
    lab_b = lab_image[:, :, 2].astype(np.float32) - 128.0

    # Pink/red must have real positive LAB a* chroma and limited yellow b*.
    # This rejects weak red fringes created by gold reflections and JPEG edges.
    pink_chroma = (
        (lab_a >= 8.0)
        & (lab_b <= 20.0)
        & (lab_l >= 65.0)
        & (val < 252)
    )
    red_chroma = (
        (lab_a >= (12.0 / color_boost))
        & (lab_b <= (28.0 + (6.0 * (color_boost - 1.0))))
        & (lab_l >= 45.0)
        & (val < 252)
    )
    masks["Pink"] = cv2.bitwise_and(
        masks["Pink"],
        (pink_chroma.astype(np.uint8) * 255),
    )
    masks["Red"] = cv2.bitwise_and(
        masks["Red"],
        (red_chroma.astype(np.uint8) * 255),
    )

    # Selective green recovery works in LAB and only inside the jewel mask.
    # Increasing the option relaxes the negative-a* threshold; it does not
    # alter pink/red/gold pixels or globally increase image saturation.
    green_gain = normalization["green_recovery"]
    dark_gain = normalization["dark_green_recovery"]
    green_a_limit = -(7.5 / green_gain)
    dark_green_a_limit = -(5.5 / dark_gain)
    green_lab = (
        (lab_a <= green_a_limit)
        & (lab_b >= -28.0)
        & (lab_b <= 58.0)
        & (lab_l >= 58.0)
        & (lab_l <= 235.0)
    )
    dark_green_lab = (
        (lab_a <= dark_green_a_limit)
        & (lab_b >= -25.0)
        & (lab_b <= 48.0)
        & (lab_l >= 24.0)
        & (lab_l < 105.0 + (18.0 * (dark_gain - 1.0)))
    )
    recovered_green = ((green_lab | dark_green_lab).astype(np.uint8) * 255)
    masks["Green"] = cv2.bitwise_or(masks["Green"], recovered_green)
    masks["Green"] = cv2.bitwise_and(masks["Green"], jewel_mask)

    white_mask = np.where((sat <= 55) & (val >= 185), 255, 0).astype(np.uint8)
    if white_exclude_mask is not None:
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(white_exclude_mask))
    black_mask = np.where(val <= 60, 255, 0).astype(np.uint8)
    masks["White/Colorless"] = cv2.bitwise_and(white_mask, jewel_mask)
    masks["Black"] = cv2.bitwise_and(black_mask, jewel_mask)
    return masks


def filter_color_masks(
    raw_masks: dict[str, np.ndarray],
    broad_gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for color_name, mask in raw_masks.items():
        if color_name in {"Orange", "Yellow/Gold"}:
            filtered[color_name] = cv2.bitwise_and(mask, cv2.bitwise_not(strict_gold_mask))
        else:
            filtered[color_name] = cv2.bitwise_and(mask, cv2.bitwise_not(broad_gold_mask))
    return filtered


def cleanup_color_mask(
    mask: np.ndarray,
    color_name: str,
    color_boost: float = 1.0,
) -> np.ndarray:
    if color_name == "White/Colorless":
        # Preserve tiny stone faces. A 2x2 opening can erase a genuine
        # 2-4 pixel stone completely at the native camera scale.
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            KERNEL_ELLIPSE_3,
            iterations=1,
        )
    open_kernel = KERNEL_ELLIPSE_2 if color_boost >= 1.35 else KERNEL_ELLIPSE_3
    return cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1),
        cv2.MORPH_CLOSE,
        KERNEL_ELLIPSE_3,
        iterations=1,
    )


def circular_hue_mean(hues: np.ndarray) -> float:
    if hues.size == 0:
        return 0.0
    angles = hues.astype(np.float32) * (2.0 * np.pi / 180.0)
    x = float(np.cos(angles).mean())
    y = float(np.sin(angles).mean())
    if x == 0.0 and y == 0.0:
        return float(hues.mean())
    angle = np.arctan2(y, x)
    if angle < 0.0:
        angle += 2.0 * np.pi
    return float(angle * 180.0 / (2.0 * np.pi))


def mean_hsv_for_mask(hsv_image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    pixels = mask > 0
    hsv_pixels = hsv_image[pixels]
    if hsv_pixels.size == 0:
        return 0.0, 0.0, 0.0
    return (
        circular_hue_mean(hsv_pixels[:, 0]),
        float(hsv_pixels[:, 1].mean()),
        float(hsv_pixels[:, 2].mean()),
    )


def mean_lab_ab_for_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    lab_image: np.ndarray | None = None,
) -> tuple[float, float, float]:
    pixels = mask > 0
    if not np.any(pixels):
        return 0.0, 0.0, 0.0
    lab = lab_image if lab_image is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab_pixels = lab[pixels].astype(np.float32)
    lab_a = lab_pixels[:, 1] - 128.0
    lab_b = lab_pixels[:, 2] - 128.0
    chroma = np.sqrt((lab_a * lab_a) + (lab_b * lab_b))
    return (
        float(lab_a.mean()),
        float(lab_b.mean()),
        float(chroma.mean()),
    )


def fallback_color_from_mean_hsv(mean_h: float, mean_s: float, mean_v: float) -> str:
    if mean_v < 50:
        return "Black"
    if mean_s < 30 and mean_v >= 170:
        return "White/Colorless"
    if mean_h < 8 or mean_h >= 170:
        if mean_h >= 175 and mean_s <= 190:
            return "Pink"
        if mean_v >= 145 and mean_s < 190:
            return "Pink"
        return "Red"
    if mean_h < 23:
        return "Orange"
    if mean_h < 40:
        return "Yellow/Gold"
    if mean_h < 85:
        return "Green"
    if mean_h < 136:
        return "Blue"
    if mean_h < 160:
        return "Purple/Violet"
    return "Pink"


def contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def _components_touching_seed(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray:
    """Keep candidate components connected to or immediately touching a seed."""
    binary = (candidate_mask > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(candidate_mask)
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(candidate_mask)
    seed_contact = cv2.dilate(
        (seed_mask > 0).astype(np.uint8),
        KERNEL_ELLIPSE_3,
        iterations=1,
    )
    touching_labels = np.unique(labels[seed_contact > 0])
    touching_labels = touching_labels[touching_labels > 0]
    if touching_labels.size == 0:
        return np.zeros_like(candidate_mask)
    return np.where(
        np.isin(labels, touching_labels),
        255,
        0,
    ).astype(np.uint8)


def _stone_growth_candidate_is_valid(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    gold_barrier_mask: np.ndarray,
    maximum_area: int,
    max_gold_overlap_share: float = 0.0,
) -> bool:
    seed_area = max(1, int(cv2.countNonZero(seed_mask)))
    area = int(cv2.countNonZero(candidate_mask))
    if area < seed_area:
        return False
    if area > maximum_area:
        return False

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0:
        return False
    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = max(width, height) / max(1, min(width, height))
    extent = area / float(max(1, width * height))
    circularity = contour_circularity(contour)
    if aspect_ratio > 3.8 or extent < 0.18 or circularity < 0.10:
        return False

    gold_overlap = int(
        cv2.countNonZero(cv2.bitwise_and(candidate_mask, gold_barrier_mask))
    )
    if gold_overlap <= STONE_REGION_GROW_MAX_GOLD_PIXELS:
        return True
    if max_gold_overlap_share <= 0.0:
        return False
    return (gold_overlap / float(max(1, area))) <= float(max_gold_overlap_share)


def build_reliable_gold_barrier(
    image_bgr: np.ndarray,
    broad_gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
    jewel_mask: np.ndarray,
    include_reflective_metal: bool = False,
) -> np.ndarray:
    """Build a hard metal barrier without treating warm gray stones as gold."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    lab_b = lab[:, :, 2] - 128.0

    reliable_broad_gold = (
        (broad_gold_mask > 0)
        & (value >= 55.0)
        & (
            (saturation >= 55.0)
            | ((saturation >= 38.0) & (lab_b >= 9.0))
        )
    )
    barrier = np.where(
        (strict_gold_mask > 0) | reliable_broad_gold,
        255,
        0,
    ).astype(np.uint8)
    if include_reflective_metal:
        gold_context = cv2.dilate(
            (broad_gold_mask > 0).astype(np.uint8) * 255,
            KERNEL_ELLIPSE_5,
            iterations=1,
        )
        reflective_metal = (
            (gold_context > 0)
            & (value >= 150.0)
            & (saturation <= 72.0)
        )
        barrier[reflective_metal] = 255
    return cv2.bitwise_and(barrier, jewel_mask)


def expand_stone_seed_to_full_region(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
    jewel_mask: np.ndarray,
    broad_gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
    color_name: str,
    jewel_area: int,
    background_match_mask: np.ndarray | None = None,
    hsv_image: np.ndarray | None = None,
    lab_image: np.ndarray | None = None,
    reliable_gold_barrier: np.ndarray | None = None,
    reflective_gold_barrier: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow a strict HSV seed to the complete local stone face.

    Growth is illumination tolerant but spatially conservative:
    - the search is restricted to a small dilation around the seed;
    - reliable broad/strict gold and calibrated background are hard barriers;
    - only components connected to the seed survive;
    - compactness, size, and gold-overlap checks prevent chain leakage.
    """
    seed = np.where(seed_mask > 0, 255, 0).astype(np.uint8)
    seed_area = int(cv2.countNonZero(seed))
    diagnostics: dict[str, Any] = {
        "enabled": bool(STONE_REGION_GROW_ENABLED),
        "seed_area_px": seed_area,
        "expanded_area_px": seed_area,
        "area_gain": 1.0,
        "method": "seed_only",
    }
    if (
        not STONE_REGION_GROW_ENABLED
        or seed_area <= 0
        or color_name in {"White/Colorless", "Yellow/Gold", "Orange"}
    ):
        return seed, diagnostics

    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return seed, diagnostics
    seed_contour = max(contours, key=cv2.contourArea)
    _x, _y, seed_width, seed_height = cv2.boundingRect(seed_contour)
    seed_circularity = contour_circularity(seed_contour)
    seed_aspect_ratio = max(seed_width, seed_height) / max(
        1,
        min(seed_width, seed_height),
    )
    if color_name != "Black" and (
        seed_area < STONE_REGION_GROW_COLOR_MIN_SEED_AREA_PX
        or seed_circularity < 0.22
        or seed_aspect_ratio > 2.8
    ):
        diagnostics["method"] = "seed_only_unqualified_color_seed"
        return seed, diagnostics

    equivalent_radius = math.sqrt(seed_area / math.pi)
    if color_name == "Black":
        growth_radius = int(
            round(
                max(
                    STONE_REGION_GROW_MIN_RADIUS_PX,
                    max(seed_width, seed_height) * 1.35,
                    equivalent_radius * 2.3,
                )
            )
        )
        growth_radius = min(STONE_REGION_GROW_MAX_RADIUS_PX, growth_radius)
        maximum_area = min(
            max(
                seed_area + 8,
                int(
                    round(
                        seed_area
                        * STONE_REGION_GROW_BLACK_MAX_SEED_MULTIPLIER
                    )
                ),
            ),
            max(
                seed_area + 8,
                int(
                    round(
                        jewel_area
                        * STONE_REGION_GROW_BLACK_MAX_JEWEL_FRACTION
                    )
                ),
            ),
        )
    else:
        growth_radius = int(
            round(
                max(
                    STONE_REGION_GROW_MIN_RADIUS_PX,
                    max(seed_width, seed_height) * 0.45,
                    equivalent_radius * 0.85,
                )
            )
        )
        growth_radius = min(
            STONE_REGION_GROW_COLOR_MAX_RADIUS_PX,
            growth_radius,
        )
        maximum_area = min(
            max(
                seed_area + 8,
                int(
                    round(
                        seed_area
                        * STONE_REGION_GROW_COLOR_MAX_SEED_MULTIPLIER
                    )
                ),
            ),
            max(
                seed_area + 8,
                int(
                    round(
                        jewel_area
                        * STONE_REGION_GROW_COLOR_MAX_JEWEL_FRACTION
                    )
                ),
            ),
        )

    relaxed_colored_growth = color_name in COLOR_STONE_REGION_GROW_COLORS
    if color_name != "Black" and reflective_gold_barrier is not None:
        reliable_barrier = reflective_gold_barrier
    elif reliable_gold_barrier is not None:
        reliable_barrier = reliable_gold_barrier
    else:
        reliable_barrier = build_reliable_gold_barrier(
            image_bgr,
            broad_gold_mask,
            strict_gold_mask,
            jewel_mask,
            include_reflective_metal=color_name != "Black",
        )
    if relaxed_colored_growth:
        # Colored gemstones often contain yellow/white specular facets that
        # fall inside the broad gold HSV range. Treat only strict gold as a
        # hard growth barrier, then validate the final compact component
        # against the broader metal mask below.
        gold_barrier = cv2.bitwise_and(strict_gold_mask, jewel_mask)
        gold_validation_barrier = reliable_barrier
        max_gold_overlap_share = STONE_REGION_GROW_COLOR_MAX_GOLD_OVERLAP_SHARE
    else:
        gold_barrier = reliable_barrier
        gold_validation_barrier = reliable_barrier
        max_gold_overlap_share = 0.0

    seed = cv2.bitwise_and(seed, cv2.bitwise_not(gold_barrier))
    seed_area = int(cv2.countNonZero(seed))
    if seed_area <= 0:
        diagnostics["method"] = "rejected_gold_seed"
        diagnostics["expanded_area_px"] = 0
        diagnostics["area_gain"] = 0.0
        return seed, diagnostics
    seed_pixels = seed > 0

    search_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * growth_radius + 1, 2 * growth_radius + 1),
    )
    search_mask = cv2.dilate(seed, search_kernel, iterations=1)
    search_mask = cv2.bitwise_and(search_mask, jewel_mask)
    search_mask = cv2.bitwise_and(
        search_mask,
        cv2.bitwise_not(gold_barrier),
    )
    maximum_area = min(
        maximum_area,
        max(seed_area + 8, int(cv2.countNonZero(search_mask))),
    )

    hsv = hsv_image if hsv_image is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = (
        lab_image.astype(np.float32, copy=False)
        if lab_image is not None
        else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    )
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    lab_a = lab[:, :, 1] - 128.0
    lab_b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(lab_a * lab_a + lab_b * lab_b)

    background = (
        (background_match_mask > 0)
        if background_match_mask is not None
        else np.zeros(seed.shape, dtype=bool)
    )
    seed_value = value[seed_pixels]
    seed_chroma = chroma[seed_pixels]
    local_sample_mask = (
        (search_mask > 0)
        & (gold_barrier == 0)
        & (~background)
    )
    local_values = value[local_sample_mask]
    local_chroma = chroma[local_sample_mask]
    if local_values.size == 0:
        return seed, diagnostics

    if color_name == "Black":
        seed_value_high = float(np.percentile(seed_value, 90))
        seed_chroma_high = float(np.percentile(seed_chroma, 90))
        value_limit = min(
            200.0,
            max(
                105.0,
                seed_value_high + 105.0,
                float(np.percentile(local_values, 72)),
            ),
        )
        chroma_limit = min(
            90.0,
            max(
                32.0,
                seed_chroma_high + 35.0,
                float(np.percentile(local_chroma, 78)),
            ),
        )
        very_dark = value <= min(145.0, seed_value_high + 62.0)
        appearance_match = (
            (value <= value_limit)
            & ((chroma <= chroma_limit) | very_dark)
        )
        diagnostics["black_value_limit"] = round(value_limit, 2)
        diagnostics["black_chroma_limit"] = round(chroma_limit, 2)
    else:
        seed_hue = circular_hue_mean(hsv[:, :, 0][seed_pixels])
        seed_sat = saturation[seed_pixels]
        seed_lab_center = np.median(lab[seed_pixels], axis=0)
        hue_spread = float(
            np.percentile(
                hue_distance(hsv[:, :, 0][seed_pixels], seed_hue),
                90,
            )
        )
        hue_limit = min(24.0, max(9.0, hue_spread + 8.0))
        chroma_distance = np.linalg.norm(
            lab[:, :, 1:3] - seed_lab_center[1:3].reshape(1, 1, 2),
            axis=2,
        )
        saturation_floor = max(
            10.0,
            float(np.percentile(seed_sat, 15)) * 0.08,
        )
        value_ceiling = min(
            238.0,
            float(np.percentile(seed_value, 90)) + 110.0,
        )
        neutral_facet = (
            (saturation <= 58.0)
            & (chroma <= 48.0)
            & (value <= value_ceiling)
        )
        appearance_match = (
            (
                hue_distance(hsv[:, :, 0], seed_hue) <= hue_limit
            )
            & (saturation >= saturation_floor)
            & (value <= value_ceiling)
        ) | (
            (chroma_distance <= 48.0)
            & (value <= value_ceiling)
        ) | neutral_facet

    initial_candidate = np.where(
        (search_mask > 0)
        & appearance_match
        & (gold_barrier == 0)
        & (~background),
        255,
        0,
    ).astype(np.uint8)
    initial_candidate = cv2.bitwise_or(initial_candidate, seed)
    initial_candidate = cv2.morphologyEx(
        initial_candidate,
        cv2.MORPH_CLOSE,
        KERNEL_ELLIPSE_5,
        iterations=1,
    )
    initial_candidate = cv2.bitwise_and(
        initial_candidate,
        cv2.bitwise_not(gold_barrier),
    )
    threshold_region = _components_touching_seed(initial_candidate, seed)
    threshold_region = fill_component_holes(
        threshold_region,
        max_growth_ratio=1.75,
    )
    threshold_region = cv2.bitwise_and(
        threshold_region,
        cv2.bitwise_not(gold_barrier),
    )

    candidates: list[tuple[str, np.ndarray]] = [("adaptive_region_grow", threshold_region)]

    # GrabCut uses the adaptive region as probable foreground and can recover
    # highlighted facets inside the same local edge boundary.
    if (
        STONE_REGION_GROW_GRABCUT_ITERATIONS > 0
        and cv2.countNonZero(threshold_region) > seed_area
    ):
        search_points = cv2.findNonZero(search_mask)
        if search_points is None:
            return seed, diagnostics
        grab_x, grab_y, grab_w, grab_h = cv2.boundingRect(search_points)
        grab_x1 = max(0, grab_x - 2)
        grab_y1 = max(0, grab_y - 2)
        grab_x2 = min(seed.shape[1], grab_x + grab_w + 2)
        grab_y2 = min(seed.shape[0], grab_y + grab_h + 2)
        grab_slice = np.s_[grab_y1:grab_y2, grab_x1:grab_x2]
        local_search = search_mask[grab_slice]
        local_threshold = threshold_region[grab_slice]
        local_seed = seed[grab_slice]
        local_gold_barrier = gold_barrier[grab_slice]
        local_background = background[grab_slice]
        grab_mask = np.full(local_seed.shape, cv2.GC_BGD, dtype=np.uint8)
        search_pixels = local_search > 0
        grab_mask[search_pixels] = cv2.GC_PR_BGD
        grab_mask[local_threshold > 0] = cv2.GC_PR_FGD
        grab_mask[local_seed > 0] = cv2.GC_FGD
        grab_mask[(local_gold_barrier > 0) | local_background] = cv2.GC_BGD
        try:
            background_model = np.zeros((1, 65), np.float64)
            foreground_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                image_bgr[grab_slice],
                grab_mask,
                None,
                background_model,
                foreground_model,
                STONE_REGION_GROW_GRABCUT_ITERATIONS,
                cv2.GC_INIT_WITH_MASK,
            )
            local_grab_region = np.where(
                (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            grab_region = np.zeros_like(seed)
            grab_region[grab_slice] = local_grab_region
            grab_region = cv2.bitwise_and(grab_region, search_mask)
            grab_region = cv2.bitwise_and(
                grab_region,
                cv2.bitwise_not(gold_barrier),
            )
            grab_region = _components_touching_seed(grab_region, seed)
            grab_region = fill_component_holes(
                grab_region,
                max_growth_ratio=1.75,
            )
            grab_region = cv2.bitwise_and(
                grab_region,
                cv2.bitwise_not(gold_barrier),
            )
            candidates.append(("edge_aware_grabcut", grab_region))
        except cv2.error:
            pass

    accepted: list[tuple[str, np.ndarray, int]] = []
    for method, candidate in candidates:
        area = int(cv2.countNonZero(candidate))
        if area < int(round(seed_area * STONE_REGION_GROW_MIN_AREA_GAIN)):
            continue
        if not _stone_growth_candidate_is_valid(
            candidate,
            seed,
            gold_validation_barrier,
            maximum_area,
            max_gold_overlap_share=max_gold_overlap_share,
        ):
            continue
        accepted.append((method, candidate, area))

    if not accepted:
        return seed, diagnostics
    method, expanded, expanded_area = max(accepted, key=lambda item: item[2])
    expanded = cv2.bitwise_and(expanded, cv2.bitwise_not(gold_barrier))
    expanded_area = int(cv2.countNonZero(expanded))
    diagnostics.update(
        {
            "expanded_area_px": expanded_area,
            "area_gain": round(expanded_area / float(seed_area), 3),
            "method": method,
            "growth_radius_px": growth_radius,
            "maximum_area_px": maximum_area,
            "gold_overlap_px": int(
                cv2.countNonZero(cv2.bitwise_and(expanded, gold_validation_barrier))
            ),
            "gold_overlap_share": round(
                cv2.countNonZero(cv2.bitwise_and(expanded, gold_validation_barrier))
                / float(max(1, expanded_area)),
                3,
            ),
        }
    )
    return expanded, diagnostics


def extract_component_masks(candidate_mask: np.ndarray, min_area: int, max_area: int) -> list[tuple[np.ndarray, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    components: list[tuple[np.ndarray, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        component = np.zeros_like(candidate_mask)
        component_view = component[y:y + h, x:x + w]
        labels_view = labels[y:y + h, x:x + w]
        component_view[labels_view == label] = 255
        components.append((component, area))
    return components


def extract_region_candidates(
    candidate_masks: dict[str, np.ndarray],
    jewel_area: int,
    min_area_floor: int = 18,
    white_min_area_floor: int = 10,
    black_min_area_floor: int = 18,
    color_boost: float = 1.0,
) -> list[dict]:
    region_candidates: list[dict] = []
    boost = max(1.0, min(2.5, float(color_boost)))
    for color_name, mask in candidate_masks.items():
        cleaned = cleanup_color_mask(mask, color_name, color_boost=boost)
        colorful_floor = max(
            MICRO_STONE_MIN_AREA_PX,
            int(round(min_area_floor / (boost * boost))),
        )
        proportional_floor = int(
            round(jewel_area * 0.00006 / (boost * boost))
        )
        min_area = max(
            MICRO_STONE_MIN_AREA_PX,
            min(colorful_floor, max(MICRO_STONE_MIN_AREA_PX, proportional_floor)),
        )
        if color_name == "White/Colorless":
            min_area = max(
                MICRO_STONE_MIN_AREA_PX,
                min(
                    max(MICRO_STONE_MIN_AREA_PX, int(white_min_area_floor)),
                    max(
                        MICRO_STONE_MIN_AREA_PX,
                        int(round(jewel_area * 0.00004)),
                    ),
                ),
            )
        elif color_name == "Black":
            min_area = max(
                MICRO_STONE_MIN_AREA_PX,
                min(
                    max(MICRO_STONE_MIN_AREA_PX, int(black_min_area_floor)),
                    max(
                        MICRO_STONE_MIN_AREA_PX,
                        int(round(jewel_area * 0.00004)),
                    ),
                ),
            )
        max_area_ratio = 0.38
        if color_name in {"Orange", "Yellow/Gold"}:
            max_area_ratio = 0.18
        elif color_name in {"White/Colorless", "Black"}:
            max_area_ratio = 0.12
        max_area = max(min_area + 1, int(round(jewel_area * max_area_ratio)))

        for component, area_px in extract_component_masks(cleaned, min_area=min_area, max_area=max_area):
            region_candidates.append(
                {
                    "base_color": color_name,
                    "mask": component,
                    "area_px": int(area_px),
                }
            )

    region_candidates.sort(key=lambda item: (-item["area_px"], item["base_color"]))
    return region_candidates


def is_stone_adjacent_to_metal(
    stone_mask: np.ndarray,
    gold_mask: np.ndarray,
    proximity_px: int = METAL_PROXIMITY_PIXELS,
) -> bool:
    """
    Check if stone blob is adjacent to gold/metal setting.
    Real jewelry stones always sit inside or adjacent to gold/silver.
    This rule alone eliminates most background false positives.
    """
    if cv2.countNonZero(stone_mask) == 0 or cv2.countNonZero(gold_mask) == 0:
        return False
    
    # Dilate stone boundary to check proximity
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (proximity_px, proximity_px))
    stone_expanded = cv2.dilate(stone_mask, kernel, iterations=1)
    
    # Check overlap between expanded stone and gold
    overlap = cv2.bitwise_and(stone_expanded, gold_mask)
    metal_neighbor_px = int(cv2.countNonZero(overlap))
    
    # Must have at least some gold pixel contact after dilation
    return metal_neighbor_px > 0


def is_stone_shape_valid(
    contour: np.ndarray,
    stone_mask: np.ndarray,
    color: str,
) -> bool:
    """
    Validate stone shape using circularity.
    Real stones are compact (circularity > 0.5).
    Edge artifacts and strips are elongated (circularity < 0.2).
    """
    if cv2.countNonZero(stone_mask) == 0:
        return False
    
    circularity = contour_circularity(contour)
    
    # For white/colorless stones, be more strict about shape
    if color == "White/Colorless":
        area = int(cv2.countNonZero(stone_mask))
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        extent = area / float(max(1, width * height))
        if area <= MICRO_STONE_MAX_AREA_PX:
            return bool(
                circularity >= 0.16
                and aspect_ratio <= 3.0
                and extent >= 0.20
            )
        # Real diamonds/white stones: 0.5 < circularity < 0.95
        # Edge artifacts: circularity < 0.2 or > 0.98 (near-perfect circles are noise)
        if circularity < STONE_CIRCULARITY_MIN:
            return False  # Too elongated, likely edge strip
        if circularity > STONE_CIRCULARITY_MAX:
            return False  # Too circular, likely noise blob
        return True
    
    # For other colors, standard check (already in classify_component)
    return circularity >= 0.06


def touches_tile_boundary(
    region_bbox: tuple[int, int, int, int],
    tile_bbox: tuple[int, int, int, int] | None = None,
    dead_zone_px: int = SAHI_TILE_BORDER_DEAD_ZONE,
) -> bool:
    """
    Check if region bounding box touches SAHI tile boundary dead zone.
    After slicing, mark 5-10px border around each tile as "dead zone."
    Any detection touching this border gets flagged (for potential rejection).
    
    Args:
        region_bbox: (x, y, w, h) of region
        tile_bbox: (x, y, w, h) of SAHI tile, None if no tile context
        dead_zone_px: Dead zone width at boundaries
    
    Returns:
        True if region touches boundary dead zone
    """
    if tile_bbox is None:
        return False
    
    x, y, w, h = region_bbox
    tile_x, tile_y, tile_w, tile_h = tile_bbox
    
    # Dead zones at tile boundaries
    left_boundary = tile_x + dead_zone_px
    top_boundary = tile_y + dead_zone_px
    right_boundary = tile_x + tile_w - dead_zone_px
    bottom_boundary = tile_y + tile_h - dead_zone_px
    
    # Check if bounding box touches any boundary
    touches_left = x <= left_boundary and tile_x == 0  # At image edge
    touches_top = y <= top_boundary and tile_y == 0    # At image edge
    touches_right = (x + w) >= right_boundary          # Within dead zone
    touches_bottom = (y + h) >= bottom_boundary        # Within dead zone
    
    return touches_left or touches_top or touches_right or touches_bottom


def regions_overlap_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Calculate Intersection over Union (IoU) between two region masks.
    Used to detect duplicate detections across SAHI tile boundaries.
    """
    if cv2.countNonZero(mask_a) == 0 or cv2.countNonZero(mask_b) == 0:
        return 0.0
    
    intersection = cv2.bitwise_and(mask_a, mask_b)
    union = cv2.bitwise_or(mask_a, mask_b)
    
    intersection_px = int(cv2.countNonZero(intersection))
    union_px = int(cv2.countNonZero(union))
    
    if union_px == 0:
        return 0.0
    
    return float(intersection_px) / float(union_px)


def classify_component(
    region_id: int,
    component_mask: np.ndarray,
    raw_color_masks: dict[str, np.ndarray],
    gold_mask: np.ndarray,
    strict_gold_mask: np.ndarray,
    jewel_mask: np.ndarray,
    image_bgr: np.ndarray,
    hsv_image: np.ndarray,
    lab_image: np.ndarray,
    jewel_area: int,
    background_match_mask: np.ndarray | None = None,
    learned_profile_masks: dict[str, np.ndarray] | None = None,
    learned_profiles_by_id: dict[str, dict] | None = None,
    reliable_gold_barrier: np.ndarray | None = None,
    reflective_gold_barrier: np.ndarray | None = None,
    allowed_colors: set[str] | None = None,
) -> dict | None:
    area_px = int(cv2.countNonZero(component_mask))
    if area_px == 0:
        return None

    overlaps = {
        color_name: int(cv2.countNonZero(cv2.bitwise_and(component_mask, color_mask)))
        for color_name, color_mask in raw_color_masks.items()
    }
    breakdown = [(name, pixels) for name, pixels in overlaps.items() if pixels > 0]
    breakdown.sort(key=lambda item: (-item[1], item[0]))

    mean_h, mean_s, mean_v = mean_hsv_for_mask(hsv_image, component_mask)
    fallback_color = fallback_color_from_mean_hsv(mean_h, mean_s, mean_v)

    dominant_color = breakdown[0][0] if breakdown else fallback_color
    dominant_share = (breakdown[0][1] / area_px) if breakdown else 0.0
    if dominant_share < 0.45:
        dominant_color = fallback_color

    multicolor_drivers = [
        (name, pixels)
        for name, pixels in breakdown
        if name not in {"White/Colorless", "Black", "Yellow/Gold"}
    ]
    if len(multicolor_drivers) >= 2:
        first_share = multicolor_drivers[0][1] / area_px
        second_share = multicolor_drivers[1][1] / area_px
        if first_share >= 0.18 and second_share >= 0.18 and (first_share + second_share) >= 0.60:
            dominant_color = "Multicolor/Color-changing"

    if allowed_colors is not None and dominant_color not in allowed_colors:
        return None

    gold_overlap = int(cv2.countNonZero(cv2.bitwise_and(component_mask, gold_mask)))
    gold_share = gold_overlap / area_px
    if dominant_color in {"Orange", "Yellow/Gold"} and gold_share >= 0.55:
        return None
    if dominant_color == "Red" and gold_share >= 0.45:
        return None
    if area_px > int(jewel_area * 0.72):
        return None

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    circularity = contour_circularity(contour)
    if circularity < 0.06:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    extent = area_px / max(1, w * h)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    if extent < 0.14:
        return None
    if aspect_ratio > 6.0 and extent < 0.35:
        return None
    if dominant_color in {"White/Colorless", "Black"} and circularity < 0.14:
        return None

    # Geometric filtering: Shape circularity validation
    # Real stones are compact; edge artifacts are elongated or perfectly circular
    if dominant_color == "White/Colorless":
        if not is_stone_shape_valid(contour, component_mask, dominant_color):
            return None
        
        # Metal proximity constraint: White stones must be adjacent to gold setting
        # This rule eliminates most background false positives
        if not is_stone_adjacent_to_metal(component_mask, gold_mask, METAL_PROXIMITY_PIXELS):
            return None

    seed_area_px = area_px
    classification_seed_mask = component_mask.copy()
    component_mask, expansion = expand_stone_seed_to_full_region(
        image_bgr=image_bgr,
        seed_mask=component_mask,
        jewel_mask=jewel_mask,
        broad_gold_mask=gold_mask,
        strict_gold_mask=strict_gold_mask,
        color_name=dominant_color,
        jewel_area=jewel_area,
        background_match_mask=background_match_mask,
        hsv_image=hsv_image,
        lab_image=lab_image,
        reliable_gold_barrier=reliable_gold_barrier,
        reflective_gold_barrier=reflective_gold_barrier,
    )
    if dominant_color in COLOR_STONE_REGION_GROW_COLORS:
        # The grown colored-stone mask has already been compactness checked
        # against the broad metal mask. Do not erase warm/low-saturation
        # stone facets here just because they resemble reflective gold.
        selected_gold_barrier = cv2.bitwise_and(strict_gold_mask, jewel_mask)
    else:
        if dominant_color in {"Black", "White/Colorless"}:
            selected_gold_barrier = reliable_gold_barrier
        else:
            selected_gold_barrier = reflective_gold_barrier
        if selected_gold_barrier is None:
            selected_gold_barrier = build_reliable_gold_barrier(
                image_bgr,
                gold_mask,
                strict_gold_mask,
                jewel_mask,
                # Bright low-saturation pixels are the stone itself for white gems;
                # never subtract them as reflective metal.
                include_reflective_metal=dominant_color not in {
                    "Black",
                    "White/Colorless",
                },
            )
    component_mask = fill_component_holes(component_mask)
    component_mask = cv2.bitwise_and(
        component_mask,
        cv2.bitwise_not(selected_gold_barrier),
    )
    area_px = int(cv2.countNonZero(component_mask))
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    circularity = contour_circularity(contour)
    x, y, w, h = cv2.boundingRect(contour)
    extent = area_px / max(1, w * h)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    mean_h, mean_s, mean_v = mean_hsv_for_mask(hsv_image, component_mask)
    mean_lab_a, mean_lab_b, mean_lab_chroma = mean_lab_ab_for_mask(
        image_bgr,
        component_mask,
        lab_image=lab_image,
    )
    overlaps = {
        color_name: int(
            cv2.countNonZero(cv2.bitwise_and(component_mask, color_mask))
        )
        for color_name, color_mask in raw_color_masks.items()
    }
    breakdown = [(name, pixels) for name, pixels in overlaps.items() if pixels > 0]
    breakdown.sort(key=lambda item: (-item[1], item[0]))
    final_gold_overlap = int(
        cv2.countNonZero(
            cv2.bitwise_and(component_mask, selected_gold_barrier)
        )
    )
    final_gold_share = final_gold_overlap / max(1, area_px)

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x = x + w // 2
        center_y = y + h // 2

    color_mix = {
        name: round(pixels / max(1, area_px) * 100.0, 1)
        for name, pixels in breakdown
        if (pixels / max(1, area_px)) >= 0.08
    }
    if dominant_color not in color_mix:
        color_mix[dominant_color] = round(dominant_share * 100.0, 1)

    learned_matches: list[dict] = []
    for profile_id, profile_mask in (learned_profile_masks or {}).items():
        overlap_px = int(
            cv2.countNonZero(cv2.bitwise_and(component_mask, profile_mask))
        )
        overlap_share = overlap_px / max(1, area_px)
        seed_overlap_px = int(
            cv2.countNonZero(
                cv2.bitwise_and(classification_seed_mask, profile_mask)
            )
        )
        seed_overlap_share = seed_overlap_px / max(1, seed_area_px)
        if (
            max(overlap_px, seed_overlap_px) < 4
            or max(overlap_share, seed_overlap_share) < 0.60
        ):
            continue
        profile = (learned_profiles_by_id or {}).get(profile_id) or {}
        learned_matches.append(
            {
                "profile_id": profile_id,
                "label": str(profile.get("label") or "Learned sample"),
                "color": str(profile.get("color") or dominant_color),
                "hsv_center": list(profile.get("hsv_center") or []),
                "overlap_percent": round(
                    max(overlap_share, seed_overlap_share) * 100.0,
                    1,
                ),
            }
        )
    learned_matches.sort(
        key=lambda item: (-item["overlap_percent"], item["label"])
    )
    if learned_matches and learned_matches[0]["color"] in GEMSTONE_OPTIONS:
        dominant_color = learned_matches[0]["color"]

    classification_seed_mask = cv2.bitwise_and(
        classification_seed_mask,
        component_mask,
    )

    return {
        "region_id": region_id,
        "color": dominant_color,
        "possible_gemstones": GEMSTONE_OPTIONS[dominant_color],
        "area_px": area_px,
        "seed_area_px": seed_area_px,
        "stone_region_expansion": expansion,
        "bbox": [int(x), int(y), int(w), int(h)],
        "center": [int(center_x), int(center_y)],
        "mean_hsv": [round(mean_h, 1), round(mean_s, 1), round(mean_v, 1)],
        "mean_lab_ab": [round(mean_lab_a, 1), round(mean_lab_b, 1)],
        "lab_chroma_mean": round(mean_lab_chroma, 1),
        "gold_overlap_percent": round(final_gold_share * 100.0, 1),
        "color_mix_percent": color_mix,
        "dominant_share_percent": round(dominant_share * 100.0, 1),
        "extent": round(extent, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 3),
        "learned_matches": learned_matches,
        "seed_mask": classification_seed_mask,
        "mask": component_mask,
        "contour": contour,
    }


def detect_sparse_gold_regions(
    gold_mask: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """
    Detect if we're in a sparse/fragmented gold region where background edge
    contamination is likely when glare removal is weak.
    
    Returns: (is_sparse, gold_density_map)
    - is_sparse: True if gold is sparse and scattered
    - gold_density_map: Local density of gold pixels (dilated gold mask)
    """
    gold_px = int(cv2.countNonZero(gold_mask))
    is_sparse = gold_px < SPARSE_GOLD_REGION_THRESHOLD
    
    # Build density map: where gold is densely packed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    gold_density = cv2.dilate(gold_mask, kernel, iterations=1)
    
    return is_sparse, gold_density


def is_white_stone_in_lab(
    white_mask: np.ndarray,
    roi_bgr: np.ndarray,
) -> bool:
    """
    Validate if a white stone candidate is genuinely white/colorless using LAB color space.
    
    LAB is more robust to lighting variation than HSV:
    - L channel: brightness (independent of white background)
    - A channel: red/green axis (-127=green, +127=red)
    - B channel: yellow/blue axis (-127=blue, +127=yellow)
    
    Real white diamonds have:
    - High L (brightness)
    - Neutral A, B (near 0, 0)
    - Distinct signature from pure white background
    """
    white_px = int(cv2.countNonZero(white_mask))
    if white_px < MICRO_STONE_MIN_AREA_PX:
        return False
    
    try:
        # Convert to LAB
        lab_image = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        
        # Extract LAB values at white stone pixels
        l_values = lab_image[:, :, 0][white_mask > 0]
        a_values = lab_image[:, :, 1][white_mask > 0]
        b_values = lab_image[:, :, 2][white_mask > 0]
        
        if l_values.size == 0:
            return False
        
        # Compute statistics
        l_mean = float(np.mean(l_values))
        a_mean = float(np.mean(a_values)) - 128.0
        b_mean = float(np.mean(b_values)) - 128.0
        
        # Check brightness threshold for white/colorless
        lightness_min = (
            105.0
            if white_px <= MICRO_STONE_MAX_AREA_PX
            else float(LAB_WHITE_L_MIN)
        )
        if l_mean < lightness_min:
            return False
        
        # Check chromaticity (A-B deviation from neutral gray)
        # White diamonds have very low chromaticity
        lab_saturation = np.sqrt((a_values.astype(np.float32) - 128) ** 2 + 
                                 (b_values.astype(np.float32) - 128) ** 2)
        sat_mean = float(np.mean(lab_saturation))
        
        saturation_max = (
            14.0
            if white_px <= MICRO_STONE_MAX_AREA_PX
            else float(LAB_SATURATION_MAX)
        )
        if sat_mean > saturation_max:
            return False
        
        # Check A and B channels are in neutral range (even pure white has tiny offset)
        if white_px <= MICRO_STONE_MAX_AREA_PX:
            a_in_range = -10.0 <= a_mean <= 12.0
            b_in_range = -12.0 <= b_mean <= 15.0
        else:
            a_in_range = LAB_WHITE_A_MIN <= a_mean <= LAB_WHITE_A_MAX
            b_in_range = LAB_WHITE_B_MIN <= b_mean <= LAB_WHITE_B_MAX
        
        if not (a_in_range and b_in_range):
            return False
        
        return True
        
    except Exception:
        # If LAB conversion fails, accept the stone (don't reject it)
        return True


def is_white_stone_edge_artifact(
    white_mask: np.ndarray,
    gold_mask: np.ndarray,
    roi_mask: np.ndarray,
    is_sparse_gold: bool,
    glare_threshold: int = DEFAULT_GLARE_THRESHOLD,
    roi_bgr: np.ndarray | None = None,
) -> bool:
    """
    Detect if a white stone candidate is likely just a background edge artifact
    caused by weak glare removal in sparse gold regions.
    
    Returns True if artifact (should be ignored), False if likely real stone.
    """
    white_px = int(cv2.countNonZero(white_mask))
    if white_px == 0:
        return True
    
    # In sparse gold regions with weak glare removal, be very strict about white stones
    weak_glare_mode = glare_threshold < WEAK_GLARE_THRESHOLD
    
    if is_sparse_gold and weak_glare_mode:
        # For sparse regions with weak glare: require minimum stone size
        if white_px < SPARSE_GOLD_WHITE_STONE_MIN_SIZE:
            return True
        
        # Check if white stone is directly adjacent to gold (real stone) vs edge artifact
        # Expand white stone by 1 pixel to check adjacency
        kernel_adj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_dilated = cv2.dilate(white_mask, kernel_adj, iterations=1)
        
        # Find pixels that are dilated white but NOT original white (the boundary)
        white_boundary = cv2.subtract(white_dilated, white_mask)
        
        # Check how many boundary pixels are adjacent to gold
        gold_neighbor_px = int(cv2.countNonZero(cv2.bitwise_and(white_boundary, gold_mask)))
        boundary_px = int(cv2.countNonZero(white_boundary))
        
        if boundary_px > 0:
            gold_neighbor_share = gold_neighbor_px / boundary_px
            # If white stone has poor connectivity to gold, it's likely background artifact
            if gold_neighbor_share < SPARSE_GOLD_WHITE_GOLD_NEIGHBOR_MIN:
                return True
        
        # Additional edge filtering: white stones at ROI edges in sparse mode are suspicious
        roi_border = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        roi_border = cv2.subtract(roi_border, roi_mask)
        
        border_overlap = int(cv2.countNonZero(cv2.bitwise_and(white_mask, roi_border)))
        if border_overlap > 0:
            border_share = border_overlap / white_px
            if border_share >= SPARSE_GOLD_WHITE_BORDER_SHARE_MAX:
                return True
    
    return False


def detect_regions_in_roi(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    area_reference: int | None = None,
    min_area_floor: int = 18,
    white_min_area_floor: int = 10,
    black_min_area_floor: int = 18,
    source_label: str = "full_roi",
    white_exclude_mask: np.ndarray | None = None,
    glare_threshold: int = DEFAULT_GLARE_THRESHOLD,
    background_calibration: dict | None = None,
    analysis_normalization: dict | None = None,
    learned_stone_profiles: list[dict] | None = None,
    allowed_colors: set[str] | None = None,
) -> list[dict]:
    if cv2.countNonZero(roi_mask) == 0:
        return []

    # With a calibrated background, preserve the full jewel mask and suppress
    # matching background only inside a narrow inner boundary band. Without a
    # calibration, retain a one-pixel erosion as a conservative fallback.
    if background_calibration:
        analysis_roi_mask = roi_mask.copy()
    else:
        analysis_roi_mask = cv2.erode(roi_mask, KERNEL_ELLIPSE_3, iterations=1)
        if cv2.countNonZero(analysis_roi_mask) == 0:
            analysis_roi_mask = roi_mask.copy()

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    gold_mask = build_gold_mask(hsv, analysis_roi_mask, strict=False)
    strict_gold_mask = build_gold_mask(hsv, analysis_roi_mask, strict=True)
    if cv2.countNonZero(gold_mask) < MIN_GOLD_PIXELS_FOR_STONE_DETECTION:
        return []

    boundary_band, roi_distance = build_inner_boundary_band(
        analysis_roi_mask,
        BACKGROUND_BOUNDARY_WIDTH_PX,
    )
    background_match_mask = build_background_match_mask(roi_bgr, background_calibration)
    background_boundary_mask = cv2.bitwise_and(background_match_mask, boundary_band)
    combined_white_exclude = background_boundary_mask.copy()
    if white_exclude_mask is not None:
        combined_white_exclude = cv2.bitwise_or(combined_white_exclude, white_exclude_mask)

    raw_color_masks = build_raw_color_masks(
        hsv,
        analysis_roi_mask,
        white_exclude_mask=combined_white_exclude,
        lab_image=lab,
        analysis_normalization=analysis_normalization,
    )
    normalization = normalize_analysis_normalization(analysis_normalization)
    learned_profile_masks, learned_profiles_by_id = build_learned_profile_masks(
        hsv,
        lab,
        analysis_roi_mask,
        learned_stone_profiles,
        color_boost=normalization["color_boost"],
    )
    candidate_color_masks = filter_color_masks(raw_color_masks, gold_mask, strict_gold_mask)
    filtered_learned_masks = {
        profile_id: (
            cv2.bitwise_and(profile_mask, cv2.bitwise_not(strict_gold_mask))
            if learned_profiles_by_id[profile_id]["color"] in {"Orange", "Yellow/Gold"}
            else cv2.bitwise_and(profile_mask, cv2.bitwise_not(gold_mask))
        )
        for profile_id, profile_mask in learned_profile_masks.items()
    }
    reliable_gold_barrier = build_reliable_gold_barrier(
        roi_bgr,
        gold_mask,
        strict_gold_mask,
        analysis_roi_mask,
        include_reflective_metal=False,
    )
    reflective_gold_barrier = build_reliable_gold_barrier(
        roi_bgr,
        gold_mask,
        strict_gold_mask,
        analysis_roi_mask,
        include_reflective_metal=True,
    )

    effective_area = max(1, int(area_reference or cv2.countNonZero(roi_mask)))
    region_candidates = extract_region_candidates(
        candidate_color_masks,
        jewel_area=effective_area,
        min_area_floor=min_area_floor,
        white_min_area_floor=white_min_area_floor,
        black_min_area_floor=black_min_area_floor,
        color_boost=normalization["color_boost"],
    )
    for profile_id, profile_mask in filtered_learned_masks.items():
        profile = learned_profiles_by_id[profile_id]
        learned_min_area = max(
            5,
            min(
                30,
                int(
                    round(
                        float(profile.get("sample_component_area_px", 0) or 0)
                        * 0.15
                    )
                ),
            ),
        )
        for component, area_px in extract_component_masks(
            profile_mask,
            min_area=learned_min_area,
            max_area=max(6, int(round(effective_area * 0.38))),
        ):
            region_candidates.append(
                {
                    "base_color": profile["color"],
                    "mask": component,
                    "area_px": int(area_px),
                    "learned_profile_id": profile_id,
                }
            )
    region_candidates.sort(
        key=lambda item: (
            0 if item.get("learned_profile_id") else 1,
            -item["area_px"],
            item["base_color"],
        )
    )

    regions: list[dict] = []
    occupied_mask = np.zeros_like(roi_mask)
    
    # Detect sparse gold regions for better edge artifact filtering
    is_sparse_gold, gold_density_map = detect_sparse_gold_regions(gold_mask, roi_mask)
    
    gold_context_mask = cv2.dilate(gold_mask, KERNEL_ELLIPSE_5, iterations=2)
    
    # Thin/small ROIs are more susceptible to false white stones from background edge
    roi_mask_px = max(1, int(cv2.countNonZero(roi_mask)))
    analysis_roi_px = max(1, int(cv2.countNonZero(analysis_roi_mask)))
    is_thin_roi = (analysis_roi_px / roi_mask_px) < TINY_ROI_EROSION_RATIO or effective_area < TINY_ROI_AREA_THRESHOLD
    for region_candidate in region_candidates:
        component_mask = region_candidate["mask"]
        area_px = max(1, int(cv2.countNonZero(component_mask)))
        overlap_px = int(cv2.countNonZero(cv2.bitwise_and(component_mask, occupied_mask)))
        overlap_share = overlap_px / area_px
        if overlap_share >= 0.55:
            continue
        if overlap_share > 0.12:
            component_mask = cv2.bitwise_and(component_mask, cv2.bitwise_not(occupied_mask))
            if cv2.countNonZero(component_mask) < min_area_floor:
                continue

        region = classify_component(
            region_id=len(regions) + 1,
            component_mask=component_mask,
            raw_color_masks=raw_color_masks,
            gold_mask=gold_mask,
            strict_gold_mask=strict_gold_mask,
            jewel_mask=analysis_roi_mask,
            image_bgr=roi_bgr,
            hsv_image=hsv,
            lab_image=lab,
            jewel_area=effective_area,
            background_match_mask=background_match_mask,
            learned_profile_masks=filtered_learned_masks,
            learned_profiles_by_id=learned_profiles_by_id,
            reliable_gold_barrier=reliable_gold_barrier,
            reflective_gold_barrier=reflective_gold_barrier,
            allowed_colors=allowed_colors,
        )
        if region is not None:
            learned_profile_id = region_candidate.get("learned_profile_id")
            if learned_profile_id:
                learned_match = next(
                    (
                        match
                        for match in region.get("learned_matches") or []
                        if match.get("profile_id") == learned_profile_id
                    ),
                    None,
                )
                if learned_match is None:
                    continue
                if (
                    float(region.get("circularity", 0.0)) < 0.16
                    or float(region.get("extent", 0.0)) < 0.22
                    or float(region.get("aspect_ratio", 99.0)) > 4.5
                ):
                    continue
            if region["color"] in {"White/Colorless", "Black"}:
                region_area = max(1, region["area_px"])
                nearby_gold_px = int(cv2.countNonZero(cv2.bitwise_and(region["mask"], gold_context_mask)))
                nearby_gold_share = nearby_gold_px / region_area
                if nearby_gold_share < 0.18:
                    continue

            # White-on-white background can leak into the segmented jewel only
            # along its outline. A calibrated background is therefore rejected
            # only in the inner boundary band; interior white stones remain valid.
            if region["color"] == "White/Colorless":
                if not is_white_stone_in_lab(region["mask"], roi_bgr):
                    continue

                if is_white_stone_edge_artifact(
                    region["mask"],
                    gold_mask,
                    roi_mask,
                    is_sparse_gold,
                    glare_threshold,
                    roi_bgr,
                ):
                    continue

                region_distances = roi_distance[region["mask"] > 0]
                max_depth = float(region_distances.max()) if region_distances.size else 0.0
                deep_share = float((region_distances >= 2.0).mean()) if region_distances.size else 0.0
                boundary_overlap = int(
                    cv2.countNonZero(cv2.bitwise_and(region["mask"], boundary_band))
                )
                boundary_share = boundary_overlap / max(1, region["area_px"])
                background_overlap = int(
                    cv2.countNonZero(cv2.bitwise_and(region["mask"], background_match_mask))
                )
                background_share = background_overlap / max(1, region["area_px"])
                region["boundary_share"] = round(boundary_share, 3)
                region["background_match_share"] = round(background_share, 3)
                region["max_boundary_depth_px"] = round(max_depth, 2)

                if background_calibration:
                    if (
                        background_share >= BACKGROUND_MATCH_REJECT_SHARE
                        and boundary_share >= 0.35
                        and max_depth <= BACKGROUND_BOUNDARY_WIDTH_PX + 1.5
                    ):
                        continue
                else:
                    min_depth = TINY_MAX_DEPTH_MIN if is_thin_roi else 2.20
                    min_deep_share = TINY_DEEP_SHARE_MIN if is_thin_roi else 0.06
                    if max_depth < min_depth:
                        continue
                    if boundary_share >= 0.55 and deep_share < min_deep_share:
                        continue
            elif region["color"] == "Black":
                boundary_overlap = int(cv2.countNonZero(cv2.bitwise_and(region["mask"], boundary_band)))
                if boundary_overlap > 0 and (boundary_overlap / max(1, region["area_px"])) >= 0.65:
                    continue
            region["source"] = source_label
            regions.append(region)
            occupied_mask = cv2.bitwise_or(occupied_mask, region["mask"])

    return regions


def region_touches_internal_slice_border(
    region: dict,
    slice_shape: tuple[int, ...],
    slice_bbox: Rect,
    full_shape: tuple[int, ...],
    margin: int = 2,
) -> bool:
    x, y, w, h = region["bbox"]
    slice_h, slice_w = slice_shape[:2]
    slice_x, slice_y, slice_width, slice_height = slice_bbox
    full_h, full_w = full_shape[:2]

    touches_left = x <= margin and slice_x > 0
    touches_top = y <= margin and slice_y > 0
    touches_right = (x + w) >= (slice_w - margin) and (slice_x + slice_width) < full_w
    touches_bottom = (y + h) >= (slice_h - margin) and (slice_y + slice_height) < full_h
    return touches_left or touches_top or touches_right or touches_bottom


def offset_region_to_full_roi(
    region: dict,
    slice_bbox: Rect,
    full_shape: tuple[int, ...],
) -> dict:
    slice_x, slice_y, _, _ = slice_bbox
    region_mask = region["mask"]
    full_mask = np.zeros(full_shape[:2], dtype=np.uint8)
    full_mask[slice_y:slice_y + region_mask.shape[0], slice_x:slice_x + region_mask.shape[1]] = region_mask

    contour = region["contour"].copy()
    contour[:, 0, 0] += slice_x
    contour[:, 0, 1] += slice_y

    offset_region = dict(region)
    offset_region["bbox"] = [
        int(region["bbox"][0] + slice_x),
        int(region["bbox"][1] + slice_y),
        int(region["bbox"][2]),
        int(region["bbox"][3]),
    ]
    offset_region["center"] = [
        int(region["center"][0] + slice_x),
        int(region["center"][1] + slice_y),
    ]
    if region.get("seed_mask") is not None:
        seed_mask = region["seed_mask"]
        full_seed_mask = np.zeros(full_shape[:2], dtype=np.uint8)
        full_seed_mask[
            slice_y:slice_y + seed_mask.shape[0],
            slice_x:slice_x + seed_mask.shape[1],
        ] = seed_mask
        offset_region["seed_mask"] = full_seed_mask
    offset_region["mask"] = full_mask
    offset_region["contour"] = contour
    offset_region["source"] = "sahi_slice"
    return offset_region


def region_quality_score(region: dict) -> float:
    dominant_share = float(region.get("dominant_share_percent", 0.0)) / 100.0
    circularity = min(1.0, float(region.get("circularity", 0.0)))
    extent = min(1.0, float(region.get("extent", 0.0)))
    area_bonus = min(0.12, float(region["area_px"]) / 2000.0 * 0.12)
    source_bonus = 0.04 if region.get("source") == "sahi_slice" else 0.0
    return dominant_share * 0.55 + circularity * 0.18 + extent * 0.11 + area_bonus + source_bonus


def regions_are_duplicates(region_a: dict, region_b: dict) -> bool:
    ax, ay, aw, ah = region_a["bbox"]
    bx, by, bw, bh = region_b["bbox"]
    overlap_x1 = max(ax, bx)
    overlap_y1 = max(ay, by)
    overlap_x2 = min(ax + aw, bx + bw)
    overlap_y2 = min(ay + ah, by + bh)
    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
        return False
    overlap_slice = np.s_[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
    overlap_px = int(
        cv2.countNonZero(
            cv2.bitwise_and(
                region_a["mask"][overlap_slice],
                region_b["mask"][overlap_slice],
            )
        )
    )
    if overlap_px <= 0:
        return False

    min_area = max(1, min(region_a["area_px"], region_b["area_px"]))
    overlap_share = overlap_px / min_area
    if overlap_share >= 0.55:
        return True

    union_px = region_a["area_px"] + region_b["area_px"] - overlap_px
    if union_px > 0 and (overlap_px / union_px) >= 0.30:
        return True

    a_center_x, a_center_y = region_a["center"]
    b_center_x, b_center_y = region_b["center"]
    return (
        overlap_share >= 0.32
        and point_in_bbox(a_center_x, a_center_y, region_b["bbox"])
        and point_in_bbox(b_center_x, b_center_y, region_a["bbox"])
    )


def merge_region_detections(regions: list[dict]) -> list[dict]:
    ordered = sorted(
        regions,
        key=lambda item: (
            -region_quality_score(item),
            -item["area_px"],
            item["bbox"][1],
            item["bbox"][0],
        ),
    )

    kept: list[dict] = []
    for region in ordered:
        if any(regions_are_duplicates(region, existing) for existing in kept):
            continue
        kept.append(region)

    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept


def build_combined_region_mask(mask_shape: tuple[int, ...], regions: list[dict]) -> np.ndarray:
    combined = np.zeros(mask_shape[:2], dtype=np.uint8)
    for region in regions:
        combined = cv2.bitwise_or(combined, region["mask"])
    return combined


def region_mask_overlap_share(region_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    region_area = int(cv2.countNonZero(region_mask))
    if region_area <= 0:
        return 0.0
    points = cv2.findNonZero(region_mask)
    if points is None:
        return 0.0
    x, y, width, height = cv2.boundingRect(points)
    region_crop = region_mask[y:y + height, x:x + width]
    reference_crop = reference_mask[y:y + height, x:x + width]
    overlap_px = int(cv2.countNonZero(cv2.bitwise_and(region_crop, reference_crop)))
    return overlap_px / region_area


def filter_supplemental_sahi_regions(
    sliced_regions: list[dict],
    base_regions: list[dict],
    mask_shape: tuple[int, ...],
) -> list[dict]:
    if not sliced_regions:
        return []

    accepted: list[dict] = []
    occupied_mask = build_combined_region_mask(mask_shape, base_regions)
    ordered = sorted(
        sliced_regions,
        key=lambda item: (
            -region_quality_score(item),
            -item["area_px"],
            item["bbox"][1],
            item["bbox"][0],
        ),
    )

    for region in ordered:
        overlap_share = region_mask_overlap_share(region["mask"], occupied_mask)
        if overlap_share >= 0.18:
            continue

        circularity = float(region.get("circularity", 0.0))
        aspect_ratio = float(region.get("aspect_ratio", 0.0))
        if circularity < 0.18:
            continue
        if aspect_ratio > 2.60 and region["area_px"] < 1200:
            continue
        if region["color"] == "Black" and circularity < 0.28:
            continue

        min_quality = 0.62
        if region["color"] in {"Black", "White/Colorless"}:
            min_quality = 0.70
        if region_quality_score(region) < min_quality:
            continue

        accepted.append(region)
        occupied_mask = cv2.bitwise_or(occupied_mask, region["mask"])

    return accepted


def detect_regions_with_sahi_slicing(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    slice_size: int,
    overlap_ratio: float,
    full_jewel_area: int,
    white_exclude_mask: np.ndarray | None = None,
    glare_threshold: int = DEFAULT_GLARE_THRESHOLD,
    background_calibration: dict | None = None,
    analysis_normalization: dict | None = None,
    learned_stone_profiles: list[dict] | None = None,
) -> tuple[list[dict], list[Rect]]:
    tight_bgr, tight_mask, tight_bbox = crop_to_nonzero_mask(roi_bgr, roi_mask)
    if tight_bgr is None or tight_mask is None or tight_bbox is None:
        return [], []
    tight_x, tight_y, _, _ = tight_bbox
    tight_white_exclude_mask = None
    if white_exclude_mask is not None:
        tight_white_exclude_mask = white_exclude_mask[tight_y:tight_y + tight_mask.shape[0], tight_x:tight_x + tight_mask.shape[1]]

    local_slice_bboxes = generate_sahi_slice_bboxes(
        tight_bgr.shape,
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
    )
    if not local_slice_bboxes:
        return [], []

    sliced_regions: list[dict] = []
    slice_bboxes: list[Rect] = []
    for local_slice_bbox in local_slice_bboxes:
        local_x, local_y, slice_w, slice_h = local_slice_bbox
        slice_bbox = (tight_x + local_x, tight_y + local_y, slice_w, slice_h)
        slice_bboxes.append(slice_bbox)
        slice_x, slice_y, slice_w, slice_h = slice_bbox
        slice_bgr = tight_bgr[local_y:local_y + slice_h, local_x:local_x + slice_w]
        slice_mask = tight_mask[local_y:local_y + slice_h, local_x:local_x + slice_w]
        slice_white_exclude_mask = None
        if tight_white_exclude_mask is not None:
            slice_white_exclude_mask = tight_white_exclude_mask[local_y:local_y + slice_h, local_x:local_x + slice_w]
        slice_area = int(cv2.countNonZero(slice_mask))
        if slice_area < 20:
            continue
        effective_area = max(slice_area, int(round(full_jewel_area * 0.28)))

        local_regions = detect_regions_in_roi(
            slice_bgr,
            slice_mask,
            area_reference=effective_area,
            min_area_floor=12,
            white_min_area_floor=8,
            black_min_area_floor=18,
            source_label="sahi_slice",
            white_exclude_mask=slice_white_exclude_mask,
            glare_threshold=glare_threshold,
            background_calibration=background_calibration,
            analysis_normalization=analysis_normalization,
            learned_stone_profiles=learned_stone_profiles,
        )
        for region in local_regions:
            # Standard boundary check
            if region_touches_internal_slice_border(region, slice_bgr.shape, slice_bbox, roi_bgr.shape):
                continue
            
            # Geometric filtering: SAHI tile-edge exclusion dead zone
            # Mark 5-10px border around each tile as "dead zone" to eliminate boundary artifacts
            region_bbox = tuple(region["bbox"])
            if touches_tile_boundary(region_bbox, (local_slice_bbox[0], local_slice_bbox[1], 
                                                    local_slice_bbox[2], local_slice_bbox[3]), 
                                    SAHI_TILE_BORDER_DEAD_ZONE):
                # Region touches SAHI tile boundary - can still keep if it's white and high confidence
                # Otherwise reject as likely artifact
                if region["color"] != "White/Colorless":
                    continue
            
            sliced_regions.append(offset_region_to_full_roi(region, slice_bbox, roi_bgr.shape))

    return sliced_regions, slice_bboxes


def create_alpha_crop(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    alpha = mask.copy()
    return cv2.merge([b, g, r, alpha])


def blend_mask(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    pixels = mask > 0
    if not np.any(pixels):
        return
    image_bgr[pixels] = (
        image_bgr[pixels].astype(np.float32) * (1.0 - alpha)
        + np.array(color_bgr, dtype=np.float32) * alpha
    ).astype(np.uint8)


@lru_cache(maxsize=32)
def load_annotation_font(font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf")
    )
    font_paths = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]

    for font_path in font_paths:
        if font_path.is_file():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except OSError:
                continue

    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def draw_professional_text(
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color_bgr: tuple[int, int, int] = ANNOTATION_TEXT_BGR,
    font_size: int = 14,
    bold: bool = False,
    background_bgr: tuple[int, int, int] | None = None,
    border_bgr: tuple[int, int, int] | None = None,
    padding: tuple[int, int] = (0, 0),
    corner_radius: int = 0,
) -> tuple[int, int]:
    """Draw crisp single-color text, optionally on a compact label card."""
    if not text:
        return (0, 0)

    pil_image = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = load_annotation_font(max(8, int(font_size)), bold=bold)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = max(1, text_box[2] - text_box[0])
    text_height = max(1, text_box[3] - text_box[1])
    pad_x, pad_y = padding
    card_width = text_width + (2 * pad_x)
    card_height = text_height + (2 * pad_y)

    image_height, image_width = image_bgr.shape[:2]
    x = min(max(0, int(origin[0])), max(0, image_width - card_width))
    y = min(max(0, int(origin[1])), max(0, image_height - card_height))

    if background_bgr is not None:
        card_box = (x, y, x + card_width - 1, y + card_height - 1)
        fill_rgb = tuple(reversed(background_bgr))
        border_rgb = tuple(reversed(border_bgr)) if border_bgr is not None else None
        draw.rounded_rectangle(
            card_box,
            radius=max(0, int(corner_radius)),
            fill=fill_rgb,
            outline=border_rgb,
            width=1 if border_rgb is not None else 0,
        )

    text_x = x + pad_x - text_box[0]
    text_y = y + pad_y - text_box[1]
    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=tuple(reversed(color_bgr)),
    )
    image_bgr[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return (card_width, card_height)


def _draw_color_legend(
    image_bgr: np.ndarray,
    color_entries: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Draw a compact color-swatch legend at the top-left corner of the image."""
    swatch = 12
    pad = 6
    font_size = 11
    row_h = swatch + 8
    total_h = pad + len(color_entries) * row_h + pad
    total_w = pad + swatch + 6 + 130 + pad

    h, w = image_bgr.shape[:2]
    x0, y0 = 8, 8
    x1 = min(w - 2, x0 + total_w)
    y1 = min(h - 2, y0 + total_h)

    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), ANNOTATION_CARD_BGR, -1)
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), ANNOTATION_BORDER_BGR, 1)

    for i, (color_name, color_bgr) in enumerate(color_entries):
        iy = y0 + pad + i * row_h
        sx0 = x0 + pad
        sx1 = sx0 + swatch
        sy0 = iy + (row_h - swatch) // 2
        sy1 = sy0 + swatch
        cv2.rectangle(image_bgr, (sx0, sy0), (sx1, sy1), color_bgr, -1)
        cv2.rectangle(image_bgr, (sx0, sy0), (sx1, sy1), (130, 130, 130), 1)
        draw_professional_text(
            image_bgr,
            color_name,
            (sx1 + 5, iy + 1),
            color_bgr=ANNOTATION_TEXT_BGR,
            font_size=font_size,
        )


def _render_jewel_outputs_legacy(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    regions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Retained only for compatibility with any external imports of the old
    # private helper. Delegate to the clean renderer defined below.
    return render_jewel_outputs(roi_bgr, roi_mask, regions)

    masked_preview = np.full_like(roi_bgr, 255)
    masked_preview[roi_mask > 0] = roi_bgr[roi_mask > 0]

    overlay = masked_preview.copy()
    color_mask_vis = np.full_like(roi_bgr, 20)

    for region in regions:
        color = COLOR_DRAW_BGR[region["color"]]
        mask = region["mask"]
        contour = region["contour"]
        blend_mask(overlay, mask, color, alpha=0.52)
        color_mask_vis[mask > 0] = color
        cv2.drawContours(overlay, [contour], -1, color, 1)
        cv2.drawContours(color_mask_vis, [contour], -1, (245, 245, 245), 1)
        learned_matches = region.get("learned_matches") or []
        if learned_matches:
            learned = learned_matches[0]
            hsv_center = learned.get("hsv_center") or []
            hsv_text = (
                f" HSV {float(hsv_center[0]):.0f},{float(hsv_center[1]):.0f},"
                f"{float(hsv_center[2]):.0f}"
                if len(hsv_center) == 3
                else ""
            )
            x, y, _w, _h = region["bbox"]
            draw_professional_text(
                overlay,
                f"{region['color']} · {learned['label']}{hsv_text}",
                (x, max(0, y - 22)),
                color_bgr=ANNOTATION_TEXT_BGR,
                font_size=11,
                bold=True,
                background_bgr=ANNOTATION_CARD_BGR,
                border_bgr=color,
                padding=(4, 3),
                corner_radius=3,
            )

    return create_alpha_crop(roi_bgr, roi_mask), overlay, color_mask_vis


def color_summary(regions: list[dict]) -> list[dict]:
    totals: dict[str, dict] = {}
    for region in regions:
        entry = totals.setdefault(
            region["color"],
            {
                "color": region["color"],
                "region_count": 0,
                "mask": np.zeros_like(region["mask"]),
                "possible_gemstones": region["possible_gemstones"],
                "learned_labels": set(),
            },
        )
        entry["region_count"] += 1
        entry["mask"] = cv2.bitwise_or(entry["mask"], region["mask"])
        for learned in region.get("learned_matches") or []:
            entry["learned_labels"].add(str(learned.get("label") or "Learned sample"))

    summary: list[dict] = []
    for entry in totals.values():
        summary.append(
            {
                "color": entry["color"],
                "region_count": entry["region_count"],
                "area_px": int(cv2.countNonZero(entry["mask"])),
                "possible_gemstones": entry["possible_gemstones"],
                "learned_labels": sorted(entry["learned_labels"]),
            }
        )
    return sorted(summary, key=lambda item: (-item["area_px"], item["color"]))


def learned_representation_summary(regions: list[dict]) -> list[dict]:
    summaries: dict[str, dict] = {}
    for region in regions:
        for match in region.get("learned_matches") or []:
            profile_id = str(match.get("profile_id") or "").strip()
            if not profile_id:
                continue
            entry = summaries.setdefault(
                profile_id,
                {
                    "profile_id": profile_id,
                    "label": str(match.get("label") or "Learned sample"),
                    "color": str(match.get("color") or region["color"]),
                    "hsv_center": list(match.get("hsv_center") or []),
                    "region_count": 0,
                    "area_px": 0,
                },
            )
            entry["region_count"] += 1
            entry["area_px"] += int(region.get("area_px", 0))
    return sorted(
        summaries.values(),
        key=lambda item: (item["color"], item["label"], item["profile_id"]),
    )


def render_jewel_outputs(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    regions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render clean contours with one compact marker per learned profile."""
    masked_preview = np.full_like(roi_bgr, 255)
    masked_preview[roi_mask > 0] = roi_bgr[roi_mask > 0]

    overlay = masked_preview.copy()
    color_mask_vis = np.full_like(roi_bgr, 20)
    for region in regions:
        color = COLOR_DRAW_BGR[region["color"]]
        mask = region["mask"]
        contour = region["contour"]
        blend_mask(overlay, mask, color, alpha=0.52)
        color_mask_vis[mask > 0] = color
        cv2.drawContours(overlay, [contour], -1, color, 1)
        cv2.drawContours(color_mask_vis, [contour], -1, (245, 245, 245), 1)

    representations = learned_representation_summary(regions)
    for representation_index, representation in enumerate(representations, start=1):
        representative_region = max(
            (
                region
                for region in regions
                if any(
                    match.get("profile_id") == representation["profile_id"]
                    for match in region.get("learned_matches") or []
                )
            ),
            key=lambda region: region["area_px"],
            default=None,
        )
        if representative_region is None:
            continue
        x, y, _w, _h = representative_region["bbox"]
        color = COLOR_DRAW_BGR[representative_region["color"]]
        draw_professional_text(
            overlay,
            f"L{representation_index}",
            (x, max(0, y - 16)),
            color_bgr=ANNOTATION_TEXT_BGR,
            font_size=9,
            bold=True,
            background_bgr=ANNOTATION_CARD_BGR,
            border_bgr=color,
            padding=(3, 2),
            corner_radius=2,
        )

    return create_alpha_crop(roi_bgr, roi_mask), overlay, color_mask_vis


def simplify_nested_regions(regions: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for region in sorted(regions, key=lambda item: item["area_px"], reverse=True):
        center_x, center_y = region["center"]
        suppress = False
        for existing in kept:
            if {region["color"], existing["color"]} != {"Red", "Pink"}:
                continue
            x, y, w, h = existing["bbox"]
            inside = x <= center_x <= (x + w) and y <= center_y <= (y + h)
            if inside and existing["area_px"] >= int(region["area_px"] * 1.8):
                suppress = True
                break
        if not suppress:
            kept.append(region)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept


def mark_repeated_micro_stone_groups(
    regions: list[dict],
    jewel_mask: np.ndarray,
) -> list[dict]:
    """Support repeated tiny stones without accepting isolated highlights."""
    for region in regions:
        region["micro_stone_supported"] = False

    for color_name in GEMSTONE_OPTIONS:
        candidate_indexes = [
            index
            for index, region in enumerate(regions)
            if region.get("color") == color_name
            and MICRO_STONE_MIN_AREA_PX
            <= int(region.get("area_px", 0))
            <= MICRO_STONE_MAX_AREA_PX
            and float(region.get("circularity", 0.0)) >= 0.16
            and float(region.get("extent", 0.0)) >= 0.20
            and float(region.get("aspect_ratio", 99.0)) <= 3.0
            and float(region.get("gold_overlap_percent", 100.0)) == 0.0
        ]
        if len(candidate_indexes) < MICRO_STONE_MIN_GROUP_COUNT:
            continue

        parents = {index: index for index in candidate_indexes}

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parents[root_right] = root_left

        for offset, left in enumerate(candidate_indexes):
            left_x, left_y = regions[left]["center"]
            for right in candidate_indexes[offset + 1:]:
                right_x, right_y = regions[right]["center"]
                if (
                    math.hypot(left_x - right_x, left_y - right_y)
                    <= MICRO_STONE_LINK_DISTANCE_PX
                ):
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in candidate_indexes:
            groups.setdefault(find(index), []).append(index)

        for group in groups.values():
            if len(group) < MICRO_STONE_MIN_GROUP_COUNT:
                continue
            total_area = sum(int(regions[index]["area_px"]) for index in group)
            if total_area < MICRO_STONE_MIN_GROUP_AREA_PX:
                continue

            padding = 5
            x0 = max(
                0,
                min(int(regions[index]["bbox"][0]) for index in group) - padding,
            )
            y0 = max(
                0,
                min(int(regions[index]["bbox"][1]) for index in group) - padding,
            )
            x1 = min(
                jewel_mask.shape[1],
                max(
                    int(
                        regions[index]["bbox"][0]
                        + regions[index]["bbox"][2]
                    )
                    for index in group
                )
                + padding,
            )
            y1 = min(
                jewel_mask.shape[0],
                max(
                    int(
                        regions[index]["bbox"][1]
                        + regions[index]["bbox"][3]
                    )
                    for index in group
                )
                + padding,
            )
            width = max(1, x1 - x0)
            height = max(1, y1 - y0)
            group_aspect = max(width, height) / max(1, min(width, height))
            local_jewel_density = (
                cv2.countNonZero(jewel_mask[y0:y1, x0:x1])
                / float(width * height)
            )
            if (
                group_aspect > 4.5
                or local_jewel_density < MICRO_STONE_MIN_LOCAL_JEWEL_DENSITY
            ):
                continue

            for index in group:
                regions[index]["micro_stone_supported"] = True
                regions[index]["micro_stone_group_count"] = len(group)
                regions[index]["micro_stone_group_area_px"] = total_area
                regions[index]["micro_stone_local_jewel_density"] = round(
                    local_jewel_density,
                    3,
                )

    return regions


def is_warm_white_reflection_signature(region: dict) -> bool:
    if region.get("color") != "White/Colorless":
        return False
    mean_hsv = region.get("mean_hsv") or [0.0, 0.0, 0.0]
    mean_lab_ab = region.get("mean_lab_ab") or [0.0, 0.0]
    try:
        mean_h = float(mean_hsv[0])
        mean_s = float(mean_hsv[1])
        mean_v = float(mean_hsv[2])
        mean_b = float(mean_lab_ab[1])
    except (TypeError, ValueError, IndexError):
        return False

    warm_hue = REFLECTIVE_WHITE_WARM_HUE_MIN <= mean_h <= REFLECTIVE_WHITE_WARM_HUE_MAX
    yellow_cast = mean_b >= REFLECTIVE_WHITE_WARM_LAB_B_MIN
    return bool(
        mean_v >= REFLECTIVE_WHITE_WARM_VALUE_MIN
        and yellow_cast
        and (warm_hue or mean_b >= REFLECTIVE_WHITE_WARM_LAB_B_MIN + 3.0)
        and mean_s >= REFLECTIVE_WHITE_WARM_SAT_MIN
    )


def is_low_confidence_edge_white(region: dict) -> bool:
    if region.get("color") != "White/Colorless":
        return False
    if int(region.get("area_px", 0)) > REFLECTIVE_EDGE_MICRO_AREA_MAX_PX:
        return False
    return bool(
        float(region.get("boundary_share", 0.0)) >= 0.70
        and float(region.get("max_boundary_depth_px", 999.0)) <= 2.0
        and not bool(region.get("learned_matches"))
    )


def is_yellow_gold_driven_multicolor(region: dict) -> bool:
    if region.get("color") != "Multicolor/Color-changing":
        return False
    if region.get("learned_matches"):
        return False
    color_mix = region.get("color_mix_percent") or {}
    try:
        yellow_share = float(color_mix.get("Yellow/Gold", 0.0))
    except (TypeError, ValueError):
        yellow_share = 0.0
    return yellow_share >= 55.0


def suppress_reflective_artifact_regions(
    regions: list[dict],
    jewel_area: int,
) -> list[dict]:
    if not regions:
        return regions

    filtered: list[dict] = []
    for region in regions:
        if is_low_confidence_edge_white(region):
            continue
        if is_yellow_gold_driven_multicolor(region):
            continue
        if (
            region.get("color") in COLOR_STONE_REGION_GROW_COLORS
            and int(region.get("seed_area_px", region.get("area_px", 0)))
            < STONE_REGION_GROW_COLOR_MIN_SEED_AREA_PX
            and not bool(region.get("micro_stone_supported"))
            and not bool(region.get("learned_matches"))
        ):
            continue
        filtered.append(region)

    if not filtered:
        return filtered

    non_white_regions = [
        region
        for region in filtered
        if region.get("color") != "White/Colorless"
    ]
    white_regions = [
        region
        for region in filtered
        if region.get("color") == "White/Colorless"
        and not bool(region.get("learned_matches"))
    ]
    if non_white_regions or len(white_regions) < REFLECTIVE_WHITE_ONLY_MIN_REGIONS:
        return filtered

    total_white_area = sum(int(region.get("area_px", 0)) for region in white_regions)
    if total_white_area <= 0 or jewel_area <= 0:
        return filtered

    white_percent = total_white_area / float(jewel_area) * 100.0
    largest_white = max(int(region.get("area_px", 0)) for region in white_regions)
    largest_limit = max(
        REFLECTIVE_WHITE_ONLY_MAX_LARGEST_REGION_PX,
        int(round(jewel_area * REFLECTIVE_WHITE_ONLY_MAX_LARGEST_JEWEL_FRACTION)),
    )
    warm_count = sum(
        1 for region in white_regions if is_warm_white_reflection_signature(region)
    )
    warm_share = warm_count / float(max(1, len(white_regions)))

    if (
        white_percent <= REFLECTIVE_WHITE_ONLY_MAX_PERCENT
        and largest_white <= largest_limit
        and warm_share >= 0.55
    ):
        return [
            region
            for region in filtered
            if region.get("color") != "White/Colorless"
            or bool(region.get("learned_matches"))
        ]

    return filtered


def separate_touching_stone_regions(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    regions: list[dict],
) -> tuple[list[dict], dict[str, int | bool]]:
    """Separate touching grown masks using their conservative HSV/LAB seeds.

    Region growth is useful for recovering dim and reflective stone facets, but
    neighbouring grown masks can touch and then be measured as one large stone.
    Pixels shared by touching masks are assigned to the nearest original seed;
    one boundary pixel is removed between competing assignments so connected-
    component measurement preserves the individual stone instances.
    """
    diagnostics: dict[str, int | bool] = {
        "enabled": True,
        "input_region_count": len(regions),
        "touching_cluster_count": 0,
        "separated_region_count": 0,
        "overlap_pixels_resolved": 0,
        "boundary_pixels_removed": 0,
        "disconnected_pixels_removed": 0,
    }
    if len(regions) < 2:
        return regions, diagnostics

    valid_indexes = [
        index
        for index, region in enumerate(regions)
        if cv2.countNonZero(region["mask"]) > 0
    ]
    if len(valid_indexes) < 2:
        return regions, diagnostics

    parents = {index: index for index in valid_indexes}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    touch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_masks = {
        index: cv2.dilate(regions[index]["mask"], touch_kernel, iterations=1)
        for index in valid_indexes
    }
    for offset, left in enumerate(valid_indexes):
        for right in valid_indexes[offset + 1:]:
            if regions[left]["color"] != regions[right]["color"]:
                continue
            left_x, left_y, left_width, left_height = regions[left]["bbox"]
            right_x, right_y, right_width, right_height = regions[right]["bbox"]
            if (
                left_x + left_width + 1 < right_x
                or right_x + right_width + 1 < left_x
                or left_y + left_height + 1 < right_y
                or right_y + right_height + 1 < left_y
            ):
                continue
            if cv2.countNonZero(
                cv2.bitwise_and(dilated_masks[left], regions[right]["mask"])
            ) > 0:
                union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in valid_indexes:
        clusters.setdefault(find(index), []).append(index)
    touching_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]
    diagnostics["touching_cluster_count"] = len(touching_clusters)
    if not touching_clusters:
        return regions, diagnostics

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    updated_regions = list(regions)
    separated_indexes: set[int] = set()

    for cluster in touching_clusters:
        cluster_union = np.zeros(jewel_mask.shape, dtype=np.uint8)
        input_area_sum = 0
        for index in cluster:
            cluster_union = cv2.bitwise_or(cluster_union, regions[index]["mask"])
            input_area_sum += int(cv2.countNonZero(regions[index]["mask"]))
        cluster_union = cv2.bitwise_and(cluster_union, jewel_mask)
        union_area = int(cv2.countNonZero(cluster_union))
        diagnostics["overlap_pixels_resolved"] += max(0, input_area_sum - union_area)

        x, y, width, height = cv2.boundingRect(cluster_union)
        local_union = cluster_union[y:y + height, x:x + width]
        labels = np.zeros(local_union.shape, dtype=np.int32)
        best_distance = np.full(local_union.shape, np.inf, dtype=np.float32)

        for label, index in enumerate(cluster, start=1):
            local_region = regions[index]["mask"][y:y + height, x:x + width]
            seed_mask = regions[index].get("seed_mask")
            if seed_mask is not None:
                local_seed = cv2.bitwise_and(
                    seed_mask[y:y + height, x:x + width],
                    local_region,
                )
            else:
                local_seed = np.zeros(local_region.shape, dtype=np.uint8)
            if cv2.countNonZero(local_seed) == 0:
                interior_distance = cv2.distanceTransform(
                    local_region,
                    cv2.DIST_L2,
                    3,
                )
                center_y, center_x = np.unravel_index(
                    int(np.argmax(interior_distance)),
                    interior_distance.shape,
                )
                local_seed[center_y, center_x] = 255

            distance_input = np.full(local_seed.shape, 255, dtype=np.uint8)
            distance_input[local_seed > 0] = 0
            seed_distance = cv2.distanceTransform(
                distance_input,
                cv2.DIST_L2,
                3,
            )
            replace = (local_region > 0) & (seed_distance < best_distance)
            labels[replace] = label
            best_distance[replace] = seed_distance[replace]

        remove_boundary = np.zeros(labels.shape, dtype=bool)
        local_height, local_width = labels.shape
        for delta_y, delta_x in ((0, 1), (1, -1), (1, 0), (1, 1)):
            if delta_x >= 0:
                left_x = slice(0, local_width - delta_x)
                right_x = slice(delta_x, local_width)
            else:
                left_x = slice(-delta_x, local_width)
                right_x = slice(0, local_width + delta_x)
            left_y = slice(0, local_height - delta_y)
            right_y = slice(delta_y, local_height)
            left_labels = labels[left_y, left_x]
            right_labels = labels[right_y, right_x]
            competing = (
                (left_labels > 0)
                & (right_labels > 0)
                & (left_labels != right_labels)
            )
            if not np.any(competing):
                continue
            left_distances = best_distance[left_y, left_x]
            right_distances = best_distance[right_y, right_x]
            remove_left = competing & (left_distances >= right_distances)
            remove_right = competing & ~remove_left
            left_removal = remove_boundary[left_y, left_x]
            right_removal = remove_boundary[right_y, right_x]
            left_removal[remove_left] = True
            right_removal[remove_right] = True

        labels[remove_boundary] = 0
        diagnostics["boundary_pixels_removed"] += int(np.count_nonzero(remove_boundary))

        for label, index in enumerate(cluster, start=1):
            local_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            updated_mask = np.zeros(jewel_mask.shape, dtype=np.uint8)
            updated_mask[y:y + height, x:x + width] = local_mask
            if cv2.countNonZero(updated_mask) == 0:
                continue

            component_count, component_labels, component_stats, _ = (
                cv2.connectedComponentsWithStats(
                    updated_mask,
                    connectivity=8,
                )
            )
            if component_count > 2:
                seed_mask = regions[index].get("seed_mask")
                best_component = 1
                best_score = (-1, -1)
                for component_id in range(1, component_count):
                    component_area = int(
                        component_stats[component_id, cv2.CC_STAT_AREA]
                    )
                    seed_overlap = 0
                    if seed_mask is not None:
                        seed_overlap = int(
                            np.count_nonzero(
                                (component_labels == component_id)
                                & (seed_mask > 0)
                            )
                        )
                    score = (seed_overlap, component_area)
                    if score > best_score:
                        best_component = component_id
                        best_score = score
                retained_mask = np.where(
                    component_labels == best_component,
                    255,
                    0,
                ).astype(np.uint8)
                diagnostics["disconnected_pixels_removed"] += max(
                    0,
                    int(cv2.countNonZero(updated_mask))
                    - int(cv2.countNonZero(retained_mask)),
                )
                updated_mask = retained_mask

            updated_region = dict(regions[index])
            previous_area = int(updated_region.get("area_px", 0))
            updated_region["mask"] = updated_mask
            area_px = int(cv2.countNonZero(updated_mask))
            contours, _ = cv2.findContours(
                updated_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contour = max(contours, key=cv2.contourArea)
            region_x, region_y, region_width, region_height = cv2.boundingRect(updated_mask)
            moments = cv2.moments(updated_mask, binaryImage=True)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            mean_h, mean_s, mean_v = mean_hsv_for_mask(hsv_image, updated_mask)
            mean_lab_a, mean_lab_b, mean_lab_chroma = mean_lab_ab_for_mask(
                image_bgr,
                updated_mask,
            )
            seed_area = max(1, int(updated_region.get("seed_area_px", area_px)))
            expansion = dict(updated_region.get("stone_region_expansion") or {})
            expansion.update(
                {
                    "expanded_area_px": area_px,
                    "area_gain": round(area_px / float(seed_area), 3),
                    "seeded_instance_separation": True,
                    "pre_separation_area_px": previous_area,
                }
            )
            updated_region.update(
                {
                    "area_px": area_px,
                    "bbox": [region_x, region_y, region_width, region_height],
                    "center": [center_x, center_y],
                    "mean_hsv": [round(mean_h, 1), round(mean_s, 1), round(mean_v, 1)],
                    "mean_lab_ab": [round(mean_lab_a, 1), round(mean_lab_b, 1)],
                    "lab_chroma_mean": round(mean_lab_chroma, 1),
                    "extent": round(
                        area_px / max(1, region_width * region_height),
                        3,
                    ),
                    "aspect_ratio": round(
                        max(region_width, region_height)
                        / max(1, min(region_width, region_height)),
                        3,
                    ),
                    "circularity": round(contour_circularity(contour), 3),
                    "stone_region_expansion": expansion,
                    "contour": contour,
                }
            )
            updated_regions[index] = updated_region
            separated_indexes.add(index)

    diagnostics["separated_region_count"] = len(separated_indexes)
    for region_id, region in enumerate(updated_regions, start=1):
        region["region_id"] = region_id
    return updated_regions, diagnostics


def serialize_regions(regions: list[dict]) -> list[dict]:
    serialized_regions = []
    for region in regions:
        serialized_regions.append(
            {
                key: value
                for key, value in region.items()
                if key not in {"mask", "seed_mask", "contour", "source"}
            }
        )
    return serialized_regions


def build_jewel_report_dict(
    jewel_index: int,
    roi_mask: np.ndarray,
    regions: list[dict],
    bbox_global: tuple[int, int, int, int] | None = None,
    outputs: dict[str, str] | None = None,
) -> dict:
    jewel_area_px = int(cv2.countNonZero(roi_mask))
    stone_mask = build_combined_region_mask(roi_mask.shape, regions)
    stone_area_px = min(int(cv2.countNonZero(stone_mask)), jewel_area_px)
    stone_seed_area_px = sum(
        max(0, int(region.get("seed_area_px", region.get("area_px", 0))))
        for region in regions
    )
    report = {
        "jewel_id": jewel_index,
        "jewel_area_px": jewel_area_px,
        "stone_area_px": stone_area_px,
        "stone_seed_area_px": stone_seed_area_px,
        "stone_mask_area_gain": round(
            stone_area_px / float(stone_seed_area_px)
            if stone_seed_area_px > 0
            else 1.0,
            3,
        ),
        "stone_percentage": round(
            stone_area_px / float(jewel_area_px) * 100.0
            if jewel_area_px > 0
            else 0.0,
            2,
        ),
        "outputs": outputs or {},
        "detected_colors": color_summary(regions),
        "learned_representations": learned_representation_summary(regions),
        "regions": serialize_regions(regions),
    }
    if bbox_global is not None:
        report["bbox_global"] = [int(v) for v in bbox_global]
    return report


def write_jewel_output_images(
    output_dir: Path,
    jewel_index: int,
    masked_png: np.ndarray,
    overlay_png: np.ndarray,
    color_mask_png: np.ndarray,
) -> dict[str, str]:
    jewel_prefix = f"jewel_{jewel_index:02d}"
    masked_path = output_dir / f"{jewel_prefix}_masked.png"
    overlay_path = output_dir / f"{jewel_prefix}_overlay.png"
    color_mask_path = output_dir / f"{jewel_prefix}_color_mask.png"

    cv2.imwrite(str(masked_path), masked_png)
    cv2.imwrite(str(overlay_path), overlay_png)
    cv2.imwrite(str(color_mask_path), color_mask_png)

    return {
        "masked_png": str(masked_path),
        "overlay_png": str(overlay_path),
        "color_mask_png": str(color_mask_path),
    }


def resize_to_fit(image_bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return image_bgr

    scale = min(max_width / width, max_height / height)
    if scale >= 1.0:
        return image_bgr.copy()

    resized = cv2.resize(
        image_bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def gallery_summary_text(detected_colors: list[dict], reflection: dict | None = None) -> str:
    color_text = (
        ", ".join(
            "Multicolor / Mixed Appearance"
            if entry["color"] == "Multicolor/Color-changing"
            else entry["color"]
            for entry in detected_colors[:3]
        )
        if detected_colors
        else "No gemstone-colored regions"
    )
    if reflection and reflection.get("flagged"):
        return f"RISK | {color_text} | Transparent stones possible"
    return color_text


def gallery_learned_text(learned_representations: list[dict]) -> str:
    parts: list[str] = []
    for index, representation in enumerate(learned_representations[:3], start=1):
        hsv = representation.get("hsv_center") or []
        hsv_text = (
            f" HSV {float(hsv[0]):.0f},{float(hsv[1]):.0f},{float(hsv[2]):.0f}"
            if len(hsv) == 3
            else ""
        )
        count = int(representation.get("region_count", 0))
        parts.append(
            f"L{index} {representation.get('label', 'Learned sample')} "
            f"({representation.get('color', '-')}, {count} region{'s' if count != 1 else ''})"
            f"{hsv_text}"
        )
    return " | ".join(parts)


def build_result_gallery(jewel_views: list[dict]) -> np.ndarray:
    if not jewel_views:
        canvas = np.full((480, 720, 3), 250, dtype=np.uint8)
        draw_professional_text(
            canvas,
            "No jewelry candidates found",
            (36, 110),
            font_size=22,
            bold=True,
        )
        draw_professional_text(
            canvas,
            "Load a clearer gold-jewelry image and analyze again",
            (36, 154),
            color_bgr=ANNOTATION_MUTED_TEXT_BGR,
            font_size=15,
        )
        return canvas

    cols = 1 if len(jewel_views) == 1 else 2
    rows = (len(jewel_views) + cols - 1) // cols
    tile_width = 520
    tile_height = 360
    margin = 24
    header_height = 76

    canvas_height = margin + rows * (header_height + tile_height + margin)
    canvas_width = margin + cols * (tile_width + margin)
    canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)

    for idx, jewel_view in enumerate(jewel_views):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (tile_width + margin)
        y0 = margin + row * (header_height + tile_height + margin)

        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + tile_width, y0 + header_height + tile_height),
            ANNOTATION_BORDER_BGR,
            1,
        )
        draw_professional_text(
            canvas,
            f"Jewel {jewel_view['jewel_id']}",
            (x0 + 14, y0 + 8),
            font_size=17,
            bold=True,
        )
        draw_professional_text(
            canvas,
            gallery_summary_text(
                jewel_view["detected_colors"],
                jewel_view.get("reflection"),
            ),
            (x0 + 14, y0 + 33),
            color_bgr=ANNOTATION_MUTED_TEXT_BGR,
            font_size=13,
        )
        learned_text = gallery_learned_text(
            jewel_view.get("learned_representations") or []
        )
        if learned_text:
            draw_professional_text(
                canvas,
                learned_text,
                (x0 + 14, y0 + 54),
                color_bgr=ANNOTATION_MUTED_TEXT_BGR,
                font_size=10,
            )

        preview = resize_to_fit(jewel_view["overlay_bgr"], tile_width - 20, tile_height - 20)
        ph, pw = preview.shape[:2]
        px = x0 + (tile_width - pw) // 2
        py = y0 + header_height + (tile_height - ph) // 2
        canvas[py:py + ph, px:px + pw] = preview

    return canvas


def build_report_text(report: dict) -> str:
    slicing = report.get("sahi_slicing", {})
    if slicing.get("enabled"):
        slicing_text = (
            f"SAHI slicing: on | size={slicing.get('slice_size', DEFAULT_SAHI_SLICE_SIZE)}"
            f" | overlap={float(slicing.get('overlap_ratio', DEFAULT_SAHI_OVERLAP)):.2f}"
        )
    else:
        slicing_text = "SAHI slicing: off"

    glare = report.get("glare_preprocessing", {})
    if glare.get("enabled"):
        glare_text = (
            f"Glare removal: on | gray>={glare.get('threshold', DEFAULT_GLARE_THRESHOLD)}"
            f" | patch={glare.get('patch_size', DEFAULT_GLARE_PATCH_SIZE)}"
            f" | sat<={glare.get('saturation_max', DEFAULT_GLARE_SATURATION_MAX)}"
        )
    else:
        glare_text = "Glare removal: off"

    lines = [
        f"Image: {report['source_image']}",
        f"Jewels found: {report['jewel_count']}",
        (
            "Overall status: RISK - dense reflection indicates possible "
            "additional transparent/colorless gemstones."
            if report.get("reflection_flagged")
            else "Overall reflection status: NORMAL"
        ),
        (
            "Ignored ROI(s): "
            + ", ".join(str(tuple(region)) for region in report.get("ignored_regions", []))
        )
        if report.get("ignored_regions")
        else "Ignored ROI(s): none",
        slicing_text,
        glare_text,
        report["note"],
        "",
    ]

    if not report["jewels"]:
        lines.append("No jewelry candidates found.")
        return "\n".join(lines)

    for jewel in report["jewels"]:
        bbox = jewel.get("bbox_global", [])
        lines.append(
            f"Jewel {jewel['jewel_id']} | area={jewel['jewel_area_px']} px"
            + (f" | bbox={tuple(bbox)}" if bbox else "")
        )
        glare_px = int(jewel.get("glare_inpaint_px", jewel.get("glare_mask_px", 0)))
        glare_regions = int(jewel.get("glare_region_count", 0))
        if glare_px > 0:
            lines.append(f"  Glare inpainted: {glare_px} px across {glare_regions} region(s).")
        reflection = jewel.get("reflection") or {}
        if reflection.get("flagged"):
            lines.append(
                "  RISK JEWEL: Dense reflection indicates possible additional transparent/colorless gemstones. "
                f"{reflection.get('level', 'elevated')} | "
                f"coverage={float(reflection.get('coverage_percent', 0.0)):.2f}% | "
                f"local density={float(reflection.get('local_density_percent', 0.0)):.2f}%"
            )
        lines.append(
            f"  Stone coverage: {float(jewel.get('stone_percentage', 0.0)):.2f}% "
            f"({int(jewel.get('stone_area_px', 0))}/{int(jewel.get('jewel_area_px', 0))} px)"
        )
        if not jewel["detected_colors"]:
            lines.append("  No gemstone-colored regions detected.")
        else:
            lines.append("  Detected colors")
            for entry in jewel["detected_colors"]:
                display_color = (
                    "Multicolor / Mixed Appearance"
                    if entry["color"] == "Multicolor/Color-changing"
                    else entry["color"]
                )
                lines.append(
                    f"  - {display_color}: {entry['region_count']} region(s), {entry['area_px']} px"
                )
                lines.append(f"    Possible gemstones: {', '.join(entry['possible_gemstones'])}")
                if entry.get("learned_labels"):
                    lines.append(
                        f"    Learned representation: {', '.join(entry['learned_labels'])}"
                    )

        if jewel["regions"]:
            lines.append("  Regions")
            for region in jewel["regions"]:
                lines.append(
                    f"  - Region {region['region_id']}: {region['color']} | "
                    f"area={region['area_px']} px | HSV={tuple(region['mean_hsv'])}"
                )
                if region.get("learned_matches"):
                    lines.append(
                        "    Matched learned profile(s): "
                        + ", ".join(
                            str(match.get("label") or "Learned sample")
                            for match in region["learned_matches"]
                        )
                    )
        lines.append("")

    return "\n".join(lines).rstrip()


def analyze_jewel_candidate(
    jewel_index: int,
    jewel: dict,
    zoom_scale: int,
    use_glare_removal: bool,
    glare_threshold: int,
    glare_patch_size: int,
    use_sahi_slicing: bool,
    sahi_slice_size: int,
    sahi_overlap_ratio: float,
    color_correction: dict | None,
    background_calibration: dict | None,
    analysis_normalization: dict | None,
    measurement_scale: dict | None,
    learned_stone_profiles: list[dict] | None,
    fastsam_model: object | None = None,
    fastsam_lock: object | None = None,
    stone_v2_debug_dir: Path | None = None,
) -> tuple[dict, dict]:
    refined_crop_mask, mask_cleanup = remove_enclosed_background_from_jewel_mask(
        jewel["crop_bgr"],
        jewel["crop_mask"],
        calibration=background_calibration,
    )
    raw_zoomed_bgr, zoomed_mask = zoom_pair(
        jewel["crop_bgr"],
        refined_crop_mask,
        zoom_scale,
    )
    mask_cleanup["analysis_zoom_scale"] = int(max(1, zoom_scale))
    mask_cleanup["background_hole_pixels_removed_analysis"] = int(
        mask_cleanup["background_hole_pixels_removed"]
        * max(1, zoom_scale)
        * max(1, zoom_scale)
    )
    original_area = int(cv2.countNonZero(zoomed_mask))
    normalized_bgr = build_normalized_analysis_image(
        raw_zoomed_bgr,
        settings=analysis_normalization,
        background_calibration=background_calibration,
    )
    display_bgr = apply_color_correction(
        raw_zoomed_bgr,
        color_correction,
        mask=zoomed_mask,
    )
    gold_px, gold_ratio = gold_support_stats(normalized_bgr, zoomed_mask)
    if gold_px < MIN_GOLD_PIXELS_FOR_STONE_DETECTION or gold_ratio < MIN_GOLD_RATIO_FOR_STONE_DETECTION:
        empty_regions: list[dict] = []
        masked_png, overlay_png, color_mask_png = render_jewel_outputs(
            display_bgr,
            zoomed_mask,
            empty_regions,
        )
        jewel_report = build_jewel_report_dict(
            jewel_index=jewel_index,
            roi_mask=zoomed_mask,
            regions=empty_regions,
            bbox_global=jewel["bbox_global"],
        )
        if _STONE_AREA_CALC_AVAILABLE:
            try:
                empty_area_stats = _stone_area_calc.calculate_stone_area_statistics(
                    zoomed_mask,
                    {},
                    min_component_area_pixels=MIN_STONE_COMPONENT_AREA_PX,
                )
            except Exception as exc:
                empty_area_stats = {"success": False, "error": str(exc)}
        else:
            empty_area_stats = {
                "success": False,
                "error": "stone_area_calculator not available",
            }
        jewel_report["stone_area_statistics"] = empty_area_stats
        jewel_report["mask_cleanup"] = mask_cleanup
        jewel_report["area_denominator"] = "segmented_jewel_mask"
        jewel_report["glare_mask_px"] = 0
        jewel_report["glare_region_count"] = 0
        jewel_report["glare_inpaint_px"] = 0
        jewel_report["skipped_stone_detection"] = True
        jewel_report["skip_reason"] = "Gold region too small for reliable stone detection"
        jewel_report["gold_support_px"] = int(gold_px)
        jewel_report["gold_support_ratio"] = round(float(gold_ratio), 4)
        jewel_report["reflection"] = reflection_metrics(
            np.zeros_like(zoomed_mask),
            zoomed_mask,
        )
        jewel_view = {
            "jewel_id": jewel_index,
            "bbox_global": [int(v) for v in jewel["bbox_global"]],
            "detected_colors": jewel_report["detected_colors"],
            "learned_representations": jewel_report["learned_representations"],
            "reflection": jewel_report["reflection"],
            "masked_bgra": masked_png,
            "overlay_bgr": overlay_png,
            "color_mask_bgr": color_mask_png,
        }
        return jewel_report, jewel_view

    pre_glare_white_regions = [
        region
        for region in detect_regions_in_roi(
            normalized_bgr,
            zoomed_mask,
            area_reference=original_area,
            min_area_floor=18,
            white_min_area_floor=10,
            black_min_area_floor=18,
            source_label="pre_glare_white",
            white_exclude_mask=None,
            glare_threshold=glare_threshold,
            background_calibration=background_calibration,
            analysis_normalization=analysis_normalization,
            learned_stone_profiles=learned_stone_profiles,
            allowed_colors={"White/Colorless"},
        )
        if region["color"] == "White/Colorless"
    ]

    glare_processed_bgr, glare_white_exclude_mask, glare_stats = remove_specular_glare(
        normalized_bgr,
        zoomed_mask,
        enabled=use_glare_removal,
        threshold=glare_threshold,
        patch_size=glare_patch_size,
    )
    jewel_area = original_area
    regions = detect_regions_in_roi(
        glare_processed_bgr,
        zoomed_mask,
        area_reference=jewel_area,
        min_area_floor=18,
        white_min_area_floor=10,
        black_min_area_floor=18,
        source_label="full_roi",
        white_exclude_mask=glare_white_exclude_mask,
        glare_threshold=glare_threshold,
        background_calibration=background_calibration,
        analysis_normalization=analysis_normalization,
        learned_stone_profiles=learned_stone_profiles,
    )
    if pre_glare_white_regions:
        regions = merge_region_detections(regions + pre_glare_white_regions)
    if use_sahi_slicing:
        sliced_regions, _ = detect_regions_with_sahi_slicing(
            glare_processed_bgr,
            zoomed_mask,
            slice_size=sahi_slice_size,
            overlap_ratio=sahi_overlap_ratio,
            full_jewel_area=jewel_area,
            white_exclude_mask=glare_white_exclude_mask,
            glare_threshold=glare_threshold,
            background_calibration=background_calibration,
            analysis_normalization=analysis_normalization,
            learned_stone_profiles=learned_stone_profiles,
        )
        sliced_regions = filter_supplemental_sahi_regions(sliced_regions, regions, zoomed_mask.shape)
        regions = merge_region_detections(regions + sliced_regions)

    regions = simplify_nested_regions(regions)
    regions = mark_repeated_micro_stone_groups(regions, zoomed_mask)
    for region_id, region in enumerate(regions, start=1):
        region["region_id"] = region_id

    white_only_area = sum(region["area_px"] for region in regions if region["color"] == "White/Colorless")
    has_non_white = any(region["color"] != "White/Colorless" for region in regions)
    has_supported_micro_stones = any(
        bool(region.get("micro_stone_supported"))
        for region in regions
    )
    if (
        regions
        and not has_non_white
        and not has_supported_micro_stones
        and white_only_area < max(100, int(round(jewel_area * 0.001)))
    ):
        regions = []
    low_confidence_neutral_colors = {"White/Colorless", "Black"}
    for neutral_color in low_confidence_neutral_colors:
        neutral_area = sum(region["area_px"] for region in regions if region["color"] == neutral_color)
        neutral_has_micro_support = any(
            region.get("color") == neutral_color
            and bool(region.get("micro_stone_supported"))
            for region in regions
        )
        if (
            neutral_area
            and not neutral_has_micro_support
            and neutral_area < max(120, int(round(jewel_area * 0.003)))
        ):
            regions = [
                region
                for region in regions
                if region["color"] != neutral_color
            ]

    # Filter: suppress colors where every individual detected region is too small
    # to be a real stone. Gold chain links and reflective surfaces produce many
    # scattered tiny detections (30-70 px each); a real stone—even a small one—
    # produces at least one region above this floor.
    if regions:
        color_boost = normalize_analysis_normalization(
            analysis_normalization
        )["color_boost"]
        _single_px_floor = max(
            12,
            int(round(MIN_SINGLE_STONE_REGION_PX / (color_boost * color_boost))),
            int(
                round(
                    jewel_area
                    * MIN_SINGLE_STONE_REGION_JEWEL_FRACTION
                    / color_boost
                )
            ),
        )
        _color_max_single: dict[str, int] = {}
        for _r in regions:
            _c = _r["color"]
            if _r["area_px"] > _color_max_single.get(_c, 0):
                _color_max_single[_c] = _r["area_px"]
        regions = [
            _r for _r in regions
            if (
                (
                    _r["color"] in {"White/Colorless", "Black"}
                    and _r["area_px"] >= _single_px_floor
                )
                or (
                    _r["color"] not in {"White/Colorless", "Black"}
                    and _color_max_single.get(_r["color"], 0) >= _single_px_floor
                )
                or bool(_r.get("learned_matches"))
                or bool(_r.get("micro_stone_supported"))
            )
        ]

        regions = [
            _r for _r in regions
            if not (
                _r["color"] == "White/Colorless"
                and float(_r.get("boundary_share", 0.0)) >= 0.90
                and float(_r.get("max_boundary_depth_px", 999.0))
                <= BACKGROUND_BOUNDARY_WIDTH_PX
                and not bool(_r.get("learned_matches"))
            )
        ]

        for neutral_color in low_confidence_neutral_colors:
            neutral_area = sum(
                region["area_px"]
                for region in regions
                if region["color"] == neutral_color
            )
            neutral_has_micro_support = any(
                region.get("color") == neutral_color
                and bool(region.get("micro_stone_supported"))
                for region in regions
            )
            if (
                neutral_area
                and not neutral_has_micro_support
                and neutral_area < max(120, int(round(jewel_area * 0.003)))
            ):
                regions = [
                    region
                    for region in regions
                    if region["color"] != neutral_color
                ]

    regions = suppress_reflective_artifact_regions(regions, jewel_area)
    _stone_v2_diagnostics: dict = {
        "enabled": False,
        "fallback": "legacy_regions",
        "candidate_count": len(regions),
        "stone_instance_count": len(regions),
        "segmentation_method_counts": {"legacy": len(regions)} if regions else {},
    }
    _stone_v2_debug_artifacts: dict[str, str] = {}
    if _STONE_V2_AVAILABLE:
        try:
            v2_gold_mask = build_gold_mask(
                cv2.cvtColor(glare_processed_bgr, cv2.COLOR_BGR2HSV),
                zoomed_mask,
                strict=False,
            )
            v2_strict_gold_mask = build_gold_mask(
                cv2.cvtColor(glare_processed_bgr, cv2.COLOR_BGR2HSV),
                zoomed_mask,
                strict=True,
            )
            color_measurement_bgr = _stone_v2.build_color_measurement_image(
                raw_zoomed_bgr,
                zoomed_mask,
                background_calibration,
            )
            color_measurement_hsv = cv2.cvtColor(
                color_measurement_bgr,
                cv2.COLOR_BGR2HSV,
            )
            color_measurement_lab = cv2.cvtColor(
                color_measurement_bgr,
                cv2.COLOR_BGR2LAB,
            )
            v2_candidates = _stone_v2.generate_stone_candidates(
                glare_processed_bgr,
                zoomed_mask,
                regions,
                v2_gold_mask,
                glare_mask=glare_white_exclude_mask,
            )
            v2_regions, _stone_v2_diagnostics = (
                _stone_v2.build_final_stone_instances(
                    glare_processed_bgr,
                    zoomed_mask,
                    v2_candidates,
                    v2_gold_mask,
                    v2_strict_gold_mask,
                    fastsam_model=fastsam_model,
                    inference_lock=fastsam_lock,
                )
            )
            for region in v2_regions:
                mask = region["mask"]
                classification = _stone_v2.classify_stone_instance_color(
                    color_measurement_bgr,
                    mask,
                    gold_mask=v2_gold_mask,
                    lab_image=color_measurement_lab,
                    hsv_image=color_measurement_hsv,
                )
                color = str(classification["color"])
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                area_px = int(cv2.countNonZero(mask))
                x, y, width, height = cv2.boundingRect(contour)
                moments = cv2.moments(mask, binaryImage=True)
                center = (
                    [
                        int(round(moments["m10"] / moments["m00"])),
                        int(round(moments["m01"] / moments["m00"])),
                    ]
                    if moments["m00"] > 0
                    else [x + width // 2, y + height // 2]
                )
                mean_h, mean_s, mean_v = mean_hsv_for_mask(
                    color_measurement_hsv,
                    mask,
                )
                mean_lab_a, mean_lab_b, mean_lab_chroma = mean_lab_ab_for_mask(
                    color_measurement_bgr,
                    mask,
                    lab_image=color_measurement_lab,
                )
                region.update(
                    {
                        "color": color,
                        "display_color": classification["display_color"],
                        "possible_gemstones": GEMSTONE_OPTIONS[color],
                        "color_confidence": classification["color_confidence"],
                        "color_classification": classification,
                        "area_px": area_px,
                        "bbox": [int(x), int(y), int(width), int(height)],
                        "center": center,
                        "contour": contour,
                        "mean_hsv": [round(mean_h, 1), round(mean_s, 1), round(mean_v, 1)],
                        "mean_lab_ab": [round(mean_lab_a, 1), round(mean_lab_b, 1)],
                        "lab_chroma_mean": round(mean_lab_chroma, 1),
                        "color_mix_percent": {color: 100.0},
                        "dominant_share_percent": round(
                            float(classification["color_confidence"]) * 100.0,
                            1,
                        ),
                        "extent": round(area_px / float(max(1, width * height)), 3),
                        "aspect_ratio": round(
                            max(width, height) / float(max(1, min(width, height))),
                            3,
                        ),
                        "circularity": round(contour_circularity(contour), 3),
                    }
                )
            regions = v2_regions
            _stone_v2_diagnostics["enabled"] = True
            _stone_v2_diagnostics["color_measurement_image"] = (
                "captured_original_with_conservative_white_reference"
            )
            if stone_v2_debug_dir is not None:
                _stone_v2_debug_artifacts = _stone_v2.save_debug_artifacts(
                    stone_v2_debug_dir,
                    color_measurement_bgr,
                    zoomed_mask,
                    v2_candidates,
                    regions,
                )
        except Exception as _exc:
            _stone_v2_diagnostics = {
                "enabled": False,
                "fallback": "legacy_regions",
                "error": str(_exc),
                "candidate_count": len(regions),
                "stone_instance_count": len(regions),
                "segmentation_method_counts": {"legacy": len(regions)} if regions else {},
            }
    regions, _seeded_separation = separate_touching_stone_regions(
        glare_processed_bgr,
        zoomed_mask,
        regions,
    )

    # Calculate accurate stone-area statistics from filled masks BEFORE
    # visualization so contour lines and overlays do not affect pixel counts.
    _stone_masks_by_color: dict[str, np.ndarray] = {}
    for _r in regions:
        _c = _r["color"]
        if _c not in _stone_masks_by_color:
            _stone_masks_by_color[_c] = np.zeros(zoomed_mask.shape, dtype=np.uint8)
        _stone_masks_by_color[_c] = cv2.bitwise_or(
            _stone_masks_by_color[_c], _r["mask"]
        )
    if _STONE_AREA_CALC_AVAILABLE:
        try:
            _stone_area_stats = _stone_area_calc.calculate_stone_area_statistics(
                zoomed_mask,
                _stone_masks_by_color,
                min_component_area_pixels=MIN_STONE_COMPONENT_AREA_PX,
            )
        except Exception as _exc:
            _stone_area_stats = {"success": False, "error": str(_exc)}
    else:
        _stone_area_stats = {"success": False, "error": "stone_area_calculator not available"}

    _stone_measurements = {
        "success": False,
        "error": "Metric calibration is unavailable.",
        "instances": [],
    }
    if (
        _STONE_AREA_CALC_AVAILABLE
        and measurement_scale
        and hasattr(_stone_area_calc, "calculate_stone_measurements")
    ):
        try:
            _stone_measurements = _stone_area_calc.calculate_stone_measurements(
                _stone_masks_by_color,
                float(measurement_scale["mm_per_pixel_x"]),
                float(measurement_scale["mm_per_pixel_y"]),
                min_component_area_pixels=MIN_STONE_COMPONENT_AREA_PX,
            )
        except Exception as _exc:
            _stone_measurements = {
                "success": False,
                "error": str(_exc),
                "instances": [],
            }

    masked_png, overlay_png, color_mask_png = render_jewel_outputs(
        display_bgr,
        zoomed_mask,
        regions,
    )
    jewel_report = build_jewel_report_dict(
        jewel_index=jewel_index,
        roi_mask=zoomed_mask,
        regions=regions,
        bbox_global=jewel["bbox_global"],
    )
    jewel_report["stone_area_statistics"] = _stone_area_stats
    if _stone_area_stats.get("success"):
        jewel_report["jewel_area_px"] = int(
            _stone_area_stats["jewel_area_pixels"]
        )
        jewel_report["stone_area_px"] = int(
            _stone_area_stats["stone_area_pixels"]
        )
        jewel_report["stone_percentage"] = float(
            _stone_area_stats["stone_percentage"]
        )
        jewel_report["metal_area_px"] = int(
            _stone_area_stats["metal_area_pixels"]
        )
        jewel_report["metal_percentage"] = float(
            _stone_area_stats["metal_percentage"]
        )
        jewel_report["area_denominator"] = "segmented_jewel_mask"
    jewel_report["mask_cleanup"] = mask_cleanup
    jewel_report["seeded_instance_separation"] = _seeded_separation
    jewel_report["stone_analysis_v2"] = _stone_v2_diagnostics
    jewel_report["stone_instance_count"] = len(regions)
    jewel_report["segmentation_method_counts"] = dict(
        _stone_v2_diagnostics.get("segmentation_method_counts") or {}
    )
    jewel_report["debug_artifacts"] = _stone_v2_debug_artifacts
    jewel_report["stone_measurements"] = _stone_measurements
    jewel_report["glare_mask_px"] = int(glare_stats["glare_mask_px"])
    jewel_report["glare_region_count"] = int(glare_stats["glare_region_count"])
    jewel_report["glare_inpaint_px"] = int(glare_stats["glare_inpaint_px"])
    jewel_report["reflection"] = glare_stats["reflection"]
    jewel_report["skipped_stone_detection"] = False
    jewel_report["gold_support_px"] = int(gold_px)
    jewel_report["gold_support_ratio"] = round(float(gold_ratio), 4)
    jewel_view = {
        "jewel_id": jewel_index,
        "bbox_global": [int(v) for v in jewel["bbox_global"]],
        "detected_colors": jewel_report["detected_colors"],
        "learned_representations": jewel_report["learned_representations"],
        "reflection": jewel_report["reflection"],
        "masked_bgra": masked_png,
        "overlay_bgr": overlay_png,
        "color_mask_bgr": color_mask_png,
    }
    return jewel_report, jewel_view


def analyze_image_bgr(
    image_bgr: np.ndarray,
    source_name: str = "",
    zoom_scale: int = 3,
    extraction_mode: str = "default",
    preset_candidates: list[dict] | None = None,
    ignore_regions: list[Rect] | None = None,
    use_glare_removal: bool = DEFAULT_GLARE_REMOVAL_ENABLED,
    glare_threshold: int = DEFAULT_GLARE_THRESHOLD,
    glare_patch_size: int = DEFAULT_GLARE_PATCH_SIZE,
    use_sahi_slicing: bool = DEFAULT_SAHI_ENABLED,
    sahi_slice_size: int = DEFAULT_SAHI_SLICE_SIZE,
    sahi_overlap_ratio: float = DEFAULT_SAHI_OVERLAP,
    external_mask: np.ndarray | None = None,
    color_correction: dict | None = None,
    background_calibration: dict | None = None,
    analysis_normalization: dict | None = None,
    measurement_scale: dict | None = None,
    learned_stone_profiles: list[dict] | None = None,
    fastsam_model: object | None = None,
    fastsam_lock: object | None = None,
    stone_v2_debug_dir: Path | None = None,
) -> dict:
    working_bgr, normalized_ignore_regions = apply_ignore_regions_to_image(image_bgr, ignore_regions)
    if preset_candidates is not None:
        jewels = list(preset_candidates)
    else:
        jewels = extract_jewel_candidates(
            working_bgr,
            extraction_mode=extraction_mode,
            external_mask=external_mask,
        )
    jewels = [
        jewel for jewel in jewels
        if has_enough_gold_for_stone_detection(jewel["crop_bgr"], jewel["crop_mask"])
    ]
    report: dict = {
        "source_image": source_name or "<in-memory>",
        "image_shape": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
        "ignored_regions": [[int(v) for v in region] for region in normalized_ignore_regions],
        "extraction_mode": extraction_mode,
        "zoom_scale": int(max(1, zoom_scale)),
        "sahi_slicing": {
            "enabled": bool(use_sahi_slicing),
            "slice_size": int(max(64, sahi_slice_size)),
            "overlap_ratio": round(clamp_overlap_ratio(sahi_overlap_ratio), 2),
            "provider": "sahi" if sahi_get_slice_bboxes is not None else "fallback",
        },
        "glare_preprocessing": {
            "enabled": bool(use_glare_removal),
            "threshold": int(max(0, min(255, glare_threshold))),
            "patch_size": int(max(1, glare_patch_size)),
            "saturation_max": int(DEFAULT_GLARE_SATURATION_MAX),
        },
        "color_correction": normalize_color_correction(color_correction),
        "analysis_normalization": normalize_analysis_normalization(
            analysis_normalization
        ),
        "background_calibration": background_calibration or None,
        "learned_stone_profiles": normalize_learned_stone_profiles(
            learned_stone_profiles
        ),
        "jewel_count": len(jewels),
        "note": "Gemstone names are color-based possibilities only, not a lab-grade identification.",
        "jewels": [],
    }
    jewel_views: list[dict] = []

    for jewel_index, jewel in enumerate(jewels, start=1):
        jewel_report, jewel_view = analyze_jewel_candidate(
            jewel_index,
            jewel,
            max(1, zoom_scale),
            use_glare_removal=bool(use_glare_removal),
            glare_threshold=max(0, min(255, int(glare_threshold))),
            glare_patch_size=max(1, int(glare_patch_size)),
            use_sahi_slicing=use_sahi_slicing,
            sahi_slice_size=max(64, int(sahi_slice_size)),
            sahi_overlap_ratio=clamp_overlap_ratio(sahi_overlap_ratio),
            color_correction=color_correction,
            background_calibration=background_calibration,
            analysis_normalization=analysis_normalization,
            measurement_scale=measurement_scale,
            learned_stone_profiles=learned_stone_profiles,
            fastsam_model=fastsam_model,
            fastsam_lock=fastsam_lock,
            stone_v2_debug_dir=(
                stone_v2_debug_dir / f"jewel_{jewel_index:02d}"
                if stone_v2_debug_dir is not None
                else None
            ),
        )
        report["jewels"].append(jewel_report)
        jewel_views.append(jewel_view)

    separation_results = [
        jewel.get("seeded_instance_separation") or {}
        for jewel in report["jewels"]
    ]
    report["seeded_instance_separation"] = {
        "enabled": True,
        "touching_cluster_count": sum(
            int(item.get("touching_cluster_count", 0))
            for item in separation_results
        ),
        "separated_region_count": sum(
            int(item.get("separated_region_count", 0))
            for item in separation_results
        ),
        "overlap_pixels_resolved": sum(
            int(item.get("overlap_pixels_resolved", 0))
            for item in separation_results
        ),
        "boundary_pixels_removed": sum(
            int(item.get("boundary_pixels_removed", 0))
            for item in separation_results
        ),
        "disconnected_pixels_removed": sum(
            int(item.get("disconnected_pixels_removed", 0))
            for item in separation_results
        ),
    }
    report["jewel_area_px_total"] = sum(
        int(jewel.get("jewel_area_px", 0))
        for jewel in report["jewels"]
    )
    report["stone_area_px_total"] = sum(
        int(jewel.get("stone_area_px", 0))
        for jewel in report["jewels"]
    )
    report["stone_seed_area_px_total"] = sum(
        int(jewel.get("stone_seed_area_px", 0))
        for jewel in report["jewels"]
    )
    report["stone_mask_area_gain"] = round(
        report["stone_area_px_total"]
        / float(report["stone_seed_area_px_total"])
        if report["stone_seed_area_px_total"] > 0
        else 1.0,
        3,
    )
    report["stone_percentage"] = round(
        report["stone_area_px_total"] / float(report["jewel_area_px_total"]) * 100.0
        if report["jewel_area_px_total"] > 0
        else 0.0,
        2,
    )
    report["area_denominator"] = "sum_of_segmented_jewel_masks"
    report["stone_surface_coverage_percent"] = report["stone_percentage"]
    report["stone_instance_count"] = sum(
        int(jewel.get("stone_instance_count", 0))
        for jewel in report["jewels"]
    )
    segmentation_method_counts: dict[str, int] = {}
    for jewel in report["jewels"]:
        for method, count in (jewel.get("segmentation_method_counts") or {}).items():
            segmentation_method_counts[str(method)] = (
                segmentation_method_counts.get(str(method), 0) + int(count)
            )
    report["segmentation_method_counts"] = segmentation_method_counts
    report["background_hole_pixels_removed"] = sum(
        int(
            (jewel.get("mask_cleanup") or {}).get(
                "background_hole_pixels_removed_analysis",
                0,
            )
        )
        for jewel in report["jewels"]
    )
    report["background_hole_count"] = sum(
        int((jewel.get("mask_cleanup") or {}).get("background_hole_count", 0))
        for jewel in report["jewels"]
    )
    report["reflection_flagged"] = any(
        bool((jewel.get("reflection") or {}).get("flagged"))
        for jewel in report["jewels"]
    )
    report["reflection_risk"] = bool(report["reflection_flagged"])
    report["risk_jewel"] = bool(report["reflection_flagged"])
    report["risk_reasons"] = (
        ["dense reflection indicates possible additional transparent/colorless gemstones"]
        if report["reflection_flagged"]
        else []
    )
    if _STONE_V2_AVAILABLE:
        report["stone_surface_risk"] = _stone_v2.calculate_stone_surface_risk(
            report["stone_percentage"],
            report["stone_instance_count"],
            reflection_risk=report["reflection_risk"],
        )
    else:
        report["stone_surface_risk"] = {
            "level": "HIGH" if report["stone_percentage"] > 40.0 else "LOW",
            "status": (
                "HIGH STONE AREA — RISK"
                if report["stone_percentage"] > 40.0
                else "LOW STONE AREA — NORMAL"
            ),
            "stone_surface_coverage_percent": report["stone_percentage"],
            "stone_instance_count": report["stone_instance_count"],
            "high_risk": report["stone_percentage"] > 40.0,
        }
    report["stone_surface_risk_level"] = report["stone_surface_risk"]["level"]
    report["stone_surface_status"] = report["stone_surface_risk"]["status"]
    if report["stone_surface_risk"].get("high_risk"):
        report["risk_jewel"] = True
        report["risk_reasons"].append("high stone surface coverage")
    measurement_results = [
        jewel.get("stone_measurements") or {}
        for jewel in report["jewels"]
        if (jewel.get("stone_measurements") or {}).get("success")
    ]
    report["stone_measurements"] = {
        "success": bool(measurement_results),
        "instance_count": sum(
            int(item.get("instance_count", 0))
            for item in measurement_results
        ),
        "estimated_total_average_ct": round(
            sum(
                float(item.get("estimated_total_average_ct", 0.0))
                for item in measurement_results
            ),
            4,
        ),
        "estimated_total_minimum_ct": round(
            sum(
                float(item.get("estimated_total_minimum_ct", 0.0))
                for item in measurement_results
            ),
            4,
        ),
        "estimated_total_maximum_ct": round(
            sum(
                float(item.get("estimated_total_maximum_ct", 0.0))
                for item in measurement_results
            ),
            4,
        ),
        "estimated_total_average_g": round(
            sum(
                float(
                    item.get(
                        "estimated_total_average_g",
                        float(item.get("estimated_total_average_ct", 0.0)) * 0.2,
                    )
                )
                for item in measurement_results
            ),
            4,
        ),
        "estimated_total_typical_g": round(
            sum(
                float(
                    item.get(
                        "estimated_total_typical_g",
                        item.get(
                            "estimated_total_average_g",
                            float(item.get("estimated_total_average_ct", 0.0)) * 0.2,
                        ),
                    )
                )
                for item in measurement_results
            ),
            4,
        ),
        "estimated_total_minimum_g": round(
            sum(
                float(
                    item.get(
                        "estimated_total_minimum_g",
                        float(item.get("estimated_total_minimum_ct", 0.0)) * 0.2,
                    )
                )
                for item in measurement_results
            ),
            4,
        ),
        "estimated_total_maximum_g": round(
            sum(
                float(
                    item.get(
                        "estimated_total_maximum_g",
                        float(item.get("estimated_total_maximum_ct", 0.0)) * 0.2,
                    )
                )
                for item in measurement_results
            ),
            4,
        ),
        "instances": [
            instance
            for item in measurement_results
            for instance in item.get("instances", [])
        ],
        "note": (
            "Face-up geometry is measured; hidden depth and material density "
            "remain uncertain."
        ),
    }

    return {
        "report": report,
        "jewel_views": jewel_views,
        "result_gallery_bgr": build_result_gallery(jewel_views),
    }


def analyze_single_image(
    image_path: Path,
    output_root: Path,
    zoom_scale: int,
    ignore_regions: list[Rect] | None = None,
    use_glare_removal: bool = DEFAULT_GLARE_REMOVAL_ENABLED,
    glare_threshold: int = DEFAULT_GLARE_THRESHOLD,
    glare_patch_size: int = DEFAULT_GLARE_PATCH_SIZE,
    use_sahi_slicing: bool = DEFAULT_SAHI_ENABLED,
    sahi_slice_size: int = DEFAULT_SAHI_SLICE_SIZE,
    sahi_overlap_ratio: float = DEFAULT_SAHI_OVERLAP,
    external_mask: np.ndarray | None = None,
) -> dict:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_image_bgr(
        image_bgr,
        source_name=str(image_path),
        zoom_scale=zoom_scale,
        ignore_regions=ignore_regions,
        use_glare_removal=use_glare_removal,
        glare_threshold=glare_threshold,
        glare_patch_size=glare_patch_size,
        use_sahi_slicing=use_sahi_slicing,
        sahi_slice_size=sahi_slice_size,
        sahi_overlap_ratio=sahi_overlap_ratio,
        external_mask=external_mask,
    )
    report = analysis["report"]

    for jewel_report, jewel_view in zip(report["jewels"], analysis["jewel_views"]):
        jewel_report["outputs"] = write_jewel_output_images(
            output_dir=output_dir,
            jewel_index=jewel_report["jewel_id"],
            masked_png=jewel_view["masked_bgra"],
            overlay_png=jewel_view["overlay_bgr"],
            color_mask_png=jewel_view["color_mask_bgr"],
        )

    report_path = output_dir / f"{image_path.stem}_gem_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def print_report(report: dict) -> None:
    print()
    print(build_report_text(report))
    if not report["jewels"]:
        if "report_path" in report:
            print(f"\nReport: {report['report_path']}")
        return

    for jewel in report["jewels"]:
        overlay_path = jewel.get("outputs", {}).get("overlay_png")
        if overlay_path:
            print(f"  Overlay: {overlay_path}")
    if "report_path" in report:
        print(f"\nReport: {report['report_path']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract clean jewels with Otsu masking, then highlight gemstone colors with HSV masks.",
    )
    parser.add_argument("inputs", nargs="+", help="Image file(s), folder(s), or glob(s) to process.")
    parser.add_argument(
        "--output",
        default="gem_hsv_outputs",
        help="Output root folder. Default: %(default)s",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=3,
        help="Zoom factor before HSV analysis. Default: %(default)s",
    )
    parser.add_argument(
        "--ignore-roi",
        action="append",
        default=[],
        type=parse_roi_argument,
        metavar="X,Y,W,H",
        help="Blank this ROI before masking and stone detection. Repeat to ignore multiple areas.",
    )
    parser.add_argument(
        "--no-glare-removal",
        action="store_false",
        dest="use_glare_removal",
        help="Disable specular glare removal on the Otsu-cleaned jewel crop before HSV and SAHI analysis.",
    )
    parser.add_argument(
        "--glare-threshold",
        type=int,
        default=DEFAULT_GLARE_THRESHOLD,
        help="Grayscale threshold used to mark bright glare pixels. Default: %(default)s",
    )
    parser.add_argument(
        "--glare-patch-size",
        type=int,
        default=DEFAULT_GLARE_PATCH_SIZE,
        help="Patch size / inpaint radius used for glare removal after zooming. Default: %(default)s",
    )
    parser.add_argument(
        "--no-sahi",
        action="store_false",
        dest="use_sahi_slicing",
        help="Disable SAHI slice-based HSV pass.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=DEFAULT_SAHI_SLICE_SIZE,
        help="SAHI slice size in pixels after zooming. Default: %(default)s",
    )
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=DEFAULT_SAHI_OVERLAP,
        help="SAHI slice overlap ratio from 0.0 to 0.49. Default: %(default)s",
    )
    return parser


def parse_roi_argument(raw_value: str) -> Rect:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must have 4 comma-separated integers: {raw_value}")
    try:
        x, y, w, h = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"ROI must have 4 comma-separated integers: {raw_value}") from exc
    return x, y, w, h


def main() -> int:
    args = build_arg_parser().parse_args()
    image_paths = expand_inputs(args.inputs)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    ignore_regions = args.ignore_roi

    for image_path in image_paths:
        report = analyze_single_image(
            image_path,
            output_root=output_root,
            zoom_scale=max(1, args.zoom),
            ignore_regions=ignore_regions,
            use_glare_removal=bool(args.use_glare_removal),
            glare_threshold=max(0, min(255, int(args.glare_threshold))),
            glare_patch_size=max(1, int(args.glare_patch_size)),
            use_sahi_slicing=bool(args.use_sahi_slicing),
            sahi_slice_size=max(64, int(args.slice_size)),
            sahi_overlap_ratio=clamp_overlap_ratio(args.slice_overlap),
        )
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
