#!/usr/bin/env python3
"""
Accurate visible-area percentage calculation for gold jewelry stone detection.

All percentages represent visible two-dimensional projected area only.
They do NOT represent stone weight, carat percentage, volume percentage,
purity, or monetary value.

Integration order (must be called before visualization):
    capture original image
        -> generate and clean the jewelry mask
        -> detect stones and produce filled masks per color class
        -> clean small false-positive components
        -> create stone_union_mask  (logical OR of all color masks)
        -> clip stone_union_mask to the cleaned jewelry mask
        -> calculate_stone_area_statistics(...)   <-- call here
        -> store results
        -> draw contours and visualization
        -> display or return result

Do NOT call calculate_stone_area_statistics on the post-visualization
image because contour lines, legends, and fill colors alter pixel counts.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum connected-component area (pixels) below which an isolated region
# is treated as false-positive noise and removed before area calculation.
# Raise this value to be more aggressive about noise removal.
MIN_STONE_COMPONENT_AREA_PX: int = 5
MAX_REPORTED_STONE_WEIGHT_RANGE_G: float = 3.0
MAX_REPORTED_STONE_WEIGHT_RANGE_RATIO: float = 0.30
STONE_SETTING_PROFILE_FRONT_ONLY = "front_only_shallow"
STONE_SETTING_PROFILE_OPEN_BACK = "open_back_faceted"
STONE_SETTING_PROFILE_UNKNOWN = "unknown"
DEFAULT_STONE_SETTING_PROFILE = STONE_SETTING_PROFILE_FRONT_ONLY
FRONT_ONLY_AREAL_MASS_G_PER_MM2: float = 0.001695
FRONT_ONLY_WEIGHT_UNCERTAINTY_RATIO: float = 0.15
FRONT_ONLY_CALIBRATION_SAMPLE_COUNT: int = 1
STONE_WEIGHT_MAX_JEWEL_SHARE: float = 1.0
STONE_MATERIAL_PROFILES: dict[str, dict[str, float]] = {
    "lightweight_imitation": {"density_min_g_cm3": 1.8, "density_max_g_cm3": 2.6},
    "glass_like": {"density_min_g_cm3": 2.3, "density_max_g_cm3": 3.0},
    "natural_gem_general": {"density_min_g_cm3": 2.6, "density_max_g_cm3": 4.3},
}
STONE_UNKNOWN_DENSITY_RANGE_G_CM3 = (1.8, 3.1, 4.3)
STONE_SHAPE_DEPTH_PROFILES: dict[str, tuple[float, float, float, float]] = {
    # minimum, typical and maximum depth/minor-axis ratios, then volume factor
    "round": (0.32, 0.52, 0.72, math.pi / 6.0),
    "oval": (0.25, 0.45, 0.66, math.pi / 6.0),
    "rectangular": (0.22, 0.42, 0.62, 0.55),
    "pear": (0.24, 0.45, 0.68, 0.48),
    "irregular": (0.18, 0.38, 0.65, 0.42),
}

# Reflection-risk thresholds. A jewel is flagged when reflections cover a
# meaningful share of the jewel, or when a smaller but sufficiently large
# local hotspot is extremely dense (common with transparent/colorless stones).
REFLECTION_COVERAGE_RISK_PERCENT: float = 1.50
REFLECTION_MIXED_COVERAGE_MIN_PERCENT: float = 0.55
REFLECTION_LOCAL_DENSITY_RISK_PERCENT: float = 18.0
REFLECTION_LOCAL_HOTSPOT_RISK_PERCENT: float = 32.0
REFLECTION_MIN_HOTSPOT_PIXELS: int = 12

# Set True to apply a single morphological closing pass (3×3 elliptical
# kernel, 1 iteration) to each stone mask before area calculation.
# Useful only when the existing masks contain small internal gaps.
# Disabled by default because it slightly inflates area measurements.
STONE_MASK_CLOSE_GAPS: bool = False

_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Business-average face-up weight tables supplied for this project. These are
# estimates only; actual weight depends on depth, profile, material density,
# and cut proportions.
ROUND_WEIGHT_TABLE = [
    (1.0, 0.00575, 0.004, 0.008),
    (2.0, 0.0400, 0.030, 0.060),
    (3.0, 0.1250, 0.100, 0.170),
    (4.0, 0.3125, 0.250, 0.400),
    (5.0, 0.6425, 0.500, 0.850),
    (6.0, 1.0825, 0.850, 1.400),
    (7.0, 1.6625, 1.250, 2.200),
    (8.0, 2.4625, 1.900, 3.300),
]

OVAL_WEIGHT_TABLE = [
    (5.0, 3.0, 0.2833, 0.250, 0.350),
    (6.0, 4.0, 0.5400, 0.500, 0.600),
    (7.0, 5.0, 0.8867, 0.750, 1.060),
    (8.0, 6.0, 1.4367, 1.260, 1.550),
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def to_binary_mask(mask: Any) -> np.ndarray:
    """
    Validate and convert any mask representation to a single-channel
    uint8 binary mask (0 = background, 255 = foreground).

    Handles: None check, 3-channel → single channel, float, bool,
    any non-zero value → 255.

    Raises:
        ValueError: if mask is None or has unexpected dimensions.
    """
    if mask is None:
        raise ValueError("to_binary_mask: mask must not be None")

    arr = np.asarray(mask)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if arr.ndim != 2:
        raise ValueError(
            f"to_binary_mask: expected 2-D mask, got shape {arr.shape}"
        )

    return np.where(arr > 0, np.uint8(255), np.uint8(0))


def remove_small_components(
    mask: np.ndarray,
    min_area_pixels: int = MIN_STONE_COMPONENT_AREA_PX,
) -> np.ndarray:
    """
    Remove isolated connected components whose pixel area < min_area_pixels.

    Uses cv2.connectedComponentsWithStats with 8-connectivity for efficient
    component analysis without iterating over individual pixels.

    Returns:
        Clean binary uint8 mask with small noise components zeroed out.
    """
    if not np.any(mask):
        return mask.copy()

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    clean = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= max(1, int(min_area_pixels)):
            clean[labels == label] = 255

    return clean


def build_filled_mask_from_contours(
    image_shape: tuple[int, int],
    contours: list,
) -> np.ndarray:
    """
    Convert stone contour detections into a filled binary mask.

    Uses cv2.FILLED thickness so the complete interior of each contour
    is included — not just the contour boundary pixels.

    Args:
        image_shape: (height, width) of the target mask.
        contours:    list of OpenCV contours (Nx1x2 int32 arrays).

    Returns:
        uint8 mask with interior of each contour filled to 255.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(
            mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED
        )
    return mask


def _interpolate_round_weight(diameter_mm: float) -> dict[str, float]:
    """Linearly interpolate the supplied round-stone weight table."""
    diameter = max(0.0, float(diameter_mm))
    lower = ROUND_WEIGHT_TABLE[0]
    upper = ROUND_WEIGHT_TABLE[-1]
    if diameter <= lower[0]:
        scale = (diameter / lower[0]) ** 3 if lower[0] > 0 else 0.0
        values = [lower[index] * scale for index in range(1, 4)]
    elif diameter >= upper[0]:
        scale = (diameter / upper[0]) ** 3
        values = [upper[index] * scale for index in range(1, 4)]
    else:
        values = []
        for left, right in zip(ROUND_WEIGHT_TABLE, ROUND_WEIGHT_TABLE[1:]):
            if left[0] <= diameter <= right[0]:
                fraction = (diameter - left[0]) / (right[0] - left[0])
                values = [
                    left[index] + fraction * (right[index] - left[index])
                    for index in range(1, 4)
                ]
                break
    average, minimum, maximum = values
    return {
        "average_ct": round(max(0.0, average), 5),
        "minimum_ct": round(max(0.0, minimum), 5),
        "maximum_ct": round(max(0.0, maximum), 5),
        "method": "round face-up size table",
    }


def _estimate_oval_weight(major_mm: float, minor_mm: float) -> dict[str, float]:
    """Use the nearest supplied oval size, scaled by face-up area."""
    major = max(float(major_mm), float(minor_mm))
    minor = min(float(major_mm), float(minor_mm))
    reference = min(
        OVAL_WEIGHT_TABLE,
        key=lambda row: math.hypot(major - row[0], minor - row[1]),
    )
    reference_area = max(0.01, reference[0] * reference[1])
    area_scale = max(0.05, (major * minor) / reference_area)
    average = reference[2] * area_scale
    minimum = reference[3] * area_scale
    maximum = reference[4] * area_scale
    return {
        "average_ct": round(average, 5),
        "minimum_ct": round(minimum, 5),
        "maximum_ct": round(maximum, 5),
        "method": (
            f"oval face-up size table scaled from "
            f"{reference[0]:g}x{reference[1]:g} mm"
        ),
    }


def _ellipse_residual(contour: np.ndarray) -> float | None:
    if contour is None or len(contour) < 5:
        return None
    try:
        (cx, cy), (axis_a, axis_b), angle_deg = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    if axis_a <= 0 or axis_b <= 0:
        return None
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points = contour.reshape(-1, 2).astype(np.float64)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    radius = np.sqrt(
        (x_rot / max(axis_a / 2.0, 1e-6)) ** 2
        + (y_rot / max(axis_b / 2.0, 1e-6)) ** 2
    )
    return float(np.mean(np.abs(radius - 1.0)))


def _pear_asymmetry(mask: np.ndarray, contour: np.ndarray) -> float:
    """Estimate narrow-end/broad-end asymmetry along the major rectangle axis."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    width, height = rect[1]
    if width <= 1 or height <= 1:
        return 0.0
    out_w = max(8, int(round(max(width, height))))
    out_h = max(6, int(round(min(width, height))))
    ordered = box[np.argsort(box[:, 1])]
    top = ordered[:2][np.argsort(ordered[:2, 0])]
    bottom = ordered[2:][np.argsort(ordered[2:, 0])]
    source = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    target = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    warped = cv2.warpPerspective(mask, cv2.getPerspectiveTransform(source, target), (out_w, out_h))
    if out_w < out_h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    profile = np.count_nonzero(warped > 0, axis=0).astype(np.float32)
    quarter = max(1, profile.size // 4)
    left = float(profile[:quarter].mean())
    right = float(profile[-quarter:].mean())
    return abs(left - right) / max(left, right, 1.0)


def _classify_shape(
    contour: np.ndarray,
    mask: np.ndarray,
    aspect_ratio: float,
    circularity: float,
    rectangularity: float,
    ellipse_residual: float | None,
) -> str:
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    if (
        4 <= len(polygon) <= 6
        and rectangularity >= 0.78
        and circularity < 0.86
    ):
        return "rectangular"
    pear_asymmetry = _pear_asymmetry(mask, contour)
    if pear_asymmetry >= 0.28 and 1.15 <= aspect_ratio <= 2.3:
        return "pear"
    if aspect_ratio <= 1.15 and circularity >= 0.68:
        return "round"
    if aspect_ratio > 1.15 and ellipse_residual is not None and ellipse_residual <= 0.22:
        return "oval"
    return "irregular"


def calculate_stone_measurements(
    stone_masks: dict[str, np.ndarray],
    mm_per_pixel_x: float,
    mm_per_pixel_y: float,
    min_component_area_pixels: int = MIN_STONE_COMPONENT_AREA_PX,
) -> dict[str, Any]:
    """
    Measure each connected stone mask in millimetres and estimate face-up weight.

    The weight result is deliberately labelled as an estimate. It is based on
    the supplied average face-up size tables, not gemstone depth or density.
    """
    scale_x = float(mm_per_pixel_x)
    scale_y = float(mm_per_pixel_y)
    if scale_x <= 0 or scale_y <= 0:
        return {
            "success": False,
            "error": "Valid mm-per-pixel X and Y scales are required.",
            "instances": [],
            "estimated_total_average_ct": 0.0,
            "estimated_total_minimum_ct": 0.0,
            "estimated_total_maximum_ct": 0.0,
        }

    instances: list[dict[str, Any]] = []
    instance_id = 0
    for color_name, raw_mask in stone_masks.items():
        try:
            mask = remove_small_components(
                to_binary_mask(raw_mask),
                min_component_area_pixels,
            )
        except Exception:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, count):
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if area_px < max(1, int(min_component_area_pixels)):
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour_area_px = max(1.0, float(cv2.contourArea(contour)))
            physical_contour = contour.astype(np.float32).copy()
            physical_contour[:, 0, 0] *= scale_x
            physical_contour[:, 0, 1] *= scale_y
            perimeter_mm = float(cv2.arcLength(physical_contour, True))
            contour_area_mm2 = max(
                1e-6,
                float(cv2.contourArea(physical_contour)),
            )
            rect = cv2.minAreaRect(physical_contour)
            rect_w_mm, rect_h_mm = rect[1]
            major_mm = max(rect_w_mm, rect_h_mm)
            minor_mm = min(rect_w_mm, rect_h_mm)
            area_mm2 = area_px * scale_x * scale_y
            equivalent_diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
            aspect_ratio = major_mm / max(minor_mm, 1e-6)
            circularity = (
                4.0 * math.pi * contour_area_mm2 / (perimeter_mm ** 2)
                if perimeter_mm > 0
                else 0.0
            )
            hull = cv2.convexHull(physical_contour)
            hull_area = max(1e-6, float(cv2.contourArea(hull)))
            solidity = contour_area_mm2 / hull_area
            rect_area = max(1e-6, rect_w_mm * rect_h_mm)
            rectangularity = contour_area_mm2 / rect_area
            ellipse_residual = _ellipse_residual(contour)
            shape = _classify_shape(
                contour,
                component,
                aspect_ratio,
                circularity,
                rectangularity,
                ellipse_residual,
            )
            if shape == "oval":
                weight = _estimate_oval_weight(major_mm, minor_mm)
            else:
                weight = _interpolate_round_weight(equivalent_diameter_mm)
                if shape != "round":
                    weight["method"] = (
                        f"equivalent-round approximation for {shape} face-up mask"
                    )
            weight["average_g"] = round(weight["average_ct"] * 0.2, 5)
            weight["minimum_g"] = round(weight["minimum_ct"] * 0.2, 5)
            weight["maximum_g"] = round(weight["maximum_ct"] * 0.2, 5)

            instance_id += 1
            instances.append(
                {
                    "instance_id": instance_id,
                    "color": str(color_name),
                    "shape": shape,
                    "area_pixels": area_px,
                    "area_mm2": round(area_mm2, 4),
                    "equivalent_diameter_mm": round(equivalent_diameter_mm, 3),
                    "major_axis_mm": round(major_mm, 3),
                    "minor_axis_mm": round(minor_mm, 3),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "perimeter_mm": round(perimeter_mm, 3),
                    "circularity": round(circularity, 3),
                    "solidity": round(solidity, 3),
                    "rectangularity": round(rectangularity, 3),
                    "ellipse_residual": (
                        round(ellipse_residual, 4)
                        if ellipse_residual is not None
                        else None
                    ),
                    "estimated_weight": weight,
                }
            )

    total_average = sum(item["estimated_weight"]["average_ct"] for item in instances)
    total_minimum = sum(item["estimated_weight"]["minimum_ct"] for item in instances)
    total_maximum = sum(item["estimated_weight"]["maximum_ct"] for item in instances)
    return {
        "success": True,
        "scale_mm_per_pixel_x": scale_x,
        "scale_mm_per_pixel_y": scale_y,
        "instance_count": len(instances),
        "instances": instances,
        "estimated_total_average_ct": round(total_average, 4),
        "estimated_total_minimum_ct": round(total_minimum, 4),
        "estimated_total_maximum_ct": round(total_maximum, 4),
        "estimated_total_average_g": round(total_average * 0.2, 4),
        "estimated_total_minimum_g": round(total_minimum * 0.2, 4),
        "estimated_total_maximum_g": round(total_maximum * 0.2, 4),
        "note": (
            "Estimated average weight from face-up size only; depth, cut profile, "
            "and gemstone density are not measured."
        ),
    }


def estimate_stone_weight_range(
    measurement_result: dict[str, Any],
    setting_profile: Any,
    jewel_weight_g: float | None = None,
) -> dict[str, Any]:
    """Estimate a broad mass range from face-up geometry.

    The image supplies length, width and area.  Hidden depth and material
    density remain explicit uncertainty inputs; visible color never selects a
    mineral or density profile.
    """
    profile = normalize_stone_setting_profile(setting_profile)
    instances = list(measurement_result.get("instances") or [])
    if not measurement_result.get("success") or not instances:
        return {
            **measurement_result,
            "success": bool(measurement_result.get("success")) and not instances,
            "stone_setting_profile": profile,
            "weight_estimate_suppressed": False,
            "estimated_total_typical_g": 0.0,
            "weight_confidence": "Low",
            "weight_confidence_score": 0.0,
            "weight_method": "face-up geometry with depth and density uncertainty",
            "weight_warnings": ["No measurable final stone instances were available."],
        }

    density_min, density_typical, density_max = STONE_UNKNOWN_DENSITY_RANGE_G_CM3
    depth_scale = 0.68 if profile == STONE_SETTING_PROFILE_FRONT_ONLY else 1.0
    minimum_total = 0.0
    typical_total = 0.0
    maximum_total = 0.0
    irregular_count = 0
    implausibly_large_count = 0
    enriched_instances: list[dict[str, Any]] = []
    for instance in instances:
        shape = str(instance.get("shape") or "irregular")
        depth_min, depth_typical, depth_max, shape_factor = (
            STONE_SHAPE_DEPTH_PROFILES.get(
                shape,
                STONE_SHAPE_DEPTH_PROFILES["irregular"],
            )
        )
        if shape == "irregular":
            irregular_count += 1
        major = max(0.0, float(instance.get("major_axis_mm", 0.0)))
        minor = max(0.0, float(instance.get("minor_axis_mm", 0.0)))
        if major <= 0.0 or minor <= 0.0:
            continue
        if major > 30.0 or minor > 25.0:
            implausibly_large_count += 1
        minimum_depth = minor * depth_min * depth_scale
        typical_depth = minor * depth_typical * depth_scale
        maximum_depth = minor * depth_max * depth_scale
        minimum_volume = major * minor * minimum_depth * shape_factor
        typical_volume = major * minor * typical_depth * shape_factor
        maximum_volume = major * minor * maximum_depth * shape_factor
        minimum_g = minimum_volume / 1000.0 * density_min
        typical_g = typical_volume / 1000.0 * density_typical
        maximum_g = maximum_volume / 1000.0 * density_max
        minimum_total += minimum_g
        typical_total += typical_g
        maximum_total += maximum_g
        enriched = dict(instance)
        enriched["estimated_depth_range_mm"] = [
            round(minimum_depth, 3),
            round(typical_depth, 3),
            round(maximum_depth, 3),
        ]
        enriched["estimated_weight_range_g"] = [
            round(minimum_g, 5),
            round(typical_g, 5),
            round(maximum_g, 5),
        ]
        enriched_instances.append(enriched)

    warnings = [
        "Stone depth is not directly visible in the captured image.",
        "Gemstone material and density are not identified from visible color.",
    ]
    confidence_score = 0.68
    if profile == STONE_SETTING_PROFILE_FRONT_ONLY:
        warnings.append(
            "Front-only setting uses a shallower depth profile because the stone back is hidden."
        )
        confidence_score -= 0.08
    if irregular_count:
        warnings.append(
            f"{irregular_count} irregular instance(s) have wider shape uncertainty."
        )
        confidence_score -= min(0.18, irregular_count / float(len(instances)) * 0.18)
    if implausibly_large_count:
        warnings.append(
            "One or more very large regions may contain merged stones or residual metal."
        )
        confidence_score -= 0.25
    confidence_score = max(0.10, min(0.90, confidence_score))
    confidence = "High" if confidence_score >= 0.78 else "Medium" if confidence_score >= 0.50 else "Low"
    result = {
        **measurement_result,
        "success": True,
        "instances": enriched_instances,
        "stone_setting_profile": profile,
        "weight_model": "geometry_depth_density_range",
        "weight_method": "face-up geometry with depth and density uncertainty",
        "v2_geometry_estimate": True,
        "material_profile": "unknown_general_range",
        "density_range_g_cm3": [density_min, density_typical, density_max],
        "estimated_total_minimum_g": round(max(0.0, minimum_total), 4),
        "estimated_total_typical_g": round(max(0.0, typical_total), 4),
        "estimated_total_average_g": round(max(0.0, typical_total), 4),
        "estimated_total_maximum_g": round(max(0.0, maximum_total), 4),
        "estimated_total_minimum_ct": round(max(0.0, minimum_total) * 5.0, 4),
        "estimated_total_average_ct": round(max(0.0, typical_total) * 5.0, 4),
        "estimated_total_maximum_ct": round(max(0.0, maximum_total) * 5.0, 4),
        "weight_confidence": confidence,
        "weight_confidence_score": round(confidence_score, 3),
        "weight_warnings": warnings,
        "note": (
            "Approximate image-based range; hidden depth, cut and material density "
            "are not measured."
        ),
    }
    return calibrate_weight_estimate_to_jewel_weight(result, jewel_weight_g)


def calibrate_weight_estimate_to_jewel_weight(
    weight_estimate: dict[str, Any],
    jewel_weight_g: float | None,
) -> dict[str, Any]:
    """Constrain an estimated stone-weight range to the complete jewel weight."""
    calibrated = dict(weight_estimate)
    entered_weight = (
        float(jewel_weight_g)
        if jewel_weight_g is not None and float(jewel_weight_g) > 0
        else None
    )
    calibrated["entered_jewel_weight_g"] = entered_weight
    if not calibrated.get("success") or entered_weight is None:
        return calibrated

    raw_values: dict[str, float] = {}
    for name in ("minimum", "average", "maximum"):
        raw_key = f"raw_estimated_total_{name}_g"
        raw_values[name] = max(
            0.0,
            float(
                calibrated.get(
                    raw_key,
                    calibrated.get(
                        f"estimated_total_{name}_g",
                        float(calibrated.get(f"estimated_total_{name}_ct", 0.0)) * 0.2,
                    ),
                )
            ),
        )

    raw_maximum = raw_values["maximum"]
    physical_upper_bound = entered_weight * STONE_WEIGHT_MAX_JEWEL_SHARE
    calibration_factor = (
        min(1.0, physical_upper_bound / raw_maximum)
        if raw_maximum > 0
        else 1.0
    )
    calibration_applied = calibration_factor < 1.0
    bounded_values = {
        name: raw_value_g * calibration_factor
        for name, raw_value_g in raw_values.items()
    }
    original_span_g = max(
        0.0,
        bounded_values["maximum"] - bounded_values["minimum"],
    )
    target_span_g = (
        original_span_g
        if calibrated.get("v2_geometry_estimate")
        else min(
            original_span_g,
            MAX_REPORTED_STONE_WEIGHT_RANGE_G,
            bounded_values["average"] * MAX_REPORTED_STONE_WEIGHT_RANGE_RATIO,
        )
    )
    range_narrowing_applied = target_span_g < original_span_g
    if range_narrowing_applied:
        narrowed_minimum = bounded_values["average"] - target_span_g / 2.0
        narrowed_maximum = bounded_values["average"] + target_span_g / 2.0
        if narrowed_minimum < bounded_values["minimum"]:
            narrowed_maximum += bounded_values["minimum"] - narrowed_minimum
            narrowed_minimum = bounded_values["minimum"]
        if narrowed_maximum > bounded_values["maximum"]:
            narrowed_minimum -= narrowed_maximum - bounded_values["maximum"]
            narrowed_maximum = bounded_values["maximum"]
        bounded_values["minimum"] = max(0.0, narrowed_minimum)
        bounded_values["maximum"] = min(entered_weight, narrowed_maximum)

    if calibration_applied or range_narrowing_applied:
        for name, value_g in raw_values.items():
            calibrated[f"raw_estimated_total_{name}_g"] = round(value_g, 4)
            calibrated[f"raw_estimated_total_{name}_ct"] = round(value_g * 5.0, 4)

    for name, value_g in bounded_values.items():
        calibrated[f"estimated_total_{name}_g"] = round(value_g, 4)
        calibrated[f"estimated_total_{name}_ct"] = round(value_g * 5.0, 4)
    calibrated["estimated_total_typical_g"] = calibrated[
        "estimated_total_average_g"
    ]

    calibrated["calibration_applied"] = calibration_applied
    calibrated["range_narrowing_applied"] = range_narrowing_applied
    calibrated["jewel_weight_calibration_factor"] = round(calibration_factor, 6)
    calibrated["reported_range_span_g"] = round(
        calibrated["estimated_total_maximum_g"]
        - calibrated["estimated_total_minimum_g"],
        4,
    )
    calibrated["reported_range_maximum_span_g"] = MAX_REPORTED_STONE_WEIGHT_RANGE_G
    calibrated["estimated_stone_share_of_jewel_weight_percent"] = round(
        calibrated["estimated_total_average_g"] / entered_weight * 100.0,
        2,
    )
    calibrated["estimated_stone_share_range_percent"] = [
        round(calibrated["estimated_total_minimum_g"] / entered_weight * 100.0, 2),
        round(calibrated["estimated_total_maximum_g"] / entered_weight * 100.0, 2),
    ]
    calibrated["reference_check"] = (
        "calibrated_to_jewel_weight" if calibration_applied else "plausible"
    )
    warnings = list(calibrated.get("weight_warnings") or [])
    if calibration_applied:
        warnings.append(
            "Visible stone geometry conflicted with the complete OCR jewel weight; "
            "the raw range is retained and the reported range is safety-bounded."
        )
        calibrated["weight_confidence_score"] = round(
            max(0.05, float(calibrated.get("weight_confidence_score", 0.5)) - 0.25),
            3,
        )
        calibrated["weight_confidence"] = (
            "Low"
            if calibrated["weight_confidence_score"] < 0.50
            else "Medium"
        )
    calibrated["weight_warnings"] = warnings
    calibrated["sanity_constraint_applied"] = calibration_applied
    calibrated["stone_weight_max_jewel_share"] = STONE_WEIGHT_MAX_JEWEL_SHARE
    reference_notes = [
        (
            "The captured OCR jewel weight is a hard upper bound. The raw stone-weight "
            "range was proportionally scaled because its maximum exceeded that bound."
            if calibration_applied
            else "The captured OCR jewel weight was used as a hard upper-bound check."
        )
    ]
    if range_narrowing_applied:
        reference_notes.append(
            "The reported range was narrowed around the average estimate to a maximum "
            "of 3.00 g or 30% of the average estimate, whichever is smaller."
        )
    calibrated["reference_note"] = " ".join(reference_notes)
    return calibrated


def normalize_stone_setting_profile(value: Any) -> str:
    profile = str(value or "").strip().lower()
    if profile in {
        STONE_SETTING_PROFILE_FRONT_ONLY,
        STONE_SETTING_PROFILE_OPEN_BACK,
        STONE_SETTING_PROFILE_UNKNOWN,
    }:
        return profile
    return DEFAULT_STONE_SETTING_PROFILE


def apply_stone_setting_weight_model(
    face_up_estimate: dict[str, Any],
    setting_profile: Any,
    visible_stone_area_mm2: float | None,
    jewel_weight_g: float | None,
) -> dict[str, Any]:
    """Choose the weight model appropriate for the observed stone setting."""
    profile = normalize_stone_setting_profile(setting_profile)
    visible_area = (
        max(0.0, float(visible_stone_area_mm2))
        if visible_stone_area_mm2 is not None
        else None
    )

    if profile == STONE_SETTING_PROFILE_UNKNOWN:
        return {
            "success": False,
            "weight_estimate_suppressed": True,
            "stone_setting_profile": profile,
            "weight_model": "visible_area_only",
            "visible_stone_area_mm2": (
                round(visible_area, 4) if visible_area is not None else None
            ),
            "entered_jewel_weight_g": jewel_weight_g,
            "error": (
                "Stone weight is not estimated until the setting type is known."
            ),
        }

    if face_up_estimate.get("success") and face_up_estimate.get("instances"):
        estimated = estimate_stone_weight_range(
            face_up_estimate,
            profile,
            jewel_weight_g,
        )
        estimated["visible_stone_area_mm2"] = (
            round(visible_area, 4) if visible_area is not None else None
        )
        return estimated

    if profile == STONE_SETTING_PROFILE_OPEN_BACK:
        estimated = calibrate_weight_estimate_to_jewel_weight(
            face_up_estimate,
            jewel_weight_g,
        )
        estimated["stone_setting_profile"] = profile
        estimated["weight_model"] = "face_up_size_table_fallback"
        estimated["weight_method"] = "legacy face-up size-table fallback"
        estimated["weight_confidence"] = "Low"
        estimated["weight_confidence_score"] = 0.30
        estimated["weight_warnings"] = [
            "Per-instance geometry was unavailable; legacy face-up tables were used."
        ]
        estimated["visible_stone_area_mm2"] = (
            round(visible_area, 4) if visible_area is not None else None
        )
        return estimated

    if visible_area is None:
        return {
            "success": False,
            "stone_setting_profile": profile,
            "weight_model": "front_only_areal_calibration",
            "weight_method": "front-only visible-area calibration fallback",
            "weight_confidence": "Low",
            "weight_confidence_score": 0.30,
            "weight_warnings": [
                "Per-instance geometry was unavailable; provisional visible-area calibration was used."
            ],
            "visible_stone_area_mm2": None,
            "entered_jewel_weight_g": jewel_weight_g,
            "error": "Metric calibration is required for front-only stone weight.",
        }

    average_g = visible_area * FRONT_ONLY_AREAL_MASS_G_PER_MM2
    minimum_g = average_g * (1.0 - FRONT_ONLY_WEIGHT_UNCERTAINTY_RATIO)
    maximum_g = average_g * (1.0 + FRONT_ONLY_WEIGHT_UNCERTAINTY_RATIO)
    estimated = calibrate_weight_estimate_to_jewel_weight(
        {
            "success": True,
            "estimated_total_average_g": round(average_g, 4),
            "estimated_total_minimum_g": round(minimum_g, 4),
            "estimated_total_maximum_g": round(maximum_g, 4),
            "estimated_total_average_ct": round(average_g * 5.0, 4),
            "estimated_total_minimum_ct": round(minimum_g * 5.0, 4),
            "estimated_total_maximum_ct": round(maximum_g * 5.0, 4),
            "weight_method": "front-only visible-area calibration fallback",
            "weight_confidence": "Low",
            "weight_confidence_score": 0.30,
            "weight_warnings": [
                "Per-instance geometry was unavailable; provisional visible-area calibration was used."
            ],
        },
        jewel_weight_g,
    )
    estimated.update(
        {
            "stone_setting_profile": profile,
            "weight_model": "front_only_areal_calibration",
            "visible_stone_area_mm2": round(visible_area, 4),
            "calibration_g_per_mm2": FRONT_ONLY_AREAL_MASS_G_PER_MM2,
            "calibration_sample_count": FRONT_ONLY_CALIBRATION_SAMPLE_COUNT,
            "provisional_calibration": True,
            "uncertainty_percent": round(
                FRONT_ONLY_WEIGHT_UNCERTAINTY_RATIO * 100.0,
                1,
            ),
            "note": (
                "Front-only shallow-stone estimate from visible area using one "
                "physical peeled-stone calibration sample."
            ),
        }
    )
    return estimated


# ---------------------------------------------------------------------------
# Reflection risk calculation
# ---------------------------------------------------------------------------

def calculate_reflection_risk(
    glare_mask: np.ndarray,
    jewel_mask: np.ndarray,
    coverage_risk_percent: float = REFLECTION_COVERAGE_RISK_PERCENT,
    local_density_risk_percent: float = REFLECTION_LOCAL_DENSITY_RISK_PERCENT,
    local_hotspot_risk_percent: float = REFLECTION_LOCAL_HOTSPOT_RISK_PERCENT,
) -> dict[str, Any]:
    """
    Measure dense specular reflection inside a jewelry mask.

    Dense reflection is not proof of a gemstone. It is treated as a risk
    indicator because transparent/colorless stones often produce concentrated
    highlights that color-based stone detection can miss.
    """
    glare_bin = to_binary_mask(glare_mask)
    jewel_bin = to_binary_mask(jewel_mask)
    if glare_bin.shape != jewel_bin.shape:
        raise ValueError(
            "calculate_reflection_risk: glare_mask and jewel_mask must have "
            f"the same shape, got {glare_bin.shape} and {jewel_bin.shape}"
        )

    glare_bin = cv2.bitwise_and(glare_bin, jewel_bin)
    jewel_px = int(np.count_nonzero(jewel_bin))
    glare_px = int(np.count_nonzero(glare_bin))
    if jewel_px <= 0 or glare_px <= 0:
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

    binary_glare = (glare_bin > 0).astype(np.float32)
    binary_jewel = (jewel_bin > 0).astype(np.float32)
    height, width = glare_bin.shape[:2]
    window = max(9, min(41, int(round(min(height, width) * 0.12))))
    if window % 2 == 0:
        window += 1
    local_glare = cv2.boxFilter(
        binary_glare, -1, (window, window), normalize=False
    )
    local_jewel = cv2.boxFilter(
        binary_jewel, -1, (window, window), normalize=False
    )
    local_density = np.divide(
        local_glare,
        np.maximum(local_jewel, 1.0),
        out=np.zeros_like(local_glare),
        where=local_jewel > 0,
    )
    local_density_percent = float(local_density.max() * 100.0)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (glare_bin > 0).astype(np.uint8),
        connectivity=8,
    )
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 3
    ]
    largest_region_px = max(component_areas, default=0)
    coverage_percent = glare_px / float(jewel_px) * 100.0
    largest_region_percent = largest_region_px / float(jewel_px) * 100.0

    min_hotspot_px = max(
        REFLECTION_MIN_HOTSPOT_PIXELS,
        int(round(jewel_px * 0.0003)),
    )
    min_connected_hotspot_px = max(6, int(round(jewel_px * 0.0001)))
    substantial_hotspot = bool(
        glare_px >= min_hotspot_px
        and largest_region_px >= min_connected_hotspot_px
    )

    coverage_trigger = coverage_percent >= float(coverage_risk_percent)
    mixed_trigger = bool(
        coverage_percent >= REFLECTION_MIXED_COVERAGE_MIN_PERCENT
        and local_density_percent >= float(local_density_risk_percent)
    )
    local_hotspot_trigger = bool(
        substantial_hotspot
        and local_density_percent >= float(local_hotspot_risk_percent)
    )
    flagged = bool(coverage_trigger or mixed_trigger or local_hotspot_trigger)

    if coverage_percent >= 3.0 or local_hotspot_trigger:
        level = "high"
    elif flagged:
        level = "elevated"
    else:
        level = "normal"

    message = (
        "Dense reflection detected; possible additional transparent/colorless gemstones "
        "may be present. Treat this jewel as RISK."
        if flagged
        else "Reflection is present but not dense enough to classify the jewel as RISK."
    )
    return {
        "coverage_percent": round(coverage_percent, 2),
        "local_density_percent": round(local_density_percent, 2),
        "region_count": len(component_areas),
        "largest_region_px": largest_region_px,
        "largest_region_percent": round(largest_region_percent, 2),
        "flagged": flagged,
        "risk_status": "RISK" if flagged else "NORMAL",
        "possible_transparent_stones": flagged,
        "level": level,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calculate_stone_area_statistics(
    jewel_mask: np.ndarray,
    stone_masks: dict[str, np.ndarray],
    min_component_area_pixels: int = MIN_STONE_COMPONENT_AREA_PX,
    valid_roi: np.ndarray | None = None,
    include_stones_outside_jewel: bool = False,
) -> dict[str, Any]:
    """
    Calculate visible projected-area percentages for stones and metal.

    MATHEMATICAL DEFINITION
    -----------------------
    By default, every stone mask is clipped to the supplied jewel mask:

    complete_jewel_mask  = jewel_mask
    stone_union_mask     = stone_union_mask AND jewel_mask

    This guarantees that the denominator is the segmented visible jewel area,
    never the full crop/image, and prevents background false positives from
    expanding both the stone count and the denominator.

    For legacy callers that intentionally provide stone masks outside an
    incomplete jewel mask, include_stones_outside_jewel=True restores:

    complete_jewel_mask  = jewel_mask OR stone_union_mask

    jewel_area_pixels    = count_nonzero(complete_jewel_mask)
    stone_area_pixels    = count_nonzero(stone_union_mask)
    metal_area_pixels    = jewel_area_pixels - stone_area_pixels
    stone_percentage     = stone_area_pixels / jewel_area_pixels * 100
    metal_percentage     = metal_area_pixels / jewel_area_pixels * 100

    For each stone color:
        percentage_of_jewel  = color_area / jewel_area_pixels * 100
        percentage_of_stones = color_area / stone_area_pixels * 100

    MASK REQUIREMENTS
    -----------------
    All masks must share the same height and width.
    Background = 0, foreground = any non-zero value.
    All masks are converted to binary uint8 internally.

    The necklace interior and small hollow regions are excluded when they are
    zero in the supplied cleaned jewel mask.

    Args:
        jewel_mask:               Otsu binary mask of the complete jewelry.
        stone_masks:              Dict mapping color label → binary stone mask.
                                  Example keys: "White/Colorless", "Pink", etc.
        min_component_area_pixels: Components below this area are treated as
                                   noise and removed before calculation.
        valid_roi:                Optional mask to restrict the analysis area.
        include_stones_outside_jewel:
                                   Legacy opt-in that allows stone masks to
                                   expand the jewel denominator. False by
                                   default for strict mask-based percentages.

    Returns:
        dict with keys:
            success               bool
            jewel_area_pixels     int
            stone_area_pixels     int
            metal_area_pixels     int
            stone_percentage      float  (0–100)
            metal_percentage      float  (0–100)
            overlapping_class_pixels  int
            colors                dict[color_name → {
                                      area_pixels        int,
                                      percentage_of_jewel   float,
                                      percentage_of_stones  float,
                                      component_count    int,
                                  }]
        On empty jewel mask:
            success = False, error = str, all area values = 0
    """
    # -- Validate and convert Otsu mask ---------------------------------
    try:
        otsu_bin = to_binary_mask(jewel_mask)
    except Exception as exc:
        logger.error("[StoneArea] Invalid jewel_mask: %s", exc)
        return _empty_failure(f"Invalid jewel_mask: {exc}")

    # Optional ROI restriction is applied to the denominator first so every
    # subsequent count uses exactly the same analyzed pixel domain.
    roi_bin: np.ndarray | None = None
    if valid_roi is not None:
        try:
            roi_bin = to_binary_mask(valid_roi)
            if roi_bin.shape != otsu_bin.shape:
                raise ValueError(
                    f"valid_roi shape {roi_bin.shape} does not match "
                    f"jewel_mask shape {otsu_bin.shape}"
                )
            otsu_bin = cv2.bitwise_and(otsu_bin, roi_bin)
        except Exception as exc:
            logger.warning("[StoneArea] valid_roi ignored: %s", exc)
            roi_bin = None

    supplied_jewel_area_px = int(np.count_nonzero(otsu_bin))

    # -- Clean and union stone masks ------------------------------------
    cleaned: dict[str, np.ndarray] = {}
    rejected_outside_jewel_px = 0
    for color_name, raw in stone_masks.items():
        if raw is None or not np.any(raw):
            continue
        try:
            bin_mask = to_binary_mask(raw)
        except Exception as exc:
            logger.warning("[StoneArea] Skipping %s mask: %s", color_name, exc)
            continue
        if bin_mask.shape != otsu_bin.shape:
            logger.warning(
                "[StoneArea] Skipping %s mask: shape %s does not match jewel "
                "mask shape %s",
                color_name,
                bin_mask.shape,
                otsu_bin.shape,
            )
            continue
        if roi_bin is not None:
            bin_mask = cv2.bitwise_and(bin_mask, roi_bin)
        if STONE_MASK_CLOSE_GAPS:
            bin_mask = cv2.morphologyEx(
                bin_mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL, iterations=1
            )
        bin_mask = remove_small_components(bin_mask, min_component_area_pixels)
        if not include_stones_outside_jewel:
            outside = cv2.bitwise_and(bin_mask, cv2.bitwise_not(otsu_bin))
            rejected_outside_jewel_px += int(np.count_nonzero(outside))
            bin_mask = cv2.bitwise_and(bin_mask, otsu_bin)
        if np.any(bin_mask):
            cleaned[color_name] = bin_mask

    # Union of all stone masks (a pixel counted only once regardless of
    # how many color classes overlap)
    stone_union = np.zeros(otsu_bin.shape, dtype=np.uint8)
    for mask in cleaned.values():
        stone_union = np.logical_or(stone_union, mask).astype(np.uint8) * 255

    # -- Complete jewel mask --------------------------------------------
    if include_stones_outside_jewel:
        complete_jewel = np.logical_or(otsu_bin, stone_union).astype(np.uint8) * 255
        denominator_source = "jewel_mask_or_stone_union"
    else:
        complete_jewel = otsu_bin.copy()
        stone_union = cv2.bitwise_and(stone_union, complete_jewel)
        denominator_source = "segmented_jewel_mask"

    # -- Core area counts -----------------------------------------------
    jewel_area_px = int(np.count_nonzero(complete_jewel))
    stone_area_px = int(np.count_nonzero(stone_union))

    if jewel_area_px == 0:
        logger.error(
            "[StoneArea] Jewelry mask is empty — cannot calculate percentages"
        )
        return _empty_failure("Jewelry mask is empty")

    metal_area_px = max(0, jewel_area_px - stone_area_px)
    stone_pct = round(stone_area_px / jewel_area_px * 100.0, 2)
    metal_pct = round(metal_area_px / jewel_area_px * 100.0, 2)

    # -- Detect overlapping color-class pixels --------------------------
    if len(cleaned) >= 2:
        overlap_accum = np.zeros(otsu_bin.shape, dtype=np.uint8)
        for mask in cleaned.values():
            overlap_accum = overlap_accum + (mask > 0).astype(np.uint8)
        overlapping_px = int(np.count_nonzero(overlap_accum > 1))
    else:
        overlapping_px = 0

    # -- Per-color statistics -------------------------------------------
    color_stats: dict[str, dict[str, Any]] = {}
    for color_name, mask in cleaned.items():
        color_area_px = int(np.count_nonzero(mask))
        # Count connected components (subtract 1 for background label)
        _, _, comp_stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        component_count = max(0, int(comp_stats.shape[0]) - 1)

        pct_of_jewel = round(color_area_px / jewel_area_px * 100.0, 2)
        pct_of_stones = (
            round(color_area_px / stone_area_px * 100.0, 2)
            if stone_area_px > 0
            else 0.0
        )
        color_stats[color_name] = {
            "area_pixels": color_area_px,
            "percentage_of_jewel": pct_of_jewel,
            "percentage_of_stones": pct_of_stones,
            "component_count": component_count,
        }

    # -- Logging --------------------------------------------------------
    logger.info("[StoneArea] Jewelry area:       %d px", jewel_area_px)
    logger.info("[StoneArea] Stone area:         %d px", stone_area_px)
    logger.info("[StoneArea] Metal area:         %d px", metal_area_px)
    logger.info("[StoneArea] Total stone coverage: %.2f%%", stone_pct)
    for cname, cs in color_stats.items():
        logger.info(
            "[StoneArea] %-22s %.2f%% of jewelry",
            cname + ":",
            cs["percentage_of_jewel"],
        )
    if overlapping_px:
        logger.info("[StoneArea] Overlapping class pixels: %d", overlapping_px)

    return {
        "success": True,
        "jewel_area_pixels": jewel_area_px,
        "stone_area_pixels": stone_area_px,
        "metal_area_pixels": metal_area_px,
        "stone_percentage": stone_pct,
        "metal_percentage": metal_pct,
        "overlapping_class_pixels": overlapping_px,
        "denominator_source": denominator_source,
        "supplied_jewel_mask_pixels": supplied_jewel_area_px,
        "stone_pixels_outside_jewel_rejected": rejected_outside_jewel_px,
        "colors": color_stats,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_failure(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": message,
        "jewel_area_pixels": 0,
        "stone_area_pixels": 0,
        "metal_area_pixels": 0,
        "stone_percentage": 0.0,
        "metal_percentage": 0.0,
        "overlapping_class_pixels": 0,
        "denominator_source": "segmented_jewel_mask",
        "supplied_jewel_mask_pixels": 0,
        "stone_pixels_outside_jewel_rejected": 0,
        "colors": {},
    }


# ---------------------------------------------------------------------------
# Validation tests (run with: python stone_area_calculator.py)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Self-contained validation tests for calculate_stone_area_statistics."""
    import sys

    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal failed
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
            failed += 1

    print("Running stone_area_calculator validation tests …\n")

    h, w = 10, 10

    # Test 1: 100 jewelry pixels, 20 stone pixels
    j1 = np.ones((h, w), dtype=np.uint8) * 255
    s1 = np.zeros((h, w), dtype=np.uint8)
    s1[:2, :10] = 255  # 20 pixels
    r1 = calculate_stone_area_statistics(j1, {"pink": s1})
    check("T1 stone_percentage=20.0", r1["stone_percentage"] == 20.0,
          str(r1["stone_percentage"]))
    check("T1 metal_percentage=80.0", r1["metal_percentage"] == 80.0,
          str(r1["metal_percentage"]))

    # Test 2: 100 jewelry pixels, no stones
    j2 = np.ones((h, w), dtype=np.uint8) * 255
    r2 = calculate_stone_area_statistics(j2, {})
    check("T2 stone_percentage=0.0", r2["stone_percentage"] == 0.0)
    check("T2 metal_percentage=100.0", r2["metal_percentage"] == 100.0)

    # Test 3: stone pixels outside the jewel mask are rejected by default.
    otsu3 = np.zeros((h, w), dtype=np.uint8)
    otsu3[2:, :] = 255   # 80 metal pixels
    white3 = np.zeros((h, w), dtype=np.uint8)
    white3[:2, :] = 255  # 20 stone pixels Otsu missed
    r3 = calculate_stone_area_statistics(otsu3, {"White/Colorless": white3})
    check("T3 jewel_area=80", r3["jewel_area_pixels"] == 80,
          str(r3["jewel_area_pixels"]))
    check("T3 stone_percentage=0.0", r3["stone_percentage"] == 0.0,
          str(r3["stone_percentage"]))
    check("T3 outside pixels rejected=20",
          r3["stone_pixels_outside_jewel_rejected"] == 20,
          str(r3["stone_pixels_outside_jewel_rejected"]))

    # Test 3b: legacy opt-in can still expand an incomplete jewel mask.
    r3b = calculate_stone_area_statistics(
        otsu3,
        {"White/Colorless": white3},
        include_stones_outside_jewel=True,
    )
    check("T3b legacy jewel_area=100", r3b["jewel_area_pixels"] == 100,
          str(r3b["jewel_area_pixels"]))
    check("T3b legacy stone_percentage=20.0", r3b["stone_percentage"] == 20.0,
          str(r3b["stone_percentage"]))

    # Test 4: Pink and Red overlap in 5 pixels
    j4 = np.ones((h, w), dtype=np.uint8) * 255
    pink4 = np.zeros((h, w), dtype=np.uint8)
    pink4[:, :7] = 255   # 70 px
    red4 = np.zeros((h, w), dtype=np.uint8)
    red4[:, 5:] = 255    # 50 px, overlap with pink in columns 5-6 → 20 px overlap
    r4 = calculate_stone_area_statistics(j4, {"pink": pink4, "red": red4})
    expected_union = 100  # all 100 px covered
    check("T4 union = 100 (no double-count)", r4["stone_area_pixels"] == expected_union,
          str(r4["stone_area_pixels"]))
    check("T4 overlapping_class_pixels > 0", r4["overlapping_class_pixels"] > 0,
          str(r4["overlapping_class_pixels"]))

    # Test 5: 1-pixel noise removed when min_area=5
    j5 = np.ones((h, w), dtype=np.uint8) * 255
    s5 = np.zeros((h, w), dtype=np.uint8)
    s5[0, 0] = 255   # single isolated pixel
    r5 = calculate_stone_area_statistics(j5, {"pink": s5}, min_component_area_pixels=5)
    check("T5 noise removed → stone_area=0", r5["stone_area_pixels"] == 0,
          str(r5["stone_area_pixels"]))

    # Test 6: Large necklace with empty interior — interior not counted
    # Simulate: ring-shaped Otsu mask (border pixels only)
    j6 = np.zeros((10, 10), dtype=np.uint8)
    j6[0, :] = j6[9, :] = j6[:, 0] = j6[:, 9] = 255  # border ring
    r6 = calculate_stone_area_statistics(j6, {})
    # Interior 8×8 = 64 pixels must NOT be in jewel_area
    border_px = int(np.count_nonzero(j6))
    check("T6 interior excluded", r6["jewel_area_pixels"] == border_px,
          f"got {r6['jewel_area_pixels']}, expected {border_px}")

    # Test 7: Empty jewel mask → controlled failure
    j7 = np.zeros((h, w), dtype=np.uint8)
    r7 = calculate_stone_area_statistics(j7, {})
    check("T7 success=False on empty mask", r7["success"] is False)
    check("T7 no exception", True)  # reaching here means no exception raised

    # Test 8: Dense local reflection hotspot marks the jewel as RISK
    j8 = np.ones((100, 100), dtype=np.uint8) * 255
    glare8 = np.zeros_like(j8)
    glare8[46:54, 46:54] = 255
    r8 = calculate_reflection_risk(glare8, j8)
    check("T8 dense reflection hotspot => RISK", r8["risk_status"] == "RISK",
          str(r8))

    # Test 9: A few isolated reflection pixels remain NORMAL
    glare9 = np.zeros_like(j8)
    glare9[10, 10] = glare9[50, 50] = glare9[90, 90] = 255
    r9 = calculate_reflection_risk(glare9, j8)
    check("T9 isolated reflections => NORMAL", r9["risk_status"] == "NORMAL",
          str(r9))

    # Test 10: metric weight totals are exposed in grams as well as internal carats
    stone10 = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(stone10, (40, 40), 10, 255, -1)
    r10 = calculate_stone_measurements(
        {"Red": stone10},
        mm_per_pixel_x=0.1,
        mm_per_pixel_y=0.1,
    )
    check("T10 gram weight available", r10["estimated_total_average_g"] > 0.0,
          str(r10))
    check(
        "T10 gram range ordered",
        r10["estimated_total_minimum_g"]
        <= r10["estimated_total_average_g"]
        <= r10["estimated_total_maximum_g"],
        str(r10),
    )

    # Test 11: V2 uses per-instance geometry with explicit uncertainty.
    r11 = apply_stone_setting_weight_model(
        r10,
        STONE_SETTING_PROFILE_FRONT_ONLY,
        visible_stone_area_mm2=1061.6565,
        jewel_weight_g=18.44,
    )
    check(
        "T11 geometry range ordered",
        r11["estimated_total_minimum_g"]
        <= r11["estimated_total_typical_g"]
        <= r11["estimated_total_maximum_g"],
        str(r11),
    )
    check(
        "T11 typical preserves legacy average key",
        r11["estimated_total_typical_g"] == r11["estimated_total_average_g"],
        str(r11),
    )
    check(
        "T11 depth and density warnings exposed",
        len(r11.get("weight_warnings") or []) >= 2,
        str(r11),
    )

    # Test 12: unknown setting exposes area but deliberately suppresses grams.
    r12 = apply_stone_setting_weight_model(
        r10,
        STONE_SETTING_PROFILE_UNKNOWN,
        visible_stone_area_mm2=1061.6565,
        jewel_weight_g=18.44,
    )
    check(
        "T12 unknown setting suppresses weight",
        r12["success"] is False and r12["weight_estimate_suppressed"] is True,
        str(r12),
    )

    print(f"\n{'All tests passed.' if failed == 0 else f'{failed} test(s) FAILED.'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    _run_tests()
