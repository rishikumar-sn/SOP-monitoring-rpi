"""
Classical CV bangle OD/ID detector with CLI and PyQt6 GUI.

Examples:
    python bangle_detector.py --image path/to/bangle.jpg
    python bangle_detector.py --image path/to/bangle.jpg --scale 0.085
    python bangle_detector.py --image path/to/bangle.jpg --debug
    python bangle_detector.py --gui

Dependencies:
    pip install opencv-python numpy scipy matplotlib PyQt6

The detector is intentionally classical computer vision only: CLAHE,
bilateral filtering, adaptive/Otsu thresholding, morphology, color masking,
radial circle measurement, and HoughCircles fallback.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# ----------------------------- Data model -----------------------------


@dataclass
class CircleInfo:
    """Circle representation for OD/ID measurement."""

    center: tuple[float, float]
    radius: float
    method: str
    support_count: int = 0
    ellipse: Optional[tuple[tuple[float, float], tuple[float, float], float]] = None

    @property
    def diameter(self) -> float:
        return float(self.radius * 2.0)


# ----------------------------- Core pipeline -----------------------------


def preprocess(img: np.ndarray) -> np.ndarray:
    """Return CLAHE-enhanced, bilateral-filtered grayscale image."""
    if img is None or img.size == 0:
        raise ValueError("Empty image supplied to preprocess().")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # CLAHE reduces the effect of slow illumination gradients and shadows before
    # edge/threshold operations see the image.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Bilateral filtering suppresses sensor noise while preserving hard ring
    # boundaries better than Gaussian blur.
    return cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)


def _auto_canny(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Canny thresholds derived from image median for one sigma setting."""
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 40, 120
    return cv2.Canny(gray, lower, upper, L2gradient=True)


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Return multi-scale Canny edges using two sigma passes."""
    edges_tight = _auto_canny(gray, sigma=0.25)
    edges_loose = _auto_canny(gray, sigma=0.55)
    edges = cv2.bitwise_or(edges_tight, edges_loose)

    # A light close bridges tiny breaks caused by glare or low-contrast patches.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def segment_bangle(img: np.ndarray) -> np.ndarray:
    """Binary mask via adaptive threshold + Otsu OR, then morphology cleanup."""
    gray = preprocess(img)

    # Adaptive threshold catches local contrast under uneven lighting.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        3,
    )

    # Otsu catches globally separated foreground/background cases.
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick the Otsu polarity whose foreground area is plausible; this avoids
    # accidentally selecting the whole background on bright metal scenes.
    total = gray.shape[0] * gray.shape[1]
    inv_area = cv2.countNonZero(otsu_inv) / float(total)
    otsu_mask = otsu_inv if inv_area < 0.55 else otsu

    combined = cv2.bitwise_or(adaptive, otsu_mask)

    min_dim = min(gray.shape[:2])
    k_close = max(5, int(round(min_dim * 0.018)) | 1)
    k_open = max(3, int(round(min_dim * 0.006)) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return cleaned


def normalize_illumination(gray: np.ndarray, shadow_strength: int = 70) -> np.ndarray:
    """Dimension-calib-style shadow normalization before Otsu thresholding."""
    image_h, image_w = gray.shape[:2]
    min_size = min(image_w, image_h)
    bg_size = _odd_kernel(min_size * 0.22, 41)
    sh_size = _odd_kernel(min_size * 0.08, 15)
    strength = float(np.clip(shadow_strength, 0, 100)) / 100.0

    background = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
    divided = cv2.divide(gray, background, scale=255)

    shadow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (sh_size, sh_size)
    )
    dark_halo = cv2.morphologyEx(divided, cv2.MORPH_BLACKHAT, shadow_kernel)
    bright_halo = cv2.morphologyEx(divided, cv2.MORPH_TOPHAT, shadow_kernel)
    corrected = cv2.addWeighted(divided, 1.0, dark_halo, strength, 0)
    corrected = cv2.addWeighted(corrected, 1.0, bright_halo, -0.35 * strength, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(corrected)
    return cv2.GaussianBlur(normalized, (5, 5), 0)


def clean_otsu_mask(mask: np.ndarray) -> np.ndarray:
    """Connected-component and morphology cleanup from dimension_calib.py."""
    image_h, image_w = mask.shape[:2]
    min_size = min(image_w, image_h)
    cleaned = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(30, int(min_size * min_size * 0.0005))

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        touches_border = (
            x <= 1 or y <= 1 or x + w >= image_w - 1 or y + h >= image_h - 1
        )
        too_large = w > image_w * 0.75 or h > image_h * 0.75
        if area >= min_area and not touches_border and not too_large:
            cleaned[labels == label] = 255

    k = _odd_kernel(min_size * 0.018, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def otsu_bangle_mask(img: np.ndarray, threshold_offset: int = 0) -> np.ndarray:
    """Build an Otsu mask using the process from dimension_calib.py."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    corrected = normalize_illumination(gray)

    otsu_value, _ = cv2.threshold(
        corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = int(np.clip(otsu_value + threshold_offset, 0, 255))
    _, thresholded = cv2.threshold(corrected, threshold_value, 255, cv2.THRESH_BINARY)
    thresholded = cv2.bitwise_not(thresholded)

    kernel = np.ones((3, 3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    return clean_otsu_mask(thresholded)


def gold_bangle_mask(img: np.ndarray) -> np.ndarray:
    """Return a color-prior mask for gold/brown bangle pixels.

    The reference images have a pale background and a gold bangle. This mask is
    not used alone for measurement; it acts as a strong object prior so black
    ArUco markers, paper edges, and gray shadows are less likely to be chosen.
    
    Shadow removal is applied to prevent false inner diameter detection caused
    by shadow pixels being misclassified as bangle material.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)

    # Gold appears as yellow/orange/brown in HSV with enough saturation. The
    # value upper bound avoids swallowing white glare/background.
    hsv_gold = cv2.inRange(hsv, np.array([5, 22, 35]), np.array([45, 255, 245]))

    # Extra BGR evidence catches darker brown shadowed gold where hue can wobble.
    warm_excess = np.maximum(r, g).astype(np.int16) - b.astype(np.int16)
    bgr_gold = (
        (warm_excess > 18)
        & (r > 70)
        & (g > 55)
        & (s > 16)
        & (v < 250)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_or(hsv_gold, bgr_gold)
    
    # **Shadow detection and removal**
    # Shadows are detected as regions with:
    # - Valid gold hue (5-45 degrees)
    # - BUT significantly darker than surrounding gold (v < 80)
    # - AND lower saturation typical of shadows (s < 80)
    shadow_region = (
        (h >= 5) & (h <= 45)
        & (v < 80)
        & (s < 80)
    ).astype(np.uint8) * 255
    
    # Dilate shadow region slightly to ensure complete removal
    shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_region = cv2.dilate(shadow_region, shadow_kernel, iterations=1)
    
    # Remove detected shadows from the mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_region))
    
    marker_ignore = detect_marker_ignore_mask(img)
    mask[marker_ignore > 0] = 0

    min_dim = min(mask.shape[:2])
    open_k = max(3, int(round(min_dim * 0.004)) | 1)
    close_k = max(9, int(round(min_dim * 0.018)) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return mask


def detect_marker_ignore_mask(img: np.ndarray) -> np.ndarray:
    """Detect and mask square fiducial markers in the upper-left image area."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    dark = cv2.inRange(gray, 0, 95)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ignore = np.zeros_like(gray)

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if x > w * 0.30 or y > h * 0.30:
            continue
        if not (w * 0.04 <= bw <= w * 0.22 and h * 0.04 <= bh <= h * 0.22):
            continue
        aspect = bw / float(max(bh, 1))
        fill = area / float(max(bw * bh, 1))
        if 0.65 <= aspect <= 1.35 and fill > 0.12:
            pad = int(round(max(bw, bh) * 0.18))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            ignore[y0:y1, x0:x1] = 255
    return ignore


def _contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def find_ring_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find contour candidates filtered by area and circularity."""
    h, w = mask.shape[:2]
    img_area = float(h * w)
    min_area = max(80.0, img_area * 0.0004)
    max_area = img_area * 0.92

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    candidates: list[np.ndarray] = []

    for contour in contours:
        if len(contour) < 20:
            continue
        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < w * 0.03 or bh < h * 0.03:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        circularity = _contour_circularity(contour)
        if circularity < 0.12:
            continue
        candidates.append(contour)

    candidates.sort(key=cv2.contourArea, reverse=True)
    return candidates


def _odd_kernel(value: float, minimum: int) -> int:
    size = max(minimum, int(round(value)))
    return size + 1 if size % 2 == 0 else size


def _measure_contour_circle(contour: np.ndarray, method: str) -> Optional[CircleInfo]:
    """Measure a contour with an ellipse fit or true circle."""
    if contour is None or len(contour) == 0:
        return None
    
    if len(contour) >= 5:
        # Preferred: Ellipse fit (more robust for bangles at slight angles)
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (d1, d2), angle = ellipse
            avg_diameter = (d1 + d2) / 2.0
            return CircleInfo(
                center=(float(cx), float(cy)),
                radius=float(avg_diameter / 2.0),
                method=method,
                support_count=int(len(contour)),
                ellipse=ellipse,
            )
        except cv2.error:
            pass

    # Fallback: Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if radius <= 2.0 or not np.isfinite([cx, cy, radius]).all():
        return None
    return CircleInfo(
        center=(float(cx), float(cy)),
        radius=float(radius),
        method=method,
        support_count=int(len(contour)),
    )


def _find_inner_from_filled(mask: np.ndarray, outer_contour: np.ndarray) -> Optional[np.ndarray]:
    """Find the bangle hole when contour hierarchy did not expose a child."""
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [outer_contour], -1, 255, -1)
    hole_mask = cv2.subtract(filled, mask)
    hole_mask = cv2.morphologyEx(
        hole_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    hole_contours, _ = cv2.findContours(
        hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_hole = cv2.contourArea(outer_contour) * 0.03
    valid = [c for c in hole_contours if cv2.contourArea(c) > min_hole]
    return max(valid, key=cv2.contourArea) if valid else None


def contour_circle_detection(mask: np.ndarray) -> Optional[tuple[CircleInfo, CircleInfo, np.ndarray, np.ndarray]]:
    """Dimension-calib-style contour hierarchy detection, measured as circles/ellipses."""
    image_h, image_w = mask.shape[:2]
    min_size = min(image_w, image_h)
    k = _odd_kernel(min_size * 0.025, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    best_outer_idx = -1
    best_inner_idx = -1
    best_score = -1.0

    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        if w < min_size * 0.03 or h < min_size * 0.03:
            continue
        if w > image_w * 0.80 or h > image_h * 0.80:
            continue

        children = [
            ci for ci, item in enumerate(hierarchy)
            if item[3] == index and cv2.contourArea(contours[ci]) > 0
        ]
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.25:
            continue

        child_idx = -1
        child_area = 0.0
        if children:
            child_idx = int(max(children, key=lambda ci: cv2.contourArea(contours[ci])))
            child_area = cv2.contourArea(contours[child_idx])
            if child_area < area * 0.05:
                child_idx = -1
                child_area = 0.0

        score = area + child_area + circularity * 5000.0
        if score > best_score:
            best_outer_idx = index
            best_inner_idx = child_idx
            best_score = score

    if best_outer_idx == -1:
        # Fallback: Find largest top-level contour
        top_level = []
        for i, h_node in enumerate(hierarchy):
            if h_node[3] == -1:
                top_level.append((i, contours[i]))
        
        if not top_level:
            return None
        best_outer_idx, best_outer_contour = max(top_level, key=lambda x: cv2.contourArea(x[1]))
    else:
        best_outer_contour = contours[best_outer_idx]

    if best_inner_idx == -1:
        best_inner_contour = _find_inner_from_filled(connected, best_outer_contour)
    else:
        best_inner_contour = contours[best_inner_idx]

    outer = _measure_contour_circle(best_outer_contour, "contour_circle")
    inner = _measure_contour_circle(best_inner_contour, "contour_circle")
    if outer is None or inner is None:
        return None
    if inner.radius >= outer.radius:
        return None

    center_dist = math.hypot(outer.center[0] - inner.center[0], outer.center[1] - inner.center[1])
    if center_dist > math.hypot(image_w, image_h) * 0.15:
        return None
    if inner.radius / outer.radius < 0.62:
        return None
    return outer, inner, best_outer_contour, best_inner_contour


def hough_circle_detection(
    gray: np.ndarray, ignore_mask: Optional[np.ndarray] = None
) -> Optional[tuple[CircleInfo, CircleInfo]]:
    """Fallback OD/ID detection using HoughCircles."""
    if ignore_mask is not None:
        gray = gray.copy()
        # Paint ignored regions as background so square markers cannot become
        # fallback circles if radial measurement fails.
        bg_value = int(np.median(gray[ignore_mask == 0])) if np.any(ignore_mask == 0) else 220
        gray[ignore_mask > 0] = bg_value
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 8),
        param1=90,
        param2=24,
        minRadius=max(5, int(min_dim * 0.03)),
        maxRadius=int(min_dim * 0.48),
    )
    if circles is None:
        return None

    detected = np.squeeze(circles, axis=0)
    if detected.ndim != 2 or detected.shape[0] < 2:
        return None

    # Prefer two circles with nearby centers and substantially different radii.
    best_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
    best_score = float("inf")
    diag = math.hypot(w, h)
    for i in range(len(detected)):
        for j in range(i + 1, len(detected)):
            c1, c2 = detected[i], detected[j]
            r1, r2 = float(c1[2]), float(c2[2])
            if abs(r1 - r2) < min_dim * 0.035:
                continue
            center_dist = math.hypot(float(c1[0] - c2[0]), float(c1[1] - c2[1]))
            if center_dist > diag * 0.15:
                continue
            score = center_dist - abs(r1 - r2) * 0.02
            if score < best_score:
                best_score = score
                best_pair = (c1, c2)

    if best_pair is None:
        return None

    outer_c, inner_c = sorted(best_pair, key=lambda c: c[2], reverse=True)
    outer = CircleInfo(
        center=(float(outer_c[0]), float(outer_c[1])),
        radius=float(outer_c[2]),
        method="hough_circle",
    )
    inner = CircleInfo(
        center=(float(inner_c[0]), float(inner_c[1])),
        radius=float(inner_c[2]),
        method="hough_circle",
    )
    return outer, inner


def _largest_bangle_component(mask: np.ndarray) -> Optional[np.ndarray]:
    """Keep the largest non-border gold component as the bangle body."""
    h, w = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best_label = 0
    best_area = 0

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < max(250, int(h * w * 0.0005)):
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        if area > best_area:
            best_label = label
            best_area = int(area)

    if best_label == 0:
        return None
    component = np.zeros_like(mask)
    component[labels == best_label] = 255
    return component


def radial_circle_detection(mask: np.ndarray) -> Optional[tuple[CircleInfo, CircleInfo]]:
    """Measure OD/ID as circles from radial distances of bangle mask pixels."""
    component = _largest_bangle_component(mask)
    if component is None:
        return None

    points_y, points_x = np.where(component > 0)
    if len(points_x) < 100:
        return None

    x, y, bw, bh = cv2.boundingRect(component)
    center_x = x + bw / 2.0
    center_y = y + bh / 2.0
    distances = np.hypot(points_x.astype(np.float64) - center_x, points_y.astype(np.float64) - center_y)
    distances = distances[np.isfinite(distances)]
    if distances.size < 100:
        return None

    # Robust percentiles ignore isolated decorative bumps and tiny mask specks.
    inner_radius = float(np.percentile(distances, 4.0))
    outer_radius = float(np.percentile(distances, 98.5))
    if inner_radius <= 2.0 or outer_radius <= inner_radius:
        return None
    if inner_radius / outer_radius < 0.62:
        return None

    center = (float(center_x), float(center_y))
    outer = CircleInfo(center=center, radius=outer_radius, method="radial_mask", support_count=int(distances.size))
    inner = CircleInfo(center=center, radius=inner_radius, method="radial_mask", support_count=int(distances.size))
    return outer, inner


def _finger_ring_color_mask(img: np.ndarray) -> np.ndarray:
    """Return a minimally processed mask that preserves small ring holes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    _l, _a, lab_b = cv2.split(lab)
    b, g, r = cv2.split(img)

    warm_hsv = (
        (h >= 2)
        & (h <= 48)
        & (s >= 12)
        & (v >= 28)
        & (v <= 252)
    )
    warm_bgr = (
        (r.astype(np.int16) - b.astype(np.int16) >= 8)
        & (g.astype(np.int16) - b.astype(np.int16) >= 4)
        & (r >= 55)
        & (v <= 250)
    )
    yellow_lab = (lab_b >= 132) & (s >= 8) & (v >= 35) & (v <= 250)
    mask = (warm_hsv | warm_bgr | yellow_lab).astype(np.uint8) * 255

    # Small fixed kernels are intentional. Frame-relative bangle kernels can be
    # wider than a finger ring and fill its center before contour extraction.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _finger_ring_radius_limits(
    image_shape: tuple[int, ...],
    scale: Optional[float],
) -> tuple[float, float, float, float]:
    """Return inner/outer radius limits in original-image pixels."""
    min_dim = float(min(image_shape[:2]))
    if scale is not None and scale > 0:
        # Deliberately broad physical limits cover small children's rings and
        # large decorative rings while excluding bangle-sized candidates.
        inner_min = 2.5 / scale
        inner_max = 18.0 / scale
        outer_min = 4.0 / scale
        outer_max = 25.0 / scale
    else:
        inner_min = min_dim * 0.004
        inner_max = min_dim * 0.14
        outer_min = min_dim * 0.006
        outer_max = min_dim * 0.18

    inner_min = max(3.0, inner_min)
    outer_min = max(inner_min + 1.5, 5.0, outer_min)
    inner_max = min(max(inner_min + 2.0, inner_max), min_dim * 0.22)
    outer_max = min(max(outer_min + 3.0, outer_max), min_dim * 0.26)
    return inner_min, inner_max, outer_min, outer_max


def _candidate_ring_rois(
    img: np.ndarray,
    color_mask: np.ndarray,
    outer_min: float,
    outer_max: float,
) -> list[tuple[int, int, int, int]]:
    """Locate small circular/color components before expensive zoomed Hough."""
    image_h, image_w = img.shape[:2]
    candidates: list[tuple[float, float, float]] = []

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_mask, connectivity=8
    )
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        size = float(max(bw, bh))
        if area < 8:
            continue
        if size < outer_min * 0.65 or size > outer_max * 3.2:
            continue
        aspect = bw / float(max(bh, 1))
        if not 0.35 <= aspect <= 2.85:
            continue
        cx, cy = centroids[index]
        candidates.append((float(cx), float(cy), max(size, outer_min * 2.0)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 115)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 12:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not outer_min * 0.65 <= radius <= outer_max * 1.25:
            continue
        area = float(abs(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if area > 0 and perimeter > 0
            else 0.0
        )
        if circularity < 0.20:
            continue
        candidates.append((float(cx), float(cy), float(radius * 2.0)))

    candidates.sort(key=lambda item: item[2])
    rois: list[tuple[int, int, int, int]] = []
    centers: list[tuple[float, float, float]] = []
    for cx, cy, size in candidates:
        if any(
            math.hypot(cx - old_x, cy - old_y) < max(8.0, min(size, old_size) * 0.35)
            for old_x, old_y, old_size in centers
        ):
            continue
        half = int(
            round(
                min(
                    outer_max * 1.45,
                    max(size * 0.95, outer_min * 1.8, 20.0),
                )
            )
        )
        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(image_w, int(round(cx)) + half + 1)
        y1 = min(image_h, int(round(cy)) + half + 1)
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            rois.append((x0, y0, x1, y1))
            centers.append((cx, cy, size))
        if len(rois) >= 24:
            break

    # A full-frame fallback covers low-saturation/white-gold rings. Candidate
    # ROIs remain first, so the common gold-ring path stays inexpensive.
    rois.append((0, 0, image_w, image_h))
    return rois


def _dedupe_circles(
    circles: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Merge duplicate Hough detections while retaining the strongest one."""
    deduped: list[tuple[float, float, float, float]] = []
    for cx, cy, radius, zoom in sorted(circles, key=lambda item: item[2]):
        duplicate_index = None
        for index, (old_x, old_y, old_radius, _old_zoom) in enumerate(deduped):
            if (
                math.hypot(cx - old_x, cy - old_y) <= max(2.0, radius * 0.08)
                and abs(radius - old_radius) <= max(1.5, radius * 0.07)
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            deduped.append((cx, cy, radius, zoom))
        elif zoom > deduped[duplicate_index][3]:
            deduped[duplicate_index] = (cx, cy, radius, zoom)
    return deduped


def _zoomed_hough_circles(
    img: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    inner_min: float,
    outer_max: float,
) -> tuple[list[tuple[float, float, float, float]], float]:
    """Detect small circle boundaries in candidate crops and map them back."""
    circles_out: list[tuple[float, float, float, float]] = []
    max_zoom_used = 1.0

    for roi_index, (x0, y0, x1, y1) in enumerate(rois):
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_h, crop_w = crop.shape[:2]
        full_frame = roi_index == len(rois) - 1
        if full_frame and len(circles_out) >= 2:
            break
        if full_frame:
            zoom = min(1.35, 1800.0 / max(crop_h, crop_w))
            zoom = max(1.0, zoom)
        else:
            desired = 18.0 / max(inner_min, 1.0)
            zoom = float(np.clip(desired, 1.5, 4.0))
            zoom = min(zoom, 1400.0 / max(crop_h, crop_w))
            zoom = max(1.0, zoom)
        max_zoom_used = max(max_zoom_used, zoom)

        enlarged = cv2.resize(
            crop,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_CUBIC if zoom > 1.0 else cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        variants = [
            cv2.GaussianBlur(gray, (5, 5), 0),
            cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0),
        ]
        min_radius = max(3, int(math.floor(inner_min * zoom * 0.78)))
        max_radius = min(
            int(math.ceil(outer_max * zoom * 1.08)),
            int(min(enlarged.shape[:2]) * 0.48),
        )
        if max_radius <= min_radius + 2:
            continue

        local_circles: list[np.ndarray] = []
        for variant in variants:
            for param2 in (18, 14, 11):
                detected = cv2.HoughCircles(
                    variant,
                    cv2.HOUGH_GRADIENT,
                    dp=1.0,
                    minDist=max(6, int(inner_min * zoom * 0.45)),
                    param1=85,
                    param2=param2,
                    minRadius=min_radius,
                    maxRadius=max_radius,
                )
                if detected is not None:
                    local_circles.extend(np.asarray(detected[0], dtype=np.float32))
                if len(local_circles) >= 12:
                    break
            if len(local_circles) >= 12:
                break

        for circle in local_circles:
            cx = x0 + float(circle[0]) / zoom
            cy = y0 + float(circle[1]) / zoom
            radius = float(circle[2]) / zoom
            circles_out.append((cx, cy, radius, zoom))

        if len(circles_out) >= 160:
            break

    return _dedupe_circles(circles_out), max_zoom_used


def _circle_edge_support(
    edges: np.ndarray,
    center: tuple[float, float],
    radius: float,
    tolerance_px: Optional[int] = None,
) -> float:
    """Measure how much of a proposed circle is supported by nearby edges."""
    if radius <= 1:
        return 0.0
    samples = int(np.clip(round(2.0 * math.pi * radius), 72, 720))
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    tolerance = (
        max(1, int(round(radius * 0.035)))
        if tolerance_px is None
        else max(0, int(tolerance_px))
    )
    supported = np.zeros(samples, dtype=bool)
    for offset in range(-tolerance, tolerance + 1):
        sample_radius = radius + offset
        xs = np.rint(center[0] + np.cos(angles) * sample_radius).astype(np.int32)
        ys = np.rint(center[1] + np.sin(angles) * sample_radius).astype(np.int32)
        valid = (
            (xs >= 0)
            & (xs < edges.shape[1])
            & (ys >= 0)
            & (ys < edges.shape[0])
        )
        supported[valid] |= edges[ys[valid], xs[valid]] > 0
    return float(supported.mean())


def _annulus_color_support(
    color_mask: np.ndarray,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> float:
    """Measure gold/warm pixel support inside the proposed ring wall."""
    x0 = max(0, int(math.floor(center[0] - outer_radius - 1)))
    y0 = max(0, int(math.floor(center[1] - outer_radius - 1)))
    x1 = min(color_mask.shape[1], int(math.ceil(center[0] + outer_radius + 2)))
    y1 = min(color_mask.shape[0], int(math.ceil(center[1] + outer_radius + 2)))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    distances_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    annulus = (
        (distances_sq >= max(0.0, inner_radius - 1.5) ** 2)
        & (distances_sq <= (outer_radius + 1.5) ** 2)
    )
    if not np.any(annulus):
        return 0.0
    return float(np.mean(color_mask[y0:y1, x0:x1][annulus] > 0))


def _finger_ring_pair_score(
    outer: CircleInfo,
    inner: CircleInfo,
    edges: np.ndarray,
    color_mask: np.ndarray,
    scale: Optional[float],
) -> Optional[float]:
    """Validate and score a concentric finger-ring OD/ID pair."""
    if outer.radius <= inner.radius:
        return None
    ratio = inner.radius / outer.radius
    if not 0.34 <= ratio <= 0.96:
        return None

    center_distance = math.hypot(
        outer.center[0] - inner.center[0],
        outer.center[1] - inner.center[1],
    )
    if center_distance > max(3.5, outer.radius * 0.18):
        return None

    wall_px = outer.radius - inner.radius
    if wall_px < 1.0:
        return None
    if scale is not None and scale > 0:
        od_mm = outer.diameter * scale
        id_mm = inner.diameter * scale
        wall_mm = wall_px * scale
        if not 8.0 <= od_mm <= 50.0:
            return None
        if not 5.0 <= id_mm <= 36.0:
            return None
        if not 0.20 <= wall_mm <= 10.0:
            return None

    outer_support = _circle_edge_support(edges, outer.center, outer.radius)
    inner_support = _circle_edge_support(edges, inner.center, inner.radius)
    annulus_support = _annulus_color_support(
        color_mask,
        (
            (outer.center[0] + inner.center[0]) / 2.0,
            (outer.center[1] + inner.center[1]) / 2.0,
        ),
        inner.radius,
        outer.radius,
    )
    if outer_support + inner_support < 0.20:
        return None
    if max(outer_support, inner_support) < 0.12:
        return None

    center_penalty = center_distance / max(outer.radius, 1.0)
    wall_ratio = wall_px / outer.radius
    plausible_wall_bonus = 1.0 - min(abs(wall_ratio - 0.16) / 0.30, 1.0)
    return float(
        outer_support * 4.0
        + inner_support * 4.0
        + annulus_support * 3.0
        + plausible_wall_bonus * 0.5
        - center_penalty * 8.0
    )


def _best_radial_inner_circle(
    outer: CircleInfo,
    edges: np.ndarray,
    inner_min: float,
    inner_max: float,
) -> Optional[CircleInfo]:
    """Infer an inner boundary when Hough reports only the outer boundary."""
    low = max(inner_min, outer.radius * 0.36)
    high = min(inner_max, outer.radius * 0.95)
    if high <= low + 1:
        return None

    radii = np.arange(math.ceil(low), math.floor(high) + 0.5, 1.0)
    if radii.size == 0:
        return None
    supports = np.array(
        [_circle_edge_support(edges, outer.center, float(radius)) for radius in radii]
    )
    best_index = int(np.argmax(supports))
    if supports[best_index] < 0.13:
        return None
    return CircleInfo(
        center=outer.center,
        radius=float(radii[best_index]),
        method="finger_ring_radial_inner",
        support_count=int(round(supports[best_index] * 1000)),
    )


def _radial_ring_pairs(
    center: tuple[float, float],
    edges: np.ndarray,
    inner_min: float,
    inner_max: float,
    outer_min: float,
    outer_max: float,
) -> list[tuple[CircleInfo, CircleInfo]]:
    """Find separate ID/OD edge peaks around a Hough-estimated center."""
    low = max(2, int(math.floor(inner_min * 0.75)))
    high = min(
        int(math.ceil(outer_max * 1.08)),
        int(
            min(
                center[0],
                center[1],
                edges.shape[1] - 1 - center[0],
                edges.shape[0] - 1 - center[1],
            )
        ),
    )
    if high <= low + 3:
        return []

    radii = np.arange(low, high + 1, dtype=np.float32)
    exact_support = np.array(
        [
            _circle_edge_support(
                edges,
                center,
                float(radius),
                tolerance_px=0,
            )
            for radius in radii
        ],
        dtype=np.float32,
    )
    if exact_support.size >= 3:
        exact_support = np.convolve(
            exact_support,
            np.array([0.20, 0.60, 0.20], dtype=np.float32),
            mode="same",
        )

    peak_indices = [
        index
        for index in range(1, len(radii) - 1)
        if exact_support[index] >= 0.08
        and exact_support[index] >= exact_support[index - 1]
        and exact_support[index] >= exact_support[index + 1]
    ]
    peak_indices.sort(key=lambda index: float(exact_support[index]), reverse=True)

    selected: list[int] = []
    for index in peak_indices:
        if any(abs(float(radii[index] - radii[old])) < 2.0 for old in selected):
            continue
        selected.append(index)
        if len(selected) >= 14:
            break

    pairs: list[tuple[CircleInfo, CircleInfo]] = []
    for first_pos, first_index in enumerate(selected):
        for second_index in selected[first_pos + 1:]:
            first_radius = float(radii[first_index])
            second_radius = float(radii[second_index])
            inner_radius, outer_radius = sorted((first_radius, second_radius))
            if not inner_min * 0.75 <= inner_radius <= inner_max * 1.10:
                continue
            if not outer_min * 0.75 <= outer_radius <= outer_max * 1.10:
                continue
            if outer_radius - inner_radius < 2.0:
                continue
            outer = CircleInfo(
                center=center,
                radius=outer_radius,
                method="finger_ring_radial_profile",
                support_count=int(round(exact_support[second_index] * 1000)),
            )
            inner = CircleInfo(
                center=center,
                radius=inner_radius,
                method="finger_ring_radial_profile",
                support_count=int(round(exact_support[first_index] * 1000)),
            )
            pairs.append((outer, inner))
    return pairs


def finger_ring_circle_detection(
    img: np.ndarray,
    scale: Optional[float],
) -> tuple[CircleInfo, CircleInfo, float]:
    """Detect small finger-ring OD/ID circles using calibrated zoomed crops."""
    color_mask = _finger_ring_color_mask(img)
    inner_min, inner_max, outer_min, outer_max = _finger_ring_radius_limits(
        img.shape, scale
    )
    rois = _candidate_ring_rois(img, color_mask, outer_min, outer_max)
    detected, max_zoom = _zoomed_hough_circles(
        img,
        rois,
        inner_min=inner_min,
        outer_max=outer_max,
    )
    if not detected:
        raise RuntimeError(
            "Could not find a finger-ring circle. Place the ring flat inside "
            "the processing ROI with its center hole visible."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.bitwise_or(
        cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 100),
        cv2.Canny(cv2.GaussianBlur(normalized, (3, 3), 0), 35, 120),
    )

    candidates: list[tuple[float, CircleInfo, CircleInfo]] = []
    for first_index, first in enumerate(detected):
        for second in detected[first_index + 1:]:
            if first[2] >= second[2]:
                outer_raw, inner_raw = first, second
            else:
                outer_raw, inner_raw = second, first
            if not outer_min * 0.75 <= outer_raw[2] <= outer_max * 1.10:
                continue
            if not inner_min * 0.75 <= inner_raw[2] <= inner_max * 1.10:
                continue
            outer = CircleInfo(
                center=(outer_raw[0], outer_raw[1]),
                radius=outer_raw[2],
                method="finger_ring_zoom_hough",
            )
            inner = CircleInfo(
                center=(inner_raw[0], inner_raw[1]),
                radius=inner_raw[2],
                method="finger_ring_zoom_hough",
            )
            score = _finger_ring_pair_score(
                outer, inner, edges, color_mask, scale
            )
            if score is not None:
                candidates.append((score, outer, inner))

    # Partial glare often causes Hough to return only one of the two circles.
    # Use the same center and inspect radial edge support for the missing ID.
    for cx, cy, radius, _zoom in detected:
        if not outer_min * 0.75 <= radius <= outer_max * 1.10:
            continue
        outer = CircleInfo(
            center=(cx, cy),
            radius=radius,
            method="finger_ring_zoom_hough",
        )
        inner = _best_radial_inner_circle(
            outer, edges, inner_min=inner_min, inner_max=inner_max
        )
        if inner is None:
            continue
        score = _finger_ring_pair_score(outer, inner, edges, color_mask, scale)
        if score is not None:
            candidates.append((score - 0.15, outer, inner))

    # Hough is excellent at finding the center of a small ring but glare can
    # bias its radius toward the bright side of a thick band. Re-scan radial
    # edge peaks around the strongest centers to recover the true OD and ID.
    center_seeds: list[tuple[float, float]] = []
    ranked_detections = sorted(
        detected,
        key=lambda item: _circle_edge_support(
            edges, (item[0], item[1]), item[2]
        ),
        reverse=True,
    )
    for cx, cy, _radius, _zoom in ranked_detections:
        if any(math.hypot(cx - old_x, cy - old_y) < 2.5 for old_x, old_y in center_seeds):
            continue
        center_seeds.append((cx, cy))
        if len(center_seeds) >= 24:
            break

    for center in center_seeds:
        for outer, inner in _radial_ring_pairs(
            center,
            edges,
            inner_min=inner_min,
            inner_max=inner_max,
            outer_min=outer_min,
            outer_max=outer_max,
        ):
            score = _finger_ring_pair_score(
                outer, inner, edges, color_mask, scale
            )
            if score is not None:
                candidates.append((score + 0.20, outer, inner))

    if not candidates:
        raise RuntimeError(
            "A small circular object was found, but valid concentric OD/ID "
            "boundaries were not visible. Place the finger ring flat and keep "
            "the center hole clear."
        )

    _score, outer, inner = max(candidates, key=lambda item: item[0])
    return outer, inner, max_zoom


def detect_bangle(
    image_path: str | os.PathLike[str],
    scale: Optional[float] = None,
    debug: bool = False,
    jewel_type: Optional[str] = None,
) -> dict[str, Any]:
    """Full bangle detection pipeline. Returns a serializable result dict."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    normalized_jewel_type = str(jewel_type or "").strip().lower()
    is_finger_ring = "finger" in normalized_jewel_type and "ring" in normalized_jewel_type

    if is_finger_ring:
        outer, inner, zoom_factor = finger_ring_circle_detection(img, scale)
        od_px = outer.diameter
        id_px = inner.diameter
        wall_px = (od_px - id_px) / 2.0
        result: dict[str, Any] = {
            "image_path": str(image_path),
            "outer": asdict(outer),
            "inner": asdict(inner),
            "od_px": float(od_px),
            "id_px": float(id_px),
            "wall_thickness_px": float(wall_px),
            "scale_mm_per_px": float(scale) if scale is not None else None,
            "used_fallback": False,
            "detection_mode": "finger_ring_zoom",
            "zoom_factor": float(zoom_factor),
        }
        if scale is not None:
            result.update(
                {
                    "od_mm": float(od_px * scale),
                    "id_mm": float(id_px * scale),
                    "wall_thickness_mm": float(wall_px * scale),
                }
            )

        annotated = draw_results(img, result)
        output_path = _result_path(image_path)
        cv2.imwrite(str(output_path), annotated)
        result["annotated_path"] = str(output_path)
        return result

    gray = preprocess(img)
    mask = segment_bangle(img)
    edges = edge_map(gray)
    gold_mask = gold_bangle_mask(img)
    otsu_mask = otsu_bangle_mask(img)
    marker_ignore = detect_marker_ignore_mask(img)

    # Remove fiducial markers and paper borders from generic threshold/edge
    # sources, then add Otsu support near the gold prior for this use case.
    mask[marker_ignore > 0] = 0
    edges[marker_ignore > 0] = 0
    otsu_mask[marker_ignore > 0] = 0

    min_dim = min(mask.shape[:2])
    prior_k = _odd_kernel(min_dim * 0.035, 15)
    prior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prior_k, prior_k))
    gold_neighborhood = cv2.dilate(gold_mask, prior_kernel, iterations=1)
    otsu_near_gold = cv2.bitwise_and(otsu_mask, gold_neighborhood)
    mask = cv2.bitwise_or(mask, cv2.bitwise_or(gold_mask, otsu_near_gold))

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
    closed_edges[marker_ignore > 0] = 0
    contour_source = cv2.bitwise_or(mask, closed_edges)
    contour_source[marker_ignore > 0] = 0

    measurement_mask = cv2.bitwise_or(gold_mask, otsu_near_gold)
    pair_data = contour_circle_detection(measurement_mask)
    used_fallback = False
    outer_contour = None
    inner_contour = None

    if pair_data is not None:
        outer, inner, outer_contour, inner_contour = pair_data
    else:
        pair = radial_circle_detection(measurement_mask)
        if pair is not None:
            outer, inner = pair
        else:
            pair = hough_circle_detection(gray, marker_ignore)
            used_fallback = pair is not None
            if pair is not None:
                outer, inner = pair
            else:
                raise RuntimeError("Could not detect valid OD/ID circles.")

    od_px = outer.diameter
    id_px = inner.diameter
    wall_px = (od_px - id_px) / 2.0

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "outer": asdict(outer),
        "inner": asdict(inner),
        "od_px": float(od_px),
        "id_px": float(id_px),
        "wall_thickness_px": float(wall_px),
        "scale_mm_per_px": float(scale) if scale is not None else None,
        "used_fallback": bool(used_fallback),
        "detection_mode": "bangle",
        "zoom_factor": 1.0,
    }

    if scale is not None:
        result.update(
            {
                "od_mm": float(od_px * scale),
                "id_mm": float(id_px * scale),
                "wall_thickness_mm": float(wall_px * scale),
            }
        )

    annotated = draw_results(img, result, outer_contour=outer_contour, inner_contour=inner_contour)
    output_path = _result_path(image_path)
    cv2.imwrite(str(output_path), annotated)
    result["annotated_path"] = str(output_path)

    if debug:
        _show_debug(img, gray, mask, edges, contour_source, annotated, otsu_mask, measurement_mask)

    return result


def _circle_from_result(data: dict[str, Any]) -> CircleInfo:
    ellipse_data = data.get("ellipse")
    if ellipse_data is not None:
        # data["ellipse"] might be a list of lists from JSON serialization
        ellipse = (
            (float(ellipse_data[0][0]), float(ellipse_data[0][1])),
            (float(ellipse_data[1][0]), float(ellipse_data[1][1])),
            float(ellipse_data[2]),
        )
    else:
        ellipse = None

    return CircleInfo(
        center=tuple(data["center"]),
        radius=float(data["radius"]),
        method=str(data["method"]),
        support_count=int(data.get("support_count", 0)),
        ellipse=ellipse,
    )


def draw_results(
    img: np.ndarray,
    result: dict[str, Any],
    outer_contour: Optional[np.ndarray] = None,
    inner_contour: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw OD/ID circles and measurement labels on a BGR image."""
    out = img.copy()
    outer = _circle_from_result(result["outer"])
    inner = _circle_from_result(result["inner"])

    # Colors: Green for Outer, Blue for Inner
    COLOR_OUTER = (0, 220, 0)
    COLOR_INNER = (255, 0, 0)

    # Draw raw contours if provided (thickness 2)
    if outer_contour is not None:
        cv2.drawContours(out, [outer_contour], -1, COLOR_OUTER, 2, cv2.LINE_AA)
    if inner_contour is not None:
        cv2.drawContours(out, [inner_contour], -1, COLOR_INNER, 2, cv2.LINE_AA)

    # Draw fits (Ellipse or Circle) - thickness 1
    for circle, color in [(outer, COLOR_OUTER), (inner, COLOR_INNER)]:
        if circle.ellipse is not None:
            cv2.ellipse(out, circle.ellipse, color, 1, cv2.LINE_AA)
        else:
            center = tuple(np.round(circle.center).astype(int))
            cv2.circle(out, center, int(round(circle.radius)), color, 1, cv2.LINE_AA)
        
        # Center point
        cv2.circle(out, tuple(np.round(circle.center).astype(int)), 3, color, -1, cv2.LINE_AA)

    # Annotate labels near center, like in dimension_calib_finale.py
    cx, cy = np.round(outer.center).astype(int)
    
    od_label = f"OD={result['od_mm']:.2f}mm" if result.get("od_mm") else f"OD={result['od_px']:.1f}px"
    id_label = f"ID={result['id_mm']:.2f}mm" if result.get("id_mm") else f"ID={result['id_px']:.1f}px"

    font_scale = 0.55 if outer.radius < 50 else 0.7
    text_thickness = 1 if outer.radius < 50 else 2
    outline_thickness = text_thickness + 2
    if outer.radius < 50:
        label_x = cx + int(round(outer.radius)) + 8
        max_label_width = max(
            cv2.getTextSize(od_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0][0],
            cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0][0],
        )
        if label_x + max_label_width >= out.shape[1] - 5:
            label_x = max(5, cx - int(round(outer.radius)) - max_label_width - 8)
        od_y = max(18, cy - 3)
        id_y = min(out.shape[0] - 5, cy + 18)
    else:
        label_x = max(5, cx - 90)
        od_y = max(18, cy - 12)
        id_y = min(out.shape[0] - 5, cy + 18)

    # Background outline for readability
    cv2.putText(out, od_label, (label_x, od_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(out, od_label, (label_x, od_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_OUTER, text_thickness, cv2.LINE_AA)
    
    cv2.putText(out, id_label, (label_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(out, id_label, (label_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_INNER, text_thickness, cv2.LINE_AA)

    return out


def _result_path(image_path: str | os.PathLike[str]) -> Path:
    path = Path(image_path)
    return path.with_name(f"{path.stem}_result.jpg")


def _show_debug(
    img: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    contour_source: np.ndarray,
    annotated: np.ndarray,
    otsu_mask: Optional[np.ndarray] = None,
    measurement_mask: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Debug plotting unavailable: {exc}")
        return

    panels = [
        ("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "rgb"),
        ("CLAHE + Bilateral", gray, "gray"),
        ("Otsu Mask", otsu_mask if otsu_mask is not None else mask, "gray"),
        ("Measurement Mask", measurement_mask if measurement_mask is not None else mask, "gray"),
        ("Multi-scale Canny", edges, "gray"),
        ("Contour Source", contour_source, "gray"),
        ("Annotated", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "rgb"),
    ]
    plt.figure(figsize=(16, 8))
    for idx, (title, data, mode) in enumerate(panels, start=1):
        plt.subplot(2, 4, idx)
        plt.title(title)
        plt.axis("off")
        if mode == "gray":
            plt.imshow(data, cmap="gray")
        else:
            plt.imshow(data)
    plt.tight_layout()
    plt.show()


def print_result(result: dict[str, Any]) -> None:
    """Console output requested by the CLI requirements."""
    outer = _circle_from_result(result["outer"])
    inner = _circle_from_result(result["inner"])

    print(f"OD: {result['od_px']:.3f} px")
    print(f"ID: {result['id_px']:.3f} px")
    print(f"Wall thickness: {result['wall_thickness_px']:.3f} px")
    if result.get("scale_mm_per_px") is not None:
        print(f"OD: {result['od_mm']:.3f} mm")
        print(f"ID: {result['id_mm']:.3f} mm")
        print(f"Wall thickness: {result['wall_thickness_mm']:.3f} mm")

    print(
        "Outer circle: "
        f"center=({outer.center[0]:.2f}, {outer.center[1]:.2f}), "
        f"radius={outer.radius:.2f} px, "
        f"method={outer.method}"
    )
    print(
        "Inner circle: "
        f"center=({inner.center[0]:.2f}, {inner.center[1]:.2f}), "
        f"radius={inner.radius:.2f} px, "
        f"method={inner.method}"
    )
    print(f"Annotated image: {result['annotated_path']}")
    if result.get("used_fallback"):
        print("Fallback: HoughCircles was used.")


# ----------------------------- PyQt6 GUI -----------------------------


class _QtImportError(RuntimeError):
    pass


def _load_pyqt6():
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QDoubleSpinBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - import depends on environment.
        raise _QtImportError("PyQt6 is required for the GUI: pip install PyQt6") from exc
    return locals()


def run_gui() -> int:
    qt = _load_pyqt6()
    QApplication = qt["QApplication"]
    QMainWindow = qt["QMainWindow"]
    QWidget = qt["QWidget"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QTextEdit = qt["QTextEdit"]
    QVBoxLayout = qt["QVBoxLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    QDoubleSpinBox = qt["QDoubleSpinBox"]
    QCheckBox = qt["QCheckBox"]
    QImage = qt["QImage"]
    QPixmap = qt["QPixmap"]
    Qt = qt["Qt"]

    class ImageLabel(QLabel):
        def __init__(self) -> None:
            super().__init__("Load an image to detect OD / ID")
            self._bgr: Optional[np.ndarray] = None
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(760, 520)
            self.setStyleSheet("background:#181818;color:#aaa;border:1px solid #444;")

        def set_bgr(self, image: np.ndarray) -> None:
            self._bgr = image.copy()
            self._refresh()

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._refresh()

        def _refresh(self) -> None:
            if self._bgr is None:
                return
            rgb = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    class BangleWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Bangle OD / ID Detector")
            self.resize(1100, 780)
            self.image_path: Optional[str] = None
            self.original: Optional[np.ndarray] = None
            self.annotated: Optional[np.ndarray] = None

            root = QWidget()
            layout = QVBoxLayout(root)

            controls = QHBoxLayout()
            self.btn_load = QPushButton("Load Image")
            self.btn_detect = QPushButton("Detect OD / ID")
            self.btn_save = QPushButton("Save Annotated")
            self.scale_check = QCheckBox("Use scale")
            self.scale_spin = QDoubleSpinBox()
            self.scale_spin.setDecimals(6)
            self.scale_spin.setRange(0.000001, 1000.0)
            self.scale_spin.setValue(0.085)
            self.scale_spin.setSuffix(" mm/px")
            self.btn_detect.setEnabled(False)
            self.btn_save.setEnabled(False)
            controls.addWidget(self.btn_load)
            controls.addWidget(self.btn_detect)
            controls.addWidget(self.scale_check)
            controls.addWidget(self.scale_spin)
            controls.addWidget(self.btn_save)
            controls.addStretch(1)

            self.image_label = ImageLabel()
            self.output = QTextEdit()
            self.output.setReadOnly(True)
            self.output.setMaximumHeight(150)
            self.output.setStyleSheet(
                "background:#101010;color:#8cffb2;font-family:Consolas,monospace;"
            )
            self.output.setPlainText("Load a bangle image, then click Detect OD / ID.")

            layout.addLayout(controls)
            layout.addWidget(self.image_label, 1)
            layout.addWidget(self.output)
            self.setCentralWidget(root)

            self.btn_load.clicked.connect(self.load_image)
            self.btn_detect.clicked.connect(self.detect)
            self.btn_save.clicked.connect(self.save_annotated)

        def load_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open bangle image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            )
            if not path:
                return
            image = cv2.imread(path)
            if image is None:
                QMessageBox.critical(self, "Load failed", "OpenCV could not load this image.")
                return
            self.image_path = path
            self.original = image
            self.annotated = None
            self.image_label.set_bgr(image)
            self.btn_detect.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.output.setPlainText(f"Loaded: {path}\nShape: {image.shape[1]} x {image.shape[0]} px")

        def detect(self) -> None:
            if not self.image_path:
                return
            scale = self.scale_spin.value() if self.scale_check.isChecked() else None
            try:
                result = detect_bangle(self.image_path, scale=scale, debug=False)
            except Exception as exc:
                QMessageBox.critical(self, "Detection failed", str(exc))
                return

            annotated = cv2.imread(result["annotated_path"])
            if annotated is not None:
                self.annotated = annotated
                self.image_label.set_bgr(annotated)
                self.btn_save.setEnabled(True)

            lines = [
                f"OD: {result['od_px']:.3f} px",
                f"ID: {result['id_px']:.3f} px",
                f"Wall thickness: {result['wall_thickness_px']:.3f} px",
            ]
            if scale is not None:
                lines.extend(
                    [
                        f"OD: {result['od_mm']:.3f} mm",
                        f"ID: {result['id_mm']:.3f} mm",
                        f"Wall thickness: {result['wall_thickness_mm']:.3f} mm",
                    ]
                )
            outer = _circle_from_result(result["outer"])
            inner = _circle_from_result(result["inner"])
            lines.extend(
                [
                    "",
                    f"Outer radius: {outer.radius:.2f} px | center=({outer.center[0]:.1f}, {outer.center[1]:.1f})",
                    f"Inner radius: {inner.radius:.2f} px | center=({inner.center[0]:.1f}, {inner.center[1]:.1f})",
                    f"Annotated saved: {result['annotated_path']}",
                ]
            )
            if result.get("used_fallback"):
                lines.append("Fallback: HoughCircles was used.")
            self.output.setPlainText("\n".join(lines))

        def save_annotated(self) -> None:
            if self.annotated is None:
                return
            default = _result_path(self.image_path or "bangle.jpg")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save annotated image", str(default), "JPEG (*.jpg);;PNG (*.png)"
            )
            if path:
                cv2.imwrite(path, self.annotated)
                self.output.append(f"\nSaved copy: {path}")

    app = QApplication(sys.argv)
    window = BangleWindow()
    window.show()
    return app.exec()


# ----------------------------- CLI -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect bangle OD/ID from an image.")
    parser.add_argument("--image", help="Path to bangle image.")
    parser.add_argument("--scale", type=float, default=None, help="Optional mm/pixel scale.")
    parser.add_argument("--debug", action="store_true", help="Show matplotlib debug panels.")
    parser.add_argument("--gui", action="store_true", help="Open PyQt6 GUI.")
    args = parser.parse_args()

    if args.gui or not args.image:
        return run_gui()

    result = detect_bangle(args.image, scale=args.scale, debug=args.debug)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
