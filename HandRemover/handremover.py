from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HAND_HEF_PATH = Path(__file__).resolve().with_name("handremover.hef")
HAND_REMOVAL_PIPELINE_VERSION = "2026-06-30-bangle-envelope-v2"
_HAND_MODEL: Any = None
_OWNED_HAILO_RUNTIME: Any = None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


HAND_CONFIDENCE = min(0.95, max(0.05, _env_float("HAND_HEF_CONFIDENCE", 0.25)))
HAND_MASK_THRESHOLD = min(0.95, max(0.05, _env_float("HAND_MASK_THRESHOLD", 0.55)))
HAND_NMS_IOU = min(0.95, max(0.05, _env_float("HAND_NMS_IOU", 0.50)))
MIN_OUTPUT_COMPONENT_AREA = max(1, _env_int("HAND_OUTPUT_MIN_AREA", 12))
JEWELRY_MIN_SATURATION = max(
    0,
    min(255, _env_int("HAND_JEWELRY_MIN_SATURATION", 34)),
)
JEWELRY_MAX_FRAME_FRACTION = min(
    0.80,
    max(0.01, _env_float("HAND_JEWELRY_MAX_FRAME_FRACTION", 0.22)),
)
JEWELRY_GRABCUT_ITERATIONS = max(
    1,
    min(8, _env_int("HAND_JEWELRY_GRABCUT_ITERATIONS", 3)),
)
JEWELRY_ENVELOPE_PADDING = max(
    2,
    min(30, _env_int("HAND_JEWELRY_ENVELOPE_PADDING", 5)),
)


def get_hand_model(hailo_runtime: Any = None) -> Any:
    """Return the configured hand-segmentation HEF model.

    The integrated app passes its shared HailoRuntime during startup. A private
    runtime is created only for standalone use of this module.
    """
    global _HAND_MODEL, _OWNED_HAILO_RUNTIME

    if _HAND_MODEL is not None:
        return _HAND_MODEL
    if not HAND_HEF_PATH.is_file():
        raise FileNotFoundError(f"Hand removal HEF not found at {HAND_HEF_PATH}")

    runtime = hailo_runtime
    if runtime is None:
        from hailo_model_runner import (  # Imported lazily for non-Hailo dev PCs.
            DEFAULT_HAILO_BATCH_SIZE,
            DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
            HailoRuntime,
        )

        _OWNED_HAILO_RUNTIME = HailoRuntime()
        runtime = _OWNED_HAILO_RUNTIME
        timeout_ms = DEFAULT_HAILO_INFERENCE_TIMEOUT_MS
        batch_size = DEFAULT_HAILO_BATCH_SIZE
    else:
        timeout_ms = None
        batch_size = 1

    _HAND_MODEL = runtime.create_model(
        str(HAND_HEF_PATH),
        "HandRemoval",
        timeout_ms=timeout_ms,
        batch_size=batch_size,
    )
    if _HAND_MODEL is None:
        detail = str(getattr(runtime, "last_model_error", "") or "unknown Hailo error")
        raise RuntimeError(f"Could not configure the hand-removal HEF: {detail}")
    return _HAND_MODEL


def _letterbox(
    bgr: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    height, width = bgr.shape[:2]
    scale = min(target_w / float(width), target_h / float(height))
    resized_w = max(1, min(target_w, int(round(width * scale))))
    resized_h = max(1, min(target_h, int(round(height * scale))))
    resized = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, (left, top, resized_w, resized_h)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax_last(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / (np.sum(exponentials, axis=-1, keepdims=True) + 1e-9)


def _nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int = 20,
) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_w = np.maximum(
            0.0,
            np.minimum(x2[index], x2[remaining]) - np.maximum(x1[index], x1[remaining]),
        )
        intersection_h = np.maximum(
            0.0,
            np.minimum(y2[index], y2[remaining]) - np.maximum(y1[index], y1[remaining]),
        )
        intersection = intersection_w * intersection_h
        union = areas[index] + areas[remaining] - intersection + 1e-6
        order = remaining[(intersection / union) <= iou_threshold]
    return np.asarray(keep, dtype=np.int32)


def _hailo_output_hwc(output: np.ndarray) -> np.ndarray:
    tensor = np.asarray(output)
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor[:, :, None]
    if (
        tensor.ndim == 3
        and tensor.shape[1] == tensor.shape[2]
        and tensor.shape[0] != tensor.shape[1]
    ):
        tensor = np.transpose(tensor, (1, 2, 0))
    if tensor.ndim != 3:
        raise RuntimeError(f"Unsupported hand-removal HEF output shape: {tensor.shape}")
    return tensor.astype(np.float32, copy=False)


def _decode_hailo_yolov8_seg(
    raw_outputs: dict[str, np.ndarray],
    input_h: int,
    input_w: int,
    confidence: float,
    iou_threshold: float,
) -> list[np.ndarray]:
    """Decode the 10 raw tensors emitted by the handremover YOLOv8-seg HEF."""
    tensors = [(name, _hailo_output_hwc(value)) for name, value in raw_outputs.items()]
    if len(tensors) != 10:
        shapes = {name: tuple(tensor.shape) for name, tensor in tensors}
        raise RuntimeError(
            "Hand-removal HEF must expose 10 raw YOLOv8-seg outputs; "
            f"received {len(tensors)}: {shapes}"
        )

    proto_name, prototypes = max(
        tensors,
        key=lambda item: (
            item[1].shape[0] * item[1].shape[1],
            -abs(item[1].shape[2] - 32),
        ),
    )
    prototype_channels = int(prototypes.shape[2])
    grouped: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
    for name, tensor in tensors:
        if name != proto_name:
            grouped.setdefault(tuple(tensor.shape[:2]), []).append((name, tensor))

    if len(grouped) != 3:
        raise RuntimeError(
            f"Unexpected hand-removal feature-map groups: {sorted(grouped)}"
        )

    decoded_boxes: list[np.ndarray] = []
    score_heads: list[np.ndarray] = []
    coefficient_heads: list[np.ndarray] = []

    for (height, width), scale_tensors in sorted(grouped.items()):
        bbox_candidates = [
            item
            for item in scale_tensors
            if item[1].shape[2] > prototype_channels
            and item[1].shape[2] % 4 == 0
        ]
        if len(scale_tensors) != 3 or len(bbox_candidates) != 1:
            raise RuntimeError(
                f"Unexpected hand-removal outputs at {height}x{width}: "
                f"{[(name, tensor.shape) for name, tensor in scale_tensors]}"
            )

        bbox_name, bbox_tensor = bbox_candidates[0]
        remaining = [item for item in scale_tensors if item[0] != bbox_name]
        coefficient_candidates = [
            item for item in remaining if item[1].shape[2] == prototype_channels
        ]
        if len(coefficient_candidates) != 1:
            raise RuntimeError(
                f"Could not identify mask coefficients at {height}x{width}"
            )
        coefficient_name, coefficient_tensor = coefficient_candidates[0]
        class_tensors = [item for item in remaining if item[0] != coefficient_name]
        if len(class_tensors) != 1:
            raise RuntimeError(f"Could not identify class scores at {height}x{width}")

        channels = int(bbox_tensor.shape[2])
        regression_length = channels // 4
        distribution = bbox_tensor.reshape(-1, 4, regression_length)
        distances = np.sum(
            _softmax_last(distribution)
            * np.arange(regression_length, dtype=np.float32).reshape(1, 1, -1),
            axis=-1,
        )
        stride_x = input_w / float(width)
        stride_y = input_h / float(height)
        distances[:, (0, 2)] *= stride_x
        distances[:, (1, 3)] *= stride_y

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32) + 0.5,
            np.arange(height, dtype=np.float32) + 0.5,
        )
        centers = np.stack(
            (
                grid_x.reshape(-1) * stride_x,
                grid_y.reshape(-1) * stride_y,
                grid_x.reshape(-1) * stride_x,
                grid_y.reshape(-1) * stride_y,
            ),
            axis=1,
        )
        decoded_boxes.append(
            centers + np.concatenate((-distances[:, :2], distances[:, 2:]), axis=1)
        )
        score_heads.append(class_tensors[0][1].reshape(-1, class_tensors[0][1].shape[2]))
        coefficient_heads.append(
            coefficient_tensor.reshape(-1, prototype_channels)
        )

    boxes = np.concatenate(decoded_boxes, axis=0)
    scores_by_class = np.concatenate(score_heads, axis=0)
    if scores_by_class.size and (
        float(np.min(scores_by_class)) < 0.0
        or float(np.max(scores_by_class)) > 1.0
    ):
        scores_by_class = _sigmoid(scores_by_class)
    scores = np.max(scores_by_class, axis=1)
    coefficients = np.concatenate(coefficient_heads, axis=0)

    candidate_indices = np.flatnonzero(scores >= confidence)
    if candidate_indices.size == 0:
        return []
    candidate_boxes = boxes[candidate_indices]
    candidate_scores = scores[candidate_indices]
    selected = candidate_indices[
        _nms_xyxy(candidate_boxes, candidate_scores, iou_threshold)
    ]

    prototype_flat = prototypes.reshape(-1, prototype_channels).T
    prototype_masks = _sigmoid(coefficients[selected] @ prototype_flat).reshape(
        -1,
        prototypes.shape[0],
        prototypes.shape[1],
    )

    masks: list[np.ndarray] = []
    selected_boxes = boxes[selected]
    for prototype_mask, box in zip(prototype_masks, selected_boxes):
        full_mask = cv2.resize(
            prototype_mask,
            (input_w, input_h),
            interpolation=cv2.INTER_LINEAR,
        )
        x1 = int(np.clip(np.floor(box[0]), 0, input_w))
        y1 = int(np.clip(np.floor(box[1]), 0, input_h))
        x2 = int(np.clip(np.ceil(box[2]), 0, input_w))
        y2 = int(np.clip(np.ceil(box[3]), 0, input_h))
        cropped = np.zeros_like(full_mask)
        if x2 > x1 and y2 > y1:
            cropped[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]
        masks.append(cropped)
    return masks


def _infer_hand_probability(bgr: np.ndarray) -> np.ndarray:
    model = get_hand_model()
    input_h = int(model.input_h)
    input_w = int(model.input_w)
    padded, _, (left, top, resized_w, resized_h) = _letterbox(
        bgr,
        input_h,
        input_w,
    )

    # Ultralytics YOLO models are trained with RGB input. Keep uint8/HWC so
    # HailoRT can use the HEF's native input buffer without host normalization.
    rgb = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    raw_outputs = model.run_inference(rgb)
    if not isinstance(raw_outputs, dict):
        raise RuntimeError(
            "Hand-removal HEF inference did not return named output tensors."
        )
    network_masks = _decode_hailo_yolov8_seg(
        raw_outputs,
        input_h,
        input_w,
        HAND_CONFIDENCE,
        HAND_NMS_IOU,
    )
    if not network_masks:
        return np.zeros(bgr.shape[:2], dtype=np.float32)

    combined = np.maximum.reduce(network_masks)
    unpadded = combined[top : top + resized_h, left : left + resized_w]
    if unpadded.size == 0:
        return np.zeros(bgr.shape[:2], dtype=np.float32)
    return cv2.resize(
        unpadded,
        (bgr.shape[1], bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _otsu_mask(channel: np.ndarray) -> tuple[np.ndarray, float]:
    channel_u8 = np.clip(channel, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(channel_u8, (5, 5), 0)
    threshold, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return mask, float(threshold)


def _scale_to_u8(values: np.ndarray, upper_percentile: float = 99.5) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    scale = float(np.percentile(finite, upper_percentile))
    if scale <= 1e-6:
        return np.zeros(finite.shape, dtype=np.uint8)
    return np.clip(finite * (255.0 / scale), 0, 255).astype(np.uint8)


def _estimate_paper_color(
    bgr: np.ndarray,
    raw_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the illuminated paper color from neutral border pixels."""
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    border_y = max(8, int(round(height * 0.10)))
    border_x = max(8, int(round(width * 0.10)))
    border = np.zeros((height, width), dtype=np.uint8)
    border[:border_y, :] = 255
    border[-border_y:, :] = 255
    border[:, :border_x] = 255
    border[:, -border_x:] = 255

    blocked = cv2.dilate(raw_hand, np.ones((11, 11), np.uint8), iterations=1)
    sample_mask = (
        (border > 0)
        & (blocked == 0)
        & (saturation <= 65)
        & (value >= 85)
    )
    if int(np.count_nonzero(sample_mask)) < 500:
        sample_mask = (
            (blocked == 0)
            & (saturation <= 55)
            & (value >= 110)
        )
    samples = lab[sample_mask]
    if len(samples) < 50:
        paper_lab = np.array([235.0, 128.0, 128.0], dtype=np.float32)
    else:
        paper_lab = np.median(samples, axis=0).astype(np.float32)
    return paper_lab, (sample_mask.astype(np.uint8) * 255)


def _keep_seed_components_near_hand(
    seed: np.ndarray,
    raw_hand: np.ndarray,
) -> np.ndarray:
    """Reject isolated colored objects that are unrelated to the held jewel."""
    if cv2.countNonZero(seed) == 0:
        return seed
    height, width = seed.shape
    if cv2.countNonZero(raw_hand) > 0:
        outside_hand = cv2.bitwise_not(raw_hand)
        hand_distance = cv2.distanceTransform(outside_hand, cv2.DIST_L2, 5)
        maximum_distance = max(12.0, min(height, width) * 0.09)
    else:
        hand_distance = np.zeros(seed.shape, dtype=np.float32)
        maximum_distance = float(max(height, width))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    kept = np.zeros_like(seed)
    center_x = width * 0.5
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        component = labels == label
        distance = float(np.min(hand_distance[component]))
        x = int(stats[label, cv2.CC_STAT_LEFT])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_center_x = x + component_width * 0.5
        center_distance = abs(component_center_x - center_x) / max(1.0, width)
        if distance <= maximum_distance and center_distance <= 0.42:
            kept[component] = 255
    return kept


def _seed_roi(seed: np.ndarray) -> np.ndarray:
    """Build a padded work area around the reliable gold/stone seed pixels."""
    height, width = seed.shape
    ys, xs = np.where(seed > 0)
    if len(xs) == 0:
        return np.full(seed.shape, 255, dtype=np.uint8)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    seed_width = max(1, x2 - x1)
    seed_height = max(1, y2 - y1)
    pad_x = max(12, int(round(seed_width * 0.80)), int(round(width * 0.012)))
    pad_y = max(12, int(round(seed_height * 0.35)), int(round(height * 0.012)))
    x1 = max(0, x1 - pad_x)
    x2 = min(width, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(height, y2 + pad_y)

    roi = np.zeros_like(seed)
    roi[y1:y2, x1:x2] = 255
    return roi


def _contact_shape_envelope(
    seed: np.ndarray,
    raw_hand: np.ndarray,
) -> np.ndarray | None:
    """Cut seed widening on the hand-facing side of a held jewel.

    This trim is useful when fingers become warm/colorful seed lobes beside the
    jewel. It is harmful for side-view bangles/rings where the hand-to-jewel
    vector follows the jewel's own long axis, so those cases fall back to the
    regular seed envelope below.
    """
    seed_ys, seed_xs = np.where(seed > 0)
    if len(seed_xs) < 20 or cv2.countNonZero(raw_hand) < 100:
        return None

    seed_points = np.column_stack((seed_xs, seed_ys)).astype(np.float32)
    seed_center = np.median(seed_points, axis=0)
    centered = seed_points - seed_center

    body_mask = raw_hand.copy()
    seed_exclusion = cv2.dilate(seed, np.ones((7, 7), np.uint8), iterations=1)
    body_mask[seed_exclusion > 0] = 0
    body_ys, body_xs = np.where(body_mask > 0)
    if len(body_xs) < 80:
        return None
    body_center = np.array(
        [np.median(body_xs), np.median(body_ys)],
        dtype=np.float32,
    )
    away_axis = seed_center - body_center
    norm = float(np.linalg.norm(away_axis))
    if norm < 8.0:
        return None
    away_axis /= norm

    covariance = np.cov(centered, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        eigenvalues = np.array([1.0, 1.0], dtype=np.float32)
        eigenvectors = np.eye(2, dtype=np.float32)
    order = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, order[0]].astype(np.float32)
    major_variance = max(1e-6, float(eigenvalues[order[0]]))
    minor_variance = max(1e-6, float(eigenvalues[order[1]]))
    elongation = np.sqrt(major_variance / minor_variance)
    axis_alignment = abs(float(np.dot(away_axis, major_axis)))
    if elongation >= 1.65 and axis_alignment >= 0.82:
        return None

    side_axis = np.array([-away_axis[1], away_axis[0]], dtype=np.float32)

    away_projection = centered @ away_axis
    side_projection = centered @ side_axis
    projection_span = float(np.ptp(away_projection))
    if projection_span < 12.0:
        return None

    bin_width = max(2.0, min(seed.shape) * 0.003)
    edges = np.arange(
        float(np.min(away_projection)),
        float(np.max(away_projection)) + bin_width,
        bin_width,
        dtype=np.float32,
    )
    if len(edges) < 5:
        return None

    bin_records: list[tuple[float, float, float, int]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (away_projection >= lower) & (away_projection < upper)
        count = int(np.count_nonzero(in_bin))
        if count < 3:
            continue
        values = side_projection[in_bin]
        width = float(np.percentile(values, 95) - np.percentile(values, 5))
        bin_records.append((float(lower), float(upper), width, count))
    if len(bin_records) < 4:
        return None

    contact_limit = float(np.percentile(away_projection, 35.0))
    baseline_records = [
        record for record in bin_records if record[0] > contact_limit
    ]
    if len(baseline_records) < 2:
        return None
    baseline_width = float(
        np.percentile([record[2] for record in baseline_records], 70.0)
    )
    expansion_threshold = max(
        baseline_width * 1.40,
        baseline_width + JEWELRY_ENVELOPE_PADDING * 2.0,
    )
    expanded_contact_bins = [
        record
        for record in bin_records
        if record[1] <= contact_limit and record[2] > expansion_threshold
    ]
    if not expanded_contact_bins:
        return None

    cutoff = max(record[1] for record in expanded_contact_bins)
    core = away_projection >= cutoff
    if int(np.count_nonzero(core)) < max(12, int(len(seed_points) * 0.30)):
        return None

    core_away = away_projection[core]
    core_side = side_projection[core]
    padding = float(JEWELRY_ENVELOPE_PADDING)
    contact_aspect = projection_span / max(1.0, baseline_width)
    if contact_aspect >= 2.6:
        away_padding = max(5.0, padding * 2.0)
        side_half_width = baseline_width * 0.72 + max(4.0, padding * 0.8)
    else:
        away_padding = max(2.0, padding * 0.5)
        side_half_width = baseline_width * 0.42 + max(2.0, padding * 0.4)
    away_low = float(np.min(core_away)) - away_padding
    away_high = float(np.max(core_away)) + padding
    side_center = float(np.median(core_side))
    side_low = side_center - side_half_width
    side_high = side_center + side_half_width
    corners = np.array(
        [
            seed_center + away_axis * away_low + side_axis * side_low,
            seed_center + away_axis * away_low + side_axis * side_high,
            seed_center + away_axis * away_high + side_axis * side_high,
            seed_center + away_axis * away_high + side_axis * side_low,
        ],
        dtype=np.float32,
    )
    envelope = np.zeros(seed.shape, dtype=np.uint8)
    cv2.fillConvexPoly(envelope, np.round(corners).astype(np.int32), 255)
    envelope = cv2.dilate(
        envelope,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    seed_pixels = cv2.countNonZero(seed)
    retained_pixels = cv2.countNonZero(cv2.bitwise_and(seed, envelope))
    if seed_pixels and retained_pixels / float(seed_pixels) < 0.55:
        return None

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed,
        connectivity=8,
    )
    if count > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest = (labels == largest_label).astype(np.uint8) * 255
        largest_pixels = cv2.countNonZero(largest)
        largest_retained = cv2.countNonZero(cv2.bitwise_and(largest, envelope))
        if largest_pixels and largest_retained / float(largest_pixels) < 0.70:
            return None
    return envelope


def _seed_shape_envelope(
    seed: np.ndarray,
    raw_hand: np.ndarray | None = None,
) -> np.ndarray:
    """Build a robust shape prior around the jewelry color seeds.

    Contact fingers form side lobes at one end of the jewel. A robust PCA axis
    removes those lateral outliers before the retained seed hull is expanded by
    a small amount to recover reflective, low-saturation metal edges.
    """
    height, width = seed.shape
    ys, xs = np.where(seed > 0)
    if len(xs) < 8:
        return _seed_roi(seed)
    if raw_hand is not None:
        contact_envelope = _contact_shape_envelope(seed, raw_hand)
        if contact_envelope is not None:
            return contact_envelope

    shape_seed = seed
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed,
        connectivity=8,
    )
    if count > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
        seed_area = cv2.countNonZero(seed)
        if largest_area >= max(40, int(round(seed_area * 0.45))):
            shape_seed = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            ys, xs = np.where(shape_seed > 0)

    points_xy = np.column_stack((xs, ys)).astype(np.float32)
    center = np.median(points_xy, axis=0)
    centered = points_xy - center
    covariance = np.cov(centered, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        eigenvalues = np.array([1.0, 1.0], dtype=np.float32)
        eigenvectors = np.eye(2, dtype=np.float32)
    order = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, order[0]].astype(np.float32)
    minor_axis = eigenvectors[:, order[1]].astype(np.float32)

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    minor_center = float(np.median(minor_projection))
    minor_distance = np.abs(minor_projection - minor_center)
    minor_limit = max(2.0, float(np.percentile(minor_distance, 60.0)))
    major_low, major_high = np.percentile(major_projection, (1.0, 99.0))
    inliers = (
        (minor_distance <= minor_limit)
        & (major_projection >= major_low)
        & (major_projection <= major_high)
    )
    inlier_points = points_xy[inliers]
    if len(inlier_points) < max(8, int(len(points_xy) * 0.35)):
        inlier_points = points_xy

    envelope = np.zeros((height, width), dtype=np.uint8)
    major_variance = max(1e-6, float(eigenvalues[order[0]]))
    minor_variance = max(1e-6, float(eigenvalues[order[1]]))
    elongation = np.sqrt(major_variance / minor_variance)

    if elongation >= 1.65:
        padding = float(JEWELRY_ENVELOPE_PADDING)
        half_width = minor_limit + padding
        major_start = float(major_low) - padding * 1.5
        major_end = float(major_high) + padding * 1.5
        corners = np.array(
            [
                center + major_axis * major_start + minor_axis * (minor_center - half_width),
                center + major_axis * major_start + minor_axis * (minor_center + half_width),
                center + major_axis * major_end + minor_axis * (minor_center + half_width),
                center + major_axis * major_end + minor_axis * (minor_center - half_width),
            ],
            dtype=np.float32,
        )
        cv2.fillConvexPoly(envelope, np.round(corners).astype(np.int32), 255)
        envelope = cv2.dilate(
            envelope,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
    else:
        hull = cv2.convexHull(np.round(inlier_points).astype(np.int32))
        cv2.fillConvexPoly(envelope, hull, 255)
        hull_x, hull_y, hull_w, hull_h = cv2.boundingRect(hull)
        minor_size = max(1, min(hull_w, hull_h))
        padding = max(
            JEWELRY_ENVELOPE_PADDING,
            min(14, int(round(minor_size * 0.12))),
        )
        kernel_size = padding * 2 + 1
        envelope = cv2.dilate(
            envelope,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            ),
            iterations=1,
        )
    return envelope


def _grabcut_refine(
    bgr: np.ndarray,
    candidate: np.ndarray,
    seed: np.ndarray,
    roi: np.ndarray,
    raw_hand: np.ndarray,
    hand_removal: np.ndarray,
    paper_samples: np.ndarray,
) -> np.ndarray:
    """Refine Otsu foreground with HEF/paper/jewelry trimap seeds."""
    if cv2.countNonZero(seed) < 8 or cv2.countNonZero(candidate) < 20:
        return candidate

    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return candidate
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    image_crop = np.ascontiguousarray(bgr[y1:y2, x1:x2])
    candidate_crop = candidate[y1:y2, x1:x2]
    seed_crop = seed[y1:y2, x1:x2]
    raw_hand_crop = raw_hand[y1:y2, x1:x2]
    hand_removal_crop = hand_removal[y1:y2, x1:x2]
    paper_crop = paper_samples[y1:y2, x1:x2]
    roi_crop = roi[y1:y2, x1:x2]

    grab_mask = np.full(candidate_crop.shape, cv2.GC_BGD, dtype=np.uint8)
    grab_mask[roi_crop > 0] = cv2.GC_PR_BGD
    grab_mask[candidate_crop > 0] = cv2.GC_PR_FGD

    seed_guard = cv2.dilate(seed_crop, np.ones((5, 5), np.uint8), iterations=1)
    probable_hand = (raw_hand_crop > 0) & (seed_guard == 0)
    definite_hand = (hand_removal_crop > 0) & (seed_guard == 0)
    grab_mask[probable_hand] = cv2.GC_PR_BGD
    grab_mask[definite_hand] = cv2.GC_BGD
    grab_mask[paper_crop > 0] = cv2.GC_BGD

    definite_foreground = cv2.erode(
        seed_crop,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    if cv2.countNonZero(definite_foreground) < 4:
        definite_foreground = seed_crop
    grab_mask[seed_crop > 0] = cv2.GC_PR_FGD
    grab_mask[definite_foreground > 0] = cv2.GC_FGD

    if not np.any(grab_mask == cv2.GC_FGD) or not np.any(grab_mask == cv2.GC_BGD):
        return candidate

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image_crop,
            grab_mask,
            None,
            background_model,
            foreground_model,
            JEWELRY_GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return candidate

    refined_crop = np.where(
        (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    refined_crop = cv2.bitwise_or(refined_crop, seed_crop)
    refined_crop = cv2.bitwise_and(refined_crop, roi_crop)

    refined = np.zeros_like(candidate)
    refined[y1:y2, x1:x2] = refined_crop
    return refined


def _keep_primary_jewelry_cluster(
    mask: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    """Keep the seeded jewel cluster and discard remote/border artifacts."""
    if cv2.countNonZero(mask) == 0:
        return mask

    bridge = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        bridge,
        connectivity=8,
    )
    records: list[tuple[int, int, int]] = []
    for label in range(1, count):
        component = labels == label
        original_pixels = int(np.count_nonzero((mask > 0) & component))
        seed_pixels = int(np.count_nonzero((seed > 0) & component))
        if original_pixels < MIN_OUTPUT_COMPONENT_AREA or seed_pixels == 0:
            continue
        records.append((seed_pixels, original_pixels, label))
    if not records:
        return np.zeros_like(mask)

    records.sort(reverse=True)
    best_seed_pixels, best_area, best_label = records[0]
    selected_bridge = labels == best_label
    output = np.where((mask > 0) & selected_bridge, 255, 0).astype(np.uint8)

    # Retain another seeded cluster only when it is substantial and physically
    # close to the main jewel. This accommodates reflective gaps but eliminates
    # the thin unrelated line occasionally seen at the bottom image boundary.
    main_dilated = cv2.dilate(output, np.ones((9, 9), np.uint8), iterations=1)
    for seed_pixels, area, label in records[1:]:
        component = labels == label
        if seed_pixels < max(3, int(best_seed_pixels * 0.06)):
            continue
        component_original = (mask > 0) & component
        if area >= max(12, int(best_area * 0.05)) and np.any(
            component_original & (main_dilated > 0)
        ):
            output[component_original] = 255
    return output


def _select_seeded_components(
    candidate: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    """Keep Otsu foreground components containing reliable jewelry colors."""
    height, width = candidate.shape
    image_area = height * width
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )
    selected = np.zeros_like(candidate)
    maximum_area = max(100, int(round(image_area * JEWELRY_MAX_FRAME_FRACTION)))
    component_records: list[tuple[float, int, int]] = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_OUTPUT_COMPONENT_AREA or area > maximum_area:
            continue
        component = labels == label
        seed_pixels = int(np.count_nonzero(seed[component]))
        if seed_pixels < 2:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = (
            x <= 1
            or y <= 1
            or x + component_width >= width - 1
            or y + component_height >= height - 1
        )
        seed_fraction = seed_pixels / float(max(1, area))
        if touches_border and seed_fraction < 0.05:
            continue
        score = seed_pixels * 8.0 + seed_fraction * 500.0 + np.sqrt(area)
        component_records.append((score, label, area))

    if not component_records:
        return selected

    component_records.sort(reverse=True)
    best_score = component_records[0][0]
    for score, label, _ in component_records:
        if score >= max(18.0, best_score * 0.08):
            selected[labels == label] = 255

    # Recover small anti-aliased/highlight islands immediately next to the
    # selected metal without admitting distant background components.
    nearby = cv2.dilate(selected, np.ones((5, 5), np.uint8), iterations=1)
    for label in range(1, count):
        if np.any(selected[labels == label]):
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_OUTPUT_COMPONENT_AREA or area > maximum_area:
            continue
        component = labels == label
        if np.any((nearby > 0) & component):
            selected[component] = 255
    return selected


def _segment_jewelry_otsu(
    bgr: np.ndarray,
    hand_probability: np.ndarray,
    hand_removal: np.ndarray,
    jewelry_protection: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Segment only the held jewel using background distance and Otsu masks."""
    raw_hand = (hand_probability >= HAND_MASK_THRESHOLD).astype(np.uint8) * 255
    paper_lab, paper_samples = _estimate_paper_color(bgr, raw_hand)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    difference = lab - paper_lab.reshape(1, 1, 3)

    # Give chroma more weight than illumination so gray cast shadows are less
    # likely to become foreground than gold or colored stones.
    background_distance = np.sqrt(
        (difference[:, :, 0] * 0.55) ** 2
        + (difference[:, :, 1] * 1.65) ** 2
        + (difference[:, :, 2] * 1.65) ** 2
    )
    chroma_distance = np.sqrt(
        difference[:, :, 1] ** 2 + difference[:, :, 2] ** 2
    )
    distance_u8 = _scale_to_u8(background_distance)
    chroma_u8 = _scale_to_u8(chroma_distance)

    distance_otsu, _ = _otsu_mask(distance_u8)
    chroma_otsu, _ = _otsu_mask(chroma_u8)
    saturation_otsu, saturation_threshold = _otsu_mask(saturation)

    gold_saturation = max(
        JEWELRY_MIN_SATURATION,
        min(95, int(round(saturation_threshold * 0.65))),
    )
    yellow_gold = (
        (hue >= 7)
        & (hue <= 45)
        & (saturation >= gold_saturation)
        & (value >= 38)
        & (difference[:, :, 2] >= 3.0)
    )
    colored_stones = (
        (saturation >= max(62, int(round(saturation_threshold))))
        & (value >= 32)
        & (distance_otsu > 0)
    )
    warm_metal = (
        (difference[:, :, 2] >= 8.0)
        & (difference[:, :, 1] >= -5.0)
        & (saturation >= 24)
        & (value >= 45)
    )

    general_seed = (yellow_gold | colored_stones | warm_metal).astype(np.uint8) * 255
    # Inside the neural hand silhouette trust only the per-image jewelry
    # protection mask; this prevents brown skin from becoming a "gold" seed.
    outside_hand_seed = cv2.bitwise_and(general_seed, cv2.bitwise_not(raw_hand))
    seed = cv2.bitwise_or(outside_hand_seed, jewelry_protection)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    seed = _keep_seed_components_near_hand(seed, raw_hand)

    # Otsu provides the shape candidate; color is only the reliable seed. This
    # preserves low-saturation gold highlights connected to colored metal.
    candidate = cv2.bitwise_or(distance_otsu, chroma_otsu)
    candidate = cv2.bitwise_or(candidate, saturation_otsu)
    candidate = cv2.bitwise_or(candidate, seed)

    hand_cut = cv2.dilate(hand_removal, np.ones((3, 3), np.uint8), iterations=1)
    hand_cut = cv2.bitwise_and(hand_cut, cv2.bitwise_not(jewelry_protection))
    candidate[hand_cut > 0] = 0

    roi = _seed_roi(seed)
    candidate = cv2.bitwise_and(candidate, roi)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    candidate = _remove_small(candidate, MIN_OUTPUT_COMPONENT_AREA)
    grabcut_mask = _grabcut_refine(
        bgr,
        candidate,
        seed,
        roi,
        raw_hand,
        hand_removal,
        paper_samples,
    )
    shape_envelope = _seed_shape_envelope(seed, raw_hand)
    constrained_seed = cv2.bitwise_and(seed, shape_envelope)
    constrained_candidate = cv2.bitwise_and(grabcut_mask, shape_envelope)
    final_mask = _select_seeded_components(
        constrained_candidate,
        constrained_seed,
    )
    final_mask = _keep_primary_jewelry_cluster(final_mask, constrained_seed)
    # Recover a one-pixel reflective rim rejected by the conservative shape
    # prior, but only where GrabCut/Otsu still agree and the skin model does not.
    edge_band = cv2.dilate(
        final_mask,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    recoverable_edge = cv2.bitwise_and(edge_band, constrained_candidate)
    recoverable_edge = cv2.bitwise_and(
        recoverable_edge,
        cv2.bitwise_not(hand_removal),
    )
    final_mask = cv2.bitwise_or(final_mask, recoverable_edge)

    # If quantization or extreme lighting leaves no selected component, retain
    # the cleaned seed rather than returning hand/background pixels.
    if cv2.countNonZero(final_mask) == 0 and cv2.countNonZero(constrained_seed) > 0:
        final_mask = cv2.dilate(
            constrained_seed,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        final_mask = cv2.bitwise_and(final_mask, constrained_candidate)
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    # Final skin veto for small fingers left at the jewel contact point. Keep a
    # narrow guard around reliable jewelry colors so gold borders are retained.
    seed_guard = cv2.dilate(seed, np.ones((5, 5), np.uint8), iterations=1)
    residual_skin = cv2.bitwise_and(_generic_skin_mask(bgr), raw_hand)
    residual_skin = cv2.bitwise_and(residual_skin, cv2.bitwise_not(seed_guard))
    residual_skin = cv2.bitwise_and(
        residual_skin,
        cv2.bitwise_not(jewelry_protection),
    )
    final_mask[residual_skin > 0] = 0
    final_mask = _remove_small(final_mask, MIN_OUTPUT_COMPONENT_AREA)

    return final_mask, {
        "paper_samples": paper_samples,
        "background_distance": distance_u8,
        "distance_otsu": distance_otsu,
        "chroma_otsu": chroma_otsu,
        "saturation_otsu": saturation_otsu,
        "jewelry_seed": seed,
        "seed_roi": roi,
        "otsu_candidate": candidate,
        "grabcut_mask": grabcut_mask,
        "shape_envelope": shape_envelope,
        "constrained_candidate": constrained_candidate,
    }


def _generic_skin_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hue, saturation, value = cv2.split(hsv)
    _, cr, cb = cv2.split(ycrcb)

    ycrcb_skin = (
        (cr >= 123)
        & (cr <= 185)
        & (cb >= 68)
        & (cb <= 142)
    )
    hsv_skin = (
        ((hue <= 24) | (hue >= 172))
        & (saturation >= 18)
        & (value >= 25)
    )
    return (ycrcb_skin & hsv_skin).astype(np.uint8) * 255


def _build_skin_removal_mask(
    bgr: np.ndarray,
    hand_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain the neural hand mask to skin while protecting jewelry.

    YOLO can label a ring or bangle as part of the hand silhouette. We estimate
    the wearer's skin color from the lower/interior part of the detected hand,
    then remove only connected skin-like pixels. Gold, colored stones, and
    reflective highlights are explicitly protected.
    """
    raw_hand = (hand_probability >= HAND_MASK_THRESHOLD).astype(np.uint8) * 255
    if cv2.countNonZero(raw_hand) < 100:
        return np.zeros_like(raw_hand), np.zeros_like(raw_hand)

    raw_hand = cv2.morphologyEx(
        raw_hand,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    generic_skin = _generic_skin_mask(bgr)
    interior = cv2.erode(raw_hand, np.ones((5, 5), np.uint8), iterations=1)

    ys, xs = np.where(raw_hand > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    lower_start = y_min + int(round((y_max - y_min) * 0.42))
    lower_region = np.zeros_like(raw_hand)
    lower_region[lower_start : y_max + 1] = 255
    sample_mask = cv2.bitwise_and(interior, generic_skin)
    preferred_samples = cv2.bitwise_and(sample_mask, lower_region)
    if cv2.countNonZero(preferred_samples) >= 200:
        sample_mask = preferred_samples

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sample_pixels = lab[sample_mask > 0]
    if len(sample_pixels) < 80:
        # Conservative fallback: a valid model mask is still restricted by the
        # broad skin classifier, so jewelry is not removed with the silhouette.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        obvious_jewelry = (
            (
                (hue >= 16)
                & (hue <= 42)
                & (saturation >= 48)
                & (value >= 55)
            )
            | (
                (saturation >= 135)
                & (value >= 45)
                & ((hue >= 19) | (hue <= 3) | (hue >= 165))
            )
        ).astype(np.uint8) * 255
        obvious_jewelry = cv2.dilate(
            obvious_jewelry,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        obvious_jewelry = cv2.bitwise_and(obvious_jewelry, raw_hand)
        # Skin-colored pixels are not reliable jewelry evidence merely because
        # warm lighting placed them in the broad yellow range.
        definite_stone = (
            (saturation >= 135)
            & (value >= 45)
            & ((hue >= 19) | (hue <= 3) | (hue >= 165))
        )
        skin_veto = (generic_skin > 0) & (~definite_stone)
        obvious_jewelry[skin_veto] = 0
        skin_core = cv2.bitwise_and(raw_hand, generic_skin)
        removal = cv2.dilate(skin_core, np.ones((3, 3), np.uint8), iterations=1)
        removal = cv2.bitwise_and(removal, raw_hand)
        removal = cv2.bitwise_and(removal, cv2.bitwise_not(obvious_jewelry))
        return removal, obvious_jewelry

    median = np.median(sample_pixels, axis=0)
    mad = np.median(np.abs(sample_pixels - median), axis=0) * 1.4826
    robust_scale = np.maximum(mad, np.array([12.0, 4.0, 5.0], dtype=np.float32))
    difference = np.abs(lab - median)
    chroma_distance = np.sqrt(
        (difference[:, :, 1] / robust_scale[1]) ** 2
        + (difference[:, :, 2] / robust_scale[2]) ** 2
    )
    skin_color = (
        (difference[:, :, 0] <= robust_scale[0] * 3.5)
        & (chroma_distance <= 3.8)
    )

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    lab_b = lab[:, :, 2]

    # Yellow metal and saturated stones are unlike normal skin even when YOLO
    # includes them in the hand instance. A one-pixel expansion preserves their
    # anti-aliased border as well as the center color.
    yellow_metal = (
        (hue >= 16)
        & (hue <= 42)
        & (saturation >= 48)
        & (value >= 55)
    )
    colored_stone = (
        (saturation >= 135)
        & (value >= 45)
        & ((hue >= 19) | (hue <= 3) | (hue >= 165))
    )
    yellow_lab = (
        (lab_b >= median[2] + max(10.0, robust_scale[2] * 1.8))
        & (saturation >= 38)
        & (value >= 65)
    )
    jewelry_protection = (
        (yellow_metal | colored_stone | yellow_lab).astype(np.uint8) * 255
    )
    jewelry_protection = cv2.dilate(
        jewelry_protection,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    jewelry_protection = cv2.bitwise_and(jewelry_protection, raw_hand)
    sure_skin = skin_color & (generic_skin > 0) & (~colored_stone)
    jewelry_protection[sure_skin] = 0

    skin_core = (
        (skin_color & (generic_skin > 0) & (raw_hand > 0)).astype(np.uint8) * 255
    )
    # A small expansion fills nails and narrow model gaps but cannot grow
    # outside the neural hand silhouette or into protected jewelry.
    removal = cv2.dilate(skin_core, np.ones((5, 5), np.uint8), iterations=1)
    removal = cv2.bitwise_and(removal, raw_hand)
    removal = cv2.bitwise_and(removal, cv2.bitwise_not(jewelry_protection))
    return removal, jewelry_protection


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == label] = 255
    return output


def _write_debug_masks(
    output_path: str | os.PathLike[str],
    bgr: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    debug_dir = Path(output_path).with_suffix("")
    debug_dir = debug_dir.parent / f"{debug_dir.name}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "00_input.png"), bgr)
    for index, (name, mask) in enumerate(masks.items(), start=1):
        image = np.asarray(mask)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"{index:02d}_{name}.png"), image)


def extract_bangles(
    image_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    debug: bool = False,
    mask_output_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Remove hand/background and save only the held jewelry on white."""
    debug = debug or os.environ.get("HAND_REMOVAL_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image from {image_path}")

    hand_probability = _infer_hand_probability(bgr)
    hand_removal, jewelry_protection = _build_skin_removal_mask(
        bgr,
        hand_probability,
    )
    final_mask, otsu_debug = _segment_jewelry_otsu(
        bgr,
        hand_probability,
        hand_removal,
        jewelry_protection,
    )
    print(
        "[HandRemoval] "
        f"hand={cv2.countNonZero((hand_probability >= HAND_MASK_THRESHOLD).astype(np.uint8))}px "
        f"removed_skin={cv2.countNonZero(hand_removal)}px "
        f"jewelry_seed={cv2.countNonZero(otsu_debug['jewelry_seed'])}px "
        f"final_jewelry={cv2.countNonZero(final_mask)}px"
    )

    white_background = np.full_like(bgr, 255)
    final = np.where(final_mask[:, :, None] > 0, bgr, white_background)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), final):
        raise RuntimeError(f"Could not save hand-removed image to {output}")
    if mask_output_path is not None:
        mask_output = Path(mask_output_path)
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_output), final_mask):
            raise RuntimeError(f"Could not save jewelry mask to {mask_output}")

    if debug:
        debug_masks = {
            "hand_probability": hand_probability,
            "raw_hand": (hand_probability >= HAND_MASK_THRESHOLD).astype(np.uint8)
            * 255,
            "jewelry_protection": jewelry_protection,
            "hand_removal": hand_removal,
        }
        debug_masks.update(otsu_debug)
        debug_masks["final_mask"] = final_mask
        _write_debug_masks(
            output,
            bgr,
            debug_masks,
        )
    return True
