"""Decode LCD OBB outputs and apply rotated non-maximum suppression."""

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RotatedDetection:
    center_x: float
    center_y: float
    width: float
    height: float
    angle_radians: float
    confidence: float

    @property
    def angle_degrees(self):
        return math.degrees(self.angle_radians)

    def rotated_rect(self):
        return (
            (float(self.center_x), float(self.center_y)),
            (float(self.width), float(self.height)),
            float(self.angle_degrees),
        )

    def corners(self):
        return cv2.boxPoints(self.rotated_rect()).astype(np.float32)

    def as_dict(self):
        result = asdict(self)
        result["angle_degrees"] = self.angle_degrees
        result["corners"] = self.corners().tolist()
        return result


def sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def decode_angle(angle_logits):
    """Decode standard Ultralytics YOLOv8/YOLO11 OBB angle logits."""
    return (sigmoid(angle_logits) - 0.25) * np.pi


def decode_dfl(regression_logits, reg_max: int = 16):
    logits = np.asarray(regression_logits, dtype=np.float32)
    if logits.shape[-1] != 4 * reg_max:
        raise ValueError(
            f"DFL expects {4 * reg_max} channels, received {logits.shape[-1]}"
        )
    logits = logits.reshape(*logits.shape[:-1], 4, reg_max)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32)
    return np.sum(probabilities * bins, axis=-1)


def decode_scale(
    regression,
    classification,
    angle,
    stride: int,
    confidence_threshold: float,
    reg_max: int = 16,
):
    regression = np.asarray(regression)
    classification = np.asarray(classification)
    angle = np.asarray(angle)
    if regression.ndim != 3 or regression.shape[2] != 4 * reg_max:
        raise ValueError(f"Unexpected regression shape: {regression.shape}")
    expected_head_shape = regression.shape[:2] + (1,)
    if classification.shape != expected_head_shape:
        raise ValueError(f"Unexpected classification shape: {classification.shape}")
    if angle.shape != expected_head_shape:
        raise ValueError(f"Unexpected angle shape: {angle.shape}")

    scores = sigmoid(classification[..., 0])
    grid_y, grid_x = np.nonzero(scores >= confidence_threshold)
    if grid_x.size == 0:
        return []

    distances = decode_dfl(regression[grid_y, grid_x], reg_max)
    predicted_angles = decode_angle(angle[grid_y, grid_x, 0])
    anchor_x = grid_x.astype(np.float32) + 0.5
    anchor_y = grid_y.astype(np.float32) + 0.5

    offset_x = (distances[:, 2] - distances[:, 0]) / 2.0
    offset_y = (distances[:, 3] - distances[:, 1]) / 2.0
    cosine = np.cos(predicted_angles)
    sine = np.sin(predicted_angles)
    center_x = (offset_x * cosine - offset_y * sine + anchor_x) * stride
    center_y = (offset_x * sine + offset_y * cosine + anchor_y) * stride
    widths = (distances[:, 0] + distances[:, 2]) * stride
    heights = (distances[:, 1] + distances[:, 3]) * stride

    return [
        RotatedDetection(
            center_x=float(center_x[index]),
            center_y=float(center_y[index]),
            width=float(widths[index]),
            height=float(heights[index]),
            angle_radians=float(predicted_angles[index]),
            confidence=float(scores[grid_y[index], grid_x[index]]),
        )
        for index in range(grid_x.size)
        if widths[index] > 0 and heights[index] > 0
    ]


def decode_outputs(outputs, head_specs, confidence_threshold: float, reg_max: int = 16):
    detections = []
    for regression_name, classification_name, angle_name, stride in head_specs:
        missing = [
            name
            for name in (regression_name, classification_name, angle_name)
            if name not in outputs
        ]
        if missing:
            raise KeyError(f"Missing OBB output tensors: {', '.join(missing)}")
        detections.extend(
            decode_scale(
                outputs[regression_name],
                outputs[classification_name],
                outputs[angle_name],
                stride,
                confidence_threshold,
                reg_max,
            )
        )
    return detections


def decode_onnx_output(output, confidence_threshold: float):
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[0] != 6:
        raise ValueError(f"Unexpected decoded ONNX OBB shape: {predictions.shape}")

    accepted = np.flatnonzero(predictions[4] >= confidence_threshold)
    return [
        RotatedDetection(
            center_x=float(predictions[0, index]),
            center_y=float(predictions[1, index]),
            width=float(predictions[2, index]),
            height=float(predictions[3, index]),
            angle_radians=float(predictions[5, index]),
            confidence=float(predictions[4, index]),
        )
        for index in accepted
        if predictions[2, index] > 0 and predictions[3, index] > 0
    ]


def rotated_nms(detections, score_threshold: float, iou_threshold: float):
    if not detections:
        return []
    indices = cv2.dnn.NMSBoxesRotated(
        [detection.rotated_rect() for detection in detections],
        [float(detection.confidence) for detection in detections],
        float(score_threshold),
        float(iou_threshold),
    )
    if len(indices) == 0:
        return []
    kept = [detections[int(index)] for index in np.asarray(indices).reshape(-1)]
    return sorted(kept, key=lambda detection: detection.confidence, reverse=True)


def draw_detections(model_rgb, detections):
    debug_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
    for index, detection in enumerate(detections):
        color = (0, 255, 0) if index == 0 else (0, 200, 255)
        polygon = np.round(detection.corners()).astype(np.int32)
        cv2.polylines(debug_bgr, [polygon], True, color, 2, cv2.LINE_AA)
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        cv2.circle(debug_bgr, center, 3, color, -1, cv2.LINE_AA)
        label = (
            f"LCD {detection.confidence:.3f} "
            f"c=({detection.center_x:.1f},{detection.center_y:.1f}) "
            f"wh=({detection.width:.1f},{detection.height:.1f}) "
            f"a={detection.angle_degrees:.1f}deg"
        )
        label_y = max(18, int(polygon[:, 1].min()) - 6)
        cv2.putText(
            debug_bgr,
            label,
            (5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_bgr,
            label,
            (5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug_bgr
