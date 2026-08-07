"""
Standalone testbed striping process for Hailo HEF models.

Runs the cover/bag detector inside the configured ROI first. Only after the
cover is fully laid flat in the testbed, measured by rectangularity, does it
start the strip/seal detector using strip-m.hef. The main jewel tracker is
left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QRect, QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRubberBand,
    QVBoxLayout,
    QWidget,
)


DEFAULT_HEF = "strip-m.hef"
DEFAULT_COVER_HEF = "bag.hef"
DEFAULT_ROI_CONFIG = "roi_config.json"
DEFAULT_STRIP_FP_MODEL = "hsv_fp_filter_strip.pt"
DEFAULT_STRIP_FP_CONFIDENCE = 0.60
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DEFAULT_FPS = 30.0
COVER_CONF_THRESHOLD = 0.50
STRIP_CONF_THRESHOLD = 0.80
CONF_THRESHOLD = STRIP_CONF_THRESHOLD
DEFAULT_RECT_THRESHOLD = 0.92
STRIP_DEBOUNCE_FRAMES = 4
STRIP_CONFIRM_FRAMES = 3
SEAL_CHECK_INTERVAL_SEC = 1.0
SEAL_GONE_CHECKS = 2
HAND_CLEAR_STABLE_SEC = 1.0
HAND_MOTION_PIXEL_THRESHOLD = 20
HAND_MOTION_FRACTION_THRESHOLD = 0.03
STRIP_REMOVAL_PIXEL_THRESHOLD = 25
STRIP_REMOVAL_CHANGE_FRACTION = 0.30
TARGET_STRIP_ZONE_PADDING = 0.10
TARGET_BAG_MASK_IOU = 0.50
TARGET_BAG_AREA_RATIO = (0.70, 1.30)
TARGET_STRIP_MASK_IOU = 0.20
TARGET_STRIP_AREA_RATIO = (0.50, 1.50)


def _make_strip_hsv_cnn():
    import torch.nn as nn

    class StripHSVCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 2),
            )

        def forward(self, tensor):
            return self.head(self.features(tensor))

    return StripHSVCNN()


class HSVFPFilter:
    def __init__(self, model_path, confidence=0.60):
        self.confidence = float(confidence)
        self.model = None
        self._torch = None
        if not model_path or not os.path.exists(model_path):
            print(
                f"[strip-fp] WARNING: checkpoint not found: {model_path}; "
                "secondary filter disabled, Hailo strip detections will pass through"
            )
            return
        try:
            import torch

            model = _make_strip_hsv_cnn()
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self._torch = torch
            print(f"[strip-fp] checkpoint loaded on CPU: {model_path}")
        except Exception as exc:
            print(
                f"[strip-fp] WARNING: could not load checkpoint {model_path}: {exc}; "
                "secondary filter disabled, Hailo strip detections will pass through"
            )

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def predict(self, crop_bgr) -> tuple[bool, float]:
        if self.model is None:
            return True, 1.0
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0
        try:
            hsv = cv2.cvtColor(
                cv2.resize(crop_bgr, (64, 64)),
                cv2.COLOR_BGR2HSV,
            ).astype(np.float32)
            hsv[:, :, 0] /= 180.0
            hsv[:, :, 1] /= 255.0
            hsv[:, :, 2] /= 255.0
            tensor = self._torch.from_numpy(
                hsv.transpose(2, 0, 1)
            ).unsqueeze(0)
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(self.model(tensor), dim=1)
            probability = float(probabilities[0, 1])
            return probability >= self.confidence, probability
        except Exception as exc:
            print(
                f"[strip-fp] WARNING: classifier inference failed: {exc}; "
                "secondary filter disabled, Hailo strip detections will pass through"
            )
            self.model = None
            self._torch = None
            return True, 1.0


class PreviewWindow(QMainWindow):
    def __init__(self, roi_config_path: str):
        super().__init__()
        self.closed = False
        self.restart_requested = False
        self.verify_removal_requested = False
        self.roi_config_path = roi_config_path
        self.roi = load_roi(roi_config_path)
        self._roi_drawing = False
        self._roi_origin = None
        self._rubber_band = None
        self._last_frame_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
        self._source_image = None

        self.setWindowTitle("Striping Process - strip-m.hef")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#111; border:1px solid #444;")
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        layout.addWidget(self.image_label, 1)

        controls = QHBoxLayout()
        self.draw_roi_button = QPushButton("Draw ROI")
        self.draw_roi_button.setCheckable(True)
        self.clear_roi_button = QPushButton("Clear ROI")
        self.verify_removal_button = QPushButton("Hand Clear - Verify Removal")
        self.verify_removal_button.setEnabled(False)
        self.restart_button = QPushButton("Restart Cycle")
        self.roi_info_label = QLabel()
        self.draw_roi_button.toggled.connect(self._toggle_roi_draw)
        self.clear_roi_button.clicked.connect(self._clear_roi)
        self.verify_removal_button.clicked.connect(self._request_removal_verification)
        self.restart_button.clicked.connect(self._request_restart)
        controls.addWidget(self.draw_roi_button)
        controls.addWidget(self.clear_roi_button)
        controls.addWidget(self.verify_removal_button)
        controls.addWidget(self.restart_button)
        controls.addWidget(self.roi_info_label)
        controls.addStretch()
        layout.addLayout(controls)

        self.setCentralWidget(root)
        self._update_roi_label()
        self.resize(CAMERA_WIDTH, CAMERA_HEIGHT + 70)
        self.show()

    def show_frame(self, frame: np.ndarray) -> None:
        self._last_frame_size = (frame.shape[1], frame.shape[0])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._source_image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._refresh_image()

    def _refresh_image(self) -> None:
        if self._source_image is None or self.image_label.width() <= 0:
            return
        pixmap = QPixmap.fromImage(self._source_image).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_image()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.image_label and self._roi_drawing:
            event_type = event.type()
            if (event_type == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                self._roi_origin = event.pos()
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(
                        QRubberBand.Shape.Rectangle, self.image_label)
                self._rubber_band.setGeometry(QRect(self._roi_origin, QSize()))
                self._rubber_band.show()
                return True
            if event_type == QEvent.Type.MouseMove and self._roi_origin is not None:
                self._rubber_band.setGeometry(
                    QRect(self._roi_origin, event.pos()).normalized())
                return True
            if (event_type == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton
                    and self._roi_origin is not None):
                rect = QRect(self._roi_origin, event.pos()).normalized()
                self._rubber_band.hide()
                self._roi_origin = None
                self._roi_drawing = False
                self.draw_roi_button.setChecked(False)
                if rect.width() > 4 and rect.height() > 4:
                    x1, y1 = self._label_to_frame(rect.left(), rect.top())
                    x2, y2 = self._label_to_frame(rect.right(), rect.bottom())
                    if x2 > x1 and y2 > y1:
                        self.roi = (x1, y1, x2, y2)
                        self._save_roi()
                        self._update_roi_label()
                return True
        return super().eventFilter(watched, event)

    def _label_to_frame(self, label_x: int, label_y: int) -> tuple[int, int]:
        frame_w, frame_h = self._last_frame_size
        label_w, label_h = self.image_label.width(), self.image_label.height()
        scale = min(label_w / frame_w, label_h / frame_h)
        shown_w = int(frame_w * scale)
        shown_h = int(frame_h * scale)
        offset_x = (label_w - shown_w) // 2
        offset_y = (label_h - shown_h) // 2
        frame_x = int(np.clip((label_x - offset_x) / scale, 0, frame_w - 1))
        frame_y = int(np.clip((label_y - offset_y) / scale, 0, frame_h - 1))
        return frame_x, frame_y

    def _toggle_roi_draw(self, checked: bool) -> None:
        self._roi_drawing = checked
        self.image_label.setCursor(
            Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor)
        if not checked and self._rubber_band is not None:
            self._rubber_band.hide()
            self._roi_origin = None

    def _clear_roi(self) -> None:
        self.roi = None
        self._save_roi()
        self._update_roi_label()

    def _save_roi(self) -> None:
        try:
            with open(self.roi_config_path, "w", encoding="utf-8") as f:
                json.dump({"roi": list(self.roi) if self.roi else None}, f)
        except OSError as exc:
            print(f"[roi] save error: {exc}")

    def _update_roi_label(self) -> None:
        if self.roi is None:
            self.roi_info_label.setText("No ROI set")
        else:
            x1, y1, x2, y2 = self.roi
            self.roi_info_label.setText(f"ROI: ({x1},{y1}) to ({x2},{y2})")

    def _request_restart(self) -> None:
        self.restart_requested = True

    def set_removal_verification_enabled(self, enabled: bool) -> None:
        self.verify_removal_button.setEnabled(enabled)
        if not enabled:
            self.verify_removal_requested = False

    def _request_removal_verification(self) -> None:
        self.verify_removal_requested = True
        self.verify_removal_button.setEnabled(False)

    def take_removal_verification_request(self) -> bool:
        requested = self.verify_removal_requested
        self.verify_removal_requested = False
        return requested

    def take_restart_request(self) -> bool:
        requested = self.restart_requested
        self.restart_requested = False
        return requested

    def closeEvent(self, event) -> None:
        self.closed = True
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)


def get_centroid(mask: np.ndarray):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def get_largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def measure_rectangularity(mask: np.ndarray) -> float:
    cnt = get_largest_contour(mask)
    if cnt is None:
        return 0.0
    area = float(cv2.contourArea(cnt))
    if area < 1.0:
        return 0.0
    _center, (w, h), _angle = cv2.minAreaRect(cnt)
    return area / max(float(w * h), 1.0)


def mask_inside_reference(
        mask: np.ndarray | None,
        reference_mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None or reference_mask is None:
        return None
    if mask.shape != reference_mask.shape:
        reference_mask = cv2.resize(
            reference_mask,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    centroid = get_centroid(mask)
    if centroid is None or reference_mask[centroid[1], centroid[0]] == 0:
        return None
    inside = cv2.bitwise_and(mask, reference_mask)
    return inside if np.any(inside) else None


def speak(text: str) -> None:
    def run() -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            try:
                subprocess.Popen(
                    ["espeak", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass

    threading.Thread(target=run, daemon=True).start()


def load_roi(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roi = data.get("roi")
    if not roi or len(roi) != 4:
        return None
    x1, y1, x2, y2 = (int(v) for v in roi)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def offset_mask(mask: np.ndarray | None, roi, full_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None or roi is None:
        return mask
    full_h, full_w = full_shape
    x1, y1, x2, y2 = roi
    out = np.zeros((full_h, full_w), dtype=np.uint8)
    out[y1:y2, x1:x2] = mask
    return out


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k = max(5, int(min(mask.shape[:2]) * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnt = get_largest_contour(opened)
    if cnt is None:
        return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    return out


def make_target_zone(mask: np.ndarray | None, frame_shape: tuple[int, int]):
    if mask is None:
        return None
    contour = get_largest_contour(mask)
    if contour is None:
        return None
    x, y, width, height = cv2.boundingRect(contour)
    frame_h, frame_w = frame_shape
    pad_x = max(12, int(width * TARGET_STRIP_ZONE_PADDING))
    pad_y = max(12, int(height * TARGET_STRIP_ZONE_PADDING))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(frame_w, x + width + pad_x),
        min(frame_h, y + height + pad_y),
    )


def mask_in_target_zone(mask: np.ndarray | None, zone):
    if mask is None or zone is None:
        return None
    x1, y1, x2, y2 = zone
    matched = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width // 2
        center_y = y + height // 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            cv2.drawContours(matched, [contour], -1, 255, cv2.FILLED)
    return matched if matched.sum() > 0 else None


def mask_matches_reference(reference: np.ndarray | None, candidate: np.ndarray | None,
                           min_iou: float, area_ratio_range: tuple[float, float]) -> bool:
    if reference is None or candidate is None:
        return False
    if reference.shape != candidate.shape:
        candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    reference_area = int(np.count_nonzero(reference))
    candidate_area = int(np.count_nonzero(candidate))
    if reference_area == 0 or candidate_area == 0:
        return False
    area_ratio = candidate_area / reference_area
    if not area_ratio_range[0] <= area_ratio <= area_ratio_range[1]:
        return False
    intersection = int(np.count_nonzero((reference > 0) & (candidate > 0)))
    union = int(np.count_nonzero((reference > 0) | (candidate > 0)))
    return union > 0 and intersection / union >= min_iou


def target_zone_motion_fraction(previous_gray: np.ndarray | None, frame: np.ndarray, zone):
    if zone is None:
        return None, None
    x1, y1, x2, y2 = zone
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    current_gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if previous_gray is None or previous_gray.shape != current_gray.shape:
        return current_gray, None
    difference = cv2.absdiff(previous_gray, current_gray)
    motion_fraction = float(np.mean(difference >= HAND_MOTION_PIXEL_THRESHOLD))
    return current_gray, motion_fraction


def target_zone_appearance_change_fraction(
        reference_gray: np.ndarray | None, frame: np.ndarray, zone) -> float | None:
    if reference_gray is None or zone is None:
        return None
    x1, y1, x2, y2 = zone
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    current_gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if reference_gray.shape != current_gray.shape:
        return None
    brightness_shift = float(np.median(current_gray) - np.median(reference_gray))
    normalized_current = np.clip(
        current_gray.astype(np.float32) - brightness_shift,
        0,
        255,
    ).astype(np.uint8)
    difference = cv2.absdiff(reference_gray, normalized_current)
    return float(np.mean(difference >= STRIP_REMOVAL_PIXEL_THRESHOLD))


def draw_seal_message(vis: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    height, width = vis.shape[:2]
    banner_height = max(80, height // 8)
    center_y = height // 2
    overlay = vis.copy()
    cv2.rectangle(
        overlay,
        (0, center_y - banner_height // 2),
        (width, center_y + banner_height // 2),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    scale = width / 900.0
    thickness = max(2, int(scale * 2.5))
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        vis,
        text,
        ((width - text_width) // 2, center_y + text_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


@dataclass
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    input_w: int
    input_h: int


def letterbox(frame: np.ndarray, input_w: int, input_h: int) -> tuple[np.ndarray, Letterbox]:
    h, w = frame.shape[:2]
    scale = min(input_w / w, input_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2

    canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, Letterbox(scale, pad_x, pad_y, new_w, new_h, input_w, input_h)


def undo_letterbox_mask(mask: np.ndarray, meta: Letterbox, out_size: tuple[int, int]) -> np.ndarray:
    out_w, out_h = out_size
    cropped = mask[meta.pad_y:meta.pad_y + meta.new_h, meta.pad_x:meta.pad_x + meta.new_w]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


class HailoSegModel:
    def __init__(self, hef_path: str, conf: float = CONF_THRESHOLD,
                 dump_outputs: bool = False, rgb_input: bool = True,
                 label: str = "model", hailo_model=None):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF model not found: {hef_path}")

        self.conf = float(conf)
        self.dump_outputs = dump_outputs
        self.rgb_input = rgb_input
        self.label = label
        self.last_confidence: float | None = None
        self._shared_hailo_model = hailo_model
        self.device = None

        # The integrated app owns one shared Hailo VDevice.  Reusing its
        # configured model keeps this standalone decoder usable in the packet
        # recording flow without competing for a second physical device.
        if hailo_model is not None:
            shape = tuple(int(x) for x in hailo_model.input_shape)
            if len(shape) != 3:
                raise RuntimeError(f"Unsupported HEF input shape: {shape}")
            self.input_h, self.input_w = shape[0], shape[1]
            self.input_name = str(hailo_model.input_name)
            self.output_names = list(hailo_model.output_names)
            self._InferVStreams = None
            print(f"[{self.label}] shared HEF loaded: {hef_path}")
            print(f"[{self.label}] input: {self.input_name} {shape}")
            return

        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HEF,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except Exception as exc:
            raise RuntimeError(
                "HailoRT Python package is not available. Run this on the "
                "Raspberry Pi/Hailo machine with hailo_platform installed."
            ) from exc

        self._InferVStreams = InferVStreams

        self.hef = HEF(hef_path)
        self.device = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, configure_params)[0]
        self.network_params = self.network_group.create_params()
        self.input_params = self._make_vstream_params(
            InputVStreamParams, FormatType.UINT8, quantized=True
        )
        self.output_params = self._make_vstream_params(
            OutputVStreamParams, FormatType.FLOAT32, quantized=False
        )

        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = input_info.name
        shape = tuple(int(x) for x in input_info.shape)
        if len(shape) != 3:
            raise RuntimeError(f"Unsupported HEF input shape: {shape}")
        self.input_h, self.input_w = shape[0], shape[1]
        self.output_names = [info.name for info in self.hef.get_output_vstream_infos()]

        print(f"[{self.label}] HEF loaded: {hef_path}")
        print(f"[{self.label}] input: {self.input_name} {shape}")
        for info in self.hef.get_output_vstream_infos():
            print(f"[{self.label}] output: {info.name} {tuple(info.shape)}")

    def _make_vstream_params(self, params_cls, format_type, quantized: bool):
        try:
            return params_cls.make_from_network_group(
                self.network_group, quantized=quantized, format_type=format_type
            )
        except AttributeError:
            return params_cls.make(self.network_group, format_type=format_type)

    def close(self) -> None:
        if self._shared_hailo_model is not None:
            return
        try:
            self.device.release()
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        self.last_confidence = None
        h, w = frame.shape[:2]
        canvas, meta = letterbox(frame, self.input_w, self.input_h)
        infer_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) if self.rgb_input else canvas
        batch = np.expand_dims(infer_img, axis=0).astype(np.uint8)

        if self._shared_hailo_model is not None:
            outputs = self._shared_hailo_model.run_inference(batch[0])
        else:
            with self.network_group.activate(self.network_params):
                with self._InferVStreams(
                    self.network_group, self.input_params, self.output_params
                ) as infer_pipeline:
                    outputs = infer_pipeline.infer({self.input_name: batch})

        if self.dump_outputs:
            for name, arr in outputs.items():
                print(f"[{self.label}] {name}: shape={np.asarray(arr).shape} dtype={np.asarray(arr).dtype}")
            self.dump_outputs = False

        mask = self._decode_outputs(outputs, self.input_w, self.input_h)
        if mask is None:
            return None
        mask = undo_letterbox_mask(mask, meta, (w, h))
        mask = clean_mask(mask)
        if mask is None:
            self.last_confidence = None
        return mask

    def _decode_outputs(self, outputs: dict, input_w: int, input_h: int) -> np.ndarray | None:
        # These HEFs expose the raw YOLOv8-seg head as three tensors per scale:
        # 64 DFL box values, one class-confidence value, and 32 mask
        # coefficients. A separate 96x96x32 tensor holds the mask prototypes.
        # They are not pre-decoded [x1, y1, x2, y2, score] rows.
        heads = {}
        for output in outputs.values():
            array = np.asarray(output, dtype=np.float32)
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 3:
                continue
            height, width, channels = array.shape
            head = heads.setdefault((height, width), {})
            if channels == 64:
                head["dfl"] = array
            elif channels == 1:
                head["scores"] = array[:, :, 0]
            elif channels == 32:
                head["coefficients"] = array

        prototypes = []
        detections = []
        for (grid_h, grid_w), head in heads.items():
            dfl = head.get("dfl")
            scores = head.get("scores")
            coefficients = head.get("coefficients")
            if dfl is None or scores is None or coefficients is None:
                if coefficients is not None:
                    prototypes.append(coefficients)
                continue

            stride_x = input_w / grid_w
            stride_y = input_h / grid_h
            for grid_y, grid_x in np.argwhere(scores >= self.conf):
                score = float(scores[grid_y, grid_x])
                if not np.isfinite(score):
                    continue
                distances = self._decode_dfl(dfl[grid_y, grid_x])
                center_x = (grid_x + 0.5) * stride_x
                center_y = (grid_y + 0.5) * stride_y
                x1 = int(np.clip(round(center_x - distances[0] * stride_x), 0, input_w - 1))
                y1 = int(np.clip(round(center_y - distances[1] * stride_y), 0, input_h - 1))
                x2 = int(np.clip(round(center_x + distances[2] * stride_x), 0, input_w - 1))
                y2 = int(np.clip(round(center_y + distances[3] * stride_y), 0, input_h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((score, x1, y1, x2, y2, coefficients[grid_y, grid_x]))

        if not detections:
            return None

        prototype = max(prototypes, key=lambda item: item.shape[0] * item.shape[1], default=None)
        selected = []
        for detection in sorted(detections, key=lambda item: item[0], reverse=True):
            if all(self._box_iou(detection[1:5], kept[1:5]) < 0.45 for kept in selected):
                selected.append(detection)

        combined = np.zeros((input_h, input_w), dtype=np.uint8)
        for _score, x1, y1, x2, y2, coefficients in selected:
            if prototype is None:
                cv2.rectangle(combined, (x1, y1), (x2, y2), 255, cv2.FILLED)
                continue
            mask_logits = np.tensordot(prototype, coefficients, axes=([2], [0]))
            mask_probability = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -50, 50)))
            proto_h, proto_w = mask_probability.shape
            px1 = int(np.clip(x1 * proto_w / input_w, 0, proto_w))
            py1 = int(np.clip(y1 * proto_h / input_h, 0, proto_h))
            px2 = int(np.clip(x2 * proto_w / input_w, 0, proto_w))
            py2 = int(np.clip(y2 * proto_h / input_h, 0, proto_h))
            mask_probability[:py1] = 0.0
            mask_probability[py2:] = 0.0
            mask_probability[:, :px1] = 0.0
            mask_probability[:, px2:] = 0.0
            mask = cv2.resize(mask_probability, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            combined[mask >= 0.5] = 255

        if combined.sum() <= 0:
            return None
        self.last_confidence = float(max(detection[0] for detection in selected))
        return combined

    @staticmethod
    def _decode_dfl(values: np.ndarray) -> np.ndarray:
        logits = np.asarray(values, dtype=np.float32).reshape(4, 16)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities @ np.arange(16, dtype=np.float32)

    @staticmethod
    def _box_iou(first, second) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        return intersection / max(first_area + second_area - intersection, 1)


class SegWorker(threading.Thread):
    def __init__(self, model: HailoSegModel,
                 strip_fp_filter: HSVFPFilter | None = None):
        super().__init__(daemon=True)
        self.model = model
        self.strip_fp_filter = strip_fp_filter
        self._in = queue.Queue(maxsize=1)
        self._out = queue.Queue(maxsize=1)
        self.last_confidence: float | None = None
        self.error = ""

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(frame.copy())

    def stop(self) -> None:
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(None)

    def get_result(self):
        try:
            result, confidence = self._out.get_nowait()
            self.last_confidence = confidence
            return result
        except queue.Empty:
            return None

    def run(self) -> None:
        while True:
            frame = self._in.get()
            if frame is None:
                return
            try:
                mask = self.model.detect(frame)
            except Exception as exc:
                self.error = str(exc)
                print(f"[{self.model.label}] inference stopped: {self.error}")
                return
            if mask is not None and self.strip_fp_filter is not None:
                contour = get_largest_contour(mask)
                if contour is None:
                    print("[strip-fp] empty contour accepted=False")
                    mask = None
                else:
                    x, y, width, height = cv2.boundingRect(contour)
                    frame_h, frame_w = frame.shape[:2]
                    x1 = max(0, min(x, frame_w))
                    y1 = max(0, min(y, frame_h))
                    x2 = max(0, min(x + width, frame_w))
                    y2 = max(0, min(y + height, frame_h))
                    crop = frame[y1:y2, x1:x2]
                    if x2 <= x1 or y2 <= y1 or crop.size == 0:
                        print("[strip-fp] empty bounding-box crop accepted=False")
                        mask = None
                    else:
                        accepted, probability = self.strip_fp_filter.predict(crop)
                        print(
                            f"[strip-fp] TP probability={probability:.2f} "
                            f"accepted={accepted}"
                        )
                        if not accepted:
                            mask = None
            result = (get_centroid(mask), mask) if mask is not None else (None, None)
            try:
                self._out.get_nowait()
            except queue.Empty:
                pass
            self._out.put((result, self.model.last_confidence))


def draw_status(vis: np.ndarray, state: str, strip_present: bool, miss_count: int,
                rect_score: float, rect_threshold: float,
                bag_present: bool | None, seal_gone_checks: int) -> None:
    colors = {
        "TRACKING": (50, 220, 50),
        "STRIP_MODE": (0, 200, 255),
        "STRIP_DETECTED": (0, 80, 255),
        "SEALED": (50, 220, 50),
        "BAG_REMOVED": (0, 0, 255),
    }
    color = colors.get(state, (180, 180, 180))
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 42), (20, 20, 20), cv2.FILLED)
    details = []
    if state == "TRACKING":
        details.append(f"rect={rect_score:.2f}/{rect_threshold:.2f}")
    if state == "STRIP_DETECTED":
        details.append(f"strip={'YES' if strip_present else 'NO'}")
        details.append(f"miss={miss_count}")
        if bag_present is not None:
            details.append(f"bag={'YES' if bag_present else 'NO'}")
        details.append(f"seal checks={seal_gone_checks}/{SEAL_GONE_CHECKS}")
    cv2.putText(
        vis,
        f"{state}  {'  '.join(details)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testbed striping process using strip-m.hef")
    parser.add_argument("--hef", default=DEFAULT_HEF, help="Path to strip HEF model")
    parser.add_argument("--cover-hef", default=DEFAULT_COVER_HEF, help="Path to cover/bag HEF model")
    parser.add_argument("--roi-config", default=DEFAULT_ROI_CONFIG, help="Path to roi_config.json")
    parser.add_argument("--rect-threshold", type=float, default=DEFAULT_RECT_THRESHOLD,
                        help="Cover rectangularity needed before strip detection starts")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--conf",
        type=float,
        default=STRIP_CONF_THRESHOLD,
        help="Strip detection confidence",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Requested camera FPS")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Requested camera width")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Requested camera height")
    parser.add_argument("--dump-outputs", action="store_true", help="Print HEF output tensor shapes once")
    parser.add_argument("--bgr-input", action="store_true", help="Send BGR input instead of RGB")
    parser.add_argument(
        "--strip-fp-model",
        default=DEFAULT_STRIP_FP_MODEL,
        help="Path to HSV strip false-positive checkpoint",
    )
    parser.add_argument(
        "--strip-fp-conf",
        type=float,
        default=DEFAULT_STRIP_FP_CONFIDENCE,
        help="Minimum class-1 probability for a genuine strip",
    )
    parser.add_argument(
        "--disable-strip-fp",
        action="store_true",
        help="Disable the secondary HSV strip false-positive filter",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strip_fp_filter = None
    if args.disable_strip_fp:
        print("[strip-fp] secondary filter disabled by command line")
    else:
        candidate = HSVFPFilter(args.strip_fp_model, args.strip_fp_conf)
        if candidate.enabled:
            strip_fp_filter = candidate
    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    app = QApplication.instance() or QApplication(sys.argv)
    preview = PreviewWindow(args.roi_config)

    if preview.roi is None:
        print(f"[roi] no valid ROI in {args.roi_config}; using full frame")
    else:
        print(f"[roi] loaded {preview.roi} from {args.roi_config}")

    active_model = None
    active_worker = None
    active_kind = None

    def stop_active_model() -> None:
        nonlocal active_kind, active_model, active_worker
        if active_worker is not None:
            active_worker.stop()
            active_worker.join()
            active_worker = None
        if active_model is not None:
            active_model.close()
            active_model = None
        active_kind = None

    def start_model(hef_path: str, label: str, dump_outputs: bool = False) -> None:
        nonlocal active_kind, active_model, active_worker
        stop_active_model()
        is_cover = label.startswith("cover")
        active_model = HailoSegModel(
            hef_path,
            conf=COVER_CONF_THRESHOLD if is_cover else args.conf,
            dump_outputs=dump_outputs,
            rgb_input=not args.bgr_input if is_cover else False,
            label=label,
        )
        active_worker = SegWorker(
            active_model,
            strip_fp_filter if label == "strip" else None,
        )
        active_worker.start()
        active_kind = label

    def roi_frame(frame: np.ndarray, roi) -> np.ndarray:
        if roi is None:
            return frame.copy()
        x1, y1, x2, y2 = roi
        return frame[y1:y2, x1:x2].copy()

    def offset_result(result, roi, full_shape):
        if result is None or roi is None:
            return result
        centroid, mask = result
        if centroid is not None:
            x1, y1, _x2, _y2 = roi
            centroid = (centroid[0] + x1, centroid[1] + y1)
        return centroid, offset_mask(mask, roi, full_shape)

    start_model(args.cover_hef, "cover", args.dump_outputs)

    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        stop_active_model()
        return 1

    cover_mask = None
    strip_present = False
    strip_mask = None
    strip_miss = 0
    rect_score = 0.0
    state = "TRACKING"
    bag_present_last = None
    seal_gone_checks = 0
    strip_confirm_count = 0
    target_bag_mask = None
    target_bag_zone = None
    target_strip_mask = None
    target_strip_zone = None
    removal_verification_started = False
    previous_strip_gray = None
    strip_stable_since = None
    verification_strip_misses = 0
    check_interval = max(1, int(SEAL_CHECK_INTERVAL_SEC * args.fps))
    check_remaining = check_interval
    last_submit = 0.0
    min_submit_dt = 1.0 / max(args.fps, 1.0)

    def start_cover_check(frame: np.ndarray, roi) -> None:
        nonlocal previous_strip_gray, strip_stable_since, verification_strip_misses
        previous_strip_gray = None
        strip_stable_since = None
        verification_strip_misses = 0
        start_model(args.cover_hef, "cover-check")
        active_worker.submit(roi_frame(frame, roi))

    def strip_zone_is_stable(frame: np.ndarray) -> bool:
        nonlocal previous_strip_gray, strip_stable_since
        previous_strip_gray, motion_fraction = target_zone_motion_fraction(
            previous_strip_gray, frame, target_strip_zone)
        if motion_fraction is None or motion_fraction > HAND_MOTION_FRACTION_THRESHOLD:
            strip_stable_since = None
            return False
        if strip_stable_since is None:
            strip_stable_since = time.perf_counter()
        return time.perf_counter() - strip_stable_since >= HAND_CLEAR_STABLE_SEC

    def restart_cycle() -> None:
        nonlocal bag_present_last, check_remaining, cover_mask
        nonlocal rect_score, seal_gone_checks, state, strip_mask, strip_miss, strip_present
        nonlocal strip_confirm_count, target_bag_mask, target_bag_zone
        nonlocal target_strip_mask, target_strip_zone, removal_verification_started
        nonlocal previous_strip_gray, strip_stable_since, verification_strip_misses
        state = "TRACKING"
        cover_mask = None
        strip_present = False
        strip_mask = None
        strip_miss = 0
        rect_score = 0.0
        bag_present_last = None
        seal_gone_checks = 0
        strip_confirm_count = 0
        target_bag_mask = None
        target_bag_zone = None
        target_strip_mask = None
        target_strip_zone = None
        removal_verification_started = False
        previous_strip_gray = None
        strip_stable_since = None
        verification_strip_misses = 0
        check_remaining = check_interval
        preview.set_removal_verification_enabled(False)
        start_model(args.cover_hef, "cover")
        print("[state] restarted; waiting for cover")

    print("Close the preview window, or press q or ESC, to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break

            roi = preview.roi
            now = time.perf_counter()
            if now - last_submit >= min_submit_dt:
                if active_kind == "cover" and state == "TRACKING":
                    active_worker.submit(roi_frame(frame, roi))
                elif active_kind == "strip" and state in ("STRIP_MODE", "STRIP_DETECTED"):
                    active_worker.submit(roi_frame(frame, roi))
                last_submit = now

            if state == "TRACKING":
                result = active_worker.get_result() if active_kind == "cover" else None
                if result is not None:
                    centroid, cover_mask = offset_result(result, roi, frame.shape[:2])
                    rect_score = measure_rectangularity(cover_mask) if cover_mask is not None else 0.0
                    if centroid is not None and rect_score >= args.rect_threshold:
                        target_bag_mask = cover_mask.copy()
                        target_bag_zone = make_target_zone(cover_mask, frame.shape[:2])
                        if target_bag_zone is not None:
                            state = "STRIP_MODE"
                            print(f"[state] cover laid in ROI, rect={rect_score:.2f}; starting strip detection")
                            start_model(args.hef, "strip")
            elif active_kind == "cover-check":
                result = active_worker.get_result()
                if result is not None:
                    _centroid, check_mask = offset_result(result, roi, frame.shape[:2])
                    check_mask = mask_in_target_zone(check_mask, target_bag_zone)
                    bag_present_last = mask_matches_reference(
                        target_bag_mask,
                        check_mask,
                        TARGET_BAG_MASK_IOU,
                        TARGET_BAG_AREA_RATIO,
                    )
                    if not bag_present_last:
                        state = "BAG_REMOVED"
                        seal_gone_checks = 0
                        preview.set_removal_verification_enabled(False)
                        print("[state] bag removed from its target zone; not sealed")
                        speak("Bag removed. Not sealed")
                    else:
                        if not strip_present:
                            seal_gone_checks += 1
                            if seal_gone_checks >= SEAL_GONE_CHECKS:
                                state = "SEALED"
                                preview.set_removal_verification_enabled(False)
                                print("[state] seal removed and bag present; bag closed")
                                speak("Bag sealed OK")
                        else:
                            seal_gone_checks = 0

                    if state not in ("SEALED", "BAG_REMOVED"):
                        check_remaining = check_interval
                        start_model(args.hef, "strip")
            elif active_kind == "strip":
                result = active_worker.get_result()
                target_detected = False
                if result is not None:
                    centroid, mask = offset_result(result, roi, frame.shape[:2])
                    if target_strip_zone is None:
                        target_strip_zone = make_target_zone(mask, frame.shape[:2])
                        if target_strip_zone is not None:
                            target_strip_mask = mask.copy()
                            print(f"[strip] target zone locked: {target_strip_zone}")
                    else:
                        mask = mask_in_target_zone(mask, target_strip_zone)
                        centroid = get_centroid(mask) if mask is not None else None
                        if not mask_matches_reference(
                                target_strip_mask,
                                mask,
                                TARGET_STRIP_MASK_IOU,
                                TARGET_STRIP_AREA_RATIO):
                            mask = None
                            centroid = None
                    if centroid is not None:
                        target_detected = True
                        strip_present = True
                        strip_mask = mask
                        strip_miss = 0
                    else:
                        strip_miss += 1
                        if strip_miss >= STRIP_DEBOUNCE_FRAMES:
                            strip_present = False
                            strip_mask = None

                if state == "STRIP_MODE":
                    if result is not None and target_detected:
                        strip_confirm_count += 1
                    elif result is not None:
                        strip_confirm_count = 0
                        target_strip_mask = None
                        target_strip_zone = None
                    if strip_confirm_count >= STRIP_CONFIRM_FRAMES:
                        state = "STRIP_DETECTED"
                        print("[state] seal confirmed; remove the seal")
                        speak("Remove the seal")
                        preview.set_removal_verification_enabled(True)
                elif state == "STRIP_DETECTED":
                    stable = strip_zone_is_stable(frame) if removal_verification_started else False
                    if stable and result is not None:
                        if target_detected:
                            verification_strip_misses = 0
                        else:
                            verification_strip_misses += 1
                    if stable and verification_strip_misses >= STRIP_DEBOUNCE_FRAMES:
                        start_cover_check(frame, roi)

            vis = frame.copy()
            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(vis, "ROI", (x1 + 4, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            if target_bag_zone is not None:
                x1, y1, x2, y2 = target_bag_zone
                cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.putText(vis, "TARGET BAG", (x1 + 4, max(22, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2, cv2.LINE_AA)

            if target_strip_zone is not None:
                x1, y1, x2, y2 = target_strip_zone
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(vis, "TARGET STRIP", (x1 + 4, max(22, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

            if state == "TRACKING" and cover_mask is not None:
                layer = vis.copy()
                layer[cover_mask > 0] = (50, 200, 50)
                cv2.addWeighted(layer, 0.20, vis, 0.80, 0, vis)
                cnt = get_largest_contour(cover_mask)
                if cnt is not None:
                    cv2.drawContours(vis, [cnt], -1, (50, 220, 50), 2)

            if (strip_present and strip_mask is not None
                    and state in ("STRIP_MODE", "STRIP_DETECTED")):
                layer = vis.copy()
                layer[strip_mask > 0] = (0, 100, 255)
                cv2.addWeighted(layer, 0.35, vis, 0.65, 0, vis)
                cnt = get_largest_contour(strip_mask)
                if cnt is not None:
                    cv2.drawContours(vis, [cnt], -1, (0, 160, 255), 2)
            if state == "STRIP_MODE":
                cv2.putText(vis, "STRIP MODE - waiting for seal",
                            (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 200, 255), 2, cv2.LINE_AA)
            elif state == "STRIP_DETECTED":
                if removal_verification_started:
                    draw_seal_message(vis, "VERIFYING - KEEP HAND CLEAR", (0, 200, 255))
                else:
                    draw_seal_message(vis, "REMOVE STRIP, THEN PRESS HAND CLEAR", (0, 80, 255))
            elif state == "SEALED":
                draw_seal_message(vis, "SEAL REMOVED OK, BAG CLOSED", (50, 220, 50))
            elif state == "BAG_REMOVED":
                draw_seal_message(vis, "BAG REMOVED - NOT SEALED", (0, 0, 255))

            draw_status(vis, state, strip_present, strip_miss, rect_score,
                        args.rect_threshold, bag_present_last, seal_gone_checks)
            preview.show_frame(vis)
            app.processEvents()
            if preview.closed:
                break
            if preview.take_removal_verification_request():
                removal_verification_started = True
                previous_strip_gray = None
                strip_stable_since = None
                verification_strip_misses = 0
            if preview.take_restart_request():
                restart_cycle()
    finally:
        cap.release()
        preview.close()
        stop_active_model()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
