from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QProcess, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


TOOL_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOL_DIR.parent
DATASET_DIR = TOOL_DIR / "dataset"
CHECKPOINT_DIR = TOOL_DIR / "checkpoints"
PRODUCTION_CHECKPOINT = REPO_DIR / "Segmentation" / "tassel_mobilenet_v3_small.pt"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
TRAIN_SCRIPT = TOOL_DIR / "train_mobilenet.py"
ROI_CONFIG_PATH = TOOL_DIR / "test_bed_roi.json"
MIN_SAMPLES_PER_CLASS = 8
FOCUS_WARMUP_FRAMES = 60
FOCUS_STABILITY_FRAMES = 12
FOCUS_MAX_RELATIVE_SPREAD = 0.05

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from Segmentation import segment_necklace_fastsam as segmentation  # noqa: E402


def now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_test_bed_roi(path: Path = ROI_CONFIG_PATH) -> tuple[float, float, float, float] | None:
    if not path.exists():
        return None
    try:
        values = json.loads(path.read_text(encoding="utf-8"))["normalized_roi"]
        x1, y1, x2, y2 = (float(value) for value in values)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return x1, y1, x2, y2


def save_test_bed_roi(
    roi: tuple[float, float, float, float],
    path: Path = ROI_CONFIG_PATH,
) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"normalized_roi": list(roi)}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def roi_rect_for_shape(
    roi: tuple[float, float, float, float] | None,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    if roi is None:
        return 0, 0, width, height
    x1 = max(0, min(width - 1, int(round(roi[0] * width))))
    y1 = max(0, min(height - 1, int(round(roi[1] * height))))
    x2 = max(x1 + 1, min(width, int(round(roi[2] * width))))
    y2 = max(y1 + 1, min(height, int(round(roi[3] * height))))
    return x1, y1, x2, y2


def normalized_roi_from_rect(
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    height, width = shape
    x1, y1, x2, y2 = rect
    return x1 / width, y1 / height, x2 / width, y2 / height


def crop_to_test_bed(
    image: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rect = roi_rect_for_shape(roi, image.shape[:2])
    x1, y1, x2, y2 = rect
    return image[y1:y2, x1:x2].copy(), rect


def prepare_tassel_test_bed(image: np.ndarray) -> segmentation.PreparedInput:
    """Keep disconnected thread/tassel material instead of only the main necklace."""
    original_image, _ = segmentation.composite_to_bgr(image)
    otsu_mask = segmentation.extract_primary_mask_from_otsu(original_image)
    _, color_maps = segmentation.estimate_background_mask(original_image)
    material_mask = (
        (color_maps["gold_mask"] > 0)
        | (color_maps["red_mask"] > 0)
        | (color_maps["green_mask"] > 0)
        | (color_maps["textile_mask"] > 0)
    ).astype(np.uint8)
    foreground_mask = ((otsu_mask > 0) | (material_mask > 0)).astype(np.uint8)
    foreground_mask = segmentation.close(foreground_mask, 3)
    foreground_mask = segmentation.remove_small_components(
        foreground_mask,
        max(80, foreground_mask.size // 15000),
    )
    if not foreground_mask.any():
        foreground_mask = np.ones(original_image.shape[:2], dtype=np.uint8)
    working_image, working_mask = segmentation.center_primary_jewel(
        original_image,
        foreground_mask,
    )
    return segmentation.PreparedInput(
        original_image=original_image,
        working_image=working_image,
        working_mask=working_mask.astype(np.uint8),
    )


def make_masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float = 0.12,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("Draw or predict a non-empty tassel region first.")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = max(3, int(round(max(x2 - x1, y2 - y1) * padding_ratio)))
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2] > 0
    crop[~crop_mask] = 255
    return crop


def training_shortfall(labels_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts = {
        class_name: len(list((labels_dir / class_name).glob("*.png")))
        for class_name in ("tassel", "false_positive")
    }
    missing = {
        class_name: max(0, MIN_SAMPLES_PER_CLASS - count)
        for class_name, count in counts.items()
    }
    return counts, missing


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root
        self.sessions_dir = root / "sessions"
        self.labels_dir = root / "labels"
        for path in (
            self.sessions_dir,
            self.labels_dir / "tassel",
            self.labels_dir / "false_positive",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        source_image: np.ndarray,
        working_image: np.ndarray,
        foreground_mask: np.ndarray,
        jewel_type: str,
        test_bed_image: np.ndarray | None = None,
        test_bed_roi: tuple[float, float, float, float] | None = None,
        test_bed_rect: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        )
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(session_dir / "camera_capture.png"), source_image)
        if test_bed_image is not None:
            cv2.imwrite(str(session_dir / "test_bed_crop.png"), test_bed_image)
        cv2.imwrite(str(session_dir / "working_image.png"), working_image)
        cv2.imwrite(str(session_dir / "foreground_mask.png"), foreground_mask * 255)
        manifest = {
            "session_id": session_id,
            "captured_at": now_stamp(),
            "jewel_type": jewel_type,
            "camera_capture": "camera_capture.png",
            "test_bed_crop": "test_bed_crop.png" if test_bed_image is not None else None,
            "test_bed_roi_normalized": list(test_bed_roi) if test_bed_roi else None,
            "test_bed_roi_pixels": list(test_bed_rect) if test_bed_rect else None,
            "working_image": "working_image.png",
            "foreground_mask": "foreground_mask.png",
            "prediction": None,
            "samples": {},
        }
        self.write_manifest(manifest)
        return manifest

    def session_dir(self, manifest: dict[str, Any]) -> Path:
        return self.sessions_dir / str(manifest["session_id"])

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.session_dir(manifest) / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)

    def save_prediction(
        self,
        manifest: dict[str, Any],
        mask: np.ndarray | None,
        evidence: dict[str, Any] | None,
        probability: float | None,
        predicted_positive: bool,
        checkpoint_path: Path,
    ) -> None:
        session_dir = self.session_dir(manifest)
        if mask is not None and mask.any():
            cv2.imwrite(str(session_dir / "predicted_mask.png"), mask * 255)
        manifest["prediction"] = {
            "predicted_at": now_stamp(),
            "positive": predicted_positive,
            "probability": probability,
            "checkpoint": str(checkpoint_path),
            "candidate_evidence": evidence,
            "mask": "predicted_mask.png" if mask is not None and mask.any() else None,
        }
        self.write_manifest(manifest)

    def save_sample(
        self,
        manifest: dict[str, Any],
        sample_key: str,
        label: str,
        image: np.ndarray,
        mask: np.ndarray,
        source: str,
        probability: float | None = None,
    ) -> Path:
        if label not in {"tassel", "false_positive"}:
            raise ValueError("Unknown training label.")
        crop = make_masked_crop(image, mask)
        session_id = str(manifest["session_id"])
        filename = f"{session_id}_{sample_key}.png"
        previous = (manifest.get("samples") or {}).get(sample_key) or {}
        previous_path = previous.get("label_path")
        if previous_path:
            stale = self.root / str(previous_path)
            if stale.exists():
                stale.unlink()

        label_path = self.labels_dir / label / filename
        cv2.imwrite(str(label_path), crop)
        mask_name = f"{sample_key}_mask.png"
        cv2.imwrite(str(self.session_dir(manifest) / mask_name), mask * 255)
        manifest.setdefault("samples", {})[sample_key] = {
            "label": label,
            "source": source,
            "reviewed_at": now_stamp(),
            "probability": probability,
            "mask": mask_name,
            "label_path": str(label_path.relative_to(self.root)),
        }
        self.write_manifest(manifest)
        return label_path


class MobileNetPredictor:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.transform = None
        self.threshold = 0.5
        self.load()

    def load(self) -> None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        payload = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = mobilenet_v3_small(weights=None)
        input_features = model.classifier[0].in_features
        model.classifier = torch.nn.Linear(input_features, 1)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self.model = model
        self.transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
        self.threshold = float(payload.get("threshold", 0.5))

    def predict(self, image: np.ndarray, mask: np.ndarray) -> float:
        import torch

        crop = make_masked_crop(image, mask)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        with torch.inference_mode():
            logit = self.model(self.transform(tensor).unsqueeze(0)).reshape(-1)[0]
            return float(torch.sigmoid(logit).item())


class MaskCanvas(QLabel):
    mask_changed = pyqtSignal()
    roi_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(760, 560)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0f172a; border: 1px solid #334155;")
        self.image: np.ndarray | None = None
        self.predicted_mask: np.ndarray | None = None
        self.manual_mask: np.ndarray | None = None
        self.roi_rect: tuple[int, int, int, int] | None = None
        self.roi_drawing_enabled = False
        self.brush_size = 24
        self._last_point: tuple[int, int] | None = None
        self._roi_start: tuple[int, int] | None = None
        self._roi_current: tuple[int, int] | None = None
        self._draw_value = 255
        self._display_scale = 1.0
        self._display_offset = (0, 0)

    def set_image(self, image: np.ndarray | None) -> None:
        self.image = None if image is None else image.copy()
        if image is None:
            self.predicted_mask = None
            self.manual_mask = None
        else:
            self.predicted_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            self.manual_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.render()

    def set_predicted_mask(self, mask: np.ndarray | None) -> None:
        if self.image is None:
            return
        self.predicted_mask = (
            np.zeros(self.image.shape[:2], dtype=np.uint8)
            if mask is None
            else (mask > 0).astype(np.uint8) * 255
        )
        self.render()

    def set_roi_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        self.roi_rect = rect
        self.render()

    def set_roi_drawing(self, enabled: bool) -> None:
        self.roi_drawing_enabled = enabled
        self._roi_start = None
        self._roi_current = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.render()

    def clear_manual_mask(self) -> None:
        if self.manual_mask is not None:
            self.manual_mask.fill(0)
            self.mask_changed.emit()
            self.render()

    def render(self) -> None:
        if self.image is None:
            self.clear()
            self.setText("Live camera starting...")
            return
        display = self.image.copy()
        roi_rect = self.roi_rect
        if self._roi_start is not None and self._roi_current is not None:
            x1, x2 = sorted((self._roi_start[0], self._roi_current[0]))
            y1, y2 = sorted((self._roi_start[1], self._roi_current[1]))
            roi_rect = (x1, y1, x2, y2)
        if roi_rect is not None:
            x1, y1, x2, y2 = roi_rect
            shaded = (display.astype(np.float32) * 0.30).astype(np.uint8)
            shaded[y1:y2, x1:x2] = display[y1:y2, x1:x2]
            display = shaded
            cv2.rectangle(
                display,
                (x1, y1),
                (max(x1, x2 - 1), max(y1, y2 - 1)),
                (50, 220, 80),
                3,
            )
        if self.predicted_mask is not None and self.predicted_mask.any():
            blue = np.zeros_like(display)
            blue[:] = (255, 90, 20)
            selected = self.predicted_mask > 0
            display[selected] = cv2.addWeighted(
                display[selected], 0.45, blue[selected], 0.55, 0
            )
        if self.manual_mask is not None and self.manual_mask.any():
            red = np.zeros_like(display)
            red[:] = (20, 30, 255)
            selected = self.manual_mask > 0
            display[selected] = cv2.addWeighted(
                display[selected], 0.35, red[selected], 0.65, 0
            )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display_scale = pixmap.width() / max(1, self.image.shape[1])
        self._display_offset = (
            (self.width() - pixmap.width()) // 2,
            (self.height() - pixmap.height()) // 2,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.render()

    def _image_point(self, event: QMouseEvent) -> tuple[int, int] | None:
        if self.image is None or self._display_scale <= 0:
            return None
        x = int((event.position().x() - self._display_offset[0]) / self._display_scale)
        y = int((event.position().y() - self._display_offset[1]) / self._display_scale)
        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            return x, y
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.roi_drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            point = self._image_point(event)
            if point is not None:
                self._roi_start = point
                self._roi_current = point
                self.render()
            return
        if event.button() not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            return super().mousePressEvent(event)
        point = self._image_point(event)
        if point is None or self.manual_mask is None:
            return
        self._draw_value = 255 if event.button() == Qt.MouseButton.LeftButton else 0
        self._last_point = point
        cv2.circle(
            self.manual_mask,
            point,
            max(1, self.brush_size // 2),
            self._draw_value,
            -1,
        )
        self.mask_changed.emit()
        self.render()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.roi_drawing_enabled and self._roi_start is not None:
            point = self._image_point(event)
            if point is not None:
                self._roi_current = point
                self.render()
            return
        if self._last_point is None or self.manual_mask is None:
            return super().mouseMoveEvent(event)
        point = self._image_point(event)
        if point is None:
            return
        cv2.line(
            self.manual_mask,
            self._last_point,
            point,
            self._draw_value,
            self.brush_size,
            cv2.LINE_AA,
        )
        self._last_point = point
        self.mask_changed.emit()
        self.render()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.roi_drawing_enabled and self._roi_start is not None:
            point = self._image_point(event) or self._roi_current
            start = self._roi_start
            self._roi_start = None
            self._roi_current = None
            if point is not None:
                x1, x2 = sorted((start[0], point[0]))
                y1, y2 = sorted((start[1], point[1]))
                if x2 - x1 >= 32 and y2 - y1 >= 32:
                    self.roi_rect = (x1, y1, x2, y2)
                    self.roi_changed.emit(self.roi_rect)
            self.render()
            return
        self._last_point = None
        super().mouseReleaseEvent(event)


class TasselTrainingWindow(QMainWindow):
    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.setWindowTitle("Tassel MobileNetV3 Capture and Training")
        self.resize(1380, 850)
        self.store = DatasetStore(DATASET_DIR)
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.review_image: np.ndarray | None = None
        self.foreground_mask: np.ndarray | None = None
        self.candidate_mask: np.ndarray | None = None
        self.candidate_evidence: dict[str, Any] | None = None
        self.candidate_probability: float | None = None
        self.current_manifest: dict[str, Any] | None = None
        self.training_process: QProcess | None = None
        self.focus_frame_count = 0
        self.focus_scores: deque[float] = deque(maxlen=FOCUS_STABILITY_FRAMES)
        self.focus_ready = False
        self.focus_locked = False
        self.test_bed_roi = load_test_bed_roi()
        self.model_path = LATEST_CHECKPOINT if LATEST_CHECKPOINT.exists() else PRODUCTION_CHECKPOINT
        self.predictor = MobileNetPredictor(self.model_path)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.read_frame)
        QTimer.singleShot(0, self.start_camera)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        header = QLabel("TASSEL PRESENCE LABELER + MOBILENETV3 CHECKPOINT TRAINER")
        header.setStyleSheet("font-size: 20px; font-weight: 800; color: #e2e8f0;")
        outer.addWidget(header)

        body = QHBoxLayout()
        self.canvas = MaskCanvas()
        body.addWidget(self.canvas, 1)
        controls = QVBoxLayout()
        controls.setSpacing(9)

        form = QFormLayout()
        self.jewel_type = QComboBox()
        self.jewel_type.addItems(["Necklace", "Haram"])
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.95)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(self.predictor.threshold)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(4, 100)
        self.brush_size.setValue(24)
        self.brush_size.valueChanged.connect(self.set_brush_size)
        form.addRow("Jewel type", self.jewel_type)
        form.addRow("Positive threshold", self.threshold)
        form.addRow("Brush size", self.brush_size)
        controls.addLayout(form)

        self.model_label = QLabel(f"Loaded: {self.model_path.name}")
        self.model_label.setWordWrap(True)
        controls.addWidget(self.model_label)
        self.counts_label = QLabel()
        controls.addWidget(self.counts_label)

        self.roi_label = QLabel()
        controls.addWidget(self.roi_label)
        workflow = QLabel(
            "1. Draw ROI  2. Capture Test Bed  3. Segment Tassel  "
            "4. Label the blue proposal or draw the correct tassel"
        )
        workflow.setWordWrap(True)
        workflow.setStyleSheet("color: #93c5fd; font-weight: 700;")
        controls.addWidget(workflow)

        self.roi_button = QPushButton("Draw Test Bed ROI")
        self.roi_button.setCheckable(True)
        self.clear_roi_button = QPushButton("Clear Test Bed ROI")
        self.capture_button = QPushButton("1. Capture Test Bed")
        self.predict_button = QPushButton("2. Segment Tassel / Run Prediction")
        self.live_button = QPushButton("Return to Live Feed")
        self.true_button = QPushButton("3A. Blue Region is Correct Tassel")
        self.false_button = QPushButton("3B. Blue Region is False Positive")
        self.manual_button = QPushButton("3C. Save Red Drawn Tassel")
        self.no_tassel_button = QPushButton("3D. No Tassel in This Image")
        self.clear_button = QPushButton("Clear Drawn Mask")
        self.train_button = QPushButton("Train MobileNetV3 from Latest Checkpoint")
        for button in (
            self.roi_button,
            self.clear_roi_button,
            self.capture_button,
            self.predict_button,
            self.live_button,
            self.true_button,
            self.false_button,
            self.manual_button,
            self.no_tassel_button,
            self.clear_button,
            self.train_button,
        ):
            controls.addWidget(button)

        self.roi_button.toggled.connect(self.toggle_roi_drawing)
        self.clear_roi_button.clicked.connect(self.clear_test_bed_roi)
        self.canvas.roi_changed.connect(self.test_bed_roi_changed)
        self.capture_button.clicked.connect(self.capture_image)
        self.predict_button.clicked.connect(self.run_prediction)
        self.live_button.clicked.connect(self.start_camera)
        self.true_button.clicked.connect(lambda: self.save_candidate("tassel"))
        self.false_button.clicked.connect(
            lambda: self.save_candidate("false_positive")
        )
        self.true_button.setStyleSheet("background: #166534;")
        self.false_button.setStyleSheet("background: #991b1b;")
        self.manual_button.clicked.connect(self.save_manual_tassel)
        self.no_tassel_button.clicked.connect(self.save_no_tassel)
        self.clear_button.clicked.connect(self.canvas.clear_manual_mask)
        self.train_button.clicked.connect(self.start_training)

        controls.addStretch()
        self.status = QLabel("Starting camera...")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(340)
        self.status.setStyleSheet(
            "padding: 10px; background: #111827; color: #f8fafc; border-radius: 8px;"
        )
        controls.addWidget(self.status)
        body.addLayout(controls)
        outer.addLayout(body, 1)
        root.setStyleSheet(
            "QWidget { background: #020617; color: #cbd5e1; }"
            "QPushButton { padding: 9px; background: #1e293b; border: 1px solid #475569; }"
            "QPushButton:disabled { color: #64748b; }"
        )
        self.setCentralWidget(root)
        self.refresh_counts()
        self.refresh_roi_label()
        self.set_review_controls(False)

    def refresh_roi_label(self) -> None:
        self.roi_label.setText(
            "Test bed ROI: saved and active"
            if self.test_bed_roi is not None
            else "Test bed ROI: full camera frame"
        )

    def toggle_roi_drawing(self, enabled: bool) -> None:
        self.canvas.set_roi_drawing(enabled)
        self.capture_button.setEnabled(self.focus_ready and not enabled)
        if enabled:
            self.status.setText(
                "Drag a green rectangle around only the test bed. Release to save it."
            )

    def _reset_focus_after_roi_change(self) -> None:
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.focus_frame_count = 0
        self.focus_scores.clear()
        self.focus_ready = False
        self.focus_locked = False
        self.capture_button.setEnabled(False)

    def test_bed_roi_changed(self, rect: tuple[int, int, int, int]) -> None:
        if self.canvas.image is None:
            return
        self.test_bed_roi = normalized_roi_from_rect(rect, self.canvas.image.shape[:2])
        save_test_bed_roi(self.test_bed_roi)
        self.roi_button.setChecked(False)
        self.refresh_roi_label()
        self._reset_focus_after_roi_change()
        self.status.setText(
            "Test bed ROI saved. Outside background is excluded. Hold still while focus relocks."
        )

    def clear_test_bed_roi(self) -> None:
        self.test_bed_roi = None
        try:
            ROI_CONFIG_PATH.unlink()
        except FileNotFoundError:
            pass
        self.canvas.set_roi_rect(None)
        self.roi_button.setChecked(False)
        self.refresh_roi_label()
        self._reset_focus_after_roi_change()
        self.status.setText(
            "Test bed ROI cleared. The full camera frame will be used after focus relocks."
        )

    def set_brush_size(self, value: int) -> None:
        self.canvas.brush_size = value

    def set_review_controls(self, enabled: bool) -> None:
        self.predict_button.setEnabled(enabled)
        self.live_button.setEnabled(enabled)
        self.manual_button.setEnabled(enabled)
        self.no_tassel_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.true_button.setEnabled(enabled and self.candidate_mask is not None)
        self.false_button.setEnabled(enabled and self.candidate_mask is not None)

    def refresh_counts(self) -> None:
        counts, missing = training_shortfall(self.store.labels_dir)
        self.counts_label.setText(
            f"Training samples: {counts['tassel']} tassel / "
            f"{counts['false_positive']} false positive\n"
            f"Need: {missing['tassel']} more tassel / "
            f"{missing['false_positive']} more false positive"
        )

    def start_camera(self) -> None:
        if self.training_process is not None:
            return
        if self.capture is None:
            capture = cv2.VideoCapture(self.camera_index)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            if not capture.isOpened():
                self.status.setText(f"Could not open camera index {self.camera_index}.")
                capture.release()
                return
            self.capture = capture
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.focus_frame_count = 0
        self.focus_scores.clear()
        self.focus_ready = False
        self.focus_locked = False
        self.current_manifest = None
        self.review_image = None
        self.foreground_mask = None
        self.candidate_mask = None
        self.candidate_evidence = None
        self.candidate_probability = None
        self.canvas.set_predicted_mask(None)
        self.canvas.set_roi_drawing(False)
        self.roi_button.setChecked(False)
        self.roi_button.setEnabled(True)
        self.clear_roi_button.setEnabled(self.test_bed_roi is not None)
        self.capture_button.setEnabled(False)
        self.set_review_controls(False)
        self.timer.start()
        self.status.setText(
            "Live feed. Hold the jewellery still while autofocus stabilizes."
        )

    def read_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.status.setText("Camera frame read failed.")
            return
        self.latest_frame = frame.copy()
        self.canvas.set_image(frame)
        self.canvas.set_roi_rect(
            roi_rect_for_shape(self.test_bed_roi, frame.shape[:2])
            if self.test_bed_roi is not None
            else None
        )
        self.clear_roi_button.setEnabled(self.test_bed_roi is not None)
        if not self.focus_ready:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.test_bed_roi is not None:
                x1, y1, x2, y2 = roi_rect_for_shape(
                    self.test_bed_roi,
                    gray.shape,
                )
                focus_region = gray[y1:y2, x1:x2]
            else:
                height, width = gray.shape
                margin_x = width // 8
                margin_y = height // 8
                focus_region = gray[
                    margin_y : height - margin_y,
                    margin_x : width - margin_x,
                ]
            focus_score = float(cv2.Laplacian(focus_region, cv2.CV_64F).var())
            self.focus_frame_count += 1
            if self.focus_frame_count >= FOCUS_WARMUP_FRAMES:
                self.focus_scores.append(focus_score)
            if len(self.focus_scores) == FOCUS_STABILITY_FRAMES:
                mean_score = float(np.mean(self.focus_scores))
                relative_spread = (
                    (max(self.focus_scores) - min(self.focus_scores))
                    / max(mean_score, 1e-6)
                )
                if relative_spread <= FOCUS_MAX_RELATIVE_SPREAD:
                    self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    self.focus_ready = True
                    self.focus_locked = True
                    self.capture_button.setEnabled(True)
                    self.status.setText(
                        f"Focus locked (sharpness {focus_score:.1f}). Capture is ready."
                    )
            if not self.focus_ready and self.focus_frame_count % 15 == 0:
                self.status.setText(
                    f"Focusing: {self.focus_frame_count}/{FOCUS_WARMUP_FRAMES} warm-up frames. "
                    "Keep the jewellery still."
                )

    def capture_image(self) -> None:
        if self.latest_frame is None or self.training_process is not None:
            return
        self.timer.stop()
        source = self.latest_frame.copy()
        test_bed_image, test_bed_rect = crop_to_test_bed(source, self.test_bed_roi)
        prepared = prepare_tassel_test_bed(test_bed_image)
        self.review_image = prepared.working_image
        self.foreground_mask = prepared.working_mask
        self.current_manifest = self.store.create_session(
            source,
            self.review_image,
            self.foreground_mask,
            self.jewel_type.currentText(),
            test_bed_image,
            self.test_bed_roi,
            test_bed_rect,
        )
        self.candidate_mask = None
        self.candidate_evidence = None
        self.candidate_probability = None
        self.canvas.set_image(self.review_image)
        self.canvas.set_roi_rect(None)
        self.canvas.set_roi_drawing(False)
        self.roi_button.setChecked(False)
        self.roi_button.setEnabled(False)
        self.clear_roi_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.set_review_controls(True)
        self.status.setText(
            "Captured test bed only. Run Prediction. Left mouse draws tassel; right mouse erases."
        )

    def run_prediction(self) -> None:
        if self.review_image is None or self.foreground_mask is None:
            return
        self.status.setText("Finding tassel proposals and checking them with MobileNetV3...")
        QApplication.processEvents()
        _, color_maps = segmentation.estimate_background_mask(self.review_image)
        proposal_masks = (
            (
                "gold_or_pale_thread",
                ((color_maps["gold_mask"] > 0) & (self.foreground_mask > 0)).astype(
                    np.uint8
                ),
            ),
            (
                "coloured_thread",
                (
                    (
                        (color_maps["red_mask"] > 0)
                        | (color_maps["green_mask"] > 0)
                        | (color_maps["textile_mask"] > 0)
                    )
                    & (self.foreground_mask > 0)
                ).astype(np.uint8),
            ),
            ("complete_foreground", self.foreground_mask),
        )
        proposals: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        proposal_summary = []
        for source, proposal_mask in proposal_masks:
            proposal_mask = segmentation.remove_small_components(
                segmentation.close(proposal_mask, 3),
                max(40, proposal_mask.size // 25000),
            )
            if not proposal_mask.any():
                continue
            seed, _, _, evidence = segmentation.detect_tassel_seed(
                proposal_mask,
                self.review_image,
                color_maps["textile_mask"],
                self.jewel_type.currentText(),
                use_classifier=False,
            )
            candidate = seed.copy()
            if not candidate.any() and evidence and evidence.get("bbox"):
                x1, y1, x2, y2 = (int(value) for value in evidence["bbox"])
                candidate = np.zeros_like(self.foreground_mask)
                candidate[y1 : y2 + 1, x1 : x2 + 1] = proposal_mask[
                    y1 : y2 + 1,
                    x1 : x2 + 1,
                ]
            if not candidate.any():
                continue
            probability = self.predictor.predict(self.review_image, candidate)
            candidate_evidence = dict(evidence or {})
            candidate_evidence["proposal_source"] = source
            proposals.append((probability, candidate, candidate_evidence))
            proposal_summary.append(
                {
                    "source": source,
                    "probability": round(float(probability), 6),
                    "bbox": candidate_evidence.get("bbox"),
                }
            )

        self.candidate_mask = None
        self.candidate_evidence = None
        self.candidate_probability = None
        predicted_positive = False
        if proposals:
            probability, candidate, evidence = max(
                proposals,
                key=lambda item: item[0],
            )
            evidence["proposal_candidates"] = proposal_summary
            self.candidate_mask = candidate
            self.candidate_evidence = evidence
            self.candidate_probability = probability
            geometry_accepted = bool(
                (evidence or {}).get(
                    "geometry_accepted",
                    (evidence or {}).get("accepted", False),
                )
            )
            predicted_positive = bool(probability >= self.threshold.value())
            self.canvas.set_predicted_mask(self.candidate_mask)
            decision = "TASSEL" if predicted_positive else "NO TASSEL"
            self.status.setText(
                f"Prediction: {decision}; MobileNetV3 probability {probability:.1%}; "
                f"proposal source {evidence.get('proposal_source', 'unknown')}; "
                f"geometry {'supported' if geometry_accepted else 'uncertain'}. "
                "If blue is tassel choose 3A. If blue is necklace/thread/jewellery choose 3B. "
                "For a missed tassel, draw it in red and choose 3C."
            )
        else:
            self.canvas.set_predicted_mask(None)
            self.status.setText(
                "Prediction: NO TASSEL and no candidate region. If a tassel exists, draw it in red; "
                "otherwise choose Correct Rejection / No Tassel."
            )
        self.true_button.setEnabled(self.candidate_mask is not None)
        self.false_button.setEnabled(self.candidate_mask is not None)
        if self.current_manifest is not None:
            self.store.save_prediction(
                self.current_manifest,
                self.candidate_mask,
                self.candidate_evidence,
                self.candidate_probability,
                predicted_positive,
                self.model_path,
            )

    def save_candidate(self, label: str) -> None:
        if (
            self.current_manifest is None
            or self.review_image is None
            or self.candidate_mask is None
        ):
            return
        path = self.store.save_sample(
            self.current_manifest,
            "auto_candidate",
            label,
            self.review_image,
            self.candidate_mask,
            "model_prediction",
            self.candidate_probability,
        )
        self.refresh_counts()
        self.status.setText(f"Saved {label.replace('_', ' ')} sample: {path.name}")

    def save_manual_tassel(self) -> None:
        if self.current_manifest is None or self.review_image is None:
            return
        manual_mask = self.canvas.manual_mask
        if manual_mask is None or not manual_mask.any():
            self.status.setText("Draw the tassel region in red before saving.")
            return
        mask = (manual_mask > 0).astype(np.uint8)
        try:
            path = self.store.save_sample(
                self.current_manifest,
                "manual_tassel",
                "tassel",
                self.review_image,
                mask,
                "manual_brush",
            )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.refresh_counts()
        self.status.setText(f"Saved manually drawn tassel sample: {path.name}")

    def save_no_tassel(self) -> None:
        if (
            self.current_manifest is None
            or self.review_image is None
            or self.foreground_mask is None
        ):
            return
        mask = (
            self.candidate_mask
            if self.candidate_mask is not None
            else self.foreground_mask
        )
        source = (
            "rejected_candidate"
            if self.candidate_mask is not None
            else "no_candidate_full_object"
        )
        path = self.store.save_sample(
            self.current_manifest,
            "no_tassel",
            "false_positive",
            self.review_image,
            mask,
            source,
            self.candidate_probability,
        )
        self.refresh_counts()
        self.status.setText(f"Saved no-tassel training sample: {path.name}")

    def start_training(self) -> None:
        if self.training_process is not None:
            return
        counts, missing = training_shortfall(self.store.labels_dir)
        missing_text = []
        if missing["tassel"]:
            missing_text.append(f"{missing['tassel']} more tassel")
        if missing["false_positive"]:
            missing_text.append(f"{missing['false_positive']} more false-positive")
        if missing_text:
            self.status.setText(
                "Training not started: capture " + " and ".join(missing_text) + " sample(s)."
            )
            return
        self.timer.stop()
        self.capture_button.setEnabled(False)
        self.set_review_controls(False)
        self.roi_button.setEnabled(False)
        self.clear_roi_button.setEnabled(False)
        self.train_button.setEnabled(False)
        base_checkpoint = LATEST_CHECKPOINT if LATEST_CHECKPOINT.exists() else PRODUCTION_CHECKPOINT
        process = QProcess(self)
        process.setWorkingDirectory(str(TOOL_DIR))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_training_output)
        process.finished.connect(self.training_finished)
        self.training_process = process
        process.start(
            sys.executable,
            [
                "-u",
                str(TRAIN_SCRIPT),
                "--dataset",
                str(self.store.labels_dir),
                "--base-checkpoint",
                str(base_checkpoint),
                "--checkpoint-dir",
                str(CHECKPOINT_DIR),
            ],
        )
        if not process.waitForStarted(3000):
            self.training_process = None
            self.train_button.setEnabled(True)
            self.status.setText(f"Could not start training: {process.errorString()}")
            return
        self.status.setText(
            f"Fine-tuning from {base_checkpoint.name}; {counts['tassel']} tassel and "
            f"{counts['false_positive']} false-positive samples."
        )

    def read_training_output(self) -> None:
        if self.training_process is None:
            return
        text = bytes(self.training_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            self.status.setText(lines[-1])

    def training_finished(self, exit_code: int, _exit_status) -> None:
        process = self.training_process
        remainder = (
            bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if process is not None
            else ""
        )
        self.training_process = None
        self.train_button.setEnabled(True)
        if exit_code == 0 and LATEST_CHECKPOINT.exists():
            self.model_path = LATEST_CHECKPOINT
            self.predictor = MobileNetPredictor(self.model_path)
            self.threshold.setValue(self.predictor.threshold)
            self.model_label.setText(f"Loaded: {self.model_path.name}")
            self.status.setText("Training complete. Latest checkpoint loaded for prediction.")
        else:
            lines = [line for line in remainder.splitlines() if line.strip()]
            detail = lines[-1] if lines else f"exit code {exit_code}"
            self.status.setText(f"Training failed: {detail}")
        self.start_camera()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        if self.training_process is not None:
            self.training_process.terminate()
            self.training_process.waitForFinished(3000)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tassel MobileNetV3 data collection tool")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = TasselTrainingWindow(camera_index=args.camera)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
