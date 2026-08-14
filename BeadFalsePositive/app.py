from __future__ import annotations

import argparse
import filecmp
import json
import math
import platform
import re
import shutil
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QProcess, QRect, QSize, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


TOOL_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOL_DIR.parent
DEFAULT_MODEL_PATH = REPO_DIR / "models" / "bead_finder.hef"
PROJECTS_DIR = TOOL_DIR / "model_projects"
DEFAULT_PROJECT_ID = "bead_finder"
LEGACY_DATASET_DIR = TOOL_DIR / "dataset"
RESNET_DIR = TOOL_DIR / "ResNet18"
LEGACY_RESNET_MODEL_PATH = RESNET_DIR / "bead_fp_resnet18_test.pt"
RESNET_MODEL_NAME = "beadcheck.pt"
CANDIDATE_RESNET_MODEL_NAME = "beadcheck_candidate.pt"
CANDIDATE_REPORT_NAME = "beadcheck_candidate_training_report.json"
MOBILENET_DIR_NAME = "MobileNetV3"
MOBILENET_MODEL_NAME = "beadcheck_mobilenet_v3_candidate.pt"
MOBILENET_REPORT_NAME = "beadcheck_mobilenet_v3_candidate_training_report.json"
LOCKED_EVALUATION_NAME = "locked_crop_evaluation_manifest.json"
RESNET_TRAIN_SCRIPT = RESNET_DIR / "train_resnet18.py"
YOLO_SCORE_THRESHOLD = 0.50
TRUE_DETECTION_THRESHOLD = 0.75
CROP_PADDING_RATIO = 0.05
FOCUS_WARMUP_FRAMES = 60
FOCUS_STABILITY_FRAMES = 12
FOCUS_MAX_RELATIVE_SPREAD = 0.05
VALID_LABELS = {"unreviewed", "true_detection", "false_positive"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from hailo_model_runner import HailoRuntime  # noqa: E402


def inspect_detection_hef(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file() or model_path.suffix.lower() != ".hef":
        raise ValueError("Select an existing .hef model file.")
    try:
        from hailo_platform import HEF

        hef = HEF(str(model_path))
        inputs = list(hef.get_input_vstream_infos())
        outputs = list(hef.get_output_vstream_infos())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not inspect HEF {model_path.name}: {exc}") from exc
    if len(inputs) != 1 or len(inputs[0].shape) != 3:
        raise ValueError("The HEF must have one three-dimensional image input.")
    input_h, input_w, input_c = (int(value) for value in inputs[0].shape)
    input_order = str(inputs[0].format.order)
    if input_c != 3 or "NHWC" not in input_order:
        raise ValueError(
            f"The HEF must use a three-channel NHWC input; received {inputs[0].shape} {input_order}."
        )
    output_order = str(outputs[0].format.order) if len(outputs) == 1 else ""
    if len(outputs) == 1 and "HAILO_NMS_BY_CLASS" in output_order:
        output_decoder = "hailo_nms"
        number_of_classes = int(outputs[0].nms_shape.number_of_classes)
    elif is_single_class_yolov8_seg_shape([tuple(output.shape) for output in outputs]):
        output_decoder = "yolov8_seg_raw"
        output_order = "RAW_YOLOV8_SEG"
        number_of_classes = 1
    else:
        raise ValueError(
            "This HEF must expose Hailo NMS detections or the supported single-class "
            "raw YOLOv8-seg output used by bestnewacid.hef."
        )
    return {
        "input_height": input_h,
        "input_width": input_w,
        "input_channels": input_c,
        "output_format": output_order,
        "output_decoder": output_decoder,
        "number_of_classes": number_of_classes,
    }


def is_single_class_yolov8_seg_shape(shapes: list[tuple[int, ...]]) -> bool:
    grouped: dict[tuple[int, int], list[int]] = {}
    for shape in shapes:
        if len(shape) != 3:
            return False
        height, width, channels = (int(value) for value in shape)
        grouped.setdefault((height, width), []).append(channels)
    head_groups = [sorted(channels) for channels in grouped.values() if len(channels) == 3]
    prototype_groups = [channels for channels in grouped.values() if len(channels) == 1]
    return (
        len(shapes) == 10
        and len(head_groups) == 3
        and all(channels == [1, 32, 64] for channels in head_groups)
        and len(prototype_groups) == 1
        and prototype_groups[0] == [32]
    )


def project_id_from_model(model_path: Path) -> str:
    project_id = re.sub(r"[^A-Za-z0-9_-]+", "_", model_path.stem).strip("_")
    if not project_id:
        raise ValueError("The HEF filename must contain a letter or number.")
    return project_id


def normalize_legacy_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    for candidate in manifest.get("candidates", []):
        if candidate.get("label") == "true_bead":
            candidate["label"] = "true_detection"
        if candidate.get("resnet_prediction") == "true_bead":
            candidate["resnet_prediction"] = "true_detection"
        if "true_bead_probability" in candidate:
            candidate["true_detection_probability"] = candidate.pop("true_bead_probability")
        if "bead_number" in candidate:
            candidate["detection_number"] = candidate.pop("bead_number")
    if "bead_count" in manifest:
        manifest["detection_count"] = manifest.pop("bead_count")
    return manifest


class ModelProjectManager:
    def __init__(self, root: Path, inspector=inspect_detection_hef):
        self.root = root
        self.inspector = inspector
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_root(self, project_id: str) -> Path:
        if not SAFE_ID.fullmatch(project_id or ""):
            raise ValueError("Invalid project ID.")
        return self.root / project_id

    def import_hef(self, source_path: Path) -> dict[str, Any]:
        source_path = source_path.resolve()
        hef_info = self.inspector(source_path)
        project_id = project_id_from_model(source_path)
        project_root = self._project_root(project_id)
        project_root.mkdir(parents=True, exist_ok=True)
        existing_models = list(project_root.glob("*.hef"))
        target_path = project_root / source_path.name
        for existing_path in existing_models:
            if existing_path.name != source_path.name or not filecmp.cmp(
                existing_path, source_path, shallow=False
            ):
                raise ValueError(
                    f"Project {project_id} already contains a different HEF. "
                    "Rename the new HEF file to create a separate project."
                )
        if source_path != target_path and not target_path.exists():
            shutil.copy2(source_path, target_path)
        for path in (
            project_root / "dataset" / "sessions",
            project_root / "dataset" / "labels" / "true_detection",
            project_root / "dataset" / "labels" / "false_positive",
            project_root / "ResNet18",
            project_root / MOBILENET_DIR_NAME,
        ):
            path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "project_id": project_id,
            "hef_file": target_path.name,
            "resnet_model": RESNET_MODEL_NAME,
            "mobilenet_model": (
                MOBILENET_MODEL_NAME
                if project_id == DEFAULT_PROJECT_ID
                else f"{project_id}_mobilenet_v3_candidate.pt"
            ),
            "mobilenet_report": (
                MOBILENET_REPORT_NAME
                if project_id == DEFAULT_PROJECT_ID
                else f"{project_id}_mobilenet_v3_candidate_training_report.json"
            ),
            **hef_info,
        }
        metadata_path = project_root / "project.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(metadata_path)
        return self.load_project(project_id)

    def load_project(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        metadata_path = project_root / "project.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Unknown HEF project: {project_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mobilenet_model = str(
            metadata.get("mobilenet_model")
            or (
                MOBILENET_MODEL_NAME
                if project_id == DEFAULT_PROJECT_ID
                else f"{project_id}_mobilenet_v3_candidate.pt"
            )
        )
        mobilenet_report = str(
            metadata.get("mobilenet_report")
            or (
                MOBILENET_REPORT_NAME
                if project_id == DEFAULT_PROJECT_ID
                else f"{project_id}_mobilenet_v3_candidate_training_report.json"
            )
        )
        return {
            **metadata,
            "root": project_root,
            "model_path": project_root / str(metadata["hef_file"]),
            "dataset_path": project_root / "dataset",
            "resnet_dir": project_root / "ResNet18",
            "resnet_model_path": project_root / "ResNet18" / str(metadata["resnet_model"]),
            "candidate_model_path": project_root / "ResNet18" / CANDIDATE_RESNET_MODEL_NAME,
            "candidate_report_path": project_root / "ResNet18" / CANDIDATE_REPORT_NAME,
            "locked_evaluation_path": project_root / "ResNet18" / LOCKED_EVALUATION_NAME,
            "mobilenet_model_path": project_root / MOBILENET_DIR_NAME / mobilenet_model,
            "mobilenet_report_path": project_root / MOBILENET_DIR_NAME / mobilenet_report,
        }

    def list_projects(self) -> list[dict[str, Any]]:
        projects = []
        for metadata_path in sorted(self.root.glob("*/project.json")):
            try:
                projects.append(self.load_project(metadata_path.parent.name))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return projects


def migrate_legacy_bead_project(project: dict[str, Any]) -> None:
    dataset_path = Path(project["dataset_path"])
    sessions_path = dataset_path / "sessions"
    labels_path = dataset_path / "labels"
    if LEGACY_DATASET_DIR.exists() and not any(sessions_path.glob("*/manifest.json")):
        shutil.copytree(LEGACY_DATASET_DIR / "sessions", sessions_path, dirs_exist_ok=True)
        shutil.copytree(
            LEGACY_DATASET_DIR / "labels" / "false_positive",
            labels_path / "false_positive",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            LEGACY_DATASET_DIR / "labels" / "true_bead",
            labels_path / "true_detection",
            dirs_exist_ok=True,
        )
        for manifest_path in sessions_path.glob("*/manifest.json"):
            manifest = normalize_legacy_manifest(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    resnet_model_path = Path(project["resnet_model_path"])
    if LEGACY_RESNET_MODEL_PATH.exists() and not resnet_model_path.exists():
        shutil.copy2(LEGACY_RESNET_MODEL_PATH, resnet_model_path)


def normalize_nms_rows(output: Any) -> np.ndarray:
    rows = np.squeeze(np.asarray(output, dtype=np.float32))
    if rows.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    elif rows.ndim == 3 and rows.shape[1] == 5:
        rows = rows.transpose(0, 2, 1).reshape(-1, 5)
    elif rows.ndim > 2 and rows.shape[-1] == 5:
        rows = rows.reshape(-1, 5)
    if rows.ndim == 2 and rows.shape[1] != 5 and rows.shape[0] == 5:
        rows = rows.T
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise RuntimeError(f"Unexpected detection HEF output shape: {np.asarray(output).shape}.")
    return rows


def decode_detections(
    rows: np.ndarray,
    source_shape: tuple[int, int],
    scale: float,
    left: int,
    top: int,
    threshold: float = YOLO_SCORE_THRESHOLD,
    model_shape: tuple[int, int] = (640, 640),
) -> list[dict[str, Any]]:
    source_h, source_w = source_shape
    model_h, model_w = model_shape
    detections: list[dict[str, Any]] = []
    for y1, x1, y2, x2, score in rows:
        values = np.asarray([y1, x1, y2, x2, score], dtype=np.float32)
        if not np.isfinite(values).all() or float(score) < threshold:
            continue
        box_x1 = int(round((float(x1) * model_w - left) / scale))
        box_y1 = int(round((float(y1) * model_h - top) / scale))
        box_x2 = int(round((float(x2) * model_w - left) / scale))
        box_y2 = int(round((float(y2) * model_h - top) / scale))
        box_x1 = max(0, min(source_w - 1, box_x1))
        box_y1 = max(0, min(source_h - 1, box_y1))
        box_x2 = max(0, min(source_w - 1, box_x2))
        box_y2 = max(0, min(source_h - 1, box_y2))
        if box_x2 <= box_x1 or box_y2 <= box_y1:
            continue
        detections.append(
            {
                "bbox": [box_x1, box_y1, box_x2, box_y2],
                "yolo_score": float(score),
            }
        )
    return detections


def decode_yolov8_seg_rows(
    output: dict[str, Any],
    model_shape: tuple[int, int],
    threshold: float = YOLO_SCORE_THRESHOLD,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    model_h, model_w = model_shape
    grouped: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for value in output.values():
        tensor = np.asarray(value, dtype=np.float32)
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            continue
        height, width, channels = tensor.shape
        if channels in (1, 32, 64):
            grouped.setdefault((height, width), {})[channels] = tensor

    candidates: list[list[float]] = []
    regression_bins = np.arange(16, dtype=np.float32)
    for (grid_h, grid_w), tensors in grouped.items():
        if 64 not in tensors or 1 not in tensors or 32 not in tensors:
            continue
        dfl = tensors[64]
        scores = tensors[1][:, :, 0]
        if scores.size and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
        stride_x = model_w / grid_w
        stride_y = model_h / grid_h
        for grid_y, grid_x in np.argwhere(scores >= threshold):
            logits = dfl[grid_y, grid_x].reshape(4, 16)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            left_dist, top_dist, right_dist, bottom_dist = probabilities @ regression_bins
            center_x = (grid_x + 0.5) * stride_x
            center_y = (grid_y + 0.5) * stride_y
            x1 = max(0.0, center_x - float(left_dist) * stride_x)
            y1 = max(0.0, center_y - float(top_dist) * stride_y)
            x2 = min(float(model_w), center_x + float(right_dist) * stride_x)
            y2 = min(float(model_h), center_y + float(bottom_dist) * stride_y)
            if x2 > x1 and y2 > y1:
                candidates.append([x1, y1, x2, y2, float(scores[grid_y, grid_x])])

    selected: list[list[float]] = []
    for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, _score = candidate
        keep = True
        for kept in selected:
            intersection = max(0.0, min(x2, kept[2]) - max(x1, kept[0])) * max(
                0.0, min(y2, kept[3]) - max(y1, kept[1])
            )
            union = (x2 - x1) * (y2 - y1) + (kept[2] - kept[0]) * (
                kept[3] - kept[1]
            ) - intersection
            if intersection / max(union, 1e-6) >= iou_threshold:
                keep = False
                break
        if keep:
            selected.append(candidate)

    return np.asarray(
        [
            [y1 / model_h, x1 / model_w, y2 / model_h, x2 / model_w, score]
            for x1, y1, x2, y2, score in selected
        ],
        dtype=np.float32,
    ).reshape(-1, 5)


def make_square_context_crop(
    image_bgr: np.ndarray,
    bbox: list[int],
    padding_ratio: float = CROP_PADDING_RATIO,
) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    crop_width = max(1, int(math.ceil(width * (1.0 + 2.0 * padding_ratio))))
    crop_height = max(1, int(math.ceil(height * (1.0 + 2.0 * padding_ratio))))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(math.floor(center_x - crop_width / 2.0))
    crop_y1 = int(math.floor(center_y - crop_height / 2.0))
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    image_h, image_w = image_bgr.shape[:2]
    source_x1 = max(0, crop_x1)
    source_y1 = max(0, crop_y1)
    source_x2 = min(image_w, crop_x2)
    source_y2 = min(image_h, crop_y2)
    rectangular_crop = np.full((crop_height, crop_width, 3), 114, dtype=np.uint8)
    if source_x2 > source_x1 and source_y2 > source_y1:
        target_x1 = source_x1 - crop_x1
        target_y1 = source_y1 - crop_y1
        rectangular_crop[
            target_y1 : target_y1 + (source_y2 - source_y1),
            target_x1 : target_x1 + (source_x2 - source_x1),
        ] = image_bgr[source_y1:source_y2, source_x1:source_x2]
    side = max(crop_width, crop_height)
    crop = np.full((side, side, 3), 114, dtype=np.uint8)
    letterbox_x = (side - crop_width) // 2
    letterbox_y = (side - crop_height) // 2
    crop[
        letterbox_y : letterbox_y + crop_height,
        letterbox_x : letterbox_x + crop_width,
    ] = rectangular_crop
    return crop


def roi_sharpness_score(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = roi
    roi_image = image_bgr[y1:y2, x1:x2]
    if roi_image.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def focus_scores_are_stable(scores: list[float]) -> bool:
    if len(scores) < FOCUS_STABILITY_FRAMES:
        return False
    recent = np.asarray(scores[-FOCUS_STABILITY_FRAMES:], dtype=np.float64)
    mean_score = float(recent.mean())
    if not np.isfinite(recent).all() or mean_score <= 0.0:
        return False
    relative_spread = float((recent.max() - recent.min()) / mean_score)
    return relative_spread <= FOCUS_MAX_RELATIVE_SPREAD


def set_camera_continuous_autofocus(
    capture: cv2.VideoCapture,
    enabled: bool,
) -> bool:
    if not capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enabled else 0):
        return False
    actual = float(capture.get(cv2.CAP_PROP_AUTOFOCUS))
    if actual in (0.0, 1.0):
        return bool(round(actual)) is enabled
    return True


def classifier_training_shortfall(
    labels_dir: Path,
    locked_manifest_path: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    counts = {
        class_name: len(list((labels_dir / class_name).glob("*.png")))
        for class_name in ("false_positive", "true_detection")
    }
    locked_counts = {"false_positive": 1, "true_detection": 1}
    if locked_manifest_path.exists():
        try:
            payload = json.loads(locked_manifest_path.read_text(encoding="utf-8"))
            locked_counts = {"false_positive": 0, "true_detection": 0}
            for sample in payload.get("samples", []):
                class_name = str(sample.get("class_name"))
                if class_name in locked_counts:
                    locked_counts[class_name] += 1
        except (OSError, json.JSONDecodeError):
            pass
    missing = {
        class_name: max(0, 7 + locked_counts[class_name] - counts[class_name])
        for class_name in counts
    }
    return counts, missing


class BeadDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.runtime: HailoRuntime | None = None
        self.model: Any = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Detection HEF not found: {self.model_path}")
        self.runtime = HailoRuntime()
        self.model = self.runtime.create_model(str(self.model_path), "DetectionLabelCollector")
        if self.model is None:
            detail = self.runtime.last_model_error if self.runtime is not None else "unknown error"
            raise RuntimeError(f"Could not load the detection HEF: {detail}")

    def detect(
        self,
        image_bgr: np.ndarray,
        threshold: float = YOLO_SCORE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        with self.lock:
            self._load()
            model_h = int(self.model.input_h)
            model_w = int(self.model.input_w)
            if int(self.model.input_c) != 3:
                raise RuntimeError(
                    f"Detection HEF must use a three-channel input; received {self.model.input_shape}."
                )

            source_h, source_w = image_bgr.shape[:2]
            scale = min(model_w / source_w, model_h / source_h)
            resized_w = int(round(source_w * scale))
            resized_h = int(round(source_h * scale))
            resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            left = (model_w - resized_w) // 2
            top = (model_h - resized_h) // 2
            model_image = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
            model_image[top : top + resized_h, left : left + resized_w] = resized
            model_input = np.ascontiguousarray(cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB))
            output = self.model.run_inference(model_input)
            if isinstance(output, dict):
                rows = decode_yolov8_seg_rows(
                    output,
                    (model_h, model_w),
                    threshold,
                )
            else:
                rows = normalize_nms_rows(output)
            return decode_detections(
                rows,
                (source_h, source_w),
                scale,
                left,
                top,
                threshold,
                (model_h, model_w),
            )

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
        self.runtime = None
        self.model = None


class ResNet18Classifier:
    def __init__(
        self,
        model_path: Path,
        architecture: str = "resnet18",
        true_detection_threshold: float = TRUE_DETECTION_THRESHOLD,
    ):
        self.model_path = model_path
        self.architecture = architecture
        self.true_detection_threshold = true_detection_threshold
        self.model: Any = None
        self.torch: Any = None
        self.transform: Any = None
        self.image_type: Any = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PT model not found: {self.model_path}. Train the classifier first."
            )
        import torch
        from PIL import Image
        from torch import nn
        from torchvision import models, transforms

        if self.architecture == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, 2),
            )
        elif self.architecture == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        else:
            raise ValueError(f"Unsupported classifier architecture: {self.architecture}")
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        self.torch = torch
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.image_type = Image

    def predict_crops(self, crops_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        if not crops_bgr:
            return []
        with self.lock:
            self._load()
            tensors = []
            for crop_bgr in crops_bgr:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(self.image_type.fromarray(rgb)))
            inputs = self.torch.stack(tensors)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(inputs), dim=1).cpu().numpy()
        results = []
        for probability in probabilities:
            class_index = int(float(probability[1]) >= self.true_detection_threshold)
            results.append(
                {
                    "resnet_prediction": "true_detection" if class_index == 1 else "false_positive",
                    "resnet_confidence": float(probability[class_index]),
                    "true_detection_probability": float(probability[1]),
                }
            )
        return results

    def reset(self) -> None:
        with self.lock:
            self.model = None
            self.torch = None
            self.transform = None
            self.image_type = None


class DatasetStore:
    def __init__(self, root: Path, model_name: str = DEFAULT_MODEL_PATH.name):
        self.root = root
        self.model_name = model_name
        self.sessions_dir = root / "sessions"
        self.labels_dir = root / "labels"
        self.lock = threading.Lock()
        for path in (
            self.sessions_dir,
            self.labels_dir / "true_detection",
            self.labels_dir / "false_positive",
        ):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(value: str, name: str) -> str:
        if not SAFE_ID.fullmatch(value or ""):
            raise ValueError(f"Invalid {name}.")
        return value

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / self._validate_id(session_id, "session ID")

    def _manifest_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "manifest.json"

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        path = self._manifest_path(str(manifest["session_id"]))
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)

    def create_session(
        self,
        image_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        original_filename: str,
        confidence_threshold: float = YOLO_SCORE_THRESHOLD,
        roi: tuple[int, int, int, int] | None = None,
        analysis_mode: str = "yolo_only",
        resnet_model: str | None = None,
        classifier_architecture: str | None = None,
        classifier_threshold: float | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        session_dir = self._session_dir(session_id)
        crops_dir = session_dir / "crops"
        crops_dir.mkdir(parents=True)
        cv2.imwrite(str(session_dir / "original.png"), image_bgr)

        annotated = image_bgr.copy()
        if roi is not None and analysis_mode != "yolo_resnet_pt":
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            cv2.rectangle(
                annotated,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                (0, 255, 120),
                3,
                cv2.LINE_AA,
            )
        candidates: list[dict[str, Any]] = []
        detection_count = 0
        for index, detection in enumerate(detections, start=1):
            candidate_id = f"candidate_{index:03d}"
            crop = make_square_context_crop(image_bgr, detection["bbox"])
            crop_relative = Path("sessions") / session_id / "crops" / f"{candidate_id}.png"
            cv2.imwrite(str(self.root / crop_relative), crop)
            x1, y1, x2, y2 = detection["bbox"]
            prediction = detection.get("resnet_prediction")
            show_detection = analysis_mode != "yolo_resnet_pt" or prediction == "true_detection"
            if prediction == "true_detection":
                detection_count += 1
            if show_detection:
                color = (0, 200, 0) if prediction == "true_detection" else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                annotation = str(detection_count if prediction == "true_detection" else index)
                if prediction:
                    annotation += f" detection {float(detection['resnet_confidence']):.0%}"
                cv2.putText(
                    annotated,
                    annotation,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            candidate = {
                "id": candidate_id,
                "number": index,
                "bbox": detection["bbox"],
                "yolo_score": detection["yolo_score"],
                "crop": crop_relative.as_posix(),
                "label": "unreviewed",
                "reviewed_at": None,
            }
            for key in ("resnet_prediction", "resnet_confidence", "true_detection_probability"):
                if key in detection:
                    candidate[key] = detection[key]
            if prediction == "true_detection":
                candidate["detection_number"] = detection_count
            candidates.append(candidate)
        if analysis_mode == "yolo_resnet_pt":
            cv2.rectangle(annotated, (10, 10), (260, 54), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                f"Detection count: {detection_count}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(session_dir / "annotated.png"), annotated)

        manifest = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "original_filename": Path(original_filename).name,
            "original_image": (Path("sessions") / session_id / "original.png").as_posix(),
            "annotated_image": (Path("sessions") / session_id / "annotated.png").as_posix(),
            "model": self.model_name,
            "yolo_threshold": float(confidence_threshold),
            "roi": list(roi) if roi is not None else None,
            "prediction_source": "roi" if roi is not None else "full_image",
            "analysis_mode": analysis_mode,
            "resnet_model": resnet_model,
            "classifier_architecture": classifier_architecture,
            "classifier_threshold": classifier_threshold,
            "yolo_candidate_count": len(detections),
            "detection_count": detection_count if analysis_mode == "yolo_resnet_pt" else None,
            "crop_padding_ratio": CROP_PADDING_RATIO,
            "candidates": candidates,
        }
        self._write_manifest(manifest)
        return manifest

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self._manifest_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown labeling session: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in sorted(self.sessions_dir.glob("*/manifest.json"), reverse=True):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            counts = self.label_counts(manifest)
            sessions.append(
                {
                    "session_id": manifest["session_id"],
                    "created_at": manifest.get("created_at", ""),
                    "original_filename": manifest.get("original_filename", ""),
                    "counts": counts,
                }
            )
        return sessions[:30]

    @staticmethod
    def label_counts(manifest: dict[str, Any]) -> dict[str, int]:
        counts = {label: 0 for label in VALID_LABELS}
        for candidate in manifest.get("candidates", []):
            label = str(candidate.get("label") or "unreviewed")
            if label in counts:
                counts[label] += 1
        return counts

    def set_label(self, session_id: str, candidate_id: str, label: str) -> dict[str, Any]:
        self._validate_id(candidate_id, "candidate ID")
        if label not in VALID_LABELS:
            raise ValueError("Label must be true_detection, false_positive, or unreviewed.")
        with self.lock:
            manifest = self.load_session(session_id)
            candidate = next(
                (item for item in manifest["candidates"] if item.get("id") == candidate_id),
                None,
            )
            if candidate is None:
                raise ValueError("Candidate was not found in this session.")

            crop_path = self.root / candidate["crop"]
            labeled_name = f"{session_id}_{candidate_id}.png"
            for class_name in ("true_detection", "false_positive"):
                labeled_path = self.labels_dir / class_name / labeled_name
                if class_name == label:
                    shutil.copy2(crop_path, labeled_path)
                elif labeled_path.exists():
                    labeled_path.unlink()

            candidate["label"] = label
            candidate["reviewed_at"] = (
                datetime.now().isoformat(timespec="seconds") if label != "unreviewed" else None
            )
            self._write_manifest(manifest)
            return {
                "candidate": candidate,
                "counts": self.label_counts(manifest),
            }


class DetectionWorker(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        detector: BeadDetector,
        store: DatasetStore,
        captured_frame: np.ndarray,
        capture_name: str,
        roi: tuple[int, int, int, int],
        confidence_threshold: float,
        classifier: ResNet18Classifier | None = None,
    ):
        super().__init__()
        self.detector = detector
        self.store = store
        self.captured_frame = captured_frame.copy()
        self.capture_name = capture_name
        self.roi = roi
        self.confidence_threshold = confidence_threshold
        self.classifier = classifier

    @pyqtSlot()
    def run(self) -> None:
        try:
            x1, y1, x2, y2 = self.roi
            roi_frame = self.captured_frame[y1:y2, x1:x2].copy()
            detections = self.detector.detect(
                roi_frame,
                threshold=self.confidence_threshold,
            )
            for detection in detections:
                box_x1, box_y1, box_x2, box_y2 = detection["bbox"]
                detection["bbox"] = [
                    box_x1 + x1,
                    box_y1 + y1,
                    box_x2 + x1,
                    box_y2 + y1,
                ]
            if self.classifier is not None:
                crops = [
                    make_square_context_crop(self.captured_frame, detection["bbox"])
                    for detection in detections
                ]
                predictions = self.classifier.predict_crops(crops)
                for detection, prediction in zip(detections, predictions):
                    detection.update(prediction)
            manifest = self.store.create_session(
                self.captured_frame,
                detections,
                self.capture_name,
                confidence_threshold=self.confidence_threshold,
                roi=self.roi,
                analysis_mode="yolo_resnet_pt" if self.classifier is not None else "yolo_only",
                resnet_model=(self.classifier.model_path.name if self.classifier is not None else None),
                classifier_architecture=(
                    getattr(self.classifier, "architecture", "resnet18")
                    if self.classifier is not None
                    else None
                ),
                classifier_threshold=(
                    float(
                        getattr(
                            self.classifier,
                            "true_detection_threshold",
                            TRUE_DETECTION_THRESHOLD,
                        )
                    )
                    if self.classifier is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "out of physical devices" in message.lower():
                message += " Stop the main jewellery application before running this standalone detector."
            self.failed.emit(message)
            return
        self.completed.emit(manifest)


class ScaledImageLabel(QLabel):
    roi_selected = pyqtSignal(object)

    def __init__(self, minimum_size: tuple[int, int] = (640, 360)):
        super().__init__(alignment=Qt.AlignmentFlag.AlignCenter)
        self._source_image: QImage | None = None
        self._roi_selection_enabled = False
        self._roi_origin = None
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setMinimumSize(*minimum_size)
        self.setStyleSheet("background: #05070a; border: 1px solid #263142; border-radius: 10px;")

    def set_bgr_image(self, image_bgr: np.ndarray | None) -> None:
        if image_bgr is None:
            self._source_image = None
            self.clear()
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._source_image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._refresh()

    def _refresh(self) -> None:
        if self._source_image is None or self.width() <= 0 or self.height() <= 0:
            return
        pixmap = QPixmap.fromImage(self._source_image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def set_roi_selection_enabled(self, enabled: bool) -> None:
        self._roi_selection_enabled = enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        if not enabled:
            self._roi_origin = None
            self._rubber_band.hide()

    def _label_to_image(self, label_x: int, label_y: int) -> tuple[int, int] | None:
        if self._source_image is None:
            return None
        image_w = self._source_image.width()
        image_h = self._source_image.height()
        scale = min(self.width() / image_w, self.height() / image_h)
        shown_w = image_w * scale
        shown_h = image_h * scale
        offset_x = (self.width() - shown_w) / 2.0
        offset_y = (self.height() - shown_h) / 2.0
        image_x = int(np.clip((label_x - offset_x) / scale, 0, image_w - 1))
        image_y = int(np.clip((label_y - offset_y) / scale, 0, image_h - 1))
        return image_x, image_y

    def mousePressEvent(self, event) -> None:
        if (
            self._roi_selection_enabled
            and self._source_image is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._roi_origin = event.position().toPoint()
            self._rubber_band.setGeometry(QRect(self._roi_origin, QSize()))
            self._rubber_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._roi_selection_enabled and self._roi_origin is not None:
            self._rubber_band.setGeometry(
                QRect(self._roi_origin, event.position().toPoint()).normalized()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            self._roi_selection_enabled
            and self._roi_origin is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            rect = QRect(self._roi_origin, event.position().toPoint()).normalized()
            self._roi_origin = None
            self._rubber_band.hide()
            first = self._label_to_image(rect.left(), rect.top())
            second = self._label_to_image(rect.right(), rect.bottom())
            if first is not None and second is not None:
                x1, y1 = first
                x2, y2 = second
                if x2 - x1 >= 20 and y2 - y1 >= 20:
                    self.roi_selected.emit((x1, y1, x2, y2))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CandidateCard(QFrame):
    def __init__(
        self,
        candidate: dict[str, Any],
        crop_path: Path,
        on_label,
        show_label_actions: bool = True,
        classifier_label: str = "ResNet PT",
    ):
        super().__init__()
        self.candidate_id = str(candidate["id"])
        self._on_label = on_label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        heading = QHBoxLayout()
        title = QLabel(
            f"Detection {candidate['detection_number']}"
            if "detection_number" in candidate
            else f"Candidate {candidate['number']}"
        )
        title.setStyleSheet("font-weight: 700; font-size: 15px;")
        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight)
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.status)
        layout.addLayout(heading)

        prediction = candidate.get("resnet_prediction")
        if prediction:
            confidence = float(candidate.get("resnet_confidence", 0.0))
            prediction_label = QLabel(
                f"{classifier_label}: {str(prediction).replace('_', ' ').upper()} ({confidence:.1%})"
            )
            prediction_label.setStyleSheet(
                "color: #86efac; font-weight: 800;"
                if prediction == "true_detection"
                else "color: #fca5a5; font-weight: 800;"
            )
            layout.addWidget(prediction_label)

        crop_label = ScaledImageLabel((160, 160))
        crop_label.setFixedHeight(190)
        crop_label.set_bgr_image(cv2.imread(str(crop_path), cv2.IMREAD_COLOR))
        layout.addWidget(crop_label)

        if show_label_actions:
            actions = QGridLayout()
            true_button = QPushButton("True Detection")
            false_button = QPushButton("False Positive")
            clear_button = QPushButton("Unreviewed")
            true_button.setObjectName("trueButton")
            false_button.setObjectName("falseButton")
            clear_button.setObjectName("clearButton")
            true_button.clicked.connect(lambda: self._on_label(self, "true_detection"))
            false_button.clicked.connect(lambda: self._on_label(self, "false_positive"))
            clear_button.clicked.connect(lambda: self._on_label(self, "unreviewed"))
            actions.addWidget(true_button, 0, 0)
            actions.addWidget(false_button, 0, 1)
            actions.addWidget(clear_button, 1, 0, 1, 2)
            layout.addLayout(actions)
            self.set_label(str(candidate.get("label") or "unreviewed"))
        else:
            self.status.setText("TRUE DETECTION")
            self.setStyleSheet(
                "CandidateCard { border: 2px solid #22c55e; border-radius: 12px; "
                "background: #0e2118; }"
            )

    def set_label(self, label: str) -> None:
        colors = {
            "true_detection": ("#22c55e", "#0e2118"),
            "false_positive": ("#ef4444", "#2a1215"),
            "unreviewed": ("#334155", "#111827"),
        }
        border, background = colors[label]
        self.setStyleSheet(
            f"CandidateCard {{ border: 2px solid {border}; border-radius: 12px; "
            f"background: {background}; }}"
        )
        self.status.setText(label.replace("_", " ").upper())


class BeadFalsePositiveWindow(QMainWindow):
    def __init__(
        self,
        project_manager: ModelProjectManager,
        project: dict[str, Any],
        camera_index: int = 0,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: float = 30.0,
    ):
        super().__init__()
        self.project_manager = project_manager
        self.active_project = project
        self.detector = BeadDetector(Path(project["model_path"]))
        self.classifier = ResNet18Classifier(Path(project["resnet_model_path"]))
        self.candidate_classifier = ResNet18Classifier(Path(project["candidate_model_path"]))
        self.mobilenet_classifier = ResNet18Classifier(
            Path(project["mobilenet_model_path"]),
            architecture="mobilenet_v3_small",
        )
        self.store = DatasetStore(Path(project["dataset_path"]), str(project["hef_file"]))
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.capture: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self._focus_frame_count = 0
        self._focus_scores: deque[float] = deque(maxlen=FOCUS_STABILITY_FRAMES)
        self._focus_ready = False
        self._focus_locked = False
        self._latest_focus_score = 0.0
        self.current_manifest: dict[str, Any] | None = None
        self._detect_thread: QThread | None = None
        self._detect_worker: DetectionWorker | None = None
        self._training_process: QProcess | None = None
        self._training_architecture = ""
        self._training_last_line = ""
        self._close_when_finished = False
        self._candidate_cards: dict[str, CandidateCard] = {}

        self.setWindowTitle("HEF Detection False-Positive Labeler")
        self.resize(1440, 900)
        self.setMinimumSize(980, 650)
        self._build_ui()

        self.camera_timer = QTimer(self)
        self.camera_timer.setInterval(max(1, int(round(1000.0 / max(camera_fps, 1.0)))))
        self.camera_timer.timeout.connect(self._read_camera_frame)
        self._refresh_sessions()
        self._refresh_candidate_controls()
        QTimer.singleShot(0, self.show_live_feed)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        heading = QHBoxLayout()
        heading_text = QVBoxLayout()
        eyebrow = QLabel("FALSE-POSITIVE DATASET TOOL")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("HEF Detection Candidate Labeler")
        title.setObjectName("title")
        subtitle = QLabel("Capture one live frame, run the active HEF, then label its detection candidates.")
        subtitle.setObjectName("subtitle")
        heading_text.addWidget(eyebrow)
        heading_text.addWidget(title)
        heading_text.addWidget(subtitle)
        heading.addLayout(heading_text)
        heading.addStretch()
        heading.addWidget(QLabel("Active HEF"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(190)
        self.project_combo.currentTextChanged.connect(self._project_selected)
        heading.addWidget(self.project_combo)
        self.load_hef_button = QPushButton("Load HEF")
        self.load_hef_button.clicked.connect(self.load_hef)
        heading.addWidget(self.load_hef_button)
        self.train_button = QPushButton("Train ResNet18")
        self.train_button.setObjectName("trainButton")
        self.train_button.clicked.connect(self.train_resnet18)
        heading.addWidget(self.train_button)
        self.train_mobilenet_button = QPushButton("Train MobileNetV3")
        self.train_mobilenet_button.setObjectName("trainButton")
        self.train_mobilenet_button.clicked.connect(self.train_mobilenet_v3)
        heading.addWidget(self.train_mobilenet_button)
        self.promote_button = QPushButton("Promote Candidate")
        self.promote_button.setEnabled(False)
        self.promote_button.clicked.connect(self.promote_candidate)
        heading.addWidget(self.promote_button)
        outer.addLayout(heading)
        self._refresh_project_combo(str(self.active_project["project_id"]))

        content = QHBoxLayout()
        content.setSpacing(12)
        sidebar = QFrame()
        sidebar.setObjectName("panel")
        sidebar.setFixedWidth(245)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QLabel("Saved sessions"))
        self.session_list = QListWidget()
        self.session_list.itemClicked.connect(self._open_saved_session)
        sidebar_layout.addWidget(self.session_list, 1)
        content.addWidget(sidebar)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_camera_page())
        self.pages.addWidget(self._build_review_page())
        content.addWidget(self.pages, 1)
        outer.addLayout(content, 1)

        self.status_label = QLabel("Opening live camera...")
        self.status_label.setObjectName("status")
        outer.addWidget(self.status_label)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #090c12; color: #f6f7fb; font-size: 13px; }
            QLabel#eyebrow { color: #94a3b8; font-size: 11px; font-weight: 700; }
            QLabel#title { font-size: 28px; font-weight: 800; }
            QLabel#subtitle, QLabel#status { color: #a8b2c3; }
            QFrame#panel, QFrame#reviewPanel { background: #0f141f; border: 1px solid #263142; border-radius: 12px; }
            QListWidget { background: #111827; border: 0; border-radius: 8px; padding: 4px; }
            QListWidget::item { padding: 9px 7px; border-radius: 6px; }
            QListWidget::item:selected { background: #1d4ed8; }
            QComboBox { background: #111827; border: 1px solid #374151; border-radius: 7px; padding: 7px; }
            QPushButton { border: 0; border-radius: 8px; padding: 9px 13px; background: #374151; color: white; font-weight: 700; }
            QPushButton:hover { background: #4b5563; }
            QPushButton:disabled { color: #94a3b8; background: #1f2937; }
            QPushButton#captureButton { background: #2563eb; font-size: 16px; padding: 12px 26px; }
            QPushButton#checkButton { background: #7c3aed; font-size: 14px; padding: 12px 18px; }
            QPushButton#trainButton { background: #d97706; padding: 11px 18px; }
            QPushButton#liveButton { background: #2563eb; }
            QPushButton#trueButton { background: #15803d; }
            QPushButton#falseButton { background: #b91c1c; }
            QPushButton#clearButton { padding: 6px; }
            QPushButton#drawRoiButton:checked { background: #d97706; }
            QDoubleSpinBox { background: #111827; border: 1px solid #374151; border-radius: 7px; padding: 7px; }
            QScrollArea { background: transparent; border: 0; }
            """
        )

    def _refresh_project_combo(self, selected_project_id: str) -> None:
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for project in self.project_manager.list_projects():
            self.project_combo.addItem(str(project["project_id"]))
        index = self.project_combo.findText(selected_project_id)
        if index >= 0:
            self.project_combo.setCurrentIndex(index)
        self.project_combo.blockSignals(False)

    @pyqtSlot(str)
    def _project_selected(self, project_id: str) -> None:
        if not project_id or project_id == str(self.active_project["project_id"]):
            return
        if self._detect_thread is not None or self._training_process is not None:
            self._refresh_project_combo(str(self.active_project["project_id"]))
            return
        try:
            project = self.project_manager.load_project(project_id)
        except (FileNotFoundError, ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            self.status_label.setText(f"Could not open HEF project: {exc}")
            self._refresh_project_combo(str(self.active_project["project_id"]))
            return
        was_live = self.camera_timer.isActive()
        self.camera_timer.stop()
        self.detector.close()
        self.classifier.reset()
        self.candidate_classifier.reset()
        self.mobilenet_classifier.reset()
        self.active_project = project
        self.detector = BeadDetector(Path(project["model_path"]))
        self.classifier = ResNet18Classifier(Path(project["resnet_model_path"]))
        self.candidate_classifier = ResNet18Classifier(Path(project["candidate_model_path"]))
        self.mobilenet_classifier = ResNet18Classifier(
            Path(project["mobilenet_model_path"]),
            architecture="mobilenet_v3_small",
        )
        self.store = DatasetStore(Path(project["dataset_path"]), str(project["hef_file"]))
        self.current_manifest = None
        self.pages.setCurrentIndex(0)
        self._reset_focus_tracking()
        self._refresh_sessions()
        self._refresh_project_combo(project_id)
        self._refresh_candidate_controls()
        self.status_label.setText(
            f"Active HEF: {project['hef_file']}. Dataset and classifier models switched to {project_id}."
        )
        if was_live and self.capture is not None:
            self.camera_timer.start()

    @pyqtSlot()
    def load_hef(self) -> None:
        if self._detect_thread is not None or self._training_process is not None:
            return
        filename, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load compatible detection HEF",
            str(REPO_DIR),
            "Hailo Executable Format (*.hef)",
        )
        if not filename:
            return
        try:
            project = self.project_manager.import_hef(Path(filename))
        except (ValueError, OSError) as exc:
            self.status_label.setText(f"HEF not loaded: {exc}")
            return
        self._refresh_project_combo(str(project["project_id"]))
        if str(project["project_id"]) == str(self.active_project["project_id"]):
            self.status_label.setText(f"HEF project already active: {project['hef_file']}.")
            return
        self._project_selected(str(project["project_id"]))

    def _load_candidate_report(self) -> dict[str, Any] | None:
        report_path = Path(self.active_project["candidate_report_path"])
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _refresh_candidate_controls(self) -> None:
        candidate_path = Path(self.active_project["candidate_model_path"])
        working_path = Path(self.active_project["resnet_model_path"])
        report = self._load_candidate_report()
        different_from_working = bool(
            candidate_path.exists()
            and working_path.exists()
            and not filecmp.cmp(candidate_path, working_path, shallow=False)
        )
        self.promote_button.setEnabled(
            bool(
                report
                and report.get("promotion_recommended")
                and different_from_working
                and self._training_process is None
                and self._detect_thread is None
            )
        )
        self._update_capture_availability()

    @pyqtSlot()
    def promote_candidate(self) -> None:
        if self._training_process is not None or self._detect_thread is not None:
            return
        report = self._load_candidate_report()
        candidate_path = Path(self.active_project["candidate_model_path"])
        working_path = Path(self.active_project["resnet_model_path"])
        if not report or not report.get("promotion_recommended"):
            self.status_label.setText(
                "Candidate promotion blocked. It must reduce false-positive acceptance and improve F0.5 on the locked crops."
            )
            return
        if not candidate_path.exists():
            self.status_label.setText(f"Candidate PT model not found: {candidate_path}")
            return
        shutil.copy2(candidate_path, working_path)
        self.classifier.reset()
        self._refresh_candidate_controls()
        self.status_label.setText(
            f"Candidate promoted to {working_path.name} for this standalone project. "
            "The production model in EMBSYS-AI/models was not changed."
        )

    def _build_camera_page(self) -> QWidget:
        page = QFrame()
        page.setObjectName("panel")
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        live_title = QLabel("LIVE CAMERA")
        live_title.setStyleSheet("font-size: 18px; font-weight: 800;")
        self.camera_name_label = QLabel(f"Camera {self.camera_index}")
        self.camera_name_label.setObjectName("subtitle")
        top.addWidget(live_title)
        top.addStretch()
        top.addWidget(self.camera_name_label)
        layout.addLayout(top)
        self.live_image = ScaledImageLabel()
        self.live_image.roi_selected.connect(self._set_roi)
        layout.addWidget(self.live_image, 1)
        controls = QHBoxLayout()
        self.draw_roi_button = QPushButton("Draw ROI")
        self.draw_roi_button.setObjectName("drawRoiButton")
        self.draw_roi_button.setCheckable(True)
        self.draw_roi_button.toggled.connect(self.live_image.set_roi_selection_enabled)
        self.draw_roi_button.toggled.connect(self._roi_draw_toggled)
        self.clear_roi_button = QPushButton("Clear ROI")
        self.clear_roi_button.clicked.connect(self._clear_roi)
        self.roi_label = QLabel("ROI: not set")
        self.roi_label.setObjectName("subtitle")
        controls.addWidget(self.draw_roi_button)
        controls.addWidget(self.clear_roi_button)
        controls.addWidget(self.roi_label)
        self.focus_label = QLabel("Focus: warming up")
        self.focus_label.setObjectName("subtitle")
        controls.addWidget(self.focus_label)
        controls.addStretch()
        controls.addWidget(QLabel("Confidence"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.20, 0.95)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(YOLO_SCORE_THRESHOLD)
        controls.addWidget(self.confidence_spin)
        self.capture_button = QPushButton("Capture")
        self.capture_button.setObjectName("captureButton")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_current_frame)
        controls.addWidget(self.capture_button)
        self.check_button = QPushButton("Check with ResNet and PT")
        self.check_button.setObjectName("checkButton")
        self.check_button.setEnabled(False)
        self.check_button.clicked.connect(self.check_with_resnet)
        controls.addWidget(self.check_button)
        self.check_candidate_button = QPushButton("Check Candidate PT")
        self.check_candidate_button.setObjectName("checkButton")
        self.check_candidate_button.setEnabled(False)
        self.check_candidate_button.clicked.connect(self.check_with_candidate)
        controls.addWidget(self.check_candidate_button)
        self.check_mobilenet_button = QPushButton("Check HEF BBoxes + MobileNet")
        self.check_mobilenet_button.setObjectName("checkButton")
        self.check_mobilenet_button.setEnabled(False)
        self.check_mobilenet_button.clicked.connect(self.check_with_mobilenet)
        controls.addWidget(self.check_mobilenet_button)
        layout.addLayout(controls)
        return page

    def _build_review_page(self) -> QWidget:
        page = QFrame()
        page.setObjectName("reviewPanel")
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        headings = QVBoxLayout()
        self.review_source = QLabel("CAPTURED FRAME")
        self.review_source.setObjectName("eyebrow")
        review_title = QLabel("Review candidates")
        review_title.setStyleSheet("font-size: 20px; font-weight: 800;")
        self.review_title = review_title
        self.review_settings = QLabel()
        self.review_settings.setObjectName("subtitle")
        headings.addWidget(self.review_source)
        headings.addWidget(review_title)
        headings.addWidget(self.review_settings)
        top.addLayout(headings)
        top.addStretch()
        self.counts_label = QLabel()
        top.addWidget(self.counts_label)
        self.live_button = QPushButton("Live Feed")
        self.live_button.setObjectName("liveButton")
        self.live_button.clicked.connect(self.show_live_feed)
        top.addWidget(self.live_button)
        layout.addLayout(top)

        body = QHBoxLayout()
        self.annotated_image = ScaledImageLabel((480, 320))
        body.addWidget(self.annotated_image, 3)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.candidate_container = QWidget()
        self.candidate_layout = QGridLayout(self.candidate_container)
        self.candidate_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.candidate_container)
        body.addWidget(scroll, 2)
        layout.addLayout(body, 1)
        return page

    def _open_camera(self) -> bool:
        backend = cv2.CAP_ANY
        if platform.system() == "Linux":
            backend = cv2.CAP_V4L2
        elif platform.system() == "Windows":
            backend = cv2.CAP_DSHOW
        self.capture = cv2.VideoCapture(self.camera_index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            self.status_label.setText(f"Could not open camera {self.camera_index}.")
            self.capture_button.setEnabled(False)
            self.check_button.setEnabled(False)
            return False
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_value = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        self.camera_name_label.setText(
            f"Camera {self.camera_index} | {actual_width}x{actual_height} {fourcc.strip()}"
        )
        return True

    @pyqtSlot()
    def _read_camera_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.status_label.setText(f"Camera {self.camera_index} frame read failed.")
            return
        self.latest_frame = frame.copy()
        self._show_live_frame(frame)
        self._update_focus_state(frame)

    def _reset_focus_tracking(self) -> None:
        self._focus_frame_count = 0
        self._focus_scores.clear()
        self._focus_ready = False
        self._focus_locked = False
        self._latest_focus_score = 0.0
        if self.capture is not None:
            set_camera_continuous_autofocus(self.capture, True)
        self.focus_label.setText("Focus: warming up")
        self.focus_label.setStyleSheet("color: #fbbf24;")
        self.capture_button.setEnabled(False)
        self.check_button.setEnabled(False)

    def _update_focus_state(self, frame: np.ndarray) -> None:
        self._focus_frame_count += 1
        if self.roi is not None:
            self._latest_focus_score = roi_sharpness_score(frame, self.roi)
            self._focus_scores.append(self._latest_focus_score)

        if self._focus_frame_count < FOCUS_WARMUP_FRAMES:
            self._focus_ready = False
            self.focus_label.setText(
                f"Focus: warming up {self._focus_frame_count}/{FOCUS_WARMUP_FRAMES}"
            )
            self.focus_label.setStyleSheet("color: #fbbf24;")
            self.status_label.setText(
                "Focusing camera... Draw the ROI and keep the jewellery still."
            )
        elif self.roi is None:
            self._focus_ready = False
            self.focus_label.setText("Focus: draw ROI")
            self.focus_label.setStyleSheet("color: #fbbf24;")
            self.status_label.setText("Autofocus warmed up. Draw an ROI around the jewellery.")
        elif not focus_scores_are_stable(list(self._focus_scores)):
            self._focus_ready = False
            self.focus_label.setText(f"Focus: adjusting ({self._latest_focus_score:.0f})")
            self.focus_label.setStyleSheet("color: #fbbf24;")
            self.status_label.setText(
                "Hold the jewellery still while focus stabilizes inside the ROI."
            )
        else:
            self._focus_ready = True
            if not self._focus_locked and self.capture is not None:
                self._focus_locked = set_camera_continuous_autofocus(
                    self.capture,
                    False,
                )
            focus_state = "locked" if self._focus_locked else "ready"
            self.focus_label.setText(
                f"Focus: {focus_state} ({self._latest_focus_score:.0f})"
            )
            self.focus_label.setStyleSheet("color: #86efac;")
            self.status_label.setText(
                f"Focus {focus_state}. Confidence {self.confidence_spin.value():.2f}. Select Capture."
            )
        self._update_capture_availability()

    def _update_capture_availability(self) -> None:
        enabled = (
            self.roi is not None
            and self._focus_ready
            and self._detect_thread is None
            and self._training_process is None
            and self.capture is not None
            and self.camera_timer.isActive()
            and not self.draw_roi_button.isChecked()
        )
        self.capture_button.setEnabled(enabled)
        self.check_button.setEnabled(enabled and self.classifier.model_path.exists())
        self.check_candidate_button.setEnabled(
            enabled and self.candidate_classifier.model_path.exists()
        )
        self.check_mobilenet_button.setEnabled(
            enabled and self.mobilenet_classifier.model_path.exists()
        )

    def _show_live_frame(self, frame: np.ndarray) -> None:
        display = frame.copy()
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(
                display,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (0, 255, 120),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "HEF ROI",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )
        self.live_image.set_bgr_image(display)

    @pyqtSlot(bool)
    def _roi_draw_toggled(self, checked: bool) -> None:
        if checked:
            self.status_label.setText("Drag on the live image to draw the HEF inference ROI.")
        self._update_capture_availability()

    @pyqtSlot(object)
    def _set_roi(self, selected: tuple[int, int, int, int]) -> None:
        if self.latest_frame is None:
            return
        frame_h, frame_w = self.latest_frame.shape[:2]
        x1, y1, x2, y2 = selected
        self.roi = (
            max(0, min(frame_w - 1, x1)),
            max(0, min(frame_h - 1, y1)),
            max(1, min(frame_w, x2 + 1)),
            max(1, min(frame_h, y2 + 1)),
        )
        self._focus_scores.clear()
        self._focus_ready = False
        self._focus_locked = False
        if self.capture is not None:
            set_camera_continuous_autofocus(self.capture, True)
        self.draw_roi_button.setChecked(False)
        self.roi_label.setText(
            f"ROI: {self.roi[2] - self.roi[0]} x {self.roi[3] - self.roi[1]}"
        )
        self.focus_label.setText("Focus: adjusting")
        self.focus_label.setStyleSheet("color: #fbbf24;")
        self._update_capture_availability()
        self._show_live_frame(self.latest_frame)
        self.status_label.setText("Hold the jewellery still while focus stabilizes inside the ROI.")

    @pyqtSlot()
    def _clear_roi(self) -> None:
        self.roi = None
        self._focus_scores.clear()
        self._focus_ready = False
        self.draw_roi_button.setChecked(False)
        self.roi_label.setText("ROI: not set")
        self.focus_label.setText("Focus: draw ROI")
        self.focus_label.setStyleSheet("color: #fbbf24;")
        self._update_capture_availability()
        if self.latest_frame is not None:
            self._show_live_frame(self.latest_frame)
        self.status_label.setText("ROI cleared. Draw an ROI around the jewellery.")

    @pyqtSlot()
    def show_live_feed(self) -> None:
        if self._detect_thread is not None:
            return
        self.pages.setCurrentIndex(0)
        self.session_list.clearSelection()
        if self.capture is None and not self._open_camera():
            return
        self._reset_focus_tracking()
        self.camera_timer.start()
        self.status_label.setText("Starting live camera feed and autofocus...")

    @pyqtSlot()
    def capture_current_frame(self) -> None:
        self._capture_and_detect(classifier=None)

    @pyqtSlot()
    def check_with_resnet(self) -> None:
        self._capture_and_detect(classifier=self.classifier)

    @pyqtSlot()
    def check_with_candidate(self) -> None:
        self._capture_and_detect(classifier=self.candidate_classifier)

    @pyqtSlot()
    def check_with_mobilenet(self) -> None:
        self._capture_and_detect(classifier=self.mobilenet_classifier)

    def _capture_and_detect(self, classifier: ResNet18Classifier | None) -> None:
        if (
            self.latest_frame is None
            or self.roi is None
            or not self._focus_ready
            or self._detect_thread is not None
            or self._training_process is not None
        ):
            return
        captured_frame = self.latest_frame.copy()
        captured_roi = self.roi
        confidence_threshold = float(self.confidence_spin.value())
        capture_name = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.camera_timer.stop()
        self._show_live_frame(captured_frame)
        self.capture_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.check_candidate_button.setEnabled(False)
        self.check_mobilenet_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.train_mobilenet_button.setEnabled(False)
        self.promote_button.setEnabled(False)
        self.load_hef_button.setEnabled(False)
        self.project_combo.setEnabled(False)
        self.draw_roi_button.setEnabled(False)
        self.clear_roi_button.setEnabled(False)
        self.confidence_spin.setEnabled(False)
        self.session_list.setEnabled(False)
        action = (
            f"Running the HEF, then checking each candidate with {classifier.model_path.name}"
            if classifier is not None
            else "Running the active HEF"
        )
        self.status_label.setText(
            f"Captured frame frozen. {action} inside the ROI at {confidence_threshold:.2f} confidence..."
        )

        thread = QThread(self)
        worker = DetectionWorker(
            self.detector,
            self.store,
            captured_frame,
            capture_name,
            captured_roi,
            confidence_threshold,
            classifier=classifier,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._detection_completed)
        worker.failed.connect(self._detection_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._detection_thread_stopped)
        thread.finished.connect(thread.deleteLater)
        self._detect_thread = thread
        self._detect_worker = worker
        thread.start()

    @pyqtSlot(object)
    def _detection_completed(self, manifest: dict[str, Any]) -> None:
        self._show_manifest(manifest)
        self._refresh_sessions(manifest["session_id"])
        self.pages.setCurrentIndex(1)
        count = len(manifest.get("candidates", []))
        if manifest.get("analysis_mode") == "yolo_resnet_pt":
            architecture = str(manifest.get("classifier_architecture") or "resnet18")
            classifier_name = (
                "MobileNetV3-Small" if architecture == "mobilenet_v3_small" else "ResNet18"
            )
            self.status_label.setText(
                f"The HEF produced {count} candidate(s) for {classifier_name}. "
                f"Final detection count: {int(manifest.get('detection_count', 0))}."
            )
        else:
            self.status_label.setText(f"The HEF found {count} candidate(s). Review each detection.")

    @pyqtSlot(str)
    def _detection_failed(self, message: str) -> None:
        self.status_label.setText(f"HEF inference failed: {message}")
        self._reset_focus_tracking()
        self.camera_timer.start()

    @pyqtSlot()
    def _detection_thread_stopped(self) -> None:
        self._detect_thread = None
        self._detect_worker = None
        self.session_list.setEnabled(True)
        self.draw_roi_button.setEnabled(True)
        self.clear_roi_button.setEnabled(True)
        self.confidence_spin.setEnabled(True)
        self.train_button.setEnabled(True)
        self.train_mobilenet_button.setEnabled(True)
        self.load_hef_button.setEnabled(True)
        self.project_combo.setEnabled(True)
        self._refresh_candidate_controls()
        if self._close_when_finished:
            self.close()

    @pyqtSlot()
    def train_resnet18(self) -> None:
        self._start_classifier_training("resnet18")

    @pyqtSlot()
    def train_mobilenet_v3(self) -> None:
        self._start_classifier_training("mobilenet_v3_small")

    def _start_classifier_training(self, architecture: str) -> None:
        if self._training_process is not None or self._detect_thread is not None:
            return
        if not RESNET_TRAIN_SCRIPT.exists():
            self.status_label.setText(f"Training script not found: {RESNET_TRAIN_SCRIPT}")
            return
        is_mobilenet = architecture == "mobilenet_v3_small"
        classifier = self.mobilenet_classifier if is_mobilenet else self.candidate_classifier
        model_label = "MobileNetV3-Small" if is_mobilenet else "ResNet18"
        counts, missing = classifier_training_shortfall(
            self.store.labels_dir,
            Path(self.active_project["locked_evaluation_path"]),
        )
        missing_parts = []
        if missing["false_positive"]:
            missing_parts.append(f"{missing['false_positive']} more False Positive")
        if missing["true_detection"]:
            missing_parts.append(f"{missing['true_detection']} more True Detection")
        if missing_parts:
            self.status_label.setText(
                f"{model_label} training not started: label "
                + " and ".join(missing_parts)
                + f" candidate(s). Current labels: {counts['false_positive']} false, "
                f"{counts['true_detection']} true."
            )
            return
        base_model_path = (
            Path(self.active_project["root"])
            / MOBILENET_DIR_NAME
            / f"{self.active_project['project_id']}_mobilenet_v3_working.pt"
            if is_mobilenet
            else self.classifier.model_path
        )
        self._resume_camera_after_training = self.camera_timer.isActive()
        self._training_last_line = ""
        self._training_architecture = architecture
        self.camera_timer.stop()
        self.train_button.setEnabled(False)
        self.train_mobilenet_button.setEnabled(False)
        self.load_hef_button.setEnabled(False)
        self.project_combo.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.check_candidate_button.setEnabled(False)
        self.check_mobilenet_button.setEnabled(False)
        self.promote_button.setEnabled(False)
        self.candidate_container.setEnabled(False)
        self.status_label.setText(f"Training isolated {model_label} candidate...")

        process = QProcess(self)
        process.setWorkingDirectory(str(RESNET_DIR))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_training_output)
        process.finished.connect(self._training_finished)
        self._training_process = process
        process.start(
            sys.executable,
            [
                "-u",
                str(RESNET_TRAIN_SCRIPT),
                "--dataset",
                str(self.store.labels_dir),
                "--output",
                str(classifier.model_path),
                "--base-model",
                str(base_model_path),
                "--locked-evaluation-manifest",
                str(self.active_project["locked_evaluation_path"]),
                "--architecture",
                architecture,
                "--learning-rate",
                "0.0001" if is_mobilenet else "0.001",
                "--false-positive-weight",
                "2.0",
                "--true-detection-threshold",
                str(TRUE_DETECTION_THRESHOLD),
            ],
        )
        if not process.waitForStarted(3000):
            self._training_process = None
            self._training_architecture = ""
            self.train_button.setEnabled(True)
            self.train_mobilenet_button.setEnabled(True)
            self.load_hef_button.setEnabled(True)
            self.project_combo.setEnabled(True)
            self.candidate_container.setEnabled(True)
            self.status_label.setText(
                f"Could not start {model_label} training: {process.errorString()}"
            )
            if self._resume_camera_after_training and self.capture is not None:
                self.camera_timer.start()
            self._refresh_candidate_controls()

    @pyqtSlot()
    def _read_training_output(self) -> None:
        if self._training_process is None:
            return
        output = bytes(self._training_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            self._training_last_line = lines[-1]
            model_label = (
                "MobileNetV3-Small"
                if self._training_architecture == "mobilenet_v3_small"
                else "ResNet18"
            )
            self.status_label.setText(f"{model_label} training: {self._training_last_line}")

    def _training_finished(self, exit_code: int, _exit_status) -> None:
        process = self._training_process
        if process is not None:
            remaining = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        else:
            remaining = ""
        architecture = self._training_architecture
        is_mobilenet = architecture == "mobilenet_v3_small"
        classifier = self.mobilenet_classifier if is_mobilenet else self.candidate_classifier
        report_path = (
            Path(self.active_project["mobilenet_report_path"])
            if is_mobilenet
            else Path(self.active_project["candidate_report_path"])
        )
        model_label = "MobileNetV3-Small" if is_mobilenet else "ResNet18"
        self._training_process = None
        self._training_architecture = ""
        self.train_button.setEnabled(True)
        self.train_mobilenet_button.setEnabled(True)
        self.load_hef_button.setEnabled(True)
        self.project_combo.setEnabled(True)
        self.candidate_container.setEnabled(True)
        if exit_code == 0 and classifier.model_path.exists():
            classifier.reset()
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}
            candidate_metrics = report.get("candidate_locked_metrics") or {}
            promotion_recommended = bool(report.get("promotion_recommended"))
            self.status_label.setText(
                f"{model_label} training complete: locked accuracy "
                f"{float(candidate_metrics.get('accuracy', 0.0)):.1%}; "
                f"false accepted {float(candidate_metrics.get('false_positive_acceptance_rate', 0.0)):.1%}; "
                f"true recall {float(candidate_metrics.get('true_detection_recall', 0.0)):.1%}. "
                + (
                    "Candidate passed; Promote Candidate is available."
                    if promotion_recommended and not is_mobilenet
                    else "Experiment saved; the main application was not changed."
                )
            )
        else:
            lines = [line.strip() for line in remaining.splitlines() if line.strip()]
            detail = lines[-1] if lines else self._training_last_line
            detail = detail or f"process exited with code {exit_code}"
            self.status_label.setText(f"{model_label} training failed: {detail}")
        if self._resume_camera_after_training and self.capture is not None:
            self.camera_timer.start()
        self._refresh_candidate_controls()
        if self._close_when_finished:
            self.close()

    def _refresh_sessions(self, selected_session_id: str | None = None) -> None:
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session in self.store.list_sessions():
            counts = session["counts"]
            text = (
                f"{session['created_at']}\n"
                f"{counts['false_positive']} false | {counts['true_detection']} true | "
                f"{counts['unreviewed']} open"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, session["session_id"])
            self.session_list.addItem(item)
            if session["session_id"] == selected_session_id:
                self.session_list.setCurrentItem(item)
        self.session_list.blockSignals(False)

    @pyqtSlot(QListWidgetItem)
    def _open_saved_session(self, item: QListWidgetItem) -> None:
        if self._detect_thread is not None:
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole))
        try:
            manifest = self.store.load_session(session_id)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.status_label.setText(str(exc))
            return
        self.camera_timer.stop()
        self._show_manifest(manifest)
        self.pages.setCurrentIndex(1)
        self.status_label.setText("Saved session loaded. Labels can be updated.")

    def _show_manifest(self, manifest: dict[str, Any]) -> None:
        self.current_manifest = manifest
        self.review_source.setText(str(manifest.get("original_filename") or "CAPTURED FRAME"))
        roi = manifest.get("roi")
        source_text = "ROI" if roi else "full frame"
        is_resnet_check = manifest.get("analysis_mode") == "yolo_resnet_pt"
        self.review_title.setText(
            "True detections after PT checking" if is_resnet_check else "Review candidates"
        )
        self.review_settings.setText(
            f"HEF source: {source_text}  |  Confidence: {float(manifest.get('yolo_threshold', YOLO_SCORE_THRESHOLD)):.2f}"
            + (f"  |  PT: {manifest.get('resnet_model')}" if is_resnet_check else "")
        )
        annotated = cv2.imread(
            str(self.store.root / manifest["annotated_image"]),
            cv2.IMREAD_COLOR,
        )
        self.annotated_image.set_bgr_image(annotated)

        while self.candidate_layout.count():
            item = self.candidate_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._candidate_cards.clear()
        candidates = manifest.get("candidates", [])
        if is_resnet_check:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("resnet_prediction") == "true_detection"
            ]
        if candidates:
            classifier_label = (
                "MobileNetV3 PT"
                if manifest.get("classifier_architecture") == "mobilenet_v3_small"
                else "ResNet18 PT"
            )
            for index, candidate in enumerate(candidates):
                card = CandidateCard(
                    candidate,
                    self.store.root / candidate["crop"],
                    self._set_candidate_label,
                    show_label_actions=not is_resnet_check,
                    classifier_label=classifier_label,
                )
                self._candidate_cards[str(candidate["id"])] = card
                self.candidate_layout.addWidget(card, index, 0)
        else:
            empty = QLabel(
                "No true detections remained after PT checking."
                if is_resnet_check
                else "The active HEF did not return any candidates for this frame."
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color: #a8b2c3; padding: 30px;")
            self.candidate_layout.addWidget(empty, 0, 0)
        self._update_counts()

    def _set_candidate_label(self, card: CandidateCard, label: str) -> None:
        if self.current_manifest is None:
            return
        try:
            result = self.store.set_label(
                str(self.current_manifest["session_id"]),
                card.candidate_id,
                label,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.status_label.setText(f"Could not save label: {exc}")
            return
        card.set_label(label)
        self.current_manifest = self.store.load_session(str(self.current_manifest["session_id"]))
        self._update_counts(result["counts"])
        self._refresh_sessions(str(self.current_manifest["session_id"]))
        self.status_label.setText(f"Candidate saved as {label.replace('_', ' ')}.")

    def _update_counts(self, counts: dict[str, int] | None = None) -> None:
        if counts is None:
            counts = self.store.label_counts(self.current_manifest or {})
        label_text = (
            f"{counts['true_detection']} true  |  {counts['false_positive']} false  |  "
            f"{counts['unreviewed']} unreviewed"
        )
        if (self.current_manifest or {}).get("analysis_mode") == "yolo_resnet_pt":
            label_text = f"Detection count: {int((self.current_manifest or {}).get('detection_count', 0))}"
        self.counts_label.setText(label_text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._training_process is not None:
            self._close_when_finished = True
            self.status_label.setText("Waiting for ResNet18 training to finish before closing...")
            event.ignore()
            return
        if self._detect_thread is not None:
            self._close_when_finished = True
            self.status_label.setText("Waiting for the current HEF inference to finish before closing...")
            event.ignore()
            return
        self.camera_timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.detector.close()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="PyQt HEF detection false-positive labeling tool.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--projects-dir", type=Path, default=PROJECTS_DIR)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS")
    args = parser.parse_args()

    application = QApplication.instance() or QApplication(sys.argv)
    project_manager = ModelProjectManager(args.projects_dir.resolve())
    project = project_manager.import_hef(args.model.resolve())
    if str(project["project_id"]) == DEFAULT_PROJECT_ID:
        migrate_legacy_bead_project(project)
    window = BeadFalsePositiveWindow(
        project_manager,
        project,
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
    )
    window.show()
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()
