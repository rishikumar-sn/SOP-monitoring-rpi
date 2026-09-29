#!/usr/bin/env python3
"""Review saved Stone Analysis regions as true stones or false positives."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


TOOL_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOL_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
DEFAULT_RUNTIME_ROOT = REPO_DIR / "runtime_sessions"
DEFAULT_DATASET_DIR = TOOL_DIR / "stone_review_dataset"
DEFAULT_IMAGE_ROOT = REPO_DIR.parent / "Necklace"
DEFAULT_ANALYSIS_CACHE = DEFAULT_DATASET_DIR / "analysis_cache"

STONE_COLORS = [
    "Red",
    "Orange",
    "Yellow/Gold",
    "Green",
    "Blue",
    "Purple/Violet",
    "Pink",
    "Black",
    "White/Colorless",
    "Multicolor/Color-changing",
]

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


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _existing_artifact(path_value: str | None, report_path: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_file():
        return path
    fallback = report_path.parent / path.name
    return fallback if fallback.is_file() else None


def _composite_masked_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4:
        return image[:, :, :3].copy()
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    bgr = image[:, :, :3].astype(np.float32)
    return np.clip(bgr * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def reconstruct_region_mask(
    color_mask_bgr: np.ndarray,
    region: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    """Recover a saved final mask from its rendered fill and white contour."""
    height, width = color_mask_bgr.shape[:2]
    x, y, box_width, box_height = (int(value) for value in region["bbox"])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + box_width), min(height, y + box_height)
    output = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return output, False

    predicted_color = str(region.get("color") or "")
    fill_bgr = COLOR_DRAW_BGR.get(predicted_color)
    if fill_bgr is None:
        return output, False

    crop = color_mask_bgr[y1:y2, x1:x2]
    fill = np.all(crop == np.asarray(fill_bgr, dtype=np.uint8), axis=2)
    contour = np.all(crop == np.asarray((245, 245, 245), dtype=np.uint8), axis=2)
    possible = np.where(fill | contour, 255, 0).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        possible,
        8,
    )
    if component_count <= 1:
        return output, False

    center_x, center_y = (int(value) for value in region.get("center", [x, y]))
    local_x = min(max(center_x - x1, 0), x2 - x1 - 1)
    local_y = min(max(center_y - y1, 0), y2 - y1 - 1)
    selected = int(labels[local_y, local_x])
    if selected == 0:
        selected = min(
            range(1, component_count),
            key=lambda label: (
                (float(centroids[label, 0]) - local_x) ** 2
                + (float(centroids[label, 1]) - local_y) ** 2
            ),
        )
    local_mask = np.where(labels == selected, 255, 0).astype(np.uint8)
    output[y1:y2, x1:x2] = local_mask
    expected_area = int(region.get("area_px") or 0)
    exact = expected_area > 0 and cv2.countNonZero(local_mask) == expected_area
    return output, exact


def make_context_crop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    context_scale: float = 2.5,
    minimum_side: int = 48,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Cannot export an empty region mask.")
    x, y, width, height = cv2.boundingRect(points)
    side = max(minimum_side, int(round(max(width, height) * context_scale)))
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    crop_x1 = int(round(center_x - side / 2.0))
    crop_y1 = int(round(center_y - side / 2.0))
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    crop = np.full((side, side, 3), 114, dtype=np.uint8)
    crop_mask = np.zeros((side, side), dtype=np.uint8)
    image_height, image_width = image_bgr.shape[:2]
    source_x1, source_y1 = max(0, crop_x1), max(0, crop_y1)
    source_x2, source_y2 = min(image_width, crop_x2), min(image_height, crop_y2)
    if source_x2 > source_x1 and source_y2 > source_y1:
        target_x1, target_y1 = source_x1 - crop_x1, source_y1 - crop_y1
        target_x2 = target_x1 + source_x2 - source_x1
        target_y2 = target_y1 + source_y2 - source_y1
        crop[target_y1:target_y2, target_x1:target_x2] = image_bgr[
            source_y1:source_y2,
            source_x1:source_x2,
        ]
        crop_mask[target_y1:target_y2, target_x1:target_x2] = mask[
            source_y1:source_y2,
            source_x1:source_x2,
        ]
    return crop, crop_mask, [crop_x1, crop_y1, side, side]


@dataclass
class ReviewRegion:
    data: dict[str, Any]
    mask: np.ndarray
    exact_reconstruction: bool
    sample_id: str

    @property
    def region_id(self) -> int:
        return int(self.data.get("region_id") or 0)


@dataclass
class ReviewImage:
    report_path: Path
    session_name: str
    stage_name: str
    jewel_id: int
    image_bgr: np.ndarray
    bbox_global: list[int]
    regions: list[ReviewRegion]
    source_image_path: Path | None = None
    analysis_backend: str = "saved_session"

    @property
    def title(self) -> str:
        return f"{self.session_name} / {self.stage_name} / Jewel {self.jewel_id}"


def discover_dataset_image_paths(image_root: Path) -> list[Path]:
    """Use ROI captures when the folder contains paired full/ROI images."""
    roi_paths = sorted(
        path for path in image_root.glob("*_roi.*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    )
    if roi_paths:
        return roi_paths
    return sorted(
        path for path in image_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    )


def _sample_id(report_path: Path, runtime_root: Path, jewel_id: int, region_id: int) -> str:
    try:
        report_key = str(report_path.resolve().relative_to(runtime_root.resolve()))
    except ValueError:
        report_key = str(report_path.resolve())
    raw = f"{report_key}|{jewel_id}|{region_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def discover_review_images(runtime_root: Path) -> tuple[list[ReviewImage], list[str]]:
    items: list[tuple[float, ReviewImage]] = []
    errors: list[str] = []
    for report_path in runtime_root.rglob("*gem_report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            session_name = report_path.parent.parent.name
            stage_name = report_path.parent.name
            for jewel in report.get("jewels") or []:
                outputs = jewel.get("outputs") or {}
                masked_path = _existing_artifact(outputs.get("masked_png"), report_path)
                color_mask_path = _existing_artifact(outputs.get("color_mask_png"), report_path)
                if masked_path is None or color_mask_path is None:
                    raise ValueError("masked or color-mask artifact is missing")
                masked = _read_image(masked_path, cv2.IMREAD_UNCHANGED)
                image_bgr = _composite_masked_image(masked)
                color_mask = _read_image(color_mask_path)
                if image_bgr.shape[:2] != color_mask.shape[:2]:
                    raise ValueError("masked image and color mask have different sizes")
                jewel_id = int(jewel.get("jewel_id") or 1)
                regions = []
                for region_data in jewel.get("regions") or []:
                    mask, exact = reconstruct_region_mask(color_mask, region_data)
                    region_id = int(region_data.get("region_id") or len(regions) + 1)
                    regions.append(
                        ReviewRegion(
                            data=region_data,
                            mask=mask,
                            exact_reconstruction=exact,
                            sample_id=_sample_id(
                                report_path,
                                runtime_root,
                                jewel_id,
                                region_id,
                            ),
                        )
                    )
                items.append(
                    (
                        report_path.stat().st_mtime,
                        ReviewImage(
                            report_path=report_path,
                            session_name=session_name,
                            stage_name=stage_name,
                            jewel_id=jewel_id,
                            image_bgr=image_bgr,
                            bbox_global=[int(v) for v in jewel.get("bbox_global", [0, 0, 0, 0])],
                            regions=regions,
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - continue past damaged old sessions.
            errors.append(f"{report_path}: {exc}")
    items.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in items], errors


class DatasetAnalyzer:
    """Run the current stone pipeline lazily and cache its review artifacts."""

    def __init__(self, image_root: Path, cache_root: Path):
        self.image_root = image_root
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.settings = self._load_settings()
        self.runtime: Any = None
        self.fastsam_model: Any = None
        self.inference_lock: Any = threading.RLock()
        self.backend = "not_initialized"
        self.backend_error = ""

    @staticmethod
    def _load_settings() -> dict[str, Any]:
        settings: dict[str, Any] = {}
        settings_path = REPO_DIR / "roi_config.json"
        if settings_path.is_file():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                settings = {}
        return {
            "color_correction": settings.get("color_correction"),
            "background_calibration": settings.get("background_calibration"),
            "analysis_normalization": settings.get("analysis_normalization"),
            "learned_stone_profiles": settings.get("learned_stone_profiles"),
        }

    def image_paths(self) -> list[Path]:
        return discover_dataset_image_paths(self.image_root)

    def _initialize_backend(self) -> None:
        if self.backend != "not_initialized":
            return
        try:
            segmentation_dir = REPO_DIR / "Segmentation"
            if str(segmentation_dir) not in sys.path:
                sys.path.insert(0, str(segmentation_dir))
            from hailo_model_runner import HailoRuntime
            from Segmentation.segment_necklace_fastsam import FastSamOnnx

            model_path = segmentation_dir / "fast_sam_s.hef"
            self.runtime = HailoRuntime()
            hailo_model = self.runtime.create_model(str(model_path), "FastSAM")
            if hailo_model is None:
                raise RuntimeError(self.runtime.last_model_error or "FastSAM HEF could not be loaded")
            self.fastsam_model = FastSamOnnx(
                model_path,
                providers=["CPUExecutionProvider"],
                input_size=640,
                hailo_model=hailo_model,
            )
            self.backend = "fastsam_hailo"
        except Exception as exc:  # noqa: BLE001 - production also has an OpenCV fallback.
            if self.runtime is not None:
                try:
                    self.runtime.close()
                except Exception:
                    pass
            self.runtime = None
            self.fastsam_model = None
            self.inference_lock = threading.RLock()
            self.backend = "opencv_fallback"
            self.backend_error = str(exc)

    def close(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.close()
            finally:
                self.runtime = None

    def _cache_dir(self, image_path: Path) -> Path:
        stat = image_path.stat()
        pipeline_files = [
            TOOL_DIR / "jewel_gem_hsv_report.py",
            TOOL_DIR / "stone_analysis_v2.py",
            REPO_DIR / "roi_config.json",
        ]
        signature = {
            "source": str(image_path.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "backend": self.backend,
            "pipeline_mtime_ns": {
                str(path): path.stat().st_mtime_ns if path.is_file() else None
                for path in pipeline_files
            },
            "settings": self.settings,
        }
        digest = hashlib.sha1(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_root / f"{image_path.stem}_{digest}"

    def analyze(self, image_path: Path) -> ReviewImage:
        self._initialize_backend()
        output_dir = self._cache_dir(image_path)
        report_path = output_dir / "gem_report.json"
        if not report_path.is_file():
            import jewel_gem_hsv_report as stone_detection

            image_bgr = _read_image(image_path)
            analysis = stone_detection.analyze_image_bgr(
                image_bgr,
                source_name=str(image_path),
                zoom_scale=1,
                use_glare_removal=True,
                glare_threshold=stone_detection.DEFAULT_GLARE_THRESHOLD,
                glare_patch_size=stone_detection.DEFAULT_GLARE_PATCH_SIZE,
                use_sahi_slicing=True,
                sahi_slice_size=stone_detection.DEFAULT_SAHI_SLICE_SIZE,
                sahi_overlap_ratio=stone_detection.DEFAULT_SAHI_OVERLAP,
                color_correction=self.settings["color_correction"],
                background_calibration=self.settings["background_calibration"],
                analysis_normalization=self.settings["analysis_normalization"],
                learned_stone_profiles=self.settings["learned_stone_profiles"],
                fastsam_model=self.fastsam_model,
                fastsam_lock=self.inference_lock,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            report = analysis["report"]
            for jewel_report, jewel_view in zip(
                report["jewels"], analysis["jewel_views"], strict=False
            ):
                jewel_report["outputs"] = stone_detection.write_jewel_output_images(
                    output_dir,
                    jewel_report["jewel_id"],
                    jewel_view["masked_bgra"],
                    jewel_view["overlay_bgr"],
                    jewel_view["color_mask_bgr"],
                )
            report["review_analysis_backend"] = self.backend
            report["review_analysis_backend_error"] = self.backend_error or None
            temporary = report_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(report_path)
        return self._load_cached_item(image_path, report_path)

    def _load_cached_item(self, image_path: Path, report_path: Path) -> ReviewImage:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        image_bgr = _read_image(image_path)
        image_height, image_width = image_bgr.shape[:2]
        regions: list[ReviewRegion] = []
        for jewel in report.get("jewels") or []:
            outputs = jewel.get("outputs") or {}
            color_mask_path = _existing_artifact(outputs.get("color_mask_png"), report_path)
            if color_mask_path is None:
                raise ValueError(f"Cached color mask is missing for {image_path.name}")
            color_mask = _read_image(color_mask_path)
            global_x, global_y, _, _ = (
                int(value) for value in jewel.get("bbox_global", [0, 0, 0, 0])
            )
            jewel_id = int(jewel.get("jewel_id") or 1)
            for region_data in jewel.get("regions") or []:
                local_mask, exact = reconstruct_region_mask(color_mask, region_data)
                global_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                paste_width = min(local_mask.shape[1], image_width - max(0, global_x))
                paste_height = min(local_mask.shape[0], image_height - max(0, global_y))
                if paste_width > 0 and paste_height > 0:
                    source_x = max(0, -global_x)
                    source_y = max(0, -global_y)
                    target_x = max(0, global_x)
                    target_y = max(0, global_y)
                    paste_width = min(paste_width, local_mask.shape[1] - source_x)
                    paste_height = min(paste_height, local_mask.shape[0] - source_y)
                    global_mask[
                        target_y : target_y + paste_height,
                        target_x : target_x + paste_width,
                    ] = local_mask[
                        source_y : source_y + paste_height,
                        source_x : source_x + paste_width,
                    ]
                data = copy.deepcopy(region_data)
                x, y, width, height = (int(value) for value in data["bbox"])
                data["bbox_local"] = [x, y, width, height]
                data["bbox"] = [global_x + x, global_y + y, width, height]
                center_x, center_y = (int(value) for value in data.get("center", [x, y]))
                data["center"] = [global_x + center_x, global_y + center_y]
                region_id = int(data.get("region_id") or len(regions) + 1)
                sample_raw = (
                    f"dataset|{image_path.resolve()}|{report_path.parent.name}|"
                    f"{jewel_id}|{region_id}"
                ).encode("utf-8")
                regions.append(
                    ReviewRegion(
                        data=data,
                        mask=global_mask,
                        exact_reconstruction=exact,
                        sample_id=hashlib.sha1(sample_raw).hexdigest()[:20],
                    )
                )
        return ReviewImage(
            report_path=report_path,
            session_name=image_path.stem,
            stage_name="Necklace dataset",
            jewel_id=1,
            image_bgr=image_bgr,
            bbox_global=[0, 0, image_width, image_height],
            regions=regions,
            source_image_path=image_path,
            analysis_backend=str(report.get("review_analysis_backend") or self.backend),
        )


class ReviewStore:
    def __init__(self, dataset_dir: Path, runtime_root: Path):
        self.dataset_dir = dataset_dir
        self.runtime_root = runtime_root
        self.samples_dir = dataset_dir / "samples"
        self.manifest_path = dataset_dir / "labels.json"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"version": 1, "labels": {}}
        self.manifest.setdefault("labels", {})

    def get(self, sample_id: str) -> dict[str, Any] | None:
        return self.manifest["labels"].get(sample_id)

    def counts(self) -> dict[str, int]:
        counts = {"true_stone": 0, "false_positive": 0}
        for entry in self.manifest["labels"].values():
            label = entry.get("label")
            if label in counts:
                counts[label] += 1
        return counts

    def label_region(
        self,
        item: ReviewImage,
        region: ReviewRegion,
        label: str,
        color: str | None,
    ) -> dict[str, Any]:
        if label not in {"true_stone", "false_positive"}:
            raise ValueError(f"Unsupported label: {label}")
        if label == "true_stone" and color not in STONE_COLORS:
            raise ValueError("A valid color is required for a true stone.")
        if cv2.countNonZero(region.mask) == 0:
            raise ValueError("This historical region mask could not be reconstructed.")

        sample_dir = self.samples_dir / region.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        crop, crop_mask, crop_bbox = make_context_crop(item.image_bgr, region.mask)
        crop_path = sample_dir / "crop.png"
        mask_path = sample_dir / "mask.png"
        if not crop_path.exists() and not cv2.imwrite(str(crop_path), crop):
            raise OSError(f"Could not save {crop_path}")
        if not mask_path.exists() and not cv2.imwrite(str(mask_path), crop_mask):
            raise OSError(f"Could not save {mask_path}")

        try:
            report_value = str(item.report_path.resolve().relative_to(self.runtime_root.resolve()))
        except ValueError:
            report_value = str(item.report_path.resolve())
        entry = {
            "sample_id": region.sample_id,
            "label": label,
            "color": color if label == "true_stone" else None,
            "session": item.session_name,
            "stage": item.stage_name,
            "jewel_id": item.jewel_id,
            "region_id": region.region_id,
            "report_path": report_value,
            "source_image_path": (
                str(item.source_image_path.resolve())
                if item.source_image_path is not None
                else None
            ),
            "analysis_backend": item.analysis_backend,
            "bbox": [int(v) for v in region.data.get("bbox", [])],
            "bbox_global": item.bbox_global,
            "crop_bbox": crop_bbox,
            "recorded_area_px": int(region.data.get("area_px") or 0),
            "reconstructed_area_px": int(cv2.countNonZero(region.mask)),
            "mask_reconstruction_exact": bool(region.exact_reconstruction),
            "predicted_color": region.data.get("color"),
            "source_methods": list(region.data.get("source_methods") or []),
            "segmentation_method": region.data.get("segmentation_method"),
            "crop_path": str(crop_path.relative_to(self.dataset_dir)),
            "mask_path": str(mask_path.relative_to(self.dataset_dir)),
            "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.manifest["labels"][region.sample_id] = entry
        self._save()
        return entry

    def clear_label(self, sample_id: str) -> None:
        self.manifest["labels"].pop(sample_id, None)
        self._save()

    def _save(self) -> None:
        self.manifest["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)


class MaskCanvas(QWidget):
    regionSelected = pyqtSignal(int)

    def __init__(self, show_masks: bool = True):
        super().__init__()
        self.show_masks = show_masks
        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.item: ReviewImage | None = None
        self.labels: dict[str, dict[str, Any]] = {}
        self.selected_index = -1
        self._image_rect = QRectF()
        self._view_origin = (0, 0)
        self._view_shape = (0, 0)

    def set_item(
        self,
        item: ReviewImage | None,
        labels: dict[str, dict[str, Any]],
        selected_index: int = -1,
    ) -> None:
        self.item = item
        self.labels = labels
        self.selected_index = selected_index
        self.update()

    def set_selected_index(self, index: int) -> None:
        self.selected_index = index
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override.
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(36, 39, 44))
        if self.item is None:
            painter.setPen(QColor(225, 225, 225))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No reviewable sessions found")
            return

        display = self.item.image_bgr.copy()
        for index, region in enumerate(self.item.regions if self.show_masks else []):
            entry = self.labels.get(region.sample_id)
            if entry and entry.get("label") == "true_stone":
                overlay_color = COLOR_DRAW_BGR.get(entry.get("color"), (60, 200, 70))
            elif entry and entry.get("label") == "false_positive":
                overlay_color = (60, 60, 220)
            else:
                overlay_color = (0, 210, 255)
            mask_pixels = region.mask > 0
            if np.any(mask_pixels):
                color_array = np.asarray(overlay_color, dtype=np.float32)
                display[mask_pixels] = np.clip(
                    display[mask_pixels].astype(np.float32) * 0.45 + color_array * 0.55,
                    0,
                    255,
                ).astype(np.uint8)
                contours, _ = cv2.findContours(region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                line_color = (255, 255, 0) if index == self.selected_index else overlay_color
                cv2.drawContours(display, contours, -1, line_color, 2 if index == self.selected_index else 1)

        view = display
        self._view_origin = (0, 0)
        if self.show_masks and 0 <= self.selected_index < len(self.item.regions):
            region = self.item.regions[self.selected_index]
            x, y, width, height = (int(v) for v in region.data["bbox"])
            side = max(96, max(width, height) * 6)
            center_x = x + width // 2
            center_y = y + height // 2
            x1 = max(0, center_x - side // 2)
            y1 = max(0, center_y - side // 2)
            x2 = min(display.shape[1], x1 + side)
            y2 = min(display.shape[0], y1 + side)
            x1 = max(0, x2 - side)
            y1 = max(0, y2 - side)
            cv2.rectangle(
                display,
                (max(0, x - 3), max(0, y - 3)),
                (min(display.shape[1] - 1, x + width + 2), min(display.shape[0] - 1, y + height + 2)),
                (255, 255, 0),
                2,
            )
            view = display[y1:y2, x1:x2]
            self._view_origin = (x1, y1)
        self._view_shape = view.shape[:2]

        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        available = self.rect().adjusted(10, 10, -10, -10)
        scaled = pixmap.scaled(
            available.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        left = available.left() + (available.width() - scaled.width()) / 2
        top = available.top() + (available.height() - scaled.height()) / 2
        self._image_rect = QRectF(left, top, scaled.width(), scaled.height())
        painter.drawPixmap(int(left), int(top), scaled)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override.
        if self.item is None or not self._image_rect.contains(QPointF(event.position())):
            return
        view_height, view_width = self._view_shape
        if view_width <= 0 or view_height <= 0:
            return
        x = self._view_origin[0] + int(
            (event.position().x() - self._image_rect.left())
            * view_width
            / self._image_rect.width()
        )
        y = self._view_origin[1] + int(
            (event.position().y() - self._image_rect.top())
            * view_height
            / self._image_rect.height()
        )
        matches = [
            (index, cv2.countNonZero(region.mask))
            for index, region in enumerate(self.item.regions)
            if 0 <= y < region.mask.shape[0]
            and 0 <= x < region.mask.shape[1]
            and region.mask[y, x] > 0
        ]
        if matches:
            self.regionSelected.emit(min(matches, key=lambda value: value[1])[0])
            return
        boxed = []
        for index, region in enumerate(self.item.regions):
            bx, by, bw, bh = (int(v) for v in region.data["bbox"])
            if bx <= x < bx + bw and by <= y < by + bh:
                cx, cy = region.data.get("center", [bx + bw // 2, by + bh // 2])
                boxed.append((index, (cx - x) ** 2 + (cy - y) ** 2))
        if boxed:
            self.regionSelected.emit(min(boxed, key=lambda value: value[1])[0])


class StoneReviewWindow(QMainWindow):
    def __init__(
        self,
        runtime_root: Path,
        dataset_dir: Path,
        image_root: Path | None = None,
    ):
        super().__init__()
        self.runtime_root = runtime_root
        self.store = ReviewStore(dataset_dir, runtime_root)
        self.dataset_analyzer = (
            DatasetAnalyzer(image_root, dataset_dir / "analysis_cache")
            if image_root is not None
            else None
        )
        self.source_paths: list[Path] = []
        self.items: list[ReviewImage | None] = []
        self.discovery_errors: list[str] = []
        self.item_index = -1
        self.region_index = -1
        self.setWindowTitle("Stone Mask Review")
        self.resize(1320, 820)

        self.raw_canvas = MaskCanvas(show_masks=False)
        self.canvas = MaskCanvas(show_masks=True)
        self.canvas.regionSelected.connect(self.select_region)
        self.region_list = QListWidget()
        self.region_list.currentRowChanged.connect(self.select_region)
        self.title_label = QLabel()
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.progress_label = QLabel()
        self.detail_label = QLabel("Select a stone-analysis region.")
        self.detail_label.setWordWrap(True)
        self.color_combo = QComboBox()
        self.color_combo.addItems(STONE_COLORS)
        self.true_button = QPushButton(
            "True Stone" if self.dataset_analyzer is not None else "True Stone + Color"
        )
        self.false_button = QPushButton("False Positive")
        self.clear_button = QPushButton("Clear Label")
        self.previous_button = QPushButton("Previous Image")
        self.next_button = QPushButton("Next Image")
        self.next_unreviewed_button = QPushButton("Next Unreviewed")
        self.reload_button = QPushButton(
            "Reload Dataset" if self.dataset_analyzer is not None else "Reload Sessions"
        )

        self.true_button.clicked.connect(self.mark_true)
        self.false_button.clicked.connect(self.mark_false)
        self.clear_button.clicked.connect(self.clear_label)
        self.previous_button.clicked.connect(lambda: self.show_item(self.item_index - 1))
        self.next_button.clicked.connect(lambda: self.show_item(self.item_index + 1))
        self.next_unreviewed_button.clicked.connect(self.next_unreviewed)
        self.reload_button.clicked.connect(self.reload_sessions)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Detected regions"))
        controls.addWidget(self.region_list, 1)
        controls.addWidget(self.detail_label)
        self.color_label = QLabel("True-stone color")
        controls.addWidget(self.color_label)
        controls.addWidget(self.color_combo)
        self.color_label.setVisible(self.dataset_analyzer is None)
        self.color_combo.setVisible(self.dataset_analyzer is None)
        controls.addWidget(self.true_button)
        controls.addWidget(self.false_button)
        controls.addWidget(self.clear_button)
        controls.addSpacing(12)
        controls.addWidget(self.next_unreviewed_button)
        controls.addStretch()

        right = QWidget()
        right.setLayout(controls)
        raw_panel = QWidget()
        raw_layout = QVBoxLayout(raw_panel)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.addWidget(QLabel("Raw ROI image"))
        raw_layout.addWidget(self.raw_canvas, 1)
        mask_panel = QWidget()
        mask_layout = QVBoxLayout(mask_panel)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_layout.addWidget(QLabel("Selected stone mask (zoomed, outlined in cyan)"))
        mask_layout.addWidget(self.canvas, 1)

        splitter = QSplitter()
        splitter.addWidget(raw_panel)
        splitter.addWidget(mask_panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500, 300])

        navigation = QHBoxLayout()
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.reload_button)
        navigation.addStretch()
        navigation.addWidget(self.progress_label)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addLayout(navigation)
        layout.addWidget(splitter, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        self._add_shortcuts()
        self.reload_sessions()

    def _add_shortcuts(self) -> None:
        shortcuts = [
            ("F", self.mark_false),
            ("T", self.mark_true),
            ("N", self.next_unreviewed),
            (QKeySequence.StandardKey.MoveToNextChar, lambda: self.show_item(self.item_index + 1)),
            (QKeySequence.StandardKey.MoveToPreviousChar, lambda: self.show_item(self.item_index - 1)),
        ]
        for key, callback in shortcuts:
            action = QAction(self)
            action.setShortcut(key)
            action.triggered.connect(callback)
            self.addAction(action)

    def reload_sessions(self) -> None:
        previous_key = None
        if 0 <= self.item_index < len(self.items):
            item = self.items[self.item_index]
            if item is not None:
                previous_key = (str(item.report_path), item.jewel_id)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self.dataset_analyzer is not None:
                self.source_paths = self.dataset_analyzer.image_paths()
                self.items = [None] * len(self.source_paths)
                self.discovery_errors = []
            else:
                loaded, self.discovery_errors = discover_review_images(self.runtime_root)
                self.items = list(loaded)
        finally:
            QApplication.restoreOverrideCursor()
        new_index = 0
        if previous_key:
            for index, item in enumerate(self.items):
                if item is not None and (str(item.report_path), item.jewel_id) == previous_key:
                    new_index = index
                    break
        self.show_item(new_index if self.items else -1)
        source_name = "dataset images" if self.dataset_analyzer is not None else "review images"
        message = f"Loaded {len(self.items)} {source_name}"
        if self.discovery_errors:
            message += f"; skipped {len(self.discovery_errors)} damaged/incomplete reports"
        self.statusBar().showMessage(message, 8000)

    def show_item(self, index: int) -> None:
        if not self.items:
            self.item_index = self.region_index = -1
            self.title_label.setText(f"No stone-analysis reports under {self.runtime_root}")
            self.region_list.clear()
            self.raw_canvas.set_item(None, {})
            self.canvas.set_item(None, {})
            self._update_progress()
            return
        self.item_index = min(max(index, 0), len(self.items) - 1)
        self.region_index = -1
        item = self.items[self.item_index]
        if item is None:
            image_path = self.source_paths[self.item_index]
            self.statusBar().showMessage(f"Running Stone Analysis: {image_path.name}")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                if self.dataset_analyzer is None:
                    raise RuntimeError("Dataset analyzer is not configured")
                item = self.dataset_analyzer.analyze(image_path)
                self.items[self.item_index] = item
            except Exception as exc:  # noqa: BLE001 - surface model/image failures in the UI.
                QMessageBox.critical(self, "Stone Analysis failed", f"{image_path}\n\n{exc}")
                self.title_label.setText(f"Stone Analysis failed: {image_path.name}")
                self.region_list.clear()
                self.raw_canvas.set_item(None, {})
                self.canvas.set_item(None, {})
                return
            finally:
                QApplication.restoreOverrideCursor()
        self.title_label.setText(f"{item.title} | backend: {item.analysis_backend}")
        self.region_list.blockSignals(True)
        self.region_list.clear()
        for region in item.regions:
            self.region_list.addItem(self._region_text(region))
        self.region_list.blockSignals(False)
        labels = {
            region.sample_id: self.store.get(region.sample_id)
            for region in item.regions
            if self.store.get(region.sample_id) is not None
        }
        self.raw_canvas.set_item(item, {})
        self.canvas.set_item(item, labels)
        self.previous_button.setEnabled(self.item_index > 0)
        self.next_button.setEnabled(self.item_index < len(self.items) - 1)
        if item.regions:
            first_unreviewed = next(
                (i for i, region in enumerate(item.regions) if self.store.get(region.sample_id) is None),
                0,
            )
            self.select_region(first_unreviewed)
        else:
            self.detail_label.setText("This image has no accepted Stone Analysis regions.")
        self._update_progress()

    def _region_text(self, region: ReviewRegion) -> str:
        entry = self.store.get(region.sample_id)
        if entry is None:
            status = "UNREVIEWED"
        elif entry.get("label") == "true_stone":
            status = f"TRUE: {entry.get('color')}"
        else:
            status = "FALSE POSITIVE"
        approximate = " ~mask" if not region.exact_reconstruction else ""
        return (
            f"#{region.region_id:02d}  {status}{approximate}  "
            f"pred={region.data.get('color')}  area={region.data.get('area_px')}"
        )

    def select_region(self, index: int) -> None:
        if not (0 <= self.item_index < len(self.items)):
            return
        item = self.items[self.item_index]
        if item is None:
            return
        if not (0 <= index < len(item.regions)):
            return
        self.region_index = index
        self.region_list.blockSignals(True)
        self.region_list.setCurrentRow(index)
        self.region_list.blockSignals(False)
        self.canvas.set_selected_index(index)
        region = item.regions[index]
        predicted = str(region.data.get("color") or "")
        if predicted in STONE_COLORS:
            self.color_combo.setCurrentText(predicted)
        exact_text = "exact" if region.exact_reconstruction else "APPROXIMATE - review carefully"
        self.detail_label.setText(
            f"Region {region.region_id} | predicted {predicted} | "
            f"recorded {region.data.get('area_px')} px | reconstructed "
            f"{cv2.countNonZero(region.mask)} px ({exact_text})\n"
            f"Sources: {', '.join(region.data.get('source_methods') or [])}"
        )

    def _selected(self) -> tuple[ReviewImage, ReviewRegion] | None:
        if not (0 <= self.item_index < len(self.items)):
            return None
        item = self.items[self.item_index]
        if item is None:
            return None
        if not (0 <= self.region_index < len(item.regions)):
            return None
        return item, item.regions[self.region_index]

    def mark_true(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        item, region = selected
        color = (
            str(region.data.get("color") or "")
            if self.dataset_analyzer is not None
            else self.color_combo.currentText()
        )
        self._save_label(item, region, "true_stone", color)

    def mark_false(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        self._save_label(*selected, "false_positive", None)

    def _save_label(
        self,
        item: ReviewImage,
        region: ReviewRegion,
        label: str,
        color: str | None,
    ) -> None:
        try:
            self.store.label_region(item, region, label, color)
        except Exception as exc:  # noqa: BLE001 - show file/model data failures in UI.
            QMessageBox.critical(self, "Could not save label", str(exc))
            return
        self._refresh_current_region()
        self._advance_within_item()

    def clear_label(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        _, region = selected
        self.store.clear_label(region.sample_id)
        self._refresh_current_region()

    def _refresh_current_region(self) -> None:
        if not (0 <= self.item_index < len(self.items)):
            return
        item = self.items[self.item_index]
        if item is None:
            return
        self.region_list.blockSignals(True)
        for index, region in enumerate(item.regions):
            self.region_list.item(index).setText(self._region_text(region))
        self.region_list.blockSignals(False)
        self.canvas.labels = {
            region.sample_id: self.store.get(region.sample_id)
            for region in item.regions
            if self.store.get(region.sample_id) is not None
        }
        self.canvas.update()
        self._update_progress()

    def _advance_within_item(self) -> None:
        item = self.items[self.item_index]
        if item is None:
            return
        for offset in range(1, len(item.regions) + 1):
            index = (self.region_index + offset) % len(item.regions)
            if self.store.get(item.regions[index].sample_id) is None:
                self.select_region(index)
                return
        self.next_unreviewed()

    def next_unreviewed(self) -> None:
        if not self.items:
            return
        start_item = max(0, self.item_index)
        for item_offset in range(len(self.items)):
            item_index = (start_item + item_offset) % len(self.items)
            item = self.items[item_index]
            if item is None:
                self.show_item(item_index)
                item = self.items[item_index]
            if item is None:
                continue
            for region_index, region in enumerate(item.regions):
                if self.store.get(region.sample_id) is None:
                    if item_index != self.item_index:
                        self.show_item(item_index)
                    self.select_region(region_index)
                    return
        self.statusBar().showMessage("Every discovered region has been reviewed.", 5000)

    def _update_progress(self) -> None:
        loaded_items = [item for item in self.items if item is not None]
        total = sum(len(item.regions) for item in loaded_items)
        reviewed = sum(
            self.store.get(region.sample_id) is not None
            for item in loaded_items
            for region in item.regions
        )
        counts = self.store.counts()
        image_text = f"Image {self.item_index + 1}/{len(self.items)}" if self.items else "Image 0/0"
        self.progress_label.setText(
            f"{image_text} | Reviewed {reviewed}/{total} | "
            f"True {counts['true_stone']} | False {counts['false_positive']}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override.
        if self.dataset_analyzer is not None:
            self.dataset_analyzer.close()
        super().closeEvent(event)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Folder of images to analyze; paired folders process only *_roi images.",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Review previously saved runtime sessions instead of the image dataset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv[:1])
    image_root = None if args.runtime_only else args.image_root.resolve()
    window = StoneReviewWindow(
        args.runtime_root.resolve(),
        args.dataset_dir.resolve(),
        image_root=image_root,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
