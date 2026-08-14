import logging
from pathlib import Path
import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from config import PROJECT_DIR, ModelConfig
from inference.obb_decoder import (
    decode_onnx_output,
    draw_detections,
    rotated_nms,
)
from inference.onnx_obb import OnnxObbModel
from ocr.paddle_client import PaddleOCRClient
from vision.letterbox import letterbox
from vision.obb_mapping import draw_polygon_copy, map_model_corners
from vision.perspective import rectify_lcd


LOGGER = logging.getLogger(__name__)


def prepare_model_input(bgr_image, width: int, height: int):
    letterboxed_bgr, transform = letterbox(bgr_image, width, height)
    model_input = np.ascontiguousarray(
        cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    )
    return model_input, transform


def prepare_phase3_input(bgr_image, width: int, height: int):
    """Compatibility helper for the Phase 3 raw-inference test."""
    model_input, _transform = prepare_model_input(bgr_image, width, height)
    return model_input


class InferenceWorker(QObject):
    model_ready = pyqtSignal(object)
    model_failed = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    inference_failed = pyqtSignal(str)
    processing_changed = pyqtSignal(bool)

    def __init__(self, config: ModelConfig):
        super().__init__()
        self._config = config
        self.model = None
        self.ocr = None
        self._busy = False

    @pyqtSlot()
    def initialize(self):
        try:
            self.model = OnnxObbModel(self._config.onnx_path)
            self.ocr = PaddleOCRClient()
            contract = self.model.contract()
            contract["ocr_backend"] = "PaddleOCR en_PP-OCRv5_mobile_rec"
            self.model_ready.emit(contract)
        except Exception as exc:
            LOGGER.exception("Failed to initialize LCD detector")
            self.model_failed.emit(str(exc))

    @pyqtSlot(object)
    def process_roi(self, request):
        if self._busy:
            self.inference_failed.emit("LCD inference is already running")
            return
        if self.model is None:
            self.inference_failed.emit("LCD model is not ready")
            return

        self._busy = True
        self.processing_changed.emit(True)
        try:
            if isinstance(request, dict):
                bgr_roi = request["roi"]
                full_frame = request["full_frame"]
                roi_bounds = tuple(request["roi_bounds"])
                debug_tag = request.get("debug_tag")
            else:
                bgr_roi = request
                full_frame = None
                roi_bounds = None
                debug_tag = None
            debug_dir = Path(PROJECT_DIR) / "debug"
            if debug_tag:
                debug_dir = debug_dir / str(debug_tag)
            debug_dir.mkdir(parents=True, exist_ok=True)
            model_input, transform = prepare_model_input(
                bgr_roi,
                self._config.input_width,
                self._config.input_height,
            )
            outputs = self.model.infer(model_input)
            decode_started = time.perf_counter()
            candidates = decode_onnx_output(
                outputs[self.model.output_names[0]],
                self._config.confidence_threshold,
            )
            detections = rotated_nms(
                candidates,
                self._config.confidence_threshold,
                self._config.rotated_nms_iou,
            )
            detections = detections[:1]
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            debug_image = draw_detections(model_input, detections)
            debug_path = debug_dir / "obb_decoded.png"
            if not cv2.imwrite(str(debug_path), debug_image):
                raise RuntimeError(f"Failed to save OBB debug image: {debug_path}")

            roi_corners = None
            full_corners = None
            phase6_roi_debug = None
            phase6_full_debug = None
            lcd_raw_path = None
            lcd_rectified_path = None
            lcd_raw_shape = None
            lcd_rectified_shape = None
            digit_result = None
            lcd_decode_ms = None
            paddle_primary_path = None
            paddle_secondary_path = None
            if detections and full_frame is not None and roi_bounds is not None:
                roi_corners, full_corners = map_model_corners(
                    detections[0].corners(),
                    transform,
                    roi_bounds,
                    full_frame.shape,
                )
                label = f"LCD {detections[0].confidence:.3f}"
                roi_overlay = draw_polygon_copy(bgr_roi, roi_corners, label)
                full_overlay = draw_polygon_copy(full_frame, full_corners, label)
                phase6_roi_debug = debug_dir / "phase6_roi_obb.png"
                phase6_full_debug = debug_dir / "phase6_full_obb.png"
                if not cv2.imwrite(str(phase6_roi_debug), roi_overlay):
                    raise RuntimeError(
                        f"Failed to save ROI-space OBB image: {phase6_roi_debug}"
                    )
                if not cv2.imwrite(str(phase6_full_debug), full_overlay):
                    raise RuntimeError(
                        f"Failed to save full-frame OBB image: {phase6_full_debug}"
                    )
                lcd_raw, lcd_rectified = rectify_lcd(
                    bgr_roi,
                    roi_corners,
                    self._config.lcd_inner_margin,
                    self._config.lcd_quad_expand_x,
                    self._config.lcd_quad_expand_y,
                )
                lcd_raw_path = debug_dir / "lcd_raw.png"
                lcd_rectified_path = debug_dir / "lcd_rectified.png"
                if not cv2.imwrite(str(lcd_raw_path), lcd_raw):
                    raise RuntimeError(f"Failed to save raw LCD warp: {lcd_raw_path}")
                if not cv2.imwrite(str(lcd_rectified_path), lcd_rectified):
                    raise RuntimeError(
                        f"Failed to save rectified LCD: {lcd_rectified_path}"
                    )
                lcd_raw_shape = tuple(int(value) for value in lcd_raw.shape)
                lcd_rectified_shape = tuple(
                    int(value) for value in lcd_rectified.shape
                )
                digit_started = time.perf_counter()
                height, width = lcd_rectified.shape[:2]
                x1 = int(round(width * self._config.ocr_crop_x[0]))
                x2 = int(round(width * self._config.ocr_crop_x[1]))
                bottom = int(round(height * self._config.ocr_crop_bottom))
                primary = lcd_rectified[
                    int(round(height * self._config.ocr_primary_top)) : bottom,
                    x1:x2,
                ]
                secondary = lcd_rectified[
                    int(round(height * self._config.ocr_secondary_top)) : bottom,
                    x1:x2,
                ]
                paddle_primary_path = debug_dir / "paddle_primary.png"
                paddle_secondary_path = debug_dir / "paddle_secondary.png"
                if not cv2.imwrite(str(paddle_primary_path), primary):
                    raise RuntimeError("Failed to save primary PaddleOCR crop")
                if not cv2.imwrite(str(paddle_secondary_path), secondary):
                    raise RuntimeError("Failed to save secondary PaddleOCR crop")
                paddle_result = self.ocr.recognize(
                    (paddle_primary_path, paddle_secondary_path)
                )
                success = (
                    paddle_result["agreed"]
                    and paddle_result["confidence"]
                    >= self._config.ocr_min_confidence
                )
                digit_result = {
                    "success": success,
                    "digits": paddle_result["digits"] if success else None,
                    "confidence": float(paddle_result["confidence"]),
                    "failed_slots": [],
                    "variants": paddle_result["variants"],
                    "backend": "PaddleOCR en_PP-OCRv5_mobile_rec",
                }
                lcd_decode_ms = (time.perf_counter() - digit_started) * 1000.0

            tensor_stats = []
            for name in self.model.output_names:
                tensor = np.asarray(outputs[name])
                if not np.isfinite(tensor).all():
                    raise RuntimeError(f"Output {name} contains non-finite values")
                stats = {
                    "name": name,
                    "shape": tuple(int(value) for value in tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                }
                tensor_stats.append(stats)
                LOGGER.info(
                    "%s shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                    stats["name"],
                    stats["shape"],
                    stats["dtype"],
                    stats["min"],
                    stats["max"],
                    stats["mean"],
                )
            self.result_ready.emit(
                {
                    "tensors": tensor_stats,
                    "backend": self.model.contract()["backend"],
                    "inference_ms": self.model.last_inference_ms,
                    "inference_count": self.model.inference_count,
                    "configuration_count": self.model.configuration_count,
                    "letterbox_transform": transform.as_dict(),
                    "candidate_count": len(candidates),
                    "detections": [detection.as_dict() for detection in detections],
                    "obb_decode_ms": decode_ms,
                    "debug_image": str(debug_path),
                    "roi_bounds": roi_bounds,
                    "roi_corners": None if roi_corners is None else roi_corners.tolist(),
                    "full_corners": (
                        None if full_corners is None else full_corners.tolist()
                    ),
                    "phase6_roi_debug": (
                        None if phase6_roi_debug is None else str(phase6_roi_debug)
                    ),
                    "phase6_full_debug": (
                        None if phase6_full_debug is None else str(phase6_full_debug)
                    ),
                    "lcd_raw": None if lcd_raw_path is None else str(lcd_raw_path),
                    "lcd_rectified": (
                        None if lcd_rectified_path is None else str(lcd_rectified_path)
                    ),
                    "lcd_raw_shape": lcd_raw_shape,
                    "lcd_rectified_shape": lcd_rectified_shape,
                    "digit_result": digit_result,
                    "lcd_decode_ms": lcd_decode_ms,
                    "ocr_backend": "PaddleOCR en_PP-OCRv5_mobile_rec",
                    "paddle_primary": (
                        None if paddle_primary_path is None else str(paddle_primary_path)
                    ),
                    "paddle_secondary": (
                        None
                        if paddle_secondary_path is None
                        else str(paddle_secondary_path)
                    ),
                }
            )
        except Exception as exc:
            LOGGER.exception("LCD inference failed")
            self.inference_failed.emit(str(exc))
        finally:
            self._busy = False
            self.processing_changed.emit(False)

    @pyqtSlot()
    def shutdown(self):
        if self.ocr is not None:
            self.ocr.close()
            self.ocr = None
        self.model = None
        QThread.currentThread().quit()
