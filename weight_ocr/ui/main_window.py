import logging
from datetime import datetime
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import MODEL, CameraConfig, ModelConfig
from ui.video_canvas import VideoCanvas
from validation.sample_saver import save_validation_sample
from workers.camera_worker import CameraWorker
from workers.inference_worker import InferenceWorker


LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    inference_requested = pyqtSignal(object)
    inference_shutdown_requested = pyqtSignal()

    def __init__(
        self,
        camera_config: CameraConfig,
        model_config: ModelConfig = MODEL,
    ):
        super().__init__()
        self.camera_config = camera_config
        self.model_config = model_config
        self.latest_full_resolution_frame = None
        self.captured_frame = None
        self.captured_roi = None
        self.captured_roi_bounds = None
        self._captured_at = None
        self._result_pixmap = None
        self._camera_info = None
        self._capture_fps = 0.0
        self._model_ready = False
        self._model_status = "Initializing"
        self._model_backend = "Detector"
        self._last_inference_result = None
        self._last_inference_error = None
        self._capture_path = (
            Path(__file__).resolve().parents[1] / "captures" / "latest_roi.png"
        )
        self._full_capture_path = (
            Path(__file__).resolve().parents[1] / "captures" / "latest_full.png"
        )
        self._validation_root = (
            Path(__file__).resolve().parents[1]
            / "validation"
            / "phase8"
            / "samples"
        )

        self.setWindowTitle("LCD Weight Reader")
        self.resize(1280, 760)
        self._build_ui()
        self._start_camera()
        self._start_inference()

    def _build_ui(self):
        title = QLabel("LCD Weight Reader")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 600; padding: 8px;")

        self.video_canvas = VideoCanvas()

        live_title = QLabel("LIVE CAMERA")
        live_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        live_layout = QVBoxLayout()
        live_layout.addWidget(live_title)
        live_layout.addWidget(self.video_canvas, 1)

        self.result_title = QLabel("Captured ROI")
        self.result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.result_label = QLabel("No ROI captured")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(360, 260)
        self.result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.result_label.setStyleSheet(
            "background: #111; color: #ddd; font-size: 18px;"
        )
        self.roi_status_label = QLabel("ROI: not set")
        self.roi_status_label.setWordWrap(True)
        self.roi_status_label.setStyleSheet("padding: 6px;")
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_label, 1)
        result_layout.addWidget(self.roi_status_label)

        panels = QHBoxLayout()
        panels.addLayout(live_layout, 2)
        panels.addLayout(result_layout, 1)

        self.draw_roi_button = QPushButton("Draw ROI")
        self.clear_roi_button = QPushButton("Clear ROI")
        self.capture_button = QPushButton("Capture && Read")
        self.capture_button.setEnabled(False)
        self.draw_roi_button.clicked.connect(self._begin_roi_drawing)
        self.clear_roi_button.clicked.connect(self._clear_roi)
        self.capture_button.clicked.connect(self._capture_and_read)
        self.video_canvas.roi_changed.connect(self._on_roi_changed)

        controls = QHBoxLayout()
        controls.addWidget(self.draw_roi_button)
        controls.addWidget(self.clear_roi_button)
        controls.addWidget(self.capture_button)
        controls.addStretch(1)

        self.true_reading_input = QLineEdit()
        self.true_reading_input.setPlaceholderText("True LCD, e.g. 1.17")
        self.true_reading_input.setMaximumWidth(180)
        self.condition_combo = QComboBox()
        self.condition_combo.addItems(
            (
                "Normal",
                "Bright LCD",
                "Dark LCD",
                "Glare",
                "Rotated left",
                "Rotated right",
                "Jewellery near display",
                "Other",
            )
        )
        self.save_validation_button = QPushButton("Save Validation Sample")
        self.save_validation_button.setEnabled(False)
        self.save_validation_button.clicked.connect(self._save_validation_sample)
        validation_controls = QHBoxLayout()
        validation_controls.addWidget(QLabel("True reading:"))
        validation_controls.addWidget(self.true_reading_input)
        validation_controls.addWidget(QLabel("Condition:"))
        validation_controls.addWidget(self.condition_combo)
        validation_controls.addWidget(self.save_validation_button)
        validation_controls.addStretch(1)
        self.validation_status_label = QLabel(
            "Validation: capture and run inference before saving"
        )
        self.validation_status_label.setWordWrap(True)

        self.status_label = QLabel(
            f"Camera: {self.camera_config.device} | Opening…"
        )
        self.status_label.setStyleSheet("padding: 8px; font-size: 14px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(title)
        layout.addLayout(panels, 1)
        layout.addLayout(controls)
        layout.addLayout(validation_controls)
        layout.addWidget(self.validation_status_label)
        layout.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _start_camera(self):
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(self.camera_config)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.camera_opened.connect(self._on_camera_opened)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.stats_updated.connect(self._on_stats)
        self.camera_worker.error.connect(self._on_camera_error)
        self.camera_worker.stopped.connect(self._on_camera_stopped)
        self.camera_thread.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.start()

    def _start_inference(self):
        self.inference_thread = QThread(self)
        self.inference_worker = InferenceWorker(self.model_config)
        self.inference_worker.moveToThread(self.inference_thread)
        self.inference_thread.started.connect(self.inference_worker.initialize)
        self.inference_requested.connect(self.inference_worker.process_roi)
        self.inference_shutdown_requested.connect(self.inference_worker.shutdown)
        self.inference_worker.model_ready.connect(self._on_model_ready)
        self.inference_worker.model_failed.connect(self._on_model_failed)
        self.inference_worker.result_ready.connect(self._on_raw_inference_result)
        self.inference_worker.inference_failed.connect(self._on_inference_failed)
        self.inference_worker.processing_changed.connect(self._on_processing_changed)
        self.inference_thread.finished.connect(self.inference_worker.deleteLater)
        self.inference_thread.start()

    def _on_camera_opened(self, info):
        self._camera_info = info
        if (
            info.width != self.camera_config.width
            or info.height != self.camera_config.height
        ):
            self.status_label.setStyleSheet(
                "padding: 8px; font-size: 14px; color: #b45309;"
            )
        self._update_system_status()

    def _on_frame(self, frame):
        self.latest_full_resolution_frame = frame
        self.video_canvas.set_bgr_frame(frame)

    def _on_stats(self, capture_fps: float, _captured_frames: int):
        self._capture_fps = capture_fps
        self._update_system_status()

    def _on_camera_error(self, message: str):
        self.video_canvas.set_message("CAMERA ERROR")
        self.status_label.setText(f"Camera error: {message}")
        self.status_label.setStyleSheet(
            "padding: 8px; font-size: 14px; color: #b91c1c;"
        )

    def _on_camera_stopped(self):
        LOGGER.info("Camera worker stopped")

    def _update_system_status(self):
        if self._camera_info is None:
            return
        info = self._camera_info
        self.status_label.setText(
            f"Camera: {info.device} | Actual: {info.width}x{info.height} | "
            f"FPS: {self._capture_fps:.1f} | {info.fourcc} | "
            f"{self._model_backend}: {self._model_status}"
        )

    def _on_model_ready(self, contract):
        self._model_ready = True
        self._model_backend = contract["backend"]
        self._model_status = "Model Ready"
        self.capture_button.setEnabled(self.video_canvas.roi is not None)
        self._update_system_status()
        LOGGER.info("LCD model contract: %s", contract)

    def _on_model_failed(self, message: str):
        self._model_ready = False
        self._model_status = "MODEL_NOT_READY"
        self.capture_button.setEnabled(False)
        self.roi_status_label.setText(f"MODEL_NOT_READY: {message}")
        self._update_system_status()

    def _on_raw_inference_result(self, result):
        self._last_inference_result = result
        self._last_inference_error = None
        self.save_validation_button.setEnabled(self.captured_roi is not None)
        self.validation_status_label.setText(
            "Validation: enter the visible LCD reading, then save this sample"
        )
        if result["detections"]:
            best = result["detections"][0]
            if result["lcd_rectified"]:
                rectified = cv2.imread(result["lcd_rectified"])
                if rectified is not None:
                    self.result_title.setText("Rectified LCD")
                    self._set_result_image(rectified)
            digit_result = result["digit_result"]
            if digit_result and digit_result["success"]:
                weight = int(digit_result["digits"]) / 100.0
                self.result_title.setText(f"WEIGHT: {weight:.2f} g")
                self.roi_status_label.setText(
                    f"WEIGHT: {weight:.2f} g | digits {digit_result['digits']} | "
                    f"confidence {digit_result['confidence']:.3f} | "
                    f"LCD {best['confidence']:.3f} | "
                    f"detector {result['inference_ms']:.1f} ms | "
                    f"PaddleOCR {result['lcd_decode_ms']:.1f} ms"
                )
            else:
                self.result_title.setText("READ FAILED")
                variants = (digit_result or {}).get("variants", [])
                readings = ", ".join(
                    f"{item.get('digits') or 'FAIL'} ({item.get('confidence', 0):.2f})"
                    for item in variants
                )
                self.roi_status_label.setText(
                    "READ FAILED: PaddleOCR crops did not agree or confidence was "
                    f"below {self.model_config.ocr_min_confidence:.2f} | "
                    f"[{readings}]"
                )
        else:
            self.roi_status_label.setText(
                f"LCD_NOT_FOUND: no OBB over "
                f"{self.model_config.confidence_threshold:.2f} confidence"
            )
        LOGGER.info("LCD inference result: %s", result)

    def _on_inference_failed(self, message: str):
        self._last_inference_result = None
        self._last_inference_error = message
        self.save_validation_button.setEnabled(self.captured_roi is not None)
        self.validation_status_label.setText(
            "Validation: inference failed, but this sample can still be saved"
        )
        self.roi_status_label.setText(f"LCD inference failed: {message}")

    def _on_processing_changed(self, processing: bool):
        self.capture_button.setText(
            "Processing…" if processing else "Capture && Read"
        )
        self.capture_button.setEnabled(
            not processing
            and self._model_ready
            and self.video_canvas.roi is not None
        )
        self.save_validation_button.setEnabled(
            not processing
            and self.captured_roi is not None
            and (
                self._last_inference_result is not None
                or self._last_inference_error is not None
            )
        )

    def _begin_roi_drawing(self):
        if self.latest_full_resolution_frame is None:
            self.roi_status_label.setText("ROI: camera frame is not ready")
            return
        self.video_canvas.set_drawing_enabled(True)
        self.roi_status_label.setText("ROI: drag over the white board")

    def _on_roi_changed(self, roi):
        self.video_canvas.set_drawing_enabled(False)
        self.capture_button.setEnabled(self._model_ready)
        x1, y1, x2, y2 = roi
        self.roi_status_label.setText(
            f"ROI: ({x1}, {y1}) to ({x2}, {y2}) — {x2 - x1}x{y2 - y1}"
        )

    def _clear_roi(self):
        self.video_canvas.set_drawing_enabled(False)
        self.video_canvas.clear_roi()
        self.capture_button.setEnabled(False)
        self.captured_frame = None
        self.captured_roi = None
        self.captured_roi_bounds = None
        self._captured_at = None
        self._last_inference_result = None
        self._last_inference_error = None
        self.save_validation_button.setEnabled(False)
        self._result_pixmap = None
        self.result_label.clear()
        self.result_label.setText("No ROI captured")
        self.result_title.setText("Captured ROI")
        self.roi_status_label.setText("ROI: not set")
        self.validation_status_label.setText(
            "Validation: capture and run inference before saving"
        )

    def _capture_and_read(self):
        if not self._model_ready:
            self.roi_status_label.setText("MODEL_NOT_READY: detector is not initialized")
            return
        if not self._capture_roi():
            return
        self._last_inference_result = None
        self._last_inference_error = None
        self.capture_button.setEnabled(False)
        self.save_validation_button.setEnabled(False)
        self.validation_status_label.setText(
            "Validation: waiting for PaddleOCR result"
        )
        self.roi_status_label.setText(
            "Detecting LCD and running PaddleOCR…"
        )
        self.inference_requested.emit(
            {
                "roi": self.captured_roi,
                "full_frame": self.captured_frame,
                "roi_bounds": self.captured_roi_bounds,
            }
        )

    def _capture_roi(self):
        roi = self.video_canvas.roi
        if roi is None:
            self.roi_status_label.setText("ROI: draw an ROI before capture")
            return False

        captured_frame = self.camera_worker.latest_full_resolution_frame()
        if captured_frame is None:
            self.roi_status_label.setText("ROI: no camera frame available")
            return False

        x1, y1, x2, y2 = roi
        frame_height, frame_width = captured_frame.shape[:2]
        x1 = max(0, min(frame_width, x1))
        x2 = max(0, min(frame_width, x2))
        y1 = max(0, min(frame_height, y1))
        y2 = max(0, min(frame_height, y2))
        if x2 <= x1 or y2 <= y1:
            self.roi_status_label.setText("ROI: selected area is invalid")
            return False

        self.captured_frame = captured_frame
        self.captured_roi = captured_frame[y1:y2, x1:x2].copy()
        self.captured_roi_bounds = (x1, y1, x2, y2)
        self._captured_at = datetime.now().astimezone()
        self._capture_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self._capture_path), self.captured_roi):
            self.roi_status_label.setText("ROI: failed to save captured image")
            return False
        if not cv2.imwrite(str(self._full_capture_path), self.captured_frame):
            self.roi_status_label.setText("ROI: failed to save full captured image")
            return False

        self._set_result_image(self.captured_roi)
        self.result_title.setText("Captured ROI")
        self.roi_status_label.setText(
            f"Captured {x2 - x1}x{y2 - y1} ROI — {self._capture_path.name}"
        )
        return True

    def _save_validation_sample(self):
        if (
            self.captured_roi is None
            or self.captured_frame is None
            or self.captured_roi_bounds is None
        ):
            self.validation_status_label.setText(
                "Validation: capture an ROI before saving"
            )
            return
        try:
            sample_dir, metadata = save_validation_sample(
                self._validation_root,
                self.true_reading_input.text(),
                self.condition_combo.currentText(),
                self.captured_roi,
                self.captured_frame,
                self.captured_roi_bounds,
                inference_result=self._last_inference_result,
                inference_error=self._last_inference_error,
                captured_at=self._captured_at,
            )
        except Exception as exc:
            self.validation_status_label.setText(f"Validation save failed: {exc}")
            return

        self.save_validation_button.setEnabled(False)
        self.validation_status_label.setText(
            f"Saved {metadata['true_reading']} ({metadata['inference_status']}) "
            f"to {sample_dir.name}"
        )

    def _set_result_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        )
        self._result_pixmap = QPixmap.fromImage(image)
        self._scale_result_preview()

    def _scale_result_preview(self):
        if self._result_pixmap is None:
            return
        self.result_label.setPixmap(
            self._result_pixmap.scaled(
                self.result_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_result_preview()

    def closeEvent(self, event: QCloseEvent):
        self.camera_worker.request_stop()
        self.inference_shutdown_requested.emit()
        camera_stopped = self.camera_thread.wait(3000)
        inference_stopped = self.inference_thread.wait(5000)
        if not camera_stopped or not inference_stopped:
            LOGGER.error(
                "Shutdown timed out: camera_stopped=%s inference_stopped=%s",
                camera_stopped,
                inference_stopped,
            )
            event.ignore()
            return
        event.accept()
