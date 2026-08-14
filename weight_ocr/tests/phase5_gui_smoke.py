"""Exercise one ONNX OBB inference through the PyQt worker path."""

from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import CAMERA, MODEL
from ui.main_window import MainWindow


def main():
    report = json.loads(
        (PROJECT_DIR / "validation" / "phase8" / "paddleocr_report.json").read_text()
    )
    source = next(
        row
        for row in report["agreement"]["samples"]
        if row["predicted_digits"] == row["true_digits"]
        and row["confidence"] >= MODEL.ocr_min_confidence
    )
    source_dir = PROJECT_DIR / "validation" / "phase8" / "samples" / source["sample_id"]
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    source_full = cv2.imread(str(source_dir / "full.png"))
    source_roi = cv2.imread(str(source_dir / "roi.png"))
    if source_full is None or source_roi is None:
        print(f"FAIL: saved GUI smoke source is incomplete: {source_dir}")
        return 1

    app = QApplication([])
    window = MainWindow(CAMERA)
    validation_directory = tempfile.TemporaryDirectory()
    window._validation_root = Path(validation_directory.name) / "samples"
    window.show()
    results = []
    errors = []
    state = {"started": False, "finished": False}

    window.inference_worker.result_ready.connect(results.append)
    window.inference_worker.model_failed.connect(errors.append)
    window.inference_worker.inference_failed.connect(errors.append)
    window.camera_worker.error.connect(errors.append)

    def start_when_ready():
        if errors:
            finish()
            return
        if not window._model_ready:
            QTimer.singleShot(100, start_when_ready)
            return
        if state["started"]:
            return
        state["started"] = True
        window.captured_frame = source_full.copy()
        window.captured_roi = source_roi.copy()
        window.captured_roi_bounds = tuple(source_metadata["roi_bounds"])
        window._captured_at = datetime.fromisoformat(source_metadata["captured_at"])
        window.inference_requested.emit(
            {
                "roi": window.captured_roi,
                "full_frame": window.captured_frame,
                "roi_bounds": window.captured_roi_bounds,
            }
        )

    def finish():
        if state["finished"]:
            return
        state["finished"] = True
        evidence = PROJECT_DIR / "debug" / "phase8_gui.png"
        if not window.grab().save(str(evidence)):
            errors.append("failed to save Phase 5 GUI evidence")
        window.close()
        app.quit()

    def result_received(_result):
        window.true_reading_input.setText(source_metadata["true_reading"])
        window.condition_combo.setCurrentText("Glare")
        window._save_validation_sample()
        QTimer.singleShot(250, finish)

    window.inference_worker.result_ready.connect(result_received)
    QTimer.singleShot(100, start_when_ready)
    QTimer.singleShot(20_000, finish)
    app.exec()

    if errors:
        print(f"FAIL: errors={errors}")
        return 1
    if len(results) != 1:
        print(f"FAIL: expected one inference result, received {len(results)}")
        return 1
    sample_directories = list(window._validation_root.glob("*"))
    if len(sample_directories) != 1:
        print(
            f"FAIL: expected one saved validation sample, received "
            f"{len(sample_directories)}"
        )
        return 1
    if not (sample_directories[0] / "metadata.json").is_file():
        print("FAIL: validation metadata was not saved")
        return 1
    metadata = json.loads((sample_directories[0] / "metadata.json").read_text())
    result = window._last_inference_result
    if result is None:
        print("FAIL: GUI did not produce an inference result")
        return 1
    if metadata.get("ocr_backend") != result.get("ocr_backend"):
        print("FAIL: validation metadata does not identify the OCR backend")
        return 1
    if result["backend"] != "ONNX Runtime CPU":
        print(f"FAIL: unexpected backend {result['backend']}")
        return 1
    if len(result["detections"]) != 1:
        print(f"FAIL: expected one LCD, received {len(result['detections'])}")
        return 1
    if result["roi_corners"] is None or result["full_corners"] is None:
        print("FAIL: GUI result does not contain Phase 6 mapped corners")
        return 1
    roi_corners = np.asarray(result["roi_corners"])
    full_corners = np.asarray(result["full_corners"])
    expected_offset = np.asarray(result["roi_bounds"][:2])
    offset_error = float(np.max(np.abs(full_corners - roi_corners - expected_offset)))
    if offset_error > 1e-4:
        print(f"FAIL: GUI full-frame mapping offset error={offset_error}")
        return 1
    if not Path(result["phase6_roi_debug"]).is_file():
        print("FAIL: ROI-space debug image was not saved")
        return 1
    if not Path(result["phase6_full_debug"]).is_file():
        print("FAIL: full-frame debug image was not saved")
        return 1
    if not Path(result["lcd_raw"]).is_file():
        print("FAIL: raw perspective warp was not saved")
        return 1
    if not Path(result["lcd_rectified"]).is_file():
        print("FAIL: rectified LCD image was not saved")
        return 1
    raw_height, raw_width = result["lcd_raw_shape"][:2]
    rectified_height, rectified_width = result["lcd_rectified_shape"][:2]
    if raw_width <= raw_height or rectified_width <= rectified_height:
        print("FAIL: GUI perspective output is not horizontal")
        return 1
    for debug_key in ("paddle_primary", "paddle_secondary"):
        if not Path(result[debug_key]).is_file():
            print(f"FAIL: PaddleOCR crop is missing: {debug_key}")
            return 1
    digit_result = result["digit_result"]
    if not digit_result["backend"].startswith("PaddleOCR"):
        print(f"FAIL: unexpected OCR backend: {digit_result['backend']}")
        return 1
    if digit_result["success"] and not digit_result["digits"]:
        print("FAIL: successful digit result is empty")
        return 1
    if not digit_result["success"] and digit_result["digits"] is not None:
        print("FAIL: failed digit result silently returned digits")
        return 1
    if not digit_result["success"] or digit_result["digits"] != source["true_digits"]:
        print(
            f"FAIL: expected saved LCD {source['true_digits']}, "
            f"received {digit_result['digits']}"
        )
        return 1
    print(
        f"backend={result['backend']} candidates={result['candidate_count']} "
        f"kept={len(result['detections'])} inference_ms={result['inference_ms']:.2f} "
        f"session_count={result['configuration_count']} offset_error={offset_error:.6f} "
        f"rectified={rectified_width}x{rectified_height} "
        f"digits={digit_result['digits']} digit_confidence={digit_result['confidence']:.3f}"
    )
    print(
        f"PASS: GUI retained, mapped, rectified, and safely read one LCD with PaddleOCR "
        f"at confidence {MODEL.confidence_threshold:.2f} and rotated NMS IoU "
        f"{MODEL.rotated_nms_iou:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
