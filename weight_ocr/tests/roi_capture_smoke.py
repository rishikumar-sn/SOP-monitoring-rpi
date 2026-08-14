"""Exercise five Phase 2 ROI captures using the real 2K camera."""

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import CAMERA
from ui.main_window import MainWindow


ROIS = {
    "top-left": (0, 0, 640, 360),
    "top-right": (1920, 0, 2560, 360),
    "bottom-left": (0, 1080, 640, 1440),
    "bottom-right": (1920, 1080, 2560, 1440),
    "center": (960, 540, 1600, 900),
}


def main() -> int:
    app = QApplication([])
    window = MainWindow(CAMERA)
    window.show()
    errors = []
    results = []
    window.camera_worker.error.connect(errors.append)

    def run_captures():
        if window.latest_full_resolution_frame is None:
            QTimer.singleShot(250, run_captures)
            return
        try:
            for name, roi in ROIS.items():
                window.video_canvas.set_roi(roi)
                window._capture_roi()
                x1, y1, x2, y2 = roi
                expected = window.captured_frame[y1:y2, x1:x2]
                saved = cv2.imread(str(window._capture_path), cv2.IMREAD_COLOR)
                passed = (
                    window.captured_roi is not None
                    and np.array_equal(window.captured_roi, expected)
                    and saved is not None
                    and np.array_equal(saved, expected)
                )
                results.append((name, roi, window.captured_roi.shape, passed))
            app.processEvents()
            debug_path = PROJECT_DIR / "debug" / "phase2_roi_gui.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(debug_path)):
                errors.append("failed to save Phase 2 GUI evidence image")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            window.close()
            app.quit()

    QTimer.singleShot(2500, run_captures)
    QTimer.singleShot(15000, app.quit)
    app.exec()

    for name, roi, shape, passed in results:
        print(f"{name}: roi={roi} crop_shape={shape} exact_pixels={passed}")
    if errors:
        print(f"FAIL: errors={errors}")
        return 1
    if len(results) != len(ROIS) or not all(result[3] for result in results):
        print("FAIL: not all ROI captures matched the original full-frame slice")
        return 1
    print("PASS: all five real-camera ROI captures matched exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
