import logging
import threading
import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from camera.usb_camera import USBCamera
from config import CameraConfig


LOGGER = logging.getLogger(__name__)


class CameraWorker(QObject):
    frame_ready = pyqtSignal(object)
    camera_opened = pyqtSignal(object)
    stats_updated = pyqtSignal(float, int)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, config: CameraConfig):
        super().__init__()
        self._config = config
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_full_resolution_frame = None

    @pyqtSlot()
    def run(self):
        camera = USBCamera(self._config)
        captured_frames = 0
        consecutive_failures = 0
        stats_started = time.monotonic()
        preview_interval = 1.0 / self._config.preview_fps
        next_preview_at = stats_started

        try:
            info = camera.open()
            self.camera_opened.emit(info)

            while not self._stop_event.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        raise RuntimeError("Webcam stopped returning frames")
                    continue

                consecutive_failures = 0
                captured_frames += 1
                with self._frame_lock:
                    self._latest_full_resolution_frame = frame

                now = time.monotonic()
                if now >= next_preview_at:
                    self.frame_ready.emit(frame)
                    next_preview_at = now + preview_interval

                elapsed = now - stats_started
                if elapsed >= 1.0:
                    self.stats_updated.emit(captured_frames / elapsed, captured_frames)
                    captured_frames = 0
                    stats_started = now
        except Exception as exc:
            LOGGER.exception("Camera worker failed")
            self.error.emit(str(exc))
        finally:
            camera.close()
            self.stopped.emit()
            QThread.currentThread().quit()

    def latest_full_resolution_frame(self):
        with self._frame_lock:
            if self._latest_full_resolution_frame is None:
                return None
            return self._latest_full_resolution_frame.copy()

    def request_stop(self):
        self._stop_event.set()
