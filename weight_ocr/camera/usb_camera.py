from dataclasses import dataclass
import logging

import cv2

from config import CameraConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraInfo:
    device: str
    width: int
    height: int
    fps: float
    fourcc: str


class USBCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture = None

    def open(self) -> CameraInfo:
        capture = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"USB camera unavailable: {self.config.device}")

        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.config.fourcc),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        info = CameraInfo(
            device=self.config.device,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            fourcc=self._decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
        )
        LOGGER.info(
            "Camera negotiated %dx%d at %.2f FPS using %s",
            info.width,
            info.height,
            info.fps,
            info.fourcc,
        )
        return info

    def read(self):
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        return self._capture.read()

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @staticmethod
    def _decode_fourcc(value: float) -> str:
        encoded = int(value)
        return "".join(chr((encoded >> (8 * index)) & 0xFF) for index in range(4))
