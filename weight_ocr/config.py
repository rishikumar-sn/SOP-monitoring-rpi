from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CameraConfig:
    device: str = "/dev/video0"
    width: int = 2560
    height: int = 1440
    fps: int = 30
    fourcc: str = "MJPG"
    preview_fps: int = 15


CAMERA = CameraConfig()


@dataclass(frozen=True)
class ModelConfig:
    onnx_path: Path
    input_width: int = 512
    input_height: int = 512
    confidence_threshold: float = 0.25
    rotated_nms_iou: float = 0.30
    lcd_quad_expand_x: float = 0.08
    lcd_quad_expand_y: float = 0.04
    lcd_inner_margin: float = 0.0
    ocr_crop_x: tuple = (0.32, 0.92)
    ocr_crop_bottom: float = 0.88
    ocr_primary_top: float = 0.03
    ocr_secondary_top: float = 0.08
    ocr_min_confidence: float = 0.70


MODEL = ModelConfig(
    onnx_path=PROJECT_DIR / "models" / "lcd_obb.onnx",
)
