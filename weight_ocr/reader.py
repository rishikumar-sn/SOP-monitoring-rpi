"""Reusable, non-Qt LCD weight reader for the integrated workflow."""

from pathlib import Path

import cv2
import numpy as np

from .config import MODEL
from .inference.obb_decoder import decode_onnx_output, rotated_nms
from .inference.onnx_obb import OnnxObbModel
from .ocr.paddle_client import PaddleOCRClient
from .vision.letterbox import letterbox
from .vision.obb_mapping import model_corners_to_roi
from .vision.perspective import rectify_lcd


class WeightReader:
    def __init__(self):
        self.model = OnnxObbModel(MODEL.onnx_path)
        self.ocr = PaddleOCRClient()

    def close(self):
        self.ocr.close()

    def read(self, raw_bgr: np.ndarray, output_dir: Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        letterboxed_bgr, transform = letterbox(
            raw_bgr,
            MODEL.input_width,
            MODEL.input_height,
        )
        model_input = np.ascontiguousarray(
            cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
        )
        outputs = self.model.infer(model_input)
        candidates = decode_onnx_output(
            outputs[self.model.output_names[0]],
            MODEL.confidence_threshold,
        )
        detections = rotated_nms(
            candidates,
            MODEL.confidence_threshold,
            MODEL.rotated_nms_iou,
        )[:1]
        if not detections:
            return {
                "success": False,
                "status": "lcd_not_found",
                "message": "Weight could not be read. Place the scale LCD inside the box and recapture.",
            }

        image_corners = model_corners_to_roi(detections[0].corners(), transform)
        _lcd_raw, lcd_rectified = rectify_lcd(
            raw_bgr,
            image_corners,
            MODEL.lcd_inner_margin,
            MODEL.lcd_quad_expand_x,
            MODEL.lcd_quad_expand_y,
        )
        lcd_path = output_dir / "lcd_rectified.png"
        if not cv2.imwrite(str(lcd_path), lcd_rectified):
            raise RuntimeError("Could not save the rectified weight display image.")

        height, width = lcd_rectified.shape[:2]
        x1 = int(round(width * MODEL.ocr_crop_x[0]))
        x2 = int(round(width * MODEL.ocr_crop_x[1]))
        bottom = int(round(height * MODEL.ocr_crop_bottom))
        primary = lcd_rectified[
            int(round(height * MODEL.ocr_primary_top)) : bottom,
            x1:x2,
        ]
        secondary = lcd_rectified[
            int(round(height * MODEL.ocr_secondary_top)) : bottom,
            x1:x2,
        ]
        primary_path = output_dir / "ocr_primary.png"
        secondary_path = output_dir / "ocr_secondary.png"
        if not cv2.imwrite(str(primary_path), primary):
            raise RuntimeError("Could not save the primary OCR crop.")
        if not cv2.imwrite(str(secondary_path), secondary):
            raise RuntimeError("Could not save the secondary OCR crop.")

        paddle_result = self.ocr.recognize((primary_path, secondary_path))
        accepted = bool(
            paddle_result["agreed"]
            and paddle_result["confidence"] >= MODEL.ocr_min_confidence
        )
        digits = str(paddle_result.get("digits") or "") if accepted else ""
        if not digits.isdigit():
            accepted = False
        if not accepted:
            return {
                "success": False,
                "status": "read_failed",
                "message": "Weight could not be read clearly. Keep the scale steady inside the box and recapture.",
                "lcd_path": lcd_path,
            }

        weight_g = int(digits) / 100.0
        if weight_g <= 0:
            return {
                "success": False,
                "status": "invalid_weight",
                "message": "The detected weight is zero. Place the jewel on the scale and recapture.",
                "lcd_path": lcd_path,
            }
        return {
            "success": True,
            "status": "captured",
            "message": f"Actual weight captured: {weight_g:.2f} g",
            "digits": digits,
            "weight_g": round(weight_g, 2),
            "lcd_path": lcd_path,
        }
