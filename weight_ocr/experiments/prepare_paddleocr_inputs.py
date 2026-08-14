"""Prepare current OBB/perspective crops without importing PaddlePaddle."""

import json
from pathlib import Path
import sys

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import MODEL
from inference.obb_decoder import decode_onnx_output, rotated_nms
from inference.onnx_obb import OnnxObbModel
from vision.letterbox import letterbox
from vision.obb_mapping import model_corners_to_roi
from vision.perspective import rectify_lcd


SAMPLES_DIR = PROJECT_DIR / "validation" / "phase8" / "samples"
OUTPUT_DIR = PROJECT_DIR / "validation" / "phase8" / "paddleocr_inputs"


def input_variants(rectified):
    height, width = rectified.shape[:2]
    x1 = int(round(width * 0.32))
    x2 = int(round(width * 0.92))
    y2 = int(round(height * 0.88))
    digit_crop = rectified[
        int(round(height * 0.03)) : y2,
        x1:x2,
    ]
    digit_top08 = rectified[int(round(height * 0.08)) : y2, x1:x2]
    gray = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _threshold, binary = cv2.threshold(
        cv2.GaussianBlur(clahe, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return {
        "full_color": rectified,
        "digit_color": digit_crop,
        "digit_top08": digit_top08,
        "digit_clahe": cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        "digit_otsu": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
    }


def rectify_sample(detector, roi):
    model_bgr, transform = letterbox(
        roi,
        MODEL.input_width,
        MODEL.input_height,
    )
    outputs = detector.infer(cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB))
    candidates = decode_onnx_output(
        outputs[detector.output_names[0]],
        MODEL.confidence_threshold,
    )
    detections = rotated_nms(
        candidates,
        MODEL.confidence_threshold,
        MODEL.rotated_nms_iou,
    )[:1]
    if not detections:
        return None
    corners = model_corners_to_roi(detections[0].corners(), transform)
    _raw, rectified = rectify_lcd(
        roi,
        corners,
        MODEL.lcd_inner_margin,
        MODEL.lcd_quad_expand_x,
        MODEL.lcd_quad_expand_y,
    )
    return rectified


def main():
    detector = OnnxObbModel(MODEL.onnx_path)
    manifest = []
    for metadata_path in sorted(SAMPLES_DIR.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        roi = cv2.imread(str(metadata_path.parent / "roi.png"))
        rectified = rectify_sample(detector, roi)
        if rectified is None:
            continue

        files = {}
        for variant_name, image in input_variants(rectified).items():
            variant_dir = OUTPUT_DIR / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            path = variant_dir / f"{metadata['sample_id']}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Failed to save PaddleOCR input: {path}")
            files[variant_name] = str(path)
        manifest.append(
            {
                "sample_id": metadata["sample_id"],
                "true_digits": str(metadata["label_digits"]),
                "files": files,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} labeled LCD crops: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
