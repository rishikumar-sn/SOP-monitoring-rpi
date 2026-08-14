"""Verify LCD perspective rectification for straight and rotated captures."""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import MODEL
from inference.obb_decoder import decode_onnx_output, rotated_nms
from inference.onnx_obb import OnnxObbModel
from vision.letterbox import letterbox
from vision.obb_mapping import model_corners_to_roi
from vision.perspective import rectify_lcd


ANGLES = (-30, -20, -10, 0, 10, 20, 30)


def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def detect_corners(model, roi_image):
    letterboxed_bgr, transform = letterbox(roi_image, 512, 512)
    model_rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    outputs = model.infer(model_rgb)
    candidates = decode_onnx_output(
        outputs[model.output_names[0]], MODEL.confidence_threshold
    )
    detections = rotated_nms(
        candidates,
        MODEL.confidence_threshold,
        MODEL.rotated_nms_iou,
    )
    if len(detections) != 1:
        raise RuntimeError(f"Expected one LCD, received {len(detections)}")
    return (
        model_corners_to_roi(detections[0].corners(), transform),
        detections[0].confidence,
    )


def normalized_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1 = int(round(MODEL.ocr_crop_x[0] * gray.shape[1]))
    x2 = int(round(MODEL.ocr_crop_x[1] * gray.shape[1]))
    y1 = int(round(MODEL.ocr_primary_top * gray.shape[0]))
    y2 = int(round(MODEL.ocr_crop_bottom * gray.shape[0]))
    return cv2.resize(gray[y1:y2, x1:x2], (300, 100), interpolation=cv2.INTER_LINEAR)


def main():
    clean_roi = cv2.imread(str(PROJECT_DIR / "captures" / "latest_roi.png"))
    if clean_roi is None:
        print("FAIL: captures/latest_roi.png is required")
        return 1
    original_pixels = clean_roi.copy()
    model = OnnxObbModel(MODEL.onnx_path)
    output_dir = PROJECT_DIR / "debug" / "phase7"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []

    for angle in ANGLES:
        rotated_roi = rotate_image(clean_roi, angle)
        corners, confidence = detect_corners(model, rotated_roi)
        raw, rectified = rectify_lcd(
            rotated_roi,
            corners,
            MODEL.lcd_inner_margin,
            MODEL.lcd_quad_expand_x,
            MODEL.lcd_quad_expand_y,
        )
        if raw.shape[1] <= raw.shape[0] or rectified.shape[1] <= rectified.shape[0]:
            failures.append(f"angle {angle:+d}: rectified LCD is not horizontal")
        aspect_ratio = rectified.shape[1] / rectified.shape[0]
        if not 2.0 <= aspect_ratio <= 4.0:
            failures.append(
                f"angle {angle:+d}: unexpected LCD aspect ratio {aspect_ratio:.2f}"
            )
        cv2.imwrite(str(output_dir / f"lcd_raw_angle_{angle:+03d}.png"), raw)
        cv2.imwrite(
            str(output_dir / f"lcd_rectified_angle_{angle:+03d}.png"),
            rectified,
        )
        results[angle] = (raw, rectified, confidence)

    reference = normalized_gray(results[0][1])
    raw_montage = []
    rectified_montage = []
    for angle in ANGLES:
        raw, rectified, confidence = results[angle]
        normalized = normalized_gray(rectified)
        correlation = float(
            cv2.matchTemplate(normalized, reference, cv2.TM_CCOEFF_NORMED)[0, 0]
        )
        upside_down = cv2.rotate(normalized, cv2.ROTATE_180)
        upside_down_correlation = float(
            cv2.matchTemplate(upside_down, reference, cv2.TM_CCOEFF_NORMED)[0, 0]
        )
        if correlation < upside_down_correlation + 0.20:
            failures.append(
                f"angle {angle:+d}: inconsistent digit orientation "
                f"({correlation:.3f} vs upside-down {upside_down_correlation:.3f})"
            )
        raw_montage.append(cv2.resize(raw, (320, 128)))
        enlarged = cv2.resize(rectified, (320, 128))
        cv2.putText(
            enlarged,
            f"{angle:+d} deg corr {correlation:.2f}",
            (5, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            enlarged,
            f"{angle:+d} deg corr {correlation:.2f}",
            (5, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        rectified_montage.append(enlarged)
        print(
            f"angle={angle:+3d} confidence={confidence:.3f} "
            f"raw={raw.shape[1]}x{raw.shape[0]} "
            f"rectified={rectified.shape[1]}x{rectified.shape[0]} "
            f"correlation={correlation:.3f} upside_down={upside_down_correlation:.3f}"
        )

    montage = np.vstack((np.hstack(raw_montage), np.hstack(rectified_montage)))
    cv2.imwrite(str(PROJECT_DIR / "debug" / "phase7_rectification_montage.jpg"), montage)
    if not np.array_equal(clean_roi, original_pixels):
        failures.append("clean ROI pixels were modified during rectification")

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PASS: all {len(ANGLES)} LCDs are horizontal with consistent digit orientation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
