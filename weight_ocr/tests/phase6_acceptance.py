"""Verify model-to-ROI-to-full-frame OBB mapping for straight/rotated LCDs."""

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
from vision.obb_mapping import draw_polygon_copy, map_model_corners


ANGLES = (-30, -20, -10, 0, 10, 20, 30)


def locate_roi(full_frame, roi_image):
    scale = 0.25
    small_full = cv2.resize(
        full_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    small_roi = cv2.resize(
        roi_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    coarse_match = cv2.matchTemplate(small_full, small_roi, cv2.TM_CCOEFF_NORMED)
    _minimum, _score, _minimum_location, coarse = cv2.minMaxLoc(coarse_match)
    coarse_x = round(coarse[0] / scale)
    coarse_y = round(coarse[1] / scale)

    margin = 8
    roi_height, roi_width = roi_image.shape[:2]
    search_x1 = max(0, coarse_x - margin)
    search_y1 = max(0, coarse_y - margin)
    search_x2 = min(full_frame.shape[1], coarse_x + roi_width + margin)
    search_y2 = min(full_frame.shape[0], coarse_y + roi_height + margin)
    search = full_frame[search_y1:search_y2, search_x1:search_x2]
    exact_match = cv2.matchTemplate(search, roi_image, cv2.TM_CCOEFF_NORMED)
    _minimum, score, _minimum_location, exact = cv2.minMaxLoc(exact_match)
    x1 = search_x1 + exact[0]
    y1 = search_y1 + exact[1]
    if score < 0.99:
        raise RuntimeError(f"Saved ROI/full-frame pair does not match: {score:.3f}")
    return (x1, y1, x1 + roi_width, y1 + roi_height), score


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


def infer(model, roi_image):
    letterboxed_bgr, transform = letterbox(roi_image, 512, 512)
    model_rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    outputs = model.infer(model_rgb)
    candidates = decode_onnx_output(
        outputs[model.output_names[0]], MODEL.confidence_threshold
    )
    return transform, rotated_nms(
        candidates,
        MODEL.confidence_threshold,
        MODEL.rotated_nms_iou,
    )


def main():
    clean_roi = cv2.imread(str(PROJECT_DIR / "captures" / "latest_roi.png"))
    clean_full = cv2.imread(str(PROJECT_DIR / "captures" / "latest_full.png"))
    if clean_roi is None or clean_full is None:
        print("FAIL: paired latest_roi.png/latest_full.png captures are required")
        return 1

    roi_bounds, match_score = locate_roi(clean_full, clean_roi)
    x1, y1, x2, y2 = roi_bounds
    model = OnnxObbModel(MODEL.onnx_path)
    output_dir = PROJECT_DIR / "debug" / "phase6"
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    roi_montage = []
    full_montage = []

    print(f"roi_bounds={roi_bounds} exact_pair_match={match_score:.6f}")
    for angle in ANGLES:
        rotated_roi = rotate_image(clean_roi, angle)
        case_full = clean_full.copy()
        case_full[y1:y2, x1:x2] = rotated_roi
        transform, detections = infer(model, rotated_roi)
        if len(detections) != 1:
            failures.append(f"angle {angle:+d}: retained {len(detections)} boxes")
            continue

        detection = detections[0]
        roi_corners, full_corners = map_model_corners(
            detection.corners(),
            transform,
            roi_bounds,
            case_full.shape,
        )
        offset_error = np.max(
            np.abs(full_corners - roi_corners - np.array((x1, y1)))
        )
        roundtrip = np.array(
            [transform.roi_to_model_point(x, y) for x, y in roi_corners],
            dtype=np.float32,
        )
        roundtrip_error = float(np.max(np.abs(roundtrip - detection.corners())))
        if offset_error > 1e-4:
            failures.append(f"angle {angle:+d}: full-frame offset error {offset_error}")
        if roundtrip_error > 1e-3:
            failures.append(
                f"angle {angle:+d}: letterbox round-trip error {roundtrip_error}"
            )

        label = f"LCD {detection.confidence:.3f} angle {angle:+d}"
        roi_overlay = draw_polygon_copy(rotated_roi, roi_corners, label)
        full_overlay = draw_polygon_copy(case_full, full_corners, label)
        cv2.imwrite(str(output_dir / f"roi_angle_{angle:+03d}.png"), roi_overlay)
        cv2.imwrite(str(output_dir / f"full_angle_{angle:+03d}.png"), full_overlay)
        roi_montage.append(cv2.resize(roi_overlay, (320, 214)))
        full_montage.append(cv2.resize(full_overlay, (320, 180)))
        print(
            f"angle={angle:+3d} confidence={detection.confidence:.3f} "
            f"offset_error={offset_error:.6f} roundtrip_error={roundtrip_error:.6f}"
        )

    if roi_montage and full_montage:
        montage = np.vstack((np.hstack(roi_montage), np.hstack(full_montage)))
        cv2.imwrite(
            str(PROJECT_DIR / "debug" / "phase6_mapping_montage.jpg"),
            montage,
        )

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PASS: all {len(ANGLES)} ROI/full-frame mappings are aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
