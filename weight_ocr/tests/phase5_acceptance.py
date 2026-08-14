"""Run the Phase 5 OBB acceptance matrix on three live camera frames."""

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import CAMERA, MODEL
from inference.obb_decoder import decode_onnx_output, draw_detections, rotated_nms
from inference.onnx_obb import OnnxObbModel
from vision.letterbox import letterbox


ANGLES = (-30, -20, -10, 0, 10, 20, 30)


def capture_frames(count):
    camera = cv2.VideoCapture(CAMERA.device, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA.fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA.device}")
    frames = []
    try:
        for _ in range(20):
            ok, _frame = camera.read()
            if not ok:
                raise RuntimeError("Camera warm-up frame failed")
        for _ in range(count):
            for _ in range(5):
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera capture failed")
            frames.append(frame.copy())
    finally:
        camera.release()
    return frames


def locate_saved_roi(frame, template):
    scale = 0.25
    small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_template = cv2.resize(
        template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    match = cv2.matchTemplate(small_frame, small_template, cv2.TM_CCOEFF_NORMED)
    _minimum, score, _minimum_location, location = cv2.minMaxLoc(match)
    if score < 0.80:
        raise RuntimeError(f"Could not relocate saved ROI in live frame: score={score:.3f}")
    x = round(location[0] / scale)
    y = round(location[1] / scale)
    height, width = template.shape[:2]
    return (x, y, x + width, y + height), score


def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def long_edge_orientation(detection):
    angle = detection.angle_degrees
    if detection.height > detection.width:
        angle += 90.0
    return angle % 180.0


def orientation_error(actual, expected):
    return abs((actual - expected + 90.0) % 180.0 - 90.0)


def infer(model, bgr_image):
    letterboxed_bgr, _transform = letterbox(bgr_image, 512, 512)
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
    return model_rgb, candidates, detections


def main():
    template = cv2.imread(str(PROJECT_DIR / "captures" / "latest_roi.png"))
    if template is None:
        print("FAIL: captures/latest_roi.png is missing")
        return 1

    frames = capture_frames(3)
    roi, match_score = locate_saved_roi(frames[0], template)
    x1, y1, x2, y2 = roi
    model = OnnxObbModel(MODEL.onnx_path)
    capture_dir = PROJECT_DIR / "captures" / "phase5"
    debug_dir = PROJECT_DIR / "debug" / "phase5"
    capture_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    montage_images = []
    print(f"live_roi={roi} template_match={match_score:.3f}")
    print(
        f"threshold={MODEL.confidence_threshold:.2f} "
        f"rotated_nms_iou={MODEL.rotated_nms_iou:.2f}"
    )
    for image_index, frame in enumerate(frames, start=1):
        roi_image = frame[y1:y2, x1:x2].copy()
        cv2.imwrite(str(capture_dir / f"straight_{image_index}.png"), roi_image)
        _rgb, _candidates, baseline = infer(model, roi_image)
        if len(baseline) != 1:
            failures.append(
                f"image {image_index}: baseline retained {len(baseline)} boxes"
            )
            continue
        baseline_orientation = long_edge_orientation(baseline[0])

        for angle in ANGLES:
            rotated = rotate_image(roi_image, angle)
            model_rgb, candidates, detections = infer(model, rotated)
            case_name = f"image_{image_index}_angle_{angle:+03d}"
            if len(detections) != 1:
                failures.append(
                    f"{case_name}: retained {len(detections)} boxes after NMS"
                )
                print(
                    f"{case_name}: raw={len(candidates)} kept={len(detections)} FAIL"
                )
                continue

            detection = detections[0]
            actual_orientation = long_edge_orientation(detection)
            expected_orientation = (baseline_orientation - angle) % 180.0
            error = orientation_error(actual_orientation, expected_orientation)
            long_side = max(detection.width, detection.height)
            short_side = min(detection.width, detection.height)
            if error > 8.0:
                failures.append(f"{case_name}: orientation error {error:.1f} degrees")
            if short_side <= 0 or long_side / short_side < 1.8:
                failures.append(f"{case_name}: box does not follow full LCD aspect")

            overlay = draw_detections(model_rgb, detections)
            cv2.putText(
                overlay,
                f"frame {image_index} rotation {angle:+d} deg",
                (8, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(debug_dir / f"{case_name}.png"), overlay)
            montage_images.append(cv2.resize(overlay, (256, 256)))
            print(
                f"{case_name}: raw={len(candidates)} kept=1 "
                f"confidence={detection.confidence:.3f} "
                f"orientation={actual_orientation:.1f} "
                f"expected={expected_orientation:.1f} error={error:.1f}"
            )

    if montage_images:
        rows = []
        for start in range(0, len(montage_images), len(ANGLES)):
            rows.append(np.hstack(montage_images[start : start + len(ANGLES)]))
        montage = np.vstack(rows)
        cv2.imwrite(str(PROJECT_DIR / "debug" / "phase5_acceptance_montage.jpg"), montage)

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PASS: all {len(frames) * len(ANGLES)} OBB cases retained one rotating LCD box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
