"""Compare the original ONNX prediction with the Phase 5 acceptance threshold."""

from pathlib import Path
import sys

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import MODEL
from inference.obb_decoder import decode_onnx_output, draw_detections, rotated_nms
from inference.onnx_obb import OnnxObbModel
from vision.letterbox import letterbox


def main() -> int:
    image = cv2.imread(str(PROJECT_DIR / "captures" / "latest_roi.png"))
    if image is None:
        print("FAIL: captures/latest_roi.png is missing")
        return 1
    letterboxed_bgr, _transform = letterbox(image, 512, 512)
    model_rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    model = OnnxObbModel(MODEL.onnx_path)
    output = model.infer(model_rgb)[model.output_names[0]]
    if output.shape != (1, 6, 5376):
        print(f"FAIL: unexpected ONNX output shape {output.shape}")
        return 1
    predictions = output[0]
    maximum_confidence = float(predictions[4].max())
    candidates = decode_onnx_output(output, MODEL.confidence_threshold)
    detections = rotated_nms(
        candidates,
        MODEL.confidence_threshold,
        MODEL.rotated_nms_iou,
    )
    debug_path = PROJECT_DIR / "debug" / "onnx_obb_decoded.png"
    cv2.imwrite(
        str(debug_path),
        draw_detections(model_rgb, detections),
    )

    print(f"output_shape={output.shape}")
    print(f"maximum_confidence={maximum_confidence:.6f}")
    print(
        f"raw_candidates_at_{MODEL.confidence_threshold:.2f}="
        f"{len(candidates)}"
    )
    print(
        f"detections_after_rotated_nms_{MODEL.rotated_nms_iou:.2f}="
        f"{len(detections)}"
    )
    for detection in detections:
        print(detection)

    if len(detections) != 1:
        print(f"FAIL: expected one LCD after rotated NMS, received {len(detections)}")
        return 1
    print("PASS: rotated NMS retained exactly one full-LCD OBB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
