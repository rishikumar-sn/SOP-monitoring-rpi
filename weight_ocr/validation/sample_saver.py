from datetime import datetime
import json
from pathlib import Path
import re
import shutil

import cv2


ARTIFACTS = {
    "debug_image": "model_obb.png",
    "phase6_roi_debug": "roi_obb.png",
    "phase6_full_debug": "full_obb.png",
    "lcd_raw": "lcd_raw.png",
    "lcd_rectified": "lcd_rectified.png",
    "paddle_primary": "paddle_primary.png",
    "paddle_secondary": "paddle_secondary.png",
}


def normalize_reading(reading):
    value = str(reading).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        raise ValueError("True LCD reading must contain only digits and one decimal point")
    digits = value.replace(".", "")
    if not 1 <= len(digits) <= 4:
        raise ValueError("True LCD reading must contain between one and four digits")
    return value, digits


def _condition_slug(condition):
    slug = re.sub(r"[^a-z0-9]+", "_", str(condition).strip().lower()).strip("_")
    if not slug:
        raise ValueError("A validation condition is required")
    return slug


def _write_image(path, image):
    if image is None or image.size == 0 or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save validation image: {path}")


def save_validation_sample(
    root,
    true_reading,
    condition,
    roi_image,
    full_image,
    roi_bounds,
    inference_result=None,
    inference_error=None,
    captured_at=None,
):
    display_reading, label_digits = normalize_reading(true_reading)
    condition_slug = _condition_slug(condition)
    timestamp = captured_at or datetime.now().astimezone()
    sample_id = (
        f"{timestamp:%Y%m%d_%H%M%S_%f}_{label_digits}_{condition_slug}"
    )
    sample_dir = Path(root) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)

    _write_image(sample_dir / "roi.png", roi_image)
    _write_image(sample_dir / "full.png", full_image)
    saved_files = ["roi.png", "full.png"]
    result = inference_result or {}
    for key, destination_name in ARTIFACTS.items():
        source_value = result.get(key)
        if not source_value:
            continue
        source = Path(source_value)
        if source.is_file():
            shutil.copy2(source, sample_dir / destination_name)
            saved_files.append(destination_name)

    digit_result = result.get("digit_result") or {}
    detections = result.get("detections") or []
    if inference_error:
        inference_status = "inference_error"
    elif not inference_result:
        inference_status = "capture_only"
    elif not detections:
        inference_status = "lcd_not_found"
    elif digit_result.get("success"):
        inference_status = "decoded"
    else:
        inference_status = "read_failed"

    metadata = {
        "sample_id": sample_id,
        "captured_at": timestamp.isoformat(),
        "true_reading": display_reading,
        "label_digits": label_digits,
        "condition": condition_slug,
        "inference_status": inference_status,
        "inference_error": inference_error,
        "predicted_digits": digit_result.get("digits"),
        "digit_confidence": digit_result.get("confidence"),
        "failed_slots": digit_result.get("failed_slots", []),
        "detector_confidence": (
            detections[0].get("confidence") if detections else None
        ),
        "backend": result.get("backend"),
        "ocr_backend": result.get("ocr_backend"),
        "roi_bounds": list(roi_bounds),
        "saved_files": saved_files,
    }
    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sample_dir, metadata
