from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

PARTS = [
    ("pendant", "Pendant", (255, 60, 160)),
    ("chain", "Chain Region", (70, 190, 70)),
    ("tassel", "Tassel / Thread", (40, 110, 255)),
]

FEEDBACK_TEMPLATE_SIZE = 192
MAX_SIMILAR_FEEDBACK_SCORE = 32.0
MAX_NEGATIVE_FEEDBACK_SCORE = 8.0
PENDANT_AUTO_MIN_SCORE = 7.25
TASSEL_AUTO_MIN_SCORE_HARAM = 9.0
TASSEL_AUTO_MIN_SCORE_OTHER = 11.5
TASSEL_AUTO_DISABLED_TYPES = frozenset({"necklace", "dollar chain", "dollar"})

# FastSAM is class-agnostic, so these prompts are also the explicit policy used
# by the geometric/color evidence gates below and are exposed in debug output.
PART_DETECTION_PROMPTS = {
    "pendant": (
        "Pendant: detect only a compact round, oval, or drop-shaped "
        "ornament attached to the chain near the bottom middle of the jewel. "
        "Do not invent a pendant when the chain continues without a distinct "
        "compact hanging ornament."
    ),
    "tassel": (
        "Tassel/thread: automatic detection is disabled for Necklace and "
        "Dollar chain. Haram uses conservative detection, while other jewel "
        "types require very strong terminal textile-strand evidence. Manual "
        "correction remains authoritative."
    ),
}


@dataclass
class DetectionMask:
    score: float
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    fill_ratio: float
    aspect_ratio: float
    red_ratio: float
    gold_ratio: float


@dataclass
class PreparedInput:
    original_image: np.ndarray
    working_image: np.ndarray
    working_mask: np.ndarray


@dataclass
class ManualPartFeedback:
    working_size: Tuple[int, int]
    pendant_bbox: Tuple[int, int, int, int] | None = None
    tassel_bbox: Tuple[int, int, int, int] | None = None
    pendant_mask: np.ndarray | None = None
    tassel_mask: np.ndarray | None = None
    no_pendant: bool = False
    no_tassel: bool = False
    feedback_path: str | None = None
    match_type: str | None = None
    match_score: float | None = None
    alignment_score: float | None = None
    source_jewel_type: str | None = None


@dataclass(frozen=True)
class FeedbackSignature:
    exact_hash: str
    image_dhash: str
    mask_dhash: str
    object_aspect_ratio: float
    object_fill_ratio: float
    object_density: float | None = None
    edge_dhash: str = ""
    color_histogram: Tuple[float, ...] = ()

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "exact_hash": self.exact_hash,
            "image_dhash": self.image_dhash,
            "mask_dhash": self.mask_dhash,
            "object_aspect_ratio": round(float(self.object_aspect_ratio), 6),
            "object_fill_ratio": round(float(self.object_fill_ratio), 6),
        }
        if self.object_density is not None:
            payload["object_density"] = round(float(self.object_density), 6)
        if self.edge_dhash:
            payload["edge_dhash"] = self.edge_dhash
        if self.color_histogram:
            payload["color_histogram"] = [
                round(float(value), 6)
                for value in self.color_histogram
            ]
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a jewelry image into pendant, chain region, and tassel/thread using Fast-SAM plus conventional masking."
    )
    parser.add_argument("--image", help="Input jewelry image path.")
    parser.add_argument("--model", default="fast_sam_s.onnx", help="Fast-SAM ONNX model path.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory. A per-image folder will be created inside it.",
    )
    parser.add_argument("--input-size", type=int, default=640, help="Model inference size.")
    parser.add_argument("--conf-thres", type=float, default=0.20, help="Detection confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.85, help="Detection NMS IoU threshold.")
    parser.add_argument("--mask-thres", type=float, default=0.50, help="Mask binarization threshold.")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["CPUExecutionProvider"],
        help="onnxruntime execution providers.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a simple desktop GUI to choose an input image and preview the result.",
    )
    parser.add_argument(
        "--feedback-dir",
        default="feedback",
        help="Directory used to store per-image manual pendant/tassel corrections.",
    )
    parser.add_argument(
        "--jewel-type",
        default="",
        help="Classified jewel type used to select the tassel detection policy.",
    )
    return parser.parse_args()


def letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def clip_box(box: Iterable[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _safe_feedback_stem(image_path: Path) -> str:
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_path.stem).strip("_")
    return safe_stem[:48] if safe_stem else "image"


def feedback_file_path(
    image_path: Path,
    feedback_dir: Path,
    signature: FeedbackSignature | None = None,
) -> Path:
    if signature is not None:
        digest = signature.exact_hash[:16]
    else:
        resolved = str(image_path.resolve()).lower()
        digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return feedback_dir / f"{_safe_feedback_stem(image_path)}_{digest}.json"


def _infer_signature_mask(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None and np.any(mask):
        inferred = mask.astype(np.uint8)
    else:
        inferred = extract_primary_mask_from_otsu(image).astype(np.uint8)
        if not inferred.any():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inferred = (gray < 250).astype(np.uint8)

    inferred = close(inferred, 3)
    inferred = remove_small_components(inferred, max(20, inferred.size // 30000))
    if inferred.any():
        inferred = largest_component(inferred)
    if not inferred.any():
        inferred = np.ones(image.shape[:2], dtype=np.uint8)
    return inferred.astype(np.uint8)


def _canonical_feedback_view(
    image: np.ndarray,
    mask: np.ndarray,
    size: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    if not mask.any():
        mask = np.ones(image.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = bounding_box(mask)
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    pad = max(8, int(round(max(box_w, box_h) * 0.08)))
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    crop_image = image[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    crop_h, crop_w = crop_image.shape[:2]
    side = max(crop_h, crop_w) + 2 * pad

    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - crop_h) // 2
    offset_x = (side - crop_w) // 2

    image_region = canvas[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    mask_region = canvas_mask[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    image_region[crop_mask > 0] = crop_image[crop_mask > 0]
    mask_region[crop_mask > 0] = 1

    resized_image = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(canvas_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized_image, (resized_mask > 0).astype(np.uint8)


def _object_template_view(
    image: np.ndarray,
    object_mask: np.ndarray,
    part_mask: np.ndarray | None = None,
    size: int = FEEDBACK_TEMPLATE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Normalize an object and optional corrected part into an object-local square."""
    if not object_mask.any():
        object_mask = np.ones(image.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = bounding_box(object_mask)
    crop_image = image[y1 : y2 + 1, x1 : x2 + 1]
    crop_object = object_mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    crop_part = (
        part_mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
        if part_mask is not None
        else None
    )
    crop_h, crop_w = crop_object.shape
    side = max(1, crop_h, crop_w)
    offset_y = (side - crop_h) // 2
    offset_x = (side - crop_w) // 2

    image_canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    object_canvas = np.zeros((side, side), dtype=np.uint8)
    part_canvas = np.zeros((side, side), dtype=np.uint8) if crop_part is not None else None
    image_region = image_canvas[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    object_region = object_canvas[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    image_region[crop_object > 0] = crop_image[crop_object > 0]
    object_region[crop_object > 0] = 1
    if part_canvas is not None and crop_part is not None:
        part_canvas[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w][crop_part > 0] = 1

    normalized_image = cv2.resize(image_canvas, (size, size), interpolation=cv2.INTER_AREA)
    normalized_object = cv2.resize(object_canvas, (size, size), interpolation=cv2.INTER_NEAREST)
    normalized_part = (
        cv2.resize(part_canvas, (size, size), interpolation=cv2.INTER_NEAREST)
        if part_canvas is not None
        else None
    )
    return (
        normalized_image,
        (normalized_object > 0).astype(np.uint8),
        (normalized_part > 0).astype(np.uint8) if normalized_part is not None else None,
    )


def _restore_object_template_mask(
    normalized_mask: np.ndarray,
    object_mask: np.ndarray,
) -> np.ndarray:
    """Map an object-local template mask back to the current image coordinates."""
    restored = np.zeros_like(object_mask, dtype=np.uint8)
    if not normalized_mask.any() or not object_mask.any():
        return restored

    x1, y1, x2, y2 = bounding_box(object_mask)
    crop_h = y2 - y1 + 1
    crop_w = x2 - x1 + 1
    side = max(1, crop_h, crop_w)
    square_mask = cv2.resize(
        normalized_mask.astype(np.uint8),
        (side, side),
        interpolation=cv2.INTER_NEAREST,
    )
    offset_y = (side - crop_h) // 2
    offset_x = (side - crop_w) // 2
    crop = square_mask[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    restored[y1 : y2 + 1, x1 : x2 + 1] = crop
    return (restored & object_mask.astype(np.uint8)).astype(np.uint8)


def _encode_png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode feedback template.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_png_base64(value: object, grayscale: bool = False) -> np.ndarray | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, TypeError):
        return None
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    return decoded


def _feedback_template_key(part_key: str) -> str:
    return part_key.replace("_bbox", "_mask_png")


def _feedback_alignment_feature(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = cv2.bitwise_and(255 - gray, 255 - gray, mask=(mask > 0).astype(np.uint8))
    foreground = cv2.GaussianBlur(foreground, (5, 5), 0)
    maximum = float(foreground.max())
    if maximum > 0:
        foreground = np.clip(foreground.astype(np.float32) / maximum, 0.0, 1.0)
    else:
        foreground = foreground.astype(np.float32)
    return foreground


def _align_saved_template(
    saved_image: np.ndarray,
    saved_object_mask: np.ndarray,
    saved_part_mask: np.ndarray,
    current_image: np.ndarray,
    current_object_mask: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """Align a saved object-local correction to the current object-local view."""
    size = current_object_mask.shape[0]
    if saved_image.shape[:2] != (size, size):
        saved_image = cv2.resize(saved_image, (size, size), interpolation=cv2.INTER_AREA)
    if saved_object_mask.shape[:2] != (size, size):
        saved_object_mask = cv2.resize(
            saved_object_mask,
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
    if saved_part_mask.shape[:2] != (size, size):
        saved_part_mask = cv2.resize(
            saved_part_mask,
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )

    aligned_part = (saved_part_mask > 0).astype(np.uint8)
    alignment_score: float | None = None
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        current_feature = _feedback_alignment_feature(current_image, current_object_mask)
        saved_feature = _feedback_alignment_feature(saved_image, saved_object_mask)
        score, estimated = cv2.findTransformECC(
            current_feature,
            saved_feature,
            warp,
            cv2.MOTION_EUCLIDEAN,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                60,
                1e-5,
            ),
            None,
            3,
        )
        rotation_scale = estimated[:, :2]
        determinant = float(np.linalg.det(rotation_scale))
        translation = float(np.linalg.norm(estimated[:, 2]))
        if 0.80 <= determinant <= 1.20 and translation <= size * 0.22:
            aligned_part = cv2.warpAffine(
                aligned_part,
                estimated,
                (size, size),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            alignment_score = float(score)
    except cv2.error:
        pass

    aligned_part = (aligned_part > 0).astype(np.uint8)
    aligned_part &= current_object_mask.astype(np.uint8)
    aligned_part = close(aligned_part, 3)
    aligned_part = remove_small_components(
        aligned_part,
        max(4, aligned_part.size // 8000),
    )
    return aligned_part.astype(np.uint8), alignment_score


def _dhash_hex(image: np.ndarray, hash_size: int = 8) -> str:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] >= resized[:, :-1]
    value = 0
    for bit in diff.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return f"{value:0{diff.size // 4}x}"


def _hash_bytes(*arrays: np.ndarray) -> str:
    sha1 = hashlib.sha1()
    for array in arrays:
        sha1.update(np.ascontiguousarray(array).tobytes())
    return sha1.hexdigest()


def _masked_color_histogram(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, ...]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        (mask > 0).astype(np.uint8) * 255,
        [8, 3],
        [0, 180, 0, 256],
    ).reshape(-1)
    total = float(histogram.sum())
    if total > 0:
        histogram = histogram / total
    return tuple(float(value) for value in histogram)


def build_feedback_signature(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> FeedbackSignature:
    inferred_mask = _infer_signature_mask(image, mask)
    canonical_image, canonical_mask = _canonical_feedback_view(image, inferred_mask)
    x1, y1, x2, y2 = bounding_box(inferred_mask)
    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)

    return FeedbackSignature(
        exact_hash=_hash_bytes(canonical_image, canonical_mask),
        image_dhash=_dhash_hex(canonical_image),
        mask_dhash=_dhash_hex((canonical_mask * 255).astype(np.uint8)),
        object_aspect_ratio=bbox_w / float(max(1, bbox_h)),
        object_fill_ratio=float(inferred_mask.sum()) / float(max(1, inferred_mask.size)),
        object_density=float(inferred_mask.sum()) / float(max(1, bbox_w * bbox_h)),
        edge_dhash=_dhash_hex(
            cv2.Canny(
                cv2.cvtColor(canonical_image, cv2.COLOR_BGR2GRAY),
                45,
                135,
            )
        ),
        color_histogram=_masked_color_histogram(canonical_image, canonical_mask),
    )


def _read_feedback_payload(feedback_path: Path) -> dict[str, object] | None:
    try:
        with open(feedback_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _iter_feedback_entries(feedback_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    if not feedback_dir.exists():
        return []
    entries: list[tuple[Path, dict[str, object]]] = []
    for feedback_path in sorted(feedback_dir.glob("*.json")):
        payload = _read_feedback_payload(feedback_path)
        if payload is not None:
            entries.append((feedback_path, payload))
    return entries


def _signature_from_payload(payload: dict[str, object]) -> FeedbackSignature | None:
    raw_signature = payload.get("signature")
    if not isinstance(raw_signature, dict):
        return None

    exact_hash = str(raw_signature.get("exact_hash") or "").strip().lower()
    image_dhash = str(raw_signature.get("image_dhash") or "").strip().lower()
    mask_dhash = str(raw_signature.get("mask_dhash") or "").strip().lower()
    if not exact_hash or not image_dhash or not mask_dhash:
        return None

    try:
        object_aspect_ratio = float(raw_signature.get("object_aspect_ratio", 1.0))
        object_fill_ratio = float(raw_signature.get("object_fill_ratio", 0.0))
        object_density_value = raw_signature.get("object_density")
        object_density = (
            float(object_density_value)
            if object_density_value is not None
            else None
        )
        edge_dhash = str(raw_signature.get("edge_dhash") or "").strip().lower()
        color_histogram_value = raw_signature.get("color_histogram")
        color_histogram = (
            tuple(float(value) for value in color_histogram_value)
            if isinstance(color_histogram_value, list)
            else ()
        )
    except (TypeError, ValueError):
        return None

    return FeedbackSignature(
        exact_hash=exact_hash,
        image_dhash=image_dhash,
        mask_dhash=mask_dhash,
        object_aspect_ratio=object_aspect_ratio,
        object_fill_ratio=object_fill_ratio,
        object_density=object_density,
        edge_dhash=edge_dhash,
        color_histogram=color_histogram,
    )


def _hamming_distance_hex(left: str, right: str) -> int | None:
    if not left or not right or len(left) != len(right):
        return None
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return None


def _feedback_similarity_score(
    current_signature: FeedbackSignature,
    saved_signature: FeedbackSignature,
) -> float | None:
    image_distance = _hamming_distance_hex(current_signature.image_dhash, saved_signature.image_dhash)
    mask_distance = _hamming_distance_hex(current_signature.mask_dhash, saved_signature.mask_dhash)
    if image_distance is None or mask_distance is None:
        return None
    edge_distance = None
    if current_signature.edge_dhash and saved_signature.edge_dhash:
        edge_distance = _hamming_distance_hex(
            current_signature.edge_dhash,
            saved_signature.edge_dhash,
        )
        if edge_distance is None or edge_distance > 18:
            return None

    color_distance = None
    if (
        current_signature.color_histogram
        and saved_signature.color_histogram
        and len(current_signature.color_histogram)
        == len(saved_signature.color_histogram)
    ):
        color_distance = 0.5 * sum(
            abs(current - saved)
            for current, saved in zip(
                current_signature.color_histogram,
                saved_signature.color_histogram,
            )
        )
        if color_distance > 0.58:
            return None

    aspect_delta = abs(
        math.log(
            max(saved_signature.object_aspect_ratio, 1e-6)
            / max(current_signature.object_aspect_ratio, 1e-6)
        )
    )
    if (
        saved_signature.object_density is not None
        and current_signature.object_density is not None
    ):
        fill_delta = abs(
            saved_signature.object_density - current_signature.object_density
        )
    else:
        fill_delta = abs(
            saved_signature.object_fill_ratio - current_signature.object_fill_ratio
        )

    if image_distance > 12 or mask_distance > 12 or aspect_delta > 0.30 or fill_delta > 0.18:
        return None

    score = (
        float(image_distance)
        + float(mask_distance) * 1.4
        + aspect_delta * 20.0
        + fill_delta * 120.0
    )
    if edge_distance is not None:
        score += float(edge_distance) * 0.25
    if color_distance is not None:
        score += float(color_distance) * 6.0
    return score


def _feedback_mtime_ns(feedback_path: Path) -> int:
    try:
        return feedback_path.stat().st_mtime_ns
    except OSError:
        return 0


def _feedback_from_payload(
    payload: dict[str, object],
    current_shape: Tuple[int, int],
    current_object_bbox: Tuple[int, int, int, int] | None,
    prepared_image: np.ndarray,
    prepared_mask: np.ndarray | None,
    feedback_path: Path,
    match_type: str,
    match_score: float | None,
) -> ManualPartFeedback | None:
    size_values = payload.get("working_size")
    if not isinstance(size_values, list) or len(size_values) != 2:
        return None

    saved_shape = (max(1, int(size_values[0])), max(1, int(size_values[1])))
    current_object_mask = _infer_signature_mask(prepared_image, prepared_mask)
    pendant_mask, pendant_alignment = _load_feedback_template_mask(
        payload,
        "pendant_bbox",
        prepared_image,
        current_object_mask,
    )
    tassel_mask, tassel_alignment = _load_feedback_template_mask(
        payload,
        "tassel_bbox",
        prepared_image,
        current_object_mask,
    )
    pendant_bbox = bounding_box(pendant_mask) if pendant_mask is not None and pendant_mask.any() else _load_feedback_box(
        payload,
        "pendant_bbox",
        current_shape,
        saved_shape,
        current_object_bbox,
    )
    tassel_bbox = bounding_box(tassel_mask) if tassel_mask is not None and tassel_mask.any() else _load_feedback_box(
        payload,
        "tassel_bbox",
        current_shape,
        saved_shape,
        current_object_bbox,
    )
    saved_signature = _signature_from_payload(payload)
    rich_signature = bool(
        saved_signature is not None
        and saved_signature.edge_dhash
        and saved_signature.color_histogram
    )
    strong_negative_match = (
        match_type == "exact_signature"
        or match_type == "legacy_path"
        or (
            rich_signature
            and match_score is not None
            and match_score <= MAX_NEGATIVE_FEEDBACK_SCORE
        )
    )
    no_pendant = bool(payload.get("no_pendant", False)) and strong_negative_match
    no_tassel = bool(payload.get("no_tassel", False)) and strong_negative_match

    has_object_relative_pendant = isinstance(
        payload.get(_feedback_relative_box_key("pendant_bbox")),
        list,
    )
    has_object_relative_tassel = isinstance(
        payload.get(_feedback_relative_box_key("tassel_bbox")),
        list,
    )
    if match_type in {"similar_signature", "same_path_similar"}:
        if pendant_mask is None and not has_object_relative_pendant:
            pendant_bbox = None
        if tassel_mask is None and not has_object_relative_tassel:
            tassel_bbox = None

    if pendant_bbox is None and tassel_bbox is None and not no_pendant and not no_tassel:
        return None

    current_h, current_w = current_shape
    alignment_scores = [
        score
        for score in (pendant_alignment, tassel_alignment)
        if score is not None
    ]
    return ManualPartFeedback(
        working_size=(current_h, current_w),
        pendant_bbox=pendant_bbox,
        tassel_bbox=tassel_bbox,
        pendant_mask=pendant_mask,
        tassel_mask=tassel_mask,
        no_pendant=no_pendant,
        no_tassel=no_tassel,
        feedback_path=str(feedback_path),
        match_type=match_type,
        match_score=match_score,
        alignment_score=max(alignment_scores) if alignment_scores else None,
        source_jewel_type=str(payload.get("jewel_type") or "").strip() or None,
    )


def _legacy_path_feedback_is_current(
    image_path: Path,
    feedback_path: Path,
) -> bool:
    """Accept path-only legacy feedback only while the source file is unchanged."""
    try:
        return image_path.exists() and image_path.stat().st_mtime_ns <= feedback_path.stat().st_mtime_ns
    except OSError:
        return False


def _find_feedback_match(
    image_path: Path,
    feedback_dir: Path,
    current_signature: FeedbackSignature,
) -> tuple[Path, dict[str, object], str, float | None] | None:
    current_resolved = str(image_path.resolve()).lower()
    exact_matches: list[tuple[int, Path, dict[str, object]]] = []
    path_matches: list[tuple[int, Path, dict[str, object]]] = []
    path_similar_matches: list[tuple[float, int, Path, dict[str, object]]] = []
    similar_matches: list[tuple[float, int, Path, dict[str, object]]] = []

    for feedback_path, payload in _iter_feedback_entries(feedback_dir):
        saved_signature = _signature_from_payload(payload)
        if saved_signature is not None and saved_signature.exact_hash == current_signature.exact_hash:
            exact_matches.append((_feedback_mtime_ns(feedback_path), feedback_path, payload))
            continue

        payload_image_path = str(payload.get("image_path") or "").strip().lower()
        if payload_image_path and payload_image_path == current_resolved:
            if saved_signature is None:
                if _legacy_path_feedback_is_current(image_path, feedback_path):
                    path_matches.append((_feedback_mtime_ns(feedback_path), feedback_path, payload))
                continue
            similarity_score = _feedback_similarity_score(current_signature, saved_signature)
            if similarity_score is not None and similarity_score <= MAX_SIMILAR_FEEDBACK_SCORE:
                path_similar_matches.append(
                    (similarity_score, _feedback_mtime_ns(feedback_path), feedback_path, payload)
                )
            continue

        if saved_signature is None:
            continue

        similarity_score = _feedback_similarity_score(current_signature, saved_signature)
        if similarity_score is not None and similarity_score <= MAX_SIMILAR_FEEDBACK_SCORE:
            similar_matches.append((similarity_score, _feedback_mtime_ns(feedback_path), feedback_path, payload))

    if exact_matches:
        _, feedback_path, payload = max(exact_matches, key=lambda item: item[0])
        return feedback_path, payload, "exact_signature", 0.0

    if path_matches:
        _, feedback_path, payload = max(path_matches, key=lambda item: item[0])
        return feedback_path, payload, "legacy_path", None

    if path_similar_matches:
        path_similar_matches.sort(key=lambda item: (item[0], -item[1]))
        score, _, feedback_path, payload = path_similar_matches[0]
        return feedback_path, payload, "same_path_similar", score

    if similar_matches:
        similar_matches.sort(key=lambda item: (item[0], -item[1]))
        score, _, feedback_path, payload = similar_matches[0]
        return feedback_path, payload, "similar_signature", score

    return None


def _feedback_relative_box_key(key: str) -> str:
    return f"{key}_object"


def _normalize_box_to_object(
    bbox: Tuple[int, int, int, int],
    object_bbox: Tuple[int, int, int, int],
) -> list[float]:
    x1, y1, x2, y2 = bbox
    object_x1, object_y1, object_x2, object_y2 = object_bbox
    object_w = max(1.0, float(object_x2 - object_x1))
    object_h = max(1.0, float(object_y2 - object_y1))
    return [
        round((float(x1) - object_x1) / object_w, 6),
        round((float(y1) - object_y1) / object_h, 6),
        round((float(x2) - object_x1) / object_w, 6),
        round((float(y2) - object_y1) / object_h, 6),
    ]


def _load_relative_box(
    values: object,
    object_bbox: Tuple[int, int, int, int],
    current_shape: Tuple[int, int],
) -> Tuple[int, int, int, int] | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        relative = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and -2.0 <= value <= 3.0 for value in relative):
        return None

    object_x1, object_y1, object_x2, object_y2 = object_bbox
    object_w = max(1.0, float(object_x2 - object_x1))
    object_h = max(1.0, float(object_y2 - object_y1))
    current_h, current_w = current_shape
    return clip_box(
        (
            object_x1 + relative[0] * object_w,
            object_y1 + relative[1] * object_h,
            object_x1 + relative[2] * object_w,
            object_y1 + relative[3] * object_h,
        ),
        current_w,
        current_h,
    )


def _feedback_object_bbox(
    prepared_image: np.ndarray,
    prepared_mask: np.ndarray | None,
) -> Tuple[int, int, int, int]:
    reference_mask = _infer_signature_mask(prepared_image, prepared_mask)
    return bounding_box(reference_mask)


def _load_feedback_template_mask(
    payload: dict[str, object],
    part_key: str,
    prepared_image: np.ndarray,
    current_object_mask: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    object_image = _decode_png_base64(payload.get("object_template_png"))
    object_mask = _decode_png_base64(payload.get("object_template_mask_png"), grayscale=True)
    part_mask = _decode_png_base64(payload.get(_feedback_template_key(part_key)), grayscale=True)
    if object_image is None or object_mask is None or part_mask is None:
        return None, None

    try:
        template_size = int(payload.get("template_size") or FEEDBACK_TEMPLATE_SIZE)
    except (TypeError, ValueError):
        template_size = FEEDBACK_TEMPLATE_SIZE
    template_size = max(64, min(512, template_size))
    current_image, current_mask, _ = _object_template_view(
        prepared_image,
        current_object_mask,
        size=template_size,
    )
    aligned_part, alignment_score = _align_saved_template(
        object_image,
        (object_mask > 0).astype(np.uint8),
        (part_mask > 0).astype(np.uint8),
        current_image,
        current_mask,
    )
    restored = _restore_object_template_mask(aligned_part, current_object_mask)
    if not restored.any():
        return None, alignment_score
    return restored.astype(np.uint8), alignment_score


def _payload_object_bbox(
    payload: dict[str, object],
    saved_shape: Tuple[int, int],
) -> Tuple[int, int, int, int] | None:
    values = payload.get("object_bbox")
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        saved_h, saved_w = saved_shape
        return clip_box(tuple(float(value) for value in values), saved_w, saved_h)
    except (TypeError, ValueError):
        return None


def _legacy_feedback_object_bbox(
    payload: dict[str, object],
    saved_shape: Tuple[int, int],
) -> Tuple[int, int, int, int] | None:
    image_path_value = str(payload.get("image_path") or "").strip()
    if not image_path_value:
        return None

    image_path = Path(image_path_value)
    session_dir = image_path.parent.parent
    segmentation_dir = session_dir / "segmentation" / image_path.stem
    saved_h, saved_w = saved_shape

    mask_path = segmentation_dir / "input_mask.png"
    saved_mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.exists()
        else None
    )
    if saved_mask is not None:
        if saved_mask.shape[:2] != saved_shape:
            saved_mask = cv2.resize(
                saved_mask,
                (saved_w, saved_h),
                interpolation=cv2.INTER_NEAREST,
            )
        saved_mask = (saved_mask > 0).astype(np.uint8)
        if saved_mask.any():
            reference_image = np.full((saved_h, saved_w, 3), 255, dtype=np.uint8)
            return _feedback_object_bbox(reference_image, saved_mask)

    prepared_path = segmentation_dir / "input_preprocessed.png"
    saved_image = (
        cv2.imread(str(prepared_path), cv2.IMREAD_COLOR)
        if prepared_path.exists()
        else None
    )
    if saved_image is None:
        return None
    if saved_image.shape[:2] != saved_shape:
        saved_image = cv2.resize(saved_image, (saved_w, saved_h), interpolation=cv2.INTER_AREA)
    return _feedback_object_bbox(saved_image, None)


def _load_feedback_box(
    payload: dict,
    key: str,
    current_shape: Tuple[int, int],
    saved_shape: Tuple[int, int],
    current_object_bbox: Tuple[int, int, int, int] | None = None,
) -> Tuple[int, int, int, int] | None:
    bbox_values = payload.get(key)
    if not isinstance(bbox_values, list) or len(bbox_values) != 4:
        return None

    if current_object_bbox is not None:
        relative_key = _feedback_relative_box_key(key)
        relative_box = _load_relative_box(
            payload.get(relative_key),
            current_object_bbox,
            current_shape,
        )
        if relative_box is not None:
            return relative_box

        saved_object_bbox = _payload_object_bbox(payload, saved_shape)
        if saved_object_bbox is None:
            saved_object_bbox = _legacy_feedback_object_bbox(payload, saved_shape)
        if saved_object_bbox is not None:
            try:
                saved_bbox = tuple(int(round(float(value))) for value in bbox_values)
            except (TypeError, ValueError):
                saved_bbox = None
            if saved_bbox is not None:
                relative_values = _normalize_box_to_object(saved_bbox, saved_object_bbox)
                relative_box = _load_relative_box(
                    relative_values,
                    current_object_bbox,
                    current_shape,
                )
                if relative_box is not None:
                    return relative_box

    saved_h = max(1, int(saved_shape[0]))
    saved_w = max(1, int(saved_shape[1]))
    current_h, current_w = current_shape
    scale_x = current_w / float(saved_w)
    scale_y = current_h / float(saved_h)

    x1 = float(bbox_values[0]) * scale_x
    y1 = float(bbox_values[1]) * scale_y
    x2 = float(bbox_values[2]) * scale_x
    y2 = float(bbox_values[3]) * scale_y
    return clip_box((x1, y1, x2, y2), current_w, current_h)


def load_manual_feedback(
    image_path: Path,
    feedback_dir: Path,
    current_shape: Tuple[int, int],
    prepared_image: np.ndarray,
    prepared_mask: np.ndarray | None = None,
) -> ManualPartFeedback | None:
    if not feedback_dir.exists():
        return None

    current_signature = build_feedback_signature(prepared_image, prepared_mask)
    current_object_bbox = _feedback_object_bbox(prepared_image, prepared_mask)
    matched = _find_feedback_match(image_path, feedback_dir, current_signature)
    if matched is None:
        return None

    feedback_path, payload, match_type, match_score = matched
    return _feedback_from_payload(
        payload,
        current_shape,
        current_object_bbox,
        prepared_image,
        prepared_mask,
        feedback_path,
        match_type,
        match_score,
    )


def _resolve_feedback_storage_path(
    image_path: Path,
    feedback_dir: Path,
    signature: FeedbackSignature | None,
) -> Path:
    current_resolved = str(image_path.resolve()).lower()
    for feedback_path, payload in _iter_feedback_entries(feedback_dir):
        saved_signature = _signature_from_payload(payload)
        if signature is not None and saved_signature is not None and saved_signature.exact_hash == signature.exact_hash:
            return feedback_path
        payload_image_path = str(payload.get("image_path") or "").strip().lower()
        if signature is None and payload_image_path and payload_image_path == current_resolved:
            return feedback_path
    return feedback_file_path(image_path, feedback_dir, signature)


def _save_feedback_box(
    image_path: Path,
    feedback_dir: Path,
    part_key: str,
    bbox: Tuple[int, int, int, int],
    working_shape: Tuple[int, int],
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
    jewel_type: str | None = None,
) -> Path:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    current_h, current_w = working_shape
    clipped = clip_box(bbox, current_w, current_h)
    signature = build_feedback_signature(prepared_image, prepared_mask) if prepared_image is not None else None
    object_bbox = (
        _feedback_object_bbox(prepared_image, prepared_mask)
        if prepared_image is not None
        else None
    )
    feedback_path = _resolve_feedback_storage_path(image_path, feedback_dir, signature)
    payload: dict[str, object] = {}
    if feedback_path.exists():
        try:
            with open(feedback_path, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
            if isinstance(existing, dict):
                payload.update(existing)
        except (json.JSONDecodeError, OSError):
            payload = {}

    payload["image_path"] = str(image_path.resolve())
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if jewel_type:
        payload["jewel_type"] = str(jewel_type).strip()
    if signature is not None:
        payload["signature"] = signature.to_payload()
    payload[part_key] = list(clipped)
    if object_bbox is not None:
        payload["feedback_version"] = 3
        payload["object_bbox"] = list(object_bbox)
        payload[_feedback_relative_box_key(part_key)] = _normalize_box_to_object(
            clipped,
            object_bbox,
        )
    if prepared_image is not None and object_bbox is not None:
        object_mask = _infer_signature_mask(prepared_image, prepared_mask)
        correction_mask = build_feedback_support_mask(
            object_mask,
            clipped,
            close_ksize=3,
            min_component=max(4, object_mask.size // 80000),
        )
        template_image, template_object_mask, template_part_mask = _object_template_view(
            prepared_image,
            object_mask,
            correction_mask,
            size=FEEDBACK_TEMPLATE_SIZE,
        )
        if template_part_mask is not None and template_part_mask.any():
            payload["template_size"] = FEEDBACK_TEMPLATE_SIZE
            payload["object_template_png"] = _encode_png_base64(template_image)
            payload["object_template_mask_png"] = _encode_png_base64(
                template_object_mask * 255
            )
            payload[_feedback_template_key(part_key)] = _encode_png_base64(
                template_part_mask * 255
            )
    if part_key == "pendant_bbox":
        payload.pop("no_pendant", None)
    elif part_key == "tassel_bbox":
        payload.pop("no_tassel", None)
    payload["working_size"] = [int(current_h), int(current_w)]
    with open(feedback_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return feedback_path


def save_pendant_feedback(
    image_path: Path,
    feedback_dir: Path,
    pendant_bbox: Tuple[int, int, int, int],
    working_shape: Tuple[int, int],
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
    jewel_type: str | None = None,
) -> Path:
    return _save_feedback_box(
        image_path,
        feedback_dir,
        "pendant_bbox",
        pendant_bbox,
        working_shape,
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
        jewel_type=jewel_type,
    )


def save_tassel_feedback(
    image_path: Path,
    feedback_dir: Path,
    tassel_bbox: Tuple[int, int, int, int],
    working_shape: Tuple[int, int],
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
    jewel_type: str | None = None,
) -> Path:
    return _save_feedback_box(
        image_path,
        feedback_dir,
        "tassel_bbox",
        tassel_bbox,
        working_shape,
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
        jewel_type=jewel_type,
    )


def _save_no_part_feedback(
    image_path: Path,
    feedback_dir: Path,
    part_key: str,
    working_shape: Tuple[int, int],
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
    jewel_type: str | None = None,
) -> Path:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    current_h, current_w = working_shape
    signature = build_feedback_signature(prepared_image, prepared_mask) if prepared_image is not None else None
    object_bbox = (
        _feedback_object_bbox(prepared_image, prepared_mask)
        if prepared_image is not None
        else None
    )
    feedback_path = _resolve_feedback_storage_path(image_path, feedback_dir, signature)
    payload: dict[str, object] = {}
    if feedback_path.exists():
        try:
            with open(feedback_path, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
            if isinstance(existing, dict):
                payload.update(existing)
        except (json.JSONDecodeError, OSError):
            payload = {}

    payload["image_path"] = str(image_path.resolve())
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if jewel_type:
        payload["jewel_type"] = str(jewel_type).strip()
    if signature is not None:
        payload["signature"] = signature.to_payload()
    if object_bbox is not None:
        payload["feedback_version"] = 3
        payload["object_bbox"] = list(object_bbox)
    payload[part_key] = True
    if part_key == "no_pendant":
        payload.pop("pendant_bbox", None)
        payload.pop(_feedback_relative_box_key("pendant_bbox"), None)
        payload.pop(_feedback_template_key("pendant_bbox"), None)
    elif part_key == "no_tassel":
        payload.pop("tassel_bbox", None)
        payload.pop(_feedback_relative_box_key("tassel_bbox"), None)
        payload.pop(_feedback_template_key("tassel_bbox"), None)
    payload["working_size"] = [int(current_h), int(current_w)]
    with open(feedback_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return feedback_path


def save_no_pendant_feedback(
    image_path: Path,
    feedback_dir: Path,
    working_shape: Tuple[int, int],
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
    jewel_type: str | None = None,
) -> Path:
    return _save_no_part_feedback(
        image_path,
        feedback_dir,
        "no_pendant",
        working_shape,
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
        jewel_type=jewel_type,
    )


def save_no_tassel_feedback(
    image_path: Path,
    feedback_dir: Path,
    working_shape: Tuple[int, int],
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
    jewel_type: str | None = None,
) -> Path:
    return _save_no_part_feedback(
        image_path,
        feedback_dir,
        "no_tassel",
        working_shape,
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
        jewel_type=jewel_type,
    )


def _find_feedback_path_for_clear(
    image_path: Path,
    feedback_dir: Path,
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
) -> Path | None:
    signature = build_feedback_signature(prepared_image, prepared_mask) if prepared_image is not None else None
    current_resolved = str(image_path.resolve()).lower()
    for feedback_path, payload in _iter_feedback_entries(feedback_dir):
        saved_signature = _signature_from_payload(payload)
        if signature is not None and saved_signature is not None and saved_signature.exact_hash == signature.exact_hash:
            return feedback_path
        payload_image_path = str(payload.get("image_path") or "").strip().lower()
        if signature is None and payload_image_path and payload_image_path == current_resolved:
            return feedback_path
    return None


def _clear_feedback_box(
    image_path: Path,
    feedback_dir: Path,
    part_key: str,
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
) -> bool:
    feedback_path = _find_feedback_path_for_clear(
        image_path,
        feedback_dir,
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
    )
    if feedback_path is None or not feedback_path.exists():
        return False

    try:
        with open(feedback_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (json.JSONDecodeError, OSError):
        feedback_path.unlink()
        return True

    if not isinstance(payload, dict) or part_key not in payload:
        return False

    payload.pop(part_key, None)
    if part_key in {"pendant_bbox", "tassel_bbox"}:
        payload.pop(_feedback_relative_box_key(part_key), None)
        payload.pop(_feedback_template_key(part_key), None)
    if (
        "pendant_bbox" not in payload
        and "tassel_bbox" not in payload
        and "no_pendant" not in payload
        and "no_tassel" not in payload
    ):
        feedback_path.unlink()
        return True

    with open(feedback_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return True


def clear_pendant_feedback(
    image_path: Path,
    feedback_dir: Path,
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
) -> bool:
    return _clear_feedback_box(
        image_path,
        feedback_dir,
        "pendant_bbox",
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
    )


def clear_tassel_feedback(
    image_path: Path,
    feedback_dir: Path,
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
) -> bool:
    return _clear_feedback_box(
        image_path,
        feedback_dir,
        "tassel_bbox",
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
    )


def clear_no_pendant_feedback(
    image_path: Path,
    feedback_dir: Path,
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
) -> bool:
    return _clear_feedback_box(
        image_path,
        feedback_dir,
        "no_pendant",
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
    )


def clear_no_tassel_feedback(
    image_path: Path,
    feedback_dir: Path,
    prepared_image: np.ndarray | None = None,
    prepared_mask: np.ndarray | None = None,
) -> bool:
    return _clear_feedback_box(
        image_path,
        feedback_dir,
        "no_tassel",
        prepared_image=prepared_image,
        prepared_mask=prepared_mask,
    )


def _load_feedback_prepared_source(
    payload: dict[str, object],
) -> tuple[Path, PreparedInput] | None:
    image_path_value = str(payload.get("image_path") or "").strip()
    if not image_path_value:
        return None
    image_path = Path(image_path_value)

    session_dir = image_path.parent.parent
    segmentation_dir = session_dir / "segmentation" / image_path.stem
    prepared_path = segmentation_dir / "input_preprocessed.png"
    mask_path = segmentation_dir / "input_mask.png"
    prepared_image = (
        cv2.imread(str(prepared_path), cv2.IMREAD_COLOR)
        if prepared_path.exists()
        else None
    )
    prepared_mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.exists()
        else None
    )
    if prepared_image is not None:
        if prepared_mask is None:
            gray = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2GRAY)
            prepared_mask = (gray < 250).astype(np.uint8)
        elif prepared_mask.shape[:2] != prepared_image.shape[:2]:
            prepared_mask = cv2.resize(
                prepared_mask,
                (prepared_image.shape[1], prepared_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return image_path, PreparedInput(
            original_image=prepared_image,
            working_image=prepared_image,
            working_mask=(prepared_mask > 0).astype(np.uint8),
        )

    if not image_path.exists():
        return None
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw_image is None:
        return None
    original_image, alpha_mask = composite_to_bgr(raw_image)
    size_values = payload.get("working_size")
    saved_shape = None
    if isinstance(size_values, list) and len(size_values) == 2:
        try:
            saved_shape = (max(1, int(size_values[0])), max(1, int(size_values[1])))
        except (TypeError, ValueError):
            saved_shape = None

    centered = prepare_segmentation_input(raw_image)
    if saved_shape is not None and centered.working_image.shape[:2] == saved_shape:
        return image_path, centered

    if saved_shape is not None and original_image.shape[:2] != saved_shape:
        original_image = cv2.resize(
            original_image,
            (saved_shape[1], saved_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        if alpha_mask is not None:
            alpha_mask = cv2.resize(
                alpha_mask,
                (saved_shape[1], saved_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    working_mask = _infer_signature_mask(original_image, alpha_mask)
    return image_path, PreparedInput(
        original_image=original_image,
        working_image=original_image,
        working_mask=working_mask,
    )


def _upgrade_feedback_payload(
    payload: dict[str, object],
    prepared: PreparedInput,
) -> dict[str, object]:
    upgraded = dict(payload)
    image = prepared.working_image
    object_mask = _infer_signature_mask(image, prepared.working_mask)
    object_bbox = bounding_box(object_mask)
    current_shape = image.shape[:2]
    size_values = payload.get("working_size")
    if isinstance(size_values, list) and len(size_values) == 2:
        saved_shape = (max(1, int(size_values[0])), max(1, int(size_values[1])))
    else:
        saved_shape = current_shape

    signature = build_feedback_signature(image, object_mask)
    upgraded["signature"] = signature.to_payload()
    upgraded["feedback_version"] = 3
    upgraded["object_bbox"] = list(object_bbox)
    upgraded["working_size"] = [int(current_shape[0]), int(current_shape[1])]

    template_image, template_object_mask, _ = _object_template_view(
        image,
        object_mask,
        size=FEEDBACK_TEMPLATE_SIZE,
    )
    upgraded["template_size"] = FEEDBACK_TEMPLATE_SIZE
    upgraded["object_template_png"] = _encode_png_base64(template_image)
    upgraded["object_template_mask_png"] = _encode_png_base64(
        template_object_mask * 255
    )

    for part_key in ("pendant_bbox", "tassel_bbox"):
        bbox = _load_feedback_box(
            payload,
            part_key,
            current_shape,
            saved_shape,
            object_bbox,
        )
        if bbox is None:
            continue
        part_mask = build_feedback_support_mask(
            object_mask,
            bbox,
            close_ksize=3,
            min_component=max(4, object_mask.size // 80000),
        )
        _, _, template_part_mask = _object_template_view(
            image,
            object_mask,
            part_mask,
            size=FEEDBACK_TEMPLATE_SIZE,
        )
        upgraded[part_key] = list(bbox)
        upgraded[_feedback_relative_box_key(part_key)] = _normalize_box_to_object(
            bbox,
            object_bbox,
        )
        if template_part_mask is not None and template_part_mask.any():
            upgraded[_feedback_template_key(part_key)] = _encode_png_base64(
                template_part_mask * 255
            )

    return upgraded


def upgrade_feedback_library(feedback_dir: Path) -> int:
    """Upgrade reachable legacy feedback to position-invariant version-3 templates."""
    upgraded_count = 0
    for feedback_path, payload in _iter_feedback_entries(feedback_dir):
        try:
            feedback_version = int(payload.get("feedback_version") or 0)
        except (TypeError, ValueError):
            feedback_version = 0
        needs_upgrade = feedback_version < 3
        if not needs_upgrade:
            for part_key in ("pendant_bbox", "tassel_bbox"):
                if part_key in payload and _feedback_template_key(part_key) not in payload:
                    needs_upgrade = True
                    break
        if not needs_upgrade:
            continue

        source = _load_feedback_prepared_source(payload)
        if source is None:
            continue
        image_path, prepared = source
        upgraded = _upgrade_feedback_payload(payload, prepared)
        upgraded["image_path"] = str(image_path.resolve())
        upgraded.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
        with open(feedback_path, "w", encoding="utf-8") as fp:
            json.dump(upgraded, fp, indent=2)
        upgraded_count += 1
    return upgraded_count


def migrate_feedback_library(source_dirs: Iterable[Path], target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    migrated = 0

    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            continue

        for _, payload in _iter_feedback_entries(source_path):
            image_path_value = str(payload.get("image_path") or "").strip()
            if not image_path_value:
                continue

            image_path = Path(image_path_value)
            if not image_path.exists():
                continue

            raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if raw_image is None:
                continue

            prepared = prepare_segmentation_input(raw_image)
            signature = build_feedback_signature(prepared.working_image, prepared.working_mask)
            target_path = _resolve_feedback_storage_path(image_path, target_dir, signature)

            merged_payload: dict[str, object] = {}
            existing_payload = _read_feedback_payload(target_path)
            if existing_payload is not None:
                merged_payload.update(existing_payload)

            merged_payload.update(payload)
            merged_payload["image_path"] = str(image_path.resolve())
            merged_payload = _upgrade_feedback_payload(merged_payload, prepared)
            merged_payload.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))

            with open(target_path, "w", encoding="utf-8") as fp:
                json.dump(merged_payload, fp, indent=2)
            migrated += 1

    return migrated


def rectangle_mask(shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = clip_box(bbox, w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 1
    return cleaned


def component_touches_border(component_mask: np.ndarray) -> bool:
    return bool(
        component_mask[0, :].any()
        or component_mask[-1, :].any()
        or component_mask[:, 0].any()
        or component_mask[:, -1].any()
    )


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def largest_or_required_components(mask: np.ndarray, required_mask: np.ndarray | None = None) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num_labels):
        component = (labels == idx).astype(np.uint8)
        if idx == largest_idx or (required_mask is not None and np.any(component & required_mask)):
            keep |= component
    return keep.astype(np.uint8)


def keep_primary_jewelry_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        8,
    )
    if num_labels <= 1:
        return mask.astype(np.uint8)

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    total_area = int(stats[1:, cv2.CC_STAT_AREA].sum())
    keep = np.zeros_like(mask, dtype=np.uint8)

    for idx in range(1, num_labels):
        component = (labels == idx).astype(np.uint8)
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if idx == largest_idx:
            keep |= component
            continue
        if component_touches_border(component):
            continue

        area_ratio_to_largest = area / float(max(1, largest_area))
        area_ratio_to_total = area / float(max(1, total_area))
        if area >= 60 and (
            area_ratio_to_largest >= 0.003
            or area_ratio_to_total >= 0.002
        ):
            keep |= component

    return keep.astype(np.uint8) if keep.any() else largest_component(mask)


def dilate(mask: np.ndarray, ksize: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def erode(mask: np.ndarray, ksize: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def close(mask: np.ndarray, ksize: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def open_mask(mask: np.ndarray, ksize: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 1, 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def contour_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def compute_solidity(mask: np.ndarray) -> float:
    """Compute solidity = contour_area / convex_hull_area (how 'filled' the shape is)."""
    contours = contour_from_mask(mask)
    if not contours:
        return 1.0
    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return 1.0
    return contour_area / hull_area


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _softmax_last(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / (np.sum(exp_values, axis=-1, keepdims=True) + 1e-9)


def _nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thres: float,
    max_detections: int = 100,
) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    while order.size and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break

        remaining = order[1:]
        intersection_w = np.maximum(
            0.0,
            np.minimum(x2[index], x2[remaining]) - np.maximum(x1[index], x1[remaining]),
        )
        intersection_h = np.maximum(
            0.0,
            np.minimum(y2[index], y2[remaining]) - np.maximum(y1[index], y1[remaining]),
        )
        intersection = intersection_w * intersection_h
        union = areas[index] + areas[remaining] - intersection + 1e-6
        order = remaining[(intersection / union) <= iou_thres]

    return np.asarray(keep, dtype=np.int32)


def _hailo_output_hwc(output: np.ndarray) -> np.ndarray:
    tensor = np.asarray(output)
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor[:, :, None]
    if (
        tensor.ndim == 3
        and tensor.shape[1] == tensor.shape[2]
        and tensor.shape[0] != tensor.shape[1]
    ):
        tensor = np.transpose(tensor, (1, 2, 0))
    if tensor.ndim != 3 or tensor.shape[0] != tensor.shape[1]:
        raise RuntimeError(f"Unsupported FastSAM Hailo output shape: {tensor.shape}")
    return tensor.astype(np.float32, copy=False)


def _decode_hailo_fastsam(
    raw_outputs: Dict[str, np.ndarray],
    input_size: int,
    conf_thres: float,
    iou_thres: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = [(name, _hailo_output_hwc(output)) for name, output in raw_outputs.items()]
    if len(tensors) != 10:
        shapes = {name: tuple(tensor.shape) for name, tensor in tensors}
        raise RuntimeError(
            "FastSAM HEF must expose 10 raw YOLOv8-seg outputs; "
            f"received {len(tensors)}: {shapes}"
        )

    proto_name, protos = max(
        tensors,
        key=lambda item: (item[1].shape[0] * item[1].shape[1], -abs(item[1].shape[2] - 32)),
    )
    proto_channels = int(protos.shape[2])
    grouped: Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]] = {}
    for name, tensor in tensors:
        if name == proto_name:
            continue
        grouped.setdefault(tensor.shape[:2], []).append((name, tensor))

    scale_shapes = sorted(grouped)
    if len(scale_shapes) != 3:
        raise RuntimeError(f"Unexpected FastSAM feature-map groups: {scale_shapes}")

    raw_boxes: List[np.ndarray] = []
    raw_scores: List[np.ndarray] = []
    raw_coeffs: List[np.ndarray] = []
    strides: List[int] = []

    for height, width in scale_shapes:
        scale_tensors = grouped[(height, width)]
        bbox_candidates = [
            item
            for item in scale_tensors
            if item[1].shape[2] > proto_channels and item[1].shape[2] % 4 == 0
        ]
        if len(scale_tensors) != 3 or not bbox_candidates:
            raise RuntimeError(
                f"Unexpected FastSAM outputs at {height}x{width}: "
                f"{[(name, tensor.shape) for name, tensor in scale_tensors]}"
            )

        bbox_name, bbox_tensor = max(bbox_candidates, key=lambda item: item[1].shape[2])
        remaining = [item for item in scale_tensors if item[0] != bbox_name]
        coeff_candidates = [item for item in remaining if item[1].shape[2] == proto_channels]
        if len(coeff_candidates) != 1:
            raise RuntimeError(f"Could not identify FastSAM mask coefficients at {height}x{width}")
        coeff_name, coeff_tensor = coeff_candidates[0]
        score_tensors = [item for item in remaining if item[0] != coeff_name]
        if len(score_tensors) != 1:
            raise RuntimeError(f"Could not identify FastSAM scores at {height}x{width}")

        raw_boxes.append(bbox_tensor)
        raw_scores.append(score_tensors[0][1])
        raw_coeffs.append(coeff_tensor)
        strides.append(int(round(input_size / height)))

    decoded_boxes: List[np.ndarray] = []
    for box_tensor, stride in zip(raw_boxes, strides):
        height, width, channels = box_tensor.shape
        regression_length = channels // 4
        box_distribution = box_tensor.reshape(1, height * width, 4, regression_length)
        distances = np.sum(
            _softmax_last(box_distribution)
            * np.arange(regression_length, dtype=np.float32).reshape(1, 1, 1, -1),
            axis=-1,
        ) * stride

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32) + 0.5,
            np.arange(height, dtype=np.float32) + 0.5,
        )
        centers = np.stack(
            (grid_x.reshape(-1), grid_y.reshape(-1), grid_x.reshape(-1), grid_y.reshape(-1)),
            axis=1,
        ) * stride
        xyxy = centers[None] + np.concatenate((-distances[:, :, :2], distances[:, :, 2:]), axis=-1)
        decoded_boxes.append(xyxy)

    boxes = np.concatenate(decoded_boxes, axis=1)[0]
    scores_by_class = np.concatenate(
        [tensor.reshape(1, -1, tensor.shape[2]) for tensor in raw_scores],
        axis=1,
    )[0]
    if scores_by_class.size and (
        float(np.min(scores_by_class)) < 0.0 or float(np.max(scores_by_class)) > 1.0
    ):
        scores_by_class = _sigmoid_array(scores_by_class)
    scores = np.max(scores_by_class, axis=1)
    coefficients = np.concatenate(
        [tensor.reshape(1, -1, proto_channels) for tensor in raw_coeffs],
        axis=1,
    )[0]

    candidate_indices = np.flatnonzero(scores > conf_thres)
    if candidate_indices.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, input_size, input_size), dtype=np.float32),
        )

    candidate_boxes = boxes[candidate_indices]
    candidate_scores = scores[candidate_indices]
    nms_indices = _nms_xyxy(candidate_boxes, candidate_scores, iou_thres)
    selected = candidate_indices[nms_indices]
    selected_boxes = boxes[selected].astype(np.float32)
    selected_scores = scores[selected].astype(np.float32)

    proto_flat = protos.reshape(-1, proto_channels).T
    masks = _sigmoid_array(coefficients[selected] @ proto_flat).reshape(
        -1,
        protos.shape[0],
        protos.shape[1],
    )
    full_masks = np.empty((len(selected), input_size, input_size), dtype=np.float32)
    for index, (mask, box) in enumerate(zip(masks, selected_boxes)):
        resized = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = np.clip(np.ceil(box).astype(int), 0, input_size)
        cropped = np.zeros_like(resized)
        if x2 > x1 and y2 > y1:
            cropped[y1:y2, x1:x2] = resized[y1:y2, x1:x2]
        full_masks[index] = cropped

    return selected_boxes, selected_scores, full_masks


class FastSamOnnx:
    def __init__(self, model_path: Path, providers: List[str], input_size: int = 640, hailo_model: Any = None) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.hailo_model = hailo_model
        
        if self.hailo_model is None:
            print(f"Using ONNX model: {model_path}")
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        else:
            print(f"Using Hailo HEF model via shared runtime: {self.hailo_model.hef_path}")

    def infer(
        self,
        image: np.ndarray,
        conf_thres: float,
        iou_thres: float,
        mask_thres: float,
    ) -> List[DetectionMask]:
        original_h, original_w = image.shape[:2]
        padded, scale, (pad_x, pad_y) = letterbox(image, self.input_size)
        # Ensure the blob is C-contiguous as transpose/slice operations can make it non-contiguous,
        # which the Hailo runtime requires for the input buffer.
        blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        if self.hailo_model is not None:
            raw_outputs = self.hailo_model.run_inference(blob[None])
            if not isinstance(raw_outputs, dict):
                raise RuntimeError("FastSAM Hailo inference did not return named output tensors.")
            boxes, scores, masks = _decode_hailo_fastsam(
                raw_outputs,
                self.input_size,
                conf_thres,
                iou_thres,
            )
        else:
            outputs = self.session.run(None, {self.input_name: blob[None]})
            predictions = outputs[0][0].T
            protos = outputs[1][0]

            keep = predictions[:, 4] > conf_thres
            predictions = predictions[keep]
            if len(predictions) == 0:
                return []

            boxes = xywh_to_xyxy(predictions[:, :4])
            scores = predictions[:, 4]
            mask_coeffs = predictions[:, 5:]
            indices = _nms_xyxy(boxes, scores, iou_thres)
            boxes = boxes[indices]
            scores = scores[indices]
            mask_coeffs = mask_coeffs[indices]

            proto_h, proto_w = protos.shape[1:]
            masks = _sigmoid_array(mask_coeffs @ protos.reshape(protos.shape[0], -1))
            masks = masks.reshape(-1, proto_h, proto_w)

        detections: List[DetectionMask] = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for score, box, proto_mask in zip(scores, boxes, masks):
            resized_mask = cv2.resize(proto_mask, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            cropped = resized_mask[pad_y : pad_y + int(round(original_h * scale)), pad_x : pad_x + int(round(original_w * scale))]
            full_mask = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            binary = (full_mask >= mask_thres).astype(np.uint8)
            area = int(binary.sum())
            if area == 0:
                continue

            ys, xs = np.where(binary > 0)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox_w = max(1, x2 - x1 + 1)
            bbox_h = max(1, y2 - y1 + 1)
            fill_ratio = area / float(bbox_w * bbox_h)
            aspect_ratio = bbox_w / float(bbox_h)

            hue = hsv[:, :, 0][binary > 0]
            sat = hsv[:, :, 1][binary > 0]
            red_ratio = float((((hue < 10) | (hue > 170)) & (sat > 60)).mean())
            gold_ratio = float((((hue > 8) & (hue < 40) & (sat > 40)).mean()))

            detections.append(
                DetectionMask(
                    score=float(score),
                    mask=binary,
                    bbox=(x1, y1, x2, y2),
                    centroid=(float(xs.mean()), float(ys.mean())),
                    area=area,
                    fill_ratio=fill_ratio,
                    aspect_ratio=aspect_ratio,
                    red_ratio=red_ratio,
                    gold_ratio=gold_ratio,
                )
            )
        return detections


def composite_to_bgr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        composite = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
        alpha_mask = (image[:, :, 3] > 0).astype(np.uint8)
        return composite, alpha_mask
    return image[:, :, :3].copy(), None


def extract_primary_mask_from_otsu(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return remove_small_components((otsu > 0).astype(np.uint8), max(80, (h * w) // 12000))

    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    min_area = max(120.0, float((h * w) // 8000))
    small_hole_area = max(24.0, float((h * w) // 12000))
    best_score = -1.0
    best_mask = np.zeros((h, w), dtype=np.uint8)

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if hierarchy[0][idx][3] != -1:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)

        child = hierarchy[0][idx][2]
        while child != -1:
            child_area = cv2.contourArea(contours[child])
            if child_area > small_hole_area:
                cv2.drawContours(mask, [contours[child]], -1, 0, -1)
            child = hierarchy[0][child][0]

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        comp_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
        center_dist = float(np.linalg.norm((comp_center - center) / np.array([max(1.0, w), max(1.0, h)], dtype=np.float32)))
        bbox_w = int(xs.max()) - int(xs.min()) + 1
        bbox_h = int(ys.max()) - int(ys.min()) + 1
        fill_ratio = float(mask.sum()) / float(max(1, bbox_w * bbox_h))
        score = float(mask.sum()) * (1.0 + 0.35 * fill_ratio) / (1.0 + 3.5 * center_dist)

        if score > best_score:
            best_score = score
            best_mask = mask

    if not best_mask.any():
        best_mask = (otsu > 0).astype(np.uint8)

    best_mask = close(best_mask, 5)
    best_mask = remove_small_components(best_mask, max(80, (h * w) // 12000))
    if best_mask.any():
        best_mask = largest_component(best_mask)
    return best_mask.astype(np.uint8)


def remove_background_like_holes(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Cut uniform bed-colored islands back out after morphological closing."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_width = max(2, min(12, min(binary.shape) // 20))
    border = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    center = np.median(border, axis=0)
    mad = np.median(np.abs(border - center), axis=0)
    tolerance = np.array(
        [
            max(4.0, min(14.0, 3.0 * float(mad[0]) + 3.0)),
            max(3.0, min(10.0, 3.0 * float(mad[1]) + 2.0)),
            max(3.0, min(10.0, 3.0 * float(mad[2]) + 2.0)),
        ],
        dtype=np.float32,
    )
    background_match = np.all(
        np.abs(lab - center.reshape(1, 1, 3)) <= tolerance,
        axis=2,
    )
    candidates = (background_match & (binary > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates,
        connectivity=8,
    )
    refined = binary.copy()
    jewel_area = int(binary.sum())
    max_area = max(8, int(round(jewel_area * 0.20)))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > max_area:
            continue
        component = (labels == label).astype(np.uint8)
        ring = cv2.subtract(
            cv2.dilate(component, ring_kernel, iterations=1),
            component,
        )
        ring_area = int(ring.sum())
        if ring_area <= 0:
            continue
        ring_support = int((ring & binary).sum()) / float(ring_area)
        if ring_support < 0.70:
            continue
        pixels = lab[component > 0]
        if (
            float(np.std(pixels[:, 0])) > 11.0
            or float(np.std(pixels[:, 1])) > 6.0
            or float(np.std(pixels[:, 2])) > 6.0
        ):
            continue
        refined[component > 0] = 0
    return refined.astype(np.uint8)


def center_primary_jewel(image: np.ndarray, mask: np.ndarray, pad: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    if not mask.any():
        return image.copy(), np.ones(image.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = bounding_box(mask)
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    crop_img = image[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    crop_h, crop_w = crop_img.shape[:2]
    canvas_side = max(320, max(crop_h, crop_w) + 2 * pad)

    canvas = np.full((canvas_side, canvas_side, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((canvas_side, canvas_side), dtype=np.uint8)

    offset_y = (canvas_side - crop_h) // 2
    offset_x = (canvas_side - crop_w) // 2

    paste = canvas[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    paste_mask = canvas_mask[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
    paste[crop_mask > 0] = crop_img[crop_mask > 0]
    paste_mask[crop_mask > 0] = 1
    return canvas, canvas_mask


def prepare_segmentation_input(
    raw_image: np.ndarray,
    external_mask: np.ndarray | None = None,
) -> PreparedInput:
    original_image, alpha_mask = composite_to_bgr(raw_image)
    if external_mask is not None:
        primary_mask = np.asarray(external_mask).astype(np.uint8)
        if primary_mask.ndim == 3:
            primary_mask = primary_mask.squeeze()
        if primary_mask.shape[:2] != original_image.shape[:2]:
            primary_mask = cv2.resize(
                primary_mask,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        primary_mask = (primary_mask > 0).astype(np.uint8)
    else:
        primary_mask = extract_primary_mask_from_otsu(original_image)

    if alpha_mask is not None and alpha_mask.any():
        alpha_mask = close(alpha_mask.astype(np.uint8), 3)
        alpha_mask = remove_small_components(alpha_mask, max(20, alpha_mask.size // 30000))
        if alpha_mask.any():
            primary_mask = alpha_mask

    if not primary_mask.any():
        fallback_mask, _ = estimate_background_mask(original_image)
        primary_mask = fallback_mask

    primary_mask = close(primary_mask.astype(np.uint8), 5)
    primary_mask = remove_background_like_holes(original_image, primary_mask)
    primary_mask = remove_small_components(primary_mask, max(80, primary_mask.size // 15000))
    if primary_mask.any():
        primary_mask = keep_primary_jewelry_components(primary_mask)

    working_image, working_mask = center_primary_jewel(original_image, primary_mask)
    return PreparedInput(
        original_image=original_image,
        working_image=working_image,
        working_mask=working_mask.astype(np.uint8),
    )


def estimate_background_mask(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = image.shape[:2]

    border = np.concatenate(
        [
            lab[:20, :, :].reshape(-1, 3),
            lab[-20:, :, :].reshape(-1, 3),
            lab[:, :20, :].reshape(-1, 3),
            lab[:, -20:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_lab = np.median(border, axis=0).astype(np.float32)
    lab_diff = np.linalg.norm(lab.astype(np.float32) - bg_lab[None, None, :], axis=2)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red_mask = (
        (((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) & (saturation > 70) & (value > 40))
    ).astype(np.uint8)
    gold_mask = (
        ((hsv[:, :, 0] > 8) & (hsv[:, :, 0] < 40) & (saturation > 35) & (value > 40))
    ).astype(np.uint8)
    green_mask = (
        ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 100) & (saturation > 70) & (value > 35))
    ).astype(np.uint8)
    saturated_thread_mask = (
        (saturation > 85)
        & (value > 30)
        & (
            (red_mask > 0)
            | (green_mask > 0)
            | ((hsv[:, :, 0] >= 125) & (hsv[:, :, 0] <= 170))
        )
    ).astype(np.uint8)

    color_foreground = (
        (lab_diff > 18)
        | ((saturation > 35) & (value < 250))
        | (red_mask > 0)
        | (green_mask > 0)
    ).astype(np.uint8)
    color_foreground = open_mask(color_foreground, 3)
    color_foreground = close(color_foreground, 5)
    color_foreground = remove_small_components(color_foreground, max(80, (h * w) // 12000))

    return color_foreground, {
        "red_mask": red_mask,
        "gold_mask": gold_mask,
        "green_mask": green_mask,
        "textile_mask": saturated_thread_mask,
        "lab_diff": lab_diff,
    }


def build_proposal_union(
    detections: List[DetectionMask],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, List[DetectionMask]]:
    h, w = shape
    union = np.zeros((h, w), dtype=np.uint8)
    selected: List[DetectionMask] = []
    max_area = h * w

    for det in detections:
        area_ratio = det.area / float(max_area)
        jewelry_score = max(det.gold_ratio, det.red_ratio * 1.7)
        if det.area < 100:
            continue
        if area_ratio > 0.22:
            continue
        if jewelry_score < 0.42 and det.score < 0.55:
            continue
        union |= det.mask.astype(np.uint8)
        selected.append(det)

    union = close(union, 7)
    union = dilate(union, 5)
    union = remove_small_components(union, max(100, (h * w) // 10000))
    return union, selected


def refine_necklace_mask(
    image: np.ndarray,
    color_mask: np.ndarray,
    proposal_mask: np.ndarray,
    color_maps: Dict[str, np.ndarray],
) -> np.ndarray:
    h, w = image.shape[:2]
    gold_mask = color_maps["gold_mask"]
    red_mask = color_maps["red_mask"]
    textile_mask = color_maps.get("textile_mask", red_mask)

    strong_color = (
        (gold_mask > 0)
        | (red_mask > 0)
        | (textile_mask > 0)
    ).astype(np.uint8)
    strong_color = dilate(strong_color, 3)

    if proposal_mask.any():
        necklace = ((color_mask > 0) & ((dilate(proposal_mask, 11) > 0) | (strong_color > 0))).astype(np.uint8)
    else:
        necklace = ((color_mask > 0) & (strong_color > 0)).astype(np.uint8)

    necklace |= strong_color
    necklace = close(necklace, 5)
    necklace = remove_small_components(necklace, max(100, (h * w) // 12000))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(necklace, 8)
    cleaned = np.zeros_like(necklace, dtype=np.uint8)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        component = (labels == idx).astype(np.uint8)
        if area < max(100, (h * w) // 12000):
            continue
        if component_touches_border(component) and area > (h * w) * 0.04:
            overlap = float((component & dilate(proposal_mask, 5)).sum()) / float(max(1, component.sum()))
            if overlap < 0.08:
                continue
        cleaned |= component

    cleaned = close(cleaned, 3)
    cleaned = remove_small_components(cleaned, max(120, (h * w) // 12000))
    return cleaned


def is_valid_thread(mask: np.ndarray, necklace_mask: np.ndarray) -> bool:
    """Check whether a region can be a cord or fan-shaped tassel."""
    if not mask.any():
        return False

    area = int(mask.sum())
    necklace_area = int(necklace_mask.sum())

    area_ratio = area / max(1, necklace_area)
    if area_ratio < 0.004 or area_ratio > 0.52:
        return False

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False

    bbox_w = int(xs.max()) - int(xs.min()) + 1
    bbox_h = int(ys.max()) - int(ys.min()) + 1
    fill_ratio = area / max(1, bbox_w * bbox_h)
    solidity = compute_solidity(mask)

    # Threads are sparse or concave. A broad fan tassel may be wider than tall,
    # so aspect ratio alone is deliberately not used as a rejection criterion.
    if fill_ratio > 0.72 and solidity > 0.88:
        return False

    necklace_x1, necklace_y1, necklace_x2, necklace_y2 = bounding_box(necklace_mask)
    range_x = max(1.0, float(necklace_x2 - necklace_x1))
    range_y = max(1.0, float(necklace_y2 - necklace_y1))
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    x_edge = min(
        abs(center_x - necklace_x1),
        abs(necklace_x2 - center_x),
    ) / range_x
    y_edge = min(
        abs(center_y - necklace_y1),
        abs(necklace_y2 - center_y),
    ) / range_y
    if min(x_edge, y_edge) > 0.42 and fill_ratio > 0.48 and solidity > 0.75:
        return False

    return True


def build_run_based_seeds(
    necklace_mask: np.ndarray,
    red_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = necklace_mask.shape
    thread_seed = np.zeros_like(necklace_mask, dtype=np.uint8)
    chain_seed = np.zeros_like(necklace_mask, dtype=np.uint8)

    _, y1, _, y2 = bounding_box(necklace_mask)
    top_limit = y1 + int(round((y2 - y1 + 1) * 0.60))

    red_components = remove_small_components((red_mask & necklace_mask).astype(np.uint8), max(30, (h * w) // 40000))
    red_anchor = largest_component(red_components) if red_components.any() else np.zeros_like(necklace_mask, dtype=np.uint8)

    for x in range(w):
        ys = np.where(necklace_mask[:, x] > 0)[0]
        if ys.size == 0:
            continue
        runs: List[Tuple[int, int]] = []
        run_start = int(ys[0])
        prev = int(ys[0])
        for y in ys[1:]:
            y = int(y)
            if y - prev <= 4:
                prev = y
            else:
                runs.append((run_start, prev))
                run_start = y
                prev = y
        runs.append((run_start, prev))

        top_start, top_end = runs[0]
        if top_end <= top_limit:
            thread_seed[top_start : top_end + 1, x] = 1

        bottom_start, bottom_end = runs[-1]
        chain_seed[bottom_start : bottom_end + 1, x] = 1
        if len(runs) >= 2:
            mid_start, mid_end = runs[1]
            chain_seed[mid_start : mid_end + 1, x] = 1

    thread_seed = close(thread_seed, 13)
    chain_seed = close(chain_seed, 13)
    thread_seed &= necklace_mask
    chain_seed &= necklace_mask

    # Validate thread seed - if it doesn't look like thread, clear it
    if not is_valid_thread(thread_seed, necklace_mask):
        thread_seed = np.zeros_like(necklace_mask, dtype=np.uint8)

    if red_anchor.any():
        red_zone = dilate(red_anchor, 21)
        if thread_seed.any():
            thread_seed |= (necklace_mask & red_zone)
        chain_seed &= (1 - dilate(red_anchor, 25))

    chain_seed &= (1 - erode(thread_seed, 5))
    thread_seed = remove_small_components(thread_seed, max(80, (h * w) // 20000))
    chain_seed = remove_small_components(chain_seed, max(100, (h * w) // 18000))
    return thread_seed, chain_seed


def _normalized_jewel_type(jewel_type: str | None) -> str:
    return " ".join(str(jewel_type or "").strip().lower().split())


def _is_haram_type(jewel_type: str | None) -> bool:
    return _normalized_jewel_type(jewel_type) == "haram"


def _tassel_auto_policy(jewel_type: str | None) -> str:
    normalized_type = _normalized_jewel_type(jewel_type)
    if normalized_type in TASSEL_AUTO_DISABLED_TYPES:
        return "disabled"
    if normalized_type == "haram":
        return "haram"
    return "very_low"


def _largest_contour_shape_metrics(mask: np.ndarray) -> Tuple[float, float]:
    contours = contour_from_mask(mask.astype(np.uint8))
    if not contours:
        return 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (
        4.0 * math.pi * contour_area / (perimeter * perimeter)
        if perimeter > 0.0 and contour_area > 0.0
        else 0.0
    )
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    contour_solidity = contour_area / hull_area if hull_area > 0.0 else 0.0
    return float(circularity), float(contour_solidity)


def pendant_candidate_evidence(
    mask: np.ndarray,
    necklace_mask: np.ndarray,
    image: np.ndarray,
) -> Dict[str, object]:
    """Measure whether a region is a distinct bottom-middle compact pendant."""
    evidence: Dict[str, object] = {
        "accepted": False,
        "score": -999.0,
        "reason": "No pendant candidate.",
    }
    if not mask.any() or not necklace_mask.any():
        return evidence

    area = int(mask.sum())
    necklace_area = max(1, int(necklace_mask.sum()))
    area_ratio = area / float(necklace_area)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return evidence

    x1, y1, x2, y2 = bounding_box(mask)
    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)
    aspect = bbox_w / float(bbox_h)
    fill_ratio = area / float(max(1, bbox_w * bbox_h))
    solidity = compute_solidity(mask)
    circularity, contour_solidity = _largest_contour_shape_metrics(mask)

    necklace_x1, necklace_y1, necklace_x2, necklace_y2 = bounding_box(necklace_mask)
    range_x = max(1.0, float(necklace_x2 - necklace_x1))
    range_y = max(1.0, float(necklace_y2 - necklace_y1))
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    x_fraction = (center_x - necklace_x1) / range_x
    y_fraction = (center_y - necklace_y1) / range_y
    center_offset = abs(x_fraction - 0.5)
    bottom_reach = (float(y2) - necklace_y1) / range_y

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][mask > 0]
    sat = hsv[:, :, 1][mask > 0]
    val = hsv[:, :, 2][mask > 0]
    gold_ratio = float(((hue > 8) & (hue < 40) & (sat > 40)).mean())
    red_ratio = float((((hue < 10) | (hue > 170)) & (sat > 60)).mean())
    bright_metal_ratio = float(((sat < 75) & (val > 85) & (val < 250)).mean())
    jewelry_ratio = max(gold_ratio, red_ratio * 0.82, bright_metal_ratio * 0.72)
    mean_value = float(val.mean())
    dark_ratio = float((val < 100).mean())

    outside_candidate = (
        necklace_mask.astype(np.uint8)
        & (1 - mask.astype(np.uint8))
    )
    contact_shell = dilate(mask.astype(np.uint8), 5) & outside_candidate
    shell_y, shell_x = np.where(contact_shell > 0)
    top_center_contacts = 0
    top_contact_ratio = 0.0
    if len(shell_x):
        top_limit = y1 + int(round(bbox_h * 0.48))
        center_left = x1 + int(round(bbox_w * 0.20))
        center_right = x2 - int(round(bbox_w * 0.20))
        top_center_contacts = int(
            (
                (shell_y <= top_limit)
                & (shell_x >= center_left)
                & (shell_x <= center_right)
            ).sum()
        )
        top_contact_ratio = float((shell_y <= top_limit).mean())

    thickness = cv2.distanceTransform(necklace_mask.astype(np.uint8), cv2.DIST_L2, 5)
    candidate_peak = float(thickness[mask > 0].max()) if area else 0.0
    comparison_mask = necklace_mask & (1 - dilate(mask.astype(np.uint8), 5))
    comparison_values = thickness[comparison_mask > 0]
    chain_reference = (
        float(np.percentile(comparison_values, 80.0))
        if comparison_values.size
        else 1.0
    )
    thickness_ratio = candidate_peak / max(1.0, chain_reference)

    score = 0.0
    if 0.48 <= aspect <= 1.75:
        score += 1.5
    elif 0.36 <= aspect <= 2.15:
        score += 0.65

    if fill_ratio >= 0.42:
        score += 1.25
    elif fill_ratio >= 0.24:
        score += 0.75

    if solidity >= 0.68:
        score += 1.2
    elif solidity >= 0.42:
        score += 0.65

    if circularity >= 0.58:
        score += 1.25
    elif circularity >= 0.32:
        score += 0.65

    score += min(2.25, jewelry_ratio * 3.2)
    if mean_value > 145:
        score += 0.65

    if y_fraction >= 0.68:
        score += 1.35
    elif y_fraction >= 0.58:
        score += 0.7
    if bottom_reach >= 0.88:
        score += 1.0
    elif bottom_reach >= 0.78:
        score += 0.45

    if center_offset <= 0.16:
        score += 1.35
    elif center_offset <= 0.27:
        score += 0.7

    if top_center_contacts >= max(3, int(round(math.sqrt(area) * 0.06))):
        score += 0.85
    elif top_contact_ratio >= 0.30:
        score += 0.35

    if thickness_ratio >= 1.8:
        score += 0.9
    elif thickness_ratio >= 1.35:
        score += 0.4

    if dark_ratio > 0.55:
        score -= 2.0
    elif dark_ratio > 0.35:
        score -= 0.8
    if area_ratio > 0.36:
        score -= 1.0
    if center_offset > 0.34:
        score -= 2.0
    if bottom_reach < 0.74:
        score -= 2.0

    compact_votes = sum(
        (
            0.42 <= aspect <= 1.95,
            fill_ratio >= 0.22,
            solidity >= 0.38,
            circularity >= 0.28,
            contour_solidity >= 0.42,
        )
    )
    accepted = bool(
        0.02 <= area_ratio <= 0.48
        and jewelry_ratio >= 0.18
        and y_fraction >= 0.55
        and bottom_reach >= 0.76
        and center_offset <= 0.34
        and 0.34 <= aspect <= 2.20
        and fill_ratio >= 0.18
        and thickness_ratio >= 1.20
        and (
            circularity >= 0.22
            or (solidity >= 0.58 and thickness_ratio >= 1.55)
        )
        and compact_votes >= 3
        and score >= PENDANT_AUTO_MIN_SCORE
    )

    if accepted:
        reason = "Compact jewelry ornament attached near the bottom middle."
    elif center_offset > 0.34 or bottom_reach < 0.76:
        reason = "Candidate is not in the bottom-middle pendant zone."
    elif (
        compact_votes < 3
        or fill_ratio < 0.18
        or thickness_ratio < 1.20
        or (
            circularity < 0.22
            and not (solidity >= 0.58 and thickness_ratio >= 1.55)
        )
    ):
        reason = "Candidate is not compact/round/oval enough for a pendant."
    elif jewelry_ratio < 0.18:
        reason = "Candidate lacks jewelry-material evidence."
    else:
        reason = "Pendant evidence is below the conservative threshold."

    evidence.update(
        {
            "accepted": accepted,
            "score": round(float(score), 3),
            "reason": reason,
            "area_ratio": round(float(area_ratio), 4),
            "aspect_ratio": round(float(aspect), 3),
            "fill_ratio": round(float(fill_ratio), 3),
            "solidity": round(float(solidity), 3),
            "circularity": round(float(circularity), 3),
            "jewelry_ratio": round(float(jewelry_ratio), 3),
            "x_center_fraction": round(float(x_fraction), 3),
            "y_center_fraction": round(float(y_fraction), 3),
            "bottom_reach": round(float(bottom_reach), 3),
            "top_attachment_pixels": top_center_contacts,
            "thickness_ratio": round(float(thickness_ratio), 3),
        }
    )
    return evidence


def is_valid_pendant(mask: np.ndarray, necklace_mask: np.ndarray, image: np.ndarray) -> bool:
    return bool(pendant_candidate_evidence(mask, necklace_mask, image)["accepted"])


def _skeleton_endpoint_count(mask: np.ndarray) -> int:
    skeleton = _skeletonize(mask.astype(np.uint8))
    if not skeleton.any():
        return 0
    neighbors = cv2.filter2D(
        skeleton.astype(np.uint8),
        cv2.CV_16U,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    return int(((skeleton > 0) & (neighbors == 2)).sum())


def _parallel_line_metrics(mask: np.ndarray, image: np.ndarray) -> Tuple[int, float]:
    if not mask.any():
        return 0, 0.0
    x1, y1, x2, y2 = bounding_box(mask)
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    crop_gray = cv2.cvtColor(
        image[y1 : y2 + 1, x1 : x2 + 1],
        cv2.COLOR_BGR2GRAY,
    )
    edges = cv2.Canny(crop_gray, 40, 120)
    edges[crop_mask == 0] = 0
    max_dimension = max(crop_mask.shape)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(8, int(round(max_dimension * 0.05))),
        minLineLength=max(8, int(round(max_dimension * 0.12))),
        maxLineGap=max(3, int(round(max_dimension * 0.035))),
    )
    if lines is None:
        return 0, 0.0

    angles: List[float] = []
    weights: List[float] = []
    for line in lines[:, 0, :]:
        dx = float(line[2] - line[0])
        dy = float(line[3] - line[1])
        length = math.hypot(dx, dy)
        if length < max(8.0, max_dimension * 0.12):
            continue
        angles.append(math.atan2(dy, dx))
        weights.append(length)
    if not angles:
        return 0, 0.0

    doubled = np.asarray(angles, dtype=np.float32) * 2.0
    weight_array = np.asarray(weights, dtype=np.float32)
    vector = np.sum(weight_array * np.exp(1j * doubled))
    concentration = abs(vector) / max(1e-6, float(weight_array.sum()))
    return len(angles), float(concentration)


def tassel_candidate_evidence(
    mask: np.ndarray,
    necklace_mask: np.ndarray,
    image: np.ndarray,
    jewel_type: str | None = None,
) -> Dict[str, object]:
    """Require colorful, filament-like, terminal evidence for tassels."""
    evidence: Dict[str, object] = {
        "accepted": False,
        "score": -999.0,
        "reason": "No tassel candidate.",
    }
    if not mask.any():
        return evidence

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return evidence

    bbox_w = int(xs.max()) - int(xs.min()) + 1
    bbox_h = int(ys.max()) - int(ys.min()) + 1
    area = int(mask.sum())

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][mask > 0]
    sat = hsv[:, :, 1][mask > 0]
    val = hsv[:, :, 2][mask > 0]

    gold_ratio = float(((hue > 8) & (hue < 40) & (sat > 40)).mean())
    textile_pixels = (
        (sat > 80)
        & (val > 30)
        & (
            (hue < 10)
            | (hue > 170)
            | ((hue >= 35) & (hue <= 170))
        )
    )
    textile_ratio = float(textile_pixels.mean())
    saturated_ratio = float((sat > 85).mean())
    colorful_hues = hue[(sat > 90) & (val > 35)]
    hue_diversity = 0
    if colorful_hues.size:
        histogram, _ = np.histogram(colorful_hues, bins=6, range=(0, 180))
        hue_diversity = int((histogram >= max(2, colorful_hues.size * 0.05)).sum())

    solidity = compute_solidity(mask)
    aspect_ratio = bbox_w / max(1, bbox_h)
    fill_ratio = area / max(1, bbox_w * bbox_h)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 45, 135)
    edge_ratio = float((edge_map[mask > 0] > 0).mean())
    necklace_area = max(1, int(necklace_mask.sum()))
    area_ratio = area / float(necklace_area)
    skeleton = _skeletonize(mask.astype(np.uint8))
    skeleton_ratio = float(skeleton.sum()) / float(max(1, area))
    endpoint_count = _skeleton_endpoint_count(mask)
    long_line_count, line_concentration = _parallel_line_metrics(mask, image)

    score = 0.0

    if textile_ratio >= 0.42:
        score += 3.0
    elif textile_ratio >= 0.25:
        score += 2.3
    elif textile_ratio >= 0.16:
        score += 1.3

    if saturated_ratio >= 0.55:
        score += 1.2
    elif saturated_ratio >= 0.32:
        score += 0.65
    if hue_diversity >= 3:
        score += 0.8
    elif hue_diversity >= 2:
        score += 0.4

    if long_line_count >= 7:
        score += 2.4
    elif long_line_count >= 4:
        score += 1.7
    elif long_line_count >= 2:
        score += 0.65
    if line_concentration >= 0.62:
        score += 1.35
    elif line_concentration >= 0.42:
        score += 0.8

    if fill_ratio <= 0.32:
        score += 1.35
    elif fill_ratio <= 0.48:
        score += 0.75
    if solidity <= 0.55:
        score += 1.25
    elif solidity <= 0.72:
        score += 0.7

    if skeleton_ratio >= 0.10:
        score += 0.9
    elif skeleton_ratio >= 0.055:
        score += 0.45
    if endpoint_count >= 8:
        score += 1.1
    elif endpoint_count >= 4:
        score += 0.55
    if edge_ratio >= 0.20:
        score += 0.7
    elif edge_ratio >= 0.12:
        score += 0.35

    necklace_x1, necklace_y1, necklace_x2, necklace_y2 = bounding_box(necklace_mask)
    range_x = max(1.0, float(necklace_x2 - necklace_x1))
    range_y = max(1.0, float(necklace_y2 - necklace_y1))
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    x_fraction = (center_x - necklace_x1) / range_x
    y_fraction = (center_y - necklace_y1) / range_y
    terminal_distance = min(
        x_fraction,
        1.0 - x_fraction,
        y_fraction,
        1.0 - y_fraction,
    )
    if terminal_distance <= 0.16:
        score += 1.15
    elif terminal_distance <= 0.28:
        score += 0.5

    auto_policy = _tassel_auto_policy(jewel_type)
    if auto_policy == "haram":
        score += 0.25
        threshold = TASSEL_AUTO_MIN_SCORE_HARAM
    elif auto_policy == "disabled":
        score -= 2.0
        threshold = 999.0
    else:
        score -= 1.5
        threshold = TASSEL_AUTO_MIN_SCORE_OTHER

    if gold_ratio > 0.55 and textile_ratio < 0.16:
        score -= 3.0
    if fill_ratio > 0.58 or solidity > 0.82:
        score -= 2.0

    if auto_policy == "very_low":
        colorful_threads = (
            textile_ratio >= 0.25
            and saturated_ratio >= 0.40
            and hue_diversity >= 2
        )
        filament_bundle = (
            long_line_count >= 5
            and line_concentration >= 0.48
            and skeleton_ratio >= 0.06
            and endpoint_count >= 5
        )
        sparse_bundle = fill_ratio <= 0.45 and solidity <= 0.70
        terminal_region = terminal_distance <= 0.22
        max_area_ratio = 0.45
    else:
        colorful_threads = textile_ratio >= 0.16 and saturated_ratio >= 0.30
        filament_bundle = (
            long_line_count >= 4
            and line_concentration >= 0.38
            and (skeleton_ratio >= 0.045 or endpoint_count >= 4)
        )
        sparse_bundle = fill_ratio <= 0.52 and solidity <= 0.78
        terminal_region = terminal_distance <= 0.30
        max_area_ratio = 0.62
    accepted = bool(
        auto_policy != "disabled"
        and 0.012 <= area_ratio <= max_area_ratio
        and colorful_threads
        and filament_bundle
        and sparse_bundle
        and terminal_region
        and score >= threshold
    )

    if auto_policy == "disabled":
        reason = (
            "Automatic tassel detection is disabled for this jewel type; "
            "use a manual correction when a tassel is present."
        )
    elif accepted:
        reason = "Colorful terminal bundle with multiple aligned thread-like strands."
    elif not colorful_threads:
        reason = "No strong colorful textile-thread evidence."
    elif not filament_bundle:
        reason = "Region lacks multiple elongated, aligned strand features."
    elif not terminal_region:
        reason = "Region is not terminal enough for a tassel."
    elif not sparse_bundle:
        reason = "Region is too solid/compact and looks like jewelry rather than thread."
    else:
        reason = "Tassel confidence is below the conservative threshold."

    evidence.update(
        {
            "accepted": accepted,
            "score": round(float(score), 3),
            "threshold": round(float(threshold), 3),
            "reason": reason,
            "jewel_type": str(jewel_type or ""),
            "auto_policy": auto_policy,
            "haram_prior": _is_haram_type(jewel_type),
            "area_ratio": round(float(area_ratio), 4),
            "aspect_ratio": round(float(aspect_ratio), 3),
            "fill_ratio": round(float(fill_ratio), 3),
            "solidity": round(float(solidity), 3),
            "gold_ratio": round(float(gold_ratio), 3),
            "textile_ratio": round(float(textile_ratio), 3),
            "saturated_ratio": round(float(saturated_ratio), 3),
            "hue_diversity": hue_diversity,
            "edge_ratio": round(float(edge_ratio), 3),
            "skeleton_ratio": round(float(skeleton_ratio), 4),
            "skeleton_endpoints": endpoint_count,
            "long_line_count": long_line_count,
            "line_concentration": round(float(line_concentration), 3),
            "terminal_distance": round(float(terminal_distance), 3),
        }
    )
    return evidence


def score_tassel_candidate(
    mask: np.ndarray,
    necklace_mask: np.ndarray,
    image: np.ndarray,
    jewel_type: str | None = None,
) -> float:
    return float(
        tassel_candidate_evidence(mask, necklace_mask, image, jewel_type)["score"]
    )


def is_likely_tassel(
    mask: np.ndarray,
    necklace_mask: np.ndarray,
    image: np.ndarray,
    jewel_type: str | None = None,
) -> bool:
    return bool(
        tassel_candidate_evidence(mask, necklace_mask, image, jewel_type)["accepted"]
    )


def _legacy_score_pendant_candidate(
    blob: np.ndarray, necklace_mask: np.ndarray, image: np.ndarray,
) -> float:
    """Score a candidate region for pendant-ness (higher = more pendant-like)."""
    if not blob.any():
        return -999.0

    h, w = necklace_mask.shape
    area = int(blob.sum())
    necklace_area = max(1, int(necklace_mask.sum()))
    area_ratio = area / necklace_area

    # Too small or too large → reject
    if area_ratio < 0.02 or area_ratio > 0.58:
        return -999.0

    ys, xs = np.where(blob > 0)
    bbox_w = int(xs.max()) - int(xs.min()) + 1
    bbox_h = int(ys.max()) - int(ys.min()) + 1
    aspect = bbox_w / max(1, bbox_h)
    fill = area / max(1, bbox_w * bbox_h)
    solidity = compute_solidity(blob)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][blob > 0]
    sat = hsv[:, :, 1][blob > 0]
    val = hsv[:, :, 2][blob > 0]
    gold_ratio = float(((hue > 8) & (hue < 40) & (sat > 40)).mean())
    red_ratio  = float((((hue < 10) | (hue > 170)) & (sat > 60)).mean())
    bright_metal_ratio = float(((sat < 75) & (val > 85) & (val < 250)).mean())
    mean_val   = float(val.mean())
    dark_ratio = float((val < 100).mean())

    # --- Build score ---
    score = 0.0

    # Gold/jewelry colour is the strongest pendant indicator
    jewelry = max(gold_ratio, red_ratio * 0.8, bright_metal_ratio * 0.72)
    score += jewelry * 6.0          # max ~6

    # Brightness: pendants are bright (gold reflects light)
    if mean_val > 150:
        score += 2.0
    elif mean_val > 120:
        score += 1.0

    # Solidity: pendants are solid shapes
    score += solidity * 2.0          # max ~2

    # Fill ratio: pendants fill their bbox
    score += fill * 1.5              # max ~1.5

    # Compact aspect ratio close to 1.0
    if 0.45 <= aspect <= 1.7:
        score += 1.0
    elif 0.28 <= aspect <= 2.8:
        score += 0.5

    # Location: prefer lower parts of necklace
    necklace_ys = np.where(necklace_mask > 0)[0]
    y_range = float(necklace_ys.max() - necklace_ys.min()) if len(necklace_ys) > 1 else 1.0
    y_frac = (float(ys.mean()) - float(necklace_ys.min())) / max(1.0, y_range)
    score += y_frac * 1.0            # max ~1 for bottom

    if area_ratio > 0.35:
        if jewelry < 0.42 or fill < 0.28 or solidity < 0.48 or y_frac < 0.55:
            return -999.0
        score += 0.75

    # --- Penalties for tassel characteristics ---
    if dark_ratio > 0.5:
        score -= 3.0
    elif dark_ratio > 0.3:
        score -= 1.5

    if gold_ratio < 0.20 and bright_metal_ratio < 0.28 and red_ratio < 0.20:
        score -= 2.0

    if solidity < 0.36:
        score -= 2.0

    return score


def _score_pendant_candidate(
    blob: np.ndarray,
    necklace_mask: np.ndarray,
    image: np.ndarray,
) -> float:
    """Score a candidate using the conservative pendant evidence policy."""
    return float(pendant_candidate_evidence(blob, necklace_mask, image)["score"])


def pick_pendant_seed(
    necklace_mask: np.ndarray,
    image: np.ndarray,
    jewel_type: str | None = None,
) -> np.ndarray:
    """Find the pendant region using multi-candidate blob scoring.

    Instead of a single density-peak ellipse (which can land on tassels),
    we find multiple dense blob candidates and score each for pendant-ness.
    """
    h, w = necklace_mask.shape
    necklace_area = max(1, int(necklace_mask.sum()))

    # ---- Strategy 1: density-peak ellipse (original approach) ----
    mask_f = necklace_mask.astype(np.float32)
    sigma = max(8.0, min(w, h) / 40.0)
    density = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    y_grid = np.linspace(0.90, 1.20, h, dtype=np.float32)[:, None]
    weighted = density * y_grid
    peak_y, peak_x = np.unravel_index(int(np.argmax(weighted)), weighted.shape)

    rx = max(40, int(round(w * 0.092)))
    ry = max(40, int(round(h * 0.092)))
    yy, xx = np.ogrid[:h, :w]
    ellipse = ((((xx - peak_x) / float(rx)) ** 2 + ((yy - peak_y) / float(ry)) ** 2) <= 1.0).astype(np.uint8)
    density_seed = ellipse & necklace_mask
    density_seed = close(density_seed, 9)
    density_seed = remove_small_components(density_seed, max(80, (h * w) // 20000))

    # ---- Strategy 2: thick-blob connected components ----
    # Erode the necklace mask aggressively to isolate thick/dense regions (pendants)
    # then find connected components as candidates
    blob_candidates: List[np.ndarray] = []
    for ksize in (15, 21, 29):
        eroded = erode(necklace_mask, ksize)
        eroded = remove_small_components(eroded, max(200, necklace_area // 500))
        if not eroded.any():
            continue
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded.astype(np.uint8), 8)
        for idx in range(1, n_labels):
            comp = (labels == idx).astype(np.uint8)
            comp_area = stats[idx, cv2.CC_STAT_AREA]
            if comp_area < max(200, necklace_area // 500):
                continue
            # Dilate back to recover original extent
            candidate = dilate(comp, ksize + 4) & necklace_mask
            candidate = close(candidate, 7)
            blob_candidates.append(candidate)

    # ---- Strategy 3: local thickness peaks ----
    # Some pendants stay connected to the chain so strongly that erosion does not
    # isolate them well. The distance-transform core highlights the thickest local
    # ornament region and lets us regrow it back into a pendant candidate.
    thickness = cv2.distanceTransform(necklace_mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_thickness = float(thickness.max())
    if max_thickness >= 10.0:
        expand_radius = max(12, int(round(max_thickness * 1.8)))
        for frac in (0.55, 0.60):
            thick_core = (thickness >= max_thickness * frac).astype(np.uint8)
            thick_core = remove_small_components(thick_core, max(20, necklace_area // 4000))
            if not thick_core.any():
                continue
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_core.astype(np.uint8), 8)
            for idx in range(1, n_labels):
                if stats[idx, cv2.CC_STAT_AREA] < max(20, necklace_area // 5000):
                    continue
                comp = (labels == idx).astype(np.uint8)
                candidate = dilate(comp, expand_radius) & necklace_mask
                candidate = close(candidate, 7)
                blob_candidates.append(candidate)

    # ---- Strategy 4: gold-colour blob ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold_mask = (((hsv[:, :, 0] > 8) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 80))).astype(np.uint8)
    gold_on_necklace = gold_mask & necklace_mask
    gold_on_necklace = close(gold_on_necklace, 9)
    gold_on_necklace = remove_small_components(gold_on_necklace, max(200, necklace_area // 400))
    if gold_on_necklace.any():
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gold_on_necklace.astype(np.uint8), 8)
        for idx in range(1, n_labels):
            comp = (labels == idx).astype(np.uint8)
            if stats[idx, cv2.CC_STAT_AREA] >= max(200, necklace_area // 400):
                blob_candidates.append(comp)

    # ---- Collect all candidates and score ----
    all_candidates = [density_seed] + blob_candidates
    best_score = -999.0
    best_seed = np.zeros_like(necklace_mask, dtype=np.uint8)

    for cand in all_candidates:
        if not cand.any():
            continue
        # Skip candidates that look like tassels
        if is_likely_tassel(cand, necklace_mask, image, jewel_type):
            continue
        if not is_valid_pendant(cand, necklace_mask, image):
            continue
        sc = _score_pendant_candidate(cand, necklace_mask, image)
        if sc > best_score:
            best_score = sc
            best_seed = cand.copy()

    return best_seed


def detect_pendant_seed(
    necklace_mask: np.ndarray,
    image: np.ndarray,
    jewel_type: str | None = None,
) -> Tuple[np.ndarray, float]:
    candidate = pick_pendant_seed(necklace_mask, image, jewel_type)
    if not candidate.any():
        return np.zeros_like(necklace_mask, dtype=np.uint8), -999.0

    score = _score_pendant_candidate(candidate, necklace_mask, image)
    evidence = pendant_candidate_evidence(candidate, necklace_mask, image)
    if not evidence["accepted"]:
        return np.zeros_like(necklace_mask, dtype=np.uint8), score
    return candidate.astype(np.uint8), score


def build_feedback_support_mask(
    support_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    close_ksize: int = 5,
    min_component: int = 12,
) -> np.ndarray:
    region = rectangle_mask(support_mask.shape, bbox).astype(np.uint8)
    candidate = (support_mask & region).astype(np.uint8)
    if not candidate.any():
        candidate = region
    candidate = close(candidate, close_ksize)
    candidate = remove_small_components(candidate, min_component)
    return candidate.astype(np.uint8)


def build_pendant_seed_from_feedback(
    necklace_mask: np.ndarray,
    feedback_mask: np.ndarray,
) -> np.ndarray:
    candidate = (necklace_mask & feedback_mask).astype(np.uint8)
    candidate = close(candidate, 7)
    candidate = remove_small_components(candidate, max(20, candidate.size // 30000))
    if not candidate.any():
        return np.zeros_like(necklace_mask, dtype=np.uint8)

    thickness = cv2.distanceTransform(candidate.astype(np.uint8), cv2.DIST_L2, 5)
    max_thickness = float(thickness.max())
    if max_thickness < 3.0:
        return candidate

    thick_core = (thickness >= max(2.0, max_thickness * 0.45)).astype(np.uint8)
    thick_core = remove_small_components(thick_core, max(8, int(candidate.sum()) // 2500))
    if not thick_core.any():
        return candidate

    expand_radius = max(5, int(round(max_thickness * 2.0)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_core.astype(np.uint8), 8)
    best_seed = np.zeros_like(candidate, dtype=np.uint8)
    best_metric = -1.0
    for idx in range(1, n_labels):
        if stats[idx, cv2.CC_STAT_AREA] < max(8, int(candidate.sum()) // 3000):
            continue
        comp = (labels == idx).astype(np.uint8)
        seed = (dilate(comp, expand_radius) & candidate).astype(np.uint8)
        seed = close(seed, 7)
        metric = float(seed.sum()) * max(0.1, compute_solidity(seed))
        if metric > best_metric:
            best_metric = metric
            best_seed = seed

    if not best_seed.any():
        best_seed = candidate

    best_seed &= necklace_mask
    best_seed = close(best_seed, 9)
    best_seed = remove_small_components(best_seed, max(20, int(candidate.sum()) // 4000))
    return best_seed.astype(np.uint8)


def build_tassel_seed_from_feedback(
    necklace_mask: np.ndarray,
    feedback_mask: np.ndarray,
) -> np.ndarray:
    candidate = (necklace_mask & feedback_mask).astype(np.uint8)
    candidate = close(candidate, 5)
    candidate = remove_small_components(candidate, max(10, int(candidate.sum()) // 5000))
    if not candidate.any():
        return np.zeros_like(necklace_mask, dtype=np.uint8)
    candidate &= necklace_mask
    return candidate.astype(np.uint8)


def detect_tassel_seed(
    necklace_mask: np.ndarray,
    image: np.ndarray,
    textile_mask: np.ndarray,
    jewel_type: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = necklace_mask.shape
    if _tassel_auto_policy(jewel_type) == "disabled":
        return (
            np.zeros_like(necklace_mask, dtype=np.uint8),
            necklace_mask.copy().astype(np.uint8),
            -999.0,
        )

    topology_seed, chain_hint = build_run_based_seeds(necklace_mask, textile_mask)
    candidates: List[np.ndarray] = []
    topology_textile_overlap = (
        float((topology_seed & textile_mask).sum())
        / float(max(1, int(topology_seed.sum())))
        if topology_seed.any()
        else 0.0
    )
    if topology_seed.any() and topology_textile_overlap >= 0.10:
        candidates.append(topology_seed)

    color_components = remove_small_components(
        (textile_mask & necklace_mask).astype(np.uint8),
        max(24, (h * w) // 50000),
    )
    if color_components.any():
        grouped_color = close(dilate(color_components, 5), 11) & necklace_mask
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grouped_color.astype(np.uint8),
            8,
        )
        expand_radius = max(18, int(round(min(h, w) * 0.06)))
        for idx in range(1, n_labels):
            if stats[idx, cv2.CC_STAT_AREA] < max(24, (h * w) // 50000):
                continue
            core = (labels == idx).astype(np.uint8)
            distance = distance_to_seed(core)
            candidate = ((distance <= expand_radius) & (necklace_mask > 0)).astype(np.uint8)
            candidate = close(candidate, 9)
            candidate = remove_small_components(candidate, max(60, (h * w) // 22000))
            if candidate.any():
                candidates.append(candidate)

    # False positives are more costly than misses. Texture proposals are enabled
    # only when colorful textile pixels already exist; monochrome/gold thread
    # can still be supplied by manual feedback.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = (cv2.Canny(gray, 45, 135) > 0).astype(np.float32)
    density_sigma = max(3.0, min(h, w) / 70.0)
    edge_density = cv2.GaussianBlur(edges, (0, 0), density_sigma)
    necklace_values = edge_density[necklace_mask > 0]
    if necklace_values.size and color_components.any():
        density_threshold = float(np.percentile(necklace_values, 72.0))
        x1, y1, x2, y2 = bounding_box(necklace_mask)
        yy, xx = np.ogrid[:h, :w]
        range_x = max(1.0, float(x2 - x1))
        range_y = max(1.0, float(y2 - y1))
        endpoint_zone = (
            (xx <= x1 + 0.24 * range_x)
            | (xx >= x2 - 0.24 * range_x)
            | (yy <= y1 + 0.18 * range_y)
            | (yy >= y1 + 0.70 * range_y)
        )
        texture_core = (
            (edge_density >= max(0.015, density_threshold))
            & endpoint_zone
            & (necklace_mask > 0)
        ).astype(np.uint8)
        texture_core = close(dilate(texture_core, 5), 9)
        texture_core = remove_small_components(
            texture_core,
            max(30, (h * w) // 45000),
        )
        if texture_core.any():
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                texture_core,
                8,
            )
            component_indices = sorted(
                range(1, n_labels),
                key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
                reverse=True,
            )[:6]
            for idx in component_indices:
                core = (labels == idx).astype(np.uint8)
                distance = distance_to_seed(core)
                candidate = (
                    (distance <= max(12, int(round(min(h, w) * 0.04))))
                    & (necklace_mask > 0)
                ).astype(np.uint8)
                candidate = close(candidate, 7)
                candidate = remove_small_components(
                    candidate,
                    max(50, (h * w) // 25000),
                )
                if candidate.any():
                    candidates.append(candidate)

        # Fine saturated strands can be disconnected by the foreground mask.
        # Group them by proximity before regrowing into the necklace support.
        strand_groups = close(dilate(color_components, 9), 17)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            strand_groups.astype(np.uint8),
            8,
        )
        for idx in range(1, n_labels):
            if stats[idx, cv2.CC_STAT_AREA] < max(80, (h * w) // 25000):
                continue
            group = (labels == idx).astype(np.uint8)
            distance = distance_to_seed(group)
            candidate = (
                (distance <= max(12, int(round(min(h, w) * 0.035))))
                & (necklace_mask > 0)
            ).astype(np.uint8)
            candidate = close(candidate, 7)
            candidate = remove_small_components(
                candidate,
                max(50, (h * w) // 25000),
            )
            if candidate.any():
                candidates.append(candidate)

    best_seed = np.zeros_like(necklace_mask, dtype=np.uint8)
    best_score = -999.0
    best_evidence: Dict[str, object] | None = None
    for cand in candidates:
        evidence = tassel_candidate_evidence(
            cand,
            necklace_mask,
            image,
            jewel_type,
        )
        score = float(evidence["score"])
        if (
            best_evidence is None
            or (
                bool(evidence["accepted"])
                and not bool(best_evidence["accepted"])
            )
            or (
                bool(evidence["accepted"]) == bool(best_evidence["accepted"])
                and score > best_score
            )
        ):
            best_score = score
            best_seed = cand.copy()
            best_evidence = evidence

    if best_evidence is None or not best_evidence["accepted"]:
        return np.zeros_like(necklace_mask, dtype=np.uint8), chain_hint, best_score

    best_seed = close(best_seed, 9) & necklace_mask
    best_seed = remove_small_components(best_seed, max(80, (h * w) // 20000))
    chain_hint = (chain_hint & (1 - dilate(best_seed, 5))).astype(np.uint8)
    return best_seed.astype(np.uint8), chain_hint, best_score


def distance_to_seed(seed: np.ndarray) -> np.ndarray:
    inv = np.where(seed > 0, 0, 1).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 5)


def assign_parts_by_seed(
    necklace_mask: np.ndarray,
    tassel_seed: np.ndarray,
    chain_seed: np.ndarray,
    pendant_seed: np.ndarray,
) -> Dict[str, np.ndarray]:
    h, w = necklace_mask.shape
    base_seeds = {
        "tassel": tassel_seed.astype(np.uint8),
        "chain": chain_seed.astype(np.uint8),
        "pendant": pendant_seed.astype(np.uint8),
    }

    if not base_seeds["chain"].any():
        base_seeds["chain"] = necklace_mask & (1 - dilate(base_seeds["tassel"], 17)) & (1 - dilate(base_seeds["pendant"], 17))

    seed_masks = {name: remove_small_components(mask & necklace_mask, 40) for name, mask in base_seeds.items()}
    parts = {name: np.zeros((h, w), dtype=np.uint8) for name in ["pendant", "chain", "tassel"]}
    active_seeds = [(name, seed_masks[name]) for name in ["pendant", "chain", "tassel"] if seed_masks[name].any()]

    if active_seeds:
        distances = [distance_to_seed(seed) for name, seed in active_seeds]
        names = [name for name, _ in active_seeds]
        stack = np.stack(distances, axis=0)
        assignment = np.argmin(stack, axis=0)

        for idx, name in enumerate(names):
            parts[name] = ((assignment == idx) & (necklace_mask > 0)).astype(np.uint8)
            parts[name] |= seed_masks[name]
            parts[name] &= necklace_mask
            parts[name] = close(parts[name], 5)
    else:
        parts["chain"] = necklace_mask.copy()

    return parts


def post_refine_parts(parts: Dict[str, np.ndarray], necklace_mask: np.ndarray) -> Dict[str, np.ndarray]:
    refined = {name: mask.copy() for name, mask in parts.items()}

    if refined["pendant"].any():
        refined["pendant"] = close(refined["pendant"], 11)
    if refined["tassel"].any():
        refined["tassel"] = close(refined["tassel"], 9)

    for key in ("pendant", "chain", "tassel"):
        refined[key] &= necklace_mask
        refined[key] = remove_small_components(refined[key], 40)

    refined["tassel"] &= (1 - refined["pendant"])
    reserved = refined["pendant"] | refined["tassel"]
    refined["chain"] = (necklace_mask & (1 - reserved)).astype(np.uint8)
    refined["chain"] = close(refined["chain"], 9)
    refined["chain"] &= necklace_mask
    refined["chain"] = remove_small_components(refined["chain"], 40)

    return refined


def make_cutout(image: np.ndarray, mask: np.ndarray, pad: int = 20, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    if not mask.any():
        return np.full((256, 256, 3), bg_color, dtype=np.uint8)

    x1, y1, x2, y2 = bounding_box(mask)
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    crop_img = image[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    bg = np.full_like(crop_img, bg_color)
    bg[crop_mask > 0] = crop_img[crop_mask > 0]
    return bg


def fit_into_box(image: np.ndarray, width: int, height: int, margin: int = 18) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min((width - 2 * margin) / max(1, w), (height - 2 * margin) / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def draw_dashed_contour(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    for contour in contour_from_mask(mask):
        if len(contour) < 2:
            continue
        pts = contour[:, 0, :]
        dash_on = True
        dash_remaining = 12.0
        for idx in range(len(pts)):
            p1 = pts[idx]
            p2 = pts[(idx + 1) % len(pts)]
            segment = p2.astype(np.float32) - p1.astype(np.float32)
            length = float(np.linalg.norm(segment))
            if length < 1e-6:
                continue
            direction = segment / length
            progress = 0.0
            while progress < length:
                step = min(dash_remaining, length - progress)
                start = p1.astype(np.float32) + direction * progress
                end = p1.astype(np.float32) + direction * (progress + step)
                if dash_on:
                    cv2.line(
                        image,
                        tuple(np.round(start).astype(int)),
                        tuple(np.round(end).astype(int)),
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                progress += step
                if abs(step - dash_remaining) < 1e-6:
                    dash_on = not dash_on
                    dash_remaining = 12.0
                else:
                    dash_remaining -= step


def draw_badge(image: np.ndarray, center: Tuple[int, int], index: int, color: Tuple[int, int, int]) -> None:
    cv2.circle(image, center, 22, color, -1, cv2.LINE_AA)
    cv2.circle(image, center, 22, (255, 255, 255), 2, cv2.LINE_AA)
    text = str(index)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        image,
        text,
        (center[0] - text_w // 2, center[1] + text_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def build_layout(image: np.ndarray, parts: Dict[str, np.ndarray], bead_result: Dict[str, object] | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    panel_w = max(360, int(w * 0.35))
    start_x = w + 40
    start_y = 20
    bottom_margin = 20
    title_block_h = 56
    inner_bottom_pad = 14
    analysis_panel_h = 130 if bead_result is not None else 0
    box_h = max(160, int(math.ceil((h - start_y - bottom_margin) / max(1, len(PARTS)))))
    canvas_h = max(h, start_y + len(PARTS) * box_h + analysis_panel_h + bottom_margin + 40)

    canvas = np.full((canvas_h, w + panel_w + 80, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = image.copy()

    label_positions = {}
    for idx, (part_key, _, color) in enumerate(PARTS, start=1):
        mask = parts[part_key]
        if not mask.any():
            continue
        draw_dashed_contour(canvas[:, :w], mask, color, thickness=2)
        cx, cy = centroid(mask)
        x1, y1, x2, y2 = bounding_box(mask)
        badge_x = min(w - 30, max(30, int(round(cx))))
        badge_y = max(30, int(round(y1)) - 22)
        if part_key == "pendant":
            badge_x = min(w - 30, x2 - 18)
            badge_y = min(h - 30, y2 + 30)
        elif part_key == "tassel":
            badge_x = max(30, x1 + 16)
            badge_y = max(30, y1 - 18)
        draw_badge(canvas[:, :w], (badge_x, badge_y), idx, color)
        label_positions[part_key] = (badge_x, badge_y)

    for idx, (part_key, title, color) in enumerate(PARTS, start=1):
        box_y = start_y + (idx - 1) * box_h
        box_x2 = start_x + panel_w
        box_y2 = min(canvas_h - bottom_margin, box_y + box_h - 22)
        cv2.rectangle(canvas, (start_x, box_y), (box_x2, box_y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{idx}",
            (start_x + 20, box_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            title,
            (start_x + 58, box_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        if not parts[part_key].any() and part_key in {"pendant", "tassel"}:
            note = "Not detected"
            cv2.putText(
                canvas,
                note,
                (start_x + 24, box_y + 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (105, 105, 105),
                1,
                cv2.LINE_AA,
            )
            continue
        cutout = make_cutout(image, parts[part_key])
        inner_y = box_y + title_block_h
        inner_x = start_x + 20
        available_inner_h = max(1, box_y2 - inner_y - inner_bottom_pad)
        box_inner = fit_into_box(cutout, panel_w - 40, available_inner_h)
        ih, iw = box_inner.shape[:2]
        paste_h = min(ih, canvas_h - inner_y)
        paste_w = min(iw, canvas.shape[1] - inner_x)
        if paste_h > 0 and paste_w > 0:
            canvas[inner_y : inner_y + paste_h, inner_x : inner_x + paste_w] = box_inner[:paste_h, :paste_w]

    if bead_result is not None:
        analysis_y = start_y + len(PARTS) * box_h + 30
        box_x2 = start_x + panel_w

        risk = str(bead_result.get("risk", "Low"))
        beads_detected = bool(bead_result.get("beads_detected", False))
        thickness_peaks = int(bead_result.get("thickness_peaks", 0))
        blob_count = int(bead_result.get("blob_count", 0))
        variation = float(bead_result.get("width_variation", 0.0))
        mean_w = float(bead_result.get("chain_mean_width", 0.0))

        risk_color = (0, 180, 0) if risk == "Low" else (0, 0, 220)
        risk_bg_color = (220, 255, 220) if risk == "Low" else (255, 220, 220)

        cv2.rectangle(canvas, (start_x, analysis_y), (box_x2, analysis_y + analysis_panel_h), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (start_x + 1, analysis_y + 1), (box_x2 - 1, analysis_y + analysis_panel_h - 1), (248, 248, 248), -1)
        cv2.putText(
            canvas,
            "Bead Analysis",
            (start_x + 20, analysis_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (start_x + 20, analysis_y + 42), (box_x2 - 20, analysis_y + 68), risk_bg_color, -1)
        cv2.rectangle(canvas, (start_x + 20, analysis_y + 42), (box_x2 - 20, analysis_y + 68), risk_color, 1, cv2.LINE_AA)
        risk_label = f"Chain RISK: {risk}"
        if beads_detected:
            risk_label += " | Decorative elements detected"
        cv2.putText(
            canvas,
            risk_label,
            (start_x + 30, analysis_y + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            risk_color,
            2,
            cv2.LINE_AA,
        )

        detail_text = (
            f"Thickness Peaks: {thickness_peaks} | Blobs: {blob_count} | "
            f"Width Var: {variation:.2f} | Avg Width: {mean_w:.1f}px"
        )
        cv2.putText(
            canvas,
            detail_text,
            (start_x + 20, analysis_y + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    alpha = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_path), alpha)


def mask_summary(mask: np.ndarray) -> Dict[str, object]:
    if not mask.any():
        return {"area": 0, "bbox": None, "centroid": None}
    x1, y1, x2, y2 = bounding_box(mask)
    cx, cy = centroid(mask)
    return {
        "area": int(mask.sum()),
        "bbox": [x1, y1, x2, y2],
        "centroid": [round(cx, 2), round(cy, 2)],
    }


def reclassify_pendant_if_tassel(
    parts: Dict[str, np.ndarray],
    necklace_mask: np.ndarray,
    image: np.ndarray,
    lock_pendant: bool = False,
    lock_tassel: bool = False,
    jewel_type: str | None = None,
) -> Dict[str, np.ndarray]:
    """Post-processing: if the final pendant region looks like a tassel, move it to the tassel/thread class."""
    pendant = parts.get("pendant")
    if pendant is None or not pendant.any():
        return parts
    if lock_pendant or lock_tassel:
        return parts

    if is_likely_tassel(pendant, necklace_mask, image, jewel_type):
        parts["tassel"] = ((parts["tassel"] | pendant) & necklace_mask).astype(np.uint8)
        parts["pendant"] = np.zeros_like(necklace_mask, dtype=np.uint8)
        parts["tassel"] = close(parts["tassel"], 9)
        parts["tassel"] = remove_small_components(parts["tassel"], 40)
        parts["tassel"] &= necklace_mask
        parts["chain"] = necklace_mask & (1 - parts["pendant"]) & (1 - parts["tassel"])

    return parts


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    if _SKIMAGE_AVAILABLE:
        return _skimage_skeletonize(mask.astype(np.uint8)).astype(np.uint8)
    temp = mask.astype(np.uint8)
    skeleton = np.zeros_like(temp)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(temp, kernel)
        dilated = cv2.dilate(eroded, kernel)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton


def detect_chain_beads(chain_mask: np.ndarray, full_image: np.ndarray) -> Dict[str, object]:
    result: Dict[str, object] = {
        "beads_detected": False,
        "risk": "Low",
        "bead_count": 0,
        "thickness_peaks": 0,
        "blob_count": 0,
        "candidate_count": 0,
        "width_variation": 0.0,
        "chain_min_width": 0.0,
        "chain_max_width": 0.0,
        "chain_mean_width": 0.0,
        "baseline_radius": 0.0,
        "max_thickness_ratio": 0.0,
        "candidate_size_cv": 0.0,
        "decision_reason": "No repeated compact bead pattern found.",
    }

    if not chain_mask.any():
        return result

    x1, y1, x2, y2 = bounding_box(chain_mask)
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1
    if crop_w < 10 or crop_h < 10:
        return result

    chain_crop = chain_mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8)
    chain_crop = close(chain_crop, 3)
    chain_crop = remove_small_components(chain_crop, 20)
    if not chain_crop.any():
        return result

    analysis_pad = 2
    analysis_mask = cv2.copyMakeBorder(
        chain_crop,
        analysis_pad,
        analysis_pad,
        analysis_pad,
        analysis_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    analysis_h, analysis_w = analysis_mask.shape
    dt = cv2.distanceTransform(analysis_mask, cv2.DIST_L2, 5)
    max_dt = float(dt.max())
    min_dt = float(dt.min())
    if max_dt > 1e10 or abs(max_dt - min_dt) < 0.5:
        return result

    skeleton = _skeletonize(analysis_mask)
    skel_pts = np.where(skeleton > 0)
    if len(skel_pts[0]) < 10:
        return result

    widths = dt[skel_pts]
    if len(widths) == 0:
        return result

    median_w = float(np.median(widths))
    lower_width_limit = float(np.percentile(widths, 60))
    baseline_widths = widths[widths <= lower_width_limit]
    baseline_radius = float(np.median(baseline_widths)) if len(baseline_widths) else median_w
    baseline_radius = max(1.0, baseline_radius)
    result["chain_min_width"] = float(widths.min())
    result["chain_max_width"] = float(widths.max())
    result["chain_mean_width"] = float(widths.mean())
    result["width_variation"] = float(np.std(widths)) / max(1.0, median_w)
    result["baseline_radius"] = baseline_radius
    result["max_thickness_ratio"] = float(widths.max()) / baseline_radius

    # Skeleton pixels returned by np.where are in row-major image order, not
    # chain-path order. Detect compact 2D thickness cores instead of treating
    # that unordered sequence as a signal.
    core_threshold = max(baseline_radius * 1.80, baseline_radius + 1.75)
    thick_core = (dt >= core_threshold).astype(np.uint8)
    if baseline_radius >= 2.0:
        thick_core = open_mask(thick_core, 3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thick_core,
        8,
    )
    result["thickness_peaks"] = max(0, num_labels - 1)

    chain_area = int(analysis_mask.sum())
    min_core_area = max(4, int(round(math.pi * (baseline_radius * 0.35) ** 2)))
    max_core_area = max(min_core_area + 1, int(round(chain_area * 0.12)))
    shoulder_threshold = max(
        baseline_radius * 1.25,
        baseline_radius + 0.75,
    )
    shoulder_mask = (dt >= shoulder_threshold).astype(np.uint8)
    (
        _shoulder_count,
        shoulder_labels,
        shoulder_stats,
        _shoulder_centroids,
    ) = cv2.connectedComponentsWithStats(shoulder_mask, 8)
    seen_shoulder_labels: set[int] = set()
    candidates: List[Dict[str, object]] = []

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bx = int(stats[idx, cv2.CC_STAT_LEFT])
        by = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < min_core_area or area > max_core_area or bw < 2 or bh < 2:
            continue
        if (
            bx <= analysis_pad
            or by <= analysis_pad
            or bx + bw >= analysis_w - analysis_pad
            or by + bh >= analysis_h - analysis_pad
        ):
            continue

        aspect = min(bw, bh) / float(max(bw, bh))
        fill_ratio = area / float(max(1, bw * bh))
        if aspect < 0.58 or fill_ratio < 0.42:
            continue

        analysis_component = (labels == idx).astype(np.uint8)
        contours = contour_from_mask(analysis_component)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        if perimeter <= 0.0 or contour_area <= 0.0:
            continue
        circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 0.0 else 0.0
        if circularity < 0.50 or solidity < 0.72:
            continue

        peak_radius = float(dt[labels == idx].max())
        thickness_ratio = peak_radius / baseline_radius
        if thickness_ratio < 1.85:
            continue

        cx, cy = centroids[idx]
        center_x = int(round(cx))
        center_y = int(round(cy))
        shoulder_idx = int(shoulder_labels[center_y, center_x])
        if shoulder_idx <= 0 or shoulder_idx in seen_shoulder_labels:
            continue

        shoulder_area = int(shoulder_stats[shoulder_idx, cv2.CC_STAT_AREA])
        shoulder_w = int(shoulder_stats[shoulder_idx, cv2.CC_STAT_WIDTH])
        shoulder_h = int(shoulder_stats[shoulder_idx, cv2.CC_STAT_HEIGHT])
        shoulder_aspect = min(shoulder_w, shoulder_h) / float(
            max(shoulder_w, shoulder_h)
        )
        if shoulder_area > max_core_area * 3 or shoulder_aspect < 0.58:
            continue

        analysis_shoulder = (shoulder_labels == shoulder_idx).astype(np.uint8)
        shoulder_contours = contour_from_mask(analysis_shoulder)
        if not shoulder_contours:
            continue
        shoulder_contour = max(shoulder_contours, key=cv2.contourArea)
        shoulder_perimeter = float(cv2.arcLength(shoulder_contour, True))
        shoulder_contour_area = float(cv2.contourArea(shoulder_contour))
        if shoulder_perimeter <= 0.0 or shoulder_contour_area <= 0.0:
            continue
        shoulder_circularity = (
            4.0
            * math.pi
            * shoulder_contour_area
            / (shoulder_perimeter * shoulder_perimeter)
        )
        shoulder_hull = cv2.convexHull(shoulder_contour)
        shoulder_hull_area = float(cv2.contourArea(shoulder_hull))
        shoulder_solidity = (
            shoulder_contour_area / shoulder_hull_area
            if shoulder_hull_area > 0.0
            else 0.0
        )
        if shoulder_circularity < 0.46 or shoulder_solidity < 0.72:
            continue

        seen_shoulder_labels.add(shoulder_idx)
        component = analysis_shoulder[
            analysis_pad : analysis_pad + crop_h,
            analysis_pad : analysis_pad + crop_w,
        ]
        candidates.append(
            {
                "center": (
                    int(round(cx - analysis_pad)),
                    int(round(cy - analysis_pad)),
                ),
                "diameter": peak_radius * 2.0,
                "peak_radius": peak_radius,
                "thickness_ratio": thickness_ratio,
                "circularity": min(circularity, shoulder_circularity),
                "solidity": min(solidity, shoulder_solidity),
                "aspect": min(aspect, shoulder_aspect),
                "component": component,
            }
        )

    result["candidate_count"] = len(candidates)
    result["blob_count"] = len(candidates)

    inlier_candidates: List[Dict[str, object]] = []
    size_cv = 0.0
    if len(candidates) >= 3:
        diameters = np.array(
            [float(candidate["diameter"]) for candidate in candidates],
            dtype=np.float32,
        )
        median_diameter = float(np.median(diameters))
        tolerance = max(2.0, median_diameter * 0.38)
        inlier_candidates = [
            candidate
            for candidate in candidates
            if abs(float(candidate["diameter"]) - median_diameter) <= tolerance
        ]
        if inlier_candidates:
            inlier_sizes = np.array(
                [float(candidate["diameter"]) for candidate in inlier_candidates],
                dtype=np.float32,
            )
            size_cv = float(np.std(inlier_sizes)) / max(
                1.0,
                float(np.mean(inlier_sizes)),
            )

    high_quality_count = sum(
        1
        for candidate in inlier_candidates
        if float(candidate["circularity"]) >= 0.68
        and float(candidate["solidity"]) >= 0.82
        and float(candidate["aspect"]) >= 0.72
        and float(candidate["thickness_ratio"]) >= 2.05
    )
    mean_thickness_ratio = (
        float(
            np.mean(
                [
                    float(candidate["thickness_ratio"])
                    for candidate in inlier_candidates
                ]
            )
        )
        if inlier_candidates
        else 0.0
    )

    repeated_pattern = len(inlier_candidates) >= 4 and size_cv <= 0.40
    strong_three_pattern = (
        len(inlier_candidates) >= 3
        and high_quality_count >= 3
        and size_cv <= 0.28
        and mean_thickness_ratio >= 2.10
    )
    beads = repeated_pattern or strong_three_pattern
    result["candidate_size_cv"] = size_cv
    result["bead_count"] = len(inlier_candidates) if beads else 0
    result["beads_detected"] = beads
    result["risk"] = "High" if beads else "Low"
    if beads:
        result["decision_reason"] = (
            f"{len(inlier_candidates)} repeated compact round expansions "
            "with consistent size."
        )
    elif candidates:
        result["decision_reason"] = (
            f"{len(candidates)} thick regions found, but they were not a "
            "consistent repeated bead pattern."
        )

    vis = full_image[y1:y2 + 1, x1:x2 + 1].copy()
    if beads:
        bead_mask = np.zeros_like(chain_crop, dtype=np.uint8)
        for candidate in inlier_candidates:
            bead_mask |= candidate["component"]
            cx, cy = candidate["center"]
            radius = max(
                4,
                int(round(float(candidate["peak_radius"]) * 1.25)),
            )
            cv2.circle(vis, (cx, cy), radius, (0, 0, 255), 2, cv2.LINE_AA)
        overlay = np.zeros_like(vis)
        overlay[dilate(bead_mask, 3) > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.78, overlay, 0.22, 0)
    result["visualization"] = vis

    return result


def segment_necklace(
    image: np.ndarray,
    primary_mask: np.ndarray,
    sampler: FastSamOnnx,
    args: argparse.Namespace,
    manual_feedback: ManualPartFeedback | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    jewel_type = str(getattr(args, "jewel_type", "") or "").strip()
    authoritative_feedback = bool(
        manual_feedback is not None
        and manual_feedback.match_type in {"exact_signature", "legacy_path"}
    )
    authoritative_pendant_feedback = bool(
        authoritative_feedback
        and manual_feedback is not None
        and (
            manual_feedback.pendant_mask is not None
            or manual_feedback.pendant_bbox is not None
        )
    )
    authoritative_tassel_feedback = bool(
        authoritative_feedback
        and manual_feedback is not None
        and (
            manual_feedback.tassel_mask is not None
            or manual_feedback.tassel_bbox is not None
        )
    )
    detections = sampler.infer(image, args.conf_thres, args.iou_thres, args.mask_thres)
    color_mask, color_maps = estimate_background_mask(image)
    proposal_union, selected_detections = build_proposal_union(detections, image.shape[:2])

    support_mask = dilate(primary_mask.astype(np.uint8), 11) if primary_mask.any() else np.ones_like(color_mask, dtype=np.uint8)
    color_mask &= support_mask
    color_maps["red_mask"] &= support_mask
    color_maps["gold_mask"] &= support_mask
    proposal_union &= support_mask

    refined_mask = refine_necklace_mask(image, color_mask, proposal_union, color_maps)
    necklace_mask = primary_mask.astype(np.uint8).copy()
    if refined_mask.any():
        necklace_mask |= (refined_mask & support_mask)

    manual_pendant_mask = np.zeros_like(primary_mask, dtype=np.uint8)
    manual_tassel_mask = np.zeros_like(primary_mask, dtype=np.uint8)
    manual_keep_mask = np.zeros_like(primary_mask, dtype=np.uint8)
    if manual_feedback is not None:
        if manual_feedback.pendant_mask is not None and manual_feedback.pendant_mask.any():
            manual_pendant_mask = manual_feedback.pendant_mask.astype(np.uint8)
            if manual_pendant_mask.shape != primary_mask.shape:
                manual_pendant_mask = cv2.resize(
                    manual_pendant_mask,
                    (primary_mask.shape[1], primary_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            manual_pendant_mask &= primary_mask.astype(np.uint8)
            if authoritative_feedback:
                manual_keep_mask |= manual_pendant_mask
        elif manual_feedback.pendant_bbox is not None:
            manual_pendant_mask = build_feedback_support_mask(
                primary_mask.astype(np.uint8),
                manual_feedback.pendant_bbox,
                close_ksize=5,
                min_component=max(12, primary_mask.size // 50000),
            )
            if authoritative_feedback:
                manual_keep_mask |= manual_pendant_mask
        if manual_feedback.tassel_mask is not None and manual_feedback.tassel_mask.any():
            manual_tassel_mask = manual_feedback.tassel_mask.astype(np.uint8)
            if manual_tassel_mask.shape != primary_mask.shape:
                manual_tassel_mask = cv2.resize(
                    manual_tassel_mask,
                    (primary_mask.shape[1], primary_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            manual_tassel_mask &= primary_mask.astype(np.uint8)
            if authoritative_feedback:
                manual_keep_mask |= manual_tassel_mask
        elif manual_feedback.tassel_bbox is not None:
            manual_tassel_mask = build_feedback_support_mask(
                primary_mask.astype(np.uint8),
                manual_feedback.tassel_bbox,
                close_ksize=5,
                min_component=max(12, primary_mask.size // 60000),
            )
            if authoritative_feedback:
                manual_keep_mask |= manual_tassel_mask
    if manual_keep_mask.any():
        necklace_mask |= manual_keep_mask

    necklace_mask = close(necklace_mask, 7)
    necklace_mask = remove_small_components(necklace_mask, max(80, necklace_mask.size // 15000))
    if manual_keep_mask.any():
        necklace_mask |= manual_keep_mask
        necklace_mask = close(necklace_mask, 5)
        necklace_mask = largest_or_required_components(necklace_mask, manual_keep_mask)
    elif necklace_mask.any():
        necklace_mask = keep_primary_jewelry_components(necklace_mask)

    tassel_seed, chain_seed_hint, tassel_score = detect_tassel_seed(
        necklace_mask,
        image,
        color_maps.get("textile_mask", color_maps["red_mask"]),
        jewel_type,
    )
    pendant_seed, pendant_score = detect_pendant_seed(
        necklace_mask,
        image,
        jewel_type,
    )
    tassel_locked = False
    pendant_locked = False
    tassel_feedback_rejected_reason: str | None = None
    pendant_feedback_rejected_reason: str | None = None

    if manual_tassel_mask.any():
        feedback_seed = build_tassel_seed_from_feedback(necklace_mask, manual_tassel_mask)
        if feedback_seed.any():
            tassel_feedback_evidence = tassel_candidate_evidence(
                feedback_seed,
                necklace_mask,
                image,
                jewel_type,
            )
            source_type = _normalized_jewel_type(manual_feedback.source_jewel_type)
            current_type = _normalized_jewel_type(jewel_type)
            compatible_type = bool(
                not source_type
                or not current_type
                or source_type == current_type
            )
            if authoritative_feedback or (
                compatible_type and tassel_feedback_evidence["accepted"]
            ):
                tassel_seed = feedback_seed
                tassel_score = (
                    99.0
                    if authoritative_feedback
                    else float(tassel_feedback_evidence["score"])
                )
                tassel_locked = True
            elif not compatible_type:
                tassel_feedback_rejected_reason = (
                    "Similar tassel feedback came from a different jewel type."
                )
            else:
                tassel_feedback_rejected_reason = str(
                    tassel_feedback_evidence["reason"]
                )

    if manual_pendant_mask.any():
        feedback_seed = build_pendant_seed_from_feedback(necklace_mask, manual_pendant_mask)
        if feedback_seed.any():
            pendant_feedback_evidence = pendant_candidate_evidence(
                feedback_seed,
                necklace_mask,
                image,
            )
            if authoritative_feedback or pendant_feedback_evidence["accepted"]:
                pendant_seed = feedback_seed
                pendant_score = (
                    99.0
                    if authoritative_feedback
                    else float(pendant_feedback_evidence["score"])
                )
                pendant_locked = True
            else:
                pendant_feedback_rejected_reason = str(
                    pendant_feedback_evidence["reason"]
                )

    if manual_feedback is not None and manual_feedback.no_pendant:
        pendant_seed = np.zeros_like(necklace_mask, dtype=np.uint8)
        pendant_score = -999.0
        pendant_locked = True
    if manual_feedback is not None and manual_feedback.no_tassel:
        tassel_seed = np.zeros_like(necklace_mask, dtype=np.uint8)
        tassel_score = -999.0
        tassel_locked = True
    no_pendant_requested = bool(
        manual_feedback is not None and manual_feedback.no_pendant
    )
    no_tassel_requested = bool(
        manual_feedback is not None and manual_feedback.no_tassel
    )

    if pendant_locked and pendant_seed.any():
        tassel_seed = (tassel_seed & (1 - pendant_seed)).astype(np.uint8)
    if tassel_locked and tassel_seed.any():
        pendant_seed = (pendant_seed & (1 - tassel_seed)).astype(np.uint8)

    if (
        not pendant_locked
        and not tassel_locked
        and pendant_seed.any()
        and is_likely_tassel(
            pendant_seed,
            necklace_mask,
            image,
            jewel_type,
        )
    ):
        tassel_seed |= pendant_seed
        pendant_seed = np.zeros_like(necklace_mask, dtype=np.uint8)
        pendant_score = -999.0

    chain_seed = chain_seed_hint & (1 - dilate(pendant_seed, 7)) & (1 - dilate(tassel_seed, 5))
    if not pendant_seed.any() and not tassel_seed.any():
        chain_seed = necklace_mask.copy()
    elif not chain_seed.any():
        chain_seed = necklace_mask & (1 - dilate(pendant_seed, 17)) & (1 - dilate(tassel_seed, 17))

    seed_parts = assign_parts_by_seed(necklace_mask, tassel_seed, chain_seed, pendant_seed)
    parts = post_refine_parts(seed_parts, necklace_mask)
    parts = reclassify_pendant_if_tassel(
        parts,
        necklace_mask,
        image,
        lock_pendant=pendant_locked,
        lock_tassel=tassel_locked,
        jewel_type=jewel_type,
    )

    # Voronoi assignment can grow a good seed into an implausibly large region.
    # Keep exact manual corrections authoritative; otherwise re-check the final
    # auxiliary part and fall back to the accepted seed before giving it up.
    if parts["pendant"].any() and not authoritative_pendant_feedback:
        final_pendant_evidence = pendant_candidate_evidence(
            parts["pendant"],
            necklace_mask,
            image,
        )
        if not final_pendant_evidence["accepted"]:
            seed_evidence = pendant_candidate_evidence(
                pendant_seed,
                necklace_mask,
                image,
            )
            parts["pendant"] = (
                pendant_seed.copy()
                if pendant_seed.any() and seed_evidence["accepted"]
                else np.zeros_like(necklace_mask, dtype=np.uint8)
            )

    if parts["tassel"].any() and not authoritative_tassel_feedback:
        final_tassel_evidence = tassel_candidate_evidence(
            parts["tassel"],
            necklace_mask,
            image,
            jewel_type,
        )
        if not final_tassel_evidence["accepted"]:
            seed_evidence = tassel_candidate_evidence(
                tassel_seed,
                necklace_mask,
                image,
                jewel_type,
            )
            parts["tassel"] = (
                tassel_seed.copy()
                if tassel_seed.any() and seed_evidence["accepted"]
                else np.zeros_like(necklace_mask, dtype=np.uint8)
            )

    # Negative feedback is a hard final constraint. No later classification or
    # refinement step may recreate an explicitly excluded auxiliary part.
    if no_pendant_requested:
        parts["pendant"] = np.zeros_like(necklace_mask, dtype=np.uint8)
    if no_tassel_requested:
        parts["tassel"] = np.zeros_like(necklace_mask, dtype=np.uint8)

    parts["tassel"] &= (1 - parts["pendant"])
    parts["chain"] = (
        necklace_mask & (1 - parts["pendant"]) & (1 - parts["tassel"])
    ).astype(np.uint8)

    if not parts["pendant"].any() and not parts["tassel"].any():
        parts["chain"] = necklace_mask.copy()

    chain_mask = parts.get("chain", np.zeros(image.shape[:2], dtype=np.uint8))
    bead_result = detect_chain_beads(chain_mask, image)
    pendant_evidence = pendant_candidate_evidence(
        parts["pendant"],
        necklace_mask,
        image,
    )
    tassel_evidence = tassel_candidate_evidence(
        parts["tassel"],
        necklace_mask,
        image,
        jewel_type,
    )
    debug = {
        "bead_analysis": bead_result,
        "jewel_type": jewel_type,
        "tassel_auto_policy": _tassel_auto_policy(jewel_type),
        "part_detection_prompts": PART_DETECTION_PROMPTS,
        "pendant_detected": bool(parts["pendant"].any()),
        "tassel_detected": bool(parts["tassel"].any()),
        "pendant_evidence": pendant_evidence,
        "tassel_evidence": tassel_evidence,
        "detections": len(detections),
        "selected_detections": len(selected_detections),
        "primary_mask_area": int(primary_mask.sum()),
        "necklace_mask_area": int(necklace_mask.sum()),
        "tassel_seed_area": int(tassel_seed.sum()),
        "chain_seed_area": int(chain_seed.sum()),
        "pendant_seed_area": int(pendant_seed.sum()),
        "tassel_score": round(float(tassel_score), 3),
        "pendant_score": round(float(pendant_score), 3),
        "feedback_found": manual_feedback is not None,
        "feedback_match_type": manual_feedback.match_type if manual_feedback is not None else None,
        "feedback_match_score": round(float(manual_feedback.match_score), 3) if manual_feedback is not None and manual_feedback.match_score is not None else None,
        "feedback_alignment_score": round(float(manual_feedback.alignment_score), 4) if manual_feedback is not None and manual_feedback.alignment_score is not None else None,
        "feedback_file": Path(manual_feedback.feedback_path).name if manual_feedback is not None and manual_feedback.feedback_path else None,
        "feedback_source_jewel_type": manual_feedback.source_jewel_type if manual_feedback is not None else None,
        "manual_feedback_applied": bool(pendant_locked or tassel_locked),
        "manual_feedback_area": int(manual_keep_mask.sum()),
        "tassel_feedback_applied": tassel_locked,
        "tassel_feedback_rejected_reason": tassel_feedback_rejected_reason,
        "tassel_feedback_bbox": list(manual_feedback.tassel_bbox) if tassel_locked and manual_feedback is not None and manual_feedback.tassel_bbox is not None else None,
        "pendant_feedback_applied": pendant_locked,
        "pendant_feedback_rejected_reason": pendant_feedback_rejected_reason,
        "pendant_feedback_bbox": list(manual_feedback.pendant_bbox) if pendant_locked and manual_feedback is not None and manual_feedback.pendant_bbox is not None else None,
        "no_pendant": manual_feedback is not None and manual_feedback.no_pendant,
        "no_tassel": manual_feedback is not None and manual_feedback.no_tassel,
    }
    return parts, debug


def save_outputs(
    image: np.ndarray,
    parts: Dict[str, np.ndarray],
    debug: Dict[str, object],
    output_dir: Path,
    primary_mask: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bead_result = debug.get("bead_analysis")
    layout = build_layout(image, parts, bead_result)
    cv2.imwrite(str(output_dir / "composite_layout.png"), layout)
    cv2.imwrite(str(output_dir / "input_preprocessed.png"), image)
    if primary_mask is not None:
        save_mask_png(primary_mask, output_dir / "input_mask.png")

    necklace_overlay = image.copy()
    for idx, (part_key, _, color) in enumerate(PARTS, start=1):
        mask = parts[part_key]
        if mask.any():
            color_arr = np.array(color, dtype=np.uint8)
            necklace_overlay[mask > 0] = (
                0.58 * necklace_overlay[mask > 0].astype(np.float32) + 0.42 * color_arr.astype(np.float32)
            ).astype(np.uint8)
            draw_dashed_contour(necklace_overlay, mask, color, thickness=2)
    cv2.imwrite(str(output_dir / "overlay.png"), necklace_overlay)

    part_summaries = {}
    for part_key, _, _ in PARTS:
        mask = parts[part_key]
        cutout = make_cutout(image, mask)
        cv2.imwrite(str(output_dir / f"{part_key}_cutout.png"), cutout)
        save_mask_png(mask, output_dir / f"{part_key}_mask.png")
        part_summaries[part_key] = mask_summary(mask)

    bead_analysis_data = debug.get("bead_analysis")
    if bead_analysis_data is not None:
        bead_vis = bead_analysis_data.pop("visualization", None)
        if bead_vis is not None:
            cv2.imwrite(str(output_dir / "bead_analysis.png"), bead_vis)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump({"parts": part_summaries, "debug": debug}, fp, indent=2, default=str)


def get_output_dir(image_path: Path, base_output_dir: str | Path) -> Path:
    return Path(base_output_dir) / image_path.stem


def run_segmentation(
    image_path: Path,
    model_path: Path,
    sampler: FastSamOnnx,
    args: argparse.Namespace,
    preprocessed_image: np.ndarray | None = None,
    preprocessed_mask: np.ndarray | None = None,
) -> Tuple[Path, Dict[str, object]]:
    if preprocessed_image is not None:
        if preprocessed_mask is None:
            gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
            preprocessed_mask = (gray < 250).astype(np.uint8)
        if preprocessed_mask.shape[:2] != preprocessed_image.shape[:2]:
            preprocessed_mask = cv2.resize(
                preprocessed_mask,
                (preprocessed_image.shape[1], preprocessed_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        prepared = PreparedInput(
            original_image=preprocessed_image,
            working_image=preprocessed_image,
            working_mask=(preprocessed_mask > 0).astype(np.uint8),
        )
    else:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        prepared = prepare_segmentation_input(raw_image)
    feedback = load_manual_feedback(
        image_path,
        Path(args.feedback_dir),
        prepared.working_image.shape[:2],
        prepared.working_image,
        prepared.working_mask,
    )
    output_dir = get_output_dir(image_path, args.output_dir)
    parts, debug = segment_necklace(
        prepared.working_image,
        prepared.working_mask,
        sampler,
        args,
        manual_feedback=feedback,
    )
    save_outputs(prepared.working_image, parts, debug, output_dir, prepared.working_mask)
    return output_dir, debug


def build_preview_image(image_path: Path, max_width: int = 920, max_height: int = 700) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return image


def launch_gui(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    sampler = FastSamOnnx(model_path, providers=args.providers, input_size=args.input_size)
    feedback_dir = Path(args.feedback_dir)

    root = tk.Tk()
    root.title("Jewelry Segmentation")
    root.geometry("1180x860")
    root.minsize(900, 700)

    selected_image = tk.StringVar(value="")
    status_text = tk.StringVar(value="Choose a jewelry image to start segmentation.")
    result_text = tk.StringVar(value="")

    def set_preview(preview_path: Path) -> None:
        preview = build_preview_image(preview_path)
        preview_photo = ImageTk.PhotoImage(preview)
        image_panel.configure(image=preview_photo, text="")
        image_panel.image = preview_photo

    def run_selected_segmentation() -> Tuple[Path, Dict[str, object]]:
        if not selected_image.get():
            raise RuntimeError("Please choose an input image first.")
        image_path = Path(selected_image.get())
        output_dir, debug = run_segmentation(image_path, model_path, sampler, args)
        preview_path = output_dir / "composite_layout.png"
        set_preview(preview_path)
        result_text.set(f"Saved to: {preview_path}")
        feedback_note = ""
        if debug.get("manual_feedback_applied"):
            match_type = str(debug.get("feedback_match_type") or "")
            if match_type == "similar_signature":
                feedback_note = " with similar-image feedback"
            else:
                feedback_note = " with saved feedback"
        status_text.set(
            f"Done{feedback_note}. detections={debug['detections']}, selected={debug['selected_detections']}, area={debug['necklace_mask_area']}"
        )
        return output_dir, debug

    def process_selected() -> None:
        if not selected_image.get():
            messagebox.showinfo("No image selected", "Please choose an input image first.")
            return
        try:
            status_text.set("Running segmentation...")
            root.update_idletasks()
            run_selected_segmentation()
        except Exception as exc:
            status_text.set("Segmentation failed.")
            messagebox.showerror("Segmentation error", str(exc))

    def correct_pendant() -> None:
        if not selected_image.get():
            messagebox.showinfo("No image selected", "Please choose an input image first.")
            return

        image_path = Path(selected_image.get())
        output_dir = get_output_dir(image_path, args.output_dir)
        preprocessed_path = output_dir / "input_preprocessed.png"

        try:
            if not preprocessed_path.exists():
                status_text.set("Preparing image for manual pendant correction...")
                root.update_idletasks()
                run_selected_segmentation()

            preprocessed = cv2.imread(str(preprocessed_path), cv2.IMREAD_COLOR)
            if preprocessed is None:
                raise RuntimeError(f"Could not read preprocessed image: {preprocessed_path}")
            preprocessed_mask = cv2.imread(
                str(preprocessed_path.with_name("input_mask.png")),
                cv2.IMREAD_GRAYSCALE,
            )
            if preprocessed_mask is not None:
                preprocessed_mask = (preprocessed_mask > 0).astype(np.uint8)

            messagebox.showinfo(
                "Correct Pendant",
                "Draw a box around the correct pendant, then press Enter or Space. Press C to cancel.",
            )
            window_name = "Draw Pendant Correction"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            x, y, w, h = cv2.selectROI(window_name, preprocessed, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name)
            if w <= 0 or h <= 0:
                status_text.set("Pendant correction cancelled.")
                return

            bbox = (int(x), int(y), int(x + w - 1), int(y + h - 1))
            feedback_path = save_pendant_feedback(
                image_path,
                feedback_dir,
                bbox,
                preprocessed.shape[:2],
                prepared_image=preprocessed,
                prepared_mask=preprocessed_mask,
            )
            status_text.set("Pendant correction saved. Re-running segmentation...")
            root.update_idletasks()
            run_selected_segmentation()
            result_text.set(f"Saved to: {output_dir / 'composite_layout.png'} | Feedback: {feedback_path.name}")
        except Exception as exc:
            status_text.set("Pendant correction failed.")
            try:
                cv2.destroyWindow("Draw Pendant Correction")
            except cv2.error:
                pass
            messagebox.showerror("Pendant correction error", str(exc))

    def clear_pendant_correction() -> None:
        if not selected_image.get():
            messagebox.showinfo("No image selected", "Please choose an input image first.")
            return

        image_path = Path(selected_image.get())
        output_dir = get_output_dir(image_path, args.output_dir)
        preprocessed_path = output_dir / "input_preprocessed.png"
        try:
            prepared_image = None
            prepared_mask = None
            if preprocessed_path.exists():
                prepared_image = cv2.imread(str(preprocessed_path), cv2.IMREAD_COLOR)
                prepared_mask = cv2.imread(
                    str(preprocessed_path.with_name("input_mask.png")),
                    cv2.IMREAD_GRAYSCALE,
                )
                if prepared_mask is not None:
                    prepared_mask = (prepared_mask > 0).astype(np.uint8)
            removed = clear_pendant_feedback(
                image_path,
                feedback_dir,
                prepared_image=prepared_image,
                prepared_mask=prepared_mask,
            )
            if not removed:
                status_text.set("No saved pendant correction found for this image.")
                return
            status_text.set("Saved pendant correction cleared. Re-running segmentation...")
            root.update_idletasks()
            output_dir, _ = run_selected_segmentation()
            result_text.set(f"Saved to: {output_dir / 'composite_layout.png'}")
        except Exception as exc:
            status_text.set("Could not clear pendant correction.")
            messagebox.showerror("Clear correction error", str(exc))

    def choose_image() -> None:
        file_path = filedialog.askopenfilename(
            title="Choose input jewelry image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        selected_image.set(file_path)
        status_text.set("Image selected. Click Run Segmentation.")
        result_text.set("")
        try:
            set_preview(Path(file_path))
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    controls = ttk.Frame(root, padding=(12, 12, 12, 0))
    controls.pack(fill="x")

    ttk.Label(controls, text="Input Image:").grid(row=0, column=0, sticky="w")
    ttk.Entry(controls, textvariable=selected_image, width=85).grid(row=0, column=1, padx=8, sticky="ew")
    ttk.Button(controls, text="Browse", command=choose_image).grid(row=0, column=2, padx=(0, 8))
    ttk.Button(controls, text="Run Segmentation", command=process_selected).grid(row=0, column=3, padx=(0, 8))
    ttk.Button(controls, text="Correct Pendant", command=correct_pendant).grid(row=0, column=4, padx=(0, 8))
    ttk.Button(controls, text="Clear Feedback", command=clear_pendant_correction).grid(row=0, column=5)
    controls.columnconfigure(1, weight=1)

    info_bar = ttk.Frame(root, padding=(12, 4, 12, 0))
    info_bar.pack(fill="x")
    ttk.Label(info_bar, textvariable=status_text).pack(anchor="w")
    tk.Label(info_bar, textvariable=result_text, fg="#2d7d2d", anchor="w").pack(anchor="w", pady=(2, 0))

    image_panel = ttk.Label(root, text="Result preview will appear here.", anchor="center")
    image_panel.pack(fill="both", expand=True, padx=12, pady=(8, 12))

    if args.image:
        selected_image.set(str(Path(args.image)))
        try:
            set_preview(Path(args.image))
            status_text.set("Image loaded. Click Run Segmentation.")
        except Exception:
            status_text.set("Choose a jewelry image to start segmentation.")

    root.mainloop()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.gui or not args.image:
        launch_gui(args)
        return

    image_path = Path(args.image)
    sampler = FastSamOnnx(model_path, providers=args.providers, input_size=args.input_size)
    output_dir, debug = run_segmentation(image_path, model_path, sampler, args)
    print(f"Saved outputs to: {output_dir}")
    print(json.dumps(debug, indent=2))


if __name__ == "__main__":
    main()
