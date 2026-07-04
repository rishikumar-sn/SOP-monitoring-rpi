from __future__ import annotations
"""
run_local2.py - Standalone Gold Testing System (Hailo + rubbing audio)
Updated with clear Visual + Audio Sync Status
"""

import cv2
import numpy as np
import os
import sys
import json
import threading
import queue
import time
import logging
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import scipy.signal

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    print("[WARN] sounddevice not installed. Audio detection disabled.")
    SOUNDDEVICE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("[WARN] librosa not installed.")
    LIBROSA_AVAILABLE = False

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    PYQT_AVAILABLE = True
except ImportError:
    print("[WARN] PyQt5/pyqtgraph not installed. GUI disabled.")
    QtCore = SimpleNamespace()
    QtGui = SimpleNamespace(QCloseEvent=object)
    QtWidgets = SimpleNamespace(QMainWindow=object)
    pg = None
    PYQT_AVAILABLE = False

AUDIO_AVAILABLE = SOUNDDEVICE_AVAILABLE and LIBROSA_AVAILABLE

try:
    from hailo_platform import HEF, VDevice, FormatType
    try:
        from hailo_platform import HailoSchedulingAlgorithm
    except Exception:
        HailoSchedulingAlgorithm = None
    HAILO_AVAILABLE = True
except ImportError:
    HEF = None
    VDevice = None
    FormatType = None
    HailoSchedulingAlgorithm = None
    HAILO_AVAILABLE = False

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ====================== PATHS ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_model_path(*candidates: str) -> str:
    for rel in candidates:
        p = os.path.join(BASE_DIR, rel)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, candidates[0])

MODEL_GOLD_PATH = resolve_model_path("models/gold.hef")
MODEL_STONE_PATH = resolve_model_path("models/yolov8s_seg.hef")
MODEL_ACID_PATH = resolve_model_path("models/bestnewacid.hef")
SOUND_MODEL_DIR = resolve_model_path("new_audio_rubbing/models")
SOUND_MODEL_PATH = resolve_model_path(
    "new_audio_rubbing/models/gold_rub_cnn.tflite",
    "new_audio_rubbing/models/gold_rub_cnn.keras",
)

# ====================== CONFIG ======================
CAMERA_ID = 0
FRAME_W = 640
FRAME_H = 480
INFER_SKIP = 2

CAM_ROTATE_90_CLOCKWISE = os.environ.get("CAM_ROTATE_90_CLOCKWISE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    CAM_MAX_PROBE_DIM = max(2048, int(os.environ.get("CAM_MAX_PROBE_DIM", "10000")))
except Exception:
    CAM_MAX_PROBE_DIM = 10000
try:
    CAM_PROBE_GRABS = max(0, int(os.environ.get("CAM_PROBE_GRABS", "3")))
except Exception:
    CAM_PROBE_GRABS = 3
ROI_CONFIG_FILE = os.path.join(BASE_DIR, "roi_config.json")
ROI_CONFIG = {"x1": None, "y1": None, "x2": None, "y2": None}
ROI_CONFIG_LOCK = threading.Lock()
ROI_CONFIG_LAST_MODIFIED = 0.0

STONE_CONF_THRESH = 0.20
STONE_IOU_THRESH = 0.45
STONE_MIN_AREA_RATIO = 0.02

GOLD_CONF_THRESH = 0.70
GOLD_IOU_THRESH = 0.45
GOLD_MASK_THRESH = 0.45
GOLD_MIN_MASK_PIXELS = 80
GOLD_MIN_OVERLAP_PIXELS = 40
GOLD_CROP_PAD_RATIO = 0.90
GOLD_MIN_INSIDE_STONE_RATIO = 0.50
try:
    GOLD_FULL_FRAME_FALLBACK_EVERY = max(
        1,
        int(os.environ.get("GOLD_FULL_FRAME_FALLBACK_EVERY", "6")),
    )
except (TypeError, ValueError):
    GOLD_FULL_FRAME_FALLBACK_EVERY = 6

ACID_CONF_THRESH = 0.90
ACID_IOU_THRESH = 0.45
ACID_MIN_AREA_PX = 140
ACID_CONFIRM_FRAMES = 3

# ====================== IMPROVED AUDIO & SYNC ======================
AUDIO_WINDOW_SEC = 1.0
AUDIO_HOP_RATIO = 0.3
AUDIO_CONF_THRESH = 0.90
AUDIO_OK_STREAK_REQUIRED = 1
AUDIO_NOK_STREAK_REQUIRED = 1
AUDIO_SYNC_WINDOW_SEC = 1.5
RUBBING_SYNC_CONFIRM_FRAMES = 2
RUBBING_MIN_CENTROID_MOVE = 0.5
GOLD_AUDIO_GRACE_SEC = 2.0

AUDIO_DEVICE_KEYWORDS = ("walmart ab13x", "ab13x", "usb audio", "headset", "adapter")
AUDIO_WEBCAM_KEYWORDS = ("brio", "logitech", "webcam", "camera")
AUDIO_DEVICE_AUTO = "__AUTO__"
AUDIO_DEVICE_NAME_PREFIX = "name:"

AUDIO_N_MELS = 64
JEWEL_RUB_OK_LABEL = "JEWEL_RUB_OK"
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_PEAK_THRESHOLD = 0.03
SILENCE_ACTIVE_THRESHOLD = 0.02
SILENCE_MIN_ACTIVE_RATIO = 0.02

STONE_BOX_COLOR = (0, 0, 255)
GOLD_OVERLAY_COLOR = (0, 215, 255)

# ====================== GLOBAL STATE ======================
STATE = {
    "stage": "RUBBING",
    "rubbing_done": False,
    "recent_distances": deque(maxlen=10),
    "prev_centroid": None,
    "sound_status": "Waiting...",
    "audio_label": "Waiting...",
    "audio_decision": "Waiting...",
    "audio_confidence": 0.0,
    "audio_probabilities": {},
    "acid_positive_streak": 0,
    "last_stone_bbox": None,
    "stone_visible_now": False,
    "audio_ok_recent_until": 0.0,
    "gold_detected_recent_until": 0.0,
    "visual_rubbing_recent_until": 0.0,
    "gold_visible_now": False,
    "rubbing_sync_hits": 0,
    "last_rubbing_bbox": None,
    "last_rubbing_mask": None,
    "last_acid_bbox": None,
    "gold_full_frame_miss_count": 0,
}

def reset_state() -> None:
    STATE["stage"] = "RUBBING"
    STATE["rubbing_done"] = False
    STATE["recent_distances"].clear()
    STATE["prev_centroid"] = None
    STATE["sound_status"] = "Waiting..."
    STATE["audio_label"] = "Waiting..."
    STATE["audio_decision"] = "Waiting..."
    STATE["audio_confidence"] = 0.0
    STATE["audio_probabilities"] = {}
    STATE["acid_positive_streak"] = 0
    STATE["last_stone_bbox"] = None
    STATE["stone_visible_now"] = False
    STATE["audio_ok_recent_until"] = 0.0
    STATE["gold_detected_recent_until"] = 0.0
    STATE["visual_rubbing_recent_until"] = 0.0
    STATE["gold_visible_now"] = False
    STATE["rubbing_sync_hits"] = 0
    STATE["last_rubbing_bbox"] = None
    STATE["last_rubbing_mask"] = None
    STATE["last_acid_bbox"] = None
    STATE["gold_full_frame_miss_count"] = 0
    logger.info("State reset to RUBBING.")

# ====================== VIDEO CAPTURE ======================
def _read_camera_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    except Exception:
        return 0, 0


def _display_camera_resolution(width: int, height: int) -> Tuple[int, int]:
    if CAM_ROTATE_90_CLOCKWISE and width > 0 and height > 0:
        return height, width
    return width, height


def _request_camera_resolution(cap: cv2.VideoCapture, width: int, height: int, warmup_grabs: int = 0) -> Tuple[int, int]:
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    except Exception:
        return _read_camera_resolution(cap)

    for _ in range(max(0, int(warmup_grabs or 0))):
        try:
            cap.grab()
        except Exception:
            break
    return _read_camera_resolution(cap)


def _best_effort_set_max_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
    candidates = [
        (4032, 3024),
        (3840, 2160),
        (3264, 2448),
        (2592, 1944),
        (2560, 1440),
        (2304, 1536),
        (2048, 1536),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]
    best = _read_camera_resolution(cap)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    probe_dim = max(2048, int(CAM_MAX_PROBE_DIM or 10000))
    rw, rh = _request_camera_resolution(cap, probe_dim, probe_dim, warmup_grabs=CAM_PROBE_GRABS)
    if rw * rh > best[0] * best[1]:
        best = rw, rh

    if best[0] * best[1] <= 640 * 480:
        for w, h in candidates:
            rw, rh = _request_camera_resolution(cap, w, h, warmup_grabs=1)
            if rw > 0 and rh > 0 and rw * rh > best[0] * best[1]:
                best = rw, rh

    if best[0] > 0 and best[1] > 0:
        best = _request_camera_resolution(cap, best[0], best[1], warmup_grabs=1)
    return best


def _transform_camera_frame(frame: np.ndarray) -> np.ndarray:
    if CAM_ROTATE_90_CLOCKWISE:
        try:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        except Exception:
            return frame
    return frame


def _load_roi_config(verbose: bool = True) -> bool:
    global ROI_CONFIG, ROI_CONFIG_LAST_MODIFIED
    if not os.path.exists(ROI_CONFIG_FILE):
        with ROI_CONFIG_LOCK:
            ROI_CONFIG = {"x1": None, "y1": None, "x2": None, "y2": None}
            ROI_CONFIG_LAST_MODIFIED = 0.0
        if verbose:
            logger.info("[ROI] Config file not found: %s", ROI_CONFIG_FILE)
        return False

    try:
        current_mtime = os.path.getmtime(ROI_CONFIG_FILE)
        if current_mtime == ROI_CONFIG_LAST_MODIFIED and ROI_CONFIG.get("x1") is not None:
            return True
        with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        roi = config.get("roi", {})
        if not all(isinstance(roi.get(k), (int, float, type(None))) for k in ("x1", "y1", "x2", "y2")):
            raise ValueError("ROI fields must be numeric or null")
        if not all(roi.get(k) is not None for k in ("x1", "y1", "x2", "y2")):
            raise ValueError("ROI is incomplete")
        roi_int = {k: int(roi[k]) for k in ("x1", "y1", "x2", "y2")}
        with ROI_CONFIG_LOCK:
            ROI_CONFIG = roi_int
            ROI_CONFIG_LAST_MODIFIED = current_mtime
        if verbose:
            logger.info(
                "[ROI] Loaded %s | size=%sx%s",
                roi_int,
                roi_int["x2"] - roi_int["x1"],
                roi_int["y2"] - roi_int["y1"],
            )
        return True
    except Exception as e:
        if verbose:
            logger.warning("[ROI] Could not load %s: %s", ROI_CONFIG_FILE, e)
        return False


def _apply_roi(frame: np.ndarray) -> np.ndarray:
    with ROI_CONFIG_LOCK:
        roi = dict(ROI_CONFIG)
    if roi.get("x1") is None or not all(v is not None for v in roi.values()):
        return frame

    x1 = max(0, int(roi["x1"]))
    y1 = max(0, int(roi["y1"]))
    x2 = min(frame.shape[1], int(roi["x2"]))
    y2 = min(frame.shape[0], int(roi["y2"]))
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    return frame


class VideoStreamWidget:
    def __init__(self, src: int = 0, w: int = 640, h: int = 480):
        if os.name == "nt":
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(src)
        cap_w, cap_h = _best_effort_set_max_resolution(self.capture)
        show_w, show_h = _display_camera_resolution(cap_w, cap_h)
        logger.info("[Camera] Capture opened at %sx%s, processing frame=%sx%s", cap_w, cap_h, show_w, show_h)
        _load_roi_config(verbose=True)
        self.ret, self.frame = self.capture.read()
        if self.ret and self.frame is not None:
            self.frame = _apply_roi(_transform_camera_frame(self.frame))
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self) -> None:
        while self.running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    frame = _apply_roi(_transform_camera_frame(frame))
                    self.ret, self.frame = True, frame
                else:
                    self.ret = False
            time.sleep(0.01)

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.ret, self.frame.copy() if self.frame is not None else self.frame

    def release(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()
        
# ====================== HAILO HELPERS ======================
def _format_type_to_numpy(format_type: Any) -> np.dtype:
    name = str(format_type).split(".")[-1].upper()
    mapping = {
        "FLOAT32": np.float32,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "INT8": np.int8,
        "INT16": np.int16,
    }
    return mapping.get(name, np.float32)


def _letterbox(image: np.ndarray, new_size: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Resize image with unchanged aspect ratio and pad to model input size.
    Returns padded image and metadata for restoring boxes/masks to original frame.
    """
    target_h, target_w = new_size
    src_h, src_w = image.shape[:2]

    if src_h == 0 or src_w == 0:
        raise ValueError("Invalid frame shape for letterbox")

    scale = min(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    meta = {
        "scale": float(scale),
        "left": int(left),
        "top": int(top),
        "resized_w": int(resized_w),
        "resized_h": int(resized_h),
        "input_w": int(target_w),
        "input_h": int(target_h),
    }
    return padded, meta


def _normalize_bbox_xyxy(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _clip_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(round(x1), 0, max(0, w - 1)))
    y1 = int(np.clip(round(y1), 0, max(0, h - 1)))
    x2 = int(np.clip(round(x2), 0, max(0, w - 1)))
    y2 = int(np.clip(round(y2), 0, max(0, h - 1)))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


def _simple_nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45, max_dets: int = 300) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    while order.size > 0 and len(keep) < max_dets:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_thresh]

    return np.asarray(keep, dtype=np.int32)


def _mz_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)


def _mz_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _mz_xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _mz_crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    n_masks, _, _ = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1)
    for k in range(n_masks):
        masks[k, : y1[k, 0], :] = 0
        masks[k, y2[k, 0] :, :] = 0
        masks[k, :, : x1[k, 0]] = 0
        masks[k, :, x2[k, 0] :] = 0
    return masks


def _mz_process_mask(protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray, shape: Tuple[int, int], upsample: bool = True) -> Optional[np.ndarray]:
    mh, mw, c = protos.shape
    ih, iw = shape
    masks = _mz_sigmoid(masks_in @ protos.reshape((-1, c)).transpose((1, 0))).reshape((-1, mh, mw))

    if upsample:
        if not masks.shape[0]:
            return None
        masks = cv2.resize(np.transpose(masks, axes=(1, 2, 0)), shape, interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        masks = np.transpose(masks, axes=(2, 0, 1))

    masks = _mz_crop_mask(masks, bboxes)
    return masks


def _mz_non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nm: int = 32,
    multi_label: bool = True,
) -> List[Dict[str, np.ndarray]]:
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    mi = 5 + nc
    output = []

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4), dtype=np.float32),
                    "mask": np.zeros((0, nm), dtype=np.float32),
                    "detection_classes": np.zeros((0,), dtype=np.int32),
                    "detection_scores": np.zeros((0,), dtype=np.float32),
                }
            )
            continue

        x[:, 5:] *= x[:, 4:5]
        boxes = _mz_xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        multi_label = bool(multi_label and nc > 1)
        if not multi_label:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)
            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate((boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1)

        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4), dtype=np.float32),
                    "mask": np.zeros((0, nm), dtype=np.float32),
                    "detection_classes": np.zeros((0,), dtype=np.int32),
                    "detection_scores": np.zeros((0,), dtype=np.float32),
                }
            )
            continue

        x = x[x[:, 4].argsort()[::-1]]
        cls_shift = x[:, 5:6] * max_wh
        boxes_for_nms = x[:, :4] + cls_shift
        conf = x[:, 4]
        keep = _simple_nms_indices(boxes_for_nms.astype(np.float32), conf.astype(np.float32), iou_thres, max_det)
        out = x[keep]

        output.append(
            {
                "detection_boxes": out[:, :4].astype(np.float32),
                "mask": out[:, 6:].astype(np.float32),
                "detection_classes": out[:, 5].astype(np.int32),
                "detection_scores": out[:, 4].astype(np.float32),
            }
        )

    return output


def _mz_yolov8_decoding(raw_boxes: List[np.ndarray], strides: List[int], image_dims: Tuple[int, int], reg_max: int) -> np.ndarray:
    boxes = None
    for box_distribute, stride in zip(raw_boxes, strides):
        shape = [int(x / stride) for x in image_dims]
        grid_x = np.arange(shape[1]) + 0.5
        grid_y = np.arange(shape[0]) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        box_distribute = np.reshape(
            box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1)
        )
        box_distance = _mz_softmax(box_distribute)
        reg_range = np.arange(reg_max + 1, dtype=np.float32)
        box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
        box_distance = np.sum(box_distance, axis=-1) * stride

        box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
        decode_box = np.expand_dims(center, axis=0) + box_distance

        xmin = decode_box[:, :, 0]
        ymin = decode_box[:, :, 1]
        xmax = decode_box[:, :, 2]
        ymax = decode_box[:, :, 3]
        xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes


def _mz_order_yolov8_seg_endnodes(raw_output: Dict[str, np.ndarray]) -> Optional[Tuple[List[np.ndarray], Dict[str, Any]]]:
    entries = []
    for name_, tensor in raw_output.items():
        if not isinstance(tensor, np.ndarray) or tensor.ndim != 3:
            continue
        entries.append((name_, tensor.astype(np.float32, copy=False)))

    if len(entries) < 10:
        return None

    proto_name, proto_tensor = max(entries, key=lambda item: (item[1].shape[0] * item[1].shape[1], -abs(item[1].shape[2] - 32)))
    proto_channels = proto_tensor.shape[2]

    grouped: Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]] = {}
    for name_, tensor in entries:
        if name_ == proto_name:
            continue
        grouped.setdefault(tensor.shape[:2], []).append((name_, tensor))

    ordered_scales = sorted(grouped.keys(), key=lambda item: item[0])
    if len(ordered_scales) != 3:
        return None

    endnodes: List[np.ndarray] = []
    score_channels = None
    reg_max = None
    feature_strides: List[int] = []

    for (h, w) in ordered_scales:
        tensors = grouped[(h, w)]
        if len(tensors) != 3:
            return None

        bbox_candidates = [(n, t) for n, t in tensors if t.shape[2] % 4 == 0 and t.shape[2] > proto_channels]
        if not bbox_candidates:
            return None
        bbox_name, bbox_tensor = max(bbox_candidates, key=lambda item: item[1].shape[2])

        remaining = [(n, t) for n, t in tensors if n != bbox_name]
        coeff_candidates = [(n, t) for n, t in remaining if t.shape[2] == proto_channels]
        if coeff_candidates:
            coeff_name, coeff_tensor = coeff_candidates[0]
        else:
            coeff_name, coeff_tensor = min(remaining, key=lambda item: abs(item[1].shape[2] - proto_channels))

        score_name = [n for n, _ in remaining if n != coeff_name]
        if len(score_name) != 1:
            return None
        score_name = score_name[0]
        score_tensor = [t for n, t in remaining if n == score_name][0]

        endnodes.extend([
            np.expand_dims(bbox_tensor, axis=0),
            np.expand_dims(score_tensor, axis=0),
            np.expand_dims(coeff_tensor, axis=0),
        ])

        score_channels = int(score_tensor.shape[2])
        reg_max = int(bbox_tensor.shape[2] // 4) - 1
        feature_strides.append(int(proto_tensor.shape[0] * 4 // h))

    endnodes.append(np.expand_dims(proto_tensor, axis=0))
    meta = {
        "classes": int(score_channels),
        "regression_length": int(reg_max),
        "proto_channels": int(proto_channels),
        "feature_strides": list(reversed(feature_strides)),
    }
    return endnodes, meta


def _mz_yolov8_seg_postprocess(
    endnodes: List[np.ndarray],
    img_dims: Tuple[int, int],
    classes: int,
    regression_length: int,
    strides: List[int],
    score_threshold: float,
    nms_iou_thresh: float,
    max_dets: int,
    include_masks: bool = True,
) -> List[Dict[str, Any]]:
    raw_boxes = endnodes[:7:3]
    score_tensors = [np.reshape(s, (-1, s.shape[1] * s.shape[2], classes)) for s in endnodes[1:8:3]]
    scores = np.concatenate(score_tensors, axis=1)
    if np.min(scores) < 0.0 or np.max(scores) > 1.0:
        scores = _mz_sigmoid(scores)

    decoded_boxes = _mz_yolov8_decoding(raw_boxes, list(reversed(strides)), img_dims, regression_length)
    proto_data = endnodes[9]
    batch_size, _, _, n_masks = proto_data.shape

    fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1), dtype=np.float32)
    scores_obj = np.concatenate([fake_objectness, scores], axis=-1)

    coeff_tensors = [np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]]
    coeffs = np.concatenate(coeff_tensors, axis=1)

    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
    nms_res = _mz_non_max_suppression(
        predictions,
        conf_thres=score_threshold,
        iou_thres=nms_iou_thresh,
        max_det=max_dets,
        nm=n_masks,
        multi_label=classes > 1,
    )

    outputs = []
    for batch_idx in range(batch_size):
        boxes = nms_res[batch_idx]["detection_boxes"]
        masks = (
            _mz_process_mask(
                proto_data[batch_idx],
                nms_res[batch_idx]["mask"],
                boxes,
                img_dims,
                upsample=True,
            )
            if include_masks
            else None
        )

        output = {
            "detection_boxes": np.array(boxes, dtype=np.float32) / np.tile((img_dims[1], img_dims[0]), 2),
            "detection_scores": np.array(nms_res[batch_idx]["detection_scores"], dtype=np.float32),
            "detection_classes": np.array(nms_res[batch_idx]["detection_classes"], dtype=np.int32),
            "mask": np.transpose(masks, (0, 1, 2)) if masks is not None else None,
        }
        outputs.append(output)

    return outputs


class HailoHEFModel:
    """Small wrapper around a single Hailo HEF model for synchronous single-frame inference."""

    def __init__(self, vdevice: VDevice, hef_path: str, name: str):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF not found for {name}: {hef_path}")

        self.name = name
        self.hef_path = hef_path
        self.hef = HEF(hef_path)

        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.input_shape = tuple(self.input_info.shape)
        if len(self.input_shape) != 3:
            raise RuntimeError(f"[{self.name}] Unsupported input shape: {self.input_shape}")

        if self.input_shape[2] in (1, 3, 4):
            self.input_layout = "HWC"
            self.input_h, self.input_w, self.input_c = self.input_shape
        elif self.input_shape[0] in (1, 3, 4):
            self.input_layout = "CHW"
            self.input_c, self.input_h, self.input_w = self.input_shape
        else:
            raise RuntimeError(f"[{self.name}] Could not infer input layout from shape {self.input_shape}")

        self.native_input_dtype = _format_type_to_numpy(
            self.input_info.format.type
        )
        requested_input_format = os.environ.get(
            "HAILO_INPUT_FORMAT",
            "native",
        ).strip().lower()
        if requested_input_format not in ("native", "uint8", "float32"):
            raise ValueError(
                "HAILO_INPUT_FORMAT must be native, uint8, or float32; "
                f"received {requested_input_format!r}."
            )
        self.input_dtype = self.native_input_dtype

        if requested_input_format == "float32":
            try:
                self.infer_model.input().set_format_type(FormatType.FLOAT32)
                self.input_dtype = np.float32
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.name}] Could not configure FLOAT32 input buffers: {exc}"
                ) from exc
        elif requested_input_format == "uint8":
            try:
                self.infer_model.input().set_format_type(FormatType.UINT8)
                self.input_dtype = np.uint8
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.name}] Could not configure UINT8 input buffers: {exc}"
                ) from exc
        elif self.input_dtype != np.float32:
            try:
                self.infer_model.input().set_format_type(FormatType.UINT8)
                self.input_dtype = np.uint8
            except Exception:
                pass
        else:
            try:
                self.infer_model.input().set_format_type(FormatType.FLOAT32)
                self.input_dtype = np.float32
            except Exception:
                pass

        self.output_infos = list(self.hef.get_output_vstream_infos())
        self.output_names = [info.name for info in self.output_infos]

        self.output_dtypes: Dict[str, np.dtype] = {}
        for info in self.output_infos:
            name_ = info.name
            dtype = _format_type_to_numpy(info.format.type)
            try:
                self.infer_model.output(name_).set_format_type(FormatType.FLOAT32)
                dtype = np.float32
            except Exception:
                pass
            self.output_dtypes[name_] = dtype

        self._config_ctx = self.infer_model.configure()
        self.configured_model = self._config_ctx.__enter__()

        self._logged_output_schema = False
        self._logged_decoder = False
        self._warned_no_decode = False
        self._empty_predict_counter = 0

        logger.info(
            "[%s] HEF loaded: %s | input=%s(%s,%s,%s) | "
            "native_dtype=%s host_dtype=%s | outputs=%s",
            self.name,
            os.path.basename(self.hef_path),
            self.input_layout,
            self.input_h,
            self.input_w,
            self.input_c,
            np.dtype(self.native_input_dtype),
            np.dtype(self.input_dtype),
            self.output_names,
        )

    def _prepare_input(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        model_input, meta = _letterbox(frame_bgr, (self.input_h, self.input_w))

        if self.input_c == 1:
            model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2GRAY)
            model_input = np.expand_dims(model_input, axis=-1)
        elif self.input_c == 3:
            # Match the original Ultralytics preprocessing path, which converts OpenCV BGR frames to RGB.
            model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        elif self.input_c == 4:
            model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGBA)

        if self.input_layout == "CHW":
            model_input = np.transpose(model_input, (2, 0, 1))

        if self.input_dtype == np.float32:
            model_input = model_input.astype(np.float32) / 255.0
        else:
            model_input = model_input.astype(self.input_dtype)

        return np.ascontiguousarray(model_input), meta

    def _create_output_buffers(self) -> Dict[str, np.ndarray]:
        buffers: Dict[str, np.ndarray] = {}
        for name_ in self.output_names:
            shape = tuple(self.infer_model.output(name_).shape)
            dtype = self.output_dtypes.get(name_, np.float32)
            buffers[name_] = np.empty(shape, dtype=dtype)
        return buffers

    def _run_inference(self, input_tensor: np.ndarray) -> Any:
        output_buffers = self._create_output_buffers()
        binding = self.configured_model.create_bindings(output_buffers=output_buffers)
        binding.input().set_buffer(input_tensor)

        done = threading.Event()
        completion_holder: Dict[str, Any] = {"info": None}

        def _callback(completion_info: Any) -> None:
            completion_holder["info"] = completion_info
            done.set()

        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        job = self.configured_model.run_async([binding], _callback)
        job.wait(10000)

        if not done.wait(timeout=2.0):
            raise TimeoutError(f"[{self.name}] Inference callback timeout")

        completion_info = completion_holder.get("info")
        if completion_info is not None and getattr(completion_info, "exception", None):
            raise RuntimeError(f"[{self.name}] Inference exception: {completion_info.exception}")

        if len(self.output_names) == 1:
            return binding.output().get_buffer()

        return {name_: binding.output(name_).get_buffer() for name_ in self.output_names}

    def _describe_output(self, output: Any) -> Any:
        if isinstance(output, dict):
            return {k: self._describe_output(v) for k, v in output.items()}
        if isinstance(output, np.ndarray):
            return {"type": "ndarray", "shape": list(output.shape), "dtype": str(output.dtype)}
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                return {"type": type(output).__name__, "len": 0}
            return {
                "type": type(output).__name__,
                "len": len(output),
                "first": self._describe_output(output[0]),
            }
        return str(type(output))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax_last(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)

    @staticmethod
    def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45, max_dets: int = 100) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(scores)[::-1]
        keep: List[int] = []

        while order.size > 0 and len(keep) < max_dets:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[rest] - inter + 1e-6
            iou = inter / union
            order = rest[iou <= iou_thresh]

        return np.asarray(keep, dtype=np.int32)

    def _decode_yolov8_seg_raw(
        self,
        raw_output: Any,
        conf_thresh: float,
        iou_thresh: float,
        max_dets: int,
        include_masks: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Decode raw YOLOv8-seg heads using the same postprocess flow as Hailo Model Zoo.
        Returns None when output does not look like raw YOLOv8-seg heads.
        """
        if not isinstance(raw_output, dict):
            return None

        ordered = _mz_order_yolov8_seg_endnodes(raw_output)
        if ordered is None:
            return None

        endnodes, meta = ordered
        outputs = _mz_yolov8_seg_postprocess(
            endnodes=endnodes,
            img_dims=(self.input_h, self.input_w),
            classes=meta["classes"],
            regression_length=meta["regression_length"],
            strides=meta["feature_strides"],
            score_threshold=conf_thresh,
            nms_iou_thresh=iou_thresh,
            max_dets=max_dets,
            include_masks=include_masks,
        )

        if not outputs:
            return []

        output = outputs[0]
        boxes = output["detection_boxes"]
        scores = output["detection_scores"]
        classes = output["detection_classes"]
        masks = output["mask"]

        detections: List[Dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            x1 = float(box[0] * self.input_w)
            y1 = float(box[1] * self.input_h)
            x2 = float(box[2] * self.input_w)
            y2 = float(box[3] * self.input_h)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": float(scores[idx]),
                    "class_id": int(classes[idx]),
                    "mask": None if masks is None else masks[idx],
                }
            )

        return detections

    def _raw_score_peak(self, raw_output: Any) -> Optional[float]:
        ordered = _mz_order_yolov8_seg_endnodes(raw_output) if isinstance(raw_output, dict) else None
        if ordered is None:
            return None

        endnodes, _ = ordered
        score_tensors = [np.reshape(s, (-1, s.shape[1] * s.shape[2], s.shape[3])) for s in endnodes[1:8:3]]
        scores = np.concatenate(score_tensors, axis=1)
        if scores.size == 0:
            return None
        if np.min(scores) < 0.0 or np.max(scores) > 1.0:
            scores = _mz_sigmoid(scores)
        return float(np.max(scores))

    @staticmethod
    def _get_attr_or_call(obj: Any, attr_name: str) -> Any:
        if not hasattr(obj, attr_name):
            return None
        val = getattr(obj, attr_name)
        if callable(val):
            try:
                return val()
            except Exception:
                return None
        return val

    def _extract_bbox_from_object(self, obj: Any) -> Optional[Tuple[float, float, float, float]]:
        x1 = self._get_attr_or_call(obj, "x_min")
        y1 = self._get_attr_or_call(obj, "y_min")
        x2 = self._get_attr_or_call(obj, "x_max")
        y2 = self._get_attr_or_call(obj, "y_max")
        if None not in (x1, y1, x2, y2):
            return _normalize_bbox_xyxy(float(x1), float(y1), float(x2), float(y2))

        bbox = self._get_attr_or_call(obj, "bbox")
        if bbox is not None:
            bx1 = self._get_attr_or_call(bbox, "x_min")
            by1 = self._get_attr_or_call(bbox, "y_min")
            bx2 = self._get_attr_or_call(bbox, "x_max")
            by2 = self._get_attr_or_call(bbox, "y_max")
            if None not in (bx1, by1, bx2, by2):
                return _normalize_bbox_xyxy(float(bx1), float(by1), float(bx2), float(by2))

            try:
                arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
                if arr.size >= 4:
                    return _normalize_bbox_xyxy(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
            except Exception:
                pass

        get_bbox = self._get_attr_or_call(obj, "get_bbox")
        if get_bbox is not None:
            try:
                arr = np.asarray(get_bbox, dtype=np.float32).reshape(-1)
                if arr.size >= 4:
                    return _normalize_bbox_xyxy(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
            except Exception:
                pass

        return None

    def _extract_mask_from_object(self, obj: Any) -> Optional[np.ndarray]:
        for key in ("mask", "segmentation_mask", "byte_mask"):
            mask_val = self._get_attr_or_call(obj, key)
            if mask_val is None:
                continue
            try:
                mask_np = np.asarray(mask_val)
            except Exception:
                continue

            if mask_np.size == 0:
                continue
            if mask_np.ndim == 3:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[..., 0]
            if mask_np.ndim != 2:
                continue
            return mask_np

        return None

    def _extract_from_object(self, obj: Any, class_hint: int = 0) -> Optional[Dict[str, Any]]:
        bbox = self._extract_bbox_from_object(obj)
        if bbox is None:
            return None

        score = None
        for key in ("score", "confidence", "conf", "probability"):
            s = self._get_attr_or_call(obj, key)
            if s is not None:
                score = float(s)
                break
        if score is None:
            score = 1.0

        class_id = None
        for key in ("class_id", "label", "label_id", "class_index"):
            c = self._get_attr_or_call(obj, key)
            if c is not None:
                class_id = int(c)
                break
        if class_id is None:
            class_id = int(class_hint)

        return {
            "bbox": bbox,
            "score": score,
            "class_id": class_id,
            "mask": self._extract_mask_from_object(obj),
        }

    def _row_to_det(self, row: np.ndarray, class_hint: int = 0) -> Optional[Dict[str, Any]]:
        vals = np.asarray(row, dtype=np.float32).reshape(-1)
        if vals.size < 5:
            return None

        # Supported row layouts:
        # 1) [x1, y1, x2, y2, score, class(optional)]
        # 2) [score, x1, y1, x2, y2, class(optional)]
        # 3) [class, score, x1, y1, x2, y2]
        score = None
        class_id = int(class_hint)

        if 0.0 <= vals[4] <= 1.0:
            x1, y1, x2, y2 = vals[:4]
            score = float(vals[4])
            if vals.size >= 6 and np.isfinite(vals[5]):
                class_id = int(max(0, round(float(vals[5]))))
        elif 0.0 <= vals[0] <= 1.0:
            score = float(vals[0])
            x1, y1, x2, y2 = vals[1:5]
            if vals.size >= 6 and np.isfinite(vals[5]):
                class_id = int(max(0, round(float(vals[5]))))
        elif vals.size >= 6 and np.isfinite(vals[0]) and np.isfinite(vals[1]) and 0.0 <= vals[1] <= 1.0:
            class_id = int(max(0, round(float(vals[0]))))
            score = float(vals[1])
            x1, y1, x2, y2 = vals[2:6]
        else:
            return None

        x1, y1, x2, y2 = _normalize_bbox_xyxy(float(x1), float(y1), float(x2), float(y2))

        return {
            "bbox": (x1, y1, x2, y2),
            "score": float(score),
            "class_id": class_id,
            "mask": None,
        }

    def _extract_from_array(self, arr: np.ndarray, conf_thresh: float, class_hint: int = 0) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []

        arr = np.asarray(arr)
        if arr.size == 0:
            return dets

        arr = np.squeeze(arr)
        if arr.ndim == 0:
            return dets

        if arr.dtype == object:
            for item in arr.reshape(-1):
                det = self._extract_from_object(item, class_hint=class_hint)
                if det is not None and det["score"] >= conf_thresh:
                    dets.append(det)
            return dets

        if arr.ndim == 1:
            det = self._row_to_det(arr, class_hint=class_hint)
            if det is not None and det["score"] >= conf_thresh:
                dets.append(det)
            return dets

        if arr.ndim == 2:
            if arr.shape[1] < 5:
                return dets
            # Keep parsing bounded in case tensor is unexpectedly large.
            for row in arr[:500]:
                det = self._row_to_det(row, class_hint=class_hint)
                if det is not None and det["score"] >= conf_thresh:
                    dets.append(det)
            return dets

        if arr.ndim >= 3:
            if arr.shape[-1] < 5 or arr.shape[-1] > 16:
                return dets

            if arr.ndim > 3:
                arr = arr.reshape((-1, arr.shape[-1]))
                for row in arr:
                    det = self._row_to_det(row, class_hint=class_hint)
                    if det is not None and det["score"] >= conf_thresh:
                        dets.append(det)
                return dets

            # arr.ndim == 3
            if arr.shape[0] == 1:
                for row in arr[0]:
                    det = self._row_to_det(row, class_hint=class_hint)
                    if det is not None and det["score"] >= conf_thresh:
                        dets.append(det)
            elif arr.shape[0] <= 80:
                # Typical class-major layout: [num_classes, max_dets, 5/6]
                for cls_idx in range(arr.shape[0]):
                    for row in arr[cls_idx]:
                        det = self._row_to_det(row, class_hint=cls_idx)
                        if det is not None and det["score"] >= conf_thresh:
                            dets.append(det)
            else:
                flat = arr.reshape((-1, arr.shape[-1]))
                for row in flat:
                    det = self._row_to_det(row, class_hint=class_hint)
                    if det is not None and det["score"] >= conf_thresh:
                        dets.append(det)

        return dets

    def _extract_raw_detections(self, raw_output: Any, conf_thresh: float) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        def _collect(value: Any, class_hint: int = 0) -> None:
            if value is None:
                return

            if isinstance(value, dict):
                for v in value.values():
                    _collect(v, class_hint=class_hint)
                return

            if isinstance(value, np.ndarray):
                detections.extend(self._extract_from_array(value, conf_thresh, class_hint=class_hint))
                return

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return

                # Try each element independently to support mixed output types.
                for item in value:
                    if isinstance(item, np.ndarray):
                        detections.extend(self._extract_from_array(item, conf_thresh, class_hint=class_hint))
                    elif isinstance(item, (list, tuple)):
                        try:
                            row = np.asarray(item, dtype=np.float32)
                            det = self._row_to_det(row, class_hint=class_hint)
                            if det is not None and det["score"] >= conf_thresh:
                                detections.append(det)
                        except Exception:
                            det = self._extract_from_object(item, class_hint=class_hint)
                            if det is not None and det["score"] >= conf_thresh:
                                detections.append(det)
                    else:
                        det = self._extract_from_object(item, class_hint=class_hint)
                        if det is not None and det["score"] >= conf_thresh:
                            detections.append(det)
                return

            det = self._extract_from_object(value, class_hint=class_hint)
            if det is not None and det["score"] >= conf_thresh:
                detections.append(det)

        _collect(raw_output, class_hint=0)

        detections.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        return detections

    def _map_bbox_to_frame(
        self,
        bbox: Tuple[float, float, float, float],
        meta: Dict[str, int],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox

        # If normalized [0,1], scale into model input pixels first.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= float(meta["input_w"])
            x2 *= float(meta["input_w"])
            y1 *= float(meta["input_h"])
            y2 *= float(meta["input_h"])

        scale = float(meta["scale"])
        left = float(meta["left"])
        top = float(meta["top"])

        x1 = (x1 - left) / scale
        x2 = (x2 - left) / scale
        y1 = (y1 - top) / scale
        y2 = (y2 - top) / scale

        return _clip_bbox_xyxy(x1, y1, x2, y2, frame_w, frame_h)

    def _map_mask_to_frame(
        self,
        mask: np.ndarray,
        meta: Dict[str, int],
        frame_shape: Tuple[int, int],
        mask_thresh: float = 0.5,
    ) -> Optional[np.ndarray]:
        if mask is None:
            return None

        try:
            mask_np = np.asarray(mask)
        except Exception:
            return None

        if mask_np.ndim == 3:
            mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[..., 0]
        if mask_np.ndim != 2:
            return None

        input_h = int(meta["input_h"])
        input_w = int(meta["input_w"])

        if mask_np.shape != (input_h, input_w):
            mask_np = cv2.resize(mask_np.astype(np.float32), (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        left = int(meta["left"])
        top = int(meta["top"])
        resized_w = int(meta["resized_w"])
        resized_h = int(meta["resized_h"])

        right = min(input_w, left + resized_w)
        bottom = min(input_h, top + resized_h)

        if right <= left or bottom <= top:
            return None

        mask_unpadded = mask_np[top:bottom, left:right]
        if mask_unpadded.size == 0:
            return None

        frame_h, frame_w = frame_shape
        mask_full = cv2.resize(mask_unpadded.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        if mask_full.max() > 1.0:
            threshold = 255.0 * mask_thresh
            mask_bin = (mask_full > threshold).astype(np.uint8) * 255
        else:
            mask_bin = (mask_full > mask_thresh).astype(np.uint8) * 255

        return mask_bin

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        mask_thresh: float = 0.5,
        max_dets: int = 100,
        include_masks: bool = True,
    ) -> List[Dict[str, Any]]:
        input_tensor, meta = self._prepare_input(frame_bgr)
        raw_output = self._run_inference(input_tensor)

        if not self._logged_output_schema:
            self._logged_output_schema = True
            logger.info("[%s] Raw output schema: %s", self.name, self._describe_output(raw_output))

        raw_dets = self._decode_yolov8_seg_raw(
            raw_output,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_dets=max_dets,
            include_masks=include_masks,
        )
        if raw_dets is not None and not self._logged_decoder:
            self._logged_decoder = True
            logger.info("[%s] Using Hailo Model Zoo style YOLOv8-seg postprocessing.", self.name)
        if raw_dets is None:
            raw_dets = self._extract_raw_detections(raw_output, conf_thresh=conf_thresh)

        frame_h, frame_w = frame_bgr.shape[:2]
        mapped_dets: List[Dict[str, Any]] = []

        for det in raw_dets[:max_dets]:
            x1, y1, x2, y2 = self._map_bbox_to_frame(det["bbox"], meta, frame_w, frame_h)
            mapped = {
                "bbox": (x1, y1, x2, y2),
                "score": float(det.get("score", 0.0)),
                "class_id": int(det.get("class_id", 0)),
                "mask": None,
            }

            raw_mask = det.get("mask", None)
            if raw_mask is not None:
                mapped_mask = self._map_mask_to_frame(raw_mask, meta, (frame_h, frame_w), mask_thresh=mask_thresh)
                mapped["mask"] = mapped_mask

            mapped_dets.append(mapped)

        if raw_dets is None and not mapped_dets and not self._warned_no_decode:
            self._warned_no_decode = True
            logger.warning(
                "[%s] No detections parsed from HEF outputs. If your HEF exports raw YOLO heads, add custom post-process.",
                self.name,
            )

        if not mapped_dets:
            self._empty_predict_counter += 1
            if self.name == "GOLD" and self._empty_predict_counter % 40 == 0:
                peak_score = self._raw_score_peak(raw_output)
                if peak_score is not None:
                    logger.info("[GOLD] No detections this frame batch. Peak raw class score: %.4f", peak_score)
        else:
            self._empty_predict_counter = 0

        return mapped_dets

    def close(self) -> None:
        if self._config_ctx is not None:
            try:
                self._config_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._config_ctx = None


class HailoRuntime:
    """Shared VDevice context for multiple HEF models."""

    def __init__(self):
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "hailo_platform is not installed. Install HailoRT Python bindings on Raspberry Pi."
            )

        self.vdevice = self._create_vdevice()
        self.models: List[HailoHEFModel] = []

    def _create_vdevice(self) -> VDevice:
        try:
            params = VDevice.create_params()
            if HailoSchedulingAlgorithm is not None and hasattr(params, "scheduling_algorithm"):
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            if hasattr(params, "group_id"):
                params.group_id = "GOLD_TESTING_SHARED"
            return VDevice(params)
        except Exception as e:
            logger.warning("Could not create shared VDevice params (%s). Falling back to default VDevice().", e)
            return VDevice()

    def create_model(self, hef_path: str, name: str) -> HailoHEFModel:
        model = HailoHEFModel(self.vdevice, hef_path, name)
        self.models.append(model)
        return model

    def close(self) -> None:
        for model in self.models:
            model.close()
        self.models.clear()

        if self.vdevice is not None and hasattr(self.vdevice, "release"):
            try:
                self.vdevice.release()
            except Exception:
                pass


def pad_or_trim_audio(audio: np.ndarray, sample_count: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size >= sample_count:
        return audio[:sample_count]
    return np.pad(audio, (0, sample_count - audio.size), mode="constant")


def extract_log_mel(audio: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    sample_rate = int(config.get("sample_rate", 48000) or 48000)
    duration = float(config.get("duration", AUDIO_WINDOW_SEC) or AUDIO_WINDOW_SEC)
    sample_count = int(round(sample_rate * duration))
    fixed = pad_or_trim_audio(audio, sample_count)
    mel = librosa.feature.melspectrogram(
        y=fixed,
        sr=sample_rate,
        n_fft=int(config.get("n_fft", 512) or 512),
        hop_length=int(config.get("hop_length", 160) or 160),
        win_length=int(config.get("win_length", 400) or 400),
        n_mels=int(config.get("n_mels", AUDIO_N_MELS) or AUDIO_N_MELS),
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)
    mean = float(config.get("normalization_mean", float(np.mean(log_mel))) or 0.0)
    std = max(float(config.get("normalization_std", float(np.std(log_mel))) or 1.0), 1e-6)
    return ((log_mel - mean) / std)[..., np.newaxis].astype(np.float32)

def analyze_audio_activity(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(abs_x.max())
    active_ratio = float(np.mean(abs_x >= SILENCE_ACTIVE_THRESHOLD))
    return rms, peak, active_ratio

def is_silent_window(x: np.ndarray) -> Tuple[bool, float, float, float]:
    rms, peak, active_ratio = analyze_audio_activity(x)
    silent = (rms < SILENCE_RMS_THRESHOLD and peak < SILENCE_PEAK_THRESHOLD and active_ratio < SILENCE_MIN_ACTIVE_RATIO)
    return silent, rms, peak, active_ratio

class RubbingAudioModel:
    def __init__(self, model_dir: str, model_path: str) -> None:
        self.model_dir = model_dir
        self.model_path = model_path
        self.backend = ""
        self.config: Dict[str, Any] = {}
        self.classes: List[str] = []
        self.model: Any = None
        self.interpreter: Any = None
        self.input_detail: Dict[str, Any] = {}
        self.output_detail: Dict[str, Any] = {}

    @property
    def sample_rate(self) -> int:
        return int(self.config.get("sample_rate", 48000) or 48000)

    @property
    def duration(self) -> float:
        return float(self.config.get("duration", AUDIO_WINDOW_SEC) or AUDIO_WINDOW_SEC)

    @property
    def confidence_threshold(self) -> float:
        return float(self.config.get("confidence_threshold", AUDIO_CONF_THRESH) or AUDIO_CONF_THRESH)

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.duration))

    def load(self) -> None:
        config_path = os.path.join(self.model_dir, "config.json")
        labels_path = os.path.join(self.model_dir, "labels.json")
        missing = [
            path
            for path in (self.model_path, config_path, labels_path)
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError("Missing rubbing audio model file(s): " + ", ".join(missing))

        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = json.load(fh)
        with open(labels_path, "r", encoding="utf-8") as fh:
            self.classes = [str(label) for label in json.load(fh)]
        if not self.classes:
            raise ValueError("Rubbing audio labels.json is empty.")

        import tensorflow as tf

        if self.model_path.lower().endswith(".tflite"):
            thread_count = max(1, int(os.environ.get("PURITY_TFLITE_THREADS", "2")))
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path,
                num_threads=thread_count,
            )
            self.interpreter.allocate_tensors()
            self.input_detail = self.interpreter.get_input_details()[0]
            self.output_detail = self.interpreter.get_output_details()[0]
            self.backend = f"tflite/{thread_count}t"
        else:
            self.model = tf.keras.models.load_model(self.model_path)
            self.backend = "keras"

    def predict(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        feature = extract_log_mel(audio, self.config)
        input_batch = feature[np.newaxis, ...]

        if self.backend.startswith("tflite"):
            input_index = int(self.input_detail["index"])
            output_index = int(self.output_detail["index"])
            input_dtype = self.input_detail.get("dtype", np.float32)
            expected_shape = tuple(int(v) for v in self.input_detail.get("shape", input_batch.shape))
            if tuple(input_batch.shape) != expected_shape:
                self.interpreter.resize_tensor_input(input_index, input_batch.shape, strict=False)
                self.interpreter.allocate_tensors()
                self.input_detail = self.interpreter.get_input_details()[0]
                self.output_detail = self.interpreter.get_output_details()[0]
                input_index = int(self.input_detail["index"])
                output_index = int(self.output_detail["index"])
                input_dtype = self.input_detail.get("dtype", np.float32)
            self.interpreter.set_tensor(input_index, input_batch.astype(input_dtype))
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(output_index)[0]
        else:
            probabilities = self.model.predict(input_batch, verbose=0)[0]

        probabilities = np.asarray(probabilities, dtype=np.float32).ravel()
        if probabilities.size != len(self.classes):
            raise ValueError(
                f"Rubbing audio model returned {probabilities.size} probabilities "
                f"for {len(self.classes)} labels."
            )
        best_index = int(np.argmax(probabilities))
        all_probabilities = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.classes)
        }
        return self.classes[best_index], float(probabilities[best_index]), all_probabilities

def load_audio_model() -> Optional[Tuple[Any, Any]]:
    model_path = SOUND_MODEL_PATH
    if not os.path.exists(model_path):
        fallback_paths = [
            os.path.join(SOUND_MODEL_DIR, "gold_rub_cnn.tflite"),
            os.path.join(SOUND_MODEL_DIR, "gold_rub_cnn.keras"),
        ]
        model_path = next((path for path in fallback_paths if os.path.exists(path)), model_path)
    if not os.path.exists(model_path):
        logger.warning("Rubbing audio model not found at %s.", model_path)
        return None
    try:
        model_dir = os.path.dirname(model_path) or SOUND_MODEL_DIR
        audio_model = RubbingAudioModel(model_dir=model_dir, model_path=model_path)
        audio_model.load()
        logger.info(
            "Rubbing audio model loaded: backend=%s path=%s classes=%s sr=%sHz window=%.2fs threshold=%.2f",
            audio_model.backend,
            model_path,
            audio_model.classes,
            audio_model.sample_rate,
            audio_model.duration,
            audio_model.confidence_threshold,
        )
        return audio_model, None
    except Exception as e:
        logger.error("Failed to load rubbing audio model: %s", e)
        return None

def _normalize_audio_device_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def audio_device_stable_id(index: int, name: str) -> str:
    return f"{AUDIO_DEVICE_NAME_PREFIX}{_normalize_audio_device_name(name)}"


class AudioWorker:
    def __init__(self, model: Any, device_ctx: Any, sample_rate: Optional[int] = None,
                 window_sec: Optional[float] = None,
                 hop_ratio: float = AUDIO_HOP_RATIO,
                 confidence_threshold: Optional[float] = None,
                 device: Any = None, allow_fallback: bool = True):
        self.model: RubbingAudioModel = model
        self.device_ctx = device_ctx
        self.model_sr = int(sample_rate or getattr(model, "sample_rate", 48000) or 48000)
        self.input_sr: Optional[int] = None
        self.device_default_sr = self.model_sr
        self.window_sec = float(window_sec or getattr(model, "duration", AUDIO_WINDOW_SEC) or AUDIO_WINDOW_SEC)
        self.hop_ratio = hop_ratio
        self.device = device
        self.allow_fallback = allow_fallback
        self.conf_thresh = float(
            confidence_threshold
            if confidence_threshold is not None
            else getattr(model, "confidence_threshold", AUDIO_CONF_THRESH)
        )

        try:
            device_info = sd.query_devices(device if device is not None else sd.default.device[0], "input")
            self.device_default_sr = int(float(device_info.get("default_samplerate", self.model_sr)))
            logger.info(
                "[Audio] Device default sample rate: %sHz (rubbing model needs %sHz)",
                self.device_default_sr,
                self.model_sr,
            )
        except Exception:
            self.device_default_sr = self.model_sr

        self.win_samples = int(self.model_sr * self.window_sec)
        self.hop_samples = max(1, int(self.win_samples * self.hop_ratio))

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional[queue.Queue] = None
        self._last_status_log_ts = 0.0
        self._ok_streak = 0
        self._nok_streak = 0
        self._stream = None
        self._wave_lock = threading.Lock()
        self._latest_waveform = np.zeros(1024, dtype=np.float32)
        self._debug_lock = threading.Lock()
        self._debug: Dict[str, Any] = {
            "selected_device": self.device,
            "selected_device_name": get_audio_device_name(self.device),
            "input_sr": self.input_sr or self.model_sr,
            "model_sr": self.model_sr,
            "device_default_sr": self.device_default_sr,
            "model_backend": getattr(self.model, "backend", ""),
            "model_path": getattr(self.model, "model_path", ""),
            "threshold": self.conf_thresh,
            "last_label": "Waiting...",
            "last_conf": 0.0,
            "last_decision": "Waiting...",
            "ok_prob": 0.0,
            "nok_prob": 0.0,
            "probabilities": {},
            "rms": 0.0,
            "peak": 0.0,
            "active_ratio": 0.0,
            "stream_open": False,
            "last_error": "",
        }

    def _update_window_sizes(self) -> None:
        rate = int(self.input_sr or self.model_sr)
        self.win_samples = max(1, int(rate * self.window_sec))
        self.hop_samples = max(1, int(self.win_samples * self.hop_ratio))

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            now = time.time()
            if now - self._last_status_log_ts >= 2.0:
                logger.warning("Audio status: %s", status)
                self._last_status_log_ts = now

        mono = np.nan_to_num(
            indata[:, 0].astype(np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        with self._wave_lock:
            self._latest_waveform = mono.copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                pass

    def _inference_loop(self) -> None:
        accum: List[float] = []
        last_silence_log_ts = 0.0

        while self._running:
            try:
                chunk = self._queue.get(timeout=0.1)
                accum.extend(chunk.tolist())
            except queue.Empty:
                continue

            if len(accum) < self.win_samples:
                continue

            x = np.array(accum[-self.win_samples:], dtype=np.float32)
            if self.input_sr and int(self.input_sr) != self.model_sr:
                target_len = int(round(len(x) * (self.model_sr / float(self.input_sr))))
                x = scipy.signal.resample(x, target_len).astype(np.float32)

            silent, rms, peak, active_ratio = is_silent_window(x)
            if silent:
                now = time.time()
                if now - last_silence_log_ts >= 2.0:
                    logger.info(
                        "[Audio] Low-energy window classified by rubbing model (rms=%.4f peak=%.4f active=%.3f)",
                        rms,
                        peak,
                        active_ratio,
                    )
                    last_silence_log_ts = now

            try:
                label, conf, probabilities = self.model.predict(x)
                ok_prob = float(probabilities.get(JEWEL_RUB_OK_LABEL, 0.0) or 0.0)
                nok_prob = max(
                    [float(prob) for name, prob in probabilities.items() if name != JEWEL_RUB_OK_LABEL]
                    or [0.0]
                )
                decision = "OK" if label == JEWEL_RUB_OK_LABEL and conf >= self.conf_thresh else "NOK"

                STATE["audio_label"] = label
                STATE["audio_decision"] = decision
                STATE["audio_confidence"] = conf
                STATE["audio_probabilities"] = dict(probabilities)

                with self._debug_lock:
                    self._debug.update(
                        {
                            "input_sr": self.input_sr,
                            "model_sr": self.model_sr,
                            "last_label": label,
                            "last_conf": conf,
                            "last_decision": decision,
                            "ok_prob": ok_prob,
                            "nok_prob": nok_prob,
                            "probabilities": dict(probabilities),
                            "rms": rms,
                            "peak": peak,
                            "active_ratio": active_ratio,
                        }
                    )

                if decision == "OK":
                    self._ok_streak += 1
                    self._nok_streak = 0
                    if self._ok_streak >= AUDIO_OK_STREAK_REQUIRED:
                        STATE["sound_status"] = "OK"
                        STATE["audio_ok_recent_until"] = time.time() + AUDIO_SYNC_WINDOW_SEC
                        logger.info(
                            "[Audio] Detected: %s => OK (conf %.3f, streak %s)",
                            label,
                            conf,
                            self._ok_streak,
                        )
                else:
                    self._nok_streak += 1
                    self._ok_streak = 0
                    if self._nok_streak >= AUDIO_NOK_STREAK_REQUIRED:
                        STATE["sound_status"] = "NOK"
                        logger.info(
                            "[Audio] Detected: %s => NOK (conf %.3f, streak %s)",
                            label,
                            conf,
                            self._nok_streak,
                        )
            except Exception as e:
                logger.error("Rubbing audio inference error: %s", e)
                with self._debug_lock:
                    self._debug.update({"last_error": str(e)})

            accum = accum[self.hop_samples:]

    def _open_stream(self, device: Any, sample_rate: int, label: str) -> None:
        self.input_sr = int(sample_rate)
        self._update_window_sizes()
        self._stream = sd.InputStream(
            samplerate=self.input_sr,
            channels=1,
            dtype="float32",
            blocksize=0,
            latency="high",
            callback=self._audio_callback,
            device=device,
        )
        self._stream.start()
        with self._debug_lock:
            self._debug.update(
                {
                    "selected_device": device,
                    "selected_device_name": get_audio_device_name(device),
                    "input_sr": self.input_sr,
                    "model_sr": self.model_sr,
                    "stream_open": True,
                    "last_error": "",
                }
            )
        logger.info("[Audio] Microphone stream started on %s at %sHz", label, self.input_sr)

    def _try_open(self, device: Any, sample_rate: int, label: str) -> Optional[Exception]:
        try:
            self._open_stream(device, sample_rate, label)
            return None
        except Exception as exc:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None
            logger.warning("[Audio] Could not open %s at %sHz: %s", label, sample_rate, exc)
            return exc

    def start(self) -> None:
        self._running = True
        self._queue = queue.Queue(maxsize=24)
        dev_name = get_audio_device_name(self.device)

        first_error = self._try_open(
            self.device,
            self.model_sr,
            f"preferred device {dev_name} at model rate",
        )
        if first_error is not None and self.device_default_sr != self.model_sr:
            first_error = self._try_open(
                self.device,
                self.device_default_sr,
                f"preferred device {dev_name} at default rate",
            )

        if first_error is not None and self.allow_fallback:
            logger.info("[Audio] Trying fallback to default input device...")
            default_sr = self.model_sr
            try:
                dev_info = sd.query_devices(sd.default.device[0], "input")
                default_sr = int(float(dev_info.get("default_samplerate", self.model_sr)))
            except Exception:
                pass
            first_error = self._try_open(None, self.model_sr, "default device at model rate")
            if first_error is not None and default_sr != self.model_sr:
                first_error = self._try_open(None, default_sr, "default device at default rate")

        if first_error is not None:
            self._running = False
            with self._debug_lock:
                self._debug.update(
                    {
                        "selected_device": self.device,
                        "selected_device_name": get_audio_device_name(self.device),
                        "stream_open": False,
                        "last_error": str(first_error),
                    }
                )

        if self._running and self._stream is not None:
            self._thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._running = False

        if hasattr(self, "_stream"):
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        with self._debug_lock:
            self._debug.update({"stream_open": False})
        logger.info("[Audio] Microphone stream stopped.")

    def get_latest_waveform(self) -> np.ndarray:
        with self._wave_lock:
            return self._latest_waveform.copy()

    def get_debug_snapshot(self) -> Dict[str, Any]:
        with self._debug_lock:
            return dict(self._debug)

    def set_confidence_threshold(self, confidence_threshold: float) -> float:
        self.conf_thresh = max(0.0, min(1.0, float(confidence_threshold)))
        with self._debug_lock:
            last_label = str(self._debug.get("last_label", "Waiting...") or "Waiting...")
            last_conf = float(self._debug.get("last_conf", 0.0) or 0.0)
            if last_label == "Waiting...":
                decision = str(self._debug.get("last_decision", "Waiting...") or "Waiting...")
            else:
                decision = "OK" if last_label == JEWEL_RUB_OK_LABEL and last_conf >= self.conf_thresh else "NOK"
            self._debug.update({"threshold": self.conf_thresh, "last_decision": decision})
        if last_label != "Waiting...":
            STATE["audio_decision"] = decision
            if decision == "OK":
                STATE["sound_status"] = "OK"
                STATE["audio_ok_recent_until"] = time.time() + AUDIO_SYNC_WINDOW_SEC
            else:
                STATE["sound_status"] = "NOK"
                STATE["audio_ok_recent_until"] = 0.0
        logger.info("[Audio] OK confidence threshold set to %.2f", self.conf_thresh)
        return self.conf_thresh


# ====================== MODEL INIT ======================
HAILO_RUNTIME: Optional[HailoRuntime] = None
MODEL_STONE: Optional[HailoHEFModel] = None
MODEL_GOLD: Optional[HailoHEFModel] = None
MODEL_ACID: Optional[HailoHEFModel] = None


def init_hailo_models() -> Tuple[HailoRuntime, HailoHEFModel, HailoHEFModel, HailoHEFModel]:
    missing = [p for p in (MODEL_STONE_PATH, MODEL_GOLD_PATH, MODEL_ACID_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing HEF files: {missing}")

    runtime = HailoRuntime()

    logger.info("Loading Hailo HEF models...")
    model_stone = runtime.create_model(MODEL_STONE_PATH, "STONE")
    model_gold = runtime.create_model(MODEL_GOLD_PATH, "GOLD")
    model_acid = runtime.create_model(MODEL_ACID_PATH, "ACID")
    logger.info("Hailo HEF models loaded.")

    return runtime, model_stone, model_gold, model_acid


# ====================== DETECTION FUNCTIONS ======================
def _bbox_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))


def _expand_bbox(box: Tuple[int, int, int, int], pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def _bbox_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_intersection_area(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return np.zeros_like(mask_u8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def _normalize_binary_mask(mask: np.ndarray, thresh: float) -> np.ndarray:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[..., 0]
    if mask_np.max() > 1.0:
        mask_np = (mask_np > (255.0 * thresh)).astype(np.uint8)
    else:
        mask_np = (mask_np > thresh).astype(np.uint8)
    return _largest_connected_component(mask_np)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _should_run_gold_full_frame_fallback() -> bool:
    miss_count = int(STATE.get("gold_full_frame_miss_count", 0) or 0) + 1
    STATE["gold_full_frame_miss_count"] = miss_count
    return miss_count == 1 or miss_count % GOLD_FULL_FRAME_FALLBACK_EVERY == 0


def process_rubbing_frame(frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    H, W = frame.shape[:2]
    annotated = frame.copy()
    inference_error = ""

    gold_clipped_full = np.zeros((H, W), dtype=np.uint8)
    gold_mask_pct = 0.0
    stone_bbox: Optional[Tuple[int, int, int, int]] = None
    STATE["last_stone_bbox"] = None
    STATE["stone_visible_now"] = False
    STATE["gold_visible_now"] = False
    STATE["last_rubbing_bbox"] = None
    STATE["last_rubbing_mask"] = None

    # Stage 1A: stone detection. The model wrapper owns resize/letterbox and
    # maps detections back to this frame's coordinate space.
    try:
        stone_dets = MODEL_STONE.predict(
            frame,
            conf_thresh=STONE_CONF_THRESH,
            iou_thresh=STONE_IOU_THRESH,
            max_dets=20,
            include_masks=False,
        )

        frame_area = H * W
        stone_dets = [
            det for det in stone_dets
            if _bbox_area(det["bbox"]) >= int(frame_area * STONE_MIN_AREA_RATIO)
        ]

        if stone_dets:
            largest = max(stone_dets, key=lambda det: _bbox_area(det["bbox"]) * max(det.get("score", 0.01), 0.01))
            x1, y1, x2, y2 = largest["bbox"]
            stone_bbox = (x1, y1, x2, y2)
            STATE["last_stone_bbox"] = stone_bbox
            STATE["stone_visible_now"] = True
            logger.debug("[STONE] bbox=(%s,%s)-(%s,%s) score=%.3f frame=%sx%s", x1, y1, x2, y2, float(largest.get("score", 0.0)), W, H)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), STONE_BOX_COLOR, 2)
            cv2.putText(
                annotated,
                f"Stone {largest['score']:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                STONE_BOX_COLOR,
                2,
            )

    except Exception as e:
        inference_error = f"Stone model inference failed: {e}"
        logger.exception(inference_error)

    # Stage 1B: gold detection constrained to the detected stone region
    if stone_bbox:
        sx1, sy1, sx2, sy2 = stone_bbox

        try:
            w = sx2 - sx1
            h = sy2 - sy1
            stone_area = max(1, _bbox_area(stone_bbox))
            stone_mask = np.zeros((H, W), dtype=np.uint8)
            stone_mask[sy1:sy2, sx1:sx2] = 1
            crop_pad = max(20, int(GOLD_CROP_PAD_RATIO * max(w, h)))
            cx1, cy1, cx2, cy2 = _expand_bbox((sx1, sy1, sx2, sy2), crop_pad, W, H)
            crop = frame[cy1:cy2, cx1:cx2]

            gold_candidates: List[Dict[str, Any]] = []

            if crop.size != 0:
                crop_h, crop_w = crop.shape[:2]
                crop_dets = MODEL_GOLD.predict(
                    crop,
                    conf_thresh=GOLD_CONF_THRESH,
                    iou_thresh=GOLD_IOU_THRESH,
                    mask_thresh=GOLD_MASK_THRESH,
                    max_dets=20,
                )

                for det in crop_dets:
                    mask_crop = det.get("mask")
                    if mask_crop is None:
                        continue

                    mask_crop = _normalize_binary_mask(mask_crop, GOLD_MASK_THRESH)
                    if mask_crop.shape[:2] != (crop_h, crop_w):
                        mask_crop = cv2.resize(mask_crop.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    if int(mask_crop.sum()) < GOLD_MIN_MASK_PIXELS:
                        continue

                    mask_full = np.zeros((H, W), dtype=np.uint8)
                    mask_full[cy1:cy2, cx1:cx2] = mask_crop * 255

                    bx1, by1, bx2, by2 = det["bbox"]
                    gold_candidates.append(
                        {
                            "bbox": (int(bx1 + cx1), int(by1 + cy1), int(bx2 + cx1), int(by2 + cy1)),
                            "score": float(det.get("score", 0.0)),
                            "mask": mask_full,
                            "source": "crop",
                        }
                    )

            if gold_candidates:
                STATE["gold_full_frame_miss_count"] = 0

            # Full-frame fallback can still recover missed detections, but every
            # candidate is clipped back to the detected stone area below. Run
            # this expensive second Gold inference periodically, not on every
            # crop miss, so the live preview remains responsive.
            if not gold_candidates and _should_run_gold_full_frame_fallback():
                full_dets = MODEL_GOLD.predict(
                    frame,
                    conf_thresh=max(0.08, GOLD_CONF_THRESH * 0.75),
                    iou_thresh=GOLD_IOU_THRESH,
                    mask_thresh=GOLD_MASK_THRESH,
                    max_dets=20,
                )
                for det in full_dets:
                    if det.get("mask") is None:
                        continue
                    mask_full = _normalize_binary_mask(det["mask"], GOLD_MASK_THRESH) * 255
                    if int((mask_full > 0).sum()) < GOLD_MIN_MASK_PIXELS:
                        continue
                    gold_candidates.append(
                        {
                            "bbox": tuple(int(v) for v in det["bbox"]),
                            "score": float(det.get("score", 0.0)),
                            "mask": mask_full.astype(np.uint8),
                            "source": "full",
                        }
                    )

            if gold_candidates:
                ranked_candidates: List[Tuple[float, np.ndarray, Dict[str, Any], Tuple[int, int, int, int], float, float]] = []
                for det in gold_candidates:
                    mask_full = (det["mask"] > 0).astype(np.uint8)
                    raw_mask_pixels = int(mask_full.sum())
                    if raw_mask_pixels < GOLD_MIN_MASK_PIXELS:
                        continue

                    mask_on_stone = _largest_connected_component(mask_full * stone_mask)
                    mask_on_stone_pixels = int(mask_on_stone.sum())
                    if mask_on_stone_pixels < max(GOLD_MIN_MASK_PIXELS, GOLD_MIN_OVERLAP_PIXELS):
                        continue

                    inside_ratio = float(mask_on_stone_pixels) / float(max(1, raw_mask_pixels))
                    if inside_ratio < GOLD_MIN_INSIDE_STONE_RATIO:
                        continue

                    clipped_bbox = _mask_bbox(mask_on_stone)
                    if clipped_bbox is None:
                        continue

                    stone_overlap_pct = float(mask_on_stone_pixels) / float(stone_area) * 100.0
                    score = float(det["score"]) * inside_ratio * float(mask_on_stone_pixels)
                    ranked_candidates.append(
                        (
                            score,
                            (mask_on_stone * 255).astype(np.uint8),
                            det,
                            clipped_bbox,
                            stone_overlap_pct,
                            inside_ratio,
                        )
                    )

                if ranked_candidates:
                    _, gold_clipped_full, best, best_bbox, gold_mask_pct, inside_ratio = max(
                        ranked_candidates, key=lambda item: item[0]
                    )
                    STATE["gold_visible_now"] = True
                    STATE["last_rubbing_bbox"] = tuple(int(v) for v in best_bbox)
                    STATE["last_rubbing_mask"] = gold_clipped_full.copy()
                    STATE["gold_detected_recent_until"] = max(
                        float(STATE.get("gold_detected_recent_until", 0.0)),
                        time.time() + GOLD_AUDIO_GRACE_SEC,
                    )
                    annotated[gold_clipped_full > 0] = GOLD_OVERLAY_COLOR

                    gx1, gy1, gx2, gy2 = best_bbox
                    cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), GOLD_OVERLAY_COLOR, 2)
                    cv2.putText(
                        annotated,
                        f"Gold {best['score']:.2f} | {gold_mask_pct:.1f}% stone | {inside_ratio * 100.0:.0f}% inside",
                        (max(0, gx1), max(20, gy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        GOLD_OVERLAY_COLOR,
                        2,
                    )

        except Exception as e:
            inference_error = f"Gold model inference failed: {e}"
            logger.exception(inference_error)
    else:
        STATE["gold_full_frame_miss_count"] = 0

    return annotated, {
        "mask": gold_clipped_full,
        "mask_pct": gold_mask_pct,
        "stone_bbox": stone_bbox,
        "gold_present": bool(STATE["gold_visible_now"]),
        "error": inference_error,
    }


def compute_rubbing(annotated: np.ndarray, gold_info: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    if not gold_info or gold_info.get("stone_bbox") is None:
        STATE["prev_centroid"] = None
        STATE["recent_distances"].clear()   # Clear when no stone
        return annotated, False

    mask = gold_info["mask"]
    if int((mask > 0).sum()) == 0:
        STATE["prev_centroid"] = None
        STATE["recent_distances"].clear()
        return annotated, False

    M = cv2.moments(mask)
    if M["m00"] == 0:
        STATE["prev_centroid"] = None
        return annotated, False

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

    prev_centroid = STATE.get("prev_centroid")
    centroid_move = 0.0
    if prev_centroid is not None:
        centroid_move = float(np.hypot(cx - prev_centroid[0], cy - prev_centroid[1]))
        cv2.line(annotated, prev_centroid, (cx, cy), (255, 255, 0), 2)

    STATE["prev_centroid"] = (cx, cy)
    STATE["recent_distances"].append(centroid_move)

    # === IMPROVED VISUAL RUBBING LOGIC ===
    recent_moves = list(STATE["recent_distances"])
    max_recent = max(recent_moves[-5:]) if recent_moves else 0.0   # Look at last 5 frames
    avg_recent = sum(recent_moves[-5:]) / 5 if len(recent_moves) >= 5 else 0.0

    # More responsive: Require consistent movement
    visual_rubbing = (max_recent >= RUBBING_MIN_CENTROID_MOVE * 1.2) and (avg_recent >= RUBBING_MIN_CENTROID_MOVE * 0.6)

    return annotated, visual_rubbing


def _now_ts() -> float:
    return time.time()


def update_visual_rubbing_grace(is_rubbing: bool, gold_info: Dict[str, Any], now: Optional[float] = None) -> None:
    """
    Hold a short visual window after gold is confirmed inside the stone.
    Audio prediction is asynchronous, so audio OK may arrive slightly after the
    valid gold-inside-stone frame and still count as synchronized rubbing.
    """
    now = _now_ts() if now is None else float(now)
    gold_present = bool(gold_info and gold_info.get("gold_present"))
    stone_present = bool(gold_info and gold_info.get("stone_bbox") is not None)
    gold_recent = now <= float(STATE.get("gold_detected_recent_until", 0.0) or 0.0)

    if stone_present and (gold_present or gold_recent):
        STATE["visual_rubbing_recent_until"] = max(
            float(STATE.get("visual_rubbing_recent_until", 0.0) or 0.0),
            now + GOLD_AUDIO_GRACE_SEC,
        )


def is_audio_ok_recent(now: Optional[float] = None) -> bool:
    now = _now_ts() if now is None else float(now)
    return (
        STATE.get("sound_status") == "OK"
        and now <= float(STATE.get("audio_ok_recent_until", 0.0) or 0.0)
    )


def is_visual_rubbing_recent(now: Optional[float] = None) -> bool:
    now = _now_ts() if now is None else float(now)
    return now <= float(STATE.get("visual_rubbing_recent_until", 0.0) or 0.0)


def rubbing_sync_ready(is_rubbing: bool, gold_info: Dict[str, Any], now: Optional[float] = None) -> Tuple[bool, bool, bool]:
    """
    Return (combined_sync_ok, visual_recent, audio_recent).
    A valid gold-inside-stone frame starts a GOLD_AUDIO_GRACE_SEC window; audio
    OK may arrive during that window and still count as synchronized rubbing.
    """
    now = _now_ts() if now is None else float(now)
    update_visual_rubbing_grace(is_rubbing, gold_info, now=now)
    visual_recent = is_visual_rubbing_recent(now)
    audio_recent = is_audio_ok_recent(now)
    return visual_recent and audio_recent, visual_recent, audio_recent


def process_acid_frame(frame: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    annotated = frame.copy()
    acid_detected = False
    acid_bbox: Optional[Tuple[int, int, int, int]] = None
    inference_error = ""
    STATE["last_acid_bbox"] = None

    try:
        acid_dets = MODEL_ACID.predict(
            frame,
            conf_thresh=ACID_CONF_THRESH,
            iou_thresh=ACID_IOU_THRESH,
            max_dets=20,
            include_masks=False,
        )

        for det in acid_dets:
            conf = float(det.get("score", 0.0))
            x1, y1, x2, y2 = det["bbox"]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            if conf < ACID_CONF_THRESH or _bbox_area((x1, y1, x2, y2)) < ACID_MIN_AREA_PX:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(
                annotated,
                f"Acid {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            acid_detected = True
            acid_bbox = (x1, y1, x2, y2)
            STATE["last_acid_bbox"] = acid_bbox

    except Exception as e:
        inference_error = f"Acid model inference failed: {e}"
        logger.exception(inference_error)

    return annotated, acid_detected, {
        "acid_bbox": acid_bbox,
        "error": inference_error,
    }


def draw_status(frame: np.ndarray) -> np.ndarray:
    """Draw minimal status indicators at bottom-right corner."""
    stage = STATE["stage"]
    sound = STATE["sound_status"]
    H, W = frame.shape[:2]
    
    # Minimal compact status at bottom-right
    indicators = []
    stage_color = (0, 255, 255) if stage == "ACID" else (255, 220, 0)
    indicators.append((f"Stage: {stage}", stage_color))
    
    # Stone detected indicator
    stone_bbox = STATE.get("last_stone_bbox")
    stone_ok = stone_bbox is not None
    stone_symbol = "✓" if stone_ok else "✗"
    stone_color = (0, 255, 0) if stone_ok else (0, 0, 255)
    indicators.append((f"Stone {stone_symbol}", stone_color))
    
    # Gold visible indicator
    gold_visible = STATE.get("gold_visible_now", False)
    gold_symbol = "✓" if gold_visible else "✗"
    gold_color = (0, 255, 0) if gold_visible else (150, 150, 150)
    indicators.append((f"Gold {gold_symbol}", gold_color))
    
    # Audio status
    audio_ok = (sound == "OK")
    audio_text = "Audio ✓" if audio_ok else "Audio ✗"
    audio_color = (0, 255, 0) if audio_ok else (0, 0, 255)
    indicators.append((audio_text, audio_color))

    if stage == "RUBBING":
        sync_ok = is_visual_rubbing_recent() and is_audio_ok_recent()
        sync_text = "Sync OK" if sync_ok else "Sync --"
        sync_color = (0, 255, 0) if sync_ok else (0, 165, 255)
        indicators.append((sync_text, sync_color))
    elif stage == "ACID":
        acid_hits = int(STATE.get("acid_positive_streak", 0) or 0)
        indicators.append((f"Acid search {acid_hits}/{ACID_CONFIRM_FRAMES}", (0, 255, 255)))
    
    # Draw compact indicators at bottom-right
    padding = 10
    line_height = 20
    text_x = max(5, W - 120)
    text_y = max(20, H - (len(indicators) * line_height + padding + 5))
    
    for idx, (text, color) in enumerate(indicators):
        cv2.putText(
            frame,
            text,
            (text_x, text_y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    
    return frame


# ====================== UTIL ======================
def list_audio_devices() -> List[Dict[str, Any]]:
    """Return available input audio devices from PortAudio/sounddevice."""
    if not SOUNDDEVICE_AVAILABLE:
        return []

    devices: List[Dict[str, Any]] = []
    try:
        devs = sd.query_devices()
        for i, dev in enumerate(devs):
            max_inputs = int(dev.get("max_input_channels", 0))
            if max_inputs <= 0:
                continue
            devices.append(
                {
                    "index": int(i),
                    "name": str(dev.get("name", f"Input {i}")),
                    "id": audio_device_stable_id(i, str(dev.get("name", f"Input {i}"))),
                    "default_samplerate": int(float(dev.get("default_samplerate", 16000))),
                    "max_input_channels": max_inputs,
                }
            )
        logger.info("[Audio] Found input devices: %s", [(d["index"], d["name"]) for d in devices])
    except Exception as e:
        logger.error("Error listing audio devices: %s", e)

    return devices


def resolve_audio_input_device(value: Any) -> Optional[int]:
    """Resolve a UI/device token to the current PortAudio input index."""
    if value is None or value == AUDIO_DEVICE_AUTO:
        return None
    if isinstance(value, int):
        return int(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        pass
    normalized = _normalize_audio_device_name(raw)
    if raw.startswith(AUDIO_DEVICE_NAME_PREFIX):
        normalized = _normalize_audio_device_name(raw[len(AUDIO_DEVICE_NAME_PREFIX):])
    for device in list_audio_devices():
        name = _normalize_audio_device_name(device.get("name", ""))
        stable_id = str(device.get("id", ""))
        if stable_id == raw or name == normalized:
            return int(device["index"])
    for device in list_audio_devices():
        if normalized and normalized in _normalize_audio_device_name(device.get("name", "")):
            return int(device["index"])
    return None


def score_audio_input_device(name: str) -> int:
    lower = str(name).lower()
    score = 0

    if any(keyword in lower for keyword in ("walmart", "ab13x")):
        score += 220
    if any(keyword in lower for keyword in ("headset", "adapter")):
        score += 80
    if "usb audio" in lower:
        score += 20
    if "usb" in lower:
        score += 35
    if any(keyword in lower for keyword in ("audio", "mic", "microphone", "headset", "adapter")):
        score += 20
    if any(keyword in lower for keyword in AUDIO_WEBCAM_KEYWORDS):
        score -= 40

    return score


def get_audio_device_name(device_index: Optional[int]) -> str:
    if not SOUNDDEVICE_AVAILABLE:
        return "Unavailable"

    if device_index is None:
        try:
            default_index = sd.default.device[0]
            dev = sd.query_devices(default_index)
            return f"default:{default_index} {dev['name']}"
        except Exception:
            return "Default input device"

    try:
        dev = sd.query_devices(device_index)
        return f"{device_index}: {dev['name']}"
    except Exception:
        return f"{device_index}"


def find_preferred_audio_input_device() -> Optional[int]:
    """Pick the most likely external microphone over webcam/default inputs."""
    devices = list_audio_devices()
    if not devices:
        return None

    ranked = sorted(devices, key=lambda item: (score_audio_input_device(item["name"]), item["index"]), reverse=True)
    best = ranked[0]

    if score_audio_input_device(best["name"]) > 0:
        logger.info("[Audio] Preferred input device: %s (%s)", best["index"], best["name"])
        return int(best["index"])

    logger.warning("[Audio] No strong preferred mic match found. Using first available input device.")
    return int(devices[0]["index"])


# ====================== GUI ======================
class GoldTestingWindow(QtWidgets.QMainWindow):
    def __init__(self, cap: VideoStreamWidget, audio_bundle: Optional[Tuple[Any, Any]]):
        super().__init__()
        self.cap = cap
        self.audio_bundle = audio_bundle
        self.audio_worker: Optional[AudioWorker] = None
        self.frame_count = 0
        self.last_annotated: Optional[np.ndarray] = None
        self._closed = False
        self._tick_busy = False

        self.setWindowTitle("Gold Testing Dashboard")
        self.resize(920, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        status_grid = QtWidgets.QGridLayout()
        root.addLayout(status_grid)

        self.stage_value = QtWidgets.QLabel("RUBBING")
        self.sound_value = QtWidgets.QLabel("Waiting...")
        self.pred_value = QtWidgets.QLabel("No prediction yet")
        self.audio_info = QtWidgets.QLabel("Audio stream not started")

        for label in (self.stage_value, self.sound_value, self.pred_value, self.audio_info):
            label.setWordWrap(True)

        status_grid.addWidget(QtWidgets.QLabel("Stage"), 0, 0)
        status_grid.addWidget(self.stage_value, 0, 1)
        status_grid.addWidget(QtWidgets.QLabel("Pipeline Sound"), 1, 0)
        status_grid.addWidget(self.sound_value, 1, 1)
        status_grid.addWidget(QtWidgets.QLabel("Last Audio Pred"), 2, 0)
        status_grid.addWidget(self.pred_value, 2, 1)
        status_grid.addWidget(QtWidgets.QLabel("Audio Debug"), 3, 0)
        status_grid.addWidget(self.audio_info, 3, 1)

        audio_row = QtWidgets.QHBoxLayout()
        root.addLayout(audio_row)

        self.mic_combo = QtWidgets.QComboBox()
        self.refresh_mic_btn = QtWidgets.QPushButton("Refresh Mics")
        self.apply_mic_btn = QtWidgets.QPushButton("Apply Mic")
        self.reset_btn = QtWidgets.QPushButton("Reset State")
        self.quit_btn = QtWidgets.QPushButton("Quit")

        audio_row.addWidget(QtWidgets.QLabel("Mic Device"))
        audio_row.addWidget(self.mic_combo, stretch=1)
        audio_row.addWidget(self.refresh_mic_btn)
        audio_row.addWidget(self.apply_mic_btn)
        audio_row.addStretch(1)
        audio_row.addWidget(self.reset_btn)
        audio_row.addWidget(self.quit_btn)

        self.video_label = QtWidgets.QLabel("Camera view loading...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background: #101010; color: white; border: 1px solid #333333; }")
        root.addWidget(self.video_label, stretch=2)

        self.wave_plot = pg.PlotWidget(title="Raw Input Audio Waveform")
        self.wave_plot.setBackground("k")
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wave_plot.setYRange(-1.05, 1.05)
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen("y", width=2))
        root.addWidget(self.wave_plot, stretch=1)

        self.note_label = QtWidgets.QLabel(
            "Camera video is shown inside this dashboard. "
            "Use Apply Mic after selecting the webcam mic or the USB mic. "
            "Keyboard shortcuts: R = reset, Q/Esc = quit."
        )
        self.note_label.setWordWrap(True)
        root.addWidget(self.note_label)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Initializing...")

        self.refresh_mic_btn.clicked.connect(self.refresh_audio_devices)
        self.apply_mic_btn.clicked.connect(self.apply_selected_audio_device)
        self.reset_btn.clicked.connect(self.handle_reset)
        self.quit_btn.clicked.connect(self.close)

        self.refresh_audio_devices(select_preferred=True)
        self._apply_audio_availability()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(15)

    def _apply_audio_availability(self) -> None:
        audio_ready = AUDIO_AVAILABLE and self.audio_bundle is not None
        self.mic_combo.setEnabled(audio_ready)
        self.refresh_mic_btn.setEnabled(audio_ready)
        self.apply_mic_btn.setEnabled(audio_ready)

        if not AUDIO_AVAILABLE:
            self.status_bar.showMessage("Audio dependencies missing. Install sounddevice, tensorflow, librosa, PyQt5, pyqtgraph.")
            return

        if self.audio_bundle is None:
            self.status_bar.showMessage(f"Audio model not loaded from {SOUND_MODEL_PATH}")
            return

        self.start_audio_worker(self.mic_combo.currentData())

    def refresh_audio_devices(self, select_preferred: bool = False) -> None:
        previous = self.mic_combo.currentData() if self.mic_combo.count() else None
        preferred = find_preferred_audio_input_device() if select_preferred else previous
        devices = list_audio_devices()

        self.mic_combo.clear()
        self.mic_combo.addItem("Auto (prefer USB adapter)", AUDIO_DEVICE_AUTO)
        self.mic_combo.addItem("Default input device", None)
        for device in devices:
            self.mic_combo.addItem(
                f"{device['index']}: {device['name']} ({device['default_samplerate']} Hz, ch={device['max_input_channels']})",
                int(device["index"]),
            )

        target = preferred if preferred is not None else previous
        if target is not None:
            combo_index = self.mic_combo.findData(target)
            if combo_index >= 0:
                self.mic_combo.setCurrentIndex(combo_index)
        elif select_preferred:
            self.mic_combo.setCurrentIndex(0)

        self.status_bar.showMessage(f"Found {len(devices)} input microphone device(s)")

    def start_audio_worker(self, device_index: Any) -> None:
        if not (AUDIO_AVAILABLE and self.audio_bundle is not None):
            return

        self.stop_audio_worker()

        selected_data = device_index
        allow_fallback = False
        if selected_data == AUDIO_DEVICE_AUTO:
            selected_data = find_preferred_audio_input_device()
            allow_fallback = True
        elif selected_data is None:
            allow_fallback = True
        else:
            selected_data = resolve_audio_input_device(selected_data)

        audio_model, audio_device_ctx = self.audio_bundle
        self.audio_worker = AudioWorker(
            audio_model,
            audio_device_ctx,
            device=selected_data,
            allow_fallback=allow_fallback,
        )
        self.audio_worker.start()

        if selected_data is None:
            selected_text = "Default input device"
        else:
            selected_text = get_audio_device_name(selected_data)

        debug = self.audio_worker.get_debug_snapshot()
        if bool(debug.get("stream_open", False)):
            self.status_bar.showMessage(f"Audio worker started on {debug.get('selected_device_name', selected_text)}")
        else:
            self.status_bar.showMessage(
                f"Audio open failed on {selected_text}: {debug.get('last_error', 'unknown error')}"
            )

    def stop_audio_worker(self) -> None:
        if self.audio_worker is not None:
            self.audio_worker.stop()
            self.audio_worker = None

    def apply_selected_audio_device(self) -> None:
        self.start_audio_worker(self.mic_combo.currentData())

    def handle_reset(self) -> None:
        reset_state()
        self.status_bar.showMessage("State reset to RUBBING")

    def _update_audio_dashboard(self) -> None:
        self.stage_value.setText(str(STATE["stage"]))
        self.sound_value.setText(str(STATE["sound_status"]))

        if self.audio_worker is None:
            self.pred_value.setText("Audio worker not running")
            self.audio_info.setText("No waveform available")
            self.wave_curve.setData(np.zeros(1024, dtype=np.float32))
            return

        waveform = self.audio_worker.get_latest_waveform()
        if waveform.size > 0:
            self.wave_curve.setData(waveform)

        debug = self.audio_worker.get_debug_snapshot()
        self.pred_value.setText(
            f"{debug.get('last_label', 'Waiting...')} ({float(debug.get('last_conf', 0.0)):.2f})"
            f" -> {debug.get('last_decision', 'Waiting...')}"
        )
        probabilities = debug.get("probabilities", {}) or {}
        top_probs = sorted(
            ((str(name), float(prob)) for name, prob in probabilities.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        probability_text = ", ".join(f"{name}={prob:.2f}" for name, prob in top_probs) or "no probabilities"
        self.audio_info.setText(
            f"mic={debug.get('selected_device_name', 'unknown')} | "
            f"{probability_text} | "
            f"rms={float(debug.get('rms', 0.0)):.4f} | "
            f"peak={float(debug.get('peak', 0.0)):.4f} | "
            f"active={float(debug.get('active_ratio', 0.0)):.3f} | "
            f"stream={'open' if bool(debug.get('stream_open', False)) else 'closed'} | "
            f"sr={int(debug.get('input_sr', 0) or 0)}/{int(debug.get('model_sr', 0) or 0)} | "
            f"backend={debug.get('model_backend', '')} | "
            f"err={debug.get('last_error', '')}"
        )

    def _display_frame_in_gui(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimg)

        target_size = self.video_label.size()
        if target_size.width() > 1 and target_size.height() > 1:
            pixmap = pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        self.video_label.setPixmap(pixmap)

    def _process_one_frame(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        if self.frame_count % INFER_SKIP == 0:
            if not STATE["rubbing_done"]:
                annotated, info = process_rubbing_frame(frame)
                annotated, is_rubbing = compute_rubbing(annotated, info)

                combined_sync_ok, visual_recent, audio_sync_active = rubbing_sync_ready(is_rubbing, info)
                if combined_sync_ok:
                    STATE["rubbing_sync_hits"] += 1
                else:
                    STATE["rubbing_sync_hits"] = 0

                if STATE["rubbing_sync_hits"] >= RUBBING_SYNC_CONFIRM_FRAMES:
                    STATE["rubbing_done"] = True
                    STATE["stage"] = "ACID"
                    STATE["acid_positive_streak"] = 0
                    STATE["rubbing_sync_hits"] = 0
                    logger.info(">>> RUBBING CONFIRMED - switching to ACID stage.")

                if STATE["rubbing_done"]:
                    frame_h = int(annotated.shape[0])
                    cv2.putText(
                        annotated,
                        "RUBBING CONFIRMED! -> ACID TEST",
                        (30, max(60, frame_h // 2)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.9,
                        (0, 255, 255),
                        3,
                    )
            else:
                annotated, acid_detected, _acid_info = process_acid_frame(frame)
                if acid_detected:
                    STATE["acid_positive_streak"] += 1
                else:
                    STATE["acid_positive_streak"] = 0

                if STATE["acid_positive_streak"] >= ACID_CONFIRM_FRAMES:
                    STATE["stage"] = "COMPLETED"
                    logger.info(">>> ACID DETECTED - COMPLETED.")

            self.last_annotated = annotated.copy()
        else:
            self.last_annotated = frame if self.last_annotated is None else self.last_annotated

        display = draw_status(self.last_annotated.copy())
        self._display_frame_in_gui(display)

    def on_tick(self) -> None:
        if self._tick_busy or self._closed:
            return

        self._tick_busy = True
        try:
            self._update_audio_dashboard()
            self._process_one_frame()
        finally:
            self._tick_busy = False

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        if key == QtCore.Qt.Key_R:
            self.handle_reset()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._closed:
            event.accept()
            return

        self._closed = True
        logger.info("Shutting down...")

        if hasattr(self, "timer"):
            self.timer.stop()

        self.stop_audio_worker()
        self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if HAILO_RUNTIME is not None:
            HAILO_RUNTIME.close()

        event.accept()


# ====================== MAIN ======================
def main() -> None:
    global HAILO_RUNTIME, MODEL_STONE, MODEL_GOLD, MODEL_ACID

    if not PYQT_AVAILABLE:
        logger.error("PyQt5 and pyqtgraph are required for the requested dashboard UI.")
        return

    if not HAILO_AVAILABLE:
        logger.error("hailo_platform import failed. Install HailoRT Python bindings on your Raspberry Pi first.")
        return

    cap = VideoStreamWidget(CAMERA_ID, FRAME_W, FRAME_H)
    logger.info("[Camera] Threaded capture started. Frames are processed at camera/ROI resolution.")

    try:
        HAILO_RUNTIME, MODEL_STONE, MODEL_GOLD, MODEL_ACID = init_hailo_models()
    except Exception as e:
        logger.error("Failed to initialize Hailo models: %s", e)
        cap.release()
        return

    audio_bundle = load_audio_model() if AUDIO_AVAILABLE else None
    if AUDIO_AVAILABLE and audio_bundle is None:
        logger.warning("Audio model not loaded - dashboard will show audio status only.")
    elif not AUDIO_AVAILABLE:
        logger.warning("Audio runtime dependencies are missing.")
    else:
        logger.info("[Audio] Dashboard device list: %s", [(d["index"], d["name"]) for d in list_audio_devices()])

    logger.info("Starting PyQt dashboard. Press Q/ESC in the OpenCV window or use Quit in the dashboard.")

    if hasattr(QtCore.Qt, "AA_UseSoftwareOpenGL"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    window = GoldTestingWindow(cap, audio_bundle)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
