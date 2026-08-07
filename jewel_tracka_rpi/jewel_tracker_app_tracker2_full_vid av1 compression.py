"""
Jewel Bag Tracker
=================
PyQt5 application for tracking jewels being placed into a bag using
two segmentation models (bag + jewel). Counts placements and raises
alerts on suspicious activity.

Inference runs in background threads every INFER_INTERVAL frames.
Lucas-Kanade optical flow + Kalman filtering bridges the gaps for
smooth, low-latency tracking on resource-constrained hardware (RPi 5).
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import platform
import queue
import threading
import subprocess
import shutil
import glob
from collections import deque
from datetime import datetime

# Load ONNX Runtime before PyQt5 on Windows. Loading it lazily from the count
# worker after Qt is active can fail DLL initialization.
import siglip_jewel_counter as sjc

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QLineEdit, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QListWidget, QListWidgetItem,
    QComboBox, QFileDialog, QSizePolicy,
    QRubberBand
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QSize, QEvent
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPalette


# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_BACKEND = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
DEFAULT_FPS    = 30   # processing/capture frame rate (editable in the UI)

# ── Inference ─────────────────────────────────────────────────────────────────
INFER_INTERVAL       = 1         # run each model every N frames
INFER_SIZE           = (320, 320) # input resize before inference (width, height)
FP_FILTER_MODEL_PATH_BAG   = "fp_filter_resnet18_bag.pt"
HSV_FP_FILTER_STRIP_PATH   = "hsv_fp_filter_strip.pt"
FP_FILTER_CONF             = 0.6
BAG_IOU_THRESHOLD    = 0.45
CONF_THRESHOLD       = 0.50       # detection confidence (hardcoded, not in UI)

BAG_MODEL_FILE         = "best_coverbag.pt"
STRIP_MODEL_FILE       = "strip-l.pt"
DEFAULT_RECT_THRESHOLD = 0.92   # bag contour-area / bounding-rect-area → "rectangular"
DEFAULT_SEAL_WAIT_SEC  = 2.0    # seconds after seal disappears before announcing closed
STRIP_DEBOUNCE_FRAMES  = 4      # consecutive no-detect frames before "seal gone"

# ── Tracking ──────────────────────────────────────────────────────────────────
TEMPLATE_SIZE = 31
SEARCH_MARGIN = 50
FB_THRESHOLD  = 2.0
NCC_THRESHOLD = 0.50
MAX_DROPOUT   = 15

# ── Mean-shift (jewel tracker only) ───────────────────────────────────────────
N_BINS         = 16    # H×S histogram bins per channel  (16×16 = 256 total)
MS_MAX_ITERS   = 20    # iteration cap per frame
MS_EPS         = 0.5   # convergence threshold (pixels)
BHATT_GOOD     = 0.75  # ρ ≥ this → healthy mean-shift (green)
BHATT_COAST    = 0.50  # ρ ≥ this → marginal; Kalman corrects (yellow)
MS_MIN_HALF    = 20    # minimum search-window half-size (pixels)
MS_MAX_DROPOUT = 20    # consecutive weak mean-shift frames before dropping
MS_SEG_MISS_MAX = 4    # consecutive "model found nothing" frames before dropping
                       # absorbs 1-2 frame (None,None) bursts from overlap clipping

# ── IMM filter (jewel tracker only) ───────────────────────────────────────────
# Interacting Multiple Model filter: two constant-velocity models mixed by a
# Markov chain. A "steady" model (low process noise) stays smooth on slow/linear
# motion; a "maneuver" model (high process noise) reacts fast when the jewel
# darts, so the coasted estimate stops wandering during quick moves.
IMM_R           = 0.1    # measurement noise (px²) — how much to trust detections
IMM_Q_STEADY    = 1e-2   # process noise of the smooth/steady model
IMM_Q_MANEUVER  = 5.0    # process noise of the agile/maneuvering model
IMM_P0          = 10.0   # initial state covariance
IMM_TRANS       = ((0.95, 0.05),   # P(stay steady),    P(steady → maneuver)
                   (0.05, 0.95))   # P(maneuver → steady), P(stay maneuver)

_LK_PARAMS = dict(
    winSize=(25, 25), maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


# ── LK / Kalman helpers ───────────────────────────────────────────────────────

def make_kalman(x: float, y: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix   = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    kf.statePost           = np.array([[x], [y], [0.], [0.]], dtype=np.float32)
    return kf


class IMMFilter:
    """
    Interacting Multiple Model filter (Blom & Bar-Shalom) for 2-D position
    tracking — a drop-in replacement for the constant-velocity Kalman used by
    the jewel tracker.  Runs two constant-velocity sub-filters in parallel:

      • model 0 "steady"   — low  process noise → smooth on slow/linear motion
      • model 1 "maneuver" — high process noise → agile when the jewel darts

    A Markov chain mixes the two each cycle and the measurement likelihood
    re-weights them, so the combined estimate locks onto fast, direction-
    changing motion that a single CV model overshoots (the "wandering" point).

    Exposes the subset of cv2.KalmanFilter the trackers rely on:
        predict()            -> (4,1) combined predicted state
        correct(z: (2,1))    -> (4,1) combined updated state
        statePost: (4,1)      = combined state  [x, y, vx, vy]
    """

    def __init__(self, x: float, y: float):
        self._n = 4
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * IMM_R
        self.Q = [np.eye(4, dtype=np.float64) * IMM_Q_STEADY,
                  np.eye(4, dtype=np.float64) * IMM_Q_MANEUVER]
        self.Pi = np.array(IMM_TRANS, dtype=np.float64)   # transition matrix
        self.mu = np.array([0.5, 0.5], dtype=np.float64)  # model probabilities

        x0 = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self.x = [x0.copy(), x0.copy()]                    # per-model states
        self.P = [np.eye(4) * IMM_P0, np.eye(4) * IMM_P0]  # per-model covariances
        self._xpre = [x0.copy(), x0.copy()]                # predicted (pre-update)
        self._Ppre = [self.P[0].copy(), self.P[1].copy()]
        self._cbar = self.mu.copy()
        self.statePost = x0.astype(np.float32)

    def _mix(self):
        """Blend per-model states using the Markov mixing probabilities."""
        cbar = self.Pi.T @ self.mu
        cbar = np.maximum(cbar, 1e-12)
        x0, P0 = [], []
        for j in range(2):
            mix = (self.Pi[:, j] * self.mu) / cbar[j]      # μ_{i|j}
            xj = mix[0] * self.x[0] + mix[1] * self.x[1]
            Pj = np.zeros((4, 4))
            for i in range(2):
                d = self.x[i] - xj
                Pj += mix[i] * (self.P[i] + d @ d.T)
            x0.append(xj)
            P0.append(Pj)
        self._cbar = cbar
        return x0, P0

    def predict(self):
        x0, P0 = self._mix()
        for j in range(2):
            self._xpre[j] = self.F @ x0[j]
            self._Ppre[j] = self.F @ P0[j] @ self.F.T + self.Q[j]
            # with no correction this cycle the posterior == the prediction
            self.x[j] = self._xpre[j].copy()
            self.P[j] = self._Ppre[j].copy()
        # let the mode probabilities evolve through the chain while coasting
        self.mu = self._cbar.copy()
        xc = self.mu[0] * self._xpre[0] + self.mu[1] * self._xpre[1]
        self.statePost = xc.astype(np.float32)
        return self.statePost

    def correct(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(2, 1)
        like = np.zeros(2)
        for j in range(2):
            xpre, Ppre = self._xpre[j], self._Ppre[j]
            S    = self.H @ Ppre @ self.H.T + self.R
            Sinv = np.linalg.inv(S)
            K    = Ppre @ self.H.T @ Sinv
            innov = z - self.H @ xpre
            self.x[j] = xpre + K @ innov
            self.P[j] = (np.eye(4) - K @ self.H) @ Ppre
            det     = max(float(np.linalg.det(2 * np.pi * S)), 1e-12)
            md      = float((innov.T @ Sinv @ innov)[0, 0])
            like[j] = np.exp(-0.5 * md) / np.sqrt(det)
        # update model probabilities  μ_j = c̄_j · Λ_j / Σ
        post = self._cbar * like
        s = post.sum()
        self.mu = post / s if s > 1e-12 else np.array([0.5, 0.5])
        xc = self.mu[0] * self.x[0] + self.mu[1] * self.x[1]
        self.statePost = xc.astype(np.float32)
        return self.statePost


def make_imm(x: float, y: float) -> IMMFilter:
    """Constant-velocity IMM (steady + maneuver) for the jewel tracker."""
    return IMMFilter(x, y)


def extract_patch(gray: np.ndarray, x: float, y: float) -> np.ndarray | None:
    half = TEMPLATE_SIZE // 2
    xi, yi = int(round(x)), int(round(y))
    if (xi - half < 0 or yi - half < 0
            or xi + half + 1 > gray.shape[1]
            or yi + half + 1 > gray.shape[0]):
        return None
    return gray[yi - half: yi + half + 1, xi - half: xi + half + 1].copy()


def template_search(gray: np.ndarray, px: float, py: float,
                    tmpl: np.ndarray) -> tuple[float, float, float]:
    half = TEMPLATE_SIZE // 2
    xi, yi = int(round(px)), int(round(py))
    x1 = max(xi - half - SEARCH_MARGIN, 0)
    y1 = max(yi - half - SEARCH_MARGIN, 0)
    x2 = min(xi + half + SEARCH_MARGIN + 1, gray.shape[1])
    y2 = min(yi + half + SEARCH_MARGIN + 1, gray.shape[0])
    region = gray[y1:y2, x1:x2]
    if region.shape[0] < TEMPLATE_SIZE or region.shape[1] < TEMPLATE_SIZE:
        return px, py, 0.0
    result = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    return float(x1 + loc[0] + half), float(y1 + loc[1] + half), float(score)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def get_centroid(mask: np.ndarray):
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def get_largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def point_in_contour(point, contour) -> bool:
    if contour is None or point is None:
        return False
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def mask_coverage(jewel_mask: np.ndarray, bag_mask: np.ndarray) -> float:
    if jewel_mask is None or bag_mask is None:
        return 0.0
    h1, w1 = jewel_mask.shape[:2]
    h2, w2 = bag_mask.shape[:2]
    if (h1, w1) != (h2, w2):
        bag_mask = cv2.resize(bag_mask, (w1, h1))
    intersection = cv2.bitwise_and(jewel_mask, bag_mask)
    return float(intersection.sum()) / (float(jewel_mask.sum()) + 1e-6)


def trajectory_goes_toward(trajectory: deque, target_xy) -> bool:
    if len(trajectory) < 5 or target_xy is None:
        return False
    start, end = trajectory[0], trajectory[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    to_target = (target_xy[0] - end[0], target_xy[1] - end[1])
    return dx * to_target[0] + dy * to_target[1] > 0


def measure_rectangularity(mask: np.ndarray) -> float:
    """Ratio of contour area to its axis-aligned bounding-rect area.
    Returns 1.0 for a perfect rectangle, lower for irregular shapes."""
    cnt = get_largest_contour(mask)
    if cnt is None:
        return 0.0
    area = float(cv2.contourArea(cnt))
    if area < 1.0:
        return 0.0
    _x, _y, w, h = cv2.boundingRect(cnt)
    return area / max(float(w * h), 1.0)


# ── TTS helper ───────────────────────────────────────────────────────────────

def _speak(text: str) -> None:
    """Fire-and-forget TTS — tries pyttsx3 then Windows SAPI PowerShell fallback."""
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            try:
                import subprocess
                safe = text.replace("'", "").replace('"', "")
                subprocess.Popen(
                    ["powershell", "-WindowStyle", "Hidden", "-c",
                     f"Add-Type -AssemblyName System.Speech; "
                     f"(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                     f".Speak('{safe}')"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()


# ── False-positive filter (ResNet18) ─────────────────────────────────────────

class FPFilter:
    """
    Classifies YOLO detection crops as true positive or false positive.
    Applied per-detection to both the jewel and bag models (each with its
    own ResNet18 weights). Falls back to True (assume TP) when the model
    file is absent.
    """

    def __init__(self, model_path: str, conf: float = FP_FILTER_CONF,
                 tp_class: int = 1):
        self.conf     = conf
        self.tp_class = tp_class   # output class index that means "true positive"
        self.model    = None
        self.device   = None
        if not model_path or not os.path.exists(model_path):
            print(f"[FPFilter] model not found: {model_path} — filter disabled")
            return
        try:
            import torch
            import torchvision.models as tvm
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model  = tvm.resnet18(weights=None)
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(self.model.fc.in_features, 2),
            )
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[FPFilter] ResNet18 loaded from {model_path}")
        except Exception as e:
            print(f"[FPFilter] load error: {e}")
            self.model = None

    def is_true_positive(self, crop_bgr: np.ndarray) -> bool:
        if self.model is None:
            return True
        try:
            import torch
            import torchvision.transforms as T
            rgb = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
            t   = T.Compose([T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
            x = t(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = torch.nn.functional.softmax(self.model(x), dim=1)
            return float(p[0, self.tp_class]) >= self.conf
        except Exception as e:
            print(f"[FPFilter] inference error: {e}")
            return True


# ── HSV-CNN false-positive filter (strip only) ────────────────────────────────
# Architecture must match strip_fp_trainer.py exactly.  Train with that app,
# which saves hsv_fp_filter_strip.pt to the script directory.

_HSV_CROP_SIZE = 64   # must equal CROP_SIZE in strip_fp_trainer.py


def _make_strip_hsv_cnn():
    """Build a fresh StripHSVCNN (same 3-block architecture as the trainer)."""
    import torch.nn as nn
    d = _HSV_CROP_SIZE // 8
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * d * d, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.4), nn.Linear(128, 2),
            )
        def forward(self, x):
            return self.head(self.features(x))
    return _Net()


class HSVFPFilter:
    """
    Strip FP filter backed by a lightweight HSV CNN trained with
    strip_fp_trainer.py.  Falls back to True (pass-through) when the
    model file is absent — identical fail-safe to FPFilter.
    Class 1 = True Positive (real seal strip).
    """

    def __init__(self, model_path: str, conf: float = FP_FILTER_CONF):
        self.conf   = conf
        self.model  = None
        self.device = None
        if not model_path or not os.path.exists(model_path):
            print(f"[HSVFPFilter] model not found: {model_path} — filter disabled")
            return
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = _make_strip_hsv_cnn()
            net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = net.to(self.device).eval()
            print(f"[HSVFPFilter] HSV CNN loaded from {model_path}")
        except Exception as e:
            print(f"[HSVFPFilter] load error: {e}")
            self.model = None

    def is_true_positive(self, crop_bgr: np.ndarray) -> bool:
        if self.model is None:
            return True
        try:
            import torch
            hsv = cv2.cvtColor(cv2.resize(crop_bgr, (_HSV_CROP_SIZE, _HSV_CROP_SIZE)),
                               cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] /= 180.0
            hsv[:, :, 1] /= 255.0
            hsv[:, :, 2] /= 255.0
            x = torch.from_numpy(hsv.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = torch.nn.functional.softmax(self.model(x), dim=1)
            return float(p[0, 1]) >= self.conf
        except Exception as e:
            print(f"[HSVFPFilter] inference error: {e}")
            return True


# ── Segmentation model wrapper ────────────────────────────────────────────────

class SegmentationModel:
    """
    Wraps either a YOLO-seg (ultralytics) or RF-DETR-seg (rfdetr) model.
    Returns a combined binary uint8 mask (same H×W as input frame), or None.
    """

    def __init__(self, model_path: str, label: str,
                 conf_threshold: float = 0.5, model_type: str = "yolo",
                 fp_filter: "FPFilter | None" = None,
                 iou_threshold: float = 0.45):
        self.label          = label
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.model_type     = model_type.lower()
        self.fp_filter      = fp_filter   # FPFilter instance, or None to disable
        self.model          = None
        if not model_path:
            return
        try:
            if self.model_type == "rfdetr":
                self.model = self._load_rfdetr(model_path)
            else:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            print(f"[{label}] {self.model_type.upper()} loaded: {model_path}")
        except Exception as e:
            print(f"[{label}] Model load error: {e}")

    def _load_rfdetr(self, model_path: str):
        """Auto-detect and load RF-DETR model (detection or segmentation)."""
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Try segmentation models first (most common for this app)
        seg_models = ["RFDETRSegNano", "RFDETRSegTiny", "RFDETRSegSmall", "RFDETRSegBase"]
        for model_class in seg_models:
            try:
                if hasattr(__import__("rfdetr"), model_class):
                    ModelCls = getattr(__import__("rfdetr"), model_class)
                    return ModelCls(pretrain_weights=model_path)
            except Exception:
                continue

        # Fall back to detection models if segmentation fails
        det_models = ["RFDETRNano", "RFDETRTiny", "RFDETRSmall", "RFDETRBase"]
        for model_class in det_models:
            try:
                if hasattr(__import__("rfdetr"), model_class):
                    ModelCls = getattr(__import__("rfdetr"), model_class)
                    return ModelCls(pretrain_weights=model_path)
            except Exception:
                continue

        # Last resort: try RFDETRBase (legacy)
        from rfdetr import RFDETRBase
        return RFDETRBase(pretrain_weights=model_path)

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        if self.model is None:
            return None
        h, w = frame.shape[:2]
        try:
            if self.model_type == "rfdetr":
                mask = self._detect_rfdetr(frame, h, w)
                return self._clean_mask(mask) if mask is not None else None
            else:
                mask = self._detect_yolo_seg(frame, h, w)
                return self._clean_mask(mask) if mask is not None else None
        except Exception as e:
            print(f"[{self.label}] Inference error: {e}")
            return None

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """
        Remove jagged boundary protrusions from a segmentation mask.

        Steps:
          1. Morphological opening (erode→dilate) — cuts off thin spikes
             without significantly shrinking the main blob.
          2. Keep only the largest contour and fill it solid — eliminates
             detached noise fragments and gives a smooth, filled shape.
        """
        # Kernel ≈ 2 % of the shorter image dimension, minimum 5 px
        k = max(5, int(min(mask.shape[:2]) * 0.02) | 1)  # odd number
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask   # nothing survived — return original rather than blank

        out = np.zeros_like(mask)
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255,
                         cv2.FILLED)
        return out

    def _detect_yolo_seg(self, frame, h, w):
        """Use segmentation masks from the YOLO-seg head.
        Falls back to a filled bounding-box if the loaded model has no seg head
        (so a plain detection .pt still works without crashing)."""
        results = self.model(frame, conf=self.conf_threshold,
                             iou=self.iou_threshold, verbose=False)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            has_masks = r.masks is not None and len(r.masks) > 0
            combined  = np.zeros((h, w), dtype=np.uint8)
            for idx, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                # FP filter on bbox crop (works for both seg and det models)
                if self.fp_filter is not None:
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size > 0 and not self.fp_filter.is_true_positive(crop):
                        continue
                if has_masks:
                    mask_np = r.masks.data[idx].cpu().numpy()
                    mask_np = cv2.resize(mask_np, (w, h),
                                         interpolation=cv2.INTER_LINEAR)
                    combined[mask_np > 0.5] = 255
                else:
                    # plain detection model — solid rectangle fallback
                    cv2.rectangle(combined,
                                  (max(0, x1), max(0, y1)),
                                  (min(w - 1, x2), min(h - 1, y2)),
                                  255, cv2.FILLED)
            if combined.sum() > 0:
                return combined
        return None

    def _detect_rfdetr(self, frame, h, w):
        from PIL import Image as _PILImage
        pil = _PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            dets = self.model.predict(pil, threshold=self.conf_threshold,
                                      iou_threshold=self.iou_threshold)
        except TypeError:
            dets = self.model.predict(pil, threshold=self.conf_threshold)
        if dets.mask is not None and len(dets.mask) > 0:
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in dets.mask:
                combined = cv2.bitwise_or(
                    combined,
                    cv2.resize(m.astype(np.uint8) * 255, (w, h),
                               interpolation=cv2.INTER_NEAREST))
            if combined.sum() > 0:
                return combined
        return None

    def set_threshold(self, val: float):
        self.conf_threshold = val

    def set_iou(self, val: float):
        self.iou_threshold = val


# ── Segmentation background worker ────────────────────────────────────────────

class SegWorker(threading.Thread):
    """
    Runs a SegmentationModel in a daemon thread (never blocks the UI/capture loop).
    Resizes input to INFER_SIZE for speed; returns centroid + full-res mask.

    get_result() returns:
        None            → inference not finished yet  (keep LK bridging)
        (None, None)    → model ran, nothing detected (object gone → drop point)
        ((cx,cy), mask) → model found the object      (snap + refresh template)
    """

    def __init__(self, seg_model: SegmentationModel):
        super().__init__(daemon=True)
        self._model = seg_model
        self._in  = queue.Queue(maxsize=1)
        self._out = queue.Queue(maxsize=1)

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(frame)

    def get_result(self):
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def run(self) -> None:
        while True:
            frame = self._in.get()
            orig_h, orig_w = frame.shape[:2]
            iw, ih = INFER_SIZE

            # Aspect-preserving letterbox: scale to fit, pad the rest with black.
            # Squashing a non-square crop into the square inference size distorts
            # the object and makes the mask clamp/protrude at the crop edges.
            scale = min(iw / orig_w, ih / orig_h)
            nw, nh = max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))
            px, py = (iw - nw) // 2, (ih - nh) // 2
            canvas = np.zeros((ih, iw, 3), dtype=frame.dtype)
            canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))

            mask_canvas = self._model.detect(canvas)
            if mask_canvas is not None:
                # Strip the padding, then map the mask back to the crop size.
                mask_small = mask_canvas[py:py + nh, px:px + nw]
                mask     = cv2.resize(mask_small, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
                centroid = get_centroid(mask)
                result   = (centroid, mask)
            else:
                result = (None, None)
            try:
                self._out.get_nowait()
            except queue.Empty:
                pass
            self._out.put(result)


# ── Per-object centroid tracker ───────────────────────────────────────────────

class CentroidTracker:
    """
    Hybrid LK + Kalman tracker for a single object's centroid.

    Call update() every frame with the latest grayscale frame and the
    SegWorker result for this object.  When the worker fires:
      (centroid, mask)  → snap to model centroid, refresh LK template
      (None, None)      → object gone, drop tracking point
    Between model fires, LK optical flow + Kalman keep the position smooth.
    """

    def __init__(self):
        self.position:  tuple | None      = None  # current (cx, cy) float
        self.last_mask: np.ndarray | None = None  # most recent segmentation mask
        self.snapped = False   # True on the frame the model just anchored the point
        self._old_gray = None
        self._pt       = None  # (1, 1, 2) float32 for calcOpticalFlowPyrLK
        self._kf       = None
        self._tmpl     = None
        self._dropout  = 0
        self._seg_miss = 0  # consecutive frames the model returned (None, None)

    def update(self, gray: np.ndarray, seg_result) -> tuple:
        """Returns (position, last_mask)."""
        self.snapped = False

        if seg_result is not None:
            centroid, mask = seg_result
            if centroid is not None:
                self._snap(gray, float(centroid[0]), float(centroid[1]))
                self.last_mask = mask
                self._seg_miss = 0
            else:
                self._seg_miss += 1
                if self._seg_miss > MS_SEG_MISS_MAX:
                    self._drop()

        if self._pt is not None and self._old_gray is not None:
            self._lk_step(gray)

        self._old_gray = gray.copy()
        return self.position, self.last_mask

    # ── internal ──────────────────────────────────────────────────────────────

    def _snap(self, gray: np.ndarray, cx: float, cy: float) -> None:
        """Anchor tracking to model detection — always wins over LK."""
        self._pt = np.array([[[cx, cy]]], dtype=np.float32)
        if self._kf is None:
            self._kf = make_kalman(cx, cy)
        else:
            self._kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        tmpl = extract_patch(gray, cx, cy)
        if tmpl is not None:
            self._tmpl = tmpl
        self._dropout = 0
        self.snapped  = True
        self.position = (cx, cy)

    def _drop(self) -> None:
        """Object gone — clear all tracking state."""
        self._pt = self._kf = self._tmpl = None
        self._dropout  = 0
        self._seg_miss = 0
        self.position  = None
        self.last_mask = None

    def _lk_step(self, gray: np.ndarray) -> None:
        """LK forward–backward + template NCC + Kalman — runs every frame."""
        fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._old_gray, gray, self._pt, None, **_LK_PARAMS)

        lk_ok = False
        if fwd is not None and st_fwd is not None and st_fwd[0, 0] == 1:
            bwd, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._old_gray, fwd, None, **_LK_PARAMS)
            fb    = np.linalg.norm(self._pt.reshape(2) - bwd.reshape(2))
            lk_ok = st_bwd[0, 0] == 1 and fb < FB_THRESHOLD

        pred = self._kf.predict() if self._kf else None
        kx   = float(pred[0, 0]) if pred is not None else self.position[0]
        ky   = float(pred[1, 0]) if pred is not None else self.position[1]

        fx, fy = kx, ky
        good   = False

        if lk_ok:
            lkx = float(fwd[0, 0, 0])
            lky = float(fwd[0, 0, 1])
            if self._tmpl is not None:
                tx, ty, score = template_search(gray, lkx, lky, self._tmpl)
                if score >= NCC_THRESHOLD:
                    fx, fy = tx, ty
                    good   = True
            else:
                fx, fy = lkx, lky
                good   = True

        if self._kf is not None and good:
            self._kf.correct(np.array([[fx], [fy]], dtype=np.float32))
            fx = float(self._kf.statePost[0, 0])
            fy = float(self._kf.statePost[1, 0])
            self._dropout = 0
        else:
            self._dropout += 1

        if self._dropout > MAX_DROPOUT:
            self._drop()
            return

        h, w = gray.shape
        if not (0 <= fx < w and 0 <= fy < h):
            self._drop()
            return

        self._pt      = np.array([[[fx, fy]]], dtype=np.float32)
        self.position = (fx, fy)


# ── Comaniciu Mean-Shift Tracker ──────────────────────────────────────────────

class MeanShiftTracker:
    """
    Kernel-based mean-shift tracker — Comaniciu, Ramesh & Meer (2003).

    Target model q:  2-D H×S histogram weighted by Epanechnikov kernel
                     k(r²) = max(0, 1 − r²).  V channel dropped for
                     illumination invariance.
    Per-frame step:  iterative mean-shift maximising Bhattacharyya ρ = Σ√(q·p).
    Update rule:     y_{t+1} = Σ x_i g_i w_i / Σ g_i w_i
                     g_i = max(0, 1−r²)  (Epanechnikov soft window)
                     w_i = √(q[b(x_i)] / p[b(x_i)])  (Eq. 13 in paper)
    """

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins   = n_bins
        self.n_total  = n_bins * n_bins
        self.target_q: np.ndarray | None          = None
        self.pos:      tuple[float, float] | None = None
        self.bw:       tuple[float, float] | None = None  # (half_w, half_h)

    def init(self, hsv: np.ndarray,
             cx: float, cy: float,
             half_w: float, half_h: float) -> None:
        self.bw       = (max(half_w, MS_MIN_HALF), max(half_h, MS_MIN_HALF))
        self.pos      = (cx, cy)
        self.target_q = self._histogram(hsv, cx, cy)

    def update(self, hsv: np.ndarray) -> tuple[float, float, float] | None:
        """Returns (cx, cy, bhattacharyya_score) or None if uninitialised."""
        if self.target_q is None or self.pos is None or self.bw is None:
            return None

        H, W   = hsv.shape[:2]
        bw, bh = self.bw
        cx, cy = float(self.pos[0]), float(self.pos[1])

        for _ in range(MS_MAX_ITERS):
            p_y    = self._histogram(hsv, cx, cy)
            safe_p = np.maximum(p_y, 1e-10)
            dw     = np.sqrt(self.target_q / safe_p)

            x1 = max(0, int(cx - bw));  x2 = min(W, int(cx + bw) + 1)
            y1 = max(0, int(cy - bh));  y2 = min(H, int(cy + bh) + 1)
            roi = hsv[y1:y2, x1:x2]
            if roi.size == 0:
                break

            yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
            dx = (xx - cx) / bw;  dy = (yy - cy) / bh
            g  = np.maximum(0.0, 1.0 - (dx * dx + dy * dy))

            pix_w = g.ravel() * dw[self._bin_indices(roi)]
            tot   = pix_w.sum()
            if tot < 1e-10:
                break

            new_cx = float((xx.ravel() * pix_w).sum() / tot)
            new_cy = float((yy.ravel() * pix_w).sum() / tot)
            shift  = np.hypot(new_cx - cx, new_cy - cy)
            cx = float(np.clip(new_cx, bw, W - bw - 1))
            cy = float(np.clip(new_cy, bh, H - bh - 1))
            if shift < MS_EPS:
                break

        self.pos    = (cx, cy)
        p_final     = self._histogram(hsv, cx, cy)
        score       = float(np.sqrt(
            np.maximum(self.target_q, 0.0) * np.maximum(p_final, 0.0)).sum())
        return cx, cy, score

    def reanchor(self, hsv: np.ndarray,
                 cx: float, cy: float,
                 half_w: float | None = None,
                 half_h: float | None = None) -> None:
        if half_w is not None:
            self.bw = (max(half_w, MS_MIN_HALF),
                       max(half_h if half_h is not None else half_w, MS_MIN_HALF))
        self.pos      = (cx, cy)
        self.target_q = self._histogram(hsv, cx, cy)

    def _histogram(self, hsv: np.ndarray, cx: float, cy: float) -> np.ndarray:
        H, W   = hsv.shape[:2]
        bw, bh = self.bw  # type: ignore[misc]
        x1 = max(0, int(cx - bw));  x2 = min(W, int(cx + bw) + 1)
        y1 = max(0, int(cy - bh));  y2 = min(H, int(cy + bh) + 1)
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            return np.full(self.n_total, 1.0 / self.n_total, dtype=np.float32)
        yy, xx  = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        dx = (xx - cx) / bw;  dy = (yy - cy) / bh
        w   = np.maximum(0.0, 1.0 - (dx * dx + dy * dy)).ravel()
        hist = np.bincount(self._bin_indices(roi), weights=w,
                           minlength=self.n_total).astype(np.float32)
        tot = hist.sum()
        return hist / tot if tot > 0 else np.full(self.n_total, 1.0 / self.n_total, dtype=np.float32)

    def _bin_indices(self, roi: np.ndarray) -> np.ndarray:
        hi = (roi[:, :, 0].astype(np.float32) * (self.n_bins / 180.0)
              ).astype(np.int32).clip(0, self.n_bins - 1)
        si = (roi[:, :, 1].astype(np.float32) * (self.n_bins / 256.0)
              ).astype(np.int32).clip(0, self.n_bins - 1)
        return (hi * self.n_bins + si).ravel()


class MeanShiftCentroidTracker:
    """
    Drop-in replacement for CentroidTracker (jewel only).

    Same interface — update(frame_bgr, seg_result) → (position, last_mask) —
    but bridges between segmentation detections using Comaniciu mean-shift
    on the HSV frame instead of Lucas-Kanade optical flow.

    Colour legend on the overlay:
      Orange  — segmentation just anchored  (snapped this frame)
      Green   — mean-shift healthy  ρ ≥ BHATT_GOOD
      Yellow  — mean-shift marginal ρ ≥ BHATT_COAST  (Kalman corrects)
      Red     — Kalman coasting     ρ <  BHATT_COAST
    """

    def __init__(self):
        self.position:  tuple | None      = None
        self.last_mask: np.ndarray | None = None
        self.snapped  = False
        self.score    = 0.0               # last Bhattacharyya ρ (for overlay)
        self._ms      = MeanShiftTracker()
        self._kf: cv2.KalmanFilter | None = None
        self._dropout  = 0
        self._seg_miss = 0  # consecutive frames the model returned (None, None)

    def update(self, frame_bgr: np.ndarray, seg_result) -> tuple:
        """
        Returns (position, last_mask).

        Loop order mirrors Comaniciu's hybrid tracker (point_tracking_app):
          ① mean-shift bridges between detections every frame; on weak frames
             the search window itself follows the Kalman prediction so a fast
             jewel can be re-acquired,
          ② the segmentation result then re-anchors the target model cleanly,
             so detection frames report the exact detection (no drift).
        """
        self.snapped = False
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # ── ① Mean-shift tracking between detections ──────────────────────
        if self._ms.pos is not None and self.position is not None:
            result = self._ms.update(hsv)
            if result is not None:
                ms_cx, ms_cy, score = result
                self.score = score

                if score >= BHATT_COAST:
                    # Healthy / marginal: trust mean-shift, correct Kalman.
                    if self._kf is not None:
                        self._kf.predict()
                        self._kf.correct(
                            np.array([[ms_cx], [ms_cy]], dtype=np.float32))
                        ms_cx = float(self._kf.statePost[0, 0])
                        ms_cy = float(self._kf.statePost[1, 0])
                    self._dropout = 0
                else:
                    # Lost: coast on the Kalman prediction.
                    self._dropout += 1
                    if self._kf is not None:
                        pred  = self._kf.predict()
                        ms_cx = float(pred[0, 0])
                        ms_cy = float(pred[1, 0])
                    if self._dropout > MS_MAX_DROPOUT:
                        self._drop()

                # Move the search window to the (possibly coasted) estimate so
                # re-acquisition tracks where the jewel is predicted to be.
                if self.position is not None:
                    self._ms.pos  = (ms_cx, ms_cy)
                    self.position = (ms_cx, ms_cy)

        # ── ② Segmentation re-anchor (overrides mean-shift cleanly) ───────
        if seg_result is not None:
            centroid, mask = seg_result
            if centroid is not None:
                cx, cy = float(centroid[0]), float(centroid[1])
                hw, hh = self._bw_from_mask(mask, cx, cy)
                self._ms.reanchor(hsv, cx, cy, hw, hh)
                if self._kf is None:
                    self._kf = make_imm(cx, cy)
                else:
                    self._kf.correct(np.array([[cx], [cy]], dtype=np.float32))
                self.last_mask = mask
                self.position  = (cx, cy)
                self._dropout  = 0
                self._seg_miss = 0
                self.snapped   = True
            else:
                # Buffer consecutive misses — mean-shift keeps bridging during
                # this window, absorbing the 1-2 frame (None,None) bursts that
                # come from the jewel/bag overlap clipping.
                self._seg_miss += 1
                if self._seg_miss > MS_SEG_MISS_MAX:
                    self._drop()

        return self.position, self.last_mask

    def _bw_from_mask(self, mask: np.ndarray | None,
                      cx: float, cy: float) -> tuple[float, float]:
        """Estimate search-window half-size from the segmentation mask bbox."""
        if mask is None:
            return MS_MIN_HALF, MS_MIN_HALF
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            r = max(float(np.sqrt((mask > 0).sum() / np.pi)), MS_MIN_HALF)
            return r, r
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return max(w / 2.0, MS_MIN_HALF), max(h / 2.0, MS_MIN_HALF)

    def _drop(self) -> None:
        self._ms       = MeanShiftTracker()
        self._kf       = None
        self._dropout  = 0
        self._seg_miss = 0
        self.score     = 0.0
        self.position  = None
        self.last_mask = None


# ── Recording ─────────────────────────────────────────────────────────────────
PRE_BUFFER_SEC  = 0.5   # seconds of rolling pre-appearance buffer (shorter → smaller clips)
POST_RECORD_SEC = 0.5   # seconds to continue recording after jewel vanishes

# ── Saved-clip compression ────────────────────────────────────────────────────
# Applied ONLY to written files — the live preview stays full resolution / rate.
# Half resolution (¼ the pixels) + frame-skipping to a low FPS + H.264 keeps each
# clip tiny.  Codecs are tried best-compression-first (see _select_writer).
REC_SCALE  = 0.5   # downscale factor for saved clips (0.4 → 512×288 from 1280×720)
REC_FPS    = 6     # frame rate of saved clips; capture frames are skipped to match
REC_CODECS = ("avc1", "H264", "X264", "mp4v")  # preferred → fallback (H.264 first)

# ── AV1 post-record transcode ─────────────────────────────────────────────────
# Clips are recorded live with the fast H.264 codec above (never blocks capture).
# When a clip closes, it is re-encoded to AV1 (SVT-AV1) in a detached background
# thread — typically ~4× smaller than H.264 — and the AV1 file atomically
# replaces the original .mp4, so the path the UI already knows stays valid.
# Requires ffmpeg with libsvtav1 on PATH; if absent, the H.264 clip is kept.
AV1_TRANSCODE = True
AV1_FFMPEG    = "ffmpeg"   # ffmpeg executable name or absolute path
AV1_PRESET    = 8          # SVT-AV1 speed preset: 0 = slow/best … 13 = fastest
AV1_CRF       = 35         # SVT-AV1 quality: lower = better/larger (≈30–40 good here)

_ffmpeg_path_cache = None   # resolved once, then reused


def _resolve_ffmpeg():
    """Locate an ffmpeg executable. Tries AV1_FFMPEG (PATH or absolute), then —
    on Windows — the winget package dir, since a freshly winget-installed ffmpeg
    isn't on PATH for shells opened before the install. Returns a path or None."""
    global _ffmpeg_path_cache
    if _ffmpeg_path_cache is not None:
        return _ffmpeg_path_cache or None   # "" means "looked, found nothing"

    found = shutil.which(AV1_FFMPEG) or (
        AV1_FFMPEG if os.path.isabs(AV1_FFMPEG) and os.path.exists(AV1_FFMPEG) else None)
    if not found and os.name == "nt":
        pat = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                           "Microsoft", "WinGet", "Packages",
                           "Gyan.FFmpeg*", "**", "ffmpeg.exe")
        hits = glob.glob(pat, recursive=True)
        found = hits[0] if hits else None

    _ffmpeg_path_cache = found or ""
    return found


def _spawn_av1_transcode(src_path: str):
    """Re-encode a finished H.264 clip to AV1 in a daemon thread, then replace
    the original file in place. Never raises into the caller and never blocks the
    capture loop; on any failure (including missing ffmpeg) the H.264 clip is kept."""
    if not (AV1_TRANSCODE and src_path and os.path.exists(src_path)):
        return
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        print("[AV1] ffmpeg/libsvtav1 not found — kept H.264 clip.")
        return

    def _work():
        tmp = src_path + ".av1.tmp.mp4"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-i", src_path,
               "-c:v", "libsvtav1", "-preset", str(AV1_PRESET), "-crf", str(AV1_CRF),
               "-pix_fmt", "yuv420p", "-an", tmp]
        try:
            r = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, src_path)          # AV1 now lives at the original path
                print(f"[AV1] transcoded → {src_path}")
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"[AV1] transcode failed (kept H.264): {(r.stderr or '').strip()[:200]}")
        except FileNotFoundError:
            print("[AV1] ffmpeg/libsvtav1 not found on PATH — kept H.264 clip.")
        except Exception as e:                     # pragma: no cover (defensive)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            print(f"[AV1] transcode error (kept H.264): {e}")

    threading.Thread(target=_work, daemon=True, name="av1-transcode").start()


class RecordingController:
    """
    Triggers clip recording when a jewel is present in the tracked region.

    States:
      IDLE      — jewel absent; pre-buffer rolling, not writing
      RECORDING — jewel present; writer open
      TAIL      — jewel just disappeared; counting down before closing writer

    Call push_frame() every frame (keeps pre-buffer current), then update().
    update() returns (new_state, flush_frames): flush_frames is the pre-buffer
    snapshot to write at IDLE→RECORDING; empty otherwise.
    """

    def __init__(self, fps: float = DEFAULT_FPS,
                 pre_sec: float = PRE_BUFFER_SEC,
                 post_sec: float = POST_RECORD_SEC):
        self.fps       = fps
        self._pre_sec  = pre_sec
        self._post_sec = post_sec
        self.state     = "IDLE"
        self._tail_rem = 0
        self._pre_buf  = deque(maxlen=max(1, int(pre_sec * fps)))

    def set_fps(self, fps: float):
        self.fps = fps
        new_max = max(1, int(self._pre_sec * fps))
        old = list(self._pre_buf)
        self._pre_buf = deque(old[-new_max:], maxlen=new_max)

    def set_pre_sec(self, sec: float):
        self._pre_sec = sec
        new_max = max(1, int(sec * self.fps))
        old = list(self._pre_buf)
        self._pre_buf = deque(old[-new_max:], maxlen=new_max)

    def set_post_sec(self, sec: float):
        self._post_sec = sec

    def push_frame(self, frame: np.ndarray):
        """Always called — keeps the rolling pre-buffer fresh."""
        self._pre_buf.append(frame.copy())

    def update(self, jewel_present: bool) -> tuple[str, list]:
        """
        Returns (new_state, flush_frames).
        flush_frames is non-empty only on IDLE→RECORDING (pre-buffer contents).
        """
        flush = []

        if self.state == "IDLE":
            if jewel_present:
                flush          = list(self._pre_buf)
                self.state     = "RECORDING"
                self._tail_rem = 0

        elif self.state == "RECORDING":
            if not jewel_present:
                self.state     = "TAIL"
                self._tail_rem = max(1, int(self._post_sec * self.fps))

        elif self.state == "TAIL":
            if jewel_present:
                self.state     = "RECORDING"
                self._tail_rem = 0
            else:
                self._tail_rem -= 1
                if self._tail_rem <= 0:
                    self.state = "IDLE"

        return self.state, flush

    def reset(self):
        self.state     = "IDLE"
        self._tail_rem = 0
        self._pre_buf.clear()


# ── Video capture thread ──────────────────────────────────────────────────────

class VideoThread(QThread):
    frame_ready       = pyqtSignal(np.ndarray)
    rec_state_ready   = pyqtSignal(str)   # "IDLE" / "RECORDING" / "TAIL"
    recording_started = pyqtSignal(str)   # clip path
    recording_stopped = pyqtSignal(str)   # clip path
    alert_fired       = pyqtSignal(str)
    seal_state_changed = pyqtSignal(str)  # TRACKING/STRIP_MODE/STRIP_DETECTED/SEAL_WAIT/SEALED

    def __init__(self, bag_model: SegmentationModel,
                 strip_model: SegmentationModel,
                 save_folder: str,
                 camera_id: int = 0,
                 fps: float = DEFAULT_FPS,
                 pre_sec: float = PRE_BUFFER_SEC,
                 post_sec: float = POST_RECORD_SEC,
                 rect_threshold: float = DEFAULT_RECT_THRESHOLD,
                 seal_wait_sec: float = DEFAULT_SEAL_WAIT_SEC):
        super().__init__()
        self.bag_model      = bag_model
        self.strip_model    = strip_model
        self.save_folder    = save_folder
        self.camera_id      = camera_id
        self._running       = False
        self.roi            = None
        self.target_fps     = float(fps)
        self.pre_sec        = pre_sec
        self.post_sec       = post_sec
        self.rect_threshold = float(rect_threshold)
        self.seal_wait_sec  = float(seal_wait_sec)
        self._clean_frame   = None          # latest RAW camera frame (no overlays)
        self._clean_lock    = threading.Lock()
        self._rec_start_requested = False    # set True by the Count Jewels button

    def get_clean_frame(self):
        """Thread-safe snapshot of the latest raw camera frame (no overlays drawn).
        Used by the jewel-count feature so segmentation never sees the burned-in
        masks/ROI box/text that the live preview frame carries."""
        with self._clean_lock:
            return None if self._clean_frame is None else self._clean_frame.copy()

    def request_recording_start(self):
        """Begin a recording on the next loop iteration (Count Jewels trigger).
        The pre-buffer is flushed on open, so the clip starts pre_sec seconds
        before this call."""
        self._rec_start_requested = True

    @staticmethod
    def _select_writer(path_base: str, fps: float, size: tuple):
        """Open a VideoWriter with the best-compressing codec that works on this
        machine (H.264 first, MPEG-4 Part 2 fallback).  Returns (writer, path)
        or (None, path) if none could be opened."""
        path = path_base + ".mp4"
        for code in REC_CODECS:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
            if writer.isOpened():
                print(f"[Recording] codec={code} → {path}")
                return writer, path
            writer.release()
        return None, path

    def set_roi(self, roi):
        self.roi = roi

    def set_fps(self, fps: float):
        self.target_fps = max(1.0, float(fps))

    def set_rect_threshold(self, val: float):
        self.rect_threshold = float(val)

    def set_seal_wait_sec(self, val: float):
        self.seal_wait_sec = float(val)

    def set_save_folder(self, folder: str):
        """Switch the destination folder for subsequently-recorded clips."""
        self.save_folder = folder
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"[VideoThread] cannot create folder {folder}: {e}")

    def run(self):
        cap = cv2.VideoCapture(self.camera_id, CAMERA_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FPS,          self.target_fps)

        if not cap.isOpened():
            self.alert_fired.emit("Cannot open camera.")
            return

        bag_tracker   = CentroidTracker()

        bag_worker   = SegWorker(self.bag_model)
        strip_worker = SegWorker(self.strip_model)
        bag_worker.start()
        strip_worker.start()

        # ── Seal detection state machine ──────────────────────────────────────
        # TRACKING      → normal bag+jewel inference
        # STRIP_MODE    → bag went rectangular; strip.pt active; bag/jewel idle
        # STRIP_DETECTED→ strip.pt found the seal; say "remove the seal"
        # SEAL_WAIT     → seal gone; counting down before announcing closed
        # SEALED        → announced "seal removed bag closed"; terminal state
        _seal_state    = "TRACKING"
        _strip_present = False          # last known positive strip detection
        _strip_mask: np.ndarray | None = None
        _strip_miss    = 0              # consecutive no-detect inference results
        _spoke_remove  = False
        _spoke_closed  = False
        rect_score     = 0.0

        # Periodic 1-second checkpoint driving STRIP_DETECTED logic
        _check_int        = max(1, int(1.0 * self.target_fps))
        _check_rem        = _check_int
        _bag_present_last = None  # latest bag detection result (True / False / None)
        _bag_miss_count   = 0     # consecutive 1-s checks with bag absent
        _seal_gone_count  = 0     # consecutive 1-s checks with bag present + seal absent

        # Single continuous recording: (bag first seen − pre_sec) → (SEALED + post_sec)
        _bag_recording_active = False
        bag_writer    = None
        bag_clip_path = None
        rec_stride    = 1
        rec_fps       = float(REC_FPS)
        rec_w = rec_h = 0
        rec_widx      = 0
        _tail_rem     = 0      # frames left in the post-record tail (0 = not tailing)
        pre_buf       = None   # rolling pre-buffer of compressed frames (set on 1st frame)

        os.makedirs(self.save_folder, exist_ok=True)

        self._running = True
        frame_count   = 0

        while self._running:
            loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                self.alert_fired.emit("Lost camera feed.")
                break

            # Stash the clean frame (before any overlays) for jewel counting.
            with self._clean_lock:
                self._clean_frame = frame.copy()

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % INFER_INTERVAL == 0:
                roi = self.roi
                if roi is not None:
                    rx1, ry1, rx2, ry2 = roi
                    submit = frame[ry1:ry2, rx1:rx2].copy()
                else:
                    submit = frame.copy()
                if _seal_state == "TRACKING":
                    bag_worker.submit(submit)
                else:
                    strip_worker.submit(submit)

            roi = self.roi

            if _seal_state == "TRACKING":
                bag_raw   = self._offset_result(bag_worker.get_result(),   roi, frame.shape[:2])

                bag_pos,   bag_mask   = bag_tracker.update(gray, bag_raw)
                # Jewel detection is handled by the SigLIP count button, not a model.

                # Measure bag rectangularity and trigger strip mode when threshold met
                if bag_mask is not None:
                    rect_score = measure_rectangularity(bag_mask)
                    if rect_score >= self.rect_threshold:
                        _seal_state = "STRIP_MODE"
                        self.seal_state_changed.emit(_seal_state)
                else:
                    rect_score = 0.0

            else:
                # Bag worker is idle; only strip worker runs
                bag_pos = bag_mask = None

                strip_result = self._offset_result(
                    strip_worker.get_result(), roi, frame.shape[:2])
                if strip_result is not None:
                    s_centroid, s_mask = strip_result
                    if s_centroid is not None:
                        _strip_present = True
                        _strip_mask    = s_mask
                        _strip_miss    = 0
                    else:
                        _strip_present = False
                        _strip_miss   += 1

                # ── State transitions ─────────────────────────────────────
                if _seal_state == "STRIP_MODE":
                    if _strip_present:
                        _seal_state = "STRIP_DETECTED"
                        self.seal_state_changed.emit(_seal_state)
                        if not _spoke_remove:
                            _speak("Remove the seal")
                            _spoke_remove = True
                        # Pre-submit bag check so result is ready by first 2-s checkpoint
                        _roi = self.roi
                        _chk = (frame[_roi[1]:_roi[3], _roi[0]:_roi[2]].copy()
                                if _roi else frame.copy())
                        bag_worker.submit(_chk)
                        _check_rem = _check_int  # reset countdown from now

                elif _seal_state == "STRIP_DETECTED":
                    # Collect latest bag check result (async; submitted at last checkpoint)
                    _bag_chk = bag_worker.get_result()
                    if _bag_chk is not None:
                        _, _bag_chk_mask = _bag_chk
                        _bag_present_last = (_bag_chk_mask is not None)

                    # ── 2-second checkpoint ────────────────────────────────────
                    _check_rem -= 1
                    if _check_rem <= 0:
                        _check_rem = _check_int

                        if _bag_present_last is not None:
                            if not _bag_present_last:
                                # Bag absent this check
                                _bag_miss_count  += 1
                                _seal_gone_count  = 0
                                if _bag_miss_count >= 2:
                                    _speak("Warning: bag not present")
                            else:
                                # Bag present — evaluate seal status
                                _bag_miss_count = 0
                                if not _strip_present:
                                    _seal_gone_count += 1
                                    if _seal_gone_count >= 2:
                                        # Bag present, seal gone 3 checks → sealed.
                                        # Keep recording for post_sec seconds (tail)
                                        # before the writer is closed.
                                        _seal_state = "SEALED"
                                        if bag_writer is not None and _tail_rem == 0:
                                            _tail_rem = max(1, int(self.post_sec * self.target_fps))
                                        self.seal_state_changed.emit(_seal_state)
                                        if not _spoke_closed:
                                            _speak("Bag sealed OK")
                                            _spoke_closed = True
                                else:
                                    # Seal still present — reset seal counter
                                    _seal_gone_count = 0

                        # Submit fresh bag check for the next checkpoint
                        if _seal_state == "STRIP_DETECTED":
                            _roi = self.roi
                            _chk = (frame[_roi[1]:_roi[3], _roi[0]:_roi[2]].copy()
                                    if _roi else frame.copy())
                            bag_worker.submit(_chk)
                # SEALED is terminal — no further transitions

            # ── Overlays ──────────────────────────────────────────────────────
            vis = frame.copy()

            if bag_mask is not None:
                layer = vis.copy()
                layer[bag_mask > 0] = (50, 200, 50)
                cv2.addWeighted(layer, 0.20, vis, 0.80, 0, vis)
                cnt = get_largest_contour(bag_mask)
                if cnt is not None:
                    cv2.drawContours(vis, [cnt], -1, (50, 220, 50), 2)

            if bag_pos is not None:
                bx, by = int(bag_pos[0]), int(bag_pos[1])
                col = (0, 140, 255) if bag_tracker.snapped else (50, 220, 50)
                cv2.circle(vis, (bx, by), 10, col, 2)
                cv2.circle(vis, (bx, by),  3, col, -1)
                cv2.putText(vis, "BAG", (bx + 12, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            if self.roi is not None:
                rx1, ry1, rx2, ry2 = self.roi
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                cv2.putText(vis, "ROI", (rx1 + 4, ry1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # ── Rectangularity score (shown during TRACKING when bag visible) ─
            if _seal_state == "TRACKING" and bag_mask is not None:
                fh, fw = vis.shape[:2]
                cv2.putText(vis,
                            f"Rect: {rect_score:.2f} / {self.rect_threshold:.2f}",
                            (fw - 240, fh - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

            # ── Strip mask overlay ────────────────────────────────────────────
            if (_strip_present and _strip_mask is not None
                    and _seal_state in ("STRIP_MODE", "STRIP_DETECTED")):
                layer = vis.copy()
                layer[_strip_mask > 0] = (0, 100, 255)
                cv2.addWeighted(layer, 0.35, vis, 0.65, 0, vis)
                cnt = get_largest_contour(_strip_mask)
                if cnt is not None:
                    cv2.drawContours(vis, [cnt], -1, (0, 140, 255), 2)

            # ── Seal state overlay ────────────────────────────────────────────
            if _seal_state == "STRIP_MODE":
                cv2.putText(vis, "STRIP MODE — waiting for seal…",
                            (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            elif _seal_state == "STRIP_DETECTED":
                self._draw_seal_message(vis, "REMOVE THE SEAL", (0, 80, 255))
            elif _seal_state in ("SEAL_WAIT", "SEALED"):
                self._draw_seal_message(vis, "SEAL REMOVED OK, BAG CLOSED", (50, 220, 50))

            # ── Bag recording with pre-buffer + post-record tail ──────────────
            # Pre-buffer  : the clip starts pre_sec seconds BEFORE the bag is
            #               first detected (rolling buffer flushed on open).
            # Post-record : after the bag is SEALED, keep writing for post_sec
            #               seconds before the file is closed.
            if rec_w == 0:   # one-time setup once the frame size is known
                h_v, w_v   = vis.shape[:2]
                rec_stride = max(1, int(round(self.target_fps / REC_FPS)))
                rec_fps    = self.target_fps / rec_stride
                rec_w = (int(w_v * REC_SCALE) // 2) * 2
                rec_h = (int(h_v * REC_SCALE) // 2) * 2
                pre_buf = deque(maxlen=max(1, int(self.pre_sec * rec_fps)))

            # Compressed snapshot of this frame, on the save-rate (stride) cadence.
            write_this = (rec_widx % rec_stride == 0)
            rec_widx  += 1
            small = (cv2.resize(vis, (rec_w, rec_h), interpolation=cv2.INTER_AREA)
                     if write_this else None)

            # Open the writer when Count Jewels is clicked, flushing the
            # pre-buffer so the clip begins pre_sec seconds before the click.
            if self._rec_start_requested and bag_writer is None and not _bag_recording_active:
                self._rec_start_requested = False
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path_base = os.path.join(self.save_folder, f"bag_{ts}")
                bag_writer, bag_clip_path = self._select_writer(
                    path_base, rec_fps, (rec_w, rec_h))
                if bag_writer is None:
                    self.alert_fired.emit("Could not open video writer (no codec).")
                else:
                    _bag_recording_active = True
                    _tail_rem = 0
                    for buffered in pre_buf:
                        bag_writer.write(buffered)
                    pre_buf.clear()
                    self.recording_started.emit(bag_clip_path)

            # Route the compressed frame: into the writer while recording,
            # otherwise into the rolling pre-buffer.
            if small is not None:
                if bag_writer is not None:
                    bag_writer.write(small)
                else:
                    pre_buf.append(small)

            # Post-record tail: once SEALED, keep writing for post_sec seconds.
            if bag_writer is not None and _seal_state == "SEALED":
                _tail_rem -= 1
                if _tail_rem <= 0:
                    bag_writer.release()
                    _spawn_av1_transcode(bag_clip_path)   # → AV1 in background
                    bag_writer = None
                    _bag_recording_active = False
                    self.recording_stopped.emit(bag_clip_path or "")
                    bag_clip_path = None

            # REC indicator drawn on live view only (after writer.write so it
            # does not burn into the saved clip)
            if bag_writer is not None:
                cv2.rectangle(vis, (0, 0), (90, 40), (30, 30, 180), -1)
                cv2.circle(vis, (18, 20), 8, (0, 0, 255), -1)
                cv2.putText(vis, "REC", (32, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            self.frame_ready.emit(vis)
            self.rec_state_ready.emit("RECORDING" if bag_writer is not None else "IDLE")

            target_interval = 1.0 / max(self.target_fps, 1.0)
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        if bag_writer is not None:
            bag_writer.release()
            _spawn_av1_transcode(bag_clip_path)   # → AV1 in background
            self.recording_stopped.emit(bag_clip_path or "")

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

    @staticmethod
    def _clip_jewel_to_outside_bag(jewel_raw, bag_mask):
        if jewel_raw is None:
            return None
        centroid, mask = jewel_raw
        if mask is None or bag_mask is None:
            return jewel_raw
        if mask.shape != bag_mask.shape:
            bag_resized = cv2.resize(bag_mask, (mask.shape[1], mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            bag_resized = bag_mask
        filtered = mask.copy()
        filtered[bag_resized > 0] = 0
        if filtered.sum() == 0:
            return (None, None)
        return (get_centroid(filtered), filtered)

    @staticmethod
    def _offset_result(result, roi, full_shape):
        if result is None or roi is None:
            return result
        centroid, mask = result
        if centroid is None and mask is None:
            return result
        rx1, ry1, rx2, ry2 = roi
        ox, oy = rx1, ry1
        fh, fw = full_shape
        if centroid is not None:
            centroid = (centroid[0] + ox, centroid[1] + oy)
        if mask is not None:
            full_mask = np.zeros((fh, fw), dtype=np.uint8)
            mh, mw = mask.shape[:2]
            dy = min(mh, fh - oy)
            dx = min(mw, fw - ox)
            full_mask[oy:oy + dy, ox:ox + dx] = mask[:dy, :dx]
            mask = full_mask
        return centroid, mask

    @staticmethod
    def _draw_seal_message(vis: np.ndarray, text: str, color: tuple) -> None:
        """Centred semi-transparent banner for seal state messages."""
        h, w = vis.shape[:2]
        banner_h = max(80, h // 8)
        cy = h // 2
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, cy - banner_h // 2),
                      (w, cy + banner_h // 2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
        scale = w / 900.0
        thick = max(2, int(scale * 2.5))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(vis, text, ((w - tw) // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    jewel_count_finished = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jewel Recorder  •  v3.0")
        self.resize(1400, 850)

        self.bag_model    = None
        self.video_thread = None
        self._clip_count  = 0

        self._last_frame      = None
        self._roi_drawing     = False
        self._roi_origin      = None
        self._rubber_band     = None
        self._current_roi     = None
        self._last_frame_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
        self._roi_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "roi_config.json")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._bag_model_path   = os.path.join(script_dir, BAG_MODEL_FILE)
        self._strip_model_path = os.path.join(script_dir, STRIP_MODEL_FILE)
        self._placing_folder = os.path.join(script_dir, "placing")
        self._removal_folder = os.path.join(script_dir, "removal")
        self._active_folder  = self._placing_folder

        self._apply_dark_theme()
        self._build_ui()
        self._load_roi()
        self.jewel_count_finished.connect(self._on_jewel_count_finished)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)
        root_layout.addWidget(self._make_video_panel(),   stretch=3)
        root_layout.addWidget(self._make_control_panel(), stretch=0)

    def _make_video_panel(self):
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        logo_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "Embsys Intelligence logo transparent.png")
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        pix = QPixmap(logo_path)
        if not pix.isNull():
            logo_label.setPixmap(pix.scaledToHeight(48, Qt.SmoothTransformation))
        layout.addWidget(logo_label)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "background:#111; border-radius:10px; color:#555; font-size:16px;")
        self.video_label.setText("◉  Camera feed will appear here")
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)
        layout.addWidget(self.video_label)

        roi_row = QHBoxLayout()
        self.draw_roi_btn = QPushButton("✎  Draw ROI")
        self.draw_roi_btn.setCheckable(True)
        self.draw_roi_btn.setFixedHeight(28)
        self.draw_roi_btn.setStyleSheet(
            "QPushButton{background:#555;color:#f0f0f0;border-radius:5px;padding:4px 10px;font-size:12px;}"
            "QPushButton:checked{background:#d4a017;color:#1e1e1e;font-weight:bold;}")
        self.clear_roi_btn = QPushButton("✕  Clear ROI")
        self.clear_roi_btn.setFixedHeight(28)
        self.clear_roi_btn.setStyleSheet(
            "background:#555;color:#f0f0f0;border-radius:5px;padding:4px 10px;font-size:12px;")
        self.roi_info_label = QLabel("No ROI set")
        self.roi_info_label.setStyleSheet("color:#888;font-size:11px;")
        self.draw_roi_btn.toggled.connect(self._toggle_roi_draw)
        self.clear_roi_btn.clicked.connect(self._clear_roi)
        roi_row.addWidget(self.draw_roi_btn)
        roi_row.addWidget(self.clear_roi_btn)
        roi_row.addSpacing(10)
        roi_row.addWidget(self.roi_info_label)
        roi_row.addStretch()
        roi_widget = QWidget()
        roi_widget.setLayout(roi_row)
        layout.addWidget(roi_widget)
        return panel

    def _make_control_panel(self):
        panel  = QWidget()
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Jewel Recorder")
        title.setFont(QFont("Arial", 17, QFont.Bold))
        title.setStyleSheet("color:#f0c040; margin-bottom:4px;")
        layout.addWidget(title)
        layout.addWidget(self._make_folder_group())
        layout.addWidget(self._make_config_group())
        layout.addWidget(self._make_buttons())
        layout.addWidget(self._make_rec_status_group())
        layout.addWidget(self._make_clip_log_group())
        layout.addStretch()
        return panel

    def _make_folder_group(self):
        grp    = QGroupBox("Save Folder")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        radio_row = QHBoxLayout()
        self.placing_btn = QPushButton("Placing")
        self.placing_btn.setCheckable(True)
        self.placing_btn.setChecked(True)
        self.removal_btn = QPushButton("Removal")
        self.removal_btn.setCheckable(True)
        btn_style = (
            "QPushButton{{background:#555;color:#f0f0f0;"
            "border-radius:5px;padding:5px 14px;font-size:12px;}}"
            "QPushButton:checked{{background:{};color:white;font-weight:bold;}}")
        self.placing_btn.setStyleSheet(btn_style.format("#27ae60"))
        self.removal_btn.setStyleSheet(btn_style.format("#c0392b"))
        self.placing_btn.clicked.connect(self._select_placing)
        self.removal_btn.clicked.connect(self._select_removal)
        radio_row.addWidget(self.placing_btn)
        radio_row.addWidget(self.removal_btn)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        path_row = QHBoxLayout()
        self.folder_path_label = QLabel(self._placing_folder)
        self.folder_path_label.setStyleSheet(
            "color:#aaa;font-size:10px;background:#2a2a2a;"
            "border-radius:4px;padding:3px 6px;")
        self.folder_path_label.setWordWrap(True)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse_folder)
        path_row.addWidget(self.folder_path_label, stretch=1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)
        return grp

    def _make_config_group(self):
        grp  = QGroupBox("Session Configuration")
        form = QFormLayout(grp)
        form.setSpacing(8)

        self.camera_combo = QComboBox()
        for i in range(5):
            self.camera_combo.addItem(f"Camera {i}", i)
        form.addRow("Camera:", self.camera_combo)

        self.pre_spin = QDoubleSpinBox()
        self.pre_spin.setRange(0.0, 10.0)
        self.pre_spin.setSingleStep(0.5)
        self.pre_spin.setDecimals(1)
        self.pre_spin.setValue(PRE_BUFFER_SEC)
        self.pre_spin.setSuffix(" s")
        self.pre_spin.setFixedWidth(75)
        self.pre_spin.setToolTip("Seconds of footage prepended before the jewel appears")
        form.addRow("Pre-buffer:", self.pre_spin)

        self.post_spin = QDoubleSpinBox()
        self.post_spin.setRange(0.0, 10.0)
        self.post_spin.setSingleStep(0.1)
        self.post_spin.setDecimals(1)
        self.post_spin.setValue(POST_RECORD_SEC)
        self.post_spin.setSuffix(" s")
        self.post_spin.setFixedWidth(75)
        self.post_spin.setToolTip("Seconds to keep recording after the jewel disappears")
        form.addRow("Post-record:", self.post_spin)

        self.rect_thresh_spin = QDoubleSpinBox()
        self.rect_thresh_spin.setRange(0.50, 0.99)
        self.rect_thresh_spin.setSingleStep(0.01)
        self.rect_thresh_spin.setDecimals(2)
        self.rect_thresh_spin.setValue(DEFAULT_RECT_THRESHOLD)
        self.rect_thresh_spin.setFixedWidth(75)
        self.rect_thresh_spin.setToolTip(
            "Bag shape threshold: contour area ÷ bounding-rect area.\n"
            "Higher value = more strictly rectangular before switching to strip mode.")
        form.addRow("Rect threshold:", self.rect_thresh_spin)

        self.seal_wait_spin = QDoubleSpinBox()
        self.seal_wait_spin.setRange(0.5, 15.0)
        self.seal_wait_spin.setSingleStep(0.5)
        self.seal_wait_spin.setDecimals(1)
        self.seal_wait_spin.setValue(DEFAULT_SEAL_WAIT_SEC)
        self.seal_wait_spin.setSuffix(" s")
        self.seal_wait_spin.setFixedWidth(75)
        self.seal_wait_spin.setToolTip(
            "Seconds to wait after the seal disappears before\n"
            "announcing 'Seal removed — bag closed'.")
        form.addRow("Seal wait:", self.seal_wait_spin)
        return grp

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Save Folder", self._active_folder)
        if not folder:
            return
        if self.placing_btn.isChecked():
            self._placing_folder = folder
        else:
            self._removal_folder = folder
        self._active_folder = folder
        self.folder_path_label.setText(folder)
        if self.video_thread:
            self.video_thread.set_save_folder(folder)

    def _make_buttons(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.stop_btn  = QPushButton("■  Stop")
        self.start_btn.setStyleSheet(
            "background:#27ae60;color:white;font-weight:bold;"
            "padding:8px;border-radius:6px;font-size:13px;")
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;"
            "padding:8px;border-radius:6px;font-size:13px;")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        layout.addLayout(row)

        # Restart prompt — hidden until the bag is sealed. SEALED is a terminal
        # state in the video thread, so a fresh session is needed for the next
        # bag; this button invites the operator to start one.
        self.restart_btn = QPushButton("⟳  Bag sealed — click to restart")
        self.restart_btn.setStyleSheet(
            "background:#f0c040;color:#1e1e1e;font-weight:bold;"
            "padding:10px;border-radius:6px;font-size:13px;")
        self.restart_btn.clicked.connect(self.restart_tracking)
        self.restart_btn.hide()
        layout.addWidget(self.restart_btn)

        self.count_jewels_btn = QPushButton("Count Jewels")
        self.count_jewels_btn.setStyleSheet(
            "background:#1a6ea8;color:white;font-weight:bold;"
            "padding:8px;border-radius:6px;font-size:13px;")
        self.count_jewels_btn.clicked.connect(self._count_jewels)
        layout.addWidget(self.count_jewels_btn)

        self.jewel_count_label = QLabel("Jewels: —")
        self.jewel_count_label.setAlignment(Qt.AlignCenter)
        self.jewel_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.jewel_count_label.setStyleSheet(
            "color:#4fc3f7;background:#1a2a3a;border-radius:6px;padding:6px;")
        layout.addWidget(self.jewel_count_label)
        return w

    def _count_jewels(self):
        # Count on the CLEAN camera frame (no overlays/ROI box/text burned in),
        # so threshold segmentation sees only the real scene — same as feeding a
        # photo to simple_bg_remover.
        if self.video_thread is None:
            self.jewel_count_label.setText("Jewels: start feed first")
            return
        frame = self.video_thread.get_clean_frame()
        if frame is None:
            self.jewel_count_label.setText("Jewels: no frame")
            return

        # Crop to ROI if one is set — count only what's in the region of interest.
        roi = self._current_roi
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            h, w = frame.shape[:2]
            rx1, rx2 = max(0, min(rx1, w)), max(0, min(rx2, w))
            ry1, ry2 = max(0, min(ry1, h)), max(0, min(ry2, h))
            crop = frame[ry1:ry2, rx1:rx2].copy()
        else:
            crop = frame.copy()

        if crop.size == 0:
            self.jewel_count_label.setText("Jewels: invalid ROI")
            return

        # Begin recording at this moment (pre-buffer flushed so the clip starts
        # pre_sec seconds before this click); it stops after the bag is sealed.
        self.video_thread.request_recording_start()

        # Count this snapshot in the background while the live feed keeps running.
        self.count_jewels_btn.setEnabled(False)
        self.jewel_count_label.setText("Jewels: counting…")

        def _run():
            items = []
            error = None
            classifier_error = None
            try:
                # Qt-free copy of simple_bg_remover's threshold/contour +
                # SigLIP jewelry-prompt pipeline.
                result = sjc.count_jewels_bgr(crop)
                items = result.get("items", [])
                classifier_error = result.get("classifier_error")
                if classifier_error:
                    print(f"[JewelCount] classifier: {classifier_error}")
            except Exception as exc:
                error = str(exc)
                print(f"[JewelCount] error: {exc}")

            self.jewel_count_finished.emit(
                items, {"error": error, "classifier_error": classifier_error})

        threading.Thread(target=_run, daemon=True).start()

    def _on_jewel_count_finished(self, items, status):
        error = status.get("error")
        ts = datetime.now().strftime("%H:%M:%S")

        if error:
            self.jewel_count_label.setText("Jewels: error")
            item_w = QListWidgetItem(f"[{ts}]  Jewel count error: {error}")
            item_w.setForeground(QColor("#e74c3c"))
            self.clip_list.insertItem(0, item_w)
        else:
            count = len(items)
            self.jewel_count_label.setText(f"Jewels: {count}")
            for i, item in enumerate(reversed(items), 1):
                label = item.get("label", "?")
                confidence = item.get("confidence", 0.0)
                source = item.get("source", "")
                entry = QListWidgetItem(
                    f"[{ts}]  #{i} {label}  ({confidence:.0%})  [{source}]")
                entry.setForeground(QColor("#4fc3f7"))
                self.clip_list.insertItem(0, entry)

            summary = QListWidgetItem(f"[{ts}]  -- Jewel count: {count} --")
            summary.setForeground(QColor("#f0c040"))
            self.clip_list.insertItem(0, summary)

        self.count_jewels_btn.setEnabled(True)

    def _make_rec_status_group(self):
        grp    = QGroupBox("Recording Status")
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)
        self.rec_badge = QLabel("IDLE")
        self.rec_badge.setAlignment(Qt.AlignCenter)
        self.rec_badge.setFont(QFont("Arial", 18, QFont.Bold))
        self.rec_badge.setStyleSheet(
            "color:#888;background:#2a2a2a;border-radius:8px;padding:6px;")
        layout.addWidget(self.rec_badge)
        self.clip_count_label = QLabel("Clips saved: 0")
        self.clip_count_label.setAlignment(Qt.AlignCenter)
        self.clip_count_label.setStyleSheet("color:#aaa;font-size:12px;")
        layout.addWidget(self.clip_count_label)

        self.seal_badge = QLabel("TRACKING")
        self.seal_badge.setAlignment(Qt.AlignCenter)
        self.seal_badge.setFont(QFont("Arial", 11, QFont.Bold))
        self.seal_badge.setStyleSheet(
            "color:#555;background:#1a1a1a;border-radius:6px;padding:4px;"
            "border:1px solid #333;")
        self.seal_badge.setToolTip("Seal detection state machine")
        layout.addWidget(self.seal_badge)
        return grp

    def _make_clip_log_group(self):
        grp    = QGroupBox("Saved Clips")
        layout = QVBoxLayout(grp)
        self.clip_list = QListWidget()
        self.clip_list.setFixedHeight(130)
        self.clip_list.setStyleSheet(
            "font-size:11px;background:#1a1a1a;border:none;")
        layout.addWidget(self.clip_list)
        clr = QPushButton("Clear log")
        clr.setStyleSheet(
            "background:#333;color:#aaa;border-radius:4px;padding:3px;font-size:11px;")
        clr.clicked.connect(self.clip_list.clear)
        layout.addWidget(clr)
        return grp

    # ── Tracking control ──────────────────────────────────────────────────────

    def start_tracking(self):
        bag_fp_filter   = FPFilter(FP_FILTER_MODEL_PATH_BAG,   FP_FILTER_CONF)
        strip_fp_filter = HSVFPFilter(HSV_FP_FILTER_STRIP_PATH, FP_FILTER_CONF)
        self.bag_model   = SegmentationModel(self._bag_model_path,   "bag",
                                             CONF_THRESHOLD, "yolo",
                                             fp_filter=bag_fp_filter,
                                             iou_threshold=BAG_IOU_THRESHOLD)
        self.strip_model = SegmentationModel(self._strip_model_path, "strip",
                                             CONF_THRESHOLD, "yolo",
                                             fp_filter=strip_fp_filter)

        cam_id = self.camera_combo.currentData()
        self.video_thread = VideoThread(
            self.bag_model, self.strip_model,
            save_folder=self._active_folder,
            camera_id=cam_id,
            fps=DEFAULT_FPS,
            pre_sec=float(self.pre_spin.value()),
            post_sec=float(self.post_spin.value()),
            rect_threshold=float(self.rect_thresh_spin.value()),
            seal_wait_sec=float(self.seal_wait_spin.value()))
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.rec_state_ready.connect(self._on_rec_state)
        self.video_thread.recording_started.connect(self._on_recording_started)
        self.video_thread.recording_stopped.connect(self._on_recording_stopped)
        self.video_thread.alert_fired.connect(self._on_alert)
        self.video_thread.seal_state_changed.connect(self._on_seal_state)
        if self._current_roi:
            self.video_thread.set_roi(self._current_roi)
        self.video_thread.start()

        self._clip_count = 0
        self.clip_count_label.setText("Clips saved: 0")
        self.seal_badge.setText("TRACKING")
        self.seal_badge.setStyleSheet(
            "color:#555;background:#1a1a1a;border-radius:6px;padding:4px;"
            "border:1px solid #333;")
        self.restart_btn.hide()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_tracking(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.restart_btn.hide()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.rec_badge.setText("IDLE")
        self.rec_badge.setStyleSheet(
            "color:#888;background:#2a2a2a;border-radius:8px;padding:6px;")
        self.seal_badge.setText("TRACKING")
        self.seal_badge.setStyleSheet(
            "color:#555;background:#1a1a1a;border-radius:6px;padding:4px;"
            "border:1px solid #333;")

    def restart_tracking(self):
        """Tear down the sealed session and start a fresh one for the next bag."""
        self.stop_tracking()
        self.start_tracking()

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        self._last_frame = frame.copy()
        self._last_frame_size = (frame.shape[1], frame.shape[0])
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(),
                       Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_rec_state(self, state: str):
        colors = {"IDLE": "#888888", "RECORDING": "#e74c3c", "TAIL": "#f39c12"}
        col = colors.get(state, "#888888")
        self.rec_badge.setText(state)
        self.rec_badge.setStyleSheet(
            f"color:{col};background:#2a2a2a;"
            f"border-radius:8px;padding:6px;border:2px solid {col};")

    def _on_recording_started(self, path: str):
        pass  # badge already updated via _on_rec_state

    def _on_recording_stopped(self, path: str):
        self._clip_count += 1
        self.clip_count_label.setText(f"Clips saved: {self._clip_count}")
        ts   = datetime.now().strftime("%H:%M:%S")
        name = os.path.basename(path)
        item = QListWidgetItem(f"[{ts}]  {name}")
        item.setForeground(QColor("#2ecc71"))
        self.clip_list.insertItem(0, item)

    def _on_alert(self, msg: str):
        ts   = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{ts}]  {msg}")
        item.setForeground(QColor("#e74c3c"))
        self.clip_list.insertItem(0, item)

    def _on_seal_state(self, state: str):
        styles = {
            "TRACKING":       ("TRACKING",        "#555",    "#1a1a1a", "#333"),
            "STRIP_MODE":     ("STRIP MODE",       "#00c8ff", "#0a2030", "#00c8ff"),
            "STRIP_DETECTED": ("REMOVE THE SEAL",  "#ff4040", "#300000", "#ff4040"),
            "SEAL_WAIT":      ("SEAL WAIT…",       "#f0c040", "#2a2000", "#f0c040"),
            "SEALED":         ("BAG CLOSED",       "#32dc5a", "#002010", "#32dc5a"),
        }
        label, fg, bg, border = styles.get(state, ("?", "#888", "#1a1a1a", "#333"))
        self.seal_badge.setText(label)
        self.seal_badge.setStyleSheet(
            f"color:{fg};background:{bg};border-radius:6px;padding:4px;"
            f"border:1px solid {border};")
        # Once the bag is sealed (terminal state), invite a restart for the next bag.
        self.restart_btn.setVisible(state == "SEALED")

    # ── Folder selection ──────────────────────────────────────────────────────

    def _select_placing(self):
        self.placing_btn.setChecked(True)
        self.removal_btn.setChecked(False)
        self._active_folder = self._placing_folder
        self.folder_path_label.setText(self._active_folder)
        if self.video_thread:
            self.video_thread.set_save_folder(self._active_folder)

    def _select_removal(self):
        self.removal_btn.setChecked(True)
        self.placing_btn.setChecked(False)
        self._active_folder = self._removal_folder
        self.folder_path_label.setText(self._active_folder)
        if self.video_thread:
            self.video_thread.set_save_folder(self._active_folder)

    # ── ROI drawing ───────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.video_label and self._roi_drawing:
            t = event.type()
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._roi_origin = event.pos()
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(QRubberBand.Rectangle,
                                                    self.video_label)
                self._rubber_band.setGeometry(QRect(self._roi_origin, QSize()))
                self._rubber_band.show()
            elif t == QEvent.MouseMove and self._roi_origin is not None:
                self._rubber_band.setGeometry(
                    QRect(self._roi_origin, event.pos()).normalized())
            elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton \
                    and self._roi_origin is not None:
                rect = QRect(self._roi_origin, event.pos()).normalized()
                self._rubber_band.hide()
                self._roi_origin  = None
                self._roi_drawing = False
                self.draw_roi_btn.setChecked(False)
                self.video_label.setCursor(Qt.ArrowCursor)
                if rect.width() > 4 and rect.height() > 4:
                    x1, y1 = self._label_to_frame(rect.left(),  rect.top())
                    x2, y2 = self._label_to_frame(rect.right(), rect.bottom())
                    if x2 > x1 and y2 > y1:
                        self._current_roi = (x1, y1, x2, y2)
                        self.roi_info_label.setText(
                            f"ROI: ({x1},{y1})→({x2},{y2})")
                        if self.video_thread:
                            self.video_thread.set_roi(self._current_roi)
                        self._save_roi()
        return super().eventFilter(obj, event)

    def _label_to_frame(self, lx: int, ly: int) -> tuple[int, int]:
        fw, fh = self._last_frame_size
        lw, lh = self.video_label.width(), self.video_label.height()
        scale  = min(lw / fw, lh / fh)
        ox = (lw - int(fw * scale)) // 2
        oy = (lh - int(fh * scale)) // 2
        fx = int(np.clip((lx - ox) / scale, 0, fw - 1))
        fy = int(np.clip((ly - oy) / scale, 0, fh - 1))
        return fx, fy

    def _toggle_roi_draw(self, checked: bool):
        self._roi_drawing = checked
        self.video_label.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)
        if not checked and self._rubber_band:
            self._rubber_band.hide()
            self._roi_origin = None

    def _clear_roi(self):
        self._current_roi = None
        self.roi_info_label.setText("No ROI set")
        if self.video_thread:
            self.video_thread.set_roi(None)
        self._save_roi()

    def _save_roi(self):
        try:
            roi = list(self._current_roi) if self._current_roi else None
            with open(self._roi_config_path, "w") as f:
                json.dump({"roi": roi}, f)
        except Exception as e:
            print(f"[ROI] save error: {e}")

    def _load_roi(self):
        try:
            if not os.path.exists(self._roi_config_path):
                return
            with open(self._roi_config_path) as f:
                roi = json.load(f).get("roi")
            if roi and len(roi) == 4:
                x1, y1, x2, y2 = (int(v) for v in roi)
                self._current_roi = (x1, y1, x2, y2)
                self.roi_info_label.setText(
                    f"ROI: ({x1},{y1})→({x2},{y2})  [saved]")
        except Exception as e:
            print(f"[ROI] load error: {e}")

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget   { background:#1e1e1e; color:#f0f0f0; }
            QGroupBox {
                font-weight:bold; font-size:12px;
                border:1px solid #3a3a3a; border-radius:8px;
                margin-top:10px; padding-top:10px;
            }
            QGroupBox::title {
                subcontrol-origin:margin; left:10px; padding:0 4px; color:#aaa;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background:#2a2a2a; border:1px solid #555;
                border-radius:5px; padding:4px 6px; color:#f0f0f0;
            }
            QListWidget { background:#1a1a1a; border:none; color:#ddd; }
            QSlider::groove:horizontal { height:4px; background:#444; border-radius:2px; }
            QSlider::handle:horizontal {
                background:#f0c040; width:14px; height:14px;
                margin:-5px 0; border-radius:7px;
            }
            QSlider::sub-page:horizontal { background:#f0c040; border-radius:2px; }
            QPushButton { border-radius:6px; padding:6px 10px; }
            QPushButton:disabled { background:#333; color:#666; }
        """)

    def closeEvent(self, event):
        self.stop_tracking()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
