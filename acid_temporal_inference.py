from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASSES = ("gold_22k", "non_gold", "invalid")
INPUT_SIZE = 224
NORMALIZATION_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZATION_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
GOLD_CONFIDENCE = 0.85
GENERAL_CONFIDENCE = 0.65


def _letterbox(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 24, dtype=np.uint8)
    x = (INPUT_SIZE - resized_width) // 2
    y = (INPUT_SIZE - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def prepare_temporal_triplet(
    baseline_frames: list[np.ndarray],
    transition_frames: list[np.ndarray],
    observation_frames: list[np.ndarray],
) -> np.ndarray:
    frame_groups = (baseline_frames, transition_frames, observation_frames)
    if any(not frames for frames in frame_groups):
        raise ValueError("Baseline, transition, and observation frames are required")
    if any(frame is None or not frame.size for frames in frame_groups for frame in frames):
        raise ValueError("Baseline, transition, and observation frames must be valid")

    def median(frames: list[np.ndarray], tail_count: int | None = None) -> np.ndarray:
        selected = frames[-tail_count:] if tail_count else frames
        return np.median(np.stack(selected), axis=0).astype(np.uint8)

    return np.stack((
        _letterbox(median(baseline_frames)),
        _letterbox(median(transition_frames, tail_count=3)),
        _letterbox(median(observation_frames, tail_count=5)),
    ))


class TemporalAcidClassifier:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Temporal acid result model not found: {self.model_path}"
            )
        self.sha256 = self._sha256(self.model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        if (
            input_info.name != "temporal_triplet"
            or input_info.shape[-4:] != [3, 3, INPUT_SIZE, INPUT_SIZE]
        ):
            raise RuntimeError("Temporal acid result model input contract is incompatible")
        if output_info.name != "logits" or output_info.shape[-1] != len(CLASSES):
            raise RuntimeError("Temporal acid result model output contract is incompatible")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def predict(
        self,
        baseline_frames: list[np.ndarray],
        transition_frames: list[np.ndarray],
        observation_frames: list[np.ndarray],
    ) -> dict[str, Any]:
        triplet = prepare_temporal_triplet(
            baseline_frames,
            transition_frames,
            observation_frames,
        ).astype(np.float32) / 255.0
        triplet = (triplet - NORMALIZATION_MEAN) / NORMALIZATION_STD
        tensor = np.ascontiguousarray(triplet.transpose(0, 3, 1, 2)[None])
        started_at = time.perf_counter()
        logits = self.session.run(["logits"], {"temporal_triplet": tensor})[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        values = probabilities[0]
        index = int(np.argmax(values))
        confidence = float(values[index])
        prediction = CLASSES[index]
        if confidence < GENERAL_CONFIDENCE or (
            prediction == "gold_22k" and confidence < GOLD_CONFIDENCE
        ):
            prediction = "invalid"
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                label: float(values[position])
                for position, label in enumerate(CLASSES)
            },
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        }
