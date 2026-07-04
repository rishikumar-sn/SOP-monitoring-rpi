from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.feature_extractor import FeatureConfig, extract_log_mel
from app.utils import models_root


class AudioPredictor:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = (model_dir or models_root()).resolve()
        self.model = None
        self.classes: list[str] = []
        self.config: dict[str, object] = {}

    def load(self) -> None:
        import tensorflow as tf

        model_path = self.model_dir / "gold_rub_cnn.keras"
        config_path = self.model_dir / "config.json"
        labels_path = self.model_dir / "labels.json"
        if not model_path.exists() or not config_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Train a model first. Required model files were not found.")
        self.model = tf.keras.models.load_model(model_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.classes = json.loads(labels_path.read_text(encoding="utf-8"))

    def predict(self, audio: np.ndarray) -> tuple[str, float, dict[str, float]]:
        if self.model is None:
            self.load()
        feature_config = FeatureConfig(
            sample_rate=int(self.config["sample_rate"]),
            duration=float(self.config["duration"]),
            n_mels=int(self.config["n_mels"]),
            n_fft=int(self.config["n_fft"]),
            hop_length=int(self.config["hop_length"]),
            win_length=int(self.config["win_length"]),
        )
        feature = extract_log_mel(
            audio,
            feature_config,
            mean=float(self.config["normalization_mean"]),
            std=float(self.config["normalization_std"]),
        )
        probabilities = self.model.predict(feature[np.newaxis, ...], verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        all_probabilities = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.classes)
        }
        print(f"Predicted class: {self.classes[best_index]}, probability: {probabilities[best_index]:.4f}")
        return self.classes[best_index], float(probabilities[best_index]), all_probabilities
