from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.feature_extractor import FeatureConfig, extract_log_mel


class RealtimePredictor:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.resolve()
        self.model = None
        self.inference_model = None
        self.classes: list[str] = []
        self.config: dict[str, object] = {}
        self._class_embedding_vectors: np.ndarray | None = None

    def load(self) -> None:
        import tensorflow as tf

        model_path = self.model_dir / "gold_rub_cnn.keras"
        config_path = self.model_dir / "config.json"
        labels_path = self.model_dir / "labels.json"

        errors = []
        if not model_path.exists():
            errors.append(f"Missing: {model_path}")
        if not config_path.exists():
            errors.append(f"Missing: {config_path}")
        if not labels_path.exists():
            errors.append(f"Missing: {labels_path}")
        if errors:
            raise FileNotFoundError("; ".join(errors))

        self.model = tf.keras.models.load_model(str(model_path))
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.classes = json.loads(labels_path.read_text(encoding="utf-8"))

        embedding_layer = self.model.get_layer("embedding")
        output_layer = self.model.layers[-1]
        output_weights = output_layer.get_weights()
        if not output_weights or output_weights[0].shape[1] != len(self.classes):
            raise ValueError("Model output layer is incompatible with labels.json")

        self.inference_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.model.output, embedding_layer.output],
        )
        # Each output-layer column is the learned class direction in embedding space.
        class_vectors = np.asarray(output_weights[0], dtype=np.float32).T
        vector_norms = np.linalg.norm(class_vectors, axis=1, keepdims=True)
        self._class_embedding_vectors = class_vectors / np.maximum(vector_norms, 1e-8)

        self._feature_config = FeatureConfig(
            sample_rate=int(self.config["sample_rate"]),
            duration=float(self.config["duration"]),
            n_mels=int(self.config["n_mels"]),
            n_fft=int(self.config["n_fft"]),
            hop_length=int(self.config["hop_length"]),
            win_length=int(self.config["win_length"]),
        )
        self._mean = float(self.config.get("normalization_mean", 0.0))
        self._std = float(self.config.get("normalization_std", 1.0))
        self._has_norm = "normalization_mean" in self.config

    @property
    def sample_rate(self) -> int:
        return self._feature_config.sample_rate

    @property
    def duration(self) -> float:
        return self._feature_config.duration

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.duration))

    def predict(
        self,
        audio: np.ndarray,
    ) -> tuple[str, float, dict[str, float], bool, dict[str, float]]:
        if self.model is None or self.inference_model is None:
            self.load()

        mean = self._mean if self._has_norm else None
        std = self._std if self._has_norm else None

        feature = extract_log_mel(audio, self._feature_config, mean=mean, std=std)
        input_batch = feature[np.newaxis, ...]
        probabilities_batch, embedding_batch = self.inference_model.predict(
            input_batch,
            verbose=0,
        )
        probabilities = probabilities_batch[0]
        embedding = np.asarray(embedding_batch[0], dtype=np.float32)
        best_index = int(np.argmax(probabilities))
        all_probabilities = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.classes)
        }

        embedding_norm = float(np.linalg.norm(embedding))
        if embedding_norm <= 1e-8 or self._class_embedding_vectors is None:
            similarities = np.full(len(self.classes), -1.0, dtype=np.float32)
        else:
            similarities = self._class_embedding_vectors @ (embedding / embedding_norm)
        embedding_similarities = {
            class_name: float(similarities[index])
            for index, class_name in enumerate(self.classes)
        }

        jewel_index = self.classes.index("JEWEL_RUB_OK")
        strongest_index = int(np.argmax(similarities))
        negative_similarity = max(
            float(similarities[index])
            for index in range(len(self.classes))
            if index != jewel_index
        )
        jewel_similarity = float(similarities[jewel_index])
        embedding_verified = (
            best_index == jewel_index
            and strongest_index == jewel_index
            and jewel_similarity >= negative_similarity + 0.05
        )

        return (
            self.classes[best_index],
            float(probabilities[best_index]),
            all_probabilities,
            embedding_verified,
            embedding_similarities,
        )
