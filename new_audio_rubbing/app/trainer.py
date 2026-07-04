from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.dataset_manager import DatasetManager
from app.feature_extractor import (
    FeatureConfig,
    load_wav_mono,
    raw_log_mel_from_file,
)
from app.utils import CLASSES, DEFAULT_CONFIDENCE_THRESHOLD, models_root


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class TrainingConfig:
    features: FeatureConfig
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    minimum_epochs: int = 15
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


def build_model(input_shape: tuple[int, ...], num_classes: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.RandomTranslation(
        height_factor=0.0,
        width_factor=0.08,
        fill_mode="constant",
        fill_value=0.0,
        name="time_shift_augmentation",
    )(inputs)
    x = tf.keras.layers.GaussianNoise(0.03, name="feature_noise_augmentation")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def configure_tensorflow_acceleration(tf, log: LogCallback) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        version_parts = tf.__version__.split(".")
        major_minor = tuple(int(part) for part in version_parts[:2] if part.isdigit())
        if sys.platform == "win32" and major_minor >= (2, 11):
            log(
                "GPU acceleration: native Windows TensorFlow >= 2.11 does not "
                "support CUDA GPU training. Use WSL2 with tensorflow[and-cuda], "
                "or use the TensorFlow-DirectML plugin with a compatible Python/"
                "TensorFlow environment."
            )
        log("GPU acceleration: no TensorFlow-compatible GPU detected; training will use CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            log(f"GPU acceleration: could not change memory growth for {gpu.name}: {exc}")

    logical_gpus = tf.config.list_logical_devices("GPU")
    gpu_names = ", ".join(gpu.name for gpu in gpus)
    log(
        "GPU acceleration enabled: "
        f"{len(gpus)} physical GPU(s), {len(logical_gpus)} logical GPU(s): {gpu_names}"
    )


def audit_dataset(
    samples: list[tuple[Path, str]],
    config: FeatureConfig,
) -> dict[str, object]:
    expected_samples = int(round(config.sample_rate * config.duration))
    malformed_duration: list[str] = []
    impulse_artifacts: list[str] = []
    mostly_clipped: list[str] = []
    very_low_energy: list[str] = []

    for path, class_name in samples:
        audio = load_wav_mono(path, config.sample_rate)
        if len(audio) != expected_samples:
            malformed_duration.append(str(path))
        if audio.size:
            clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
            if clipped_fraction > 0.05:
                mostly_clipped.append(str(path))
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            if rms < 0.0001 and class_name != "NO_RUB_NOK":
                very_low_energy.append(str(path))
            # Repeated full-scale negative impulses are typical of a capture or
            # conversion fault, not contact-microphone rubbing.
            negative_full_scale = np.flatnonzero(audio <= -0.999)
            if len(negative_full_scale) >= 4:
                intervals = np.diff(negative_full_scale)
                if np.any((intervals >= 1000) & (intervals <= 2000)):
                    impulse_artifacts.append(str(path))

    return {
        "total": len(samples),
        "malformed_duration": malformed_duration,
        "impulse_artifacts": impulse_artifacts,
        "mostly_clipped": mostly_clipped,
        "very_low_energy": very_low_energy,
    }


def train_model(
    dataset_manager: DatasetManager,
    config: TrainingConfig,
    log: LogCallback,
    progress: ProgressCallback,
) -> dict[str, object]:
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    configure_tensorflow_acceleration(tf, log)

    samples = dataset_manager.scan_files()
    present_classes = {class_name for _, class_name in samples}
    missing = [name for name in CLASSES if name not in present_classes]
    if missing:
        raise ValueError("Missing samples for classes: " + ", ".join(missing))
    if len(samples) < len(CLASSES) * 2:
        raise ValueError("At least two samples per class are required for train/validation split.")

    audit = audit_dataset(samples, config.features)
    for key, description in (
        ("malformed_duration", "files do not match the configured duration"),
        ("impulse_artifacts", "files contain repeated full-scale impulse artifacts"),
        ("mostly_clipped", "files are more than 5% clipped"),
        ("very_low_energy", "non-silence files have extremely low energy"),
    ):
        count = len(audit[key])
        if count:
            log(f"DATA WARNING: {count} {description}.")

    log(f"Loading {len(samples)} WAV files...")
    raw_features: list[np.ndarray] = []
    labels: list[int] = []
    for index, (path, class_name) in enumerate(samples, start=1):
        raw_features.append(raw_log_mel_from_file(path, config.features))
        labels.append(CLASSES.index(class_name))
        progress(min(20, int(index / len(samples) * 20)))

    raw_x = np.stack(raw_features).astype(np.float32)
    y_indices = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y_indices, minlength=len(CLASSES))
    if np.any(counts < 2):
        sparse = [CLASSES[i] for i, count in enumerate(counts) if count < 2]
        raise ValueError("Need at least two samples in each class: " + ", ".join(sparse))

    test_size = max(config.validation_split, len(CLASSES) / len(samples))
    if test_size >= 1.0:
        raise ValueError("Dataset is too small for a stratified validation split.")
    raw_train, raw_val, idx_train, idx_val = train_test_split(
        raw_x,
        y_indices,
        test_size=test_size,
        random_state=42,
        stratify=y_indices,
    )
    dataset_mean = float(np.mean(raw_train))
    dataset_std = max(float(np.std(raw_train)), 1e-6)
    x_train = ((raw_train - dataset_mean) / dataset_std)[..., np.newaxis].astype(
        np.float32
    )
    x_val = ((raw_val - dataset_mean) / dataset_std)[..., np.newaxis].astype(
        np.float32
    )
    y_train = tf.keras.utils.to_categorical(
        idx_train,
        num_classes=len(CLASSES),
    )
    y_val = tf.keras.utils.to_categorical(
        idx_val,
        num_classes=len(CLASSES),
    )
    log(
        f"Training samples: {len(x_train)} | Validation samples: {len(x_val)}"
    )
    log(
        "Class distribution: "
        + " | ".join(
            f"{CLASSES[index]}={int(count)}"
            for index, count in enumerate(counts)
        )
    )
    log(
        f"Maximum epochs: {config.epochs} | Minimum epochs: "
        f"{config.minimum_epochs} | Early-stop patience: "
        f"{config.early_stopping_patience}"
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=idx_train,
    )
    class_weights = {index: float(weight) for index, weight in enumerate(weights)}
    model = build_model(tuple(x_train.shape[1:]), len(CLASSES))
    output_dir = models_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / "gold_rub_cnn.keras"

    class LiveCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            values = logs or {}
            learning_rate = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
            log(
                "Epoch {}/{} - loss: {:.4f} - accuracy: {:.4f} - "
                "val_loss: {:.4f} - val_accuracy: {:.4f} - lr: {:.2e}".format(
                    epoch + 1,
                    config.epochs,
                    values.get("loss", 0.0),
                    values.get("accuracy", 0.0),
                    values.get("val_loss", 0.0),
                    values.get("val_accuracy", 0.0),
                    learning_rate,
                )
            )
            progress(20 + int((epoch + 1) / config.epochs * 70))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=config.early_stopping_patience,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=max(0, config.minimum_epochs - 1),
    )
    callbacks = [
        LiveCallback(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_delta=0.001,
            min_lr=1e-6,
            mode="min",
        ),
        early_stopping,
        tf.keras.callbacks.ModelCheckpoint(
            keras_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )
    completed_epochs = len(history.history.get("loss", []))
    best_accuracy_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_accuracy = float(np.max(history.history["val_accuracy"]))
    best_loss_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_loss = float(np.min(history.history["val_loss"]))
    if completed_epochs < config.epochs:
        log(
            f"Early stopping ended training after epoch {completed_epochs}: "
            f"validation loss did not improve for "
            f"{config.early_stopping_patience} monitored epochs."
        )
    else:
        log(f"Completed all {config.epochs} requested epochs.")
    log(
        f"Best validation accuracy: {best_accuracy:.4f} at epoch "
        f"{best_accuracy_epoch}"
    )
    log(f"Lowest validation loss: {best_loss:.4f} at epoch {best_loss_epoch}")

    model = tf.keras.models.load_model(keras_path)
    probabilities = model.predict(x_val, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(idx_val, predictions, labels=np.arange(len(CLASSES)))
    report = classification_report(
        idx_val,
        predictions,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    ok_index = CLASSES.index("JEWEL_RUB_OK")
    actual_ok = idx_val == ok_index
    predicted_ok = predictions == ok_index
    true_ok = int(np.sum(actual_ok & predicted_ok))
    false_ok = int(np.sum(~actual_ok & predicted_ok))
    missed_ok = int(np.sum(actual_ok & ~predicted_ok))
    true_nok = int(np.sum(~actual_ok & ~predicted_ok))
    ok_recall = true_ok / max(1, true_ok + missed_ok)
    false_accept_rate = false_ok / max(1, false_ok + true_nok)
    log(
        f"Production validation: jewel OK recall={ok_recall:.4f} | "
        f"NOK false-accept rate={false_accept_rate:.4f}"
    )
    for class_name in CLASSES:
        values = report[class_name]
        log(
            f"{class_name}: precision={values['precision']:.4f} | "
            f"recall={values['recall']:.4f} | f1={values['f1-score']:.4f}"
        )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_path = output_dir / "gold_rub_cnn.tflite"
    tflite_path.write_bytes(converter.convert())
    (output_dir / "labels.json").write_text(
        json.dumps(CLASSES, indent=2),
        encoding="utf-8",
    )
    model_config = {
        "classes": CLASSES,
        **config.features.to_dict(),
        "confidence_threshold": config.confidence_threshold,
        "normalization_mean": dataset_mean,
        "normalization_std": dataset_std,
    }
    (output_dir / "config.json").write_text(
        json.dumps(model_config, indent=2),
        encoding="utf-8",
    )
    metrics = {
        "classification_report": report,
        "ok_recall": ok_recall,
        "nok_false_accept_rate": false_accept_rate,
        "dataset_audit": {
            key: len(value) if isinstance(value, list) else value
            for key, value in audit.items()
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    progress(100)
    log(f"Saved best Keras model to {keras_path}")
    log(f"Saved TensorFlow Lite model to {tflite_path}")

    return {
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "confusion_matrix": matrix.tolist(),
        "validation_labels": idx_val.tolist(),
        "validation_predictions": predictions.tolist(),
        "classification_report": report,
        "ok_recall": ok_recall,
        "nok_false_accept_rate": false_accept_rate,
        "dataset_audit": audit,
    }
