from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from app.audio_recorder import audio_levels
from app.utils import CLASSES, dataset_root, ensure_project_folders, sanitize_filename


METADATA_FIELDS = [
    "filepath",
    "class_name",
    "timestamp",
    "operator",
    "duration_sec",
    "sample_rate",
    "mic_device_name",
    "notes",
    "rms_energy",
    "peak_amplitude",
]


class DatasetManager:
    _metadata_lock = threading.Lock()

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or dataset_root()).resolve()
        self.metadata_path = self.root / "metadata.csv"

    def create_folders(self) -> None:
        if self.root == dataset_root().resolve():
            ensure_project_folders()
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            for class_name in CLASSES:
                (self.root / class_name).mkdir(parents=True, exist_ok=True)
        self._ensure_metadata_file()

    def _ensure_metadata_file(self) -> None:
        if self.metadata_path.exists():
            return
        with self._metadata_lock:
            if not self.metadata_path.exists():
                with self.metadata_path.open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=METADATA_FIELDS).writeheader()

    def counts(self) -> dict[str, int]:
        return {
            class_name: len(list((self.root / class_name).glob("*.wav")))
            if (self.root / class_name).exists()
            else 0
            for class_name in CLASSES
        }

    def next_sequence(self, class_name: str, timestamp_prefix: str) -> int:
        class_dir = self.root / class_name
        return len(list(class_dir.glob(f"{timestamp_prefix}_*.wav"))) + 1

    def save_sample(
        self,
        audio: np.ndarray,
        class_name: str,
        operator: str,
        duration_sec: float,
        sample_rate: int,
        mic_device_name: str,
        notes: str,
    ) -> tuple[Path, dict[str, str]]:
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")

        self.create_folders()
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
        filename_prefix = now.strftime("%Y%m%d_%H%M%S")
        operator_filename = sanitize_filename(operator)
        sequence = self.next_sequence(class_name, filename_prefix)
        filename = f"{filename_prefix}_{operator_filename}_{sequence:04d}.wav"
        path = self.root / class_name / filename

        source_audio = np.asarray(audio)
        if source_audio.ndim > 1:
            source_audio = np.mean(source_audio.astype(np.float32), axis=1)
        if np.issubdtype(source_audio.dtype, np.integer):
            integer_info = np.iinfo(source_audio.dtype)
            scale = float(max(abs(integer_info.min), integer_info.max))
            source_audio = source_audio.astype(np.float32) / scale
        else:
            source_audio = source_audio.astype(np.float32)

        expected_samples = int(round(duration_sec * sample_rate))
        if len(source_audio) != expected_samples:
            raise ValueError(
                f"Recording length mismatch: received {len(source_audio)} samples "
                f"but expected {expected_samples} for {duration_sec:.3f} seconds "
                f"at {sample_rate} Hz. The sample was not saved."
            )

        clean_audio = np.nan_to_num(
            source_audio,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        pcm = np.round(np.clip(clean_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(path, sample_rate, pcm)

        rms, peak = audio_levels(clean_audio)
        row = {
            "filepath": path.relative_to(self.root).as_posix(),
            "class_name": class_name,
            "timestamp": timestamp,
            "operator": operator.strip(),
            "duration_sec": f"{duration_sec:.3f}",
            "sample_rate": str(sample_rate),
            "mic_device_name": mic_device_name,
            "notes": notes.strip(),
            "rms_energy": f"{rms:.8f}",
            "peak_amplitude": f"{peak:.8f}",
        }
        self._append_metadata(row)
        return path, row

    def _append_metadata(self, row: dict[str, str]) -> None:
        self._ensure_metadata_file()
        with self._metadata_lock:
            with self.metadata_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
                writer.writerow(row)

    def metadata_for(self, path: Path) -> dict[str, str] | None:
        if not self.metadata_path.exists():
            return None
        try:
            relative = path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return None
        with self.metadata_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("filepath") == relative:
                    return row
        return None

    def delete_sample(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.suffix.lower() != ".wav" or self.root not in resolved.parents:
            raise ValueError("Only WAV files inside the dataset folder may be deleted.")
        if resolved.exists():
            resolved.unlink()
        self._remove_metadata_row(resolved)

    def _remove_metadata_row(self, path: Path) -> None:
        if not self.metadata_path.exists():
            return
        relative = path.relative_to(self.root).as_posix()
        with self._metadata_lock:
            with self.metadata_path.open("r", newline="", encoding="utf-8") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row.get("filepath") != relative
                ]
            temporary = self.metadata_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(self.metadata_path)

    def scan_files(self) -> list[tuple[Path, str]]:
        samples: list[tuple[Path, str]] = []
        for class_name in CLASSES:
            class_dir = self.root / class_name
            if class_dir.exists():
                samples.extend((path, class_name) for path in sorted(class_dir.glob("*.wav")))
        return samples
