from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 48000
    duration: float = 1.0
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_wav_mono(path: Path, target_sample_rate: int) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    original_dtype = audio.dtype
    if audio.ndim > 1:
        audio = np.mean(audio.astype(np.float32), axis=1)

    if np.issubdtype(original_dtype, np.integer):
        scale = float(max(abs(np.iinfo(original_dtype).min), np.iinfo(original_dtype).max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)

    audio = np.nan_to_num(audio)
    if sample_rate != target_sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
        )
    return audio.astype(np.float32)


def pad_or_trim(audio: np.ndarray, sample_count: int) -> np.ndarray:
    if len(audio) >= sample_count:
        return audio[:sample_count]
    return np.pad(audio, (0, sample_count - len(audio)), mode="constant")


def extract_log_mel(
    audio: np.ndarray,
    config: FeatureConfig,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    sample_count = int(round(config.sample_rate * config.duration))
    fixed = pad_or_trim(np.asarray(audio, dtype=np.float32), sample_count)
    mel = librosa.feature.melspectrogram(
        y=fixed,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)
    if mean is None or std is None:
        mean = float(np.mean(log_mel))
        std = float(np.std(log_mel))
    normalized = (log_mel - mean) / max(float(std), 1e-6)
    return normalized[..., np.newaxis].astype(np.float32)


def feature_from_file(
    path: Path,
    config: FeatureConfig,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    audio = load_wav_mono(path, config.sample_rate)
    return extract_log_mel(audio, config, mean=mean, std=std)


def raw_log_mel_from_file(path: Path, config: FeatureConfig) -> np.ndarray:
    audio = load_wav_mono(path, config.sample_rate)
    sample_count = int(round(config.sample_rate * config.duration))
    fixed = pad_or_trim(audio, sample_count)
    mel = librosa.feature.melspectrogram(
        y=fixed,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)

