from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_sample_rate: float
    hostapi: int

    @property
    def display_name(self) -> str:
        return f"{self.index}: {self.name}"


def list_input_devices() -> list[InputDevice]:
    devices: list[InputDevice] = []
    for index, raw in enumerate(sd.query_devices()):
        if int(raw["max_input_channels"]) > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=str(raw["name"]),
                    channels=int(raw["max_input_channels"]),
                    default_sample_rate=float(raw["default_samplerate"]),
                    hostapi=int(raw["hostapi"]),
                )
            )
    return devices


def device_details(device_index: int) -> dict[str, Any]:
    return dict(sd.query_devices(device_index, "input"))


def record_audio(
    device_index: int,
    duration_sec: float,
    sample_rate: int = 48000,
    chunk_callback: Any = None,
) -> np.ndarray:
    """Record an exact number of mono samples with optional live updates."""
    if duration_sec <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    frames = max(1, int(round(duration_sec * sample_rate)))
    chunk_size = max(1, sample_rate // 10)  # ~100ms chunks for live updates
    
    try:
        if chunk_callback is not None:
            recording_data: list[np.ndarray] = []
            completed = threading.Event()
            received_frames = 0

            def audio_callback(indata, callback_frames, time_info, status):
                del callback_frames, time_info
                nonlocal received_frames
                if status and status.input_overflow:
                    completed.set()
                    raise sd.CallbackAbort

                remaining = frames - received_frames
                if remaining <= 0:
                    completed.set()
                    raise sd.CallbackStop

                chunk = np.asarray(indata[:remaining, 0], dtype=np.float32).copy()
                recording_data.append(chunk)
                received_frames += chunk.size
                chunk_callback(chunk)
                if received_frames >= frames:
                    completed.set()
                    raise sd.CallbackStop

            stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=sample_rate,
                callback=audio_callback,
                blocksize=chunk_size,
                dtype="float32",
            )
            with stream:
                timeout_sec = max(duration_sec + 5.0, duration_sec * 2.0)
                if not completed.wait(timeout_sec):
                    raise RuntimeError(
                        "Timed out while waiting for microphone audio frames."
                    )
            if received_frames != frames:
                raise RuntimeError(
                    f"Microphone stream ended after {received_frames} of "
                    f"{frames} requested samples."
                )
            recording = np.concatenate(recording_data)
        else:
            recording = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device_index,
                blocking=True,
            )
            recording = recording[:, 0]
    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

    return np.nan_to_num(recording, nan=0.0, posinf=1.0, neginf=-1.0)


def audio_levels(audio: np.ndarray) -> tuple[float, float]:
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    return rms, peak
