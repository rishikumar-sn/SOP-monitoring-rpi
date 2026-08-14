from __future__ import annotations
import atexit
import base64
import copy
import glob
import hashlib
import io
import json
import math
import os
import queue
import re
import sys
import threading
import time
import tempfile
import uuid
import wave
import subprocess
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape as xml_escape

import cv2
import numpy as np
from PIL import Image as PILImage
from flask import Flask, abort, jsonify, request, send_file, send_from_directory, Response
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, PageBreak

BASE_DIR = Path(__file__).resolve().parent
WEBUI_DIR = BASE_DIR / "webui"
RUNTIME_DIR = BASE_DIR / "runtime_sessions"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
PLEDGE_DIR = RUNTIME_DIR / "_pledges"
PLEDGE_DIR.mkdir(parents=True, exist_ok=True)
PLEDGE_MEDIA_DIR = PLEDGE_DIR / "media"
PLEDGE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_DIR = BASE_DIR / "Classification"
DIMENSION_DIR = BASE_DIR / "Dimension"
HANDREMOVER_DIR = BASE_DIR / "HandRemover"
SEGMENTATION_DIR = BASE_DIR / "Segmentation"
STONE_DIR = BASE_DIR / "StoneDetection"

for module_dir in (
    CLASSIFICATION_DIR,
    DIMENSION_DIR,
    HANDREMOVER_DIR,
    SEGMENTATION_DIR,
    STONE_DIR,
):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from Classification.jewelry_classifier import (  # noqa: E402
    DEFAULT_CACHE_FILE,
    DEFAULT_ONNX_MODEL,
    DEFAULT_PROMPT_FILE,
    DEFAULT_TEXT_MODEL_ID,
    JewelryZeroShotClassifier,
)
from Dimension.bangle_detector import detect_bangle  # noqa: E402
from HandRemover.handremover import (  # noqa: E402
    HAND_REMOVAL_PIPELINE_VERSION,
    extract_bangles,
    get_hand_model,
)
import StoneDetection.jewel_gem_hsv_report as stone_detection  # noqa: E402
import StoneDetection.stone_analysis_v2 as stone_analysis_v2  # noqa: E402
import StoneDetection.stone_area_calculator as stone_area_calculator  # noqa: E402
import Segmentation.segment_necklace_fastsam as necklace_segmentation  # noqa: E402
from hailo_model_runner import (
    DEFAULT_HAILO_BATCH_SIZE,
    DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
    HailoRuntime,
)
from super_resolution.real_esrgan_hailo import (
    DEFAULT_HEF_PATH as SUPER_RESOLUTION_HEF_PATH,
    MODEL_SCALE as SUPER_RESOLUTION_SCALE,
    RealESRGANHailoX2,
)
from purity_test_manager import PurityTestManager
from shutterspeedset_robust import align_to_powerline, denominator_to_exposure_absolute  # noqa: E402
from weight_ocr.reader import WeightReader


# 16x2 I2C LCD display (PCF8574 backpack, default address 0x27)
LCD_ENABLED = os.environ.get("LCD_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
try:
    LCD_I2C_ADDR = int(os.environ.get("LCD_I2C_ADDR", "0x27"), 0)
except Exception:
    LCD_I2C_ADDR = 0x27
try:
    LCD_I2C_BUS = int(os.environ.get("LCD_I2C_BUS", "1"))
except Exception:
    LCD_I2C_BUS = 1
LCD_COLUMNS = 16
LCD_ROWS = 2
LCD_BACKLIGHT = 0x08
LCD_ENABLE = 0x04
LCD_COMMAND = 0x00
LCD_DATA = 0x01


class I2CLcd16x2:
    def __init__(self, bus: int = LCD_I2C_BUS, address: int = LCD_I2C_ADDR):
        self.address = address
        self._lock = threading.RLock()
        self._bus = self._open_bus(bus)
        self._init_display()

    @staticmethod
    def _open_bus(bus: int):
        try:
            import smbus2 as smbus_module  # type: ignore
        except Exception:
            import smbus as smbus_module  # type: ignore
        return smbus_module.SMBus(bus)

    def _write_byte(self, value: int) -> None:
        self._bus.write_byte(self.address, value | LCD_BACKLIGHT)

    def _pulse(self, value: int) -> None:
        self._write_byte(value | LCD_ENABLE)
        time.sleep(0.0005)
        self._write_byte(value & ~LCD_ENABLE)
        time.sleep(0.0001)

    def _write4(self, value: int) -> None:
        self._write_byte(value)
        self._pulse(value)

    def _send(self, value: int, mode: int) -> None:
        self._write4(mode | (value & 0xF0))
        self._write4(mode | ((value << 4) & 0xF0))

    def _command(self, value: int) -> None:
        self._send(value, LCD_COMMAND)

    def _write_char(self, char: str) -> None:
        self._send(ord(char), LCD_DATA)

    def _init_display(self) -> None:
        time.sleep(0.05)
        for value in (0x30, 0x30, 0x30, 0x20):
            self._write4(value)
            time.sleep(0.005)
        self._command(0x28)  # 4-bit mode, 2 lines, 5x8 font
        self._command(0x0C)  # display on, cursor off
        self._command(0x06)  # entry mode: increment
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self._command(0x01)
            time.sleep(0.002)

    def show_lines(self, lines: list[str]) -> None:
        with self._lock:
            for row in range(LCD_ROWS):
                text = str(lines[row] if row < len(lines) else "")[:LCD_COLUMNS].ljust(LCD_COLUMNS)
                self._command(0x80 if row == 0 else 0xC0)
                for char in text:
                    self._write_char(char)

    def close(self) -> None:
        try:
            self.clear()
        except Exception:
            pass
        try:
            self._bus.close()
        except Exception:
            pass


import shutil

# ── Voice Config (Piper TTS) ────────────────────────────────
def find_piper_bin():
    # 1. Check system PATH
    sys_piper = shutil.which("piper")
    if sys_piper:
        return str(Path(sys_piper).resolve())
    # 2. Check bundled Linux and Windows binaries.
    for candidate in (
        BASE_DIR / "piper" / "build" / "piper",
        BASE_DIR / "piper" / "build" / "piper.exe",
        BASE_DIR / "piper" / "piper",
        BASE_DIR / "piper" / "piper.exe",
    ):
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return "piper"  # Final fallback to PATH

PIPER_BIN   = find_piper_bin()
PIPER_MODEL = (WEBUI_DIR / "en_US-amy-medium.onnx").resolve()
ESPEAK_DATA = (
    str((BASE_DIR / "piper" / "espeak-ng-data").resolve())      # bundled inside release ← preferred
    if (BASE_DIR / "piper" / "espeak-ng-data").exists()
    else "/usr/lib/aarch64-linux-gnu/espeak-ng-data"  # system fallback
)

# Version the cache so every workflow phrase is regenerated with the same
# full-volume normalization instead of mixing older, quieter WAV files with
# newly generated prompts.
TTS_CACHE_DIR = BASE_DIR / "piper" / "cache" / "full_volume_v1"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TTS_OUTPUT_AUTO = "__AUTO__"
TTS_OUTPUT_USB_ID_PREFIX = "alsa_usb:"
TTS_OUTPUT_ALSA_ID_PREFIX = "alsa_id:"
TTS_OUTPUT_DIRECT_PREFIX = "alsa_device:"
TTS_PREFERRED_OUTPUT_USB_IDS = {
    value.strip().lower()
    for value in os.environ.get("TTS_PREFERRED_OUTPUT_USB_IDS", "1b3f:2008").split(",")
    if value.strip()
}
TTS_OUTPUT_KEYWORDS = tuple(
    keyword.strip().lower()
    for keyword in os.environ.get("TTS_OUTPUT_KEYWORDS", "generalplus,usb audio device").split(",")
    if keyword.strip()
)
TTS_OUTPUT_EXCLUDE_KEYWORDS = tuple(
    keyword.strip().lower()
    for keyword in os.environ.get("TTS_OUTPUT_EXCLUDE_KEYWORDS", "ab13x,walmart,headset,adapter").split(",")
    if keyword.strip()
)
_TTS_OUTPUT_DEVICE: str | None = TTS_OUTPUT_AUTO
PURITY_AUDIO_SENSOR_USB_ID = "001f:0b21"


def _normalize_audio_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _usb_ids_for_alsa_card(card_index: int) -> tuple[str, str] | None:
    """Return (vendor, product) for an ALSA card when it is backed by USB."""
    card_path = Path(f"/sys/class/sound/card{int(card_index)}")
    try:
        current = card_path.resolve()
    except Exception:
        return None
    for path in (current, *current.parents):
        vendor_path = path / "idVendor"
        product_path = path / "idProduct"
        if vendor_path.exists() and product_path.exists():
            try:
                vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
                product = product_path.read_text(encoding="utf-8").strip().lower()
                if vendor and product:
                    return vendor, product
            except Exception:
                return None
    return None


def purity_audio_sensor_state() -> dict[str, Any]:
    vendor_id, product_id = PURITY_AUDIO_SENSOR_USB_ID.split(":", 1)
    connected = False
    for device_path in Path("/sys/bus/usb/devices").glob("*"):
        try:
            vendor = (device_path / "idVendor").read_text(encoding="utf-8").strip().lower()
            product = (device_path / "idProduct").read_text(encoding="utf-8").strip().lower()
        except (FileNotFoundError, OSError):
            continue
        if vendor == vendor_id and product == product_id:
            connected = True
            break
    return {
        "connected": connected,
        "usb_id": PURITY_AUDIO_SENSOR_USB_ID,
        "status": "Audio sensor connected." if connected else "Audio sensor not connected.",
    }


def list_tts_output_devices() -> list[dict[str, Any]]:
    """Return ALSA playback devices with stable IDs for the TTS speaker."""
    aplay_bin = shutil.which("aplay")
    if not aplay_bin:
        return []
    try:
        result = subprocess.run(
            [aplay_bin, "-l"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    devices: list[dict[str, Any]] = []
    pattern = re.compile(
        r"card\s+(?P<card>\d+):\s+(?P<card_id>[^\s\[]+)\s+\[(?P<card_name>[^\]]+)\],\s+"
        r"device\s+(?P<device>\d+):\s+(?P<device_id>[^\[]+)\[(?P<device_name>[^\]]+)\]",
        re.IGNORECASE,
    )
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        card_index = int(match.group("card"))
        device_index = int(match.group("device"))
        card_id = match.group("card_id").strip()
        card_name = match.group("card_name").strip()
        device_name = match.group("device_name").strip()
        alsa_device = f"plughw:CARD={card_id},DEV={device_index}"
        usb_ids = _usb_ids_for_alsa_card(card_index)
        if usb_ids:
            stable_id = f"{TTS_OUTPUT_USB_ID_PREFIX}{usb_ids[0]}:{usb_ids[1]}"
        else:
            stable_id = f"{TTS_OUTPUT_ALSA_ID_PREFIX}{_normalize_audio_name(card_id)}:{device_index}"
        devices.append(
            {
                "id": stable_id,
                "index": card_index,
                "card": card_index,
                "card_id": card_id,
                "card_name": card_name,
                "device": device_index,
                "device_name": device_name,
                "name": f"{card_name} ({device_name})",
                "alsa_device": alsa_device,
                "usb_id": f"{usb_ids[0]}:{usb_ids[1]}" if usb_ids else "",
                "preferred": False,
            }
        )
    for device in devices:
        device["preferred"] = _score_tts_output_device(device) > 0
    return devices


def _score_tts_output_device(device: dict[str, Any]) -> int:
    name = _normalize_audio_name(f"{device.get('card_name', '')} {device.get('device_name', '')}")
    usb_id = str(device.get("usb_id", "")).lower()
    score = 0
    if usb_id and usb_id in TTS_PREFERRED_OUTPUT_USB_IDS:
        score += 200
    if any(keyword and keyword in name for keyword in TTS_OUTPUT_KEYWORDS):
        score += 80
    if "usb" in name:
        score += 25
    if any(keyword and keyword in name for keyword in TTS_OUTPUT_EXCLUDE_KEYWORDS):
        score -= 120
    return score


def select_tts_output_device(value: Any) -> str:
    global _TTS_OUTPUT_DEVICE
    raw = str(value or "").strip()
    _TTS_OUTPUT_DEVICE = raw or TTS_OUTPUT_AUTO
    return _TTS_OUTPUT_DEVICE


def resolve_tts_output_device(value: Any | None = None) -> dict[str, Any]:
    selected = _TTS_OUTPUT_DEVICE if value is None else value
    selected_text = str(selected or TTS_OUTPUT_AUTO).strip()
    devices = list_tts_output_devices()
    if selected_text and selected_text != TTS_OUTPUT_AUTO:
        for device in devices:
            if selected_text in {
                str(device.get("id", "")),
                str(device.get("alsa_device", "")),
                f"{TTS_OUTPUT_DIRECT_PREFIX}{device.get('alsa_device', '')}",
            }:
                return {"mode": "selected", "connected": True, "device": device}
        return {"mode": "selected", "connected": False, "device": None}

    preferred = sorted(devices, key=_score_tts_output_device, reverse=True)
    if preferred and _score_tts_output_device(preferred[0]) > 0:
        return {"mode": "auto", "connected": True, "device": preferred[0]}
    return {"mode": "auto", "connected": False, "device": None}


def tts_output_state() -> dict[str, Any]:
    resolved = resolve_tts_output_device()
    device = resolved.get("device")
    return {
        "selected": _TTS_OUTPUT_DEVICE or TTS_OUTPUT_AUTO,
        "connected": bool(resolved.get("connected")),
        "resolved_device": device,
        "status": (
            f"Speaker ready: {device.get('name')}"
            if device
            else "Preferred USB speaker not connected."
        ),
    }

TTS_PHRASES = {
    "Processing started": "processing_started.wav",
    "Jewel type complete": "jewel_type_complete.wav",
    "Acid test has been started, keep the rubbing stone inside the camera feed.": "acid_test_started_keep_rubbing_stone.wav",
    "Acid test skipped.": "acid_test_skipped.wav",
    "Enter the Pledge ID": "enter_pledge_id.wav",
    "Enter the jewel count for this pledge.": "enter_jewel_count.wav",
    "Please place ornament": "please_place_ornament.wav",
    "Gold detected": "gold_detected.wav",
    "Starting rubbing analysis": "starting_rubbing_analysis.wav",
    "Please apply acid": "please_apply_acid.wav",
    "Starting acid analysis": "starting_acid_analysis.wav",
    "Test complete": "test_complete.wav"
}

TTS_WORKFLOW_PHRASES = {
    "Enter the Pledge ID",
    "Enter the jewel count for this pledge.",
    "Pledge ID set. Ready for the jewel workflow.",
    "Jewel count saved. Click Next to start the jewel workflow.",
    "Jewel count cleared. Enter the jewel count again.",
    "Ready for the next jewel. Capture the jewel image, then run jewel type.",
    "Jewel image is captured. Run jewel type.",
    "Jewel image is captured. Running jewel type.",
    "This does not look like gold jewelry. Click Yes if correct, or No to override.",
    "Jewel type confirmed. Click next for Jewel Weight Extraction.",
    "Weight could not be detected. Place the weight scale inside the box region and recapture.",
    "Side image captured. Click Start Stone Analysis.",
    "Stone analysis completed. Click next for Acid Test.",
    "Jewellery analysis completed. Click next for Stone Detection.",
    "Jewelry risk analysis initiated.",
    "Stone detection completed. Click next for Acid Test.",
    "Current jewel completed. Click Next Jewel to continue.",
    "All jewels are completed. Capture the final jewel count.",
    "All jewels are completed. Capture the final jewel count.",
    "Current jewel completed. Continue the workflow.",
    "Not gold jewelry confirmed. It was not counted. Capture another item for the same jewel number.",
    "Please enter Pledge ID first.",
    "Please enter and save the jewel count first.",
    "Complete all jewels and acid tests before final count capture.",
    "Place all jewels on the test bed and capture the jewel count.",
    "Jewel count captured. Start packet sealing.",
    "Jewel count mismatch. Please verify before packet sealing.",
    "Capture the final jewel count before packet sealing.",
    "Packet sealing recording started.",
    "Packet sealing recording started. Put all jewels into the packet and seal it.",
    "Packet sealing stopped. Video is still compressing.",
    "Packet sealing video saved. Final report is ready.",
    "Remove the packet strip, then press Hand Clear.",
    "Packet sealed.",
    "Packet not sealed.",
    "Acid test has been started, keep the rubbing stone inside the camera feed.",
    "Now Rubbing stone is detected, use the jewelry to run on it",
    "Jewelry is now inside the stone region, now start rubbing for acid test",
    "Visual and audio synchronization is okay. Now, apply the acid to complete the purity test.",
    "Acid detected. Purity test completed. Click Stop to continue.",
}


def _all_tts_preload_phrases() -> list[str]:
    phrases = set(TTS_PHRASES)
    phrases.update(TTS_WORKFLOW_PHRASES)
    phrases.update(
        side_capture_voice_prompt(label)
        for label in ("Bangle", "Finger ring", "Necklace")
    )
    class_labels = set(globals().get("ALL_DISPLAY_LABELS", []))
    class_labels.update(globals().get("MODEL_LABELS", []))
    for label in class_labels:
        phrases.add(f"Jewel type predicted as {label}. Click Yes if correct, or No to override.")
    return sorted(phrase for phrase in phrases if phrase)


def _tts_cache_path(text: str) -> Path:
    known_filename = TTS_PHRASES.get(text)
    if known_filename:
        return TTS_CACHE_DIR / known_filename
    readable = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "voice"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return TTS_CACHE_DIR / f"{readable}_{digest}.wav"


def _piper_environment() -> dict[str, str]:
    env = os.environ.copy()
    if os.path.exists(ESPEAK_DATA):
        env["ESPEAK_DATA_PATH"] = ESPEAK_DATA

    bin_dir = Path(PIPER_BIN).parent
    lib_dirs = [bin_dir, bin_dir.parent / "lib", bin_dir.parent / "build" / "lib"]
    existing_lib_dirs = [str(path) for path in lib_dirs if path.exists()]
    if existing_lib_dirs:
        current_ld = env.get("LD_LIBRARY_PATH", "")
        new_ld = os.pathsep.join(existing_lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{new_ld}{os.pathsep}{current_ld}" if current_ld else new_ld
    return env


def _normalize_pcm_wav(source_path: Path, output_path: Path) -> bool:
    """Peak-normalize Piper PCM16 output when SoX is unavailable."""
    try:
        with wave.open(str(source_path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())
        if params.sampwidth != 2 or not frames:
            return False
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 0:
            return False
        normalized = np.clip(samples * (0.95 * 32767.0 / peak), -32768, 32767).astype("<i2")
        with wave.open(str(output_path), "wb") as output:
            output.setparams(params)
            output.writeframes(normalized.tobytes())
        return True
    except Exception as exc:
        print(f"Voice PCM normalization failed: {exc}")
        return False


def _ensure_tts_cached(text: str) -> Path | None:
    cache_path = _tts_cache_path(text)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    with TTS_CACHE_LOCK:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path
        if os.path.isabs(PIPER_BIN) and not os.path.exists(PIPER_BIN):
            print(f"Voice disabled: Piper binary not found at {PIPER_BIN}")
            return None
        if not os.path.isabs(PIPER_BIN) and not shutil.which(PIPER_BIN):
            print(f"Voice disabled: Piper binary not found: {PIPER_BIN}")
            return None
        if not PIPER_MODEL.exists():
            print(f"Voice disabled: Model not found at {PIPER_MODEL}")
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_file = tempfile.NamedTemporaryFile(
            prefix="piper_raw_",
            suffix=".wav",
            dir=TTS_CACHE_DIR,
            delete=False,
        )
        raw_path = Path(raw_file.name)
        raw_file.close()
        processed_path = raw_path.with_name(f"{raw_path.stem}_processed.wav")
        try:
            result = subprocess.run(
                [PIPER_BIN, "--model", str(PIPER_MODEL), "--output_file", str(raw_path)],
                input=text.encode("utf-8"),
                env=_piper_environment(),
                capture_output=True,
                timeout=20,
            )
            if result.returncode != 0:
                stderr_text = result.stderr.decode(errors="replace").strip()
                print(f"Voice cache generation failed for {text!r}: {stderr_text}")
                return None

            source_path = raw_path
            sox_bin = shutil.which("sox")
            if sox_bin:
                sox_result = subprocess.run(
                    [sox_bin, str(raw_path), str(processed_path), "gain", "-n", "-1"],
                    capture_output=True,
                    timeout=10,
                )
                if sox_result.returncode == 0 and processed_path.exists():
                    source_path = processed_path
                else:
                    print(
                        f"Voice volume boost failed for {text!r}; using unboosted audio: "
                        f"{sox_result.stderr.decode(errors='replace').strip()}"
                    )
            if source_path == raw_path and _normalize_pcm_wav(raw_path, processed_path):
                source_path = processed_path

            os.replace(source_path, cache_path)
            print(f"Cached TTS: {text!r} -> {cache_path.name}")
            return cache_path
        except subprocess.TimeoutExpired:
            print(f"Voice cache generation timed out for {text!r}")
        except Exception as exc:
            print(f"Voice cache generation failed for {text!r}: {exc}")
        finally:
            for temporary_path in (raw_path, processed_path):
                if temporary_path.exists():
                    try:
                        temporary_path.unlink()
                    except Exception:
                        pass
    return None


def preload_tts_phrases():
    """Pre-generate known workflow phrases; new phrases are cached on first use."""
    print("=" * 60)
    print("PRELOADING TTS PHRASES (This may take a moment)")
    print("=" * 60)
    phrases = _all_tts_preload_phrases()
    cached_count = 0
    for phrase in phrases:
        if _ensure_tts_cached(phrase):
            cached_count += 1
    print(f"TTS cache ready: {cached_count}/{len(phrases)} known commands")
    print("=" * 60)

def _tts_worker() -> None:
    """Background worker: processes TTS requests from the queue one at a time."""
    while True:
        text = TTS_QUEUE.get()
        if text is None:
            break
        try:
            _speak_sync(text)
        except Exception as exc:
            print(f"⚠️  TTS worker error: {exc}")
        TTS_QUEUE.task_done()


def ensure_tts_worker() -> None:
    """Start the TTS background worker thread (idempotent)."""
    global _TTS_WORKER_STARTED
    if _TTS_WORKER_STARTED:
        return
    _TTS_WORKER_STARTED = True
    t = threading.Thread(target=_tts_worker, name="tts-worker", daemon=True)
    t.start()


def speak(text: str) -> None:
    """Speak text using Piper TTS (non-blocking) with deduplication."""
    global _LAST_SPOKEN_TEXT, _LAST_SPOKEN_TIME
    text = str(text or "").strip()
    if not text:
        return
    now = time.time()
    last_for_text = _TTS_LAST_BY_TEXT.get(text, 0.0)
    if now - last_for_text < TTS_DEDUP_SECONDS:
        return
    _LAST_SPOKEN_TEXT = text
    _LAST_SPOKEN_TIME = now
    _TTS_LAST_BY_TEXT[text] = now
    ensure_tts_worker()
    try:
        TTS_QUEUE.put_nowait(text)
    except queue.Full:
        print(f"⚠️  TTS queue full — dropping: {text!r}")


def _speak_sync(text: str) -> None:
    """Synchronous TTS implementation (called by the background worker)."""
    cache_path = _ensure_tts_cached(text)
    if cache_path is None:
        return

    if os.path.isabs(PIPER_BIN) and os.path.exists(PIPER_BIN):
        try:
            os.chmod(PIPER_BIN, 0o755)
        except Exception:
            pass

    try:
        print(f"Playing cached TTS: {cache_path}")
        with AUDIO_LOCK:
            if os.name == "nt":
                import winsound

                winsound.PlaySound(str(cache_path), winsound.SND_FILENAME)
            else:
                aplay_bin = shutil.which("aplay")
                if not aplay_bin:
                    print("Voice playback disabled: 'aplay' is not installed.")
                    return
                playback_device = "default"
                output_resolution = resolve_tts_output_device()
                resolved_device = output_resolution.get("device")
                if resolved_device:
                    playback_device = str(resolved_device.get("alsa_device") or "default")
                elif output_resolution.get("mode") == "selected":
                    print("Voice playback skipped: selected USB speaker is not connected.")
                    return
                else:
                    print("Voice playback warning: preferred USB speaker not found; using ALSA default.")
                aplay_res = subprocess.run(
                    [aplay_bin, "-D", playback_device, str(cache_path)],
                    capture_output=True,
                    timeout=15,
                )
                if aplay_res.returncode != 0:
                    print(f"Voice playback failed: {aplay_res.stderr.decode(errors='replace').strip()}")
    except subprocess.TimeoutExpired:
        print("Voice playback timed out.")
    except Exception as exc:
        print(f"Voice playback failed: {exc}")


APP_PORT = 5050
ESTIMATED_TASSEL_WEIGHT_G = 1.5

STAGE_DISPLAY_NAMES = {
    "weight_extraction": "Jewel Weight Extraction",
    "dimension": "Dimension Analysis",
    "side_stone": "Stone Detection",
    "jewellery_analysis": "Jewellery Analysis",
    "stone_detection": "Stone Detection",
    "acid_test": "Acid Test",
    "final_count": "Final Jewel Count",
    "packet_sealing": "Packet Sealing Video",
}

def find_working_camera():
    """Automatically find the first working camera index."""
    env_cam = os.environ.get("CAM_DEVICE", "").strip()
    if env_cam:
        print(f"Using CAM_DEVICE from environment: {env_cam}")
        return env_cam

    print("Probing for working camera...")
    # Try indices 0 to 4
    for i in range(5):
        # Use V4L2 for Linux, DSHOW for Windows
        api = cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                print(f"✓ Found working camera at index: {i}")
                return str(i)
            cap.release()
    
    print("⚠️  No working camera found during probe. Defaulting to '0'.")
    return "0"

CAM_DEVICE = find_working_camera()
CAM_ROTATE_90_CLOCKWISE = os.environ.get("CAM_ROTATE_90_CLOCKWISE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    CAM_TARGET_WIDTH = max(1, int(os.environ.get("CAM_TARGET_WIDTH", "2560")))
except Exception:
    CAM_TARGET_WIDTH = 2560
try:
    CAM_TARGET_HEIGHT = max(1, int(os.environ.get("CAM_TARGET_HEIGHT", "1440")))
except Exception:
    CAM_TARGET_HEIGHT = 1440
try:
    PURITY_CAM_WIDTH = max(1, int(os.environ.get("PURITY_CAM_WIDTH", "1280")))
except Exception:
    PURITY_CAM_WIDTH = 1280
try:
    PURITY_CAM_HEIGHT = max(1, int(os.environ.get("PURITY_CAM_HEIGHT", "720")))
except Exception:
    PURITY_CAM_HEIGHT = 720
try:
    CAM_PROBE_GRABS = max(0, int(os.environ.get("CAM_PROBE_GRABS", "3")))
except Exception:
    CAM_PROBE_GRABS = 3
try:
    CAM_PREVIEW_JPEG_QUALITY = max(60, min(95, int(os.environ.get("CAM_PREVIEW_JPEG_QUALITY", "75"))))
except Exception:
    CAM_PREVIEW_JPEG_QUALITY = 75
try:
    CAMERA_FRAME_LOOP_DELAY_S = max(0.01, float(os.environ.get("CAMERA_FRAME_LOOP_DELAY_S", "0.03")))
except Exception:
    CAMERA_FRAME_LOOP_DELAY_S = 0.03
try:
    CAMERA_REOPEN_DELAY_S = max(0.25, float(os.environ.get("CAMERA_REOPEN_DELAY_S", "1.0")))
except Exception:
    CAMERA_REOPEN_DELAY_S = 1.0
CAMERA_FOCUS_WARMUP_FRAMES = 60
CAMERA_FOCUS_STABILITY_FRAMES = 12
CAMERA_FOCUS_MAX_RELATIVE_SPREAD = 0.05

# Shutter-speed / exposure configuration (from shutterspeedset_robust.py)
CAMERA_EXPOSURE_MODE = os.environ.get(
    "CAMERA_EXPOSURE_MODE",
    "native_auto",
).strip().lower()
if CAMERA_EXPOSURE_MODE not in {"native_auto", "startup_fixed"}:
    CAMERA_EXPOSURE_MODE = "native_auto"
try:
    SHUTTER_DENOMINATOR = int(os.environ.get("SHUTTER_DENOMINATOR", "100"))
except Exception:
    SHUTTER_DENOMINATOR = 100
try:
    POWER_LINE_HZ = int(os.environ.get("POWER_LINE_HZ", "50"))
except Exception:
    POWER_LINE_HZ = 50
try:
    EXPOSURE_FLUSH_FRAMES = max(1, int(os.environ.get("EXPOSURE_FLUSH_FRAMES", "10")))
except Exception:
    EXPOSURE_FLUSH_FRAMES = 10
CAMERA_STARTUP_EXPOSURE_CALIBRATION = os.environ.get(
    "CAMERA_STARTUP_EXPOSURE_CALIBRATION",
    "true",
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    EXPOSURE_TARGET_LUMA = max(80.0, min(235.0, float(os.environ.get("EXPOSURE_TARGET_LUMA", "195"))))
except Exception:
    EXPOSURE_TARGET_LUMA = 195.0
try:
    EXPOSURE_TARGET_PERCENTILE = max(50.0, min(99.0, float(os.environ.get("EXPOSURE_TARGET_PERCENTILE", "90"))))
except Exception:
    EXPOSURE_TARGET_PERCENTILE = 90.0
try:
    EXPOSURE_HIGHLIGHT_PERCENTILE = max(
        EXPOSURE_TARGET_PERCENTILE,
        min(99.9, float(os.environ.get("EXPOSURE_HIGHLIGHT_PERCENTILE", "98"))),
    )
except Exception:
    EXPOSURE_HIGHLIGHT_PERCENTILE = 98.0
try:
    EXPOSURE_HIGHLIGHT_LIMIT = max(180.0, min(254.0, float(os.environ.get("EXPOSURE_HIGHLIGHT_LIMIT", "238"))))
except Exception:
    EXPOSURE_HIGHLIGHT_LIMIT = 238.0
try:
    EXPOSURE_DEADBAND_LUMA = max(1.0, min(30.0, float(os.environ.get("EXPOSURE_DEADBAND_LUMA", "6"))))
except Exception:
    EXPOSURE_DEADBAND_LUMA = 6.0
try:
    EXPOSURE_MAX_STEP_FRACTION = max(0.05, min(0.75, float(os.environ.get("EXPOSURE_MAX_STEP_FRACTION", "0.35"))))
except Exception:
    EXPOSURE_MAX_STEP_FRACTION = 0.35
try:
    EXPOSURE_MIN_ABSOLUTE = max(1, int(os.environ.get("EXPOSURE_MIN_ABSOLUTE", "5")))
except Exception:
    EXPOSURE_MIN_ABSOLUTE = 5
try:
    EXPOSURE_MAX_ABSOLUTE = max(EXPOSURE_MIN_ABSOLUTE, int(os.environ.get("EXPOSURE_MAX_ABSOLUTE", "250")))
except Exception:
    EXPOSURE_MAX_ABSOLUTE = 250
try:
    EXPOSURE_CENTER_ROI_FRACTION = max(0.20, min(1.0, float(os.environ.get("EXPOSURE_CENTER_ROI_FRACTION", "0.72"))))
except Exception:
    EXPOSURE_CENTER_ROI_FRACTION = 0.72
try:
    EXPOSURE_STATS_MAX_PIXELS = max(10000, int(os.environ.get("EXPOSURE_STATS_MAX_PIXELS", "250000")))
except Exception:
    EXPOSURE_STATS_MAX_PIXELS = 250000
CAMERA_DISABLE_AUTO_GAIN = os.environ.get("CAMERA_DISABLE_AUTO_GAIN", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
try:
    CAMERA_NATIVE_GAIN = max(0, min(255, int(os.environ.get("CAMERA_NATIVE_GAIN", "0"))))
except Exception:
    CAMERA_NATIVE_GAIN = 0
try:
    EXPOSURE_STARTUP_SETTLE_FRAMES = max(
        5,
        int(os.environ.get("EXPOSURE_STARTUP_SETTLE_FRAMES", "35")),
    )
except Exception:
    EXPOSURE_STARTUP_SETTLE_FRAMES = 35
try:
    EXPOSURE_STARTUP_SAMPLE_FRAMES = max(
        1,
        int(os.environ.get("EXPOSURE_STARTUP_SAMPLE_FRAMES", "5")),
    )
except Exception:
    EXPOSURE_STARTUP_SAMPLE_FRAMES = 5
try:
    EXPOSURE_STARTUP_MAX_ADJUSTMENTS = max(
        1,
        int(os.environ.get("EXPOSURE_STARTUP_MAX_ADJUSTMENTS", "8")),
    )
except Exception:
    EXPOSURE_STARTUP_MAX_ADJUSTMENTS = 8
try:
    EXPOSURE_STARTUP_FLUSH_FRAMES = max(
        1,
        int(os.environ.get("EXPOSURE_STARTUP_FLUSH_FRAMES", "4")),
    )
except Exception:
    EXPOSURE_STARTUP_FLUSH_FRAMES = 4

CLASS_PROMPT_PATH = CLASSIFICATION_DIR / DEFAULT_PROMPT_FILE
SEG_MODEL_PATH = SEGMENTATION_DIR / "fast_sam_s.hef"
HAND_MODEL_PATH = HANDREMOVER_DIR / "handremover.hef"
BEAD_MODEL_PATH = BASE_DIR / "models" / "bead_finder.hef"
BEAD_YOLO_SCORE_THRESHOLD = 0.50
BEAD_CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "beadcheck_mobilenet_v3.pt"
BEAD_CLASSIFIER_TRUE_THRESHOLD = 0.50
CLASS_MODEL_PATH = CLASSIFICATION_DIR / "siglip2-base-patch32-256_vision_encoder.sim.onnx"
SEGMENTATION_FEEDBACK_DIR = SEGMENTATION_DIR / "feedback"
SEGMENTATION_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
ARUCO_AVAILABLE = bool(
    hasattr(cv2, "aruco")
    and hasattr(cv2.aruco, "DICT_APRILTAG_36h11")
)
ARUCO_DICTS = (
    {"AprilTag_36h11": cv2.aruco.DICT_APRILTAG_36h11}
    if ARUCO_AVAILABLE
    else {}
)
APRILTAG_DEFAULT_LENGTH_MM = float(os.environ.get("APRILTAG_LENGTH_MM", "20.0"))
APRILTAG_DEFAULT_BREADTH_MM = float(
    os.environ.get("APRILTAG_BREADTH_MM", str(APRILTAG_DEFAULT_LENGTH_MM))
)
APRILTAG_DEFAULT_ID = int(os.environ.get("APRILTAG_MARKER_ID", "1"))
CAMERA_TO_BED_DEFAULT_MM = float(os.environ.get("CAMERA_TO_BED_MM", "500.0"))

legacy_feedback_dirs = [
    path
    for path in RUNTIME_DIR.glob("*/segmentation_feedback")
    if path.is_dir()
]
if legacy_feedback_dirs:
    try:
        necklace_segmentation.migrate_feedback_library(legacy_feedback_dirs, SEGMENTATION_FEEDBACK_DIR)
    except Exception:
        pass
try:
    necklace_segmentation.upgrade_feedback_library(SEGMENTATION_FEEDBACK_DIR)
except Exception as exc:
    print(f"Segmentation feedback upgrade skipped: {exc}")

DIMENSION_CLASSES = {"Bangle", "Finger ring", "Finger Ring"}
DIRECT_STONE_CLASSES = {
    "bracelet",
    "Bracelet",
    "Mattal",
    "chain",
    "Chain",
    "Earing / Jumkha",
    "Earrings/ Jhumki",
    "Anklet",
    "Armlet",
    "Baby Jewellery",
    "Brooch",
    "Button Set",
    "Dollar",
    "Cufflinks",
    "Nose Ring",
    "Pendant",
    "Pin",
    "Tiara (Head)",
    "Toe Ring",
    "Waist Belt",
    "Watch Strap",
    "Jada billa",
    "Hair Ornament/ Maang Tikka",
    "Hair Ornament/ Nethi Chutti",
    "Hair Ornament/ Hair Clips",
    "Hair Ornament/ Juda Pin",
}
SEGMENTATION_CLASSES = {
    "Haram",
    "Necklace",
    "Dollar chain",
    "Dollar Chain",
    "Kasu Mala",
    "Kasu Malai",
    "Mangalsutra",
}
HIGH_RISK_STONE_THRESHOLD = 40.0
STONE_ANALYSIS_V2_DEBUG = os.environ.get(
    "STONE_ANALYSIS_V2_DEBUG",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
COUNT_CAPTURE_MIN_AREA_RATIO = float(os.environ.get("COUNT_CAPTURE_MIN_AREA_RATIO", "0.00045"))
COUNT_CAPTURE_MIN_AREA_PX = int(os.environ.get("COUNT_CAPTURE_MIN_AREA_PX", "350"))
COUNT_CAPTURE_PADDING_PX = int(os.environ.get("COUNT_CAPTURE_PADDING_PX", "14"))
PACKET_REC_SCALE = float(os.environ.get("PACKET_REC_SCALE", "0.75"))
PACKET_REC_MAX_DIMENSION = int(os.environ.get("PACKET_REC_MAX_DIMENSION", "720"))
PACKET_REC_FPS = float(os.environ.get("PACKET_REC_FPS", "6"))
PACKET_REC_PLAYBACK_SPEED = float(os.environ.get("PACKET_REC_PLAYBACK_SPEED", "2"))
PACKET_REC_CODEC = os.environ.get("PACKET_REC_CODEC", "mp4v")
PACKET_AV1_TRANSCODE = os.environ.get("PACKET_AV1_TRANSCODE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PACKET_AV1_FFMPEG = os.environ.get("PACKET_AV1_FFMPEG", "ffmpeg")
PACKET_AV1_PRESET = int(os.environ.get("PACKET_AV1_PRESET", "10"))
PACKET_AV1_CRF = int(os.environ.get("PACKET_AV1_CRF", "34"))
PACKET_TARGET_SIZE_BYTES = int(os.environ.get("PACKET_TARGET_SIZE_BYTES", "2000000"))
PACKET_TARGET_SIZE_MARGIN = float(os.environ.get("PACKET_TARGET_SIZE_MARGIN", "0.90"))
PACKET_TARGET_MIN_VIDEO_KBPS = int(os.environ.get("PACKET_TARGET_MIN_VIDEO_KBPS", "160"))
PACKET_TARGET_MAX_VIDEO_KBPS = int(os.environ.get("PACKET_TARGET_MAX_VIDEO_KBPS", "1800"))
STRIPING_PROCESS_DIR = BASE_DIR / "jewel_tracka_rpi"
STRIPING_BAG_HEF_PATH = STRIPING_PROCESS_DIR / "bag.hef"
STRIPING_HEF_PATH = STRIPING_PROCESS_DIR / "strip-m.hef"

PROMPT_CONFIG = json.loads(CLASS_PROMPT_PATH.read_text(encoding="utf-8"))
MODEL_LABELS = list(PROMPT_CONFIG["classes"].keys())
ALL_DISPLAY_LABELS = [
    "Not Gold Jewelry",
    "Anklet",
    "Armlet",
    "Baby Jewellery",
    "Bangle",
    "Bracelet",
    "Brooch",
    "Button Set",
    "chain",
    "Dollar chain",
    "Dollar",
    "Cufflinks",
    "Earing / Jumkha",
    "Finger ring",
    "Hair Ornament/ Maang Tikka",
    "Hair Ornament/ Nethi Chutti",
    "Hair Ornament/ Hair Clips",
    "Hair Ornament/ Juda Pin",
    "Necklace",
    "Nose Ring",
    "Pendant",
    "Pin",
    "Tiara (Head)",
    "Toe Ring",
    "Waist Belt",
    "Watch Strap",
    "Mattal",
    "Haram",
    "Jada billa",
    "Kasu Mala",
    "Mangalsutra",
]
VALID_CLASSIFICATION_LABELS = set(ALL_DISPLAY_LABELS) | set(MODEL_LABELS)


def _label_key(label: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label or "").casefold()).strip()


MODEL_LOCK = threading.RLock()
STATE_LOCK = threading.Lock()
CAMERA_LOCK = threading.Lock()
AUDIO_LOCK = threading.Lock()
TTS_CACHE_LOCK = threading.Lock()
TTS_QUEUE: queue.Queue[str | None] = queue.Queue(maxsize=8)
_TTS_WORKER_STARTED = False
_LAST_SPOKEN_TEXT: str = ""
_LAST_SPOKEN_TIME: float = 0.0
TTS_DEDUP_SECONDS: float = 60.0
_TTS_LAST_BY_TEXT: dict[str, float] = {}
CLASSIFIER: JewelryZeroShotClassifier | None = None
SEGMENTER: necklace_segmentation.FastSamOnnx | None = None
BEAD_MODEL: Any | None = None
BEAD_CLASSIFIER_FILTER: Any | None = None
SUPER_RESOLUTION_RUNNER: RealESRGANHailoX2 | None = None
CURRENT_STATE: dict[str, Any] = {}
PURITY_MANAGER: PurityTestManager | None = None
PACKET_RECORDER: "PacketSealingRecorder | None" = None
WEIGHT_READER: WeightReader | None = None
WEIGHT_READER_LOCK = threading.Lock()
PACKET_STRIP_HAILO_MODELS: dict[str, Any] | None = None
LCD_DISPLAY: I2CLcd16x2 | None = None
LCD_INIT_ATTEMPTED = False

DEFAULT_STONE_SUPER_RESOLUTION = {
    "enabled": False,
    "scale": int(SUPER_RESOLUTION_SCALE),
    "model": "real_esrgan_x2",
}

DEFAULT_PURITY_AUDIO_SETTINGS = {
    "ok_confidence_threshold": 0.70,
}

PERSISTENT_ROIS = {
    "processing_roi": None,
    "aruco_roi": None,
    "weight_roi": None,
    "color_correction": dict(stone_detection.DEFAULT_COLOR_CORRECTION),
    "analysis_normalization": dict(stone_detection.DEFAULT_ANALYSIS_NORMALIZATION),
    "background_calibration": None,
    "learned_stone_profiles": [],
    "stone_super_resolution": dict(DEFAULT_STONE_SUPER_RESOLUTION),
    "audio_settings": dict(DEFAULT_PURITY_AUDIO_SETTINGS),
    "calibration_config": {
        "aruco_dict": "AprilTag_36h11",
        "marker_id": APRILTAG_DEFAULT_ID,
        "marker_length_mm": APRILTAG_DEFAULT_LENGTH_MM,
        "marker_breadth_mm": APRILTAG_DEFAULT_BREADTH_MM,
        "camera_to_bed_mm": CAMERA_TO_BED_DEFAULT_MM,
        "nominal_stone_height_mm": 0.0,
        "camera_matrix": None,
        "dist_coeffs": None,
        "four_marker_enabled": False,
        "four_marker_ids": [0, 1, 2, 3],
        "marker_center_width_mm": 0.0,
        "marker_center_height_mm": 0.0,
    },
}

ROI_CONFIG_PATH = BASE_DIR / "roi_config.json"
CAMERA_CALIBRATION_PATH = Path(
    os.environ.get(
        "CAMERA_CALIBRATION_PATH",
        str(BASE_DIR / "camera_calibration.json"),
    )
).expanduser()


def normalize_metric_calibration_config(settings: dict | None) -> dict[str, Any]:
    raw = settings or {}
    try:
        marker_id = int(raw.get("marker_id", APRILTAG_DEFAULT_ID))
    except (TypeError, ValueError):
        marker_id = APRILTAG_DEFAULT_ID
    marker_side = float(
        raw.get("marker_side_mm", raw.get("marker_length_mm", APRILTAG_DEFAULT_LENGTH_MM))
    )
    marker_length = max(0.1, float(raw.get("marker_length_mm", marker_side)))
    marker_breadth = max(0.1, float(raw.get("marker_breadth_mm", marker_side)))
    camera_to_bed = max(
        1.0,
        float(raw.get("camera_to_bed_mm", CAMERA_TO_BED_DEFAULT_MM)),
    )
    stone_height = max(
        0.0,
        min(camera_to_bed * 0.5, float(raw.get("nominal_stone_height_mm", 0.0))),
    )
    raw_ids = raw.get("four_marker_ids", [0, 1, 2, 3])
    if isinstance(raw_ids, str):
        raw_ids = [part.strip() for part in raw_ids.split(",") if part.strip()]
    try:
        marker_ids = [int(value) for value in list(raw_ids)[:4]]
    except Exception:
        marker_ids = [0, 1, 2, 3]
    if len(marker_ids) != 4 or len(set(marker_ids)) != 4:
        marker_ids = [0, 1, 2, 3]

    camera_matrix = raw.get("camera_matrix")
    dist_coeffs = raw.get("dist_coeffs")
    if not (
        isinstance(camera_matrix, list)
        and len(camera_matrix) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in camera_matrix)
    ):
        camera_matrix = None
    if not isinstance(dist_coeffs, list) or len(dist_coeffs) < 4:
        dist_coeffs = None

    return {
        "aruco_dict": str(raw.get("aruco_dict") or "AprilTag_36h11"),
        "marker_id": marker_id,
        "marker_length_mm": marker_length,
        "marker_breadth_mm": marker_breadth,
        "camera_to_bed_mm": camera_to_bed,
        "nominal_stone_height_mm": stone_height,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "four_marker_enabled": bool(raw.get("four_marker_enabled", False)),
        "four_marker_ids": marker_ids,
        "marker_center_width_mm": max(
            0.0,
            float(raw.get("marker_center_width_mm", 0.0)),
        ),
        "marker_center_height_mm": max(
            0.0,
            float(raw.get("marker_center_height_mm", 0.0)),
        ),
    }


def normalize_stone_super_resolution_settings(settings: dict | None) -> dict[str, Any]:
    raw = settings or {}
    return {
        "enabled": bool(raw.get("enabled", DEFAULT_STONE_SUPER_RESOLUTION["enabled"])),
        "scale": int(SUPER_RESOLUTION_SCALE),
        "model": "real_esrgan_x2",
    }


def normalize_purity_audio_settings(settings: dict | None) -> dict[str, Any]:
    raw = settings or {}
    try:
        threshold = float(
            raw.get(
                "ok_confidence_threshold",
                DEFAULT_PURITY_AUDIO_SETTINGS["ok_confidence_threshold"],
            )
        )
    except (TypeError, ValueError):
        threshold = DEFAULT_PURITY_AUDIO_SETTINGS["ok_confidence_threshold"]
    return {
        "ok_confidence_threshold": max(0.50, min(0.99, threshold)),
    }


def get_lcd() -> I2CLcd16x2 | None:
    """Return the optional 16x2 LCD instance, initializing it once."""
    global LCD_DISPLAY, LCD_INIT_ATTEMPTED
    if not LCD_ENABLED:
        return None
    if LCD_DISPLAY is None and not LCD_INIT_ATTEMPTED:
        LCD_INIT_ATTEMPTED = True
        try:
            LCD_DISPLAY = I2CLcd16x2(bus=LCD_I2C_BUS, address=LCD_I2C_ADDR)
        except Exception as exc:
            print(f"LCD init failed at 0x{LCD_I2C_ADDR:02X}: {exc}")
            LCD_DISPLAY = None
    return LCD_DISPLAY


def lcd_show(lines: list[str]) -> None:
    """Display two short lines on the optional 16x2 LCD."""
    try:
        lcd = get_lcd()
        if lcd:
            lcd.show_lines(lines[:LCD_ROWS])
    except Exception:
        pass


def lcd_show_pledge_status(
    pledge_id: str | None,
    jewel_count: Any = None,
    jewel_index: Any = None,
) -> None:
    pledge = str(pledge_id or "").strip()
    if not pledge:
        lcd_show(["Enter Pledge ID", ""])
        return

    line1 = f"ID:{pledge}"
    try:
        count = int(jewel_count or 0)
    except Exception:
        count = 0
    try:
        index = int(jewel_index or 0)
    except Exception:
        index = 0

    if count > 0 and index > 0:
        line2 = f"Jewel {index}/{count}"
    elif count > 0:
        line2 = f"Jewels: {count}"
    else:
        line2 = "Jewels: --"
    lcd_show([line1, line2])


def load_persistent_rois() -> None:
    global PERSISTENT_ROIS
    if ROI_CONFIG_PATH.exists():
        try:
            with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                PERSISTENT_ROIS["processing_roi"] = data.get("processing_roi")
                PERSISTENT_ROIS["aruco_roi"] = data.get("aruco_roi")
                PERSISTENT_ROIS["weight_roi"] = data.get("weight_roi")
                PERSISTENT_ROIS["color_correction"] = stone_detection.normalize_color_correction(
                    data.get("color_correction")
                )
                PERSISTENT_ROIS["analysis_normalization"] = (
                    stone_detection.normalize_analysis_normalization(
                        data.get("analysis_normalization")
                    )
                )
                PERSISTENT_ROIS["background_calibration"] = data.get("background_calibration")
                PERSISTENT_ROIS["learned_stone_profiles"] = (
                    stone_detection.normalize_learned_stone_profiles(
                        data.get("learned_stone_profiles")
                    )
                )
                PERSISTENT_ROIS["stone_super_resolution"] = (
                    normalize_stone_super_resolution_settings(
                        data.get("stone_super_resolution")
                    )
                )
                PERSISTENT_ROIS["audio_settings"] = normalize_purity_audio_settings(
                    data.get("audio_settings")
                )
                PERSISTENT_ROIS["calibration_config"] = normalize_metric_calibration_config(
                    data.get("calibration_config")
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load persistent ROIs: {exc}")


def load_project_camera_calibration() -> bool:
    """Load lens intrinsics from the project calibration JSON when available."""
    calibration_path = CAMERA_CALIBRATION_PATH
    if not calibration_path.is_file():
        print(f"Camera calibration JSON not found: {calibration_path}")
        return False

    try:
        with open(calibration_path, "r", encoding="utf-8") as calibration_file:
            data = json.load(calibration_file)

        camera_matrix = data.get("camera_matrix", data.get("cameraMatrix"))
        dist_coeffs = data.get(
            "dist_coeffs",
            data.get("distCoeffs", data.get("distortion_coefficients")),
        )
        if (
            not isinstance(camera_matrix, list)
            or len(camera_matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in camera_matrix)
        ):
            raise ValueError("camera_matrix must be a 3x3 array")
        if (
            isinstance(dist_coeffs, list)
            and len(dist_coeffs) == 1
            and isinstance(dist_coeffs[0], list)
        ):
            dist_coeffs = dist_coeffs[0]
        if not isinstance(dist_coeffs, list) or len(dist_coeffs) < 4:
            raise ValueError("dist_coeffs must contain at least four values")

        matrix_array = np.asarray(camera_matrix, dtype=np.float64)
        distortion_array = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        if matrix_array.shape != (3, 3) or not np.all(np.isfinite(matrix_array)):
            raise ValueError("camera_matrix contains invalid values")
        if not np.all(np.isfinite(distortion_array)):
            raise ValueError("dist_coeffs contains invalid values")

        current_config = dict(PERSISTENT_ROIS.get("calibration_config") or {})
        current_config["camera_matrix"] = matrix_array.tolist()
        current_config["dist_coeffs"] = distortion_array.tolist()
        PERSISTENT_ROIS["calibration_config"] = normalize_metric_calibration_config(
            current_config
        )

        image_size = data.get("image_size")
        size_text = ""
        if isinstance(image_size, list) and len(image_size) == 2:
            size_text = f" ({image_size[0]}x{image_size[1]})"
        print(f"Loaded camera calibration: {calibration_path}{size_text}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load camera calibration JSON {calibration_path}: {exc}")
        return False


def save_persistent_rois() -> None:
    try:
        with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(PERSISTENT_ROIS, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to save persistent ROIs: {exc}")


def stone_settings_for_state(state: dict[str, Any] | None = None) -> dict[str, Any]:
    source = (state or {}).get("source") or {}
    return {
        "color_correction": stone_detection.normalize_color_correction(
            source.get("color_correction") or PERSISTENT_ROIS.get("color_correction")
        ),
        "analysis_normalization": stone_detection.normalize_analysis_normalization(
            source.get("analysis_normalization")
            or PERSISTENT_ROIS.get("analysis_normalization")
        ),
        "background_calibration": copy.deepcopy(
            source.get("background_calibration")
            if source.get("background_calibration") is not None
            else PERSISTENT_ROIS.get("background_calibration")
        ),
        "learned_stone_profiles": stone_detection.normalize_learned_stone_profiles(
            source.get("learned_stone_profiles")
            if source.get("learned_stone_profiles") is not None
            else PERSISTENT_ROIS.get("learned_stone_profiles")
        ),
        "stone_super_resolution": normalize_stone_super_resolution_settings(
            source.get("stone_super_resolution")
            if source.get("stone_super_resolution") is not None
            else PERSISTENT_ROIS.get("stone_super_resolution")
        ),
        "calibration_config": normalize_metric_calibration_config(
            source.get("calibration_config")
            or PERSISTENT_ROIS.get("calibration_config")
        ),
    }


def purity_audio_settings() -> dict[str, Any]:
    return normalize_purity_audio_settings(PERSISTENT_ROIS.get("audio_settings"))


def calibrate_background_sample(
    image_bgr: np.ndarray,
    point: dict[str, Any],
    radius: int = 8,
) -> dict[str, Any]:
    height, width = image_bgr.shape[:2]
    x = max(0, min(width - 1, int(round(float(point.get("x", 0))))))
    y = max(0, min(height - 1, int(round(float(point.get("y", 0))))))
    radius = max(3, min(24, int(radius)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("Could not sample the selected background point.")

    hsv_pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab_pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    bgr_pixels = patch.reshape(-1, 3).astype(np.float32)

    hsv_center = [
        float(stone_detection.circular_hue_mean(hsv_pixels[:, 0])),
        float(np.median(hsv_pixels[:, 1])),
        float(np.median(hsv_pixels[:, 2])),
    ]
    hue_deviation = stone_detection.hue_distance(hsv_pixels[:, 0], hsv_center[0])
    hsv_deviation = [
        hue_deviation,
        np.abs(hsv_pixels[:, 1] - hsv_center[1]),
        np.abs(hsv_pixels[:, 2] - hsv_center[2]),
    ]
    hsv_tolerance = [
        min(35.0, max(4.0, float(np.percentile(hsv_deviation[0], 95)) + 3.0)),
        min(90.0, max(14.0, float(np.percentile(hsv_deviation[1], 95)) + 8.0)),
        min(90.0, max(12.0, float(np.percentile(hsv_deviation[2], 95)) + 8.0)),
    ]

    lab_center_array = np.median(lab_pixels, axis=0)
    lab_tolerance = [
        min(60.0, max(10.0, float(np.percentile(np.abs(lab_pixels[:, channel] - lab_center_array[channel]), 95)) + 6.0))
        for channel in range(3)
    ]
    bgr_center = np.median(bgr_pixels, axis=0)
    return {
        "point": {"x": x, "y": y},
        "sample_radius": radius,
        "sample_count": int(hsv_pixels.shape[0]),
        "hsv_center": [round(value, 2) for value in hsv_center],
        "hsv_tolerance": [round(value, 2) for value in hsv_tolerance],
        "lab_center": [round(float(value), 2) for value in lab_center_array],
        "lab_tolerance": [round(float(value), 2) for value in lab_tolerance],
        "bgr_center": [round(float(value), 2) for value in bgr_center],
        "low_saturation": bool(hsv_center[1] <= 40.0),
        "sampled_at": now_stamp(),
    }


def calibrate_learned_stone_sample(
    image_bgr: np.ndarray,
    point: dict[str, Any],
    expected_color: str,
    label: str,
    radius: int = 8,
    analysis_normalization: dict | None = None,
    background_calibration: dict | None = None,
) -> dict[str, Any]:
    if expected_color not in stone_detection.GEMSTONE_OPTIONS:
        raise ValueError("Select a valid expected gemstone color.")

    normalized_bgr = stone_detection.build_normalized_analysis_image(
        image_bgr,
        settings=analysis_normalization,
        background_calibration=background_calibration,
    )
    height, width = normalized_bgr.shape[:2]
    x = max(0, min(width - 1, int(round(float(point.get("x", 0))))))
    y = max(0, min(height - 1, int(round(float(point.get("y", 0))))))
    radius = max(4, min(24, int(radius)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = normalized_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("Could not sample the selected gemstone point.")

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    local_x = x - x0
    local_y = y - y0
    seed_radius = min(2, radius)
    sx0, sx1 = max(0, local_x - seed_radius), min(patch.shape[1], local_x + seed_radius + 1)
    sy0, sy1 = max(0, local_y - seed_radius), min(patch.shape[0], local_y + seed_radius + 1)
    seed_hsv = hsv_patch[sy0:sy1, sx0:sx1].reshape(-1, 3).astype(np.float32)
    seed_center = [
        float(stone_detection.circular_hue_mean(seed_hsv[:, 0])),
        float(np.median(seed_hsv[:, 1])),
        float(np.median(seed_hsv[:, 2])),
    ]

    hsv_pixels = hsv_patch.reshape(-1, 3).astype(np.float32)
    hue_delta = stone_detection.hue_distance(hsv_pixels[:, 0], seed_center[0])
    expected_selection = np.zeros(hsv_pixels.shape[0], dtype=bool)
    if expected_color in stone_detection.HSV_COLOR_RANGES:
        for lower, upper in stone_detection.HSV_COLOR_RANGES[expected_color]:
            lower_h, lower_s, lower_v = (float(value) for value in lower)
            upper_h, upper_s, upper_v = (float(value) for value in upper)
            expected_selection |= (
                (hsv_pixels[:, 0] >= lower_h)
                & (hsv_pixels[:, 0] <= upper_h)
                & (hsv_pixels[:, 1] >= max(18.0, lower_s * 0.45))
                & (hsv_pixels[:, 1] <= upper_s)
                & (hsv_pixels[:, 2] >= max(20.0, lower_v - 25.0))
                & (hsv_pixels[:, 2] <= upper_v)
            )
    if expected_color == "Red":
        expected_selection |= (
            ((hsv_pixels[:, 0] <= 16.0) | (hsv_pixels[:, 0] >= 162.0))
            & (hsv_pixels[:, 1] >= 35.0)
            & (hsv_pixels[:, 2] >= 30.0)
        )

    if int(np.count_nonzero(expected_selection)) >= 6:
        selected = expected_selection
    elif expected_color in {"White/Colorless", "Black"}:
        selected = (
            (np.abs(hsv_pixels[:, 1] - seed_center[1]) <= 55.0)
            & (np.abs(hsv_pixels[:, 2] - seed_center[2]) <= 80.0)
        )
    else:
        selected = (
            (hue_delta <= 16.0)
            & (hsv_pixels[:, 1] >= max(18.0, seed_center[1] * 0.38))
            & (np.abs(hsv_pixels[:, 2] - seed_center[2]) <= 95.0)
        )
    if int(np.count_nonzero(selected)) < 6:
        selected = np.ones(hsv_pixels.shape[0], dtype=bool)

    selected_mask = selected.reshape(hsv_patch.shape[:2]).astype(np.uint8)
    component_count, component_labels, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(selected_mask, connectivity=8)
    )
    target_component = int(component_labels[local_y, local_x])
    if target_component <= 0 and component_count > 1:
        valid_labels = [
            component_index
            for component_index in range(1, component_count)
            if int(component_stats[component_index, cv2.CC_STAT_AREA]) >= 3
        ]
        if valid_labels:
            target_component = min(
                valid_labels,
                key=lambda component_index: (
                    float(component_centroids[component_index][0]) - local_x
                ) ** 2
                + (
                    float(component_centroids[component_index][1]) - local_y
                ) ** 2,
            )
    if target_component > 0:
        selected = component_labels.reshape(-1) == target_component
    if int(np.count_nonzero(selected)) < 4:
        raise ValueError(
            "The clicked point did not contain a stable gemstone-color region. "
            "Click nearer the center of the missed stone."
        )

    selected_hsv = hsv_pixels[selected]
    selected_lab = lab_patch.reshape(-1, 3).astype(np.float32)[selected]
    hsv_center = [
        float(stone_detection.circular_hue_mean(selected_hsv[:, 0])),
        float(np.median(selected_hsv[:, 1])),
        float(np.median(selected_hsv[:, 2])),
    ]
    hsv_tolerance = [
        min(
            24.0,
            max(
                3.0,
                float(
                    np.percentile(
                        stone_detection.hue_distance(
                            selected_hsv[:, 0],
                            hsv_center[0],
                        ),
                        85,
                    )
                )
                + 2.0,
            ),
        ),
        min(
            110.0,
            max(
                16.0,
                float(
                    np.percentile(
                        np.abs(selected_hsv[:, 1] - hsv_center[1]),
                        85,
                    )
                )
                + 6.0,
            ),
        ),
        min(
            110.0,
            max(
                16.0,
                float(
                    np.percentile(
                        np.abs(selected_hsv[:, 2] - hsv_center[2]),
                        85,
                    )
                )
                + 6.0,
            ),
        ),
    ]
    hsv_tolerance[0] = min(hsv_tolerance[0], 10.0)
    hsv_tolerance[1] = min(hsv_tolerance[1], 45.0)
    hsv_tolerance[2] = min(hsv_tolerance[2], 50.0)
    lab_center = np.median(selected_lab, axis=0)
    lab_tolerance = [
        min(
            70.0,
            max(
                8.0,
                float(
                    np.percentile(
                        np.abs(selected_lab[:, channel] - lab_center[channel]),
                        85,
                    )
                )
                + 5.0,
            ),
        )
        for channel in range(3)
    ]
    lab_tolerance = [min(value, 32.0) for value in lab_tolerance]
    profile = {
        "id": uuid.uuid4().hex,
        "label": (str(label or "").strip() or f"Learned {expected_color}")[:80],
        "color": expected_color,
        "point": {"x": x, "y": y},
        "sample_radius": radius,
        "sample_count": int(selected_hsv.shape[0]),
        "sample_component_area_px": int(selected_hsv.shape[0]),
        "hsv_center": [round(value, 2) for value in hsv_center],
        "hsv_tolerance": [round(value, 2) for value in hsv_tolerance],
        "lab_center": [round(float(value), 2) for value in lab_center],
        "lab_tolerance": [round(float(value), 2) for value in lab_tolerance],
        "sampled_at": now_stamp(),
    }
    normalized = stone_detection.normalize_learned_stone_profiles([profile])
    if not normalized:
        raise ValueError("The selected point could not produce a valid color profile.")
    return {**profile, **normalized[0]}


def _is_windows() -> bool:
    return os.name == "nt"


def _camera_index_from_device(device: str) -> int:
    raw = str(device or "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except Exception:
        pass

    lowered = raw.lower()
    if lowered.startswith("/dev/video"):
        suffix = raw[len("/dev/video") :]
        try:
            return int(suffix)
        except Exception:
            pass
    raise ValueError(f"Unsupported camera device value: {device!r}. Use a numeric index like 0 or /dev/video0.")


CAM_INDEX = _camera_index_from_device(CAM_DEVICE)


def _read_camera_resolution(cap: cv2.VideoCapture) -> tuple[int, int]:
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    except Exception:
        return (0, 0)


def _display_camera_resolution(width: int, height: int) -> tuple[int, int]:
    if CAM_ROTATE_90_CLOCKWISE and width > 0 and height > 0:
        return (height, width)
    return (width, height)


def _transform_camera_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    if CAM_ROTATE_90_CLOCKWISE:
        try:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        except Exception:
            pass
    return frame


def _bounded_exposure_absolute(value: int | float) -> int:
    lower = min(EXPOSURE_MIN_ABSOLUTE, EXPOSURE_MAX_ABSOLUTE)
    upper = max(EXPOSURE_MIN_ABSOLUTE, EXPOSURE_MAX_ABSOLUTE)
    try:
        numeric = int(round(float(value)))
    except Exception:
        numeric = denominator_to_exposure_absolute(SHUTTER_DENOMINATOR)
    return max(lower, min(upper, numeric))


def _powerline_aligned_exposure_absolute(value: int | float, hz: int = POWER_LINE_HZ) -> int:
    bounded = _bounded_exposure_absolute(value)
    try:
        step = max(1, int(hz))
    except Exception:
        return bounded

    max_denom = max(step, int(math.ceil(10000 / max(1, EXPOSURE_MIN_ABSOLUTE))) + step)
    candidates: set[int] = set()
    for denominator in range(step, max_denom + 1, step):
        candidate = _bounded_exposure_absolute(denominator_to_exposure_absolute(denominator))
        if EXPOSURE_MIN_ABSOLUTE <= candidate <= EXPOSURE_MAX_ABSOLUTE:
            candidates.add(candidate)

    if not candidates:
        return bounded
    return min(candidates, key=lambda candidate: (abs(candidate - bounded), candidate))


def _exposure_luma_stats(frame: np.ndarray) -> dict[str, float] | None:
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None

    roi_fraction = EXPOSURE_CENTER_ROI_FRACTION
    roi_w = max(1, int(round(width * roi_fraction)))
    roi_h = max(1, int(round(height * roi_fraction)))
    x0 = max(0, (width - roi_w) // 2)
    y0 = max(0, (height - roi_h) // 2)
    roi = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]

    try:
        if roi.ndim == 3 and roi.shape[2] >= 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
    except Exception:
        return None

    pixel_count = int(gray.size)
    if pixel_count <= 0:
        return None

    if pixel_count > EXPOSURE_STATS_MAX_PIXELS:
        scale = math.sqrt(EXPOSURE_STATS_MAX_PIXELS / pixel_count)
        sample_w = max(1, int(round(gray.shape[1] * scale)))
        sample_h = max(1, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (sample_w, sample_h), interpolation=cv2.INTER_AREA)

    values = gray.reshape(-1).astype(np.float32)
    target_luma = float(np.percentile(values, EXPOSURE_TARGET_PERCENTILE))
    highlight_luma = float(np.percentile(values, EXPOSURE_HIGHLIGHT_PERCENTILE))
    clipped_fraction = float(np.mean(values >= 250.0))
    return {
        "target_luma": target_luma,
        "highlight_luma": highlight_luma,
        "mean_luma": float(np.mean(values)),
        "clipped_fraction": clipped_fraction,
    }


def _request_camera_resolution(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    warmup_grabs: int = 0,
) -> tuple[int, int]:
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


def _best_effort_set_camera_resolution(
    cap: cv2.VideoCapture,
    target_width: int = CAM_TARGET_WIDTH,
    target_height: int = CAM_TARGET_HEIGHT,
) -> tuple[int, int]:
    target = (max(1, int(target_width)), max(1, int(target_height)))
    candidates = list(dict.fromkeys([
        target,
        (2560, 1440),
        (2048, 1080),
        (2048, 1536),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]))

    best = _read_camera_resolution(cap)
    best_score = float("inf")
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

    for index, (width, height) in enumerate(candidates):
        warmup = CAM_PROBE_GRABS if index == 0 else 1
        rw, rh = _request_camera_resolution(cap, width, height, warmup_grabs=warmup)
        if rw > 0 and rh > 0:
            score = abs(rw - target[0]) + abs(rh - target[1])
            if score < best_score:
                best = (rw, rh)
                best_score = score
            if abs(rw - target[0]) <= 16 and abs(rh - target[1]) <= 16:
                break

    if best[0] <= 0 or best[1] <= 0:
        best = _read_camera_resolution(cap)
    elif best[0] > 0 and best[1] > 0:
        best = _request_camera_resolution(cap, best[0], best[1], warmup_grabs=1)
    return best


# ── v4l2 helpers for exposure configuration (adapted from shutterspeedset_robust.py) ──

def _v4l2_ctl(args: list[str]) -> tuple[bool, str]:
    """Run v4l2-ctl and return (success, message)."""
    try:
        r = subprocess.run(
            ["v4l2-ctl"] + args,
            capture_output=True, text=True, check=True
        )
        return True, r.stdout.strip()
    except FileNotFoundError:
        return False, "v4l2-ctl not found – install v4l-utils"
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        return False, msg or "v4l2-ctl returned non-zero"


def _v4l2_get(device_path: str, control: str) -> str | None:
    """Read back a single v4l2 control value (returns None on failure)."""
    ok, out = _v4l2_ctl(["-d", device_path, "--get-ctrl", control])
    if ok and ":" in out:
        return out.split(":", 1)[1].strip()
    return None


def _v4l2_set(device_path: str, control: str, value) -> bool:
    """Set a v4l2 control.  Returns True on success."""
    ok, msg = _v4l2_ctl(["-d", device_path, "-c", f"{control}={value}"])
    if not ok:
        print(f"  ⚠  v4l2 {control}={value} on {device_path} → {msg}")
    return ok


def _v4l2_control_names(device_path: str) -> set[str]:
    """Return the control names advertised by the active camera driver."""
    for args in (
        ["-d", device_path, "--list-ctrls-menus"],
        ["-d", device_path, "--list-ctrls"],
    ):
        ok, out = _v4l2_ctl(args)
        if not ok:
            continue
        controls: set[str] = set()
        for line in out.splitlines():
            match = re.match(r"^\s*([A-Za-z0-9_]+)\s+0x[0-9A-Fa-f]+\s+\(", line)
            if match:
                controls.add(match.group(1))
        if controls:
            return controls
    return set()


def _first_supported_control(controls: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in controls:
            return candidate
    return None


def _control_int(device_path: str, control: str | None) -> int | None:
    if not control:
        return None
    raw = _v4l2_get(device_path, control)
    if raw is None:
        return None
    match = re.search(r"-?\d+", raw)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


class CameraBackend:
    def __init__(self, device: str):
        self.device = str(device)
        self._camera_index = CAM_INDEX if self.device == CAM_DEVICE else _camera_index_from_device(self.device)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_counter = 0
        self._last_frame_at = 0.0
        self._status = f"Camera idle on index {self._camera_index}."
        self._last_error = ""
        self._opened_resolution: tuple[int, int] = (0, 0)
        self._frame_resolution: tuple[int, int] = (0, 0)
        self._requested_resolution: tuple[int, int] = (CAM_TARGET_WIDTH, CAM_TARGET_HEIGHT)
        self._output_resolution: tuple[int, int] | None = None
        self._device_path = f"/dev/video{self._camera_index}"
        self._startup_calibration_enabled = bool(
            CAMERA_STARTUP_EXPOSURE_CALIBRATION
            and CAMERA_EXPOSURE_MODE not in {"native_auto", "auto", "camera_auto"}
            and not _is_windows()
        )
        self._startup_calibrated = False
        self._current_exposure_abs = _powerline_aligned_exposure_absolute(
            denominator_to_exposure_absolute(SHUTTER_DENOMINATOR),
            POWER_LINE_HZ,
        )
        self._last_exposure_stats: dict[str, float] = {}
        self._last_exposure_message = ""
        self._exposure_confirmed = False
        self._controls: set[str] = set()
        self._auto_exposure_control: str | None = None
        self._exposure_control: str | None = None
        self._auto_gain_control: str | None = None
        self._gain_control: str | None = None
        self._autofocus_enabled = False
        self._focus_frame_count = 0
        self._focus_scores: deque[float] = deque(maxlen=CAMERA_FOCUS_STABILITY_FRAMES)
        self._focus_roi: dict[str, int] | None = None
        self._focus_score = 0.0
        self._focus_relative_spread: float | None = None
        self._focus_ready = False
        self._focus_status = "Draw the Processing ROI before capture."

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.snapshot()
            self._status = f"Opening camera index {self._camera_index}..."
            self._last_error = ""
            self._stop_event.clear()
            try:
                self._cap = self._open_capture()
                requested_width, requested_height = self._requested_resolution
                opened = _best_effort_set_camera_resolution(
                    self._cap,
                    requested_width,
                    requested_height,
                )
                try:
                    self._autofocus_enabled = bool(
                        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    )
                except Exception:
                    self._autofocus_enabled = False
                self._focus_frame_count = 0
                self._focus_scores.clear()
                self._focus_roi = None
                self._focus_score = 0.0
                self._focus_relative_spread = None
                self._focus_ready = False
                self._focus_status = "Draw the Processing ROI before capture."
                # Apply exposure AFTER resolution/FPS — those can reset exposure to auto
                if not _is_windows():
                    try:
                        self.configure_exposure()
                    except Exception as exc:
                        print(f"  ⚠  Camera exposure configuration failed (non-fatal): {exc}")
                self._opened_resolution = _display_camera_resolution(*opened)
                width, height = self._opened_resolution
                if width > 0 and height > 0:
                    self._status = f"Camera streaming from index {self._camera_index} at {width}x{height}."
                else:
                    self._status = f"Camera streaming from index {self._camera_index}."
                self._thread = threading.Thread(target=self._camera_loop, name="rpi-camera-loop", daemon=True)
                self._thread.start()
            except Exception as exc:  # noqa: BLE001
                self._cap = None
                self._thread = None
                self._last_error = str(exc)
                self._status = f"Camera unavailable on index {self._camera_index}: {exc}"
        return self.snapshot()

    def stop(self) -> None:
        thread: threading.Thread | None = None
        cap: cv2.VideoCapture | None = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._status = f"Camera stopped on index {self._camera_index}."

    def set_resolution(self, width: int, height: int) -> dict[str, Any]:
        target = (max(1, int(width)), max(1, int(height)))
        with self._lock:
            already_running = bool(self._thread is not None and self._thread.is_alive())
            already_requested = self._requested_resolution == target
        if already_running and already_requested:
            return self.snapshot()

        if already_running:
            self.stop()
        with self._lock:
            self._requested_resolution = target
            self._opened_resolution = (0, 0)
            self._frame_resolution = (0, 0)
            self._latest_frame = None

        status = self.start()
        if not status.get("running"):
            raise RuntimeError(
                status.get("last_error")
                or f"Could not start camera at {target[0]}x{target[1]}."
            )
        return status

    def set_output_resolution(self, width: int | None, height: int | None) -> dict[str, Any]:
        with self._lock:
            if width is None or height is None:
                self._output_resolution = None
            else:
                self._output_resolution = (max(1, int(width)), max(1, int(height)))
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            capture_width, capture_height = self._frame_resolution
            if capture_width <= 0 or capture_height <= 0:
                capture_width, capture_height = self._opened_resolution
            if self._output_resolution is not None:
                width, height = self._output_resolution
            else:
                width, height = capture_width, capture_height
            frame_age_s = (
                max(0.0, time.monotonic() - self._last_frame_at)
                if self._last_frame_at > 0.0
                else None
            )
            return {
                "device": self.device,
                "camera_index": int(self._camera_index),
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "status": self._status,
                "last_error": self._last_error,
                "width": int(width),
                "height": int(height),
                "capture_width": int(capture_width),
                "capture_height": int(capture_height),
                "requested_width": int(self._requested_resolution[0]),
                "requested_height": int(self._requested_resolution[1]),
                "output_width": int(self._output_resolution[0]) if self._output_resolution else 0,
                "output_height": int(self._output_resolution[1]) if self._output_resolution else 0,
                "has_frame": self._latest_frame is not None,
                "frame_counter": int(self._frame_counter),
                "frame_age_s": frame_age_s,
                "rotate_90_clockwise": CAM_ROTATE_90_CLOCKWISE,
                "adaptive_exposure": False,
                "exposure_mode": CAMERA_EXPOSURE_MODE,
                "startup_exposure_calibration": bool(self._startup_calibration_enabled),
                "startup_calibrated": bool(self._startup_calibrated),
                "exposure_absolute": int(self._current_exposure_abs or 0),
                "exposure_confirmed": bool(self._exposure_confirmed),
                "exposure_stats": dict(self._last_exposure_stats),
                "exposure_status": self._last_exposure_message,
                "focus": self._focus_payload_unlocked(),
            }

    def _focus_payload_unlocked(self) -> dict[str, Any]:
        return {
            "ready": bool(self._focus_ready),
            "status": self._focus_status,
            "score": float(self._focus_score),
            "relative_spread": self._focus_relative_spread,
            "sample_count": len(self._focus_scores),
            "required_samples": CAMERA_FOCUS_STABILITY_FRAMES,
            "warmup_frames": min(self._focus_frame_count, CAMERA_FOCUS_WARMUP_FRAMES),
            "required_warmup_frames": CAMERA_FOCUS_WARMUP_FRAMES,
            "roi": copy.deepcopy(self._focus_roi),
            "autofocus_enabled": bool(self._autofocus_enabled),
        }

    def focus_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._focus_payload_unlocked()

    def _record_focus_unlocked(
        self,
        roi: dict[str, int] | None,
        score: float | None,
    ) -> None:
        self._focus_frame_count += 1
        if roi != self._focus_roi:
            self._focus_roi = copy.deepcopy(roi)
            self._focus_scores.clear()
            self._focus_ready = False
            self._focus_relative_spread = None

        if roi is None or score is None:
            self._focus_score = 0.0
            self._focus_ready = False
            self._focus_status = "Draw the Processing ROI before capture."
            return

        self._focus_score = float(score)
        self._focus_scores.append(float(score))
        if self._focus_frame_count < CAMERA_FOCUS_WARMUP_FRAMES:
            self._focus_ready = False
            self._focus_status = "Camera autofocus is warming up."
            return
        if len(self._focus_scores) < CAMERA_FOCUS_STABILITY_FRAMES:
            self._focus_ready = False
            self._focus_status = "Camera focus is stabilizing."
            return

        mean_score = float(np.mean(self._focus_scores))
        spread = (
            float(max(self._focus_scores) - min(self._focus_scores)) / mean_score
            if mean_score > 0.0
            else float("inf")
        )
        self._focus_relative_spread = spread
        self._focus_ready = bool(
            math.isfinite(spread) and spread <= CAMERA_FOCUS_MAX_RELATIVE_SPREAD
        )
        self._focus_status = (
            "Camera focus is ready."
            if self._focus_ready
            else "Camera autofocus is adjusting."
        )

    def configure_exposure(
        self,
        denominator: int = SHUTTER_DENOMINATOR,
        hz: int = POWER_LINE_HZ,
        flush_n: int = EXPOSURE_FLUSH_FRAMES,
    ) -> bool:
        """Calibrate exposure once during startup, then leave it fixed."""
        device_path = self._device_path
        cap = self._cap
        print(f"\n-- Camera Startup Calibration (device={device_path}) --")

        self._controls = _v4l2_control_names(device_path)
        self._auto_exposure_control = _first_supported_control(
            self._controls,
            ("auto_exposure", "exposure_auto"),
        )
        self._exposure_control = _first_supported_control(
            self._controls,
            ("exposure_time_absolute", "exposure_absolute"),
        )
        self._auto_gain_control = _first_supported_control(
            self._controls,
            ("gain_automatic", "auto_gain"),
        )
        self._gain_control = _first_supported_control(
            self._controls,
            ("gain",),
        )

        relevant_controls = sorted(
            control
            for control in self._controls
            if any(
                token in control
                for token in ("exposure", "gain", "power_line")
            )
        )
        if relevant_controls:
            print(f"  Camera controls: {', '.join(relevant_controls)}")
        else:
            print("  Camera did not report exposure-related V4L2 controls.")

        aligned = align_to_powerline(denominator, hz)
        baseline_abs = _powerline_aligned_exposure_absolute(
            denominator_to_exposure_absolute(aligned),
            hz,
        )
        self._current_exposure_abs = baseline_abs

        print(f"\n  Step 1: Set anti-flicker frequency to {hz} Hz")
        plf_value = 1 if hz == 50 else 2
        ok_plf = (
            _v4l2_set(device_path, "power_line_frequency", plf_value)
            if not self._controls or "power_line_frequency" in self._controls
            else False
        )
        if not ok_plf:
            print("    power_line_frequency is not available.")
        else:
            plf_readback = _v4l2_get(device_path, "power_line_frequency")
            print(f"    power_line_frequency read-back: {plf_readback}")

        if cap is None:
            self._last_exposure_message = "Startup calibration skipped: camera is not open."
            return False

        if CAMERA_EXPOSURE_MODE in {"native_auto", "auto", "camera_auto"}:
            return self._configure_native_auto(cap, flush_n)

        print("\n  Step 2: Let auto exposure settle")
        auto_started = self._set_auto_exposure_mode(True)
        for _ in range(EXPOSURE_STARTUP_SETTLE_FRAMES):
            try:
                cap.read()
            except Exception:
                break
        settled_abs = self._read_exposure_absolute()
        if settled_abs is None:
            settled_abs = baseline_abs
        settled_abs = _bounded_exposure_absolute(settled_abs)
        print(
            f"    settled exposure_absolute={settled_abs}; "
            f"auto mode {'enabled' if auto_started else 'not confirmed'}"
        )

        print("\n  Step 3: Lock exposure and gain")
        manual_locked = self._set_auto_exposure_mode(False)
        exposure_applied = self._write_exposure_absolute(settled_abs)
        if CAMERA_DISABLE_AUTO_GAIN and self._auto_gain_control:
            _v4l2_set(device_path, self._auto_gain_control, 0)
        print(
            f"    manual exposure {'locked' if manual_locked else 'not confirmed'}; "
            f"{self._exposure_control or 'OpenCV exposure fallback'}={settled_abs}"
        )

        can_calibrate_manually = bool(manual_locked and exposure_applied)
        if self._startup_calibration_enabled and can_calibrate_manually:
            print(
                f"\n  Step 4: One-time luma calibration "
                f"(target={EXPOSURE_TARGET_LUMA:.0f})"
            )
            current_abs = settled_abs
            for adjustment in range(EXPOSURE_STARTUP_MAX_ADJUSTMENTS + 1):
                stats = self._sample_exposure_stats(cap, EXPOSURE_STARTUP_SAMPLE_FRAMES)
                if not stats:
                    break

                stats["current_exposure_abs"] = float(current_abs)
                self._last_exposure_stats = stats
                target_luma = stats["target_luma"]
                highlight_luma = stats["highlight_luma"]
                clipped_fraction = stats["clipped_fraction"]
                highlights_ok = (
                    highlight_luma < EXPOSURE_HIGHLIGHT_LIMIT
                    and clipped_fraction < 0.01
                )
                luma_ok = abs(target_luma - EXPOSURE_TARGET_LUMA) <= EXPOSURE_DEADBAND_LUMA

                print(
                    f"    sample {adjustment + 1}: luma={target_luma:.1f}, "
                    f"highlight={highlight_luma:.1f}, clipped={clipped_fraction:.3f}, "
                    f"exposure={current_abs}"
                )
                if luma_ok and highlights_ok:
                    break
                if adjustment >= EXPOSURE_STARTUP_MAX_ADJUSTMENTS:
                    break

                ratio = EXPOSURE_TARGET_LUMA / max(1.0, target_luma)
                if not highlights_ok:
                    ratio = min(
                        ratio,
                        EXPOSURE_HIGHLIGHT_LIMIT / max(1.0, highlight_luma),
                        0.85,
                    )
                desired_abs = current_abs * max(0.25, min(2.0, ratio))
                max_step = max(1.0, current_abs * EXPOSURE_MAX_STEP_FRACTION)
                desired_abs = max(
                    current_abs - max_step,
                    min(current_abs + max_step, desired_abs),
                )
                next_abs = _bounded_exposure_absolute(desired_abs)
                if next_abs == current_abs:
                    break
                if not self._write_exposure_absolute(next_abs):
                    break
                current_abs = next_abs
                for _ in range(EXPOSURE_STARTUP_FLUSH_FRAMES):
                    cap.read()
        elif self._startup_calibration_enabled:
            print(
                "\n  Step 4: One-time luma calibration skipped because "
                "manual exposure could not be locked"
            )

        print("\n  Step 5: Leave camera color controls unchanged")

        print(f"\n  Step 6: Flush {flush_n} startup frames")
        for _ in range(flush_n):
            cap.read()

        self._startup_calibrated = bool(
            manual_locked and (exposure_applied or self._exposure_confirmed)
        )
        if self._startup_calibrated:
            self._last_exposure_message = (
                f"startup exposure fixed at exposure_absolute={self._current_exposure_abs}; "
                "white balance/color controls unchanged; no live exposure updates"
            )
        else:
            self._last_exposure_message = (
                "startup exposure could not be confirmed; white balance/color controls unchanged; no live exposure updates"
            )

        print(f"  Final exposure_absolute: {self._current_exposure_abs}")
        print("  Live exposure updates: disabled")
        print("-- Camera startup calibration complete --\n")
        return bool(self._startup_calibrated)

    def _configure_native_auto(
        self,
        cap: cv2.VideoCapture,
        flush_n: int,
    ) -> bool:
        """Use the WB3023's hardware auto exposure without application color tuning."""
        print("\n  Step 2: Apply Dell WB3023 native exposure controls")
        exposure_auto = self._set_auto_exposure_mode(True)
        if self._auto_gain_control:
            _v4l2_set(self._device_path, self._auto_gain_control, 0)
        if self._gain_control:
            _v4l2_set(self._device_path, self._gain_control, CAMERA_NATIVE_GAIN)
        print(
            f"    exposure mode: {'camera auto' if exposure_auto else 'camera default'}"
        )
        print(
            f"    gain={CAMERA_NATIVE_GAIN}; white balance/color controls unchanged"
        )

        print(
            f"\n  Step 3: Let camera exposure settle "
            f"({EXPOSURE_STARTUP_SETTLE_FRAMES} frames)"
        )
        for _ in range(EXPOSURE_STARTUP_SETTLE_FRAMES):
            try:
                cap.read()
            except Exception:
                break

        exposure_value = self._read_exposure_absolute()
        if exposure_value is not None:
            self._current_exposure_abs = exposure_value
        stats = self._sample_exposure_stats(cap, EXPOSURE_STARTUP_SAMPLE_FRAMES)
        if stats:
            stats["current_exposure_abs"] = float(self._current_exposure_abs)
            self._last_exposure_stats = stats
            print(
                f"    settled luma={stats['target_luma']:.1f}, "
                f"highlight={stats['highlight_luma']:.1f}, "
                f"clipped={stats['clipped_fraction']:.3f}"
            )
        print(
            f"    exposure={exposure_value if exposure_value is not None else 'camera managed'}"
        )

        print(f"\n  Step 4: Flush {flush_n} startup frames")
        for _ in range(flush_n):
            cap.read()

        self._startup_calibrated = bool(exposure_auto)
        self._exposure_confirmed = bool(exposure_auto)
        self._last_exposure_message = (
            "WB3023 native auto exposure active; "
            "application performs no white balance/color-control writes"
        )
        print("  Exposure control: WB3023 hardware auto")
        print("  Application white balance/color-control writes: disabled")
        print("-- Camera native auto setup complete --\n")
        return True

    def _set_auto_exposure_mode(self, automatic: bool) -> bool:
        control = self._auto_exposure_control
        if control:
            values = (0, 3) if automatic else (1,)
            for value in values:
                if not _v4l2_set(self._device_path, control, value):
                    continue
                time.sleep(0.05)
                readback = _control_int(self._device_path, control)
                if readback is None or readback == value:
                    return True

        cap = self._cap
        if cap is None:
            return False
        try:
            # V4L2/OpenCV convention: 0.75 = auto, 0.25 = manual.
            return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if automatic else 0.25))
        except Exception:
            return False

    def _read_exposure_absolute(self) -> int | None:
        value = _control_int(self._device_path, self._exposure_control)
        if value is not None and value > 0:
            return value

        cap = self._cap
        if cap is None:
            return None
        try:
            raw = float(cap.get(cv2.CAP_PROP_EXPOSURE))
            if math.isfinite(raw) and raw > 0:
                return int(round(raw))
        except Exception:
            pass
        return None

    def _write_exposure_absolute(self, exposure_abs: int | float) -> bool:
        target_abs = _bounded_exposure_absolute(exposure_abs)
        v4l2_ok = False
        if self._exposure_control:
            v4l2_ok = _v4l2_set(self._device_path, self._exposure_control, target_abs)

        cv_ok = False
        if not v4l2_ok and self._cap is not None:
            try:
                cv_ok = bool(self._cap.set(cv2.CAP_PROP_EXPOSURE, float(target_abs)))
            except Exception:
                pass

        readback = self._read_exposure_absolute()
        confirmed = readback == target_abs if readback is not None else False
        self._current_exposure_abs = target_abs
        self._exposure_confirmed = bool(confirmed or v4l2_ok or cv_ok)
        return self._exposure_confirmed

    @staticmethod
    def _sample_exposure_stats(
        cap: cv2.VideoCapture,
        frame_count: int,
    ) -> dict[str, float] | None:
        samples: list[dict[str, float]] = []
        for _ in range(max(1, frame_count)):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _transform_camera_frame(frame)
            stats = _exposure_luma_stats(frame)
            if stats:
                samples.append(stats)
        if not samples:
            return None
        return {
            key: float(np.median([sample[key] for sample in samples]))
            for key in samples[0]
        }

    def get_frame_copy(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
            output_resolution = self._output_resolution
        if output_resolution is not None and frame.shape[1::-1] != output_resolution:
            interpolation = (
                cv2.INTER_AREA
                if output_resolution[0] <= frame.shape[1] and output_resolution[1] <= frame.shape[0]
                else cv2.INTER_LINEAR
            )
            frame = cv2.resize(frame, output_resolution, interpolation=interpolation)
        return frame

    def get_jpeg_bytes(self, quality: int = CAM_PREVIEW_JPEG_QUALITY) -> bytes | None:
        frame = self.get_frame_copy()
        if frame is None:
            return None
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            return None
        return buffer.tobytes()

    def _open_capture(self) -> cv2.VideoCapture:
        if _is_windows():
            capture = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)

        if capture is None or not capture.isOpened():
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            raise RuntimeError(f"Could not open camera index {self._camera_index} with OpenCV.")

        return capture

    def _camera_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            # CPython GIL makes a single attribute read safe without a lock
            capture = self._cap
            if capture is None:
                break

            ok, frame = capture.read()
            frame = _transform_camera_frame(frame)
            if not ok or frame is None:
                consecutive_failures += 1
                self._last_error = f"Camera frame read failed ({consecutive_failures})."
                threading.Event().wait(0.02)
                continue

            consecutive_failures = 0
            frame_height, frame_width = frame.shape[:2]
            focus_roi = normalize_rect(
                PERSISTENT_ROIS.get("processing_roi"),
                frame_width,
                frame_height,
            )
            focus_score: float | None = None
            if focus_roi is not None:
                x = focus_roi["x"]
                y = focus_roi["y"]
                w = focus_roi["w"]
                h = focus_roi["h"]
                focus_crop = frame[y : y + h, x : x + w]
                if focus_crop.size:
                    gray = cv2.cvtColor(focus_crop, cv2.COLOR_BGR2GRAY)
                    focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            with self._lock:           # Only ONE lock acquisition per frame
                self._latest_frame = frame.copy()
                self._frame_resolution = (int(frame.shape[1]), int(frame.shape[0]))
                self._frame_counter += 1
                self._last_frame_at = time.monotonic()
                self._last_error = ""
                self._record_focus_unlocked(focus_roi, focus_score)
            threading.Event().wait(0.015)   # 15ms — matches riskanalyser.py


CAMERA_BACKEND: CameraBackend | None = None


def get_camera_backend() -> CameraBackend:
    global CAMERA_BACKEND
    if CAMERA_BACKEND is None:
        with CAMERA_LOCK:
            if CAMERA_BACKEND is None:
                CAMERA_BACKEND = CameraBackend(CAM_DEVICE)
    return CAMERA_BACKEND


def _start_purity_camera_mode() -> None:
    expected_width, expected_height = _display_camera_resolution(
        PURITY_CAM_WIDTH,
        PURITY_CAM_HEIGHT,
    )
    status = get_camera_backend().set_output_resolution(expected_width, expected_height)
    if not status.get("running") or not status.get("has_frame"):
        raise RuntimeError(
            status.get("last_error")
            or "Camera is not producing frames for the purity test."
        )
    actual_width = int(status.get("width", 0) or 0)
    actual_height = int(status.get("height", 0) or 0)
    if (
        abs(actual_width - expected_width) > 16
        or abs(actual_height - expected_height) > 16
    ):
        raise RuntimeError(
            "Camera did not enter 720p mode for the purity test: "
            f"requested {expected_width}x{expected_height}, "
            f"opened {actual_width}x{actual_height}."
        )
    print(
        "Purity test 720p output mode: "
        f"{actual_width}x{actual_height} "
        f"(capture remains {status.get('capture_width', 0)}x{status.get('capture_height', 0)}, "
        f"frame={status.get('frame_counter', 0)}, age={float(status.get('frame_age_s', 0.0) or 0.0):.3f}s)"
    )


def _stop_purity_camera_mode() -> None:
    status = get_camera_backend().set_output_resolution(None, None)
    print(
        "Normal camera output restored without restarting capture: "
        f"{status.get('width', 0)}x{status.get('height', 0)} "
        f"(capture {status.get('capture_width', 0)}x{status.get('capture_height', 0)})"
    )


def _purity_processing_roi_for_frame(
    frame: np.ndarray,
) -> dict[str, int] | None:
    """Scale the saved processing ROI into the current purity-frame size."""
    frame_height, frame_width = frame.shape[:2]
    source = CURRENT_STATE.get("source") if isinstance(CURRENT_STATE, dict) else None
    source = source if isinstance(source, dict) else {}

    if "processing_roi" in source:
        raw_roi = source.get("processing_roi")
    else:
        raw_roi = PERSISTENT_ROIS.get("processing_roi")
    if not raw_roi:
        return None

    image_size = source.get("image_size")
    if isinstance(image_size, dict):
        try:
            reference_width = int(image_size.get("width", 0) or 0)
            reference_height = int(image_size.get("height", 0) or 0)
        except (TypeError, ValueError):
            reference_width = 0
            reference_height = 0
    else:
        reference_width = 0
        reference_height = 0

    if reference_width <= 0 or reference_height <= 0:
        reference_width, reference_height = _display_camera_resolution(
            CAM_TARGET_WIDTH,
            CAM_TARGET_HEIGHT,
        )

    try:
        scaled_roi = {
            "x": float(raw_roi["x"]) * frame_width / reference_width,
            "y": float(raw_roi["y"]) * frame_height / reference_height,
            "w": float(raw_roi["w"]) * frame_width / reference_width,
            "h": float(raw_roi["h"]) * frame_height / reference_height,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return normalize_rect(scaled_roi, frame_width, frame_height)


def get_purity_camera_frame() -> np.ndarray | None:
    """Return only the processing ROI used by the purity HEF models."""
    frame = get_camera_backend().get_frame_copy()
    if frame is None:
        return None
    processing_roi = _purity_processing_roi_for_frame(frame)
    return crop_image(frame, processing_roi)


def get_purity_manager() -> PurityTestManager:
    global PURITY_MANAGER
    if PURITY_MANAGER is None:
        with MODEL_LOCK:
            if PURITY_MANAGER is None:
                PURITY_MANAGER = PurityTestManager(
                    base_dir=BASE_DIR,
                    frame_getter=get_purity_camera_frame,
                    speak_fn=speak,
                    session_start_fn=_start_purity_camera_mode,
                    session_stop_fn=_stop_purity_camera_mode,
                )
                PURITY_MANAGER.set_audio_ok_confidence_threshold(
                    purity_audio_settings()["ok_confidence_threshold"]
                )
        PURITY_MANAGER.set_audio_ok_confidence_threshold(
            purity_audio_settings()["ok_confidence_threshold"]
        )
    return PURITY_MANAGER


app = Flask(__name__, static_folder=str(WEBUI_DIR), static_url_path="/webui")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


try:
    STORAGE_CLEANUP_THRESHOLD_PERCENT = max(
        1.0,
        min(99.0, float(os.environ.get("STORAGE_CLEANUP_THRESHOLD_PERCENT", "90"))),
    )
except ValueError:
    STORAGE_CLEANUP_THRESHOLD_PERCENT = 90.0
try:
    STORAGE_CLEANUP_INTERVAL_SECONDS = max(
        30,
        int(os.environ.get("STORAGE_CLEANUP_INTERVAL_SECONDS", "300")),
    )
except ValueError:
    STORAGE_CLEANUP_INTERVAL_SECONDS = 300


def cleanup_old_runtime_sessions() -> list[str]:
    """Delete oldest inactive session directories until disk use is safe."""
    deleted: list[str] = []
    try:
        active_session_id = str(CURRENT_STATE.get("session_id") or "")
        candidates = [
            path
            for path in RUNTIME_DIR.iterdir()
            if path.is_dir()
            and path.name != PLEDGE_DIR.name
            and path.name != active_session_id
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime)

        for session_path in candidates:
            usage = shutil.disk_usage(RUNTIME_DIR)
            used_percent = usage.used * 100.0 / usage.total
            if used_percent <= STORAGE_CLEANUP_THRESHOLD_PERCENT:
                break
            shutil.rmtree(session_path)
            deleted.append(session_path.name)
            print(
                f"Storage cleanup: deleted oldest inactive session {session_path.name} "
                f"(disk was {used_percent:.1f}% used)."
            )
    except Exception as exc:
        print(f"Storage cleanup failed safely: {exc}")
    return deleted


def storage_cleanup_worker() -> None:
    while True:
        cleanup_old_runtime_sessions()
        time.sleep(STORAGE_CLEANUP_INTERVAL_SECONDS)


def new_session_id(pledge_id: str | None = None) -> str:
    prefix = f"{pledge_id}_" if pledge_id else ""
    return f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def build_empty_purity_state() -> dict[str, Any]:
    manager = get_purity_manager()
    manager.reset(stop_running=True)
    return manager.snapshot()


def build_empty_state() -> dict[str, Any]:
    return {
        "status": "Ready",
        "session_id": None,
        "pledge_id": None,
        "pledge_started_at": None,
        "pledge_result_generated_at": None,
        "jewel_count": None,
        "jewel_count_saved": False,
        "jewel_index": None,
        "completed_jewels": 0,
        "pledge_complete": False,
        "next_jewel_available": False,
        "retry_jewel_available": False,
        "is_last_jewel": False,
        "jewel_count_verification": None,
        "packet_sealing": _default_packet_sealing_state(),
        "updated_at": now_stamp(),
        "branch": None,
        "stage_skips": {},
        "weight_details": {
            "jewel_weight_g": None,
            "appraiser_stone_weight_g": None,
        },
        "weight_extraction": {
            "success": False,
            "status": "not_captured",
            "message": "Place the weight scale inside the guide box region.",
            "captured_at": None,
            "captured_image": None,
            "roi_image": None,
            "lcd_image": None,
        },
        "source": {
            "kind": None,
            "filename": None,
            "image_size": None,
            "processing_roi": PERSISTENT_ROIS["processing_roi"],
            "aruco_roi": PERSISTENT_ROIS["aruco_roi"],
            "working_aruco_roi": None,
            "calibration_config": copy.deepcopy(PERSISTENT_ROIS["calibration_config"]),
            "stone_calibration": None,
            "color_correction": copy.deepcopy(PERSISTENT_ROIS["color_correction"]),
            "analysis_normalization": copy.deepcopy(PERSISTENT_ROIS["analysis_normalization"]),
            "background_calibration": copy.deepcopy(PERSISTENT_ROIS["background_calibration"]),
            "learned_stone_profiles": copy.deepcopy(PERSISTENT_ROIS["learned_stone_profiles"]),
            "stone_super_resolution": copy.deepcopy(PERSISTENT_ROIS["stone_super_resolution"]),
            "original_image": None,
            "working_image": None,
            "preprocessed_image": None,
            "preprocessed_mask": None,
            "roi_preview": None,
        },
        "classification": {
            "predicted_label": None,
            "confidence": None,
            "confirmed_label": None,
            "confirmed": False,
            "scores": [],
            "original_preview": None,
            "cropped_preview": None,
            "is_gold_jewelry": True,
            "gold_verification_reason": "",
        },
        "dimension": {
            "done": False,
        },
        "segmentation": {
            "done": False,
            "bead_risk": None,
            "bead_analysis": None,
            "no_pendant": False,
            "no_tassel": False,
            "pendant_absent": False,
            "tassel_absent": False,
        },
        "stone_detection": {
            "setting_profile": stone_area_calculator.DEFAULT_STONE_SETTING_PROFILE,
            "main": None,
            "side": None,
        },
        "side_capture": {
            "filename": None,
            "raw_image": None,
            "original_image": None,
            "preview": None,
            "hand_removed_image": None,
            "hand_removal_applied": False,
            "hand_removal_version": None,
        },
        "purity_test": build_empty_purity_state(),
        "final": {
            "ready": False,
            "headline": "Awaiting source image",
            "lines": [],
            "artifacts": [],
        },
    }


def normalize_positive_weight(value: Any, field_name: str, *, required: bool) -> float | None:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"{field_name} is required.")
        return None
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid number in grams.") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field_name} must be greater than 0 grams.")
    return round(weight, 4)


def stage_is_skipped(state: dict[str, Any], stage_key: str) -> bool:
    return bool((state.get("stage_skips") or {}).get(stage_key))


def mark_stage_skipped(state: dict[str, Any], stage_key: str) -> dict[str, str]:
    if stage_key not in STAGE_DISPLAY_NAMES:
        raise ValueError("Choose a valid optional workflow stage.")
    skipped = {
        "status": "skipped",
        "skipped_at": now_stamp(),
        "display_name": STAGE_DISPLAY_NAMES[stage_key],
    }
    state.setdefault("stage_skips", {})[stage_key] = skipped
    return skipped


def clear_stage_skip(state: dict[str, Any], stage_key: str) -> None:
    (state.get("stage_skips") or {}).pop(stage_key, None)


def tassel_weight_for_state(state: dict[str, Any]) -> float:
    segmentation = state.get("segmentation") or {}
    part_summary = segmentation.get("part_summary") or {}
    tassel_area = int((part_summary.get("tassel") or {}).get("area", 0) or 0)
    detected = bool(segmentation.get("tassel_detected") or tassel_area > 0)
    if segmentation.get("no_tassel") or stage_is_skipped(state, "jewellery_analysis"):
        detected = False
    return ESTIMATED_TASSEL_WEIGHT_G if detected else 0.0


def ensure_state() -> dict[str, Any]:
    global CURRENT_STATE
    if not CURRENT_STATE:
        CURRENT_STATE = build_empty_state()
    return CURRENT_STATE


def session_dir_for(state: dict[str, Any]) -> Path:
    session_id = state.get("session_id")
    if not session_id:
        raise RuntimeError("No active session.")
    return RUNTIME_DIR / str(session_id)


def _purity_artifact_path_keys() -> dict[str, str]:
    return {
        "rubbing_image": "rubbing_image_path",
        "rubbing_zoom_image": "rubbing_zoom_image_path",
        "acid_stage_image": "acid_stage_image_path",
        "acid_success_image": "acid_success_image_path",
        "acid_zoom_image": "acid_zoom_image_path",
        "final_image": "final_image_path",
    }


def refresh_purity_state(state: dict[str, Any]) -> None:
    previous_purity = state.get("purity_test") or {}
    skipped_at = previous_purity.get("skipped_at") if previous_purity.get("skipped") else None
    purity_raw = get_purity_manager().snapshot()
    if purity_raw is None:
        purity_raw = {}
    purity_state: dict[str, Any] = dict(purity_raw)
    for artifact_key, path_key in _purity_artifact_path_keys().items():
        raw_path = purity_raw.get(path_key)
        artifact = artifact_payload(state, Path(raw_path)) if raw_path else None
        purity_state[artifact_key] = artifact
    if skipped_at and not purity_state.get("started_at") and not purity_state.get("running"):
        purity_state.update(
            {
                "skipped": True,
                "skipped_at": skipped_at,
                "status": "Acid test skipped.",
                "result": "Skipped",
                "acid_ok": False,
            }
        )
    state["purity_test"] = purity_state
    if purity_state.get("started_at"):
        state["status"] = purity_state.get("status") or state.get("status", "Acid test running")
        state["updated_at"] = now_stamp()


def reset_purity_state(state: dict[str, Any], *, stop_running: bool = True) -> None:
    get_purity_manager().reset(stop_running=stop_running)
    state["purity_test"] = {}
    refresh_purity_state(state)


def sanitize_filename(name: str | None, default: str) -> str:
    raw = (name or default).strip()
    if not raw:
        raw = default
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)
    return safe or default


def pledge_media_dir(pledge_id: str | None) -> Path:
    safe_id = sanitize_filename(pledge_id, "pledge")
    path = PLEDGE_MEDIA_DIR / safe_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def pledge_artifact_payload(pledge_id: str | None, path: Path | None) -> dict[str, str] | None:
    if not pledge_id or path is None:
        return None
    base = pledge_media_dir(pledge_id).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(base).as_posix()
    except ValueError:
        return None
    version = ""
    if resolved.exists():
        version = f"?v={resolved.stat().st_mtime_ns}"
    return {
        "name": resolved.name,
        "path": str(resolved),
        "url": f"/pledge-artifacts/{base.name}/{relative}{version}",
    }


def _default_packet_sealing_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "recording": False,
        "compressing": False,
        "started_at": None,
        "stopped_at": None,
        "video": None,
        "av1": None,
        "striping": {
            "status": "idle",
            "sealed": None,
            "reason": "",
            "hand_clear_enabled": False,
            "evidence_image": None,
            "error": "",
        },
        "error": "",
    }


def pledge_meta_path(pledge_id: str) -> Path:
    safe_id = sanitize_filename(pledge_id, "pledge")
    return PLEDGE_DIR / f"{safe_id}.json"


def _empty_pledge_metadata(pledge_id: str) -> dict[str, Any]:
    stamp = now_stamp()
    return {
        "pledge_id": pledge_id,
        "started_at": stamp,
        "updated_at": stamp,
        "result_generated_at": None,
        "jewel_count": None,
        "count_saved": False,
        "jewel_count_verification": None,
        "packet_sealing": _default_packet_sealing_state(),
    }


def load_pledge_metadata(pledge_id: str | None) -> dict[str, Any] | None:
    if not pledge_id:
        return None
    path = pledge_meta_path(pledge_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        data.setdefault("pledge_id", pledge_id)
        data.setdefault("started_at", now_stamp())
        data.setdefault("updated_at", data.get("started_at") or now_stamp())
        data.setdefault("result_generated_at", None)
        data.setdefault("jewel_count", None)
        data.setdefault("count_saved", bool(data.get("jewel_count")))
        data.setdefault("jewel_count_verification", None)
        packet = data.get("packet_sealing")
        if not isinstance(packet, dict):
            packet = {}
        data["packet_sealing"] = {
            **_default_packet_sealing_state(),
            **packet,
        }
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load pledge metadata for {pledge_id}: {exc}")
        return None


def save_pledge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    pledge_id = str(metadata.get("pledge_id") or "").strip()
    if not pledge_id:
        raise ValueError("Pledge ID is required.")
    metadata["pledge_id"] = pledge_id
    metadata["updated_at"] = now_stamp()
    path = pledge_meta_path(pledge_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def get_or_create_pledge_metadata(pledge_id: str) -> dict[str, Any]:
    metadata = load_pledge_metadata(pledge_id)
    if metadata is None:
        metadata = _empty_pledge_metadata(pledge_id)
        save_pledge_metadata(metadata)
    return metadata


def normalize_jewel_count(value: Any) -> int:
    try:
        count = int(value)
    except Exception as exc:
        raise ValueError("Jewel count must be a whole number.") from exc
    if count <= 0:
        raise ValueError("Jewel count must be at least 1.")
    return count


def apply_pledge_metadata_to_state(
    state: dict[str, Any],
    metadata: dict[str, Any] | None,
    *,
    jewel_index: int | None = None,
) -> None:
    if not metadata:
        return
    state["pledge_id"] = metadata.get("pledge_id")
    state["pledge_started_at"] = metadata.get("started_at")
    state["pledge_result_generated_at"] = metadata.get("result_generated_at")
    count = metadata.get("jewel_count")
    state["jewel_count"] = int(count) if count else None
    state["jewel_count_saved"] = bool(metadata.get("count_saved") and count)
    state["jewel_count_verification"] = metadata.get("jewel_count_verification")
    packet = metadata.get("packet_sealing")
    state["packet_sealing"] = (
        {**_default_packet_sealing_state(), **packet}
        if isinstance(packet, dict)
        else _default_packet_sealing_state()
    )
    if jewel_index is not None:
        state["jewel_index"] = int(jewel_index)
    elif state.get("jewel_index") is None:
        state["jewel_index"] = 1 if state["jewel_count_saved"] else None


def pledge_context_from_state(state: dict[str, Any]) -> dict[str, Any]:
    pledge_id = state.get("pledge_id")
    metadata = load_pledge_metadata(pledge_id) if pledge_id else None
    if metadata is None and pledge_id:
        metadata = {
            "pledge_id": pledge_id,
            "started_at": state.get("pledge_started_at") or now_stamp(),
            "updated_at": now_stamp(),
            "result_generated_at": state.get("pledge_result_generated_at"),
            "jewel_count": state.get("jewel_count"),
            "count_saved": bool(state.get("jewel_count_saved")),
            "jewel_count_verification": state.get("jewel_count_verification"),
            "packet_sealing": state.get("packet_sealing") or _default_packet_sealing_state(),
        }
    return {
        "metadata": metadata,
        "jewel_index": state.get("jewel_index"),
    }


def apply_pledge_context_to_state(
    state: dict[str, Any],
    context: dict[str, Any],
    *,
    jewel_index: int | None = None,
) -> None:
    metadata = context.get("metadata")
    resolved_index = jewel_index if jewel_index is not None else context.get("jewel_index")
    apply_pledge_metadata_to_state(state, metadata, jewel_index=resolved_index)


_PACKET_FFMPEG_PATH_CACHE: str | None = None


def _resolve_packet_ffmpeg() -> str | None:
    global _PACKET_FFMPEG_PATH_CACHE
    if _PACKET_FFMPEG_PATH_CACHE is not None:
        return _PACKET_FFMPEG_PATH_CACHE or None

    found = shutil.which(PACKET_AV1_FFMPEG) or (
        PACKET_AV1_FFMPEG
        if os.path.isabs(PACKET_AV1_FFMPEG) and os.path.exists(PACKET_AV1_FFMPEG)
        else None
    )
    if not found and os.name == "nt":
        pattern = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg*",
            "**",
            "ffmpeg.exe",
        )
        hits = glob.glob(pattern, recursive=True)
        found = hits[0] if hits else None

    _PACKET_FFMPEG_PATH_CACHE = found or ""
    return found


def _even_video_dimension(value: int | float, minimum: int = 2) -> int:
    return max(minimum, int(value) // 2 * 2)


def _packet_recording_dimensions(width: int, height: int) -> tuple[int, int]:
    rec_w = _even_video_dimension(max(2, int(width * PACKET_REC_SCALE)))
    rec_h = _even_video_dimension(max(2, int(height * PACKET_REC_SCALE)))
    max_dimension = max(0, int(PACKET_REC_MAX_DIMENSION or 0))
    if max_dimension > 0 and max(rec_w, rec_h) > max_dimension:
        scale = max_dimension / float(max(rec_w, rec_h))
        rec_w = _even_video_dimension(rec_w * scale)
        rec_h = _even_video_dimension(rec_h * scale)
    return rec_w, rec_h


def _packet_capture_fps() -> float:
    return max(1.0, float(PACKET_REC_FPS))


def _packet_playback_speed() -> float:
    return max(1.0, float(PACKET_REC_PLAYBACK_SPEED))


def _packet_output_fps() -> float:
    return max(1.0, _packet_capture_fps() * _packet_playback_speed())


def _packet_video_duration_seconds(path: Path) -> float:
    cap: cv2.VideoCapture | None = None
    try:
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frames > 0:
            return max(1.0, frames / fps)
    except Exception:
        pass
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return 30.0


def _packet_target_video_kbps(path: Path) -> int:
    target_bytes = max(1, int(PACKET_TARGET_SIZE_BYTES or 0))
    duration_s = _packet_video_duration_seconds(path)
    usable_bytes = max(1, int(target_bytes * max(0.1, min(0.95, PACKET_TARGET_SIZE_MARGIN))))
    kbps = int((usable_bytes * 8) / (duration_s * 1000))
    return max(
        int(PACKET_TARGET_MIN_VIDEO_KBPS),
        min(max(int(PACKET_TARGET_MAX_VIDEO_KBPS), int(PACKET_TARGET_MIN_VIDEO_KBPS)), kbps),
    )


def _transcode_packet_video_to_target_size(
    raw_path: Path,
    final_path: Path,
    ffmpeg: str,
) -> dict[str, Any] | None:
    target_bytes = max(0, int(PACKET_TARGET_SIZE_BYTES or 0))
    if target_bytes <= 0:
        return None

    base_kbps = _packet_target_video_kbps(raw_path)
    result: dict[str, Any] | None = None
    encoder_sets: list[tuple[str, list[str]]] = [
        (
            "libsvtav1-target",
            ["-c:v", "libsvtav1", "-preset", str(max(PACKET_AV1_PRESET, 10))],
        ),
        (
            "libaom-av1-target",
            ["-c:v", "libaom-av1", "-cpu-used", "8"],
        ),
        (
            "libx264-target",
            ["-c:v", "libx264", "-preset", "veryfast"],
        ),
    ]

    for factor in (1.0, 0.65, 0.45):
        kbps = max(int(PACKET_TARGET_MIN_VIDEO_KBPS), int(base_kbps * factor))
        rate_args = [
            "-b:v",
            f"{kbps}k",
            "-maxrate",
            f"{kbps}k",
            "-bufsize",
            f"{max(kbps * 2, 12)}k",
        ]
        for encoder, encoder_args in encoder_sets:
            tmp_path = final_path.with_suffix(final_path.suffix + f".{encoder}.{kbps}k.tmp")
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-r",
                f"{_packet_output_fps():.3f}",
                *encoder_args,
                *rate_args,
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(tmp_path),
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                    size = tmp_path.stat().st_size
                    candidate = {
                        "applied": True,
                        "encoder": encoder,
                        "final_size_bytes": size,
                        "target_size_bytes": target_bytes,
                        "target_video_kbps": kbps,
                        "error": "" if size <= target_bytes else f"Target encode completed but exceeded {target_bytes} bytes.",
                    }
                    current_size = int((result or {}).get("final_size_bytes") or 0)
                    current_applied = bool((result or {}).get("applied"))
                    if result is None or not current_applied or current_size <= 0 or size < current_size:
                        result = candidate
                        os.replace(tmp_path, final_path)
                    else:
                        tmp_path.unlink()
                    if size <= target_bytes:
                        return result
            except Exception as exc:  # noqa: BLE001
                if result is None:
                    result = {
                        "applied": False,
                        "encoder": encoder,
                        "final_size_bytes": 0,
                        "target_size_bytes": target_bytes,
                        "target_video_kbps": kbps,
                        "error": str(exc),
                    }
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
    return result


def _transcode_packet_video_to_av1(raw_path: Path, final_path: Path) -> dict[str, Any]:
    result = {
        "requested": bool(PACKET_AV1_TRANSCODE),
        "applied": False,
        "encoder": None,
        "source_size_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
        "final_size_bytes": 0,
        "target_size_bytes": max(0, int(PACKET_TARGET_SIZE_BYTES or 0)),
        "capture_fps": round(_packet_capture_fps(), 3),
        "playback_speed": round(_packet_playback_speed(), 3),
        "output_fps": round(_packet_output_fps(), 3),
        "error": "",
    }
    if not PACKET_AV1_TRANSCODE:
        shutil.move(str(raw_path), str(final_path))
        result["final_size_bytes"] = final_path.stat().st_size if final_path.exists() else 0
        return result

    ffmpeg = _resolve_packet_ffmpeg()
    if not ffmpeg:
        shutil.move(str(raw_path), str(final_path))
        result["error"] = "ffmpeg not found; kept OpenCV MP4."
        result["final_size_bytes"] = final_path.stat().st_size if final_path.exists() else 0
        return result

    encoders: list[tuple[str, list[str]]] = [
        (
            "libsvtav1",
            [
                "-c:v",
                "libsvtav1",
                "-preset",
                str(PACKET_AV1_PRESET),
                "-crf",
                str(PACKET_AV1_CRF),
            ],
        ),
        (
            "libaom-av1",
            [
                "-c:v",
                "libaom-av1",
                "-cpu-used",
                "8",
                "-crf",
                str(PACKET_AV1_CRF),
                "-b:v",
                "0",
            ],
        ),
    ]
    last_error = ""
    for encoder, encoder_args in encoders:
        tmp_path = final_path.with_suffix(final_path.suffix + f".{encoder}.tmp")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            *encoder_args,
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(tmp_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                os.replace(tmp_path, final_path)
                final_size = final_path.stat().st_size
                result.update(
                    {
                        "applied": True,
                        "encoder": encoder,
                        "final_size_bytes": final_size,
                        "error": "",
                    }
                )
                if (
                    result["target_size_bytes"] > 0
                    and final_size > int(result["target_size_bytes"])
                ):
                    target_result = _transcode_packet_video_to_target_size(
                        raw_path,
                        final_path,
                        ffmpeg,
                    )
                    if target_result and target_result.get("applied"):
                        result.update(target_result)
                try:
                    raw_path.unlink()
                except Exception:
                    pass
                return result
            last_error = (completed.stderr or "").strip()[:300]
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    shutil.move(str(raw_path), str(final_path))
    result["error"] = last_error or "AV1 transcode failed; kept OpenCV MP4."
    result["final_size_bytes"] = final_path.stat().st_size if final_path.exists() else 0
    return result


def get_packet_striping_hailo_models(
    *,
    create_if_missing: bool = False,
) -> dict[str, Any]:
    """Return striping HEFs that were configured during the app startup preload."""
    global PACKET_STRIP_HAILO_MODELS
    if PACKET_STRIP_HAILO_MODELS is None:
        if not create_if_missing:
            raise RuntimeError(
                "Packet striping HEFs were not loaded during application startup."
            )
        with MODEL_LOCK:
            if PACKET_STRIP_HAILO_MODELS is None:
                runtime = get_hailo_runtime()
                created_models = []
                try:
                    bag_model = runtime.create_model(
                        str(STRIPING_BAG_HEF_PATH),
                        "PacketStripBag",
                        timeout_ms=DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
                        batch_size=DEFAULT_HAILO_BATCH_SIZE,
                    )
                    if bag_model is not None:
                        created_models.append(bag_model)
                    strip_model = runtime.create_model(
                        str(STRIPING_HEF_PATH),
                        "PacketStrip",
                        timeout_ms=DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
                        batch_size=DEFAULT_HAILO_BATCH_SIZE,
                    )
                    if strip_model is not None:
                        created_models.append(strip_model)
                    if bag_model is None or strip_model is None:
                        raise RuntimeError(
                            runtime.last_model_error
                            or "Could not load the packet striping HEF models."
                        )
                    PACKET_STRIP_HAILO_MODELS = {
                        "bag": bag_model,
                        "strip": strip_model,
                    }
                except Exception:
                    for model in created_models:
                        model.close()
                        try:
                            runtime.models.remove(model)
                        except ValueError:
                            pass
                    raise
    return PACKET_STRIP_HAILO_MODELS


def refresh_packet_striping_hailo_models() -> dict[str, Any]:
    """Reconfigure the startup-loaded packet HEFs before their final-stage use."""
    global PACKET_STRIP_HAILO_MODELS
    with MODEL_LOCK:
        runtime = get_hailo_runtime()
        for model in (PACKET_STRIP_HAILO_MODELS or {}).values():
            try:
                model.close()
            finally:
                try:
                    runtime.models.remove(model)
                except ValueError:
                    pass
        PACKET_STRIP_HAILO_MODELS = None
        models = get_packet_striping_hailo_models(create_if_missing=True)
        try:
            for model in models.values():
                model.validate_runtime_contract(probe_runs=1)
        except Exception:
            for model in models.values():
                model.close()
                try:
                    runtime.models.remove(model)
                except ValueError:
                    pass
            PACKET_STRIP_HAILO_MODELS = None
            raise
        print(
            "[PacketStriping] Refreshed and tested packet HEFs before recording."
        )
        return models


class PacketStripingVerifier:
    """Run the existing striping HEFs beside packet video recording."""

    def __init__(
        self,
        pledge_id: str,
        processing_roi: dict[str, int] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._pledge_id = pledge_id
        self._media_dir = pledge_media_dir(pledge_id)
        self._support = None
        self._roi = None
        if processing_roi:
            x, y, width, height = rect_to_tuple(processing_roi)
            self._roi = (x, y, x + width, y + height)
        self._bag_hailo_model = None
        self._strip_hailo_model = None
        self._strip_fp_filter = None
        self._active_model = None
        self._active_worker = None
        self._active_kind = None
        self._status = "idle"
        self._sealed: bool | None = None
        self._reason = ""
        self._error = ""
        self._evidence_path: Path | None = None
        self._started_at: str | None = None
        self._updated_at: str | None = None
        self._hand_clear_requested = False
        self._cover_mask = None
        self._cover_confidence: float | None = None
        self._rectangularity = 0.0
        self._target_bag_mask = None
        self._target_bag_zone = None
        self._current_strip_mask = None
        self._strip_appearance_change = 0.0
        self._strip_confidence: float | None = None
        self._strip_present = False
        self._strip_confirm_count = 0
        self._seal_gone_checks = 0
        self._verification_strip_misses = 0
        self._last_frame: np.ndarray | None = None

    def start(self) -> None:
        with self._lock:
            self._reset_cycle()
            self._started_at = now_stamp()
            self._updated_at = self._started_at
            try:
                from jewel_tracka_rpi import striping_process_hef as striping

                self._support = striping
                models = get_packet_striping_hailo_models()
                self._bag_hailo_model = models["bag"]
                self._strip_hailo_model = models["strip"]
                fp_filter = striping.HSVFPFilter(
                    str(STRIPING_PROCESS_DIR / striping.DEFAULT_STRIP_FP_MODEL),
                    striping.DEFAULT_STRIP_FP_CONFIDENCE,
                )
                self._strip_fp_filter = fp_filter if fp_filter.enabled else None
                self._status = "tracking"
                self._start_model("cover")
            except Exception as exc:  # noqa: BLE001
                self._status = "unavailable"
                self._error = str(exc)
                self._reason = "Strip verification could not start."
                self._updated_at = now_stamp()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "sealed": self._sealed,
                "reason": self._reason,
                "error": self._error,
                "started_at": self._started_at,
                "updated_at": self._updated_at,
                "hand_clear_enabled": self._status == "strip_detected",
                "evidence_image": pledge_artifact_payload(
                    self._pledge_id, self._evidence_path
                ),
                "overlay": {
                    "bag": self._mask_overlay(self._cover_mask),
                    "strip": self._mask_overlay(self._current_strip_mask),
                    "bag_confidence": self._cover_confidence,
                    "strip_confidence": self._strip_confidence,
                    "strip_appearance_change": round(
                        float(self._strip_appearance_change), 3
                    ),
                    "rectangularity": round(float(self._rectangularity), 3),
                },
            }

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw the current packet masks into the live MJPEG preview."""
        with self._lock:
            annotated = frame.copy()
            bag_label = "PACKET"
            if self._cover_confidence is not None:
                bag_label += f" C {self._cover_confidence:.3f}"
            bag_label += f" R {self._rectangularity:.2f}"
            strip_label = "PACKET STRIP"
            if self._strip_confidence is not None:
                strip_label += f" C {self._strip_confidence:.3f}"
            masks = (
                (self._cover_mask, (50, 220, 50), bag_label),
                (self._current_strip_mask, (0, 140, 255), strip_label),
            )
            for mask, color, label in masks:
                if mask is None or mask.shape[:2] != annotated.shape[:2]:
                    continue
                selected = mask > 0
                if not np.any(selected):
                    continue
                annotated[selected] = (
                    annotated[selected].astype(np.float32) * 0.68
                    + np.asarray(color, dtype=np.float32) * 0.32
                ).astype(np.uint8)
                contours, _hierarchy = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(annotated, [contour], -1, color, 4, cv2.LINE_AA)
                x, y, width, _height = cv2.boundingRect(contour)
                cv2.putText(
                    annotated,
                    label,
                    (x, max(28, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            return annotated

    def _mask_overlay(self, mask: np.ndarray | None) -> dict[str, Any] | None:
        if mask is None or self._support is None:
            return None
        contour = self._support.get_largest_contour(mask)
        if contour is None:
            return None
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, max(2.0, perimeter * 0.003), True)
        points = [
            {"x": int(point[0][0]), "y": int(point[0][1])}
            for point in simplified[:64]
        ]
        if len(points) < 3:
            return None
        x, y, width, height = cv2.boundingRect(contour)
        return {
            "contour": points,
            "rect": {
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
            },
        }

    def request_hand_clear(self) -> None:
        with self._lock:
            if self._status != "strip_detected":
                raise ValueError("Wait until the strip is detected before pressing Hand Clear.")
            self._hand_clear_requested = True
            self._verification_strip_misses = 0
            self._strip_appearance_change = 0.0
            self._reason = "Confirming that the strip is absent inside the packet mask."
            self._updated_at = now_stamp()

    def restart(self, *, reload_hailo_models: bool = False) -> None:
        with self._lock:
            if self._support is None:
                raise RuntimeError(self._error or "Strip verification is unavailable.")
            self._reset_cycle()
            if reload_hailo_models:
                models = refresh_packet_striping_hailo_models()
                self._bag_hailo_model = models["bag"]
                self._strip_hailo_model = models["strip"]
            self._status = "tracking"
            self._reason = "Waiting for the packet to lie rectangular."
            self._updated_at = now_stamp()
            self._start_model("cover")

    def skip(self) -> None:
        with self._lock:
            if self._status in {"sealed", "not_sealed", "skipped"}:
                return
            self._sealed = False
            self._status = "skipped"
            self._reason = "Strip verification skipped; packet is not verified as sealed."
            self._updated_at = now_stamp()
            self._save_evidence(self._last_frame)
            self._clear_live_overlays()
            self._stop_active_model()

    def process_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if frame is None or frame.size == 0:
                return
            self._last_frame = frame.copy()
            if self._status not in {"tracking", "strip_mode", "strip_detected", "cover_check"}:
                return
            if self._active_worker is None or self._support is None:
                return
            if not self._active_worker.is_alive():
                self._status = "unavailable"
                self._error = (
                    str(getattr(self._active_worker, "error", "") or "")
                    or "Strip verification worker stopped."
                )
                self._reason = "Strip verification could not continue."
                self._updated_at = now_stamp()
                print(f"[PacketStriping] {self._error}")
                return

            active_worker = self._active_worker
            active_worker.submit(self._roi_frame(frame))
            worker_result = active_worker.get_result()
            confidence = (
                float(active_worker.last_confidence)
                if worker_result is not None
                and active_worker.last_confidence is not None
                else None
            )
            result = self._offset_result(
                worker_result, frame.shape[:2]
            )

            if self._status == "tracking":
                self._process_cover_result(result, frame.shape[:2], confidence)
            elif self._status == "cover_check":
                self._process_cover_check(result, confidence)
            else:
                self._process_strip_result(result, confidence)

    def finalize(self, frame: np.ndarray | None = None) -> None:
        with self._lock:
            if frame is not None and frame.size:
                self._last_frame = frame.copy()
            if self._status not in {"sealed", "not_sealed", "skipped"}:
                self._sealed = False
                self._status = "not_sealed"
                self._reason = (
                    "Strip verification was unavailable."
                    if self._error
                    else "Strip removal was not verified before recording stopped."
                )
                self._updated_at = now_stamp()
                self._save_evidence(self._last_frame)
            self._clear_live_overlays()
            self._stop_active_model()

    def _clear_live_overlays(self) -> None:
        self._cover_mask = None
        self._cover_confidence = None
        self._rectangularity = 0.0
        self._current_strip_mask = None
        self._strip_confidence = None

    def _reset_cycle(self) -> None:
        self._stop_active_model()
        self._status = "idle"
        self._sealed = None
        self._reason = ""
        self._error = ""
        self._evidence_path = None
        self._hand_clear_requested = False
        self._cover_mask = None
        self._cover_confidence = None
        self._rectangularity = 0.0
        self._target_bag_mask = None
        self._target_bag_zone = None
        self._current_strip_mask = None
        self._strip_appearance_change = 0.0
        self._strip_confidence = None
        self._strip_present = False
        self._strip_confirm_count = 0
        self._seal_gone_checks = 0
        self._verification_strip_misses = 0

    def _start_model(self, kind: str) -> None:
        self._stop_active_model()
        if self._support is None:
            return
        is_cover = kind in {"cover", "cover_check"}
        model = self._support.HailoSegModel(
            str(STRIPING_BAG_HEF_PATH if is_cover else STRIPING_HEF_PATH),
            conf=(
                self._support.COVER_CONF_THRESHOLD
                if is_cover
                else self._support.STRIP_CONF_THRESHOLD
            ),
            rgb_input=is_cover,
            label=f"packet-{kind}",
            hailo_model=self._bag_hailo_model if is_cover else self._strip_hailo_model,
        )
        worker = self._support.SegWorker(
            model,
            self._strip_fp_filter if not is_cover else None,
        )
        worker.start()
        self._active_model = model
        self._active_worker = worker
        self._active_kind = kind

    def _stop_active_model(self) -> None:
        worker = self._active_worker
        model = self._active_model
        self._active_worker = None
        self._active_model = None
        self._active_kind = None
        if worker is not None:
            worker.stop()
            worker.join()
        if model is not None:
            model.close()

    def _effective_roi(self, frame: np.ndarray):
        if self._roi is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._roi
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _roi_frame(self, frame: np.ndarray) -> np.ndarray:
        roi = self._effective_roi(frame)
        if roi is None:
            return frame.copy()
        x1, y1, x2, y2 = roi
        return frame[y1:y2, x1:x2].copy()

    def _offset_result(self, result, full_shape: tuple[int, int]):
        if result is None:
            return None
        centroid, mask = result
        roi = self._effective_roi(self._last_frame)
        if roi is None:
            return centroid, mask
        x1, y1, _x2, _y2 = roi
        if centroid is not None:
            centroid = centroid[0] + x1, centroid[1] + y1
        return centroid, self._support.offset_mask(mask, roi, full_shape)

    def _process_cover_result(
        self,
        result,
        frame_shape: tuple[int, int],
        confidence: float | None,
    ) -> None:
        if result is None:
            return
        self._cover_confidence = confidence
        _centroid, self._cover_mask = result
        self._rectangularity = (
            self._support.measure_rectangularity(self._cover_mask)
            if self._cover_mask is not None
            else 0.0
        )
        if self._cover_mask is None or self._rectangularity < self._support.DEFAULT_RECT_THRESHOLD:
            return
        self._target_bag_mask = self._cover_mask.copy()
        self._target_bag_zone = self._support.make_target_zone(
            self._cover_mask, frame_shape
        )
        if self._target_bag_zone is None:
            return
        self._status = "strip_mode"
        self._reason = "Packet is rectangular. Looking for the strip."
        self._updated_at = now_stamp()
        print(
            f"[PacketStriping] Rectangular packet detected "
            f"(score={self._rectangularity:.3f}); starting strip HEF."
        )
        self._start_model("strip")

    def _process_strip_result(
        self,
        result,
        confidence: float | None,
    ) -> None:
        if result is None:
            return

        self._strip_confidence = None
        _centroid, mask = result
        mask = self._support.mask_inside_reference(mask, self._target_bag_mask)
        strip_detected = (
            self._support.get_centroid(mask) is not None
            if mask is not None
            else False
        )
        self._current_strip_mask = mask.copy() if strip_detected else None
        if strip_detected:
            self._strip_confidence = confidence
            self._strip_present = True

        if self._status == "strip_mode":
            if strip_detected:
                self._strip_confirm_count += 1
            else:
                self._strip_confirm_count = 0
            if self._strip_confirm_count >= self._support.STRIP_CONFIRM_FRAMES:
                self._status = "strip_detected"
                self._reason = "Remove the strip, then press Hand Clear."
                self._updated_at = now_stamp()
                print(
                    "[PacketStriping] Strip confirmed inside packet mask "
                    f"(pixels={int(np.count_nonzero(self._current_strip_mask))}); "
                    "live preview overlay active."
                )
                speak("Remove the packet strip, then press Hand Clear.")
            return

        if not self._hand_clear_requested:
            return
        if strip_detected:
            self._verification_strip_misses = 0
        else:
            self._verification_strip_misses += 1
        if self._verification_strip_misses >= self._support.STRIP_DEBOUNCE_FRAMES:
            self._strip_present = False
            self._current_strip_mask = None
            self._strip_confidence = None
            self._status = "cover_check"
            self._reason = "Checking that the packet remains in place."
            self._updated_at = now_stamp()
            self._start_model("cover_check")

    def _process_cover_check(self, result, confidence: float | None) -> None:
        if result is None:
            return
        self._cover_confidence = confidence
        _centroid, check_mask = result
        check_mask = self._support.mask_in_target_zone(check_mask, self._target_bag_zone)
        bag_present = self._support.mask_matches_reference(
            self._target_bag_mask,
            check_mask,
            self._support.TARGET_BAG_MASK_IOU,
            self._support.TARGET_BAG_AREA_RATIO,
        )
        if not bag_present:
            self._set_terminal(False, "Packet was removed before sealing was confirmed.")
            return
        if not self._strip_present:
            self._seal_gone_checks += 1
            if self._seal_gone_checks >= self._support.SEAL_GONE_CHECKS:
                self._set_terminal(True, "Strip removed and packet remained in place.")
                return
        else:
            self._seal_gone_checks = 0

        self._verification_strip_misses = 0
        self._status = "strip_detected"
        self._reason = "Confirming that the strip remains absent inside the packet mask."
        self._updated_at = now_stamp()
        self._start_model("strip")

    def _set_terminal(self, sealed: bool, reason: str) -> None:
        self._sealed = sealed
        self._current_strip_mask = None
        self._strip_confidence = None
        self._status = "sealed" if sealed else "not_sealed"
        self._reason = reason
        self._updated_at = now_stamp()
        self._save_evidence(self._last_frame)
        self._clear_live_overlays()
        self._stop_active_model()
        speak("Packet sealed." if sealed else "Packet not sealed.")

    def _save_evidence(self, frame: np.ndarray | None) -> None:
        if frame is None or frame.size == 0:
            return
        sealed = bool(self._sealed)
        vis = frame.copy()
        label = "SEALED" if sealed else "NOT SEALED"
        color = (50, 220, 50) if sealed else (0, 0, 255)
        banner_height = max(72, vis.shape[0] // 9)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], banner_height), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            vis,
            f"STRIP CHECK: {label}",
            (20, int(banner_height * 0.66)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, vis.shape[1] / 1300.0),
            color,
            2,
            cv2.LINE_AA,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._media_dir / f"packet_striping_{timestamp}_{'sealed' if sealed else 'not_sealed'}.png"
        try:
            save_bgr(path, vis)
            self._evidence_path = path
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)


class PacketSealingRecorder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pledge_id: str | None = None
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._raw_path: Path | None = None
        self._final_path: Path | None = None
        self._striping: PacketStripingVerifier | None = None
        self._error = ""
        self._av1: dict[str, Any] | None = None
        self._capturing = False
        self._compressing = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = bool(self._capturing)
            compressing = bool(self._compressing)
            video = (
                pledge_artifact_payload(self._pledge_id, self._final_path)
                if (
                    not compressing
                    and self._pledge_id
                    and self._final_path
                    and self._final_path.exists()
                )
                else None
            )
            striping = (
                self._striping.snapshot()
                if self._striping is not None
                else _default_packet_sealing_state()["striping"]
            )
            return {
                **_default_packet_sealing_state(),
                "status": (
                    "recording"
                    if running
                    else "compressing"
                    if compressing
                    else "saved"
                    if video
                    else "idle"
                ),
                "recording": running,
                "compressing": compressing,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "video": video,
                "av1": self._av1,
                "striping": striping,
                "error": self._error,
            }

    def is_recording(self) -> bool:
        with self._lock:
            return bool(self._capturing)

    def is_compressing(self) -> bool:
        with self._lock:
            return bool(self._compressing)

    def annotate_preview(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            striping = self._striping
        if striping is None:
            return frame
        return striping.annotate_frame(frame)

    def start(
        self,
        pledge_id: str,
        processing_roi: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        pledge_id = str(pledge_id or "").strip()
        if not pledge_id:
            raise ValueError("Pledge ID is required for packet sealing.")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Packet sealing video recording or compression is already running.")
            purity = get_purity_manager()
            if purity.worker_is_active():
                purity.stop("Preparing packet sealing")
            if purity.worker_is_active():
                raise RuntimeError(
                    "The acid-test Hailo worker is still stopping. "
                    "Wait a moment before starting packet sealing."
                )
            refresh_packet_striping_hailo_models()
            media_dir = pledge_media_dir(pledge_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_id = sanitize_filename(pledge_id, "pledge")
            self._pledge_id = pledge_id
            self._started_at = now_stamp()
            self._stopped_at = None
            self._error = ""
            self._av1 = None
            self._capturing = True
            self._compressing = False
            self._raw_path = media_dir / f"packet_sealing_{safe_id}_{timestamp}.raw.mp4"
            self._final_path = media_dir / f"packet_sealing_{safe_id}_{timestamp}.mp4"
            self._striping = PacketStripingVerifier(pledge_id, processing_roi)
            self._striping.start()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._record_loop,
                name="packet-sealing-recorder",
                daemon=True,
            )
            self._thread.start()
            return self.snapshot()

    def request_striping_hand_clear(self) -> dict[str, Any]:
        with self._lock:
            if not self._capturing or self._striping is None:
                raise RuntimeError("Start packet sealing recording before verifying strip removal.")
            self._striping.request_hand_clear()
            return self.snapshot()

    def restart_striping(self) -> dict[str, Any]:
        with self._lock:
            if not self._capturing or self._striping is None:
                raise RuntimeError("Start packet sealing recording before restarting strip verification.")
            self._striping.restart(reload_hailo_models=True)
            return self.snapshot()

    def skip_striping(self) -> dict[str, Any]:
        with self._lock:
            if not self._capturing or self._striping is None:
                raise RuntimeError("Start packet sealing recording before skipping strip verification.")
            self._striping.skip()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return self.snapshot()
            self._capturing = False
            self._compressing = True
            self._stopped_at = self._stopped_at or now_stamp()
            self._stop_event.set()
            return self.snapshot()

    def _record_loop(self) -> None:
        writer: cv2.VideoWriter | None = None
        raw_path: Path | None
        final_path: Path | None
        last_frame: np.ndarray | None = None
        with self._lock:
            raw_path = self._raw_path
            final_path = self._final_path
        try:
            if raw_path is None or final_path is None:
                raise RuntimeError("Packet sealing output path was not prepared.")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            capture_fps = _packet_capture_fps()
            output_fps = _packet_output_fps()
            frame_interval = 1.0 / capture_fps
            next_frame_at = 0.0
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_frame_at:
                    time.sleep(min(0.05, next_frame_at - now))
                    continue
                next_frame_at = now + frame_interval

                frame = get_camera_backend().get_frame_copy()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_frame = frame.copy()
                with self._lock:
                    striping = self._striping
                if striping is not None:
                    striping.process_frame(frame)

                height, width = frame.shape[:2]
                rec_w, rec_h = _packet_recording_dimensions(width, height)
                small = cv2.resize(frame, (rec_w, rec_h), interpolation=cv2.INTER_AREA)

                if writer is None:
                    writer = cv2.VideoWriter(
                        str(raw_path),
                        cv2.VideoWriter_fourcc(*PACKET_REC_CODEC[:4]),
                        output_fps,
                        (rec_w, rec_h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not open packet sealing video writer.")

                writer.write(small)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = str(exc)
        finally:
            if writer is not None:
                writer.release()
            with self._lock:
                striping = self._striping
            if striping is not None:
                striping.finalize(last_frame)
            with self._lock:
                self._capturing = False
                self._compressing = True
                self._stopped_at = self._stopped_at or now_stamp()
            av1_result: dict[str, Any] | None = None
            try:
                if raw_path and raw_path.exists() and raw_path.stat().st_size > 0 and final_path:
                    av1_result = _transcode_packet_video_to_av1(raw_path, final_path)
                elif raw_path and raw_path.exists() and final_path:
                    shutil.move(str(raw_path), str(final_path))
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = str(exc)
            with self._lock:
                if av1_result is not None:
                    self._av1 = av1_result
                self._compressing = False
                completed_packet_state = self.snapshot()
                completed_pledge_id = str(self._pledge_id or "").strip()
                self._thread = None
            self._persist_completed_state(completed_packet_state, completed_pledge_id)

    def _persist_completed_state(
        self,
        packet_state: dict[str, Any] | None = None,
        pledge_id: str | None = None,
    ) -> None:
        packet_state = packet_state or self.snapshot()
        pledge_id = str(pledge_id or self._pledge_id or "").strip()
        if not pledge_id:
            return
        try:
            notify_active_pledge = False
            with STATE_LOCK:
                metadata = get_or_create_pledge_metadata(pledge_id)
                metadata["packet_sealing"] = packet_state
                save_pledge_metadata(metadata)
                state = ensure_state()
                if str(state.get("pledge_id") or "") == pledge_id:
                    notify_active_pledge = True
                    apply_pledge_metadata_to_state(state, metadata)
                    state["status"] = (
                        "Packet sealing video saved."
                        if packet_state.get("video")
                        else packet_state.get("error") or "Packet sealing video processing failed."
                    )
                    state["updated_at"] = now_stamp()
            if packet_state.get("video") and notify_active_pledge:
                speak("Packet sealing video saved. Final report is ready.")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not persist packet sealing video completion: {exc}")


def get_packet_recorder() -> PacketSealingRecorder:
    global PACKET_RECORDER
    if PACKET_RECORDER is None:
        PACKET_RECORDER = PacketSealingRecorder()
    return PACKET_RECORDER


def decode_image_data(image_data: str) -> np.ndarray:
    if not image_data or "," not in image_data:
        raise ValueError("Image payload is empty or not a data URL.")
    _, encoded = image_data.split(",", 1)
    buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image payload.")
    return image_bgr


def build_camera_status_frame(message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (19, 26, 43)

    title = "Raspberry Pi Camera Status"
    cv2.putText(
        frame,
        title,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    lines: list[str] = []
    for raw_line in str(message or "Camera frame is not available.").splitlines():
        text = raw_line.strip()
        while len(text) > 78:
            lines.append(text[:78])
            text = text[78:]
        if text:
            lines.append(text)

    if not lines:
        lines = ["Camera frame is not available."]

    y = 150
    for line in lines[:10]:
        cv2.putText(
            frame,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (225, 232, 245),
            2,
            cv2.LINE_AA,
        )
        y += 46

    cv2.putText(
        frame,
        "Check CAM_DEVICE and verify the USB camera opens in OpenCV on the Pi.",
        (40, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (124, 156, 219),
        2,
        cv2.LINE_AA,
    )
    return frame


def get_live_camera_frame() -> np.ndarray:
    camera = get_camera_backend()
    frame = camera.get_frame_copy()
    if frame is None:
        status = camera.snapshot()
        message = status.get("last_error") or status.get("status") or "Live camera frame is not available."
        raise RuntimeError(message)
    return frame


def get_weight_reader() -> WeightReader:
    global WEIGHT_READER
    if WEIGHT_READER is None:
        with WEIGHT_READER_LOCK:
            if WEIGHT_READER is None:
                WEIGHT_READER = WeightReader()
    return WEIGHT_READER


def preload_ocr_model() -> None:
    """Load the LCD detector and PaddleOCR service before serving requests."""
    print("Loading Weight OCR models...")
    get_weight_reader()
    print("Weight OCR models loaded")


def weight_extraction_ready(state: dict[str, Any]) -> bool:
    weight_state = state.get("weight_extraction") or {}
    weight_g = (state.get("weight_details") or {}).get("jewel_weight_g")
    return bool(
        stage_is_skipped(state, "weight_extraction")
        or (weight_state.get("success") and weight_g is not None and float(weight_g) > 0)
    )


def load_image_from_payload(
    payload: dict[str, Any],
    default_filename: str,
) -> tuple[np.ndarray, str, bool]:
    use_live_frame = bool(payload.get("use_live_frame"))
    filename = sanitize_filename(payload.get("filename"), default_filename)
    if use_live_frame:
        image_bgr = get_live_camera_frame()
    else:
        image_bgr = decode_image_data(str(payload.get("image_data", "")))
    return image_bgr, filename, use_live_frame


def normalize_rect(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if not raw:
        return None

    if isinstance(raw, dict):
        x = raw.get("x", 0)
        y = raw.get("y", 0)
        w = raw.get("w", 0)
        h = raw.get("h", 0)
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        x, y, w, h = raw
    else:
        return None

    x0 = max(0, min(width, int(round(x))))
    y0 = max(0, min(height, int(round(y))))
    x1 = max(0, min(width, int(round(x + w))))
    y1 = max(0, min(height, int(round(y + h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
    }


def rect_to_tuple(rect: dict[str, int] | None) -> tuple[int, int, int, int] | None:
    if not rect:
        return None
    return int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])


def crop_image(image_bgr: np.ndarray, rect: dict[str, int] | None) -> np.ndarray:
    if not rect:
        return image_bgr.copy()
    x, y, w, h = rect_to_tuple(rect)
    return image_bgr[y : y + h, x : x + w].copy()


def translate_rect_to_crop(
    rect: dict[str, int] | None,
    crop_rect: dict[str, int] | None,
) -> dict[str, int] | None:
    if not rect:
        return None
    if not crop_rect:
        return dict(rect)

    x0 = max(rect["x"], crop_rect["x"])
    y0 = max(rect["y"], crop_rect["y"])
    x1 = min(rect["x"] + rect["w"], crop_rect["x"] + crop_rect["w"])
    y1 = min(rect["y"] + rect["h"], crop_rect["y"] + crop_rect["h"])
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "x": x0 - crop_rect["x"],
        "y": y0 - crop_rect["y"],
        "w": x1 - x0,
        "h": y1 - y0,
    }


def translate_points_to_crop(
    points: list[dict[str, float]],
    crop_rect: dict[str, int] | None,
) -> list[tuple[int, int]]:
    translated: list[tuple[int, int]] = []
    for point in points[:2]:
        x = int(round(point["x"]))
        y = int(round(point["y"]))
        if crop_rect:
            x -= int(crop_rect["x"])
            y -= int(crop_rect["y"])
        translated.append((x, y))
    return translated


def save_bgr(path: Path, image_bgr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
    return path


def save_pil(path: Path, image: PILImage.Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_labeled_rect(
    image_bgr: np.ndarray,
    rect: dict[str, int] | None,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if not rect:
        return
    x, y, w, h = rect_to_tuple(rect)
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 3)
    if not label:
        return
    cv2.rectangle(image_bgr, (x, y - 28), (x + 180, y), color, -1)
    cv2.putText(
        image_bgr,
        label,
        (x + 8, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def build_roi_preview(
    source_bgr: np.ndarray,
    processing_roi: dict[str, int] | None,
    aruco_roi: dict[str, int] | None,
) -> np.ndarray:
    preview = source_bgr.copy()
    draw_labeled_rect(preview, processing_roi, "", (37, 99, 235))
    draw_labeled_rect(preview, aruco_roi, "", (22, 163, 74))
    return preview


def artifact_payload(state: dict[str, Any], path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    session_id = state.get("session_id")
    if not session_id:
        return None
    try:
        relative = path.relative_to(session_dir_for(state)).as_posix()
    except ValueError:
        return None
    version = ""
    if path.exists():
        version = f"?v={path.stat().st_mtime_ns}"
    return {
        "name": path.name,
        "path": str(path),
        "url": f"/artifacts/{session_id}/{relative}{version}",
    }


HAILO_RT: HailoRuntime | None = None

def get_hailo_runtime() -> HailoRuntime:
    global HAILO_RT
    if HAILO_RT is None:
        with MODEL_LOCK:
            if HAILO_RT is None:
                print("\n[DEBUG] Initializing HailoRuntime...")
                try:
                    HAILO_RT = HailoRuntime()
                    print("[DEBUG] ✓ HailoRuntime initialized successfully")
                except Exception as e:
                    print(f"[DEBUG] ✗ HailoRuntime initialization failed: {e}")
                    raise
    return HAILO_RT


def get_super_resolution_runner() -> RealESRGANHailoX2:
    global SUPER_RESOLUTION_RUNNER
    if SUPER_RESOLUTION_RUNNER is None:
        with MODEL_LOCK:
            if SUPER_RESOLUTION_RUNNER is None:
                print("\n[DEBUG] Initializing Real-ESRGAN x2 Hailo model...")
                runtime = get_hailo_runtime()
                SUPER_RESOLUTION_RUNNER = RealESRGANHailoX2(
                    runtime.vdevice,
                    hef_path=SUPER_RESOLUTION_HEF_PATH,
                    inference_lock=runtime.inference_lock,
                    timeout_ms=max(60000, DEFAULT_HAILO_INFERENCE_TIMEOUT_MS),
                    allow_manual_activation=False,
                )
                print(f"[DEBUG] ✓ Real-ESRGAN x2 loaded: {SUPER_RESOLUTION_HEF_PATH.name}")
    return SUPER_RESOLUTION_RUNNER


def get_classifier() -> JewelryZeroShotClassifier:
    global CLASSIFIER
    if CLASSIFIER is None:
        with MODEL_LOCK:
            if CLASSIFIER is None:
                print("\n[DEBUG] Initializing classification model...")
                try:
                    print("Initializing classification model (this may take 10-30 seconds on first run)...")
                    CLASSIFIER = JewelryZeroShotClassifier(
                        onnx_model_path=CLASS_MODEL_PATH,
                        prompt_path=CLASSIFICATION_DIR / DEFAULT_PROMPT_FILE,
                        text_model_id=DEFAULT_TEXT_MODEL_ID,
                        embedding_cache_path=CLASSIFICATION_DIR / DEFAULT_CACHE_FILE,
                    )
                    print("[DEBUG] ✓ JewelryZeroShotClassifier initialized")
                except Exception as e:
                    print(f"[DEBUG] ✗ Classification model initialization failed: {e}")
                    raise
    return CLASSIFIER


def get_segmenter() -> necklace_segmentation.FastSamOnnx:
    global SEGMENTER
    if SEGMENTER is None:
        with MODEL_LOCK:
            if SEGMENTER is None:
                print("\n[DEBUG] Initializing segmentation model...")
                try:
                    runtime = get_hailo_runtime()
                    hailo_model = runtime.create_model(
                        str(SEG_MODEL_PATH),
                        "FastSAM",
                        timeout_ms=DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
                        batch_size=DEFAULT_HAILO_BATCH_SIZE,
                    )
                    if hailo_model is None:
                        raise RuntimeError(
                            "Could not create the FastSAM Hailo model. Check the preceding "
                            "HEF/HailoRT compatibility error in the application log."
                        )
                    print("[DEBUG] ✓ Hailo model created for segmentation")
                    SEGMENTER = necklace_segmentation.FastSamOnnx(
                        SEG_MODEL_PATH,
                        providers=["CPUExecutionProvider"],
                        input_size=640,
                        hailo_model=hailo_model,
                    )
                    print("[DEBUG] ✓ FastSamOnnx initialized")
                except Exception as e:
                    print(f"[DEBUG] ✗ Segmentation model initialization failed: {e}")
                    raise
    return SEGMENTER


def make_bead_classifier_crop(
    image_bgr: np.ndarray,
    bbox: list[int],
    padding_ratio: float = 0.05,
) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    crop_width = max(1, int(math.ceil(width * (1.0 + 2.0 * padding_ratio))))
    crop_height = max(1, int(math.ceil(height * (1.0 + 2.0 * padding_ratio))))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(math.floor(center_x - crop_width / 2.0))
    crop_y1 = int(math.floor(center_y - crop_height / 2.0))
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    image_h, image_w = image_bgr.shape[:2]
    source_x1 = max(0, crop_x1)
    source_y1 = max(0, crop_y1)
    source_x2 = min(image_w, crop_x2)
    source_y2 = min(image_h, crop_y2)
    rectangular_crop = np.full((crop_height, crop_width, 3), 114, dtype=np.uint8)
    if source_x2 > source_x1 and source_y2 > source_y1:
        target_x1 = source_x1 - crop_x1
        target_y1 = source_y1 - crop_y1
        rectangular_crop[
            target_y1 : target_y1 + (source_y2 - source_y1),
            target_x1 : target_x1 + (source_x2 - source_x1),
        ] = image_bgr[source_y1:source_y2, source_x1:source_x2]
    side = max(crop_width, crop_height)
    crop = np.full((side, side, 3), 114, dtype=np.uint8)
    letterbox_x = (side - crop_width) // 2
    letterbox_y = (side - crop_height) // 2
    crop[
        letterbox_y : letterbox_y + crop_height,
        letterbox_x : letterbox_x + crop_width,
    ] = rectangular_crop
    return crop


class BeadMobileNetV3Filter:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model: Any = None
        self.torch: Any = None
        self.transform: Any = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Bead MobileNetV3 PT model not found: {self.model_path}")

        import torch
        from torch import nn
        from torchvision import models, transforms

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        self.torch = torch
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_crops(self, crops_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        if not crops_bgr:
            return []
        with self.lock:
            self._load()
            tensors = []
            for crop_bgr in crops_bgr:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(PILImage.fromarray(rgb)))
            inputs = self.torch.stack(tensors)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(inputs), dim=1).cpu().numpy()

        results = []
        for probability in probabilities:
            class_index = int(float(probability[1]) >= BEAD_CLASSIFIER_TRUE_THRESHOLD)
            results.append(
                {
                    "resnet_prediction": "true_bead" if class_index == 1 else "false_positive",
                    "resnet_confidence": float(probability[class_index]),
                    "true_bead_probability": float(probability[1]),
                }
            )
        return results


def get_bead_classifier_filter() -> BeadMobileNetV3Filter:
    global BEAD_CLASSIFIER_FILTER
    if BEAD_CLASSIFIER_FILTER is None:
        with MODEL_LOCK:
            if BEAD_CLASSIFIER_FILTER is None:
                BEAD_CLASSIFIER_FILTER = BeadMobileNetV3Filter(BEAD_CLASSIFIER_MODEL_PATH)
    return BEAD_CLASSIFIER_FILTER


def get_bead_model() -> Any:
    global BEAD_MODEL
    if BEAD_MODEL is None:
        with MODEL_LOCK:
            if BEAD_MODEL is None:
                runtime = get_hailo_runtime()
                BEAD_MODEL = runtime.create_model(
                    str(BEAD_MODEL_PATH),
                    "BeadFinder",
                    timeout_ms=DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
                    batch_size=DEFAULT_HAILO_BATCH_SIZE,
                )
                if BEAD_MODEL is None:
                    raise RuntimeError(
                        "Could not create the bead-finder Hailo model. Check the preceding "
                        "HEF/HailoRT compatibility error in the application log."
                    )
    return BEAD_MODEL


def run_full_image_bead_detection(
    image_bgr: np.ndarray,
    processing_roi: dict[str, int] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    model = get_bead_model()
    if (int(model.input_h), int(model.input_w), int(model.input_c)) != (640, 640, 3):
        raise RuntimeError(
            f"Bead-finder HEF input must be 640x640x3; received {model.input_shape}."
        )

    full_h, full_w = image_bgr.shape[:2]
    roi = normalize_rect(processing_roi, full_w, full_h)
    if roi is None:
        detection_image = image_bgr
        offset_x = 0
        offset_y = 0
        prediction_source = "full_image"
    else:
        offset_x = roi["x"]
        offset_y = roi["y"]
        detection_image = image_bgr[
            offset_y : offset_y + roi["h"],
            offset_x : offset_x + roi["w"],
        ]
        prediction_source = "processing_roi"

    source_h, source_w = detection_image.shape[:2]
    scale = min(640.0 / source_w, 640.0 / source_h)
    resized_w = int(round(source_w * scale))
    resized_h = int(round(source_h * scale))
    resized = cv2.resize(detection_image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left = (640 - resized_w) // 2
    top = (640 - resized_h) // 2
    model_image = np.full((640, 640, 3), 114, dtype=np.uint8)
    model_image[top : top + resized_h, left : left + resized_w] = resized
    model_input = np.ascontiguousarray(cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB))

    output = np.asarray(model.run_inference(model_input), dtype=np.float32)
    rows = np.squeeze(output)
    if rows.size == 0:
        rows = np.empty((0, 5), dtype=np.float32)
    elif rows.ndim == 1:
        rows = rows.reshape(1, -1)
    elif rows.ndim > 2:
        rows = rows.reshape(-1, rows.shape[-1])
    if rows.ndim == 2 and rows.shape[1] != 5 and rows.shape[0] == 5:
        rows = rows.T
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise RuntimeError(f"Unexpected bead-finder output shape: {output.shape}.")

    yolo_candidates: list[dict[str, Any]] = []
    for y1, x1, y2, x2, score in rows:
        if (
            not np.isfinite([y1, x1, y2, x2, score]).all()
            or float(score) < BEAD_YOLO_SCORE_THRESHOLD
        ):
            continue
        box_x1 = int(round((float(x1) * 640.0 - left) / scale))
        box_y1 = int(round((float(y1) * 640.0 - top) / scale))
        box_x2 = int(round((float(x2) * 640.0 - left) / scale))
        box_y2 = int(round((float(y2) * 640.0 - top) / scale))
        box_x1 = max(0, min(source_w - 1, box_x1))
        box_y1 = max(0, min(source_h - 1, box_y1))
        box_x2 = max(0, min(source_w - 1, box_x2))
        box_y2 = max(0, min(source_h - 1, box_y2))
        if box_x2 <= box_x1 or box_y2 <= box_y1:
            continue
        yolo_candidates.append(
            {
                "bbox": [
                    box_x1 + offset_x,
                    box_y1 + offset_y,
                    box_x2 + offset_x,
                    box_y2 + offset_y,
                ],
                "score": float(score),
            }
        )

    crops = [
        make_bead_classifier_crop(image_bgr, candidate["bbox"])
        for candidate in yolo_candidates
    ]
    classifier_predictions = get_bead_classifier_filter().predict_crops(crops)
    detections: list[dict[str, Any]] = []
    annotated = image_bgr.copy()
    for candidate, prediction in zip(yolo_candidates, classifier_predictions):
        if prediction["resnet_prediction"] != "true_bead":
            continue
        detection = {**candidate, **prediction}
        detections.append(detection)
        box_x1, box_y1, box_x2, box_y2 = detection["bbox"]
        cv2.rectangle(
            annotated,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (0, 200, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(len(detections)),
            (box_x1, max(24, box_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    bead_count = len(detections)
    candidate_count = len(yolo_candidates)
    cv2.rectangle(annotated, (10, 10), (260, 54), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"Bead count: {bead_count}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )
    result = {
        "beads_detected": bead_count > 0,
        "risk": "High" if bead_count > 0 else "Low",
        "bead_count": bead_count,
        "candidate_count": candidate_count,
        "false_positive_count": candidate_count - bead_count,
        "model": BEAD_MODEL_PATH.name,
        "yolo_threshold": BEAD_YOLO_SCORE_THRESHOLD,
        "post_filter_model": BEAD_CLASSIFIER_MODEL_PATH.name,
        "post_filter_architecture": "mobilenet_v3_small",
        "post_filter_threshold": BEAD_CLASSIFIER_TRUE_THRESHOLD,
        "resnet_model": BEAD_CLASSIFIER_MODEL_PATH.name,
        "model_input_size": 640,
        "prediction_source": prediction_source,
        "processing_roi": roi,
        "detections": detections,
        "decision_reason": (
            f"{bead_count} of {candidate_count} YOLO bead candidate(s) accepted by MobileNetV3."
            if bead_count
            else f"No true beads remained after MobileNetV3 checked {candidate_count} YOLO candidate(s)."
        ),
    }
    return result, annotated


def preload_hef_models() -> None:
    """Load and self-test every HEF once before the web workflow starts."""
    print("=" * 60)
    print("LOADING AND TESTING ALL HEF MODELS AT STARTUP")
    print("=" * 60)
    try:
        runtime = get_hailo_runtime()

        print("\n[1/8] Loading Jewellery Analysis HEFs (FastSAM and Bead Finder)...")
        segmenter = get_segmenter()
        get_bead_model()
        print("[1/8] Jewellery Analysis HEFs loaded")
        fastsam_model = getattr(segmenter, "hailo_model", None)
        if fastsam_model is None:
            raise RuntimeError("FastSAM did not retain its configured Hailo model.")
        baseline = fastsam_model.validate_runtime_contract(probe_runs=1)
        print(
            f"[HAILO BASELINE PASS] FastSAM before purity load "
            f"{baseline['last_inference_ms']:.1f} ms"
        )

        print("\n[2/8] Loading Purity HEFs (Stone, Gold, Acid)...")
        purity = get_purity_manager()
        readiness = purity.preload(runtime)
        if not readiness.get("available") or not readiness.get("models_loaded"):
            raise RuntimeError(
                readiness.get("last_error")
                or readiness.get("error")
                or "Purity models could not be loaded at startup."
            )
        print("[2/8] Purity HEFs loaded")

        print("\n[3/8] Loading Hand Removal HEF (YOLOv8-seg)...")
        hand_model = get_hand_model(runtime)
        print(f"[3/8] Hand Removal HEF loaded: {HAND_MODEL_PATH.name}")

        print("\n[4/8] Loading Super Resolution HEF (Real-ESRGAN x2)...")
        super_resolution = get_super_resolution_runner()
        sr_result = super_resolution.self_test()
        print(
            f"[4/8] Super Resolution HEF loaded: "
            f"{sr_result['output_shape']} in {sr_result['runtime_seconds']:.3f}s"
        )

        print("\n[5/8] Loading Packet Strip HEFs (Bag and Strip)...")
        get_packet_striping_hailo_models(create_if_missing=True)
        print("[5/8] Packet Strip HEFs loaded")

        print("\n[6/8] Running startup inference self-test for every core HEF...")
        expected_hailo_models = {
            "FastSAM",
            "BeadFinder",
            "PurityStone",
            "PurityGold",
            "PurityAcid",
            "HandRemoval",
            "PacketStripBag",
            "PacketStrip",
        }
        loaded_hailo_models = {
            str(getattr(model, "name", "unknown")) for model in runtime.models
        }
        missing_hailo_models = sorted(expected_hailo_models - loaded_hailo_models)
        if missing_hailo_models:
            raise RuntimeError(
                "Startup HEF load is incomplete: " + ", ".join(missing_hailo_models)
            )
        for model in runtime.models:
            if str(getattr(model, "name", "")) == "BeadFinder":
                bead_probe = np.full(model.input_shape, 114, dtype=model.input_dtype)
                bead_output = np.asarray(model.run_inference(bead_probe))
                print(
                    f"[HAILO SELF-TEST PASS] BeadFinder "
                    f"{model.last_inference_ms:.1f} ms output={bead_output.shape}"
                )
                continue
            result = model.validate_runtime_contract(probe_runs=1)
            print(
                f"[HAILO SELF-TEST PASS] {result['name']} "
                f"{result['last_inference_ms']:.1f} ms"
            )
        if hand_model not in runtime.models:
            raise RuntimeError("Hand-removal HEF did not retain the shared Hailo runtime.")
        print("[6/8] ALL CORE HAILO HEF SELF-TESTS PASSED")

        print("\n[7/8] Loading Classification Model (SigLIP2)...")
        get_classifier()
        print("[7/8] Classification Model loaded")
        
        print("\n[8/8] Confirming startup model set...")
        loaded_models = ", ".join(
            str(getattr(model, "name", "unknown"))
            for model in runtime.models
        )
        print(
            f"[8/8] Loaded Hailo models: {loaded_models}; "
            f"SuperResolution={SUPER_RESOLUTION_HEF_PATH.name}"
        )
        
        print("\n" + "=" * 60)
        print("STARTUP MODEL LOAD COMPLETE; NO HEF WILL LOAD DURING THE WORKFLOW")
        print("=" * 60)
    except Exception as exc:
        # Hailo device not available - warn but don't crash
        error_msg = str(exc)
        if "HAILO_OUT_OF_PHYSICAL_DEVICES" in error_msg or "not enough free devices" in error_msg:
            print("\n" + "=" * 60)
            print("⚠️  WARNING: HAILO DEVICE NOT AVAILABLE")
            print("=" * 60)
            print("The Hailo accelerator is not detected or already in use.")
            print("Possible solutions:")
            print("  1. Check if another instance is using Hailo (kill it first)")
            print("  2. Restart the Hailo service: sudo systemctl restart hailo-vdevice")
            print("  3. Reboot the RPi5")
            print("\nStartup HEF preload is unavailable for this process.")
            print("=" * 60 + "\n")
        else:
            # Other errors - show and crash
            print("\n" + "=" * 60)
            print(f"ERROR LOADING HEF MODELS: {exc}")
            print("=" * 60)
            raise


def shutdown_runtime_resources() -> None:
    weight_reader = WEIGHT_READER
    if weight_reader is not None:
        try:
            weight_reader.close()
        except Exception:
            pass

    purity = PURITY_MANAGER
    if purity is not None:
        try:
            purity.shutdown()
        except Exception:
            pass

    camera = CAMERA_BACKEND
    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass

    super_resolution = SUPER_RESOLUTION_RUNNER
    if super_resolution is not None:
        try:
            super_resolution.close()
        except Exception:
            pass

    runtime = HAILO_RT
    if runtime is not None:
        try:
            runtime.close()
        except Exception:
            pass

    lcd = LCD_DISPLAY
    if lcd is not None:
        try:
            lcd.close()
        except Exception:
            pass


atexit.register(shutdown_runtime_resources)


def branch_for_label(label: str | None) -> dict[str, str] | None:
    if not label:
        return None
    if label in DIMENSION_CLASSES:
        return {
            "key": "dimension",
            "label": "Dimension -> Stone Detection",
        }
    if label in DIRECT_STONE_CLASSES:
        return {
            "key": "direct_stone",
            "label": "Direct Stone Detection",
        }
    if label in SEGMENTATION_CLASSES:
        return {
            "key": "segmentation",
            "label": "Jewellery Analysis -> Stone Detection",
        }
    return {
        "key": "direct_stone",
        "label": "Direct Stone Detection",
    }


def side_capture_voice_prompt(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if "bangle" in normalized:
        item = "bangle"
    elif "ring" in normalized:
        item = "ring"
    else:
        item = "jewel"
    return f"Dimension measurement completed. Now show the side of the {item} for stone detection, then capture the side image."


def _stone_coverage_for_state(state: dict[str, Any]) -> float:
    stones = state.get("stone_detection") or {}
    total = 0.0
    for stone_key in ("main", "side"):
        result = stones.get(stone_key)
        if not isinstance(result, dict):
            continue
        if result.get("stone_percentage") is not None:
            try:
                total += float(result.get("stone_percentage") or 0.0)
                continue
            except Exception:
                pass
        for entry in result.get("summary_entries") or []:
            try:
                total += float(entry.get("stone_percentage") or 0.0)
            except Exception:
                continue
    return total


def _segmentation_bead_risk_high(segmentation: dict[str, Any] | None) -> bool:
    if not isinstance(segmentation, dict):
        return False
    bead_analysis = segmentation.get("bead_analysis")
    if not isinstance(bead_analysis, dict):
        bead_analysis = (segmentation.get("debug") or {}).get("bead_analysis")
    bead_risk = str(segmentation.get("bead_risk") or "").strip().lower()
    analysis_risk = str((bead_analysis or {}).get("risk") or "").strip().lower()
    return bead_risk == "high" or analysis_risk == "high" or bool((bead_analysis or {}).get("beads_detected"))


def is_confirmed_not_gold_state(state: dict[str, Any]) -> bool:
    classification = state.get("classification") or {}
    if not classification.get("confirmed"):
        return False
    confirmed_label = str(classification.get("confirmed_label") or "").strip()
    return (
        confirmed_label == "Not Gold Jewelry"
        or classification.get("is_gold_jewelry") is False
    )


def is_appraised_jewel_state(state: dict[str, Any]) -> bool:
    classification = state.get("classification") or {}
    return bool(
        state.get("final", {}).get("ready")
        and classification.get("confirmed")
        and not is_confirmed_not_gold_state(state)
    )


def weight_summary_for_state(state: dict[str, Any]) -> dict[str, Any]:
    weights = state.get("weight_details") or {}
    jewel_weight = weights.get("jewel_weight_g")
    appraiser_stone_weight = weights.get("appraiser_stone_weight_g")
    stones = state.get("stone_detection") or {}
    stone_setting_profile = stone_area_calculator.normalize_stone_setting_profile(
        stones.get("setting_profile")
    )
    stone_setting_profile_label = {
        stone_area_calculator.STONE_SETTING_PROFILE_FRONT_ONLY: "Half cut / front-only stones",
        stone_area_calculator.STONE_SETTING_PROFILE_OPEN_BACK: "Full cut / open-back stones",
        stone_area_calculator.STONE_SETTING_PROFILE_UNKNOWN: "Unknown - visible area only",
    }[stone_setting_profile]

    estimated_stone_weight: float | None = None
    minimum_stone_weight = 0.0
    maximum_stone_weight = 0.0
    estimated_parts = 0
    stone_weight_calibration_applied = False
    stone_weight_range_narrowed = False
    weight_confidences: list[str] = []
    weight_methods: list[str] = []
    v2_geometry_estimate = False
    for stone_key in ("main", "side"):
        result = stones.get(stone_key)
        if not isinstance(result, dict):
            continue
        estimate = result.get("weight_estimate") or {}
        if estimate.get("success"):
            estimated_stone_weight = float(estimated_stone_weight or 0.0) + float(
                estimate.get(
                    "estimated_total_average_g",
                    float(estimate.get("estimated_total_average_ct", 0.0)) * 0.2,
                )
            )
            minimum_stone_weight += float(
                estimate.get(
                    "estimated_total_minimum_g",
                    float(estimate.get("estimated_total_minimum_ct", 0.0)) * 0.2,
                )
            )
            maximum_stone_weight += float(
                estimate.get(
                    "estimated_total_maximum_g",
                    float(estimate.get("estimated_total_maximum_ct", 0.0)) * 0.2,
                )
            )
            estimated_parts += 1
            weight_confidences.append(str(estimate.get("weight_confidence") or "Low"))
            if estimate.get("weight_method"):
                weight_methods.append(str(estimate["weight_method"]))
            v2_geometry_estimate = (
                v2_geometry_estimate or bool(estimate.get("v2_geometry_estimate"))
            )
            stone_weight_calibration_applied = (
                stone_weight_calibration_applied
                or bool(estimate.get("calibration_applied"))
            )
            stone_weight_range_narrowed = (
                stone_weight_range_narrowed
                or bool(estimate.get("range_narrowing_applied"))
            )
        elif float(result.get("stone_percentage") or 0.0) <= 0:
            estimated_stone_weight = float(estimated_stone_weight or 0.0)

    if estimated_parts and estimated_stone_weight is not None and jewel_weight is not None:
        aggregate_estimate = stone_area_calculator.calibrate_weight_estimate_to_jewel_weight(
            {
                "success": True,
                "estimated_total_average_g": estimated_stone_weight,
                "estimated_total_minimum_g": minimum_stone_weight,
                "estimated_total_maximum_g": maximum_stone_weight,
                "v2_geometry_estimate": v2_geometry_estimate,
                "weight_confidence": (
                    min(
                        weight_confidences,
                        key=lambda value: {"Low": 0, "Medium": 1, "High": 2}.get(value, 0),
                    )
                    if weight_confidences
                    else "Low"
                ),
                "weight_confidence_score": 0.60,
            },
            jewel_weight,
        )
        estimated_stone_weight = float(aggregate_estimate["estimated_total_average_g"])
        minimum_stone_weight = float(aggregate_estimate["estimated_total_minimum_g"])
        maximum_stone_weight = float(aggregate_estimate["estimated_total_maximum_g"])
        stone_weight_calibration_applied = (
            stone_weight_calibration_applied
            or bool(aggregate_estimate.get("calibration_applied"))
        )
        stone_weight_range_narrowed = (
            stone_weight_range_narrowed
            or bool(aggregate_estimate.get("range_narrowing_applied"))
        )

    tassel_weight = tassel_weight_for_state(state)
    estimated_deduction = None
    estimated_net_weight = None
    if estimated_stone_weight is not None:
        estimated_deduction = estimated_stone_weight + tassel_weight
        if jewel_weight is not None:
            estimated_net_weight = max(0.0, float(jewel_weight) - estimated_deduction)

    appraiser_deduction = None
    appraiser_net_weight = None
    if appraiser_stone_weight is not None:
        appraiser_deduction = float(appraiser_stone_weight) + tassel_weight
        if jewel_weight is not None:
            appraiser_net_weight = max(0.0, float(jewel_weight) - appraiser_deduction)

    return {
        "jewel_weight_g": round(float(jewel_weight), 4) if jewel_weight is not None else None,
        "estimated_stone_weight_g": (
            round(float(estimated_stone_weight), 4)
            if estimated_stone_weight is not None
            else None
        ),
        "estimated_stone_weight_minimum_g": round(minimum_stone_weight, 4) if estimated_parts else None,
        "estimated_stone_weight_maximum_g": round(maximum_stone_weight, 4) if estimated_parts else None,
        "weight_confidence": (
            min(
                weight_confidences,
                key=lambda value: {"Low": 0, "Medium": 1, "High": 2}.get(value, 0),
            )
            if weight_confidences
            else None
        ),
        "weight_method": "; ".join(sorted(set(weight_methods))) or None,
        "stone_weight_calibration_applied": stone_weight_calibration_applied,
        "stone_weight_range_narrowed": stone_weight_range_narrowed,
        "stone_setting_profile": stone_setting_profile,
        "stone_setting_profile_label": stone_setting_profile_label,
        "appraiser_stone_weight_g": (
            round(float(appraiser_stone_weight), 4)
            if appraiser_stone_weight is not None
            else None
        ),
        "tassel_present": tassel_weight > 0,
        "estimated_tassel_weight_g": round(tassel_weight, 4),
        "estimated_total_deduction_g": round(estimated_deduction, 4) if estimated_deduction is not None else None,
        "estimated_net_weight_g": round(estimated_net_weight, 4) if estimated_net_weight is not None else None,
        "appraiser_total_deduction_g": round(appraiser_deduction, 4) if appraiser_deduction is not None else None,
        "appraiser_net_weight_g": round(appraiser_net_weight, 4) if appraiser_net_weight is not None else None,
        "note": "Stone and tassel weights are estimates; the jewel weight is the captured OCR scale reading.",
    }


def build_final_summary(state: dict[str, Any]) -> None:
    classification = state["classification"]
    source = state["source"]
    dimension = state["dimension"]
    segmentation = state["segmentation"]
    stones = state["stone_detection"]
    purity = state.get("purity_test") or {}
    branch = state.get("branch") or {}

    lines: list[str] = []
    artifacts: list[dict[str, str]] = []
    upstream_ready = False
    label = classification.get("confirmed_label") or classification.get("predicted_label")

    not_gold_confirmed = is_confirmed_not_gold_state(state)
    # Confirmed non-gold items finish without appraisal stages.
    purity_ready = bool(not_gold_confirmed or purity.get("skipped") or (
        purity.get("acid_ok")
        and not purity.get("running")
        and purity.get("stopped_at")
    ))

    if label:
        lines.append(f"Jewel Type: {label}")
    
    branch_key = branch.get("key")
    total_stone_coverage = _stone_coverage_for_state(state)
    stone_risk_high = total_stone_coverage > HIGH_RISK_STONE_THRESHOLD
    bead_risk_high = _segmentation_bead_risk_high(segmentation) if branch_key == "segmentation" else False
    reflective_surface_flag = any(
        bool(
            (stones.get(key) or {}).get("reflection_risk")
            or (stones.get(key) or {}).get("reflection_flagged")
        )
        for key in ("main", "side")
    )
    risk_reasons: list[str] = []
    if branch_key == "dimension":
        if dimension.get("done"):
            if dimension.get("od_mm") is not None:
                lines.append(
                    f"OD {dimension['od_mm']:.2f} mm | ID {dimension['id_mm']:.2f} mm"
                )
            else:
                lines.append("Dimension measurement completed.")
            if dimension.get("result_image"):
                artifacts.append(dimension["result_image"])

        if stones.get("side"):
            lines.append("Side stone analysis completed.")
            if stones["side"].get("gallery"):
                artifacts.append(stones["side"]["gallery"])
            upstream_ready = True

        if stones.get("main"):
            lines.append("Top stone analysis completed.")
            if stones["main"].get("gallery"):
                artifacts.append(stones["main"]["gallery"])

        if stage_is_skipped(state, "side_stone"):
            upstream_ready = True

    elif branch_key == "segmentation":
        if segmentation.get("done"):
            lines.append("Jewellery analysis completed.")
            if segmentation.get("no_pendant"):
                lines.append("Pendant excluded by operator feedback.")
            elif segmentation.get("pendant_absent"):
                lines.append("No distinct pendant region detected.")
            if segmentation.get("no_tassel"):
                lines.append("Tassel excluded by operator feedback.")
            elif segmentation.get("tassel_absent"):
                lines.append("No tassel region detected.")
            bead_risk = segmentation.get("bead_risk")
            if bead_risk_high:
                lines.append("RISK JEWEL: Round beads/decorative elements detected in chain")
                risk_reasons.append("round beads detected in chain")
            elif bead_risk:
                lines.append(f"Chain bead risk: {bead_risk}")
            if segmentation.get("composite_layout"):
                artifacts.append(segmentation["composite_layout"])

        if stones.get("main"):
            lines.append("Stone analysis completed.")
            if stones["main"].get("gallery"):
                artifacts.append(stones["main"]["gallery"])
            upstream_ready = True
        elif stage_is_skipped(state, "stone_detection"):
            upstream_ready = True

    else:
        # Direct stone branch
        if stones.get("main"):
            lines.append("Stone analysis completed.")
            if stones["main"].get("gallery"):
                artifacts.append(stones["main"]["gallery"])
            upstream_ready = True
        elif stage_is_skipped(state, "stone_detection"):
            upstream_ready = True
        elif not_gold_confirmed:
            # The disposition is complete, but this is not an appraised jewel.
            upstream_ready = True

    weight_summary = weight_summary_for_state(state)
    if weight_summary["jewel_weight_g"] is not None:
        lines.append(f"Actual OCR jewel weight: {weight_summary['jewel_weight_g']:.2f} g")
    if stones.get("main") or stones.get("side"):
        lines.append(
            f"Stone setting type: {weight_summary['stone_setting_profile_label']}"
        )
        total_stone_instances = sum(
            int((stones.get(key) or {}).get("stone_instance_count", 0))
            for key in ("main", "side")
        )
        surface_risk = stone_analysis_v2.calculate_stone_surface_risk(
            total_stone_coverage,
            total_stone_instances,
            reflection_risk=reflective_surface_flag,
            high_threshold=HIGH_RISK_STONE_THRESHOLD,
        )
        lines.append(f"Stone Surface Status: {surface_risk['status']}")
        lines.append(f"Visible Stone Coverage: {total_stone_coverage:.1f}%")
        lines.append(f"Detected Stone Regions: {total_stone_instances}")
    if weight_summary["estimated_stone_weight_g"] is not None:
        minimum_g = weight_summary["estimated_stone_weight_minimum_g"]
        maximum_g = weight_summary["estimated_stone_weight_maximum_g"]
        minimum_g = weight_summary["estimated_stone_weight_g"] if minimum_g is None else minimum_g
        maximum_g = weight_summary["estimated_stone_weight_g"] if maximum_g is None else maximum_g
        typical_g = weight_summary["estimated_stone_weight_g"]
        lines.append(f"Estimated stone weight range: {minimum_g:.2f}-{maximum_g:.2f} g")
        lines.append(f"Typical stone weight estimate: {typical_g:.2f} g")
        lines.append(
            f"Stone weight confidence: {weight_summary['weight_confidence'] or 'Low'}"
        )
    if weight_summary["appraiser_stone_weight_g"] is not None:
        lines.append(f"Appraiser stone weight: {weight_summary['appraiser_stone_weight_g']:.4f} g")
    if weight_summary["tassel_present"]:
        lines.append(
            f"Tassel region detected; estimated tassel weight: "
            f"{weight_summary['estimated_tassel_weight_g']:.4f} g"
        )
    if weight_summary["estimated_total_deduction_g"] is not None:
        lines.append(
            f"Estimated deductions (stone + tassel): {weight_summary['estimated_total_deduction_g']:.4f} g"
        )
    if weight_summary["estimated_net_weight_g"] is not None:
        lines.append(f"Estimated net jewel weight: {weight_summary['estimated_net_weight_g']:.4f} g")
    if weight_summary["appraiser_net_weight_g"] is not None:
        lines.append(
            f"Net weight using appraiser stone weight: {weight_summary['appraiser_net_weight_g']:.4f} g"
        )
    if weight_summary["jewel_weight_g"] is not None:
        lines.append("Stone and tassel deductions are estimated weights.")

    for stage_key, skipped in (state.get("stage_skips") or {}).items():
        if stage_key in {"acid_test", "final_count", "packet_sealing"}:
            continue
        lines.append(f"{skipped.get('display_name') or STAGE_DISPLAY_NAMES.get(stage_key, stage_key)}: Skipped")

    if not_gold_confirmed:
        current_index = int(state.get("jewel_index") or 0)
        slot_text = f" for Jewel {current_index}" if current_index > 0 else ""
        lines.append(
            f"Not counted as an appraised jewel{slot_text}. Capture another item for the same slot."
        )

    if stone_risk_high:
        lines.append("RISK JEWEL: High stone coverage detected")
        risk_reasons.append("high stone coverage detected")
    if reflective_surface_flag:
        lines.append(
            "RISK JEWEL: Dense reflection indicates possible additional transparent/colorless gemstones."
        )
        risk_reasons.append(
            "dense reflection indicates possible additional transparent/colorless gemstones"
        )

    if purity.get("running"):
        lines.append(f"Acid Test: {purity.get('status') or 'Running'}")
    elif purity.get("acid_ok"):
        lines.append("Acid Test: Conducted | Acid OK")
        for key in ("rubbing_image", "acid_success_image", "acid_zoom_image"):
            artifact = purity.get(key)
            if artifact:
                artifacts.append(artifact)
    elif purity.get("skipped"):
        lines.append("Acid Test: Skipped")
    elif purity.get("stopped_at") and purity.get("result"):
        lines.append(f"Acid Test: {purity.get('result')}")

    final_ready = bool(upstream_ready and purity_ready and weight_extraction_ready(state))
    state["final"] = {
        "ready": final_ready,
        "appraised": bool(
            final_ready
            and classification.get("confirmed")
            and not not_gold_confirmed
        ),
        "excluded_from_pledge_count": bool(not_gold_confirmed),
        "headline": label or "Awaiting classification",
        "lines": lines,
        "artifacts": artifacts,
        "risk_jewel": bool(
            stone_risk_high or bead_risk_high or reflective_surface_flag
        ),
        "risk_reasons": risk_reasons,
        "high_risk": bool(
            stone_risk_high or bead_risk_high or reflective_surface_flag
        ),
        "stone_risk_high": bool(stone_risk_high),
        "total_stone_coverage": round(total_stone_coverage, 2),
        "bead_risk_high": bool(bead_risk_high),
        "reflective_surface_flag": bool(reflective_surface_flag),
        "weight_summary": weight_summary,
        "stone_weight_summary": {
            "success": weight_summary["estimated_stone_weight_g"] is not None,
            "estimated_total_average_g": weight_summary["estimated_stone_weight_g"] or 0.0,
            "estimated_total_typical_g": weight_summary["estimated_stone_weight_g"] or 0.0,
            "estimated_total_minimum_g": weight_summary["estimated_stone_weight_minimum_g"] or 0.0,
            "estimated_total_maximum_g": weight_summary["estimated_stone_weight_maximum_g"] or 0.0,
            "weight_confidence": weight_summary["weight_confidence"],
            "entered_jewel_weight_g": weight_summary["jewel_weight_g"],
        },
    }


def purity_upstream_ready(state: dict[str, Any]) -> bool:
    branch = state.get("branch") or {}
    branch_key = branch.get("key")
    stones = state.get("stone_detection") or {}

    if branch_key == "dimension":
        return bool(stones.get("side") or stage_is_skipped(state, "side_stone"))
    return bool(stones.get("main") or stage_is_skipped(state, "stone_detection"))


def _pdf_image(
    img_path: Path,
    max_width: float = 6.5 * inch,
    max_height: float = 4.0 * inch,
) -> ReportLabImage:
    """Create a ReportLab Image fitted inside a box without changing aspect ratio."""
    try:
        with PILImage.open(str(img_path)) as pil_img:
            iw, ih = pil_img.size
    except Exception:
        iw, ih = 400, 300
    if iw <= 0 or ih <= 0:
        iw, ih = 400, 300

    scale = min(max_width / iw, max_height / ih, 1.0)
    img = ReportLabImage(str(img_path), width=iw * scale, height=ih * scale)
    img.hAlign = "CENTER"
    return img


def _pdf_text(value: Any) -> str:
    return xml_escape(str(value if value is not None else "-"))


def _jewel_type_for_state(state: dict[str, Any]) -> str:
    classification = state.get("classification") or {}
    return (
        classification.get("confirmed_label")
        or classification.get("predicted_label")
        or "-"
    )


def _acid_status_for_state(state: dict[str, Any]) -> str:
    purity = state.get("purity_test") or {}
    if purity.get("skipped"):
        return "Skipped"
    if purity.get("started_at") or purity.get("stopped_at") or purity.get("acid_ok"):
        return "Conducted"
    return "Not conducted"


def _summary_states_by_index(states: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    fallback_index = 1
    for state in states:
        try:
            index = int(state.get("jewel_index") or 0)
        except Exception:
            index = 0
        if index <= 0:
            while fallback_index in by_index:
                fallback_index += 1
            index = fallback_index
        by_index[index] = state
    return by_index


def _expected_count_labels_for_pledge(
    pledge_id: str,
    active_state: dict[str, Any] | None = None,
) -> list[str]:
    states = [
        state
        for state in get_states_for_pledge(pledge_id)
        if is_appraised_jewel_state(state)
    ]
    if (
        active_state
        and active_state.get("pledge_id") == pledge_id
        and is_appraised_jewel_state(active_state)
        and all(
            state.get("session_id") != active_state.get("session_id")
            for state in states
        )
    ):
        states.append(copy.deepcopy(active_state))

    labels: list[str] = []
    for _index, state in sorted(_summary_states_by_index(states).items()):
        label = str(_jewel_type_for_state(state) or "").strip()
        if label and label != "-" and label != "Not Gold Jewelry":
            labels.append(label)
    return labels


def _prediction_score_by_label_key(prediction: PredictionResult) -> dict[str, ScoreEntry]:
    by_key: dict[str, ScoreEntry] = {}
    for score in getattr(prediction, "scores", []) or []:
        key = _label_key(score.label)
        if key and (
            key not in by_key
            or float(score.confidence) > float(by_key[key].confidence)
        ):
            by_key[key] = score
    return by_key


def _count_capture_label_for_prediction(
    prediction: PredictionResult,
    expected_labels: list[str],
    item_index: int,
) -> dict[str, Any]:
    raw_label = str(prediction.label or "").strip()
    if not expected_labels:
        return {
            "label": raw_label,
            "confidence": to_python_scalar(prediction.confidence),
            "expected_label": None,
            "raw_label": raw_label,
            "label_source": "classifier",
            "label_corrected": False,
        }

    expected_keys = {_label_key(label): label for label in expected_labels}
    raw_key = _label_key(raw_label)
    if raw_key in expected_keys:
        return {
            "label": expected_keys[raw_key],
            "confidence": to_python_scalar(prediction.confidence),
            "expected_label": expected_keys[raw_key],
            "raw_label": raw_label,
            "label_source": "gallery" if prediction.gallery_match else "classifier",
            "label_corrected": expected_keys[raw_key] != raw_label,
        }

    score_by_key = _prediction_score_by_label_key(prediction)
    scored_expected: list[tuple[float, str]] = []
    for expected_label in expected_labels:
        score = score_by_key.get(_label_key(expected_label))
        if score is not None:
            scored_expected.append((float(score.confidence), expected_label))
    if scored_expected:
        confidence, label = max(scored_expected, key=lambda item: item[0])
        return {
            "label": label,
            "confidence": to_python_scalar(confidence),
            "expected_label": label,
            "raw_label": raw_label,
            "label_source": "expected_score",
            "label_corrected": True,
        }

    fallback_index = min(max(0, item_index - 1), len(expected_labels) - 1)
    fallback_label = expected_labels[fallback_index]
    return {
        "label": fallback_label,
        "confidence": to_python_scalar(prediction.confidence),
        "expected_label": fallback_label,
        "raw_label": raw_label,
        "label_source": "expected_order",
        "label_corrected": True,
    }


def generate_pdf_report(
    states: list[dict[str, Any]],
    pledge_metadata: dict[str, Any] | None = None,
) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=(0.2, 0.2, 0.2),
        spaceAfter=30,
        alignment=1,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=(0.3, 0.3, 0.3),
        spaceAfter=12,
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
    )

    if pledge_metadata is None and states:
        pledge_metadata = load_pledge_metadata(states[0].get("pledge_id")) or {}

    pledge_id = (
        (pledge_metadata or {}).get("pledge_id")
        or (states[0].get("pledge_id") if states else None)
        or "N/A"
    )
    pledge_started_at = (
        (pledge_metadata or {}).get("started_at")
        or (states[0].get("pledge_started_at") if states else None)
        or "N/A"
    )
    result_generated_at = (
        (pledge_metadata or {}).get("result_generated_at")
        or now_stamp()
    )
    states_by_index = _summary_states_by_index(states)
    declared_count = (pledge_metadata or {}).get("jewel_count")
    try:
        jewel_count = int(declared_count or 0)
    except Exception:
        jewel_count = 0
    if jewel_count <= 0:
        jewel_count = max(states_by_index.keys(), default=len(states))
    session_ids = ", ".join(str(s.get("session_id") or "-") for s in states) or "N/A"

    story.append(Paragraph("Jewelry Analysis Report", title_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"<b>Pledge ID:</b> {_pdf_text(pledge_id)}", normal_style))
    story.append(Paragraph(f"<b>Pledge Started:</b> {_pdf_text(pledge_started_at)}", normal_style))
    story.append(Paragraph(f"<b>Report Generated:</b> {_pdf_text(result_generated_at)}", normal_style))
    story.append(Paragraph(f"<b>User Entered Jewel Count:</b> {_pdf_text(jewel_count)}", normal_style))
    story.append(Paragraph(f"<b>Session ID(s):</b> {_pdf_text(session_ids)}", normal_style))
    story.append(Spacer(1, 0.25 * inch))

    count_verification = (pledge_metadata or {}).get("jewel_count_verification") or {}
    packet_sealing = (pledge_metadata or {}).get("packet_sealing") or {}
    striping = packet_sealing.get("striping") or {}
    if packet_sealing.get("skipped"):
        strip_status = "Skipped"
    elif striping.get("sealed") is True:
        strip_status = "Closed / sealed"
    elif striping.get("sealed") is False:
        strip_status = "Not closed / not sealed"
    else:
        strip_status = "Not completed"
    if count_verification.get("skipped"):
        count_status = "Skipped"
    elif count_verification:
        count_status = (
            f"{count_verification.get('predicted_count', '-')} detected / "
            f"{count_verification.get('user_entered_count', jewel_count)} expected"
        )
    else:
        count_status = "Not captured"

    overview_table = Table(
        [
            ["TOTAL JEWELS", "FINAL COUNT", "STRIP STATUS"],
            [str(jewel_count), count_status, strip_status],
        ],
        colWidths=[1.4 * inch, 2.7 * inch, 2.4 * inch],
    )
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), (0.12, 0.29, 0.48)),
        ("TEXTCOLOR", (0, 0), (-1, 0), (1, 1, 1)),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, 1), 12),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.6, (0.65, 0.68, 0.74)),
        ("BACKGROUND", (0, 1), (-1, 1), (0.94, 0.97, 1.0)),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 0.2 * inch))

    summary_data = [["Jewel", "Jewel Type", "Actual Weight", "Acid Test", "Skipped Stages", "Risk"]]
    for index in range(1, jewel_count + 1):
        state = states_by_index.get(index)
        if not state:
            summary_data.append([f"Jewel {index}", "-", "-", "-", "-", "-"])
            continue
        stones = state.get("stone_detection") or {}
        reflection_risk = any(
            bool(
                (stones.get(key) or {}).get("reflection_risk")
                or (stones.get(key) or {}).get("reflection_flagged")
            )
            for key in ("main", "side")
        )
        risk_jewel = bool(
            (state.get("final") or {}).get("risk_jewel")
            or reflection_risk
        )
        actual_weight = (state.get("weight_details") or {}).get("jewel_weight_g")
        skipped_names = [
            str(value.get("display_name") or STAGE_DISPLAY_NAMES.get(key, key))
            for key, value in (state.get("stage_skips") or {}).items()
            if key not in {"final_count", "packet_sealing"}
        ]
        summary_data.append(
            [
                f"Jewel {index}",
                _jewel_type_for_state(state),
                f"{float(actual_weight):.2f} g" if actual_weight is not None else "-",
                _acid_status_for_state(state),
                ", ".join(skipped_names) if skipped_names else "None",
                "RISK" if risk_jewel else "NORMAL",
            ]
        )

    summary_table = Table(
        summary_data,
        colWidths=[0.65 * inch, 1.35 * inch, 0.85 * inch, 0.9 * inch, 1.65 * inch, 0.8 * inch],
    )
    summary_style = [
        ("BACKGROUND", (0, 0), (-1, 0), (0.85, 0.88, 0.94)),
        ("TEXTCOLOR", (0, 0), (-1, 0), (0, 0, 0)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("ALIGN", (4, 1), (4, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, (0.65, 0.68, 0.74)),
    ]
    if len(summary_data) > 1:
        summary_style.append(("BACKGROUND", (0, 1), (-1, -1), (0.97, 0.98, 1.0)))
    for row_index, row in enumerate(summary_data[1:], start=1):
        if row[-1] == "RISK":
            summary_style.extend([
                ("TEXTCOLOR", (-1, row_index), (-1, row_index), (0.8, 0.0, 0.0)),
                ("FONTNAME", (-1, row_index), (-1, row_index), "Helvetica-Bold"),
            ])
    summary_table.setStyle(TableStyle(summary_style))
    story.append(summary_table)
    story.append(PageBreak())
    
    for i, state in enumerate(states):
        if i > 0:
            story.append(PageBreak())
        
        session_id = state.get("session_id", "N/A")
        updated_at = state.get("updated_at", "N/A")
        jewel_index = int(state.get("jewel_index") or (i + 1))
        
        story.append(Paragraph(f"<b>Jewel {jewel_index} Analysis</b>", heading_style))
        story.append(Paragraph(f"<b>Session ID:</b> {_pdf_text(session_id)}", normal_style))
        story.append(Paragraph(f"<b>Updated:</b> {_pdf_text(updated_at)}", normal_style))
        story.append(Spacer(1, 0.3 * inch))

        classification = state.get("classification", {})
        if classification:
            story.append(Paragraph("Jewel Type", heading_style))
            predicted_label = classification.get("predicted_label", "N/A")
            confirmed_label = classification.get("confirmed_label", "N/A")
            
            story.append(Paragraph(f"<b>Predicted Jewel Type:</b> {_pdf_text(predicted_label)}", normal_style))
            story.append(Paragraph(f"<b>Confirmed Jewel Type:</b> {_pdf_text(confirmed_label)}", normal_style))
            
            if classification.get("cropped_preview"):
                story.append(Paragraph("Cropped Jewel Image", heading_style))
                try:
                    img_path = Path(classification["cropped_preview"]["path"])
                    if img_path.exists():
                        img = _pdf_image(img_path)
                        story.append(img)
                        story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.2 * inch))

        weight_summary = weight_summary_for_state(state)
        story.append(Paragraph("Weight Summary", heading_style))
        weight_rows = [["Weight Item", "Value", "Basis"]]
        if weight_summary["jewel_weight_g"] is not None:
            weight_rows.append(["Actual Jewel Weight", f"{weight_summary['jewel_weight_g']:.2f} g", "OCR scale reading"])
        if weight_summary["estimated_stone_weight_g"] is not None:
            minimum_g = weight_summary["estimated_stone_weight_minimum_g"]
            maximum_g = weight_summary["estimated_stone_weight_maximum_g"]
            minimum_g = weight_summary["estimated_stone_weight_g"] if minimum_g is None else minimum_g
            maximum_g = weight_summary["estimated_stone_weight_g"] if maximum_g is None else maximum_g
            basis_parts = ["Estimated"]
            if weight_summary["weight_method"]:
                basis_parts.append(weight_summary["weight_method"])
            if weight_summary["stone_weight_calibration_applied"]:
                basis_parts.append("OCR-weight constrained")
            basis = "; ".join(basis_parts)
            weight_rows.append(["Stone Weight Range", f"{minimum_g:.2f}-{maximum_g:.2f} g", basis])
            weight_rows.append([
                "Typical Stone Weight",
                f"{weight_summary['estimated_stone_weight_g']:.2f} g",
                f"Confidence: {weight_summary['weight_confidence'] or 'Low'}",
            ])
        else:
            unavailable_basis = (
                "Unknown setting; visible area only"
                if weight_summary["stone_setting_profile"]
                == stone_area_calculator.STONE_SETTING_PROFILE_UNKNOWN
                else "Analysis skipped or metric unavailable"
            )
            weight_rows.append(["Stone Weight", "Unavailable", unavailable_basis])
        if weight_summary["appraiser_stone_weight_g"] is not None:
            weight_rows.append(["Appraiser Stone Weight", f"{weight_summary['appraiser_stone_weight_g']:.4f} g", "Appraiser input"])
        if weight_summary["tassel_present"]:
            weight_rows.append(["Tassel Region Weight", f"{weight_summary['estimated_tassel_weight_g']:.4f} g", "Estimated"])
        if weight_summary["estimated_total_deduction_g"] is not None:
            weight_rows.append(["Total Deduction", f"{weight_summary['estimated_total_deduction_g']:.4f} g", "Estimated stone + tassel"])
        if weight_summary["estimated_net_weight_g"] is not None:
            weight_rows.append(["Net Jewel Weight", f"{weight_summary['estimated_net_weight_g']:.4f} g", "Estimated"])
        if weight_summary["appraiser_net_weight_g"] is not None:
            weight_rows.append(["Net Weight Using Appraiser Stone Weight", f"{weight_summary['appraiser_net_weight_g']:.4f} g", "Appraiser stone + estimated tassel"])
        weight_table = Table(weight_rows, colWidths=[2.5 * inch, 1.4 * inch, 2.6 * inch])
        weight_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), (0.12, 0.29, 0.48)),
            ("TEXTCOLOR", (0, 0), (-1, 0), (1, 1, 1)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, (0.72, 0.77, 0.82)),
            ("BACKGROUND", (0, 1), (-1, -1), (0.97, 0.98, 0.99)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(weight_table)
        story.append(Paragraph(
            "Stone and tassel values are estimated weights and should be reviewed by the appraiser.",
            normal_style,
        ))
        weight_capture = (state.get("weight_extraction") or {}).get("captured_image")
        if weight_capture:
            try:
                weight_capture_path = Path(weight_capture["path"])
                if weight_capture_path.exists():
                    story.append(Paragraph("Weight Capture", heading_style))
                    story.append(_pdf_image(weight_capture_path))
            except Exception:
                pass
        story.append(Spacer(1, 0.2 * inch))

        skipped_stages = {
            key: value
            for key, value in (state.get("stage_skips") or {}).items()
            if key not in {"final_count", "packet_sealing"}
        }
        if skipped_stages:
            story.append(Paragraph("Skipped Stages", heading_style))
            for stage_key, skipped in skipped_stages.items():
                stage_name = skipped.get("display_name") or STAGE_DISPLAY_NAMES.get(stage_key, stage_key)
                story.append(Paragraph(
                    f"<b>{_pdf_text(stage_name)}:</b> Skipped at {_pdf_text(skipped.get('skipped_at', 'N/A'))}",
                    normal_style,
                ))
            story.append(Spacer(1, 0.2 * inch))

        dimension = state.get("dimension", {})
        if dimension.get("done"):
            story.append(PageBreak())
            story.append(Paragraph("Dimension Analysis", heading_style))
            
            if dimension.get("od_mm") is not None:
                story.append(Paragraph(f"<b>Outer Diameter (OD):</b> {dimension['od_mm']:.2f} mm", normal_style))
                story.append(Paragraph(f"<b>Inner Diameter (ID):</b> {dimension['id_mm']:.2f} mm", normal_style))
                story.append(Paragraph(f"<b>Wall Thickness:</b> {dimension['wall_thickness_mm']:.2f} mm", normal_style))
            else:
                story.append(Paragraph("Dimension measurement completed; metric values are unavailable.", normal_style))
            
            calibration = dimension.get("calibration", {})
            if calibration:
                method = calibration.get("method", "N/A")
                story.append(Paragraph(f"<b>Calibration Method:</b> {method}", normal_style))
                
                if method == "aruco":
                    story.append(Paragraph(f"<b>Marker ID:</b> {calibration.get('marker_id', 'N/A')}", normal_style))
                    story.append(Paragraph(f"<b>Marker Size:</b> {calibration.get('marker_length_mm', 0):.2f} x {calibration.get('marker_breadth_mm', 0):.2f} mm", normal_style))
                elif method == "line":
                    story.append(Paragraph(f"<b>Known Distance:</b> {calibration.get('known_distance_mm', 0):.2f} mm", normal_style))
            
            if dimension.get("result_image"):
                try:
                    img_path = Path(dimension["result_image"]["path"])
                    if img_path.exists():
                        img = _pdf_image(img_path)
                        story.append(img)
                        story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.2 * inch))

        segmentation = state.get("segmentation", {})
        if segmentation.get("done"):
            story.append(PageBreak())
            story.append(Paragraph("Jewellery Analysis", heading_style))
            
            story.append(Paragraph("Jewellery analysis completed.", normal_style))
            if segmentation.get("no_pendant"):
                story.append(Paragraph("Pendant excluded by operator feedback.", normal_style))
            elif segmentation.get("pendant_absent"):
                story.append(Paragraph("No distinct pendant region detected.", normal_style))
            if segmentation.get("no_tassel"):
                story.append(Paragraph("Tassel excluded by operator feedback.", normal_style))
            elif segmentation.get("tassel_absent"):
                story.append(Paragraph("No tassel region detected.", normal_style))
            bead_risk = segmentation.get("bead_risk")
            if _segmentation_bead_risk_high(segmentation):
                story.append(Paragraph(
                    '<font color="red"><b>RISK JEWEL: Round beads/decorative elements detected in chain</b></font>',
                    normal_style,
                ))
            elif bead_risk:
                story.append(Paragraph(f"<b>Chain Bead Risk:</b> {_pdf_text(bead_risk)}", normal_style))
            
            if segmentation.get("composite_layout"):
                try:
                    img_path = Path(segmentation["composite_layout"]["path"])
                    if img_path.exists():
                        img = _pdf_image(img_path)
                        story.append(img)
                        story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.2 * inch))

        stones = state.get("stone_detection", {})
        main_stones = stones.get("main")
        if main_stones:
            story.append(PageBreak())
            story.append(Paragraph("Stone Analysis - Main Image", heading_style))
            story.append(Paragraph(
                f"<b>Stone Setting Type:</b> "
                f"{_pdf_text(weight_summary['stone_setting_profile_label'])}",
                normal_style,
            ))
            
            story.append(Paragraph("Stone Surface Assessment", heading_style))
            main_surface = main_stones.get("stone_surface_risk") or {}
            main_surface_status = (
                main_stones.get("stone_surface_status")
                or main_surface.get("status")
                or "NO SIGNIFICANT STONES DETECTED"
            )
            story.append(Paragraph(
                f"<b>Status:</b> {_pdf_text(main_surface_status)}",
                normal_style,
            ))
            story.append(Paragraph(
                f"<b>Visible Stone Coverage:</b> "
                f"{float(main_stones.get('stone_surface_coverage_percent', main_stones.get('stone_percentage', 0.0))):.1f}%",
                normal_style,
            ))
            story.append(Paragraph(
                f"<b>Stone Instances:</b> {int(main_stones.get('stone_instance_count', 0))}",
                normal_style,
            ))
            main_weight = stone_area_calculator.calibrate_weight_estimate_to_jewel_weight(
                main_stones.get("weight_estimate") or {},
                weight_summary["jewel_weight_g"],
            )
            if main_weight.get("success"):
                main_minimum_g = float(
                    main_weight.get(
                        "estimated_total_minimum_g",
                        float(main_weight.get("estimated_total_minimum_ct", 0.0)) * 0.2,
                    )
                )
                main_maximum_g = float(
                    main_weight.get(
                        "estimated_total_maximum_g",
                        float(main_weight.get("estimated_total_maximum_ct", 0.0)) * 0.2,
                    )
                )
                story.append(Paragraph(
                    f"<b>Estimated Stone Weight Range:</b> "
                    f"{main_minimum_g:.2f}-{main_maximum_g:.2f} g",
                    normal_style,
                ))
                main_typical_g = float(
                    main_weight.get(
                        "estimated_total_typical_g",
                        main_weight.get("estimated_total_average_g", 0.0),
                    )
                )
                story.append(Paragraph(
                    f"<b>Typical Estimate:</b> {main_typical_g:.2f} g",
                    normal_style,
                ))
                story.append(Paragraph(
                    f"<b>Weight Confidence:</b> "
                    f"{_pdf_text(main_weight.get('weight_confidence', 'Low'))}",
                    normal_style,
                ))
                if main_weight.get("calibration_applied"):
                    story.append(Paragraph(
                        "Range constrained by the captured OCR jewel weight; raw model values remain in the saved analysis data.",
                        normal_style,
                    ))
            else:
                story.append(Paragraph(
                    "<b>Estimated Stone Weight:</b> Unavailable for the selected setting or metric calibration.",
                    normal_style,
                ))
            if main_stones.get("reflection_risk") or main_stones.get("reflection_flagged"):
                story.append(Paragraph(
                    '<font color="red"><b>RISK JEWEL: Dense reflection indicates possible '
                    'additional transparent/colorless gemstones.</b></font>',
                    normal_style,
                ))
                story.append(Paragraph(
                    f"<b>Reflection Analysis:</b> "
                    f"{_pdf_text(main_stones.get('reflection_summary', 'Dense reflection detected.'))}",
                    normal_style,
                ))
            
            summary_entries = main_stones.get("summary_entries", [])
            if summary_entries:
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph("<b>Stone Details:</b>", normal_style))
                
                table_data = [["Detected Stone Color", "Detected Regions"]]
                for entry in summary_entries:
                    table_data.append([
                        (
                            "Multicolor / Mixed Appearance"
                            if entry.get("color") == "Multicolor/Color-changing"
                            else entry.get("color", "N/A")
                        ),
                        str(entry.get("region_count", 0)),
                    ])
                
                table = Table(table_data, colWidths=[2.5 * inch, 1.5 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), (0.7, 0.7, 0.7)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), (0.95, 0.95, 0.95)),
                    ('GRID', (0, 0), (-1, -1), 1, (0.7, 0.7, 0.7)),
                ]))
                story.append(table)
                story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(
                "Stone weight is an approximate image-based estimate. Actual weight "
                "varies with hidden depth, cut and material density.",
                normal_style,
            ))
            
            if main_stones.get("gallery"):
                try:
                    img_path = Path(main_stones["gallery"]["path"])
                    if img_path.exists():
                        img = _pdf_image(img_path)
                        story.append(img)
                        story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.2 * inch))

        side_stones = stones.get("side")
        if side_stones:
            story.append(PageBreak())
            story.append(Paragraph("Stone Analysis - Side Image", heading_style))
            story.append(Paragraph(
                f"<b>Stone Setting Type:</b> "
                f"{_pdf_text(weight_summary['stone_setting_profile_label'])}",
                normal_style,
            ))
            
            story.append(Paragraph("Stone Surface Assessment", heading_style))
            side_surface = side_stones.get("stone_surface_risk") or {}
            side_surface_status = (
                side_stones.get("stone_surface_status")
                or side_surface.get("status")
                or "NO SIGNIFICANT STONES DETECTED"
            )
            story.append(Paragraph(
                f"<b>Status:</b> {_pdf_text(side_surface_status)}",
                normal_style,
            ))
            story.append(Paragraph(
                f"<b>Visible Stone Coverage:</b> "
                f"{float(side_stones.get('stone_surface_coverage_percent', side_stones.get('stone_percentage', 0.0))):.1f}%",
                normal_style,
            ))
            story.append(Paragraph(
                f"<b>Stone Instances:</b> {int(side_stones.get('stone_instance_count', 0))}",
                normal_style,
            ))
            side_weight = stone_area_calculator.calibrate_weight_estimate_to_jewel_weight(
                side_stones.get("weight_estimate") or {},
                weight_summary["jewel_weight_g"],
            )
            if side_weight.get("success"):
                side_minimum_g = float(
                    side_weight.get(
                        "estimated_total_minimum_g",
                        float(side_weight.get("estimated_total_minimum_ct", 0.0)) * 0.2,
                    )
                )
                side_maximum_g = float(
                    side_weight.get(
                        "estimated_total_maximum_g",
                        float(side_weight.get("estimated_total_maximum_ct", 0.0)) * 0.2,
                    )
                )
                story.append(Paragraph(
                    f"<b>Estimated Stone Weight Range:</b> "
                    f"{side_minimum_g:.2f}-{side_maximum_g:.2f} g",
                    normal_style,
                ))
                side_typical_g = float(
                    side_weight.get(
                        "estimated_total_typical_g",
                        side_weight.get("estimated_total_average_g", 0.0),
                    )
                )
                story.append(Paragraph(
                    f"<b>Typical Estimate:</b> {side_typical_g:.2f} g",
                    normal_style,
                ))
                story.append(Paragraph(
                    f"<b>Weight Confidence:</b> "
                    f"{_pdf_text(side_weight.get('weight_confidence', 'Low'))}",
                    normal_style,
                ))
                if side_weight.get("calibration_applied"):
                    story.append(Paragraph(
                        "Range constrained by the captured OCR jewel weight; raw model values remain in the saved analysis data.",
                        normal_style,
                    ))
            else:
                story.append(Paragraph(
                    "<b>Estimated Stone Weight:</b> Unavailable for the selected setting or metric calibration.",
                    normal_style,
                ))
            if side_stones.get("reflection_risk") or side_stones.get("reflection_flagged"):
                story.append(Paragraph(
                    '<font color="red"><b>RISK JEWEL: Dense reflection indicates possible '
                    'additional transparent/colorless gemstones.</b></font>',
                    normal_style,
                ))
                story.append(Paragraph(
                    f"<b>Reflection Analysis:</b> "
                    f"{_pdf_text(side_stones.get('reflection_summary', 'Dense reflection detected.'))}",
                    normal_style,
                ))
            
            summary_entries = side_stones.get("summary_entries", [])
            if summary_entries:
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph("<b>Stone Details:</b>", normal_style))
                
                table_data = [["Detected Stone Color", "Detected Regions"]]
                for entry in summary_entries:
                    table_data.append([
                        (
                            "Multicolor / Mixed Appearance"
                            if entry.get("color") == "Multicolor/Color-changing"
                            else entry.get("color", "N/A")
                        ),
                        str(entry.get("region_count", 0)),
                    ])
                
                table = Table(table_data, colWidths=[2.5 * inch, 1.5 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), (0.7, 0.7, 0.7)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), (0.95, 0.95, 0.95)),
                    ('GRID', (0, 0), (-1, -1), 1, (0.7, 0.7, 0.7)),
                ]))
                story.append(table)
                story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(
                "Stone weight is an approximate image-based estimate. Actual weight "
                "varies with hidden depth, cut and material density.",
                normal_style,
            ))
            
            if side_stones.get("gallery"):
                try:
                    img_path = Path(side_stones["gallery"]["path"])
                    if img_path.exists():
                        img = _pdf_image(img_path)
                        story.append(img)
                        story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.2 * inch))

        purity = state.get("purity_test", {})
        if purity and (
            purity.get("started_at")
            or purity.get("running")
            or purity.get("stopped_at")
            or purity.get("acid_ok")
            or purity.get("skipped")
        ):
            story.append(PageBreak())
            story.append(Paragraph("Acid Test", heading_style))
            if purity.get("skipped"):
                story.append(Paragraph("<b>Status:</b> Skipped", normal_style))
                story.append(Paragraph(f"<b>Skipped At:</b> {_pdf_text(purity.get('skipped_at', 'N/A'))}", normal_style))
            else:
                story.append(Paragraph(f"<b>Status:</b> {_pdf_text(purity.get('status', 'N/A'))}", normal_style))
                story.append(Paragraph(f"<b>Stage:</b> {_pdf_text(purity.get('stage', 'N/A'))}", normal_style))
                story.append(Paragraph(f"<b>Result:</b> {_pdf_text(purity.get('result', 'N/A'))}", normal_style))
                story.append(Paragraph(f"<b>Started:</b> {_pdf_text(purity.get('started_at', 'N/A'))}", normal_style))
                story.append(Paragraph(f"<b>Stopped:</b> {_pdf_text(purity.get('stopped_at', 'N/A'))}", normal_style))
                story.append(Paragraph(f"<b>Completed:</b> {_pdf_text(purity.get('completed_at', 'N/A'))}", normal_style))
                story.append(Paragraph(f"<b>Rubbing OK:</b> {'Yes' if purity.get('rubbing_ok') else 'No'}", normal_style))
                story.append(Paragraph(f"<b>Acid OK:</b> {'Yes' if purity.get('acid_ok') else 'No'}", normal_style))
            if purity.get("error"):
                story.append(Paragraph(f"<b>Error:</b> {_pdf_text(purity['error'])}", normal_style))

            purity_artifacts = [] if purity.get("skipped") else [
                purity.get("rubbing_image"),
                purity.get("rubbing_zoom_image"),
                purity.get("acid_success_image") or purity.get("final_image"),
                purity.get("acid_zoom_image"),
            ]
            for artifact in purity_artifacts:
                if not artifact:
                    continue
                try:
                    img_path = Path(artifact["path"])
                    if img_path.exists():
                        img = _pdf_image(img_path)
                        story.append(img)
                        story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.2 * inch))

        final = state.get("final", {})
        if final.get("ready"):
            story.append(PageBreak())
            story.append(Paragraph("Final Summary", heading_style))
            story.append(Paragraph(f"<b>{final.get('headline', 'N/A')}</b>", normal_style))
            
            lines = final.get("lines", [])
            if lines:
                story.append(Spacer(1, 0.1 * inch))
                for line in lines:
                    story.append(Paragraph(f"• {line}", normal_style))
            if final.get("risk_jewel"):
                reasons = ", ".join(final.get("risk_reasons") or ["risk condition detected"])
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph(
                    f'<font color="red"><b>RISK JEWEL: {_pdf_text(reasons)}</b></font>',
                    normal_style,
                ))
            
            story.append(Spacer(1, 0.2 * inch))

    count_verification = (pledge_metadata or {}).get("jewel_count_verification")
    packet_sealing = (pledge_metadata or {}).get("packet_sealing") or {}
    if count_verification or packet_sealing.get("video"):
        story.append(PageBreak())
        story.append(Paragraph("Pledge Closure", heading_style))
        if count_verification:
            story.append(Paragraph("Final Jewel Count Capture", heading_style))
            if count_verification.get("skipped"):
                story.append(Paragraph("<b>Status:</b> Skipped", normal_style))
                story.append(Paragraph(
                    f"<b>Skipped At:</b> {_pdf_text(count_verification.get('skipped_at', 'N/A'))}",
                    normal_style,
                ))
            else:
                story.append(Paragraph(
                    f"<b>User Entered Jewel Count:</b> "
                    f"{_pdf_text(count_verification.get('user_entered_count', jewel_count))}",
                    normal_style,
                ))
                story.append(Paragraph(
                    f"<b>Predicted Jewel Count:</b> "
                    f"{_pdf_text(count_verification.get('predicted_count', '-'))}",
                    normal_style,
                ))
                match = count_verification.get("match")
                if match is not None:
                    story.append(Paragraph(
                        f"<b>Count Match:</b> {'Yes' if match else 'No'}",
                        normal_style,
                    ))
                story.append(Paragraph(
                    f"<b>Captured At:</b> {_pdf_text(count_verification.get('captured_at', 'N/A'))}",
                    normal_style,
                ))
                artifact = count_verification.get("result_image")
                if artifact:
                    try:
                        img_path = Path(artifact["path"])
                        if img_path.exists():
                            story.append(_pdf_image(img_path))
                            story.append(Spacer(1, 0.1 * inch))
                    except Exception:
                        pass

        video = packet_sealing.get("video") if isinstance(packet_sealing, dict) else None
        if packet_sealing.get("skipped"):
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph("Packet Sealing Video", heading_style))
            story.append(Paragraph("<b>Status:</b> Skipped", normal_style))
            story.append(Paragraph(
                f"<b>Skipped At:</b> {_pdf_text(packet_sealing.get('skipped_at', 'N/A'))}",
                normal_style,
            ))
        if video:
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph("Packet Sealing", heading_style))
            story.append(Paragraph(
                f"<b>Recording Started:</b> {_pdf_text(packet_sealing.get('started_at', 'N/A'))}",
                normal_style,
            ))
            story.append(Paragraph(
                f"<b>Recording Stopped:</b> {_pdf_text(packet_sealing.get('stopped_at', 'N/A'))}",
                normal_style,
            ))
            av1 = packet_sealing.get("av1") or {}
            story.append(Paragraph(
                f"<b>Video File:</b> {_pdf_text(video.get('name', 'packet_sealing.mp4'))}",
                normal_style,
            ))
            if av1:
                story.append(Paragraph(
                    f"<b>AV1 Compression:</b> "
                    f"{'Applied' if av1.get('applied') else 'Fallback kept'}"
                    f"{' (' + _pdf_text(av1.get('encoder')) + ')' if av1.get('encoder') else ''}",
                    normal_style,
                ))
            striping = packet_sealing.get("striping") or {}
            sealed = striping.get("sealed")
            if sealed is not None:
                story.append(Paragraph(
                    f"<b>Strip Removal Seal Status:</b> {'Sealed' if sealed else 'Not sealed'}",
                    normal_style,
                ))
                if striping.get("reason"):
                    story.append(Paragraph(
                        f"<b>Strip Check:</b> {_pdf_text(striping['reason'])}",
                        normal_style,
                    ))
                evidence = striping.get("evidence_image")
                if evidence:
                    try:
                        img_path = Path(evidence["path"])
                        if img_path.exists():
                            story.append(_pdf_image(img_path))
                            story.append(Spacer(1, 0.1 * inch))
                    except Exception:
                        pass

    doc.build(story)
    buffer.seek(0)
    return buffer


def snapshot_state() -> dict[str, Any]:
    with STATE_LOCK:
        state = ensure_state()
        recorder = PACKET_RECORDER
        if (
            recorder is not None
            and recorder.is_recording()
            and not state.get("pledge_id")
            and getattr(recorder, "_pledge_id", None)
        ):
            recorder_pledge_id = str(recorder._pledge_id)
            metadata = get_or_create_pledge_metadata(recorder_pledge_id)
            apply_pledge_metadata_to_state(state, metadata)
        refresh_purity_state(state)
        build_final_summary(state)
        update_pledge_progress(state)
        if (
            recorder is not None
            and state.get("pledge_id")
            and str(getattr(recorder, "_pledge_id", "") or "") == str(state["pledge_id"])
        ):
            state["packet_sealing"] = recorder.snapshot()
        state_copy = copy.deepcopy(state)
        
        session_id = state_copy.get("session_id")
        if session_id:
            try:
                session_dir = RUNTIME_DIR / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                state_file = session_dir / "state.json"
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state_copy, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save state to disk for session {session_id}: {e}")
                
        return state_copy


def parse_post_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def to_python_scalar(val: Any) -> float | int | None:
    """Safely convert numpy arrays and other types to Python scalar values."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        # Handle numpy arrays by extracting the scalar
        if val.size == 0:
            return None
        if val.size == 1:
            return float(val.item())
        # For multi-element arrays, take the first element
        return float(val.flat[0])
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return None
        return float(val[0])
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fail(message: str, status_code: int = 400):
    response = jsonify({"ok": False, "error": message})
    response.status_code = status_code
    return response


def detect_aruco_marker(
    image_bgr: np.ndarray,
    dict_name: str,
    roi: dict[str, int] | None = None,
    expected_marker_id: int | None = None,
) -> dict[str, Any]:
    if not ARUCO_AVAILABLE:
        raise RuntimeError(
            "OpenCV ArUco/AprilTag support is unavailable. Install an "
            "opencv-contrib-python build that matches this environment."
        )
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"Unsupported ArUco dictionary: {dict_name}")

    dict_id = ARUCO_DICTS[dict_name]
    height, width = image_bgr.shape[:2]
    detection_regions: list[dict[str, int] | None] = []
    if roi:
        padding = max(8, int(round(max(roi["w"], roi["h"]) * 0.12)))
        padded_roi = normalize_rect(
            {
                "x": roi["x"] - padding,
                "y": roi["y"] - padding,
                "w": roi["w"] + (padding * 2),
                "h": roi["h"] + (padding * 2),
            },
            width,
            height,
        )
        if padded_roi:
            detection_regions.append(padded_roi)
        detection_regions.append(None)
    else:
        detection_regions.append(None)

    aruco_dict = (
        cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "getPredefinedDictionary")
        else cv2.aruco.Dictionary_get(dict_id)
    )

    detected_corners = None
    detected_ids = None
    detected_region = None
    for detection_region in detection_regions:
        working = crop_image(image_bgr, detection_region)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        gray_variants = [gray]
        equalized = cv2.equalizeHist(gray)
        if not np.array_equal(equalized, gray):
            gray_variants.append(equalized)

        for candidate_gray in gray_variants:
            if hasattr(cv2.aruco, "ArucoDetector"):
                params = cv2.aruco.DetectorParameters()
                if (
                    hasattr(params, "cornerRefinementMethod")
                    and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX")
                ):
                    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(candidate_gray)
            else:
                params = cv2.aruco.DetectorParameters_create()
                if (
                    hasattr(params, "cornerRefinementMethod")
                    and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX")
                ):
                    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                corners, ids, _ = cv2.aruco.detectMarkers(
                    candidate_gray,
                    aruco_dict,
                    parameters=params,
                )

            if ids is not None and len(corners) > 0:
                detected_corners = corners
                detected_ids = ids
                detected_region = detection_region
                break
        if detected_ids is not None:
            break

    if detected_ids is None or not detected_corners:
        location = "the selected ROI or full image" if roi else "the image"
        raise RuntimeError(f"No AprilTag marker was detected in {location}.")

    candidate_indexes = list(range(len(detected_corners)))
    if expected_marker_id is not None:
        matching = [
            index
            for index in candidate_indexes
            if int(detected_ids[index][0]) == int(expected_marker_id)
        ]
        if not matching:
            found = ", ".join(str(int(value[0])) for value in detected_ids)
            raise RuntimeError(
                f"AprilTag ID {expected_marker_id} was not detected. Found: {found or 'none'}."
            )
        candidate_indexes = matching
    marker_index = max(
        candidate_indexes,
        key=lambda index: abs(
            cv2.contourArea(detected_corners[index][0].astype(np.float32))
        ),
    )
    marker = detected_corners[marker_index][0].astype(np.float32)
    offset_x = int(detected_region["x"]) if detected_region else 0
    offset_y = int(detected_region["y"]) if detected_region else 0
    marker[:, 0] += offset_x
    marker[:, 1] += offset_y

    side_top = np.linalg.norm(marker[0] - marker[1])
    side_right = np.linalg.norm(marker[1] - marker[2])
    side_bottom = np.linalg.norm(marker[2] - marker[3])
    side_left = np.linalg.norm(marker[3] - marker[0])

    return {
        "marker_id": int(detected_ids[marker_index][0]),
        "corners": marker.tolist(),
        "horizontal_px": float((side_top + side_bottom) / 2.0),
        "vertical_px": float((side_left + side_right) / 2.0),
    }


def undistort_captured_still(
    image_bgr: np.ndarray,
    calibration_config: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    camera_matrix = calibration_config.get("camera_matrix")
    dist_coeffs = calibration_config.get("dist_coeffs")
    if camera_matrix is None or dist_coeffs is None:
        return image_bgr.copy(), False
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if matrix.shape != (3, 3) or distortion.size < 4:
        return image_bgr.copy(), False
    return cv2.undistort(image_bgr, matrix, distortion), True


def rectify_with_four_apriltags(
    image_bgr: np.ndarray,
    calibration_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not calibration_config.get("four_marker_enabled"):
        return image_bgr, None
    marker_ids = calibration_config.get("four_marker_ids") or []
    width_mm = float(calibration_config.get("marker_center_width_mm", 0.0) or 0.0)
    height_mm = float(calibration_config.get("marker_center_height_mm", 0.0) or 0.0)
    if len(marker_ids) != 4 or width_mm <= 0 or height_mm <= 0:
        raise ValueError(
            "Four-marker rectification requires four marker IDs and measured "
            "marker-center width/height in millimetres."
        )

    markers = [
        detect_aruco_marker(
            image_bgr,
            calibration_config["aruco_dict"],
            roi=None,
            expected_marker_id=int(marker_id),
        )
        for marker_id in marker_ids
    ]
    source_points = np.asarray(
        [
            np.asarray(marker["corners"], dtype=np.float32).mean(axis=0)
            for marker in markers
        ],
        dtype=np.float32,
    )
    top_px = float(np.linalg.norm(source_points[1] - source_points[0]))
    bottom_px = float(np.linalg.norm(source_points[2] - source_points[3]))
    left_px = float(np.linalg.norm(source_points[3] - source_points[0]))
    right_px = float(np.linalg.norm(source_points[2] - source_points[1]))
    px_per_mm_x = max(0.1, ((top_px + bottom_px) / 2.0) / width_mm)
    px_per_mm_y = max(0.1, ((left_px + right_px) / 2.0) / height_mm)
    output_width = max(32, int(round(width_mm * px_per_mm_x)))
    output_height = max(32, int(round(height_mm * px_per_mm_y)))
    target_points = np.asarray(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_points, target_points)
    rectified = cv2.warpPerspective(
        image_bgr,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    scale_x = width_mm / max(1.0, output_width - 1.0)
    scale_y = height_mm / max(1.0, output_height - 1.0)
    camera_to_bed = float(calibration_config.get("camera_to_bed_mm", 0.0) or 0.0)
    stone_height = float(calibration_config.get("nominal_stone_height_mm", 0.0) or 0.0)
    height_factor = (
        max(0.5, min(1.0, (camera_to_bed - stone_height) / camera_to_bed))
        if camera_to_bed > 0 and stone_height > 0
        else 1.0
    )
    calibration = {
        "method": "four_marker_homography",
        "dict_name": calibration_config["aruco_dict"],
        "marker_ids": [int(value) for value in marker_ids],
        "marker_center_width_mm": width_mm,
        "marker_center_height_mm": height_mm,
        "horizontal_scale": scale_x,
        "vertical_scale": scale_y,
        "effective_horizontal_scale": scale_x * height_factor,
        "effective_vertical_scale": scale_y * height_factor,
        "height_scale_factor": height_factor,
        "camera_to_bed_mm": camera_to_bed or None,
        "nominal_stone_height_mm": stone_height,
        "homography": homography.tolist(),
        "source_marker_centers": source_points.tolist(),
        "rectified_size": [output_width, output_height],
    }
    return rectified, calibration


def translate_corners_to_crop(
    corners: list[list[float]] | None,
    crop_rect: dict[str, int] | None,
    crop_shape: tuple[int, int, int],
) -> list[list[float]] | None:
    if not corners:
        return None

    translated = np.array(corners, dtype=np.float32)
    if crop_rect:
        translated[:, 0] -= float(crop_rect["x"])
        translated[:, 1] -= float(crop_rect["y"])

    height, width = crop_shape[:2]
    if (
        float(np.max(translated[:, 0])) < 0
        or float(np.max(translated[:, 1])) < 0
        or float(np.min(translated[:, 0])) >= width
        or float(np.min(translated[:, 1])) >= height
    ):
        return None
    return translated.tolist()


def compute_scale_mm_per_px(
    image_bgr: np.ndarray,
    payload: dict[str, Any],
    processing_roi: dict[str, int] | None,
    working_aruco_roi: dict[str, int] | None,
) -> tuple[float, dict[str, Any]]:
    method = str(payload.get("calibration_method", "aruco")).strip().lower()

    if method == "manual":
        manual_scale = float(payload.get("manual_scale_mm_per_px", 0.0))
        if manual_scale <= 0:
            raise ValueError("Manual mm/px must be greater than zero.")
        return manual_scale, {
            "method": "manual",
            "scale_mm_per_px": manual_scale,
        }

    if method == "line":
        points = payload.get("line_points") or []
        if len(points) != 2:
            raise ValueError("Draw two calibration points for manual line calibration.")
        known_mm = float(payload.get("known_distance_mm", 0.0))
        if known_mm <= 0:
            raise ValueError("Known distance must be greater than zero.")

        translated = translate_points_to_crop(points, processing_roi)
        (x1, y1), (x2, y2) = translated
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if pixel_distance <= 1.0:
            raise ValueError("Calibration points are too close together.")
        scale = known_mm / pixel_distance
        return scale, {
            "method": "line",
            "known_distance_mm": known_mm,
            "pixel_distance": pixel_distance,
            "scale_mm_per_px": scale,
            "line_points": translated,
        }

    dict_name = str(payload.get("aruco_dict", "AprilTag_36h11")).strip() or "AprilTag_36h11"
    marker_length_mm = float(payload.get("marker_length_mm", 20.0))
    marker_breadth_mm = float(payload.get("marker_breadth_mm", marker_length_mm))
    if marker_length_mm <= 0 or marker_breadth_mm <= 0:
        raise ValueError("Marker length and breadth must be greater than zero.")

    marker_id = payload.get("marker_id")
    aruco = detect_aruco_marker(
        image_bgr,
        dict_name,
        roi=working_aruco_roi,
        expected_marker_id=int(marker_id) if marker_id is not None else None,
    )
    scale_horizontal = marker_length_mm / aruco["horizontal_px"]
    scale_vertical = marker_breadth_mm / aruco["vertical_px"]
    scale = (scale_horizontal + scale_vertical) / 2.0
    return scale, {
        "method": "aruco",
        "dict_name": dict_name,
        "marker_length_mm": marker_length_mm,
        "marker_breadth_mm": marker_breadth_mm,
        "marker_id": aruco["marker_id"],
        "horizontal_px": aruco["horizontal_px"],
        "vertical_px": aruco["vertical_px"],
        "horizontal_scale": scale_horizontal,
        "vertical_scale": scale_vertical,
        "scale_mm_per_px": scale,
        "corners": aruco["corners"],
    }


def april_tag_calibration_config(
    payload: dict[str, Any],
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = {
        **(PERSISTENT_ROIS.get("calibration_config") or {}),
        **(fallback or {}),
        **(payload or {}),
    }
    return normalize_metric_calibration_config(merged)


def detect_stone_area_calibration(
    image_bgr: np.ndarray,
    config: dict[str, Any],
    aruco_roi: dict[str, int] | None,
) -> dict[str, Any]:
    calibration_payload = {
        "calibration_method": "aruco",
        **config,
    }
    _scale, calibration = compute_scale_mm_per_px(
        image_bgr=image_bgr,
        payload=calibration_payload,
        processing_roi=None,
        working_aruco_roi=aruco_roi,
    )
    camera_to_bed = float(config.get("camera_to_bed_mm", 0.0) or 0.0)
    stone_height = float(config.get("nominal_stone_height_mm", 0.0) or 0.0)
    height_factor = 1.0
    if camera_to_bed > 0 and stone_height > 0:
        height_factor = max(0.5, min(1.0, (camera_to_bed - stone_height) / camera_to_bed))
    calibration["camera_to_bed_mm"] = camera_to_bed or None
    calibration["nominal_stone_height_mm"] = stone_height
    calibration["height_scale_factor"] = height_factor
    calibration["effective_horizontal_scale"] = (
        float(calibration["horizontal_scale"]) * height_factor
    )
    calibration["effective_vertical_scale"] = (
        float(calibration["vertical_scale"]) * height_factor
    )
    return calibration


def _calibration_area_per_pixel(calibration: dict[str, Any] | None) -> float | None:
    if not calibration:
        return None
    horizontal = to_python_scalar(
        calibration.get("effective_horizontal_scale")
        or calibration.get("horizontal_scale")
        or calibration.get("scale_mm_per_px")
    )
    vertical = to_python_scalar(
        calibration.get("effective_vertical_scale")
        or calibration.get("vertical_scale")
        or calibration.get("scale_mm_per_px")
    )
    if horizontal is None or vertical is None or horizontal <= 0 or vertical <= 0:
        return None
    return float(horizontal * vertical)


def _measurement_scale_for_calibration(
    calibration: dict[str, Any] | None,
) -> dict[str, float] | None:
    if not calibration:
        return None
    scale_x = to_python_scalar(
        calibration.get("effective_horizontal_scale")
        or calibration.get("horizontal_scale")
        or calibration.get("scale_mm_per_px")
    )
    scale_y = to_python_scalar(
        calibration.get("effective_vertical_scale")
        or calibration.get("vertical_scale")
        or calibration.get("scale_mm_per_px")
    )
    if scale_x is None or scale_y is None or scale_x <= 0 or scale_y <= 0:
        return None
    return {
        "mm_per_pixel_x": float(scale_x),
        "mm_per_pixel_y": float(scale_y),
    }


def stone_calibration_for_state(
    state: dict[str, Any],
    *,
    side_image: bool = False,
) -> dict[str, Any] | None:
    if side_image:
        side_calibration = (state.get("side_capture") or {}).get("stone_calibration")
        if _calibration_area_per_pixel(side_calibration):
            return side_calibration

    dimension_calibration = (state.get("dimension") or {}).get("calibration")
    if _calibration_area_per_pixel(dimension_calibration):
        return dimension_calibration

    source_calibration = (state.get("source") or {}).get("stone_calibration")
    if _calibration_area_per_pixel(source_calibration):
        return source_calibration
    return None


def build_aruco_ignore_mask(
    image_shape: tuple[int, int, int],
    aruco_roi: dict[str, int] | None,
    aruco_corners: list[list[float]] | None = None,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if aruco_roi:
        x, y, w, h = rect_to_tuple(aruco_roi)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    if aruco_corners:
        corners = np.array(aruco_corners, dtype=np.float32)
        center = corners.mean(axis=0)
        padded = corners.copy()
        padded[:, 0] = center[0] + (padded[:, 0] - center[0]) * 1.30
        padded[:, 1] = center[1] + (padded[:, 1] - center[1]) * 1.30
        cv2.fillPoly(mask, [padded.astype(np.int32)], 255)

    return mask


def _threshold_jewelry_mask(
    image_bgr: np.ndarray,
    bg_is_white: bool,
    threshold_value: int = 220,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if bg_is_white:
        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        _, fixed = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )
        thresholded = cv2.bitwise_or(otsu, fixed)
    else:
        thresholded = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    thresholded = cv2.bitwise_or(thresholded, edges)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored_jewelry = (
        (saturation > 45)
        & (value > 28)
        & (
            ((hue >= 5) & (hue <= 110))
            | (hue >= 125)
        )
    ).astype(np.uint8) * 255
    colored_jewelry = cv2.dilate(
        colored_jewelry,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    thresholded = cv2.bitwise_or(thresholded, colored_jewelry)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    return (thresholded > 0).astype(np.uint8)


def _score_threshold_jewelry_mask(mask: np.ndarray) -> float:
    mask_u8 = (mask > 0).astype(np.uint8)
    mask_ratio = float(mask_u8.mean()) if mask_u8.size else 1.0
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    image_h, image_w = mask_u8.shape[:2]
    min_component_area = max(80, int(mask_u8.size * 0.0005))

    useful_components = 0
    largest_border_area = 0
    largest_inner_area = 0
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        touches_border = (
            x <= 1
            or y <= 1
            or (x + w) >= image_w - 1
            or (y + h) >= image_h - 1
        )
        if touches_border:
            largest_border_area = max(largest_border_area, area)
        else:
            useful_components += 1
            largest_inner_area = max(largest_inner_area, area)

    if useful_components == 0:
        return -1_000_000.0

    score = float(largest_inner_area) + useful_components * 250.0
    if mask_ratio < 0.002:
        score -= 100_000.0
    if mask_ratio > 0.45:
        score -= mask_ratio * mask_u8.size
    if largest_border_area > largest_inner_area:
        score -= largest_border_area * 0.75
    return score


def build_shared_jewelry_mask(
    image_bgr: np.ndarray,
    erase_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    corners = [
        image_bgr[0, 0],
        image_bgr[0, -1],
        image_bgr[-1, 0],
        image_bgr[-1, -1],
    ]
    preferred_bg_is_white = float(np.mean([corner.mean() for corner in corners])) > 180.0
    candidates: list[tuple[float, np.ndarray]] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for bg_is_white in (preferred_bg_is_white, not preferred_bg_is_white):
        candidate = _threshold_jewelry_mask(image_bgr, bg_is_white=bg_is_white)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidates.append((_score_threshold_jewelry_mask(candidate), candidate))

    jewelry_mask = max(candidates, key=lambda item: item[0])[1].astype(np.uint8)
    if erase_mask is not None and np.any(erase_mask):
        padded_erase = (erase_mask > 0).astype(np.uint8)
        erase_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        padded_erase = cv2.dilate(padded_erase, erase_kernel, iterations=1)
        jewelry_mask[padded_erase > 0] = 0

    preprocessed = np.full_like(image_bgr, 255)
    preprocessed[jewelry_mask > 0] = image_bgr[jewelry_mask > 0]
    return preprocessed, jewelry_mask


def _put_readable_text(
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(
        image_bgr,
        (max(0, x - 3), max(0, y - th - baseline - 5)),
        (min(image_bgr.shape[1] - 1, x + tw + 5), min(image_bgr.shape[0] - 1, y + baseline + 5)),
        (20, 24, 33),
        -1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _component_items_from_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    min_area_px: int,
    pad_px: int,
) -> list[dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    height, width = binary.shape[:2]
    items: list[dict[str, Any]] = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if w < 10 or h < 10:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        extent = area / max(1, w * h)
        if aspect > 10.0 and area < min_area_px * 6:
            continue
        if extent < 0.01 and area < min_area_px * 10:
            continue

        x1 = max(0, x - pad_px)
        y1 = max(0, y - pad_px)
        x2 = min(width, x + w + pad_px)
        y2 = min(height, y + h + pad_px)
        component_mask = labels[y1:y2, x1:x2] == idx
        crop = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
        crop_source = image_bgr[y1:y2, x1:x2]
        crop[component_mask] = crop_source[component_mask]
        items.append(
            {
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "area_px": area,
                "crop_bgr": crop,
            }
        )
    items.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    return items


def analyze_pledge_jewel_count_capture(
    pledge_id: str,
    frame_bgr: np.ndarray,
    declared_count: int | None,
    expected_labels: list[str] | None = None,
    processing_roi_raw: Any = None,
    aruco_roi_raw: Any = None,
) -> dict[str, Any]:
    if frame_bgr is None or frame_bgr.size == 0:
        raise RuntimeError("Live camera frame is not available for count capture.")

    media_dir = pledge_media_dir(pledge_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = save_bgr(media_dir / f"jewel_count_capture_{timestamp}.png", frame_bgr)

    height, width = frame_bgr.shape[:2]
    processing_roi = normalize_rect(
        processing_roi_raw if processing_roi_raw is not None else PERSISTENT_ROIS.get("processing_roi"),
        width,
        height,
    )
    aruco_roi = normalize_rect(
        aruco_roi_raw if aruco_roi_raw is not None else PERSISTENT_ROIS.get("aruco_roi"),
        width,
        height,
    )
    working = crop_image(frame_bgr, processing_roi)
    working_aruco_roi = translate_rect_to_crop(aruco_roi, processing_roi)
    ignore_mask = build_aruco_ignore_mask(working.shape, working_aruco_roi)
    cleaned = working.copy()
    cleaned[ignore_mask > 0] = 255
    preprocessed, mask = build_shared_jewelry_mask(cleaned, erase_mask=ignore_mask)

    mask_path = media_dir / f"jewel_count_mask_{timestamp}.png"
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    min_area = max(
        COUNT_CAPTURE_MIN_AREA_PX,
        int(mask.shape[0] * mask.shape[1] * COUNT_CAPTURE_MIN_AREA_RATIO),
    )
    raw_items = _component_items_from_mask(
        preprocessed,
        mask,
        min_area_px=min_area,
        pad_px=max(0, COUNT_CAPTURE_PADDING_PX),
    )

    items: list[dict[str, Any]] = []
    classifier = get_classifier() if raw_items else None
    expected_labels = list(expected_labels or [])
    remaining_expected_labels = list(expected_labels)
    for index, item in enumerate(raw_items, start=1):
        crop_rgb = cv2.cvtColor(item["crop_bgr"], cv2.COLOR_BGR2RGB)
        prediction = classifier.classify_image(PILImage.fromarray(crop_rgb))
        label_result = _count_capture_label_for_prediction(
            prediction,
            remaining_expected_labels,
            index,
        )
        used_expected = str(label_result.get("expected_label") or label_result.get("label") or "").strip()
        for expected_index, expected_label in enumerate(remaining_expected_labels):
            if _label_key(expected_label) == _label_key(used_expected):
                remaining_expected_labels.pop(expected_index)
                break
        bbox = dict(item["bbox"])
        full_bbox = dict(bbox)
        if processing_roi:
            full_bbox["x"] += int(processing_roi["x"])
            full_bbox["y"] += int(processing_roi["y"])
        items.append(
            {
                "index": index,
                "label": label_result["label"],
                "confidence": label_result["confidence"],
                "raw_label": label_result["raw_label"],
                "expected_label": label_result["expected_label"],
                "label_source": label_result["label_source"],
                "label_corrected": bool(label_result["label_corrected"]),
                "gallery_match": bool(prediction.gallery_match),
                "gallery_similarity": to_python_scalar(prediction.gallery_similarity),
                "is_gold_jewelry": bool(getattr(prediction, "is_gold_jewelry", True)),
                "gold_verification_reason": str(
                    getattr(prediction, "gold_verification_reason", "")
                ),
                "bbox": bbox,
                "full_bbox": full_bbox,
                "area_px": int(item["area_px"]),
            }
        )

    annotated = frame_bgr.copy()
    draw_labeled_rect(annotated, processing_roi, "Count ROI", (37, 99, 235))
    draw_labeled_rect(annotated, aruco_roi, "Aruco Ignore", (22, 163, 74))
    for item in items:
        bbox = item["full_bbox"]
        x, y, w, h = rect_to_tuple(bbox)
        color = (20, 184, 166) if item["is_gold_jewelry"] else (0, 0, 220)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label = str(item["label"] or "").strip() or "Jewel"
        _put_readable_text(
            annotated,
            label[:44],
            (x, max(24, y - 8)),
            0.62,
            (255, 255, 255),
            2,
        )

    declared_text = declared_count if declared_count is not None else "-"
    _put_readable_text(
        annotated,
        f"Predicted jewel count: {len(items)} | User entered: {declared_text}",
        (24, 44),
        0.85,
        (255, 255, 255),
        2,
    )
    result_path = save_bgr(media_dir / f"jewel_count_prediction_{timestamp}.png", annotated)

    verification = {
        "captured_at": now_stamp(),
        "user_entered_count": declared_count,
        "predicted_count": len(items),
        "match": (
            bool(declared_count is not None and int(declared_count) == len(items))
            if declared_count is not None
            else None
        ),
        "processing_roi": processing_roi,
        "aruco_roi": aruco_roi,
        "min_area_px": min_area,
        "expected_labels": expected_labels,
        "items": items,
        "captured_image": pledge_artifact_payload(pledge_id, original_path),
        "mask_image": pledge_artifact_payload(pledge_id, mask_path),
        "result_image": pledge_artifact_payload(pledge_id, result_path),
    }
    return verification


def load_or_create_shared_jewelry_input(
    state: dict[str, Any],
) -> tuple[Path, np.ndarray, np.ndarray]:
    source = state.get("source") or {}
    image_artifact = source.get("preprocessed_image")
    mask_artifact = source.get("preprocessed_mask")
    if image_artifact and mask_artifact:
        image_path = Path(image_artifact.get("path", ""))
        mask_path = Path(mask_artifact.get("path", ""))
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is not None and mask is not None:
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image_bgr.shape[1], image_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            return image_path, image_bgr, (mask > 0).astype(np.uint8)

    working_artifact = source.get("working_image")
    if not working_artifact or not working_artifact.get("path"):
        raise RuntimeError("Capture or upload the main jewelry image first.")
    working_path = Path(working_artifact["path"])
    working_bgr = cv2.imread(str(working_path), cv2.IMREAD_COLOR)
    if working_bgr is None:
        raise RuntimeError(f"Could not load the saved working image: {working_path}")

    ignore_mask = build_aruco_ignore_mask(
        working_bgr.shape,
        source.get("working_aruco_roi"),
    )
    cleaned_working = working_bgr.copy()
    cleaned_working[ignore_mask > 0] = 255
    preprocessed, mask = build_shared_jewelry_mask(
        cleaned_working,
        erase_mask=ignore_mask,
    )

    source_dir = session_dir_for(state) / "source"
    preprocessed_path = save_bgr(source_dir / "preprocessed.png", preprocessed)
    mask_path = source_dir / "preprocessed_mask.png"
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    source["preprocessed_image"] = artifact_payload(state, preprocessed_path)
    source["preprocessed_mask"] = artifact_payload(state, mask_path)
    state["source"] = source
    return preprocessed_path, preprocessed, mask


def run_dimension_measurement(state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    source = state["source"]
    jewel_type = (state.get("classification") or {}).get("confirmed_label") or ""
    source_path = Path(source["working_image"]["path"])
    image_bgr = cv2.imread(str(source_path))
    if image_bgr is None:
        raise RuntimeError("Could not load the saved working image for dimension analysis.")

    calibration_method = str(
        payload.get("calibration_method", "aruco")
    ).strip().lower()
    source_metric_calibration = source.get("stone_calibration") or {}
    if (
        calibration_method == "aruco"
        and source_metric_calibration.get("method") == "four_marker_homography"
        and _measurement_scale_for_calibration(source_metric_calibration)
    ):
        scale_pair = _measurement_scale_for_calibration(source_metric_calibration) or {}
        scale_mm_per_px = (
            float(scale_pair["mm_per_pixel_x"])
            + float(scale_pair["mm_per_pixel_y"])
        ) / 2.0
        calibration = copy.deepcopy(source_metric_calibration)
        calibration["detection_source"] = "rectified_source"
    elif calibration_method == "aruco":
        calibration_image = image_bgr
        calibration_roi = source.get("working_aruco_roi")
        detection_source = "working"

        original_artifact = source.get("original_image") or {}
        original_path = Path(str(original_artifact.get("path") or ""))
        original_bgr = cv2.imread(str(original_path)) if original_path.is_file() else None
        if original_bgr is not None:
            calibration_image = original_bgr
            calibration_roi = source.get("aruco_roi")
            detection_source = "original"

        scale_mm_per_px, calibration = compute_scale_mm_per_px(
            image_bgr=calibration_image,
            payload=payload,
            processing_roi=None,
            working_aruco_roi=calibration_roi,
        )
        if detection_source == "original":
            original_corners = calibration.get("corners")
            calibration["corners_original"] = original_corners
            calibration["corners"] = translate_corners_to_crop(
                original_corners,
                source.get("processing_roi"),
                image_bgr.shape,
            )
        calibration["detection_source"] = detection_source
    else:
        scale_mm_per_px, calibration = compute_scale_mm_per_px(
            image_bgr=image_bgr,
            payload=payload,
            processing_roi=source.get("processing_roi"),
            working_aruco_roi=source.get("working_aruco_roi"),
        )

    ignore_mask = build_aruco_ignore_mask(
        image_bgr.shape,
        source.get("working_aruco_roi"),
        calibration.get("corners"),
    )
    measurement_input = image_bgr.copy()
    measurement_input[ignore_mask > 0] = 255

    session_dir = session_dir_for(state)
    measurement_input_path = save_bgr(session_dir / "dimension" / "measurement_input.png", measurement_input)
    result = detect_bangle(
        str(measurement_input_path),
        scale=scale_mm_per_px,
        debug=False,
        jewel_type=jewel_type,
    )

    result_image_bgr = cv2.imread(result["annotated_path"])
    if result_image_bgr is None:
        raise RuntimeError("Dimension result image was not generated.")

    if source.get("working_aruco_roi"):
        roi = source["working_aruco_roi"]
        draw_labeled_rect(result_image_bgr, roi, "Ignored Aruco ROI", (0, 220, 220))
    if calibration.get("corners"):
        corners = np.array(calibration["corners"], dtype=np.int32)
        cv2.polylines(result_image_bgr, [corners], True, (0, 220, 220), 2)

    dimension_result_path = save_bgr(session_dir / "dimension" / "dimension_result.png", result_image_bgr)

    return {
        "done": True,
        "calibration": calibration,
        "scale_mm_per_px": to_python_scalar(scale_mm_per_px),
        "od_px": to_python_scalar(result["od_px"]),
        "id_px": to_python_scalar(result["id_px"]),
        "wall_thickness_px": to_python_scalar(result["wall_thickness_px"]),
        "od_mm": to_python_scalar(result.get("od_mm")),
        "id_mm": to_python_scalar(result.get("id_mm")),
        "wall_thickness_mm": to_python_scalar(result.get("wall_thickness_mm")),
        "result_image": artifact_payload(state, dimension_result_path),
        "measurement_input": artifact_payload(state, measurement_input_path),
        "used_fallback": bool(result.get("used_fallback")),
        "detection_mode": result.get("detection_mode", "bangle"),
        "zoom_factor": to_python_scalar(result.get("zoom_factor")) or 1.0,
    }


def segmentation_args_for(state: dict[str, Any]) -> SimpleNamespace:
    session_dir = session_dir_for(state)
    jewel_type = (
        state.get("classification", {}).get("confirmed_label")
        or state.get("classification", {}).get("predicted_label")
        or ""
    )
    return SimpleNamespace(
        image=state["source"]["working_image"]["path"],
        model=str(SEG_MODEL_PATH),
        output_dir=str(session_dir / "segmentation"),
        input_size=640,
        conf_thres=0.20,
        iou_thres=0.85,
        mask_thres=0.50,
        providers=["CPUExecutionProvider"],
        gui=False,
        feedback_dir=str(SEGMENTATION_FEEDBACK_DIR),
        jewel_type=str(jewel_type),
    )


def run_segmentation_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    args = segmentation_args_for(state)
    source_path = Path(state["source"]["working_image"]["path"])
    full_image_artifact = state["source"].get("original_image") or {}
    full_image_path = Path(str(full_image_artifact.get("path") or ""))
    full_image = cv2.imread(str(full_image_path), cv2.IMREAD_COLOR)
    if full_image is None:
        raise RuntimeError("Could not load the full captured image for bead detection.")
    bead_analysis, bead_detection_image = run_full_image_bead_detection(
        full_image,
        state.get("source", {}).get("processing_roi"),
    )
    _preprocessed_path, preprocessed_image, preprocessed_mask = (
        load_or_create_shared_jewelry_input(state)
    )
    output_dir, debug = necklace_segmentation.run_segmentation(
        source_path,
        SEG_MODEL_PATH,
        get_segmenter(),
        args,
        preprocessed_image=preprocessed_image,
        preprocessed_mask=preprocessed_mask,
        bead_analysis_override=bead_analysis,
    )
    bead_detection_path = save_bgr(output_dir / "bead_finder_full_image.png", bead_detection_image)
    summary_path = output_dir / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    bead_analysis = debug.get("bead_analysis") or (summary_payload.get("debug") or {}).get("bead_analysis") or {}
    bead_risk = str(bead_analysis.get("risk", "Low")) if bead_analysis else None
    bead_risk_high = str(bead_risk or "").strip().lower() == "high" or bool((bead_analysis or {}).get("beads_detected"))

    return {
        "done": True,
        "debug": debug,
        "bead_risk": bead_risk,
        "bead_risk_high": bead_risk_high,
        "bead_analysis": bead_analysis or None,
        "no_pendant": bool(debug.get("no_pendant", False)),
        "no_tassel": bool(debug.get("no_tassel", False)),
        "feedback_match_type": debug.get("feedback_match_type"),
        "feedback_match_score": debug.get("feedback_match_score"),
        "feedback_alignment_score": debug.get("feedback_alignment_score"),
        "pendant_detected": bool(debug.get("pendant_detected", False)),
        "tassel_detected": bool(debug.get("tassel_detected", False)),
        "pendant_absent": bool(debug.get("pendant_absent", False)),
        "tassel_absent": bool(debug.get("tassel_absent", False)),
        "pendant_evidence": debug.get("pendant_evidence"),
        "tassel_evidence": debug.get("tassel_evidence"),
        "part_detection_prompts": debug.get("part_detection_prompts"),
        "part_summary": summary_payload.get("parts", {}),
        "summary_json": artifact_payload(state, summary_path),
        "composite_layout": artifact_payload(state, output_dir / "composite_layout.png"),
        "bead_analysis_image": artifact_payload(state, output_dir / "bead_analysis.png") if (output_dir / "bead_analysis.png").exists() else None,
        "bead_finder_image": artifact_payload(state, bead_detection_path),
        "preprocessed_image": artifact_payload(state, output_dir / "input_preprocessed.png"),
        "overlay_image": artifact_payload(state, output_dir / "overlay.png"),
        "part_masks": {
            "pendant": artifact_payload(state, output_dir / "pendant_mask.png"),
            "chain": artifact_payload(state, output_dir / "chain_mask.png"),
            "tassel": artifact_payload(state, output_dir / "tassel_mask.png"),
        },
    }


def apply_segmentation_feedback(
    state: dict[str, Any],
    part: str,
    bbox: dict[str, int],
) -> dict[str, Any]:
    args = segmentation_args_for(state)
    working_path = Path(state["source"]["working_image"]["path"])
    preprocessed_path = Path(args.output_dir) / working_path.stem / "input_preprocessed.png"
    preprocessed = cv2.imread(str(preprocessed_path))
    if preprocessed is None:
        raise RuntimeError("Run jewellery analysis once before applying a manual correction.")
    preprocessed_mask = cv2.imread(
        str(preprocessed_path.with_name("input_mask.png")),
        cv2.IMREAD_GRAYSCALE,
    )
    if preprocessed_mask is not None:
        preprocessed_mask = (preprocessed_mask > 0).astype(np.uint8)

    x, y, w, h = rect_to_tuple(bbox)
    feedback_bbox = (x, y, x + w - 1, y + h - 1)
    feedback_dir = Path(args.feedback_dir)
    if part == "pendant":
        necklace_segmentation.save_pendant_feedback(
            working_path,
            feedback_dir,
            feedback_bbox,
            preprocessed.shape[:2],
            prepared_image=preprocessed,
            prepared_mask=preprocessed_mask,
            jewel_type=args.jewel_type,
        )
    elif part == "tassel":
        necklace_segmentation.save_tassel_feedback(
            working_path,
            feedback_dir,
            feedback_bbox,
            preprocessed.shape[:2],
            prepared_image=preprocessed,
            prepared_mask=preprocessed_mask,
            jewel_type=args.jewel_type,
        )
    else:
        raise RuntimeError(f"Unsupported jewellery analysis correction part: {part}")
    return run_segmentation_pipeline(state)


def apply_segmentation_no_part(
    state: dict[str, Any],
    part: str,
) -> dict[str, Any]:
    args = segmentation_args_for(state)
    working_path = Path(state["source"]["working_image"]["path"])
    preprocessed_path = Path(args.output_dir) / working_path.stem / "input_preprocessed.png"
    preprocessed = cv2.imread(str(preprocessed_path))
    if preprocessed is None:
        raise RuntimeError("Run jewellery analysis once before excluding a region.")
    preprocessed_mask = cv2.imread(
        str(preprocessed_path.with_name("input_mask.png")),
        cv2.IMREAD_GRAYSCALE,
    )
    if preprocessed_mask is not None:
        preprocessed_mask = (preprocessed_mask > 0).astype(np.uint8)

    feedback_dir = Path(args.feedback_dir)
    if part == "pendant":
        necklace_segmentation.save_no_pendant_feedback(
            working_path,
            feedback_dir,
            preprocessed.shape[:2],
            prepared_image=preprocessed,
            prepared_mask=preprocessed_mask,
            jewel_type=args.jewel_type,
        )
    elif part == "tassel":
        necklace_segmentation.save_no_tassel_feedback(
            working_path,
            feedback_dir,
            preprocessed.shape[:2],
            prepared_image=preprocessed,
            prepared_mask=preprocessed_mask,
            jewel_type=args.jewel_type,
        )
    else:
        raise RuntimeError(f"Unsupported jewellery analysis exclusion part: {part}")
    result = run_segmentation_pipeline(state)
    remaining_area = int(
        ((result.get("part_summary") or {}).get(part) or {}).get("area", 0)
    )
    if remaining_area > 0:
        raise RuntimeError(
            f"{part.title()} exclusion failed: {remaining_area} pixels remained "
            "after the jewellery analysis rerun."
        )
    return result


def aggregate_stone_summary(
    report: dict[str, Any],
    gold_total_px: int,
    jewel_total_px: int,
    calibration: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    color_totals: dict[str, dict[str, Any]] = {}

    for jewel in report.get("jewels", []):
        for entry in jewel.get("detected_colors", []):
            color = str(entry["color"])
            item = color_totals.setdefault(
                color,
                {
                    "color": color,
                    "area_px": 0,
                    "region_count": 0,
                    "possible_gemstones": list(entry.get("possible_gemstones", [])),
                    "learned_labels": [],
                },
            )
            item["area_px"] += int(entry.get("area_px", 0))
            item["region_count"] += int(entry.get("region_count", 0))
            item["learned_labels"] = sorted(
                {
                    *item.get("learned_labels", []),
                    *[
                        str(value)
                        for value in entry.get("learned_labels", [])
                        if str(value).strip()
                    ],
                }
            )

    summary_entries = sorted(
        color_totals.values(),
        key=lambda item: (-item["area_px"], item["color"]),
    )

    denominator_px = max(
        0,
        int(jewel_total_px)
        or sum(int(jewel.get("jewel_area_px", 0)) for jewel in report.get("jewels", [])),
    )
    measured_stone_px = sum(
        int(jewel.get("stone_area_px", 0))
        for jewel in report.get("jewels", [])
    )
    if measured_stone_px <= 0:
        measured_stone_px = sum(
            max(0, int(entry["area_px"]))
            for entry in summary_entries
        )
    total_stone_px = min(measured_stone_px, denominator_px)
    color_area_total = sum(max(0, int(entry["area_px"])) for entry in summary_entries)
    color_scale = (
        min(1.0, total_stone_px / float(color_area_total))
        if color_area_total > 0
        else 1.0
    )
    gold_region_px = min(max(0, int(gold_total_px)), denominator_px)
    area_per_pixel = _calibration_area_per_pixel(calibration)
    jewel_denom_area = (
        float(denominator_px) * area_per_pixel
        if area_per_pixel is not None
        else float(denominator_px)
    )

    for entry in summary_entries:
        area_px = min(
            max(0, int(round(int(entry["area_px"]) * color_scale))),
            total_stone_px,
        )
        entry["area_px"] = area_px
        area = (
            float(area_px) * area_per_pixel
            if area_per_pixel is not None
            else float(area_px)
        )
        percent = (area / jewel_denom_area * 100.0) if jewel_denom_area > 0 else 0.0
        entry["stone_percentage"] = round(percent, 2)
        if area_per_pixel is not None:
            entry["area_mm2"] = round(area, 4)

        if gold_region_px > 0:
            gold_rel_percent = area_px / float(gold_region_px) * 100.0
        else:
            gold_rel_percent = 0.0
        entry["gold_percentage"] = round(gold_rel_percent, 2)

    if not summary_entries:
        return [], "No stones detected"

    total_stone_area = (
        float(total_stone_px) * area_per_pixel
        if area_per_pixel is not None
        else float(total_stone_px)
    )
    total_stone_percent = min(
        (total_stone_area / jewel_denom_area * 100.0)
        if jewel_denom_area > 0
        else 0.0,
        100.0,
    )

    text = f"Stone Percentage: {total_stone_percent:.2f}%"

    return summary_entries, text


def load_artifact_mask(
    artifact: dict[str, Any] | None,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if not artifact:
        return None
    mask_path = artifact.get("path")
    if not mask_path:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def mask_image_for_stone_detection(
    image_bgr: np.ndarray,
    ignore_regions: list[tuple[int, int, int, int]] | None,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    masked_image = image_bgr.copy()
    if ignore_mask is not None and np.any(ignore_mask):
        masked_image[ignore_mask > 0] = (255, 255, 255)
    return stone_detection.apply_ignore_regions_to_image(masked_image, ignore_regions)


def compute_gold_pixel_total(
    image_bgr: np.ndarray,
    extraction_mode: str = "default",
    preset_candidates: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    if preset_candidates is not None:
        jewels = list(preset_candidates)
    else:
        jewels = stone_detection.extract_jewel_candidates(
            image_bgr,
            extraction_mode=extraction_mode,
        )
    gold_total = 0
    jewel_total = 0
    for jewel in jewels:
        jewel_total += int(cv2.countNonZero(jewel["crop_mask"]))
        hsv = cv2.cvtColor(jewel["crop_bgr"], cv2.COLOR_BGR2HSV)
        gold_mask = stone_detection.build_gold_mask(hsv, jewel["crop_mask"], strict=False)
        gold_total += int(cv2.countNonZero(gold_mask))
    return gold_total, jewel_total


def _aggregate_stone_area_stats(jewels: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-jewel stone_area_statistics into a single combined result."""
    total_jewel_px = 0
    total_stone_px = 0
    color_stats: dict[str, dict[str, Any]] = {}

    for jewel in jewels:
        stats = jewel.get("stone_area_statistics") or {}
        if not stats.get("success"):
            continue
        j_px = int(stats.get("jewel_area_pixels", 0))
        s_px = int(stats.get("stone_area_pixels", 0))
        total_jewel_px += j_px
        total_stone_px += s_px
        for color, cdata in (stats.get("colors") or {}).items():
            acc = color_stats.setdefault(color, {"area_pixels": 0, "component_count": 0})
            acc["area_pixels"] += int(cdata.get("area_pixels", 0))
            acc["component_count"] += int(cdata.get("component_count", 0))

    stone_percentage = (
        min(total_stone_px / float(total_jewel_px) * 100.0, 100.0)
        if total_jewel_px > 0
        else 0.0
    )
    metal_percentage = max(0.0, 100.0 - stone_percentage)

    colors: dict[str, Any] = {}
    for color, acc in color_stats.items():
        a_px = acc["area_pixels"]
        pct_jewel = (a_px / float(total_jewel_px) * 100.0) if total_jewel_px > 0 else 0.0
        pct_stones = (a_px / float(total_stone_px) * 100.0) if total_stone_px > 0 else 0.0
        colors[color] = {
            "area_pixels": a_px,
            "percentage_of_jewel": round(pct_jewel, 2),
            "percentage_of_stones": round(pct_stones, 2),
            "component_count": acc["component_count"],
        }

    return {
        "success": total_jewel_px > 0,
        "jewel_area_pixels": total_jewel_px,
        "stone_area_pixels": total_stone_px,
        "metal_area_pixels": max(0, total_jewel_px - total_stone_px),
        "stone_percentage": round(stone_percentage, 2),
        "metal_percentage": round(metal_percentage, 2),
        "colors": colors,
    }


def _scaled_measurement_scale(
    measurement_scale: dict[str, float] | None,
    analysis_scale: int,
) -> dict[str, float] | None:
    if not measurement_scale:
        return None
    scale = max(1, int(analysis_scale))
    if scale <= 1:
        return dict(measurement_scale)
    return {
        "mm_per_pixel_x": float(measurement_scale["mm_per_pixel_x"]) / scale,
        "mm_per_pixel_y": float(measurement_scale["mm_per_pixel_y"]) / scale,
    }


def _normalized_binary_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    mask_bin = np.asarray(mask, dtype=np.uint8)
    if mask_bin.ndim == 3:
        mask_bin = mask_bin.squeeze()
    if mask_bin.shape[:2] != image_shape:
        mask_bin = cv2.resize(
            mask_bin,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (mask_bin > 0).astype(np.uint8)


def _enhance_masked_jewel_with_super_resolution(
    image_bgr: np.ndarray,
    jewel_mask: np.ndarray,
    runner: RealESRGANHailoX2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enhance only jewel pixels, then return to the calibrated native grid."""
    mask_bin = _normalized_binary_mask(jewel_mask, image_bgr.shape[:2])
    points = cv2.findNonZero(mask_bin)
    if points is None:
        raise RuntimeError("The threshold jewel mask is empty.")

    x, y, width, height = cv2.boundingRect(points)
    padding = 16
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_bgr.shape[1], x + width + padding)
    y2 = min(image_bgr.shape[0], y + height + padding)

    crop_bgr = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask_bin[y1:y2, x1:x2]
    masked_crop = np.zeros_like(crop_bgr)
    masked_crop[crop_mask > 0] = crop_bgr[crop_mask > 0]

    super_resolved_crop = runner.process_bgr(masked_crop)
    enhanced_crop = cv2.resize(
        super_resolved_crop,
        (masked_crop.shape[1], masked_crop.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    # ESRGAN improves luminance detail but may reduce saturation. Preserve the
    # calibrated source chroma so HSV stone classification remains stable.
    enhanced_lab = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2LAB)
    enhanced_lab[:, :, 1:] = source_lab[:, :, 1:]
    enhanced_crop = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_crop[crop_mask == 0] = 0

    enhanced_bgr = np.zeros_like(image_bgr)
    enhanced_bgr[y1:y2, x1:x2] = enhanced_crop
    enhanced_bgr[mask_bin == 0] = 0
    return enhanced_bgr, {
        "masked_crop_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
        "model_output_width": int(super_resolved_crop.shape[1]),
        "model_output_height": int(super_resolved_crop.shape[0]),
        "measurement_grid_scale": 1,
    }


def run_stone_pipeline(
    state: dict[str, Any],
    image_path: Path,
    output_name: str,
    ignore_regions: list[tuple[int, int, int, int]] | None = None,
    ignore_mask: np.ndarray | None = None,
    preset_binary_mask: np.ndarray | None = None,
    calibration: dict[str, Any] | None = None,
    entered_jewel_weight_g: float | None = None,
    stone_setting_profile: str = stone_area_calculator.DEFAULT_STONE_SETTING_PROFILE,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not load image for stone detection: {image_path}")

    confirmed_label = (state.get("classification") or {}).get("confirmed_label") or ""
    # Default to native pixels. Real-ESRGAN x2 is applied only after the
    # jewelry mask is built, because generic upscaling can create edge halos
    # that look like white/colorless stones on side images.
    zoom_scale = 1
    extraction_mode = "earring" if confirmed_label in {"Earing / Jumkha", "Earrings/ Jhumki"} else "default"

    output_dir = session_dir_for(state) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    stone_settings = stone_settings_for_state(state)
    analysis_input, normalized_ignore_regions = mask_image_for_stone_detection(
        image_bgr,
        ignore_regions,
        ignore_mask=ignore_mask,
    )
    analysis_scale = 1
    analysis_image_path: Path | None = None
    analysis_mask_path: Path | None = None
    super_resolution_settings = stone_settings["stone_super_resolution"]
    super_resolution_info: dict[str, Any] = {
        "requested": bool(super_resolution_settings.get("enabled")),
        "applied": False,
        "scale": 1,
        "model": super_resolution_settings.get("model") or "real_esrgan_x2",
        "hef_path": str(SUPER_RESOLUTION_HEF_PATH),
        "runtime_seconds": 0.0,
        "tile_count": 0,
    }
    preset_mask_for_analysis = (
        _normalized_binary_mask(preset_binary_mask, analysis_input.shape[:2])
        if preset_binary_mask is not None
        else None
    )

    if super_resolution_info["requested"]:
        try:
            if preset_mask_for_analysis is None:
                preset_mask_for_analysis = (
                    stone_detection.otsu_clean_mask(analysis_input) > 0
                ).astype(np.uint8)
            with MODEL_LOCK:
                runner = get_super_resolution_runner()
                analysis_input, enhancement_info = (
                    _enhance_masked_jewel_with_super_resolution(
                        analysis_input,
                        preset_mask_for_analysis,
                        runner,
                    )
                )
                super_resolution_info["runtime_seconds"] = round(
                    float(runner.last_runtime_seconds),
                    3,
                )
                super_resolution_info["tile_count"] = int(runner.last_tile_count)
            # Prediction stays on the calibrated native grid. ESRGAN contributes
            # detail, while pixel areas and mm-per-pixel remain directly
            # comparable with the non-SR path.
            analysis_scale = 1
            super_resolution_info.update(enhancement_info)
            super_resolution_info["applied"] = True
            super_resolution_info["scale"] = int(SUPER_RESOLUTION_SCALE)
            analysis_image_path = save_bgr(
                output_dir / "enhanced_masked_analysis_input.png",
                analysis_input,
            )
            analysis_mask_path = output_dir / "analysis_jewel_mask.png"
            cv2.imwrite(
                str(analysis_mask_path),
                (preset_mask_for_analysis * 255).astype(np.uint8),
            )
        except Exception as exc:
            super_resolution_info["error"] = str(exc)
            super_resolution_info["fallback"] = "native"
            super_resolution_info["applied"] = False
            super_resolution_info["scale"] = 1
            analysis_scale = 1
            print(f"Stone super resolution failed; using native analysis: {exc}")

    preset_candidates = None
    if preset_mask_for_analysis is not None:
        try:
            mask_bin = _normalized_binary_mask(
                preset_mask_for_analysis,
                analysis_input.shape[:2],
            )
            preset_candidates = stone_detection.build_candidates_from_component_mask(
                analysis_input,
                mask_bin,
                min_area=max(80, int(mask_bin.size * 0.0005)),
                max_candidates=12,
                # This is an authoritative threshold/hand-removal mask. A jewel
                # touching the Processing ROI edge is still a valid candidate.
                reject_border_touching=False,
                min_area_ratio_to_largest=0.01,
            )
            preset_candidates = [
                candidate for candidate in preset_candidates
                if stone_detection.has_enough_gold_for_stone_detection(
                    candidate["crop_bgr"],
                    candidate["crop_mask"],
                )
            ]
        except Exception as exc:
            print(f"Preset threshold mask candidate build failed: {exc}")
            preset_candidates = []

    native_measurement_scale = _measurement_scale_for_calibration(calibration)
    analysis_measurement_scale = _scaled_measurement_scale(
        native_measurement_scale,
        analysis_scale,
    )
    fastsam_model = None
    try:
        # get_segmenter() reuses the application's single HailoRuntime and
        # cached FastSAM HEF.  Failure is non-fatal because V2 has an OpenCV
        # and conservative-seed fallback for every candidate.
        fastsam_model = get_segmenter()
    except Exception as exc:
        print(f"Stone FastSAM refinement unavailable; using CPU fallback: {exc}")
    analysis = stone_detection.analyze_image_bgr(
        analysis_input,
        source_name=str(image_path),
        zoom_scale=zoom_scale,
        extraction_mode=extraction_mode,
        preset_candidates=preset_candidates,
        external_mask=preset_mask_for_analysis,
        ignore_regions=None,
        use_glare_removal=True,
        glare_threshold=stone_detection.DEFAULT_GLARE_THRESHOLD,
        glare_patch_size=stone_detection.DEFAULT_GLARE_PATCH_SIZE,
        use_sahi_slicing=True,
        sahi_slice_size=stone_detection.DEFAULT_SAHI_SLICE_SIZE,
        sahi_overlap_ratio=stone_detection.DEFAULT_SAHI_OVERLAP,
        color_correction=stone_settings["color_correction"],
        background_calibration=stone_settings["background_calibration"],
        analysis_normalization=stone_settings["analysis_normalization"],
        measurement_scale=analysis_measurement_scale,
        learned_stone_profiles=stone_settings["learned_stone_profiles"],
        fastsam_model=fastsam_model,
        fastsam_lock=MODEL_LOCK,
        stone_v2_debug_dir=(
            output_dir / "stone_analysis_v2_debug"
            if STONE_ANALYSIS_V2_DEBUG
            else None
        ),
    )
    report = analysis["report"]
    report["stone_super_resolution"] = {
        **super_resolution_info,
        "analysis_image_path": str(analysis_image_path) if analysis_image_path else None,
        "analysis_mask_path": str(analysis_mask_path) if analysis_mask_path else None,
    }
    report["analysis_calibration"] = {
        "analysis_scale": int(analysis_scale),
        "native_mm_per_pixel_x": (
            (native_measurement_scale or {}).get("mm_per_pixel_x")
        ),
        "native_mm_per_pixel_y": (
            (native_measurement_scale or {}).get("mm_per_pixel_y")
        ),
        "analysis_mm_per_pixel_x": (
            (analysis_measurement_scale or {}).get("mm_per_pixel_x")
        ),
        "analysis_mm_per_pixel_y": (
            (analysis_measurement_scale or {}).get("mm_per_pixel_y")
        ),
    }
    face_up_weight_estimate = copy.deepcopy(report.get("stone_measurements") or {})
    entered_weight = (
        float(entered_jewel_weight_g)
        if entered_jewel_weight_g is not None and float(entered_jewel_weight_g) > 0
        else None
    )
    visible_stone_area_mm2 = None
    if analysis_measurement_scale:
        visible_stone_area_mm2 = (
            float(report.get("stone_area_px_total", 0))
            * float(analysis_measurement_scale["mm_per_pixel_x"])
            * float(analysis_measurement_scale["mm_per_pixel_y"])
        )
    weight_estimate = stone_area_calculator.apply_stone_setting_weight_model(
        face_up_weight_estimate,
        stone_setting_profile,
        visible_stone_area_mm2,
        entered_weight,
    )
    report["stone_setting_profile"] = weight_estimate["stone_setting_profile"]
    report["stone_weight_estimate"] = weight_estimate
    surface_risk = report.get("stone_surface_risk") or (
        stone_analysis_v2.calculate_stone_surface_risk(
            report.get("stone_percentage", 0.0),
            report.get("stone_instance_count", 0),
            reflection_risk=bool(report.get("reflection_risk")),
        )
    )
    report["stone_surface_risk"] = surface_risk
    report["stone_weight_feature_vector"] = (
        stone_analysis_v2.build_stone_weight_feature_vector(
            report,
            weight_estimate,
            jewel_type=confirmed_label,
            total_jewel_weight_g=entered_weight,
            view="side" if "side" in output_name else "main",
        )
    )

    gallery_path = save_bgr(output_dir / "stone_gallery.png", analysis["result_gallery_bgr"])

    for jewel_report, jewel_view in zip(report["jewels"], analysis["jewel_views"], strict=False):
        jewel_report["outputs"] = stone_detection.write_jewel_output_images(
            output_dir=output_dir,
            jewel_index=jewel_report["jewel_id"],
            masked_png=jewel_view["masked_bgra"],
            overlay_png=jewel_view["overlay_bgr"],
            color_mask_png=jewel_view["color_mask_bgr"],
        )

    report_path = output_dir / f"{image_path.stem}_gem_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    gold_total_px, fallback_jewel_total_px = compute_gold_pixel_total(
        analysis_input,
        extraction_mode=extraction_mode,
        preset_candidates=preset_candidates,
    )
    stone_area_statistics = _aggregate_stone_area_stats(report.get("jewels", []))
    if stone_area_statistics.get("success"):
        jewel_total_px = int(stone_area_statistics["jewel_area_pixels"])
        stone_area_px = int(stone_area_statistics["stone_area_pixels"])
        stone_percentage = float(stone_area_statistics["stone_percentage"])
    else:
        jewel_total_px = max(0, int(fallback_jewel_total_px))
        stone_area_px = min(
            sum(
                int(jewel.get("stone_area_px", 0))
                for jewel in report.get("jewels", [])
            ),
            jewel_total_px,
        )
        stone_percentage = (
            stone_area_px / float(jewel_total_px) * 100.0
            if jewel_total_px > 0
            else 0.0
        )
    gold_total_px = min(max(0, int(gold_total_px)), jewel_total_px)

    summary_entries, summary_text = aggregate_stone_summary(
        report,
        gold_total_px,
        jewel_total_px,
        calibration=calibration,
    )
    reflective_jewels = [
        {
            "jewel_id": int(jewel.get("jewel_id", 0)),
            **(jewel.get("reflection") or {}),
        }
        for jewel in report.get("jewels", [])
        if (jewel.get("reflection") or {}).get("flagged")
    ]
    max_reflection_coverage = max(
        (float(item.get("coverage_percent", 0.0)) for item in reflective_jewels),
        default=0.0,
    )
    max_reflection_density = max(
        (float(item.get("local_density_percent", 0.0)) for item in reflective_jewels),
        default=0.0,
    )
    reflection_summary = (
        f"More/dense reflection detected in {len(reflective_jewels)} jewel area(s) "
        f"(coverage up to {max_reflection_coverage:.2f}%, local density up to "
        f"{max_reflection_density:.2f}%); possible additional transparent/colorless "
        "gemstones may be present."
        if reflective_jewels
        else "No dense reflection risk detected."
    )

    return {
        "report_path": artifact_payload(state, report_path),
        "gallery": artifact_payload(state, gallery_path),
        "jewel_count": int(report.get("jewel_count", 0)),
        "gold_total_px": gold_total_px,
        "jewel_total_px": jewel_total_px,
        "stone_area_px": stone_area_px,
        "stone_percentage": round(stone_percentage, 2),
        "stone_surface_coverage_percent": round(stone_percentage, 2),
        "stone_surface_risk": surface_risk,
        "stone_surface_risk_level": surface_risk.get("level"),
        "stone_surface_status": surface_risk.get("status"),
        "stone_instance_count": int(report.get("stone_instance_count", 0)),
        "segmentation_method_counts": dict(
            report.get("segmentation_method_counts") or {}
        ),
        "stone_seed_area_px": int(report.get("stone_seed_area_px_total", 0)),
        "stone_mask_area_gain": float(report.get("stone_mask_area_gain", 1.0)),
        "seeded_instance_separation": report.get(
            "seeded_instance_separation"
        ) or {},
        "ignored_regions": normalized_ignore_regions,
        "ignored_mask_px": int(cv2.countNonZero(ignore_mask)) if ignore_mask is not None else 0,
        "summary_entries": summary_entries,
        "summary_text": summary_text,
        "stone_area_statistics": stone_area_statistics,
        "area_denominator": "segmented_jewel_mask",
        "fallback_extraction_jewel_total_px": int(fallback_jewel_total_px),
        "background_hole_pixels_removed": int(
            report.get("background_hole_pixels_removed", 0)
        ),
        "background_hole_count": int(report.get("background_hole_count", 0)),
        "reflection_flagged": bool(reflective_jewels),
        "reflection_risk": bool(reflective_jewels),
        "risk_jewel": bool(reflective_jewels),
        "reflective_jewels": reflective_jewels,
        "reflection_summary": reflection_summary,
        "color_correction": stone_settings["color_correction"],
        "analysis_normalization": stone_settings["analysis_normalization"],
        "background_calibration": stone_settings["background_calibration"],
        "learned_stone_profiles": stone_settings["learned_stone_profiles"],
        "stone_super_resolution": {
            **super_resolution_info,
            "analysis_image": artifact_payload(state, analysis_image_path),
            "analysis_mask": artifact_payload(state, analysis_mask_path),
        },
        "weight_estimate": weight_estimate,
        "estimated_total_minimum_g": weight_estimate.get(
            "estimated_total_minimum_g"
        ),
        "estimated_total_typical_g": weight_estimate.get(
            "estimated_total_typical_g",
            weight_estimate.get("estimated_total_average_g"),
        ),
        "estimated_total_maximum_g": weight_estimate.get(
            "estimated_total_maximum_g"
        ),
        "weight_confidence": weight_estimate.get("weight_confidence"),
        "weight_confidence_score": weight_estimate.get(
            "weight_confidence_score"
        ),
        "weight_method": weight_estimate.get("weight_method"),
        "weight_warnings": list(weight_estimate.get("weight_warnings") or []),
        "stone_weight_feature_vector": report["stone_weight_feature_vector"],
        "area_calibration": {
            "method": calibration.get("method") if calibration else None,
            "applied": bool(_calibration_area_per_pixel(calibration)),
            "marker_id": calibration.get("marker_id") if calibration else None,
            "marker_length_mm": calibration.get("marker_length_mm") if calibration else None,
            "marker_breadth_mm": calibration.get("marker_breadth_mm") if calibration else None,
            "mm_per_pixel_x": (
                (native_measurement_scale or {}).get("mm_per_pixel_x")
            ),
            "mm_per_pixel_y": (
                (native_measurement_scale or {}).get("mm_per_pixel_y")
            ),
            "analysis_scale": int(analysis_scale),
            "analysis_mm_per_pixel_x": (
                (analysis_measurement_scale or {}).get("mm_per_pixel_x")
            ),
            "analysis_mm_per_pixel_y": (
                (analysis_measurement_scale or {}).get("mm_per_pixel_y")
            ),
            "height_scale_factor": (
                calibration.get("height_scale_factor") if calibration else None
            ),
        },
        "report": report,
    }


@app.route("/")
def index():
    return send_file(WEBUI_DIR / "main.html")

@app.route("/report")
def report_page():
    return send_file(WEBUI_DIR / "report.html")

@app.route('/favicon.ico')
def favicon():
    return send_file(WEBUI_DIR / "embsys_logo.png", mimetype='image/png')


@app.route("/api/config")
def api_config():
    return jsonify(
        {
            "ok": True,
            "class_labels": sorted(ALL_DISPLAY_LABELS, key=str.casefold),
            "aruco_dicts": list(ARUCO_DICTS.keys()),
            "branches": {
                "dimension": sorted(list(DIMENSION_CLASSES)),
                "direct_stone": sorted(list(DIRECT_STONE_CLASSES)),
                "segmentation": sorted(list(SEGMENTATION_CLASSES)),
            },
            "persistent_rois": PERSISTENT_ROIS,
            "stone_settings": stone_settings_for_state(),
            "audio_settings": purity_audio_settings(),
            "camera": {
                "transport": "server-frame",
                "device": CAM_DEVICE,
                "camera_index": CAM_INDEX,
                "rotate_90_clockwise": CAM_ROTATE_90_CLOCKWISE,
                "target_width": CAM_TARGET_WIDTH,
                "target_height": CAM_TARGET_HEIGHT,
                "exposure_mode": CAMERA_EXPOSURE_MODE,
            },
            "max_upload_mb": int(app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)),
        }
    )


@app.route("/api/state")
def api_state():
    return jsonify(
        {
            "ok": True,
            "state": snapshot_state(),
            "camera_focus": get_camera_backend().focus_snapshot(),
        }
    )


@app.route("/api/purity/audio-devices")
def api_purity_audio_devices():
    manager = get_purity_manager()
    return jsonify(
        {
            "ok": True,
            "devices": manager.list_audio_devices(),
            "audio_sensor": purity_audio_sensor_state(),
            "outputs": list_tts_output_devices(),
            "tts_output": tts_output_state(),
            "state": manager.snapshot(),
        }
    )


@app.route("/api/purity/audio-output", methods=["POST"])
def api_purity_audio_output():
    payload = parse_post_payload()
    selected = select_tts_output_device(payload.get("output_device"))
    return jsonify(
        {
            "ok": True,
            "selected": selected,
            "outputs": list_tts_output_devices(),
            "tts_output": tts_output_state(),
        }
    )


@app.route("/api/purity/audio-settings", methods=["POST"])
def api_purity_audio_settings():
    try:
        payload = parse_post_payload()
        settings = normalize_purity_audio_settings(payload)
        PERSISTENT_ROIS["audio_settings"] = settings
        save_persistent_rois()
        manager = get_purity_manager()
        applied_threshold = manager.set_audio_ok_confidence_threshold(
            settings["ok_confidence_threshold"]
        )
        settings["ok_confidence_threshold"] = applied_threshold
        return jsonify(
            {
                "ok": True,
                "audio_settings": settings,
                "state": snapshot_state(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/purity/start", methods=["POST"])
def api_purity_start():
    with STATE_LOCK:
        state = ensure_state()
        if not state.get("session_id"):
            return fail("Capture or upload the main jewelry image first.")
        if not purity_upstream_ready(state):
            return fail("Finish the current workflow branch before starting the acid test.")

        try:
            payload = parse_post_payload()
            audio_device = payload.get("audio_device")
            manager = get_purity_manager()
            readiness = manager.snapshot()
            if not readiness.get("available") or not readiness.get("models_loaded"):
                raise RuntimeError(
                    readiness.get("last_error")
                    or readiness.get("error")
                    or "Purity HEFs did not pass startup loading. Restart the application."
                )
            print("[DEBUG] Purity start uses startup-loaded HEFs; no model load performed")
            manager.start(session_dir_for(state) / "purity_test", audio_device=audio_device)
            clear_stage_skip(state, "acid_test")
            refresh_purity_state(state)
            state["status"] = state["purity_test"].get("status") or "Acid test running"
            state["updated_at"] = now_stamp()
            build_final_summary(state)
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/purity/stop", methods=["POST"])
def api_purity_stop():
    with STATE_LOCK:
        state = ensure_state()
        try:
            payload = parse_post_payload()
            reason = str(payload.get("reason") or "Stopped by user")
            manager = get_purity_manager()
            stop_state = manager.stop(reason=reason)
            refresh_purity_state(state)
            state["status"] = state["purity_test"].get("status") or "Acid test stopped"
            state["updated_at"] = now_stamp()
            build_final_summary(state)

            if not stop_state.get("running"):
                speak(post_jewel_voice_prompt(state, "acid test completed"))
        except Exception as exc:  # noqa: BLE001
            print(f"Could not stop purity test: {exc}")
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/purity/skip", methods=["POST"])
def api_purity_skip():
    with STATE_LOCK:
        state = ensure_state()
        if not state.get("session_id"):
            return fail("Capture or upload the main jewelry image first.")
        if not purity_upstream_ready(state):
            return fail("Finish the current workflow branch before skipping the acid test.")
        purity = state.get("purity_test") or {}
        if purity.get("running"):
            return fail("Stop the running acid test before skipping it.")

        try:
            manager = get_purity_manager()
            manager.reset(stop_running=True)
            state["purity_test"] = {
                **(manager.snapshot() or {}),
                "skipped": True,
                "skipped_at": now_stamp(),
                "status": "Acid test skipped.",
                "result": "Skipped",
                "acid_ok": False,
            }
            mark_stage_skipped(state, "acid_test")
            state["status"] = "Acid test skipped."
            state["updated_at"] = now_stamp()
            build_final_summary(state)

            speak(post_jewel_voice_prompt(state, "acid test skipped"))
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/stage/skip", methods=["POST"])
def api_stage_skip():
    with STATE_LOCK:
        state = ensure_state()
        update_pledge_progress(state)
        if not state.get("session_id"):
            return fail("Capture the jewel image and run jewel type before skipping an optional stage.")
        if not (state.get("classification") or {}).get("confirmed"):
            return fail("Confirm the jewel type before skipping an optional stage.")

        try:
            payload = parse_post_payload()
            stage_key = str(payload.get("stage") or "").strip()
            branch_key = (state.get("branch") or {}).get("key")
            allowed = {"weight_extraction", "acid_test"}
            if branch_key == "dimension":
                allowed.update({"dimension", "side_stone"})
            elif branch_key == "segmentation":
                allowed.update({"jewellery_analysis", "stone_detection"})
            elif branch_key == "direct_stone":
                allowed.add("stone_detection")
            if state.get("pledge_complete"):
                allowed.update({"final_count", "packet_sealing"})
            if stage_key not in allowed:
                raise ValueError("This stage is not available to skip in the current workflow.")

            if stage_key == "weight_extraction":
                skipped_at = now_stamp()
                state["weight_details"] = {
                    "jewel_weight_g": None,
                    "appraiser_stone_weight_g": None,
                }
                state["weight_extraction"] = {
                    "success": False,
                    "status": "skipped",
                    "message": "Jewel weight extraction skipped.",
                    "captured_at": None,
                    "captured_image": None,
                    "roi_image": None,
                    "lcd_image": None,
                    "skipped": True,
                    "skipped_at": skipped_at,
                }
            elif stage_key == "acid_test":
                purity = state.get("purity_test") or {}
                if purity.get("running"):
                    raise ValueError("Stop the running acid test before skipping it.")
                manager = get_purity_manager()
                manager.reset(stop_running=True)
                state["purity_test"] = {
                    **(manager.snapshot() or {}),
                    "skipped": True,
                    "skipped_at": now_stamp(),
                    "status": "Acid test skipped.",
                    "result": "Skipped",
                    "acid_ok": False,
                }
            elif stage_key == "dimension":
                state["dimension"] = {"done": False, "skipped": True}
            elif stage_key == "jewellery_analysis":
                state["segmentation"] = {"done": False, "skipped": True}
                state["stone_detection"]["main"] = None
            elif stage_key == "side_stone":
                state["stone_detection"]["side"] = None
            elif stage_key == "stone_detection":
                state["stone_detection"]["main"] = None
            elif stage_key in {"final_count", "packet_sealing"}:
                pledge_id = str(state.get("pledge_id") or "").strip()
                if not pledge_id:
                    raise ValueError("Set the Pledge ID before skipping a closure stage.")
                metadata = get_or_create_pledge_metadata(pledge_id)
                if stage_key == "final_count":
                    skipped_at = now_stamp()
                    metadata["jewel_count_verification"] = {
                        "skipped": True,
                        "skipped_at": skipped_at,
                        "status": "Skipped",
                    }
                else:
                    metadata["packet_sealing"] = {
                        **_default_packet_sealing_state(),
                        "status": "skipped",
                        "skipped": True,
                        "skipped_at": now_stamp(),
                    }
                save_pledge_metadata(metadata)
                apply_pledge_metadata_to_state(state, metadata)

            skipped = mark_stage_skipped(state, stage_key)
            state["status"] = f"{skipped['display_name']} skipped."
            state["updated_at"] = now_stamp()
            build_final_summary(state)
            if stage_key == "acid_test":
                speak(post_jewel_voice_prompt(state, "acid test skipped"))
            else:
                speak(f"{skipped['display_name']} skipped. Continue to the next stage.")
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


def _preview_jpeg_bytes() -> bytes | None:
    recorder = PACKET_RECORDER
    if recorder is not None and recorder.is_recording():
        packet_frame = get_camera_backend().get_frame_copy()
        if packet_frame is not None:
            packet_frame = recorder.annotate_preview(packet_frame)
            ok, buffer = cv2.imencode(
                ".jpg",
                packet_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(CAM_PREVIEW_JPEG_QUALITY)],
            )
            if ok and buffer is not None:
                return buffer.tobytes()

    manager = get_purity_manager()
    purity_running = manager.is_running()
    # During purity inference, the latest annotated result remains the
    # authoritative preview until the next result replaces it. Expiring it
    # after 400 ms caused the stream to fall back to raw camera frames.
    purity_frame = manager.get_display_frame_copy(
        max_age_s=None if purity_running else 0.5
    )
    if purity_frame is None and purity_running:
        live_frame = get_purity_camera_frame()
        if live_frame is not None:
            purity_frame = manager.build_live_preview(live_frame)
    if purity_frame is not None:
        ok, buffer = cv2.imencode(
            ".jpg",
            purity_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(CAM_PREVIEW_JPEG_QUALITY)],
        )
        if ok and buffer is not None:
            return buffer.tobytes()
    return get_camera_backend().get_jpeg_bytes(CAM_PREVIEW_JPEG_QUALITY)


@app.route("/api/frame.jpg")
def api_frame():
    camera = get_camera_backend()
    jpeg_bytes = _preview_jpeg_bytes()
    if not jpeg_bytes:
        status = camera.snapshot()
        message = status.get("last_error") or status.get("status") or "Live camera frame is not available."
        debug_frame = build_camera_status_frame(message)
        ok, buffer = cv2.imencode(
            ".jpg",
            debug_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(CAM_PREVIEW_JPEG_QUALITY)],
        )
        if ok and buffer is not None:
            jpeg_bytes = buffer.tobytes()
        else:
            return (
                message,
                503,
                {
                    "Content-Type": "text/plain; charset=utf-8",
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

    return (
        jpeg_bytes,
        200,
        {
            "Content-Type": "image/jpeg",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

def generate_mjpeg_stream():
    camera = get_camera_backend()
    while True:
        jpeg_bytes = _preview_jpeg_bytes()
        if jpeg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        else:
            status = camera.snapshot()
            message = status.get("last_error") or status.get("status") or "Live camera frame is not available."
            debug_frame = build_camera_status_frame(message)
            ok, buffer = cv2.imencode(
                ".jpg",
                debug_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(CAM_PREVIEW_JPEG_QUALITY)],
            )
            if ok and buffer is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
        time.sleep(0.04)

@app.route("/api/video_feed")
def api_video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global CURRENT_STATE
    recorder = PACKET_RECORDER
    if recorder is not None and recorder.is_recording():
        return fail("Stop packet sealing recording before resetting the workflow.")
    with STATE_LOCK:
        CURRENT_STATE = build_empty_state()
    lcd_show_pledge_status(None)
    speak("Enter the Pledge ID")
    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/rois", methods=["POST"])
def api_rois():
    try:
        payload = parse_post_payload()
        # We don't have width/height here, but normalize_rect can handle dicts.
        # However, it's better to just trust the frontend's absolute px values here
        # or pass image dimensions if needed. 
        # For simplicity, since the frontend sends valid rects:
        if "processing_roi" in payload:
            PERSISTENT_ROIS["processing_roi"] = payload.get("processing_roi")
        if "aruco_roi" in payload:
            PERSISTENT_ROIS["aruco_roi"] = payload.get("aruco_roi")
        if "weight_roi" in payload:
            PERSISTENT_ROIS["weight_roi"] = payload.get("weight_roi")
        save_persistent_rois()
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/stone-settings", methods=["POST"])
def api_stone_settings():
    try:
        payload = parse_post_payload()
        if "color_correction" in payload:
            PERSISTENT_ROIS["color_correction"] = stone_detection.normalize_color_correction(
                payload.get("color_correction")
            )
        if "analysis_normalization" in payload:
            PERSISTENT_ROIS["analysis_normalization"] = (
                stone_detection.normalize_analysis_normalization(
                    payload.get("analysis_normalization")
                )
            )
        if payload.get("clear_background"):
            PERSISTENT_ROIS["background_calibration"] = None
        elif "background_calibration" in payload:
            PERSISTENT_ROIS["background_calibration"] = payload.get("background_calibration")
        if payload.get("clear_learned_stone_profiles"):
            PERSISTENT_ROIS["learned_stone_profiles"] = []
        elif "learned_stone_profiles" in payload:
            PERSISTENT_ROIS["learned_stone_profiles"] = (
                stone_detection.normalize_learned_stone_profiles(
                    payload.get("learned_stone_profiles")
                )
            )
        if "stone_super_resolution" in payload:
            PERSISTENT_ROIS["stone_super_resolution"] = (
                normalize_stone_super_resolution_settings(
                    payload.get("stone_super_resolution")
                )
            )
        save_persistent_rois()

        with STATE_LOCK:
            if CURRENT_STATE:
                source = CURRENT_STATE.get("source") or {}
                source["color_correction"] = copy.deepcopy(PERSISTENT_ROIS["color_correction"])
                source["analysis_normalization"] = copy.deepcopy(
                    PERSISTENT_ROIS["analysis_normalization"]
                )
                source["background_calibration"] = copy.deepcopy(PERSISTENT_ROIS["background_calibration"])
                source["learned_stone_profiles"] = copy.deepcopy(
                    PERSISTENT_ROIS["learned_stone_profiles"]
                )
                source["stone_super_resolution"] = copy.deepcopy(
                    PERSISTENT_ROIS["stone_super_resolution"]
                )
                CURRENT_STATE["source"] = source
                CURRENT_STATE["stone_detection"]["main"] = None
                CURRENT_STATE["stone_detection"]["side"] = None
                CURRENT_STATE["updated_at"] = now_stamp()
                reset_purity_state(CURRENT_STATE)
                build_final_summary(CURRENT_STATE)
        return jsonify(
            {
                "ok": True,
                "stone_settings": stone_settings_for_state(CURRENT_STATE),
                "state": snapshot_state(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/calibration-settings", methods=["POST"])
def api_calibration_settings():
    try:
        payload = parse_post_payload()
        config = normalize_metric_calibration_config(payload)
        PERSISTENT_ROIS["calibration_config"] = config
        save_persistent_rois()
        with STATE_LOCK:
            if CURRENT_STATE:
                source = CURRENT_STATE.get("source") or {}
                source["calibration_config"] = copy.deepcopy(config)
                CURRENT_STATE["source"] = source
                CURRENT_STATE["updated_at"] = now_stamp()
        return jsonify(
            {
                "ok": True,
                "calibration_config": config,
                "state": snapshot_state(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/stone-settings/background-calibrate", methods=["POST"])
def api_stone_background_calibrate():
    try:
        payload = parse_post_payload()
        point = payload.get("point")
        if not isinstance(point, dict):
            return fail("Click a valid background point on the captured image.")
        image_bgr, _filename, _use_live = load_image_from_payload(
            payload,
            "background_calibration.png",
        )
        calibration = calibrate_background_sample(
            image_bgr,
            point,
            radius=int(payload.get("sample_radius", 8)),
        )
        PERSISTENT_ROIS["background_calibration"] = calibration
        save_persistent_rois()

        with STATE_LOCK:
            if CURRENT_STATE:
                source = CURRENT_STATE.get("source") or {}
                source["background_calibration"] = copy.deepcopy(calibration)
                CURRENT_STATE["source"] = source
                CURRENT_STATE["stone_detection"]["main"] = None
                CURRENT_STATE["stone_detection"]["side"] = None
                CURRENT_STATE["updated_at"] = now_stamp()
                reset_purity_state(CURRENT_STATE)
                build_final_summary(CURRENT_STATE)
        return jsonify(
            {
                "ok": True,
                "background_calibration": calibration,
                "stone_settings": stone_settings_for_state(CURRENT_STATE),
                "state": snapshot_state(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/stone-settings/learn-color", methods=["POST"])
def api_stone_learn_color():
    try:
        payload = parse_post_payload()
        point = payload.get("point")
        if not isinstance(point, dict):
            return fail("Click the center of the missed gemstone on the captured image.")
        image_bgr, _filename, _use_live = load_image_from_payload(
            payload,
            "learned_stone_color.png",
        )
        settings = stone_settings_for_state(CURRENT_STATE)
        profile = calibrate_learned_stone_sample(
            image_bgr,
            point,
            expected_color=str(payload.get("expected_color") or "").strip(),
            label=str(payload.get("label") or "").strip(),
            radius=int(payload.get("sample_radius", 8)),
            analysis_normalization=settings["analysis_normalization"],
            background_calibration=settings["background_calibration"],
        )
        existing_profiles = [
            existing
            for existing in (PERSISTENT_ROIS.get("learned_stone_profiles") or [])
            if not (
                str(existing.get("color") or "") == profile["color"]
                and str(existing.get("label") or "").strip().casefold()
                == str(profile["label"]).strip().casefold()
            )
        ]
        profiles = stone_detection.normalize_learned_stone_profiles(
            [*existing_profiles, profile]
        )
        PERSISTENT_ROIS["learned_stone_profiles"] = profiles[-30:]
        save_persistent_rois()

        with STATE_LOCK:
            if CURRENT_STATE:
                source = CURRENT_STATE.get("source") or {}
                source["learned_stone_profiles"] = copy.deepcopy(
                    PERSISTENT_ROIS["learned_stone_profiles"]
                )
                CURRENT_STATE["source"] = source
                CURRENT_STATE["stone_detection"]["main"] = None
                CURRENT_STATE["stone_detection"]["side"] = None
                CURRENT_STATE["updated_at"] = now_stamp()
                reset_purity_state(CURRENT_STATE)
                build_final_summary(CURRENT_STATE)
        return jsonify(
            {
                "ok": True,
                "learned_profile": profile,
                "stone_settings": stone_settings_for_state(CURRENT_STATE),
                "state": snapshot_state(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/pledge", methods=["POST"])
def api_pledge():
    global CURRENT_STATE
    try:
        payload = parse_post_payload()
        pledge_id = str(payload.get("pledge_id") or "").strip()
        if not pledge_id:
            return fail("Pledge ID is required.")
        
        with STATE_LOCK:
            state = ensure_state()
            if state.get("session_id") and state.get("pledge_id") != pledge_id:
                CURRENT_STATE = build_empty_state()
                state = CURRENT_STATE

            metadata = get_or_create_pledge_metadata(pledge_id)
            jewel_index = state.get("jewel_index")
            if not jewel_index and metadata.get("count_saved"):
                jewel_index = next_jewel_index_for_pledge(pledge_id, metadata.get("jewel_count"))
            apply_pledge_metadata_to_state(state, metadata, jewel_index=jewel_index)
            if metadata.get("count_saved"):
                state["status"] = f"Pledge ID set: {pledge_id}. Jewel count saved."
            else:
                state["status"] = f"Pledge ID set: {pledge_id}. Enter jewel count."
            state["updated_at"] = now_stamp()
            lcd_count = metadata.get("jewel_count") if metadata.get("count_saved") else None
            lcd_index = jewel_index if metadata.get("count_saved") else None
            
        lcd_show_pledge_status(pledge_id, lcd_count, lcd_index)
        if lcd_count:
            speak("Pledge ID set. Ready for the jewel workflow.")
        else:
            speak("Enter the jewel count for this pledge.")
        
        return jsonify({"ok": True, "state": snapshot_state()})
    except Exception as exc:
        return fail(str(exc))


@app.route("/api/pledge/count", methods=["POST"])
def api_pledge_count():
    try:
        payload = parse_post_payload()
        clear_count = bool(payload.get("clear"))

        with STATE_LOCK:
            state = ensure_state()
            pledge_id = str(state.get("pledge_id") or payload.get("pledge_id") or "").strip()
            if not pledge_id:
                return fail("Set the Pledge ID before entering jewel count.")

            metadata = get_or_create_pledge_metadata(pledge_id)
            completed_indexes = completed_jewel_indexes_for_pledge(pledge_id)
            analysis_started = bool(get_states_for_pledge(pledge_id) or state.get("session_id"))

            if clear_count:
                if analysis_started:
                    return fail("Jewel count cannot be cleared after jewel analysis has started.")
                metadata["jewel_count"] = None
                metadata["count_saved"] = False
                save_pledge_metadata(metadata)
                apply_pledge_metadata_to_state(state, metadata, jewel_index=None)
                state["jewel_index"] = None
                state["status"] = "Jewel count cleared. Enter the count again."
                state["updated_at"] = now_stamp()
                lcd_show_pledge_status(pledge_id)
                speak("Jewel count cleared. Enter the jewel count again.")
                return jsonify({"ok": True, "state": copy.deepcopy(state)})

            count = normalize_jewel_count(payload.get("jewel_count"))
            minimum_count = max(completed_indexes or {0})
            if state.get("session_id") and state.get("jewel_index"):
                minimum_count = max(minimum_count, int(state["jewel_index"]))
            if count < minimum_count:
                return fail(f"Jewel count cannot be less than the current completed jewel count ({minimum_count}).")

            metadata["jewel_count"] = count
            metadata["count_saved"] = True
            save_pledge_metadata(metadata)
            jewel_index = state.get("jewel_index") or next_jewel_index_for_pledge(pledge_id, count)
            apply_pledge_metadata_to_state(state, metadata, jewel_index=jewel_index)
            state["status"] = f"Jewel count saved: {count}. Ready for jewel {state.get('jewel_index') or 1}."
            state["updated_at"] = now_stamp()

        lcd_show_pledge_status(pledge_id, count, jewel_index)
        speak("Jewel count saved. Click Next to start the jewel workflow.")
        return jsonify({"ok": True, "state": snapshot_state()})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


def pledge_analysis_complete(
    pledge_id: str,
    metadata: dict[str, Any],
    active_state: dict[str, Any] | None = None,
) -> bool:
    try:
        count = int(metadata.get("jewel_count") or 0)
    except Exception:
        count = 0
    if count <= 0:
        return False
    completed = completed_jewel_indexes_for_pledge(pledge_id)
    if (
        active_state
        and active_state.get("pledge_id") == pledge_id
        and is_appraised_jewel_state(active_state)
        and active_state.get("jewel_index")
    ):
        try:
            completed.add(int(active_state["jewel_index"]))
        except Exception:
            pass
    return len(completed) >= count


@app.route("/api/pledge/jewel-count/capture", methods=["POST"])
def api_pledge_jewel_count_capture():
    try:
        with STATE_LOCK:
            state = ensure_state()
            pledge_id = str(state.get("pledge_id") or "").strip()
            if not pledge_id:
                return fail("Set the Pledge ID before capturing the final jewel count.")
            metadata = get_or_create_pledge_metadata(pledge_id)
            if not pledge_analysis_complete(pledge_id, metadata, state):
                speak("Complete all jewels and acid tests before final count capture.")
                return fail("Complete all jewels and acid tests before final count capture.")
            declared_count = int(metadata.get("jewel_count") or 0) or None
            source = state.get("source") or {}
            processing_roi = source.get("processing_roi", PERSISTENT_ROIS.get("processing_roi"))
            aruco_roi = source.get("aruco_roi", PERSISTENT_ROIS.get("aruco_roi"))
            expected_labels = _expected_count_labels_for_pledge(pledge_id, state)

        speak("Place all jewels on the test bed and capture the jewel count.")
        frame = get_live_camera_frame()
        verification = analyze_pledge_jewel_count_capture(
            pledge_id,
            frame,
            declared_count,
            expected_labels,
            processing_roi,
            aruco_roi,
        )

        with STATE_LOCK:
            metadata = get_or_create_pledge_metadata(pledge_id)
            metadata["jewel_count_verification"] = verification
            save_pledge_metadata(metadata)
            state = ensure_state()
            apply_pledge_metadata_to_state(state, metadata)
            clear_stage_skip(state, "final_count")
            state["status"] = (
                f"Predicted jewel count: {verification['predicted_count']}."
            )
            state["updated_at"] = now_stamp()

        if verification.get("match") is False:
            speak("Jewel count mismatch. Please verify before packet sealing.")
        else:
            speak("Jewel count captured. Start packet sealing.")
        return jsonify({"ok": True, "state": snapshot_state()})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/packet-sealing/start", methods=["POST"])
def api_packet_sealing_start():
    global PACKET_RECORDER
    try:
        with STATE_LOCK:
            state = ensure_state()
            pledge_id = str(state.get("pledge_id") or "").strip()
            if not pledge_id:
                return fail("Set the Pledge ID before packet sealing.")
            metadata = get_or_create_pledge_metadata(pledge_id)
            if not pledge_analysis_complete(pledge_id, metadata, state):
                speak("Complete all jewels and acid tests before final count capture.")
                return fail("Complete all jewels before packet sealing.")
            if not metadata.get("jewel_count_verification"):
                speak("Capture the final jewel count before packet sealing.")
                return fail("Capture the final jewel count before packet sealing.")
            source = state.get("source") or {}
            processing_roi = source.get("processing_roi") or PERSISTENT_ROIS.get(
                "processing_roi"
            )

        recorder = get_packet_recorder()
        previous_pledge_id = str(getattr(recorder, "_pledge_id", "") or "").strip()
        if recorder.is_compressing() and previous_pledge_id != pledge_id:
            PACKET_RECORDER = PacketSealingRecorder()
            recorder = PACKET_RECORDER
        packet_state = recorder.start(pledge_id, processing_roi)
        with STATE_LOCK:
            metadata = get_or_create_pledge_metadata(pledge_id)
            metadata["packet_sealing"] = packet_state
            save_pledge_metadata(metadata)
            state = ensure_state()
            apply_pledge_metadata_to_state(state, metadata)
            clear_stage_skip(state, "packet_sealing")
            state["status"] = "Packet sealing recording started."
            state["updated_at"] = now_stamp()

        speak("Packet sealing recording started. Put all jewels into the packet and seal it.")
        return jsonify({"ok": True, "state": snapshot_state()})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


def _update_packet_striping_state(packet_state: dict[str, Any]) -> dict[str, Any]:
    with STATE_LOCK:
        state = ensure_state()
        pledge_id = str(state.get("pledge_id") or "").strip()
        if not pledge_id:
            raise ValueError("No active packet sealing pledge was found.")
        metadata = get_or_create_pledge_metadata(pledge_id)
        metadata["packet_sealing"] = packet_state
        save_pledge_metadata(metadata)
        apply_pledge_metadata_to_state(state, metadata)
        clear_stage_skip(state, "packet_sealing")
        state["updated_at"] = now_stamp()
    return snapshot_state()


@app.route("/api/packet-sealing/striping/hand-clear", methods=["POST"])
def api_packet_striping_hand_clear():
    try:
        packet_state = get_packet_recorder().request_striping_hand_clear()
        state = _update_packet_striping_state(packet_state)
        return jsonify({"ok": True, "state": state})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/packet-sealing/striping/restart", methods=["POST"])
def api_packet_striping_restart():
    try:
        packet_state = get_packet_recorder().restart_striping()
        state = _update_packet_striping_state(packet_state)
        return jsonify({"ok": True, "state": state})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/packet-sealing/striping/skip", methods=["POST"])
def api_packet_striping_skip():
    try:
        packet_state = get_packet_recorder().skip_striping()
        state = _update_packet_striping_state(packet_state)
        return jsonify({"ok": True, "state": state})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/packet-sealing/stop", methods=["POST"])
def api_packet_sealing_stop():
    try:
        recorder = get_packet_recorder()
        packet_state = recorder.stop()
        pledge_id = str(getattr(recorder, "_pledge_id", "") or "")
        with STATE_LOCK:
            packet_state = recorder.snapshot()
            state = ensure_state()
            pledge_id = str(state.get("pledge_id") or pledge_id).strip()
            if not pledge_id:
                return fail("No active packet sealing pledge was found.")
            metadata = get_or_create_pledge_metadata(pledge_id)
            metadata["packet_sealing"] = packet_state
            save_pledge_metadata(metadata)
            apply_pledge_metadata_to_state(state, metadata)
            clear_stage_skip(state, "packet_sealing")
            state["status"] = (
                "Packet sealing video saved."
                if packet_state.get("video")
                else packet_state.get("error") or "Packet sealing stopped."
            )
            state["updated_at"] = now_stamp()

        if packet_state.get("compressing"):
            speak("Packet sealing stopped. Video is still compressing.")
        return jsonify({"ok": True, "state": snapshot_state()})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/jewel/next", methods=["POST"])
def api_next_jewel():
    global CURRENT_STATE
    with STATE_LOCK:
        state = ensure_state()
        pledge_id = state.get("pledge_id")
        if not pledge_id:
            return fail("Set the Pledge ID first.")
        metadata = get_or_create_pledge_metadata(str(pledge_id))
        if not metadata.get("count_saved") or not metadata.get("jewel_count"):
            return fail("Save the jewel count before moving to the next jewel.")
        if not state.get("final", {}).get("ready"):
            return fail("Complete or skip the acid test for the current jewel before moving ahead.")

        count = int(metadata["jewel_count"])
        current_index = int(state.get("jewel_index") or next_jewel_index_for_pledge(str(pledge_id), count))
        retry_same_index = is_confirmed_not_gold_state(state)
        if not retry_same_index and not is_appraised_jewel_state(state):
            return fail("The current item is not ready to advance.")
        if not retry_same_index and current_index >= count:
            return fail("All jewels for this pledge are already completed.")

        next_index = current_index if retry_same_index else current_index + 1
        CURRENT_STATE = build_empty_state()
        state = CURRENT_STATE
        apply_pledge_metadata_to_state(state, metadata, jewel_index=next_index)
        if retry_same_index:
            state["status"] = (
                f"Not-gold item excluded. Ready to capture replacement for jewel "
                f"{next_index} of {count}."
            )
        else:
            state["status"] = f"Ready for jewel {next_index} of {count}."
        state["updated_at"] = now_stamp()
        reset_purity_state(state)

    lcd_show_pledge_status(str(pledge_id), count, next_index)
    if retry_same_index:
        speak("Not gold jewelry confirmed. It was not counted. Capture another item for the same jewel number.")
    else:
        speak("Ready for the next jewel. Capture the jewel image, then run jewel type.")
    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/source", methods=["POST"])
def api_source():
    global CURRENT_STATE
    try:
        payload = parse_post_payload()
        image_bgr, filename, use_live_frame = load_image_from_payload(payload, "source.png")
        raw_image_bgr = image_bgr.copy()
        source_kind = str(payload.get("source_kind") or ("camera" if use_live_frame else "unknown"))
        if source_kind == "camera" or use_live_frame:
            raw_height, raw_width = raw_image_bgr.shape[:2]
            focus_roi = normalize_rect(
                payload.get("processing_roi"),
                raw_width,
                raw_height,
            )
            focus_state = get_camera_backend().focus_snapshot()
            if focus_roi is None:
                raise RuntimeError("Draw the Processing ROI before capturing the jewel image.")
            if focus_state.get("roi") != focus_roi or not focus_state.get("ready"):
                raise RuntimeError(
                    str(focus_state.get("status") or "Wait for camera focus to stabilize before capture.")
                )
        calibration_config = april_tag_calibration_config(payload)
        if source_kind == "upload":
            lens_undistorted = False
            four_marker_calibration = None
        else:
            image_bgr, lens_undistorted = undistort_captured_still(
                image_bgr,
                calibration_config,
            )
            image_bgr, four_marker_calibration = rectify_with_four_apriltags(
                image_bgr,
                calibration_config,
            )
        height, width = image_bgr.shape[:2]
        processing_roi = (
            None
            if four_marker_calibration
            else normalize_rect(payload.get("processing_roi"), width, height)
        )
        aruco_roi = (
            None
            if four_marker_calibration
            else normalize_rect(payload.get("aruco_roi"), width, height)
        )
        working_aruco_roi = translate_rect_to_crop(aruco_roi, processing_roi)
        if four_marker_calibration:
            stone_calibration = four_marker_calibration
        else:
            try:
                stone_calibration = detect_stone_area_calibration(
                    image_bgr,
                    calibration_config,
                    aruco_roi,
                )
            except Exception as exc:
                stone_calibration = {
                    "method": "aruco",
                    **calibration_config,
                    "error": str(exc),
                }

        if source_kind != "upload":
            PERSISTENT_ROIS["processing_roi"] = processing_roi
            PERSISTENT_ROIS["aruco_roi"] = aruco_roi
            PERSISTENT_ROIS["calibration_config"] = copy.deepcopy(calibration_config)
            save_persistent_rois()

        with STATE_LOCK:
            state = ensure_state()
            current_pledge = state.get("pledge_id")
            if not current_pledge:
                return fail("Pledge ID is required. Please enter a Pledge ID first.")
            pledge_context = pledge_context_from_state(state)
            metadata = pledge_context.get("metadata")
            if not metadata or not metadata.get("count_saved") or not metadata.get("jewel_count"):
                return fail("Jewel count is required. Save the jewel count before classification.")
            jewel_index = int(state.get("jewel_index") or next_jewel_index_for_pledge(str(current_pledge), metadata.get("jewel_count")))

        session_id = new_session_id(pledge_id=current_pledge)
        session_dir = RUNTIME_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        raw_original_path = save_bgr(
            session_dir / "source" / f"raw_original_{filename}",
            raw_image_bgr,
        )
        original_path = save_bgr(session_dir / "source" / f"original_{filename}", image_bgr)
        working_bgr = crop_image(image_bgr, processing_roi)
        aruco_ignore_mask = build_aruco_ignore_mask(
            working_bgr.shape,
            working_aruco_roi,
        )
        working_bgr[aruco_ignore_mask > 0] = 255
        working_path = save_bgr(session_dir / "source" / "working_source.png", working_bgr)
        preprocessed_bgr, preprocessed_mask = build_shared_jewelry_mask(
            working_bgr,
            erase_mask=aruco_ignore_mask,
        )
        preprocessed_path = save_bgr(
            session_dir / "source" / "preprocessed.png",
            preprocessed_bgr,
        )
        preprocessed_mask_path = session_dir / "source" / "preprocessed_mask.png"
        cv2.imwrite(
            str(preprocessed_mask_path),
            (preprocessed_mask * 255).astype(np.uint8),
        )
        roi_preview_path = save_bgr(
            session_dir / "source" / "roi_preview.png",
            build_roi_preview(image_bgr, processing_roi, aruco_roi),
        )

        with STATE_LOCK:
            CURRENT_STATE = build_empty_state()
            state = CURRENT_STATE
            apply_pledge_context_to_state(state, pledge_context, jewel_index=jewel_index)
            state["status"] = "Source image captured."
            state["session_id"] = session_id
            state["updated_at"] = now_stamp()
            state["source"] = {
                "kind": source_kind,
                "filename": filename,
                "image_size": {"width": width, "height": height},
                "processing_roi": processing_roi,
                "aruco_roi": aruco_roi,
                "working_aruco_roi": working_aruco_roi,
                "calibration_config": calibration_config,
                "stone_calibration": stone_calibration,
                "color_correction": copy.deepcopy(PERSISTENT_ROIS["color_correction"]),
                "analysis_normalization": copy.deepcopy(PERSISTENT_ROIS["analysis_normalization"]),
                "background_calibration": copy.deepcopy(PERSISTENT_ROIS["background_calibration"]),
                "learned_stone_profiles": copy.deepcopy(PERSISTENT_ROIS["learned_stone_profiles"]),
                "stone_super_resolution": copy.deepcopy(PERSISTENT_ROIS["stone_super_resolution"]),
                "lens_undistorted": lens_undistorted,
                "four_marker_rectified": bool(four_marker_calibration),
                "raw_original_image": artifact_payload(state, raw_original_path),
                "original_image": artifact_payload(state, original_path),
                "working_image": artifact_payload(state, working_path),
                "preprocessed_image": artifact_payload(state, preprocessed_path),
                "preprocessed_mask": artifact_payload(state, preprocessed_mask_path),
                "roi_preview": artifact_payload(state, roi_preview_path),
            }
            build_final_summary(state)

        # Instruction 3: Jewelry image is captured
        speak("Jewel image is captured. Run jewel type.")
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    with STATE_LOCK:
        state = ensure_state()
        print("\n" + "=" * 60)
        print("DEBUG: /api/classify called")
        print(f"Session ID: {state.get('session_id')}")
        print(f"Source: {state['source']}")
        print(f"Working image: {state['source'].get('working_image')}")
        print("=" * 60 + "\n")
        
        if not state.get("session_id") or not state["source"].get("working_image"):
            error_msg = "Capture or upload the primary jewelry image first."
            print(f"VALIDATION ERROR: {error_msg}")
            return fail(error_msg)

        try:
            payload = parse_post_payload()
            jewel_weight = (state.get("weight_details") or {}).get("jewel_weight_g")
            appraiser_stone_weight = normalize_positive_weight(
                payload.get("appraiser_stone_weight_g"),
                "Appraiser stone weight",
                required=False,
            )
            if (
                jewel_weight is not None
                and appraiser_stone_weight is not None
                and appraiser_stone_weight > jewel_weight
            ):
                raise ValueError("Appraiser stone weight cannot exceed the current jewel weight.")
            state["weight_details"] = {
                "jewel_weight_g": jewel_weight,
                "appraiser_stone_weight_g": appraiser_stone_weight,
            }
            working_path = Path(state["source"]["working_image"]["path"])
            print(f"Loading image from: {working_path}")
            print(f"Image exists: {working_path.exists()}")
            prediction = get_classifier().classify_path(working_path)
            session_dir = session_dir_for(state)
            original_preview_path = save_pil(
                session_dir / "classification" / "original_preview.png",
                prediction.original_image,
            )
            cropped_preview_path = save_pil(
                session_dir / "classification" / "cropped_preview.png",
                prediction.cropped_image,
            )

            # Helper function to safely convert numpy arrays to Python scalars
            def to_scalar(val):
                return to_python_scalar(val)
            
            is_gold = bool(getattr(prediction, "is_gold_jewelry", True))
            state["classification"] = {
                "predicted_label": prediction.label,
                "confidence": to_scalar(prediction.confidence),
                "confirmed_label": prediction.label,
                "confirmed": False,
                "gallery_match": prediction.gallery_match,
                "gallery_similarity": to_scalar(prediction.gallery_similarity),
                "scores": [
                    {
                        "label": score.label,
                        "confidence": to_scalar(score.confidence),
                        "similarity": to_scalar(score.similarity),
                    }
                    for score in prediction.scores
                ],
                "original_preview": artifact_payload(state, original_preview_path),
                "cropped_preview": artifact_payload(state, cropped_preview_path),
                "is_gold_jewelry": is_gold,
                "gold_verification_reason": str(getattr(prediction, "gold_verification_reason", "")),
            }
            state["branch"] = branch_for_label(prediction.label) if is_gold else {"key": "none", "label": "Not Gold"}
            state["status"] = f"Classification completed: {prediction.label}"
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)

            if not is_gold:
                speak("This does not look like gold jewelry. Click Yes if correct, or No to override.")
            else:
                speak(f"Jewel type predicted as {prediction.label}. Click Yes if correct, or No to override.")
        except Exception as exc:  # noqa: BLE001
            import traceback
            error_trace = traceback.format_exc()
            print("\n" + "=" * 60)
            print("ERROR IN /api/classify")
            print("=" * 60)
            print(error_trace)
            print("=" * 60 + "\n")
            
            # Check if it's a Hailo error
            if "HAILO" in str(exc).upper() or "PHYSICAL_DEVICES" in str(exc):
                error_msg = f"Hailo device error: {str(exc)}. Make sure the Hailo accelerator is connected and not in use by another process."
            else:
                error_msg = f"Classification failed: {str(exc)}"
            
            return fail(error_msg)

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/speak", methods=["POST"])
def api_speak():
    """Speak an arbitrary phrase on behalf of the frontend (e.g. stage-arrival guidance
    for transitions that happen purely client-side, with no other backend call to hang the
    announcement off of)."""
    try:
        payload = parse_post_payload()
    except ValueError as exc:
        return fail(str(exc))
    text = str(payload.get("text", "")).strip()
    if text:
        speak(text)
    return jsonify({"ok": True})


@app.route("/api/classification/confirm", methods=["POST"])
def api_classification_confirm():
    with STATE_LOCK:
        state = ensure_state()
        if not state["classification"].get("predicted_label"):
            return fail("Run classification first.")

        try:
            payload = parse_post_payload()
            label = str(payload.get("confirmed_label", "")).strip()
            if label not in VALID_CLASSIFICATION_LABELS:
                return fail("Choose a valid jewelry class.")

            old_prediction = state["classification"].get("predicted_label")

            is_not_gold = (label == "Not Gold Jewelry")
            state["classification"]["confirmed_label"] = label
            state["classification"]["confirmed"] = True
            state["classification"]["is_gold_jewelry"] = not is_not_gold
            if is_not_gold:
                state["classification"]["predicted_label"] = "Not Gold Jewelry"
                state["classification"]["scores"] = []
            state["branch"] = branch_for_label(label) if not is_not_gold else {"key": "none", "label": "Not Gold"}
            state["status"] = f"Classification confirmed: {label}"
            state["updated_at"] = now_stamp()
            state["dimension"] = {"done": False}
            state["segmentation"] = {"done": False}
            state["stone_detection"]["main"] = None
            state["stone_detection"]["side"] = None
            state["stage_skips"] = {}
            reset_purity_state(state)
            build_final_summary(state)

            learn_from_confirmation = state.get("source", {}).get("kind") != "upload"

            if is_not_gold:
                speak("Jewel type confirmed. Click next for Jewel Weight Extraction.")
                # Store "Not Gold Jewelry" in gallery so similar non-gold items
                # will be recognised as non-gold in future classifications
                if learn_from_confirmation:
                    try:
                        working_path = Path(state["source"]["working_image"]["path"])
                        get_classifier().learn_correction(working_path, "Not Gold Jewelry")
                        state["status"] += " (Learned as non-gold for gallery)"
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to learn non-gold correction: {exc}")
                # NB: can NOT call snapshot_state() here — STATE_LOCK is already held,
                # and snapshot_state() re-acquires STATE_LOCK → self-deadlock.
                return jsonify({"ok": True, "state": copy.deepcopy(state)})

            # Instruction 4 continued: Click next for ____next-stage___
            speak("Jewel type confirmed. Click next for Jewel Weight Extraction.")

            # Gallery corrections support every UI class, including classes that
            # do not have a dedicated SigLIP text prompt (for example Mangalsutra).
            if label != old_prediction and learn_from_confirmation:
                try:
                    working_path = Path(state["source"]["working_image"]["path"])
                    get_classifier().learn_correction(working_path, label)
                    state["status"] += " (Correction learned for gallery)"
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to learn correction: {exc}")

        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/weight/capture", methods=["POST"])
def api_weight_capture():
    try:
        payload = parse_post_payload()
        with STATE_LOCK:
            state = ensure_state()
            if not (state.get("classification") or {}).get("confirmed"):
                return fail("Confirm the jewel type before capturing its weight.")
            if not state.get("session_id"):
                return fail("Capture the jewellery image before capturing its weight.")
            session_id = str(state["session_id"])
            source = state.get("source") or {}
            processing_roi_raw = (
                source.get("processing_roi")
                or PERSISTENT_ROIS.get("processing_roi")
            )
            weight_roi_raw = PERSISTENT_ROIS.get("weight_roi")
            appraiser_stone_weight = normalize_positive_weight(
                payload.get("appraiser_stone_weight_g"),
                "Appraiser stone weight",
                required=False,
            )

        frame = get_live_camera_frame()
        height, width = frame.shape[:2]
        processing_roi = normalize_rect(processing_roi_raw, width, height)
        if not processing_roi:
            return fail("Draw and save the Processing ROI before capturing weight.")
        weight_roi = normalize_rect(weight_roi_raw, width, height)
        if not weight_roi:
            return fail("Draw and save the Weight Scale ROI in Setup Tools before capturing weight.")
        roi_bgr = crop_image(frame, weight_roi)
        processing_bgr = crop_image(frame, processing_roi)

        with STATE_LOCK:
            state = ensure_state()
            if str(state.get("session_id") or "") != session_id:
                return fail("The active jewel changed before weight capture completed. Recapture its weight.")
            output_dir = session_dir_for(state) / "weight"
        captured_path = save_bgr(output_dir / "weight_capture.png", frame)
        roi_path = save_bgr(output_dir / "weight_roi.png", roi_bgr)

        try:
            reader = get_weight_reader()
            with WEIGHT_READER_LOCK:
                result = reader.read(processing_bgr, output_dir)
        except Exception as exc:  # noqa: BLE001
            result = {
                "success": False,
                "status": "reader_error",
                "message": f"Weight reader is unavailable: {exc}. Recapture after checking the OCR service.",
            }

        evidence_frame = frame.copy()
        evidence_label = (
            f"ACTUAL WEIGHT {float(result['weight_g']):.2f} g"
            if result.get("success")
            else "WEIGHT READ FAILED"
        )
        draw_labeled_rect(evidence_frame, weight_roi, evidence_label, (153, 72, 236))
        save_bgr(captured_path, evidence_frame)

        with STATE_LOCK:
            state = ensure_state()
            if str(state.get("session_id") or "") != session_id:
                return fail("The active jewel changed before weight capture completed. Recapture its weight.")
            lcd_path = result.get("lcd_path")
            weight_state = {
                "success": bool(result.get("success")),
                "status": str(result.get("status") or "read_failed"),
                "message": str(result.get("message") or "Weight could not be read. Recapture."),
                "captured_at": now_stamp(),
                "captured_image": artifact_payload(state, captured_path),
                "roi_image": artifact_payload(state, roi_path),
                "lcd_image": artifact_payload(state, Path(lcd_path)) if lcd_path else None,
            }
            if result.get("success"):
                weight_g = normalize_positive_weight(
                    result.get("weight_g"),
                    "OCR jewel weight",
                    required=True,
                )
                if appraiser_stone_weight is not None and appraiser_stone_weight > weight_g:
                    appraiser_stone_weight = None
                    weight_state["message"] += " Appraiser stone weight was cleared because it exceeded the actual weight."
                weight_state["weight_g"] = weight_g
                weight_state["digits"] = str(result.get("digits") or "")
                state["weight_details"] = {
                    "jewel_weight_g": weight_g,
                    "appraiser_stone_weight_g": appraiser_stone_weight,
                }
            else:
                state["weight_details"] = {
                    "jewel_weight_g": None,
                    "appraiser_stone_weight_g": appraiser_stone_weight,
                }
            state["weight_extraction"] = weight_state
            clear_stage_skip(state, "weight_extraction")
            state["status"] = weight_state["message"]
            state["updated_at"] = now_stamp()
            build_final_summary(state)

        if result.get("success"):
            speak(f"Actual jewel weight is {float(result['weight_g']):.2f} grams. Click next.")
        else:
            speak("Weight could not be detected. Place the weight scale inside the box region and recapture.")
        return jsonify({"ok": True, "state": snapshot_state()})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.route("/api/dimension/run", methods=["POST"])
def api_dimension_run():
    with STATE_LOCK:
        state = ensure_state()
        if not weight_extraction_ready(state):
            return fail("Capture or skip the actual jewel weight before dimension analysis.")
        label = state["classification"].get("confirmed_label")
        if label not in DIMENSION_CLASSES:
            return fail("Dimension measurement is only available for Bangle and Finger ring.")
        try:
            payload = parse_post_payload()
            state["dimension"] = run_dimension_measurement(state, payload)
            clear_stage_skip(state, "dimension")
            state["status"] = "Dimension measurement completed."
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)

            speak(side_capture_voice_prompt(label))
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/side-source", methods=["POST"])
def api_side_source():
    with STATE_LOCK:
        state = ensure_state()
        if not weight_extraction_ready(state):
            return fail("Capture or skip the actual jewel weight before side-image analysis.")
        if not state.get("session_id"):
            return fail("Capture the main jewelry image first.")
        try:
            payload = parse_post_payload()
            image_bgr, filename, _use_live_frame = load_image_from_payload(payload, "side.png")
            raw_image_bgr = image_bgr.copy()
            session_dir = session_dir_for(state)
            
            # 1. Apply the exact same ROI processing as the main image
            source_info = state.get("source", {})
            
            # Prefer payload ROIs (frontend state), fallback to backend state, fallback to persistent ROIs
            raw_processing = payload.get("processing_roi") or source_info.get("processing_roi") or PERSISTENT_ROIS.get("processing_roi")
            raw_aruco = payload.get("aruco_roi") or source_info.get("aruco_roi") or PERSISTENT_ROIS.get("aruco_roi")
            
            calibration_config = april_tag_calibration_config(
                payload,
                fallback=source_info.get("calibration_config"),
            )
            image_bgr, lens_undistorted = undistort_captured_still(
                image_bgr,
                calibration_config,
            )
            try:
                image_bgr, four_marker_calibration = rectify_with_four_apriltags(
                    image_bgr,
                    calibration_config,
                )
            except Exception:
                four_marker_calibration = None
            height, width = image_bgr.shape[:2]
            if four_marker_calibration:
                processing_roi = None
                aruco_roi = None
                side_stone_calibration = four_marker_calibration
            else:
                processing_roi = normalize_rect(raw_processing, width, height)
                aruco_roi = normalize_rect(raw_aruco, width, height)
                try:
                    side_stone_calibration = detect_stone_area_calibration(
                        image_bgr,
                        calibration_config,
                        aruco_roi,
                    )
                except Exception:
                    side_stone_calibration = source_info.get("stone_calibration")
            
            working_bgr = crop_image(image_bgr, processing_roi)
            working_aruco_roi = translate_rect_to_crop(aruco_roi, processing_roi)
            
            # 2. Ignore the ArUco marker by painting it white
            if working_aruco_roi:
                ignore_mask = build_aruco_ignore_mask(
                    working_bgr.shape,
                    working_aruco_roi,
                )
                working_bgr[ignore_mask > 0] = 255
            
            side_raw_original_path = save_bgr(
                session_dir / "side" / f"side_raw_original_{filename}",
                raw_image_bgr,
            )
            side_original_path = save_bgr(session_dir / "side" / f"side_original_{filename}", image_bgr)
            side_preview_path = save_bgr(session_dir / "side" / "side_preview.png", working_bgr)
            
            hand_removed_path = session_dir / "side" / "hand_removed_side.png"
            hand_removed_mask_path = session_dir / "side" / "hand_removed_side_mask.png"
            hand_removal_applied = False
            try:
                with MODEL_LOCK:
                    extract_bangles(
                        str(side_preview_path),
                        str(hand_removed_path),
                        debug=False,
                        mask_output_path=str(hand_removed_mask_path),
                    )
                # Use hand-removed image for subsequent steps
                active_image_path = hand_removed_path
                hand_removal_applied = True
            except Exception as e:
                print(f"Hand removal failed during side capture: {e}")
                active_image_path = side_preview_path
            
            state["side_capture"] = {
                "filename": filename,
                "raw_image": artifact_payload(state, side_original_path),
                "original_image": artifact_payload(state, side_preview_path),
                "preview": artifact_payload(state, active_image_path),
                "hand_removed_image": (
                    artifact_payload(state, hand_removed_path)
                    if hand_removal_applied
                    else None
                ),
                "hand_removed_mask": (
                    artifact_payload(state, hand_removed_mask_path)
                    if hand_removal_applied
                    else None
                ),
                "hand_removal_applied": hand_removal_applied,
                "hand_removal_version": (
                    HAND_REMOVAL_PIPELINE_VERSION
                    if hand_removal_applied
                    else None
                ),
                "stone_calibration": side_stone_calibration,
                "lens_undistorted": lens_undistorted,
                "four_marker_rectified": bool(four_marker_calibration),
                "raw_original_image": artifact_payload(state, side_raw_original_path),
            }
            state["stone_detection"]["side"] = None
            reset_purity_state(state)
            state["status"] = (
                "Side image captured and hand removal applied."
                if hand_removal_applied
                else "Side image captured; hand removal will retry during stone detection."
            )
            state["updated_at"] = now_stamp()

            speak("Side image captured. Click Start Stone Analysis.")
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/side-stones/run", methods=["POST"])
def api_side_stones_run():
    with STATE_LOCK:
        state = ensure_state()
        if not weight_extraction_ready(state):
            return fail("Capture or skip the actual jewel weight before stone analysis.")
        side_capture = state.get("side_capture") or {}
        if not side_capture.get("original_image"):
            return fail("Capture or upload the side image first.")

        try:
            payload = parse_post_payload()
            setting_profile = stone_area_calculator.normalize_stone_setting_profile(
                payload.get("stone_setting_profile")
            )
            state["stone_detection"]["setting_profile"] = setting_profile
            entered_weight = (state.get("weight_details") or {}).get("jewel_weight_g")
            session_dir = session_dir_for(state)
            hand_removed_path = session_dir / "side" / "hand_removed_side.png"
            hand_removed_mask_path = session_dir / "side" / "hand_removed_side_mask.png"
            hand_removed_artifact = side_capture.get("hand_removed_image") or {}
            hand_removed_mask_artifact = side_capture.get("hand_removed_mask") or {}
            existing_hand_removed = Path(
                str(hand_removed_artifact.get("path") or "")
            )
            existing_hand_removed_mask = Path(
                str(hand_removed_mask_artifact.get("path") or "")
            )
            existing_hand_removal_is_current = (
                existing_hand_removed.is_file()
                and existing_hand_removed_mask.is_file()
                and side_capture.get("hand_removal_version")
                == HAND_REMOVAL_PIPELINE_VERSION
            )
            if existing_hand_removal_is_current:
                hand_removed_path = existing_hand_removed
                hand_removed_mask_path = existing_hand_removed_mask
            else:
                side_original_path = Path(side_capture["original_image"]["path"])
                if side_original_path.resolve() == hand_removed_path.resolve():
                    raise RuntimeError("The saved side image is unavailable.")
                with MODEL_LOCK:
                    extract_bangles(
                        str(side_original_path),
                        str(hand_removed_path),
                        debug=False,
                        mask_output_path=str(hand_removed_mask_path),
                    )

            preset_jewelry_mask = cv2.imread(
                str(hand_removed_mask_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if preset_jewelry_mask is None:
                raise RuntimeError("The side jewelry mask could not be loaded.")
            preset_jewelry_mask = (
                (preset_jewelry_mask > 0).astype(np.uint8)
            )

            side_result = run_stone_pipeline(
                state,
                hand_removed_path,
                "side_stone_detection",
                preset_binary_mask=preset_jewelry_mask,
                calibration=stone_calibration_for_state(state, side_image=True),
                entered_jewel_weight_g=entered_weight,
                stone_setting_profile=setting_profile,
            )
            side_result["hand_removed_image"] = artifact_payload(state, hand_removed_path)
            side_result["hand_removed_mask"] = artifact_payload(
                state,
                hand_removed_mask_path,
            )
            side_capture["hand_removed_image"] = artifact_payload(state, hand_removed_path)
            side_capture["hand_removed_mask"] = artifact_payload(
                state,
                hand_removed_mask_path,
            )
            side_capture["hand_removal_applied"] = True
            side_capture["hand_removal_version"] = HAND_REMOVAL_PIPELINE_VERSION
            state["side_capture"] = side_capture

            # For bangle/dimension classes, double side stone values (only one side captured)
            confirmed_label = (state.get("classification") or {}).get("confirmed_label") or ""
            if confirmed_label in DIMENSION_CLASSES:
                jewel_total = side_result.get("jewel_total_px", 1) or 1
                jewel_total_doubled = jewel_total * 2
                doubled_entries = []
                for entry in side_result.get("summary_entries", []):
                    entry = dict(entry)
                    area = min(entry.get("area_px", 0) * 2, jewel_total_doubled)
                    percent = (area / jewel_total_doubled * 100.0) if jewel_total_doubled > 0 else 0.0
                    entry["area_px"] = area
                    entry["stone_percentage"] = round(percent, 2)
                    doubled_entries.append(entry)
                if doubled_entries:
                    side_result["summary_entries"] = doubled_entries
                    total_stone_px = sum(e.get("area_px", 0) for e in doubled_entries)
                    total_stone_percent = min(
                        (total_stone_px / jewel_total_doubled * 100.0) if jewel_total_doubled > 0 else 0.0,
                        100.0,
                    )
                    side_result["stone_area_px"] = min(total_stone_px, jewel_total_doubled)
                    side_result["stone_percentage"] = round(total_stone_percent, 2)
                    side_result["jewel_total_px"] = jewel_total_doubled
                    side_result["summary_text"] = (
                        f"Stone: {total_stone_percent:.1f}% of jewel area (side estimate)"
                    )
                    weight_estimate = side_result.get("weight_estimate") or {}
                    if weight_estimate.get("success"):
                        for key in (
                            "estimated_total_average_ct",
                            "estimated_total_minimum_ct",
                            "estimated_total_maximum_ct",
                            "estimated_total_average_g",
                            "estimated_total_typical_g",
                            "estimated_total_minimum_g",
                            "estimated_total_maximum_g",
                            "raw_estimated_total_average_ct",
                            "raw_estimated_total_minimum_ct",
                            "raw_estimated_total_maximum_ct",
                            "raw_estimated_total_average_g",
                            "raw_estimated_total_minimum_g",
                            "raw_estimated_total_maximum_g",
                            "visible_stone_area_mm2",
                        ):
                            if key in weight_estimate:
                                weight_estimate[key] = round(
                                    float(weight_estimate[key]) * 2.0,
                                    4,
                                )
                        weight_estimate["side_multiplier"] = 2
                        entered_weight_g = weight_estimate.get("entered_jewel_weight_g")
                        if entered_weight_g:
                            weight_estimate[
                                "estimated_stone_share_of_jewel_weight_percent"
                            ] = round(
                                float(weight_estimate["estimated_total_average_g"])
                                / float(entered_weight_g)
                                * 100.0,
                                2,
                            )
                            weight_estimate["estimated_stone_share_range_percent"] = [
                                round(
                                    float(weight_estimate["estimated_total_minimum_g"])
                                    / float(entered_weight_g)
                                    * 100.0,
                                    2,
                                ),
                                round(
                                    float(weight_estimate["estimated_total_maximum_g"])
                                    / float(entered_weight_g)
                                    * 100.0,
                                    2,
                                ),
                            ]
                        weight_estimate["note"] = (
                            f"{weight_estimate.get('note', '')} Total doubled because "
                            "one side was captured for a two-sided dimension-class jewel."
                        ).strip()
                        weight_estimate = stone_area_calculator.calibrate_weight_estimate_to_jewel_weight(
                            weight_estimate,
                            entered_weight_g,
                        )
                        side_result["weight_estimate"] = weight_estimate
                        side_result["estimated_total_minimum_g"] = weight_estimate.get(
                            "estimated_total_minimum_g"
                        )
                        side_result["estimated_total_typical_g"] = weight_estimate.get(
                            "estimated_total_typical_g",
                            weight_estimate.get("estimated_total_average_g"),
                        )
                        side_result["estimated_total_maximum_g"] = weight_estimate.get(
                            "estimated_total_maximum_g"
                        )
                        side_result["weight_confidence"] = weight_estimate.get(
                            "weight_confidence"
                        )
                        feature_vector = side_result.get("stone_weight_feature_vector") or {}
                        feature_vector["side_multiplier"] = 2
                        feature_vector["physics_minimum_g"] = side_result[
                            "estimated_total_minimum_g"
                        ]
                        feature_vector["physics_typical_g"] = side_result[
                            "estimated_total_typical_g"
                        ]
                        feature_vector["physics_maximum_g"] = side_result[
                            "estimated_total_maximum_g"
                        ]
                        side_result["stone_weight_feature_vector"] = feature_vector

            state["stone_detection"]["side"] = side_result
            clear_stage_skip(state, "side_stone")
            state["status"] = "Side image hand removal and stone detection completed."
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)

            speak("Stone analysis completed. Click next for Acid Test.")
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/segmentation/run", methods=["POST"])
def api_segmentation_run():
    with STATE_LOCK:
        state = ensure_state()
        if not weight_extraction_ready(state):
            return fail("Capture or skip the actual jewel weight before jewellery analysis.")
        if not state["classification"].get("confirmed"):
            return fail("Confirm the jewel type before running jewellery analysis.")
        label = state["classification"].get("confirmed_label")
        if label not in SEGMENTATION_CLASSES:
            return fail("Jewellery analysis is only available for Haram, Necklace, Dollar chain, and Kasu Mala.")
        try:
            speak("Jewelry risk analysis initiated.")
            state["segmentation"] = run_segmentation_pipeline(state)
            clear_stage_skip(state, "jewellery_analysis")
            state["stone_detection"]["main"] = None
            if state["segmentation"].get("bead_risk_high"):
                state["status"] = "Jewellery analysis completed. RISK JEWEL: Round beads detected in chain."
            else:
                state["status"] = "Jewellery analysis completed."
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)

            speak("Jewellery analysis completed. Click next for Stone Detection.")
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/segmentation/correct", methods=["POST"])
def api_segmentation_correct():
    with STATE_LOCK:
        state = ensure_state()
        label = state["classification"].get("confirmed_label")
        if label not in SEGMENTATION_CLASSES:
            return fail("This correction is only relevant for the jewellery analysis workflow.")
        try:
            payload = parse_post_payload()
            segmentation = state.get("segmentation") or {}
            preprocessed = segmentation.get("preprocessed_image")
            if not preprocessed:
                return fail("Run jewellery analysis once before applying a correction.")

            image_path = Path(preprocessed["path"])
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                return fail("Could not load the jewellery analysis input image.")

            part = str(payload.get("part") or "pendant").strip().lower()
            if part not in {"pendant", "tassel"}:
                return fail("Correction part must be either pendant or tassel.")

            raw_bbox = payload.get("bbox")
            if raw_bbox is None:
                raw_bbox = payload.get(f"{part}_bbox")

            bbox = normalize_rect(raw_bbox, image_bgr.shape[1], image_bgr.shape[0])
            if not bbox:
                return fail(f"Draw a valid {part} correction box.")

            state["segmentation"] = apply_segmentation_feedback(state, part, bbox)
            state["stone_detection"]["main"] = None
            state["status"] = (
                f"{part.title()} correction learned as a position-invariant visual template; "
                "jewellery analysis rerun."
            )
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/segmentation/no-pendant", methods=["POST"])
def api_segmentation_no_pendant():
    with STATE_LOCK:
        state = ensure_state()
        label = state["classification"].get("confirmed_label")
        if label not in SEGMENTATION_CLASSES:
            return fail("This action is only relevant for the jewellery analysis workflow.")
        if not (state.get("segmentation") or {}).get("done"):
            return fail("Run jewellery analysis once before excluding pendant.")
        try:
            state["segmentation"] = apply_segmentation_no_part(state, "pendant")
            state["stone_detection"]["main"] = None
            state["status"] = "Pendant excluded; region merged into chain. Jewellery analysis rerun."
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/segmentation/no-tassel", methods=["POST"])
def api_segmentation_no_tassel():
    with STATE_LOCK:
        state = ensure_state()
        label = state["classification"].get("confirmed_label")
        if label not in SEGMENTATION_CLASSES:
            return fail("This action is only relevant for the jewellery analysis workflow.")
        if not (state.get("segmentation") or {}).get("done"):
            return fail("Run jewellery analysis once before excluding tassel.")
        try:
            state["segmentation"] = apply_segmentation_no_part(state, "tassel")
            state["stone_detection"]["main"] = None
            state["status"] = "Tassel excluded; region merged into chain. Jewellery analysis rerun."
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/api/stone-detection/main", methods=["POST"])
def api_stone_detection_main():
    with STATE_LOCK:
        state = ensure_state()
        if not weight_extraction_ready(state):
            return fail("Capture or skip the actual jewel weight before stone analysis.")
        if not state["source"].get("working_image"):
            return fail("Capture or upload the main jewelry image first.")

        try:
            payload = parse_post_payload()
            setting_profile = stone_area_calculator.normalize_stone_setting_profile(
                payload.get("stone_setting_profile")
            )
            state["stone_detection"]["setting_profile"] = setting_profile
            entered_weight = (state.get("weight_details") or {}).get("jewel_weight_g")
            segmentation = state.get("segmentation") or {}
            working_path, working_bgr, shared_jewelry_mask = (
                load_or_create_shared_jewelry_input(state)
            )
            ignore_mask = None
            jewel_total_px_override = None

            if segmentation.get("done"):
                part_summary = segmentation.get("part_summary") or {}
                pendant_area = int((part_summary.get("pendant") or {}).get("area", 0))
                chain_area = int((part_summary.get("chain") or {}).get("area", 0))
                if pendant_area > 0 or chain_area > 0:
                    jewel_total_px_override = pendant_area + chain_area

                tassel_mask_artifact = (segmentation.get("part_masks") or {}).get("tassel")
                if tassel_mask_artifact:
                    ignore_mask = load_artifact_mask(tassel_mask_artifact, working_bgr.shape[:2])

            stone_candidate_mask = shared_jewelry_mask.copy()
            if ignore_mask is not None:
                stone_candidate_mask[ignore_mask > 0] = 0

            state["stone_detection"]["main"] = run_stone_pipeline(
                state,
                working_path,
                "main_stone_detection",
                ignore_mask=ignore_mask,
                preset_binary_mask=stone_candidate_mask,
                calibration=stone_calibration_for_state(state),
                entered_jewel_weight_g=entered_weight,
                stone_setting_profile=setting_profile,
            )
            
            # Keep the stone pipeline's cleaned jewel mask as the percentage
            # denominator. Segmentation area is useful for auditing, but it may
            # contain filled hollow/background pixels and must not overwrite the
            # exact mask used to detect stones.
            if jewel_total_px_override is not None:
                main_stones = state["stone_detection"]["main"]
                analyzed_area = int(main_stones.get("jewel_total_px", 0))
                main_stones["segmentation_jewel_area_px"] = jewel_total_px_override
                main_stones["segmentation_area_difference_px"] = (
                    jewel_total_px_override - analyzed_area
                )

            state["status"] = "Main image stone detection completed."
            clear_stage_skip(state, "stone_detection")
            state["updated_at"] = now_stamp()
            reset_purity_state(state)
            build_final_summary(state)

            speak("Stone detection completed. Click next for Acid Test.")
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))

    return jsonify({"ok": True, "state": snapshot_state()})


@app.route("/artifacts/<session_id>/<path:relative_path>")
def serve_artifact(session_id: str, relative_path: str):
    base = (RUNTIME_DIR / session_id).resolve()
    target = (base / relative_path).resolve()
    if base not in target.parents and target != base:
        abort(404)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_file(target)


@app.route("/pledge-artifacts/<safe_pledge_id>/<path:relative_path>")
def serve_pledge_artifact(safe_pledge_id: str, relative_path: str):
    safe_id = sanitize_filename(safe_pledge_id, "pledge")
    base = (PLEDGE_MEDIA_DIR / safe_id).resolve()
    target = (base / relative_path).resolve()
    if base not in target.parents and target != base:
        abort(404)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_file(target)


def get_states_for_pledge(pledge_id: str) -> list[dict[str, Any]]:
    states = []
    if not pledge_id:
        return states
    
    prefix = f"{pledge_id}_"
    for p in RUNTIME_DIR.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            state_file = p / "state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        states.append(json.load(f))
                except Exception:
                    pass
    
    states.sort(key=lambda s: (int(s.get("jewel_index") or 999999), s.get("updated_at", "")))
    return states


def completed_jewel_indexes_for_pledge(pledge_id: str) -> set[int]:
    indexes: set[int] = set()
    for state in get_states_for_pledge(pledge_id):
        if not is_appraised_jewel_state(state):
            continue
        try:
            index = int(state.get("jewel_index") or 0)
        except Exception:
            index = 0
        if index > 0:
            indexes.add(index)
    return indexes


def next_jewel_index_for_pledge(pledge_id: str, jewel_count: Any = None) -> int:
    completed = completed_jewel_indexes_for_pledge(pledge_id)
    count = int(jewel_count or 0)
    next_index = 1
    while next_index in completed:
        next_index += 1
    if count > 0:
        next_index = min(next_index, count)
    return max(1, next_index)


def update_pledge_progress(state: dict[str, Any]) -> None:
    pledge_id = state.get("pledge_id")
    metadata = load_pledge_metadata(str(pledge_id)) if pledge_id else None
    if metadata:
        apply_pledge_metadata_to_state(state, metadata)

    count = int(state.get("jewel_count") or 0)
    current_index = int(state.get("jewel_index") or 0)
    completed = completed_jewel_indexes_for_pledge(str(pledge_id)) if pledge_id else set()
    current_appraised = is_appraised_jewel_state(state)
    retry_current = bool(
        is_confirmed_not_gold_state(state)
        and state.get("final", {}).get("ready")
    )
    if current_appraised and current_index > 0:
        completed.add(current_index)

    state["completed_jewels"] = len(completed)
    state["is_last_jewel"] = bool(
        current_appraised and count and current_index >= count
    )
    state["next_jewel_available"] = bool(
        current_appraised and count and current_index < count
    )
    state["retry_jewel_available"] = bool(
        retry_current and count and len(completed) < count
    )
    state["pledge_complete"] = bool(count and len(completed) >= count)


def post_jewel_voice_prompt(state: dict[str, Any], action_label: str = "completed") -> str:
    update_pledge_progress(state)
    if state.get("retry_jewel_available"):
        return "Not gold jewelry confirmed. It was not counted. Capture another item for the same jewel number."
    if state.get("next_jewel_available"):
        return "Current jewel completed. Click Next Jewel to continue."
    if state.get("pledge_complete") or state.get("final", {}).get("ready"):
        return "All jewels are completed. Capture the final jewel count."
    return "Current jewel completed. Continue the workflow."


def _packet_video_path_from_metadata(metadata: dict[str, Any] | None) -> Path | None:
    packet = (metadata or {}).get("packet_sealing") or {}
    video = packet.get("video") if isinstance(packet, dict) else None
    if not isinstance(video, dict):
        return None
    raw_path = video.get("path")
    if not raw_path:
        return None
    try:
        path = Path(raw_path).resolve()
    except Exception:
        return None
    pledge_id = str((metadata or {}).get("pledge_id") or "").strip()
    if not pledge_id:
        return None
    base = pledge_media_dir(pledge_id).resolve()
    if base not in path.parents and path != base:
        return None
    return path if path.exists() and path.is_file() else None


def _pledge_closure_complete(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    count_verification = metadata.get("jewel_count_verification") or {}
    packet = metadata.get("packet_sealing") or {}
    count_done = bool(count_verification)
    packet_done = bool(packet.get("skipped") or _packet_video_path_from_metadata(metadata))
    return bool(count_done and packet_done)


@app.route("/api/reports", methods=["GET"])
def api_reports():
    reports: list[dict[str, Any]] = []
    for metadata_path in PLEDGE_DIR.glob("*.json"):
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pledge_id = str(raw_metadata.get("pledge_id") or "").strip()
            metadata = load_pledge_metadata(pledge_id)
            if not pledge_id or not metadata or not _pledge_closure_complete(metadata):
                continue
            states = [
                state
                for state in get_states_for_pledge(pledge_id)
                if is_appraised_jewel_state(state)
            ]
            if not states:
                continue
            jewel_count = int(metadata.get("jewel_count") or 0)
            completed_count = len(completed_jewel_indexes_for_pledge(pledge_id))
            if jewel_count and completed_count < jewel_count:
                continue
            video_path = _packet_video_path_from_metadata(metadata)
            encoded_id = quote(pledge_id, safe="")
            reports.append(
                {
                    "pledge_id": pledge_id,
                    "started_at": metadata.get("started_at"),
                    "updated_at": metadata.get("updated_at"),
                    "jewel_count": jewel_count or completed_count,
                    "pdf_url": f"/api/report/pdf?pledge_id={encoded_id}&download=1",
                    "video_url": (
                        f"/api/report/video?pledge_id={encoded_id}"
                        if video_path
                        else None
                    ),
                    "video_name": video_path.name if video_path else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Could not list pledge report {metadata_path.name}: {exc}")
    reports.sort(key=lambda report: str(report.get("started_at") or ""), reverse=True)
    return jsonify({"ok": True, "reports": reports})


@app.route("/api/report/assets", methods=["GET"])
def api_report_assets():
    pledge_id = str(request.args.get("pledge_id") or "").strip()
    if not pledge_id:
        return fail("Pledge ID is required.", 400)
    metadata = load_pledge_metadata(pledge_id)
    if not metadata:
        return fail(f"No pledge metadata found for pledge ID: {pledge_id}", 404)
    video_path = _packet_video_path_from_metadata(metadata)
    return jsonify(
        {
            "ok": True,
            "pledge_id": pledge_id,
            "has_packet_video": bool(video_path),
            "packet_video_url": (
                f"/api/report/video?pledge_id={quote(pledge_id, safe='')}"
                if video_path
                else None
            ),
            "packet_video_name": video_path.name if video_path else None,
        }
    )


@app.route("/api/report/video", methods=["GET"])
def api_report_packet_video():
    pledge_id = str(request.args.get("pledge_id") or "").strip()
    if not pledge_id:
        return fail("Pledge ID is required.", 400)
    metadata = load_pledge_metadata(pledge_id)
    video_path = _packet_video_path_from_metadata(metadata)
    if not video_path:
        return fail(f"No packet sealing video found for pledge ID: {pledge_id}", 404)
    return send_file(
        video_path,
        as_attachment=True,
        download_name=video_path.name,
        mimetype="video/mp4",
    )


@app.route("/api/report/pdf", methods=["GET"])
def api_generate_pdf():
    pledge_id = request.args.get("pledge_id")
    download_pdf = request.args.get("download") == "1"
    with STATE_LOCK:
        if pledge_id:
            metadata = load_pledge_metadata(pledge_id) or {"pledge_id": pledge_id}
            states = get_states_for_pledge(pledge_id)
            if not states:
                return fail(f"No sessions found for pledge ID: {pledge_id}", 404)
            
            states = [s for s in states if is_appraised_jewel_state(s)]
            active_state = ensure_state()
            if (
                active_state.get("pledge_id") == pledge_id
                and is_appraised_jewel_state(active_state)
                and active_state.get("session_id")
                and all(s.get("session_id") != active_state.get("session_id") for s in states)
            ):
                states.append(copy.deepcopy(active_state))
                states.sort(key=lambda s: (int(s.get("jewel_index") or 999999), s.get("updated_at", "")))
            if not states:
                return fail(f"No completed analysis found for pledge ID: {pledge_id}", 400)
            jewel_count = int(metadata.get("jewel_count") or 0)
            completed_indexes = completed_jewel_indexes_for_pledge(pledge_id)
            if is_appraised_jewel_state(active_state) and active_state.get("jewel_index"):
                completed_indexes.add(int(active_state["jewel_index"]))
            if jewel_count and len(completed_indexes) < jewel_count:
                return fail(f"Pledge report is not ready. Complete all {jewel_count} jewels first.", 400)
            if not _pledge_closure_complete(metadata):
                return fail("Pledge report is not ready. Complete or skip final count capture and packet sealing video first.", 400)
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata["result_generated_at"] = now_stamp()
                save_pledge_metadata(metadata)
                pdf_buffer = generate_pdf_report(states, metadata)
                filename = f"jewelry_report_{pledge_id}_{timestamp}.pdf"
                
                return send_file(
                    pdf_buffer,
                    as_attachment=download_pdf,
                    download_name=filename,
                    mimetype="application/pdf"
                )
            except Exception as exc:
                return fail(str(exc), 500)
        else:
            state = ensure_state()
            if not state.get("session_id"):
                return fail("No active session. Please capture or upload an image first.", 404)
            
            if not is_appraised_jewel_state(state):
                return fail("Analysis not complete. Please complete the workflow before generating a report.", 400)
            
            current_pledge = state.get("pledge_id")
            metadata = load_pledge_metadata(current_pledge) if current_pledge else None
            if current_pledge:
                states = get_states_for_pledge(current_pledge)
                states = [s for s in states if is_appraised_jewel_state(s)]
                if is_appraised_jewel_state(state):
                    current_session = state.get("session_id")
                    if current_session and all(s.get("session_id") != current_session for s in states):
                        states.append(copy.deepcopy(state))
                        states.sort(key=lambda s: (int(s.get("jewel_index") or 999999), s.get("updated_at", "")))
                if not states:
                     states = [state]
                jewel_count = int((metadata or {}).get("jewel_count") or 0)
                completed_indexes = completed_jewel_indexes_for_pledge(current_pledge)
                if is_appraised_jewel_state(state) and state.get("jewel_index"):
                    completed_indexes.add(int(state["jewel_index"]))
                if jewel_count and len(completed_indexes) < jewel_count:
                    return fail(f"Pledge report is not ready. Complete all {jewel_count} jewels first.", 400)
                if not _pledge_closure_complete(metadata):
                    return fail("Pledge report is not ready. Complete or skip final count capture and packet sealing video first.", 400)
            else:
                states = [state]

            try:
                session_id = state.get("session_id", "unknown")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if metadata is not None:
                    metadata["result_generated_at"] = now_stamp()
                    save_pledge_metadata(metadata)
                pdf_buffer = generate_pdf_report(states, metadata)
                filename = f"jewelry_report_{session_id}_{timestamp}.pdf"
                
                return send_file(
                    pdf_buffer,
                    as_attachment=download_pdf,
                    download_name=filename,
                    mimetype="application/pdf"
                )
            except Exception as exc:
                return fail(str(exc), 500)


if __name__ == "__main__":
    load_persistent_rois()
    load_project_camera_calibration()
    preload_ocr_model()
    with STATE_LOCK:
        CURRENT_STATE = build_empty_state()
    cleanup_old_runtime_sessions()
    threading.Thread(
        target=storage_cleanup_worker,
        name="storage-cleanup",
        daemon=True,
    ).start()
    get_camera_backend().start()
    print(f"Integrated UI running on http://0.0.0.0:{APP_PORT}")
    print(f"Raspberry Pi camera device: {CAM_DEVICE} (index {CAM_INDEX})")
    print(f"Camera target resolution: {CAM_TARGET_WIDTH}x{CAM_TARGET_HEIGHT}; rotation={CAM_ROTATE_90_CLOCKWISE}")
    print(f"Camera exposure mode: {CAMERA_EXPOSURE_MODE}")
    
    lcd_show(["EmbSys AI", "Ready"])
    
    # Preload the remaining inference models before accepting operator requests.
    preload_tts_phrases()
    preload_hef_models()
    lcd_show_pledge_status(None)

    # Instruction 1: Enter the Pledge ID
    def _announce_start():
        time.sleep(2.0)
        speak("Enter the Pledge ID")
    threading.Thread(target=_announce_start, daemon=True).start()

    app.run(host="0.0.0.0", port=APP_PORT, debug=False, threaded=True)
