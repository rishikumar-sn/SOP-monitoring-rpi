from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CLASSES = [
    "JEWEL_RUB_OK",
    "NAIL_RUB_NOK",
    "FINGER_RUB_NOK",
    "STONE_TAP_HANDLING_NOK",
    "EXTERNAL_NOISE_NOK",
    "NO_RUB_NOK",
    "WEAK_RUB_NOK",
]

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DURATION = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def dataset_root() -> Path:
    return project_root() / "dataset"


def models_root() -> Path:
    return project_root() / "models"


def ensure_project_folders() -> None:
    dataset_root().mkdir(parents=True, exist_ok=True)
    models_root().mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (dataset_root() / class_name).mkdir(parents=True, exist_ok=True)


def open_in_file_manager(path: Path) -> None:
    path = path.resolve()
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def sanitize_filename(value: str, fallback: str = "operator") -> str:
    cleaned = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value.strip()
    )
    return cleaned.strip("_") or fallback

