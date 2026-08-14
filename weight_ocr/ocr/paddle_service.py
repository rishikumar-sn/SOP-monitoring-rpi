"""Long-lived PaddleOCR subprocess using JSON lines over stdin/stdout."""

import json
import os
from pathlib import Path
import re
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "paddleocr"
    / "official_models"
    / "en_PP-OCRv5_mobile_rec"
)
os.environ.setdefault("PADDLE_PDX_EAGER_INIT", "False")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    str(PROJECT_DIR / "models" / "paddleocr"),
)

import cv2
from paddleocr import TextRecognition


def emit(payload):
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def normalize_digits(text):
    value = str(text).replace(".", "").replace(" ", "")
    return value if re.fullmatch(r"\d{1,4}", value) else None


def main():
    reader = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(MODEL_DIR),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=4,
    )
    emit({"status": "ready"})
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("op") == "shutdown":
                    emit({"status": "stopped"})
                    break
                paths = request.get("paths") or []
                if request.get("op") != "recognize" or len(paths) != 2:
                    raise ValueError("Expected a recognize request with two crop paths")
                images = [cv2.imread(str(path)) for path in paths]
                if any(image is None or image.size == 0 for image in images):
                    raise ValueError("PaddleOCR could not read one of the crop images")
                results = reader.predict(input=images, batch_size=2)
                variants = []
                for result in results:
                    values = result.json["res"]
                    variants.append(
                        {
                            "text": values["rec_text"],
                            "digits": normalize_digits(values["rec_text"]),
                            "confidence": float(values["rec_score"]),
                        }
                    )
                agreed = (
                    variants[0]["digits"] is not None
                    and variants[0]["digits"] == variants[1]["digits"]
                )
                emit(
                    {
                        "status": "ok",
                        "agreed": agreed,
                        "digits": variants[0]["digits"] if agreed else None,
                        "confidence": min(
                            variant["confidence"] for variant in variants
                        ),
                        "variants": variants,
                    }
                )
            except Exception as exc:
                emit({"status": "error", "message": str(exc)})
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
