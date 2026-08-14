"""Evaluate PaddleOCR recognition on the labeled Phase 8 LCD crops."""

import json
from pathlib import Path
import re

import cv2
from paddleocr import TextRecognition


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "paddleocr"
    / "official_models"
    / "en_PP-OCRv5_mobile_rec"
)
INPUT_DIR = PROJECT_DIR / "validation" / "phase8" / "paddleocr_inputs"
REPORT_PATH = PROJECT_DIR / "validation" / "phase8" / "paddleocr_report.json"
EVALUATED_VARIANTS = ("digit_color", "digit_top08")


def normalize_digits(text):
    value = str(text).replace(".", "").replace(" ", "")
    return value if re.fullmatch(r"\d{1,4}", value) else None


def score(rows, confidence_threshold):
    correct = wrong = failed = 0
    for row in rows:
        predicted = row["predicted_digits"]
        if predicted is None or row["confidence"] < confidence_threshold:
            failed += 1
        elif predicted == row["true_digits"]:
            correct += 1
        else:
            wrong += 1
    return {"correct": correct, "wrong": wrong, "failed": failed}


def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"PaddleOCR model is missing: {MODEL_DIR}")
    manifest_path = INPUT_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Prepared inputs are missing; run prepare_paddleocr_inputs.py first"
        )

    reader = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        model_dir=str(MODEL_DIR),
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=4,
    )
    prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(prepared)} labeled LCD crops")

    report = {
        "model": "en_PP-OCRv5_mobile_rec",
        "sample_count": len(prepared),
        "variants": {},
    }
    for variant_name in EVALUATED_VARIANTS:
        images = [cv2.imread(item["files"][variant_name]) for item in prepared]
        predictions = reader.predict(input=images, batch_size=8)
        rows = []
        for item, prediction in zip(prepared, predictions):
            values = prediction.json["res"]
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "true_digits": item["true_digits"],
                    "raw_text": values["rec_text"],
                    "predicted_digits": normalize_digits(values["rec_text"]),
                    "confidence": float(values["rec_score"]),
                }
            )
        scores = {
            f"{threshold:.2f}": score(rows, threshold)
            for threshold in (0.0, 0.5, 0.7, 0.8, 0.9)
        }
        report["variants"][variant_name] = {"scores": scores, "samples": rows}
        print(variant_name, scores)

    first = report["variants"][EVALUATED_VARIANTS[0]]["samples"]
    second = report["variants"][EVALUATED_VARIANTS[1]]["samples"]
    agreement_rows = []
    for left, right in zip(first, second):
        agreed = left["predicted_digits"] == right["predicted_digits"]
        agreement_rows.append(
            {
                "sample_id": left["sample_id"],
                "true_digits": left["true_digits"],
                "predicted_digits": left["predicted_digits"] if agreed else None,
                "confidence": min(left["confidence"], right["confidence"]),
            }
        )
    report["agreement"] = {
        "variants": list(EVALUATED_VARIANTS),
        "scores": {
            f"{threshold:.2f}": score(agreement_rows, threshold)
            for threshold in (0.0, 0.5, 0.7, 0.8, 0.9)
        },
        "samples": agreement_rows,
    }
    print("agreement", report["agreement"]["scores"])

    reader.close()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
