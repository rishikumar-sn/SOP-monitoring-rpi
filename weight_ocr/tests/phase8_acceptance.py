"""Check the labeled PaddleOCR evaluation report against the safety target."""

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_DIR / "validation" / "phase8" / "paddleocr_report.json"


def main():
    if not REPORT_PATH.is_file():
        print("FAIL: run experiments/paddleocr_eval.py before this check")
        return 1

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    scores = report["agreement"]["scores"]["0.70"]
    sample_count = report["sample_count"]
    if sum(scores.values()) != sample_count:
        print(f"FAIL: report count mismatch: samples={sample_count} scores={scores}")
        return 1
    if scores["wrong"] != 0 or scores["correct"] < 50:
        print(f"FAIL: PaddleOCR safety target was not met: {scores}")
        return 1

    print(
        f"PASS: PaddleOCR two-crop agreement on {sample_count} labeled samples: "
        f"{scores['correct']} correct, {scores['wrong']} wrong, "
        f"{scores['failed']} explicit READ FAILED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
