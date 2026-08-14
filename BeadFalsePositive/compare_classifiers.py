from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = TOOL_DIR / "model_projects" / "bead_finder"
CLASS_NAMES = ("false_positive", "true_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare classifier reports and dataset health.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_health(labels_dir: Path) -> dict:
    class_counts = {}
    shortest_sides = []
    for class_name in CLASS_NAMES:
        paths = sorted((labels_dir / class_name).glob("*.png"))
        class_counts[class_name] = len(paths)
        for path in paths:
            with Image.open(path) as image:
                shortest_sides.append(min(image.size))
    return {
        "class_counts": class_counts,
        "crop_count": len(shortest_sides),
        "crops_below_24_pixels": sum(side < 24 for side in shortest_sides),
        "median_shortest_side": statistics.median(shortest_sides) if shortest_sides else 0,
    }


def metric_summary(name: str, report: dict | None) -> dict:
    if report is None:
        return {"model": name, "available": False}
    test = report.get("test_metrics") or {}
    locked = report.get("candidate_locked_metrics") or {}
    confusion = locked.get("confusion_matrix") or [[0, 0], [0, 0]]
    test_accuracy = float(test.get("accuracy", 0.0))
    locked_accuracy = float(locked.get("accuracy", 0.0))
    return {
        "model": name,
        "available": True,
        "ordinary_test_accuracy": test_accuracy,
        "locked_accuracy": locked_accuracy,
        "generalization_gap": test_accuracy - locked_accuracy,
        "locked_true_detection_recall": float(locked.get("true_detection_recall", 0.0)),
        "locked_false_positive_acceptance_rate": float(
            locked.get("false_positive_acceptance_rate", 0.0)
        ),
        "locked_false_positives_accepted": int(confusion[0][1]),
        "locked_true_detections_rejected": int(confusion[1][0]),
    }


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    health = dataset_health(project / "dataset" / "labels")
    models = [
        metric_summary(
            "ResNet18",
            load_report(project / "ResNet18" / "beadcheck_candidate_training_report.json"),
        ),
        metric_summary(
            "MobileNetV3-Small",
            load_report(
                project
                / "MobileNetV3"
                / "beadcheck_mobilenet_v3_candidate_training_report.json"
            ),
        ),
    ]
    risks = []
    if health["crop_count"] and health["crops_below_24_pixels"] / health["crop_count"] > 0.25:
        risks.append("more than 25% of crops are below 24 pixels")
    if any(model.get("generalization_gap", 0.0) > 0.20 for model in models):
        risks.append("ordinary test accuracy exceeds locked accuracy by more than 20 points")
    result = {
        "project": str(project),
        "dataset": health,
        "models": models,
        "dataset_collection_risk": bool(risks),
        "risk_reasons": risks,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
