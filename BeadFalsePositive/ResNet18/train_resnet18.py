from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = SCRIPT_DIR.parent / "model_projects" / "bead_finder"
DEFAULT_DATASET = DEFAULT_PROJECT / "dataset" / "labels"
DEFAULT_BASE_MODEL = DEFAULT_PROJECT / "ResNet18" / "beadcheck.pt"
DEFAULT_OUTPUT = DEFAULT_PROJECT / "ResNet18" / "beadcheck_candidate.pt"
DEFAULT_LOCKED_EVALUATION = (
    DEFAULT_PROJECT / "ResNet18" / "locked_crop_evaluation_manifest.json"
)
CLASS_TO_INDEX = {"false_positive": 0, "true_detection": 1}
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_TARGETS = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)


def session_id_from_path(path: Path) -> str:
    return path.stem.rsplit("_candidate_", 1)[0]


def collect_samples(dataset_root: Path) -> list[dict]:
    samples = []
    for class_name, class_index in CLASS_TO_INDEX.items():
        for path in sorted((dataset_root / class_name).glob("*.png")):
            samples.append(
                {
                    "path": path,
                    "class_name": class_name,
                    "class_index": class_index,
                    "session_id": session_id_from_path(path),
                }
            )
    return samples


def split_samples_by_crop(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    split_samples = {name: [] for name in SPLIT_NAMES}
    for class_name in CLASS_TO_INDEX:
        class_samples = [sample for sample in samples if sample["class_name"] == class_name]
        if len(class_samples) < 7:
            raise RuntimeError(
                f"At least seven {class_name} crops are required for stratified splits."
            )
        order = rng.permutation(len(class_samples))
        shuffled = [class_samples[int(index)] for index in order]
        validation_count = max(1, int(round(len(shuffled) * SPLIT_TARGETS[1])))
        test_count = max(1, int(round(len(shuffled) * SPLIT_TARGETS[2])))
        train_count = len(shuffled) - validation_count - test_count
        split_samples["train"].extend(shuffled[:train_count])
        split_samples["validation"].extend(
            shuffled[train_count : train_count + validation_count]
        )
        split_samples["test"].extend(shuffled[train_count + validation_count :])
    for split_name in SPLIT_NAMES:
        rng.shuffle(split_samples[split_name])
    return split_samples


class BeadCropDataset(Dataset):
    def __init__(self, samples: list[dict], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample["class_index"]


def make_model(pretrained: bool, architecture: str = "resnet18") -> nn.Module:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 2),
        )
        return model
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def set_trainable_parameters(model: nn.Module, architecture: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if architecture == "resnet18":
        modules = [model.fc]
    elif architecture == "mobilenet_v3_small":
        modules = [model.features[-3:], model.classifier]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def keep_frozen_batch_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


def class_counts(samples: list[dict]) -> dict[str, int]:
    return {
        class_name: sum(sample["class_name"] == class_name for sample in samples)
        for class_name in CLASS_TO_INDEX
    }


def sample_manifest_entry(sample: dict, dataset_root: Path) -> dict:
    return {
        "path": str(sample["path"].relative_to(dataset_root)),
        "class_name": sample["class_name"],
        "class_index": sample["class_index"],
        "session_id": sample["session_id"],
    }


def load_or_create_locked_evaluation(
    samples: list[dict],
    dataset_root: Path,
    manifest_path: Path,
    seed: int,
) -> list[dict]:
    sample_by_path = {
        str(sample["path"].relative_to(dataset_root)): sample for sample in samples
    }
    locked_root = manifest_path.parent / manifest_path.stem.removesuffix("_manifest")
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("samples", [])
        locked_samples = []
        migrated = False
        for entry in entries:
            locked_relative = entry.get("locked_path")
            if not locked_relative:
                source = sample_by_path.get(str(entry.get("path")))
                if source is None:
                    raise RuntimeError(
                        f"Could not preserve locked evaluation sample: {entry.get('path')}"
                    )
                locked_relative = str(Path(source["class_name"]) / source["path"].name)
                target = locked_root / locked_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source["path"], target)
                entry["source_path"] = entry.pop("path")
                entry["locked_path"] = locked_relative
                migrated = True
            locked_path = locked_root / str(locked_relative)
            if not locked_path.exists():
                raise RuntimeError(f"Locked evaluation image is missing: {locked_path}")
            locked_samples.append(
                {
                    "path": locked_path,
                    "class_name": str(entry["class_name"]),
                    "class_index": int(entry["class_index"]),
                    "session_id": str(entry["session_id"]),
                    "source_relative_path": str(entry.get("source_path") or entry.get("path")),
                }
            )
        if migrated:
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return locked_samples

    initial_splits = split_samples_by_crop(samples, seed)
    locked_samples = initial_splits["test"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    preserved_samples = []
    for sample in locked_samples:
        locked_relative = Path(sample["class_name"]) / sample["path"].name
        target = locked_root / locked_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["path"], target)
        entries.append(
            {
                **sample_manifest_entry(sample, dataset_root),
                "source_path": str(sample["path"].relative_to(dataset_root)),
                "locked_path": locked_relative.as_posix(),
            }
        )
        entries[-1].pop("path")
        preserved_samples.append(
            {
                **sample,
                "path": target,
                "source_relative_path": str(sample["path"].relative_to(dataset_root)),
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "warning": "Do not train on or replace these locked evaluation crops.",
                "samples": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return preserved_samples


def evaluate(
    model,
    loader,
    criterion,
    device,
    true_detection_threshold: float,
) -> dict[str, float | list[list[int]]]:
    model.eval()
    loss_total = 0.0
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss_total += float(criterion(logits, targets)) * targets.size(0)
            true_probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (true_probabilities >= true_detection_threshold).long()
            for target, prediction in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[target, prediction] += 1

    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    total = int(confusion.sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    beta_squared = 0.25
    f0_5 = (
        (1.0 + beta_squared)
        * precision
        * recall
        / max(beta_squared * precision + recall, 1e-12)
    )
    return {
        "loss": loss_total / max(total, 1),
        "accuracy": (true_positive + true_negative) / max(total, 1),
        "true_detection_precision": precision,
        "true_detection_recall": recall,
        "true_detection_f1": f1,
        "true_detection_f0_5": f0_5,
        "false_positive_acceptance_rate": false_positive
        / max(true_negative + false_positive, 1),
        "true_detection_rejection_rate": false_negative
        / max(false_negative + true_positive, 1),
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-class detection classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--locked-evaluation-manifest",
        type=Path,
        default=DEFAULT_LOCKED_EVALUATION,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--false-positive-weight", type=float, default=2.0)
    parser.add_argument("--true-detection-threshold", type=float, default=0.75)
    parser.add_argument(
        "--architecture",
        choices=("resnet18", "mobilenet_v3_small"),
        default="resnet18",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = args.dataset.resolve()
    output_path = args.output.resolve()
    base_model_path = args.base_model.resolve()
    locked_manifest_path = args.locked_evaluation_manifest.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(dataset_root)
    if not samples:
        raise RuntimeError(f"No labeled PNG crops were found under {dataset_root}.")
    locked_samples = load_or_create_locked_evaluation(
        samples,
        dataset_root,
        locked_manifest_path,
        args.seed,
    )
    locked_source_paths = {
        str(sample.get("source_relative_path")) for sample in locked_samples
    }
    training_samples = [
        sample
        for sample in samples
        if str(sample["path"].relative_to(dataset_root)) not in locked_source_paths
    ]
    splits = split_samples_by_crop(training_samples, args.seed)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    loaders = {
        "train": DataLoader(
            BeadCropDataset(splits["train"], train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "validation": DataLoader(
            BeadCropDataset(splits["validation"], evaluation_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            BeadCropDataset(splits["test"], evaluation_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "locked_evaluation": DataLoader(
            BeadCropDataset(locked_samples, evaluation_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    print("WARNING: Validate test metrics before using this checkpoint in production")
    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: {len(splits[split_name])} images, "
            f"crop-stratified, {class_counts(splits[split_name])}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = None
    if base_model_path.exists():
        baseline_model = make_model(pretrained=False, architecture=args.architecture)
        baseline_model.load_state_dict(
            torch.load(base_model_path, map_location="cpu", weights_only=True)
        )
        model = copy.deepcopy(baseline_model)
        print(f"warm-starting candidate from working model: {base_model_path}")
    else:
        model = make_model(pretrained=True, architecture=args.architecture)
        print("working model not found; starting candidate from ImageNet weights")
    trainable_parameters = set_trainable_parameters(model, args.architecture)
    model = model.to(device)

    train_counts = class_counts(splits["train"])
    class_weights = torch.tensor(
        [
            len(splits["train"]) / (2.0 * train_counts["false_positive"]),
            len(splits["train"]) / (2.0 * train_counts["true_detection"]),
        ],
        dtype=torch.float32,
        device=device,
    )
    class_weights[CLASS_TO_INDEX["false_positive"]] *= args.false_positive_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        keep_frozen_batch_norm_eval(model)
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * targets.size(0)
            train_total += targets.size(0)
            train_correct += int((logits.argmax(dim=1) == targets).sum())

        validation = evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            args.true_detection_threshold,
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss / max(train_total, 1):.4f} "
            f"train_acc={train_correct / max(train_total, 1):.3f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_f0.5={validation['true_detection_f0_5']:.3f} "
            f"val_fp_rate={validation['false_positive_acceptance_rate']:.3f}"
        )
        if (
            best_validation is None
            or validation["true_detection_f0_5"] > best_validation["true_detection_f0_5"]
            or (
                validation["true_detection_f0_5"] == best_validation["true_detection_f0_5"]
                and validation["false_positive_acceptance_rate"]
                < best_validation["false_positive_acceptance_rate"]
            )
            or (
                validation["true_detection_f0_5"] == best_validation["true_detection_f0_5"]
                and validation["false_positive_acceptance_rate"]
                == best_validation["false_positive_acceptance_rate"]
                and validation["loss"] < best_validation["loss"]
            )
        ):
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())

    torch.save(best_state, output_path)
    reloaded = make_model(pretrained=False, architecture=args.architecture)
    reloaded.load_state_dict(torch.load(output_path, map_location="cpu", weights_only=True))
    reloaded = reloaded.to(device)
    test_metrics = evaluate(
        reloaded,
        loaders["test"],
        criterion,
        device,
        args.true_detection_threshold,
    )
    candidate_locked_metrics = evaluate(
        reloaded,
        loaders["locked_evaluation"],
        criterion,
        device,
        args.true_detection_threshold,
    )
    baseline_locked_metrics = None
    if baseline_model is not None:
        baseline_model = baseline_model.to(device)
        baseline_locked_metrics = evaluate(
            baseline_model,
            loaders["locked_evaluation"],
            criterion,
            device,
            args.true_detection_threshold,
        )

    candidate_confusion = candidate_locked_metrics["confusion_matrix"]
    baseline_confusion = (
        baseline_locked_metrics["confusion_matrix"]
        if baseline_locked_metrics is not None
        else None
    )
    metrics_improved = bool(
        baseline_locked_metrics is not None
        and candidate_locked_metrics["true_detection_f0_5"]
        > baseline_locked_metrics["true_detection_f0_5"]
        and candidate_locked_metrics["false_positive_acceptance_rate"]
        <= baseline_locked_metrics["false_positive_acceptance_rate"]
    )
    promotion_recommended = metrics_improved

    split_manifest = {
        split_name: [
            {
                **sample_manifest_entry(sample, dataset_root),
            }
            for sample in splits[split_name]
        ]
        for split_name in SPLIT_NAMES
    }
    artifact_prefix = output_path.stem
    split_manifest_path = output_path.parent / f"{artifact_prefix}_split_manifest.json"
    training_report_path = output_path.parent / f"{artifact_prefix}_training_report.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    report = {
        "warning": "Validate test metrics before using this checkpoint in production",
        "model": args.architecture,
        "output": str(output_path),
        "base_model": str(base_model_path) if base_model_path.exists() else None,
        "class_to_index": CLASS_TO_INDEX,
        "split_mode": "stratified_random_crop",
        "true_detection_threshold": args.true_detection_threshold,
        "false_positive_weight": args.false_positive_weight,
        "input_size": [224, 224],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "epochs": args.epochs,
        "seed": args.seed,
        "split_counts": {name: class_counts(splits[name]) for name in SPLIT_NAMES},
        "validation_metrics": best_validation,
        "test_metrics": test_metrics,
        "locked_evaluation_manifest": str(locked_manifest_path),
        "locked_evaluation_counts": class_counts(locked_samples),
        "baseline_locked_metrics": baseline_locked_metrics,
        "candidate_locked_metrics": candidate_locked_metrics,
        "metrics_improved": metrics_improved,
        "promotion_recommended": promotion_recommended,
    }
    training_report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"saved raw state_dict: {output_path}")
    print(f"test metrics: {json.dumps(test_metrics, sort_keys=True)}")
    print(
        "locked evaluation: "
        f"baseline={json.dumps(baseline_locked_metrics, sort_keys=True)} "
        f"candidate={json.dumps(candidate_locked_metrics, sort_keys=True)}"
    )
    print(
        "split mode: stratified random crops; "
        f"true threshold: {args.true_detection_threshold:.2f}; "
        f"promotion recommended: {promotion_recommended}"
    )


if __name__ == "__main__":
    main()
