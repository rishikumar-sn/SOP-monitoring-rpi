from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_TO_INDEX = {"false_positive": 0, "tassel": 1}
MIN_SAMPLES_PER_CLASS = 8


class TasselDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, target


def collect_samples(dataset: Path) -> dict[str, list[Path]]:
    return {
        class_name: sorted((dataset / class_name).glob("*.png"))
        for class_name in CLASS_TO_INDEX
    }


def split_samples(
    samples_by_class: dict[str, list[Path]],
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    training: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    for class_name, class_index in CLASS_TO_INDEX.items():
        paths = list(samples_by_class[class_name])
        if len(paths) < MIN_SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"At least {MIN_SAMPLES_PER_CLASS} {class_name} samples are required; "
                f"found {len(paths)}."
            )
        rng.shuffle(paths)
        validation_count = max(2, int(round(len(paths) * 0.20)))
        validation.extend((path, class_index) for path in paths[:validation_count])
        training.extend((path, class_index) for path in paths[validation_count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def make_model(checkpoint: dict) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[0].in_features
    model.classifier = nn.Linear(input_features, 1)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def set_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.features[-3:], model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def keep_frozen_batch_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


def evaluate(model, loader, criterion, threshold: float) -> dict:
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    loss_total = 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            targets = targets.to(dtype=torch.float32)
            logits = model(inputs).reshape(-1)
            loss_total += float(criterion(logits, targets)) * targets.numel()
            predictions = (torch.sigmoid(logits) >= threshold).to(dtype=torch.int64)
            for target, prediction in zip(
                targets.to(dtype=torch.int64).tolist(),
                predictions.tolist(),
            ):
                confusion[target, prediction] += 1
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    negative_recall = true_negative / max(true_negative + false_positive, 1)
    positive_recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "loss": loss_total / max(int(confusion.sum()), 1),
        "accuracy": float(np.trace(confusion) / max(int(confusion.sum()), 1)),
        "balanced_accuracy": float((negative_recall + positive_recall) / 2.0),
        "false_positive_acceptance_rate": float(
            false_positive / max(true_negative + false_positive, 1)
        ),
        "tassel_recall": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "confusion_matrix": confusion.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune tassel MobileNetV3 from an existing checkpoint."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(
            f"Base checkpoint does not exist: {args.base_checkpoint}"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

    samples_by_class = collect_samples(args.dataset)
    training_samples, validation_samples = split_samples(
        samples_by_class,
        args.seed,
    )
    print(
        f"Dataset ready: {len(samples_by_class['tassel'])} tassel, "
        f"{len(samples_by_class['false_positive'])} false positive; "
        f"train={len(training_samples)} validation={len(validation_samples)}",
        flush=True,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12, fill=255),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    validation_transform = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
    train_loader = DataLoader(
        TasselDataset(training_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TasselDataset(validation_samples, validation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model = make_model(checkpoint)
    trainable = set_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    positive_count = len(samples_by_class["tassel"])
    negative_count = len(samples_by_class["false_positive"])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / max(positive_count, 1)])
    )
    threshold = float(checkpoint.get("threshold", 0.5))
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate(model, validation_loader, criterion, threshold)
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        keep_frozen_batch_norm_eval(model)
        loss_total = 0.0
        sample_count = 0
        for inputs, targets in train_loader:
            targets = targets.to(dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).reshape(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * targets.numel()
            sample_count += targets.numel()
        metrics = evaluate(model, validation_loader, criterion, threshold)
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={loss_total / max(sample_count, 1):.4f} "
            f"val_balanced={metrics['balanced_accuracy']:.3f} "
            f"false_accept={metrics['false_positive_acceptance_rate']:.3f} "
            f"tassel_recall={metrics['tassel_recall']:.3f}",
            flush=True,
        )
        candidate_key = (
            metrics["balanced_accuracy"],
            -metrics["false_positive_acceptance_rate"],
            metrics["tassel_recall"],
        )
        best_key = (
            best_metrics["balanced_accuracy"],
            -best_metrics["false_positive_acceptance_rate"],
            best_metrics["tassel_recall"],
        )
        if candidate_key > best_key:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = args.checkpoint_dir / f"tassel_mobilenet_v3_{timestamp}.pt"
    latest_path = args.checkpoint_dir / "latest.pt"
    payload = {
        "state_dict": best_state,
        "architecture": "mobilenet_v3_small",
        "input_size": 224,
        "threshold": threshold,
        "parent_checkpoint": str(args.base_checkpoint.resolve()),
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "class_counts": {
            "tassel": positive_count,
            "false_positive": negative_count,
        },
        "validation_metrics": best_metrics,
    }
    temporary = latest_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(latest_path)
    torch.save(payload, archive_path)
    report = {
        key: value for key, value in payload.items() if key != "state_dict"
    }
    report["checkpoint"] = archive_path.name
    report["training_samples"] = [str(path) for path, _ in training_samples]
    report["validation_samples"] = [str(path) for path, _ in validation_samples]
    (args.checkpoint_dir / f"training_report_{timestamp}.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {archive_path.name}; latest.pt updated from checkpoint parent "
        f"{args.base_checkpoint.name}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
