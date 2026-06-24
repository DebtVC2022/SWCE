"""CIFAR diagnostics, component ablations, and beta-stability checks.

This runner covers the targeted supplemental experiments used in the paper:

- beta stability around the default beta=0.30;
- CIFAR synthetic-noise diagnostics for SWCE loss segments;
- CIFAR ablation variants for segmentation, soft weighting, and full SWCE.

The implementation uses ResNet-18, CIFAR train/test transforms, SGD,
beta=0.30, q1=0.80, q2=0.90, synthetic symmetric noise, and three random seeds
by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swce_common import (
    IndexedNoisyDataset,
    ResultRow,
    SWCELoss,
    build_resnet18,
    choose_device,
    evaluate_single,
    get_cifar_transforms,
    make_loader,
    make_noisy_targets,
    make_optimizer,
    maybe_subset,
    set_seed,
    write_rows,
)


@dataclass
class DiagnosticRow:
    dataset: str
    noise_type: str
    noise_ratio: float
    seed: int
    beta: float
    q1: float
    q2: float
    rule: str
    threshold: float
    flagged: int
    true_noisy: int
    true_clean: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    clean_retention_rate: float


class HardHighLossFilterLoss(nn.Module):
    """Keep samples with CE loss <= q2 batch quantile and drop the rest."""

    def __init__(self, q2: float = 0.90) -> None:
        super().__init__()
        self.q2 = q2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits, targets, reduction="none")
        threshold = torch.quantile(losses.detach(), self.q2)
        keep = losses <= threshold
        if int(keep.sum().item()) == 0:
            return losses.mean()
        return losses[keep].mean()


class TwoSegmentSoftWeightLoss(nn.Module):
    """Low-loss samples get weight 1; all samples above q1 get beta."""

    def __init__(self, beta: float = 0.30, q1: float = 0.80) -> None:
        super().__init__()
        self.beta = beta
        self.q1 = q1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits, targets, reduction="none")
        threshold = torch.quantile(losses.detach(), self.q1)
        weights = torch.ones_like(losses)
        weights = torch.where(losses > threshold, self.beta * weights, weights)
        return (weights * losses).mean()


class TransformTargetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform=None) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.targets = list(getattr(base_dataset, "targets", []))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--variant", required=True, choices=["ce", "hard_filter", "two_segment", "swce"])
    parser.add_argument("--noise-type", default="symmetric", choices=["symmetric", "asymmetric"])
    parser.add_argument("--noise-ratio", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--q1", type=float, default=0.80)
    parser.add_argument("--q2", type=float, default=0.90)
    parser.add_argument("--data-root", type=Path, default=Path("./data_cifar"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results_diagnostics"))
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples-train", type=int, default=0)
    parser.add_argument("--max-samples-test", type=int, default=0)
    parser.add_argument("--write-diagnostic", action="store_true")
    parser.add_argument("--save-per-sample", action="store_true")
    return parser.parse_args()


def make_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(epochs * 0.5)), max(1, int(epochs * 0.75))],
        gamma=0.1,
    )


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.variant == "ce":
        return nn.CrossEntropyLoss()
    if args.variant == "hard_filter":
        return HardHighLossFilterLoss(q2=args.q2)
    if args.variant == "two_segment":
        return TwoSegmentSoftWeightLoss(beta=args.beta, q1=args.q1)
    if args.variant == "swce":
        return SWCELoss(beta=args.beta, q1=args.q1, q2=args.q2)
    raise ValueError(args.variant)


def load_cifar(args: argparse.Namespace):
    train_transform, test_transform = get_cifar_transforms(args.dataset)
    if args.dataset == "cifar10":
        cls = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == "cifar100":
        cls = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(args.dataset)

    train_base = cls(root=args.data_root, train=True, transform=train_transform, download=False)
    train_eval_base = cls(root=args.data_root, train=True, transform=test_transform, download=False)
    test_base = cls(root=args.data_root, train=False, transform=test_transform, download=False)

    true_train = np.asarray(train_base.targets, dtype=np.int64)
    true_test = np.asarray(test_base.targets, dtype=np.int64)
    noisy_train = make_noisy_targets(true_train, num_classes, args.noise_ratio, args.noise_type, args.seed)
    noisy_test = make_noisy_targets(true_test, num_classes, args.noise_ratio, args.noise_type, args.seed + 10_000)

    train_data = IndexedNoisyDataset(train_base, noisy_train, true_train)
    train_eval_data = IndexedNoisyDataset(train_eval_base, noisy_train, true_train)
    test_data = IndexedNoisyDataset(test_base, noisy_test, true_test)

    train_data = maybe_subset(train_data, args.max_samples_train, args.seed)
    train_eval_data = maybe_subset(train_eval_data, args.max_samples_train, args.seed)
    test_data = maybe_subset(test_data, args.max_samples_test, args.seed + 1)
    return train_data, train_eval_data, test_data, num_classes


def train_model(args: argparse.Namespace, train_data, test_data, num_classes: int, device: torch.device):
    model = build_resnet18(num_classes, small_input=True).to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay, args.optimizer)
    scheduler = make_scheduler(optimizer, args.epochs) if args.optimizer == "sgd" else None
    criterion = build_criterion(args)
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, device)
    test_loader = make_loader(test_data, args.batch_size, False, args.num_workers, device)

    started = time.monotonic()
    model.train()
    for _ in range(args.epochs):
        for images, noisy_labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            noisy_labels = noisy_labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, noisy_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    train_seconds = time.monotonic() - started
    acc_true, acc_noisy = evaluate_single(model, test_loader, device)
    return model, acc_true, acc_noisy, train_seconds


@torch.no_grad()
def collect_train_losses(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, float | int]]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    for images, noisy_labels, true_labels, indices in loader:
        images = images.to(device, non_blocking=True)
        noisy_labels_device = noisy_labels.to(device, non_blocking=True)
        losses = F.cross_entropy(model(images), noisy_labels_device, reduction="none").detach().cpu().numpy()
        for index, noisy_label, true_label, loss in zip(
            indices.numpy().tolist(),
            noisy_labels.numpy().tolist(),
            true_labels.numpy().tolist(),
            losses.tolist(),
        ):
            rows.append(
                {
                    "index": int(index),
                    "noisy_label": int(noisy_label),
                    "true_label": int(true_label),
                    "is_noisy": int(noisy_label != true_label),
                    "ce_loss": float(loss),
                }
            )
    rows.sort(key=lambda row: int(row["index"]))
    return rows


def diagnostic_for_rule(rows: list[dict[str, float | int]], rule: str, threshold: float, args: argparse.Namespace) -> DiagnosticRow:
    losses = np.asarray([float(row["ce_loss"]) for row in rows], dtype=np.float64)
    noisy = np.asarray([int(row["is_noisy"]) for row in rows], dtype=bool)
    flagged = losses > threshold
    tp = int(np.sum(flagged & noisy))
    fp = int(np.sum(flagged & ~noisy))
    fn = int(np.sum(~flagged & noisy))
    tn = int(np.sum(~flagged & ~noisy))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    clean_retention = tn / (fp + tn) if (fp + tn) else 0.0
    return DiagnosticRow(
        dataset=args.dataset,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        beta=args.beta,
        q1=args.q1,
        q2=args.q2,
        rule=rule,
        threshold=threshold,
        flagged=int(np.sum(flagged)),
        true_noisy=int(np.sum(noisy)),
        true_clean=int(np.sum(~noisy)),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        clean_retention_rate=clean_retention,
    )


def write_dict_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics(args: argparse.Namespace, model: nn.Module, train_eval_data, device: torch.device) -> None:
    loader = make_loader(train_eval_data, args.batch_size, False, args.num_workers, device)
    rows = collect_train_losses(model, loader, device)
    losses = np.asarray([float(row["ce_loss"]) for row in rows], dtype=np.float64)
    lambda1 = float(np.quantile(losses, args.q1))
    lambda2 = float(np.quantile(losses, args.q2))
    diag_rows = [
        asdict(diagnostic_for_rule(rows, "loss_gt_lambda1", lambda1, args)),
        asdict(diagnostic_for_rule(rows, "loss_gt_lambda2", lambda2, args)),
    ]
    write_dict_rows(args.output_dir / "noise_identification_raw.csv", diag_rows)
    if args.save_per_sample:
        per_sample = []
        for row in rows:
            loss = float(row["ce_loss"])
            segment = "low"
            if loss > lambda2:
                segment = "high"
            elif loss > lambda1:
                segment = "middle"
            item = dict(row)
            item.update({"lambda1": lambda1, "lambda2": lambda2, "segment": segment})
            per_sample.append(item)
        write_dict_rows(args.output_dir / "per_sample_losses.csv", per_sample)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_data, train_eval_data, test_data, num_classes = load_cifar(args)
    model, acc_true, acc_noisy, train_seconds = train_model(args, train_data, test_data, num_classes, device)

    row = ResultRow(
        dataset=args.dataset,
        method=args.variant,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        acc_true=acc_true,
        acc_noisy=acc_noisy,
        train_seconds=train_seconds,
        epochs=args.epochs,
        beta=args.beta if args.variant in {"swce", "two_segment"} else "",
        q1=args.q1 if args.variant in {"swce", "two_segment"} else "",
        q2=args.q2 if args.variant in {"swce", "hard_filter"} else "",
        backbone="resnet18",
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        num_classes=num_classes,
        train_samples=len(train_data),
        test_samples=len(test_data),
        device=str(device),
        extra=f"lr={args.learning_rate}; weight_decay={args.weight_decay}",
    )
    write_rows(args.output_dir / "diagnostic_raw.csv", [row])
    if args.write_diagnostic and args.variant == "swce":
        write_diagnostics(args, model, train_eval_data, device)
    print(
        f"{args.dataset} {args.variant} {args.noise_type} eta={args.noise_ratio} "
        f"seed={args.seed} beta={args.beta:.2f} ACC_true={acc_true:.4f} "
        f"ACC_noisy={acc_noisy:.4f} seconds={train_seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
