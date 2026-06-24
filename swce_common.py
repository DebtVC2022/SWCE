"""Shared utilities for SWCE experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG")


@dataclass
class ResultRow:
    dataset: str
    method: str
    noise_type: str
    noise_ratio: float | str
    seed: int
    acc_true: float
    acc_noisy: float | str
    train_seconds: float
    epochs: int | str = ""
    beta: float | str = ""
    q1: float | str = ""
    q2: float | str = ""
    backbone: str = "resnet18"
    optimizer: str = ""
    batch_size: int | str = ""
    num_classes: int | str = ""
    train_samples: int | str = ""
    test_samples: int | str = ""
    device: str = ""
    extra: str = ""


class IndexedNoisyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, noisy_targets: Sequence[int], true_targets: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.noisy_targets = np.asarray(noisy_targets, dtype=np.int64)
        self.true_targets = np.asarray(true_targets, dtype=np.int64)
        if len(self.noisy_targets) != len(self.base_dataset):
            raise ValueError("noisy_targets length does not match dataset length")
        if len(self.true_targets) != len(self.base_dataset):
            raise ValueError("true_targets length does not match dataset length")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, _ = self.base_dataset[index]
        return image, int(self.noisy_targets[index]), int(self.true_targets[index]), index


class TwoViewDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, noisy_label, true_label, original_index = self.base_dataset[index]
        if isinstance(image, torch.Tensor):
            view1 = image
            view2 = image.clone()
        else:
            view1 = self.transform(image)
            view2 = self.transform(image)
        return view1, view2, noisy_label, true_label, original_index


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


class SWCELoss(nn.Module):
    def __init__(self, beta: float = 0.30, q1: float = 0.80, q2: float = 0.90) -> None:
        super().__init__()
        self.beta = beta
        self.q1 = q1
        self.q2 = q2

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits, targets, reduction="none")
        lambda1 = torch.quantile(losses.detach(), self.q1)
        lambda2 = torch.quantile(losses.detach(), self.q2)
        max_loss = torch.maximum(losses.detach().max(), losses.new_tensor(1.0))
        weights = torch.ones_like(losses)
        weights = torch.where((losses > lambda1) & (losses <= lambda2), self.beta * weights, weights)
        weights = torch.where(losses > lambda2, (self.beta / max_loss) * weights, weights)
        return (weights * losses).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_cifar_transforms(dataset_name: str):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_transform, test_transform


def get_tiny_imagenet_transforms():
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_transform, test_transform


def make_noisy_targets(targets: Sequence[int], num_classes: int, noise_ratio: float, noise_type: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = np.asarray(targets, dtype=np.int64)
    noisy = clean.copy()
    count = int(noise_ratio * clean.size)
    indices = rng.choice(clean.size, size=count, replace=False)
    if noise_type == "symmetric":
        for index in indices:
            choices = np.delete(np.arange(num_classes), clean[index])
            noisy[index] = rng.choice(choices)
    elif noise_type == "asymmetric":
        for index in indices:
            noisy[index] = (clean[index] + 1) % num_classes
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noisy


def build_resnet18(num_classes: int, small_input: bool = True) -> nn.Module:
    model = models.resnet18(weights=None)
    if small_input:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def maybe_subset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train()
    for images, noisy_labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy_labels = noisy_labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, noisy_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate_single(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct_true = 0
    correct_noisy = 0
    total = 0
    for images, noisy_labels, true_labels, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy_labels = noisy_labels.to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct_true += (pred == true_labels).sum().item()
        correct_noisy += (pred == noisy_labels).sum().item()
        total += true_labels.numel()
    return correct_true / total, correct_noisy / total


@torch.no_grad()
def evaluate_ensemble(models_: Sequence[nn.Module], loader: DataLoader, device: torch.device) -> tuple[float, float]:
    for model in models_:
        model.eval()
    correct_true = 0
    correct_noisy = 0
    total = 0
    for images, noisy_labels, true_labels, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy_labels = noisy_labels.to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        logits = torch.stack([model(images) for model in models_]).mean(dim=0)
        pred = logits.argmax(dim=1)
        correct_true += (pred == true_labels).sum().item()
        correct_noisy += (pred == noisy_labels).sum().item()
        total += true_labels.numel()
    return correct_true / total, correct_noisy / total


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, optimizer_name: str = "sgd"):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def write_rows(path: Path, rows: Iterable[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ResultRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize_rows(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["noise_type"], row["noise_ratio"])
        groups.setdefault(key, []).append(row)
    fields = [
        "dataset",
        "method",
        "noise_type",
        "noise_ratio",
        "acc_true_mean",
        "acc_true_std",
        "acc_noisy_mean",
        "acc_noisy_std",
        "train_seconds_mean",
        "num_seeds",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (dataset, method, noise_type, noise_ratio), group in sorted(groups.items()):
            acc_true = np.array([float(row["acc_true"]) for row in group])
            noisy_values = [row["acc_noisy"] for row in group if row["acc_noisy"] != ""]
            acc_noisy = np.array([float(value) for value in noisy_values]) if noisy_values else np.array([])
            seconds = np.array([float(row["train_seconds"]) for row in group])
            writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "noise_type": noise_type,
                    "noise_ratio": noise_ratio,
                    "acc_true_mean": f"{acc_true.mean():.4f}",
                    "acc_true_std": f"{acc_true.std(ddof=1) if len(acc_true) > 1 else 0.0:.4f}",
                    "acc_noisy_mean": f"{acc_noisy.mean():.4f}" if len(acc_noisy) else "",
                    "acc_noisy_std": f"{acc_noisy.std(ddof=1) if len(acc_noisy) > 1 else 0.0:.4f}" if len(acc_noisy) else "",
                    "train_seconds_mean": f"{seconds.mean():.2f}",
                    "num_seeds": len(group),
                }
            )
