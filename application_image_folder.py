"""ImageFolder application-dataset experiment with synthetic label noise."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models


class SWCELoss(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(outputs, labels)
        sorted_loss, _ = torch.sort(ce_loss, descending=True)
        loss_80 = sorted_loss[int(0.2 * ce_loss.size(0))]
        loss_90 = sorted_loss[int(0.1 * ce_loss.size(0))]

        low_mask = ce_loss <= loss_80
        mid_mask = (ce_loss > loss_80) & (ce_loss <= loss_90)
        high_mask = ce_loss > loss_90
        max_loss = torch.maximum(torch.max(ce_loss), ce_loss.new_tensor(1.0))

        weighted = ce_loss * low_mask.float()
        weighted += ce_loss * self.beta * mid_mask.float()
        weighted += ce_loss * self.beta / max_loss * high_mask.float()
        return weighted.mean()


class SmallConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-subdir", default="")
    parser.add_argument("--test-subdir", default="")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--dataset-name", default="application")
    parser.add_argument("--output-dir", type=Path, default=Path("./results_application"))
    parser.add_argument("--model", choices=["convnet", "resnet34"], default="resnet34")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--noise-type", choices=["symmetric", "asymmetric"], default="symmetric")
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def load_datasets(args: argparse.Namespace) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    transform = image_transform(args.image_size)
    train_root = args.data_root / args.train_subdir if args.train_subdir else None
    test_root = args.data_root / args.test_subdir if args.test_subdir else None

    if train_root and test_root and train_root.exists() and test_root.exists():
        train_set = datasets.ImageFolder(train_root, transform)
        test_set = datasets.ImageFolder(test_root, transform)
        return train_set, test_set, len(train_set.classes)

    full_set = datasets.ImageFolder(args.data_root, transform)
    train_size = int(args.train_fraction * len(full_set))
    test_size = len(full_set) - train_size
    train_set, test_set = random_split(full_set, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    return train_set, test_set, len(full_set.classes)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "convnet":
        return SmallConvNet(num_classes)
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(name: str, model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def corrupt_labels(labels: torch.Tensor, noise_ratio: float, noise_type: str, num_classes: int) -> torch.Tensor:
    labels_changed = labels.clone()
    num_to_change = int(noise_ratio * len(labels_changed))
    if num_to_change == 0:
        return labels_changed
    indices = np.random.choice(len(labels_changed), num_to_change, replace=False)
    for index in indices:
        label = int(labels_changed[index].item())
        if noise_type == "asymmetric":
            labels_changed[index] = (label + 1) % num_classes
        else:
            choices = [candidate for candidate in range(num_classes) if candidate != label]
            labels_changed[index] = random.choice(choices)
    return labels_changed


def evaluate(model: nn.Module, loader: DataLoader, noise_ratio: float, noise_type: str, num_classes: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct_true = 0
    correct_noisy = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            noisy_labels = corrupt_labels(labels, noise_ratio, noise_type, num_classes)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_true += (predicted == labels).sum().item()
            correct_noisy += (predicted == noisy_labels).sum().item()
    return correct_true / total, correct_noisy / total


def run_setting(args: argparse.Namespace, train_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset, num_classes: int, noise_ratio: float, beta: float) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args.model, num_classes).to(device)
    criterion = SWCELoss(beta)
    optimizer = build_optimizer(args.optimizer, model, args.learning_rate)
    losses: list[float] = []
    acc_true: list[float] = []
    acc_noisy: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        last_loss = None
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels_changed = corrupt_labels(labels, noise_ratio, args.noise_type, num_classes)
            outputs = model(images)
            loss = criterion(outputs, labels_changed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss

        if (epoch + 1) % 2 == 0 and last_loss is not None:
            true_acc, noisy_acc = evaluate(model, test_loader, noise_ratio, args.noise_type, num_classes, device)
            losses.append(float(last_loss.detach().cpu()))
            acc_true.append(true_acc)
            acc_noisy.append(noisy_acc)
            print(f"{args.dataset_name} noise={noise_ratio} beta={beta} epoch={epoch + 1} true_acc={true_acc:.4f} noisy_acc={noisy_acc:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.dataset_name}_{args.noise_type}_{noise_ratio}_{beta}"
    pd.DataFrame(losses).to_csv(args.output_dir / f"loss_{suffix}.csv", index=False)
    pd.DataFrame(acc_noisy).to_csv(args.output_dir / f"acc_noisy_{suffix}.csv", index=False)
    pd.DataFrame(acc_true).to_csv(args.output_dir / f"acc_true_{suffix}.csv", index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_set, test_set, num_classes = load_datasets(args)
    for noise_ratio in args.noise_ratios:
        for beta in args.betas:
            run_setting(args, train_set, test_set, num_classes, noise_ratio, beta)


if __name__ == "__main__":
    main()
