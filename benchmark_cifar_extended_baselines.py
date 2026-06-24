"""CIFAR baseline comparisons for the benchmark section.

Implemented methods:
- ce: ordinary cross-entropy.
- focal: Focal Loss with gamma=2 by default.
- swce: proposed SWCE loss.
- coteaching: classic two-network small-loss exchange training.
- mentornet_pd: MentorNet predefined-curriculum/self-paced weighting variant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swce_common import (
    FocalLoss,
    IndexedNoisyDataset,
    ResultRow,
    SWCELoss,
    build_resnet18,
    choose_device,
    evaluate_ensemble,
    evaluate_single,
    get_cifar_transforms,
    make_loader,
    make_noisy_targets,
    make_optimizer,
    maybe_subset,
    set_seed,
    summarize_rows,
    train_single_epoch,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"], choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ce", "focal", "swce", "coteaching", "mentornet_pd"],
        choices=["ce", "focal", "swce", "coteaching", "mentornet_pd"],
    )
    parser.add_argument("--noise-types", nargs="+", default=["symmetric", "asymmetric"], choices=["symmetric", "asymmetric"])
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--q1", type=float, default=0.80)
    parser.add_argument("--q2", type=float, default=0.90)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--data-root", type=Path, default=Path("./data_cifar"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results_cifar_baselines"))
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples-train", type=int, default=0)
    parser.add_argument("--max-samples-test", type=int, default=0)
    return parser.parse_args()


def load_cifar(args: argparse.Namespace, dataset_name: str, noise_type: str, noise_ratio: float, seed: int):
    train_transform, test_transform = get_cifar_transforms(dataset_name)
    if dataset_name == "cifar10":
        train_base = datasets.CIFAR10(root=args.data_root, train=True, transform=train_transform, download=False)
        test_base = datasets.CIFAR10(root=args.data_root, train=False, transform=test_transform, download=False)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_base = datasets.CIFAR100(root=args.data_root, train=True, transform=train_transform, download=False)
        test_base = datasets.CIFAR100(root=args.data_root, train=False, transform=test_transform, download=False)
        num_classes = 100
    else:
        raise ValueError(dataset_name)
    true_train = np.asarray(train_base.targets, dtype=np.int64)
    true_test = np.asarray(test_base.targets, dtype=np.int64)
    noisy_train = make_noisy_targets(true_train, num_classes, noise_ratio, noise_type, seed)
    noisy_test = make_noisy_targets(true_test, num_classes, noise_ratio, noise_type, seed + 10_000)
    train_data = IndexedNoisyDataset(train_base, noisy_train, true_train)
    test_data = IndexedNoisyDataset(test_base, noisy_test, true_test)
    train_data = maybe_subset(train_data, args.max_samples_train, seed)
    test_data = maybe_subset(test_data, args.max_samples_test, seed + 1)
    return train_data, test_data, num_classes


def make_single_criterion(method: str, args: argparse.Namespace):
    if method == "ce":
        return torch.nn.CrossEntropyLoss()
    if method == "focal":
        return FocalLoss(gamma=args.focal_gamma)
    if method == "swce":
        return SWCELoss(beta=args.beta, q1=args.q1, q2=args.q2)
    raise ValueError(method)


def make_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(epochs * 0.5)), max(1, int(epochs * 0.75))],
        gamma=0.1,
    )


def run_single(args: argparse.Namespace, method: str, train_data, test_data, num_classes: int, device: torch.device):
    model = build_resnet18(num_classes, small_input=True).to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay, args.optimizer)
    scheduler = make_scheduler(optimizer, args.epochs) if args.optimizer == "sgd" else None
    criterion = make_single_criterion(method, args)
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, device)
    test_loader = make_loader(test_data, args.batch_size, False, args.num_workers, device)

    started = time.monotonic()
    for _ in range(args.epochs):
        train_single_epoch(model, train_loader, criterion, optimizer, device)
        if scheduler is not None:
            scheduler.step()
    train_seconds = time.monotonic() - started
    acc_true, acc_noisy = evaluate_single(model, test_loader, device)
    return acc_true, acc_noisy, train_seconds


def remember_rate(epoch: int, epochs: int, noise_ratio: float) -> float:
    gradual = max(1, min(10, epochs // 2))
    forget = min(noise_ratio * (epoch + 1) / gradual, noise_ratio)
    return max(0.0, 1.0 - forget)


def train_coteaching_epoch(model1, model2, loader, opt1, opt2, device, rate: float) -> None:
    model1.train()
    model2.train()
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits1 = model1(images)
        logits2 = model2(images)
        loss1 = F.cross_entropy(logits1, labels, reduction="none")
        loss2 = F.cross_entropy(logits2, labels, reduction="none")
        keep = max(1, int(rate * labels.numel()))
        ind1 = torch.argsort(loss1.detach())[:keep]
        ind2 = torch.argsort(loss2.detach())[:keep]
        update_loss1 = F.cross_entropy(logits1[ind2], labels[ind2])
        update_loss2 = F.cross_entropy(logits2[ind1], labels[ind1])
        opt1.zero_grad(set_to_none=True)
        update_loss1.backward()
        opt1.step()
        opt2.zero_grad(set_to_none=True)
        update_loss2.backward()
        opt2.step()


def run_coteaching(args, train_data, test_data, num_classes: int, noise_ratio: float, device: torch.device):
    model1 = build_resnet18(num_classes, small_input=True).to(device)
    model2 = build_resnet18(num_classes, small_input=True).to(device)
    opt1 = make_optimizer(model1, args.learning_rate, args.weight_decay, args.optimizer)
    opt2 = make_optimizer(model2, args.learning_rate, args.weight_decay, args.optimizer)
    sch1 = make_scheduler(opt1, args.epochs) if args.optimizer == "sgd" else None
    sch2 = make_scheduler(opt2, args.epochs) if args.optimizer == "sgd" else None
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, device)
    test_loader = make_loader(test_data, args.batch_size, False, args.num_workers, device)
    started = time.monotonic()
    for epoch in range(args.epochs):
        train_coteaching_epoch(model1, model2, train_loader, opt1, opt2, device, remember_rate(epoch, args.epochs, noise_ratio))
        if sch1 is not None:
            sch1.step()
            sch2.step()
    train_seconds = time.monotonic() - started
    acc_true, acc_noisy = evaluate_ensemble([model1, model2], test_loader, device)
    return acc_true, acc_noisy, train_seconds


def train_mentornet_pd_epoch(model, loader, optimizer, device, rate: float) -> None:
    model.train()
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        losses = F.cross_entropy(logits, labels, reduction="none")
        keep = max(1, int(rate * labels.numel()))
        threshold = torch.topk(losses.detach(), keep, largest=False).values.max()
        weights = (losses.detach() <= threshold).float()
        loss = (weights * losses).sum() / weights.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def run_mentornet_pd(args, train_data, test_data, num_classes: int, noise_ratio: float, device: torch.device):
    model = build_resnet18(num_classes, small_input=True).to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay, args.optimizer)
    scheduler = make_scheduler(optimizer, args.epochs) if args.optimizer == "sgd" else None
    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, device)
    test_loader = make_loader(test_data, args.batch_size, False, args.num_workers, device)
    started = time.monotonic()
    for epoch in range(args.epochs):
        train_mentornet_pd_epoch(model, train_loader, optimizer, device, remember_rate(epoch, args.epochs, noise_ratio))
        if scheduler is not None:
            scheduler.step()
    train_seconds = time.monotonic() - started
    acc_true, acc_noisy = evaluate_single(model, test_loader, device)
    return acc_true, acc_noisy, train_seconds


def run_one(args, dataset_name: str, method: str, noise_type: str, noise_ratio: float, seed: int) -> ResultRow:
    set_seed(seed)
    device = choose_device(args.device)
    train_data, test_data, num_classes = load_cifar(args, dataset_name, noise_type, noise_ratio, seed)
    if method in {"ce", "focal", "swce"}:
        acc_true, acc_noisy, seconds = run_single(args, method, train_data, test_data, num_classes, device)
    elif method == "coteaching":
        acc_true, acc_noisy, seconds = run_coteaching(args, train_data, test_data, num_classes, noise_ratio, device)
    elif method == "mentornet_pd":
        acc_true, acc_noisy, seconds = run_mentornet_pd(args, train_data, test_data, num_classes, noise_ratio, device)
    else:
        raise ValueError(method)
    return ResultRow(
        dataset=dataset_name,
        method=method,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        seed=seed,
        acc_true=acc_true,
        acc_noisy=acc_noisy,
        train_seconds=seconds,
        epochs=args.epochs,
        beta=args.beta if method == "swce" else "",
        q1=args.q1 if method == "swce" else "",
        q2=args.q2 if method == "swce" else "",
        backbone="resnet18",
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        num_classes=num_classes,
        train_samples=len(train_data),
        test_samples=len(test_data),
        device=str(device),
        extra=(
            f"epochs={args.epochs}; optimizer={args.optimizer}; lr={args.learning_rate}; "
            f"weight_decay={args.weight_decay}; focal_gamma={args.focal_gamma if method == 'focal' else ''}"
        ),
    )


def main() -> None:
    args = parse_args()
    rows: list[ResultRow] = []
    for dataset_name in args.datasets:
        for method in args.methods:
            for noise_type in args.noise_types:
                for noise_ratio in args.noise_ratios:
                    for seed in args.seeds:
                        row = run_one(args, dataset_name, method, noise_type, noise_ratio, seed)
                        print(row, flush=True)
                        rows.append(row)
    raw_path = args.output_dir / "cifar_baselines_raw.csv"
    summary_path = args.output_dir / "cifar_baselines_summary.csv"
    write_rows(raw_path, rows)
    summarize_rows(raw_path, summary_path)
    print(f"Wrote {raw_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
