"""Tiny-ImageNet and CIFAR-N noisy-label benchmarks.

This script covers:
1. Tiny-ImageNet with controlled synthetic label noise for ImageNet-style
   evaluation.
2. CIFAR-10N/CIFAR-100N with real human annotation noise.

The benchmark uses CE, Focal Loss, and SWCE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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
    evaluate_single,
    get_cifar_transforms,
    get_tiny_imagenet_transforms,
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
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["tiny_imagenet", "cifar10n", "cifar100n"],
        choices=["tiny_imagenet", "cifar10n", "cifar100n"],
    )
    parser.add_argument("--methods", nargs="+", default=["ce", "focal", "swce"], choices=["ce", "focal", "swce"])
    parser.add_argument("--noise-types", nargs="+", default=["symmetric", "asymmetric"], choices=["symmetric", "asymmetric"])
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--q1", type=float, default=0.80)
    parser.add_argument("--q2", type=float, default=0.90)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--cifar10n-labels",
        nargs="+",
        default=["aggre_label", "worse_label"],
        choices=["aggre_label", "worse_label", "random_label1", "random_label2", "random_label3"],
        help="Human noisy-label variants to evaluate for CIFAR-10N.",
    )
    parser.add_argument("--data-extra-root", type=Path, default=Path("./data_extra"))
    parser.add_argument("--cifar-root", type=Path, default=Path("./data_cifar"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results_tinyimagenet_cifarn"))
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples-train", type=int, default=0)
    parser.add_argument("--max-samples-test", type=int, default=0)
    return parser.parse_args()


def make_criterion(method: str, args: argparse.Namespace):
    if method == "ce":
        import torch.nn as nn

        return nn.CrossEntropyLoss()
    if method == "focal":
        return FocalLoss(gamma=args.focal_gamma)
    if method == "swce":
        return SWCELoss(beta=args.beta, q1=args.q1, q2=args.q2)
    raise ValueError(method)


def load_tiny_imagenet(args: argparse.Namespace, noise_type: str, noise_ratio: float, seed: int):
    train_transform, test_transform = get_tiny_imagenet_transforms()
    root = args.data_extra_root / "tiny-imagenet-200"
    train_base = datasets.ImageFolder(root / "train", transform=train_transform)
    test_base = datasets.ImageFolder(root / "val_by_class", transform=test_transform)
    true_train = np.asarray(train_base.targets, dtype=np.int64)
    true_test = np.asarray(test_base.targets, dtype=np.int64)
    noisy_train = make_noisy_targets(true_train, len(train_base.classes), noise_ratio, noise_type, seed)
    noisy_test = make_noisy_targets(true_test, len(train_base.classes), noise_ratio, noise_type, seed + 10_000)
    train_data = IndexedNoisyDataset(train_base, noisy_train, true_train)
    test_data = IndexedNoisyDataset(test_base, noisy_test, true_test)
    train_data = maybe_subset(train_data, args.max_samples_train, seed)
    test_data = maybe_subset(test_data, args.max_samples_test, seed + 1)
    return "tiny_imagenet", train_data, test_data, len(train_base.classes), True


def _as_numpy_labels(values) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(values):
            return values.detach().cpu().numpy().astype(np.int64)
    except Exception:
        pass
    return np.asarray(values, dtype=np.int64)


def _map_ordered_labels_to_torchvision(labels: dict, image_order: np.ndarray) -> dict[str, np.ndarray]:
    """Map CIFAR-N ordered labels back to torchvision CIFAR train order."""
    mapped: dict[str, np.ndarray] = {}
    image_order = np.asarray(image_order, dtype=np.int64)
    for key, values in labels.items():
        arr = _as_numpy_labels(values)
        if arr.shape[0] != image_order.shape[0]:
            continue
        restored = np.empty_like(arr)
        restored[image_order] = arr
        mapped[key] = restored
    return mapped


def load_cifar_n_labels(cifar_n_root: Path, dataset_name: str) -> tuple[dict[str, np.ndarray], str]:
    """Load CIFAR-N labels in torchvision CIFAR train order.

    The ``*_human_ordered.npy`` files are ordered by ``image_order_*`` rather
    than by torchvision's native CIFAR train order. Prefer the official ``.pt``
    files because they are already aligned; keep the ordered npy files only as
    a checked fallback.
    """
    if dataset_name == "cifar10n":
        pt_path = cifar_n_root / "CIFAR-10_human.pt"
        npy_path = cifar_n_root / "CIFAR-10_human_ordered.npy"
        order_path = cifar_n_root / "image_order_c10.npy"
    elif dataset_name == "cifar100n":
        pt_path = cifar_n_root / "CIFAR-100_human.pt"
        npy_path = cifar_n_root / "CIFAR-100_human_ordered.npy"
        order_path = cifar_n_root / "image_order_c100.npy"
    else:
        raise ValueError(dataset_name)

    if pt_path.exists():
        import torch

        raw_labels = torch.load(pt_path, map_location="cpu")
        labels = {key: _as_numpy_labels(value) for key, value in raw_labels.items()}
        source = pt_path.name
    else:
        raw_labels = np.load(npy_path, allow_pickle=True).item()
        image_order = np.load(order_path)
        labels = _map_ordered_labels_to_torchvision(raw_labels, image_order)
        source = f"{npy_path.name}+{order_path.name}"

    if dataset_name == "cifar100n" and "noise_label" not in labels and "noisy_label" in labels:
        labels["noise_label"] = labels["noisy_label"]
    return labels, source


def validate_cifar_n_alignment(
    dataset_name: str,
    labels: dict[str, np.ndarray],
    train_targets: np.ndarray,
    source: str,
    noisy_key: str,
) -> float:
    true_train = labels.get("clean_label")
    noisy_train = labels.get(noisy_key)
    if true_train is None:
        raise KeyError(f"{dataset_name} clean_label not found in {source}")
    if noisy_train is None:
        raise KeyError(f"{dataset_name} noisy label key not found in {source}: {noisy_key}")
    if len(true_train) != len(train_targets) or len(noisy_train) != len(train_targets):
        raise ValueError(f"{dataset_name} CIFAR-N labels do not match train set length")
    clean_match = float(np.mean(true_train == train_targets))
    if clean_match != 1.0:
        raise ValueError(
            f"{dataset_name} clean_label is not aligned with torchvision CIFAR train order "
            f"(source={source}, match={clean_match:.6f})."
        )
    return float(np.mean(noisy_train != true_train))


def load_cifar_n(args: argparse.Namespace, dataset_name: str, seed: int, cifar10n_label: str = "aggre_label"):
    train_transform, test_transform = get_cifar_transforms("cifar10" if dataset_name == "cifar10n" else "cifar100")
    cifar_root = args.cifar_root
    cifar_n_root = args.data_extra_root / "cifar-n"
    if dataset_name == "cifar10n":
        train_base = datasets.CIFAR10(root=cifar_root, train=True, transform=train_transform, download=False)
        test_base = datasets.CIFAR10(root=cifar_root, train=False, transform=test_transform, download=False)
        labels, label_source = load_cifar_n_labels(cifar_n_root, dataset_name)
        if cifar10n_label not in labels:
            raise KeyError(f"CIFAR-10N label variant not found: {cifar10n_label}")
        noisy_train = np.asarray(labels[cifar10n_label], dtype=np.int64)
        true_train = np.asarray(labels["clean_label"], dtype=np.int64)
        num_classes = 10
        dataset_label = f"cifar10n_{cifar10n_label}"
        natural_noise_type = f"human_{cifar10n_label}"
        noisy_key = cifar10n_label
    elif dataset_name == "cifar100n":
        train_base = datasets.CIFAR100(root=cifar_root, train=True, transform=train_transform, download=False)
        test_base = datasets.CIFAR100(root=cifar_root, train=False, transform=test_transform, download=False)
        labels, label_source = load_cifar_n_labels(cifar_n_root, dataset_name)
        noisy_train = np.asarray(labels["noise_label"], dtype=np.int64)
        true_train = np.asarray(labels["clean_label"], dtype=np.int64)
        num_classes = 100
        dataset_label = "cifar100n_noise_label"
        natural_noise_type = "human_noise_label"
        noisy_key = "noise_label"
    else:
        raise ValueError(dataset_name)
    train_targets = np.asarray(train_base.targets, dtype=np.int64)
    noise_rate = validate_cifar_n_alignment(dataset_name, labels, train_targets, label_source, noisy_key)
    print(
        f"{dataset_label}: CIFAR-N labels source={label_source}; "
        f"clean_alignment=1.000000; natural_noise_rate={noise_rate:.4f}",
        flush=True,
    )
    true_test = np.asarray(test_base.targets, dtype=np.int64)
    train_data = IndexedNoisyDataset(train_base, noisy_train, true_train)
    test_data = IndexedNoisyDataset(test_base, true_test, true_test)
    train_data = maybe_subset(train_data, args.max_samples_train, seed)
    test_data = maybe_subset(test_data, args.max_samples_test, seed + 1)
    return dataset_label, train_data, test_data, num_classes, True, natural_noise_type


def run_one(
    args: argparse.Namespace,
    experiment: str,
    method: str,
    noise_type: str,
    noise_ratio: float | str,
    seed: int,
    cifar10n_label: str = "aggre_label",
):
    set_seed(seed)
    device = choose_device(args.device)
    if experiment == "tiny_imagenet":
        dataset_label, train_data, test_data, num_classes, small_input = load_tiny_imagenet(
            args, str(noise_type), float(noise_ratio), seed
        )
    elif experiment in {"cifar10n", "cifar100n"}:
        dataset_label, train_data, test_data, num_classes, small_input, natural_noise_type = load_cifar_n(
            args, experiment, seed, cifar10n_label=cifar10n_label
        )
        noise_type = natural_noise_type
        noise_ratio = "human"
    else:
        raise ValueError(experiment)

    train_loader = make_loader(train_data, args.batch_size, True, args.num_workers, device)
    test_loader = make_loader(test_data, args.batch_size, False, args.num_workers, device)
    model = build_resnet18(num_classes, small_input=small_input).to(device)
    criterion = make_criterion(method, args)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay, args.optimizer)
    scheduler = None
    if args.optimizer == "sgd":
        scheduler = __import__("torch").optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[max(1, int(args.epochs * 0.5)), max(1, int(args.epochs * 0.75))],
            gamma=0.1,
        )

    started = time.monotonic()
    for _ in range(args.epochs):
        train_single_epoch(model, train_loader, criterion, optimizer, device)
        if scheduler is not None:
            scheduler.step()
    train_seconds = time.monotonic() - started
    acc_true, acc_noisy = evaluate_single(model, test_loader, device)
    return ResultRow(
        dataset=dataset_label,
        method=method,
        noise_type=str(noise_type),
        noise_ratio=noise_ratio,
        seed=seed,
        acc_true=acc_true,
        acc_noisy=acc_noisy if experiment == "tiny_imagenet" else "",
        train_seconds=train_seconds,
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
    for experiment in args.experiments:
        for method in args.methods:
            if experiment == "tiny_imagenet":
                for noise_type in args.noise_types:
                    for noise_ratio in args.noise_ratios:
                        for seed in args.seeds:
                            row = run_one(args, experiment, method, noise_type, noise_ratio, seed)
                            print(row, flush=True)
                            rows.append(row)
            else:
                for seed in args.seeds:
                    if experiment == "cifar10n":
                        for label_name in args.cifar10n_labels:
                            row = run_one(args, experiment, method, "human", "real", seed, cifar10n_label=label_name)
                            print(row, flush=True)
                            rows.append(row)
                    else:
                        row = run_one(args, experiment, method, "human", "real", seed)
                        print(row, flush=True)
                        rows.append(row)

    raw_path = args.output_dir / "tinyimagenet_cifarn_raw.csv"
    summary_path = args.output_dir / "tinyimagenet_cifarn_summary.csv"
    write_rows(raw_path, rows)
    summarize_rows(raw_path, summary_path)
    print(f"Wrote {raw_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
