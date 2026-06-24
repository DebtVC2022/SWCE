"""Standard CIFAR noisy-label benchmark for SWCE.

This script supports CIFAR-10 and CIFAR-100 with synthetic symmetric or
asymmetric label noise and compares ordinary cross-entropy with SWCE.

Example:
    python benchmark_cifar_synthetic_noise.py \
        --datasets cifar10 --noise-types symmetric \
        --noise-ratios 0.2 0.4 --losses ce swce --epochs 100 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import random
import ssl
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


CIFAR_ARCHIVES = {
    "cifar10": {
        "folder": "cifar-10-batches-py",
        "filename": "cifar-10-python.tar.gz",
        "urls": [
            "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        ],
    },
    "cifar100": {
        "folder": "cifar-100-python",
        "filename": "cifar-100-python.tar.gz",
        "urls": [
            "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        ],
    },
}


@dataclass
class Metrics:
    dataset: str
    loss: str
    noise_type: str
    noise_ratio: float
    seed: int
    q1: float
    q2: float
    beta: float
    acc_true: float
    acc_noisy: float


class NoisyLabelDataset(Dataset):
    def __init__(self, base_dataset: Dataset, noisy_targets: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.noisy_targets = noisy_targets.astype(np.int64)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, true_label = self.base_dataset[index]
        return image, int(self.noisy_targets[index]), int(true_label)


class SWCELoss(nn.Module):
    def __init__(self, beta: float = 0.30, q1: float = 0.80, q2: float = 0.90) -> None:
        super().__init__()
        self.beta = beta
        self.q1 = q1
        self.q2 = q2
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, targets)
        lambda1 = torch.quantile(ce_loss.detach(), self.q1)
        lambda2 = torch.quantile(ce_loss.detach(), self.q2)
        max_loss = torch.maximum(ce_loss.detach().max(), ce_loss.new_tensor(1.0))

        weights = torch.ones_like(ce_loss)
        weights = torch.where((ce_loss > lambda1) & (ce_loss <= lambda2), self.beta * weights, weights)
        weights = torch.where(ce_loss > lambda2, (self.beta / max_loss) * weights, weights)
        return (weights * ce_loss).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_transforms(dataset_name: str) -> tuple[transforms.Compose, transforms.Compose]:
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


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if target_dir not in member_path.parents and member_path != target_dir:
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        tar.extractall(target_dir)


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if url.startswith("https://"):
        response_cm = urllib.request.urlopen(request, timeout=60, context=ssl._create_unverified_context())
    else:
        response_cm = urllib.request.urlopen(request, timeout=60)
    with response_cm as response:
        with destination.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def ensure_cifar_available(
    dataset_name: str,
    data_dir: Path,
    download: bool,
    archive_path: Path | None,
) -> None:
    info = CIFAR_ARCHIVES[dataset_name]
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / info["folder"]
    if extracted_dir.exists():
        return

    if archive_path:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        safe_extract_tar(archive_path, data_dir)
        return

    if not download:
        return

    destination = data_dir / info["filename"]
    errors: list[str] = []
    for url in info["urls"]:
        try:
            print(f"Downloading {dataset_name} from {url}")
            download_file(url, destination)
            safe_extract_tar(destination, data_dir)
            return
        except (urllib.error.URLError, TimeoutError, OSError, tarfile.TarError) as exc:
            errors.append(f"{url}: {exc}")
            if destination.exists():
                destination.unlink()

    raise RuntimeError(
        "Could not download the CIFAR archive. Download it manually and rerun with "
        f"--{dataset_name}-archive /path/to/{info['filename']}.\n"
        + "\n".join(errors)
    )


def load_cifar(
    dataset_name: str,
    data_dir: Path,
    download: bool,
    cifar10_archive: Path | None,
    cifar100_archive: Path | None,
):
    train_transform, test_transform = get_transforms(dataset_name)
    if dataset_name == "cifar10":
        cls = datasets.CIFAR10
        num_classes = 10
        archive_path = cifar10_archive
    elif dataset_name == "cifar100":
        cls = datasets.CIFAR100
        num_classes = 100
        archive_path = cifar100_archive
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ensure_cifar_available(dataset_name, data_dir, download, archive_path)
    train_set = cls(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = cls(root=data_dir, train=False, transform=test_transform, download=False)
    return train_set, test_set, num_classes


def make_noisy_targets(
    targets: list[int] | np.ndarray,
    num_classes: int,
    noise_ratio: float,
    noise_type: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    targets_np = np.asarray(targets, dtype=np.int64)
    noisy_targets = targets_np.copy()
    noisy_count = int(noise_ratio * targets_np.size)
    noisy_indices = rng.choice(targets_np.size, size=noisy_count, replace=False)

    if noise_type == "symmetric":
        for idx in noisy_indices:
            choices = np.delete(np.arange(num_classes), targets_np[idx])
            noisy_targets[idx] = rng.choice(choices)
    elif noise_type == "asymmetric":
        # Class-dependent cyclic flips. This is a reproducible asymmetric-noise
        # protocol for both CIFAR-10 and CIFAR-100.
        for idx in noisy_indices:
            noisy_targets[idx] = (targets_np[idx] + 1) % num_classes
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy_targets


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train()
    for images, noisy_labels, _ in loader:
        images = images.to(device, non_blocking=True)
        noisy_labels = noisy_labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, noisy_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct_true = 0
    correct_noisy = 0
    total = 0
    for images, noisy_labels, true_labels in loader:
        images = images.to(device, non_blocking=True)
        noisy_labels = noisy_labels.to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct_true += (pred == true_labels).sum().item()
        correct_noisy += (pred == noisy_labels).sum().item()
        total += true_labels.numel()
    return correct_true / total, correct_noisy / total


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    loader_args = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_args["persistent_workers"] = True
    return DataLoader(dataset, **loader_args)


def run_experiment(args: argparse.Namespace) -> list[Metrics]:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    results: list[Metrics] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        train_set, test_set, num_classes = load_cifar(
            dataset_name,
            args.data_dir,
            args.download,
            args.cifar10_archive,
            args.cifar100_archive,
        )
        train_targets = np.asarray(train_set.targets)
        test_targets = np.asarray(test_set.targets)

        for noise_type in args.noise_types:
            for noise_ratio in args.noise_ratios:
                for seed in args.seeds:
                    noisy_train = make_noisy_targets(train_targets, num_classes, noise_ratio, noise_type, seed)
                    noisy_test = make_noisy_targets(test_targets, num_classes, noise_ratio, noise_type, seed + 10_000)
                    train_data = NoisyLabelDataset(train_set, noisy_train)
                    test_data = NoisyLabelDataset(test_set, noisy_test)
                    train_loader = build_loader(
                        train_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                    )
                    test_loader = build_loader(
                        test_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                    )

                    for loss_name in args.losses:
                        set_seed(seed)
                        model = build_model(num_classes).to(device)
                        if loss_name == "ce":
                            criterion: nn.Module = nn.CrossEntropyLoss()
                        elif loss_name == "swce":
                            criterion = SWCELoss(beta=args.beta, q1=args.q1, q2=args.q2)
                        else:
                            raise ValueError(f"Unknown loss: {loss_name}")

                        optimizer = torch.optim.SGD(
                            model.parameters(),
                            lr=args.learning_rate,
                            momentum=0.9,
                            weight_decay=args.weight_decay,
                        )
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=args.lr_milestones,
                            gamma=0.1,
                        )

                        for epoch in range(args.epochs):
                            train_one_epoch(model, train_loader, criterion, optimizer, device)
                            scheduler.step()
                            if args.log_interval and (epoch + 1) % args.log_interval == 0:
                                acc_true, acc_noisy = evaluate(model, test_loader, device)
                                print(
                                    f"{dataset_name} {loss_name} {noise_type} eta={noise_ratio} "
                                    f"seed={seed} epoch={epoch + 1}: "
                                    f"ACC_true={acc_true:.4f}, ACC_noisy={acc_noisy:.4f}"
                                )

                        acc_true, acc_noisy = evaluate(model, test_loader, device)
                        result = Metrics(
                            dataset=dataset_name,
                            loss=loss_name,
                            noise_type=noise_type,
                            noise_ratio=noise_ratio,
                            seed=seed,
                            q1=args.q1,
                            q2=args.q2,
                            beta=args.beta,
                            acc_true=acc_true,
                            acc_noisy=acc_noisy,
                        )
                        results.append(result)
                        append_result(args.output_dir / "cifar_noisy_benchmark_raw.csv", result)
                        print(result)

    write_summary(args.output_dir / "cifar_noisy_benchmark_summary.csv", results)
    return results


def append_result(path: Path, result: Metrics) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.__dict__.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(result.__dict__)


def write_summary(path: Path, results: list[Metrics]) -> None:
    groups: dict[tuple[str, str, str, float], list[Metrics]] = {}
    for row in results:
        groups.setdefault((row.dataset, row.loss, row.noise_type, row.noise_ratio), []).append(row)

    with path.open("w", newline="") as f:
        fields = [
            "dataset",
            "loss",
            "noise_type",
            "noise_ratio",
            "acc_true_mean",
            "acc_true_std",
            "acc_noisy_mean",
            "acc_noisy_std",
            "num_seeds",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (dataset, loss, noise_type, noise_ratio), rows in sorted(groups.items()):
            acc_true = np.array([r.acc_true for r in rows])
            acc_noisy = np.array([r.acc_noisy for r in rows])
            writer.writerow(
                {
                    "dataset": dataset,
                    "loss": loss,
                    "noise_type": noise_type,
                    "noise_ratio": noise_ratio,
                    "acc_true_mean": f"{acc_true.mean():.4f}",
                    "acc_true_std": f"{acc_true.std(ddof=1) if len(rows) > 1 else 0.0:.4f}",
                    "acc_noisy_mean": f"{acc_noisy.mean():.4f}",
                    "acc_noisy_std": f"{acc_noisy.std(ddof=1) if len(rows) > 1 else 0.0:.4f}",
                    "num_seeds": len(rows),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cifar10"], choices=["cifar10", "cifar100"])
    parser.add_argument("--losses", nargs="+", default=["ce", "swce"], choices=["ce", "swce"])
    parser.add_argument("--noise-types", nargs="+", default=["symmetric"], choices=["symmetric", "asymmetric"])
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-milestones", nargs="+", type=int, default=[50, 75])
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--q1", type=float, default=0.80)
    parser.add_argument("--q2", type=float, default=0.90)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results_cifar"))
    parser.add_argument("--cifar10-archive", type=Path, default=None)
    parser.add_argument("--cifar100-archive", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 avoids file-descriptor exhaustion on macOS/Python 3.14.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
