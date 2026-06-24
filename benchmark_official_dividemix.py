"""Run one official DivideMix CIFAR job and export one normalized CSV row.

Download the official DivideMix implementation from LiJunnan1992/DivideMix and
pass the CIFAR code directory with ``--official-root``. For each job, this
wrapper copies the official files into an isolated work directory, applies small
runtime compatibility updates to that copy, writes a seed-specific noisy-label
JSON file that matches the paper's CIFAR noise protocol, launches the official
trainer, and parses the final test accuracy. The downloaded official source
directory is not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swce_common import make_noisy_targets


RESULT_FIELDS = [
    "dataset",
    "method",
    "noise_type",
    "noise_ratio",
    "seed",
    "acc_true",
    "acc_noisy",
    "train_seconds",
    "epochs",
    "beta",
    "q1",
    "q2",
    "backbone",
    "optimizer",
    "batch_size",
    "num_classes",
    "train_samples",
    "test_samples",
    "device",
    "extra",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("./data_cifar"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--noise-type", choices=["symmetric", "asymmetric"], required=True)
    parser.add_argument("--noise-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--lambda-u", type=float, default=25.0)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def unpickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def dataset_path(data_root: Path, dataset: str) -> Path:
    if dataset == "cifar10":
        return data_root / "cifar-10-batches-py"
    return data_root / "cifar-100-python"


def load_train_targets(data_root: Path, dataset: str) -> np.ndarray:
    root = dataset_path(data_root, dataset)
    if dataset == "cifar10":
        labels: list[int] = []
        for index in range(1, 6):
            labels.extend(unpickle(root / f"data_batch_{index}")["labels"])
        return np.asarray(labels, dtype=np.int64)
    return np.asarray(unpickle(root / "train")["fine_labels"], dtype=np.int64)


def write_noise_file(path: Path, targets: np.ndarray, num_classes: int, noise_ratio: float, noise_type: str, seed: int) -> None:
    noisy = make_noisy_targets(targets, num_classes, noise_ratio, noise_type, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([int(value) for value in noisy.tolist()], f)


def copy_official_code(official_root: Path, code_dir: Path) -> None:
    if code_dir.exists():
        shutil.rmtree(code_dir)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
    shutil.copytree(official_root, code_dir, ignore=ignore)


def update_train_cifar(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "parser.add_argument('--noise_file'" not in text:
        text = text.replace(
            "parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')\n",
            "parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')\n"
            "parser.add_argument('--noise_file', default='', type=str, help='seed-specific noisy label JSON')\n"
            "parser.add_argument('--num_workers', default=5, type=int, help='dataloader workers')\n",
        )
    text = text.replace(
        "parser.add_argument('--seed', default=123)",
        "parser.add_argument('--seed', default=123, type=int)",
    )
    text = text.replace("unlabeled_train_iter.next()", "next(unlabeled_train_iter)")
    text = text.replace(
        "num_workers=5,\\\n    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))",
        "num_workers=args.num_workers,\\\n"
        "    root_dir=args.data_path,log=stats_log,noise_file=args.noise_file if args.noise_file else '%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))",
    )
    required_snippets = [
        "parser.add_argument('--noise_file'",
        "parser.add_argument('--num_workers'",
        "parser.add_argument('--seed', default=123, type=int)",
        "next(unlabeled_train_iter)",
        "noise_file=args.noise_file if args.noise_file else",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        raise RuntimeError(f"Failed to update official Train_cifar.py; missing snippets: {missing}")
    path.write_text(text, encoding="utf-8")


def update_dataloader_cifar(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from torchnet.meter import AUCMeter\n",
        "try:\n"
        "    from torchnet.meter import AUCMeter\n"
        "except Exception:\n"
        "    class AUCMeter:\n"
        "        def reset(self):\n"
        "            self._probability = None\n"
        "            self._target = None\n"
        "        def add(self, probability, target):\n"
        "            self._probability = probability\n"
        "            self._target = target\n"
        "        def value(self):\n"
        "            return 0.0, None, None\n",
    )
    if "class AUCMeter" not in text:
        raise RuntimeError("Failed to update official dataloader_cifar.py AUCMeter fallback.")
    path.write_text(text, encoding="utf-8")


def prepare_code(args: argparse.Namespace) -> Path:
    if not (args.official_root / "Train_cifar.py").exists():
        raise FileNotFoundError(f"Official DivideMix Train_cifar.py not found in {args.official_root}")
    code_dir = args.work_dir / "code"
    copy_official_code(args.official_root, code_dir)
    update_train_cifar(code_dir / "Train_cifar.py")
    update_dataloader_cifar(code_dir / "dataloader_cifar.py")
    (code_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    return code_dir


def parse_final_accuracy(checkpoint_dir: Path, dataset: str, noise_ratio: float, noise_mode: str) -> float:
    expected = checkpoint_dir / f"{dataset}_{noise_ratio:.1f}_{noise_mode}_acc.txt"
    candidates = [expected] if expected.exists() else sorted(checkpoint_dir.glob("*_acc.txt"))
    if not candidates:
        raise FileNotFoundError(f"No DivideMix accuracy log found in {checkpoint_dir}")
    text = candidates[-1].read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        raise ValueError(f"No accuracy value found in {candidates[-1]}")
    return float(matches[-1]) / 100.0


def write_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    num_classes = 10 if args.dataset == "cifar10" else 100
    train_targets = load_train_targets(args.data_root, args.dataset)
    if len(train_targets) != 50000:
        raise ValueError(f"Unexpected {args.dataset} train target count: {len(train_targets)}")

    ratio = f"{args.noise_ratio:.2f}".replace(".", "p")
    noise_json = args.work_dir / "noise_labels" / f"{args.dataset}_{args.noise_type}_eta{ratio}_seed{args.seed}.json"
    write_noise_file(noise_json, train_targets, num_classes, args.noise_ratio, args.noise_type, args.seed)
    code_dir = prepare_code(args)
    noise_mode = "sym" if args.noise_type == "symmetric" else "asym"
    data_path = dataset_path(args.data_root, args.dataset).resolve()
    command = [
        args.python_bin,
        "Train_cifar.py",
        "--dataset",
        args.dataset,
        "--num_class",
        str(num_classes),
        "--data_path",
        str(data_path),
        "--noise_file",
        str(noise_json.resolve()),
        "--noise_mode",
        noise_mode,
        "--r",
        str(args.noise_ratio),
        "--seed",
        str(args.seed),
        "--gpuid",
        str(args.gpuid),
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--lr",
        str(args.learning_rate),
        "--lambda_u",
        str(args.lambda_u),
    ]
    preview = args.work_dir / "command.txt"
    preview.write_text(" ".join(command) + "\n", encoding="utf-8")

    started = time.monotonic()
    log_path = args.work_dir / "run.log"
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command:\n")
        log.write(" ".join(command) + "\n\n")
        log.write("Environment overrides:\n")
        log.write("MKL_THREADING_LAYER=GNU\n")
        log.write(f"OMP_NUM_THREADS={env.get('OMP_NUM_THREADS')}\n")
        log.write(f"MKL_NUM_THREADS={env.get('MKL_NUM_THREADS')}\n\n")
        completed = subprocess.run(
            command,
            cwd=code_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Official DivideMix failed with return code {completed.returncode}. See {log_path}")
    acc_true = parse_final_accuracy(code_dir / "checkpoint", args.dataset, args.noise_ratio, noise_mode)
    row = {
        "dataset": args.dataset,
        "method": "dividemix_official",
        "noise_type": args.noise_type,
        "noise_ratio": args.noise_ratio,
        "seed": args.seed,
        "acc_true": acc_true,
        "acc_noisy": "",
        "train_seconds": elapsed,
        "epochs": args.num_epochs,
        "backbone": "PreResNet18",
        "optimizer": "sgd",
        "batch_size": args.batch_size,
        "num_classes": num_classes,
        "train_samples": 50000,
        "test_samples": 10000,
        "device": f"cuda:{args.gpuid}",
        "extra": (
            f"official=LiJunnan1992/DivideMix; noise_file={noise_json.name}; "
            f"num_epochs_arg={args.num_epochs}; actual_epoch_loops={args.num_epochs + 1}; "
            f"lambda_u={args.lambda_u}; lr={args.learning_rate}; num_workers={args.num_workers}; "
            "acc_true=official clean test accuracy; acc_noisy=not reported by official runner"
        ),
    }
    write_result(args.output_csv, row)
    print(row, flush=True)
    print(f"Wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
