"""Summarize result tables used in the paper."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


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

DIAGNOSTIC_FIELDS = [
    "dataset",
    "noise_type",
    "noise_ratio",
    "seed",
    "beta",
    "q1",
    "q2",
    "rule",
    "threshold",
    "flagged",
    "true_noisy",
    "true_clean",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "clean_retention_rate",
]


def try_import_scipy():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return 0.0, 0.0
    return float(array.mean()), float(array.std(ddof=1) if len(array) > 1 else 0.0)


def exact_sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def beta_value(row: dict[str, str]) -> str:
    value = row.get("beta", "")
    if value == "":
        return ""
    return f"{float(value):.2f}"


def write_dicts(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_result_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def read_result_files(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "method" not in row and "loss" in row:
                    row["method"] = row["loss"]
                for field in RESULT_FIELDS:
                    row.setdefault(field, "")
                rows.append(row)
    return rows


def read_diagnostic_files(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for field in DIAGNOSTIC_FIELDS:
                    row.setdefault(field, "")
                rows.append(row)
    return rows


def benchmark_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["dataset"], row["noise_type"], row["noise_ratio"], row["seed"])


def benchmark_paired_comparisons(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scipy_stats = try_import_scipy()
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[benchmark_key(row)][row["method"]] = row

    methods = sorted({row["method"] for row in rows if row["method"] != "swce"})
    output: list[dict[str, str]] = []
    for method in methods:
        diffs = []
        for method_rows in grouped.values():
            if "swce" in method_rows and method in method_rows:
                diffs.append(float(method_rows["swce"]["acc_true"]) - float(method_rows[method]["acc_true"]))
        if not diffs:
            continue
        diff = np.asarray(diffs, dtype=np.float64)
        wins = int(np.sum(diff > 0))
        ties = int(np.sum(diff == 0))
        losses = int(np.sum(diff < 0))
        t_p = ""
        wilcoxon_p = ""
        if scipy_stats is not None and len(diff) >= 2:
            try:
                t_p = f"{scipy_stats.ttest_1samp(diff, 0.0).pvalue:.6g}"
            except Exception:
                t_p = ""
            try:
                wilcoxon_p = f"{scipy_stats.wilcoxon(diff, alternative='two-sided', zero_method='wilcox').pvalue:.6g}"
            except Exception:
                wilcoxon_p = ""
        output.append(
            {
                "comparison": f"SWCE vs {method}",
                "n": str(len(diff)),
                "wins": str(wins),
                "ties": str(ties),
                "losses": str(losses),
                "mean_difference": f"{diff.mean():.6f}",
                "median_difference": f"{np.median(diff):.6f}",
                "sign_test_p_value": f"{exact_sign_test_p_value(wins, losses):.6g}",
                "paired_t_p_value": t_p,
                "wilcoxon_p_value": wilcoxon_p,
            }
        )
    return output


def benchmark_overhead_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("train_seconds", "") == "":
            continue
        groups[benchmark_key(row)][row["method"]] = float(row["train_seconds"])

    methods = sorted({row["method"] for row in rows if row["method"] != "ce"})
    output: list[dict[str, str]] = []
    for method in methods:
        ratios = []
        deltas = []
        for method_times in groups.values():
            if "ce" in method_times and method in method_times and method_times["ce"] > 0:
                deltas.append(method_times[method] - method_times["ce"])
                ratios.append(method_times[method] / method_times["ce"])
        if not ratios:
            continue
        ratios_np = np.asarray(ratios, dtype=np.float64)
        deltas_np = np.asarray(deltas, dtype=np.float64)
        output.append(
            {
                "method": method,
                "n": str(len(ratios)),
                "mean_time_ratio_vs_ce": f"{ratios_np.mean():.4f}",
                "median_time_ratio_vs_ce": f"{np.median(ratios_np):.4f}",
                "mean_extra_seconds_vs_ce": f"{deltas_np.mean():.2f}",
            }
        )
    return output


def result_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("method", ""),
        row.get("noise_type", ""),
        f"{float(row.get('noise_ratio') or 0.0):.2f}",
        row.get("seed", ""),
        beta_value(row),
    )


def dedupe_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = result_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def summarize_accuracy(groups: dict[tuple[str, ...], list[dict[str, str]]], key_fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, group in sorted(groups.items()):
        acc_true_mean, acc_true_std = mean_std([float(row["acc_true"]) for row in group])
        acc_noisy_mean, acc_noisy_std = mean_std([float(row["acc_noisy"]) for row in group if row.get("acc_noisy", "") != ""])
        seconds_mean, seconds_std = mean_std([float(row["train_seconds"]) for row in group if row.get("train_seconds", "") != ""])
        item = {field: value for field, value in zip(key_fields, key)}
        item.update(
            {
                "acc_true_mean": f"{acc_true_mean:.4f}",
                "acc_true_std": f"{acc_true_std:.4f}",
                "acc_noisy_mean": f"{acc_noisy_mean:.4f}",
                "acc_noisy_std": f"{acc_noisy_std:.4f}",
                "train_seconds_mean": f"{seconds_mean:.2f}",
                "train_seconds_std": f"{seconds_std:.2f}",
                "num_seeds": str(len(group)),
            }
        )
        rows.append(item)
    return rows


def beta_grid_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("method") != "swce":
            continue
        if row.get("noise_type") != "symmetric" or abs(float(row.get("noise_ratio") or 0.0) - 0.4) > 1e-9:
            continue
        if row.get("beta", "") == "":
            continue
        key = (row["dataset"], row["noise_type"], f"{float(row['noise_ratio']):.2f}", "swce", beta_value(row))
        groups[key].append(row)
    return summarize_accuracy(groups, ["dataset", "noise_type", "noise_ratio", "method", "beta"])


def ablation_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("noise_type") != "symmetric" or abs(float(row.get("noise_ratio") or 0.0) - 0.4) > 1e-9:
            continue
        method = row.get("method", "")
        beta = beta_value(row)
        include = False
        if method in {"ce", "hard_filter"}:
            include = True
            beta = ""
        elif method in {"two_segment", "swce"} and beta == "0.10":
            include = True
        if not include:
            continue
        key = (row["dataset"], row["noise_type"], f"{float(row['noise_ratio']):.2f}", method, beta)
        groups[key].append(row)

    order = {"ce": 0, "hard_filter": 1, "two_segment": 2, "swce": 3}
    output = summarize_accuracy(groups, ["dataset", "noise_type", "noise_ratio", "method", "beta"])
    output.sort(key=lambda row: (row["dataset"], order.get(row["method"], 99), row["beta"]))
    return output


def noise_identification_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics = ["precision", "recall", "f1", "false_positive_rate", "clean_retention_rate"]
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("noise_type") != "symmetric" or abs(float(row.get("noise_ratio") or 0.0) - 0.4) > 1e-9:
            continue
        groups[(row["dataset"], row["noise_type"], f"{float(row['noise_ratio']):.2f}", beta_value(row), row["rule"])].append(row)

    output: list[dict[str, str]] = []
    for (dataset, noise_type, noise_ratio, beta, rule), group in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_ratio": noise_ratio,
            "method": "swce",
            "beta": beta,
            "rule": rule,
            "num_seeds": str(len(group)),
        }
        for metric in metrics:
            mean, std = mean_std([float(row[metric]) for row in group if row.get(metric, "") != ""])
            item[f"{metric}_mean"] = f"{mean:.4f}"
            item[f"{metric}_std"] = f"{std:.4f}"
        output.append(item)
    return output


def high_noise_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["noise_type"], f"{float(row['noise_ratio']):.2f}", beta_value(row))].append(row)
    return summarize_accuracy(groups, ["dataset", "method", "noise_type", "noise_ratio", "beta"])


def high_noise_paired_comparisons(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["seed"])
        method = row["method"]
        if method == "swce":
            method = "swce_beta_0.10"
        grouped[key][method] = row

    output: list[dict[str, str]] = []
    for baseline in ["ce", "focal"]:
        diffs: list[float] = []
        for method_rows in grouped.values():
            if "swce_beta_0.10" in method_rows and baseline in method_rows:
                diffs.append(float(method_rows["swce_beta_0.10"]["acc_true"]) - float(method_rows[baseline]["acc_true"]))
        if not diffs:
            continue
        diff = np.asarray(diffs, dtype=np.float64)
        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))
        output.append(
            {
                "comparison": f"SWCE beta=0.10 vs {baseline}",
                "n_pairs": str(len(diff)),
                "wins": str(wins),
                "ties": str(ties),
                "losses": str(losses),
                "mean_acc_true_diff": f"{diff.mean():.6f}",
                "median_acc_true_diff": f"{np.median(diff):.6f}",
                "min_acc_true_diff": f"{diff.min():.6f}",
                "max_acc_true_diff": f"{diff.max():.6f}",
                "sign_test_p_two_sided": f"{exact_sign_test_p_value(wins, losses):.6g}",
            }
        )
    return output


def run_benchmark(args: argparse.Namespace) -> None:
    rows = read_result_files(args.raw_files)
    if not rows:
        raise SystemExit("No result rows found.")
    write_dicts(args.output_dir / "paired_swce_comparisons.csv", benchmark_paired_comparisons(rows))
    write_dicts(args.output_dir / "training_time_overhead.csv", benchmark_overhead_summary(rows))
    print(f"Wrote benchmark summaries to {args.output_dir}", flush=True)


def run_diagnostics(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or (args.result_dir / "analysis")
    result_files = [*args.extra_raw_files, args.result_dir / "diagnostic_raw.csv"]
    diagnostic_files = [*args.extra_diagnostic_files, args.result_dir / "noise_identification_raw.csv"]
    rows = dedupe_result_rows(read_result_files(result_files))
    diagnostic_rows = read_diagnostic_files(diagnostic_files)

    write_result_rows(output_dir / "combined_diagnostic_raw.csv", rows)
    write_dicts(output_dir / "beta_grid_summary.csv", beta_grid_summary(rows))
    write_dicts(output_dir / "ablation_summary.csv", ablation_summary(rows))
    write_dicts(output_dir / "noise_identification_summary.csv", noise_identification_summary(diagnostic_rows))
    print(f"Wrote diagnostic summaries to {output_dir}", flush=True)


def run_high_noise(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or (args.result_dir / "analysis")
    rows = read_result_files([args.result_dir / "high_noise60_raw.csv"])
    write_result_rows(output_dir / "high_noise60_seed_level_rows.csv", rows)
    write_dicts(output_dir / "high_noise60_summary.csv", high_noise_summary(rows))
    write_dicts(output_dir / "high_noise60_paired_comparisons.csv", high_noise_paired_comparisons(rows))
    print(f"Wrote high-noise summaries to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="paired benchmark and runtime summaries")
    benchmark.add_argument("--raw-files", nargs="+", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, default=Path("./results_benchmark_statistics"))
    benchmark.set_defaults(func=run_benchmark)

    diagnostics = subparsers.add_parser("diagnostics", help="beta, ablation, and noise-identification summaries")
    diagnostics.add_argument("--result-dir", type=Path, default=Path("./results_diagnostics"))
    diagnostics.add_argument("--extra-raw-files", nargs="*", type=Path, default=[])
    diagnostics.add_argument("--extra-diagnostic-files", nargs="*", type=Path, default=[])
    diagnostics.add_argument("--output-dir", type=Path, default=None)
    diagnostics.set_defaults(func=run_diagnostics)

    high_noise = subparsers.add_parser("high-noise60", help="CIFAR symmetric 60 percent high-noise summaries")
    high_noise.add_argument("--result-dir", type=Path, default=Path("./results_high_noise60"))
    high_noise.add_argument("--output-dir", type=Path, default=None)
    high_noise.set_defaults(func=run_high_noise)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
