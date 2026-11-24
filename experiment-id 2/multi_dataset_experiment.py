"""Milestone 2B: Multi-dataset Experiment 2 CLI runner and summary reporter.

Evaluates Direct CNN (baseline) alongside EMA, Momentum-only, Kalman-only,
and Hybrid Kalman-Momentum Hopfield memory across multiple image datasets
(Fashion-MNIST, KMNIST, SVHN, CIFAR-10) while keeping Hopfield retrieval
untouched at the pre-last layer.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_2 = Path(__file__).resolve().parent
for path in (str(EXPERIMENT_2), str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from datasets import (
    DEFAULT_MULTI_DATASETS,
    DatasetSpec,
    get_dataset_spec,
    load_dataset_tensors,
)
from engine import evaluate_constraints, train_and_evaluate_seed

from benchmarks.mnist_baseline import (
    configure_threads,
    environment_info,
    resolve_device,
)
from hopfieldpc import METHODS


def build_dataset_config(
    spec: DatasetSpec,
    *,
    quick: bool = False,
    methods: Sequence[str] = METHODS,
    epochs_override: int | None = None,
    rgb_channels: tuple[int, int, int] = (32, 64, 128),
) -> dict[str, Any]:
    """Build validated Experiment 2 training and evaluation config for a dataset."""
    epochs = epochs_override if epochs_override is not None else (3 if quick else spec.recommended_epochs)
    if epochs < 2:
        raise ValueError(f"epochs must be >= 2 so warmup precedes active memory, got {epochs}.")
    warmup = 1 if quick or spec.in_channels == 1 else min(2, epochs - 1)

    corruptions = (
        {
            "gaussian": [0.0, 0.5, 1.0],
            "salt_pepper": [0.0, 0.2],
            "occlusion": [spec.occlusion_sizes[0], spec.occlusion_sizes[1]],
        }
        if quick
        else {
            "gaussian": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "salt_pepper": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
            "occlusion": list(spec.occlusion_sizes),
        }
    )

    return {
        "methods": list(methods),
        "embed_dim": 64,
        "rgb_channels": list(rgb_channels),
        "epochs": epochs,
        "warmup_epochs": warmup,
        "batch_size": 128,
        "eval_batch_size": 256,
        "learning_rate": spec.recommended_lr,
        "weight_decay": spec.recommended_weight_decay,
        "augment": True,
        "val_size": 500 if quick else spec.val_size,
        "train_subset": 2000 if quick else None,
        "test_subset": 1000 if quick else None,
        "prototypes_per_class": 8,
        "hopfield_beta": 5.0,
        "hopfield_num_updates": 1,
        "merge_distance_threshold": 0.2,
        "normalization_epsilon": 1e-8,
        "min_accuracy": 0.80,
        "max_train_val_gap": 0.07,
        "update": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 1e-8,
            "process_noise": 0.01,
            "initial_uncertainty": 1.0,
        },
        "corruptions": corruptions,
    }


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def aggregate_dataset_summary(
    spec: DatasetSpec,
    seed_runs: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute cross-seed means, SDs, robustness AUCs, and constraint verdicts."""
    if not seed_runs:
        raise ValueError("seed_runs must not be empty.")

    method_keys = ["direct", *cfg["methods"]]
    summary: dict[str, Any] = {}
    for key in method_keys:
        trains = [float(r["methods"][key]["train_accuracy"]) for r in seed_runs]
        vals = [float(r["methods"][key]["validation_accuracy"]) for r in seed_runs]
        tests = [float(r["methods"][key]["test_accuracy"]) for r in seed_runs]
        gaps = [abs(t - v) for t, v in zip(trains, vals)]

        train_mean, train_sd = _mean_sd(trains)
        val_mean, val_sd = _mean_sd(vals)
        test_mean, test_sd = _mean_sd(tests)
        gap_mean, gap_sd = _mean_sd(gaps)

        gate = evaluate_constraints(
            train_mean,
            val_mean,
            test_mean,
            min_accuracy=cfg["min_accuracy"],
            max_gap=cfg["max_train_val_gap"],
        )
        all_seeds_passed = all(
            bool(r["methods"][key]["constraints"]["all_passed"]) for r in seed_runs
        )

        entry: dict[str, Any] = {
            "train_mean": train_mean,
            "train_sd": train_sd,
            "val_mean": val_mean,
            "val_sd": val_sd,
            "test_mean": test_mean,
            "test_sd": test_sd,
            "train_val_gap_mean": gap_mean,
            "train_val_gap_sd": gap_sd,
            "constraints": {**gate, "all_seeds_passed": all_seeds_passed},
            "robust_auc_mean": {},
        }
        for corr_kind in cfg["corruptions"]:
            aucs = [float(r["methods"][key]["robust_auc"][corr_kind]) for r in seed_runs]
            entry["robust_auc_mean"][corr_kind] = _mean_sd(aucs)[0]

        if key != "direct":
            entry["active_memories_mean"] = _mean_sd(
                [float(r["methods"][key]["active_memories"]) for r in seed_runs]
            )[0]
            entry["test_entropy_mean"] = _mean_sd(
                [float(r["methods"][key]["test_entropy_mean"]) for r in seed_runs]
            )[0]
            entry["test_top1_mass_mean"] = _mean_sd(
                [float(r["methods"][key]["test_top1_mass_mean"]) for r in seed_runs]
            )[0]
            entry["helped_mean"] = _mean_sd(
                [float(r["methods"][key]["test_paired"]["helped"]) for r in seed_runs]
            )[0]
            entry["harmed_mean"] = _mean_sd(
                [float(r["methods"][key]["test_paired"]["harmed"]) for r in seed_runs]
            )[0]
        summary[key] = entry
    return summary


def _method_label(key: str) -> str:
    if key == "direct":
        return "Direct CNN (Baseline)"
    return key.replace("_", " ").title()


def write_multi_dataset_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a comprehensive Markdown summary with Train/Val/Test and constraint gates."""
    lines = [
        "# Experiment 2: Multi-Dataset Prototype Learning Summary",
        "",
        (
            f"Seeds: {', '.join(str(s) for s in payload['seeds'])} | "
            f"Device: `{payload['environment']['device']}` | "
            "Constraint Gate: `Train >= 80%`, `Test >= 80%`, `|Train - Val| <= 7%`"
        ),
        "",
        "## 1. Accuracy & Generalization Gate Across Datasets",
        "",
        "| Dataset | Method | Train | Validation | Test | |Train - Val| | Gate (>=80% & <=7%) |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]

    for ds_block in payload["datasets"].values():
        display = ds_block["display_name"]
        summary = ds_block["summary"]
        for key, stats in summary.items():
            gate = stats["constraints"]
            status = "PASS" if gate["all_passed"] else "FAIL"
            lines.append(
                f"| {display} | {_method_label(key)} | "
                f"{stats['train_mean'] * 100:.2f}% | "
                f"{stats['val_mean'] * 100:.2f}% | "
                f"{stats['test_mean'] * 100:.2f}% | "
                f"{stats['train_val_gap_mean'] * 100:.2f} pp | "
                f"**{status}** |"
            )

    lines.extend([
        "",
        "## 2. Corruption Robustness AUC & Hopfield Memory Diagnostics",
        "",
        "| Dataset | Method | Gaussian AUC | S&P AUC | Occlusion AUC | Active Slots | Test Entropy | Helped / Harmed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])

    for ds_block in payload["datasets"].values():
        display = ds_block["display_name"]
        for key, stats in ds_block["summary"].items():
            aucs = stats["robust_auc_mean"]
            g_auc = f"{aucs.get('gaussian', math.nan):.4f}"
            sp_auc = f"{aucs.get('salt_pepper', math.nan):.4f}"
            occ_auc = f"{aucs.get('occlusion', math.nan):.4f}"
            if key == "direct":
                extra = "- | - | -"
            else:
                extra = (
                    f"{stats['active_memories_mean']:.1f} | "
                    f"{stats['test_entropy_mean']:.3f} | "
                    f"+{stats['helped_mean']:.1f} / -{stats['harmed_mean']:.1f}"
                )
            lines.append(
                f"| {display} | {_method_label(key)} | {g_auc} | {sp_auc} | {occ_auc} | {extra} |"
            )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_multi_dataset(
    *,
    dataset_names: Sequence[str] = DEFAULT_MULTI_DATASETS,
    seeds: Sequence[int] = (0, 1, 2),
    methods: Sequence[str] = METHODS,
    quick: bool = False,
    epochs_override: int | None = None,
    data_dir: str | Path = "data",
    output_path: Path = Path("experiment-id 2/results/multi_dataset.json"),
    device_name: str = "auto",
    num_threads: int = 4,
    rgb_channels: tuple[int, int, int] = (32, 64, 128),
) -> dict[str, Any]:
    """Execute multi-dataset Experiment 2 and persist JSON + Markdown outputs."""
    configure_threads(num_threads)
    device = resolve_device(device_name)

    datasets_out: dict[str, Any] = {}
    for ds_name in dataset_names:
        spec = get_dataset_spec(ds_name)
        cfg = build_dataset_config(
            spec,
            quick=quick,
            methods=methods,
            epochs_override=epochs_override,
            rgb_channels=rgb_channels,
        )
        print(f"=== Loading dataset: {spec.display_name} ({spec.name}) ===", flush=True)
        data = load_dataset_tensors(spec.name, data_dir)
        seed_runs = [
            train_and_evaluate_seed(spec, cfg, seed, data, device) for seed in seeds
        ]
        summary = aggregate_dataset_summary(spec, seed_runs, cfg)
        datasets_out[spec.name] = {
            "display_name": spec.display_name,
            "in_channels": spec.in_channels,
            "image_size": spec.image_size,
            "num_classes": spec.num_classes,
            "horizontal_flip": spec.horizontal_flip,
            "config": cfg,
            "seeds": seed_runs,
            "summary": summary,
        }

    payload: dict[str, Any] = {
        "experiment_id": "exp_002_multi_dataset_hopfield",
        "quick": quick,
        "seeds": list(seeds),
        "environment": environment_info(device),
        "datasets": datasets_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = output_path.with_name(f"{output_path.stem}_summary.md")
    write_multi_dataset_report(summary_path, payload)
    print(f"Results JSON: {output_path}")
    print(f"Summary report: {summary_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_MULTI_DATASETS),
        help="Datasets to run (e.g. fashion_mnist kmnist svhn cifar10).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment-id 2/results/multi_dataset.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    output = args.output
    if args.quick and output == Path("experiment-id 2/results/multi_dataset.json"):
        output = Path("experiment-id 2/results/multi_dataset_quick.json")

    payload = run_multi_dataset(
        dataset_names=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        quick=args.quick,
        epochs_override=args.epochs,
        data_dir=args.data_dir,
        output_path=output,
        device_name=args.device,
        num_threads=args.threads,
    )
    summary_path = output.with_name(f"{output.stem}_summary.md")
    print(summary_path.read_text(encoding="utf-8"))
    _ = payload


if __name__ == "__main__":
    main()
