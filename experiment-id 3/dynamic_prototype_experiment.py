"""Experiment 3: Dynamic prototype spawning and pruning with hybrid Kalman-momentum updates.

Extends the online minibatch protocol from Experiment 1 & 2 with:
- Masked fixed-size PyTorch buffers up to K_max prototypes per class (initial online cap K_init).
- Pruning via assignment support: deactivate prototypes with |S_{j,n}| < tau_support
  for T_prune consecutive epochs (preserving at least one prototype per class).
- Spawning via uncertainty: split prototypes with P_{j,n+1} > tau_uncert along the
  first principal component of their assigned embeddings S_{j,n}, resetting V=0 and P=P_0.
- Masked centered-cosine Hopfield retrieval returning raw prototype combinations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_1 = ROOT / "experiment-id 1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(EXPERIMENT_1) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_1))

from benchmarks.mnist_baseline import (
    build_baseline_net,
    configure_threads,
    encode,
    load_mnist,
    make_corrupted_test_set,
    make_splits,
    resolve_device,
)
from hopfieldpc import (
    METHODS,
    DynamicHybridOnlineBank,
    masked_centered_retrieve,
)
from mnist_online_memory import train_step as online_train_step


@torch.no_grad()
def accuracy(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    memory: DynamicHybridOnlineBank | None = None,
) -> float:
    encoder.eval()
    head.eval()
    total = 0
    correct = 0
    for start in range(0, len(images), batch_size):
        z = encoder(images[start:start + batch_size].to(device))
        if memory is not None:
            z, _ = memory.retrieve_masked(z)
        prediction = head(z).argmax(dim=-1).cpu()
        target = labels[start:start + batch_size]
        correct += int((prediction == target).sum())
        total += len(target)
    return correct / total


def run_seed(cfg: dict[str, Any], seed: int, data: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    split = make_splits(
        data,
        seed=seed,
        val_size=cfg["val_size"],
        train_subset=cfg["train_subset"],
        test_subset=cfg["test_subset"],
    )
    permutation = torch.randperm(
        len(data[0]), generator=torch.Generator().manual_seed(seed)
    )
    train_ids = permutation[cfg["val_size"]:][:len(split.train_labels)]
    online_cfg = {
        "num_classes": 10,
        "batch_post_write_diagnostics": False,
        "eval_batch_size": cfg["eval_batch_size"],
        "max_memories_per_class": cfg["dynamic"]["k_max"],
        "hopfield_beta": 5.0,
        "hopfield_num_updates": 1,
        "merge_distance_threshold": 0.2,
        "normalization_epsilon": 1e-8,
    }
    results: dict[str, Any] = {"seed": seed, "methods": {}}
    for method in cfg["methods"]:
        print(f"seed {seed}: starting method {method}", flush=True)
        torch.manual_seed(seed)
        net = build_baseline_net(seed).to(device)
        bank = DynamicHybridOnlineBank(
            64,
            10,
            cfg["dynamic"]["k_max"],
            initial_memories_per_class=cfg["dynamic"]["k_init"],
            method=method,
            ema_alpha=cfg["update"]["alpha"],
            beta=5.0,
            num_updates=1,
            merge_distance_threshold=0.2,
            normalization_epsilon=1e-8,
            gamma=cfg["update"]["gamma"],
            process_noise=cfg["update"]["process_noise"],
            initial_uncertainty=cfg["update"]["initial_uncertainty"],
            tau_support=cfg["dynamic"]["tau_support"],
            prune_patience=cfg["dynamic"]["prune_patience"],
            tau_uncert=cfg["dynamic"]["tau_uncert"],
            split_scale=cfg["dynamic"]["split_scale"],
        ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
        shuffle = torch.Generator().manual_seed(seed)
        history: list[dict[str, Any]] = []
        for epoch in range(1, cfg["epochs"] + 1):
            print(
                f"seed {seed}, {method}: epoch {epoch}/{cfg['epochs']}",
                flush=True,
            )
            net.train()
            order = torch.randperm(len(split.train_labels), generator=shuffle)
            direct_correct = 0
            seen = 0
            for start in range(0, len(order), cfg["batch_size"]):
                idx = order[start:start + cfg["batch_size"]]
                result = online_train_step(
                    net,
                    bank,
                    optimizer,
                    split.train_images[idx].to(device),
                    split.train_labels[idx].to(device),
                    train_ids[idx],
                    epoch=epoch,
                    batch=start // cfg["batch_size"],
                    step=seen,
                    seed=seed,
                    run_id=f"experiment-3-{method}",
                    cfg=online_cfg,
                    memory_enabled=True,
                    admission_policy="accuracy_guided",
                )
                direct_correct += result["direct_correct"]
                seen += result["examples"]
            epoch_accuracy = direct_correct / seen
            bank.current_accuracy = epoch_accuracy

            features = encode(
                net.encoder,
                split.train_images,
                batch_size=cfg["eval_batch_size"],
                device=device,
            )
            if epoch == cfg["warmup_epochs"]:
                center = features.mean(dim=0)
                bank.initialize_center(center.to(device))
                adaptation = {
                    "pruned": 0,
                    "spawned": 0,
                    "active_total": int(bank.active.sum().item()),
                    "per_class_active": bank.occupancy(),
                    "mean_uncertainty": 0.0,
                }
            else:
                with torch.no_grad():
                    net.eval()
                    preds = []
                    for start in range(0, len(features), cfg["eval_batch_size"]):
                        preds.append(
                            net.head(features[start:start + cfg["eval_batch_size"]].to(device))
                            .argmax(dim=-1)
                            .cpu()
                        )
                    train_preds = torch.cat(preds)
                adaptation = bank.adapt_prototypes(
                    features,
                    split.train_labels,
                    train_preds,
                    epoch=epoch,
                    step=seen,
                    allow_spawn=(epoch < cfg["epochs"]),
                )

            history.append(
                {
                    "epoch": epoch,
                    "direct_train_accuracy": epoch_accuracy,
                    **adaptation,
                }
            )
            print(
                f"seed {seed}, {method}: epoch {epoch} "
                f"train_accuracy={epoch_accuracy:.4f} "
                f"memory_size={adaptation['active_total']} "
                f"(spawned={adaptation['spawned']}, pruned={adaptation['pruned']})",
                flush=True,
            )
        print(f"seed {seed}, {method}: evaluating clean accuracy", flush=True)
        metrics: dict[str, Any] = {
            "train_accuracy": accuracy(
                net.encoder,
                net.head,
                split.train_images,
                split.train_labels,
                device=device,
                batch_size=cfg["eval_batch_size"],
                memory=bank,
            ),
            "validation_accuracy": accuracy(
                net.encoder,
                net.head,
                split.val_images,
                split.val_labels,
                device=device,
                batch_size=cfg["eval_batch_size"],
                memory=bank,
            ),
            "test_accuracy": accuracy(
                net.encoder,
                net.head,
                split.test_images,
                split.test_labels,
                device=device,
                batch_size=cfg["eval_batch_size"],
                memory=bank,
            ),
            "final_active_prototypes": int(bank.active.sum().item()),
            "final_per_class_prototypes": bank.occupancy(),
            "total_spawned": bank.total_spawned,
            "total_pruned": bank.total_pruned,
            "epochs": history,
            "test": {},
        }
        metrics["direct_train_accuracy"] = accuracy(
            net.encoder,
            net.head,
            split.train_images,
            split.train_labels,
            device=device,
            batch_size=cfg["eval_batch_size"],
        )
        metrics["direct_validation_accuracy"] = accuracy(
            net.encoder,
            net.head,
            split.val_images,
            split.val_labels,
            device=device,
            batch_size=cfg["eval_batch_size"],
        )
        metrics["direct_test_accuracy"] = accuracy(
            net.encoder,
            net.head,
            split.test_images,
            split.test_labels,
            device=device,
            batch_size=cfg["eval_batch_size"],
        )
        metrics["direct_test_corruptions"] = {}
        print(
            f"seed {seed}, {method}: "
            f"train={metrics['train_accuracy']:.4f} "
            f"val={metrics['validation_accuracy']:.4f} "
            f"test={metrics['test_accuracy']:.4f} "
            f"direct_test={metrics['direct_test_accuracy']:.4f} "
            f"active={metrics['final_active_prototypes']}",
            flush=True,
        )
        for kind, levels in cfg["corruptions"].items():
            print(
                f"seed {seed}, {method}: evaluating {kind} corruption (memory & direct baseline)",
                flush=True,
            )
            metrics["test"][kind] = {}
            metrics["direct_test_corruptions"][kind] = {}
            for level in levels:
                images = make_corrupted_test_set(
                    split.test_images,
                    seed=seed,
                    kind=kind,
                    level=level,
                )
                mem_acc = accuracy(
                    net.encoder,
                    net.head,
                    images,
                    split.test_labels,
                    device=device,
                    batch_size=cfg["eval_batch_size"],
                    memory=bank,
                )
                base_acc = accuracy(
                    net.encoder,
                    net.head,
                    images,
                    split.test_labels,
                    device=device,
                    batch_size=cfg["eval_batch_size"],
                    memory=None,
                )
                metrics["test"][kind][str(level)] = mem_acc
                metrics["direct_test_corruptions"][kind][str(level)] = base_acc
                print(
                    f"  seed {seed}, {method}, {kind}={level}: "
                    f"baseline={base_acc:.4f} {method}={mem_acc:.4f}",
                    flush=True,
                )
        print(f"seed {seed}, {method}: corruption testing complete", flush=True)
        results["methods"][method] = metrics
    return results


def write_summary_report(path: Path, results: dict[str, Any]) -> None:
    """Write a compact report with train/validation/test accuracy, dynamic counts, and per-noise baseline comparisons."""
    methods = ["direct", *results["config"]["methods"]]
    lines = [
        "# Experiment 3 Accuracy & Dynamic Capacity Summary",
        "",
        f"Seeds: {', '.join(str(seed['seed']) for seed in results['seeds'])}",
        "",
        "## Clean Accuracy & Dynamic Prototype Allocation",
        "",
        "| Method | Train accuracy | Validation accuracy | Test accuracy | Active Prototypes | Spawned | Pruned |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        if method == "direct":
            keys = ("direct_train_accuracy", "direct_validation_accuracy", "direct_test_accuracy")
            values = [
                float(seed["methods"][results["config"]["methods"][0]][key])
                for seed in results["seeds"]
                for key in keys
            ]
            means = [sum(values[index::3]) / len(results["seeds"]) for index in range(3)]
            lines.append(
                f"| Direct CNN | {means[0] * 100:.2f}% | {means[1] * 100:.2f}% | {means[2] * 100:.2f}% | - | - | - |"
            )
        else:
            keys = ("train_accuracy", "validation_accuracy", "test_accuracy")
            values = [
                float(seed["methods"][method][key])
                for seed in results["seeds"]
                for key in keys
            ]
            means = [sum(values[index::3]) / len(results["seeds"]) for index in range(3)]
            avg_active = sum(
                seed["methods"][method]["final_active_prototypes"] for seed in results["seeds"]
            ) / len(results["seeds"])
            avg_spawned = sum(
                seed["methods"][method]["total_spawned"] for seed in results["seeds"]
            ) / len(results["seeds"])
            avg_pruned = sum(
                seed["methods"][method]["total_pruned"] for seed in results["seeds"]
            ) / len(results["seeds"])
            label = method.replace("_", " ").title()
            lines.append(
                f"| {label} | {means[0] * 100:.2f}% | {means[1] * 100:.2f}% | {means[2] * 100:.2f}% "
                f"| {avg_active:.1f} | {avg_spawned:.1f} | {avg_pruned:.1f} |"
            )

    method_names = results["config"]["methods"]
    header_cols = " | ".join(["Direct CNN", *[m.replace("_", " ").title() for m in method_names]])
    sep_cols = "|".join(["---:" for _ in range(len(method_names) + 1)])
    lines.extend(
        [
            "",
            "## Corruption Accuracy by Noise Level (Means Across Seeds)",
            "",
            f"| Corruption | Level | {header_cols} |",
            f"|---|---:|{sep_cols}|",
        ]
    )
    first_method = method_names[0]
    for kind, levels in results["config"]["corruptions"].items():
        for level in levels:
            lvl_key = str(level)
            base_mean = sum(
                float(seed["methods"][first_method]["direct_test_corruptions"][kind][lvl_key])
                for seed in results["seeds"]
            ) / len(results["seeds"])
            row_vals = [f"{base_mean * 100:.2f}%"]
            for m in method_names:
                m_mean = sum(
                    float(seed["methods"][m]["test"][kind][lvl_key])
                    for seed in results["seeds"]
                ) / len(results["seeds"])
                row_vals.append(f"{m_mean * 100:.2f}%")
            lines.append(f"| {kind} | {level} | {' | '.join(row_vals)} |")

    lines.extend(
        [
            "",
            "The full corruption and per-epoch dynamic prototype results remain in "
            "[dynamic_prototypes.json](dynamic_prototypes.json).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config(quick: bool) -> dict[str, Any]:
    return {
        "methods": list(METHODS),
        "device": "auto",
        "epochs": 3 if quick else 10,
        "warmup_epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "val_size": 500 if quick else 5000,
        "train_subset": 2000 if quick else None,
        "test_subset": 1000 if quick else None,
        "eval_batch_size": 256,
        "dynamic": {
            "k_init": 1,
            "k_max": 16,
            "tau_support": 10,
            "prune_patience": 2,
            "tau_uncert": 0.5,
            "split_scale": 0.1,
        },
        "update": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 1e-8,
            "process_noise": 0.01,
            "initial_uncertainty": 1.0,
        },
        "corruptions": (
            {"gaussian": [0.0, 0.5, 1.0], "salt_pepper": [0.0, 0.2], "occlusion": [0, 7]}
            if quick
            else {
                "gaussian": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                "salt_pepper": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
                "occlusion": [0, 7, 10, 14],
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment-id 3/results/dynamic_prototypes.json"),
    )
    args = parser.parse_args()
    cfg = config(args.quick)
    configure_threads(4)
    data = load_mnist("data")
    results = {"config": cfg, "seeds": [run_seed(cfg, seed, data) for seed in args.seeds]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = args.output.with_name("dynamic_prototypes_summary.md")
    write_summary_report(report_path, results)
    print(f"Results JSON: {args.output}")
    print(f"Summary report: {report_path}")
    print(Path(report_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
