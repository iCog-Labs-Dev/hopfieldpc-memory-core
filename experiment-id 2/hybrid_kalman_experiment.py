"""Experiment 2: hybrid Kalman-momentum prototype comparison.

The CNN is trained once per seed with direct cross-entropy. At each epoch end,
all methods receive the same class-mean embedding targets and epoch accuracy.
Memory updates are detached from CNN gradients.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import sys
from typing import Any

import torch

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
    train_baseline,
)
from hopfieldpc import (
    METHODS,
    HybridOnlineBank,
    OnlineHopfieldBank,
    PrototypeUpdater,
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
    memory: PrototypeUpdater | None = None,
) -> float:
    encoder.eval()
    head.eval()
    total = 0
    correct = 0
    for start in range(0, len(images), batch_size):
        z = encoder(images[start:start + batch_size].to(device))
        if memory is not None:
            values = memory.values.to(device).reshape(-1, memory.dim)
            query_keys = torch.nn.functional.normalize(z - memory.center.to(device), dim=-1)
            memory_keys = torch.nn.functional.normalize(
                values - memory.center.to(device), dim=-1
            )
            scores = memory.beta * (query_keys @ memory_keys.T)
            z = scores.softmax(dim=-1) @ values
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
        "num_classes": 10, "batch_post_write_diagnostics": False,
        "eval_batch_size": cfg["eval_batch_size"], "max_memories_per_class": 8,
        "hopfield_beta": 5.0, "hopfield_num_updates": 1,
        "merge_distance_threshold": 0.2, "normalization_epsilon": 1e-8,
    }
    results: dict[str, Any] = {"seed": seed, "methods": {}}
    for method in cfg["methods"]:
        print(f"seed {seed}: starting method {method}", flush=True)
        torch.manual_seed(seed)
        net = build_baseline_net(seed).to(device)
        bank = HybridOnlineBank(
            64, 10, 8, method=method, ema_alpha=cfg["update"]["alpha"],
            beta=5.0, num_updates=1, merge_distance_threshold=0.2,
            normalization_epsilon=1e-8, gamma=cfg["update"]["gamma"],
            process_noise=cfg["update"]["process_noise"],
            initial_uncertainty=cfg["update"]["initial_uncertainty"],
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
                    net, bank, optimizer, split.train_images[idx].to(device),
                    split.train_labels[idx].to(device), train_ids[idx],
                    epoch=epoch, batch=start // cfg["batch_size"], step=seen,
                    seed=seed, run_id=f"experiment-2-{method}", cfg=online_cfg,
                    memory_enabled=True, admission_policy="accuracy_guided",
                )
                direct_correct += result["direct_correct"]
                seen += result["examples"]
            epoch_accuracy = direct_correct / seen
            bank.current_accuracy = epoch_accuracy
            if epoch == cfg["warmup_epochs"]:
                center = encode(
                    net.encoder, split.train_images,
                    batch_size=cfg["eval_batch_size"], device=device,
                ).mean(dim=0)
                bank.initialize_center(center.to(device))
            history.append({"epoch": epoch, "direct_train_accuracy": epoch_accuracy})
            print(
                f"seed {seed}, {method}: epoch {epoch} "
                f"train_accuracy={epoch_accuracy:.4f} "
                f"memory_size={int(bank.active.sum())}",
                flush=True,
            )
        print(f"seed {seed}, {method}: evaluating clean accuracy", flush=True)
        metrics = {
            "train_accuracy": accuracy(
                net.encoder, net.head, split.train_images, split.train_labels,
                device=device, batch_size=cfg["eval_batch_size"],
            ),
            "validation_accuracy": accuracy(
                net.encoder, net.head, split.val_images, split.val_labels,
                device=device, batch_size=cfg["eval_batch_size"], memory=bank,
            ),
            "test_accuracy": accuracy(
                net.encoder, net.head, split.test_images, split.test_labels,
                device=device, batch_size=cfg["eval_batch_size"], memory=bank,
            ),
            "epochs": history, "test": {},
        }
        print(
            f"seed {seed}, {method}: "
            f"train={metrics['train_accuracy']:.4f} "
            f"val={metrics['validation_accuracy']:.4f} "
            f"test={metrics['test_accuracy']:.4f}",
            flush=True,
        )
        metrics["direct_train_accuracy"] = accuracy(
            net.encoder, net.head, split.train_images, split.train_labels,
            device=device, batch_size=cfg["eval_batch_size"],
        )
        metrics["direct_validation_accuracy"] = accuracy(
            net.encoder, net.head, split.val_images, split.val_labels,
            device=device, batch_size=cfg["eval_batch_size"],
        )
        metrics["direct_test_accuracy"] = accuracy(
            net.encoder, net.head, split.test_images, split.test_labels,
            device=device, batch_size=cfg["eval_batch_size"],
        )
        for kind, levels in cfg["corruptions"].items():
            print(
                f"seed {seed}, {method}: evaluating {kind} corruption",
                flush=True,
            )
            metrics["test"][kind] = {}
            for level in levels:
                images = make_corrupted_test_set(
                    split.test_images, seed=seed, kind=kind, level=level,
                )
                metrics["test"][kind][str(level)] = accuracy(
                    net.encoder, net.head, images, split.test_labels,
                    device=device, batch_size=cfg["eval_batch_size"], memory=bank,
                )
        print(f"seed {seed}, {method}: corruption testing complete", flush=True)
        results["methods"][method] = metrics
    return results


def write_summary_report(path: Path, results: dict[str, Any]) -> None:
    """Write a compact report with train/validation/test accuracy."""
    methods = ["direct", *results["config"]["methods"]]
    lines = [
        "# Experiment 2 Accuracy Summary",
        "",
        f"Seeds: {', '.join(str(seed['seed']) for seed in results['seeds'])}",
        "",
        "| Method | Train accuracy | Validation accuracy | Test accuracy |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        if method == "direct":
            keys = ("direct_train_accuracy", "direct_validation_accuracy", "direct_test_accuracy")
        else:
            keys = ("train_accuracy", "validation_accuracy", "test_accuracy")
        values = [float(seed["methods"][results["config"]["methods"][0]][key])
                  if method == "direct" else float(seed["methods"][method][key])
                  for seed in results["seeds"] for key in keys]
        means = [sum(values[index::3]) / len(results["seeds"]) for index in range(3)]
        label = "Direct CNN" if method == "direct" else method.replace("_", " ").title()
        lines.append(
            f"| {label} | {means[0] * 100:.2f}% | {means[1] * 100:.2f}% | {means[2] * 100:.2f}% |"
        )
    lines.extend([
        "",
        "The full corruption results remain in "
        "[hybrid_kalman.json](hybrid_kalman.json).",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config(quick: bool) -> dict[str, Any]:
    return {
        "methods": list(METHODS),
        "device": "auto",
        "epochs": 2 if quick else 5,
        "warmup_epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "val_size": 500 if quick else 5000,
        "train_subset": 2000 if quick else None,
        "test_subset": 1000 if quick else None,
        "eval_batch_size": 256,
        "prototypes_per_class": 8,
        "update": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 1e-8,
            "process_noise": 0.01,
            "initial_uncertainty": 1.0,
        },
        "corruptions": (
            {"gaussian": [0.0, 0.5, 1.0], "salt_pepper": [0.0, 0.2], "occlusion": [0, 7]}
            if quick else
            {"gaussian": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
             "salt_pepper": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
             "occlusion": [0, 7, 10, 14]}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("experiment-id 2/results/hybrid_kalman.json"))
    args = parser.parse_args()
    cfg = config(args.quick)
    configure_threads(4)
    data = load_mnist("data")
    results = {"config": cfg, "seeds": [run_seed(cfg, seed, data) for seed in args.seeds]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = args.output.with_name("hybrid_kalman_summary.md")
    write_summary_report(report_path, results)
    print(f"Results JSON: {args.output}")
    print(f"Summary report: {report_path}")
    print(Path(report_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
