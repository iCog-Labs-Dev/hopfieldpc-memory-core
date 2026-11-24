"""Experiment 4: Global Responsibility-Weighted Dynamic Prototype Spawning,
Starvation Pruning, and Gram Coalescence in Continuous Hopfield Networks.

Reuses modular components in ``hopfieldpc`` and ``benchmarks.mnist_baseline``,
and reads all hyperparameters and optional variant toggles from ``config.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP4_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(EXP4_DIR) not in sys.path:
    sys.path.insert(0, str(EXP4_DIR))

from benchmarks.mnist_baseline import (
    build_baseline_net,
    configure_threads,
    encode,
    load_mnist,
    make_corrupted_test_set,
    make_splits,
    resolve_device,
)
from config import ExperimentConfig, get_config
from hopfieldpc import (
    GLOBAL_ABLATION_MODES,
    GlobalDynamicHopfieldBank,
    attention_entropy,
    topk_mass,
)


@torch.no_grad()
def evaluate_accuracy(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    memory: GlobalDynamicHopfieldBank | None = None,
) -> dict[str, float]:
    """Evaluate direct or memory-augmented classification accuracy and attention diagnostics."""

    encoder.eval()
    head.eval()
    total = 0
    correct = 0
    entropy_sum = 0.0
    top1_sum = 0.0

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start : start + batch_size].to(device)
        batch_labels = labels[start : start + batch_size]
        z = encoder(batch_imgs)
        if memory is not None and memory.ready:
            z_ret, weights = memory.retrieve(z)
            entropy_sum += float(attention_entropy(weights).sum().item())
            top1_sum += float(topk_mass(weights, k=1).sum().item())
            logits = head(z_ret)
        else:
            logits = head(z)
        preds = logits.argmax(dim=-1).cpu()
        correct += int((preds == batch_labels).sum().item())
        total += len(batch_labels)

    return {
        "accuracy": correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1) if memory is not None and memory.ready else 0.0,
        "mean_top1_mass": top1_sum / max(total, 1) if memory is not None and memory.ready else 0.0,
    }


def run_ablation_for_seed(
    cfg: ExperimentConfig,
    seed: int,
    ablation_mode: str,
    data: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    """Train encoder with direct CE and adapt global dynamic memory for one seed & ablation."""

    device = resolve_device(cfg.device)
    split = make_splits(
        data,
        seed=seed,
        val_size=cfg.val_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
    )

    torch.manual_seed(seed)
    net = build_baseline_net(seed).to(device)
    mem_cfg = cfg.memory.to_global_dynamic_config(ablation_mode=ablation_mode)
    bank = GlobalDynamicHopfieldBank(
        mem_cfg,
        online_update_mode=cfg.memory.online_update_mode,
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    shuffle = torch.Generator().manual_seed(seed)

    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        order = torch.randperm(len(split.train_labels), generator=shuffle)
        direct_correct = 0
        seen = 0
        sim_losses: list[float] = []
        gains: list[float] = []

        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            imgs = split.train_images[idx].to(device)
            lbls = split.train_labels[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            z = net.encoder(imgs)
            logits = net.head(z)
            loss = F.cross_entropy(logits, lbls)
            if not torch.isfinite(loss) or not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite training loss or embedding.")
            loss.backward()
            optimizer.step()

            batch_correct = int((logits.detach().argmax(dim=-1) == lbls).sum().item())
            direct_correct += batch_correct
            seen += len(lbls)

            # Sec. 7.4 Gradient Isolation: Phase 1 memory updates use sg[f_theta(x)].
            if bank.ready:
                up = bank.online_update(z.detach(), task_score=batch_correct / len(lbls))
                sim_losses.append(up["sim_loss"])
                gains.append(up["mean_gain"])

        direct_train_acc = direct_correct / max(seen, 1)
        features = encode(
            net.encoder,
            split.train_images,
            batch_size=cfg.eval_batch_size,
            device=device,
        )

        if epoch == cfg.memory.warmup_epochs and not bank.ready:
            bank.initialize_from_embeddings(features, seed=seed)
            adaptation: dict[str, Any] = {
                "epoch": epoch,
                "active_total": bank.active_count,
                "buffer_utilization": bank.active_count / bank.max_memories,
                "pruned": 0,
                "pruned_low_mass": 0,
                "pruned_diffuse": 0,
                "coalesced": 0,
                "coalesce_candidates": 0,
                "spawned": 0,
                "mean_uncertainty": float(bank.uncertainty[bank.mask].mean().item()),
                "mean_dispersion": 0.0,
                "gram_after": {"mean_off_diag": 0.0, "max_off_diag": 0.0},
            }
        else:
            adaptation = bank.adapt_structure(
                features,
                epoch=epoch,
                task_score=direct_train_acc,
                allow_spawn=(epoch < cfg.epochs),
            )

        val_mem = evaluate_accuracy(
            net.encoder,
            net.head,
            split.val_images,
            split.val_labels,
            device=device,
            batch_size=cfg.eval_batch_size,
            memory=bank,
        )
        val_dir = evaluate_accuracy(
            net.encoder,
            net.head,
            split.val_images,
            split.val_labels,
            device=device,
            batch_size=cfg.eval_batch_size,
            memory=None,
        )

        epoch_record = {
            "epoch": epoch,
            "direct_train_accuracy": direct_train_acc,
            "direct_val_accuracy": val_dir["accuracy"],
            "memory_val_accuracy": val_mem["accuracy"],
            "retrieval_delta_val": val_mem["accuracy"] - val_dir["accuracy"],
            "mean_online_sim_loss": sum(sim_losses) / len(sim_losses) if sim_losses else 0.0,
            "mean_online_gain": sum(gains) / len(gains) if gains else 0.0,
            **{k: v for k, v in adaptation.items() if k != "support_summary"},
            "mean_kish_ess": adaptation.get("support_summary", {}).get("mean_kish_ess", 0.0),
            "mean_sharpness": adaptation.get("support_summary", {}).get("mean_sharpness", 0.0),
        }
        history.append(epoch_record)
        print(
            f"seed={seed} [{ablation_mode}] epoch={epoch}/{cfg.epochs} "
            f"dir_val={val_dir['accuracy']:.4f} mem_val={val_mem['accuracy']:.4f} "
            f"active={adaptation['active_total']} "
            f"(prune={adaptation['pruned']}, coal={adaptation['coalesced']}, spawn={adaptation['spawned']})",
            flush=True,
        )

    train_mem = evaluate_accuracy(
        net.encoder, net.head, split.train_images, split.train_labels,
        device=device, batch_size=cfg.eval_batch_size, memory=bank,
    )
    train_dir = evaluate_accuracy(
        net.encoder, net.head, split.train_images, split.train_labels,
        device=device, batch_size=cfg.eval_batch_size, memory=None,
    )
    val_mem = evaluate_accuracy(
        net.encoder, net.head, split.val_images, split.val_labels,
        device=device, batch_size=cfg.eval_batch_size, memory=bank,
    )
    val_dir = evaluate_accuracy(
        net.encoder, net.head, split.val_images, split.val_labels,
        device=device, batch_size=cfg.eval_batch_size, memory=None,
    )
    test_mem = evaluate_accuracy(
        net.encoder, net.head, split.test_images, split.test_labels,
        device=device, batch_size=cfg.eval_batch_size, memory=bank,
    )
    test_dir = evaluate_accuracy(
        net.encoder, net.head, split.test_images, split.test_labels,
        device=device, batch_size=cfg.eval_batch_size, memory=None,
    )

    corruptions_mem: dict[str, dict[str, float]] = {}
    corruptions_dir: dict[str, dict[str, float]] = {}
    for kind, levels in cfg.corruptions.items():
        corruptions_mem[kind] = {}
        corruptions_dir[kind] = {}
        for level in levels:
            c_imgs = make_corrupted_test_set(split.test_images, seed=seed, kind=kind, level=level)
            c_mem = evaluate_accuracy(
                net.encoder, net.head, c_imgs, split.test_labels,
                device=device, batch_size=cfg.eval_batch_size, memory=bank,
            )
            c_dir = evaluate_accuracy(
                net.encoder, net.head, c_imgs, split.test_labels,
                device=device, batch_size=cfg.eval_batch_size, memory=None,
            )
            corruptions_mem[kind][str(level)] = c_mem["accuracy"]
            corruptions_dir[kind][str(level)] = c_dir["accuracy"]

    active_ages = bank.age_epochs[bank.mask].tolist()
    all_lifetimes = bank.completed_lifetimes + active_ages
    avg_lifetime = sum(all_lifetimes) / len(all_lifetimes) if all_lifetimes else 0.0
    spawn_churn_rate = (
        bank.spawned_subsequently_removed / bank.total_spawned
        if bank.total_spawned > 0
        else 0.0
    )

    return {
        "train_accuracy": train_mem["accuracy"],
        "validation_accuracy": val_mem["accuracy"],
        "test_accuracy": test_mem["accuracy"],
        "direct_train_accuracy": train_dir["accuracy"],
        "direct_validation_accuracy": val_dir["accuracy"],
        "direct_test_accuracy": test_dir["accuracy"],
        "retrieval_delta_test": test_mem["accuracy"] - test_dir["accuracy"],
        "test_entropy": test_mem["mean_entropy"],
        "test_top1_mass": test_mem["mean_top1_mass"],
        "final_active_prototypes": bank.active_count,
        "buffer_utilization": bank.active_count / bank.max_memories,
        "total_pruned": bank.total_pruned,
        "total_pruned_low_mass": bank.total_pruned_low_mass,
        "total_pruned_diffuse": bank.total_pruned_diffuse,
        "total_coalesced": bank.total_coalesced,
        "total_spawned": bank.total_spawned,
        "total_turnover": bank.total_pruned + bank.total_coalesced + bank.total_spawned,
        "average_prototype_lifetime": avg_lifetime,
        "spawned_subsequently_removed_fraction": spawn_churn_rate,
        "epochs": history,
        "test": corruptions_mem,
        "direct_test_corruptions": corruptions_dir,
    }


def run_seed(cfg: ExperimentConfig, seed: int, data: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    return {
        "seed": seed,
        "ablations": {
            mode: run_ablation_for_seed(cfg, seed, mode, data)
            for mode in cfg.ablations
        },
    }


def write_summary_report(path: Path, results: dict[str, Any]) -> None:
    """Write Experiment 4 Markdown summary comparing all evaluated ablations."""

    ablations = results["config"]["ablations"]
    seeds = results["seeds"]
    n_seeds = max(len(seeds), 1)
    first_ablation = ablations[0]

    lines = [
        "# Experiment 4: Global Responsibility-Weighted Dynamic Prototypes Summary",
        "",
        f"Seeds: {', '.join(str(s['seed']) for s in seeds)}",
        "",
        "## Clean Accuracy & Structural Adaptation",
        "",
        "| Variant | Train | Val | Test | Δ vs Direct | Active | Pruned | Coalesced | Spawned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    dir_vals = [
        sum(float(s["ablations"][first_ablation][k]) for s in seeds) / n_seeds
        for k in ("direct_train_accuracy", "direct_validation_accuracy", "direct_test_accuracy")
    ]
    lines.append(
        f"| Direct CNN | {dir_vals[0]*100:.2f}% | {dir_vals[1]*100:.2f}% | {dir_vals[2]*100:.2f}% | - | - | - | - | - |"
    )

    VARIANT_DISPLAY_NAMES = {
        "fixed_capacity": "Fixed-Capacity Hopfield",
        "dynamic_adaptive": "Dynamic Adaptive Hopfield",
    }

    for mode in ablations:
        train_m = sum(float(s["ablations"][mode]["train_accuracy"]) for s in seeds) / n_seeds
        val_m = sum(float(s["ablations"][mode]["validation_accuracy"]) for s in seeds) / n_seeds
        test_m = sum(float(s["ablations"][mode]["test_accuracy"]) for s in seeds) / n_seeds
        delta_m = sum(float(s["ablations"][mode]["retrieval_delta_test"]) for s in seeds) / n_seeds
        act_m = sum(float(s["ablations"][mode]["final_active_prototypes"]) for s in seeds) / n_seeds
        prn_m = sum(float(s["ablations"][mode]["total_pruned"]) for s in seeds) / n_seeds
        coa_m = sum(float(s["ablations"][mode]["total_coalesced"]) for s in seeds) / n_seeds
        spn_m = sum(float(s["ablations"][mode]["total_spawned"]) for s in seeds) / n_seeds
        label = VARIANT_DISPLAY_NAMES.get(mode, mode.replace("_", " ").title())
        lines.append(
            f"| {label} | {train_m*100:.2f}% | {val_m*100:.2f}% | {test_m*100:.2f}% "
            f"| {delta_m*100:+.2f} pp | {act_m:.1f} | {prn_m:.1f} | {coa_m:.1f} | {spn_m:.1f} |"
        )

    headers = [VARIANT_DISPLAY_NAMES.get(m, m.replace("_", " ").title()) for m in ablations]
    lines.extend(
        [
            "",
            "## Corruption Robustness (Means Across Seeds)",
            "",
            "| Corruption | Level | Direct CNN | " + " | ".join(headers) + " |",
            "|---|---:|---:|" + "|".join(["---:" for _ in ablations]) + "|",
        ]
    )
    for kind, levels in results["config"]["corruptions"].items():
        for level in levels:
            lk = str(level)
            d_acc = (
                sum(float(s["ablations"][first_ablation]["direct_test_corruptions"][kind][lk]) for s in seeds)
                / n_seeds
            )
            row = [f"{d_acc*100:.2f}%"]
            for mode in ablations:
                m_acc = sum(float(s["ablations"][mode]["test"][kind][lk]) for s in seeds) / n_seeds
                row.append(f"{m_acc*100:.2f}%")
            lines.append(f"| {kind} | {level} | {' | '.join(row)} |")

    json_name = Path(results["config"]["output_path"]).name
    lines.extend(
        [
            "",
            f"Raw JSON metrics: [{json_name}]({json_name}).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-auto-calibrate", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--ablations", nargs="+", choices=list(GLOBAL_ABLATION_MODES), default=None)
    parser.add_argument("--online-update-mode", choices=["batch", "sequential"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config(
        quick=args.quick,
        auto_calibrate=False if args.no_auto_calibrate else None,
        ablations=tuple(args.ablations) if args.ablations else None,
        seeds=tuple(args.seeds) if args.seeds else None,
        online_update_mode=args.online_update_mode,
        output_path=args.output,
    )
    configure_threads(cfg.num_threads)
    data = load_mnist(cfg.data_dir)
    results = {
        "config": cfg.to_dict(),
        "seeds": [run_seed(cfg, seed, data) for seed in cfg.seeds],
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = cfg.output_path.with_name(cfg.output_path.stem + "_summary.md")
    write_summary_report(report_path, results)
    print(f"Results JSON: {cfg.output_path}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
