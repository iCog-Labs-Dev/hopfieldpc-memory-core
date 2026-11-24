"""Experiment 4 (CIFAR-10): Global Responsibility-Weighted Dynamic Prototype
Spawning, Starvation Pruning, and Gram Coalescence in Continuous Hopfield Networks.

Pairs the RGB CNN baseline in ``benchmarks.cifar10_baseline`` with ``hopfieldpc``'s
self-calibrating ``GlobalDynamicHopfieldBank`` and reads all CIFAR-10 hyperparameters
from ``cifar10_config.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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

from benchmarks.cifar10_baseline import (
    build_cifar10_baseline_net,
    configure_threads,
    encode,
    load_cifar10,
    make_cifar10_splits,
    make_corrupted_cifar10_test_set,
    resolve_device,
)
from cifar10_config import Cifar10ExperimentConfig, get_cifar10_config
from experiment import evaluate_accuracy, write_summary_report
from hopfieldpc import GLOBAL_ABLATION_MODES, GlobalDynamicHopfieldBank


def run_seed(
    cfg: Cifar10ExperimentConfig,
    seed: int,
    data: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    """Train the CIFAR-10 CNN backbone once for ``seed`` and adapt all configured memory banks.

    Because Phase 1 updates use gradient isolation ``sg[f_theta(x)]`` (Sec. 7.4),
    evolving all ``cfg.ablations`` memory banks in lockstep on the same CNN stream is
    mathematically identical to separate CNN runs while halving wall-clock time.
    """

    device = resolve_device(cfg.device)
    split = make_cifar10_splits(
        data,
        seed=seed,
        val_size=cfg.val_size,
        train_subset=cfg.train_subset,
        test_subset=cfg.test_subset,
    )

    c1, c2, c3 = cfg.encoder_channels
    torch.manual_seed(seed)
    net = build_cifar10_baseline_net(
        seed,
        c1=c1,
        c2=c2,
        c3=c3,
        hidden=cfg.encoder_hidden,
        embed_dim=cfg.memory.dim,
        conv_dropout=cfg.conv_dropout,
        dense_dropout=cfg.dense_dropout,
    ).to(device)

    banks: dict[str, GlobalDynamicHopfieldBank] = {
        mode: GlobalDynamicHopfieldBank(
            cfg.memory.to_global_dynamic_config(ablation_mode=mode),
            online_update_mode=cfg.memory.online_update_mode,
        ).to(device)
        for mode in cfg.ablations
    }

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    shuffle = torch.Generator().manual_seed(seed)
    has_dropout = (cfg.conv_dropout > 0.0) or (cfg.dense_dropout > 0.0)

    histories: dict[str, list[dict[str, Any]]] = {mode: [] for mode in cfg.ablations}

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        order = torch.randperm(len(split.train_labels), generator=shuffle)
        direct_correct = 0
        seen = 0
        sim_losses: dict[str, list[float]] = {mode: [] for mode in cfg.ablations}
        gains: dict[str, list[float]] = {mode: [] for mode in cfg.ablations}

        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            imgs = split.train_images[idx].to(device)
            lbls = split.train_labels[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            z = net.encoder(imgs)
            logits = net.head(z)
            loss = F.cross_entropy(logits, lbls)
            if not torch.isfinite(loss) or not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite CIFAR-10 training loss or embedding.")
            loss.backward()
            optimizer.step()

            batch_correct = int((logits.detach().argmax(dim=-1) == lbls).sum().item())
            direct_correct += batch_correct
            seen += len(lbls)

            # Sec. 7.4 Gradient Isolation: Phase 1 memory updates use sg[f_theta(x)]
            # in deterministic eval() mode so dropout noise does not perturb prototypes.
            if any(b.ready for b in banks.values()):
                if has_dropout:
                    with torch.no_grad():
                        net.encoder.eval()
                        z_sg = net.encoder(imgs)
                        net.encoder.train()
                else:
                    z_sg = z.detach()
                for mode, bank in banks.items():
                    if bank.ready:
                        up = bank.online_update(z_sg, task_score=batch_correct / len(lbls))
                        sim_losses[mode].append(up["sim_loss"])
                        gains[mode].append(up["mean_gain"])

        direct_train_acc = direct_correct / max(seen, 1)

        # Encode clean (unaugmented) train split once in eval() mode for Phase 2 adaptation.
        features = encode(
            net.encoder,
            split.train_images,
            batch_size=cfg.eval_batch_size,
            device=device,
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

        for mode, bank in banks.items():
            if epoch >= cfg.memory.warmup_epochs and not bank.ready:
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
            elif bank.ready:
                adaptation = bank.adapt_structure(
                    features,
                    epoch=epoch,
                    task_score=direct_train_acc,
                    allow_spawn=(epoch < cfg.epochs),
                )
            else:
                adaptation = {
                    "epoch": epoch,
                    "active_total": 0,
                    "buffer_utilization": 0.0,
                    "pruned": 0,
                    "pruned_low_mass": 0,
                    "pruned_diffuse": 0,
                    "coalesced": 0,
                    "coalesce_candidates": 0,
                    "spawned": 0,
                    "mean_uncertainty": 0.0,
                    "mean_dispersion": 0.0,
                    "gram_after": {"mean_off_diag": 0.0, "max_off_diag": 0.0},
                }

            val_mem = (
                evaluate_accuracy(
                    net.encoder,
                    net.head,
                    split.val_images,
                    split.val_labels,
                    device=device,
                    batch_size=cfg.eval_batch_size,
                    memory=bank,
                )
                if bank.ready
                else val_dir
            )

            epoch_record = {
                "epoch": epoch,
                "direct_train_accuracy": direct_train_acc,
                "direct_val_accuracy": val_dir["accuracy"],
                "memory_val_accuracy": val_mem["accuracy"],
                "retrieval_delta_val": val_mem["accuracy"] - val_dir["accuracy"],
                "mean_online_sim_loss": (
                    sum(sim_losses[mode]) / len(sim_losses[mode])
                    if sim_losses[mode]
                    else 0.0
                ),
                "mean_online_gain": (
                    sum(gains[mode]) / len(gains[mode]) if gains[mode] else 0.0
                ),
                **{k: v for k, v in adaptation.items() if k != "support_summary"},
                "mean_kish_ess": adaptation.get("support_summary", {}).get(
                    "mean_kish_ess", 0.0
                ),
                "mean_sharpness": adaptation.get("support_summary", {}).get(
                    "mean_sharpness", 0.0
                ),
            }
            histories[mode].append(epoch_record)
            print(
                f"seed={seed} [{mode}] epoch={epoch}/{cfg.epochs} "
                f"dir_val={val_dir['accuracy']:.4f} mem_val={val_mem['accuracy']:.4f} "
                f"active={adaptation['active_total']} "
                f"(prune={adaptation['pruned']}, coal={adaptation['coalesced']}, spawn={adaptation['spawned']})",
                flush=True,
            )

    train_dir = evaluate_accuracy(
        net.encoder,
        net.head,
        split.train_images,
        split.train_labels,
        device=device,
        batch_size=cfg.eval_batch_size,
        memory=None,
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
    test_dir = evaluate_accuracy(
        net.encoder,
        net.head,
        split.test_images,
        split.test_labels,
        device=device,
        batch_size=cfg.eval_batch_size,
        memory=None,
    )

    corruptions_dir: dict[str, dict[str, float]] = {}
    corrupted_sets: dict[tuple[str, str], torch.Tensor] = {}
    for kind, levels in cfg.corruptions.items():
        corruptions_dir[kind] = {}
        for level in levels:
            lk = str(level)
            c_imgs = make_corrupted_cifar10_test_set(
                split.test_images, seed=seed, kind=kind, level=float(level)
            )
            corrupted_sets[(kind, lk)] = c_imgs
            c_dir = evaluate_accuracy(
                net.encoder,
                net.head,
                c_imgs,
                split.test_labels,
                device=device,
                batch_size=cfg.eval_batch_size,
                memory=None,
            )
            corruptions_dir[kind][lk] = c_dir["accuracy"]

    ablation_results: dict[str, Any] = {}
    for mode, bank in banks.items():
        train_mem = evaluate_accuracy(
            net.encoder,
            net.head,
            split.train_images,
            split.train_labels,
            device=device,
            batch_size=cfg.eval_batch_size,
            memory=bank,
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
        test_mem = evaluate_accuracy(
            net.encoder,
            net.head,
            split.test_images,
            split.test_labels,
            device=device,
            batch_size=cfg.eval_batch_size,
            memory=bank,
        )

        corruptions_mem: dict[str, dict[str, float]] = {}
        for kind, levels in cfg.corruptions.items():
            corruptions_mem[kind] = {}
            for level in levels:
                lk = str(level)
                c_mem = evaluate_accuracy(
                    net.encoder,
                    net.head,
                    corrupted_sets[(kind, lk)],
                    split.test_labels,
                    device=device,
                    batch_size=cfg.eval_batch_size,
                    memory=bank,
                )
                corruptions_mem[kind][lk] = c_mem["accuracy"]

        active_ages = bank.age_epochs[bank.mask].tolist()
        all_lifetimes = bank.completed_lifetimes + active_ages
        avg_lifetime = sum(all_lifetimes) / len(all_lifetimes) if all_lifetimes else 0.0
        spawn_churn_rate = (
            bank.spawned_subsequently_removed / bank.total_spawned
            if bank.total_spawned > 0
            else 0.0
        )

        ablation_results[mode] = {
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
            "epochs": histories[mode],
            "test": corruptions_mem,
            "direct_test_corruptions": corruptions_dir,
        }

    return {"seed": seed, "ablations": ablation_results}


def run_ablation_for_seed(
    cfg: Cifar10ExperimentConfig,
    seed: int,
    ablation_mode: str,
    data: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    """Single-variant convenience wrapper matching ``experiment.run_ablation_for_seed``."""

    single_cfg = replace(cfg, ablations=(ablation_mode,))
    return run_seed(single_cfg, seed, data)["ablations"][ablation_mode]


def write_cifar10_summary_report(path: Path, results: dict[str, Any]) -> None:
    """Write Experiment 4 CIFAR-10 Markdown summary report."""

    write_summary_report(path, results)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# Experiment 4: Global Responsibility-Weighted Dynamic Prototypes Summary",
        "# Experiment 4 (CIFAR-10): Global Responsibility-Weighted Dynamic Prototypes Summary",
        1,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-auto-calibrate", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(GLOBAL_ABLATION_MODES),
        default=None,
    )
    parser.add_argument(
        "--online-update-mode",
        choices=["batch", "sequential"],
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    exp_overrides: dict[str, Any] = {}
    if args.epochs is not None:
        exp_overrides["epochs"] = args.epochs

    cfg = get_cifar10_config(
        quick=args.quick,
        auto_calibrate=False if args.no_auto_calibrate else None,
        ablations=tuple(args.ablations) if args.ablations else None,
        seeds=tuple(args.seeds) if args.seeds else None,
        online_update_mode=args.online_update_mode,
        output_path=args.output,
        experiment_overrides=exp_overrides if exp_overrides else None,
    )
    configure_threads(cfg.num_threads)
    data = load_cifar10(cfg.data_dir)
    results = {
        "config": cfg.to_dict(),
        "seeds": [run_seed(cfg, seed, data) for seed in cfg.seeds],
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = cfg.output_path.with_name(cfg.output_path.stem + "_summary.md")
    write_cifar10_summary_report(report_path, results)
    print(f"Results JSON: {cfg.output_path}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
