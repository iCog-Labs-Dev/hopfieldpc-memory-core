"""Shared-backbone training and evaluation engine for multi-dataset Experiment 2.

Trains one CNN encoder and linear classifier per seed while updating all
Hopfield prototype memory banks (EMA, Momentum-only, Kalman-only, Hybrid) in
lockstep on detached pre-last-layer embeddings.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_2 = Path(__file__).resolve().parent
for path in (str(EXPERIMENT_2), str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from backbones import BackboneNet, build_backbone
from datasets import (
    DatasetSpec,
    DatasetSplits,
    augment_batch,
    make_dataset_splits,
)

from benchmarks.mnist_baseline import encode, make_corrupted_test_set, robust_auc
from hopfieldpc import (
    METHODS,
    HybridOnlineBank,
    attention_entropy,
)


@torch.no_grad()
def evaluate_embeddings(
    head: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    bank: HybridOnlineBank | None = None,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate Direct head or Hopfield retrieval from precomputed encoder embeddings."""
    head.eval()
    snapshot = None
    if bank is not None:
        snapshot = bank.snapshot()
        if not snapshot.ready:
            raise RuntimeError(
                "Hopfield bank is not ready (missing class coverage or uninitialized center)."
            )

    total = 0
    correct = 0
    entropy_sum = 0.0
    top1_sum = 0.0
    preds_list: list[torch.Tensor] = []

    for start in range(0, len(embeddings), batch_size):
        z = embeddings[start : start + batch_size].to(device)
        batch_y = labels[start : start + batch_size]
        if bank is not None and snapshot is not None:
            retrieved, weights = bank.retrieve(z, snapshot=snapshot)
            logits = head(retrieved)
            entropy_sum += float(attention_entropy(weights).sum().item())
            top1_sum += float(weights.max(dim=-1).values.sum().item())
        else:
            logits = head(z)
        preds = logits.argmax(dim=-1).cpu()
        correct += int((preds == batch_y).sum().item())
        total += len(batch_y)
        if return_predictions:
            preds_list.append(preds)

    out: dict[str, Any] = {
        "accuracy": correct / max(total, 1),
        "examples": total,
    }
    if bank is not None:
        out["entropy_mean"] = entropy_sum / max(total, 1)
        out["top1_mass_mean"] = top1_sum / max(total, 1)
    if return_predictions:
        out["predictions"] = torch.cat(preds_list) if preds_list else torch.empty(0, dtype=torch.long)
    return out


@torch.no_grad()
def evaluate_split(
    net: BackboneNet,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    bank: HybridOnlineBank | None = None,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate Direct CNN (bank=None) or pre-last-layer Hopfield retrieval."""
    z = encode(net.encoder, images, batch_size=batch_size, device=device)
    return evaluate_embeddings(
        net.head,
        z,
        labels,
        device=device,
        batch_size=batch_size,
        bank=bank,
        return_predictions=return_predictions,
    )


def paired_comparison(
    direct_preds: torch.Tensor,
    hop_preds: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    """Count helped, harmed, both-correct, and both-wrong vs Direct CNN."""
    direct_ok = direct_preds == labels
    hop_ok = hop_preds == labels
    helped = int((~direct_ok & hop_ok).sum().item())
    harmed = int((direct_ok & ~hop_ok).sum().item())
    both_correct = int((direct_ok & hop_ok).sum().item())
    both_wrong = int((~direct_ok & ~hop_ok).sum().item())
    return {
        "helped": helped,
        "harmed": harmed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_gain": helped - harmed,
    }


@torch.no_grad()
def update_banks_on_batch(
    net: BackboneNet,
    banks: dict[str, HybridOnlineBank],
    queries: torch.Tensor,
    labels: torch.Tensor,
    direct_preds: torch.Tensor,
    example_ids: torch.Tensor,
    *,
    epoch: int,
    step: int,
) -> dict[str, dict[str, int]]:
    """Apply accuracy-guided online Hopfield writes across all banks in lockstep."""
    was_training = net.head.training
    net.head.eval()
    batch_stats: dict[str, dict[str, int]] = {}
    try:
        direct_correct = direct_preds == labels
        for method, bank in banks.items():
            if not bank.center_initialized:
                batch_stats[method] = {"writes": 0, "eligible": 0}
                continue

            snapshot = bank.snapshot()
            if snapshot.ready:
                retrieved, _ = bank.retrieve(queries, snapshot=snapshot)
                hop_preds = net.head(retrieved).argmax(dim=-1)
                eligible_mask = direct_correct & (hop_preds != labels)
            else:
                eligible_mask = direct_correct

            eligible_indices = eligible_mask.nonzero(as_tuple=False).flatten().tolist()
            writes = 0
            for idx in eligible_indices:
                proposal = bank.plan_candidate(
                    queries[idx],
                    int(labels[idx].item()),
                    bootstrap=not snapshot.ready,
                )
                if proposal.is_write:
                    bank.commit(
                        proposal,
                        example_id=int(example_ids[idx].item()),
                        step=step,
                        epoch=epoch,
                    )
                    writes += 1
            batch_stats[method] = {"writes": writes, "eligible": len(eligible_indices)}
    finally:
        net.head.train(was_training)
    return batch_stats


def build_banks(
    spec: DatasetSpec,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, HybridOnlineBank]:
    """Instantiate one HybridOnlineBank per prototype update method."""
    update_cfg = cfg["update"]
    banks: dict[str, HybridOnlineBank] = {}
    for method in cfg["methods"]:
        banks[method] = HybridOnlineBank(
            cfg["embed_dim"],
            spec.num_classes,
            cfg["prototypes_per_class"],
            method=method,
            ema_alpha=update_cfg["alpha"],
            beta=cfg["hopfield_beta"],
            num_updates=cfg["hopfield_num_updates"],
            merge_distance_threshold=cfg["merge_distance_threshold"],
            normalization_epsilon=cfg["normalization_epsilon"],
            gamma=update_cfg["gamma"],
            process_noise=update_cfg["process_noise"],
            initial_uncertainty=update_cfg["initial_uncertainty"],
        ).to(device)
    return banks


@torch.no_grad()
def _ensure_banks_ready(
    net: BackboneNet,
    banks: dict[str, HybridOnlineBank],
    split: DatasetSplits,
    *,
    device: torch.device,
    batch_size: int,
    epoch: int,
) -> None:
    """Ensure all classes have at least one prototype if a tiny subset missed a class."""
    if all(bank.ready for bank in banks.values()):
        return
    net.eval()
    filled = 0
    for start in range(0, len(split.train_images), batch_size):
        batch_x = split.train_images[start : start + batch_size].to(device)
        batch_y = split.train_labels[start : start + batch_size].to(device)
        z = net.encoder(batch_x)
        preds = net.head(z).argmax(dim=-1)
        for bank in banks.values():
            if bank.ready:
                continue
            occ = bank.occupancy()
            for i in range(len(batch_y)):
                cls_id = int(batch_y[i].item())
                if occ[cls_id] == 0 and int(preds[i].item()) == cls_id:
                    prop = bank.plan_candidate(z[i], cls_id, bootstrap=True)
                    if prop.is_write:
                        bank.commit(prop, example_id=start + i, step=0, epoch=epoch)
                        occ[cls_id] += 1
                        filled += 1
        if all(bank.ready for bank in banks.values()):
            break
    if filled > 0:
        warnings.warn(
            f"_ensure_banks_ready filled {filled} missing prototype slot(s) "
            f"post-training — training loop may have insufficient class coverage.",
            stacklevel=2,
        )


def evaluate_constraints(
    train_acc: float,
    val_acc: float,
    test_acc: float,
    *,
    min_accuracy: float = 0.80,
    max_gap: float = 0.07,
) -> dict[str, Any]:
    """Verify minimum accuracy and train-validation generalization gap bounds."""
    gap = abs(train_acc - val_acc)
    min_ok = train_acc >= min_accuracy and test_acc >= min_accuracy
    gap_ok = gap <= max_gap
    return {
        "train_val_gap": gap,
        "min_accuracy_ok": min_ok,
        "generalization_gap_ok": gap_ok,
        "all_passed": min_ok and gap_ok,
    }


def train_and_evaluate_seed(
    spec: DatasetSpec,
    cfg: dict[str, Any],
    seed: int,
    data: tuple[torch.Tensor, ...],
    device: torch.device,
) -> dict[str, Any]:
    """Train one shared CNN per seed while updating all memory banks in lockstep."""
    t0 = time.perf_counter()
    split = make_dataset_splits(
        data,
        seed=seed,
        val_size=cfg["val_size"],
        train_subset=cfg["train_subset"],
        test_subset=cfg["test_subset"],
    )
    num_train = len(split.train_labels)
    train_ids = torch.arange(num_train, dtype=torch.long)

    net = build_backbone(
        spec,
        embed_dim=cfg["embed_dim"],
        seed=seed,
        rgb_channels=tuple(cfg.get("rgb_channels", (32, 64, 128))),
    ).to(device)
    banks = build_banks(spec, cfg, device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    epochs = cfg["epochs"]
    warmup_epochs = min(cfg["warmup_epochs"], max(1, epochs - 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    shuffle_gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator().manual_seed(seed + 10_000)

    history: list[dict[str, Any]] = []
    seen = 0
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(num_train, generator=shuffle_gen)
        direct_correct = 0
        epoch_seen = 0
        loss_sum = 0.0
        write_totals = {m: 0 for m in banks}

        for start in range(0, num_train, cfg["batch_size"]):
            idx = order[start : start + cfg["batch_size"]]
            batch_x = split.train_images[idx].to(device)
            batch_y = split.train_labels[idx].to(device)
            if cfg["augment"]:
                batch_x = augment_batch(
                    batch_x,
                    random_crop=True,
                    horizontal_flip=spec.horizontal_flip,
                    padding=spec.crop_padding,
                    generator=aug_gen,
                )

            optimizer.zero_grad(set_to_none=True)
            z = net.encoder(batch_x)
            direct_logits = net.head(z)
            loss = F.cross_entropy(direct_logits, batch_y)
            if not torch.isfinite(loss) or not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite training loss or encoder embedding.")

            direct_preds = direct_logits.argmax(dim=-1).detach()
            batch_writes = update_banks_on_batch(
                net,
                banks,
                z.detach(),
                batch_y,
                direct_preds,
                train_ids[idx],
                epoch=epoch,
                step=seen,
            )
            for m, stats in batch_writes.items():
                write_totals[m] += stats["writes"]

            loss.backward()
            optimizer.step()

            batch_count = len(batch_y)
            direct_correct += int((direct_preds == batch_y).sum().item())
            epoch_seen += batch_count
            seen += batch_count
            loss_sum += float(loss.detach().item()) * batch_count

        scheduler.step()
        epoch_acc = direct_correct / max(epoch_seen, 1)
        for bank in banks.values():
            bank.current_accuracy = epoch_acc

        if epoch == warmup_epochs:
            center = encode(
                net.encoder,
                split.train_images,
                batch_size=cfg["eval_batch_size"],
                device=device,
            ).mean(dim=0)
            for bank in banks.values():
                bank.initialize_center(center.to(device))

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "direct_train_accuracy": epoch_acc,
            "mean_loss": loss_sum / max(epoch_seen, 1),
            "lr": float(scheduler.get_last_lr()[0]),
            "active_memories": {m: int(b.active.sum().item()) for m, b in banks.items()},
            "writes": write_totals,
        }
        history.append(epoch_record)
        print(
            f"[{spec.name} | seed {seed}] epoch {epoch}/{epochs} "
            f"loss={epoch_record['mean_loss']:.4f} direct_acc={epoch_acc:.4f} "
            f"mem_active={epoch_record['active_memories']}",
            flush=True,
        )

    _ensure_banks_ready(
        net,
        banks,
        split,
        device=device,
        batch_size=cfg["eval_batch_size"],
        epoch=epochs,
    )

    # Encode clean splits once and evaluate all methods on the same embeddings.
    z_train = encode(net.encoder, split.train_images, batch_size=cfg["eval_batch_size"], device=device)
    z_val = encode(net.encoder, split.val_images, batch_size=cfg["eval_batch_size"], device=device)
    z_test = encode(net.encoder, split.test_images, batch_size=cfg["eval_batch_size"], device=device)

    direct_train = evaluate_embeddings(
        net.head, z_train, split.train_labels,
        device=device, batch_size=cfg["eval_batch_size"],
    )
    direct_val = evaluate_embeddings(
        net.head, z_val, split.val_labels,
        device=device, batch_size=cfg["eval_batch_size"],
    )
    direct_test = evaluate_embeddings(
        net.head, z_test, split.test_labels,
        device=device, batch_size=cfg["eval_batch_size"], return_predictions=True,
    )

    methods_out: dict[str, Any] = {
        "direct": {
            "train_accuracy": direct_train["accuracy"],
            "validation_accuracy": direct_val["accuracy"],
            "test_accuracy": direct_test["accuracy"],
            "constraints": evaluate_constraints(
                direct_train["accuracy"],
                direct_val["accuracy"],
                direct_test["accuracy"],
                min_accuracy=cfg["min_accuracy"],
                max_gap=cfg["max_train_val_gap"],
            ),
            "corruptions": {},
            "robust_auc": {},
        }
    }

    for method, bank in banks.items():
        m_train = evaluate_embeddings(
            net.head, z_train, split.train_labels,
            device=device, batch_size=cfg["eval_batch_size"], bank=bank,
        )
        m_val = evaluate_embeddings(
            net.head, z_val, split.val_labels,
            device=device, batch_size=cfg["eval_batch_size"], bank=bank,
        )
        m_test = evaluate_embeddings(
            net.head, z_test, split.test_labels,
            device=device, batch_size=cfg["eval_batch_size"], bank=bank, return_predictions=True,
        )
        methods_out[method] = {
            "train_accuracy": m_train["accuracy"],
            "validation_accuracy": m_val["accuracy"],
            "test_accuracy": m_test["accuracy"],
            "active_memories": int(bank.active.sum().item()),
            "occupancy": bank.occupancy(),
            "val_entropy_mean": m_val["entropy_mean"],
            "val_top1_mass_mean": m_val["top1_mass_mean"],
            "test_entropy_mean": m_test["entropy_mean"],
            "test_top1_mass_mean": m_test["top1_mass_mean"],
            "test_paired": paired_comparison(
                direct_test["predictions"], m_test["predictions"], split.test_labels
            ),
            "constraints": evaluate_constraints(
                m_train["accuracy"],
                m_val["accuracy"],
                m_test["accuracy"],
                min_accuracy=cfg["min_accuracy"],
                max_gap=cfg["max_train_val_gap"],
            ),
            "corruptions": {},
            "robust_auc": {},
        }

    # Evaluate corruptions on the test set once per (kind, level) across direct + all memory methods.
    for kind, levels in cfg["corruptions"].items():
        methods_out["direct"]["corruptions"][kind] = {}
        for method in banks:
            methods_out[method]["corruptions"][kind] = {}

        for level in levels:
            corrupted_x = make_corrupted_test_set(
                split.test_images, seed=seed, kind=kind, level=level
            )
            z_corr = encode(
                net.encoder, corrupted_x, batch_size=cfg["eval_batch_size"], device=device
            )
            d_res = evaluate_embeddings(
                net.head, z_corr, split.test_labels,
                device=device, batch_size=cfg["eval_batch_size"], return_predictions=True,
            )
            methods_out["direct"]["corruptions"][kind][str(level)] = d_res["accuracy"]
            for method, bank in banks.items():
                m_res = evaluate_embeddings(
                    net.head, z_corr, split.test_labels,
                    device=device, batch_size=cfg["eval_batch_size"], bank=bank, return_predictions=True,
                )
                methods_out[method]["corruptions"][kind][str(level)] = m_res["accuracy"]

        for model_key in ("direct", *METHODS):
            if model_key not in methods_out:
                continue
            accs = [methods_out[model_key]["corruptions"][kind][str(lv)] for lv in levels]
            methods_out[model_key]["robust_auc"][kind] = (
                robust_auc(levels, accs) if len(levels) >= 2 else accs[0]
            )

    elapsed = time.perf_counter() - t0
    return {
        "dataset": spec.name,
        "seed": seed,
        "elapsed_sec": round(elapsed, 2),
        "epochs": history,
        "methods": methods_out,
    }
