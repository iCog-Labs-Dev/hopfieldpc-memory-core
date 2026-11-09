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
from hopfieldpc.api import validate_query_memory
from online_memory import OnlineHopfieldBank, validate_normalization_epsilon
from mnist_online_memory import train_step as online_train_step


METHODS = ("ema", "hybrid", "momentum_only", "kalman_only")


@torch.no_grad()
def masked_centered_retrieve(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 5.0,
    num_updates: int = 1,
    normalization_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve raw prototypes using masked softmax over centered cosine scores.

    Implements Equation (4) from Prototype Selection.pdf:
        w_j = softmax(beta * q_tilde @ M_tilde_j^T * m_j + (-inf) * (1 - m_j))
    """
    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must have shape [num_memories].")
    active_mask = mask.bool()
    if not active_mask.any():
        raise ValueError("At least one prototype must be active for retrieval.")

    q = query - center
    m = values - center
    if not torch.isfinite(q).all() or not torch.isfinite(m[active_mask]).all():
        raise FloatingPointError("Nonfinite centered query or active memory.")

    views = torch.zeros_like(m)
    views[active_mask] = F.normalize(m[active_mask], dim=-1, eps=eps)
    mask_float = active_mask.to(query.dtype)

    raw = query
    weights = torch.zeros(query.shape[0], values.shape[0], dtype=query.dtype, device=query.device)
    for _ in range(num_updates):
        query_view = F.normalize(raw - center, dim=-1, eps=eps)
        scores = beta * (query_view @ views.T) * mask_float
        scores = scores.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        raw = weights @ values
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite masked retrieval output.")
    return raw, weights


class DynamicHybridOnlineBank(OnlineHopfieldBank):
    """Online Hopfield bank with masked buffers, uncertainty-driven spawning, and support pruning."""

    def __init__(
        self,
        dim: int,
        num_classes: int = 10,
        max_memories_per_class: int = 16,
        *,
        initial_memories_per_class: int = 4,
        method: str = "hybrid",
        ema_alpha: float = 0.1,
        beta: float = 5.0,
        num_updates: int = 1,
        merge_distance_threshold: float = 0.2,
        normalization_epsilon: float = 1e-8,
        gamma: float = 0.9,
        process_noise: float = 0.01,
        initial_uncertainty: float = 1.0,
        tau_support: int = 10,
        prune_patience: int = 2,
        tau_uncert: float = 0.5,
        split_scale: float = 0.1,
    ) -> None:
        super().__init__(
            dim,
            num_classes,
            max_memories_per_class,
            ema_alpha=ema_alpha,
            beta=beta,
            num_updates=num_updates,
            merge_distance_threshold=merge_distance_threshold,
            normalization_epsilon=normalization_epsilon,
        )
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}.")
        if not 1 <= initial_memories_per_class <= max_memories_per_class:
            raise ValueError("Require 1 <= initial_memories_per_class <= max_memories_per_class.")
        if tau_support < 0 or prune_patience < 1:
            raise ValueError("tau_support must be >= 0 and prune_patience >= 1.")
        if not math.isfinite(tau_uncert) or tau_uncert <= 0:
            raise ValueError("tau_uncert must be positive and finite.")
        if not math.isfinite(split_scale) or split_scale <= 0:
            raise ValueError("split_scale must be positive and finite.")

        self.update_method = method
        self.initial_capacity = int(initial_memories_per_class)
        self.max_capacity = int(max_memories_per_class)
        # Online admission uses initial_capacity per class; dynamic spawning can grow up to max_capacity.
        self.capacity = self.initial_capacity
        self.gamma = float(gamma)
        self.process_noise = float(process_noise)
        self.initial_uncertainty = float(initial_uncertainty)
        self.tau_support = int(tau_support)
        self.prune_patience = int(prune_patience)
        self.tau_uncert = float(tau_uncert)
        self.split_scale = float(split_scale)

        n = num_classes * max_memories_per_class
        self.register_buffer("mask", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("velocity", torch.zeros_like(self.values))
        self.register_buffer("uncertainty", torch.full((n,), self.initial_uncertainty))
        self.register_buffer("low_support_streak", torch.zeros(n, dtype=torch.long))
        self.register_buffer("last_support", torch.zeros(n, dtype=torch.long))
        self.current_accuracy = 1.0
        self.total_pruned = 0
        self.total_spawned = 0

    @property
    def active(self) -> torch.Tensor:
        return self.mask & (self.entry_ids >= 0)

    def _allocate_slot(self, slot: int, label: int, value: torch.Tensor, *, step: int, epoch: int, source_id: int) -> None:
        entry_id = len(self.metadata)
        self.values[slot].copy_(value)
        self.labels[slot] = label
        self.entry_ids[slot] = entry_id
        self.mask[slot] = True
        self.velocity[slot].zero_()
        self.uncertainty[slot] = self.initial_uncertainty
        self.low_support_streak[slot] = 0
        self.last_support[slot] = 0
        self.metadata.append(
            dict(
                entry_id=entry_id,
                label=label,
                created_step=step,
                created_epoch=epoch,
                updated_step=step,
                updated_epoch=epoch,
                first_source_id=source_id,
                latest_source_id=source_id,
                accepted_exposure_count=1,
                bank_version=self.version + 1,
            )
        )
        self.next_id = len(self.metadata)

    @torch.no_grad()
    def retrieve_masked(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.ready:
            raise ValueError("Bank incomplete; caller must explicitly mark direct fallback.")
        return masked_centered_retrieve(
            z,
            self.values,
            self.center,
            self.active,
            beta=self.beta,
            num_updates=self.num_updates,
            normalization_epsilon=self.eps,
        )

    @torch.no_grad()
    def plan_candidate(self, z: torch.Tensor, label: int, *, bootstrap: bool = False):
        proposal = super().plan_candidate(z, label, bootstrap=bootstrap)
        if proposal.action != "merge" or proposal.slot is None:
            return proposal
        slot = proposal.slot
        candidate = proposal.candidate
        noise = (1.0 - self.current_accuracy) + self.eps
        gain = self.uncertainty[slot] / (self.uncertainty[slot] + noise)
        delta = candidate - self.values[slot]
        if self.update_method in ("hybrid", "momentum_only"):
            self.velocity[slot].mul_(self.gamma).add_(delta, alpha=1.0 - self.gamma)
            step = self.velocity[slot]
        else:
            step = delta
        applied_gain = 1.0 if self.update_method == "momentum_only" else gain
        if self.update_method == "ema":
            new_value = (1.0 - self.alpha) * self.values[slot] + self.alpha * candidate
        else:
            new_value = self.values[slot] + applied_gain * step

        return proposal.__class__(
            proposal.action,
            proposal.reason,
            proposal.version,
            proposal.slot,
            proposal.label,
            proposal.candidate,
            new_value,
            proposal.distance,
        )

    @torch.no_grad()
    def commit(self, proposal, *, example_id: int, step: int, epoch: int) -> dict[str, Any]:
        result = super().commit(proposal, example_id=example_id, step=step, epoch=epoch)
        if proposal.slot is not None:
            self.mask[proposal.slot] = True
            if proposal.action in ("seed", "add"):
                self.velocity[proposal.slot].zero_()
                self.uncertainty[proposal.slot] = self.initial_uncertainty
                self.low_support_streak[proposal.slot] = 0
        return result

    @staticmethod
    def _principal_direction(members: torch.Tensor, fallback_dim: int, eps: float) -> torch.Tensor:
        """Compute first principal component direction of assigned embeddings S_{j,n}."""
        if members.shape[0] >= 2:
            centered = members - members.mean(dim=0, keepdim=True)
            cov_scale = centered.norm(dim=-1).mean()
            if torch.isfinite(cov_scale) and cov_scale > eps:
                try:
                    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
                    direction = vh[0]
                    scale = s[0] / math.sqrt(max(members.shape[0] - 1, 1))
                    if torch.isfinite(direction).all() and scale > eps:
                        return F.normalize(direction, dim=-1, eps=eps) * scale
                except RuntimeError:
                    pass
        unit = torch.zeros(fallback_dim, dtype=members.dtype, device=members.device)
        unit[0] = 1.0
        return unit

    @torch.no_grad()
    def adapt_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        *,
        epoch: int,
        step: int,
        allow_spawn: bool = True,
    ) -> dict[str, Any]:
        """Epoch-end assignment support pruning and uncertainty-driven spawning."""
        if not self.ready:
            return {
                "pruned": 0,
                "spawned": 0,
                "active_total": int(self.active.sum().item()),
                "per_class_active": self.occupancy(),
                "mean_uncertainty": 0.0,
            }

        device = self.values.device
        features = features.detach().to(device)
        labels = labels.detach().to(device)
        predictions = predictions.detach().to(device)

        pruned_this_epoch = 0
        spawned_this_epoch = 0

        for class_id in range(self.num_classes):
            class_mask = labels == class_id
            class_features = features[class_mask]
            class_preds = predictions[class_mask]
            active_slots = ((self.labels == class_id) & self.active).nonzero().flatten()
            if active_slots.numel() == 0 or class_features.numel() == 0:
                continue

            # Assign class embeddings to nearest active prototype in centered-cosine space
            proto_keys = F.normalize(self.values[active_slots] - self.center, dim=-1, eps=self.eps)
            feat_keys = F.normalize(class_features - self.center, dim=-1, eps=self.eps)
            distances = (1.0 - feat_keys @ proto_keys.T).clamp(0.0, 2.0)
            nearest_idx = distances.argmin(dim=1)

            # Track support |S_{j,n}| and update cluster-level uncertainty P_{j,n+1}
            slot_members: dict[int, torch.Tensor] = {}
            for local_i, slot_tensor in enumerate(active_slots):
                slot = int(slot_tensor.item())
                assigned_mask = nearest_idx == local_i
                support = int(assigned_mask.sum().item())
                self.last_support[slot] = support
                members = class_features[assigned_mask]
                slot_members[slot] = members

                if support < self.tau_support:
                    self.low_support_streak[slot] += 1
                else:
                    self.low_support_streak[slot] = 0

                if support > 0:
                    local_acc = float((class_preds[assigned_mask] == class_id).float().mean().item())
                    local_disp = float(distances[assigned_mask, local_i].std().item()) if support > 1 else 0.0
                    # Equation (2) & (3): R_{j,n} reflects local inaccuracy and intra-cluster spread
                    r_j = (1.0 - local_acc) + 4.0 * local_disp + self.eps
                    k_j = self.uncertainty[slot] / (self.uncertainty[slot] + r_j)
                    self.uncertainty[slot] = (1.0 - k_j) * self.uncertainty[slot] + self.process_noise + local_disp

            # 1. Pruning via Assignment Support (|S_{j,n}| < tau_support for T epochs)
            # Keep at least 1 active prototype per class so retrieval remains valid.
            prunable = [
                int(s.item())
                for s in active_slots
                if int(self.low_support_streak[s].item()) >= self.prune_patience
            ]
            # Sort by lowest support first
            prunable.sort(key=lambda s: int(self.last_support[s].item()))
            remaining_active = int(active_slots.numel())
            for slot in prunable:
                if remaining_active <= 1:
                    break
                self.mask[slot] = False
                self.labels[slot] = -1
                self.entry_ids[slot] = -1
                self.velocity[slot].zero_()
                self.uncertainty[slot] = self.initial_uncertainty
                self.low_support_streak[slot] = 0
                self.last_support[slot] = 0
                remaining_active -= 1
                pruned_this_epoch += 1
                self.total_pruned += 1
                self.version += 1

            # 2. Spawning via Uncertainty (P_{j,n+1} > tau_uncert => split along PC1)
            if not allow_spawn:
                continue
            surviving_slots = ((self.labels == class_id) & self.active).nonzero().flatten()
            # Sort candidate parents by highest uncertainty first
            spawn_candidates = sorted(
                [int(s.item()) for s in surviving_slots],
                key=lambda s: float(self.uncertainty[s].item()),
                reverse=True,
            )
            current_class_count = int(surviving_slots.numel())
            for slot in spawn_candidates:
                if current_class_count >= self.max_capacity:
                    break
                free_slots = (~self.active).nonzero().flatten()
                if free_slots.numel() == 0:
                    break
                if float(self.uncertainty[slot].item()) <= self.tau_uncert:
                    continue
                members = slot_members.get(slot)
                if members is None or members.shape[0] < 2:
                    continue

                pc_step = self._principal_direction(members, self.dim, self.eps)
                delta = self.split_scale * pc_step
                parent_val = self.values[slot].clone()
                child_a = parent_val + delta
                child_b = parent_val - delta
                if self._invalid_key(child_a) or self._invalid_key(child_b):
                    continue

                new_slot = int(free_slots[0].item())
                # Update parent slot as M_j^(a) and reset V=0, P=P_0
                self.values[slot].copy_(child_a)
                self.velocity[slot].zero_()
                self.uncertainty[slot] = self.initial_uncertainty
                self.low_support_streak[slot] = 0

                # Populate inactive slot as M_j^(b) with V=0, P=P_0
                self._allocate_slot(
                    new_slot,
                    class_id,
                    child_b,
                    step=step,
                    epoch=epoch,
                    source_id=-1,
                )
                current_class_count += 1
                spawned_this_epoch += 1
                self.total_spawned += 1
                self.version += 1

        active_uncert = self.uncertainty[self.active]
        return {
            "pruned": pruned_this_epoch,
            "spawned": spawned_this_epoch,
            "active_total": int(self.active.sum().item()),
            "per_class_active": self.occupancy(),
            "mean_uncertainty": float(active_uncert.mean().item()) if active_uncert.numel() else 0.0,
        }


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
