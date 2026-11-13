"""Dynamic prototype spawning and pruning for online Hopfield memory banks.

Implements Experiment 3:
- Masked fixed-size PyTorch buffers up to K_max prototypes per class (initial online cap K_init).
- Pruning via assignment support: deactivate prototypes with |S_{j,n}| < tau_support
  for T_prune consecutive epochs (preserving at least one prototype per class).
- Spawning via uncertainty: split prototypes with P_{j,n+1} > tau_uncert along the
  first principal component of their assigned embeddings S_{j,n}, resetting V=0 and P=P_0.
- Masked centered-cosine Hopfield retrieval returning raw prototype combinations.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .api import CandidateProposal, validate_update_method
from .ema import ema_step
from .functional import masked_centered_retrieve, principal_split_direction
from .kalman import kalman_gain
from .online import OnlineHopfieldBank


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
        validate_update_method(method)
        if not 1 <= initial_memories_per_class <= max_memories_per_class:
            raise ValueError(
                "Require 1 <= initial_memories_per_class <= max_memories_per_class."
            )
        if tau_support < 0 or prune_patience < 1:
            raise ValueError(
                "tau_support must be >= 0 and prune_patience >= 1."
            )
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
        self.register_buffer(
            "uncertainty", torch.full((n,), self.initial_uncertainty)
        )
        self.register_buffer(
            "low_support_streak", torch.zeros(n, dtype=torch.long)
        )
        self.register_buffer("last_support", torch.zeros(n, dtype=torch.long))
        self.current_accuracy = 1.0
        self.total_pruned = 0
        self.total_spawned = 0

    @property
    def active(self) -> torch.Tensor:
        return self.mask & (self.entry_ids >= 0)

    def _allocate_slot(
        self,
        slot: int,
        label: int,
        value: torch.Tensor,
        *,
        step: int,
        epoch: int,
        source_id: int,
    ) -> None:
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
    def retrieve_masked(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.ready:
            raise ValueError(
                "Bank incomplete; caller must explicitly mark direct fallback."
            )
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
    def plan_candidate(
        self,
        z: torch.Tensor,
        label: int,
        *,
        bootstrap: bool = False,
    ) -> CandidateProposal:
        proposal = super().plan_candidate(z, label, bootstrap=bootstrap)
        if proposal.action != "merge" or proposal.slot is None:
            return proposal
        slot = proposal.slot
        candidate = proposal.candidate
        gain = kalman_gain(
            self.uncertainty[slot],
            accuracy=self.current_accuracy,
            epsilon=self.eps,
        )
        delta = candidate - self.values[slot]
        if self.update_method in ("hybrid", "momentum_only"):
            self.velocity[slot].mul_(self.gamma).add_(
                delta, alpha=1.0 - self.gamma
            )
            step = self.velocity[slot]
        else:
            step = delta
        applied_gain = 1.0 if self.update_method == "momentum_only" else gain
        if self.update_method == "ema":
            new_value = ema_step(self.values[slot], candidate, alpha=self.alpha)
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
    def commit(
        self,
        proposal: CandidateProposal,
        *,
        example_id: int,
        step: int,
        epoch: int,
    ) -> dict[str, Any]:
        result = super().commit(
            proposal, example_id=example_id, step=step, epoch=epoch
        )
        if proposal.slot is not None:
            self.mask[proposal.slot] = True
            if proposal.action in ("seed", "add"):
                self.velocity[proposal.slot].zero_()
                self.uncertainty[proposal.slot] = self.initial_uncertainty
                self.low_support_streak[proposal.slot] = 0
        return result

    @staticmethod
    def _principal_direction(
        members: torch.Tensor, fallback_dim: int, eps: float
    ) -> torch.Tensor:
        return principal_split_direction(members, fallback_dim, eps=eps)

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
            active_slots = (
                ((self.labels == class_id) & self.active).nonzero().flatten()
            )
            if active_slots.numel() == 0 or class_features.numel() == 0:
                continue

            # Assign class embeddings to nearest active prototype in centered-cosine space
            proto_keys = F.normalize(
                self.values[active_slots] - self.center, dim=-1, eps=self.eps
            )
            feat_keys = F.normalize(
                class_features - self.center, dim=-1, eps=self.eps
            )
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
                    local_acc = float(
                        (class_preds[assigned_mask] == class_id)
                        .float()
                        .mean()
                        .item()
                    )
                    local_disp = (
                        float(distances[assigned_mask, local_i].std().item())
                        if support > 1
                        else 0.0
                    )
                    # Equation (2) & (3): R_{j,n} reflects local inaccuracy and intra-cluster spread
                    r_j = (1.0 - local_acc) + 4.0 * local_disp + self.eps
                    k_j = self.uncertainty[slot] / (
                        self.uncertainty[slot] + r_j
                    )
                    self.uncertainty[slot] = (
                        (1.0 - k_j) * self.uncertainty[slot]
                        + self.process_noise
                        + local_disp
                    )

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
            surviving_slots = (
                ((self.labels == class_id) & self.active).nonzero().flatten()
            )
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
            "mean_uncertainty": float(active_uncert.mean().item())
            if active_uncert.numel()
            else 0.0,
        }


class DynamicEMAOnlineBank(DynamicHybridOnlineBank):
    """Dynamic online Hopfield bank specialized to EMA merges."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("method", "ema")
        super().__init__(*args, **kwargs)


class DynamicMomentumOnlineBank(DynamicHybridOnlineBank):
    """Dynamic online Hopfield bank specialized to Momentum-only merges."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("method", "momentum_only")
        super().__init__(*args, **kwargs)


class DynamicKalmanOnlineBank(DynamicHybridOnlineBank):
    """Dynamic online Hopfield bank specialized to Kalman-only merges."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("method", "kalman_only")
        super().__init__(*args, **kwargs)
