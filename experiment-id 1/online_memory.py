"""Explicit, accuracy-gated memory buffers; never optimizer-trained parameters.

The caller freezes batch eligibility before calling ``plan_candidate`` / ``commit``.
Retrieval has no label argument. Labels constrain writes only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numbers
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from hopfieldpc.api import validate_beta, validate_query_memory
from hopfieldpc.functional import hopfield_retrieve


def validate_normalization_epsilon(value: object) -> float:
    """Experiment-local epsilon validation; the library API is unchanged."""
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("normalization_epsilon must be positive and finite.")
    return float(value)


def positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class BankSnapshot:
    values: torch.Tensor
    labels: torch.Tensor
    entry_ids: torch.Tensor
    center: torch.Tensor
    version: int
    ready: bool


@dataclass(frozen=True)
class CandidateProposal:
    action: str
    reason: str
    version: int
    slot: int | None
    label: int
    candidate: torch.Tensor
    new_value: torch.Tensor | None
    distance: float | None

    @property
    def is_write(self) -> bool:
        return self.action in ("seed", "add", "merge")


def correctness_categories(direct: torch.Tensor, hopfield: torch.Tensor,
                           targets: torch.Tensor) -> list[str]:
    """Per-example correctness only; arguments are predictions, never confidences."""
    if direct.shape != targets.shape or hopfield.shape != targets.shape or targets.ndim != 1:
        raise ValueError("Predictions and targets must have matching [batch] shapes.")
    return ["harmed" if d and not h else "helped" if h and not d
            else "both_correct" if d else "both_wrong"
            for d, h in zip((direct == targets).tolist(), (hopfield == targets).tolist())]


@torch.no_grad()
def centered_retrieve(query: torch.Tensor, values: torch.Tensor, center: torch.Tensor,
                      *, beta: float = 5.0, num_updates: int = 1,
                      normalization_epsilon: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Score centered directions and return a convex combination of RAW values."""
    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if center.shape != (query.shape[1],):
        raise ValueError("center must have shape [dim].")
    q, m = query - center, values - center
    if not torch.isfinite(q).all() or not torch.isfinite(m).all():
        raise FloatingPointError("Nonfinite centered query or memory.")
    qnorm, mnorm = q.norm(dim=-1), m.norm(dim=-1)
    if (not torch.isfinite(qnorm).all() or not torch.isfinite(mnorm).all()
            or (mnorm <= eps).any()):
        raise FloatingPointError("Invalid centered query norm or degenerate memory key.")
    positive_int("num_updates", num_updates)
    # Reuse the UNMODIFIED library for similarities/softmax. Its score inputs
    # are already normalized with the experiment's epsilon; use the returned
    # weights on raw values, rather than treating score views as memory values.
    # The core also computes a score-view readout which is intentionally unused.
    views = F.normalize(m, dim=-1, eps=eps)
    raw = query
    for _ in range(num_updates):
        query_view = F.normalize(raw - center, dim=-1, eps=eps)
        _, weights = hopfield_retrieve(query_view, views, beta=beta, normalize=False)
        raw = weights @ values
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite retrieval output.")
    return raw, weights


class OnlineHopfieldBank(nn.Module):
    """Bounded raw-vector bank with stable IDs, EMA writes and frozen center.

    Buffers are preallocated as zeros, but inactive slots never participate in
    retrieval. ``ready`` describes coverage; the caller snapshots it once per
    batch, so bootstrap cannot activate comparisons halfway through a batch.
    """

    def __init__(self, dim: int, num_classes: int = 10, max_memories_per_class: int = 8,
                 *, ema_alpha: float = 0.1, beta: float = 5.0, num_updates: int = 1,
                 merge_distance_threshold: float = 0.2, normalization_epsilon: float = 1e-8):
        super().__init__()
        self.dim = positive_int("dim", dim)
        self.num_classes = positive_int("num_classes", num_classes)
        self.capacity = positive_int("max_memories_per_class", max_memories_per_class)
        self.beta = validate_beta(beta)
        self.num_updates = positive_int("num_updates", num_updates)
        self.eps = validate_normalization_epsilon(normalization_epsilon)
        self.alpha = validate_beta(ema_alpha)
        if self.alpha > 1:
            raise ValueError("ema_alpha must be <= 1.")
        if (isinstance(merge_distance_threshold, bool)
                or not isinstance(merge_distance_threshold, (int, float))
                or not 0 <= merge_distance_threshold <= 2):
            raise ValueError("merge_distance_threshold must be in [0, 2].")
        self.threshold = float(merge_distance_threshold)
        n = num_classes * self.capacity
        self.register_buffer("values", torch.zeros(n, dim))
        self.register_buffer("labels", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("entry_ids", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("center", torch.zeros(dim))
        self.metadata: list[dict[str, int]] = []
        self.center_initialized = False
        self.version = 0
        self.next_id = 0

    @property
    def active(self) -> torch.Tensor:
        return self.entry_ids >= 0

    @property
    def ready(self) -> bool:
        return self.center_initialized and all(n > 0 for n in self.occupancy())

    def occupancy(self) -> list[int]:
        return [(self.labels == c).sum().item() for c in range(self.num_classes)]

    @torch.no_grad()
    def initialize_center(self, center: torch.Tensor) -> None:
        if self.center_initialized:
            raise ValueError("The center is frozen and cannot be reinitialized.")
        if center.shape != (self.dim,) or not torch.isfinite(center).all():
            raise ValueError("Center must be a finite [dim] vector.")
        self.center.copy_(center.detach().to(self.center))
        self.center_initialized = True

    def snapshot(self) -> BankSnapshot:
        mask = self.active
        return BankSnapshot(self.values[mask].detach().clone(), self.labels[mask].clone(),
                            self.entry_ids[mask].clone(), self.center.detach().clone(),
                            self.version, self.ready)

    @torch.no_grad()
    def retrieve(self, z: torch.Tensor, *, snapshot: BankSnapshot | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        state = snapshot if snapshot is not None else self.snapshot()
        if not state.ready:
            raise ValueError("Bank incomplete; caller must explicitly mark direct fallback.")
        return centered_retrieve(z, state.values, state.center, beta=self.beta,
                                 num_updates=self.num_updates, normalization_epsilon=self.eps)

    def _invalid_key(self, value: torch.Tensor) -> str | None:
        key = value - self.center
        if not torch.isfinite(key).all() or not torch.isfinite(key.norm()):
            return "nonfinite_candidate"
        return "degenerate_centered_key" if key.norm().item() <= self.eps else None

    @torch.no_grad()
    def plan_candidate(self, z: torch.Tensor, label: int, *, bootstrap: bool = False
                       ) -> CandidateProposal:
        if z.shape != (self.dim,) or not z.is_floating_point():
            raise ValueError("Candidate must be a floating [dim] tensor.")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < self.num_classes:
            raise ValueError("Invalid candidate class.")
        candidate = z.detach().to(self.values).clone()

        def proposal(action: str, reason: str, slot: int | None = None,
                     new: torch.Tensor | None = None, distance: float | None = None) -> CandidateProposal:
            return CandidateProposal(action, reason, self.version, slot, label, candidate, new, distance)

        if not self.center_initialized:
            return proposal("skip", "warmup")
        invalid = self._invalid_key(candidate)
        if invalid:
            return proposal("skip", invalid)
        slots = (self.labels == label).nonzero().flatten()
        if bootstrap:
            if slots.numel():
                return proposal("skip", "class_already_seeded")
            slot = int((~self.active).nonzero()[0].item())
            return proposal("seed", "bootstrap_seed", slot, candidate.clone())
        if not self.ready:
            return proposal("skip", "bank_incomplete")
        # IDs and slots are insertion-ordered and never reused, so argmin's
        # first occurrence breaks exact distance ties by stable entry ID.
        keys = F.normalize(self.values[slots] - self.center, dim=-1, eps=self.eps)
        key = F.normalize(candidate - self.center, dim=-1, eps=self.eps)
        distances = (1 - keys @ key).clamp(0, 2)
        nearest = int(distances.argmin().item())
        slot, distance = int(slots[nearest].item()), float(distances[nearest].item())
        if distance <= self.threshold:
            new = (1 - self.alpha) * self.values[slot] + self.alpha * candidate
            invalid = self._invalid_key(new)
            return (proposal("skip", "invalid_proposed_update", distance=distance) if invalid
                    else proposal("merge", "within_threshold", slot, new.clone(), distance))
        if slots.numel() >= self.capacity:
            return proposal("skip", "class_capacity_reached", distance=distance)
        slot = int((~self.active).nonzero()[0].item())
        return proposal("add", "beyond_threshold", slot, candidate.clone(), distance)

    @torch.no_grad()
    def commit(self, proposal: CandidateProposal, *, example_id: int, step: int,
               epoch: int) -> dict[str, Any]:
        if not proposal.is_write or proposal.slot is None or proposal.new_value is None:
            raise ValueError("Cannot commit a skipped proposal.")
        if proposal.version != self.version:
            raise ValueError("Stale proposal: recompute assignment against the working bank.")
        if proposal.new_value.shape != (self.dim,) or proposal.candidate.shape != (self.dim,):
            raise ValueError("Invalid proposal shape.")
        if self._invalid_key(proposal.new_value):
            raise ValueError("Invalid proposed key.")
        slot = proposal.slot
        if not 0 <= slot < len(self.values):
            raise ValueError("Invalid proposal slot.")
        if proposal.action == "merge":
            if not self.active[slot] or int(self.labels[slot]) != proposal.label:
                raise ValueError("Merge must address an active same-class entry.")
        elif self.active[slot] or self.occupancy()[proposal.label] >= self.capacity:
            raise ValueError("Addition would overwrite an entry or exceed class capacity.")
        old = self.values[slot].clone() if proposal.action == "merge" else None
        size_before = len(self.metadata)
        if proposal.action != "merge":
            self.labels[slot] = proposal.label
            self.entry_ids[slot] = self.next_id
            self.metadata.append(dict(entry_id=self.next_id, label=proposal.label,
                                      created_step=step, created_epoch=epoch,
                                      first_source_id=example_id, accepted_exposure_count=0))
            self.next_id += 1
        entry = self.metadata[int(self.entry_ids[slot].item())]
        entry.update(updated_step=step, updated_epoch=epoch, latest_source_id=example_id,
                     accepted_exposure_count=entry["accepted_exposure_count"] + 1,
                     bank_version=self.version + 1)
        self.values[slot].copy_(proposal.new_value)
        self.version += 1
        return dict(action=proposal.action, reason=proposal.reason, entry_id=entry["entry_id"],
                    label=proposal.label, version_before=self.version - 1, version_after=self.version,
                    size_before=size_before, size_after=len(self.metadata), distance=proposal.distance,
                    alpha=self.alpha, threshold=self.threshold, metadata=dict(entry),
                    candidate=proposal.candidate.cpu().clone(), old=old.cpu() if old is not None else None,
                    new=self.values[slot].cpu().clone())

    def get_extra_state(self) -> dict[str, Any]:
        return dict(metadata=[dict(m) for m in self.metadata], version=self.version,
                    next_id=self.next_id, center_initialized=self.center_initialized)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.metadata = [dict(m) for m in state["metadata"]]
        self.version, self.next_id = state["version"], state["next_id"]
        self.center_initialized = state["center_initialized"]

    def storage(self, step: int) -> dict[str, Any]:
        tensor_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        n = len(self.metadata)
        occupancy = self.occupancy()
        return dict(occupancy=occupancy, ready=self.ready, version=self.version,
                    covered_classes=sum(count > 0 for count in occupancy),
                    class_coverage_fraction=sum(count > 0 for count in occupancy) / self.num_classes,
                    saturated_classes=sum(count == self.capacity for count in occupancy),
                    active_vector_bytes=n * self.dim * self.values.element_size(),
                    allocated_tensor_bytes=tensor_bytes,
                    metadata_serialized_bytes=len(json.dumps(self.get_extra_state()).encode()),
                    entry_age_steps=[step - m["updated_step"] for m in self.metadata],
                    entry_lifetime_steps=[step - m["created_step"] for m in self.metadata],
                    accepted_exposures=sum(m["accepted_exposure_count"] for m in self.metadata))
