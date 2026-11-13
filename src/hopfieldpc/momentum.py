"""Momentum prototype updates and online memory bank.

Builds on common base classes in ``updates.py`` and ``online.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .api import CandidateProposal, KalmanMomentumStepOutput
from .online import OnlineHopfieldBank
from .updates import BasePrototypeUpdater


@torch.no_grad()
def momentum_velocity_update(
    velocity: torch.Tensor,
    delta: torch.Tensor,
    *,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Update momentum velocity state in-place on a cloned tensor for float32 parity."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    return velocity.clone().mul_(gamma).add_(delta, alpha=1 - gamma)


@torch.no_grad()
def momentum_step(
    current: torch.Tensor,
    candidate: torch.Tensor,
    velocity: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    accuracy: float = 1.0,
    gamma: float = 0.9,
    epsilon: float = 1e-8,
    process_noise: float = 0.01,
) -> KalmanMomentumStepOutput:
    """Compute a momentum-only prototype step (unit gain, smoothed velocity)."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1].")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    if not math.isfinite(process_noise) or process_noise < 0:
        raise ValueError("process_noise must be finite and nonnegative.")
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be finite and in [0, 1].")

    noise = (1 - accuracy) + epsilon
    raw_gain = uncertainty / (uncertainty + noise)
    delta = candidate - current
    new_velocity = momentum_velocity_update(velocity, delta, gamma=gamma)
    applied_gain = torch.ones_like(raw_gain)
    scale = applied_gain if applied_gain.ndim == 0 else applied_gain[..., None]
    new_value = current + scale * new_velocity
    new_uncertainty = (1 - raw_gain) * uncertainty + process_noise
    return KalmanMomentumStepOutput(
        new_value=new_value,
        velocity=new_velocity,
        uncertainty=new_uncertainty,
        gain=applied_gain,
    )


class MomentumPrototypeUpdater(BasePrototypeUpdater):
    """Detached class prototypes updated with momentum velocity smoothing."""

    def __init__(
        self,
        num_classes: int,
        dim: int,
        *,
        gamma: float = 0.9,
        epsilon: float = 1e-8,
        process_noise: float = 0.01,
        initial_uncertainty: float = 1.0,
        prototypes_per_class: int = 8,
        beta: float = 5.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes,
            dim,
            method="momentum_only",
            gamma=gamma,
            epsilon=epsilon,
            process_noise=process_noise,
            initial_uncertainty=initial_uncertainty,
            prototypes_per_class=prototypes_per_class,
            beta=beta,
            seed=seed,
            **kwargs,
        )

    @torch.no_grad()
    def _apply_candidates(
        self,
        candidates: torch.Tensor,
        accuracy: float,
    ) -> dict[str, torch.Tensor]:
        step_out = momentum_step(
            self.values,
            candidates,
            self.velocity,
            self.uncertainty,
            accuracy=accuracy,
            gamma=self.gamma,
            epsilon=self.epsilon,
            process_noise=self.process_noise,
        )
        self.values.copy_(step_out.new_value)
        self.velocity.copy_(step_out.velocity)
        self.uncertainty.copy_(step_out.uncertainty)
        return {
            "gain": step_out.gain.clone(),
            "velocity": self.velocity.clone(),
        }


class MomentumOnlineBank(OnlineHopfieldBank):
    """Online Hopfield bank whose accepted merges use momentum velocity state."""

    def __init__(
        self,
        *args: Any,
        gamma: float = 0.9,
        process_noise: float = 0.01,
        initial_uncertainty: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.update_method = "momentum_only"
        self.gamma = gamma
        self.process_noise = process_noise
        self.register_buffer("velocity", torch.zeros_like(self.values))
        self.register_buffer(
            "uncertainty", torch.ones(len(self.values)) * initial_uncertainty
        )
        self.current_accuracy = 1.0

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
        gain = self.uncertainty[slot] / (
            self.uncertainty[slot] + (1 - self.current_accuracy) + self.eps
        )
        delta = candidate - self.values[slot]
        self.velocity[slot].mul_(self.gamma).add_(delta, alpha=1 - self.gamma)
        new_value = self.values[slot] + 1.0 * self.velocity[slot]
        self.uncertainty[slot].copy_(
            (1 - gain) * self.uncertainty[slot] + self.process_noise
        )
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
