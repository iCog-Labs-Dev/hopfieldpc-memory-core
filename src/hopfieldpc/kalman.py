"""Kalman and Hybrid Kalman-Momentum prototype updates and online memory banks.

Builds on common base classes in ``updates.py`` and ``online.py``, and composes
with ``momentum.py`` and ``ema.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .api import (
    CandidateProposal,
    KalmanMomentumStepOutput,
    validate_update_method,
)
from .ema import ema_step
from .momentum import momentum_velocity_update
from .online import OnlineHopfieldBank
from .updates import BasePrototypeUpdater


@torch.no_grad()
def kalman_gain(
    uncertainty: torch.Tensor,
    *,
    accuracy: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute accuracy-gated scalar Kalman gain K = P / (P + (1 - A) + eps)."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be finite and in [0, 1].")
    noise = (1 - accuracy) + epsilon
    return uncertainty / (uncertainty + noise)


@torch.no_grad()
def kalman_uncertainty_update(
    uncertainty: torch.Tensor,
    gain: torch.Tensor,
    *,
    process_noise: float = 0.01,
) -> torch.Tensor:
    """Propagate prototype uncertainty P <- (1 - K) P + Q."""

    if not math.isfinite(process_noise) or process_noise < 0:
        raise ValueError("process_noise must be finite and nonnegative.")
    return (1 - gain) * uncertainty + process_noise


@torch.no_grad()
def kalman_step(
    current: torch.Tensor,
    candidate: torch.Tensor,
    velocity: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    accuracy: float,
    epsilon: float = 1e-8,
    process_noise: float = 0.01,
) -> KalmanMomentumStepOutput:
    """Compute a Kalman-only prototype step (no momentum velocity smoothing)."""

    gain = kalman_gain(uncertainty, accuracy=accuracy, epsilon=epsilon)
    delta = candidate - current
    scale = gain if gain.ndim == 0 else gain[..., None]
    new_value = current + scale * delta
    new_uncertainty = kalman_uncertainty_update(
        uncertainty, gain, process_noise=process_noise
    )
    return KalmanMomentumStepOutput(
        new_value=new_value,
        velocity=velocity.clone(),
        uncertainty=new_uncertainty,
        gain=gain,
    )


@torch.no_grad()
def hybrid_kalman_momentum_step(
    current: torch.Tensor,
    candidate: torch.Tensor,
    velocity: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    accuracy: float,
    gamma: float = 0.9,
    epsilon: float = 1e-8,
    process_noise: float = 0.01,
) -> KalmanMomentumStepOutput:
    """Compute a full Hybrid Kalman-Momentum prototype step."""

    gain = kalman_gain(uncertainty, accuracy=accuracy, epsilon=epsilon)
    delta = candidate - current
    new_velocity = momentum_velocity_update(velocity, delta, gamma=gamma)
    scale = gain if gain.ndim == 0 else gain[..., None]
    new_value = current + scale * new_velocity
    new_uncertainty = kalman_uncertainty_update(
        uncertainty, gain, process_noise=process_noise
    )
    return KalmanMomentumStepOutput(
        new_value=new_value,
        velocity=new_velocity,
        uncertainty=new_uncertainty,
        gain=gain,
    )


class KalmanPrototypeUpdater(BasePrototypeUpdater):
    """Detached class prototypes updated with accuracy-gated Kalman gain only."""

    def __init__(
        self,
        num_classes: int,
        dim: int,
        *,
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
            method="kalman_only",
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
        step_out = kalman_step(
            self.values,
            candidates,
            self.velocity,
            self.uncertainty,
            accuracy=accuracy,
            epsilon=self.epsilon,
            process_noise=self.process_noise,
        )
        self.values.copy_(step_out.new_value)
        self.uncertainty.copy_(step_out.uncertainty)
        return {
            "gain": step_out.gain.clone(),
            "velocity": self.velocity.clone(),
        }


class HybridPrototypeUpdater(BasePrototypeUpdater):
    """Detached class prototypes updated with Hybrid Kalman-Momentum."""

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
            method="hybrid",
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
        step_out = hybrid_kalman_momentum_step(
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


class HybridOnlineBank(OnlineHopfieldBank):
    """Online Hopfield bank whose accepted merges use Kalman/momentum state."""

    def __init__(
        self,
        *args: Any,
        method: str = "hybrid",
        gamma: float = 0.9,
        process_noise: float = 0.01,
        initial_uncertainty: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        validate_update_method(method)
        self.update_method = method
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
        gain = kalman_gain(
            self.uncertainty[slot],
            accuracy=self.current_accuracy,
            epsilon=self.eps,
        )
        delta = candidate - self.values[slot]
        if self.update_method in ("hybrid", "momentum_only"):
            self.velocity[slot].mul_(self.gamma).add_(
                delta, alpha=1 - self.gamma
            )
            step = self.velocity[slot]
        else:
            step = delta
        applied_gain = 1.0 if self.update_method == "momentum_only" else gain
        if self.update_method == "ema":
            new_value = ema_step(self.values[slot], candidate, alpha=self.alpha)
        else:
            new_value = self.values[slot] + applied_gain * step
        self.uncertainty[slot].copy_(
            kalman_uncertainty_update(
                self.uncertainty[slot],
                gain,
                process_noise=self.process_noise,
            )
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


class KalmanOnlineBank(HybridOnlineBank):
    """Online Hopfield bank specialized to Kalman-only merges."""

    def __init__(
        self,
        *args: Any,
        process_noise: float = 0.01,
        initial_uncertainty: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            method="kalman_only",
            gamma=0.0,
            process_noise=process_noise,
            initial_uncertainty=initial_uncertainty,
            **kwargs,
        )
