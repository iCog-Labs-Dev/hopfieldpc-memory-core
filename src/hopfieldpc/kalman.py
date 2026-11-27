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
from .momentum import (
    momentum_velocity_update,
    responsibility_momentum_velocity_update,
)
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


@torch.no_grad()
def john_effective_noise(
    responsibility: torch.Tensor,
    observation_noise: float,
    *,
    gamma_john: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute John Effective Observation Noise R_eff = R_n / (r_ij + eps)^gamma_john."""

    if not math.isfinite(observation_noise) or observation_noise <= 0.0:
        raise ValueError("observation_noise must be positive and finite.")
    if not math.isfinite(gamma_john) or gamma_john <= 0.0:
        raise ValueError("gamma_john must be positive and finite.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite.")
    denom = torch.pow(responsibility.clamp(0.0, 1.0) + eps, gamma_john)
    return observation_noise / denom


def fractional_measurement_factors(
    prior: torch.Tensor, noise: torch.Tensor | float, mass: torch.Tensor | float, *, eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correction factor for a summed innovation and fraction of prior variance retained.

    For assignment mass N and positive observation variance R the factors are
    P/(PN+R) and R/(PN+R). At N=1 the first is the scalar measurement gain.
    Computing retention directly also works when the gain rounds to one.
    """
    total = prior * mass + noise + eps
    return prior / total, (noise + eps) / total


def predict_position_velocity(
    covariance: torch.Tensor, process: torch.Tensor, dt: float, damping: float,
) -> torch.Tensor:
    """Propagate a batch of position/velocity covariances through one damped interval."""
    transition = covariance.new_tensor([[1., dt], [0., damping]])
    return transition @ covariance @ transition.T + dt * process


def condition_position_velocity(
    covariance: torch.Tensor, mass: torch.Tensor, noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return corrections per unit summed innovation and the Joseph posterior covariance."""
    column = covariance[:, :, 0]
    factors = column / (noise + column[:, :1] * mass[:, None])
    gain = factors * mass[:, None]
    transform = torch.eye(2, device=covariance.device, dtype=covariance.dtype).expand_as(covariance).clone()
    transform[:, :, 0] -= gain
    posterior = transform @ covariance @ transform.mT + gain.unsqueeze(2) * (factors * noise).unsqueeze(1)
    return factors, (posterior + posterior.mT) / 2


@torch.no_grad()
def responsibility_kalman_gain(
    uncertainty: torch.Tensor,
    effective_noise: torch.Tensor | float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute scalar Kalman gain K_ij = P_j / (P_j + R_eff_ij + eps)."""

    gain, _ = fractional_measurement_factors(uncertainty, effective_noise, 1., eps=eps)
    return gain


@torch.no_grad()
def responsibility_uncertainty_update(
    uncertainty: torch.Tensor,
    gain: torch.Tensor,
    responsibility: torch.Tensor,
    *,
    process_noise: float = 0.01,
    gamma_john: float = 1.0,
    uncertainty_min: float = 1e-4,
    uncertainty_max: float = 10.0,
) -> torch.Tensor:
    """Propagate scalar uncertainty under responsibility-aware John Factor."""

    if not math.isfinite(process_noise) or process_noise < 0.0:
        raise ValueError("process_noise must be finite and nonnegative.")
    r_weight = torch.pow(responsibility.clamp(0.0, 1.0), gamma_john)
    new_p = (1.0 - gain) * uncertainty + r_weight * process_noise
    return new_p.clamp(min=uncertainty_min, max=uncertainty_max)


@torch.no_grad()
def responsibility_kalman_momentum_step(
    current: torch.Tensor,
    candidate: torch.Tensor,
    velocity: torch.Tensor,
    uncertainty: torch.Tensor,
    responsibility: torch.Tensor,
    *,
    observation_noise: float = 0.05,
    gamma_mom: float = 0.9,
    gamma_john: float = 1.0,
    process_noise: float = 0.01,
    eps: float = 1e-8,
    gain_eps: float | None = None,
    uncertainty_min: float = 1e-4,
    uncertainty_max: float = 10.0,
) -> KalmanMomentumStepOutput:
    """Compute one John-Factor responsibility-aware Kalman-momentum prototype step."""

    r = responsibility.to(dtype=current.dtype, device=current.device).clamp(0.0, 1.0)
    r_eff = john_effective_noise(
        r, observation_noise, gamma_john=gamma_john, eps=eps
    )
    gain = responsibility_kalman_gain(uncertainty, r_eff, eps=eps if gain_eps is None else gain_eps)
    innovation = candidate - current
    new_velocity = responsibility_momentum_velocity_update(
        velocity, innovation, r, gamma_mom=gamma_mom
    )
    scale = gain if gain.ndim == new_velocity.ndim else gain.unsqueeze(-1)
    new_value = current + scale * new_velocity
    new_uncertainty = responsibility_uncertainty_update(
        uncertainty,
        gain,
        r,
        process_noise=process_noise,
        gamma_john=gamma_john,
        uncertainty_min=uncertainty_min,
        uncertainty_max=uncertainty_max,
    )
    return KalmanMomentumStepOutput(
        new_value=new_value,
        velocity=new_velocity,
        uncertainty=new_uncertainty,
        gain=gain,
    )


@torch.no_grad()
def structural_uncertainty_update(
    uncertainty: torch.Tensor,
    sharpness: torch.Tensor,
    dispersion: torch.Tensor,
    *,
    structural_base_noise: float = 0.05,
    lambda_disp: float = 2.0,
    lambda_sharp: float = 0.5,
    process_noise: float = 0.01,
    variance_scale: float = 1.0,
    uncertainty_min: float = 1e-4,
    uncertainty_max: float = 10.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Section 6.2.1 epoch-end structural uncertainty propagation."""

    sharp = sharpness.clamp(0.0, 1.0)
    disp = dispersion.clamp_min(0.0)
    scale = float(variance_scale)
    r_struct = scale * (
        structural_base_noise + lambda_disp * disp + lambda_sharp * (1.0 - sharp)
    )
    inj = scale * sharp * (process_noise + lambda_disp * disp.pow(2))
    eff_p = sharp * uncertainty
    k_struct = eff_p / (eff_p + r_struct + eps)
    new_p = ((1.0 - k_struct) * uncertainty + inj).clamp(
        min=uncertainty_min, max=uncertainty_max
    )
    return new_p, k_struct, r_struct


@torch.no_grad()
def coalesce_uncertainty_total_variance(
    p_j: torch.Tensor,
    p_k: torch.Tensor,
    m_j: torch.Tensor,
    m_k: torch.Tensor,
    m_merged: torch.Tensor,
    omega_j: float,
    omega_k: float,
    *,
    uncertainty_min: float = 1e-4,
    uncertainty_max: float = 10.0,
) -> torch.Tensor:
    """Section 6.4.3 law-of-total-variance uncertainty fusion for coalesced prototypes."""

    dj = m_j - m_merged
    dk = m_k - m_merged
    dim = m_j.shape[-1]
    merged = omega_j * (p_j + dj.pow(2).sum() / dim) + omega_k * (
        p_k + dk.pow(2).sum() / dim
    )
    return merged.clamp(min=uncertainty_min, max=uncertainty_max)


@torch.no_grad()
def fission_uncertainty_inverse_total_variance(
    p_parent: torch.Tensor,
    delta: torch.Tensor,
    *,
    process_noise: float = 0.01,
    observation_noise: float = 0.05,
    uncertainty_min: float = 1e-4,
    uncertainty_max: float = 10.0,
) -> torch.Tensor:
    """Inverse law-of-total-variance child uncertainty after symmetric +/- delta fission."""

    p_inf = riccati_steady_state_uncertainty(process_noise, observation_noise)
    floor = min(float(uncertainty_max), max(float(uncertainty_min), p_inf))
    dim = delta.shape[-1]
    residual = p_parent - delta.pow(2).sum() / dim
    return residual.clamp(min=floor, max=uncertainty_max)


def riccati_steady_state_uncertainty(
    process_noise: float,
    observation_noise: float,
) -> float:
    """Exact positive root P_inf of the scalar Discrete Algebraic Riccati Equation.

    Solves P_inf = (1 - P_inf / (P_inf + R_n)) P_inf + Q  =>  P_inf^2 - Q P_inf - Q R_n = 0.
    """

    if process_noise < 0.0 or observation_noise <= 0.0:
        raise ValueError("Require process_noise >= 0 and observation_noise > 0.")
    q = float(process_noise)
    r = float(observation_noise)
    return 0.5 * (q + math.sqrt(q * q + 4.0 * q * r))


def auto_spawning_uncertainty_threshold(
    process_noise: float,
    observation_noise: float,
    *,
    structural_base_noise: float | None = None,
    lambda_disp: float = 2.0,
    tau_coal: float = 0.95,
    variance_scale: float = 1.0,
) -> float:
    """Analytical spawning threshold tau_uncert from DARE P_inf and coalescence angle tau_coal.

    Evaluates the epoch-end structural uncertainty equilibrium at the single-mode
    cosine dispersion boundary sigma_single = 1.4 * (1 - tau_coal), scaled by
    per-dimension embedding variance ``variance_scale``.
    """

    p_inf = riccati_steady_state_uncertainty(process_noise, observation_noise)
    r0 = float(observation_noise if structural_base_noise is None else structural_base_noise)
    sigma_single = 1.4 * max(0.0, 1.0 - float(tau_coal))
    r_struct = r0 + float(lambda_disp) * sigma_single
    p_eq = (r_struct / (p_inf + r_struct + 1e-12)) * p_inf + float(process_noise) + float(lambda_disp) * (
        sigma_single * sigma_single
    )
    return max(float(variance_scale) * p_eq, 1e-12)


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
