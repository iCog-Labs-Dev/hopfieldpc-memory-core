"""Common base classes and dispatchers for prototype update rules.

Shared by ``ema.py``, ``momentum.py``, and ``kalman.py``.
"""

from __future__ import annotations

import math

import torch

from .api import (
    UPDATE_METHODS,
    KalmanMomentumStepOutput,
    validate_update_method,
)
from .functional import kmeans_prototypes

METHODS = UPDATE_METHODS


class BasePrototypeUpdater:
    """Common detached class-prototype container and k-means/assignment lifecycle.

    Subclasses in ``ema.py``, ``momentum.py``, and ``kalman.py`` implement
    :meth:`_apply_candidates` for their specific update dynamics.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        *,
        method: str,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1e-8,
        process_noise: float = 0.01,
        initial_uncertainty: float = 1.0,
        prototypes_per_class: int = 8,
        beta: float = 5.0,
        seed: int = 0,
    ) -> None:
        validate_update_method(method)
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha must be in (0, 1] and gamma must be in [0, 1].")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite.")
        if not math.isfinite(process_noise) or process_noise < 0:
            raise ValueError("process_noise must be finite and nonnegative.")
        if not math.isfinite(initial_uncertainty) or initial_uncertainty <= 0:
            raise ValueError("initial_uncertainty must be positive and finite.")
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be positive.")
        self.method = method
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.process_noise = float(process_noise)
        self.num_classes = num_classes
        self.dim = dim
        self.beta = float(beta)
        self.prototypes_per_class = prototypes_per_class
        self.seed = seed
        self.values = torch.zeros(num_classes, prototypes_per_class, dim)
        self.center = torch.zeros(dim)
        self.velocity = torch.zeros_like(self.values)
        self.uncertainty = torch.full(
            (num_classes, prototypes_per_class), initial_uncertainty
        )
        self.initialized = False

    @staticmethod
    def _normalize(values: torch.Tensor, epsilon: float) -> torch.Tensor:
        norms = values.norm(dim=-1, keepdim=True)
        if not torch.isfinite(values).all() or (norms <= epsilon).any():
            raise FloatingPointError(
                "Prototype update produced a nonfinite or zero vector."
            )
        return values / norms

    @torch.no_grad()
    def _initialize_from_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed)
        initial = []
        for class_id in range(self.num_classes):
            class_features = features[labels == class_id]
            if class_features.shape[0] < self.prototypes_per_class:
                raise ValueError(
                    "Each class needs at least prototypes_per_class examples."
                )
            initial.append(
                kmeans_prototypes(
                    class_features,
                    self.prototypes_per_class,
                    generator=generator,
                )
            )
        self.center.copy_(features.mean(dim=0))
        self.values.copy_(torch.stack(initial))
        self.initialized = True
        return {
            "gain": torch.ones_like(self.uncertainty),
            "velocity": self.velocity.clone(),
        }

    @torch.no_grad()
    def _aggregate_candidates(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.empty_like(self.values)
        for class_id in range(self.num_classes):
            class_features = features[labels == class_id]
            distances = torch.cdist(class_features, self.values[class_id])
            assignments = distances.argmin(dim=1)
            for prototype_id in range(self.prototypes_per_class):
                members = class_features[assignments == prototype_id]
                candidates[class_id, prototype_id] = (
                    members.mean(dim=0)
                    if len(members)
                    else self.values[class_id, prototype_id]
                )
        return candidates

    @torch.no_grad()
    def _apply_candidates(
        self,
        candidates: torch.Tensor,
        accuracy: float,
    ) -> dict[str, torch.Tensor]:
        step_out = kalman_momentum_step(
            self.values,
            candidates,
            self.velocity,
            self.uncertainty,
            accuracy=accuracy,
            method=self.method,
            alpha=self.alpha,
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

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        accuracy: float,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.values.shape[2]:
            raise ValueError("features must have shape [examples, dim].")
        if labels.shape != (features.shape[0],):
            raise ValueError("labels must match the number of features.")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite.")
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("accuracy must be finite and in [0, 1].")
        features, labels = features.detach().to(self.values), labels.detach()
        if not self.initialized:
            return self._initialize_from_features(features, labels)

        candidates = self._aggregate_candidates(features, labels)
        return self._apply_candidates(candidates, accuracy)


class PrototypeUpdater(BasePrototypeUpdater):
    """Detached class prototypes with selectable EMA/Kalman/momentum updates."""


@torch.no_grad()
def kalman_momentum_step(
    current: torch.Tensor,
    candidate: torch.Tensor,
    velocity: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    accuracy: float,
    method: str = "hybrid",
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1e-8,
    process_noise: float = 0.01,
) -> KalmanMomentumStepOutput:
    """Dispatch one prototype update step to ``ema``, ``momentum``, or ``kalman``."""

    from .ema import ema_step
    from .kalman import hybrid_kalman_momentum_step, kalman_step
    from .momentum import momentum_step

    validate_update_method(method)
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("alpha must be in (0, 1] and gamma must be in [0, 1].")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    if not math.isfinite(process_noise) or process_noise < 0:
        raise ValueError("process_noise must be finite and nonnegative.")
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be finite and in [0, 1].")

    if method == "ema":
        new_value = ema_step(current, candidate, alpha=alpha)
        gain = torch.full_like(uncertainty, alpha)
        return KalmanMomentumStepOutput(
            new_value=new_value,
            velocity=velocity.clone(),
            uncertainty=uncertainty.clone(),
            gain=gain,
        )

    if method == "momentum_only":
        return momentum_step(
            current,
            candidate,
            velocity,
            uncertainty,
            accuracy=accuracy,
            gamma=gamma,
            epsilon=epsilon,
            process_noise=process_noise,
        )

    if method == "kalman_only":
        return kalman_step(
            current,
            candidate,
            velocity,
            uncertainty,
            accuracy=accuracy,
            epsilon=epsilon,
            process_noise=process_noise,
        )

    return hybrid_kalman_momentum_step(
        current,
        candidate,
        velocity,
        uncertainty,
        accuracy=accuracy,
        gamma=gamma,
        epsilon=epsilon,
        process_noise=process_noise,
    )
