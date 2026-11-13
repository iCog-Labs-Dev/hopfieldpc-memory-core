"""Exponential Moving Average (EMA) prototype updates and online memory bank.

Builds on common base classes in ``updates.py`` and ``online.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from .online import OnlineHopfieldBank
from .updates import BasePrototypeUpdater


@torch.no_grad()
def ema_step(
    current: torch.Tensor,
    candidate: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute one exponential moving average prototype update step."""

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    return (1 - alpha) * current + alpha * candidate


class EMAPrototypeUpdater(BasePrototypeUpdater):
    """Detached class prototypes updated with fixed-rate EMA."""

    def __init__(
        self,
        num_classes: int,
        dim: int,
        *,
        alpha: float = 0.1,
        prototypes_per_class: int = 8,
        beta: float = 5.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes,
            dim,
            method="ema",
            alpha=alpha,
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
        del accuracy
        self.values.copy_(ema_step(self.values, candidates, alpha=self.alpha))
        return {
            "gain": torch.full_like(self.uncertainty, self.alpha),
            "velocity": self.velocity.clone(),
        }


class EMAOnlineBank(OnlineHopfieldBank):
    """Online Hopfield memory bank using fixed-rate EMA merges."""
