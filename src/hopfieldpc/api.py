"""Public API types and shape validation for HopfieldPC memory core.

This file defines the tensor contract used by the standalone HopfieldMemory
module. Retrieval operations are implemented separately in functional.py.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch


class HopfieldShapeError(ValueError):
    """Raised when query or memory tensors violate the Hopfield shape contract."""


@dataclass(frozen=True)
class HopfieldShapeContract:
    """Resolved tensor shape information for one Hopfield retrieval call."""

    batch_size: int
    dim: int
    memory_size: int


class HopfieldMemoryOutput(NamedTuple):
    """Standard output container for HopfieldMemory.

    This tuple-compatible container supports both unpacking and named access::

        retrieved, weights, diagnostics = module(query, memory)
        output = module(query, memory)
        retrieved = output.retrieved

    Attributes:
        retrieved:
            Retrieved memory vector with shape [batch_size, dim].
        weights:
            Softmax memory weights with shape [batch_size, num_memories].
        diagnostics:
            Dictionary for optional analysis values such as entropy,
            top-k mass, retrieved distance, or Hopfield energy.
    """

    retrieved: torch.Tensor
    weights: torch.Tensor
    diagnostics: dict[str, Any]


UPDATE_METHODS: tuple[str, ...] = (
    "ema",
    "hybrid",
    "momentum_only",
    "kalman_only",
)

GLOBAL_ABLATION_MODES: tuple[str, ...] = (
    "fixed_capacity",
    "dynamic_adaptive",
)


@dataclass(frozen=True)
class BankSnapshot:
    """Immutable snapshot of active online memory bank entries."""

    values: torch.Tensor
    labels: torch.Tensor
    entry_ids: torch.Tensor
    center: torch.Tensor
    version: int
    ready: bool


@dataclass(frozen=True)
class CandidateProposal:
    """Planned candidate action for an online Hopfield memory bank."""

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


class KalmanMomentumStepOutput(NamedTuple):
    """Output container for a functional Kalman-momentum prototype step."""

    new_value: torch.Tensor
    velocity: torch.Tensor
    uncertainty: torch.Tensor
    gain: torch.Tensor


@dataclass(frozen=True)
class GlobalDynamicConfig:
    """Validated hyperparameter configuration for GlobalDynamicHopfieldBank."""

    dim: int
    max_memories: int = 80
    initial_memories: int = 10
    min_active_memories: int = 2
    warmup_epochs: int = 1
    beta_read: float | None = 5.0
    beta_write: float | None = 5.0
    num_updates: int = 1
    gamma_mom: float = 0.9
    gamma_john: float = 1.0
    initial_uncertainty: float | None = 1.0
    uncertainty_min: float | None = 1e-4
    uncertainty_max: float | None = 10.0
    process_noise: float = 0.01
    observation_noise: float = 0.05
    tau_sim: float | None = 0.85
    lambda_sim: float | None = 0.05
    tau_coal: float = 0.95
    coalescence_patience: int = 2
    tau_support: float | None = 25.0
    tau_eff: float | None = 5.0
    alpha_n: float | None = 0.5
    alpha_s: float | None = 0.5
    prune_patience: int = 2
    tau_uncert: float | None = 0.5
    n_kish_min: float | None = 10.0
    split_scale: float | None = 0.25
    lambda_disp: float = 2.0
    lambda_sharp: float = 0.5
    structural_base_noise: float | None = 0.05
    eps_stab: float = 1e-8
    eps_norm: float = 1e-8
    ablation_mode: str = "dynamic_adaptive"
    stat_chunk_size: int = 2048
    auto_calibrate: bool = False

    def __post_init__(self) -> None:
        positive_int("dim", self.dim)
        positive_int("max_memories", self.max_memories)
        positive_int("initial_memories", self.initial_memories)
        positive_int("min_active_memories", self.min_active_memories)
        positive_int("num_updates", self.num_updates)
        positive_int("coalescence_patience", self.coalescence_patience)
        positive_int("prune_patience", self.prune_patience)
        positive_int("stat_chunk_size", self.stat_chunk_size)
        if self.initial_memories > self.max_memories:
            raise ValueError("initial_memories must be <= max_memories.")
        if self.min_active_memories > self.max_memories:
            raise ValueError("min_active_memories must be <= max_memories.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0.")
        if self.beta_read is not None:
            validate_beta(self.beta_read)
        if self.beta_write is not None:
            validate_beta(self.beta_write)
        validate_normalization_epsilon(self.eps_stab)
        validate_normalization_epsilon(self.eps_norm)
        validate_global_ablation_mode(self.ablation_mode)
        if not 0.0 <= self.gamma_mom <= 1.0:
            raise ValueError("gamma_mom must be in [0, 1].")
        if not math.isfinite(self.gamma_john) or self.gamma_john <= 0.0:
            raise ValueError("gamma_john must be positive and finite.")
        if self.initial_uncertainty is not None and (
            not math.isfinite(self.initial_uncertainty) or self.initial_uncertainty <= 0.0
        ):
            raise ValueError("initial_uncertainty must be positive and finite.")
        if self.uncertainty_min is not None and (
            not math.isfinite(self.uncertainty_min) or self.uncertainty_min <= 0.0
        ):
            raise ValueError("uncertainty_min must be positive and finite.")
        if self.uncertainty_max is not None and (
            not math.isfinite(self.uncertainty_max) or self.uncertainty_max <= 0.0
        ):
            raise ValueError("uncertainty_max must be positive and finite.")
        if (
            self.uncertainty_min is not None
            and self.initial_uncertainty is not None
            and self.uncertainty_min > self.initial_uncertainty
        ):
            raise ValueError("Require uncertainty_min <= initial_uncertainty.")
        if (
            self.initial_uncertainty is not None
            and self.uncertainty_max is not None
            and self.initial_uncertainty > self.uncertainty_max
        ):
            raise ValueError("Require initial_uncertainty <= uncertainty_max.")
        if (
            self.uncertainty_min is not None
            and self.uncertainty_max is not None
            and self.uncertainty_min > self.uncertainty_max
        ):
            raise ValueError("Require uncertainty_min <= uncertainty_max.")
        if self.process_noise < 0.0 or not math.isfinite(self.process_noise):
            raise ValueError("process_noise must be finite and nonnegative.")
        if self.observation_noise <= 0.0 or not math.isfinite(self.observation_noise):
            raise ValueError("observation_noise must be positive and finite.")
        if self.structural_base_noise is not None and (
            self.structural_base_noise <= 0.0 or not math.isfinite(self.structural_base_noise)
        ):
            raise ValueError("structural_base_noise must be positive and finite.")
        if (
            (self.lambda_sim is not None and self.lambda_sim < 0.0)
            or self.lambda_disp < 0.0
            or self.lambda_sharp < 0.0
        ):
            raise ValueError("lambda_* weights must be nonnegative.")
        if (self.alpha_n is not None and not 0.0 <= self.alpha_n <= 1.0) or (
            self.alpha_s is not None and not 0.0 <= self.alpha_s <= 1.0
        ):
            raise ValueError("alpha_n and alpha_s must be in [0, 1].")
        if (
            (self.tau_support is not None and self.tau_support < 0.0)
            or (self.tau_eff is not None and self.tau_eff < 0.0)
            or (self.n_kish_min is not None and self.n_kish_min < 0.0)
        ):
            raise ValueError("Support thresholds must be nonnegative.")
        if (self.tau_uncert is not None and self.tau_uncert <= 0.0) or (
            self.split_scale is not None and self.split_scale <= 0.0
        ):
            raise ValueError("tau_uncert and split_scale must be positive.")

    @classmethod
    def auto(cls, dim: int, **overrides: Any) -> GlobalDynamicConfig:
        """Create a mathematically self-calibrating configuration (any override pins that field)."""

        defaults: dict[str, Any] = {
            "beta_read": None,
            "beta_write": None,
            "initial_uncertainty": None,
            "uncertainty_min": None,
            "uncertainty_max": None,
            "tau_sim": None,
            "lambda_sim": None,
            "tau_support": None,
            "tau_eff": None,
            "alpha_n": None,
            "alpha_s": None,
            "tau_uncert": None,
            "n_kish_min": None,
            "split_scale": None,
            "structural_base_noise": None,
            "auto_calibrate": True,
        }
        defaults.update(overrides)
        return cls(dim=dim, **defaults)


def positive_int(name: str, value: Any) -> int:
    """Validate that value is a positive integer (excluding bool)."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_normalization_epsilon(value: object) -> float:
    """Validate that normalization_epsilon is a positive finite real number."""

    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("normalization_epsilon must be positive and finite.")
    return float(value)


def validate_update_method(method: str) -> str:
    """Validate that method is a supported prototype update method."""

    if method not in UPDATE_METHODS:
        raise ValueError(f"Unknown method {method!r}.")
    return method


def validate_global_ablation_mode(mode: str) -> str:
    """Validate that mode is a supported Experiment 4 memory variant."""

    if mode not in GLOBAL_ABLATION_MODES:
        raise ValueError(
            f"Unknown global ablation mode {mode!r}; expected one of {GLOBAL_ABLATION_MODES}."
        )
    return mode


def _require_tensor(name: str, value: object) -> torch.Tensor:
    """Validate that a value is a torch.Tensor and return it."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def validate_beta(beta: object) -> float:
    """Return beta as a finite positive float."""

    if (
        isinstance(beta, bool)
        or not isinstance(beta, numbers.Real)
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError(f"beta must be positive and finite, got {beta!r}.")

    return float(beta)


def validate_query_memory(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    expected_dim: int | None = None,
) -> HopfieldShapeContract:
    """Validate query and memory tensors for Hopfield retrieval.

    Expected shape contract:
        query:  [batch_size, dim]
        memory: [num_memories, dim]

    Args:
        query:
            Query tensor. Usually a hidden state or final representation.
        memory:
            Memory bank tensor. Each row is one stored pattern.
        expected_dim:
            Optional expected embedding/hidden dimension.

    Returns:
        HopfieldShapeContract with resolved batch size, dimension, and memory size.

    Raises:
        TypeError:
            If query or memory is not a floating-point torch.Tensor, or their
            dtypes differ.
        HopfieldShapeError:
            If tensor ranks or dimensions are invalid.
    """

    query = _require_tensor("query", query)
    memory = _require_tensor("memory", memory)

    if not query.is_floating_point() or not memory.is_floating_point():
        raise TypeError(
            "query and memory must be floating-point tensors, "
            f"got {query.dtype} and {memory.dtype}."
        )

    if query.dtype != memory.dtype:
        raise TypeError(
            "query and memory must share a dtype, "
            f"got {query.dtype} and {memory.dtype}."
        )

    if query.ndim != 2:
        raise HopfieldShapeError(
            f"query must have shape [batch_size, dim], got shape {tuple(query.shape)}."
        )

    if memory.ndim != 2:
        raise HopfieldShapeError(
            f"memory must have shape [num_memories, dim], got shape {tuple(memory.shape)}."
        )

    batch_size, query_dim = query.shape
    memory_size, memory_dim = memory.shape

    if batch_size <= 0:
        raise HopfieldShapeError("query batch_size must be greater than 0.")

    if memory_size <= 0:
        raise HopfieldShapeError("memory_size must be greater than 0.")

    if query_dim <= 0:
        raise HopfieldShapeError("query dim must be greater than 0.")

    if memory_dim <= 0:
        raise HopfieldShapeError("memory dim must be greater than 0.")

    if query_dim != memory_dim:
        raise HopfieldShapeError(
            "query and memory must share the same final dimension: "
            f"query dim={query_dim}, memory dim={memory_dim}."
        )

    if expected_dim is not None and query_dim != expected_dim:
        raise HopfieldShapeError(
            f"expected dim={expected_dim}, but query/memory dim={query_dim}."
        )

    return HopfieldShapeContract(
        batch_size=int(batch_size),
        dim=int(query_dim),
        memory_size=int(memory_size),
    )