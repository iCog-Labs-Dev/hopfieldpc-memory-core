"""Hopfield energy computation for logging and analysis.

Implements the modern continuous Hopfield energy (Ramsauer et al., 2020):

    E(q, M, β) = -(1/β) · logsumexp(β · q Mᵀ) + ½ ‖q‖² + (1/β) · log(N)
"""

from __future__ import annotations

import math

import torch

from .api import validate_query_memory


def hopfield_energy(
    query: torch.Tensor,
    memory: torch.Tensor,
    beta: float = 1.0,
    *,
    include_constants: bool = False,
    reduce: str = "none",
) -> torch.Tensor:
    """Compute modern Hopfield energy for each query.

    Args:
        query: Query tensor with shape [batch_size, dim].
        memory: Memory bank tensor with shape [num_memories, dim].
        beta: Retrieval sharpness (inverse temperature). Must be positive.
        include_constants: If True, include the ½‖q‖² and (1/β)·log(N) terms.
        reduce: ``"none"`` returns per-query energy [batch_size],
                ``"mean"`` returns the scalar batch mean.

    Returns:
        Hopfield energy tensor.
    """

    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta!r}.")

    if reduce not in ("none", "mean"):
        raise ValueError(f"reduce must be 'none' or 'mean', got {reduce!r}.")

    validate_query_memory(query=query, memory=memory)

    scores = beta * torch.matmul(query, memory.transpose(0, 1))
    energy = -(1.0 / beta) * torch.logsumexp(scores, dim=-1)

    if include_constants:
        num_memories = memory.shape[0]
        query_norm_sq = 0.5 * (query ** 2).sum(dim=-1)
        log_n_term = (1.0 / beta) * math.log(num_memories)
        energy = energy + query_norm_sq + log_n_term

    if reduce == "mean":
        return energy.mean()

    return energy
