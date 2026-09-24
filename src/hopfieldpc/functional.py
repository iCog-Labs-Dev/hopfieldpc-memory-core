"""Functional Hopfield retrieval operations.

Implements the pure one-step modern Hopfield retrieval equation:

    weights = softmax(beta * query @ memory.T)
    retrieved = weights @ memory

The function in this file is intentionally independent from nn.Module so it can
be tested directly and reused by HopfieldMemory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .api import validate_query_memory


def _normalize_for_scores(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize query and memory only for similarity scoring.

    The retrieved vector is still computed from the original memory vectors.
    This keeps normalization from changing the memory values themselves.
    """

    query_norm = F.normalize(query, p=2, dim=-1, eps=eps)
    memory_norm = F.normalize(memory, p=2, dim=-1, eps=eps)
    return query_norm, memory_norm


def hopfield_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = 1.0,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-step modern Hopfield retrieval.

    Args:
        query:
            Query tensor with shape [batch_size, dim].
        memory:
            Memory bank tensor with shape [num_memories, dim].
            Each row is one stored pattern.
        beta:
            Retrieval sharpness. Larger beta produces sharper memory selection.
        normalize:
            If True, query and memory are L2-normalized for similarity scoring.
            The retrieved vector is still computed from the original memory.

    Returns:
        retrieved:
            Retrieved memory vectors with shape [batch_size, dim].
        weights:
            Softmax memory weights with shape [batch_size, num_memories].

    Formula:
        scores = beta * query @ memory.T
        weights = softmax(scores)
        retrieved = weights @ memory
    """

    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta!r}.")

    validate_query_memory(query=query, memory=memory)

    if normalize:
        query_for_scores, memory_for_scores = _normalize_for_scores(query, memory)
    else:
        query_for_scores, memory_for_scores = query, memory

    scores = beta * torch.matmul(query_for_scores, memory_for_scores.transpose(0, 1))
    weights = torch.softmax(scores, dim=-1)
    retrieved = torch.matmul(weights, memory)

    return retrieved, weights