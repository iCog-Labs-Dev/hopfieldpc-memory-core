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


def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = 1.0,
    num_updates: int = 1,
    normalize: bool = False,
    return_states: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
):
    if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
        raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")

    state = query
    states = [query]
    weights = None

    for _ in range(num_updates):
        state, weights = hopfield_retrieve(
            query=state,
            memory=memory,
            beta=beta,
            normalize=normalize,
        )
        if return_states:
            states.append(state)

    if return_states:
        return state, weights, states
    return state, weights