"""Functional Hopfield retrieval operations.

Implements the pure one-step modern Hopfield retrieval equation:

    weights = softmax(beta * query @ memory.T)
    retrieved = weights @ memory

and optional multi-step retrieval, which repeats the one-step update:

    q_{t+1} = HopfieldRetrieval(q_t, M)

The function in this file is intentionally independent from nn.Module so it can
be tested directly and reused by HopfieldMemory.
"""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
import torch.nn.functional as F

from .api import (
    positive_int,
    validate_beta,
    validate_normalization_epsilon,
    validate_query_memory,
)


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
            Finite positive retrieval sharpness. Larger beta produces sharper
            memory selection.
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

    beta = validate_beta(beta)

    validate_query_memory(query=query, memory=memory)

    input_dtype = query.dtype
    working_dtype = (
        torch.float32
        if input_dtype in (torch.float16, torch.bfloat16)
        else input_dtype
    )
    query_work = query.to(dtype=working_dtype)
    memory_work = memory.to(dtype=working_dtype)

    if normalize:
        query_for_scores, memory_for_scores = _normalize_for_scores(
            query_work, memory_work
        )
    else:
        query_for_scores, memory_for_scores = query_work, memory_work

    similarities = torch.matmul(query_for_scores, memory_for_scores.transpose(0, 1))
    # Center before beta scaling to prevent positive overflow without changing softmax.
    similarities = similarities - similarities.amax(dim=-1, keepdim=True)
    scores = beta * similarities
    weights_work = torch.softmax(scores, dim=-1)
    retrieved_work = torch.matmul(weights_work, memory_work)

    return (
        retrieved_work.to(dtype=input_dtype),
        weights_work.to(dtype=input_dtype),
    )


@overload
def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: ...


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
    """Run repeated Hopfield retrieval: q0 -> q1 -> ... -> qT.

    Args:
        query:
            Initial query q0 with shape [batch_size, dim].
        memory:
            Memory bank with shape [num_memories, dim].
        beta:
            Retrieval sharpness (positive).
        num_updates:
            Number of retrieval steps T (positive integer). Default 1
            reproduces plain one-step retrieval.
        normalize:
            Passed to every one-step retrieval (scoring-only normalization).
        return_states:
            If True, also return the list [q0, q1, ..., qT] for debugging.

    Returns:
        retrieved:
            Final state qT with shape [batch_size, dim].
        weights:
            Softmax weights from the last step, shape [batch_size, num_memories].
        states:
            Only if return_states=True: list of T + 1 tensors, q0 first.
    """

    if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
        raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")

    state = query
    states: list[torch.Tensor] | None = [query] if return_states else None
    weights: torch.Tensor | None = None

    for _ in range(num_updates):
        state, weights = hopfield_retrieve(
            query=state,
            memory=memory,
            beta=beta,
            normalize=normalize,
        )
        if states is not None:
            states.append(state)

    # num_updates > 0 is validated above, so the loop ran at least once.
    assert weights is not None

    if states is not None:
        return state, weights, states
    return state, weights


@torch.no_grad()
def centered_retrieve(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    *,
    beta: float = 5.0,
    num_updates: int = 1,
    normalization_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score centered directions and return a convex combination of RAW values."""

    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if center.shape != (query.shape[1],):
        raise ValueError("center must have shape [dim].")
    q, m = query - center, values - center
    if not torch.isfinite(q).all() or not torch.isfinite(m).all():
        raise FloatingPointError("Nonfinite centered query or memory.")
    qnorm, mnorm = q.norm(dim=-1), m.norm(dim=-1)
    if (
        not torch.isfinite(qnorm).all()
        or not torch.isfinite(mnorm).all()
        or (mnorm <= eps).any()
    ):
        raise FloatingPointError(
            "Invalid centered query norm or degenerate memory key."
        )
    positive_int("num_updates", num_updates)
    views = F.normalize(m, dim=-1, eps=eps)
    raw = query
    for _ in range(num_updates):
        query_view = F.normalize(raw - center, dim=-1, eps=eps)
        _, weights = hopfield_retrieve(
            query_view, views, beta=beta, normalize=False
        )
        raw = weights @ values
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite retrieval output.")
    return raw, weights


def kmeans_prototypes(
    x: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator,
    iters: int = 25,
) -> torch.Tensor:
    """Seeded k-means++ / Lloyd on rows of x [N, D]; returns centroids [k, D]."""

    if k >= x.shape[0]:
        raise ValueError(
            f"k ({k}) must be smaller than the number of points ({x.shape[0]})."
        )
    first = torch.randint(0, x.shape[0], (1,), generator=generator)
    centroids = x[first]
    while centroids.shape[0] < k:
        d2 = torch.cdist(x, centroids).min(dim=1).values.pow(2)
        nxt = torch.multinomial(d2 / d2.sum(), 1, generator=generator)
        centroids = torch.cat([centroids, x[nxt]])
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        for j in range(k):
            members = x[assign == j]
            if members.shape[0] > 0:
                centroids[j] = members.mean(dim=0)
    return centroids


@torch.no_grad()
def masked_centered_retrieve(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 5.0,
    num_updates: int = 1,
    normalization_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve raw prototypes using masked softmax over centered cosine scores.

    Implements Equation (4) from Prototype Selection.pdf:
        w_j = softmax(beta * q_tilde @ M_tilde_j^T * m_j + (-inf) * (1 - m_j))
    """

    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must have shape [num_memories].")
    active_mask = mask.bool()
    if not active_mask.any():
        raise ValueError("At least one prototype must be active for retrieval.")

    q = query - center
    m = values - center
    if not torch.isfinite(q).all() or not torch.isfinite(m[active_mask]).all():
        raise FloatingPointError(
            "Nonfinite centered query or active memory."
        )

    views = torch.zeros_like(m)
    views[active_mask] = F.normalize(m[active_mask], dim=-1, eps=eps)
    mask_float = active_mask.to(query.dtype)

    raw = query
    weights = torch.zeros(
        query.shape[0],
        values.shape[0],
        dtype=query.dtype,
        device=query.device,
    )
    for _ in range(num_updates):
        query_view = F.normalize(raw - center, dim=-1, eps=eps)
        scores = beta * (query_view @ views.T) * mask_float
        scores = scores.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        raw = weights @ values
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite masked retrieval output.")
    return raw, weights


@torch.no_grad()
def principal_split_direction(
    members: torch.Tensor,
    fallback_dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute first principal component direction of assigned embeddings S_{j,n}."""

    if members.shape[0] >= 2:
        centered = members - members.mean(dim=0, keepdim=True)
        cov_scale = centered.norm(dim=-1).mean()
        if torch.isfinite(cov_scale) and cov_scale > eps:
            try:
                _, s, vh = torch.linalg.svd(centered, full_matrices=False)
                direction = vh[0]
                scale = s[0] / math.sqrt(max(members.shape[0] - 1, 1))
                if torch.isfinite(direction).all() and scale > eps:
                    return F.normalize(direction, dim=-1, eps=eps) * scale
            except RuntimeError:
                pass
    unit = torch.zeros(
        fallback_dim, dtype=members.dtype, device=members.device
    )
    unit[0] = 1.0
    return unit


