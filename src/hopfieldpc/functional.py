"""Functional Hopfield retrieval operations.

Implements the pure one-step modern Hopfield retrieval equation:

    weights = softmax(beta * query @ memory.T)
    retrieved = weights @ memory

and optional multi-step retrieval, which repeats the one-step update:

    q_{t+1} = HopfieldRetrieval(q_t, M)

The function in this file is intentionally independent from module wrappers so it can
be tested directly and reused by HopfieldMemory.
"""

from __future__ import annotations

from typing import Literal, overload

import jax
import jax.numpy as jnp

from .api import validate_beta, validate_query_memory


def _normalize_for_scores(
    query: jax.Array,
    memory: jax.Array,
    *,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Normalize query and memory only for similarity scoring.

    The retrieved vector is still computed from the original memory vectors.
    This keeps normalization from changing the memory values themselves.
    """

    query_norm = query / jnp.maximum(jnp.linalg.norm(query, axis=-1, keepdims=True), eps)
    memory_norm = memory / jnp.maximum(jnp.linalg.norm(memory, axis=-1, keepdims=True), eps)
    return query_norm, memory_norm


def hopfield_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = 1.0,
    normalize: bool = False,
) -> tuple[jax.Array, jax.Array]:
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
    working_dtype = jnp.float32 if input_dtype in (jnp.float16, jnp.bfloat16) else input_dtype
    query_work = query.astype(working_dtype)
    memory_work = memory.astype(working_dtype)

    if normalize:
        query_for_scores, memory_for_scores = _normalize_for_scores(
            query_work, memory_work
        )
    else:
        query_for_scores, memory_for_scores = query_work, memory_work

    similarities = jnp.matmul(query_for_scores, memory_for_scores.T)
    # Center before beta scaling to prevent positive overflow without changing softmax.
    similarities = similarities - similarities.max(axis=-1, keepdims=True)
    scores = beta * similarities
    weights_work = jax.nn.softmax(scores, axis=-1)
    retrieved_work = jnp.matmul(weights_work, memory_work)

    return (
        retrieved_work.astype(input_dtype),
        weights_work.astype(input_dtype),
    )


@overload
def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[False] = ...,
) -> tuple[jax.Array, jax.Array]: ...


@overload
def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[True],
) -> tuple[jax.Array, jax.Array, list[jax.Array]]: ...


def multi_step_retrieve(
    query: jax.Array,
    memory: jax.Array,
    *,
    beta: float = 1.0,
    num_updates: int = 1,
    normalize: bool = False,
    return_states: bool = False,
) -> (
    tuple[jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, list[jax.Array]]
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
    states: list[jax.Array] | None = [query] if return_states else None
    weights: jax.Array | None = None

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
