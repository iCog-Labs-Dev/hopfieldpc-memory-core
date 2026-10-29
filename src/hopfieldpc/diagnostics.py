"""Diagnostics for Hopfield memory retrieval.

These utilities are for analysis/logging. They help answer whether retrieval is
focused on one memory, spread across many memories, or concentrated in a small
top-k set.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _validate_weights(weights: jax.Array) -> None:
    if not isinstance(weights, jax.Array):
        raise TypeError(f"weights must be a jax.Array, got {type(weights).__name__}.")
    if weights.ndim != 2:
        raise ValueError(
            f"weights must have shape [batch_size, num_memories], got {tuple(weights.shape)}."
        )
    if weights.shape[1] <= 0:
        raise ValueError("weights must contain at least one memory column.")


def attention_entropy(
    weights: jax.Array,
    *,
    eps: float = 1e-12,
    normalize: bool = False,
) -> jax.Array:
    """Compute entropy of Hopfield memory weights.

    Args:
        weights: Softmax weights with shape [batch_size, num_memories].
        eps: Numerical stability value for log.
        normalize: If True, divide by log(num_memories), so uniform entropy is 1.

    Returns:
        Entropy tensor with shape [batch_size]. Float16 and bfloat16 weights
        are evaluated and returned in float32 for numerical stability.
    """

    _validate_weights(weights)

    entropy_weights = (
        weights.astype(jnp.float32)
        if weights.dtype in (jnp.float16, jnp.bfloat16)
        else weights
    )
    safe_weights = jnp.maximum(entropy_weights, eps)
    entropy = -(entropy_weights * jnp.log(safe_weights)).sum(axis=-1)

    if normalize:
        num_memories = weights.shape[-1]
        if num_memories == 1:
            return jnp.zeros_like(entropy)
        entropy = entropy / jnp.log(jnp.array(float(num_memories), dtype=entropy.dtype))

    return entropy


def topk_mass(weights: jax.Array, *, k: int = 1) -> jax.Array:
    """Compute total probability mass in the top-k memories.

    Args:
        weights: Softmax weights with shape [batch_size, num_memories].
        k: Number of largest weights to sum.

    Returns:
        Tensor with shape [batch_size].
    """

    _validate_weights(weights)

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    k = min(k, weights.shape[-1])
    values, _ = jax.lax.top_k(weights, k=k)
    return values.sum(axis=-1)


def max_attention_weight(weights: jax.Array) -> jax.Array:
    """Return the maximum memory weight per query."""

    _validate_weights(weights)
    return weights.max(axis=-1)


def argmax_memory_index(weights: jax.Array) -> jax.Array:
    """Return the index of the strongest memory per query."""

    _validate_weights(weights)
    return weights.argmax(axis=-1)


def retrieval_distance(
    retrieved: jax.Array,
    target: jax.Array,
    *,
    p: float = 2.0,
) -> jax.Array:
    """Compute vector distance between retrieved vectors and target vectors.

    Args:
        retrieved: Tensor with shape [batch_size, dim].
        target: Tensor with shape [batch_size, dim].
        p: Norm order.

    Returns:
        Distance tensor with shape [batch_size].
    """

    if retrieved.shape != target.shape:
        raise ValueError(
            "retrieved and target must have the same shape, got "
            f"{tuple(retrieved.shape)} and {tuple(target.shape)}."
        )

    return jnp.linalg.vector_norm(retrieved - target, ord=p, axis=-1)


def selected_memory_distance(
    retrieved: jax.Array,
    memory: jax.Array,
    weights: jax.Array,
    *,
    p: float = 2.0,
) -> jax.Array:
    """Distance from retrieved vector to the highest-weight memory vector."""

    _validate_weights(weights)

    if retrieved.ndim != 2:
        raise ValueError(
            f"retrieved must have shape [batch_size, dim], got {tuple(retrieved.shape)}."
        )

    if memory.ndim != 2:
        raise ValueError(
            f"memory must have shape [num_memories, dim], got {tuple(memory.shape)}."
        )

    if retrieved.shape[-1] != memory.shape[-1]:
        raise ValueError(
            "retrieved and memory must share the same feature dimension, got "
            f"{retrieved.shape[-1]} and {memory.shape[-1]}."
        )

    top_indices = argmax_memory_index(weights)
    selected = memory[top_indices]
    return retrieval_distance(retrieved, selected, p=p)


def topk_recall(
    weights: jax.Array,
    target_indices: jax.Array,
    *,
    k: int,
) -> jax.Array:
    """Check whether the target memory index is inside the top-k retrieved memories.

    Args:
        weights: Softmax memory weights with shape [batch_size, num_memories].
        target_indices: Correct memory indices with shape [batch_size].
        k: Number of top memories to check.

    Returns:
        Float tensor with shape [batch_size], where 1.0 means the target
        memory is in the top-k and 0.0 means it is not.
    """

    _validate_weights(weights)

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    if target_indices.ndim != 1:
        raise ValueError(
            "target_indices must have shape [batch_size], "
            f"got {tuple(target_indices.shape)}."
        )

    if target_indices.shape[0] != weights.shape[0]:
        raise ValueError(
            "target_indices batch size must match weights batch size: "
            f"{target_indices.shape[0]} != {weights.shape[0]}."
        )

    k = min(k, weights.shape[-1])
    _, topk_indices = jax.lax.top_k(weights, k=k)
    expanded_targets = jnp.broadcast_to(target_indices[:, None], topk_indices.shape)
    return (topk_indices == expanded_targets).any(axis=-1).astype(jnp.float32)


def build_retrieval_diagnostics(
    *,
    query: jax.Array,
    memory: jax.Array,
    retrieved: jax.Array,
    weights: jax.Array,
    beta: float,
    num_updates: int,
    normalize: bool,
    top_k: int = 3,
    detach: bool = True,
) -> dict[str, Any]:
    """Build standard Hopfield retrieval diagnostics."""

    entropy = attention_entropy(weights)
    norm_entropy = attention_entropy(weights, normalize=True)
    top1 = topk_mass(weights, k=1)
    topk = topk_mass(weights, k=top_k)
    max_weight = max_attention_weight(weights)
    argmax_index = argmax_memory_index(weights)
    query_distance = retrieval_distance(retrieved, query)
    selected_distance = selected_memory_distance(retrieved, memory, weights)

    diagnostics: dict[str, Any] = {
        "beta": beta,
        "num_updates": num_updates,
        "normalize": normalize,
        "query_shape": tuple(query.shape),
        "memory_shape": tuple(memory.shape),
        "retrieved_shape": tuple(retrieved.shape),
        "weights_shape": tuple(weights.shape),
        "top_k": min(top_k, weights.shape[-1]),
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "top1_mass": top1,
        "topk_mass": topk,
        "max_attention_weight": max_weight,
        "argmax_memory_index": argmax_index,
        "query_retrieved_distance": query_distance,
        "selected_memory_distance": selected_distance,
        "entropy_mean": entropy.mean(),
        "top1_mass_mean": top1.mean(),
        "topk_mass_mean": topk.mean(),
        "selected_memory_distance_mean": selected_distance.mean(),
    }

    if detach:
        diagnostics = {
            key: value if isinstance(value, jax.Array) else value
            for key, value in diagnostics.items()
        }

    return diagnostics
