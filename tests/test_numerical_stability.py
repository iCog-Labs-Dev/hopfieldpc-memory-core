"""Numerical-stability tests for Hopfield retrieval."""

import math
from typing import Any

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hopfieldpc import (


    HopfieldMemory,
    attention_entropy,
    hopfield_energy,
    hopfield_retrieve,
    multi_step_retrieve,
)

DTYPE_TOLERANCES = {
    jnp.float16: (1e-3, 1e-3),
    jnp.bfloat16: (1e-2, 1e-2),
    jnp.float32: (1e-6, 1e-5),
    jnp.float64: (1e-12, 1e-10),
}



_TEST_KEY = jax.random.PRNGKey(42)

def _set_seed(seed: int) -> None:
    global _TEST_KEY
    _TEST_KEY = jax.random.PRNGKey(seed)

def _generator(seed: int = 42) -> object:
    return None

def _randn(*shape, **kwargs):
    global _TEST_KEY
    dtype = kwargs.get("dtype", jnp.float32)
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    _TEST_KEY, key = jax.random.split(_TEST_KEY)
    return jax.random.normal(key, shape, dtype=dtype)

def _randn_like(x):
    return _randn(*x.shape, dtype=x.dtype)

def _randint(*args, **kwargs):
    global _TEST_KEY
    if len(args) >= 3:
        low, high, shape = args[0], args[1], args[2]
    else:
        low, high, shape = 0, args[0], args[1]
    dtype = kwargs.get("dtype", jnp.int32)
    _TEST_KEY, key = jax.random.split(_TEST_KEY)
    return jax.random.randint(key, shape, low, high, dtype=dtype)

def _randperm(n):
    global _TEST_KEY
    _TEST_KEY, key = jax.random.split(_TEST_KEY)
    return jax.random.permutation(key, n)

def _cdist(a, b, p=2.0):
    return jnp.linalg.vector_norm(a[:, None, :] - b[None, :, :], ord=p, axis=-1)

def _topk(x, k, dim=-1):
    return jax.lax.top_k(x, k=k)

def _assert_probability_distribution(weights: jax.Array) -> None:
    if weights.dtype == jnp.float16:
        atol, rtol = DTYPE_TOLERANCES[jnp.float16]
    elif weights.dtype == jnp.bfloat16:
        atol, rtol = DTYPE_TOLERANCES[jnp.bfloat16]
    elif weights.dtype == jnp.float64:
        atol, rtol = DTYPE_TOLERANCES[jnp.float64]
    else:
        atol, rtol = DTYPE_TOLERANCES[jnp.float32]
    assert jnp.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    _assert_close(
        weights.sum(axis=-1),
        jnp.ones(weights.shape[0], dtype=weights.dtype),
        atol=atol,
        rtol=rtol,
    )

def _assert_close(actual, desired, atol=1e-6, rtol=1e-6):
    assert jnp.allclose(actual, desired, atol=atol, rtol=rtol), (actual, desired)


def _assert_diagnostics_finite(diagnostics: dict[str, Any]) -> None:
    for value in diagnostics.values():
        tensors = value if isinstance(value, list) else [value]
        for tensor in tensors:
            if isinstance(tensor, jax.Array) and jnp.issubdtype(tensor.dtype, jnp.inexact):
                assert jnp.isfinite(tensor).all()


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_retrieval_is_finite_across_beta_sweep(beta: float, normalize: bool) -> None:
    generator = _generator(42)
    query = _randn(8, 64, generator=generator)
    memory = _randn(20, 64, generator=generator)

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=beta,
        normalize=normalize,
    )

    assert jnp.isfinite(retrieved).all()
    _assert_probability_distribution(weights)


@pytest.mark.parametrize("beta", [20.0, 50.0, 100.0])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("num_updates", [1, 3])
def test_module_diagnostics_are_finite(
    beta: float,
    normalize: bool,
    num_updates: int,
) -> None:
    generator = _generator(7)
    query = _randn(4, 16, generator=generator)
    memory = _randn(10, 16, generator=generator)
    module = HopfieldMemory(
        dim=16,
        beta=beta,
        normalize=normalize,
        num_updates=num_updates,
        return_states=True,
    )

    output = module(query, memory)

    assert jnp.isfinite(output.retrieved).all()
    _assert_probability_distribution(output.weights)
    _assert_diagnostics_finite(output.diagnostics)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_large_low_precision_scores_use_safe_working_precision(
    dtype: jnp.dtype,
) -> None:
    query = jnp.full((1, 64), 5.0, dtype=dtype)
    memory = jnp.stack(
        [
            jnp.full((64,), 5.0, dtype=dtype),
            jnp.zeros(64, dtype=dtype),
        ]
    )

    retrieved, weights = hopfield_retrieve(query, memory, beta=100.0)

    assert retrieved.dtype == dtype
    assert weights.dtype == dtype
    assert jnp.isfinite(retrieved).all()
    _assert_probability_distribution(weights)
    assert weights.argmax(axis=-1).item() == 0


def test_centering_prevents_beta_scaling_overflow() -> None:
    query = jnp.array([[1e19, 0.0]])
    memory = jnp.array([[1e18, 0.0], [0.0, 1.0]])

    retrieved, weights = hopfield_retrieve(query, memory, beta=100.0)

    assert jnp.isfinite(retrieved).all()
    assert jnp.array_equal(weights, jnp.array([[1.0, 0.0]]))


def test_energy_is_finite_when_beta_scaling_would_overflow() -> None:
    query = jnp.array([[1e19, 0.0]])
    memory = jnp.array([[1e18, 0.0], [0.0, 1.0]])

    energy = hopfield_energy(query, memory, beta=100.0)

    assert jnp.isfinite(energy).all()
    expected = -(query @ memory.T).max(axis=-1) + 0.5 * (query ** 2).sum(axis=-1)
    _assert_close(energy, expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_energy_is_finite_for_large_scores(dtype: jnp.dtype) -> None:
    query = jnp.full((1, 64), 5.0, dtype=dtype)
    memory = jnp.stack(
        [
            jnp.full((64,), 5.0, dtype=dtype),
            jnp.zeros(64, dtype=dtype),
        ]
    )

    energy = hopfield_energy(query, memory, beta=100.0)

    assert energy.dtype == dtype
    # -1600 from the best match, +800 from the query term.
    _assert_close(energy, jnp.array([-800.0], dtype=dtype))


@pytest.mark.parametrize("beta", [0.1, 1.0, 10.0])
def test_energy_matches_float64_reference(beta: float) -> None:
    generator = _generator(5)
    query = _randn(8, 16, generator=generator, dtype=jnp.float64)
    memory = _randn(20, 16, generator=generator, dtype=jnp.float64)

    energy = hopfield_energy(query, memory, beta=beta)
    reference = (
        -jax.scipy.special.logsumexp(beta * query @ memory.T, axis=-1) / beta
        + 0.5 * (query ** 2).sum(axis=-1)
    )
    grad = jax.grad(lambda q: hopfield_energy(q, memory, beta=beta).sum())(query)
    reference_grad = jax.grad(
        lambda q: (
            -jax.scipy.special.logsumexp(beta * q @ memory.T, axis=-1) / beta
            + 0.5 * (q ** 2).sum(axis=-1)
        ).sum()
    )(query)

    _assert_close(energy, reference, atol=1e-12, rtol=1e-10)
    _assert_close(grad, reference_grad, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("beta", [0.5, 2.0, 10.0])
def test_stabilized_retrieval_matches_float64_reference(
    beta: float,
    normalize: bool,
    dtype: jnp.dtype,
) -> None:
    generator = _generator(11)
    query = (_randn(4, 8, generator=generator) * 0.1).astype(dtype)
    memory = (_randn(6, 8, generator=generator) * 0.1).astype(dtype)

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=beta,
        normalize=normalize,
    )

    query64 = query.astype(jnp.float64)
    memory64 = memory.astype(jnp.float64)
    if normalize:
        query_norm = jnp.maximum(jnp.linalg.norm(query64, axis=-1, keepdims=True), 1e-8)
        memory_norm = jnp.maximum(jnp.linalg.norm(memory64, axis=-1, keepdims=True), 1e-8)
        query_scores = query64 / query_norm
        memory_scores = memory64 / memory_norm
    else:
        query_scores, memory_scores = query64, memory64
    reference_weights = jax.nn.softmax(beta * query_scores @ memory_scores.T, axis=-1)
    reference_retrieved = reference_weights @ memory64

    atol, rtol = {
        jnp.float32: (1e-6, 1e-5),
        jnp.float64: (1e-12, 1e-10),
    }[dtype]
    assert weights.dtype == dtype
    assert retrieved.dtype == dtype
    _assert_close(
        weights.astype(jnp.float64), reference_weights, atol=atol, rtol=rtol
    )
    _assert_close(
        retrieved.astype(jnp.float64), reference_retrieved, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]
)
def test_zero_query_has_uniform_normalized_weights(dtype: jnp.dtype) -> None:
    query = jnp.zeros((2, 4), dtype=dtype)
    memory = jnp.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=dtype,
    )

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=True,
    )

    expected_weights = jnp.full((2, 2), 0.5, dtype=dtype)
    atol, rtol = DTYPE_TOLERANCES[dtype]
    _assert_close(weights, expected_weights, atol=atol, rtol=rtol)
    assert jnp.isfinite(retrieved).all()


@pytest.mark.parametrize(
    "dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]
)
@pytest.mark.parametrize("query_fill", [0.0, 1.0])
def test_all_zero_memory_is_finite_with_normalization(
    query_fill: float,
    dtype: jnp.dtype,
) -> None:
    query = jnp.full((2, 4), query_fill, dtype=dtype)
    memory = jnp.zeros((3, 4), dtype=dtype)

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=True,
    )

    assert jnp.array_equal(retrieved, jnp.zeros_like(retrieved))
    _assert_probability_distribution(weights)
    atol, rtol = DTYPE_TOLERANCES[dtype]
    _assert_close(
        weights,
        jnp.full((2, 3), 1.0 / 3.0, dtype=dtype),
        atol=atol,
        rtol=rtol,
    )


def test_normalized_weights_are_scale_invariant() -> None:
    generator = _generator(23)
    query = _randn(4, 8, generator=generator)
    memory = _randn(6, 8, generator=generator)

    _, weights = hopfield_retrieve(query, memory, beta=10.0, normalize=True)
    _, scaled_weights = hopfield_retrieve(
        query * 10_000.0,
        memory * 10_000.0,
        beta=10.0,
        normalize=True,
    )

    _assert_close(weights, scaled_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("scale", [1.0, 100.0, 10_000.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_large_float32_norms_remain_finite(scale: float, normalize: bool) -> None:
    generator = _generator(31)
    query = _randn(4, 16, generator=generator) * scale
    memory = _randn(10, 16, generator=generator) * scale

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=normalize,
    )

    assert jnp.isfinite(retrieved).all()
    _assert_probability_distribution(weights)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_internal_low_precision_memory_path_is_finite(dtype: jnp.dtype) -> None:
    module = HopfieldMemory(dim=64, memory_size=2, beta=100.0)
    assert module.memory is not None
    module.memory = module.memory.astype(dtype).at[0].set(5.0).at[1].set(0.0)
    query = jnp.full((1, 64), 5.0, dtype=dtype)

    output = module(query)

    assert output.retrieved.dtype == dtype
    assert output.weights.dtype == dtype
    assert jnp.isfinite(output.retrieved).all()
    _assert_probability_distribution(output.weights)
    _assert_diagnostics_finite(output.diagnostics)


def test_internal_memory_dtype_mismatch_raises() -> None:
    module = HopfieldMemory(dim=4, memory_size=3)
    query = _randn(2, 4, dtype=jnp.float16)

    with pytest.raises(TypeError, match="must share a dtype"):
        module(query)


def test_external_memory_dtype_mismatch_raises() -> None:
    query = _randn(2, 4, dtype=jnp.float32)
    memory = _randn(3, 4, dtype=jnp.float64)

    with pytest.raises(TypeError, match="must share a dtype"):
        hopfield_retrieve(query, memory)


@pytest.mark.parametrize(
    "dtype", [jnp.int64, jnp.int32, jnp.bool_, jnp.complex64]
)
def test_non_floating_tensors_are_rejected(dtype: jnp.dtype) -> None:
    query = jnp.ones((2, 4), dtype=dtype)
    memory = jnp.ones((3, 4), dtype=dtype)

    with pytest.raises(TypeError, match="must be floating-point tensors"):
        hopfield_retrieve(query, memory)

    with pytest.raises(TypeError, match="must be floating-point tensors"):
        HopfieldMemory(dim=4)(query, memory)

    with pytest.raises(TypeError, match="must be floating-point tensors"):
        hopfield_energy(query, memory)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_multi_step_states_are_finite(dtype: jnp.dtype) -> None:
    query = jnp.full((1, 64), 5.0, dtype=dtype)
    memory = jnp.stack(
        [
            jnp.full((64,), 5.0, dtype=dtype),
            jnp.zeros(64, dtype=dtype),
        ]
    )

    retrieved, weights, states = multi_step_retrieve(
        query,
        memory,
        beta=100.0,
        num_updates=10,
        return_states=True,
    )

    assert len(states) == 11
    for state in states:
        assert state.dtype == dtype
        assert state.shape == query.shape
        assert jnp.isfinite(state).all()
    assert retrieved.dtype == dtype
    assert jnp.isfinite(retrieved).all()
    _assert_probability_distribution(weights)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_large_beta_gradients_are_finite(
    normalize: bool,
    dtype: jnp.dtype,
) -> None:
    generator = _generator(47)
    # Unscaled vectors saturate the softmax at beta=100.
    query = _randn(4, 64, generator=generator).astype(dtype)
    memory = _randn(10, 64, generator=generator).astype(dtype)
    def loss(q, m):
        retrieved, weights = hopfield_retrieve(q, m, beta=100.0, normalize=normalize)
        return (retrieved.astype(jnp.float32)**2).mean() + (weights.astype(jnp.float32)**2).mean()
    query_grad = jax.grad(lambda q: loss(q, memory))(query)
    memory_grad = jax.grad(lambda m: loss(query, m))(memory)
    assert jnp.isfinite(query_grad).all()
    assert jnp.isfinite(memory_grad).all()


def test_large_beta_learnable_memory_gradient_is_finite() -> None:
    module = HopfieldMemory(
        dim=4,
        memory_size=3,
        beta=100.0,
        learnable_memory=True,
    )
    query = _randn(2, 4) * 0.1

    grad = jax.grad(lambda memory: (module(query, memory).retrieved**2).mean())(module.memory)
    assert grad is not None
    assert jnp.isfinite(grad).all()


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_entropy_and_gradients_are_finite(dtype: jnp.dtype) -> None:
    weights = jnp.array([[1.0, 0.0, 0.0]], dtype=dtype)

    entropy = attention_entropy(weights)
    normalized_entropy = attention_entropy(weights, normalize=True)
    grad = jax.grad(lambda w: (attention_entropy(w) + attention_entropy(w, normalize=True)).sum())(weights)

    assert entropy.dtype == jnp.float32
    assert jnp.isfinite(entropy).all()
    assert jnp.isfinite(normalized_entropy).all()
    assert jnp.isfinite(grad).all()


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_uniform_entropy_is_correct(dtype: jnp.dtype) -> None:
    weights = jnp.full((1, 4), 0.25, dtype=dtype)

    entropy = attention_entropy(weights)
    normalized_entropy = attention_entropy(weights, normalize=True)

    _assert_close(
        entropy,
        jnp.full_like(entropy, math.log(4.0)),
        atol=1e-6,
        rtol=1e-5,
    )
    _assert_close(
        normalized_entropy,
        jnp.ones_like(normalized_entropy),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "beta",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "1.0"],
)
def test_invalid_beta_is_rejected_consistently(beta: Any) -> None:
    query = _randn(2, 4)
    memory = _randn(3, 4)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        hopfield_retrieve(query, memory, beta=beta)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        HopfieldMemory(dim=4, beta=beta)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        hopfield_energy(query, memory, beta=beta)


@pytest.mark.parametrize(
    "beta",
    [1, 1.0, np.float32(1.0), np.float64(1.0)],
)
def test_real_numeric_beta_types_are_accepted(beta: Any) -> None:
    query = _randn(2, 4)
    memory = _randn(3, 4)

    retrieved, weights = hopfield_retrieve(query, memory, beta=beta)
    module = HopfieldMemory(dim=4, beta=beta)
    energy = hopfield_energy(query, memory, beta=beta)

    assert module.beta == 1.0
    assert jnp.isfinite(retrieved).all()
    _assert_probability_distribution(weights)
    assert jnp.isfinite(energy).all()
