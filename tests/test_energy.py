"""Tests for the Hopfield energy computation module."""

import math

import pytest
import jax
import jax.numpy as jnp

from hopfieldpc import HopfieldShapeError, hopfield_energy





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

def _assert_close(actual, desired, atol=1e-6, rtol=1e-6):
    assert jnp.allclose(actual, desired, atol=atol, rtol=rtol), (actual, desired)

def test_energy_returns_finite_values() -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    energy = hopfield_energy(query, memory, beta=1.0)

    assert jnp.isfinite(energy).all()


def test_energy_shape_per_query() -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    energy = hopfield_energy(query, memory, beta=1.0, reduce="none")

    assert energy.shape == (8,)


def test_energy_shape_mean() -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    energy = hopfield_energy(query, memory, beta=1.0, reduce="mean")

    assert energy.ndim == 0


@pytest.mark.parametrize(
    "batch_size, dim, num_memories",
    [(1, 2, 3), (4, 8, 5), (16, 64, 50), (32, 128, 100)],
)
def test_energy_shape_parametrized(
    batch_size: int, dim: int, num_memories: int
) -> None:
    query = _randn(batch_size, dim)
    memory = _randn(num_memories, dim)

    energy = hopfield_energy(query, memory, beta=1.0)

    assert energy.shape == (batch_size,)


@pytest.mark.parametrize(
    "beta", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
)
def test_no_nan_inf_for_tested_beta_values(beta: float) -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    energy = hopfield_energy(query, memory, beta=beta)

    assert jnp.isfinite(energy).all()


@pytest.mark.parametrize(
    "beta", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
)
def test_no_nan_inf_with_constants(beta: float) -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    energy = hopfield_energy(query, memory, beta=beta, include_constants=True)

    assert jnp.isfinite(energy).all()


def test_energy_with_constants_differs() -> None:
    query = _randn(4, 3)
    memory = _randn(5, 3)

    energy_no_const = hopfield_energy(query, memory, beta=1.0)
    energy_with_const = hopfield_energy(
        query, memory, beta=1.0, include_constants=True
    )

    assert not jnp.allclose(energy_no_const, energy_with_const)


def test_energy_constants_add_correct_offset() -> None:
    _set_seed(99)
    query = _randn(4, 3)
    memory = _randn(5, 3)
    beta = 2.0

    energy_no_const = hopfield_energy(query, memory, beta=beta)
    energy_with_const = hopfield_energy(
        query, memory, beta=beta, include_constants=True
    )

    expected_offset = (
        (1.0 / beta) * math.log(memory.shape[0])
        + 0.5 * (memory ** 2).sum(axis=-1).max()
    )
    actual_diff = energy_with_const - energy_no_const

    assert jnp.allclose(actual_diff, expected_offset, atol=1e-5)


def test_energy_decreases_for_closer_query() -> None:
    memory = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    query_close = jnp.array([[0.95, 0.05, 0.0]])
    query_far = jnp.array([[0.33, 0.33, 0.33]])

    energy_close = hopfield_energy(query_close, memory, beta=5.0)
    energy_far = hopfield_energy(query_far, memory, beta=5.0)

    assert energy_close.item() < energy_far.item()


def test_energy_invalid_beta_raises() -> None:
    query = _randn(2, 3)
    memory = _randn(4, 3)

    with pytest.raises(ValueError, match="beta must be positive"):
        hopfield_energy(query, memory, beta=0.0)

    with pytest.raises(ValueError, match="beta must be positive"):
        hopfield_energy(query, memory, beta=-1.0)


def test_energy_invalid_reduce_raises() -> None:
    query = _randn(2, 3)
    memory = _randn(4, 3)

    with pytest.raises(ValueError, match="reduce must be"):
        hopfield_energy(query, memory, beta=1.0, reduce="sum")


def test_energy_shape_validation() -> None:
    query_3d = _randn(2, 3, 1)
    memory = _randn(4, 3)

    with pytest.raises(HopfieldShapeError):
        hopfield_energy(query_3d, memory, beta=1.0)


def test_energy_dim_mismatch_raises() -> None:
    query = _randn(2, 3)
    memory = _randn(4, 5)

    with pytest.raises(HopfieldShapeError, match="must share the same"):
        hopfield_energy(query, memory, beta=1.0)


def test_energy_large_magnitude_inputs() -> None:
    query = _randn(8, 4) * 100.0
    memory = _randn(10, 4) * 100.0

    energy = hopfield_energy(query, memory, beta=10.0)

    assert jnp.isfinite(energy).all()


def test_energy_large_magnitude_with_constants() -> None:
    query = _randn(8, 4) * 100.0
    memory = _randn(10, 4) * 100.0

    energy = hopfield_energy(query, memory, beta=10.0, include_constants=True)

    assert jnp.isfinite(energy).all()


def test_reduce_mean_matches_manual_mean() -> None:
    _set_seed(42)
    query = _randn(8, 4)
    memory = _randn(10, 4)

    per_query = hopfield_energy(query, memory, beta=2.0, reduce="none")
    mean_energy = hopfield_energy(query, memory, beta=2.0, reduce="mean")

    assert jnp.allclose(mean_energy, per_query.mean(), atol=1e-6)
