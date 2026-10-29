"""Tests for optional multi-step Hopfield retrieval."""

import pytest
import jax
import jax.numpy as jnp

from hopfieldpc import HopfieldMemory, hopfield_retrieve
from hopfieldpc.functional import multi_step_retrieve





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

def test_default_num_updates_is_one() -> None:
    assert HopfieldMemory(dim=4).num_updates == 1


def test_one_step_matches_hopfield_retrieve() -> None:
    q, m = _randn(3, 4), _randn(6, 4)
    r1, w1 = hopfield_retrieve(q, m, beta=2.0)
    r2, w2 = multi_step_retrieve(q, m, beta=2.0, num_updates=1)
    assert jnp.allclose(r1, r2) and jnp.allclose(w1, w2)


@pytest.mark.parametrize("steps", [1, 2, 3, 10])
def test_shapes_valid_after_multiple_updates(steps: int) -> None:
    module = HopfieldMemory(dim=8, beta=2.0, num_updates=steps)
    out = module(_randn(5, 8), _randn(12, 8))
    assert out.retrieved.shape == (5, 8)
    assert out.weights.shape == (5, 12)


def test_no_nan_or_inf() -> None:
    module = HopfieldMemory(dim=16, beta=8.0, num_updates=20)
    out = module(_randn(4, 16) * 5, _randn(10, 16))
    assert jnp.isfinite(out.retrieved).all()
    assert jnp.isfinite(out.weights).all()


def test_weights_sum_to_one_after_multi_step() -> None:
    module = HopfieldMemory(dim=8, beta=2.0, num_updates=3)
    out = module(_randn(4, 8), _randn(9, 8))
    assert jnp.allclose(out.weights.sum(-1), jnp.ones(4), atol=1e-5)


def test_distance_to_nearest_memory_does_not_explode() -> None:
    _set_seed(0)
    memory = _randn(10, 16)
    query = memory[:4] + 0.1 * _randn(4, 16)
    out = HopfieldMemory(dim=16, beta=4.0, num_updates=5)(query, memory)
    q_end = _cdist(out.retrieved, memory).min(axis=-1)
    assert jnp.isfinite(q_end).all()
    # retrieved state is a convex combination of memories, so its norm is bounded
    retrieved_norm = jnp.linalg.norm(out.retrieved, axis=-1).max()
    memory_norm = jnp.linalg.norm(memory, axis=-1).max()
    assert retrieved_norm <= memory_norm + 1e-4


def test_multi_step_moves_toward_clear_memory_in_controlled_case() -> None:
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = jnp.array([[0.8, 0.1]])
    start_dist = _cdist(query, memory[:1]).item()
    out = HopfieldMemory(dim=2, beta=8.0, num_updates=3)(query, memory)
    end_dist = _cdist(out.retrieved, memory[:1]).item()
    assert end_dist < start_dist


def test_return_states_gives_q0_to_qT() -> None:
    q, m = _randn(2, 4), _randn(5, 4)
    r, _, states = multi_step_retrieve(q, m, beta=2.0, num_updates=3, return_states=True)
    assert len(states) == 4
    assert jnp.array_equal(states[0], q)
    assert jnp.array_equal(states[-1], r)


def test_module_exposes_states_in_diagnostics() -> None:
    module = HopfieldMemory(dim=4, beta=2.0, num_updates=3, return_states=True)
    out = module(_randn(2, 4), _randn(5, 4))
    assert len(out.diagnostics["states"]) == 4


def test_states_not_returned_by_default() -> None:
    out = HopfieldMemory(dim=4, num_updates=3)(_randn(2, 4), _randn(5, 4))
    assert "states" not in out.diagnostics


def test_return_states_respects_disabled_diagnostics() -> None:
    module = HopfieldMemory(
        dim=4,
        num_updates=3,
        return_states=True,
        return_diagnostics=False,
    )
    out = module(_randn(2, 4), _randn(5, 4))
    assert out.diagnostics == {}


def test_invalid_num_updates_raises() -> None:
    with pytest.raises(ValueError, match="num_updates must be"):
        multi_step_retrieve(_randn(2, 4), _randn(5, 4), num_updates=0)


def test_module_rejects_bool_num_updates() -> None:
    with pytest.raises(ValueError, match="num_updates must be"):
        HopfieldMemory(dim=4, num_updates=True)


def test_function_rejects_bool_num_updates() -> None:
    with pytest.raises(ValueError, match="num_updates must be"):
        multi_step_retrieve(_randn(2, 4), _randn(5, 4), num_updates=True)


def test_gradient_flows_through_multi_step() -> None:
    module = HopfieldMemory(dim=3, memory_size=4, learnable_memory=True, num_updates=3)
    query = _randn(2, 3)
    grad = jax.grad(lambda memory: module(query, memory).retrieved.sum())(module.memory)
    assert grad is not None
    assert jnp.isfinite(grad).all()
