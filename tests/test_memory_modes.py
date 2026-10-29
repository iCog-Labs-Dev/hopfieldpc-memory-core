"""Tests for HopfieldMemory memory mode options.

Validates three memory modes:
- External memory passed during forward
- Internal fixed memory (registered buffer, no gradient)
- Internal learnable memory (jax.Array represented by the module parameter list)
"""

import pytest
import jax
import jax.numpy as jnp

from hopfieldpc import HopfieldMemory





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

def test_external_memory_works() -> None:
    module = HopfieldMemory(dim=4, beta=1.0)
    query = _randn(8, 4)
    memory = _randn(10, 4)

    output = module(query, memory)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 10)
    assert module.memory is None


def test_external_memory_takes_precedence_over_internal() -> None:
    module = HopfieldMemory(dim=4, memory_size=3, beta=1.0)
    query = _randn(2, 4)
    external = _randn(7, 4)

    _, weights, diag = module(query, external)

    assert weights.shape == (2, 7)
    assert diag["memory_shape"] == (7, 4)


def test_internal_fixed_memory_works() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)
    query = _randn(8, 4)

    output = module(query)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 5)


def test_internal_fixed_memory_is_buffer() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)

    assert isinstance(module.memory, jax.Array)


def test_fixed_memory_does_not_require_gradient() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)

    assert module.memory not in module.parameters()


def test_fixed_memory_shape() -> None:
    module = HopfieldMemory(dim=8, memory_size=12, learnable_memory=False)

    assert module.memory.shape == (12, 8)


def test_internal_learnable_memory_works() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)
    query = _randn(8, 4)

    output = module(query)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 5)


def test_learnable_memory_is_parameter() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)

    assert isinstance(module.memory, jax.Array)


def test_learnable_memory_is_listed_as_parameter() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)

    assert module.memory in module.parameters()


def test_learnable_memory_shape() -> None:
    module = HopfieldMemory(dim=8, memory_size=12, learnable_memory=True)

    assert module.memory.shape == (12, 8)


def test_learnable_memory_appears_in_parameters() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)
    param_list = list(module.parameters())

    assert len(param_list) == 1
    assert param_list[0] is module.memory


def test_fixed_memory_not_in_parameters() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)
    param_list = list(module.parameters())

    assert len(param_list) == 0


def test_no_memory_raises_on_forward() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(2, 4)

    with pytest.raises(ValueError, match="No memory was provided"):
        module(query)


def test_learnable_memory_requires_memory_size() -> None:
    with pytest.raises(ValueError, match="learnable_memory=True requires"):
        HopfieldMemory(dim=4, learnable_memory=True)


def test_learnable_memory_gradient_flows() -> None:
    module = HopfieldMemory(dim=3, memory_size=4, learnable_memory=True, beta=1.0)
    query = _randn(2, 3)
    grad = jax.grad(lambda memory: module(query, memory).retrieved.sum())(module.memory)
    assert grad is not None
    assert jnp.isfinite(grad).all()
