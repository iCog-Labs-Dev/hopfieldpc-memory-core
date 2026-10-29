import pytest
import jax
import jax.numpy as jnp

from hopfieldpc import (


    HopfieldMemory,
    HopfieldMemoryOutput,
    HopfieldShapeError,
    hopfield_retrieve,
)



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

def test_memory_forward_output_contract_and_shapes() -> None:
    module = HopfieldMemory(dim=64, beta=2.0)
    query = _randn(8, 64)
    memory = _randn(20, 64)

    output = module(query, memory)
    retrieved, weights, diagnostics = output

    assert isinstance(output, HopfieldMemoryOutput)
    assert retrieved.shape == (8, 64)
    assert weights.shape == (8, 20)
    assert isinstance(diagnostics, dict)


def test_named_output_matches_unpacked_output() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(2, 4)
    memory = _randn(3, 4)

    output = module(query, memory)
    retrieved, weights, diagnostics = output

    assert output.retrieved is retrieved
    assert output.weights is weights
    assert output.diagnostics is diagnostics


def test_external_memory_takes_precedence_over_internal_memory() -> None:
    module = HopfieldMemory(dim=4, memory_size=3)
    query = _randn(2, 4)
    external_memory = _randn(5, 4)

    _, weights, diagnostics = module(query, external_memory)

    assert weights.shape == (2, 5)
    assert diagnostics["memory_shape"] == (5, 4)


def test_module_matches_functional_retrieval() -> None:
    module = HopfieldMemory(dim=4, beta=2.0)
    query = _randn(2, 4)
    memory = _randn(3, 4)

    expected_retrieved, expected_weights = hopfield_retrieve(
        query,
        memory,
        beta=2.0,
    )
    retrieved, weights, _ = module(query, memory)

    assert jnp.allclose(retrieved, expected_retrieved)
    assert jnp.allclose(weights, expected_weights)


def test_disabled_diagnostics_preserves_output_contract() -> None:
    module = HopfieldMemory(dim=4, return_diagnostics=False)
    query = _randn(2, 4)
    memory = _randn(3, 4)

    retrieved, weights, diagnostics = module(query, memory)

    assert retrieved.shape == (2, 4)
    assert weights.shape == (2, 3)
    assert diagnostics == {}


def test_forward_enforces_shape_contract() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(2, 5)
    memory = _randn(3, 5)

    with pytest.raises(HopfieldShapeError, match="expected dim=4"):
        module(query, memory)
