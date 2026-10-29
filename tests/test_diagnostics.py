import math

import pytest
import jax
import jax.numpy as jnp

from hopfieldpc import HopfieldMemory
from hopfieldpc.diagnostics import (


    argmax_memory_index,
    attention_entropy,
    build_retrieval_diagnostics,
    max_attention_weight,
    retrieval_distance,
    selected_memory_distance,
    topk_mass,
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

def test_attention_entropy_low_for_one_hot_weights() -> None:
    weights = jnp.array([[1.0, 0.0, 0.0]])

    entropy = attention_entropy(weights)

    assert jnp.allclose(entropy, jnp.array([0.0]), atol=1e-6)


def test_attention_entropy_high_for_uniform_weights() -> None:
    weights = jnp.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])

    entropy = attention_entropy(weights)

    assert jnp.allclose(entropy, jnp.array([math.log(3.0)]), atol=1e-6)


def test_normalized_entropy_uniform_is_one() -> None:
    weights = jnp.array([[0.25, 0.25, 0.25, 0.25]])

    entropy = attention_entropy(weights, normalize=True)

    assert jnp.allclose(entropy, jnp.array([1.0]), atol=1e-6)


def test_topk_mass() -> None:
    weights = jnp.array([[0.5, 0.3, 0.2]])

    assert jnp.allclose(topk_mass(weights, k=1), jnp.array([0.5]))
    assert jnp.allclose(topk_mass(weights, k=2), jnp.array([0.8]))
    assert jnp.allclose(topk_mass(weights, k=10), jnp.array([1.0]))


def test_topk_invalid_k_raises() -> None:
    weights = jnp.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="k must be"):
        topk_mass(weights, k=0)

    with pytest.raises(ValueError, match="k must be"):
        topk_mass(weights, k=True)


def test_max_attention_weight_and_argmax() -> None:
    weights = jnp.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])

    assert jnp.allclose(max_attention_weight(weights), jnp.array([0.7, 0.8]))
    assert jnp.array_equal(argmax_memory_index(weights), jnp.array([1, 0]))


def test_retrieval_distance() -> None:
    retrieved = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    target = jnp.array([[0.0, 0.0], [0.0, 0.0]])

    distance = retrieval_distance(retrieved, target)

    assert jnp.allclose(distance, jnp.array([1.0, 1.0]))


def test_retrieval_distance_shape_mismatch_raises() -> None:
    retrieved = _randn(2, 3)
    target = _randn(2, 4)

    with pytest.raises(ValueError, match="same shape"):
        retrieval_distance(retrieved, target)


def test_selected_memory_distance() -> None:
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    retrieved = jnp.array([[0.9, 0.1]])
    weights = jnp.array([[0.8, 0.2]])

    distance = selected_memory_distance(retrieved, memory, weights)

    expected = jnp.linalg.vector_norm(jnp.array([[-0.1, 0.1]]), axis=-1)
    assert jnp.allclose(distance, expected)


def test_build_retrieval_diagnostics_contains_expected_keys() -> None:
    query = jnp.array([[0.9, 0.1]])
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    weights = jnp.array([[0.8, 0.2]])
    retrieved = weights @ memory

    diagnostics = build_retrieval_diagnostics(
        query=query,
        memory=memory,
        retrieved=retrieved,
        weights=weights,
        beta=2.0,
        num_updates=1,
        normalize=False,
        top_k=2,
    )

    expected_keys = {
        "beta",
        "num_updates",
        "normalize",
        "query_shape",
        "memory_shape",
        "retrieved_shape",
        "weights_shape",
        "top_k",
        "entropy",
        "normalized_entropy",
        "top1_mass",
        "topk_mass",
        "max_attention_weight",
        "argmax_memory_index",
        "query_retrieved_distance",
        "selected_memory_distance",
        "entropy_mean",
        "top1_mass_mean",
        "topk_mass_mean",
        "selected_memory_distance_mean",
    }

    assert expected_keys.issubset(diagnostics.keys())


def test_hopfield_memory_module_returns_full_diagnostics() -> None:
    module = HopfieldMemory(dim=2, beta=5.0, diagnostics_top_k=2)
    query = jnp.array([[0.9, 0.1]])
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    output = module(query, memory)

    assert "entropy" in output.diagnostics
    assert "top1_mass" in output.diagnostics
    assert "topk_mass" in output.diagnostics
    assert "selected_memory_distance" in output.diagnostics
    assert output.diagnostics["top_k"] == 2


def test_disabled_diagnostics_still_returns_empty_dict() -> None:
    module = HopfieldMemory(dim=2, return_diagnostics=False)
    query = _randn(2, 2)
    memory = _randn(3, 2)

    output = module(query, memory)

    assert output.diagnostics == {}


def test_diagnostics_are_compatible_with_jax_grad() -> None:
    module = HopfieldMemory(dim=2)
    query = _randn(2, 2)
    memory = _randn(3, 2)
    grad = jax.grad(lambda q: module(q, memory).retrieved.sum())(query)
    assert jnp.isfinite(grad).all()
