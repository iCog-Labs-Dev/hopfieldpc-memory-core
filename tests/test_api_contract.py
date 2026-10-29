import pytest
import jax
import jax.numpy as jnp

from hopfieldpc import HopfieldMemory, HopfieldShapeError


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

def test_external_memory_shape_contract() -> None:
    module = HopfieldMemory(dim=4, beta=1.0)

    query = _randn(8, 4)
    memory = _randn(10, 4)

    contract = module.validate_inputs(query=query, memory=memory)

    assert contract.batch_size == 8
    assert contract.dim == 4
    assert contract.memory_size == 10


def test_internal_fixed_memory_shape_contract() -> None:
    module = HopfieldMemory(dim=4, memory_size=7, learnable_memory=False)

    query = _randn(8, 4)

    contract = module.validate_inputs(query=query)

    assert contract.batch_size == 8
    assert contract.dim == 4
    assert contract.memory_size == 7
    assert isinstance(module.memory, jax.Array)
    assert module.memory not in module.parameters()


def test_internal_learnable_memory_shape_contract() -> None:
    module = HopfieldMemory(dim=4, memory_size=7, learnable_memory=True)

    query = _randn(8, 4)

    contract = module.validate_inputs(query=query)

    assert contract.batch_size == 8
    assert contract.dim == 4
    assert contract.memory_size == 7
    assert isinstance(module.memory, jax.Array)
    assert module.memory in module.parameters()


def test_missing_memory_raises_error() -> None:
    module = HopfieldMemory(dim=4)

    query = _randn(8, 4)

    with pytest.raises(ValueError, match="No memory was provided"):
        module.validate_inputs(query=query)


def test_query_must_be_2d() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(8, 4, 1)
    memory = _randn(10, 4)

    with pytest.raises(HopfieldShapeError, match="query must have shape"):
        module.validate_inputs(query=query, memory=memory)


def test_memory_must_be_2d() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(8, 4)
    memory = _randn(10, 4, 1)

    with pytest.raises(HopfieldShapeError, match="memory must have shape"):
        module.validate_inputs(query=query, memory=memory)


def test_query_and_memory_dim_must_match() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(8, 4)
    memory = _randn(10, 5)

    with pytest.raises(HopfieldShapeError, match="query and memory must share"):
        module.validate_inputs(query=query, memory=memory)


def test_query_dim_must_match_module_dim() -> None:
    module = HopfieldMemory(dim=4)
    query = _randn(8, 5)
    memory = _randn(10, 5)

    with pytest.raises(HopfieldShapeError, match="expected dim=4"):
        module.validate_inputs(query=query, memory=memory)


def test_invalid_init_values_raise_errors() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        HopfieldMemory(dim=0)

    with pytest.raises(ValueError, match="memory_size must be"):
        HopfieldMemory(dim=4, memory_size=0)

    with pytest.raises(ValueError, match="beta must be positive"):
        HopfieldMemory(dim=4, beta=0.0)

    with pytest.raises(ValueError, match="num_updates must be"):
        HopfieldMemory(dim=4, num_updates=0)

    with pytest.raises(ValueError, match="learnable_memory=True requires"):
        HopfieldMemory(dim=4, learnable_memory=True)
