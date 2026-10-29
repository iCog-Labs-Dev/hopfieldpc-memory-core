import jax
import jax.numpy as jnp

from hopfieldpc import hopfield_retrieve





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

def test_higher_beta_gives_sharper_retrieval() -> None:
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query = jnp.array([[1.0, 0.0]])

    _, low_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=0.1)
    _, high_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=10.0)

    low_beta_top_weight = low_beta_weights.max(axis=-1).item()
    high_beta_top_weight = high_beta_weights.max(axis=-1).item()

    assert high_beta_top_weight > low_beta_top_weight
    assert high_beta_top_weight > 0.99


def test_low_beta_is_more_uniform_than_high_beta() -> None:
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = jnp.array([[1.0, 0.0]])

    _, low_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=0.01)
    _, high_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=10.0)

    low_beta_range = (
        low_beta_weights.max(axis=-1) - low_beta_weights.min(axis=-1)
    ).item()
    high_beta_range = (
        high_beta_weights.max(axis=-1) - high_beta_weights.min(axis=-1)
    ).item()

    assert high_beta_range > low_beta_range


def test_normalized_scoring_still_returns_original_memory_values() -> None:
    memory = jnp.array(
        [
            [10.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query = jnp.array([[1.0, 0.0]])

    retrieved, weights = hopfield_retrieve(
        query=query,
        memory=memory,
        beta=10.0,
        normalize=True,
    )

    assert weights[0, 0] > 0.99
    assert retrieved[0, 0] > 9.9
