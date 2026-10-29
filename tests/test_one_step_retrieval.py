import jax
import jax.numpy as jnp

from hopfieldpc import HopfieldMemory, hopfield_retrieve





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

def test_functional_retrieval_output_shapes() -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    retrieved, weights = hopfield_retrieve(query=query, memory=memory, beta=1.0)

    assert retrieved.shape == (8, 4)
    assert weights.shape == (8, 10)


def test_functional_weights_sum_to_one() -> None:
    query = _randn(8, 4)
    memory = _randn(10, 4)

    _, weights = hopfield_retrieve(query=query, memory=memory, beta=1.0)

    expected = jnp.ones(8)
    assert jnp.allclose(weights.sum(axis=-1), expected, atol=1e-6)


def test_module_forward_output_shapes() -> None:
    module = HopfieldMemory(dim=4, beta=1.0)

    query = _randn(8, 4)
    memory = _randn(10, 4)

    output = module(query=query, memory=memory)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 10)
    assert isinstance(output.diagnostics, dict)


def test_nearest_memory_gets_highest_weight() -> None:
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = jnp.array([[0.95, 0.05]])

    _, weights = hopfield_retrieve(query=query, memory=memory, beta=5.0)

    top_index = int(weights.argmax(axis=-1).item())
    assert top_index == 0


def test_high_beta_retrieves_close_to_target_memory() -> None:
    memory = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = jnp.array([[0.95, 0.05]])

    retrieved, weights = hopfield_retrieve(query=query, memory=memory, beta=10.0)

    target = memory[0][None, :]

    assert weights[0, 0] > 0.99
    assert jnp.allclose(retrieved, target, atol=1e-2)


def test_module_uses_internal_fixed_memory() -> None:
    module = HopfieldMemory(dim=2, memory_size=3, beta=1.0, learnable_memory=False)

    query = _randn(4, 2)

    output = module(query=query)

    assert output.retrieved.shape == (4, 2)
    assert output.weights.shape == (4, 3)


def test_module_uses_internal_learnable_memory() -> None:
    module = HopfieldMemory(dim=2, memory_size=3, beta=1.0, learnable_memory=True)

    query = _randn(4, 2)

    output = module(query=query)

    assert output.retrieved.shape == (4, 2)
    assert output.weights.shape == (4, 3)
    assert module.memory in module.parameters()


def test_retrieval_is_differentiable_for_query_and_memory() -> None:
    query = _randn(4, 3)
    memory = _randn(5, 3)
    def loss(q, m):
        retrieved, weights = hopfield_retrieve(query=q, memory=m, beta=1.0)
        return (retrieved**2).mean() + (weights**2).mean()
    query_grad = jax.grad(lambda q: loss(q, memory))(query)
    memory_grad = jax.grad(lambda m: loss(query, m))(memory)
    assert jnp.isfinite(query_grad).all()
    assert jnp.isfinite(memory_grad).all()
