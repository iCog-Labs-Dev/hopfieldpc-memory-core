"""Energy-retrieval consistency tests for hopfield_energy.

Assumptions: fixed memory, fixed beta and unnormalized retrieval. Under them
E(q) = -(1/beta) logsumexp(beta q M^T) + 1/2 |q|^2 has gradient q - retrieved(q),
with or without the q-independent constants. Tests run in float64 with explicit
tolerances.
"""

import math

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hopfieldpc import hopfield_energy, hopfield_retrieve, multi_step_retrieve



BETAS = [0.5, 2.0, 8.0]
DT = jnp.float64



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

def _setup(seed=0, n=10, d=16, batch=4):
    g = _generator(seed)
    memory = _randn(n, d, generator=g, dtype=DT)
    memory = memory / memory
    query = _randn(batch, d, generator=g, dtype=DT)
    return query, memory


def _example():
    memory = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=DT)
    return jnp.array([[0.8, 0.5]], dtype=DT), memory


def test_reference_values_for_the_worked_example():
    q0, memory = _example()
    q1, _ = hopfield_retrieve(q0, memory, beta=5.0, normalize=False)
    expected = {
        (False, 0): -0.395338,
        (False, 1): -0.474894,
        (True, 0): 0.324385,
        (True, 1): 0.244828,
    }
    for (constants, step), value in expected.items():
        q = q0 if step == 0 else q1
        got = hopfield_energy(q, memory, 5.0, include_constants=constants)
        _assert_close(got, jnp.array([value], dtype=DT), atol=1e-5, rtol=0)


@pytest.mark.parametrize("beta", BETAS)
def test_constants_do_not_depend_on_the_query(beta):
    q, m = _setup()
    offset = (1.0 / beta) * math.log(m.shape[0]) + 0.5 * (m ** 2).sum(axis=-1).max()
    diff = hopfield_energy(q, m, beta, include_constants=True) - hopfield_energy(q, m, beta)
    _assert_close(diff, jnp.full_like(diff, float(offset)), atol=1e-12, rtol=0)


@pytest.mark.parametrize("include_constants", [False, True])
@pytest.mark.parametrize("beta", BETAS)
def test_gradient_is_query_minus_retrieved(beta, include_constants):
    q, m = _setup()
    retrieved, _ = hopfield_retrieve(q, m, beta=beta, normalize=False)
    grad = jax.grad(lambda x: hopfield_energy(x, m, beta, include_constants=include_constants).sum())(q)
    _assert_close(grad, q - retrieved)


def test_mean_reduction_scales_the_gradient_by_batch_size():
    q, m = _setup(batch=5)
    retrieved, _ = hopfield_retrieve(q, m, beta=2.0, normalize=False)
    grad = jax.grad(lambda x: hopfield_energy(x, m, 2.0, reduce="mean"))(q)
    _assert_close(grad, (q - retrieved) / 5)


@pytest.mark.parametrize("include_constants", [False, True])
@pytest.mark.parametrize("beta", BETAS)
def test_energy_never_increases_under_retrieval(beta, include_constants):
    q, m = _setup()
    previous = hopfield_energy(q, m, beta, include_constants=include_constants)
    for _ in range(6):
        q, _ = multi_step_retrieve(q, m, beta=beta, num_updates=1)
        current = hopfield_energy(q, m, beta, include_constants=include_constants)
        assert jnp.all(current <= previous + 1e-10)
        previous = current


@pytest.mark.parametrize("beta", BETAS)
def test_each_step_lowers_energy_by_half_the_squared_move(beta):
    q, m = _setup()
    q_new, _ = hopfield_retrieve(q, m, beta=beta, normalize=False)
    drop = hopfield_energy(q, m, beta) - hopfield_energy(q_new, m, beta)
    bound = 0.5 * ((q - q_new) ** 2).sum(axis=-1)
    assert jnp.all(drop >= bound - 1e-10)


def test_mean_reduced_energy_also_decreases():
    q, m = _setup(batch=6)
    q_new, _ = hopfield_retrieve(q, m, beta=2.0, normalize=False)
    assert hopfield_energy(q_new, m, 2.0, reduce="mean") <= hopfield_energy(q, m, 2.0, reduce="mean") + 1e-10


def test_default_energy_falls_in_the_worked_example():
    q0, m = _example()
    q1, _ = hopfield_retrieve(q0, m, beta=5.0, normalize=False)
    assert hopfield_energy(q1, m, 5.0).item() < hopfield_energy(q0, m, 5.0).item()
