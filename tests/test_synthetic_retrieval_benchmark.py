"""Unit tests for synthetic nearest-memory retrieval benchmark and diagnostics."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.synthetic_retrieval import run_benchmark
from hopfieldpc.diagnostics import (


    attention_entropy,
    retrieval_distance,
    topk_mass,
    topk_recall,
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

def test_diagnostics_functions() -> None:
    weights = jnp.array([
        [0.7, 0.2, 0.1, 0.0],
        [0.4, 0.3, 0.2, 0.1],
    ])

    entropy = attention_entropy(weights)
    assert entropy.shape == (2,)
    assert entropy[0].item() < entropy[1].item()

    top2_mass = topk_mass(weights, k=2)
    assert top2_mass.shape == (2,)
    assert jnp.isclose(top2_mass[0], jnp.array(0.9))
    assert jnp.isclose(top2_mass[1], jnp.array(0.7))

    targets = jnp.array([0, 3])
    recall_top1 = topk_recall(weights, targets, k=1)
    assert recall_top1[0].item() == 1.0

    recall_top2 = topk_recall(weights, targets, k=2)
    assert recall_top2[0].item() == 1.0
    assert recall_top2[1].item() == 0.0

    q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    tgt = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    dist = retrieval_distance(q, tgt)
    assert jnp.isclose(dist[0], jnp.array(0.0))
    assert jnp.isclose(dist[1], jnp.array(1.0))


def test_run_benchmark_tiny_config_and_output_files() -> None:
    tiny_config = {
        "seed": 123,
        "dim": 8,
        "num_trials": 4,
        "beta_values": [1.0, 5.0],
        "noise_levels": [0.01],
        "memory_sizes": [5],
        "num_updates": [1, 2],
        "top_k": 2,
        "normalize": False,
        "capacity_threshold": 0.95,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "test_reports"
        results, capacity_summary = run_benchmark(tiny_config, output_dir=output_dir)

        assert len(results) == 4
        expected_metrics = [
            "beta",
            "noise",
            "memory_size",
            "num_updates",
            "top1_acc",
            "topk_recall",
            "query_distance",
            "retrieved_distance",
            "retrieval_improvement",
            "entropy",
            "topk_mass",
            "energy_initial",
            "energy_final",
            "energy_delta",
        ]

        for row in results:
            for metric in expected_metrics:
                assert metric in row
                assert not jnp.isnan(jnp.array(row[metric]))

        assert len(capacity_summary) == 4
        assert "capacity_at_95" in capacity_summary[0]

        assert (output_dir / "results.json").exists()
        assert (output_dir / "results.csv").exists()
        assert (output_dir / "capacity_summary.csv").exists()
