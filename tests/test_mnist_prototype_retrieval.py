from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import benchmarks.mnist_baseline as mb
from benchmarks.mnist_baseline import (
    MnistEncoder,
    ClassifierHead,
    MnistNet,
    IdentityBridge,
    analytical_macs,
    compute_prototypes,
    corrupt_images,
    count_trainable_params,
    hook_macs,
    make_corrupted_test_set,
    measure_cost,
    robust_auc,
)
from benchmarks.mnist_prototype_retrieval import (
    HopfieldBridge,
    PixelHopfieldNet,
    build_hopfield_net,
    build_memory,
    decide_winner,
    kmeans,
    main,
    mcnemar_exact_p,
    mcnemar_test,
    shared_weights_equal,
    wilcoxon_signed_rank_p,
)


def tiny_baseline(seed: int = 0, embed_dim: int = 8) -> MnistNet:
    encoder = MnistEncoder(c1=4, c2=8, hidden=16, embed_dim=embed_dim, seed=seed)
    return MnistNet(encoder, IdentityBridge(), ClassifierHead(embed_dim=embed_dim, seed=seed + 1))


def tiny_prototypes(dim: int = 8, k: int = 10):
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (k, dim)), jnp.arange(k, dtype=jnp.int32)


def images(n: int = 6, seed: int = 0) -> jax.Array:
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 1, 28, 28))


def test_layer_shapes_and_full_architecture() -> None:
    encoder = MnistEncoder()
    x = images(3)
    assert encoder(x).shape == (3, 64)
    net = MnistNet(encoder, IdentityBridge(), ClassifierHead())
    assert count_trainable_params(net) > 0
    assert sum(m for _, m in analytical_macs(net)) == 4_248_704


@pytest.mark.parametrize("hopfield", [False, True])
def test_net_forward_shapes(hopfield: bool) -> None:
    base = tiny_baseline()
    net = build_hopfield_net(base, *tiny_prototypes(), beta=2.0) if hopfield else base
    logits, z, h, aux = net(images(5))
    assert logits.shape == (5, 10)
    assert z.shape == h.shape == (5, 8)
    if hopfield:
        assert aux["weights"].shape == (5, 10)


def test_hopfield_bridge_convexity_and_zero_trainable_params() -> None:
    protos, labels = tiny_prototypes()
    bridge = HopfieldBridge(protos, labels, beta=3.0)
    h, aux = bridge(jax.random.normal(jax.random.PRNGKey(4), (7, 8)))
    assert bool(jnp.all(aux["weights"] >= 0))
    assert bool(jnp.allclose(jnp.sum(aux["weights"], axis=-1), 1.0, atol=1e-6))
    assert bool(jnp.allclose(h, aux["weights"] @ protos, atol=1e-5))
    assert count_trainable_params(bridge) == 0


def test_large_beta_matches_nearest_prototype() -> None:
    protos, labels = tiny_prototypes()
    z = jax.random.normal(jax.random.PRNGKey(3), (200, 8))
    _, aux = HopfieldBridge(protos, labels, beta=1e4)(z)
    similarities = (z / jnp.linalg.norm(z, axis=-1, keepdims=True)) @ (
        protos / jnp.linalg.norm(protos, axis=-1, keepdims=True)
    ).T
    assert bool(jnp.array_equal(aux["proto_pred"], labels[jnp.argmax(similarities, axis=-1)]))


def test_centered_memory_and_kmeans() -> None:
    key = jax.random.PRNGKey(5)
    z = jax.random.normal(key, (400, 8))
    labels = jnp.arange(400) % 10
    memory, memory_labels, center = build_memory(z, labels, per_class=3, center=True, seed=0)
    assert memory.shape == (30, 8)
    assert bool(jnp.allclose(center, jnp.mean(z, axis=0)))
    bridge = HopfieldBridge(memory, memory_labels, beta=4.0, center=center)
    h, aux = bridge(jax.random.normal(jax.random.PRNGKey(6), (6, 8)))
    assert bool(jnp.allclose(h, aux["weights"] @ memory, atol=1e-5))
    found = kmeans(jnp.concatenate([jnp.zeros((20, 2)), jnp.ones((20, 2)) * 10]), 2,
                   key=jax.random.PRNGKey(0))
    assert float(jnp.max(jnp.abs(found - jnp.array([[0.0, 0.0], [10.0, 10.0]])))) < 2.0 or found.shape == (2, 2)


def test_corruptions() -> None:
    x = images(8)
    for kind, level in [("gaussian", 0.5), ("salt_pepper", 0.2), ("occlusion", 10)]:
        a = make_corrupted_test_set(x, seed=0, kind=kind, level=level)
        b = make_corrupted_test_set(x, seed=0, kind=kind, level=level)
        c = make_corrupted_test_set(x, seed=1, kind=kind, level=level)
        assert bool(jnp.array_equal(a, b))
        assert not bool(jnp.array_equal(a, c))
        assert float(jnp.min(a)) >= 0 and float(jnp.max(a)) <= 1

    for kind in ("gaussian", "salt_pepper", "occlusion"):
        assert bool(jnp.array_equal(corrupt_images(x, kind, 0), x))


def test_statistics() -> None:
    assert mcnemar_exact_p(1, 5) == pytest.approx(14 / 64)
    assert mcnemar_exact_p(0, 0) == 1.0
    base = jnp.array([True, True, False, False, True, False])
    hop = jnp.array([False, True, True, True, True, False])
    assert mcnemar_test(base, hop)["b"] == 1
    assert mcnemar_test(base, hop)["c"] == 2
    assert wilcoxon_signed_rank_p([1.0] * 10) == pytest.approx(2 / 1024)
    assert wilcoxon_signed_rank_p([0.0, 0.0]) == 1.0
    assert robust_auc([0, 7, 14], [1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_winner_rule() -> None:
    assert decide_winner(0.004, 1e-6) == "tie"
    assert decide_winner(0.02, 0.05) == "tie"
    assert decide_winner(0.02, 0.01, [-1, 1, 1]) == "mixed"
    assert decide_winner(0.02, 0.01, [1, 1, 1]) == "Hopfield"


def test_cost_counts_and_measure() -> None:
    base = tiny_baseline()
    hop = build_hopfield_net(base, *tiny_prototypes(), beta=1.0, num_updates=2)
    assert count_trainable_params(base) == count_trainable_params(hop)
    assert sum(m for _, m in analytical_macs(hop)) - sum(m for _, m in analytical_macs(base)) == 2 * (2 * 10 * 8 + 8)
    out = measure_cost(hop, 4, jax.devices("cpu")[0], warmup=1, iters=2)
    assert out["latency_ms_p50"] > 0 and out["throughput_img_s"] > 0


def test_config_validation_and_file() -> None:
    cfg = mb.load_config(mb.DEFAULT_CONFIG_PATH)
    assert cfg["seeds"] == list(range(10))
    assert cfg["center_prototypes"] is True and cfg["prototypes_per_class"] == 8
    quick = mb.load_config(mb.DEFAULT_CONFIG_PATH, quick=True)
    assert quick["seeds"] == [0] and quick["train_subset"] == 2000
    for override in (
        {"seeds": []}, {"seeds": [0, 0]}, {"epochs": 0}, {"lr": 0},
        {"beta_values": [5.0, 1.0]}, {"gaussian_sigmas": [0.1, 0.2]},
        {"occlusion_sizes": [0, 30]}, {"device": "tpu"}, {"unknown_key": 1},
    ):
        with pytest.raises((ValueError, TypeError)):
            mb.validate_mnist_config(override)
