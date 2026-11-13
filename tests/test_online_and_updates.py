"""Unit tests for modularized Experiment 1 and Experiment 2 components in hopfieldpc."""

from __future__ import annotations

import pytest
import torch

from hopfieldpc import (
    METHODS,
    UPDATE_METHODS,
    BasePrototypeUpdater,
    CachedOccupancyBank,
    EMAOnlineBank,
    EMAPrototypeUpdater,
    HybridOnlineBank,
    HybridPrototypeUpdater,
    KalmanOnlineBank,
    KalmanPrototypeUpdater,
    MomentumOnlineBank,
    MomentumPrototypeUpdater,
    OnlineHopfieldBank,
    PrototypeUpdater,
    centered_retrieve,
    correctness_categories,
    ema_step,
    hybrid_kalman_momentum_step,
    kalman_gain,
    kalman_momentum_step,
    kalman_step,
    kalman_uncertainty_update,
    momentum_step,
    momentum_velocity_update,
)


def test_public_exports_and_methods() -> None:
    assert METHODS == UPDATE_METHODS == (
        "ema",
        "hybrid",
        "momentum_only",
        "kalman_only",
    )


def test_kalman_momentum_step_single_and_batched() -> None:
    current = torch.tensor([1.0, 0.0])
    candidate = torch.tensor([0.0, 1.0])
    velocity = torch.zeros(2)
    uncertainty = torch.tensor(1.0)

    out = kalman_momentum_step(
        current,
        candidate,
        velocity,
        uncertainty,
        accuracy=0.0,
        method="hybrid",
        gamma=0.9,
        process_noise=0.01,
    )
    assert 0.0 < out.gain.item() < 1.0
    assert out.velocity.norm().item() > 0.0
    assert out.uncertainty.item() == pytest.approx(0.51, abs=1e-6)
    assert torch.isfinite(out.new_value).all()


def test_prototype_updater_hybrid_and_ablations() -> None:
    for method in METHODS:
        updater = PrototypeUpdater(
            1,
            2,
            method=method,
            gamma=0.9,
            process_noise=0.01,
            prototypes_per_class=1,
        )
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.zeros(2, dtype=torch.long)
        updater.update(features, labels, 1.0)
        result = updater.update(
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]), labels, 0.0
        )
        assert torch.isfinite(updater.values).all()
        assert torch.isfinite(result["gain"]).all()


def test_hybrid_online_bank_merge_updates_velocity_and_uncertainty() -> None:
    for method in METHODS:
        bank = HybridOnlineBank(
            2,
            num_classes=1,
            max_memories_per_class=2,
            method=method,
            ema_alpha=0.1,
            beta=5.0,
            num_updates=1,
            merge_distance_threshold=1.5,
            normalization_epsilon=1e-8,
            gamma=0.9,
            process_noise=0.01,
            initial_uncertainty=1.0,
        )
        bank.initialize_center(torch.zeros(2))
        seed_prop = bank.plan_candidate(
            torch.tensor([1.0, 0.0]), 0, bootstrap=True
        )
        assert seed_prop.action == "seed"
        bank.commit(seed_prop, example_id=0, step=0, epoch=1)
        assert bank.ready

        bank.current_accuracy = 0.5
        merge_prop = bank.plan_candidate(torch.tensor([0.8, 0.2]), 0)
        assert merge_prop.action == "merge"
        bank.commit(merge_prop, example_id=1, step=1, epoch=1)

        retrieved, weights = bank.retrieve(torch.tensor([[1.0, 0.0]]))
        assert retrieved.shape == (1, 2)
        assert weights.shape == (1, 1)
        assert torch.isfinite(retrieved).all()
        if method in ("hybrid", "momentum_only"):
            assert bank.velocity[0].norm().item() > 0.0
        assert bank.uncertainty[0].item() < 1.0


def test_cached_occupancy_bank_and_centered_retrieve() -> None:
    bank = CachedOccupancyBank(2, num_classes=2, max_memories_per_class=2)
    bank.initialize_center(torch.zeros(2))
    for label, vec in enumerate(([1.0, 0.0], [0.0, 1.0])):
        prop = bank.plan_candidate(torch.tensor(vec), label, bootstrap=True)
        bank.commit(prop, example_id=label, step=label, epoch=1)

    assert bank.ready
    assert bank.occupancy() == [1, 1]
    retrieved, weights = centered_retrieve(
        torch.tensor([[1.0, 0.0]]),
        bank.snapshot().values,
        bank.center,
        beta=5.0,
    )
    assert retrieved.shape == (1, 2)
    assert weights.shape == (1, 2)
    cats = correctness_categories(
        torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([0, 0])
    )
    assert cats == ["harmed", "both_wrong"]


def test_dedicated_ema_momentum_kalman_modules_match_dispatcher() -> None:
    features_init = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    features_next = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    labels = torch.zeros(2, dtype=torch.long)

    pairs = [
        ("ema", EMAPrototypeUpdater(1, 2, prototypes_per_class=1)),
        ("momentum_only", MomentumPrototypeUpdater(1, 2, prototypes_per_class=1)),
        ("kalman_only", KalmanPrototypeUpdater(1, 2, prototypes_per_class=1)),
        ("hybrid", HybridPrototypeUpdater(1, 2, prototypes_per_class=1)),
    ]
    for method, dedicated in pairs:
        assert isinstance(dedicated, BasePrototypeUpdater)
        generic = PrototypeUpdater(1, 2, method=method, prototypes_per_class=1)
        dedicated.update(features_init, labels, 1.0)
        generic.update(features_init, labels, 1.0)
        out_d = dedicated.update(features_next, labels, 0.25)
        out_g = generic.update(features_next, labels, 0.25)
        torch.testing.assert_close(dedicated.values, generic.values)
        torch.testing.assert_close(dedicated.velocity, generic.velocity)
        torch.testing.assert_close(dedicated.uncertainty, generic.uncertainty)
        torch.testing.assert_close(out_d["gain"], out_g["gain"])

    bank_pairs = [
        ("ema", EMAOnlineBank(2, num_classes=1, max_memories_per_class=2, merge_distance_threshold=1.5)),
        ("momentum_only", MomentumOnlineBank(2, num_classes=1, max_memories_per_class=2, merge_distance_threshold=1.5)),
        ("kalman_only", KalmanOnlineBank(2, num_classes=1, max_memories_per_class=2, merge_distance_threshold=1.5)),
        ("hybrid", HybridOnlineBank(2, num_classes=1, max_memories_per_class=2, method="hybrid", merge_distance_threshold=1.5)),
    ]
    for method, bank in bank_pairs:
        assert isinstance(bank, OnlineHopfieldBank)
        bank.initialize_center(torch.zeros(2))
        p0 = bank.plan_candidate(torch.tensor([1.0, 0.0]), 0, bootstrap=True)
        bank.commit(p0, example_id=0, step=0, epoch=1)
        p1 = bank.plan_candidate(torch.tensor([0.8, 0.2]), 0)
        assert p1.action == "merge"
        bank.commit(p1, example_id=1, step=1, epoch=1)
        assert torch.isfinite(bank.values[0]).all()

    cur = torch.tensor([1.0, 0.0])
    cand = torch.tensor([0.0, 1.0])
    vel = torch.zeros(2)
    unc = torch.tensor(1.0)
    assert torch.allclose(ema_step(cur, cand, alpha=0.2), torch.tensor([0.8, 0.2]))
    v_next = momentum_velocity_update(vel, cand - cur, gamma=0.9)
    m_out = momentum_step(cur, cand, vel, unc, accuracy=0.5, gamma=0.9)
    assert torch.allclose(m_out.velocity, v_next)
    k_g = kalman_gain(unc, accuracy=0.5, epsilon=1e-8)
    k_u = kalman_uncertainty_update(unc, k_g, process_noise=0.01)
    k_out = kalman_step(cur, cand, vel, unc, accuracy=0.5, epsilon=1e-8, process_noise=0.01)
    h_out = hybrid_kalman_momentum_step(cur, cand, vel, unc, accuracy=0.5, gamma=0.9, epsilon=1e-8, process_noise=0.01)
    assert torch.allclose(k_out.uncertainty, k_u)
    assert torch.allclose(h_out.uncertainty, k_u)

