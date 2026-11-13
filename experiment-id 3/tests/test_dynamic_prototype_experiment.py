import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from dynamic_prototype_experiment import (
    DynamicHybridOnlineBank,
    masked_centered_retrieve,
)


def test_masked_centered_retrieve_zeros_inactive_slots():
    values = torch.tensor([[2.0, 0.0], [0.0, 3.0], [-2.0, 0.0]])
    center = torch.zeros(2)
    mask = torch.tensor([True, False, True])
    query = torch.tensor([[0.1, 2.5]])

    retrieved, weights = masked_centered_retrieve(
        query, values, center, mask, beta=5.0
    )

    assert weights.shape == (1, 3)
    assert weights[0, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isclose(weights.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(retrieved).all()


def test_spawning_splits_high_uncertainty_prototype_and_resets_state():
    bank = DynamicHybridOnlineBank(
        dim=4,
        num_classes=2,
        max_memories_per_class=4,
        initial_memories_per_class=2,
        method="hybrid",
        tau_support=1,
        prune_patience=2,
        tau_uncert=0.4,
        split_scale=0.1,
    )
    bank.initialize_center(torch.zeros(4))
    # Seed class 0 and class 1
    p0 = bank.plan_candidate(torch.tensor([1.0, 0.2, 0.0, 0.0]), 0, bootstrap=True)
    bank.commit(p0, example_id=0, step=1, epoch=1)
    p1 = bank.plan_candidate(torch.tensor([0.0, 1.0, 0.2, 0.0]), 1, bootstrap=True)
    bank.commit(p1, example_id=1, step=2, epoch=1)
    assert bank.ready

    # Provide high-variance embeddings for class 0 and incorrect predictions so P_0 > tau_uncert
    features = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0],
            [1.0, -0.9, 0.0, 0.0],
            [0.8, 0.7, 0.2, 0.0],
            [0.8, -0.7, -0.2, 0.0],
            [0.0, 1.0, 0.2, 0.0],
            [0.0, 1.0, 0.25, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    predictions = torch.tensor([1, 1, 1, 1, 1, 1])  # class 0 misclassified -> high R_j & Q_j

    stats = bank.adapt_prototypes(features, labels, predictions, epoch=2, step=10)
    assert stats["spawned"] >= 1
    assert bank.occupancy()[0] == 2
    active_class0 = ((bank.labels == 0) & bank.active).nonzero().flatten()
    for slot in active_class0:
        assert torch.allclose(bank.velocity[slot], torch.zeros(4))
        assert bank.uncertainty[slot].item() == pytest.approx(bank.initial_uncertainty)


def test_pruning_deactivates_low_support_prototype_while_keeping_one_per_class():
    bank = DynamicHybridOnlineBank(
        dim=4,
        num_classes=2,
        max_memories_per_class=4,
        initial_memories_per_class=2,
        method="hybrid",
        tau_support=3,
        prune_patience=2,
        tau_uncert=10.0,  # disable spawning in this test
    )
    bank.initialize_center(torch.zeros(4))
    p0 = bank.plan_candidate(torch.tensor([1.0, 0.0, 0.0, 0.0]), 0, bootstrap=True)
    bank.commit(p0, example_id=0, step=1, epoch=1)
    p1 = bank.plan_candidate(torch.tensor([0.0, 1.0, 0.0, 0.0]), 1, bootstrap=True)
    bank.commit(p1, example_id=1, step=2, epoch=1)

    # Add a second prototype to class 0 far away so it gets 0 assignments
    p0_second = bank.plan_candidate(torch.tensor([-1.0, 0.0, 0.0, 0.0]), 0, bootstrap=False)
    bank.commit(p0_second, example_id=2, step=3, epoch=1)
    assert bank.occupancy()[0] == 2

    features = torch.tensor(
        [
            [1.0, 0.05, 0.0, 0.0],
            [0.95, -0.05, 0.0, 0.0],
            [1.02, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.0, 1.0, 0.02, 0.0],
            [0.0, 0.98, -0.02, 0.0],
            [0.0, 1.01, 0.01, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    preds = labels.clone()

    # Epoch 2: first low-support streak on the unused class-0 prototype
    stats1 = bank.adapt_prototypes(features, labels, preds, epoch=2, step=10)
    assert stats1["pruned"] == 0
    assert bank.occupancy()[0] == 2

    # Epoch 3: second consecutive low-support epoch -> pruned
    stats2 = bank.adapt_prototypes(features, labels, preds, epoch=3, step=20)
    assert stats2["pruned"] == 1
    assert bank.occupancy()[0] == 1
