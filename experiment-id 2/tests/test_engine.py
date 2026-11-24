"""Offline unit tests for the shared-backbone training and evaluation engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXPERIMENT_2 = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_2.parent
for path in (str(EXPERIMENT_2), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from backbones import build_backbone
from datasets import get_dataset_spec
from engine import (
    evaluate_constraints,
    evaluate_split,
    paired_comparison,
    train_and_evaluate_seed,
)
from synthetic_helpers import synthetic_block_dataset

from hopfieldpc import METHODS, HybridOnlineBank


def test_evaluate_constraints_enforces_accuracy_and_gap() -> None:
    ok = evaluate_constraints(0.88, 0.83, 0.82)
    assert ok["all_passed"] is True
    assert ok["min_accuracy_ok"] is True
    assert ok["generalization_gap_ok"] is True
    assert ok["train_val_gap"] == pytest.approx(0.05)

    low_acc = evaluate_constraints(0.85, 0.80, 0.79)
    assert low_acc["min_accuracy_ok"] is False
    assert low_acc["all_passed"] is False

    overfit = evaluate_constraints(0.95, 0.85, 0.84)
    assert overfit["min_accuracy_ok"] is True
    assert overfit["generalization_gap_ok"] is False
    assert overfit["all_passed"] is False


def test_evaluate_split_requires_ready_bank_and_uses_active_retrieval() -> None:
    spec = get_dataset_spec("fashion_mnist")
    net = build_backbone(spec, embed_dim=64, seed=0)
    bank = HybridOnlineBank(64, spec.num_classes, 2, method="hybrid")
    x = torch.rand(10, 1, 28, 28)
    y = torch.arange(10)

    with pytest.raises(RuntimeError, match="not ready"):
        evaluate_split(net, x, y, device=torch.device("cpu"), batch_size=5, bank=bank)

    bank.initialize_center(torch.zeros(64))
    for c in range(spec.num_classes):
        prop = bank.plan_candidate(torch.randn(64), c, bootstrap=True)
        if prop.is_write:
            bank.commit(prop, example_id=c, step=0, epoch=1)
    assert bank.ready

    res = evaluate_split(
        net, x, y, device=torch.device("cpu"), batch_size=5, bank=bank, return_predictions=True
    )
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["entropy_mean"] >= 0.0
    assert 0.0 < res["top1_mass_mean"] <= 1.0
    assert res["predictions"].shape == (10,)


def test_paired_comparison_counts_helped_and_harmed() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    direct = torch.tensor([0, 0, 2, 0])  # T, F, T, F
    hop = torch.tensor([0, 1, 0, 0])     # T, T, F, F
    counts = paired_comparison(direct, hop, labels)
    assert counts == {
        "both_correct": 1,
        "helped": 1,
        "harmed": 1,
        "both_wrong": 1,
        "net_gain": 0,
    }


@pytest.mark.parametrize("dataset_name", ("fashion_mnist", "cifar10"))
def test_train_and_evaluate_seed_produces_distinct_memory_train_evals(dataset_name: str) -> None:
    """Verify full engine run and populated memory banks on synthetic data."""
    spec = get_dataset_spec(dataset_name)
    train_x, train_y, test_x, test_y = synthetic_block_dataset(
        n_train=160,
        n_test=40,
        channels=spec.in_channels,
        image_size=spec.image_size,
        num_classes=spec.num_classes,
    )

    cfg = {
        "methods": list(METHODS),
        "embed_dim": 64,
        "rgb_channels": (8, 16, 32),
        "epochs": 8,
        "warmup_epochs": 1,
        "batch_size": 32,
        "eval_batch_size": 64,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "augment": False,
        "val_size": 40,
        "train_subset": 120,
        "test_subset": 40,
        "prototypes_per_class": 4,
        "hopfield_beta": 5.0,
        "hopfield_num_updates": 1,
        "merge_distance_threshold": 0.2,
        "normalization_epsilon": 1e-8,
        "min_accuracy": 0.80,
        "max_train_val_gap": 0.07,
        "update": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 1e-8,
            "process_noise": 0.01,
            "initial_uncertainty": 1.0,
        },
        "corruptions": {
            "gaussian": [0.0, 0.5],
            "salt_pepper": [0.0, 0.2],
            "occlusion": [0, 7],
        },
    }
    res = train_and_evaluate_seed(
        spec, cfg, seed=0, data=(train_x, train_y, test_x, test_y), device=torch.device("cpu")
    )
    assert res["dataset"] == dataset_name
    assert "direct" in res["methods"]
    for method in METHODS:
        m = res["methods"][method]
        assert m["active_memories"] >= spec.num_classes
        assert {"train_accuracy", "validation_accuracy", "test_accuracy"} <= set(m)
        assert "gaussian" in m["robust_auc"]
        assert "all_passed" in m["constraints"]
