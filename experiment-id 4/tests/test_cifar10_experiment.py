"""Tests for benchmarks/cifar10_baseline.py and Experiment 4 CIFAR-10 config & runner."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP4_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP4_DIR) not in sys.path:
    sys.path.insert(0, str(EXP4_DIR))

import cifar10_config
from benchmarks.cifar10_baseline import (
    CIFAR10_EMBED_DIM,
    build_cifar10_baseline_net,
    corrupt_cifar10_images,
    encode,
    evaluate_cifar10_baseline,
    make_cifar10_splits,
    make_corrupted_cifar10_test_set,
    train_cifar10_baseline,
)
from cifar10_config import get_cifar10_config
from cifar10_experiment import (
    run_ablation_for_seed,
    run_seed,
    write_cifar10_summary_report,
)


def test_cifar10_baseline_splits_corruptions_and_net() -> None:
    torch.manual_seed(0)
    train_images = torch.rand(100, 3, 32, 32)
    train_labels = torch.arange(100) % 10
    test_images = torch.rand(30, 3, 32, 32)
    test_labels = torch.arange(30) % 10

    splits = make_cifar10_splits(
        (train_images, train_labels, test_images, test_labels),
        seed=0,
        val_size=20,
        train_subset=60,
        test_subset=24,
    )
    assert splits.train_images.shape == (60, 3, 32, 32)
    assert splits.val_images.shape == (20, 3, 32, 32)
    assert splits.test_images.shape == (24, 3, 32, 32)

    for kind, lvl in (("gaussian", 0.3), ("salt_pepper", 0.2), ("occlusion", 7)):
        c1 = make_corrupted_cifar10_test_set(
            splits.test_images, seed=3, kind=kind, level=lvl
        )
        c2 = make_corrupted_cifar10_test_set(
            splits.test_images, seed=3, kind=kind, level=lvl
        )
        assert c1.shape == splits.test_images.shape
        assert torch.equal(c1, c2)
        assert float(c1.min()) >= 0.0 and float(c1.max()) <= 1.0

    with pytest.raises(ValueError):
        corrupt_cifar10_images(
            splits.test_images, "occlusion", 40, torch.Generator().manual_seed(0)
        )

    device = torch.device("cpu")
    net = build_cifar10_baseline_net(0, c1=8, c2=16, c3=32, hidden=32, embed_dim=16)
    wall = train_cifar10_baseline(
        net,
        splits.train_images,
        splits.train_labels,
        epochs=1,
        batch_size=30,
        seed=0,
        device=device,
    )
    assert wall >= 0.0
    acc = evaluate_cifar10_baseline(
        net, splits.test_images, splits.test_labels, batch_size=16, device=device
    )
    assert 0.0 <= acc <= 1.0
    z = encode(net.encoder, splits.test_images, batch_size=16, device=device)
    assert z.shape == (24, 16)


def test_cifar10_config_presets_and_overrides(monkeypatch) -> None:
    assert cifar10_config.DIM == CIFAR10_EMBED_DIM == 128
    monkeypatch.setattr(cifar10_config, "MAX_MEMORIES", 48)
    monkeypatch.setattr(cifar10_config, "INITIAL_MEMORIES", 8)

    full_cfg = get_cifar10_config(quick=False)
    assert full_cfg.dataset == "cifar10"
    assert full_cfg.memory.dim == 128
    assert full_cfg.memory.max_memories == 48
    assert full_cfg.memory.initial_memories == 8
    assert full_cfg.memory.warmup_epochs == 2

    quick_cfg = get_cifar10_config(quick=True)
    assert quick_cfg.memory.warmup_epochs == 1
    assert quick_cfg.memory.to_global_dynamic_config("dynamic_adaptive").auto_calibrate is True


def test_cifar10_experiment4_end_to_end_synthetic_smoke(tmp_path: Path) -> None:
    torch.manual_seed(0)
    train_images = torch.rand(120, 3, 32, 32)
    train_labels = torch.arange(120) % 10
    test_images = torch.rand(40, 3, 32, 32)
    test_labels = torch.arange(40) % 10
    data = (train_images, train_labels, test_images, test_labels)

    cfg = get_cifar10_config(
        quick=True,
        ablations=("fixed_capacity", "dynamic_adaptive"),
        output_path=tmp_path / "cifar10_exp4_smoke.json",
        memory_overrides={"dim": 16, "max_memories": 24, "initial_memories": 6},
        experiment_overrides={
            "device": "cpu",
            "epochs": 2,
            "val_size": 20,
            "train_subset": 80,
            "test_subset": 40,
            "batch_size": 32,
            "eval_batch_size": 32,
            "encoder_channels": (8, 16, 32),
            "encoder_hidden": 32,
            "corruptions": {"gaussian": [0.0, 0.3], "occlusion": [0, 7]},
        },
    )

    seed_res = run_seed(cfg, seed=0, data=data)
    assert set(seed_res["ablations"].keys()) == {"fixed_capacity", "dynamic_adaptive"}
    fc = seed_res["ablations"]["fixed_capacity"]
    da = seed_res["ablations"]["dynamic_adaptive"]
    assert fc["direct_test_accuracy"] == da["direct_test_accuracy"]
    assert fc["final_active_prototypes"] == 6
    assert da["final_active_prototypes"] >= cfg.memory.min_active_memories
    for metrics in (fc, da):
        last_epoch = metrics["epochs"][-1]
        assert last_epoch["observation_noise"] == 1 - last_epoch["direct_train_accuracy"] + cfg.memory.eps_stab

    # Verify single-variant wrapper matches lockstep run exactly for fixed_capacity.
    single_fc = run_ablation_for_seed(cfg, seed=0, ablation_mode="fixed_capacity", data=data)
    assert single_fc["test_accuracy"] == fc["test_accuracy"]

    report_path = tmp_path / "cifar10_exp4_smoke_summary.md"
    write_cifar10_summary_report(report_path, {"config": cfg.to_dict(), "seeds": [seed_res]})
    text = report_path.read_text(encoding="utf-8")
    assert "Experiment 4 (CIFAR-10)" in text
    assert "Fixed-Capacity Hopfield" in text
    assert "Dynamic Adaptive Hopfield" in text
