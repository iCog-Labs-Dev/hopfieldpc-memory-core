"""Tests for Experiment 4 config.py and experiment.py runner."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

EXP4_DIR = Path(__file__).resolve().parents[1]
if str(EXP4_DIR) not in sys.path:
    sys.path.insert(0, str(EXP4_DIR))

import config as exp4_config
from config import get_config
from experiment import run_seed, write_summary_report


def test_experiment4_config_presets_and_variants(monkeypatch) -> None:
    # Verify get_config reads module-level config variables dynamically in both full and quick modes.
    monkeypatch.setattr(exp4_config, "MAX_MEMORIES", 24)
    monkeypatch.setattr(exp4_config, "INITIAL_MEMORIES", 6)
    monkeypatch.setattr(exp4_config, "MIN_ACTIVE_MEMORIES", 2)
    monkeypatch.setattr(exp4_config, "ONLINE_UPDATE_MODE", "sequential")

    default_quick = get_config(quick=True)
    assert default_quick.memory.max_memories == 24
    assert default_quick.memory.initial_memories == 6
    assert default_quick.memory.min_active_memories == 2
    assert default_quick.memory.online_update_mode == "sequential"

    cfg = get_config(
        quick=True,
        ablations=("fixed_capacity", "dynamic_adaptive"),
        online_update_mode="batch",
    )
    assert cfg.ablations == ("fixed_capacity", "dynamic_adaptive")
    mem_cfg = cfg.memory.to_global_dynamic_config("dynamic_adaptive")
    assert mem_cfg.auto_calibrate is True
    assert cfg.memory.online_update_mode == "batch"


def test_experiment4_end_to_end_synthetic_smoke(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # Minimal synthetic MNIST-shaped dataset (N=120, 1, 28, 28) for fast smoke validation.
    train_images = torch.randn(120, 1, 28, 28)
    train_labels = torch.arange(120) % 10
    test_images = torch.randn(40, 1, 28, 28)
    test_labels = torch.arange(40) % 10
    data = (train_images, train_labels, test_images, test_labels)

    cfg = get_config(
        quick=True,
        ablations=("fixed_capacity", "dynamic_adaptive"),
        output_path=tmp_path / "exp4_smoke.json",
    )
    cfg = replace(
        cfg,
        epochs=2,
        val_size=20,
        train_subset=80,
        test_subset=40,
        batch_size=32,
        eval_batch_size=32,
        corruptions={"gaussian": [0.0, 0.5]},
    )

    seed_res = run_seed(cfg, seed=0, data=data)
    assert set(seed_res["ablations"].keys()) == {"fixed_capacity", "dynamic_adaptive"}
    for mode, metrics in seed_res["ablations"].items():
        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert metrics["final_active_prototypes"] >= cfg.memory.min_active_memories

    report_path = tmp_path / "exp4_smoke_summary.md"
    write_summary_report(report_path, {"config": cfg.to_dict(), "seeds": [seed_res]})
    text = report_path.read_text(encoding="utf-8")
    assert "Experiment 4" in text
    assert "Fixed-Capacity Hopfield" in text
    assert "Dynamic Adaptive Hopfield" in text
