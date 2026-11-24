"""Offline end-to-end tests for Milestone 2B multi-dataset runner and reporting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENT_2 = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_2.parent
for path in (str(EXPERIMENT_2), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import multi_dataset_experiment as mde
from datasets import get_dataset_spec

from hopfieldpc import METHODS


def test_build_dataset_config_sets_orientation_safe_defaults() -> None:
    svhn_cfg = mde.build_dataset_config(get_dataset_spec("svhn"), quick=False)
    assert svhn_cfg["warmup_epochs"] == 2
    assert svhn_cfg["min_accuracy"] == pytest.approx(0.80)
    assert svhn_cfg["max_train_val_gap"] == pytest.approx(0.07)

    quick_cfg = mde.build_dataset_config(get_dataset_spec("cifar10"), quick=True)
    assert quick_cfg["epochs"] == 3
    assert quick_cfg["warmup_epochs"] == 1
    assert quick_cfg["train_subset"] == 2000


def test_run_multi_dataset_end_to_end_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify full CLI runner, JSON artifact, and Markdown summary without network calls."""

    def fake_load(name: str, data_dir: str | Path, *, download: bool = True):
        spec = get_dataset_spec(name)

        def _make(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            gen = torch.Generator().manual_seed(seed)
            labels = torch.arange(n) % spec.num_classes
            x = torch.rand(n, spec.in_channels, spec.image_size, spec.image_size, generator=gen) * 0.15
            for cls_id in range(spec.num_classes):
                r = (cls_id // 4) * 8 + 2
                col = (cls_id % 4) * 6 + 2
                if spec.in_channels == 1:
                    x[labels == cls_id, :, r : r + 6, col : col + 6] += 0.75
                else:
                    r_val = 0.2 + 0.7 * ((cls_id * 3 % 10) / 9.0)
                    g_val = 0.2 + 0.7 * (((cls_id * 7 + 2) % 10) / 9.0)
                    b_val = 0.2 + 0.7 * (((cls_id * 5 + 4) % 10) / 9.0)
                    color = torch.tensor([r_val, g_val, b_val], dtype=torch.float32).view(1, 3, 1, 1)
                    x[labels == cls_id] = 0.85 * color + 0.10 * x[labels == cls_id]
            return x.clamp(0.0, 1.0), labels

        train_x, train_y = _make(160, 0)
        test_x, test_y = _make(40, 1)
        return train_x, train_y, test_x, test_y

    monkeypatch.setattr(mde, "load_dataset_tensors", fake_load)

    # Patch build_dataset_config to use tiny splits compatible with 160 synthetic train samples.
    orig_build = mde.build_dataset_config

    def tiny_build(spec, **kwargs):
        cfg = orig_build(spec, **kwargs)
        cfg["val_size"] = 40
        cfg["train_subset"] = 120
        cfg["test_subset"] = 40
        cfg["batch_size"] = 32
        cfg["eval_batch_size"] = 64
        cfg["learning_rate"] = 5e-3
        cfg["augment"] = False
        cfg["epochs"] = 6
        cfg["warmup_epochs"] = 1
        cfg["prototypes_per_class"] = 2
        return cfg

    monkeypatch.setattr(mde, "build_dataset_config", tiny_build)

    out_json = tmp_path / "results" / "multi_test.json"
    mde.run_multi_dataset(
        dataset_names=("fashion_mnist", "svhn"),
        seeds=(0,),
        methods=METHODS,
        quick=True,
        data_dir=tmp_path,
        output_path=out_json,
        device_name="cpu",
        num_threads=1,
        rgb_channels=(8, 16, 32),
    )

    assert out_json.exists()
    summary_md = out_json.with_name("multi_test_summary.md")
    assert summary_md.exists()

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(loaded["datasets"]) == {"fashion_mnist", "svhn"}
    for ds_name in ("fashion_mnist", "svhn"):
        ds_summary = loaded["datasets"][ds_name]["summary"]
        assert set(ds_summary) == {"direct", *METHODS}
        for stats in ds_summary.values():
            assert "train_mean" in stats
            assert "val_mean" in stats
            assert "test_mean" in stats
            assert "train_val_gap_mean" in stats
            assert "all_passed" in stats["constraints"]

    md_text = summary_md.read_text(encoding="utf-8")
    assert "Fashion-MNIST" in md_text
    assert "SVHN" in md_text
    assert "Direct CNN (Baseline)" in md_text
    assert "Hybrid" in md_text
    assert "Ema" in md_text
