"""Offline tests for the MNIST Herding comparison benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import benchmarks.mnist_herding_comparison as comparison
from benchmarks import mnist_prototype_retrieval as benchmark

# ---------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("raw", False),
        ("l2", True),
        ("kmeans", False),
    ],
)
def test_normalize_selection_flag(
    selection: str,
    expected: bool,
) -> None:
    assert comparison._normalize_selection_flag(selection) is expected


def test_normalize_selection_flag_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown selection mode"):
        comparison._normalize_selection_flag("invalid")


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("raw", ("Raw Herding", "original raw embeddings", "Raw Herding")),
        ("l2", ("L2 Herding", "L2-normalized embeddings", "L2 Herding")),
        (
            "kmeans",
            (
                "K-means",
                "original raw CNN embeddings",
                "Raw Herding",
            ),
        ),
    ],
)
def test_describe_selection(
    selection: str,
    expected: tuple[str, str, str],
) -> None:
    assert comparison._describe_selection(selection) == expected


# ---------------------------------------------------------------------
# Memory construction timing
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "expected_builder"),
    [
        ("raw", "herding"),
        ("l2", "herding"),
        ("kmeans", "kmeans"),
    ],
)
def test_measure_memory_builds_returns_selected_memory(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected_builder: str,
) -> None:
    run = SimpleNamespace(seed=0)

    calls = {"herding": 0, "kmeans": 0}

    def fake_herding(run: Any, **kwargs: Any):
        calls["herding"] += 1
        elapsed = [99.0, 4.0, 6.0, 5.0][calls["herding"] - 1]
        return (
            torch.tensor([[calls["herding"]]], dtype=torch.float32),
            torch.tensor([0]),
            None,
            elapsed,
        )

    def fake_kmeans(run: Any, **kwargs: Any):
        calls["kmeans"] += 1
        elapsed = [99.0, 1.0, 3.0, 2.0][calls["kmeans"] - 1]
        return (
            torch.tensor([[calls["kmeans"]]], dtype=torch.float32),
            torch.tensor([1]),
            None,
            elapsed,
        )

    monkeypatch.setattr(
        comparison,
        "timed_herding_memory_build",
        fake_herding,
    )
    monkeypatch.setattr(
        comparison,
        "timed_kmeans_memory_build",
        fake_kmeans,
    )

    result = comparison.measure_memory_builds(
        run,
        per_class=1,
        center=True,
        num_classes=2,
        normalize_for_selection=(selection == "l2"),
        selection=selection,
    )

    memory, labels, center, herding_time, kmeans_time, herding_samples, kmeans_samples = result

    assert calls == {"herding": 4, "kmeans": 4}
    assert memory.item() == (
        calls[expected_builder]
    )
    assert labels.item() == (0 if expected_builder == "herding" else 1)
    assert center is None

    assert herding_time == pytest.approx(5.0)
    assert kmeans_time == pytest.approx(2.0)
    assert herding_samples == [4.0, 6.0, 5.0]
    assert kmeans_samples == [1.0, 3.0, 2.0]


def test_measure_memory_builds_rejects_unknown_selection() -> None:
    run = SimpleNamespace(seed=0)

    with pytest.raises(ValueError, match="Unknown selection"):
        comparison.measure_memory_builds(
            run,
            per_class=1,
            center=False,
            num_classes=2,
            normalize_for_selection=False,
            selection="invalid",
        )


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------


def test_evaluate_model_returns_predictions_and_correctness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = torch.randn(3, 2)
    labels = torch.tensor([0, 1, 0])

    outputs = {
        "head_pred": torch.tensor([0, 0, 0]),
        "proto_pred": torch.tensor([0, 1, 1]),
        "weights": torch.ones(3, 2),
        "z": embeddings,
    }

    correct = {
        "head": torch.tensor([True, False, True]),
        "proto": torch.tensor([True, True, False]),
    }

    monkeypatch.setattr(
        benchmark,
        "forward_embeddings",
        lambda *args, **kwargs: outputs,
    )
    monkeypatch.setattr(
        benchmark,
        "summarize_outputs",
        lambda *args, **kwargs: (
            {"head_acc": 2 / 3, "proto_acc": 2 / 3},
            correct,
            None,
        ),
    )

    run = SimpleNamespace(
        batch_size=4,
        device=torch.device("cpu"),
        prototypes=torch.randn(2, 2),
        memory_labels=torch.tensor([0, 1]),
    )

    result = comparison._evaluate_model(
        torch.nn.Identity(),
        embeddings,
        labels,
        run,
    )

    assert result["metrics"]["head_acc"] == pytest.approx(2 / 3)
    assert result["metrics"]["proto_acc"] == pytest.approx(2 / 3)

    np.testing.assert_array_equal(
        result["predictions"]["head"],
        [0, 0, 0],
    )
    np.testing.assert_array_equal(
        result["predictions"]["proto"],
        [0, 1, 1],
    )
    np.testing.assert_array_equal(
        result["correct"]["head"],
        [True, False, True],
    )
    np.testing.assert_array_equal(
        result["correct"]["proto"],
        [True, True, False],
    )


def test_evaluate_model_detects_inconsistent_prototype_correctness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = torch.randn(2, 2)
    labels = torch.tensor([0, 1])

    outputs = {
        "head_pred": torch.tensor([0, 1]),
        "proto_pred": torch.tensor([0, 0]),
        "weights": torch.ones(2, 2),
        "z": embeddings,
    }

    incorrect_correctness = {
        "head": torch.tensor([True, True]),
        "proto": torch.tensor([True, True]),
    }

    monkeypatch.setattr(
        benchmark,
        "forward_embeddings",
        lambda *args, **kwargs: outputs,
    )
    monkeypatch.setattr(
        benchmark,
        "summarize_outputs",
        lambda *args, **kwargs: (
            {"head_acc": 1.0, "proto_acc": 0.5},
            incorrect_correctness,
            None,
        ),
    )

    run = SimpleNamespace(
        batch_size=2,
        device=torch.device("cpu"),
        prototypes=torch.randn(2, 2),
        memory_labels=torch.tensor([0, 1]),
    )

    with pytest.raises(AssertionError, match="proto_predictions disagree"):
        comparison._evaluate_model(
            torch.nn.Identity(),
            embeddings,
            labels,
            run,
        )


# ---------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------


def test_json_safe_converts_supported_values_and_excludes_arrays() -> None:
    value = {
        "array": np.array([1, 2, 3]),
        "scalar": np.float32(0.5),
        "tensor": torch.tensor([4, 5]),
        "path": Path("reports/test"),
        "device": torch.device("cpu"),
        "prediction_artifacts": {"large": np.array([9, 9])},
        "predictions": {"head": np.array([0, 1])},
        "correct": {"head": np.array([True, False])},
    }

    result = comparison._json_safe(value)

    assert result["array"] == [1, 2, 3]
    assert result["scalar"] == pytest.approx(0.5)
    assert result["tensor"] == [4, 5]
    assert result["path"] == str(Path("reports/test"))
    assert result["device"] == "cpu"

    assert "prediction_artifacts" not in result
    assert "predictions" not in result
    assert "correct" not in result


# ---------------------------------------------------------------------
# Saving experiment results
# ---------------------------------------------------------------------


def make_save_results_fixture() -> dict[str, Any]:
    point = {
        "level": 0.0,
        "baseline_head_acc": (0.90, 0.01),
        "hopfield_head_acc": (0.92, 0.02),
        "baseline_proto_acc": (0.88, 0.01),
        "hopfield_proto_acc": (0.91, 0.02),
        "head_delta": 0.02,
        "proto_delta": 0.03,
    }

    statistics = [
        {
            "level": 0.0,
            "metric": "head",
            "delta": 0.02,
            "mcnemar_p_max": 0.05,
            "wilcoxon_p": 0.10,
            "winner": "inconclusive",
        },
        {
            "level": 0.0,
            "metric": "proto",
            "delta": 0.03,
            "mcnemar_p_max": 0.04,
            "wilcoxon_p": 0.08,
            "winner": "inconclusive",
        },
    ]

    timing_result = {
        "latency_ms_p50": {"median": 1.0},
        "latency_ms_p95": {"median": 2.0},
        "latency_per_image_us": {"median": 10.0},
        "throughput_img_s": {"median": 100.0},
        "peak_mem_mb": {"median": 1.0},
    }

    static_result = {
        "trainable_params": 100,
        "buffer_values": 20,
        "model_size_bytes": 1000,
        "macs_per_image": 200,
        "hook_macs_per_image": 200,
    }

    return {
        "selection": "raw",
        "config": {"seeds": [0]},
        "beta_selection_sets": [["clean", 0]],
        "selected_beta": 5.0,
        "selection_time_s": (0.1, 0.01),
        "herding_memory_time_s": (0.1, 0.01),
        "kmeans_memory_time_s": (0.2, 0.02),
        "prototype_time_s": (1.0, 0.1),
        "hopfield_build_time_s": (0.01, 0.001),
        "memory_timing": {
            "warmups_per_method_per_seed": 1,
            "measured_repeats_per_method_per_seed": 3,
        },
        "memory": {
            "total_patterns": 10,
            "per_class": 1,
            "embedding_dim": 2,
            "centered": True,
        },
        "cost": {
            "representative_seed": 0,
            "device": "cpu",
            "num_threads": 1,
            "e7": {
                "baseline": timing_result,
                "hopfield": timing_result,
            },
            "eval_time_s": {
                "baseline": {"median": 1.0, "iqr": 0.1},
                "hopfield": {"median": 1.1, "iqr": 0.1},
            },
            "e8": {
                "1": {
                    "baseline": timing_result,
                    "hopfield": timing_result,
                },
            },
            "static": {
                "baseline": static_result,
                "hopfield": static_result,
            },
            "hopfield_bridge": {
                "prototype_shape": [10, 2],
                "prototype_dtype": "torch.float32",
                "prototype_storage_bytes": 80,
                "model_size_bytes": 100,
                "buffer_values": 20,
                "floating_buffer_values": 20,
                "trainable_params": 0,
                "macs_per_image": 40,
                "beta": 5.0,
                "num_updates": 1,
                "centered": True,
            },
        },
        "summaries": {"clean": [point]},
        "statistics": {"clean": statistics},
        "robustness_auc": {
            "gaussian": {
                "baseline": (0.8, 0.01),
                "hopfield": (0.81, 0.02),
                "delta": 0.01,
                "n_pos": 1,
                "n_neg": 0,
                "wilcoxon_p": 0.5,
            },
        },
        "prediction_artifacts": {
            "clean__seed_0__level_0__labels": np.array([0, 1]),
        },
    }


def test_save_results_writes_all_artifacts(tmp_path: Path) -> None:
    raw = make_save_results_fixture()

    comparison.save_results(raw, tmp_path)

    expected_files = (
        "herding_comparison.json",
        "herding_comparison.csv",
        "herding_comparison.md",
        "herding_predictions.npz",
    )

    for filename in expected_files:
        assert (tmp_path / filename).is_file(), filename

    payload = json.loads(
        (tmp_path / "herding_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["selection"] == "raw"
    assert "prediction_artifacts" not in payload

    with np.load(tmp_path / "herding_predictions.npz") as archive:
        assert "clean__seed_0__level_0__labels" in archive.files
        np.testing.assert_array_equal(
            archive["clean__seed_0__level_0__labels"],
            [0, 1],
        )

    csv_text = (
        tmp_path / "herding_comparison.csv"
    ).read_text(encoding="utf-8")
    assert "baseline_head_mean" in csv_text
    assert "hopfield_head_mean" in csv_text

    report = (
        tmp_path / "herding_comparison.md"
    ).read_text(encoding="utf-8")
    assert "Raw Herding" in report
    assert "Robustness AUC" in report
    assert "Saved prediction artifacts" in report


def test_save_results_does_not_write_empty_csv(
    tmp_path: Path,
) -> None:
    raw = make_save_results_fixture()
    raw["summaries"] = {}
    raw["statistics"] = {}
    raw["robustness_auc"] = {}

    comparison.save_results(raw, tmp_path)

    assert (tmp_path / "herding_comparison.json").exists()
    assert (tmp_path / "herding_comparison.md").exists()
    assert (tmp_path / "herding_predictions.npz").exists()
    assert not (tmp_path / "herding_comparison.csv").exists()


# ---------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------


def test_main_passes_selection_and_saves_to_selection_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test.yaml")
    parser.add_argument("--retrain", action="store_true")

    monkeypatch.setattr(
        benchmark,
        "build_arg_parser",
        lambda description: parser,
    )

    cfg = {
        "reports_dir": str(tmp_path / "reports"),
        "quick": True,
    }

    monkeypatch.setattr(
        benchmark,
        "config_from_args",
        lambda args: cfg,
    )

    captured: dict[str, Any] = {}

    def fake_run_experiment(
        config: dict[str, Any],
        *,
        selection: str,
        retrain: bool = False,
    ) -> dict[str, Any]:
        captured["config"] = config
        captured["selection"] = selection
        captured["retrain"] = retrain
        return {"selection": selection}

    def fake_save_results(
        raw: dict[str, Any],
        output_dir: Path,
    ) -> None:
        captured["raw"] = raw
        captured["output_dir"] = output_dir

    monkeypatch.setattr(
        comparison,
        "run_experiment",
        fake_run_experiment,
    )
    monkeypatch.setattr(
        comparison,
        "save_results",
        fake_save_results,
    )

    comparison.main(["--selection", "l2"])

    assert captured["selection"] == "l2"
    assert captured["retrain"] is False
    assert captured["config"] is cfg
    assert captured["output_dir"] == (
        tmp_path / "reports" / "herding_comparison_l2"
    )