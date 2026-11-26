"""Evaluate ONE memory variant of the MNIST Hopfield experiment.

Pick the variant with ``--selection``:

    baseline  frozen CNN only (no memory)
    kmeans    CNN + Hopfield memory selected by K-means
    raw       CNN + Hopfield memory selected by Herding in raw-embedding space
    l2        CNN + Hopfield memory selected by Herding in L2-normalized space

Each run is fully independent. It builds and evaluates only its own model and
saves the same artifact set, covering the five criteria:

    1. classification accuracy   (head_acc, clean and at every noise level)
    2. robustness                (AUC of head_acc / proto_acc per noise type)
    3. retrieval performance     (proto_acc + every scalar summarize_outputs metric)
    4. memory selection time     (None for the baseline)
    5. computational cost        (E7, E8, eval time, params, size, MACs, bridge)

Nothing is compared inside this script. After running all four variants,
compare the saved artifacts with ``benchmarks.compare_methods``:

    python -m benchmarks.mnist_herding_comparison --selection baseline
    python -m benchmarks.mnist_herding_comparison --selection kmeans
    python -m benchmarks.mnist_herding_comparison --selection raw
    python -m benchmarks.mnist_herding_comparison --selection l2
    python -m benchmarks.compare_methods --reports-dir reports/mnist_herding_comparison
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmarks import mnist_prototype_retrieval as benchmark
from benchmarks.mnist_herding import timed_herding_memory_build

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "mnist_herding_comparision.yaml"
)

NUM_MNIST_CLASSES = 10
SELECTION_MODES = ("baseline", "raw", "l2", "kmeans", "all")
SCHEMA_VERSION = 2

# Folder / file names read by compare_memory_methods.py.
# The folder prefix differs from the old script ("herding_comparison_<mode>"),
# so old reports are never overwritten or mixed up with these.
OUTPUT_PREFIX = "memory_run_"
JSON_NAME = "herding_comparison.json"
NPZ_NAME = "herding_predictions.npz"
CSV_NAME = "herding_comparison.csv"
MD_NAME = "herding_comparison.md"

# Memory construction timing protocol (applies to the one method being run).
MEMORY_TIMING_WARMUPS = 1
MEMORY_TIMING_REPEATS = 3

ACCURACY_METRICS = ("head_acc", "proto_acc")


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def _sync_if_cuda(device: torch.device) -> None:
    """Wait for queued CUDA operations to finish before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_selection_flag(selection: str) -> bool:
    """
    Convert the selection name into the Herding normalization flag.

    Only the L2 variant normalizes. The baseline and K-means do not use
    Herding, so they map to False.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {selection!r}. "
            f"Choose one of: {', '.join(SELECTION_MODES)}."
        )

    return selection == "l2"


def _describe_selection(selection: str) -> tuple[str, str]:
    """Return (method name, selection space)."""
    if selection == "baseline":
        return "CNN baseline", "none (no memory)"
    if selection == "kmeans":
        return "K-means", "original raw CNN embeddings"
    if selection == "l2":
        return "L2 Herding", "L2-normalized embeddings"
    return "Raw Herding", "original raw embeddings"


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor into a detached CPU NumPy array."""
    return tensor.detach().cpu().numpy()


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep every scalar metric that summarize_outputs() returned."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _fingerprint_net(net: torch.nn.Module) -> str:
    """Hash of the CNN weights, so the comparison can prove a shared CNN."""
    digest = hashlib.sha256()
    for key, value in sorted(net.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _fingerprint_labels(labels: torch.Tensor) -> str:
    array = _tensor_to_numpy(labels).astype(np.int64, copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _ms(stat: dict[str, Any]) -> str:
    """Format a cost entry as mean ± std over seeds."""
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def level_key(level: float) -> str:
    """Key fragment for a noise level (compare script uses the same rule)."""
    return format(float(level), ".8g").replace(".", "p")


def prediction_key(kind: str, seed: int, level: float, name: str) -> str:
    return f"{kind}__seed_{seed}__level_{level_key(level)}__{name}"


# ---------------------------------------------------------------------
# Model evaluation and per-example predictions
# ---------------------------------------------------------------------


def _evaluate_model(
    net: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    run: benchmark.BaselineRun,
) -> dict[str, Any]:
    """Evaluate one model and retain metrics, predictions and correctness."""
    outputs = benchmark.forward_embeddings(
        net,
        embeddings,
        batch_size=run.batch_size,
        device=run.device,
    )

    metrics, correct, _ = benchmark.summarize_outputs(
        outputs,
        labels,
        run.prototypes,
        run.memory_labels,
    )

    head_predictions = outputs["head_pred"]

    if outputs.get("weights") is None:
        prototype_indices = benchmark.nearest_prototype(
            outputs["z"],
            run.prototypes,
        )
        proto_predictions = run.memory_labels[prototype_indices]
    else:
        proto_predictions = outputs["proto_pred"]

    # Verify that saved prototype predictions agree with the
    # correctness flags produced by summarize_outputs().
    expected_proto_correct = (
        proto_predictions.detach().cpu()
        == labels.detach().cpu()
    )

    actual_proto_correct = correct["proto"].detach().cpu()

    assert torch.equal(
        expected_proto_correct,
        actual_proto_correct,
    ), (
        "proto_predictions disagree with "
        "summarize_outputs correct['proto']."
    )

    return {
        # head_acc and proto_acc plus every other scalar diagnostic
        # (entropy, top-1 weight, ...) for the retrieval-performance criterion.
        "metrics": _scalar_metrics(metrics),
        "predictions": {
            "head": _tensor_to_numpy(head_predictions).astype(
                np.int64, copy=False
            ),
            "proto": _tensor_to_numpy(proto_predictions).astype(
                np.int64, copy=False
            ),
        },
        "correct": {
            "head": _tensor_to_numpy(correct["head"]).astype(
                np.bool_, copy=False
            ),
            "proto": _tensor_to_numpy(correct["proto"]).astype(
                np.bool_, copy=False
            ),
        },
    }


def evaluate_conditions(
    run: benchmark.BaselineRun,
    model: torch.nn.Module,
    noise_levels: dict[str, list[float]],
    seed: int,
) -> dict[str, dict[float, dict[str, Any]]]:
    """Evaluate ONE model on clean and every corrupted test condition."""
    labels = run.splits.test_labels
    clean_embeddings = run.test_embeddings("gaussian", 0.0)

    for noise_kind in benchmark.NOISE_TYPES:
        zero_noise_embeddings = run.test_embeddings(noise_kind, 0.0)

        if not torch.equal(zero_noise_embeddings, clean_embeddings):
            max_difference = (
                zero_noise_embeddings - clean_embeddings
            ).abs().max().item()

            raise RuntimeError(
                f"{noise_kind} at noise level 0.0 does not "
                f"match clean embeddings for seed {seed}. "
                f"Maximum absolute difference: {max_difference}."
            )

    clean_result = _evaluate_model(
        model, clean_embeddings, labels, run
    )

    results: dict[str, dict[float, dict[str, Any]]] = {
        "clean": {0.0: clean_result}
    }

    for noise_kind in benchmark.NOISE_TYPES:
        configured_levels = [
            float(level) for level in noise_levels[noise_kind]
        ]

        if not configured_levels or configured_levels[0] != 0.0:
            raise ValueError(
                f"{noise_kind} noise grid must start at 0.0."
            )

        per_level = {0.0: clean_result}

        for level in configured_levels:
            if level == 0.0:
                continue

            test_embeddings = run.test_embeddings(noise_kind, level)
            per_level[level] = _evaluate_model(
                model, test_embeddings, labels, run
            )

        results[noise_kind] = per_level

    return results


# ---------------------------------------------------------------------
# Memory construction timing (only the selected method is built/timed)
# ---------------------------------------------------------------------


def timed_kmeans_memory_build(
    run: benchmark.BaselineRun,
    *,
    per_class: int,
    center: bool,
    num_classes: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    float,
]:
    """
    Build and time the K-means memory from precomputed embeddings.

    Uses the shared benchmark builder with the run's seed, so the
    result is identical to the memory used by the K-means benchmark.
    """
    _sync_if_cuda(run.device)
    start = time.perf_counter()

    memory, memory_labels, center_offset = benchmark.build_memory(
        run.z_train,
        run.splits.train_labels,
        per_class=per_class,
        center=center,
        seed=run.seed,
        num_classes=num_classes,
    )

    _sync_if_cuda(run.device)
    elapsed = time.perf_counter() - start

    return memory, memory_labels, center_offset, elapsed


def measure_memory_build(
    run: benchmark.BaselineRun,
    *,
    per_class: int,
    center: bool,
    num_classes: int,
    selection: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    float,
    list[float],
]:
    """
    Build and time the memory of the requested selection mode only.

    The method receives unrecorded warmups and several measured
    repetitions. Both builders are deterministic for a given seed, so the
    memory from the last repetition is the memory used by the model.

    Returns:
        memory, memory_labels, center_offset,
        median build time, all timing samples.
    """
    if selection == "kmeans":

        def build():
            return timed_kmeans_memory_build(
                run,
                per_class=per_class,
                center=center,
                num_classes=num_classes,
            )

    elif selection in ("raw", "l2"):

        def build():
            return timed_herding_memory_build(
                run,
                per_class=per_class,
                center=center,
                num_classes=num_classes,
                normalize_for_selection=_normalize_selection_flag(
                    selection
                ),
            )

    else:
        raise ValueError(
            f"Selection mode {selection!r} does not build a memory."
        )

    for _ in range(MEMORY_TIMING_WARMUPS):
        build()

    samples: list[float] = []
    result = None

    for _ in range(MEMORY_TIMING_REPEATS):
        result = build()
        samples.append(float(result[3]))

    memory, memory_labels, center_offset, _ = result

    return (
        memory,
        memory_labels,
        center_offset,
        float(np.median(samples)),
        samples,
    )


# ---------------------------------------------------------------------
# Aggregation over seeds (one model)
# ---------------------------------------------------------------------


def _stat(values: list[float]) -> dict[str, Any]:
    mean, std = benchmark.mean_std(values)
    return {
        "mean": float(mean),
        "std": float(std),
        "per_seed": [float(v) for v in values],
    }


def aggregate_results(
    raw_results: dict[str, Any],
    noise_levels: dict[str, list[float]],
    seeds: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """summaries[kind] = [{level, metrics: {name: {mean, std, per_seed}}}]."""
    summaries: dict[str, list[dict[str, Any]]] = {}

    for kind, seed_results in raw_results.items():
        levels = (
            [0.0]
            if kind == "clean"
            else sorted({
                float(level) for level in noise_levels[kind]
            })
        )

        for seed in seeds:
            if seed not in seed_results:
                raise RuntimeError(
                    f"Missing {kind} results for seed {seed}."
                )

            for level in levels:
                if level not in seed_results[seed]:
                    raise RuntimeError(
                        f"Missing {kind} results for seed {seed} "
                        f"at level {level}."
                    )

        rows: list[dict[str, Any]] = []

        for level in levels:
            per_seed = [
                seed_results[seed][level]["metrics"] for seed in seeds
            ]
            names = [
                name
                for name in per_seed[0]
                if all(name in metrics for metrics in per_seed)
            ]

            rows.append({
                "level": level,
                "metrics": {
                    name: _stat([m[name] for m in per_seed])
                    for name in names
                },
            })

        summaries[kind] = rows

    return summaries


def aggregate_robustness(
    raw_results: dict[str, Any],
    noise_levels: dict[str, list[float]],
    seeds: list[int],
) -> dict[str, Any]:
    """Robustness AUC of each accuracy metric over the noise grid, per seed."""
    auc: dict[str, Any] = {}

    for kind in benchmark.NOISE_TYPES:
        levels = sorted({
            float(level) for level in noise_levels[kind]
        })

        if len(levels) < 2 or levels[0] != 0.0:
            raise ValueError(
                f"{kind} noise grid must contain at least two levels "
                "and start at 0.0 for robustness AUC."
            )

        auc[kind] = {}

        for metric in ACCURACY_METRICS:
            values = [
                benchmark.robust_auc(
                    levels,
                    [
                        raw_results[kind][seed][level]["metrics"][metric]
                        for level in levels
                    ],
                )
                for seed in seeds
            ]
            auc[kind][metric] = _stat(values)

    return auc


def build_prediction_artifacts(
    raw_results: dict[str, Any],
    runs: dict[int, benchmark.BaselineRun],
) -> dict[str, np.ndarray]:
    """Per-example predictions, correctness and labels for the NPZ file."""
    artifacts: dict[str, np.ndarray] = {}

    for kind, seed_results in raw_results.items():
        for seed, level_results in seed_results.items():
            labels = _tensor_to_numpy(
                runs[seed].splits.test_labels
            ).astype(np.int64, copy=False)

            for level, result in level_results.items():
                for name in ("head", "proto"):
                    artifacts[
                        prediction_key(kind, seed, level, f"{name}_pred")
                    ] = result["predictions"][name]

                    artifacts[
                        prediction_key(kind, seed, level, f"{name}_correct")
                    ] = result["correct"][name]

                artifacts[
                    prediction_key(kind, seed, level, "labels")
                ] = labels

    return artifacts


# ---------------------------------------------------------------------
# Computational cost measurements (this one model only)
# ---------------------------------------------------------------------


def _aggregate_over_seeds(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine per-seed cost entries into one entry per metric.

    Each input maps metric name -> {"median": ..., ...} for one seed. The
    output maps metric name -> {"median", "mean", "std", "per_seed"} where
    "per_seed" holds each seed's own median.
    """
    out: dict[str, Any] = {}

    for key, first in per_seed[0].items():
        if not (isinstance(first, dict) and "median" in first):
            continue

        values = [float(entry[key]["median"]) for entry in per_seed]
        mean, std = benchmark.mean_std(values)

        out[key] = {
            "median": float(np.median(values)),
            "mean": float(mean),
            "std": float(std),
            "per_seed": values,
        }

    return out


def measure_costs(
    cfg: dict[str, Any],
    runs: dict[int, benchmark.BaselineRun],
    nets: dict[int, torch.nn.Module],
    seeds: list[int],
    device: torch.device,
    *,
    train_times: list[float],
    prototype_times: list[float],
    hopfield_build_times: list[float],
) -> dict[str, Any]:
    """
    Measure this variant's cost on EVERY seed and aggregate across seeds.

    Latency, throughput, peak memory and full-evaluation time are measured
    once per seed (each seed has its own CNN and memory). Parameter counts,
    MACs and model size do not depend on the seed, so they are taken from
    the first seed.
    """
    c = cfg["cost"]

    timing = {
        "repeats": c["repeats"],
        "warmup": c["warmup"],
        "iters": c["iters"],
    }

    e7_seeds: list[dict[str, Any]] = []
    eval_seeds: list[dict[str, Any]] = []
    e8_seeds: dict[str, list[dict[str, Any]]] = {
        str(bs): [] for bs in c["batch_sizes"]
    }

    for seed in seeds:
        model = nets[seed]
        run = runs[seed]
        models: dict[str, torch.nn.Module] = {"model": model}

        print(
            f"  Cost, seed {seed}: E7 (batch {c['batch_size']}), "
            "full evaluation, E8 sweep."
        )

        e7_seeds.append(
            benchmark.interleaved_cost(
                models,
                c["batch_size"],
                device,
                **timing,
            )["model"]
        )

        eval_times = [
            benchmark.time_full_evaluation(
                model,
                run.splits.test_images,
                batch_size=c["batch_size"],
                device=device,
            )
            for _ in range(c["repeats"])
        ]
        eval_seeds.append({"eval": {"median": float(np.median(eval_times))}})

        for bs in c["batch_sizes"]:
            e8_seeds[str(bs)].append(
                benchmark.interleaved_cost(
                    models,
                    bs,
                    device,
                    **timing,
                )["model"]
            )

    # Seed-independent quantities, taken from the first seed.
    static_seed = seeds[0]
    model = nets[static_seed]

    assert isinstance(model, benchmark.MnistNet)

    breakdown = benchmark.analytical_macs(model)

    static = {
        "trainable_params": benchmark.count_trainable_params(model),
        "buffer_values": benchmark.count_buffer_values(model),
        "floating_buffer_values": benchmark.count_buffer_values(
            model,
            floating_only=True,
        ),
        "model_size_bytes": benchmark.model_size_bytes(model),
        "macs_breakdown": breakdown,
        "macs_per_image": sum(value for _, value in breakdown),
        "hook_macs_per_image": benchmark.hook_macs(model),
    }

    bridge_info = None

    if isinstance(model.bridge, benchmark.HopfieldBridge):
        bridge = model.bridge
        prototypes = bridge.prototypes

        bridge_info = {
            "prototype_shape": list(prototypes.shape),
            "prototype_dtype": str(prototypes.dtype),
            "prototype_storage_bytes": (
                prototypes.numel() * prototypes.element_size()
            ),
            "model_size_bytes": benchmark.model_size_bytes(bridge),
            "buffer_values": benchmark.count_buffer_values(bridge),
            "floating_buffer_values": benchmark.count_buffer_values(
                bridge,
                floating_only=True,
            ),
            "trainable_params": benchmark.count_trainable_params(
                bridge
            ),
            "macs_per_image": bridge.macs_per_image(),
            "beta": bridge.beta,
            "num_updates": bridge.num_updates,
            "centered": bridge.center is not None,
        }

    return {
        "seeds_measured": list(seeds),
        "static_seed": static_seed,
        "num_threads": torch.get_num_threads(),
        "device": str(device),
        "batch_size": c["batch_size"],
        "e7": _aggregate_over_seeds(e7_seeds),
        "e8": {
            bs: _aggregate_over_seeds(entries)
            for bs, entries in e8_seeds.items()
        },
        "eval_time_s": _aggregate_over_seeds(eval_seeds)["eval"],
        "static": static,
        "bridge": bridge_info,
        "train_time_s": benchmark.mean_std(train_times),
        "prototype_time_s": benchmark.mean_std(prototype_times),
        "hopfield_build_time_s": (
            benchmark.mean_std(hopfield_build_times)
            if hopfield_build_times
            else None
        ),
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def run_experiment(
    cfg: dict[str, Any],
    *,
    selection: str,
    retrain: bool = False,
) -> dict[str, Any]:
    """Evaluate ONE variant: the CNN baseline or CNN + one Hopfield memory."""
    normalize_for_selection = _normalize_selection_flag(selection)
    method_name, space_description = _describe_selection(selection)
    is_baseline = selection == "baseline"

    per_class = cfg["prototypes_per_class"]
    center = cfg["center_prototypes"]
    num_classes = NUM_MNIST_CLASSES
    seeds = [int(seed) for seed in cfg["seeds"]]

    threads = benchmark.configure_threads(
        cfg["cost"]["num_threads"]
    )
    device = benchmark.resolve_device(cfg["device"])

    print(
        f"Device: {device}; threads: {threads}; "
        f"torch {torch.__version__}"
    )
    print(
        f"Variant: {method_name}; "
        f"selection space = {space_description}; "
        "memory storage = raw embeddings."
    )
    if not is_baseline:
        print(
            "Memory timing: "
            f"{MEMORY_TIMING_WARMUPS} warmup and "
            f"{MEMORY_TIMING_REPEATS} measured repetitions."
        )

    data = benchmark.load_mnist(cfg["data_dir"])

    tuning = benchmark.tune_baseline(
        cfg,
        data,
        device,
        retrain=retrain,
    )

    if tuning is not None:
        print("Baseline tuning completed.")

    beta_values = cfg["beta_values"]
    validation_sets = cfg["beta_selection_sets"]
    noise_levels = benchmark.noise_grids(cfg)

    runs: dict[int, benchmark.BaselineRun] = {}
    nets: dict[int, torch.nn.Module] = {}
    fingerprints: dict[str, dict[str, str]] = {}

    curves: list[list[float]] = []

    selection_medians: list[float] = []
    selection_samples: dict[str, list[float]] = {}
    prototype_times: list[float] = []
    hopfield_build_times: list[float] = []

    # Stage 1: Prepare the CNN and, for Hopfield variants, the memory bank.
    for seed in seeds:
        run = benchmark.prepare_baseline_run(
            cfg,
            data,
            seed,
            device,
            retrain=retrain,
        )

        runs[seed] = run
        prototype_times.append(run.prototype_time_s)
        fingerprints[str(seed)] = {
            "cnn": _fingerprint_net(run.net),
            "test_labels": _fingerprint_labels(run.splits.test_labels),
        }

        if is_baseline:
            nets[seed] = run.net
            print(f"Prepared seed {seed}: baseline CNN (no memory).")
            continue

        (
            memory,
            memory_labels,
            center_offset,
            selection_elapsed,
            samples,
        ) = measure_memory_build(
            run,
            per_class=per_class,
            center=center,
            num_classes=num_classes,
            selection=selection,
        )

        selection_medians.append(selection_elapsed)
        selection_samples[str(seed)] = samples

        _sync_if_cuda(run.device)
        hopfield_build_start = time.perf_counter()

        hop_net = benchmark.build_hopfield_net(
            run.net,
            memory,
            memory_labels,
            beta=beta_values[0],
            center=center_offset,
        )

        _sync_if_cuda(run.device)

        hopfield_build_elapsed = (
            time.perf_counter() - hopfield_build_start
        )
        hopfield_build_times.append(hopfield_build_elapsed)

        nets[seed] = hop_net

        curves.append(
            benchmark.beta_curve(
                hop_net,
                run,
                beta_values,
                validation_sets,
            )
        )

        print(
            f"Prepared seed {seed}: "
            f"{per_class} memories per class; "
            f"{method_name} median = {selection_elapsed:.4f}s; "
            f"Hopfield construction = {hopfield_build_elapsed:.4f}s."
        )

    # Stage 2: Select beta using the mean validation curve (Hopfield only).
    selected_beta = None
    validation_mean_curve = None

    if not is_baseline:
        selected_beta, validation_mean_curve = benchmark.select_beta(
            beta_values,
            curves,
        )

        print(
            f"Validation-selected beta for {selection}: "
            f"{selected_beta:g}"
        )

        for seed in seeds:
            net = nets[seed]

            assert isinstance(net, benchmark.MnistNet)
            assert isinstance(net.bridge, benchmark.HopfieldBridge)

            net.bridge.set_retrieval(
                beta=selected_beta,
                num_updates=1,
            )

    # Stage 3: Evaluate clean and corrupted TEST embeddings.
    raw_results: dict[str, Any] = {
        kind: {} for kind in ("clean", *benchmark.NOISE_TYPES)
    }

    for seed in seeds:
        per_kind = evaluate_conditions(
            runs[seed],
            nets[seed],
            noise_levels,
            seed,
        )

        for kind, per_level in per_kind.items():
            raw_results[kind][seed] = per_level

        print(f"Evaluated seed {seed}.")

    # Stage 4: Aggregate accuracy / retrieval, then robustness AUC.
    summaries = aggregate_results(raw_results, noise_levels, seeds)
    robustness = aggregate_robustness(raw_results, noise_levels, seeds)

    # Stage 5: Measure computational cost on every seed (this model only).
    print("\nMeasuring computational cost on every seed.")

    cost_results = measure_costs(
        cfg,
        runs,
        nets,
        seeds,
        device,
        train_times=[runs[seed].train_time_s for seed in seeds],
        prototype_times=prototype_times,
        hopfield_build_times=hopfield_build_times,
    )

    # Stage 6: Package results.
    selection_time = None

    if not is_baseline:
        selection_time = {
            "per_seed_median_s": selection_medians,
            "mean_std_s": [
                float(v) for v in benchmark.mean_std(selection_medians)
            ],
            "samples_s": selection_samples,
            "warmups": MEMORY_TIMING_WARMUPS,
            "repeats": MEMORY_TIMING_REPEATS,
            "per_seed_summary": "median",
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "selection": selection,
        "method": selection,
        "method_label": method_name,
        "normalize_for_selection": normalize_for_selection,
        "config": cfg,
        "environment": benchmark.environment_info(device),
        "seeds": seeds,
        "fingerprints": fingerprints,
        "memory": None if is_baseline else {
            "per_class": per_class,
            "num_classes": num_classes,
            "total_patterns": per_class * num_classes,
            "centered": center,
            "embedding_dim": int(runs[seeds[0]].z_train.shape[1]),
            "method": method_name,
            "selection_normalized": normalize_for_selection,
            "selection_space": selection,
            "storage_representation": "raw",
        },
        "beta_values": list(beta_values),
        "beta_selection_sets": list(validation_sets),
        "selected_beta": selected_beta,
        "validation_mean_curve": validation_mean_curve,
        "selection_time": selection_time,
        "hopfield_build_time_s": (
            None
            if is_baseline
            else [
                float(v)
                for v in benchmark.mean_std(hopfield_build_times)
            ]
        ),
        "prototype_time_s": [
            float(v) for v in benchmark.mean_std(prototype_times)
        ],
        "summaries": summaries,
        "robustness_auc": robustness,
        "cost": cost_results,
    }

    return {
        "result": result,
        "prediction_artifacts": build_prediction_artifacts(
            raw_results, runs
        ),
    }


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """
    Convert nested results into JSON-compatible values.

    Per-example arrays live in the NPZ artifact, not in the JSON.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, torch.device):
        return str(value)

    return value


# ---------------------------------------------------------------------
# Save JSON, CSV, NPZ, and Markdown reports
# ---------------------------------------------------------------------


def save_results(
    output: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save this variant's artifacts (JSON, NPZ, CSV, Markdown)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result = output["result"]
    selection = result["selection"]

    # JSON: summaries, robustness, selection time, cost, fingerprints.
    (output_dir / JSON_NAME).write_text(
        json.dumps(_json_safe(result), indent=2, default=str),
        encoding="utf-8",
    )

    # NPZ: per-example labels, predictions and correctness.
    np.savez_compressed(
        output_dir / NPZ_NAME,
        **output["prediction_artifacts"],
    )

    # CSV: one row per (condition, level).
    rows: list[dict[str, Any]] = []

    for kind, points in result["summaries"].items():
        for point in points:
            row: dict[str, Any] = {
                "selection": selection,
                "condition": kind,
                "level": point["level"],
            }

            for name, stat in point["metrics"].items():
                row[f"{name}_mean"] = stat["mean"]
                row[f"{name}_std"] = stat["std"]

            rows.append(row)

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with (output_dir / CSV_NAME).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown report for this single variant.
    cost = result["cost"]
    method_name, space_description = _describe_selection(selection)
    selection_time = result["selection_time"]

    lines = [
        f"# MNIST memory experiment: {method_name}",
        "",
        f"- Variant: **{selection}**",
        f"- Selection space: {space_description}",
        f"- Seeds: {result['seeds']}",
    ]

    if result["memory"] is not None:
        memory = result["memory"]
        lines += [
            (
                f"- Memory: {memory['total_patterns']} patterns "
                f"({memory['per_class']} per class)"
            ),
            f"- Embedding dimension: {memory['embedding_dim']}",
            f"- Centered: {memory['centered']}",
            f"- Beta selection sets: {result['beta_selection_sets']}",
            f"- Validation-selected beta: `{result['selected_beta']:g}`",
        ]
    else:
        lines.append("- Memory: none (CNN head only)")

    # 1. Classification accuracy.
    lines += ["", "## 1. Classification accuracy (head_acc)", ""]
    lines += [
        "| Condition | Level | head_acc |",
        "|---|---:|---:|",
    ]

    for kind, points in result["summaries"].items():
        for point in points:
            stat = point["metrics"]["head_acc"]
            lines.append(
                f"| {kind} | {point['level']:g} "
                f"| {stat['mean']:.4f} ± {stat['std']:.4f} |"
            )

    # 2. Robustness.
    lines += [
        "",
        "## 2. Robustness AUC",
        "",
        "| Noise type | head_acc AUC | proto_acc AUC |",
        "|---|---:|---:|",
    ]

    for kind, stats in result["robustness_auc"].items():
        lines.append(
            f"| {kind} "
            f"| {stats['head_acc']['mean']:.4f} "
            f"± {stats['head_acc']['std']:.4f} "
            f"| {stats['proto_acc']['mean']:.4f} "
            f"± {stats['proto_acc']['std']:.4f} |"
        )

    # 3. Retrieval performance (clean).
    lines += [
        "",
        "## 3. Retrieval performance (clean test data)",
        "",
        "| Metric | Mean ± std |",
        "|---|---:|",
    ]

    for name, stat in result["summaries"]["clean"][0]["metrics"].items():
        lines.append(
            f"| {name} | {stat['mean']:.4f} ± {stat['std']:.4f} |"
        )

    # 4. Memory selection time.
    lines += ["", "## 4. Memory selection time", ""]

    if selection_time is None:
        lines.append("- Not applicable: the baseline builds no memory.")
    else:
        mean, std = selection_time["mean_std_s"]
        lines += [
            f"- {method_name}: {mean:.4f} ± {std:.4f} seconds",
            f"- Warmups per seed: {selection_time['warmups']}",
            f"- Measured repetitions per seed: {selection_time['repeats']}",
            "- Per-seed summary: median",
            (
                f"- Hopfield network construction: "
                f"{result['hopfield_build_time_s'][0]:.4f} "
                f"± {result['hopfield_build_time_s'][1]:.4f} seconds"
            ),
        ]

    lines += [
        (
            f"- Embedding extraction and class prototypes: "
            f"{result['prototype_time_s'][0]:.4f} "
            f"± {result['prototype_time_s'][1]:.4f} seconds"
        ),
    ]

    # 5. Computational cost.
    e7 = cost["e7"]
    static = cost["static"]

    lines += [
        "",
        "## 5. Computational cost",
        "",
        f"- Timing measured on seeds: {cost['seeds_measured']} "
        "(mean ± std over seeds)",
        f"- Device: `{cost['device']}`; CPU threads: {cost['num_threads']}",
        (
            f"- E7 (batch {cost['batch_size']}): latency p50 "
            f"{_ms(e7['latency_ms_p50'])} ms, p95 "
            f"{_ms(e7['latency_ms_p95'])} ms, "
            f"{_ms(e7['latency_per_image_us'])} µs/image, "
            f"{_ms(e7['throughput_img_s'])} images/s, "
            f"peak memory estimate {_ms(e7['peak_mem_mb'])} MB"
        ),
        f"- Full test-set evaluation: {_ms(cost['eval_time_s'])} s",
        (
            "- Parameters, size and MACs do not depend on the seed "
            f"(taken from seed {cost['static_seed']})"
        ),
        f"- Trainable parameters: {static['trainable_params']}",
        f"- Buffer values: {static['buffer_values']}",
        f"- Serialized size: {static['model_size_bytes']} bytes",
        f"- Analytical MACs/image: {static['macs_per_image']}",
        f"- Hook MACs/image: {static['hook_macs_per_image']}",
        "",
        "### E8: batch-size sweep",
        "",
        "| Batch size | Latency p50 (ms) | Latency/image (µs) | Throughput (images/s) |",
        "|---:|---:|---:|---:|",
    ]

    for batch_size, entry in cost["e8"].items():
        lines.append(
            f"| {batch_size} "
            f"| {_ms(entry['latency_ms_p50'])} "
            f"| {_ms(entry['latency_per_image_us'])} "
            f"| {_ms(entry['throughput_img_s'])} |"
        )

    bridge = cost["bridge"]

    if bridge is not None:
        lines += [
            "",
            "### Isolated Hopfield bridge footprint",
            "",
            f"- Prototype shape: `{bridge['prototype_shape']}`",
            f"- Prototype dtype: `{bridge['prototype_dtype']}`",
            f"- Prototype storage: {bridge['prototype_storage_bytes']} bytes",
            f"- Serialized bridge state: {bridge['model_size_bytes']} bytes",
            f"- Bridge MACs per image: {bridge['macs_per_image']}",
            f"- Retrieval beta: `{bridge['beta']:g}`",
            f"- Retrieval updates: {bridge['num_updates']}",
            f"- Centered: {bridge['centered']}",
        ]

    lines += [
        "",
        "## Notes",
        "",
        "- This run evaluates only its own model; nothing is compared here.",
        "- Compare variants with `compare_memory_methods.py`.",
        "- Per-example predictions are in the NPZ file for later paired tests.",
        "- Fingerprints of the CNN weights and test labels are saved per seed so the comparison can verify a shared CNN.",
        "- Memories are selected per class and stored in raw embedding space.",
        "- The L2 variant changes only the selection objective, not the stored representation.",
        "- The L2 variant is an iCaRL-style normalized-feature Herding procedure, not a complete reproduction of iCaRL.",
        "- Model size is the serialized state_dict size.",
        "- CPU peak-memory estimates are not measurements of total process RAM.",
        "",
    ]

    (output_dir / MD_NAME).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = benchmark.build_arg_parser(
        "Run ONE variant (baseline, K-means, raw Herding or L2-normalized "
        "Herding) of the MNIST Hopfield benchmark and save its artifacts."
    )
    parser.set_defaults(config=DEFAULT_CONFIG_PATH)

    parser.add_argument(
        "--selection",
        choices=SELECTION_MODES,
        default="raw",
        help=(
            "Variant to run: "
            "'baseline' evaluates the CNN alone; "
            "'raw' uses raw-embedding Herding; "
            "'l2' uses L2-normalized Herding; "
            "'kmeans' uses K-means; "
            "'all' runs all 4 variants and compares them. "
            "Selected memories are stored in raw embedding space. "
            "Default: raw."
        ),
    )

    args = parser.parse_args(argv)
    cfg = benchmark.config_from_args(args)

    selection = args.selection

    if selection == "all":
        variants = ["baseline", "kmeans", "raw", "l2"]
        for i, var in enumerate(variants):
            print(
                f"\n{'=' * 70}\nRunning variant '{var}' from {args.config}"
                f"{' (quick mode)' if cfg.get('quick') else ''}\n{'=' * 70}"
            )
            out = run_experiment(
                cfg,
                selection=var,
                retrain=args.retrain and (i == 0),
            )
            out_dir = Path(cfg["reports_dir"]) / f"{OUTPUT_PREFIX}{var}"
            save_results(out, out_dir)
            print(f"Variant '{var}' completed and saved to: {out_dir}")

        print(f"\n{'=' * 70}\nAll 4 variants completed. Comparing artifacts...\n{'=' * 70}")
        from benchmarks import compare_methods

        compare_methods.main(["--reports-dir", str(cfg["reports_dir"])])
        return

    print(
        f"Running variant '{selection}' from {args.config}"
        f"{' (quick mode)' if cfg.get('quick') else ''}"
    )

    output = run_experiment(
        cfg,
        selection=selection,
        retrain=args.retrain,
    )

    output_dir = Path(cfg["reports_dir"]) / f"{OUTPUT_PREFIX}{selection}"

    save_results(output, output_dir)

    print(f"\nVariant: {selection}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()