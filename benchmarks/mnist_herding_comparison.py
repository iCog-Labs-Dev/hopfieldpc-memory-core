"""Evaluate K-means, raw Herding, or L2 Herding Hopfield memories in MNIST."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks import mnist_prototype_retrieval as benchmark
from benchmarks.mnist_herding import timed_herding_memory_build

NUM_MNIST_CLASSES = 10
SELECTION_MODES = ("raw", "l2", "kmeans")

# Memory construction timing protocol.
MEMORY_TIMING_WARMUPS = 1
MEMORY_TIMING_REPEATS = 3


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

    K-means does not use this flag, so it maps to False. In a K-means
    run the Herding timing therefore uses raw-embedding Herding.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {selection!r}. "
            f"Choose one of: {', '.join(SELECTION_MODES)}."
        )

    return selection == "l2"


def _describe_selection(selection: str) -> tuple[str, str, str]:
    """Return (method name, selection space, Herding-timing label)."""
    if selection == "kmeans":
        method = "K-means"
        space = "original raw CNN embeddings"
    elif selection == "l2":
        method = "L2 Herding"
        space = "L2-normalized embeddings"
    else:
        method = "Raw Herding"
        space = "original raw embeddings"

    herding_label = "L2 Herding" if selection == "l2" else "Raw Herding"
    return method, space, herding_label


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor into a detached CPU NumPy array."""
    return tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------
# Model evaluation and per-example predictions
# ---------------------------------------------------------------------


def _evaluate_model(
    net: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    run: benchmark.BaselineRun,
) -> dict[str, Any]:
    """Evaluate one model and retain predictions and correctness."""
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
        "metrics": {
            "head_acc": float(metrics["head_acc"]),
            "proto_acc": float(metrics["proto_acc"]),
        },
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


def evaluate_pair(
    run: benchmark.BaselineRun,
    hop_net: benchmark.MnistNet,
    embeddings: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate baseline and Hopfield on exactly the same embeddings."""
    labels = run.splits.test_labels

    return {
        "baseline": _evaluate_model(
            run.net, embeddings, labels, run
        ),
        "hopfield": _evaluate_model(
            hop_net, embeddings, labels, run
        ),
    }


# ---------------------------------------------------------------------
# Memory construction timing
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


def measure_memory_builds(
    run: benchmark.BaselineRun,
    *,
    per_class: int,
    center: bool,
    num_classes: int,
    normalize_for_selection: bool,
    selection: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    float,
    float,
    list[float],
    list[float],
]:
    """
    Measure K-means and Herding memory construction fairly, and return
    the memory belonging to the requested selection mode.

    Each method receives one unrecorded warmup and several measured
    runs. The order alternates between measured repetitions.

    Returns:
        memory, memory_labels, center_offset  (of the requested mode),
        median Herding time, median K-means time,
        all Herding timing samples, all K-means timing samples.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode {selection!r}.")

    def run_kmeans():
        return timed_kmeans_memory_build(
            run,
            per_class=per_class,
            center=center,
            num_classes=num_classes,
        )

    def run_herding():
        return timed_herding_memory_build(
            run,
            per_class=per_class,
            center=center,
            num_classes=num_classes,
            normalize_for_selection=normalize_for_selection,
        )

    # Warmup: alternate which method runs first by seed.
    if run.seed % 2 == 0:
        warmup_results = (run_kmeans(), run_herding())
    else:
        warmup_results = (run_herding(), run_kmeans())

    del warmup_results

    kmeans_times: list[float] = []
    herding_times: list[float] = []
    last_results: dict[str, tuple] = {}

    for repeat in range(MEMORY_TIMING_REPEATS):
        order = (
            (("kmeans", run_kmeans), ("herding", run_herding))
            if repeat % 2 == 0
            else (("herding", run_herding), ("kmeans", run_kmeans))
        )

        for name, build in order:
            result = build()
            last_results[name] = result

            if name == "kmeans":
                kmeans_times.append(float(result[3]))
            else:
                herding_times.append(float(result[3]))

    # Both builds are deterministic, so the last repetition's memory
    # is the memory for the requested method.
    chosen = last_results["kmeans" if selection == "kmeans" else "herding"]
    memory, memory_labels, center_offset, _ = chosen

    return (
        memory,
        memory_labels,
        center_offset,
        float(np.median(herding_times)),
        float(np.median(kmeans_times)),
        herding_times,
        kmeans_times,
    )


# ---------------------------------------------------------------------
# Statistical evaluation
# ---------------------------------------------------------------------


def calculate_paired_statistics(
    per_seed_results: dict[int, dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    """Calculate paired statistics using the shared benchmark rules."""
    if metric not in ("head", "proto"):
        raise ValueError(f"Unknown statistical metric: {metric!r}")

    seed_deltas: list[float] = []
    mcnemar_results: list[dict[str, Any]] = []
    seed_signs: list[int] = []

    baseline_accuracies: list[float] = []
    hopfield_accuracies: list[float] = []

    for seed in sorted(per_seed_results):
        result = per_seed_results[seed]

        baseline = result["baseline"]
        hopfield = result["hopfield"]

        baseline_accuracy = float(
            baseline["metrics"][f"{metric}_acc"]
        )
        hopfield_accuracy = float(
            hopfield["metrics"][f"{metric}_acc"]
        )

        baseline_accuracies.append(baseline_accuracy)
        hopfield_accuracies.append(hopfield_accuracy)

        seed_deltas.append(
            hopfield_accuracy - baseline_accuracy
        )

        baseline_correct = torch.as_tensor(
            baseline["correct"][metric],
            dtype=torch.bool,
        )
        hopfield_correct = torch.as_tensor(
            hopfield["correct"][metric],
            dtype=torch.bool,
        )

        test = benchmark.mcnemar_test(
            baseline_correct,
            hopfield_correct,
        )
        mcnemar_results.append(test)

        seed_signs.append(
            (test["c"] > test["b"])
            - (test["c"] < test["b"])
        )

    if not seed_deltas:
        raise ValueError(
            "Cannot calculate statistics without seed results."
        )

    baseline_summary = benchmark.mean_std(baseline_accuracies)
    hopfield_summary = benchmark.mean_std(hopfield_accuracies)

    delta = hopfield_summary[0] - baseline_summary[0]

    seed_statistics = benchmark.seed_stats(seed_deltas)

    mcnemar_p_max = max(
        result["p"] for result in mcnemar_results
    )

    winner = benchmark.decide_winner(
        delta,
        mcnemar_p_max,
        seed_signs,
        seed_statistics["wilcoxon_p"],
    )

    return {
        "baseline": baseline_summary,
        "hopfield": hopfield_summary,
        "delta": delta,
        "mcnemar_per_seed": mcnemar_results,
        "mcnemar_p_max": mcnemar_p_max,
        "seed_deltas": seed_statistics["seed_deltas"],
        "n_pos": seed_statistics["n_pos"],
        "n_neg": seed_statistics["n_neg"],
        "wilcoxon_p": seed_statistics["wilcoxon_p"],
        "seed_signs": seed_signs,
        "winner": winner,
    }


# ---------------------------------------------------------------------
# Computational cost measurements
# ---------------------------------------------------------------------


def measure_herding_costs(
    cfg: dict[str, Any],
    run: benchmark.BaselineRun,
    hop: benchmark.MnistNet,
    beta_star: float,
    device: torch.device,
    *,
    train_times: list[float],
    prototype_times: list[float],
    selected_times: list[float],
    herding_memory_times: list[float],
    hopfield_build_times: list[float],
    kmeans_memory_times: list[float],
    representative_seed: int,
) -> dict[str, Any]:
    """Measure full-model and isolated Hopfield bridge costs."""
    c = cfg["cost"]

    assert isinstance(hop.bridge, benchmark.HopfieldBridge)

    hop.bridge.set_retrieval(
        beta=beta_star,
        num_updates=1,
    )

    models: dict[str, torch.nn.Module] = {
        "baseline": run.net,
        "hopfield": hop,
    }

    timing = {
        "repeats": c["repeats"],
        "warmup": c["warmup"],
        "iters": c["iters"],
    }

    print(
        f"  Cost E7: batch size {c['batch_size']}, "
        f"{c['repeats']} interleaved repeats."
    )

    e7 = benchmark.interleaved_cost(
        models,
        c["batch_size"],
        device,
        **timing,
    )

    print("  Cost: full test-set evaluation time.")

    eval_times: dict[str, list[float]] = {
        name: [] for name in models
    }

    model_names = list(models)

    for repeat in range(c["repeats"]):
        ordered_names = (
            model_names
            if repeat % 2 == 0
            else model_names[::-1]
        )

        for name in ordered_names:
            elapsed = benchmark.time_full_evaluation(
                models[name],
                run.splits.test_images,
                batch_size=c["batch_size"],
                device=device,
            )
            eval_times[name].append(elapsed)

    print(f"  Cost E8: batch-size sweep {c['batch_sizes']}.")

    e8 = {
        bs: benchmark.interleaved_cost(
            models,
            bs,
            device,
            **timing,
        )
        for bs in c["batch_sizes"]
    }

    print("  Cost: complete-model storage and MAC measurements.")

    static: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        assert isinstance(model, benchmark.MnistNet)

        breakdown = benchmark.analytical_macs(model)

        static[name] = {
            "trainable_params": benchmark.count_trainable_params(
                model
            ),
            "buffer_values": benchmark.count_buffer_values(model),
            "floating_buffer_values": benchmark.count_buffer_values(
                model,
                floating_only=True,
            ),
            "model_size_bytes": benchmark.model_size_bytes(model),
            "macs_breakdown": breakdown,
            "macs_per_image": sum(
                value for _, value in breakdown
            ),
            "hook_macs_per_image": benchmark.hook_macs(model),
        }

    bridge = hop.bridge
    prototypes = bridge.prototypes

    hopfield_bridge = {
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
        "trainable_params": benchmark.count_trainable_params(bridge),
        "macs_per_image": bridge.macs_per_image(),
        "beta": bridge.beta,
        "num_updates": bridge.num_updates,
        "centered": bridge.center is not None,
    }

    return {
        "representative_seed": representative_seed,
        "num_threads": torch.get_num_threads(),
        "device": str(device),
        "e7": e7,
        "eval_time_s": {
            name: benchmark.median_iqr(values)
            for name, values in eval_times.items()
        },
        "e8": {
            str(bs): result
            for bs, result in e8.items()
        },
        "static": static,
        "hopfield_bridge": hopfield_bridge,
        "train_time_s": benchmark.mean_std(train_times),
        "prototype_time_s": benchmark.mean_std(prototype_times),
        "selected_memory_time_s": benchmark.mean_std(selected_times),
        "herding_memory_time_s": benchmark.mean_std(
            herding_memory_times
        ),
        "kmeans_memory_time_s": benchmark.mean_std(kmeans_memory_times),
        "hopfield_build_time_s": benchmark.mean_std(
            hopfield_build_times
        ),
        "memory_timing_protocol": {
            "warmups_per_method_per_seed": MEMORY_TIMING_WARMUPS,
            "measured_repeats_per_method_per_seed": (
                MEMORY_TIMING_REPEATS
            ),
            "per_seed_summary": "median",
            "measurement_order": "alternating",
        },
    }


# ---------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------


def aggregate_results(
    raw_results: dict[str, Any],
    noise_levels: dict[str, list[float]],
    seeds: list[int],
    metric_names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate metrics and calculate paired statistical results."""
    summaries: dict[str, Any] = {}
    statistics: dict[str, Any] = {}

    for kind, seed_results in raw_results.items():
        levels = (
            [0.0]
            if kind == "clean"
            else sorted({
                float(level)
                for level in noise_levels[kind]
            })
        )

        summaries[kind] = []
        statistics[kind] = []

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

        for level in levels:
            per_seed = {
                seed: seed_results[seed][level]
                for seed in seeds
            }

            row: dict[str, Any] = {"level": level}

            for metric in metric_names:
                values = [
                    (
                        seed_results[seed][level]["baseline"]["metrics"][
                            metric.removeprefix("baseline_")
                        ]
                        if metric.startswith("baseline_")
                        else seed_results[seed][level]["hopfield"]["metrics"][
                            metric.removeprefix("hopfield_")
                        ]
                    )
                    for seed in seeds
                ]

                row[metric] = benchmark.mean_std(values)

            row["head_delta"] = (
                row["hopfield_head_acc"][0]
                - row["baseline_head_acc"][0]
            )
            row["proto_delta"] = (
                row["hopfield_proto_acc"][0]
                - row["baseline_proto_acc"][0]
            )

            summaries[kind].append(row)

            for metric in ("head", "proto"):
                result = calculate_paired_statistics(
                    per_seed,
                    metric=metric,
                )

                statistics[kind].append({
                    "level": level,
                    "metric": metric,
                    **result,
                })

    return summaries, statistics


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def run_experiment(
    cfg: dict[str, Any],
    *,
    selection: str,
    retrain: bool = False,
) -> dict[str, Any]:
    """Evaluate CNN and CNN + Hopfield with the selected memory method."""
    normalize_for_selection = _normalize_selection_flag(selection)
    method_name, space_description, _ = _describe_selection(selection)

    per_class = cfg["prototypes_per_class"]
    center = cfg["center_prototypes"]
    num_classes = NUM_MNIST_CLASSES

    threads = benchmark.configure_threads(
        cfg["cost"]["num_threads"]
    )
    device = benchmark.resolve_device(cfg["device"])

    print(
        f"Device: {device}; threads: {threads}; "
        f"torch {torch.__version__}"
    )
    print(
        f"Memory selection: {method_name}; "
        f"selection space = {space_description}; "
        "memory storage = raw embeddings."
    )
    print(
        "Memory timing: "
        f"{MEMORY_TIMING_WARMUPS} warmup and "
        f"{MEMORY_TIMING_REPEATS} measured repetitions per method."
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
    memories: dict[int, torch.Tensor] = {}
    memory_label_sets: dict[int, torch.Tensor] = {}
    nets: dict[int, torch.nn.Module] = {}

    curves: list[list[float]] = []

    selected_times: list[float] = []
    herding_memory_times: list[float] = []
    kmeans_memory_times: list[float] = []
    prototype_times: list[float] = []
    hopfield_build_times: list[float] = []

    herding_timing_samples: dict[int, list[float]] = {}
    kmeans_timing_samples: dict[int, list[float]] = {}

    # Stage 1: Prepare CNN, memory banks, and Hopfield networks.
    for seed in cfg["seeds"]:
        run = benchmark.prepare_baseline_run(
            cfg,
            data,
            seed,
            device,
            retrain=retrain,
        )

        prototype_times.append(run.prototype_time_s)

        (
            memory,
            memory_labels,
            center_offset,
            herding_elapsed,
            kmeans_elapsed,
            herding_samples,
            kmeans_samples,
        ) = measure_memory_builds(
            run,
            per_class=per_class,
            center=center,
            num_classes=num_classes,
            normalize_for_selection=normalize_for_selection,
            selection=selection,
        )

        # Time of the method that actually supplies the memory.
        selected_elapsed = (
            kmeans_elapsed
            if selection == "kmeans"
            else herding_elapsed
        )

        selected_times.append(selected_elapsed)
        herding_memory_times.append(herding_elapsed)
        kmeans_memory_times.append(kmeans_elapsed)

        herding_timing_samples[seed] = herding_samples
        kmeans_timing_samples[seed] = kmeans_samples

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

        runs[seed] = run
        memories[seed] = memory
        memory_label_sets[seed] = memory_labels
        nets[seed] = hop_net

        seed_curve = benchmark.beta_curve(
            hop_net,
            run,
            beta_values,
            validation_sets,
        )
        curves.append(seed_curve)

        print(
            f"Prepared seed {seed}: "
            f"{per_class} memories per class; "
            f"selected method = {selection}; "
            f"Herding median = {herding_elapsed:.4f}s; "
            f"K-means median = {kmeans_elapsed:.4f}s; "
            f"selected-method time = {selected_elapsed:.4f}s; "
            f"Hopfield construction = {hopfield_build_elapsed:.4f}s."
        )

    # Stage 2: Select beta using the mean validation curve.
    selected_beta, validation_mean_curve = benchmark.select_beta(
        beta_values,
        curves,
    )

    print(
        f"Validation-selected beta for {selection}: "
        f"{selected_beta:g}"
    )

    # Stage 3: Evaluate clean and corrupted test embeddings.
    raw_results: dict[str, Any] = {
        kind: {}
        for kind in ("clean", *benchmark.NOISE_TYPES)
    }

    for seed in cfg["seeds"]:
        run = runs[seed]
        net = nets[seed]

        assert isinstance(net, benchmark.MnistNet)
        assert isinstance(net.bridge, benchmark.HopfieldBridge)

        net.bridge.set_retrieval(
            beta=selected_beta,
            num_updates=1,
        )

        clean_embeddings = run.test_embeddings("gaussian", 0.0)

        for noise_kind in benchmark.NOISE_TYPES:
            zero_noise_embeddings = run.test_embeddings(
                noise_kind,
                0.0,
            )

            if not torch.equal(
                zero_noise_embeddings,
                clean_embeddings,
            ):
                max_difference = (
                    zero_noise_embeddings - clean_embeddings
                ).abs().max().item()

                raise RuntimeError(
                    f"{noise_kind} at noise level 0.0 does not "
                    f"match clean embeddings for seed {seed}. "
                    f"Maximum absolute difference: {max_difference}."
                )

        clean_result = evaluate_pair(
            run,
            net,
            clean_embeddings,
        )

        raw_results["clean"][seed] = {
            0.0: clean_result
        }

        for noise_kind in benchmark.NOISE_TYPES:
            configured_levels = [
                float(level)
                for level in noise_levels[noise_kind]
            ]

            if (
                not configured_levels
                or configured_levels[0] != 0.0
            ):
                raise ValueError(
                    f"{noise_kind} noise grid must start at 0.0."
                )

            seed_results = {0.0: clean_result}

            for level in configured_levels:
                if level == 0.0:
                    continue

                test_embeddings = run.test_embeddings(
                    noise_kind,
                    level,
                )

                seed_results[level] = evaluate_pair(
                    run,
                    net,
                    test_embeddings,
                )

            raw_results[noise_kind][seed] = seed_results

        print(f"Evaluated seed {seed}.")

    # Stage 4: Aggregate accuracy and statistical results.
    metric_names = (
        "baseline_head_acc",
        "hopfield_head_acc",
        "baseline_proto_acc",
        "hopfield_proto_acc",
    )

    summaries, statistics = aggregate_results(
        raw_results,
        noise_levels,
        list(cfg["seeds"]),
        metric_names,
    )

    # Stage 5: Calculate robustness AUC separately per seed.
    auc: dict[str, Any] = {}

    for kind in benchmark.NOISE_TYPES:
        levels = sorted({
            float(level)
            for level in noise_levels[kind]
        })

        if len(levels) < 2 or levels[0] != 0.0:
            raise ValueError(
                f"{kind} noise grid must contain at least two levels "
                "and start at 0.0 for robustness AUC."
            )

        per_seed: dict[str, list[float]] = {
            "baseline": [],
            "hopfield": [],
        }

        for seed in cfg["seeds"]:
            seed_results = raw_results[kind][seed]

            for model in ("baseline", "hopfield"):
                accuracies = [
                    seed_results[level][model]["metrics"]["head_acc"]
                    for level in levels
                ]

                seed_auc = benchmark.robust_auc(
                    levels,
                    accuracies,
                )
                per_seed[model].append(seed_auc)

        auc[kind] = {
            model: benchmark.mean_std(values)
            for model, values in per_seed.items()
        }

        auc_seed_deltas = [
            hopfield_auc - baseline_auc
            for baseline_auc, hopfield_auc in zip(
                per_seed["baseline"],
                per_seed["hopfield"],
            )
        ]

        auc[kind]["delta"] = (
            auc[kind]["hopfield"][0]
            - auc[kind]["baseline"][0]
        )

        # Same across-seed paired statistics as the shared benchmark.
        auc[kind].update(
            benchmark.seed_stats(auc_seed_deltas)
        )

    # Stage 6: Measure computational costs.
    cost_seed = cfg["seeds"][0]
    cost_run = runs[cost_seed]
    cost_hop = nets[cost_seed]

    assert isinstance(cost_hop, benchmark.MnistNet)

    print(
        f"\nMeasuring computational costs using seed {cost_seed}."
    )

    cost_results = measure_herding_costs(
        cfg,
        cost_run,
        cost_hop,
        selected_beta,
        device,
        train_times=[
            runs[seed].train_time_s
            for seed in cfg["seeds"]
        ],
        prototype_times=prototype_times,
        selected_times=selected_times,
        herding_memory_times=herding_memory_times,
        hopfield_build_times=hopfield_build_times,
        kmeans_memory_times=kmeans_memory_times,
        representative_seed=cost_seed,
    )

    # Stage 7: Prepare prediction artifacts.
    prediction_artifacts: dict[str, np.ndarray] = {}

    for kind, seed_results in raw_results.items():
        for seed, level_results in seed_results.items():
            for level, pair in level_results.items():
                level_key = format(
                    float(level),
                    ".8g",
                ).replace(".", "p")

                prefix = (
                    f"{kind}__seed_{seed}__level_{level_key}"
                )

                for model_name in ("baseline", "hopfield"):
                    model_result = pair[model_name]

                    for prediction_type in ("head", "proto"):
                        prediction_artifacts[
                            f"{prefix}__{model_name}_{prediction_type}_pred"
                        ] = model_result["predictions"][prediction_type]

                        prediction_artifacts[
                            f"{prefix}__{model_name}_{prediction_type}_correct"
                        ] = model_result["correct"][prediction_type]

                prediction_artifacts[f"{prefix}__labels"] = (
                    _tensor_to_numpy(
                        runs[seed].splits.test_labels
                    ).astype(np.int64, copy=False)
                )

    # Stage 8: Return experiment results.
    return {
        "config": cfg,
        "environment": benchmark.environment_info(device),
        "selection": selection,
        "normalize_for_selection": normalize_for_selection,
        "beta_selection_sets": list(validation_sets),
        "selected_beta": selected_beta,
        "validation_mean_curve": validation_mean_curve,
        # Time of the method that supplied the Hopfield memory.
        "selection_time_s": benchmark.mean_std(selected_times),
        # Both methods are timed in every run.
        "herding_memory_time_s": benchmark.mean_std(
            herding_memory_times
        ),
        "kmeans_memory_time_s": benchmark.mean_std(
            kmeans_memory_times
        ),
        "prototype_time_s": benchmark.mean_std(prototype_times),
        "hopfield_build_time_s": benchmark.mean_std(
            hopfield_build_times
        ),
        "memory_timing": {
            "warmups_per_method_per_seed": MEMORY_TIMING_WARMUPS,
            "measured_repeats_per_method_per_seed": (
                MEMORY_TIMING_REPEATS
            ),
            "per_seed_summary": "median",
            "herding_samples_s": herding_timing_samples,
            "kmeans_samples_s": kmeans_timing_samples,
        },
        "cost": cost_results,
        "raw_results": raw_results,
        "summaries": summaries,
        "statistics": statistics,
        "robustness_auc": auc,
        "prediction_artifacts": prediction_artifacts,
        "memory": {
            "per_class": per_class,
            "num_classes": num_classes,
            "total_patterns": per_class * num_classes,
            "centered": center,
            "embedding_dim": int(
                runs[cfg["seeds"][0]].z_train.shape[1]
            ),
            "method": method_name,
            "selection_normalized": normalize_for_selection,
            "selection_space": selection,
            "storage_representation": "raw",
        },
    }


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """
    Convert nested results into JSON-compatible values.

    Per-example arrays are stored in the NPZ artifact. They are omitted
    from JSON to avoid duplicating large prediction and correctness
    arrays.
    """
    excluded_keys = {
        "prediction_artifacts",
        "predictions",
        "correct",
    }

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if key not in excluded_keys
        }

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
    raw: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save experiment results and prediction artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = raw["selection"]

    # JSON: summary and scalar results, not per-example arrays.
    json_payload = _json_safe(raw)

    (output_dir / "herding_comparison.json").write_text(
        json.dumps(
            json_payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # NPZ: per-example predictions, labels, and correctness.
    np.savez_compressed(
        output_dir / "herding_predictions.npz",
        **raw["prediction_artifacts"],
    )

    # CSV.
    rows: list[dict[str, Any]] = []

    for kind, points in raw["summaries"].items():
        for point in points:
            row = {
                "selection": selection,
                "condition": kind,
                "level": point["level"],
                "baseline_head_mean": point["baseline_head_acc"][0],
                "baseline_head_std": point["baseline_head_acc"][1],
                "hopfield_head_mean": point["hopfield_head_acc"][0],
                "hopfield_head_std": point["hopfield_head_acc"][1],
                "head_delta": point["head_delta"],
                "baseline_proto_mean": point["baseline_proto_acc"][0],
                "baseline_proto_std": point["baseline_proto_acc"][1],
                "hopfield_proto_mean": point["hopfield_proto_acc"][0],
                "hopfield_proto_std": point["hopfield_proto_acc"][1],
                "proto_delta": point["proto_delta"],
            }

            for metric in ("head", "proto"):
                stat = next(
                    item
                    for item in raw["statistics"][kind]
                    if (
                        item["level"] == point["level"]
                        and item["metric"] == metric
                    )
                )

                row[f"{metric}_mcnemar_p_max"] = (
                    stat["mcnemar_p_max"]
                )
                row[f"{metric}_wilcoxon_p"] = stat["wilcoxon_p"]
                row[f"{metric}_winner"] = stat["winner"]

            rows.append(row)

    if rows:
        with (
            output_dir / "herding_comparison.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

    # Markdown report.
    cost = raw["cost"]

    method_description, selection_description, herding_label = (
        _describe_selection(selection)
    )

    timing_protocol = raw["memory_timing"]

    lines = [
        "# MNIST Memory Selection Experiment",
        "",
        f"- Memory selection method: **{method_description}**",
        f"- Selection variant: **{selection.upper()}**",
        f"- Selection space: {selection_description}",
        "- Storage representation: original raw embeddings",
        f"- Seeds: {raw['config']['seeds']}",
        (
            f"- Memory: {raw['memory']['total_patterns']} patterns "
            f"({raw['memory']['per_class']} per class)"
        ),
        f"- Embedding dimension: {raw['memory']['embedding_dim']}",
        f"- Centered: {raw['memory']['centered']}",
        f"- Beta selection sets: {raw['beta_selection_sets']}",
        "",
        "## Validation-selected beta",
        "",
        f"- {method_description}: `{raw['selected_beta']:g}`",
        "",
        "## Memory construction time",
        "",
        (
            f"- Selected method ({method_description}): "
            f"{raw['selection_time_s'][0]:.4f} "
            f"± {raw['selection_time_s'][1]:.4f} seconds"
        ),
        (
            f"- {herding_label} (timing comparison): "
            f"{raw['herding_memory_time_s'][0]:.4f} "
            f"± {raw['herding_memory_time_s'][1]:.4f} seconds"
        ),
        (
            f"- K-means (timing comparison): "
            f"{raw['kmeans_memory_time_s'][0]:.4f} "
            f"± {raw['kmeans_memory_time_s'][1]:.4f} seconds"
        ),
        (
            f"- Warmups per method per seed: "
            f"{timing_protocol['warmups_per_method_per_seed']}"
        ),
        (
            f"- Measured repetitions per method per seed: "
            f"{timing_protocol['measured_repeats_per_method_per_seed']}"
        ),
        "- Per-seed timing summary: median",
        "- Measurement order: alternating",
        "",
        (
        "Both memory construction measurements start with "
        "precomputed training embeddings."
        ),
        "",
        "## Timing components",
        "",
        (
            f"- Training embedding extraction and class-prototype "
            f"construction: {raw['prototype_time_s'][0]:.4f} "
            f"± {raw['prototype_time_s'][1]:.4f} seconds"
        ),
        (
            f"- {herding_label} memory construction: "
            f"{raw['herding_memory_time_s'][0]:.4f} "
            f"± {raw['herding_memory_time_s'][1]:.4f} seconds"
        ),
        (
            f"- K-means memory construction: "
            f"{raw['kmeans_memory_time_s'][0]:.4f} "
            f"± {raw['kmeans_memory_time_s'][1]:.4f} seconds"
        ),
        (
            f"- Hopfield network construction: "
            f"{raw['hopfield_build_time_s'][0]:.4f} "
            f"± {raw['hopfield_build_time_s'][1]:.4f} seconds"
        ),
        "",
        (
        "These are separate timing components, not a claim of "
        "complete experiment setup time."
        ),
        "",
        "## Computational cost",
        "",
        f"- Representative seed: {cost['representative_seed']}",
        f"- Device: `{cost['device']}`",
        f"- CPU threads: {cost['num_threads']}",
        "",
        "### E7: Full-model inference cost",
        "",
       (
        "| Model | Latency p50 (ms) | Latency p95 (ms) | "
        "Latency/image (µs) | Throughput (images/s) | "
        "Peak memory estimate (MB) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in ("baseline", "hopfield"):
        result = cost["e7"][name]
        lines.append(
            f"| {name} "
            f"| {result['latency_ms_p50']['median']:.4f} "
            f"| {result['latency_ms_p95']['median']:.4f} "
            f"| {result['latency_per_image_us']['median']:.4f} "
            f"| {result['throughput_img_s']['median']:.4f} "
            f"| {result['peak_mem_mb']['median']:.4f} |"
        )

    lines += [
        "",
        "### Full test-set evaluation time",
        "",
        "| Model | Median (s) | IQR (s) |",
        "|---|---:|---:|",
    ]

    for name in ("baseline", "hopfield"):
        result = cost["eval_time_s"][name]
        lines.append(
            f"| {name} | {result['median']:.4f} "
            f"| {result['iqr']:.4f} |"
        )

    lines += [
        "",
        "### E8: Inference cost across batch sizes",
        "",
       (
        "| Batch size | Model | Latency p50 (ms) | "
        "Latency/image (µs) | Throughput (images/s) |"
    ),
        "|---:|---|---:|---:|---:|",
    ]

    for batch_size, models in cost["e8"].items():
        for name in ("baseline", "hopfield"):
            result = models[name]
            lines.append(
                f"| {batch_size} | {name} "
                f"| {result['latency_ms_p50']['median']:.4f} "
                f"| {result['latency_per_image_us']['median']:.4f} "
                f"| {result['throughput_img_s']['median']:.4f} |"
            )

    lines += [
        "",
        "### Complete-model computational footprint",
        "",
       (
        "| Model | Trainable parameters | Buffer values | "
        "Serialized size (bytes) | Analytical MACs/image | "
        "Hook MACs/image |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in ("baseline", "hopfield"):
        result = cost["static"][name]
        lines.append(
            f"| {name} "
            f"| {result['trainable_params']} "
            f"| {result['buffer_values']} "
            f"| {result['model_size_bytes']} "
            f"| {result['macs_per_image']} "
            f"| {result['hook_macs_per_image']} |"
        )

    bridge = cost["hopfield_bridge"]

    lines += [
        "",
        "### Isolated Hopfield bridge footprint",
        "",
        f"- Prototype shape: `{bridge['prototype_shape']}`",
        f"- Prototype dtype: `{bridge['prototype_dtype']}`",
        (
            f"- Prototype tensor storage: "
            f"{bridge['prototype_storage_bytes']} bytes"
        ),
        (
            f"- Serialized bridge state: "
            f"{bridge['model_size_bytes']} bytes"
        ),
        f"- Bridge buffer values: {bridge['buffer_values']}",
        (
            f"- Floating-point bridge buffer values: "
            f"{bridge['floating_buffer_values']}"
        ),
        (
            f"- Trainable bridge parameters: "
            f"{bridge['trainable_params']}"
        ),
        f"- Bridge MACs per image: {bridge['macs_per_image']}",
        f"- Retrieval beta: `{bridge['beta']:g}`",
        f"- Retrieval updates: {bridge['num_updates']}",
        f"- Centered: {bridge['centered']}",
        "",
    ]

    # Accuracy and statistical results.
    for kind, points in raw["summaries"].items():
        lines += [
            f"## {kind.title()} evaluation",
            "",
            (
                "| Level | CNN head accuracy | "
                "Hopfield head accuracy | Δ head | "
                "CNN retrieval accuracy | "
                "Hopfield retrieval accuracy | Δ retrieval |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for point in points:
            lines.append(
                f"| {point['level']:g} "
                f"| {point['baseline_head_acc'][0]:.4f} "
                f"± {point['baseline_head_acc'][1]:.4f} "
                f"| {point['hopfield_head_acc'][0]:.4f} "
                f"± {point['hopfield_head_acc'][1]:.4f} "
                f"| {point['head_delta']:+.4f} "
                f"| {point['baseline_proto_acc'][0]:.4f} "
                f"± {point['baseline_proto_acc'][1]:.4f} "
                f"| {point['hopfield_proto_acc'][0]:.4f} "
                f"± {point['hopfield_proto_acc'][1]:.4f} "
                f"| {point['proto_delta']:+.4f} |"
            )

        lines += [
            "",
            f"### {kind.title()} statistical comparisons",
            "",
            (
            "| Level | Metric | Δ | McNemar max p | "
            "Wilcoxon p | Winner |"
            ),
            "|---:|---|---:|---:|---:|---|",
        ]

        for stat in raw["statistics"][kind]:
            wilcoxon_text = (
                "N/A"
                if stat["wilcoxon_p"] is None
                else f"{stat['wilcoxon_p']:.6g}"
            )

            lines.append(
                f"| {stat['level']:g} "
                f"| {stat['metric']} "
                f"| {stat['delta']:+.4f} "
                f"| {stat['mcnemar_p_max']:.6g} "
                f"| {wilcoxon_text} "
                f"| {stat['winner']} |"
            )

        lines.append("")

    # Robustness AUC, including paired across-seed statistics.
    lines += [
        "## Robustness AUC",
        "",
       (
        "| Noise type | CNN AUC | Hopfield AUC | Δ | "
        "Seeds favoring Hopfield | Seeds favoring CNN | Wilcoxon p |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for kind, stats in raw["robustness_auc"].items():
        wilcoxon_text = (
            "N/A"
            if stats["wilcoxon_p"] is None
            else f"{stats['wilcoxon_p']:.6g}"
        )

        lines.append(
            f"| {kind} "
            f"| {stats['baseline'][0]:.4f} "
            f"± {stats['baseline'][1]:.4f} "
            f"| {stats['hopfield'][0]:.4f} "
            f"± {stats['hopfield'][1]:.4f} "
            f"| {stats['delta']:+.4f} "
            f"| {stats['n_pos']} "
            f"| {stats['n_neg']} "
            f"| {wilcoxon_text} |"
        )

    lines += [
        "",
        "## Methodology and interpretation notes",
        "",
        "- K-means selects cluster representatives; Herding selects actual training embeddings.",
        "- Selection is performed independently for each class.",
        f"- The selection representation is {selection_description}.",
        "- Selected memories are stored using raw CNN embedding space.",
        "- The global Hopfield centering offset is calculated from raw training embeddings.",
        "- The CNN is shared between baseline and Hopfield evaluation.",
        "- Both evaluations use identical frozen CNN embeddings.",
        "- Both models receive identical noisy test inputs for each seed and condition.",
        "- Noise corruption is generated using the shared seeded corruption procedure.",
        "- The memory budget is fixed across all seeds.",
        "- Beta is selected using validation data.",
        f"- The Hopfield bridge uses the {method_description}-selected memory bank.",
        "- This run compares the selected Hopfield model against the CNN baseline only; direct Herding-vs-K-means tests come from the separate comparison script using the saved predictions.",
        "- McNemar tests compare paired per-example correctness.",
        "- McNemar p-values are calculated per seed; the maximum is used for winner decisions.",
        "- Wilcoxon signed-rank tests operate on per-seed accuracy differences.",
        "- Wilcoxon is calculated only when the shared minimum-seed requirement is met.",
        "- Winner decisions use the shared benchmark's decision function and thresholds.",
        "- Memory construction timings exclude CNN embedding extraction and Hopfield construction.",
        "- Memory timing uses one warmup and three measured repetitions per method and seed.",
        "- The order of measured K-means and Herding runs alternates.",
        "- Per-seed memory construction time is summarized by the median of measured repetitions.",
        "- In a K-means run, the Herding timing uses raw-embedding Herding and is for timing comparison only.",
        "- Prototype timing includes training embedding extraction and class-prototype construction.",
        "- Hopfield construction timing is reported separately.",
        "- E7 and E8 use interleaved model ordering to reduce timing-order bias.",
        "- Full-model latency includes the CNN and its bridge.",
        "- Isolated Hopfield bridge storage and MACs are reported separately.",
        "- Model size is the serialized state_dict size.",
        "- Analytical and hook-based MAC counts are reported separately.",
        "- Full test-set evaluation time is measured over the complete test images.",
        "- CPU peak-memory estimates are not measurements of total process RAM.",
        "- All selection modes use the same memory budget, centering rule, beta candidates, and evaluation protocol.",
        "- The L2 variant changes the selection objective, not the representation stored in Hopfield memory.",
        "- The L2 variant is an iCaRL-style normalized-feature Herding procedure, not a complete reproduction of iCaRL.",
        "",
        "## Saved prediction artifacts",
        "",
        "- `herding_predictions.npz` contains per-example labels, predictions, and correctness arrays.",
        "- Artifact keys identify the noise condition, seed, noise level, model, and prediction type.",
        "- These arrays permit later paired analysis without rerunning the experiment.",
        "",
    ]

    (output_dir / "herding_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = benchmark.build_arg_parser(
        "Evaluate K-means, raw Herding, or L2-normalized Herding "
        "memory selection in the MNIST Hopfield benchmark."
    )

    parser.add_argument(
        "--selection",
        choices=SELECTION_MODES,
        default="raw",
        help=(
            "Memory selection method: "
            "'raw' uses raw-embedding Herding; "
            "'l2' uses L2-normalized Herding; "
            "'kmeans' uses K-means. "
            "Selected memories are stored in raw embedding space. "
            "Default: raw."
        ),
    )

    args = parser.parse_args(argv)
    cfg = benchmark.config_from_args(args)

    selection = args.selection

    print(
        f"Running memory-selection experiment from {args.config}"
        f"{' (quick mode)' if cfg.get('quick') else ''}"
    )
    print(f"Selection variant: {selection}")

    raw = run_experiment(
        cfg,
        selection=selection,
        retrain=args.retrain,
    )

    output_dir = (
        Path(cfg["reports_dir"])
        / f"herding_comparison_{selection}"
    )

    save_results(raw, output_dir)

    print(f"\nSelection variant: {selection}")
    print(f"Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()