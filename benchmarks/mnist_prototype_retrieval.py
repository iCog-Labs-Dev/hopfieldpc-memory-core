"""MNIST Prototype Retrieval Sanity Check.

Secondary sanity check that validates whether HopfieldMemory correctly
retrieves class prototypes when queried with clean or noisy MNIST digit images.

Pipeline
--------
1. Load MNIST train + test sets from the official IDX files.
2. Flatten each 28×28 image to a 784-dimensional vector.
3. Build 10 class prototypes: prototype_i = mean vector of all digit-i samples.
4. Query the Hopfield module with clean and noisy test samples.
5. Report:
   - Prototype retrieval accuracy (clean and noisy)
   - Entropy of the attention weights
   - Top-1 memory weight (retrieval confidence)
   - Noise robustness (accuracy drop across noise levels)

Run
---
    python -m benchmarks.mnist_prototype_retrieval
    python -m benchmarks.mnist_prototype_retrieval --beta 5.0 --noise 0.3 --num-samples 500
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import struct
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import copy
import math
import benchmarks.mnist_baseline as mb

WINNER_DELTA = 0.005
ALPHA = 0.05

from hopfieldpc.functional import hopfield_retrieve
from hopfieldpc import HopfieldMemory
from hopfieldpc.api import validate_beta
from benchmarks.mnist_baseline import (
    IdentityBridge,
    MnistEncoder,
    MnistNet,
    ClassifierHead,
    compute_prototypes,
    corrupt_images,
    make_corrupted_test_set,
    load_mnist,
    noise_grids,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_mnist(
    data_dir: str | Path = "./data",
    *,
    train: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Download (if needed) and load MNIST IDX files as flat JAX tensors.

    Returns
    -------
    images : Tensor of shape [N, 784], values in [0, 1].
    labels : Tensor of shape [N], dtype=long.
    """
    cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefix = "train" if train else "t10k"
    image_name = f"{prefix}-images-idx3-ubyte.gz"
    label_name = f"{prefix}-labels-idx1-ubyte.gz"
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist"

    image_path = cache_dir / image_name
    label_path = cache_dir / label_name
    if not image_path.exists():
        urllib.request.urlretrieve(f"{base_url}/{image_name}", image_path)
    if not label_path.exists():
        urllib.request.urlretrieve(f"{base_url}/{label_name}", label_path)

    with gzip.open(image_path, "rb") as image_file:
        image_header = image_file.read(16)
        magic, num_images, rows, cols = struct.unpack(">IIII", image_header)
        if magic != 2051:
            raise RuntimeError(f"Invalid MNIST image file magic: {magic}")
        image_data = np.frombuffer(image_file.read(), dtype=np.uint8)
    expected_image_size = num_images * rows * cols
    if image_data.size != expected_image_size:
        raise RuntimeError(
            f"Invalid MNIST image data size: got {image_data.size}, expected {expected_image_size}"
        )

    with gzip.open(label_path, "rb") as label_file:
        label_header = label_file.read(8)
        magic, num_labels = struct.unpack(">II", label_header)
        if magic != 2049:
            raise RuntimeError(f"Invalid MNIST label file magic: {magic}")
        label_data = np.frombuffer(label_file.read(), dtype=np.uint8)
    if label_data.size != num_labels or num_labels != num_images:
        raise RuntimeError("MNIST image and label counts do not match")

    images = jnp.asarray(image_data.reshape(num_images, rows * cols), dtype=jnp.float32) / 255.0
    labels = jnp.asarray(label_data.astype(np.int32), dtype=jnp.int32)
    return images, labels


def build_prototypes(
    images: jax.Array,
    labels: jax.Array,
    num_classes: int = 10,
) -> jax.Array:
    """Compute one prototype per class as the mean of all class samples.

    Returns
    -------
    prototypes : Tensor of shape [num_classes, 784].
    """
    dim = images.shape[1]
    prototypes = jnp.zeros((num_classes, dim), dtype=images.dtype)
    for cls in range(num_classes):
        mask = labels == cls
        prototypes = prototypes.at[cls].set(images[mask].mean(axis=0))
    return prototypes


def add_gaussian_noise(
    images: jax.Array,
    noise_level: float,
) -> jax.Array:
    """Add zero-mean Gaussian noise and clamp to [0, 1]."""
    key = jax.random.PRNGKey(42)
    noisy = images + jax.random.normal(key, images.shape, dtype=images.dtype) * noise_level
    return jnp.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Diagnostics (inline — diagnostics.py not yet created per task sequence)
# ---------------------------------------------------------------------------

def compute_entropy(weights: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Shannon entropy per query: -sum(p * log(p)). Shape [batch]."""
    return -(weights * jnp.log(weights + eps)).sum(axis=-1)


def compute_top1_weight(weights: jax.Array) -> jax.Array:
    """Maximum attention weight per query (retrieval confidence). Shape [batch]."""
    return weights.max(axis=-1)


def compute_retrieval_distance(
    retrieved: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """L2 distance between retrieved vectors and target prototypes. Shape [batch]."""
    return jnp.linalg.vector_norm(retrieved - targets, ord=2, axis=-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    queries: jax.Array,
    true_labels: jax.Array,
    prototypes: jax.Array,
    beta: float,
    normalize: bool = False,
) -> dict[str, float]:
    """Run one Hopfield retrieval pass and return averaged metrics.

    Parameters
    ----------
    queries     : [N, 784] — clean or noisy test images.
    true_labels : [N]      — ground-truth digit class (0–9).
    prototypes  : [10, 784] — one prototype per class.
    beta        : retrieval sharpness.
    normalize   : whether to L2-normalise for scoring only.

    Returns
    -------
    Dictionary with keys: top1_acc, distance, entropy, top1_weight.
    """
    retrieved, weights = hopfield_retrieve(
        queries,
        prototypes,
        beta=beta,
        normalize=normalize,
    )

    # Predicted class = index of prototype with highest attention weight
    predicted = weights.argmax(axis=-1)
    top1_acc = (predicted == true_labels).astype(jnp.float32).mean().item()

    target_prototypes = prototypes[true_labels]  # [N, 784]
    distance = compute_retrieval_distance(retrieved, target_prototypes).mean().item()
    entropy = compute_entropy(weights).mean().item()
    top1_weight = compute_top1_weight(weights).mean().item()

    return {
        "top1_acc": top1_acc,
        "distance": distance,
        "entropy": entropy,
        "top1_weight": top1_weight,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}".encode("ascii", errors="replace").decode("ascii"))
    print("=" * width)


def print_clean_vs_noisy_table(
    beta: float,
    clean_metrics: dict[str, float],
    noisy_results: list[dict[str, float]],
    noise_levels: list[float],
) -> None:
    """Print a side-by-side comparison for clean vs noisy inputs."""
    col = 10
    header = (
        f"{'noise':>{col}} | {'top1_acc':>{col}} | {'distance':>{col}} | "
        f"{'entropy':>{col}} | {'top1_wt':>{col}}"
    )
    sep = "-" * len(header)

    print(f"\n  beta = {beta:.2f}")
    print(f"  {header}")
    print(f"  {sep}")

    def row(label: str, m: dict[str, float]) -> str:
        return (
            f"  {label:>{col}} | {m['top1_acc']:>{col}.4f} | "
            f"{m['distance']:>{col}.4f} | {m['entropy']:>{col}.4f} | "
            f"{m['top1_weight']:>{col}.4f}"
        )

    print(row("clean", clean_metrics))
    for noise, m in zip(noise_levels, noisy_results):
        print(row(f"{noise:.2f}", m))


def print_beta_sweep_table(
    beta_values: list[float],
    clean_results: list[dict[str, float]],
    noisy_results: list[dict[str, float]],
    noise_repr: float,
) -> None:
    """Print how beta controls retrieval sharpness (clean vs fixed noise)."""
    col = 10
    header = (
        f"{'beta':>{col}} | {'acc_clean':>{col}} | {'acc_noisy':>{col}} | "
        f"{'entropy':>{col}} | {'top1_wt':>{col}}"
    )
    sep = "-" * len(header)

    print(f"\n  noise (noisy column) = {noise_repr:.2f}")
    print(f"  {header}")
    print(f"  {sep}")

    for beta, c, n in zip(beta_values, clean_results, noisy_results):
        print(
            f"  {beta:>{col}.2f} | {c['top1_acc']:>{col}.4f} | "
            f"{n['top1_acc']:>{col}.4f} | {c['entropy']:>{col}.4f} | "
            f"{c['top1_weight']:>{col}.4f}"
        )


def save_sanity_reports(
    reports_dir: Path,
    *,
    args: argparse.Namespace,
    clean_metrics: dict[str, float],
    noisy_results: list[dict[str, float]],
    noise_levels: list[float],
    beta_values: list[float],
    beta_clean_results: list[dict[str, float]],
    beta_noisy_results: list[dict[str, float]],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    noise_rows = [{"noise": 0.0, **clean_metrics}]
    noise_rows.extend({"noise": noise, **metrics}
                      for noise, metrics in zip(noise_levels, noisy_results))
    beta_rows = [
        {
            "beta": beta,
            **{"clean_" + key: clean[key] for key in clean},
            **{"noisy_" + key: noisy[key] for key in noisy},
        }
        for beta, clean, noisy in zip(beta_values, beta_clean_results, beta_noisy_results)
    ]
    payload = {
        "config": {"beta": args.beta, "noise_levels": noise_levels,
                   "beta_sweep": beta_values, "normalize": args.normalize,
                   "seed": args.seed, "num_samples": args.num_samples},
        "clean": clean_metrics,
        "noise_results": noise_rows,
        "beta_sweep": beta_rows,
    }
    (reports_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(noise_rows[0].keys())
    with open(reports_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(noise_rows)

    summary = [
        "# MNIST Prototype Retrieval Results",
        "",
        f"- Beta: `{args.beta}`",
        f"- Test samples: `{args.num_samples}`",
        f"- Seed: `{args.seed}`",
        f"- Normalize: `{args.normalize}`",
        "",
        "## Clean versus noisy",
        "",
        "| Noise | Top-1 accuracy | Distance | Entropy | Top-1 weight |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in noise_rows:
        summary.append(
            f"| {row['noise']:g} | {row['top1_acc']:.4f} | {row['distance']:.4f} "
            f"| {row['entropy']:.4f} | {row['top1_weight']:.4f} |"
        )
    summary.extend(["", "## Beta sweep", "",
                    "| Beta | Clean accuracy | Noisy accuracy | Clean entropy | Clean top-1 weight |",
                    "|---:|---:|---:|---:|---:|"])
    for beta, clean, noisy in zip(beta_values, beta_clean_results, beta_noisy_results):
        summary.append(
            f"| {beta:g} | {clean['top1_acc']:.4f} | {noisy['top1_acc']:.4f} "
            f"| {clean['entropy']:.4f} | {clean['top1_weight']:.4f} |"
        )
    (reports_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([0.0, *noise_levels], [clean_metrics["top1_acc"], *[m["top1_acc"] for m in noisy_results]],
                     marker="o", label="Top-1 accuracy")
        axes[0].set(xlabel="Noise level", ylabel="Accuracy", title="Clean versus noisy retrieval")
        axes[0].grid(alpha=0.3)
        axes[1].plot(beta_values, [m["top1_acc"] for m in beta_clean_results],
                     marker="o", label="Clean")
        axes[1].plot(beta_values, [m["top1_acc"] for m in beta_noisy_results],
                     marker="o", label="Noisy")
        axes[1].set(xlabel="Beta", ylabel="Top-1 accuracy", title="Beta sweep")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(reports_dir / "plots.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_task13(cfg: dict[str, Any], *, quick: bool) -> dict[str, Any]:
    device = mb.resolve_device(cfg["device"])
    mb.configure_threads(cfg["cost"]["num_threads"])
    data = mb.load_mnist(cfg["data_dir"])
    report_dir = Path(cfg["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed in cfg["seeds"]:
        splits = mb.make_splits(
            data, seed=seed, val_size=cfg["val_size"],
            train_subset=cfg["train_subset"], test_subset=cfg["test_subset"],
        )
        baseline = mb.build_baseline_net(seed)
        mb.train_baseline(
            baseline, splits.train_images, splits.train_labels,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            seed=seed, device=device,
        )
        z_train = mb.encode(
            baseline.encoder, splits.train_images,
            batch_size=cfg["eval_batch_size"], device=device,
        )
        memory, memory_labels, center = build_memory(
            z_train, splits.train_labels,
            per_class=cfg["prototypes_per_class"],
            center=cfg["center_prototypes"], seed=seed,
        )

        best_beta = cfg["beta_values"][0]
        best_score = -float("inf")
        for beta in cfg["beta_values"]:
            hopfield = build_hopfield_net(
                baseline, memory, memory_labels, beta=beta,
                center=center, placement="embedding",
            )
            scores = []
            for kind, level in cfg["beta_selection_sets"]:
                images = splits.val_images if kind == "clean" and level == 0 else mb.make_corrupted_test_set(
                    splits.val_images, seed=seed, kind="gaussian" if kind == "gaussian" else
                    "salt_pepper", level=level,
                )
                outputs = mb.forward_images(
                    hopfield, images, batch_size=cfg["eval_batch_size"], device=device,
                )
                metrics, _, _ = mb.summarize_outputs(
                    outputs, splits.val_labels, memory, memory_labels,
                )
                scores.append(metrics["head_acc"])
            score = float(np.mean(scores))
            if score > best_score + 1e-12:
                best_score = score
                best_beta = beta

        hopfield = build_hopfield_net(
            baseline, memory, memory_labels, beta=best_beta,
            center=center, placement="embedding",
        )
        grids = mb.noise_grids(cfg)
        experiment_kinds = (
            ("E1_clean", "clean", 0.0),
            *[(f"E3a_{level:g}", "gaussian", level) for level in grids["gaussian"]],
            *[(f"E3b_{level:g}", "salt_pepper", level) for level in grids["salt_pepper"]],
            *[(f"E3c_{level:g}", "occlusion", level) for level in grids["occlusion"]],
        )
        seed_row: dict[str, Any] = {"seed": seed, "beta_star": best_beta}
        for experiment, kind, level in experiment_kinds:
            if kind == "clean" or level == 0:
                images = splits.test_images
            else:
                images = mb.make_corrupted_test_set(
                    splits.test_images, seed=seed, kind=kind, level=level,
                )
            baseline_outputs = mb.forward_images(
                baseline, images, batch_size=cfg["eval_batch_size"], device=device,
            )
            hopfield_outputs = mb.forward_images(
                hopfield, images, batch_size=cfg["eval_batch_size"], device=device,
            )
            baseline_metrics, baseline_correct, _ = mb.summarize_outputs(
                baseline_outputs, splits.test_labels, memory, memory_labels,
            )
            hopfield_metrics, hopfield_correct, _ = mb.summarize_outputs(
                hopfield_outputs, splits.test_labels, memory, memory_labels,
            )
            seed_row[f"{experiment}_baseline"] = baseline_metrics["head_acc"]
            seed_row[f"{experiment}_hopfield"] = hopfield_metrics["head_acc"]
            for variant, metrics, correct in (
                ("baseline", baseline_metrics, baseline_correct),
                ("hopfield", hopfield_metrics, hopfield_correct),
            ):
                all_rows.append({
                    "seed": seed, "experiment": experiment, "variant": variant,
                    "kind": kind, "level": level, **metrics,
                })
        seed_summaries.append(seed_row)

    payload = {
        "config": cfg,
        "environment": mb.environment_info(device),
        "beta_selection": "mean validation head_acc over clean, Gaussian 0.4, salt-and-pepper 0.15",
        "seeds": seed_summaries,
        "rows": all_rows,
    }
    (report_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    fieldnames = [
        "seed", "experiment", "variant", "kind", "level",
        "head_acc", "proto_acc", "dist_to_true_proto",
    ]
    with open(report_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    beta_values = ", ".join(f"{x:g}" for x in cfg["beta_values"])
    summary = [
        "# Task 13 MNIST Baseline vs Hopfield",
        "",
        f"- Seeds: `{len(cfg['seeds'])}`",
        f"- Beta values: `{beta_values}`",
        f"- Memory: `{cfg['prototypes_per_class']} prototypes/class`, centered `{cfg['center_prototypes']}`",
        f"- Beta selection: mean validation head accuracy across clean, Gaussian 0.4, and salt-and-pepper 0.15",
        "",
        "## Seed results",
        "",
        "| Seed | Beta* | Clean baseline | Clean Hopfield | Gaussian 0.5 baseline | Gaussian 0.5 Hopfield |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summaries:
        summary.append(
            f"| {row['seed']} | {row['beta_star']:g} | "
            f"{row.get('E1_clean_baseline', 0.0):.4f} | {row.get('E1_clean_hopfield', 0.0):.4f} | "
            f"{row.get('E3a_0.5_baseline', 0.0):.4f} | {row.get('E3a_0.5_hopfield', 0.0):.4f} |"
        )
    (report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        plot_rows: dict[str, dict[str, list[float]]] = {}
        for row in all_rows:
            key = (row["kind"], str(row["level"]))
            plot_rows.setdefault(str(key), {"baseline": [], "hopfield": []})[row["variant"]].append(row["head_acc"])
        _, ax = plt.subplots(figsize=(8, 5))
        for kind in ("gaussian", "salt_pepper", "occlusion"):
            levels = sorted(
                {float(row["level"]) for row in all_rows
                 if row["kind"] == kind and float(row["level"]) > 0},
                key=float,
            )
            if not levels:
                continue
            for variant in ("baseline", "hopfield"):
                values = []
                for level in levels:
                    matched = [
                        row["head_acc"] for row in all_rows
                        if row["kind"] == kind and float(row["level"]) == level and row["variant"] == variant
                    ]
                    values.append(float(np.mean(matched)) if matched else 0.0)
                ax.plot(levels, values, marker="o", label=f"{kind} {variant}")
        ax.set_xlabel("Corruption level")
        ax.set_ylabel("Head accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / "plots.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    print(f"Task 13 reports saved to: {report_dir}")
    return payload


def _run_quick(config_path: Path, reports_dir: Path) -> None:
    cfg = mb.load_config(config_path, quick=True)
    cfg["reports_dir"] = str(reports_dir)
    _run_task13(cfg, quick=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MNIST prototype retrieval sanity check for HopfieldMemory."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Default retrieval sharpness (default: 2.0).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of test samples to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 1.0, 2.0],
        help="Gaussian noise standard deviations to test (default: 0.1 0.3 0.5 1.0 2.0).",
    )
    parser.add_argument(
        "--beta-sweep",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Beta values for the beta-sweep table (default: 0.1 0.5 1.0 2.0 5.0 10.0).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalise query and memory for scoring (default: False).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory to download/cache MNIST (default: ./data).",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=Path, default=mb.DEFAULT_CONFIG_PATH)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--legacy-sanity",
        action="store_true",
        help="Run the lightweight pixel-prototype sanity check instead of the Task 13 experiment.",
    )
    args = parser.parse_args(argv)
    if args.quick:
        report_dir = args.reports_dir or Path("reports/mnist_prototype_retrieval_quick")
        _run_quick(args.config, report_dir)
        return
    if not args.legacy_sanity:
        cfg = mb.load_config(args.config, quick=False)
        if args.reports_dir is not None:
            cfg["reports_dir"] = str(args.reports_dir)
        _run_task13(cfg, quick=False)
        return

    try:
        cfg = mb.load_config(args.config)
    except (FileNotFoundError, ValueError, TypeError):
        cfg = {}
    report_dir = args.reports_dir or Path(cfg.get("reports_dir", "reports/mnist_prototype_retrieval"))
    if args.data_dir == Path("./data") and cfg.get("data_dir"):
        args.data_dir = Path(cfg["data_dir"])

    mnist_key = jax.random.PRNGKey(args.seed)

    # ------------------------------------------------------------------
    # 1. Load data and build prototypes
    # ------------------------------------------------------------------
    print_section("Loading MNIST and building class prototypes")

    print("  Downloading / loading MNIST train split …")
    train_images, train_labels = load_mnist(args.data_dir, train=True)

    print("  Downloading / loading MNIST test split …")
    test_images, test_labels = load_mnist(args.data_dir, train=False)

    print(f"  Train: {train_images.shape}  Test: {test_images.shape}")

    prototypes = build_prototypes(train_images, train_labels)
    print(f"  Prototypes: {prototypes.shape}  (one per digit 0-9)")

    # Sub-sample test set for speed
    num_samples = min(args.num_samples, len(test_images))
    idx = jax.random.permutation(mnist_key, len(test_images))[:num_samples]
    test_images = test_images[idx]
    test_labels = test_labels[idx]
    print(f"  Using {num_samples} test samples  (seed={args.seed})")

    # ------------------------------------------------------------------
    # 2. Clean vs Noisy retrieval
    # ------------------------------------------------------------------
    print_section("Clean vs Noisy Prototype Retrieval")

    clean_metrics = evaluate(
        test_images, test_labels, prototypes,
        beta=args.beta, normalize=args.normalize,
    )

    noisy_results: list[dict[str, float]] = []
    for noise in args.noise_levels:
        noisy_images = add_gaussian_noise(test_images, noise)
        m = evaluate(noisy_images, test_labels, prototypes,
                     beta=args.beta, normalize=args.normalize)
        noisy_results.append(m)

    print_clean_vs_noisy_table(
        args.beta, clean_metrics, noisy_results, args.noise_levels
    )

    print(
        "\n  Interpretation: accuracy should decrease as noise increases.\n"
        "  top1_weight should drop and entropy should rise under high noise."
    )

    # ------------------------------------------------------------------
    # 3. Beta sweep (clean + fixed noise)
    # ------------------------------------------------------------------
    print_section("Beta Sweep - How beta Controls Retrieval Sharpness")

    noise_repr = 0.3  # representative noise level for the noisy column
    noisy_fixed = add_gaussian_noise(test_images, noise_repr)

    beta_clean_results: list[dict[str, float]] = []
    beta_noisy_results: list[dict[str, float]] = []
    for beta in args.beta_sweep:
        beta_clean_results.append(
            evaluate(test_images, test_labels, prototypes,
                     beta=beta, normalize=args.normalize)
        )
        beta_noisy_results.append(
            evaluate(noisy_fixed, test_labels, prototypes,
                     beta=beta, normalize=args.normalize)
        )

    print_beta_sweep_table(
        args.beta_sweep, beta_clean_results, beta_noisy_results, noise_repr
    )

    print(
        "\n  Interpretation: low beta -> broad/uniform retrieval (high entropy).\n"
        "  High beta -> sharp single-prototype retrieval (low entropy, high top1_wt)."
    )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print_section("Summary")
    print(f"  Prototypes  : 10 classes, dim=784")
    print(f"  Test samples: {num_samples}")
    print(f"  Default beta: {args.beta}")
    print(f"  Normalize   : {args.normalize}")
    print()
    print(f"  Clean accuracy   : {clean_metrics['top1_acc']:.4f}")
    print(f"  Clean entropy    : {clean_metrics['entropy']:.4f}")
    print(f"  Clean top-1 wt   : {clean_metrics['top1_weight']:.4f}")
    print(f"  Clean distance   : {clean_metrics['distance']:.4f}")
    acc_drop = clean_metrics["top1_acc"] - noisy_results[-1]["top1_acc"]
    print(
        f"  Accuracy drop at noise={args.noise_levels[-1]:.1f}: "
        f"{acc_drop:.4f}"
    )
    print()

    save_sanity_reports(
        report_dir,
        args=args,
        clean_metrics=clean_metrics,
        noisy_results=noisy_results,
        noise_levels=args.noise_levels,
        beta_values=args.beta_sweep,
        beta_clean_results=beta_clean_results,
        beta_noisy_results=beta_noisy_results,
    )
    print(f"  Reports saved to: {report_dir}")


class HopfieldBridge:
    """Frozen prototype memory bridge used at layer L8."""

    def __init__(self, prototypes: jax.Array, memory_labels: jax.Array, *,
                 beta: float, num_updates: int = 1, placement: str = "embedding",
                 center: jax.Array | None = None) -> None:
        if prototypes.ndim != 2:
            raise ValueError(f"prototypes must be [K, D], got {tuple(prototypes.shape)}.")
        if memory_labels.shape != (prototypes.shape[0],):
            raise ValueError(f"memory_labels must have shape [{prototypes.shape[0]}].")
        if placement not in ("embedding", "pixel"):
            raise ValueError(f"placement must be 'embedding' or 'pixel', got {placement!r}.")
        if placement == "pixel" and center is None:
            raise ValueError("center is required for placement='pixel'.")
        self.prototypes = jnp.asarray(prototypes, dtype=jnp.float32)
        self.memory_labels = jnp.asarray(memory_labels, dtype=jnp.int32)
        self.placement = placement
        self.center = None if center is None else jnp.asarray(center, dtype=jnp.float32)
        if self.center is not None and self.center.shape != (prototypes.shape[1],):
            raise ValueError(f"center must have shape [{prototypes.shape[1]}].")
        self.beta = validate_beta(beta)
        if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
            raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")
        self.num_updates = num_updates
        self.num_classes = int(jnp.max(self.memory_labels)) + 1
        self.memory = HopfieldMemory(
            dim=prototypes.shape[1], beta=self.beta, num_updates=self.num_updates,
            normalize=True, return_diagnostics=True,
        )

    def set_retrieval(self, *, beta: float | None = None, num_updates: int | None = None) -> None:
        if beta is not None:
            self.beta = validate_beta(beta)
            self.memory.beta = self.beta
        if num_updates is not None:
            if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
                raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")
            self.num_updates = num_updates
            self.memory.num_updates = num_updates

    def macs_per_image(self) -> int:
        k, d = self.prototypes.shape
        return self.num_updates * (2 * k * d + d)

    def __call__(self, z: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        z = jnp.asarray(z, dtype=self.prototypes.dtype)
        if self.center is not None:
            output = self.memory(z - self.center, memory=self.prototypes - self.center)
            retrieved = output.retrieved + self.center
            weights = output.weights
            diagnostics = output.diagnostics
        else:
            output = self.memory(z, memory=self.prototypes)
            retrieved, weights, diagnostics = output.retrieved, output.weights, output.diagnostics
        class_mass = jnp.zeros((weights.shape[0], self.num_classes), dtype=weights.dtype)
        for cls in range(self.num_classes):
            mask = self.memory_labels == cls
            class_mass = class_mass.at[:, cls].set(jnp.sum(weights[:, mask], axis=-1))
        aux = {
            **diagnostics,
            "weights": weights,
            "class_mass": class_mass,
            "proto_pred": jnp.argmax(class_mass, axis=-1),
        }
        return retrieved, aux


class PixelHopfieldNet:
    def __init__(self, input_bridge: HopfieldBridge, encoder: MnistEncoder, head: ClassifierHead) -> None:
        self.input_bridge = input_bridge
        self.encoder = encoder
        self.head = head

    def __call__(self, images: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        retrieved, aux = self.input_bridge(images.reshape((images.shape[0], -1)))
        z = self.encoder(retrieved.reshape(images.shape))
        return self.head(z), z, z, aux

    def forward(self, images: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return self(images)


def build_hopfield_net(baseline_net: MnistNet, prototypes: jax.Array,
                       memory_labels: jax.Array, *, beta: float,
                       num_updates: int = 1, placement: str = "embedding",
                       center: jax.Array | None = None) -> MnistNet | PixelHopfieldNet:
    encoder = copy.deepcopy(baseline_net.encoder)
    head = copy.deepcopy(baseline_net.head)
    bridge = HopfieldBridge(
        prototypes, memory_labels, beta=beta, num_updates=num_updates,
        placement=placement, center=center,
    )
    if placement == "pixel":
        return PixelHopfieldNet(bridge, encoder, head)
    return MnistNet(encoder, bridge, head)


def kmeans(x: jax.Array, k: int, *, key: jax.Array | None = None,
           generator: Any | None = None, seed: int | None = None, iters: int = 25) -> jax.Array:
    if key is None:
        key = jax.random.PRNGKey(seed if seed is not None else 0)
    if k >= x.shape[0]:
        raise ValueError(f"k ({k}) must be smaller than the number of points ({x.shape[0]}).")
    key, first_key = jax.random.split(key)
    centroids = x[jax.random.randint(first_key, (), 0, x.shape[0])[None, ...]]
    for _ in range(k - 1):
        distances = jnp.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        d2 = jnp.min(distances, axis=1)
        probs = d2 / jnp.maximum(jnp.sum(d2), 1e-12)
        key, choice_key = jax.random.split(key)
        cdf = np.asarray(jnp.cumsum(probs))
        idx = int(np.searchsorted(cdf, float(np.asarray(jax.random.uniform(choice_key))), side="right"))
        idx = min(idx, x.shape[0] - 1)
        centroids = jnp.concatenate([centroids, x[idx:idx + 1]], axis=0)
    for _ in range(iters):
        distances = jnp.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        assign = jnp.argmin(distances, axis=1)
        updated = []
        for j in range(k):
            members = x[assign == j]
            updated.append(jnp.mean(members, axis=0) if members.shape[0] else centroids[j])
        centroids = jnp.stack(updated)
    return centroids


def build_memory(z_train: jax.Array, labels: jax.Array, *, per_class: int, center: bool,
                 seed: int, num_classes: int = 10) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    z_train = jnp.asarray(z_train)
    labels = jnp.asarray(labels, dtype=jnp.int32)
    center_vec = jnp.mean(z_train, axis=0) if center else None
    memory_parts, label_parts = [], []
    for cls in range(num_classes):
        features = z_train[labels == cls]
        if features.shape[0] == 0:
            raise ValueError(f"Cannot build prototypes: class {cls} has no samples.")
        if per_class == 1:
            centroids = features.mean(axis=0, keepdims=True)
        else:
            centroids = kmeans(features, min(per_class, features.shape[0]),
                               key=jax.random.PRNGKey(seed + cls * 1009))
        memory_parts.append(centroids)
        label_parts.append(jnp.full((centroids.shape[0],), cls, dtype=jnp.int32))
    memory = jnp.concatenate(memory_parts, axis=0)
    memory_labels = jnp.concatenate(label_parts, axis=0)
    return memory, memory_labels, center_vec


def shared_weights_equal(a: Any, b: Any) -> bool:
    if hasattr(a, "params") and hasattr(b, "params"):
        return all(bool(jnp.array_equal(x, y)) for x, y in zip(
            _jax_flat_arrays(a.params()), _jax_flat_arrays(b.params())))
    return True


def _jax_flat_arrays(tree: Any) -> list[jax.Array]:
    if isinstance(tree, dict):
        return sum((_jax_flat_arrays(v) for v in tree.values()), [])
    return [tree] if hasattr(tree, "shape") else []


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar probability for discordant pairs b,c."""
    n = b + c
    if n == 0:
        return 1.0
    threshold = min(b, c)
    if n > 1024:
        if b == c:
            return 1.0
        z = (abs(b - c) - 1.0) / math.sqrt(n)
        return min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))
    prob = sum(math.comb(n, i) / (2.0 ** n)
               for i in range(n + 1) if i <= threshold or i >= n - threshold)
    return min(prob, 1.0)


def mcnemar_test(correct_baseline: jax.Array, correct_hopfield: jax.Array) -> dict[str, float | int]:
    b = int(jnp.sum(correct_baseline.astype(bool) & ~correct_hopfield.astype(bool)))
    c = int(jnp.sum(~correct_baseline.astype(bool) & correct_hopfield.astype(bool)))
    return {"b": b, "c": c, "p": mcnemar_exact_p(b, c)}


def wilcoxon_signed_rank_p(diffs: Sequence[float]) -> float | None:
    values = np.asarray(diffs, dtype=float)
    values = values[values != 0]
    if values.size == 0:
        return 1.0
    n = len(values)
    if n <= 20:
        order = np.argsort(np.abs(values), kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        signs = np.sign(values)
        total = n * (n + 1) / 2
        observed = float(np.sum(ranks[signs > 0]))
        counts = {}
        for mask in range(1 << n):
            subset = sum(float(ranks[i]) for i in range(n) if mask & (1 << i))
            counts[subset] = counts.get(subset, 0) + 1
        distance = abs(observed - total / 2.0)
        tail = sum(count for subset, count in counts.items()
                   if abs(subset - total / 2.0) >= distance)
        return min(1.0, tail / (2 ** n))
    expected = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    positive = float(np.sum((np.arange(1, n + 1))[values > 0]))
    z = (positive - expected) / math.sqrt(variance)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def decide_winner(delta: float, p_value: float = 1.0,
                  seed_signs: Sequence[int] = (),
                  wilcoxon_p: float | None = None) -> str:
    if wilcoxon_p is not None:
        if abs(delta) < WINNER_DELTA or wilcoxon_p >= ALPHA:
            return "tie"
        return "Hopfield" if delta > 0 else "Baseline"
    if abs(delta) < WINNER_DELTA or p_value >= ALPHA:
        return "tie"
    direction = 1 if delta > 0 else -1
    if any(sign != direction for sign in seed_signs):
        return "mixed"
    return "Hopfield" if delta > 0 else "Baseline"


if __name__ == "__main__":
    main()