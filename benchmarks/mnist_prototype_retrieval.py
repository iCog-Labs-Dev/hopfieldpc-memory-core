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
import gzip
from pathlib import Path
import struct
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

from hopfieldpc.functional import hopfield_retrieve


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()