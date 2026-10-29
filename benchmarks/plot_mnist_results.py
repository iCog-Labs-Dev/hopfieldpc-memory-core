"""Generate Task 13 result diagrams from JSON report files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline": "#4C78A8",
    "hopfield": "#F58518",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_curve(rows: list[dict], kind: str, levels: list[float], variant: str) -> np.ndarray:
    seeds = sorted({row["seed"] for row in rows})
    curve = np.empty((len(seeds), len(levels)), dtype=np.float64)
    for row_index, seed in enumerate(seeds):
        for level_index, level in enumerate(levels):
            matches = [
                row["head_acc"]
                for row in rows
                if row["seed"] == seed
                and row["variant"] == variant
                and row["kind"] == kind
                and abs(float(row["level"]) - level) < 1e-12
            ]
            if not matches:
                raise ValueError(
                    f"Missing result for seed={seed}, variant={variant}, "
                    f"kind={kind}, level={level}"
                )
            curve[row_index, level_index] = matches[0]
    return curve


def plot_curve(ax, rows: list[dict], kind: str, levels: list[float], title: str, xlabel: str) -> None:
    for variant in ("baseline", "hopfield"):
        values = seed_curve(rows, kind, levels, variant) * 100.0
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        ax.plot(levels, mean, marker="o", linewidth=2, label=variant.title(), color=COLORS[variant])
        ax.fill_between(levels, mean - std, mean + std, alpha=0.16, color=COLORS[variant])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Head accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MNIST Task 13 JSON results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/mnist_prototype_retrieval/results.json"),
        help="Full Task 13 results JSON",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("reports/mnist_prototype_retrieval/baseline_results.json"),
        help="Baseline-only comparison JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/mnist_prototype_retrieval/diagrams"),
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    result = load_json(args.results)
    baseline_result = load_json(args.baseline_results)
    rows = result["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = result["seeds"]
    sigma_base = np.array([seed["E3a_0.5_baseline"] for seed in seeds]) * 100.0
    sigma_hop = np.array([seed["E3a_0.5_hopfield"] for seed in seeds]) * 100.0
    sigma_delta = sigma_hop - sigma_base
    clean_base = np.array([seed["E1_clean_baseline"] for seed in seeds]) * 100.0
    clean_hop = np.array([seed["E1_clean_hopfield"] for seed in seeds]) * 100.0
    clean_delta = clean_hop - clean_base

    baseline_matches = 0
    baseline_rows = baseline_result["rows"]
    for baseline_row in baseline_rows:
        matches = [
            row["head_acc"]
            for row in rows
            if row["seed"] == baseline_row["seed"]
            and row["variant"] == "baseline"
            and row["kind"] == baseline_row["kind"]
            and abs(float(row["level"]) - float(baseline_row["level"])) < 1e-12
        ]
        if matches and abs(matches[0] - baseline_row["head_acc"]) < 1e-12:
            baseline_matches += 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        "MNIST Task 13: Baseline vs Hopfield (mean \u00b1 std over 10 seeds)",
        fontsize=16,
        fontweight="bold",
    )

    plot_curve(
        axes[0, 0], rows, "gaussian",
        [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        "E3a: Gaussian noise", "Gaussian sigma",
    )
    plot_curve(
        axes[0, 1], rows, "salt_pepper",
        [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
        "E3b: Salt-and-pepper noise", "Corruption probability",
    )
    plot_curve(
        axes[0, 2], rows, "occlusion",
        [0, 7, 10, 14],
        "E3c: Occlusion", "Occlusion size (pixels)",
    )

    beta_counts = Counter(seed["beta_star"] for seed in seeds)
    betas = sorted(beta_counts)
    axes[1, 0].bar(
        [str(beta) for beta in betas],
        [beta_counts[beta] for beta in betas],
        color="#7B7B7B",
    )
    axes[1, 0].set_title("Selected beta by seed")
    axes[1, 0].set_xlabel("Beta*")
    axes[1, 0].set_ylabel("Seed count")
    axes[1, 0].set_ylim(0, max(beta_counts.values()) + 1)
    axes[1, 0].grid(axis="y", alpha=0.25)

    seed_ids = np.arange(len(seeds))
    width = 0.38
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].bar(
        seed_ids - width / 2, clean_delta, width,
        label="Clean", color=COLORS["baseline"],
    )
    axes[1, 1].bar(
        seed_ids + width / 2, sigma_delta, width,
        label="Gaussian sigma=0.5", color=COLORS["hopfield"],
    )
    axes[1, 1].set_title("Hopfield minus baseline by seed")
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Accuracy difference (pp)")
    axes[1, 1].set_xticks(seed_ids, seed_ids)
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.25)

    axes[1, 2].axis("off")
    summary = (
        "SUMMARY\n\n"
        f"Clean: {np.mean(clean_base):.2f}% -> {np.mean(clean_hop):.2f}% "
        f"({np.mean(clean_delta):+.2f} pp)\n\n"
        f"Gaussian sigma=0.5:\n"
        f"{np.mean(sigma_base):.2f}% -> {np.mean(sigma_hop):.2f}% "
        f"({np.mean(sigma_delta):+.2f} pp)\n"
        f"Hopfield wins: {int(np.sum(sigma_delta > 0))}/10 seeds\n\n"
        f"Baseline rows matched: {baseline_matches}/{len(baseline_rows)}"
    )
    axes[1, 2].text(
        0.08, 0.52, summary, va="center", fontsize=11,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F2F5F7", "edgecolor": "#D5DBDE"},
    )

    output_png = args.output_dir / "mnist_task13_results.png"
    output_pdf = args.output_dir / "mnist_task13_results.pdf"
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
