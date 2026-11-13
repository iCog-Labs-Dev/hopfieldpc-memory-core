"""Analysis and evaluation suite comparing K-Center Greedy vs K-means memory.

Covers memory geometry, retrieval dynamics, outlier analysis, computational scaling,
paired statistics (exact McNemar and Wilcoxon), and generates publication-grade figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.mnist_baseline import (
    ClassifierHead,
    MnistNet,
    load_mnist,
    prepare_baseline_run,
)
from benchmarks.mnist_prototype_retrieval import (
    build_embedding_net,
    build_memory,
    covering_radius,
    forward_embeddings,
    k_center_greedy,
    kmeans,
    robust_auc,
    wilcoxon_signed_rank_p,
)


def extract_beta_star(val: Any) -> float:
    """Extract scalar beta* value from float/int, numeric string, or per-placement dict.

    Raises ValueError if val is missing (None), empty, or cannot be parsed.
    """
    if val is None:
        raise ValueError("beta_star value is missing (None)")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if not val:
            raise ValueError("beta_star dictionary is empty")
        if "embedding" in val and val["embedding"] is not None:
            return float(val["embedding"])
        for v in val.values():
            if v is not None:
                return float(v)
        raise ValueError("beta_star dictionary contains no non-null values")
    try:
        return float(val)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Cannot extract beta* from {val!r}") from err


# ---------------------------------------------------------------------------
# Geometric and Coverage Metric Helpers
# ---------------------------------------------------------------------------


def mean_quantization_error(x: torch.Tensor, centers: torch.Tensor) -> float:
    """Mean Euclidean distance from each row of x to its nearest row in centers."""
    return float(torch.cdist(x.float(), centers.float()).min(dim=1).values.mean().item())


def coverage_fraction(x: torch.Tensor, centers: torch.Tensor, r: float) -> float:
    """Fraction of rows in x with distance <= r to the nearest row in centers."""
    dists = torch.cdist(x.float(), centers.float()).min(dim=1).values
    return float((dists <= r).float().mean().item())


def cell_balance_entropy(x: torch.Tensor, centers: torch.Tensor) -> float:
    """Entropy of Voronoi cell sizes (how many points in x are nearest to each center)."""
    k = centers.shape[0]
    assign = torch.cdist(x.float(), centers.float()).argmin(dim=1)
    counts = torch.bincount(assign, minlength=k).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -float((probs * probs.log()).sum().item())


def pairwise_distances(centers: torch.Tensor) -> tuple[float, float]:
    """Mean and minimum pairwise Euclidean distance among rows of centers."""
    k = centers.shape[0]
    if k <= 1:
        return 0.0, 0.0
    dists = torch.cdist(centers.float(), centers.float())
    mask = ~torch.eye(k, dtype=torch.bool, device=centers.device)
    off_diag = dists[mask]
    return float(off_diag.mean().item()), float(off_diag.min().item())


def inter_class_margin(memory: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean distance from each memory pattern to the nearest pattern of a different class."""
    k = memory.shape[0]
    dists = torch.cdist(memory.float(), memory.float())
    diff_class = labels.unsqueeze(0) != labels.unsqueeze(1)
    dists[~diff_class] = float("inf")
    min_inter = dists.min(dim=1).values
    return float(min_inter[min_inter.isfinite()].mean().item())


# ---------------------------------------------------------------------------
# Random-8 Control (Analysis-Only)
# ---------------------------------------------------------------------------


def build_random_memory(
    z_train: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    center: bool,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Randomly sample real embeddings per class without diversity optimization."""
    generator = torch.Generator().manual_seed(seed)
    blocks, block_labels = [], []
    for c in range(num_classes):
        zc = z_train[labels == c]
        perm = torch.randperm(zc.shape[0], generator=generator)[:per_class]
        blocks.append(zc[perm])
        block_labels.append(torch.full((per_class,), c, dtype=torch.long))
    mean = z_train.mean(dim=0) if center else None
    return torch.cat(blocks), torch.cat(block_labels), mean


# ---------------------------------------------------------------------------
# Retrieval Behavior Diagnostics
# ---------------------------------------------------------------------------


def retrieval_diagnostics(
    net: MnistNet,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 1000,
    device: torch.device,
) -> dict[str, float]:
    """Compute attention entropy, effective memories used, dead memory fraction, and cross-class mass."""
    out = forward_embeddings(net, embeddings, batch_size=batch_size, device=device)
    weights = out["weights"].cpu()  # [N, K]
    class_mass = out["class_mass"].cpu()  # [N, C]
    num_memories = weights.shape[1]

    # Attention entropy per query
    probs = weights.clamp(min=1e-12)
    entropy_per_query = -(probs * probs.log()).sum(dim=-1)
    mean_entropy = float(entropy_per_query.mean().item())
    effective_memories = float(torch.exp(entropy_per_query).mean().item())

    # Dead memories: never the top-1 memory for any query in the dataset
    top1_indices = weights.argmax(dim=-1)
    active_memories = set(top1_indices.unique().tolist())
    dead_fraction = float(1.0 - len(active_memories) / num_memories)

    # Wrong class attention mass: mass assigned to classes != true class
    correct_mass = class_mass.gather(1, labels.unsqueeze(1)).squeeze(1)
    wrong_mass = float((1.0 - correct_mass).clamp(min=0.0).mean().item())

    return {
        "entropy": mean_entropy,
        "effective_memories": effective_memories,
        "dead_fraction": dead_fraction,
        "wrong_class_mass": wrong_mass,
    }


# ---------------------------------------------------------------------------
# Outlier Diagnostics (Hypothesis H6)
# ---------------------------------------------------------------------------


def outlier_analysis(
    head: ClassifierHead,
    z_train: torch.Tensor,
    train_labels: torch.Tensor,
    selected_memories: torch.Tensor,
    memory_labels: torch.Tensor,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Measure how peripheral or misclassified selected memories are relative to class distributions."""
    with torch.no_grad():
        preds = head(selected_memories).argmax(dim=-1)
    misclassified_count = int((preds != memory_labels).sum().item())

    z_scores: list[float] = []
    for c in range(num_classes):
        zc = z_train[train_labels == c]
        mean_c = zc.mean(dim=0)
        std_c = zc.std(dim=0).clamp(min=1e-6)
        mem_c = selected_memories[memory_labels == c]
        for row in mem_c:
            std_dist = torch.sqrt(((row - mean_c) / std_c).pow(2).sum()).item()
            z_scores.append(float(std_dist))

    return {
        "misclassified": misclassified_count,
        "misclassified_rate": misclassified_count / len(selected_memories),
        "mean_z_score": float(np.mean(z_scores)),
        "std_z_score": float(np.std(z_scores)),
        "z_scores": z_scores,
    }


# ---------------------------------------------------------------------------
# Computational Scaling Benchmarks
# ---------------------------------------------------------------------------


def measure_construction_scaling() -> dict[str, Any]:
    """Measure memory construction time vs N (sample size) and vs k (prototype budget)."""
    d = 64
    n_values = [1000, 3000, 10000, 30000]
    k_fixed = 8
    gen = torch.Generator().manual_seed(42)

    times_n_kc, times_n_km = [], []
    for n in n_values:
        x = torch.randn(n, d)
        start = time.perf_counter()
        _ = k_center_greedy(x, k_fixed, generator=gen)
        times_n_kc.append(time.perf_counter() - start)
        start = time.perf_counter()
        _ = kmeans(x, k_fixed, generator=gen, iters=25)
        times_n_km.append(time.perf_counter() - start)

    k_values = [8, 16, 32, 64]
    n_fixed = 10000
    times_k_kc, times_k_km = [], []
    for k in k_values:
        x = torch.randn(n_fixed, d)
        start = time.perf_counter()
        _ = k_center_greedy(x, k, generator=gen)
        times_k_kc.append(time.perf_counter() - start)
        start = time.perf_counter()
        _ = kmeans(x, k, generator=gen, iters=25)
        times_k_km.append(time.perf_counter() - start)

    return {
        "n_values": n_values,
        "times_n_kc": times_n_kc,
        "times_n_km": times_n_km,
        "k_values": k_values,
        "times_k_kc": times_k_kc,
        "times_k_km": times_k_km,
    }


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------


def generate_figures(
    output_dir: Path,
    summary_data: dict[str, Any],
    scaling_data: dict[str, Any],
    sample_mem_km: torch.Tensor,
    sample_mem_kc: torch.Tensor,
    sample_mem_rnd: torch.Tensor,
) -> None:
    """Generate publication-ready figures fig1 through fig7 in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})

    # --- Fig 1: Coverage Tradeoff (covering_radius vs MQE, plus distances) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)
    cov = summary_data["coverage"]["train"]
    methods = ["K-means", "K-Center", "Random-8"]
    colors = ["#0969da", "#cf222e", "#8250df"]

    r_means = [cov["kmeans"]["radius_mean"], cov["kcenter"]["radius_mean"], cov["random"]["radius_mean"]]
    r_stds = [cov["kmeans"]["radius_std"], cov["kcenter"]["radius_std"], cov["random"]["radius_std"]]
    ax1.bar(methods, r_means, yerr=r_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Covering Radius (Worst-Case Distance)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Covering Radius (Euclidean)")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    q_means = [cov["kmeans"]["mqe_mean"], cov["kcenter"]["mqe_mean"], cov["random"]["mqe_mean"]]
    q_stds = [cov["kmeans"]["mqe_std"], cov["kcenter"]["mqe_std"], cov["random"]["mqe_std"]]
    ax2.bar(methods, q_means, yerr=q_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Mean Quantization Error (Average Distance)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Distance to Nearest Memory")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_coverage_tradeoff.png")
    plt.close(fig)

    # --- Fig 2: Noise Robustness (accuracy vs noise level with delta subplots) ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), dpi=150, gridspec_kw={"height_ratios": [2, 1]})
    noise_specs = [
        ("gaussian", r"Gaussian Noise ($\sigma$)", "gaussian_sigmas", axes[0, 0], axes[1, 0]),
        ("salt_pepper", "Salt & Pepper ($p$)", "salt_pepper_probs", axes[0, 1], axes[1, 1]),
        ("occlusion", "Occlusion Patch Size ($s$)", "occlusion_sizes", axes[0, 2], axes[1, 2]),
    ]
    for nt, xlabel, key, ax_top, ax_bot in noise_specs:
        grid = summary_data["noise_grids"].get(key, [])
        if grid:
            base_acc = summary_data["accuracy"][nt]["baseline"]
            km_acc = summary_data["accuracy"][nt]["kmeans"]
            kc_acc = summary_data["accuracy"][nt]["kcenter"]

            ax_top.plot(grid, base_acc, "^--", color="#6e7781", label="Baseline CNN", linewidth=1.2)
            ax_top.plot(grid, km_acc, "o-", color="#0969da", label="Hopfield (K-means)", linewidth=1.5)
            ax_top.plot(grid, kc_acc, "s-", color="#cf222e", label="Hopfield (K-Center)", linewidth=1.5)
            ax_top.set_title(nt.replace("_", " ").title(), fontsize=10, fontweight="bold")
            ax_top.set_ylabel("Head Accuracy")
            ax_top.grid(True, linestyle="--", alpha=0.4)
            if ax_top is axes[0, 0]:
                ax_top.legend(frameon=True, fontsize=8)

            delta_kc_km = [kc - km for kc, km in zip(kc_acc, km_acc)]
            ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax_bot.plot(grid, delta_kc_km, "s-", color="#cf222e", linewidth=1.2)
            ax_bot.set_xlabel(xlabel)
            ax_bot.set_ylabel(r"$\Delta$ (KC $-$ KM)")
            ax_bot.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_noise_robustness.png")
    plt.close(fig)

    # --- Fig 3: Beta Sensitivity (clean and noisy accuracy vs beta, highlighting beta*) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
    betas = summary_data["beta_sweep"]["betas"]
    ax1.plot(betas, summary_data["beta_sweep"]["clean_km"], "o-", color="#0969da", label="K-means", linewidth=1.5)
    ax1.plot(betas, summary_data["beta_sweep"]["clean_kc"], "s-", color="#cf222e", label="K-Center", linewidth=1.5)
    b_km_val = extract_beta_star(summary_data["beta_star"]["kmeans"])
    b_kc_val = extract_beta_star(summary_data["beta_star"]["kcenter"])
    ax1.axvline(b_km_val, color="#0969da", linestyle=":", alpha=0.8, label=rf"KM $\beta^*={b_km_val:g}$")
    ax1.axvline(b_kc_val, color="#cf222e", linestyle=":", alpha=0.8, label=rf"KC $\beta^*={b_kc_val:g}$")
    ax1.set_xscale("log")
    ax1.set_title(r"Clean Head Accuracy vs $\beta$ (E2)", fontsize=10, fontweight="bold")
    ax1.set_xlabel(r"Inverse Temperature $\beta$")
    ax1.set_ylabel("Head Accuracy")
    ax1.legend(frameon=True, fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(betas, summary_data["beta_sweep"]["noisy_km"], "o-", color="#0969da", label="K-means", linewidth=1.5)
    ax2.plot(betas, summary_data["beta_sweep"]["noisy_kc"], "s-", color="#cf222e", label="K-Center", linewidth=1.5)
    ax2.axvline(b_km_val, color="#0969da", linestyle=":", alpha=0.8)
    ax2.axvline(b_kc_val, color="#cf222e", linestyle=":", alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_title(r"Noisy Head Accuracy vs $\beta$ (E5, $\sigma=0.5$)", fontsize=10, fontweight="bold")
    ax2.set_xlabel(r"Inverse Temperature $\beta$")
    ax2.set_ylabel("Head Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_beta_sensitivity.png")
    plt.close(fig)

    # --- Fig 4: Retrieval Dynamics (accuracy vs T and entropy vs noise) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
    ts = summary_data["t_sweep"]["steps"]
    ax1.plot(ts, summary_data["t_sweep"]["km"], "o-", color="#0969da", label="K-means", linewidth=1.5)
    ax1.plot(ts, summary_data["t_sweep"]["kc"], "s-", color="#cf222e", label="K-Center", linewidth=1.5)
    ax1.set_title(r"Accuracy vs Retrieval Steps $T$ (E4, $\sigma=0.5$)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Retrieval Steps T")
    ax1.set_ylabel("Head Accuracy")
    ax1.set_xticks(ts)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    g_grid = summary_data["noise_grids"]["gaussian_sigmas"]
    ax2.plot(g_grid, summary_data["entropy_vs_noise"]["km"], "o-", color="#0969da", label="K-means", linewidth=1.5)
    ax2.plot(g_grid, summary_data["entropy_vs_noise"]["kc"], "s-", color="#cf222e", label="K-Center", linewidth=1.5)
    ax2.set_title(r"Attention Entropy vs Gaussian Noise ($\sigma$)", fontsize=10, fontweight="bold")
    ax2.set_xlabel(r"Gaussian Noise ($\sigma$)")
    ax2.set_ylabel("Normalized Attention Entropy")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_retrieval_dynamics.png")
    plt.close(fig)

    # --- Fig 5: Outlier Diagnostics ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
    z_km = summary_data["outliers"]["kmeans"]["z_scores"]
    z_kc = summary_data["outliers"]["kcenter"]["z_scores"]
    z_rand = summary_data["outliers"]["random"]["z_scores"]
    ax1.hist(z_km, bins=15, alpha=0.5, color="#0969da", label=f"K-means (Mean: {np.mean(z_km):.2f})", edgecolor="black")
    ax1.hist(z_kc, bins=15, alpha=0.6, color="#cf222e", label=f"K-Center (Mean: {np.mean(z_kc):.2f})", edgecolor="black")
    ax1.hist(z_rand, bins=15, alpha=0.4, color="#8250df", label=f"Random-8 (Mean: {np.mean(z_rand):.2f})", edgecolor="black")
    ax1.set_title("Outlier Distance from Class Mean", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Standardized Distance from Class Mean")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Retrieval weight on true class vs noise
    ax2.plot(g_grid, summary_data["true_weight_vs_noise"]["km"], "o-", color="#0969da", label="K-means", linewidth=1.5)
    ax2.plot(g_grid, summary_data["true_weight_vs_noise"]["kc"], "s-", color="#cf222e", label="K-Center", linewidth=1.5)
    ax2.set_title("Attention Weight on Correct Class vs Noise", fontsize=10, fontweight="bold")
    ax2.set_xlabel(r"Gaussian Noise ($\sigma$)")
    ax2.set_ylabel("Weight Mass on True Class")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_outlier_diagnostics.png")
    plt.close(fig)

    # --- Fig 6: Cost Scaling (construction time & throughput vs batch size) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
    ax1.plot(scaling_data["n_values"], scaling_data["times_n_kc"], "s-", color="#cf222e", label="K-Center Greedy", linewidth=1.5)
    ax1.plot(scaling_data["n_values"], scaling_data["times_n_km"], "o-", color="#0969da", label="K-means (25 iters)", linewidth=1.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Construction Time vs N (k=8, D=64)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Number of Data Points N")
    ax1.set_ylabel("Wall Clock Time (s)")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Inference Throughput vs Batch Size (E8)
    e8_batches = summary_data["cost_e8"]["batch_sizes"]
    thr_base = summary_data["cost_e8"]["baseline_thr"]
    thr_km = summary_data["cost_e8"]["kmeans_thr"]
    thr_kc = summary_data["cost_e8"]["kcenter_thr"]
    ax2.plot(e8_batches, thr_base, "^--", color="#6e7781", label="Baseline CNN", linewidth=1.2)
    ax2.plot(e8_batches, thr_km, "o-", color="#0969da", label="Hopfield (K-means)", linewidth=1.5)
    ax2.plot(e8_batches, thr_kc, "s-", color="#cf222e", label="Hopfield (K-Center)", linewidth=1.5)
    ax2.set_xscale("log")
    ax2.set_title("Inference Throughput vs Batch Size (E8)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (images / s)")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_cost_scaling.png")
    plt.close(fig)

    # --- Fig 7: Summary Radar Chart ---
    categories = ["Clean Acc", "Gaussian AUC", "Salt-Pepper AUC", "Occlusion AUC", "Coverage (1/r)", "Build Speed"]
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    # Compute coverage efficiency (1/radius) and build speed (1/time) from real empirical measurements
    r_km = summary_data["coverage"]["train"]["kmeans"]["radius_mean"]
    r_kc = summary_data["coverage"]["train"]["kcenter"]["radius_mean"]
    inv_r_max = max(1.0 / r_km, 1.0 / r_kc) if max(r_km, r_kc) > 0 else 1.0
    cov_km_norm = (1.0 / r_km) / inv_r_max if r_km > 0 else 0.5
    cov_kc_norm = (1.0 / r_kc) / inv_r_max if r_kc > 0 else 0.5

    t_km = scaling_data["times_n_km"][-1]
    t_kc = scaling_data["times_n_kc"][-1]
    inv_t_max = max(1.0 / t_km, 1.0 / t_kc) if max(t_km, t_kc) > 0 else 1.0
    spd_km_norm = (1.0 / t_km) / inv_t_max if t_km > 0 else 0.5
    spd_kc_norm = (1.0 / t_kc) / inv_t_max if t_kc > 0 else 0.5

    # Normalize AUC metrics dynamically against reference maximum
    max_gauss = max(summary_data["gaussian_auc"]["kmeans_mean"], summary_data["gaussian_auc"]["kcenter_mean"], summary_data["gaussian_auc"]["baseline_mean"])
    max_sp = max(summary_data["salt_pepper_auc"]["kmeans_mean"], summary_data["salt_pepper_auc"]["kcenter_mean"], summary_data["salt_pepper_auc"]["baseline_mean"])
    max_occ = max(summary_data["occlusion_auc"]["kmeans_mean"], summary_data["occlusion_auc"]["kcenter_mean"], summary_data["occlusion_auc"]["baseline_mean"])

    v_km = [
        summary_data["clean_head_acc"]["kmeans_mean"],
        summary_data["gaussian_auc"]["kmeans_mean"] / (max_gauss if max_gauss > 0 else 1.0),
        summary_data["salt_pepper_auc"]["kmeans_mean"] / (max_sp if max_sp > 0 else 1.0),
        summary_data["occlusion_auc"]["kmeans_mean"] / (max_occ if max_occ > 0 else 1.0),
        cov_km_norm,
        spd_km_norm,
    ]
    v_kc = [
        summary_data["clean_head_acc"]["kcenter_mean"],
        summary_data["gaussian_auc"]["kcenter_mean"] / (max_gauss if max_gauss > 0 else 1.0),
        summary_data["salt_pepper_auc"]["kcenter_mean"] / (max_sp if max_sp > 0 else 1.0),
        summary_data["occlusion_auc"]["kcenter_mean"] / (max_occ if max_occ > 0 else 1.0),
        cov_kc_norm,
        spd_kc_norm,
    ]
    v_km += v_km[:1]
    v_kc += v_kc[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True), dpi=150)
    ax.plot(angles, v_km, "o-", color="#0969da", linewidth=1.8, label="K-means Centroids")
    ax.fill(angles, v_km, color="#0969da", alpha=0.15)
    ax.plot(angles, v_kc, "s-", color="#cf222e", linewidth=1.8, label="K-Center Greedy")
    ax.fill(angles, v_kc, color="#cf222e", alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_title("Multi-Metric Profile: K-means vs K-Center", y=1.08, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "fig7_summary_radar.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Analysis Execution
# ---------------------------------------------------------------------------


def run_full_analysis(
    kmeans_reports_dir: str,
    kcenter_reports_dir: str,
    output_dir: str = "reports/kcenter",
    data_dir: str = "data",
) -> dict[str, Any]:
    """Execute complete analysis suite comparing K-means and K-Center."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    p_km = Path(kmeans_reports_dir)
    p_kc = Path(kcenter_reports_dir)

    print(f"Loading results from {p_km} and {p_kc}...")
    with open(p_km / "results.json", encoding="utf-8") as f:
        res_km = json.load(f)
    with open(p_kc / "results.json", encoding="utf-8") as f:
        res_kc = json.load(f)

    cfg_km = res_km["config"]
    cfg_kc = res_kc["config"]

    # Priority 1.5 & Priority 2.1-2.4: Validate all relevant experimental configurations
    if cfg_km.get("seeds") != cfg_kc.get("seeds"):
        raise ValueError(f"Mismatched seeds: {cfg_km.get('seeds')} vs {cfg_kc.get('seeds')}")

    required_matching_keys = (
        "beta_values",
        "num_updates_values",
        "beta_selection_sets",
        "prototypes_per_class",
        "center_prototypes",
        "eval_batch_size",
        "noisy_sigma",
        "placements",
        "val_size",
        "train_subset",
        "test_subset",
        "gaussian_sigmas",
        "salt_pepper_probs",
        "occlusion_sizes",
    )
    for key in required_matching_keys:
        if cfg_km.get(key) != cfg_kc.get(key):
            raise ValueError(f"Mismatched configuration for {key!r}: {cfg_km.get(key)!r} vs {cfg_kc.get(key)!r}")

    seeds = cfg_km["seeds"]
    n_seeds = len(seeds)

    # Priority 2.5: Result array lengths, condition matching, and seed ordering
    for exp_key in ("E1", "E3a", "E3b", "E3c", "E2", "E4", "E5"):
        pts_km = res_km.get("results", {}).get(exp_key, [])
        pts_kc = res_kc.get("results", {}).get(exp_key, [])
        if len(pts_km) != len(pts_kc):
            raise ValueError(f"Mismatched condition count for {exp_key}: {len(pts_km)} (KM) vs {len(pts_kc)} (KC)")
        for idx, (pt_km, pt_kc) in enumerate(zip(pts_km, pts_kc)):
            if pt_km.get("value") != pt_kc.get("value"):
                raise ValueError(f"Mismatched condition value at {exp_key}[{idx}]: {pt_km.get('value')} vs {pt_kc.get('value')}")
            for model_key in ("baseline", "hopfield"):
                if len(pt_km.get(model_key, [])) != n_seeds or len(pt_kc.get(model_key, [])) != n_seeds:
                    raise ValueError(f"Incomplete seed records in {exp_key}[{idx}][{model_key}]")

    # Fairness check: CNN baseline evaluations must match across runs
    base_km_e1 = [pt["head_acc"] for pt in res_km["results"]["E1"][0]["baseline"]]
    base_kc_e1 = [pt["head_acc"] for pt in res_kc["results"]["E1"][0]["baseline"]]
    for s_idx, (b1, b2) in enumerate(zip(base_km_e1, base_kc_e1)):
        if abs(b1 - b2) > 1e-4:
            raise ValueError(
                f"Fairness violation: seed {cfg_km['seeds'][s_idx]} baseline head_acc mismatch "
                f"({b1:.4f} vs {b2:.4f}). Runs must share the same frozen backbone weights."
            )

    # Clean Accuracies (E1)
    clean_base = [pt["head_acc"] for pt in res_km["results"]["E1"][0]["baseline"]]
    clean_km = [pt["head_acc"] for pt in res_km["results"]["E1"][0]["hopfield"]]
    clean_kc = [pt["head_acc"] for pt in res_kc["results"]["E1"][0]["hopfield"]]

    clean_proto_km = [pt["proto_acc"] for pt in res_km["results"]["E1"][0]["hopfield"]]
    clean_proto_kc = [pt["proto_acc"] for pt in res_kc["results"]["E1"][0]["hopfield"]]

    # Noise AUC per seed (E3a, E3b, E3c)
    def compute_seed_aucs(res: dict[str, Any], exp_key: str, model: str) -> list[float]:
        pts = res["results"][exp_key]
        levels = [p["value"] for p in pts]
        expected_grid_map = {
            "E3a": cfg_km["gaussian_sigmas"],
            "E3b": cfg_km["salt_pepper_probs"],
            "E3c": cfg_km["occlusion_sizes"],
        }
        if exp_key in expected_grid_map and levels != expected_grid_map[exp_key]:
            raise ValueError(f"Levels in {exp_key} {levels} do not match expected config grid {expected_grid_map[exp_key]}")
        aucs = []
        for s in range(n_seeds):
            accs = [p[model][s]["head_acc"] for p in pts]
            aucs.append(robust_auc(levels, accs))
        return aucs

    auc_gauss_base = compute_seed_aucs(res_km, "E3a", "baseline")
    auc_gauss_km = compute_seed_aucs(res_km, "E3a", "hopfield")
    auc_gauss_kc = compute_seed_aucs(res_kc, "E3a", "hopfield")

    auc_sp_base = compute_seed_aucs(res_km, "E3b", "baseline")
    auc_sp_km = compute_seed_aucs(res_km, "E3b", "hopfield")
    auc_sp_kc = compute_seed_aucs(res_kc, "E3b", "hopfield")

    auc_occ_base = compute_seed_aucs(res_km, "E3c", "baseline")
    auc_occ_km = compute_seed_aucs(res_km, "E3c", "hopfield")
    auc_occ_kc = compute_seed_aucs(res_kc, "E3c", "hopfield")

    # Wilcoxon paired tests across seeds (KC - KM)
    diff_clean = [kc - km for kc, km in zip(clean_kc, clean_km)]
    diff_gauss = [kc - km for kc, km in zip(auc_gauss_kc, auc_gauss_km)]
    diff_sp = [kc - km for kc, km in zip(auc_sp_kc, auc_sp_km)]
    diff_occ = [kc - km for kc, km in zip(auc_occ_kc, auc_occ_km)]

    p_clean = wilcoxon_signed_rank_p(diff_clean)
    p_gauss = wilcoxon_signed_rank_p(diff_gauss)
    p_sp = wilcoxon_signed_rank_p(diff_sp)
    p_occ = wilcoxon_signed_rank_p(diff_occ)

    def winner(mean_diff: float, p_val: float) -> str:
        if abs(mean_diff) < 0.005 or p_val >= 0.05:
            return "tie"
        return "K-Center" if mean_diff > 0 else "K-means"

    paired_stats = {
        "clean_head_acc": {
            "mean_diff": float(np.mean(diff_clean)),
            "wilcoxon_p": p_clean,
            "winner": winner(float(np.mean(diff_clean)), p_clean),
            "primary": True,
        },
        "gaussian_robust_auc": {
            "mean_diff": float(np.mean(diff_gauss)),
            "wilcoxon_p": p_gauss,
            "winner": winner(float(np.mean(diff_gauss)), p_gauss),
            "primary": True,
        },
        "salt_pepper_robust_auc": {
            "mean_diff": float(np.mean(diff_sp)),
            "wilcoxon_p": p_sp,
            "winner": winner(float(np.mean(diff_sp)), p_sp),
            "primary": False,
        },
        "occlusion_robust_auc": {
            "mean_diff": float(np.mean(diff_occ)),
            "wilcoxon_p": p_occ,
            "winner": winner(float(np.mean(diff_occ)), p_occ),
            "primary": False,
        },
    }

    # Noise Curves for Plotting
    noise_grids = {
        "gaussian_sigmas": [p["value"] for p in res_km["results"]["E3a"]],
        "salt_pepper_probs": [p["value"] for p in res_km["results"]["E3b"]],
        "occlusion_sizes": [p["value"] for p in res_km["results"]["E3c"]],
    }

    accuracy_curves: dict[str, dict[str, list[float]]] = {}
    for nt, exp in [("gaussian", "E3a"), ("salt_pepper", "E3b"), ("occlusion", "E3c")]:
        pts_km = res_km["results"][exp]
        pts_kc = res_kc["results"][exp]
        accuracy_curves[nt] = {
            "baseline": [float(np.mean([p["baseline"][s]["head_acc"] for s in range(n_seeds)])) for p in pts_km],
            "kmeans": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in pts_km],
            "kcenter": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in pts_kc],
        }

    # Beta Sweep (E2 / E5)
    betas_e2 = [p["value"] for p in res_km["results"]["E2"]]
    beta_sweep_data = {
        "betas": betas_e2,
        "clean_km": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in res_km["results"]["E2"]],
        "clean_kc": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in res_kc["results"]["E2"]],
        "noisy_km": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in res_km["results"]["E5"]],
        "noisy_kc": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in res_kc["results"]["E5"]],
    }

    # T Sweep (E4)
    ts_e4 = [p["value"] for p in res_km["results"]["E4"]]
    t_sweep_data = {
        "steps": ts_e4,
        "km": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in res_km["results"]["E4"]],
        "kc": [float(np.mean([p["hopfield"][s]["head_acc"] for s in range(n_seeds)])) for p in res_kc["results"]["E4"]],
    }

    # Entropy vs Noise
    entropy_vs_noise = {
        "km": [float(np.mean([p["hopfield"][s]["entropy_norm"] for s in range(n_seeds)])) for p in res_km["results"]["E3a"]],
        "kc": [float(np.mean([p["hopfield"][s]["entropy_norm"] for s in range(n_seeds)])) for p in res_kc["results"]["E3a"]],
    }
    true_weight_vs_noise = {
        "km": [float(np.mean([p["hopfield"][s]["true_proto_weight"] for s in range(n_seeds)])) for p in res_km["results"]["E3a"]],
        "kc": [float(np.mean([p["hopfield"][s]["true_proto_weight"] for s in range(n_seeds)])) for p in res_kc["results"]["E3a"]],
    }

    # Geometric Memory Coverage & Diagnostics
    data = load_mnist(data_dir)
    device = torch.device("cpu")
    cov_train_km_r, cov_train_kc_r, cov_train_rnd_r = [], [], []
    cov_train_km_q, cov_train_kc_q, cov_train_rnd_q = [], [], []
    entropy_km, entropy_kc, entropy_rnd = [], [], []
    cov_frac_km, cov_frac_kc, cov_frac_rnd = [], [], []
    pw_mean_km, pw_mean_kc, pw_mean_rnd = [], [], []
    pw_min_km, pw_min_kc, pw_min_rnd = [], [], []
    margin_km, margin_kc, margin_rnd = [], [], []
    ret_km_list, ret_kc_list = [], []
    outlier_km_list, outlier_kc_list, outlier_rnd_list = [], [], []

    sample_mem_km: torch.Tensor | None = None
    sample_mem_kc: torch.Tensor | None = None
    sample_mem_rnd: torch.Tensor | None = None

    tune_path = p_km / "baseline_tuning.json"
    if tune_path.exists():
        try:
            with open(tune_path, encoding="utf-8") as f:
                tuning_res = json.load(f)
            if "best" in tuning_res:
                cfg_km.update(tuning_res["best"])
                cfg_kc.update(tuning_res["best"])
        except Exception:
            pass

    print("Computing memory geometry and outlier diagnostics across seeds...")
    for s in seeds:
        run = prepare_baseline_run(cfg_km, data, s, device)
        zt = run.z_train
        lt = run.splits.train_labels

        m_km, l_km, _ = build_memory(zt, lt, per_class=8, center=True, seed=s, method="kmeans")
        m_kc, l_kc, _ = build_memory(zt, lt, per_class=8, center=True, seed=s, method="kcenter")
        m_rnd, l_rnd, _ = build_random_memory(zt, lt, per_class=8, center=True, seed=s)

        if sample_mem_km is None:
            sample_mem_km = m_km
            sample_mem_kc = m_kc
            sample_mem_rnd = m_rnd

        margin_km.append(inter_class_margin(m_km, l_km))
        margin_kc.append(inter_class_margin(m_kc, l_kc))
        margin_rnd.append(inter_class_margin(m_rnd, l_rnd))

        for c in range(10):
            zc = zt[lt == c]
            r_km = covering_radius(zc, m_km[l_km == c])
            r_kc = covering_radius(zc, m_kc[l_kc == c])
            r_rnd = covering_radius(zc, m_rnd[l_rnd == c])
            cov_train_km_r.append(r_km)
            cov_train_kc_r.append(r_kc)
            cov_train_rnd_r.append(r_rnd)

            cov_train_km_q.append(mean_quantization_error(zc, m_km[l_km == c]))
            cov_train_kc_q.append(mean_quantization_error(zc, m_kc[l_kc == c]))
            cov_train_rnd_q.append(mean_quantization_error(zc, m_rnd[l_rnd == c]))

            entropy_km.append(cell_balance_entropy(zc, m_km[l_km == c]))
            entropy_kc.append(cell_balance_entropy(zc, m_kc[l_kc == c]))
            entropy_rnd.append(cell_balance_entropy(zc, m_rnd[l_rnd == c]))

            cov_frac_km.append(coverage_fraction(zc, m_km[l_km == c], r=r_kc))
            cov_frac_kc.append(coverage_fraction(zc, m_kc[l_kc == c], r=r_kc))
            cov_frac_rnd.append(coverage_fraction(zc, m_rnd[l_rnd == c], r=r_kc))

            pm_km, pmin_km = pairwise_distances(m_km[l_km == c])
            pm_kc, pmin_kc = pairwise_distances(m_kc[l_kc == c])
            pm_rnd, pmin_rnd = pairwise_distances(m_rnd[l_rnd == c])
            pw_mean_km.append(pm_km)
            pw_mean_kc.append(pm_kc)
            pw_mean_rnd.append(pm_rnd)
            pw_min_km.append(pmin_km)
            pw_min_kc.append(pmin_kc)
            pw_min_rnd.append(pmin_rnd)

        bstar_km = extract_beta_star(res_km["beta_star"])
        bstar_kc = extract_beta_star(res_kc["beta_star"])
        net_km = build_embedding_net(run, 8, True, beta=bstar_km)
        net_kc = build_embedding_net(run, 8, True, beta=bstar_kc, method="kcenter")

        ret_km_list.append(retrieval_diagnostics(net_km, run.test_embeddings("clean", 0.0), run.splits.test_labels, device=device))
        ret_kc_list.append(retrieval_diagnostics(net_kc, run.test_embeddings("clean", 0.0), run.splits.test_labels, device=device))

        outlier_km_list.append(outlier_analysis(run.net.head, zt, lt, m_km, l_km))
        outlier_kc_list.append(outlier_analysis(run.net.head, zt, lt, m_kc, l_kc))
        outlier_rnd_list.append(outlier_analysis(run.net.head, zt, lt, m_rnd, l_rnd))

    coverage_summary = {
        "train": {
            "kmeans": {"radius_mean": float(np.mean(cov_train_km_r)), "radius_std": float(np.std(cov_train_km_r)),
                       "mqe_mean": float(np.mean(cov_train_km_q)), "mqe_std": float(np.std(cov_train_km_q)),
                       "cell_entropy_mean": float(np.mean(entropy_km)), "coverage_frac_mean": float(np.mean(cov_frac_km)),
                       "pairwise_mean": float(np.mean(pw_mean_km)), "margin_mean": float(np.mean(margin_km))},
            "kcenter": {"radius_mean": float(np.mean(cov_train_kc_r)), "radius_std": float(np.std(cov_train_kc_r)),
                        "mqe_mean": float(np.mean(cov_train_kc_q)), "mqe_std": float(np.std(cov_train_kc_q)),
                        "cell_entropy_mean": float(np.mean(entropy_kc)), "coverage_frac_mean": float(np.mean(cov_frac_kc)),
                        "pairwise_mean": float(np.mean(pw_mean_kc)), "margin_mean": float(np.mean(margin_kc))},
            "random": {"radius_mean": float(np.mean(cov_train_rnd_r)), "radius_std": float(np.std(cov_train_rnd_r)),
                       "mqe_mean": float(np.mean(cov_train_rnd_q)), "mqe_std": float(np.std(cov_train_rnd_q)),
                       "cell_entropy_mean": float(np.mean(entropy_rnd)), "coverage_frac_mean": float(np.mean(cov_frac_rnd)),
                       "pairwise_mean": float(np.mean(pw_mean_rnd)), "margin_mean": float(np.mean(margin_rnd))},
        }
    }

    retrieval_summary = {
        "kmeans": {
            "entropy": float(np.mean([r["entropy"] for r in ret_km_list])),
            "effective_memories": float(np.mean([r["effective_memories"] for r in ret_km_list])),
            "dead_fraction": float(np.mean([r["dead_fraction"] for r in ret_km_list])),
            "wrong_class_mass": float(np.mean([r["wrong_class_mass"] for r in ret_km_list])),
        },
        "kcenter": {
            "entropy": float(np.mean([r["entropy"] for r in ret_kc_list])),
            "effective_memories": float(np.mean([r["effective_memories"] for r in ret_kc_list])),
            "dead_fraction": float(np.mean([r["dead_fraction"] for r in ret_kc_list])),
            "wrong_class_mass": float(np.mean([r["wrong_class_mass"] for r in ret_kc_list])),
        },
    }

    outlier_summary = {
        "kmeans": {
            "misclassified_mean": float(np.mean([o["misclassified"] for o in outlier_km_list])),
            "misclassified_rate": float(np.mean([o["misclassified_rate"] for o in outlier_km_list])),
            "mean_z_score": float(np.mean([o["mean_z_score"] for o in outlier_km_list])),
            "z_scores": outlier_km_list[0]["z_scores"],
        },
        "kcenter": {
            "misclassified_mean": float(np.mean([o["misclassified"] for o in outlier_kc_list])),
            "misclassified_rate": float(np.mean([o["misclassified_rate"] for o in outlier_kc_list])),
            "mean_z_score": float(np.mean([o["mean_z_score"] for o in outlier_kc_list])),
            "z_scores": outlier_kc_list[0]["z_scores"],
        },
        "random": {
            "misclassified_mean": float(np.mean([o["misclassified"] for o in outlier_rnd_list])),
            "misclassified_rate": float(np.mean([o["misclassified_rate"] for o in outlier_rnd_list])),
            "mean_z_score": float(np.mean([o["mean_z_score"] for o in outlier_rnd_list])),
            "z_scores": outlier_rnd_list[0]["z_scores"],
        },
    }

    # E8 Cost Data (handles both full runs and --quick runs)
    e8_km = res_km.get("cost", {}).get("e8", {})
    e8_kc = res_kc.get("cost", {}).get("e8", {})
    common_batches_str = sorted(
        [b for b in e8_km.keys() if b in e8_kc],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    cost_e8 = {
        "batch_sizes": [int(b) for b in common_batches_str],
        "baseline_thr": [e8_km[b]["baseline"]["throughput_img_s"]["median"] for b in common_batches_str],
        "kmeans_thr": [e8_km[b]["hopfield"]["throughput_img_s"]["median"] for b in common_batches_str],
        "kcenter_thr": [e8_kc[b]["hopfield"]["throughput_img_s"]["median"] for b in common_batches_str],
    }

    print("Benchmarking construction scaling...")
    scaling_data = measure_construction_scaling()

    summary_data = {
        "clean_head_acc": {
            "baseline_mean": float(np.mean(clean_base)), "baseline_std": float(np.std(clean_base)),
            "kmeans_mean": float(np.mean(clean_km)), "kmeans_std": float(np.std(clean_km)),
            "kcenter_mean": float(np.mean(clean_kc)), "kcenter_std": float(np.std(clean_kc)),
        },
        "gaussian_auc": {
            "baseline_mean": float(np.mean(auc_gauss_base)), "baseline_std": float(np.std(auc_gauss_base)),
            "kmeans_mean": float(np.mean(auc_gauss_km)), "kmeans_std": float(np.std(auc_gauss_km)),
            "kcenter_mean": float(np.mean(auc_gauss_kc)), "kcenter_std": float(np.std(auc_gauss_kc)),
        },
        "salt_pepper_auc": {
            "baseline_mean": float(np.mean(auc_sp_base)), "baseline_std": float(np.std(auc_sp_base)),
            "kmeans_mean": float(np.mean(auc_sp_km)), "kmeans_std": float(np.std(auc_sp_km)),
            "kcenter_mean": float(np.mean(auc_sp_kc)), "kcenter_std": float(np.std(auc_sp_kc)),
        },
        "occlusion_auc": {
            "baseline_mean": float(np.mean(auc_occ_base)), "baseline_std": float(np.std(auc_occ_base)),
            "kmeans_mean": float(np.mean(auc_occ_km)), "kmeans_std": float(np.std(auc_occ_km)),
            "kcenter_mean": float(np.mean(auc_occ_kc)), "kcenter_std": float(np.std(auc_occ_kc)),
        },
        "paired_statistics": paired_stats,
        "coverage": coverage_summary,
        "retrieval": retrieval_summary,
        "outliers": outlier_summary,
        "noise_grids": noise_grids,
        "accuracy": accuracy_curves,
        "beta_sweep": beta_sweep_data,
        "t_sweep": t_sweep_data,
        "entropy_vs_noise": entropy_vs_noise,
        "true_weight_vs_noise": true_weight_vs_noise,
        "cost_e8": cost_e8,
        "beta_star": {"kmeans": extract_beta_star(res_km["beta_star"]), "kcenter": extract_beta_star(res_kc["beta_star"])},
    }

    with open(out_path / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(out_path / "analysis_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "K-means", "K-Center", "Delta(KC-KM)", "Wilcoxon_p", "Winner"])
        writer.writerow([
            "Clean Head Acc",
            f"{summary_data['clean_head_acc']['baseline_mean']:.4f} ± {summary_data['clean_head_acc']['baseline_std']:.4f}",
            f"{summary_data['clean_head_acc']['kmeans_mean']:.4f} ± {summary_data['clean_head_acc']['kmeans_std']:.4f}",
            f"{summary_data['clean_head_acc']['kcenter_mean']:.4f} ± {summary_data['clean_head_acc']['kcenter_std']:.4f}",
            f"{paired_stats['clean_head_acc']['mean_diff']:+.4f}",
            f"{paired_stats['clean_head_acc']['wilcoxon_p']:.4f}",
            paired_stats["clean_head_acc"]["winner"],
        ])
        writer.writerow([
            "Gaussian AUC",
            f"{summary_data['gaussian_auc']['baseline_mean']:.4f} ± {summary_data['gaussian_auc']['baseline_std']:.4f}",
            f"{summary_data['gaussian_auc']['kmeans_mean']:.4f} ± {summary_data['gaussian_auc']['kmeans_std']:.4f}",
            f"{summary_data['gaussian_auc']['kcenter_mean']:.4f} ± {summary_data['gaussian_auc']['kcenter_std']:.4f}",
            f"{paired_stats['gaussian_robust_auc']['mean_diff']:+.4f}",
            f"{paired_stats['gaussian_robust_auc']['wilcoxon_p']:.4f}",
            paired_stats["gaussian_robust_auc"]["winner"],
        ])
        writer.writerow([
            "Salt & Pepper AUC",
            f"{summary_data['salt_pepper_auc']['baseline_mean']:.4f} ± {summary_data['salt_pepper_auc']['baseline_std']:.4f}",
            f"{summary_data['salt_pepper_auc']['kmeans_mean']:.4f} ± {summary_data['salt_pepper_auc']['kmeans_std']:.4f}",
            f"{summary_data['salt_pepper_auc']['kcenter_mean']:.4f} ± {summary_data['salt_pepper_auc']['kcenter_std']:.4f}",
            f"{paired_stats['salt_pepper_robust_auc']['mean_diff']:+.4f}",
            f"{paired_stats['salt_pepper_robust_auc']['wilcoxon_p']:.4f}",
            paired_stats["salt_pepper_robust_auc"]["winner"],
        ])
        writer.writerow([
            "Occlusion AUC",
            f"{summary_data['occlusion_auc']['baseline_mean']:.4f} ± {summary_data['occlusion_auc']['baseline_std']:.4f}",
            f"{summary_data['occlusion_auc']['kmeans_mean']:.4f} ± {summary_data['occlusion_auc']['kmeans_std']:.4f}",
            f"{summary_data['occlusion_auc']['kcenter_mean']:.4f} ± {summary_data['occlusion_auc']['kcenter_std']:.4f}",
            f"{paired_stats['occlusion_robust_auc']['mean_diff']:+.4f}",
            f"{paired_stats['occlusion_robust_auc']['wilcoxon_p']:.4f}",
            paired_stats["occlusion_robust_auc"]["winner"],
        ])

    print("Generating figures fig1 through fig7...")
    assert sample_mem_km is not None and sample_mem_kc is not None and sample_mem_rnd is not None
    generate_figures(
        out_path,
        summary_data,
        scaling_data,
        sample_mem_km,
        sample_mem_kc,
        sample_mem_rnd,
    )
    print(f"Analysis complete. Outputs saved to {out_path}.")
    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Center vs K-means Analysis Suite")
    parser.add_argument("--kmeans-dir", default="reports/mnist_prototype_retrieval", help="Path to K-means reports")
    parser.add_argument("--kcenter-dir", default="reports/mnist_kcenter", help="Path to K-Center reports")
    parser.add_argument("--output-dir", default="reports/kcenter", help="Path to save analysis outputs")
    parser.add_argument("--data-dir", default="data", help="Path to MNIST data")
    args = parser.parse_args()

    run_full_analysis(
        kmeans_reports_dir=args.kmeans_dir,
        kcenter_reports_dir=args.kcenter_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
