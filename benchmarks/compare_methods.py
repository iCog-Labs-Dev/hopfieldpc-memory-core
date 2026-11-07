"""Compare K-means, raw Herding and L2 Herding Hopfield memories."""

from __future__ import annotations

import argparse
import csv
import json
from functools import cache
from pathlib import Path

import numpy as np
import torch

from benchmarks import mnist_prototype_retrieval as bm

METHODS = ("kmeans", "raw", "l2")
PAIRS = (("kmeans", "raw"), ("kmeans", "l2"), ("raw", "l2"))
METRICS = ("head", "proto")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="e.g. reports/mnist_prototype_retrieval")
    root = parser.parse_args(argv).root

    npz = {
        m: np.load(root / f"herding_comparison_{m}" / "herding_predictions.npz")
        for m in METHODS
    }
    jsons = {
        m: json.loads(
            (root / f"herding_comparison_{m}" / "herding_comparison.json").read_text()
        )
        for m in METHODS
    }
    cfg = jsons["raw"]["config"]
    seeds = cfg["seeds"]
    grids = {
        "gaussian": cfg["gaussian_sigmas"],
        "salt_pepper": cfg["salt_pepper_probs"],
        "occlusion": cfg["occlusion_sizes"],
    }

    @cache
    def correct(m, kind, seed, level, model, what):
        lk = format(float(level), ".8g").replace(".", "p")
        return npz[m][f"{kind}__seed_{seed}__level_{lk}__{model}_{what}_correct"]

    conds = [("clean", 0.0)] + [
        (k, float(l)) for k in grids for l in grids[k] if float(l) != 0.0
    ]

    # Sanity: the CNN baseline must be identical in every run, for both metrics.
    for kind, level in conds:
        for s in seeds:
            for what in METRICS:
                ref = correct("raw", kind, s, level, "baseline", what)
                for m in METHODS:
                    assert np.array_equal(
                        ref, correct(m, kind, s, level, "baseline", what)
                    ), (
                        f"baseline ({what}) differs: {m}, {kind}, level {level}, "
                        f"seed {s} (CNNs are not shared; check reports_dir)"
                    )

    def acc(m, kind, level, what, model="hopfield"):
        return float(np.mean(
            [correct(m, kind, s, level, model, what).mean() for s in seeds]
        ))

    def paired(a, b, kind, level, what):
        """Hopfield correctness for metric `what`, b minus a."""
        deltas, ps = [], []
        for s in seeds:
            ca = correct(a, kind, s, level, "hopfield", what)
            cb = correct(b, kind, s, level, "hopfield", what)
            t = bm.mcnemar_test(torch.tensor(ca), torch.tensor(cb))
            deltas.append(float(cb.mean() - ca.mean()))
            ps.append(t["p"])
        st = bm.seed_stats(deltas)
        return float(np.mean(deltas)), max(ps), st["n_pos"], st["n_neg"], st["wilcoxon_p"]

    def auc(m, kind, what, model="hopfield"):
        lv = [float(x) for x in grids[kind]]
        return [
            bm.robust_auc(
                lv, [float(correct(m, kind, s, l, model, what).mean()) for l in lv]
            )
            for s in seeds
        ]

    # ---- Classifier accuracy (head), includes the CNN column -------------
    head_rows = []
    for kind, level in conds:
        accs = {m: acc(m, kind, level, "head") for m in METHODS}
        cnn = acc("raw", kind, level, "head", model="baseline")
        for a, b in PAIRS:
            d, p, npos, nneg, wp = paired(a, b, kind, level, "head")
            head_rows.append([kind, level, cnn, accs["kmeans"], accs["raw"],
                              accs["l2"], f"{b} - {a}", d, p, npos, nneg, wp])

    with open(root / "method_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "level", "cnn", "kmeans", "herding_raw", "herding_l2",
                    "pair", "mean_delta", "mcnemar_max_p", "seeds_pos", "seeds_neg",
                    "wilcoxon_p"])
        w.writerows(head_rows)

    # ---- Retrieval accuracy (proto), Hopfield variants only ---------------
    # No CNN column: its proto_acc is nearest of 10 class means, which is not
    # comparable with retrieval over 80 stored patterns.
    proto_rows = []
    for kind, level in conds:
        accs = {m: acc(m, kind, level, "proto") for m in METHODS}
        for a, b in PAIRS:
            d, p, npos, nneg, wp = paired(a, b, kind, level, "proto")
            proto_rows.append([kind, level, accs["kmeans"], accs["raw"], accs["l2"],
                               f"{b} - {a}", d, p, npos, nneg, wp])

    with open(root / "method_retrieval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "level", "kmeans", "herding_raw", "herding_l2",
                    "pair", "mean_delta", "mcnemar_max_p", "seeds_pos", "seeds_neg",
                    "wilcoxon_p"])
        w.writerows(proto_rows)

    # ---- Robustness AUC, both metrics -------------------------------------
    auc_rows = []
    for what in METRICS:
        for kind in grids:
            A = {m: auc(m, kind, what) for m in METHODS}
            if what == "head":
                cnn_auc = auc("raw", kind, what, model="baseline")
                auc_rows.append([what, kind, "cnn", float(np.mean(cnn_auc)),
                                 "", "", "", ""])
            for m in METHODS:
                auc_rows.append([what, kind, m, float(np.mean(A[m])), "", "", "", ""])
            for a, b in PAIRS:
                st = bm.seed_stats([y - x for x, y in zip(A[a], A[b])])
                delta = float(np.mean(A[b]) - np.mean(A[a]))
                auc_rows.append([what, kind, f"{b} - {a}", "", delta,
                                 st["n_pos"], st["n_neg"], st["wilcoxon_p"]])
                print(f"AUC[{what}] {kind}: {b}-{a} = {delta:+.4f}  seeds +/-: "
                      f"{st['n_pos']}/{st['n_neg']}  Wilcoxon p={st['wilcoxon_p']}")

    with open(root / "method_auc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "noise", "method_or_pair", "mean_auc", "delta",
                    "seeds_pos", "seeds_neg", "wilcoxon_p"])
        w.writerows(auc_rows)

    # ---- Memory build time (both methods are timed in the raw run) --------
    timing_rows = [
        ["kmeans", *jsons["raw"]["kmeans_memory_time_s"]],
        ["herding_raw", *jsons["raw"]["herding_memory_time_s"]],
        ["herding_l2", *jsons["l2"]["herding_memory_time_s"]],
    ]
    with open(root / "method_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_seconds_per_seed", "std_seconds"])
        w.writerows(timing_rows)

    # ---- Inference cost of the Hopfield layer, per method ------------------
    cost_rows = []
    for m in METHODS:
        c = jsons[m]["cost"]
        st = c["static"]["hopfield"]
        e7 = c["e7"]["hopfield"]
        base7 = c["e7"]["baseline"]
        cost_rows.append([
            m,
            st["trainable_params"],
            st["buffer_values"],
            st["model_size_bytes"],
            st["macs_per_image"],
            c["hopfield_bridge"]["prototype_storage_bytes"],
            base7["latency_ms_p50"]["median"],
            e7["latency_ms_p50"]["median"],
            e7["throughput_img_s"]["median"],
        ])

    with open(root / "method_cost.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "trainable_params", "buffer_values",
                    "model_size_bytes", "macs_per_image", "memory_bytes",
                    "baseline_latency_p50_ms", "hopfield_latency_p50_ms",
                    "hopfield_throughput_img_s"])
        w.writerows(cost_rows)

    static_cols = {tuple(r[1:6]) for r in cost_rows}
    print("Static cost identical across methods:", len(static_cols) == 1)
    print(f"Saved method_comparison.csv, method_retrieval.csv, method_auc.csv, "
          f"method_timing.csv and method_cost.csv in {root}")


if __name__ == "__main__":
    main()