"""Compare the artifacts saved by ``benchmarks.mnist_herding_comparison``.

Reads ``<reports_dir>/memory_run_<variant>/{herding_comparison.json,
herding_predictions.npz}`` for
baseline, kmeans, raw and l2, and compares them on:

    1. classification accuracy   (head_acc, clean + each noise level)
    2. robustness                (AUC over noise levels)
    3. retrieval performance     (proto_acc + Hopfield retrieval diagnostics)
    4. memory selection time
    5. computational cost

Paired significance tests are computed from the saved per-example predictions
(exact McNemar per seed) and per-seed metrics (exact Wilcoxon across seeds),
using the same winner rule as the benchmark: |delta| < 0.005 or p >= 0.05 is a
tie.

Outputs (in ``--output-dir``): comparison.md, comparison.json,
pairwise_tests.csv.

    python -m benchmarks.compare_methods --reports-dir reports/mnist_herding_comparison
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_PREFIX = "memory_run_"
JSON_NAME = "herding_comparison.json"
NPZ_NAME = "herding_predictions.npz"
METHOD_ORDER = ("baseline", "kmeans", "raw", "l2")
NAMES = {
    "baseline": "CNN baseline",
    "kmeans": "K-means",
    "raw": "Raw Herding",
    "l2": "L2 Herding",
}
WINNER_DELTA = 0.005
ALPHA = 0.05
MIN_SEEDS_FOR_WILCOXON = 6
ACCURACY_METRICS = ("head_acc", "proto_acc")
DIAGNOSTICS = ("entropy_norm", "top1_weight", "true_proto_weight", "dist_to_true_proto")


# ---------------------------------------------------------------------
# Exact tests (same definitions as the benchmark)
# ---------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    term = tail = 1
    for i in range(min(b, c)):
        term = term * (n - i) // (i + 1)
        tail += term
    return min(1.0, 2 * tail / 2**n)


def wilcoxon_signed_rank_p(diffs: list[float]) -> float:
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks2 = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for t in range(i, j + 1):
            ranks2[order[t]] = i + j + 2
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks2, d) if x > 0)
    total = sum(ranks2)
    counts = {0: 1}
    for r in ranks2:
        nxt: dict[int, int] = {}
        for value, cnt in counts.items():
            nxt[value] = nxt.get(value, 0) + cnt
            nxt[value + r] = nxt.get(value + r, 0) + cnt
        counts = nxt
    observed = abs(2 * w_plus - total)
    extreme = sum(c for v, c in counts.items() if abs(2 * v - total) >= observed)
    return min(1.0, extreme / 2**n)


def seed_stats(deltas: list[float]) -> dict[str, Any]:
    eps = 1e-12
    return {
        "n_pos": sum(d > eps for d in deltas),
        "n_neg": sum(d < -eps for d in deltas),
        "wilcoxon_p": (
            wilcoxon_signed_rank_p([0.0 if abs(d) <= eps else d for d in deltas])
            if len(deltas) >= MIN_SEEDS_FOR_WILCOXON
            else None
        ),
    }


def decide_winner(
    delta: float,
    mcnemar_p: float | None,
    signs: list[int],
    wilcoxon_p: float | None,
    ref: str,
    other: str,
) -> str:
    """delta = other - ref. Returns the winning method key, tie, mixed or n/a."""
    better = other if delta > 0 else ref
    if wilcoxon_p is not None:
        if abs(delta) < WINNER_DELTA or wilcoxon_p >= ALPHA:
            return "tie"
        return better
    if mcnemar_p is None:
        return "n/a (<6 seeds)"
    if abs(delta) < WINNER_DELTA or mcnemar_p >= ALPHA:
        return "tie"
    direction = 1 if delta > 0 else -1
    if any(s != direction for s in signs):
        return "mixed"
    return better


# ---------------------------------------------------------------------
# Loading and consistency checks
# ---------------------------------------------------------------------


class Run:
    def __init__(self, method: str, directory: Path) -> None:
        self.method = method
        self.dir = directory
        self.res = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
        if self.res.get("schema_version") != 2:
            raise SystemExit(
                f"{directory} was not written by the current herding_comparison.py "
                "(missing or wrong schema_version). Re-run that variant."
            )
        self.npz = np.load(directory / NPZ_NAME)
        self._cache: dict[str, np.ndarray] = {}
        self.index = {
            (kind, round(float(row["level"]), 9)): row["metrics"]
            for kind, rows in self.res["summaries"].items()
            for row in rows
        }

    @property
    def seeds(self) -> list[int]:
        return list(self.res["seeds"])

    def array(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = self.npz[key]
        return self._cache[key]

    def metric(self, kind: str, level: float, name: str) -> dict[str, Any] | None:
        return self.index.get((kind, round(float(level), 9)), {}).get(name)


def level_key(level: float) -> str:
    return format(float(level), ".8g").replace(".", "p")


def pred_key(kind: str, seed: int, level: float, name: str) -> str:
    return f"{kind}__seed_{seed}__level_{level_key(level)}__{name}"


def load_runs(reports_dir: Path, methods: list[str]) -> dict[str, Run]:
    runs, missing = {}, []
    for method in methods:
        directory = reports_dir / f"{OUTPUT_PREFIX}{method}"
        if (directory / JSON_NAME).exists() and (directory / NPZ_NAME).exists():
            runs[method] = Run(method, directory)
        else:
            missing.append(str(directory))
    if "baseline" not in runs:
        raise SystemExit("The baseline run is required but was not found.\n"
                         + "\n".join(f"  missing: {m}" for m in missing))
    if len(runs) < 2:
        raise SystemExit("Need the baseline plus at least one other method.")
    for m in missing:
        print(f"Warning: skipping missing run {m}", file=sys.stderr)
    return runs


def check_consistency(runs: dict[str, Run]) -> list[str]:
    """Problems that would make the comparison invalid."""
    problems: list[str] = []
    ref = runs["baseline"]
    critical_config_keys = (
        "center_prototypes",
        "prototypes_per_class",
        "beta_values",
        "beta_selection_sets",
        "val_size",
        "train_subset",
        "test_subset",
    )
    for method, run in runs.items():
        if method == "baseline":
            continue
        if run.seeds != ref.seeds:
            problems.append(f"{method}: seeds {run.seeds} differ from baseline {ref.seeds}")
            continue
        for seed in ref.seeds:
            a, b = ref.res["fingerprints"][str(seed)], run.res["fingerprints"][str(seed)]
            if a["cnn"] != b["cnn"]:
                problems.append(f"{method}: CNN weights differ from baseline for seed {seed}")
            if a["test_labels"] != b["test_labels"]:
                problems.append(f"{method}: test labels differ from baseline for seed {seed}")
        if set(run.index) != set(ref.index):
            problems.append(f"{method}: noise grid differs from baseline")
        if run.res.get("selected_beta") is None:
            problems.append(f"{method}: missing selected_beta in results")
        for key in critical_config_keys:
            ref_val = ref.res["config"].get(key)
            run_val = run.res["config"].get(key)
            if run_val != ref_val:
                problems.append(f"{method}: config key '{key}' ({run_val}) differs from baseline ({ref_val})")
    return problems


def conditions(run: Run) -> list[tuple[str, float]]:
    """Clean plus every non-zero noise level (level 0 equals clean)."""
    out: list[tuple[str, float]] = [("clean", 0.0)]
    for kind, rows in run.res["summaries"].items():
        if kind == "clean":
            continue
        out += [(kind, float(r["level"])) for r in rows if float(r["level"]) != 0.0]
    return out


def cond_label(kind: str, level: float) -> str:
    return "clean" if kind == "clean" else f"{kind} {level:g}"


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines + [""]


def ms(stat: dict[str, Any] | None, digits: int = 4) -> str:
    return "—" if stat is None else f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def pair_ms(values: list[float] | None, digits: int = 4) -> str:
    return "—" if not values else f"{values[0]:.{digits}f} ± {values[1]:.{digits}f}"


def bold_best(cells: list[str], values: list[float | None], higher: bool = True) -> list[str]:
    valid = [v for v in values if v is not None]
    if len(valid) < 2 or len(set(valid)) == 1:
        return cells
    best = max(valid) if higher else min(valid)
    return [f"**{c}**" if v is not None and v == best else c for c, v in zip(cells, values)]


def pval(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


# ---------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------


def accuracy_section(runs: dict[str, Run], metric: str, title: str, note: str) -> list[str]:
    methods = list(runs)
    rows = []
    for kind, level in conditions(runs["baseline"]):
        stats = [runs[m].metric(kind, level, metric) for m in methods]
        cells = bold_best([ms(s) for s in stats], [s["mean"] if s else None for s in stats])
        rows.append([cond_label(kind, level)] + cells)
    return [f"## {title}", "", note, ""] + table(
        ["Condition"] + [NAMES[m] for m in methods], rows
    )


def robustness_section(runs: dict[str, Run]) -> list[str]:
    methods = list(runs)
    out = ["## 2. Robustness (AUC over noise levels)", "",
           "Higher is better. AUC is computed per seed from the accuracy at each noise "
           "level, then averaged over seeds.", ""]
    for metric in ACCURACY_METRICS:
        rows = []
        for kind in runs["baseline"].res["robustness_auc"]:
            stats = [runs[m].res["robustness_auc"][kind][metric] for m in methods]
            cells = bold_best([ms(s) for s in stats], [s["mean"] for s in stats])
            rows.append([kind] + cells)
        out += [f"**{metric}**", ""] + table(["Noise type"] + [NAMES[m] for m in methods], rows)
    return out


def retrieval_section(runs: dict[str, Run]) -> list[str]:
    methods = list(runs)
    out = accuracy_section(
        runs, "proto_acc", "3. Retrieval performance (proto_acc)",
        "Baseline: nearest class prototype. Hopfield variants: class with the most "
        "retrieval weight over their stored memory.",
    )
    present = [d for d in DIAGNOSTICS
               if any(runs[m].metric("clean", 0.0, d) for m in methods)]
    if present:
        rows = []
        for d in present:
            stats = [runs[m].metric("clean", 0.0, d) for m in methods]
            rows.append([d] + [ms(s) for s in stats])
        out += ["**Retrieval diagnostics (clean test data)**", ""] + table(
            ["Diagnostic"] + [NAMES[m] for m in methods], rows)
    return out


def selection_time_section(runs: dict[str, Run]) -> tuple[list[str], dict[str, Any]]:
    rows, data = [], {}
    ref_time = None
    if "kmeans" in runs:
        ref_time = runs["kmeans"].res["selection_time"]["mean_std_s"][0]
    for m, run in runs.items():
        sel = run.res["selection_time"]
        if sel is None:
            rows.append([NAMES[m], "n/a (no memory)", "—", "—"])
            continue
        mean = sel["mean_std_s"][0]
        rel = "—" if ref_time in (None, 0) else f"{mean / ref_time:.2f}×"
        rows.append([NAMES[m], pair_ms(sel["mean_std_s"]),
                     pair_ms(run.res["hopfield_build_time_s"]), rel])
        data[m] = mean
    out = ["## 4. Memory selection time", "",
           "Per-seed median of repeated builds from precomputed embeddings (seconds), "
           "mean ± std over seeds. Relative column is versus K-means. Runs are separate "
           "processes, so run them on the same idle machine for a fair timing.", ""]
    out += table(["Method", "Selection time (s)", "Hopfield build (s)", "vs K-means"], rows)
    return out, data


def _cost_std(run: Run, key: str) -> float | None:
    cost = run.res["cost"]
    if key.startswith("e7."):
        entry = cost["e7"].get(key[3:])
        return None if entry is None else float(entry["std"])
    if key == "eval_time":
        return float(cost["eval_time_s"]["std"])
    return None  # static values have no spread


def _cost_value(run: Run, key: str) -> float | None:
    cost = run.res["cost"]
    if key.startswith("e7."):
        entry = cost["e7"].get(key[3:])
        return None if entry is None else float(entry["mean"])
    if key == "eval_time":
        return float(cost["eval_time_s"]["mean"])
    return float(cost["static"][key])


COST_ROWS = [
    ("Trainable params", "trainable_params", "{:,.0f}", False),
    ("Model size (bytes)", "model_size_bytes", "{:,.0f}", False),
    ("MACs / image", "macs_per_image", "{:,.0f}", False),
    ("Latency p50 (ms)", "e7.latency_ms_p50", "{:.4f}", False),
    ("Latency p95 (ms)", "e7.latency_ms_p95", "{:.4f}", False),
    ("Latency / image (µs)", "e7.latency_per_image_us", "{:.3f}", False),
    ("Throughput (img/s)", "e7.throughput_img_s", "{:,.0f}", True),
    ("Peak memory (MB)", "e7.peak_mem_mb", "{:.3f}", False),
    ("Full test-set eval (s)", "eval_time", "{:.4f}", False),
]


def cost_section(runs: dict[str, Run]) -> tuple[list[str], dict[str, Any]]:
    methods = list(runs)
    batch = runs["baseline"].res["cost"]["batch_size"]
    rows, data = [], {}
    for label, key, fmt, higher in COST_ROWS:
        values = [_cost_value(runs[m], key) for m in methods]
        stds = [_cost_std(runs[m], key) for m in methods]
        cells = [
            "—" if v is None
            else fmt.format(v) + ("" if sd is None else " ± " + fmt.format(sd))
            for v, sd in zip(values, stds)
        ]
        rows.append([label] + bold_best(cells, values, higher=higher))
        data[key] = dict(zip(methods, values))
    base = data["e7.latency_ms_p50"].get("baseline")
    overhead = ["Latency overhead vs baseline"]
    for m in methods:
        v = data["e7.latency_ms_p50"][m]
        overhead.append("—" if m == "baseline" or not base or v is None
                        else f"{(v - base) / base * 100:+.1f} %")
    rows.append(overhead)

    out = ["## 5. Computational cost", "",
           f"Measured on every seed at batch size {batch}; timing rows are mean ± std "
           "over seeds (params, size and MACs do not vary by seed). "
           "Bold = best. Cost covers the full model (CNN + L8).", ""]
    out += table(["Metric"] + [NAMES[m] for m in methods], rows)

    sweeps = sorted({int(b) for m in methods for b in runs[m].res["cost"]["e8"]})
    if sweeps:
        rows8 = []
        for bs in sweeps:
            cells = []
            for m in methods:
                entry = runs[m].res["cost"]["e8"].get(str(bs))
                cells.append("—" if entry is None
                             else f"{entry['latency_per_image_us']['mean']:.2f} ± {entry['latency_per_image_us']['std']:.2f}")
            rows8.append([str(bs)] + cells)
        out += ["**Latency per image (µs) across batch sizes**", ""] + table(
            ["Batch size"] + [NAMES[m] for m in methods], rows8)
    return out, data


# ---------------------------------------------------------------------
# Pairwise paired tests
# ---------------------------------------------------------------------


def pairwise_condition(ref: Run, other: Run, kind: str, level: float, metric: str):
    a, b = ref.metric(kind, level, metric), other.metric(kind, level, metric)
    if a is None or b is None:
        return None
    short = metric.replace("_acc", "")
    deltas = [y - x for x, y in zip(a["per_seed"], b["per_seed"])]
    mc, signs = [], []
    for seed in ref.seeds:
        ca = ref.array(pred_key(kind, seed, level, f"{short}_correct")).astype(bool)
        cb = other.array(pred_key(kind, seed, level, f"{short}_correct")).astype(bool)
        if not np.array_equal(
            ref.array(pred_key(kind, seed, level, "labels")),
            other.array(pred_key(kind, seed, level, "labels")),
        ):
            raise SystemExit(f"Test labels differ between {ref.method} and {other.method}.")
        b_cnt = int((ca & ~cb).sum())   # reference right, other wrong
        c_cnt = int((~ca & cb).sum())   # reference wrong, other right
        mc.append(mcnemar_exact_p(b_cnt, c_cnt))
        signs.append((c_cnt > b_cnt) - (c_cnt < b_cnt))
    stats = seed_stats(deltas)
    delta = float(np.mean(b["per_seed"]) - np.mean(a["per_seed"]))
    p_max = max(mc)
    return {
        "ref": ref.method, "other": other.method, "condition": cond_label(kind, level),
        "kind": kind, "level": level, "metric": metric,
        "ref_mean": a["mean"], "other_mean": b["mean"], "delta": delta,
        "mcnemar_p_max": p_max, **stats,
        "winner": decide_winner(delta, p_max, signs, stats["wilcoxon_p"], ref.method, other.method),
    }


def pairwise_auc(ref: Run, other: Run, kind: str, metric: str):
    a = ref.res["robustness_auc"][kind][metric]
    b = other.res["robustness_auc"][kind][metric]
    deltas = [y - x for x, y in zip(a["per_seed"], b["per_seed"])]
    stats = seed_stats(deltas)
    delta = float(np.mean(b["per_seed"]) - np.mean(a["per_seed"]))
    signs = [(d > 1e-12) - (d < -1e-12) for d in deltas]
    return {
        "ref": ref.method, "other": other.method, "condition": f"AUC {kind}",
        "kind": kind, "level": None, "metric": f"auc_{metric}",
        "ref_mean": a["mean"], "other_mean": b["mean"], "delta": delta,
        "mcnemar_p_max": None, **stats,
        "winner": decide_winner(delta, None, signs, stats["wilcoxon_p"], ref.method, other.method),
    }


def run_pairwise(runs: dict[str, Run]) -> list[dict[str, Any]]:
    methods = list(runs)
    conds = conditions(runs["baseline"])
    rows = []
    for ref_m, other_m in itertools.combinations(methods, 2):
        ref, other = runs[ref_m], runs[other_m]
        for kind, level in conds:
            for metric in ACCURACY_METRICS:
                row = pairwise_condition(ref, other, kind, level, metric)
                if row:
                    rows.append(row)
        for kind in ref.res["robustness_auc"]:
            for metric in ACCURACY_METRICS:
                rows.append(pairwise_auc(ref, other, kind, metric))
    return rows


def winner_name(w: str) -> str:
    return NAMES.get(w, w)


def pairwise_section(rows: list[dict[str, Any]]) -> list[str]:
    out = ["## 6. Paired significance tests", "",
           "Delta = second method minus first. McNemar p = maximum over seeds of the exact "
           "per-image test; Wilcoxon p is the exact signed-rank test across seeds "
           f"(only with ≥ {MIN_SEEDS_FOR_WILCOXON} seeds). Winner: tie if |Δ| < "
           f"{WINNER_DELTA} or p ≥ {ALPHA}. Full results for every condition are in "
           "`pairwise_tests.csv`.", ""]
    shown = [r for r in rows if r["condition"] == "clean" or r["condition"].startswith("AUC")]
    for metric in ("head_acc", "auc_head_acc", "proto_acc", "auc_proto_acc"):
        subset = [r for r in shown if r["metric"] == metric]
        if not subset:
            continue
        body = [[
            f"{NAMES[r['ref']]} → {NAMES[r['other']]}", r["condition"],
            f"{r['ref_mean']:.4f}", f"{r['other_mean']:.4f}", f"{r['delta']:+.4f}",
            pval(r["mcnemar_p_max"]), f"{r['n_pos']}/{r['n_neg']}",
            pval(r["wilcoxon_p"]), winner_name(r["winner"]),
        ] for r in subset]
        out += [f"**{metric}**", ""] + table(
            ["Pair", "Condition", "First", "Second", "Δ", "McNemar p", "Seeds +/−",
             "Wilcoxon p", "Winner"], body)
    return out


# ---------------------------------------------------------------------
# Summary / scoreboard
# ---------------------------------------------------------------------


def scoreboard(runs: dict[str, Run], sel: dict[str, float], cost: dict[str, Any]) -> list[list[str]]:
    methods = list(runs)

    def best(values: dict[str, float | None], higher: bool, fmt: str):
        valid = {m: v for m, v in values.items() if v is not None}
        if not valid:
            return "—", "—"
        m = max(valid, key=valid.get) if higher else min(valid, key=valid.get)
        return NAMES[m], fmt.format(valid[m])

    rows = []
    for metric, label in (("head_acc", "Clean classification accuracy"),
                          ("proto_acc", "Clean retrieval accuracy")):
        v = {m: (runs[m].metric("clean", 0.0, metric) or {}).get("mean") for m in methods}
        rows.append([label, *best(v, True, "{:.4f}")])
    for kind in runs["baseline"].res["robustness_auc"]:
        v = {m: runs[m].res["robustness_auc"][kind]["head_acc"]["mean"] for m in methods}
        rows.append([f"Robustness AUC ({kind})", *best(v, True, "{:.4f}")])
    rows.append(["Fastest memory selection", *best(sel, False, "{:.4f} s")])

    # For lowest latency: if baseline is present, it does fewer MACs and typically has lowest latency.
    # If comparing among Hopfield variants (which have identical MACs and parameter footprints),
    # treat differences within timing spread as an architectural tie rather than reporting jitter.
    latencies = {m: cost["e7.latency_ms_p50"].get(m) for m in methods}
    valid_lat = {m: v for m, v in latencies.items() if v is not None}
    if valid_lat:
        min_m = min(valid_lat, key=valid_lat.get)
        min_val = valid_lat[min_m]
        hopfield_methods = [m for m in methods if m != "baseline" and m in valid_lat]
        if min_m == "baseline":
            rows.append(["Lowest latency p50", NAMES["baseline"], f"{min_val:.4f} ms"])
        elif len(hopfield_methods) > 1:
            hop_vals = [valid_lat[m] for m in hopfield_methods]
            max_hop_std = max(
                (float(runs[m].res["cost"]["e7"]["latency_ms_p50"]["std"]) for m in hopfield_methods),
                default=0.0,
            )
            if (max(hop_vals) - min(hop_vals)) <= max(2 * max_hop_std, 2.0):
                rows.append(["Lowest latency p50", "Tie (Hopfield variants identical)", f"{min_val:.4f} ms"])
            else:
                rows.append(["Lowest latency p50", NAMES[min_m], f"{min_val:.4f} ms"])
        else:
            rows.append(["Lowest latency p50", NAMES[min_m], f"{min_val:.4f} ms"])
    else:
        rows.append(["Lowest latency p50", "—", "—"])

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/mnist_herding_comparison"),
                        help="Directory holding the memory_run_<variant> folders. Default: reports/mnist_herding_comparison")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <reports-dir>/memory_comparison")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER,
                        default=list(METHOD_ORDER))
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="Continue even if the consistency checks fail.")
    args = parser.parse_args(argv)

    methods = [m for m in METHOD_ORDER if m in args.methods]
    runs = load_runs(args.reports_dir, methods)
    out_dir = args.output_dir or args.reports_dir / "memory_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = check_consistency(runs)
    if problems and not args.allow_mismatch:
        print("Consistency checks FAILED; the comparison would not be valid:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Re-run the variants with the same CNN/seeds, or pass --allow-mismatch.",
              file=sys.stderr)
        raise SystemExit(1)

    pairwise = run_pairwise(runs)
    sel_lines, sel_data = selection_time_section(runs)
    cost_lines, cost_data = cost_section(runs)
    first = next(iter(runs.values()))

    md = ["# MNIST memory methods: baseline vs K-means vs Herding", "",
          f"- Variants compared: {', '.join(NAMES[m] for m in runs)}",
          f"- Seeds: {first.seeds}",
          "- Consistency: every variant uses the same CNN weights and test labels per seed"
          + (" (CHECKS FAILED, forced with --allow-mismatch)" if problems else " (verified)"),
          "- Accuracy and retrieval numbers are on TEST data; beta was chosen on validation data.",
          "", "## Summary", ""]
    md += table(["Criterion", "Best", "Value"], scoreboard(runs, sel_data, cost_data))
    md += accuracy_section(runs, "head_acc", "1. Classification accuracy (head_acc)",
                           "Mean ± std over seeds; bold = best.")
    md += robustness_section(runs)
    md += retrieval_section(runs)
    md += sel_lines
    md += cost_lines
    md += pairwise_section(pairwise)
    md += ["## Caveats", "",
           "- Separate processes: execute on an idle machine to ensure fair, low-jitter timing comparisons.",
           "- Baseline proto_acc is nearest class mean (10 prototypes), while Hopfield proto_acc is class mass over 80 exemplars.",
           "- Hopfield variants (K-means, Raw Herding, L2 Herding) share the identical network architecture, parameter count, and MACs.",
           "- K-means uses a seeded random initialization; Herding is deterministic.",
           ""]
    (out_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    fields = ["ref", "other", "condition", "metric", "ref_mean", "other_mean", "delta",
              "mcnemar_p_max", "n_pos", "n_neg", "wilcoxon_p", "winner"]
    with (out_dir / "pairwise_tests.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairwise)

    (out_dir / "comparison.json").write_text(json.dumps({
        "methods": list(runs), "seeds": first.seeds, "consistency_problems": problems,
        "selection_time_s": sel_data, "cost": cost_data, "pairwise": pairwise,
    }, indent=2, default=str), encoding="utf-8")

    print(f"Comparison written to {out_dir}")


if __name__ == "__main__":
    main()