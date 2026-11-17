"""Time ONLY the prototype-selection step: K-Means vs K-Medoids (FasterPAM).

Run from the repository root, same command-line options as the benchmark:
    python -m benchmarks.time_selection
Uses the cached backbones (no retraining) and the TRAIN embeddings only.
"""
from __future__ import annotations

import statistics
import time

import torch

from benchmarks.mnist_baseline import (
    build_arg_parser,
    config_from_args,
    configure_threads,
    load_mnist,
    prepare_baseline_run,
    resolve_device,
    tune_baseline,
)
from benchmarks.mnist_prototype_retrieval import kmeans, kmedoids

REPEATS = 3
K = 8


def median_time(fn, x, seed):
    fn(x, K, generator=torch.Generator().manual_seed(seed))  # warm-up, discarded
    ts = []
    for _ in range(REPEATS):
        g = torch.Generator().manual_seed(seed)
        t = time.perf_counter()
        fn(x, K, generator=g)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main() -> None:
    args = build_arg_parser("Selection-time benchmark").parse_args()
    cfg = config_from_args(args)
    configure_threads(cfg["cost"]["num_threads"])
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    tune_baseline(cfg, data, device, retrain=False)  # loads cached lr/batch/epochs into cfg

    km_tot, kmd_tot = [], []
    print(f"threads={torch.get_num_threads()}  k={K}  repeats={REPEATS} (median)")
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=False)
        z = run.z_train.detach().cpu()
        y = run.splits.train_labels.cpu()
        t_km = t_kmd = 0.0
        for c in range(10):
            zc = z[y == c]
            t_km += median_time(kmeans, zc, seed)
            t_kmd += median_time(kmedoids, zc, seed)
        km_tot.append(t_km)
        kmd_tot.append(t_kmd)
        print(f"seed {seed}: kmeans {t_km:7.3f}s  kmedoids {t_kmd:7.3f}s  ratio {t_kmd / t_km:6.1f}x"
              f"  (N_c per class {int(min(torch.bincount(y)))}-{int(max(torch.bincount(y)))})")
    ms = lambda v: f"{statistics.mean(v):.3f} +/- {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    print(f"MEAN over {len(km_tot)} seeds: kmeans {ms(km_tot)} s | kmedoids {ms(kmd_tot)} s | "
          f"ratio {statistics.mean(kmd_tot) / statistics.mean(km_tot):.1f}x")


if __name__ == "__main__":
    main()