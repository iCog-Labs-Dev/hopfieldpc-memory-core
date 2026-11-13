# Experiment 2 Results

## Hybrid Kalman-Momentum Prototype Learning

## 1. Summary

Experiment 2 evaluated dynamic prototype updates for a continuous Hopfield
memory on MNIST. The study used three random seeds and compared four update
rules:

- EMA;
- Hybrid Kalman-momentum;
- Momentum-only; and
- Kalman-only.

The direct CNN achieved the highest clean accuracy. Among the memory methods,
the hybrid method achieved the best validation accuracy, while Kalman-only
achieved the strongest average test robustness on Gaussian and salt-and-pepper
corruptions.

The results support the technical feasibility of the proposed update system,
but the improvements over EMA are small and should be treated as descriptive
rather than statistically conclusive.

## 2. Experimental Configuration

| Setting | Value |
|---|---:|
| Dataset | MNIST |
| Seeds | 0, 1, 2 |
| Training epochs | 5 |
| Warm-up epochs | 1 |
| Training batch size | 128 |
| Evaluation batch size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Embedding dimension | 64 |
| Prototypes per class | 8 |
| Total memory prototypes | 80 |
| EMA coefficient | 0.1 |
| Momentum coefficient | 0.9 |
| Process noise | 0.01 |
| Initial uncertainty | 1.0 |
| Retrieval beta | 5.0 |

Retrieval used centered normalized vectors for similarity scoring and raw
prototype values for the retrieved representation.

## 3. Clean Accuracy

Values are means across seeds 0, 1, and 2.

| Method | Train accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Direct CNN | **99.51%** | **98.79%** | **99.03%** |
| EMA | 99.51% | 98.15% | 98.41% |
| Hybrid Kalman-momentum | 99.51% | **98.27%** | 98.51% |
| Momentum-only | 99.51% | 98.21% | 98.43% |
| Kalman-only | 99.51% | 98.17% | 98.47% |

The hybrid method was 0.12 percentage points above EMA on validation accuracy
in the aggregate comparison and 0.10 percentage points above EMA on clean test
accuracy.

## 4. Robustness Results

The following values are arithmetic means across all tested corruption levels
and all three seeds. They are not normalized AUC values.

| Method | Gaussian | Salt-and-pepper | Occlusion |
|---|---:|---:|---:|
| EMA | 70.91% | 76.93% | 82.29% |
| Hybrid Kalman-momentum | 70.99% | 77.13% | 82.31% |
| Momentum-only | **71.95%** | 77.96% | 82.24% |
| Kalman-only | **71.97%** | **78.33%** | **82.32%** |

The Kalman-only method produced the highest average robustness for Gaussian,
salt-and-pepper, and occlusion corruption. Momentum-only was nearly identical
and had the same Gaussian result to the displayed precision. The hybrid method
improved slightly over EMA for all three corruption families, but did not
outperform both ablations.

## 5. Training Behavior

The direct training accuracy increased consistently across all methods because
the methods share the same CNN architecture and training trajectory within each
seed.

For seed 0:

| Epoch | Direct training accuracy |
|---:|---:|
| 1 | 94.29% |
| 2 | 98.32% |
| 3 | 98.88% |
| 4 | 99.15% |
| 5 | 99.29% |

The memory begins empty during warm-up. It is then populated online until the
capacity of 80 prototypes is reached or no further candidates are admitted.

## 6. Memory Occupancy

The maximum memory size is:

$$
10\text{ classes}\times 8\text{ prototypes}=80.
$$

Observed final occupancy in the pasted run:

| Seed | EMA | Hybrid | Momentum-only | Kalman-only |
|---:|---:|---:|---:|---:|
| 0 | 80 | 80 | 80 | 76 |
| 1 | 80 | 80 | 80 | 80 |
| 2 | 80 | 79 | 80 | 76 |

Memory size is the number of active prototypes, not the allocated capacity.
A bank can finish below 80 when candidates merge into existing prototypes or
are rejected after a class reaches capacity.

## 7. Interpretation

The experiment confirms that the implementation is numerically stable and
operates at the intended memory scale. The memory methods all achieve
approximately 98% clean validation and test accuracy.

The direct CNN remains the strongest clean classifier. The memory methods
slightly reduce clean accuracy but remain competitive. The ablation results
suggest that momentum and Kalman gating each contribute to robustness, but the
current hybrid combination does not produce a clear advantage over the
individual components.

The strongest observed memory result is Kalman-only robustness, not the full
hybrid rule. This does not invalidate the proposed method, but it indicates
that the momentum coefficient, uncertainty dynamics, or gain application may
need further tuning.

## 8. Limitations

- Only three random seeds were used.
- Corruption values above are means over levels, not normalized robustness
  AUC.
- No confidence intervals or statistical significance tests were calculated.
- The direct CNN corruption metrics were not included in the saved summary
  table, so memory-vs-direct corruption conclusions require additional paired
  analysis.
- The current online implementation applies the Kalman-momentum state when an
  accepted prototype merge occurs; it does not yet implement a purely
  epoch-end update for every prototype.

## 9. Conclusion

Experiment 2 successfully trained and evaluated the four prototype-update
methods across three seeds. The hybrid method achieved 98.27% validation and
98.51% clean test accuracy, while Kalman-only achieved the strongest average
corruption robustness.

The proposed approach is promising but not conclusively superior to EMA or its
ablations. Further work should focus on a proposal-faithful epoch-end update,
paired direct-CNN corruption metrics, normalized AUC, and a larger seed count.

## 10. Reproduction

From the repository root:

```powershell
python ".\experiment-id 2\hybrid_kalman_experiment.py"
```

For a faster validation run:

```powershell
python ".\experiment-id 2\hybrid_kalman_experiment.py" --quick --seeds 0
```

Raw results are stored in
[results/hybrid_kalman.json](results/hybrid_kalman.json), and the compact
accuracy table is in
[results/hybrid_kalman_summary.md](results/hybrid_kalman_summary.md).
