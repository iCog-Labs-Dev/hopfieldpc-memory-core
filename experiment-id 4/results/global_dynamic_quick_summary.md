# Experiment 4: Global Responsibility-Weighted Dynamic Prototypes Summary

Seeds: 0

## Clean Accuracy & Structural Adaptation

| Variant | Train | Val | Test | Δ vs Direct | Active | Pruned | Coalesced | Spawned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct CNN | 92.80% | 89.00% | 89.30% | - | - | - | - | - |
| Fixed-Capacity Hopfield | 81.90% | 79.00% | 80.60% | -8.70 pp | 10.0 | 0.0 | 0.0 | 0.0 |
| Dynamic Adaptive Hopfield | 85.85% | 83.20% | 83.00% | -6.30 pp | 20.0 | 0.0 | 0.0 | 10.0 |

## Corruption Robustness (Means Across Seeds)

| Corruption | Level | Direct CNN | Fixed-Capacity Hopfield | Dynamic Adaptive Hopfield |
|---|---:|---:|---:|---:|
| gaussian | 0.0 | 89.30% | 80.60% | 83.00% |
| gaussian | 0.5 | 45.60% | 22.30% | 27.70% |
| gaussian | 1.0 | 24.60% | 13.10% | 13.20% |
| salt_pepper | 0.0 | 89.30% | 80.60% | 83.00% |
| salt_pepper | 0.2 | 55.90% | 33.20% | 38.70% |
| occlusion | 0 | 89.30% | 80.60% | 83.00% |
| occlusion | 7 | 84.30% | 76.80% | 77.90% |

Raw JSON metrics: [global_dynamic_quick.json](global_dynamic_quick.json).
