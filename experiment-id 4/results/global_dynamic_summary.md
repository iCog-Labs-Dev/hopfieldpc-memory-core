# Experiment 4: Global Responsibility-Weighted Dynamic Prototypes Summary

Seeds: 0, 1, 2

## Clean Accuracy & Structural Adaptation

| Variant | Train | Val | Test | Δ vs Direct | Active | Pruned | Coalesced | Spawned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct CNN | 99.78% | 98.99% | 99.14% | - | - | - | - | - |
| Fixed-Capacity Hopfield | 92.42% | 91.39% | 91.83% | -7.31 pp | 10.0 | 0.0 | 0.0 | 0.0 |
| Dynamic Adaptive Hopfield | 99.39% | 98.58% | 98.77% | -0.37 pp | 11.7 | 0.0 | 23.7 | 25.3 |

## Corruption Robustness (Means Across Seeds)

| Corruption | Level | Direct CNN | Fixed-Capacity Hopfield | Dynamic Adaptive Hopfield |
|---|---:|---:|---:|---:|
| gaussian | 0.0 | 99.14% | 91.83% | 98.77% |
| gaussian | 0.1 | 99.00% | 91.71% | 98.59% |
| gaussian | 0.2 | 98.35% | 91.21% | 97.88% |
| gaussian | 0.3 | 92.58% | 87.88% | 93.51% |
| gaussian | 0.5 | 53.10% | 59.61% | 61.15% |
| gaussian | 0.7 | 28.65% | 34.39% | 35.07% |
| gaussian | 1.0 | 17.18% | 19.45% | 19.93% |
| salt_pepper | 0.0 | 99.14% | 91.83% | 98.77% |
| salt_pepper | 0.05 | 97.09% | 90.19% | 96.49% |
| salt_pepper | 0.1 | 91.63% | 86.14% | 91.28% |
| salt_pepper | 0.2 | 71.69% | 71.74% | 74.25% |
| salt_pepper | 0.3 | 51.41% | 54.70% | 56.15% |
| salt_pepper | 0.4 | 34.90% | 39.06% | 39.97% |
| occlusion | 0 | 99.14% | 91.83% | 98.77% |
| occlusion | 7 | 93.58% | 86.48% | 93.05% |
| occlusion | 10 | 83.50% | 77.10% | 83.12% |
| occlusion | 14 | 61.03% | 57.56% | 61.08% |

Raw JSON metrics: [global_dynamic.json](global_dynamic.json).
