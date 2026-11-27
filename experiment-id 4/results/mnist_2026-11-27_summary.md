# Experiment 4: Global Responsibility-Weighted Dynamic Prototypes Summary

Seeds: 0, 1, 2

## Clean Accuracy & Structural Adaptation

| Variant | Train | Val | Test | Δ vs Direct | Active | Pruned | Coalesced | Spawned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct CNN | 99.77% | 98.96% | 99.02% | - | - | - | - | - |
| Fixed-Capacity Hopfield | 96.23% | 95.39% | 95.53% | -3.49 pp | 10.0 | 0.0 | 0.0 | 0.0 |
| Dynamic Adaptive Hopfield | 99.44% | 98.50% | 98.79% | -0.32 pp | 15.3 | 0.0 | 19.7 | 25.0 |

## Corruption Robustness (Means Across Seeds)

| Corruption | Level | Direct CNN | Fixed-Capacity Hopfield | Dynamic Adaptive Hopfield |
|---|---:|---:|---:|---:|
| gaussian | 0.0 | 99.02% | 95.53% | 98.79% |
| gaussian | 0.1 | 98.97% | 95.42% | 98.67% |
| gaussian | 0.2 | 98.43% | 94.86% | 98.14% |
| gaussian | 0.3 | 94.09% | 91.81% | 94.45% |
| gaussian | 0.5 | 58.18% | 64.84% | 63.26% |
| gaussian | 0.7 | 30.42% | 37.08% | 34.21% |
| gaussian | 1.0 | 18.05% | 21.31% | 19.15% |
| salt_pepper | 0.0 | 99.02% | 95.53% | 98.79% |
| salt_pepper | 0.05 | 97.20% | 93.87% | 96.94% |
| salt_pepper | 0.1 | 92.44% | 90.08% | 92.41% |
| salt_pepper | 0.2 | 75.39% | 76.78% | 76.72% |
| salt_pepper | 0.3 | 55.16% | 59.42% | 57.69% |
| salt_pepper | 0.4 | 37.51% | 42.46% | 39.67% |
| occlusion | 0 | 99.02% | 95.53% | 98.79% |
| occlusion | 7 | 93.64% | 90.29% | 93.13% |
| occlusion | 10 | 83.73% | 81.06% | 82.51% |
| occlusion | 14 | 61.32% | 60.59% | 60.45% |

Raw JSON metrics: [mnist_2026-11-27.json](mnist_2026-11-27.json).
