# Experiment 4 (CIFAR-10): Global Responsibility-Weighted Dynamic Prototypes Summary

Seeds: 0, 1, 2

## Clean Accuracy & Structural Adaptation

| Variant | Train | Val | Test | Δ vs Direct | Active | Pruned | Coalesced | Spawned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct CNN | 85.36% | 78.35% | 78.29% | - | - | - | - | - |
| Fixed-Capacity Hopfield | 76.69% | 71.37% | 71.59% | -6.70 pp | 10.0 | 0.0 | 0.0 | 0.0 |
| Dynamic Adaptive Hopfield | 82.60% | 75.98% | 76.44% | -1.85 pp | 33.0 | 0.0 | 22.7 | 45.7 |

## Corruption Robustness (Means Across Seeds)

| Corruption | Level | Direct CNN | Fixed-Capacity Hopfield | Dynamic Adaptive Hopfield |
|---|---:|---:|---:|---:|
| gaussian | 0.0 | 78.29% | 71.59% | 76.44% |
| gaussian | 0.1 | 43.32% | 42.15% | 43.55% |
| gaussian | 0.2 | 18.71% | 20.16% | 19.98% |
| gaussian | 0.3 | 13.98% | 15.16% | 14.76% |
| gaussian | 0.5 | 11.98% | 12.46% | 12.37% |
| gaussian | 0.7 | 11.29% | 11.55% | 11.53% |
| gaussian | 1.0 | 10.87% | 11.10% | 11.07% |
| salt_pepper | 0.0 | 78.29% | 71.59% | 76.44% |
| salt_pepper | 0.05 | 41.63% | 40.08% | 41.88% |
| salt_pepper | 0.1 | 21.83% | 23.31% | 23.42% |
| salt_pepper | 0.2 | 14.25% | 15.64% | 15.26% |
| salt_pepper | 0.3 | 12.84% | 13.64% | 13.42% |
| salt_pepper | 0.4 | 11.97% | 12.67% | 12.56% |
| occlusion | 0 | 78.29% | 71.59% | 76.44% |
| occlusion | 7 | 64.12% | 57.22% | 62.04% |
| occlusion | 10 | 52.78% | 47.19% | 50.87% |
| occlusion | 14 | 40.35% | 35.93% | 38.30% |

Raw JSON metrics: [cifar10_global_dynamic.json](cifar10_global_dynamic.json).
