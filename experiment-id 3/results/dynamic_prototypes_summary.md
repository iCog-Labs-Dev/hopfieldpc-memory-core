# Experiment 3 Accuracy & Dynamic Capacity Summary

Seeds: 0, 1, 2

## Clean Accuracy & Dynamic Prototype Allocation

| Method | Train accuracy | Validation accuracy | Test accuracy | Active Prototypes | Spawned | Pruned |
|---|---:|---:|---:|---:|---:|---:|
| Direct CNN | 99.78% | 98.99% | 99.14% | - | - | - |
| Ema | 99.03% | 98.11% | 98.39% | 41.7 | 13.0 | 13.0 |
| Hybrid | 99.03% | 98.08% | 98.38% | 42.3 | 7.3 | 6.7 |
| Momentum Only | 98.72% | 97.80% | 98.10% | 43.3 | 11.3 | 9.3 |
| Kalman Only | 98.70% | 97.71% | 98.04% | 42.3 | 6.7 | 5.7 |

## Corruption Accuracy by Noise Level (Means Across Seeds)

| Corruption | Level | Direct CNN | Ema | Hybrid | Momentum Only | Kalman Only |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 0.0 | 99.14% | 98.39% | 98.38% | 98.10% | 98.04% |
| gaussian | 0.1 | 99.00% | 98.28% | 98.26% | 97.98% | 97.88% |
| gaussian | 0.2 | 98.35% | 97.53% | 97.54% | 97.16% | 97.12% |
| gaussian | 0.3 | 92.58% | 93.16% | 92.90% | 92.47% | 92.21% |
| gaussian | 0.5 | 53.10% | 61.12% | 61.04% | 60.24% | 59.43% |
| gaussian | 0.7 | 28.65% | 34.97% | 35.05% | 34.25% | 35.00% |
| gaussian | 1.0 | 17.18% | 20.14% | 20.27% | 19.91% | 20.90% |
| salt_pepper | 0.0 | 99.14% | 98.39% | 98.38% | 98.10% | 98.04% |
| salt_pepper | 0.05 | 97.09% | 96.24% | 95.99% | 95.53% | 95.91% |
| salt_pepper | 0.1 | 91.63% | 91.68% | 91.15% | 90.36% | 90.93% |
| salt_pepper | 0.2 | 71.69% | 75.42% | 75.06% | 74.43% | 73.89% |
| salt_pepper | 0.3 | 51.41% | 57.44% | 57.09% | 55.94% | 55.69% |
| salt_pepper | 0.4 | 34.90% | 40.21% | 40.29% | 39.36% | 39.60% |
| occlusion | 0 | 99.14% | 98.39% | 98.38% | 98.10% | 98.04% |
| occlusion | 7 | 93.58% | 91.54% | 91.35% | 90.98% | 90.91% |
| occlusion | 10 | 83.50% | 80.39% | 79.50% | 79.33% | 79.06% |
| occlusion | 14 | 61.03% | 57.94% | 56.12% | 56.41% | 55.55% |

The full corruption and per-epoch dynamic prototype results remain in [dynamic_prototypes.json](dynamic_prototypes.json).
