# Experiment 2: Multi-Dataset Prototype Learning Summary

Seeds: 0 | Device: `cpu` | Constraint Gate: `Train >= 80%`, `Test >= 80%`, `|Train - Val| <= 7%`

## 1. Accuracy & Generalization Gate Across Datasets

| Dataset | Method | Train | Validation | Test | |Train - Val| | Gate (>=80% & <=7%) |
|---|---|---:|---:|---:|---:|:---:|
| Fashion-MNIST | Direct CNN (Baseline) | 94.32% | 93.30% | 92.92% | 1.02 pp | **PASS** |
| Fashion-MNIST | Ema | 92.19% | 91.36% | 90.65% | 0.83 pp | **PASS** |
| Fashion-MNIST | Hybrid | 92.17% | 91.24% | 90.99% | 0.93 pp | **PASS** |
| Fashion-MNIST | Momentum Only | 90.52% | 89.66% | 89.00% | 0.86 pp | **PASS** |
| Fashion-MNIST | Kalman Only | 92.11% | 91.32% | 90.76% | 0.79 pp | **PASS** |
| KMNIST | Direct CNN (Baseline) | 99.80% | 99.34% | 97.85% | 0.46 pp | **PASS** |
| KMNIST | Ema | 99.52% | 99.20% | 97.09% | 0.32 pp | **PASS** |
| KMNIST | Hybrid | 99.60% | 99.20% | 97.01% | 0.40 pp | **PASS** |
| KMNIST | Momentum Only | 99.57% | 99.20% | 96.94% | 0.37 pp | **PASS** |
| KMNIST | Kalman Only | 99.54% | 99.26% | 97.01% | 0.28 pp | **PASS** |
| SVHN | Direct CNN (Baseline) | 97.09% | 95.76% | 95.94% | 1.33 pp | **PASS** |
| SVHN | Ema | 96.36% | 95.04% | 95.47% | 1.32 pp | **PASS** |
| SVHN | Hybrid | 96.41% | 95.14% | 95.52% | 1.27 pp | **PASS** |
| SVHN | Momentum Only | 96.27% | 94.84% | 95.13% | 1.43 pp | **PASS** |
| SVHN | Kalman Only | 96.44% | 95.20% | 95.56% | 1.24 pp | **PASS** |
| CIFAR-10 | Direct CNN (Baseline) | 90.83% | 86.92% | 86.94% | 3.91 pp | **PASS** |
| CIFAR-10 | Ema | 86.62% | 83.84% | 82.71% | 2.78 pp | **PASS** |
| CIFAR-10 | Hybrid | 86.30% | 83.28% | 82.24% | 3.02 pp | **PASS** |
| CIFAR-10 | Momentum Only | 86.39% | 83.32% | 82.32% | 3.07 pp | **PASS** |
| CIFAR-10 | Kalman Only | 86.28% | 83.46% | 82.51% | 2.82 pp | **PASS** |

## 2. Corruption Robustness AUC & Hopfield Memory Diagnostics

| Dataset | Method | Gaussian AUC | S&P AUC | Occlusion AUC | Active Slots | Test Entropy | Helped / Harmed |
|---|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | Direct CNN (Baseline) | 0.2257 | 0.3401 | 0.8654 | - | - | - |
| Fashion-MNIST | Ema | 0.2076 | 0.3110 | 0.8374 | 80.0 | 3.071 | +122.0 / -349.0 |
| Fashion-MNIST | Hybrid | 0.2091 | 0.3165 | 0.8434 | 80.0 | 3.060 | +118.0 / -311.0 |
| Fashion-MNIST | Momentum Only | 0.2135 | 0.3265 | 0.8408 | 80.0 | 3.120 | +143.0 / -535.0 |
| Fashion-MNIST | Kalman Only | 0.2084 | 0.3096 | 0.8282 | 80.0 | 3.055 | +121.0 / -337.0 |
| KMNIST | Direct CNN (Baseline) | 0.3856 | 0.4551 | 0.9294 | - | - | - |
| KMNIST | Ema | 0.3234 | 0.3747 | 0.9116 | 80.0 | 3.125 | +27.0 / -103.0 |
| KMNIST | Hybrid | 0.3420 | 0.3933 | 0.9112 | 80.0 | 3.146 | +28.0 / -112.0 |
| KMNIST | Momentum Only | 0.3417 | 0.3980 | 0.9099 | 80.0 | 3.144 | +34.0 / -125.0 |
| KMNIST | Kalman Only | 0.3539 | 0.4192 | 0.9093 | 80.0 | 3.152 | +17.0 / -101.0 |
| SVHN | Direct CNN (Baseline) | 0.2283 | 0.2771 | 0.7544 | - | - | - |
| SVHN | Ema | 0.2300 | 0.2661 | 0.7407 | 80.0 | 3.214 | +178.0 / -302.0 |
| SVHN | Hybrid | 0.2193 | 0.2540 | 0.7396 | 80.0 | 3.215 | +176.0 / -286.0 |
| SVHN | Momentum Only | 0.2207 | 0.2599 | 0.7408 | 80.0 | 3.232 | +158.0 / -371.0 |
| SVHN | Kalman Only | 0.2273 | 0.2626 | 0.7401 | 80.0 | 3.254 | +179.0 / -280.0 |
| CIFAR-10 | Direct CNN (Baseline) | 0.1797 | 0.2303 | 0.7181 | - | - | - |
| CIFAR-10 | Ema | 0.1704 | 0.2150 | 0.6596 | 80.0 | 3.281 | +223.0 / -646.0 |
| CIFAR-10 | Hybrid | 0.1649 | 0.2078 | 0.6589 | 80.0 | 3.276 | +230.0 / -700.0 |
| CIFAR-10 | Momentum Only | 0.1663 | 0.2142 | 0.6656 | 80.0 | 3.300 | +234.0 / -696.0 |
| CIFAR-10 | Kalman Only | 0.1602 | 0.1997 | 0.6559 | 80.0 | 3.284 | +248.0 / -691.0 |
