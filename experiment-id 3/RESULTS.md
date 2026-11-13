# Experiment 3 Results: Dynamic Prototype Spawning and Pruning in Continuous Hopfield Networks

## 1. Executive Summary

Experiment 3 evaluates dynamic prototype allocation (uncertainty-driven spawning and support-based pruning) combined with Kalman-momentum prototype updates on MNIST across three random seeds (`0, 1, 2`). Instead of fixing the memory bank to $8$ prototypes per class ($80$ total) as in Experiments 1 and 2, each class starts with an online admission cap of $K_{\text{init}} = 4$ prototypes and dynamically spawns or prunes prototypes within a masked buffer of size $K_{\max} = 16$ per class ($160$ maximum slots).

**Key Findings:**
1. **~47% Memory Reduction with Preserved Accuracy:** Dynamic allocation converges to an average of **41.7–43.3 active prototypes** across the 10 classes (nearly half the 80 prototypes used in Experiments 1 and 2), while maintaining **98.39% (EMA)** and **98.38% (Hybrid)** clean test accuracy.
2. **Substantial Robustness Gains over Direct CNN under Noise:** Across moderate and heavy Gaussian and salt-and-pepper corruptions, the dynamic Hopfield memory outperforms the Direct CNN baseline by up to **+8.02 percentage points** (at Gaussian $\sigma = 0.5$) and **+6.03 percentage points** (at salt-and-pepper $p = 0.3$).
3. **Reduced Prototype Churn with Hybrid Kalman-Momentum:** While EMA and Hybrid achieve virtually identical clean and corrupted accuracies, the **Hybrid** rule triggers nearly **half as many spawn/prune cycles** ($7.3$ spawned / $6.7$ pruned on average) as **EMA** ($13.0$ spawned / $13.0$ pruned), demonstrating that Kalman-momentum updates stabilize prototype trajectories and prevent redundant split-then-prune oscillations.
4. **Data-Driven Multi-Modal Allocation:** Across seeds, simpler digits retain $4$ prototypes, whereas visually multi-modal digits (especially digits `8`, `9`, `3`, `7`, and `5`) dynamically allocate and retain $5$ to $7$ prototypes.

---

## 2. Experimental Configuration

| Setting | Value |
|---|---:|
| Dataset | MNIST |
| Random seeds | 0, 1, 2 |
| Training epochs | 5 (including 1 warm-up epoch) |
| Training / evaluation batch size | 128 / 256 |
| Optimizer & learning rate | Adam, 0.001 |
| Embedding dimension | 64 |
| Initial online prototypes per class ($K_{\text{init}}$) | 4 |
| Maximum buffer prototypes per class ($K_{\max}$) | 16 (160 total slots) |
| Minimum support threshold ($\tau_{\text{support}}$) | 10 examples |
| Pruning patience ($T$) | 2 consecutive epochs |
| Spawning uncertainty threshold ($\tau_{\text{uncert}}$) | 0.5 |
| Principal component split scale ($\alpha_{\text{split}}$) | 0.1 |
| EMA learning rate ($\alpha$) | 0.1 |
| Momentum coefficient ($\gamma$) | 0.9 |
| Base process noise ($Q$) | 0.01 |
| Initial uncertainty ($P_0$) | 1.0 |
| Retrieval temperature ($\beta$) & updates | 5.0, 1 step |

---

## 3. Clean Accuracy & Dynamic Prototype Allocation

Values below are means across seeds `0, 1, 2`. Note that `Train accuracy` for memory methods is evaluated through the masked Hopfield memory readout on the clean training split (rather than reusing the direct CNN train accuracy).

| Method | Train accuracy | Validation accuracy | Test accuracy | Active Prototypes | Spawned | Pruned |
|---|---:|---:|---:|---:|---:|---:|
| **Direct CNN** | **99.78%** | **98.99%** | **99.14%** | - | - | - |
| **EMA** | **99.03%** | **98.11%** | **98.39%** | 41.7 | 13.0 | 13.0 |
| **Hybrid Kalman-Momentum** | **99.03%** | 98.08% | 98.38% | 42.3 | 7.3 | 6.7 |
| **Momentum-Only** | 98.72% | 97.80% | 98.10% | 43.3 | 11.3 | 9.3 |
| **Kalman-Only** | 98.70% | 97.71% | 98.04% | 42.3 | 6.7 | 5.7 |

### Per-Seed Breakdown

| Seed | Method | Train | Validation | Test | Final Active | Spawned | Pruned | Final Per-Class Prototype Counts (Digits 0–9) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | Direct CNN | 99.74% | 98.96% | 99.15% | - | - | - | - |
| 0 | EMA | 98.76% | 97.66% | 98.18% | 40 | 2 | 5 | `[4, 4, 4, 4, 4, 4, 4, 4, 4, 4]` |
| 0 | Hybrid | 98.82% | 97.68% | 98.23% | 41 | 2 | 2 | `[4, 4, 4, 5, 4, 4, 4, 4, 4, 4]` |
| 0 | Momentum-Only | 98.69% | 97.56% | 98.21% | 42 | 8 | 7 | `[4, 4, 4, 5, 4, 4, 3, 4, 4, 6]` |
| 0 | Kalman-Only | 97.81% | 96.68% | 97.15% | 42 | 6 | 6 | `[4, 4, 4, 5, 4, 4, 4, 5, 4, 4]` |
| 1 | Direct CNN | 99.84% | 99.12% | 99.15% | - | - | - | - |
| 1 | EMA | 99.30% | 98.48% | 98.65% | 42 | 17 | 17 | `[4, 4, 4, 4, 4, 4, 4, 4, 6, 4]` |
| 1 | Hybrid | 99.27% | 98.24% | 98.63% | 41 | 7 | 9 | `[4, 4, 4, 4, 4, 4, 4, 4, 5, 4]` |
| 1 | Momentum-Only | 98.81% | 98.00% | 98.09% | 42 | 10 | 11 | `[4, 4, 4, 4, 4, 4, 4, 4, 6, 4]` |
| 1 | Kalman-Only | 99.12% | 98.16% | 98.53% | 41 | 4 | 5 | `[4, 4, 4, 4, 4, 4, 4, 4, 5, 4]` |
| 2 | Direct CNN | 99.75% | 98.90% | 99.12% | - | - | - | - |
| 2 | EMA | 99.03% | 98.20% | 98.35% | 43 | 20 | 17 | `[4, 4, 4, 4, 4, 4, 4, 4, 6, 5]` |
| 2 | Hybrid | 99.00% | 98.32% | 98.29% | 45 | 13 | 9 | `[4, 4, 4, 4, 4, 4, 4, 6, 7, 4]` |
| 2 | Momentum-Only | 98.67% | 97.84% | 98.01% | 46 | 16 | 10 | `[4, 4, 4, 5, 4, 6, 4, 4, 5, 6]` |
| 2 | Kalman-Only | 99.17% | 98.30% | 98.45% | 44 | 10 | 6 | `[4, 4, 4, 4, 6, 4, 4, 4, 5, 5]` |

---

## 4. Corruption Robustness vs. Direct CNN Baseline

### 4.1 Per-Level Corruption Accuracy (Means Across Seeds 0, 1, 2)

| Corruption | Level | Direct CNN | EMA | Hybrid | Momentum-Only | Kalman-Only | Best Memory Gain vs. Direct |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Gaussian** | 0.0 | **99.14%** | 98.39% | 98.38% | 98.10% | 98.04% | -0.75 pp |
| **Gaussian** | 0.1 | **99.00%** | 98.28% | 98.26% | 97.98% | 97.88% | -0.72 pp |
| **Gaussian** | 0.2 | **98.35%** | 97.53% | 97.54% | 97.16% | 97.12% | -0.81 pp |
| **Gaussian** | 0.3 | 92.58% | **93.16%** | 92.90% | 92.47% | 92.21% | **+0.58 pp** |
| **Gaussian** | 0.5 | 53.10% | **61.12%** | 61.04% | 60.24% | 59.43% | **+8.02 pp** |
| **Gaussian** | 0.7 | 28.65% | 34.97% | **35.05%** | 34.25% | 35.00% | **+6.40 pp** |
| **Gaussian** | 1.0 | 17.18% | 20.14% | 20.27% | 19.91% | **20.90%** | **+3.72 pp** |
| **Salt-and-pepper** | 0.0 | **99.14%** | 98.39% | 98.38% | 98.10% | 98.04% | -0.75 pp |
| **Salt-and-pepper** | 0.05 | **97.09%** | 96.24% | 95.99% | 95.53% | 95.91% | -0.85 pp |
| **Salt-and-pepper** | 0.1 | 91.63% | **91.68%** | 91.15% | 90.36% | 90.93% | **+0.05 pp** |
| **Salt-and-pepper** | 0.2 | 71.69% | **75.42%** | 75.06% | 74.43% | 73.89% | **+3.73 pp** |
| **Salt-and-pepper** | 0.3 | 51.41% | **57.44%** | 57.09% | 55.94% | 55.69% | **+6.03 pp** |
| **Salt-and-pepper** | 0.4 | 34.90% | 40.21% | **40.29%** | 39.36% | 39.60% | **+5.39 pp** |
| **Occlusion** | 0 | **99.14%** | 98.39% | 98.38% | 98.10% | 98.04% | -0.75 pp |
| **Occlusion** | 7 | **93.58%** | 91.54% | 91.35% | 90.98% | 90.91% | -2.04 pp |
| **Occlusion** | 10 | **83.50%** | 80.39% | 79.50% | 79.33% | 79.06% | -3.11 pp |
| **Occlusion** | 14 | **61.03%** | 57.94% | 56.12% | 56.41% | 55.55% | -3.09 pp |

### 4.2 Aggregate Corruption Family Means

| Method | Gaussian Mean | Salt-and-Pepper Mean | Occlusion Mean |
|---|---:|---:|---:|
| **Direct CNN** | 69.71% | 74.31% | **84.31%** |
| **EMA** | **71.94%** | **76.56%** | 82.07% |
| **Hybrid Kalman-Momentum** | 71.92% | 76.33% | 81.34% |
| **Momentum-Only** | 71.44% | 75.62% | 81.21% |
| **Kalman-Only** | 71.51% | 75.68% | 80.89% |

---

## 5. Analysis & Comparison with Experiment 2

1. **Dynamic Capacity Efficiency:**
   - In Experiment 2, every class used a fixed allocation of $8$ prototypes ($80$ total), achieving $98.41\%$ (EMA) and $98.51\%$ (Hybrid) clean test accuracy.
   - In Experiment 3, starting at $K_{\text{init}} = 4$ and allowing dynamic spawning and pruning up to $K_{\max} = 16$, the network uses only **41.7–42.3 prototypes on average** (a **47% reduction** in active memory vectors) while achieving **98.39% (EMA)** and **98.38% (Hybrid)** clean test accuracy and **higher Gaussian robustness** ($71.94\%$ / $71.92\%$ in Experiment 3 vs. $70.91\%$ / $70.99\%$ in Experiment 2).
2. **Spawning and Pruning Dynamics:**
   - During Epoch 2, multi-modal clusters with high dispersion trigger prototype fission along their first principal component, expanding the active bank to $42\text{--}57$ prototypes.
   - In Epochs 3–5, poorly positioned child prototypes that fail to attract at least $\tau_{\text{support}} = 10$ embeddings for $T = 2$ consecutive epochs are pruned.
   - Visually complex digits (most consistently digit `8` across seeds, as well as digits `3`, `7`, and `9`) retain extra prototypes ($5\text{--}7$ slots), whereas simpler digits stay at $4$.
3. **Stability of Hybrid Updates vs. EMA:**
   - Because EMA applies a fixed step without uncertainty gating or momentum smoothing, its prototypes experience higher local dispersion early in training, causing **13.0 spawns and 13.0 prunes** per seed on average.
   - The **Hybrid Kalman-Momentum** method achieves the same final accuracy while cutting structural churn nearly in half (**7.3 spawns and 6.7 prunes** per seed).

---

## 6. Limitations

- **Three Seeds:** Results across seeds `0, 1, 2` provide descriptive evidence of dynamic allocation behavior and noise robustness rather than high-powered across-seed hypothesis tests.
- **Occlusion vs. Noise:** Consistent with Experiment 1 and Task 13, prototype retrieval improves additive/impulsive noise robustness (Gaussian and salt-and-pepper) where perturbations average out across dimensions, but slightly trails the Direct CNN under structured square occlusion.
- **Single-Stage Split Perturbation:** Spawning perturbs child prototypes along the first principal component of $\mathcal{S}_{j,n}$; some child prototypes land in low-density tails and are subsequently pruned $2$ epochs later.

---

## 7. Reproduction

Run the full 3-seed study from the repository root:

```powershell
python ".\experiment-id 3\dynamic_prototype_experiment.py"
```

Run the quick 1-seed validation:

```powershell
python ".\experiment-id 3\dynamic_prototype_experiment.py" --quick --seeds 0
```

Run the Experiment 3 unit test suite:

```powershell
pytest "experiment-id 3/tests/test_dynamic_prototype_experiment.py" -q
```

**Artifacts:**
- Implementation: [dynamic_prototype_experiment.py](dynamic_prototype_experiment.py)
- Proposal: [PROPOSAL.md](PROPOSAL.md)
- Summary table: [results/dynamic_prototypes_summary.md](results/dynamic_prototypes_summary.md)
- Raw JSON metrics: [results/dynamic_prototypes.json](results/dynamic_prototypes.json)
