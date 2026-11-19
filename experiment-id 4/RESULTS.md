# Experiment 4 Results: Global Responsibility-Weighted Dynamic Prototype Spawning, Starvation Pruning, and Gram Coalescence in Continuous Hopfield Networks

## 1. Executive Summary

Experiment 4 evaluates **class-agnostic, label-free** dynamic prototype adaptation in a continuous Hopfield associative memory on MNIST across three random seeds (`0, 1, 2`). Unlike Experiments 1–3—which partitioned memory slots by ground-truth class labels during writes—Experiment 4 governs online updates, uncertainty propagation, support tracking, redundancy coalescence, and spectral spawning through a **single global soft responsibility distribution** over all active slots, with all internal operating scales derived analytically (`AUTO_CALIBRATE = True`).

Starting from $K_{\text{init}} = 10$ unsupervised k-means++ prototypes inside a static masked buffer of size $N_{\max} = 80$, **Dynamic Adaptive Hopfield** is compared against **Fixed-Capacity Hopfield** ($K \equiv 10$) and the **Direct CNN** reference pathway.

**Key Findings:**
1. **Highest Clean Accuracy and Lowest Capacity Across All Dynamic Experiments (Zero Labels Used):** **Dynamic Adaptive Hopfield** achieves **98.77%** clean test accuracy (within **$-0.37\text{ pp}$** of the **99.14%** Direct CNN ceiling) while converging to an average of only **11.7 active prototypes** (`12, 12, 11` across seeds `0, 1, 2`)—an **85.4% memory reduction** relative to the 80 supervised prototypes of Experiments 1 & 2, and a **72.3% reduction** relative to the ~42.3 supervised prototypes of Experiment 3.
2. **Complete Elimination of Unsupervised Seeding Collapse (+6.94 pp over Fixed-Capacity):** On Seeds `1` and `2`, unsupervised k-means++ at $K = 10$ double-books one digit cluster and leaves another digit unrepresented, locking **Fixed-Capacity Hopfield** at **87.46%** (Seed 1) and **89.25%** (Seed 2) test accuracy (**91.83%** 3-seed mean). **Dynamic Adaptive Hopfield** detects the overloaded multi-class prototype via structural uncertainty at Epoch 2, splits it along its leading principal component, and lifts test accuracy to **98.65%** (Seed 1) and **98.81%** (Seed 2), with cross-seed spread of just **0.19 pp** (`98.65%–98.84%`).
3. **Superior Corruption Robustness Over Both Direct CNN and Fixed-Capacity Hopfield:** Across noise families, **Dynamic Adaptive Hopfield** outperforms Direct CNN by up to **+8.05 percentage points** under Gaussian noise ($\sigma = 0.5$: `61.15%` vs. `53.10%`) and **+5.07 percentage points** under salt-and-pepper noise ($p = 0.4$: `39.97%` vs. `34.90%`), achieving the highest family-wide mean on both Gaussian (**72.13%** vs. `69.71%` Direct / `68.01%` Fixed) and salt-and-pepper (**76.15%** vs. `74.31%` Direct / `72.28%` Fixed), while virtually matching Direct CNN under square occlusion (**84.01%** vs. `84.31%`).
4. **Tight Fission–Coalescence Homeostasis Without Buffer Saturation:** Pairing Law-of-Total-Variance Gram coalescence with Inverse-Total-Variance child initialization and sharpness-scaled structural dispersion keeps peak active memory at **18–20 slots** (22.5%–25.0% of $N_{\max} = 80$), expanding during early representation discovery (`25.3` spawns) and compressing back to `11–12` clean prototypes (`23.7` merges, `0.0` starvation prunes) as the encoder converges.

---

## 2. Experimental Configuration

All internal thresholds ($\beta_{\text{read}}, \beta_{\text{write}}, P_0, P_{\min}, P_{\max}, \tau_{\text{sim}}, \lambda_{\text{sim}}, \alpha_N, \alpha_S, \tau_{\text{support}}, \tau_{\text{eff}}, N^{\text{Kish}}_{\min}, \tau_{\text{uncert}}, \alpha_{\text{split}}, R_0$) were left as `None` in [config.py](config.py) and computed analytically at runtime via `AUTO_CALIBRATE = True`.

| Setting | Value / Analytical Rule |
|---|---|
| Dataset & Splits | MNIST (`55,000` train / `5,000` val / `10,000` test) |
| Random seeds | `0, 1, 2` |
| Training epochs & warm-up | `10` epochs (`1` warm-up epoch before seeding) |
| Batch size (train / eval) | `128` / `256` |
| Optimizer & learning rate | Adam, `0.001` (direct cross-entropy; memory detached via `z.detach()`) |
| Embedding dimension ($D$) | `64` |
| Initial / Max / Min prototypes ($K_{\text{init}}, N_{\max}, K^{\text{global}}_{\min}$) | `10` / `80` / `2` |
| Relative Kalman noises ($Q, R_n$) | `0.01`, `0.05` (scaled by empirical warm-up variance $P_0 = s_{\text{var}}$) |
| Momentum & John Factor ($\gamma_{\text{mom}}, \gamma_{\text{John}}$) | `0.9`, `1.0` |
| Gram coalescence ($\tau_{\text{coal}}, T_{\text{coal}}$) | `0.95`, `2` consecutive epochs |
| Starvation patience ($T_{\text{prune}}$) | `2` consecutive epochs ($\alpha_N = \alpha_S = 1 - 1/T_{\text{prune}} = 0.5$) |
| Structural weights ($\lambda_{\text{disp}}, \lambda_{\text{sharp}}$) | `2.0`, `0.5` |
| Auto-resolved temperatures ($\beta_{\text{read}}, \beta_{\text{write}}$) | $\sqrt{D} + \ln K_{\text{active}} \approx 10.45$, $\;1.5\sqrt{D} + \ln K_{\text{active}} \approx 14.45$ |
| Auto-resolved variance & spawning ($P_0, P_\infty, \tau_{\text{uncert}}$) | $P_0 \in [8.84, 11.88]$, $\;s_{\text{var}}P_\infty \in [0.25, 0.33]$, $\;\tau_{\text{uncert}} \in [0.39, 0.52]$ |

---

## 3. Clean Accuracy & Dynamic Prototype Allocation

Values below are means across seeds `0, 1, 2`.

| Variant | Train Accuracy | Validation Accuracy | Test Accuracy | Δ vs. Direct (Test) | Final Active | Spawned | Coalesced | Pruned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Direct CNN** | **99.78%** | **98.99%** | **99.14%** | — | — | — | — | — |
| **Fixed-Capacity Hopfield** | 92.42% | 91.39% | 91.83% | -7.31 pp | 10.0 | 0.0 | 0.0 | 0.0 |
| **Dynamic Adaptive Hopfield** | **99.39%** | **98.58%** | **98.77%** | **-0.37 pp** | **11.7** | 25.3 | 23.7 | 0.0 |

### 3.1 Per-Seed Breakdown

| Seed | Variant | Train | Validation | Test | Δ vs. Direct | Final Active | Peak Active | Spawned | Coalesced | Test Entropy | Top-1 Mass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Direct CNN | 99.74% | 98.96% | 99.15% | — | — | — | — | — | — | — |
| 0 | Fixed-Capacity Hopfield | 99.43% | 98.64% | 98.77% | -0.38 pp | 10 | 10 | 0 | 0 | 0.091 | 97.69% |
| 0 | **Dynamic Adaptive Hopfield** | **99.42%** | **98.54%** | **98.84%** | **-0.31 pp** | **12** | 20 | 25 | 23 | 0.241 | 87.35% |
| 1 | Direct CNN | 99.84% | 99.12% | 99.15% | — | — | — | — | — | — | — |
| 1 | Fixed-Capacity Hopfield | 88.21% | 86.96% | 87.46% | -11.69 pp | 10 | 10 | 0 | 0 | 0.155 | 92.64% |
| 1 | **Dynamic Adaptive Hopfield** | **99.31%** | **98.62%** | **98.65%** | **-0.50 pp** | **12** | 18 | 25 | 23 | 0.215 | 88.51% |
| 2 | Direct CNN | 99.75% | 98.90% | 99.12% | — | — | — | — | — | — | — |
| 2 | Fixed-Capacity Hopfield | 89.63% | 88.58% | 89.25% | -9.87 pp | 10 | 10 | 0 | 0 | 0.169 | 92.50% |
| 2 | **Dynamic Adaptive Hopfield** | **99.45%** | **98.58%** | **98.81%** | **-0.31 pp** | **11** | 19 | 26 | 25 | 0.159 | 92.29% |

### 3.2 Epoch-by-Epoch Trajectory of Active Capacity and Validation Accuracy (`Dynamic Adaptive Hopfield`)

| Seed | Metric | Ep 1 | Ep 2 | Ep 3 | Ep 4 | Ep 5 | Ep 6 | Ep 7 | Ep 8 | Ep 9 | Ep 10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | Active Slots | 10 | 20 | 20 | 11 | 18 | 19 | 12 | 14 | 16 | **12** |
| **0** | Mem Val Acc | 87.06% | 97.94% | 97.86% | 98.26% | 98.12% | 97.82% | 98.36% | 98.64% | 98.54% | **98.54%** |
| **1** | Active Slots | 10 | 18 | 18 | 15 | 16 | 17 | 17 | 15 | 15 | **12** |
| **1** | Mem Val Acc | 85.16% | 97.20% | 97.80% | 97.84% | 98.40% | 98.44% | 98.58% | 98.26% | 98.48% | **98.62%** |
| **2** | Active Slots | 10 | 18 | 19 | 15 | 17 | 17 | 16 | 16 | 14 | **11** |
| **2** | Mem Val Acc | 87.28% | 97.84% | 98.04% | 97.76% | 98.36% | 97.22% | 98.60% | 98.60% | 98.64% | **98.58%** |

Notice the immediate self-repair at **Epoch 2**: on Seeds 1 and 2, validation accuracy jumps by **+12.04 pp** (`85.16% -> 97.20%`) and **+10.56 pp** (`87.28% -> 97.84%`) in a single epoch as spectral fission splits the merged-digit prototypes, while Fixed-Capacity remains stuck around `85%–88%` for all 10 epochs.

---

## 4. Corruption Robustness

### 4.1 Per-Level Corruption Accuracy (Means Across Seeds 0, 1, 2)

| Corruption | Level | Direct CNN | Fixed-Capacity Hopfield | Dynamic Adaptive Hopfield | Dynamic Δ vs. Direct | Dynamic Δ vs. Fixed |
|---|---:|---:|---:|---:|---:|---:|
| **Gaussian** | 0.0 | **99.14%** | 91.83% | 98.77% | -0.37 pp | **+6.94 pp** |
| **Gaussian** | 0.1 | **99.00%** | 91.71% | 98.59% | -0.41 pp | **+6.88 pp** |
| **Gaussian** | 0.2 | **98.35%** | 91.21% | 97.88% | -0.47 pp | **+6.67 pp** |
| **Gaussian** | 0.3 | 92.58% | 87.88% | **93.51%** | **+0.93 pp** | **+5.63 pp** |
| **Gaussian** | 0.5 | 53.10% | 59.61% | **61.15%** | **+8.05 pp** | **+1.54 pp** |
| **Gaussian** | 0.7 | 28.65% | 34.39% | **35.07%** | **+6.42 pp** | **+0.68 pp** |
| **Gaussian** | 1.0 | 17.18% | 19.45% | **19.93%** | **+2.75 pp** | **+0.48 pp** |
| **Salt-and-pepper** | 0.0 | **99.14%** | 91.83% | 98.77% | -0.37 pp | **+6.94 pp** |
| **Salt-and-pepper** | 0.05 | **97.09%** | 90.19% | 96.49% | -0.60 pp | **+6.30 pp** |
| **Salt-and-pepper** | 0.1 | **91.63%** | 86.14% | 91.28% | -0.35 pp | **+5.14 pp** |
| **Salt-and-pepper** | 0.2 | 71.69% | 71.74% | **74.25%** | **+2.56 pp** | **+2.51 pp** |
| **Salt-and-pepper** | 0.3 | 51.41% | 54.70% | **56.15%** | **+4.74 pp** | **+1.45 pp** |
| **Salt-and-pepper** | 0.4 | 34.90% | 39.06% | **39.97%** | **+5.07 pp** | **+0.91 pp** |
| **Occlusion** | 0 | **99.14%** | 91.83% | 98.77% | -0.37 pp | **+6.94 pp** |
| **Occlusion** | 7 | **93.58%** | 86.48% | 93.05% | -0.53 pp | **+6.57 pp** |
| **Occlusion** | 10 | **83.50%** | 77.10% | 83.12% | -0.38 pp | **+6.02 pp** |
| **Occlusion** | 14 | 61.03% | 57.56% | **61.08%** | **+0.05 pp** | **+3.52 pp** |

### 4.2 Aggregate Corruption Family Means

| Method | Gaussian Mean | Salt-and-Pepper Mean | Occlusion Mean |
|---|---:|---:|---:|
| **Direct CNN** | 69.71% | 74.31% | **84.31%** |
| **Fixed-Capacity Hopfield** | 68.01% | 72.28% | 78.24% |
| **Dynamic Adaptive Hopfield** | **72.13%** | **76.15%** | 84.01% |

Because **Dynamic Adaptive Hopfield** settles to a compact, well-separated set of `11–12` prototypes (rather than over-fragmenting into 60–80 tiny Voronoi cells), it retains **deep attractor basins** under severe additive/impulsive noise while simultaneously preserving clean class boundaries—beating Fixed-Capacity Hopfield at **every single corruption type and level** (`17 / 17` rows).

---

## 5. Cross-Experiment Comparison (Experiments 2, 3, and 4)

| Experiment & Best Dynamic/Hybrid Method | Uses Class Labels in Memory? | Active Prototypes (Mean) | Clean Test Acc | Gaussian Mean | Salt-and-Pepper Mean | Occlusion Mean |
|---|:---:|---:|---:|---:|---:|---:|
| **Exp 2 — Hybrid Kalman-Momentum** ([RESULTS.md](../experiment-id%202/RESULTS.md)) | Yes (supervised) | 79.7 / 80 | 98.51% | 70.99% | 77.13% | 82.31% |
| **Exp 3 — Class-Conditioned Dynamic** ([RESULTS.md](../experiment-id%203/RESULTS.md)) | Yes (supervised) | 42.3 / 160 | 98.38% | 71.92% | 76.33% | 81.34% |
| **Exp 4 — Global Dynamic Adaptive (Ours)** | **No (100% label-free)** | **11.7 / 80** | **98.77%** | **72.13%** | 76.15% | **84.01%** |

**Takeaway:** Removing class-label supervision and governing the entire lifecycle via global soft responsibilities, analytical self-calibration, and fission–coalescence variance duality achieves **higher clean accuracy (`98.77%`), higher Gaussian robustness (`72.13%`), and higher occlusion robustness (`84.01%`)** using **3.6× fewer prototypes** than Experiment 3 and **6.8× fewer prototypes** than Experiment 2.

---

## 6. Analysis & Limitations

- **Why Clean Accuracy Trails Direct CNN by `-0.37 pp`:** Consistent with Experiments 1–3, the linear classifier head (`net.head`) is trained exclusively on raw continuous embeddings $z \in \mathbb{R}^{64}$ and evaluated zero-shot on retrieved prototype mixtures $h = \sum_j w_{ij} M_j$ without a residual skip connection. Compressing 64-dimensional representations onto the convex hull of `11–12` unsupervised vectors trades `-0.37 pp` on ambiguous boundary digits for up to `+8.05 pp` denoising gains under heavy corruption. Co-adapting the readout head on $h$ or using an energy-gated residual skip connection can close this remaining clean gap in end-to-end architectures.
- **Role of Coalescence vs. Starvation Pruning:** Across all three seeds, `pruned = 0` while `coalesced = 23.7`. Because k-means++ seeding and principal-component fission place prototypes directly inside populated data manifolds rather than empty space, excess capacity manifests as converging duplicate prototypes—which are cleanly fused via moment-matched Gram coalescence rather than discarded by starvation pruning.
- **Sample Size:** Results reflect three random seeds (`0, 1, 2`) on MNIST; evaluating on higher-dimensional natural-image backbones (e.g., CIFAR-10/100 or predictive-coding layers in FabricPC) is the natural next step.

---

## 7. Reproduction

Run the full 3-seed study from the repository root:

```powershell
.\.venv\Scripts\python.exe "experiment-id 4\experiment.py"
```

Run the quick 1-seed validation:

```powershell
.\.venv\Scripts\python.exe "experiment-id 4\experiment.py" --quick --seeds 0
```

Run the Experiment 4 unit and integration tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_global_dynamic_bank.py "experiment-id 4/tests/test_global_dynamic_experiment.py" -q
```

**Artifacts:**
- Proposal: [PROPOSAL.md](PROPOSAL.md)
- Configuration: [config.py](config.py)
- Experiment runner: [experiment.py](experiment.py)
- Core module: [global_dynamic.py](../src/hopfieldpc/global_dynamic.py)
- Summary table: [results/global_dynamic_summary.md](results/global_dynamic_summary.md)
- Raw JSON metrics: [results/global_dynamic.json](results/global_dynamic.json)
