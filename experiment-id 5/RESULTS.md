# Experiment 5 results · MNIST · 27 November 2026

Both runs used the full MNIST splits and seed 0. Classification measures correct digit labels. Reconstruction measures pixel error against the clean target. Each result is a single-seed measurement.

## Digit classification

The linear head classified **98.95%** of the 10,000 test digits correctly. GASS classified **98.78%** correctly (-0.17 percentage points). Both read the same frozen 64-dimensional encoder. The memory started with 20 slots and finished with 11.

The supervised encoder trained for 5 epochs on 55,000 images with a batch size of 128 and learning rate 0.001. GASS associated the embeddings with ten-dimensional one-hot targets using cosine reads and cosine writes, with squared retrieval loss. Only the first memory pass treated the examples as fresh measurements. Each pass was evaluated on the same 5,000-image validation set.

| Memory pass | Validation accuracy | Active slots | Accepted edits |
|---:|---:|---:|---|
| 1 | 98.62% | 20 | None |
| 2 | 98.40% | 15 | 5 × prune |
| 3 | 98.48% | 11 | 4 × merge |

### Test accuracy by digit

| True digit | Images | Direct | GASS | GASS − direct (pp) |
|---:|---:|---:|---:|---:|
| 0 | 980 | 99.69% | 99.59% | -0.10 |
| 1 | 1135 | 99.82% | 99.47% | -0.35 |
| 2 | 1032 | 99.32% | 99.13% | -0.19 |
| 3 | 1010 | 99.21% | 99.41% | +0.20 |
| 4 | 982 | 98.88% | 98.88% | 0.00 |
| 5 | 892 | 99.44% | 99.22% | -0.22 |
| 6 | 958 | 98.54% | 98.54% | 0.00 |
| 7 | 1028 | 98.64% | 98.64% | 0.00 |
| 8 | 974 | 99.49% | 98.25% | -1.23 |
| 9 | 1009 | 96.43% | 96.63% | +0.20 |

Digit 9 had the lowest GASS recall at 96.63%. Its most frequent wrong prediction was 5, accounting for 11 of its 34 errors. [Classification settings and confusion matrices](results/classification_2026-11-27.json)

## Image reconstruction

The four readouts shared one autoencoder trained for 20 epochs on 55,000 clean MNIST images, with 5,000 held out for validation. The CUDA run took 22 minutes 57 seconds on an NVIDIA GeForce RTX 3070 Ti Laptop GPU. The benchmark evaluated 10,000 paired test images with clean inputs and Gaussian noise (σ = 0.3). Pixel MSE measures error against the clean image. Lower values are better.

| Readout | Clean pixel MSE | Noisy pixel MSE | Noisy images improved vs. direct | Final slots |
|---|---:|---:|---:|---:|
| Direct | 0.005098 | 0.037485 | — | — |
| EMA | 0.084782 | 0.102861 | 1.73% | 20 |
| Dynamic | 0.114159 | 0.132620 | 0.42% | 76 |
| GASS | 0.057312 | 0.058930 | 23.32% | 21 |

The direct decoder had the lowest mean error on both clean and noisy inputs. GASS had the lowest error among the three memory readouts. GASS’s highest per-digit noisy MSE was 0.071996 on digit 2 (direct 0.037890).

The GASS reconstruction profile used sigmoid-Dice scores and Bernoulli-KL retrieval loss. At the epoch boundaries the bank ranged from 19 to 21 active slots and finished with 21. It accepted a prune at epoch 16, a birth at epoch 17 and a split at epoch 20. Two other birth proposals were rejected. [Reconstruction settings, per-digit MSE and edit history](results/reconstruction_cuda_2026-11-27.json)

## Run settings and reproduction

Both runs used a 256-example structural window, 20 starting memories, an allowed range of 2–80 memories, patience 2, a merge threshold of 0.90, support fraction 0.10, minimum child mass 1 and a budget of five accepted edits per structural pass. The inverse temperatures were 80 for reads and 20 for writes. Run these commands from the repository root with the project Python environment (PyTorch 2.7.1+cu118). Classification ran on CPU. Reconstruction ran on the NVIDIA GeForce RTX 3070 Ti Laptop GPU using CUDA.

```powershell
.\.venv\Scripts\python.exe "experiment-id 5/experiment.py" --device cpu --gass-read-beta 80 --window-size 256 --initial-memories 20 --min-active-memories 2 --max-memories 80 --patience 2 --merge-threshold 0.90 --support-fraction 0.10 --min-child-mass 1 --edit-budget 5 --output "experiment-id 5/results/classification_2026-11-27.json"
.\.venv\Scripts\python.exe -u "experiment-id 5/mnist_autoencoder_ui_gass_proof.py" --benchmark --suite comparison --epochs 20 --seeds 0 --device cuda --window-size 256 --initial-memories 20 --min-active-memories 2 --max-memories 80 --patience 2 --merge-threshold 0.90 --support-fraction 0.10 --min-child-mass 1 --edit-budget 5 --gass-read-beta 80 --output "experiment-id 5/results/reconstruction_cuda_2026-11-27.json"
```
