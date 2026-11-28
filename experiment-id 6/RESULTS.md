# Experiment 6 results · GASS retrieval on MNIST

The following runs evaluate decoder-output memory. The full-data classification and reconstruction baselines are in the [Experiment 5 report](<../experiment-id 5/RESULTS.md>).

## Random initialization and noisy recall

In a separate 12-epoch CUDA comparison, each seed trained one decoder on 4,096 MNIST images. The three memory banks saw the same decoder outputs in the same order. They started with 20 slots and used a 256-image structural window, Dice reads and writes, read β 80 and write β 20. Both random banks began with identical keys and values. The updated bank used 784, the number of pixels, as the initial step size for each key update. The previous implementation used 1. Both applied the same feasibility and objective-decrease checks.

| Mean over seeds 0, 1, 2 | Noisy pixel MSE ↓ | Top-weighted slot held the closest stored image ↑ |
|---|---:|---:|
| Warm-up | 0.06364 | 20.6% |
| Random, previous key update | 0.06381 | 23.0% |
| Random, updated key learning | 0.05476 | 46.1% |

Each run evaluated the first 768 official test digits with Gaussian noise (σ = 0.3, corruption seed 22000 + run seed). The closest slot minimizes pixel MSE between its stored image and the clean target. The reconstruction uses the weighted blend of all active slots. The updated random bank reduced noisy pixel MSE in all three seeds and lowered the mean by 14.2%. A separate seed-0 run with 10 initial slots and read β 20 reduced noisy MSE from 0.06038 to 0.05288.

## Retrieval-protected proposal on noisy MNIST

This comparison uses the updated random Dice keys in every memory bank. For each of seeds 0, 1 and 2, one decoder trained for 12 epochs on the same 4,096 clean MNIST images. Every bank within a seed received the same initial prototypes, image order and decoder snapshots. The test used all 10,000 official MNIST images with Gaussian noise (σ = 0.3). Training used minibatches of 128, a structural window of 256 images, 20 starting slots, capacity limits of 2–80, and inverse temperatures of 80 for reads and 20 for writes. Each bank could make up to five accepted edits per structural pass.

The clean training condition associated clean decoder logits with themselves. The paired condition associated noisy cues with clean decoder logits from the same network snapshot. Ordinary GASS accepted changes based on the combined objective. Monotonic MARC also required average retrieval loss to stay level or decrease for each committed change on the current evidence. The experiment used RC-ARKU for values, with the scalar dual solver and checks on both quadratic bounds. It applied retrieval checks to keys, mixture weights and structural edits. Strict MARC additionally required a decrease proportional to value movement, using a single prespecified coefficient of 1/(8 × 784). Protection covered each fresh minibatch or structural window, with no replay or per-digit constraints.

With noisy training cues and clean targets, ordinary GASS achieved lower held-out pixel MSE than both protected modes in all three seeds. Monotonic MARC raised mean error by 12.3% and strict MARC by 13.4%. With clean self-association, monotonic MARC raised mean error by 15.6%.

| Readout and training | Test pixel MSE ↓ | Digit 2 MSE ↓ | Digit 4 MSE ↓ | Test retrieval KL ↓ | Closest-slot agreement ↑ | Final slots (seeds 0, 1, 2) |
|---|---:|---:|---:|---:|---:|---:|
| Direct decoder | 0.04297 | 0.04951 | 0.04569 | — | — | — |
| Ordinary GASS, clean | 0.05529 | 0.06750 | 0.05601 | 0.07398 | 54.1% | 19, 15, 16 |
| MARC, clean | 0.06393 | 0.07343 | 0.06225 | 0.10503 | 67.1% | 12, 3, 22 |
| Ordinary GASS, paired | 0.05387 | 0.06781 | 0.05278 | 0.07265 | 66.3% | 18, 14, 16 |
| MARC, paired | 0.06047 | 0.07166 | 0.06154 | 0.09573 | 65.7% | 12, 10, 9 |
| Strict MARC, paired | 0.06111 | 0.07616 | 0.06156 | 0.09742 | 80.2% | 5, 7, 6 |

The pixel errors compare the reconstructed image with the clean MNIST image. Retrieval KL compares the output probabilities with the clean decoder probabilities. Closest-slot agreement measures whether the slot given the most weight holds the closest stored image, measured against the clean target. Strict MARC reached 80.2% agreement with an average of six slots. Ordinary paired GASS reached 66.3% with 16 slots and achieved lower pixel error on both digits 2 and 4. Capacity limits were matched, while the number and contents of the surviving slots differed.

In the paired condition, ordinary GASS increased retrieval KL on 207 of 1,152 completed minibatch writes and 35 of 118 accepted structural edits. Monotonic MARC recorded zero increases in 1,152 writes and 79 accepted edits. Strict MARC recorded zero in 1,152 writes and 90 accepted edits. The audit used a numerical tolerance of 1e-10. These counts compare the bank immediately before and after each operation on the same examples and representation. They measure a different outcome from denoising unseen digits after training.

To reproduce the comparison from the repository root, run the following command. The JSON records each seed's test metrics, digit errors, active slots, and counts of loss-increasing training writes and edits. [Benchmark](experiment.py) · [Per-seed measurements](results/retrieval_protection_mnist.json)

```powershell
.\.venv\Scripts\python.exe -u "experiment-id 6/experiment.py" --device cuda --seeds 0 1 2 --epochs 12 --output "experiment-id 6/results/retrieval_protection_mnist.json"
```

### Using the structural reference window for every MARC write

This comparison uses noisy cues and clean targets in every arm. Each fresh write still fits the current 128-image minibatch and updates trust from that minibatch once. In reference mode, MARC evaluates candidate values, keys and mixture weights against the same 256 training images used for structural edits. The reference images and their noise realization are fixed within a run. Each write re-encodes the noisy/clean reference pairs using the same decoder snapshot as the minibatch, so the before/after check isolates the memory change. A subsequent decoder update can change the loss on those images.

Switching monotonic MARC to the fixed reference lowered test error in each of the three seeds and lowered the mean by 3.0%. It also improved mean reconstruction error for digits 2 and 4. Ordinary GASS had the lowest test MSE in each seed. Strict MARC improved with the reference in seed 0 and had higher error in seeds 1 and 2; its mean test error changed by less than 0.1%.

| Method and MARC window | Test pixel MSE ↓ | Digit 2 MSE ↓ | Digit 4 MSE ↓ | Writes raising batch KL | Writes raising reference KL | Final slots (seeds 0, 1, 2) |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary GASS | 0.05387 | 0.06781 | 0.05278 | 207 / 1152 | 318 / 1152 | 18, 14, 16 |
| Monotonic, minibatch | 0.06047 | 0.07166 | 0.06154 | 0 / 1152 | 176 / 1152 | 12, 10, 9 |
| Monotonic, reference | 0.05865 | 0.06927 | 0.05807 | 3 / 1152 | 0 / 1152 | 7, 10, 9 |
| Strict, minibatch | 0.06111 | 0.07616 | 0.06156 | 0 / 1152 | 176 / 1152 | 5, 7, 6 |
| Strict, reference | 0.06114 | 0.07048 | 0.06188 | 2 / 1152 | 0 / 1152 | 5, 15, 11 |

Both fixed-reference modes recorded zero reference regressions above 1e-10 across 1,152 writes. Monotonic reference MARC also passed the structural guard for all 78 accepted edits, and strict reference MARC for all 61. The minibatch modes each recorded 176 writes that increased reference KL. Monotonic reference MARC selected the closest stored digit on 73.7% of test images, compared with 65.7% for monotonic minibatch MARC. These percentages depend on the stored prototypes available in each bank.

The benchmark audits both minibatch and reference loss around each write, including for ordinary GASS. The comparison uses the same random starts, 4,096 training images, 12 epochs, seeds 0–2 and 10,000 official test digits as the preceding experiment. [Per-seed metrics and audit counts](results/marc_reference_window_mnist.json)

```powershell
.\.venv\Scripts\python.exe -u "experiment-id 6/experiment.py" --comparison reference --device cuda --seeds 0 1 2 --epochs 12 --output "experiment-id 6/results/marc_reference_window_mnist.json"
```

### Random versus warm-up initialization with fixed-reference MARC

Each seed trained one decoder for 12 epochs on 4,096 MNIST training images. The memory banks shared its outputs, image order, 256-image reference window, and 20 starting slots. Each finished bank reconstructed all 10,000 official test images from the same noisy cues within a seed (Gaussian σ = 0.3). Clean self-association matches the UI's training task. Paired training uses noisy cues and clean decoder targets. Both training tasks used monotonic MARC, and the paired comparison also included strict MARC.

For the UI's clean-image training with monotonic MARC, one warm-up epoch lowered noisy test reconstruction error in all three seeds. Mean pixel MSE fell by 2.9%. With noisy cues and clean targets, random initialization achieved the lowest mean error under monotonic MARC and won in seeds 0 and 2. Under strict MARC, warm-up lowered mean error by 3.4% and won in all three seeds.

| MARC and training | Initialization | Test pixel MSE ↓ | Digit 2 MSE ↓ | Digit 4 MSE ↓ | Closest-slot agreement ↑ | Final slots (seeds 0, 1, 2) |
|---|---|---:|---:|---:|---:|---:|
| Monotonic, clean training (UI) | random | 0.06201 | 0.07454 | 0.06184 | 66.7% | 14, 3, 10 |
| Monotonic, clean training (UI) | warm-up | 0.06022 | 0.07436 | 0.05994 | 46.7% | 11, 8, 7 |
| Monotonic, paired training | random | 0.05865 | 0.06927 | 0.05807 | 73.7% | 7, 10, 9 |
| Monotonic, paired training | warm-up | 0.05992 | 0.07300 | 0.05854 | 48.6% | 6, 10, 12 |
| Strict, paired training | random | 0.06114 | 0.07048 | 0.06188 | 73.6% | 5, 15, 11 |
| Strict, paired training | warm-up | 0.05905 | 0.07077 | 0.05879 | 53.8% | 12, 10, 6 |

Random initialization more often gave the highest retrieval weight to the stored prototype closest to the clean image in all three training conditions. Each bank has its own prototypes, and its reconstruction blends multiple stored values. In the UI condition, random won on closest-slot agreement (66.7% versus 46.7%) while warm-up produced the more accurate blended image. Across the six configurations, all 6,624 completed writes and 448 accepted structural edits passed the reference-window retrieval check at a tolerance of 1e-10.

This compares the two initialization modes as implemented. Random starts with Gaussian prototypes, writes from the first batch, and starts the Dice key line search at 784. Warm-up trains the decoder for the first epoch, initializes keys and value centroids from the training reference at the end of that epoch, and starts its Dice key line search at 1. Thus the result includes the different write schedules and key updates as well as the choice of initial prototypes. [Per-seed metrics and write counts](results/marc_initialization_mnist.json)

```powershell
.\.venv\Scripts\python.exe -u "experiment-id 6/experiment.py" --comparison initialization --device cuda --seeds 0 1 2 --epochs 12 --output "experiment-id 6/results/marc_initialization_mnist.json"
```
