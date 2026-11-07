# Step 1: online admission ablation

Validation only; the official test set was not opened.
Encoder and head continued training from the original empty-bank warm-up checkpoints.
CNN, Adam, and shuffle states matched the original trajectory exactly at every epoch.
Alpha=0.1, capacity=8/class, beta=5, 5 total epochs, frozen warm-up center, and all write rules were held fixed.
The sole admission change permits both-correct examples as well as direct-correct/Hopfield-wrong examples.
The fixed reference is reused from the identical original final CNN.

## Final clean validation

| Seed | Direct | Original accuracy-guided | All-direct-correct | Fixed K-means | New minus original (pp) |
|---|---:|---:|---:|---:|---:|
| 0 | 98.520% | 97.660% | 97.820% | 98.320% | +0.160 |
| 1 | 98.960% | 98.440% | 98.680% | 98.540% | +0.240 |
| 2 | 98.840% | 98.380% | 98.460% | 98.520% | +0.080 |
| Mean | 98.773% (n=3) | 98.160% (n=3) | 98.320% (n=3) | 98.460% (n=3) | +0.160 (n=3) |

## Normalized validation robustness AUC

| Method | Gaussian | Salt-and-pepper | Occlusion |
|---|---:|---:|---:|
| direct | 0.573300 (n=3) | 0.712837 (n=3) | 0.878779 (n=3) |
| accuracy_guided | 0.601357 (n=3) | 0.731171 (n=3) | 0.864310 (n=3) |
| all_direct_correct | 0.626530 (n=3) | 0.761613 (n=3) | 0.872343 (n=3) |
| fixed_kmeans_reference | 0.603753 (n=3) | 0.739717 (n=3) | 0.874795 (n=3) |

AUC is normalized accuracy integrated over the same corruption grids. These are validation measurements, not test results.
Reported seeds: [0, 1, 2]. Per-image tests and seed differences remain exploratory.
Missing-bank values are excluded, never replaced by direct fallback; n gives the available seed count.

## Bank and audit

| Seed | Entries | Seeds | Adds | Merges | Examples | Replay | Seconds |
|---|---:|---:|---:|---:|---:|---|---:|
| 0 | 80 | 10 | 70 | 211832 | 220000 | exact | 1897.0 |
| 1 | 80 | 10 | 70 | 211398 | 220000 | exact | 2599.9 |
| 2 | 80 | 10 | 70 | 211056 | 220000 | exact | 2098.3 |

Counts are exposures; repeated images are not distinct examples. Every example decision and mutation is logged.
More admissions also change update frequency; this comparison does not isolate semantic coverage from update rate.
No alpha, capacity, beta, epoch, or checkpoint selection was performed here. Incomplete-bank metrics remain unavailable.
Parent artifact hashes were checked again after training. Source, settings, environment, and split provenance are retained.

[Structured metrics](results.json) | [Per-condition CSV](results.csv) | [Manifest](manifest.json)
