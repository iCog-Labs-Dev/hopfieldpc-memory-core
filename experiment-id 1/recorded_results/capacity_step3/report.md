# Step 3: capacity comparison

Validation only. Alpha=0.03; final epoch retained. Clean accuracy is the primary criterion.
Each original-capacity bank exactly reproduces the selected Step 2 bank at every epoch.
Actual CNN training is shared across independent banks; memory cannot affect its loss.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean clean accuracy |
|---|---:|---:|---:|---:|
| direct | 98.520% | 98.960% | 98.840% | 98.773% (n=3) |
| capacity_8 | 97.840% | 98.720% | 98.560% | 98.373% (n=3) |
| capacity_16 | 98.000% | 98.760% | 98.680% | 98.480% (n=3) |
| accuracy_guided | 97.660% | 98.440% | 98.380% | 98.160% (n=3) |
| fixed_kmeans_reference | 98.320% | 98.540% | 98.520% | 98.460% (n=3) |

## Normalized validation robustness AUC

| Method | Gaussian | Salt-and-pepper | Occlusion |
|---|---:|---:|---:|
| direct | 0.573300 | 0.712837 | 0.878779 |
| capacity_8 | 0.628373 | 0.762767 | 0.873693 |
| capacity_16 | 0.623020 | 0.755183 | 0.876438 |
| accuracy_guided | 0.601357 | 0.731171 | 0.864310 |
| fixed_kmeans_reference | 0.603753 | 0.739717 | 0.874795 |

Selected capacity: 16. This is validation selection, not a new test result.
Higher robustness AUC is better. Per-class accuracy, confusion, attention, per-seed paired comparisons against direct and the capacity control, writes, costs, and storage are in the structured metrics.
The fixed reference retains its original count and post-training center. Adaptive centers remain frozen at warm-up; actual adaptive occupancy is reported.
Immediate per-write predictions are unavailable for all banks; pre-write snapshot and batch post-write diagnostics remain enabled.

[Structured metrics](results.json) | [CSV](results.csv) | [Manifest](manifest.json) | [Selection](selection.json)
