# Step 2: alpha comparison

Validation only. Clean accuracy is the primary selection criterion; noise robustness breaks ties.
Independent banks share an actual direct-CE CNN training trajectory. No backbone feature cache is used.
CNN/Adam/shuffle and the alpha=0.1 bank were checked exactly against prior checkpoints at every epoch.
Every exposure and mutation is logged. Per-write local prediction diagnostics are explicitly unavailable; batch snapshot/post-write diagnostics remain enabled.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean clean accuracy |
|---|---:|---:|---:|---:|
| direct | 98.520% | 98.960% | 98.840% | 98.773% (n=3) |
| accuracy_guided | 97.660% | 98.440% | 98.380% | 98.160% (n=3) |
| alpha_0.03 | 97.840% | 98.720% | 98.560% | 98.373% (n=3) |
| alpha_0.1 | 97.820% | 98.680% | 98.460% | 98.320% (n=3) |
| alpha_0.3 | 97.780% | 98.600% | 98.400% | 98.260% (n=3) |
| fixed_kmeans_reference | 98.320% | 98.540% | 98.520% | 98.460% (n=3) |

## Normalized validation robustness AUC

| Method | Gaussian | Salt-and-pepper | Occlusion |
|---|---:|---:|---:|
| direct | 0.573300 | 0.712837 | 0.878779 |
| accuracy_guided | 0.601357 | 0.731171 | 0.864310 |
| alpha_0.03 | 0.628373 | 0.762767 | 0.873693 |
| alpha_0.1 | 0.626530 | 0.761613 | 0.872343 |
| alpha_0.3 | 0.625217 | 0.760475 | 0.871231 |
| fixed_kmeans_reference | 0.603753 | 0.739717 | 0.874795 |

Higher AUC is better; this is accuracy integrated over corruption strength, not ROC AUC.

Selected alpha: 0.03. This selection uses validation data; no new test result or across-seed significance claim is made.

[Structured metrics](results.json) | [CSV](results.csv) | [Manifest](manifest.json) | [Selection](selection.json)
