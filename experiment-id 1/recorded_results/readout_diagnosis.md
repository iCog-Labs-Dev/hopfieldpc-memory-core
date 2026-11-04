# Validation-only diagnosis of Experiment 1

Original checkpoints, banks and test results are unchanged. No test images were opened.
Beta changes below are post-training readout ablations, not retrained online experiments.
Value refresh replays accepted source IDs through the final encoder with the original entry IDs/EMA history.
It neither reselects candidates nor recreates online training with refresh enabled.

| Variant | Validation condition | Beta | Mean accuracy | Sample std | Correct-class mass |
|---|---|---:|---:|---:|---:|
| adaptive_saved | clean:0 | 5 | 98.160% | 0.434 pp | 0.457 |
| adaptive_saved | clean:0 | 20 | 95.873% | 1.206 pp | 0.770 |
| adaptive_saved | clean:0 | 50 | 90.527% | 1.024 pp | 0.838 |
| adaptive_saved | clean:0 | 100 | 88.127% | 0.757 pp | 0.849 |
| adaptive_saved_hard | clean:0 | -1 | 85.260% | 0.925 pp | 0.852 |
| adaptive_center_only | clean:0 | 5 | 98.267% | 0.321 pp | 0.475 |
| adaptive_center_only | clean:0 | 20 | 96.000% | 1.178 pp | 0.785 |
| adaptive_center_only | clean:0 | 50 | 91.293% | 0.514 pp | 0.851 |
| adaptive_center_only | clean:0 | 100 | 89.053% | 0.058 pp | 0.863 |
| adaptive_center_only_hard | clean:0 | -1 | 86.740% | 1.146 pp | 0.867 |
| adaptive_values_only | clean:0 | 5 | 97.693% | 0.881 pp | 0.576 |
| adaptive_values_only | clean:0 | 20 | 98.367% | 0.145 pp | 0.930 |
| adaptive_values_only | clean:0 | 50 | 97.713% | 0.219 pp | 0.963 |
| adaptive_values_only | clean:0 | 100 | 97.307% | 0.477 pp | 0.966 |
| adaptive_values_only_hard | clean:0 | -1 | 96.800% | 0.624 pp | 0.967 |
| adaptive_values_and_center | clean:0 | 5 | 98.087% | 0.277 pp | 0.628 |
| adaptive_values_and_center | clean:0 | 20 | 98.373% | 0.145 pp | 0.942 |
| adaptive_values_and_center | clean:0 | 50 | 97.813% | 0.261 pp | 0.966 |
| adaptive_values_and_center | clean:0 | 100 | 97.373% | 0.492 pp | 0.968 |
| adaptive_values_and_center_hard | clean:0 | -1 | 96.933% | 0.691 pp | 0.969 |
| fixed_kmeans | clean:0 | 5 | 98.460% | 0.122 pp | 0.825 |
| fixed_kmeans | clean:0 | 20 | 98.720% | 0.122 pp | 0.984 |
| fixed_kmeans | clean:0 | 50 | 98.773% | 0.151 pp | 0.987 |
| fixed_kmeans | clean:0 | 100 | 98.773% | 0.150 pp | 0.988 |
| fixed_kmeans_hard | clean:0 | -1 | 98.780% | 0.156 pp | 0.988 |
| adaptive_saved | gaussian:0.4 | 5 | 78.047% | 9.633 pp | 0.332 |
| adaptive_saved | gaussian:0.4 | 20 | 72.727% | 13.669 pp | 0.535 |
| adaptive_saved | gaussian:0.4 | 50 | 62.800% | 12.878 pp | 0.562 |
| adaptive_saved | gaussian:0.4 | 100 | 59.480% | 12.510 pp | 0.565 |
| adaptive_saved_hard | gaussian:0.4 | -1 | 56.760% | 11.657 pp | 0.566 |
| adaptive_center_only | gaussian:0.4 | 5 | 74.700% | 7.260 pp | 0.358 |
| adaptive_center_only | gaussian:0.4 | 20 | 73.907% | 7.132 pp | 0.588 |
| adaptive_center_only | gaussian:0.4 | 50 | 67.907% | 6.282 pp | 0.629 |
| adaptive_center_only | gaussian:0.4 | 100 | 65.787% | 6.034 pp | 0.635 |
| adaptive_center_only_hard | gaussian:0.4 | -1 | 63.873% | 5.583 pp | 0.637 |
| adaptive_values_only | gaussian:0.4 | 5 | 76.367% | 15.505 pp | 0.400 |
| adaptive_values_only | gaussian:0.4 | 20 | 79.953% | 9.111 pp | 0.692 |
| adaptive_values_only | gaussian:0.4 | 50 | 76.467% | 9.531 pp | 0.732 |
| adaptive_values_only | gaussian:0.4 | 100 | 75.207% | 9.370 pp | 0.738 |
| adaptive_values_only_hard | gaussian:0.4 | -1 | 73.940% | 8.814 pp | 0.738 |
| adaptive_values_and_center | gaussian:0.4 | 5 | 77.947% | 8.123 pp | 0.452 |
| adaptive_values_and_center | gaussian:0.4 | 20 | 77.313% | 6.301 pp | 0.704 |
| adaptive_values_and_center | gaussian:0.4 | 50 | 75.007% | 5.871 pp | 0.730 |
| adaptive_values_and_center | gaussian:0.4 | 100 | 74.487% | 5.616 pp | 0.733 |
| adaptive_values_and_center_hard | gaussian:0.4 | -1 | 73.733% | 5.165 pp | 0.734 |
| fixed_kmeans | gaussian:0.4 | 5 | 79.600% | 7.721 pp | 0.596 |
| fixed_kmeans | gaussian:0.4 | 20 | 80.520% | 6.669 pp | 0.792 |
| fixed_kmeans | gaussian:0.4 | 50 | 80.387% | 6.266 pp | 0.803 |
| fixed_kmeans | gaussian:0.4 | 100 | 80.287% | 6.263 pp | 0.804 |
| fixed_kmeans_hard | gaussian:0.4 | -1 | 80.307% | 6.175 pp | 0.803 |
| adaptive_saved | salt_pepper:0.15 | 5 | 85.167% | 3.310 pp | 0.367 |
| adaptive_saved | salt_pepper:0.15 | 20 | 80.027% | 6.425 pp | 0.613 |
| adaptive_saved | salt_pepper:0.15 | 50 | 72.193% | 7.266 pp | 0.659 |
| adaptive_saved | salt_pepper:0.15 | 100 | 69.433% | 7.667 pp | 0.666 |
| adaptive_saved_hard | salt_pepper:0.15 | -1 | 67.053% | 7.947 pp | 0.670 |
| adaptive_center_only | salt_pepper:0.15 | 5 | 82.253% | 3.960 pp | 0.387 |
| adaptive_center_only | salt_pepper:0.15 | 20 | 80.353% | 4.281 pp | 0.645 |
| adaptive_center_only | salt_pepper:0.15 | 50 | 74.640% | 6.023 pp | 0.694 |
| adaptive_center_only | salt_pepper:0.15 | 100 | 72.573% | 6.552 pp | 0.703 |
| adaptive_center_only_hard | salt_pepper:0.15 | -1 | 70.580% | 7.147 pp | 0.706 |
| adaptive_values_only | salt_pepper:0.15 | 5 | 83.460% | 6.061 pp | 0.446 |
| adaptive_values_only | salt_pepper:0.15 | 20 | 84.433% | 4.538 pp | 0.753 |
| adaptive_values_only | salt_pepper:0.15 | 50 | 82.213% | 4.299 pp | 0.793 |
| adaptive_values_only | salt_pepper:0.15 | 100 | 80.973% | 4.319 pp | 0.799 |
| adaptive_values_only_hard | salt_pepper:0.15 | -1 | 80.187% | 4.260 pp | 0.799 |
| adaptive_values_and_center | salt_pepper:0.15 | 5 | 84.460% | 2.808 pp | 0.490 |
| adaptive_values_and_center | salt_pepper:0.15 | 20 | 83.260% | 3.665 pp | 0.765 |
| adaptive_values_and_center | salt_pepper:0.15 | 50 | 81.607% | 3.992 pp | 0.795 |
| adaptive_values_and_center | salt_pepper:0.15 | 100 | 80.907% | 4.076 pp | 0.799 |
| adaptive_values_and_center_hard | salt_pepper:0.15 | -1 | 80.353% | 4.051 pp | 0.800 |
| fixed_kmeans | salt_pepper:0.15 | 5 | 86.407% | 2.575 pp | 0.648 |
| fixed_kmeans | salt_pepper:0.15 | 20 | 86.673% | 2.695 pp | 0.852 |
| fixed_kmeans | salt_pepper:0.15 | 50 | 86.447% | 2.957 pp | 0.862 |
| fixed_kmeans | salt_pepper:0.15 | 100 | 86.400% | 2.998 pp | 0.863 |
| fixed_kmeans_hard | salt_pepper:0.15 | -1 | 86.367% | 3.079 pp | 0.864 |

## Actual training-log audit

```json
[
  {
    "checks_passed": true,
    "counts": {
      "seed": 10,
      "local_compared": 1539,
      "local_correct_after": 198,
      "examples": 220000,
      "capacity_skips": 841,
      "fallbacks": 128,
      "both_correct": 215124,
      "add": 70,
      "harmed": 2380,
      "both_wrong": 1812,
      "merge": 1469,
      "helped": 556
    },
    "writes": 1549,
    "seed": 0,
    "checkpoint_sha256": "15fb48d63380d1f2b1d2d639a19776ba689483a772ea4ba9da9f5b94b11302d7",
    "original_memory_health": {
      "entries": 80,
      "classified_as_own_class": 79,
      "misclassified_entry_ids": [
        56
      ],
      "mean_own_class_margin": 2.893247127532959,
      "min_own_class_margin": -1.7704076766967773,
      "mean_raw_norm": 22.88584327697754
    },
    "refreshed_memory_health": {
      "entries": 80,
      "classified_as_own_class": 80,
      "misclassified_entry_ids": [],
      "mean_own_class_margin": 5.87887716293335,
      "min_own_class_margin": 0.10605239868164062,
      "mean_raw_norm": 23.8325252532959
    },
    "fixed_memory_health": {
      "entries": 80,
      "classified_as_own_class": 80,
      "misclassified_entry_ids": [],
      "mean_own_class_margin": 11.980781555175781,
      "min_own_class_margin": 5.864774703979492,
      "mean_raw_norm": 27.933181762695312
    },
    "center_shift_l2": 10.847029685974121,
    "memory_refresh_l2": 10.847501754760742
  },
  {
    "checks_passed": true,
    "counts": {
      "seed": 10,
      "local_compared": 1556,
      "local_correct_after": 161,
      "examples": 220000,
      "capacity_skips": 1112,
      "fallbacks": 128,
      "both_correct": 215034,
      "add": 70,
      "harmed": 2668,
      "both_wrong": 1621,
      "merge": 1486,
      "helped": 549
    },
    "writes": 1566,
    "seed": 1,
    "checkpoint_sha256": "1854e8671d6a2931a0e21aadd8313045358f7a0685db8e0720dd1b7779053948",
    "original_memory_health": {
      "entries": 80,
      "classified_as_own_class": 78,
      "misclassified_entry_ids": [
        55,
        76
      ],
      "mean_own_class_margin": 3.171250104904175,
      "min_own_class_margin": -0.1817030906677246,
      "mean_raw_norm": 24.36334228515625
    },
    "refreshed_memory_health": {
      "entries": 80,
      "classified_as_own_class": 79,
      "misclassified_entry_ids": [
        77
      ],
      "mean_own_class_margin": 5.7496337890625,
      "min_own_class_margin": -0.44857168197631836,
      "mean_raw_norm": 24.860380172729492
    },
    "fixed_memory_health": {
      "entries": 80,
      "classified_as_own_class": 80,
      "misclassified_entry_ids": [],
      "mean_own_class_margin": 11.717962265014648,
      "min_own_class_margin": 6.164752960205078,
      "mean_raw_norm": 28.63137435913086
    },
    "center_shift_l2": 6.970616817474365,
    "memory_refresh_l2": 9.466867446899414
  },
  {
    "checks_passed": true,
    "counts": {
      "seed": 10,
      "local_compared": 1430,
      "local_correct_after": 181,
      "examples": 220000,
      "capacity_skips": 889,
      "fallbacks": 128,
      "both_correct": 215377,
      "helped": 481,
      "both_wrong": 1695,
      "add": 69,
      "harmed": 2319,
      "merge": 1361
    },
    "writes": 1440,
    "seed": 2,
    "checkpoint_sha256": "973a43147351482b7466c9c0c71c8a29d883a51187b896cf8f041d7b3151405a",
    "original_memory_health": {
      "entries": 79,
      "classified_as_own_class": 78,
      "misclassified_entry_ids": [
        71
      ],
      "mean_own_class_margin": 2.846391201019287,
      "min_own_class_margin": -0.4776735305786133,
      "mean_raw_norm": 23.499353408813477
    },
    "refreshed_memory_health": {
      "entries": 79,
      "classified_as_own_class": 78,
      "misclassified_entry_ids": [
        33
      ],
      "mean_own_class_margin": 5.766346454620361,
      "min_own_class_margin": -0.08732867240905762,
      "mean_raw_norm": 24.68623161315918
    },
    "fixed_memory_health": {
      "entries": 80,
      "classified_as_own_class": 80,
      "misclassified_entry_ids": [],
      "mean_own_class_margin": 12.790448188781738,
      "min_own_class_margin": 5.519989967346191,
      "mean_raw_norm": 29.693822860717773
    },
    "center_shift_l2": 6.95657205581665,
    "memory_refresh_l2": 9.350824356079102
  }
]
```

Changing beta in the actual online algorithm requires rerunning its admission/write history.
A frozen-encoder control is a different training variant, not demonstrated by this posthoc replay.
