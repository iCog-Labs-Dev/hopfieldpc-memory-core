# Matched bank-construction diagnosis in frozen space

Same final frozen-control CNN, classifier, train/validation data, center and per-class slot counts.
This is an endpoint construction diagnostic, not an online all-correct retraining experiment.
No test images were opened. No parent checkpoint or bank was modified.

| Bank | Validation condition | Beta | Accuracy | Direct accuracy |
|---|---|---:|---:|---:|
| adaptive_during_training | clean:0 | 5 | 96.640% | 98.380% |
| adaptive_during_training | clean:0 | 20 | 94.760% | 98.380% |
| adaptive_during_training | clean:0 | 50 | 90.580% | 98.380% |
| adaptive_during_training | clean:0 | 100 | 88.600% | 98.380% |
| kmeans_matched_slots | clean:0 | 5 | 97.020% | 98.380% |
| kmeans_matched_slots | clean:0 | 20 | 97.420% | 98.380% |
| kmeans_matched_slots | clean:0 | 50 | 97.560% | 98.380% |
| kmeans_matched_slots | clean:0 | 100 | 97.520% | 98.380% |
| all_correct_static_stream | clean:0 | 5 | 97.180% | 98.380% |
| all_correct_static_stream | clean:0 | 20 | 97.360% | 98.380% |
| all_correct_static_stream | clean:0 | 50 | 97.300% | 98.380% |
| all_correct_static_stream | clean:0 | 100 | 97.320% | 98.380% |
| adaptive_during_training | gaussian:0.4 | 5 | 79.320% | 77.240% |
| adaptive_during_training | gaussian:0.4 | 20 | 73.180% | 77.240% |
| adaptive_during_training | gaussian:0.4 | 50 | 67.180% | 77.240% |
| adaptive_during_training | gaussian:0.4 | 100 | 65.280% | 77.240% |
| kmeans_matched_slots | gaussian:0.4 | 5 | 81.500% | 77.240% |
| kmeans_matched_slots | gaussian:0.4 | 20 | 84.660% | 77.240% |
| kmeans_matched_slots | gaussian:0.4 | 50 | 84.460% | 77.240% |
| kmeans_matched_slots | gaussian:0.4 | 100 | 84.400% | 77.240% |
| all_correct_static_stream | gaussian:0.4 | 5 | 80.180% | 77.240% |
| all_correct_static_stream | gaussian:0.4 | 20 | 84.080% | 77.240% |
| all_correct_static_stream | gaussian:0.4 | 50 | 83.960% | 77.240% |
| all_correct_static_stream | gaussian:0.4 | 100 | 83.980% | 77.240% |
| adaptive_during_training | salt_pepper:0.15 | 5 | 77.180% | 82.740% |
| adaptive_during_training | salt_pepper:0.15 | 20 | 79.640% | 82.740% |
| adaptive_during_training | salt_pepper:0.15 | 50 | 75.000% | 82.740% |
| adaptive_during_training | salt_pepper:0.15 | 100 | 72.800% | 82.740% |
| kmeans_matched_slots | salt_pepper:0.15 | 5 | 78.860% | 82.740% |
| kmeans_matched_slots | salt_pepper:0.15 | 20 | 81.300% | 82.740% |
| kmeans_matched_slots | salt_pepper:0.15 | 50 | 81.760% | 82.740% |
| kmeans_matched_slots | salt_pepper:0.15 | 100 | 81.780% | 82.740% |
| all_correct_static_stream | salt_pepper:0.15 | 5 | 77.760% | 82.740% |
| all_correct_static_stream | salt_pepper:0.15 | 20 | 80.840% | 82.740% |
| all_correct_static_stream | salt_pepper:0.15 | 50 | 81.140% | 82.740% |
| all_correct_static_stream | salt_pepper:0.15 | 100 | 81.220% | 82.740% |

## Stored-entry health

```json
{
  "adaptive_during_training": {
    "entries": 79,
    "classified_as_own_class": 78,
    "misclassified_entry_ids": [
      65
    ],
    "mean_own_class_margin": 3.091477394104004,
    "min_own_class_margin": -1.5517785549163818,
    "mean_raw_norm": 22.136137008666992
  },
  "kmeans_matched_slots": {
    "entries": 79,
    "classified_as_own_class": 79,
    "misclassified_entry_ids": [],
    "mean_own_class_margin": 9.274687767028809,
    "min_own_class_margin": 3.189493179321289,
    "mean_raw_norm": 24.818593978881836
  },
  "all_correct_static_stream": {
    "entries": 79,
    "classified_as_own_class": 79,
    "misclassified_entry_ids": [],
    "mean_own_class_margin": 9.293739318847656,
    "min_own_class_margin": 0.9644646644592285,
    "mean_raw_norm": 25.531457901000977
  }
}
```
