# Frozen-encoder control (validation only)

Same warm-up model, optimizer, train/validation IDs and shuffle state as the primary run.
Encoder frozen after warm-up; the shared classifier continues direct-path CE training.
Bank admission/assignment/EMA/beta/capacity rules are unchanged. No test set was opened.
This is a separate control, not a replacement for Experiment 1 and not evidence across multiple seeds.

| Validation condition | Direct accuracy | Adaptive accuracy at training beta=5 |
|---|---:|---:|
| clean:0 | 98.380% | 96.640% |
| gaussian:0.4 | 77.240% | 79.320% |
| salt_pepper:0.15 | 82.740% | 77.180% |

Encoder unchanged: True. Classifier updated: True.
Bank health: `{'entries': 79, 'classified_as_own_class': 78, 'misclassified_entry_ids': [65], 'mean_own_class_margin': 3.091477394104004, 'min_own_class_margin': -1.5517785549163818, 'mean_raw_norm': 22.136137008666992}`.

The endpoint beta sweep in results.json is a post-training readout ablation only.
