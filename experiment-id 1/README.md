# Experiment 1 - Online Accuracy-Guided Hopfield Memory

Experiment ID: `exp_001_online_accuracy_guided_memory`. This is independent of
Task 13's E1–E9 tables and backbone caches. It implements explicit memory
buffers, not optimizer-trained `learnable_memory` parameters or FabricPC.

## Run

From the repository root, after `python -m pip install -r requirements.txt`:

```powershell
python -B "experiment-id 1/mnist_online_memory.py" --quick
python -B "experiment-id 1/mnist_online_memory.py"
python -B "experiment-id 1/mnist_online_memory.py" --resume <run_directory>
python -B -m pytest "experiment-id 1/tests" -q
python -B -m pytest tests "experiment-id 1/tests" -q
```

Quick runs have one seed, three epochs (one warm-up plus two online), 2,000
training images, 500 validation images and 1,000 test images. They are smoke
tests, not evidence of performance. Full runs use seeds 0/1/2 and five epochs.
The CLI does not launch a hyperparameter search. Both banks use beta=5 by
default. Other online beta/alpha/threshold settings require a fresh training run.

All Experiment 1 code, configuration, tests and outputs live in this folder. The repository's `src/` and existing benchmarks are unchanged. The local scoring adapter delegates similarities and softmax to the existing core, and applies its weights to raw values; configurable epsilon and repeated updates are local to this experiment. Original split IDs are recovered locally and verified against the unchanged split helper. Earlier smoke runs are preserved under `results/previous_layout/`; they predate this source layout and are not resumable under the new source fingerprint.

## State machine and step order

The CNN trains on clean images, using the existing normalization, encoder,
classifier, Adam optimizer and direct-path cross-entropy. Each image trains
exactly once per exposure, regardless of its memory decision.

1. **Warm-up:** bank empty, no writes. At the boundary, an eval/no-gradient
   pass over the selected training split freezes its mean embedding. This pass
   is timed separately and restores model modes and RNG states. With zero
   warm-up it occurs before training starts.
2. **Bootstrap:** only missing classes can receive a seed, and only from their
   first finite, nondegenerate, directly correct candidate. Each seeded class
   remains unchanged until coverage is complete. No random active slots or
   seed duplication are used.
3. **Active:** readiness is snapshotted for the whole minibatch. Completing
   bootstrap during a batch does not retroactively activate its comparisons.

Every batch computes embeddings once and direct logits with gradients. Under
no-grad, retrieval and the **same classifier** produce Hopfield predictions
against a frozen pre-write bank snapshot. Compute all eligibility flags before
mutating memory, then process candidates in shuffled minibatch order. Assignment
is recomputed against the current working bank, including earlier writes.
Immediate before/after predictions use the same detached candidate and fixed
classifier weights. Optional whole-batch post-write accuracy is an in-sample
diagnostic. Only afterward does the direct loss backpropagate, followed by
exactly one optimizer update. Post-write predictions never cause another write.

| Direct correct | Hopfield correct | Action |
|---|---|---|
| Yes | No | Eligible for assignment |
| No | Yes | Skip: helped |
| Yes | Yes | Skip: both correct |
| No | No | Skip: both wrong |

Correctness is argmax against the training label. Confidence, loss, entropy and
batch accuracy never override the table. The label constrains **writing** only.
Retrieval always searches all active entries; classification comes from the
shared head applied to the retrieved vector, never a memory-label vote.

## Geometry and writes

Store raw embeddings. Score with normalized `(query - center)` and
`(memory - center)`; use the unchanged core to obtain weights from these scoring views, then apply those weights to raw memories. Thus
the classifier receives a convex combination of raw vectors. Multiple updates
reuse the same frozen center. Assignment uses `1 - centered_cosine` in `[0,2]`.

For the nearest same-class entry, break exact ties by stable entry ID:

- Distance <= threshold: `new = (1-alpha)*old + alpha*candidate`.
- Distance > threshold with room: add a separate entry.
- Distance > threshold at capacity: `class_capacity_reached`; no eviction,
  unrelated merge or write to another class.

Defaults are alpha=.1, threshold=.2, epsilon=1e-8, maximum eight entries per
class. Eight is a limit, not a required occupancy. EMA is not a running mean.
Accepted-exposure counters include repeated examples across epochs. Stable IDs,
first/latest source IDs, creation/update encoder steps and full logged mutation
provenance distinguish exposures from distinct training examples.

Near-zero centered candidates and nonfinite keys are rejected. Degenerate EMA
proposals leave the bank unchanged. A zero centered query safely produces
uniform attention. Nonfinite CNN outputs, loss, gradients or retrieval stop
the run and create failure artifacts; they are not disguised as fallback.
Inactive preallocated slots never participate in retrieval.

## Evaluation and drift

Validate without writes at each epoch boundary. The checkpoint is always the
final epoch; there is no automated validation tuning or early stopping.
The final held-out comparison shares one encoder/head checkpoint:

- `direct`: classifier on the current embedding.
- `adaptive_hopfield`: the bank accumulated during training, using its frozen
  **warm-up training mean**.
- `fixed_kmeans_reference`: a separately constructed post-training bank with
  eight K-means prototypes per class, using its **post-training training mean**.

The fixed bank uses the existing construction code and matched beta/update
count/epsilon, not a beta search. Its extraction and construction cost are
separate. It never initializes or refreshes the adaptive bank. Actual counts
and center policies are reported. Degenerate fixed construction fails explicitly.

Corruptions use the existing paired seeded Gaussian, salt-and-pepper and
occlusion helpers. True labels only score predictions. Incomplete adaptive
banks report null accuracy, zero genuine comparisons and explicit fallback
counts. Fallback predictions are never presented as successful retrieval.

There is **no refresh**. A fixed train-only probe reports raw representation
displacement from warm-up and the preceding epoch. Stale embeddings and the
frozen center are limitations, not silently corrected by alignment, momentum
encoders, K-means refreshes, replay loss or other changes to the method.

## Artifacts, replay and resume

Outputs have unique paths under `experiment-id 1/results/full/<run_id>/` or
`experiment-id 1/results/quick/<run_id>/`. They never reuse Task 13 outputs.

- `resolved_config.yaml`, `manifest.json`: settings, source commit/source hash,
  dataset fingerprint, environment and terminal status.
- `summary.md`, `results.json`, `results.csv`: held-out accuracy/per-class
  accuracy, confusion, paired counts and McNemar p-values, robust AUC, bank
  occupancy/ages, costs and storage. Negative results remain intact.
- `plots/training.png`: validation accuracy, in-sample helped/harmed and growth.
  `plots/held_out.png`: separate test-corruption results.
- `seed_<n>/split_ids.npz`: original official dataset indices.
- `seed_<n>/events/*.jsonl.gz`: bounded example/mutation/center event chunks.
- `seed_<n>/writes/*.npz`: lossless compressed candidate/old/new vectors.
  Event vector references identify the chunk and row. Missing old values on
  seed/add writes are marked explicitly.
- `seed_<n>/metrics.jsonl`: per-epoch loss, training denominators, categories,
  local diagnostics, validation, bank state and drift.
- `seed_<n>/checkpoints/epoch_<n>.pt`, `latest.json`: committed model, Adam,
  memory/metadata/center/phase, probe, RNG and shuffle state, progress and log
  cursor. `fixed_reference.pt` is separate.
- `failure.json` and failed manifest status preserve numerical/runtime failures.

`replay_bank()` reconstructs raw bank values and metadata from the initial
center and committed mutation events, checking sequence/version/old-value
consistency. All post-warm-up example exposures have a decision record.
Snapshot comparisons, local pre-write comparisons, and direct fallback have
separate fields; unavailable Hopfield predictions are null.

Epoch logs are flushed before checkpoint/manifest commit. Resume verifies source,
configuration, data and splits and continues the same directory from the last
committed epoch. Uncommitted crash tails are excluded from the new cursor and
are not overwritten. Local checkpoints contain Python/NumPy RNG objects and
must be trusted before loading. Mid-batch resume and cross-device bitwise
reproducibility are not claimed; deterministic CPU epoch resume is tested.

Tensor storage distinguishes active vectors and allocated buffers. Metadata
storage is its serialized logical size, not a claim about Python heap overhead.
Log/checkpoint disk bytes and report-generation costs are separate. Component
timings include diagnostics/logging; total training wall time includes those
components and must not be summed with them. GPU timings synchronize boundaries.

Synthetic integration tests establish execution and invariants, not MNIST
performance. With three seeds, per-image paired tests and per-seed differences
do not establish strong across-seed statistical evidence.
