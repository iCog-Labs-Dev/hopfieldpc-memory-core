# Step 2: online alpha comparison

Status: the protocol was fixed before the real-data sweep. The completed
three-seed [results are reported separately](STEP02_ALPHA.md). The exact
pre-run protocol is also preserved in the training source archive.

Compare alpha **0.03, 0.1, 0.3** using the all-direct-correct admission policy.
Use seeds 0/1/2, the same original train/validation IDs, original empty-bank
warm-up states, five total epochs including one warm-up, Adam/direct-only CE,
eight slots per digit, beta=5, one retrieval update, threshold=0.2, epsilon=1e-8,
and frozen warm-up centers. Keep the final epoch; perform no checkpoint search.
The official test dataset is not loaded.

## Shared training and positive controls

All three banks receive the same detached minibatch embeddings from one actual
CNN forward/backward/optimizer trajectory per seed. This factors identical
computation: memory has no gradient path to the CNN and cannot influence its
loss or optimizer. Existing backbone feature caches are not used. Banks have
separate buffers, metadata, IDs, assignments, and log streams.

At every epoch boundary, require exact CNN/Adam/shuffle equality with the
original experiment and exact alpha=0.1 bank equality with Step 1, including
vectors, center, labels, IDs, exposure counters, timestamps, and source metadata.
Check alpha=0.1 decisions and validation metrics against Step 1 as well.
Any difference stops the run.

## Instrumentation and audit

The Step 1 experiment spent substantial time computing immediate predictions
before and after each write. Those local diagnostics are disabled for all
three banks in this sweep and explicitly recorded as unavailable, independently
of readiness/fallback. Pre-write batch comparisons, all four correctness
categories, and whole-batch post-write diagnostics remain enabled. Every
exposure and successful mutation retains its source and version records;
candidate/old/new vectors remain losslessly logged in bounded chunks.

Available disk space requires lossless residual coding rather than storing
three redundant full arrays. A preceding committed entry predicts the old
vector and the declared EMA predicts the new vector. Store their bitwise XOR
residuals, original-vector CRC checksums, and the full candidate vector. Even a
discrepancy from a predictor remains recoverable. A decoder reconstructs and
checks actual vectors before the independent policy audit; resume initializes
the predictor from the committed bank. Full old/new vectors remain recoverable.
A 64 MiB free-space reserve stops the run before logs/checkpoints exhaust the
drive; an interrupted run remains resumable from its last committed epoch.

The bank caches only occupancy counts between insertions, rebuilding after
state loading. Existing assignment, EMA, commit, and retrieval math is reused.
An independent NumPy replay checks sources, eligibility, snapshot versions,
capacity, nearest assignments, thresholds, exact float32 EMA, metadata, and final
buffers. Near ties or threshold boundaries use the original float32 Torch
expression to distinguish rounding from a policy discrepancy. The auditor is
checked against the earlier Step 1 logs before the sweep.

Checkpoint all three banks, model, Adam, RNG/private shuffle state, split IDs,
source, progress, and matching log cursors at epoch boundaries. Resume verifies
source, config, alpha grid, data, and protected parent/reference hashes; ignore
uncommitted tails. Archive the exact training source. Do not claim mid-batch
resume or cross-device bitwise reproducibility.

## Evaluation and selection

At epoch boundaries, evaluate clean validation without writes. At the final
epoch, evaluate all alpha banks, direct CNN, original accuracy-guided bank, and
fixed K-means reference on the same validation corruption grids as Step 1, using
the same seeded corruption stream. Both adaptive banks retain their warm-up
centers; the fixed reference retains its post-training training center.

Primary selection: **maximize the total correctly classified clean-validation
examples across seeds**. Break exact ties by mean normalized AUC across Gaussian,
salt-and-pepper, and occlusion grids, then prefer the alpha closest to 0.1, then
the smaller alpha. An incomplete bank is unavailable rather than direct
fallback accuracy. No alpha can be selected unless its bank is complete in
every reported seed.

Report every alpha, per-seed differences, confusion/per-class scores, paired
counts, AUC, actual occupancy, writes, costs, and limits. Selection uses validation
data and does not establish test performance. Three seeds remain exploratory.

```powershell
python -B "experiment-id 1/alpha_ablation.py" --step1-run <completed_step1_comparison_directory>
python -B "experiment-id 1/alpha_ablation.py" --resume <alpha_output_directory>
python -B "experiment-id 1/alpha_ablation.py" --aggregate <worker_run_directory> <other_worker_directory>
python -B -m pytest "experiment-id 1/tests/test_alpha_ablation.py" -q
```

Workers may use `--seeds` to run independent seeds. Aggregation rejects
incompatible sources/settings, duplicate seeds, incomplete audits, and changed
protected artifacts. It recomputes the selection over all reported seeds;
worker-local selections are not used for the combined decision.
