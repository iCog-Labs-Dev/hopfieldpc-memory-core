# Step 3: prototype capacity comparison

Status: fixed before the real-data comparison. The completed three-seed
[results are reported separately](STEP03_CAPACITY.md). The original pre-run
protocol remains preserved in the exact training source archive.

Variant: `exp001_step03_all_correct_capacity_comparison`.

Compare **8 versus 16 maximum prototypes per digit** at **alpha=0.03**, the
setting selected in Step 2. Use seeds 0/1/2, the same original 55,000/5,000
train/validation splits, the original one-epoch empty-bank warm-up, five total
epochs, the final checkpoint, direct-only CNN cross-entropy and Adam 0.001,
beta=5, one retrieval update, centered cosine assignment, threshold=0.2,
epsilon=1e-8, all-direct-correct admission, no refresh or eviction, and the frozen
warm-up training center. Capacity is a limit; report actual occupancy rather
than forcing every class to fill it. The official test set is not evaluated.

Independent banks share one actual CNN training trajectory per seed. No backbone
feature cache is used. Each capacity allocates its own empty buffers and copies
only the frozen center from warm-up. Never resize or duplicate active prototypes
to initialize the larger bank. All subsequent seed/add/merge operations follow
the unchanged policy and minibatch order.

At every continued epoch, require exact CNN, Adam, private shuffle, and optimizer
progress agreement with the original experiment. Require the eight-prototype
bank's vectors, center, labels, IDs, metadata, decisions, selected counters, and
clean validation metrics to reproduce Step 2's alpha=0.03 bank exactly. Stop on
any mismatch. Diagnostic contexts preserve model modes and RNG states.

At epoch boundaries, measure clean validation without writes. After final
training, compare direct CNN, both adaptive capacities, original accuracy-guided
memory, and the existing fixed K-means reference on the identical seeded
validation corruption grids. The fixed reference retains **8 per class** and
its post-training center. Do not rebuild it at 16, change beta, tune thresholds,
or add epochs in this comparison. Record actual memory counts and center timing.

Report per-seed clean/corruption accuracy, per-class scores, confusion, normalized
robustness AUC, paired counts/McNemar against direct, and paired 16-versus-8
predictions. AUC is accuracy integrated over corruption strength, not ROC AUC.
Raw entropy depends on entry count: interpret it with actual occupancy and top-1
mass rather than treating different raw entropy values as directly comparable.

Selection maximizes total correct clean-validation predictions across all three
seeds. Exact ties use mean Gaussian/salt-and-pepper/occlusion AUC, then smaller
capacity. An incomplete bank is unavailable; fallback must not become genuine
memory accuracy. This extends the prior clean-first preference and does not
establish an independent test result. Three seeds and reused validation data
remain exploratory.

Log every exposure and successful mutation in separate bounded streams per bank.
Use the tested lossless float32 residual codec and independent assignment/EMA
replay. Per-write immediate predictions remain disabled equally for both banks;
pre-write snapshot and whole-batch post-write diagnostics remain enabled.
Track writes/skips, saturation, ages, exposures, costs, and logical bank/log/
checkpoint storage. Distinguish in-sample diagnostics from validation metrics.

Flush log chunks before atomic epoch checkpoint/cursor commits. Save all banks,
model/Adam, split IDs, Python/NumPy/Torch RNG and private shuffle state, source,
settings, and progress. Resume verifies source/config/data/protected artifacts
and starts at the last committed epoch; uncommitted tails are excluded. A
128 MiB free-space reserve stops workers before exhausting disk. Archive the
exact training source and preserve the original and earlier-stage artifacts.
Do not claim mid-batch or cross-device bitwise resume.

Synthetic tests must establish different-capacity behavior, selected-alpha
control parity, read-only evaluation, paired counts, deterministic epoch resume,
and complete replay before the real comparison. Use independent seed workers
when helpful; aggregation rejects incompatible, duplicate, incomplete, or
unaudited seeds and recomputes the selection over all seeds.

```powershell
python -B "experiment-id 1/capacity_ablation.py" --step2-run <completed_step2_comparison_directory>
python -B "experiment-id 1/capacity_ablation.py" --step2-run <completed_step2_comparison_directory> --seeds 0
python -B "experiment-id 1/capacity_ablation.py" --resume <capacity_worker_directory>
python -B "experiment-id 1/capacity_ablation.py" --aggregate <worker_seed0_directory> <worker_seed1_directory> <worker_seed2_directory>
python -B -m pytest "experiment-id 1/tests/test_capacity_ablation.py" -q
```
