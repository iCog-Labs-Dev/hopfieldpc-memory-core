"""Step 1: actual online all-direct-correct admission, with validation-only evaluation.

Continue each primary run's empty-bank warm-up checkpoint. The encoder and head
continue direct-CE training. Compare every epoch's CNN, Adam, and shuffle state
bit-for-bit against the corresponding original checkpoint. Original artifacts
are read-only; the official test dataset is never loaded here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any
import uuid

import numpy as np
import torch

from mnist_online_memory import (
    EXPERIMENT_ROOT, build_baseline_net, evaluate, extract, make_bank, train_step,
    validate_config,
)
from benchmarks.mnist_baseline import environment_info, load_mnist_tensors, make_corrupted_test_set, robust_auc
from diagnose_memory import file_hash, memory_health
from online_memory_artifacts import (
    EventWriter, atomic_json, read_events, restore_rng, rng_state, save_checkpoint, source_identity, timed,
)

VARIANT_ID = "exp001_step01_online_all_direct_correct_admission"
POLICY = "all_direct_correct"
CPU = torch.device("cpu")


def require_equal(actual: Any, expected: Any, path: str = "state") -> None:
    """An exact comparison: no numerical tolerance can hide a trajectory change."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected.cpu()):
            raise AssertionError(f"Trajectory differs at {path}.")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"State keys differ at {path}.")
        for key in expected:
            require_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"State lengths differ at {path}.")
        for i, value in enumerate(expected):
            require_equal(actual[i], value, f"{path}[{i}]")
    elif actual != expected:
        raise AssertionError(f"State differs at {path}.")


def training_fingerprint(images: torch.Tensor, labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (images, labels):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        for part in tensor.split(1024):
            digest.update(part.numpy().tobytes())
    return digest.hexdigest()


def parent_fingerprints(parent: Path, cfg: dict[str, Any], seeds: list[int]) -> dict[str, str]:
    paths = [parent / "manifest.json"]
    for seed in seeds:
        paths.extend(parent / f"seed_{seed}" / "checkpoints" / f"epoch_{epoch}.pt"
                     for epoch in range(cfg["warmup_epochs"], cfg["total_epochs"] + 1))
        paths.append(parent / f"seed_{seed}" / "fixed_reference.pt")
    return {path.relative_to(parent).as_posix(): file_hash(path) for path in paths}


@torch.no_grad()
def audit_log(folder: Path, cursor: dict[str, Any], cfg: dict[str, Any], dim: int,
              expected_bank, train_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Bounded-memory replay, checking eligibility, assignment, EMA, and provenance."""
    replay = make_bank(cfg, dim)
    sources = set(train_ids.tolist())
    counts: Counter[str] = Counter()
    cached_name, vectors, pending, batch_key, batch_snapshot = None, None, None, None, None
    for event in read_events(folder, cursor):
        if event["type"] == "center":
            replay.initialize_center(torch.tensor(event["center"], dtype=torch.float32))
        elif event["type"] == "write":
            assert pending is None
            assert event["admission_policy"] == POLICY
            assert event["snapshot_category"] in ("bootstrap", "harmed", "both_correct")
            source = event["example_id"]
            assert source in sources and int(labels[source]) == event["label"]
            ref = event["vectors"]
            if ref["file"] != cached_name:
                with np.load(folder / ref["file"], allow_pickle=False) as data:
                    vectors = {key: data[key].copy() for key in ("candidate", "old", "new")}
                cached_name = ref["file"]
            row = ref["row"]
            candidate = torch.from_numpy(vectors["candidate"][row])
            proposal = replay.plan_candidate(candidate, event["label"], bootstrap=event["fallback"])
            assert proposal.is_write and proposal.action == event["action"]
            assert proposal.slot == event["entry_id"] and replay.version == event["version_before"]
            if proposal.distance is not None:
                assert abs(proposal.distance - event["distance"]) < 2e-6
            if event["action"] == "merge":
                require_equal(replay.values[proposal.slot], torch.from_numpy(vectors["old"][row]))
            require_equal(proposal.new_value, torch.from_numpy(vectors["new"][row]))
            committed = replay.commit(proposal, example_id=source, step=event["optimizer_step"], epoch=event["epoch"])
            assert committed["metadata"] == event["metadata"]
            assert committed["version_after"] == event["version_after"]
            counts[event["action"]] += 1
            pending = event
        elif event["type"] == "example":
            key = (event["epoch"], event["batch"])
            snapshot = (event["snapshot_version"], event["ready"])
            if key == batch_key:
                assert snapshot == batch_snapshot
            else:
                batch_key, batch_snapshot = key, snapshot
            source = event["example_id"]
            assert source in sources and int(labels[source]) == event["label"]
            assert event["admission_policy"] == POLICY
            assert event["direct_correct"] == (event["direct_prediction"] == event["label"])
            assert event["allows_consideration"] == event["direct_correct"]
            assert event["working_version_after"] == replay.version
            if event["action"] != "skip":
                assert pending is not None and event["write_seq"] == pending["seq"]
                assert source == pending["example_id"] and event["direct_correct"]
                assert event["working_version_after"] == event["working_version_before"] + 1
                pending = None
            else:
                assert pending is None and event["write_seq"] is None
                assert event["working_version_after"] == event["working_version_before"]
            counts["examples"] += 1
            counts["eligible"] += event["allows_consideration"]
            counts["fallback"] += not event["ready"]
            if event["ready"]:
                direct, hop = event["direct_correct"], event["hopfield_correct"]
                assert hop == (event["hopfield_prediction"] == event["label"])
                category = ("both_correct" if direct and hop else "harmed" if direct else
                            "helped" if hop else "both_wrong")
                assert event["category"] == category
                counts[category] += 1
    assert pending is None
    require_equal(replay.state_dict(), expected_bank.state_dict(), "replayed_bank")
    assert counts["seed"] + counts["add"] + counts["merge"] == expected_bank.version
    assert counts["both_correct"] + counts["harmed"] + counts["helped"] + counts["both_wrong"] + counts["fallback"] == counts["examples"]
    return {"replay_exact": True, "eligibility_and_mutations_verified": True, **counts}


def evaluate_comparison(model, new_bank, original_bank, fixed, images, labels, cfg) -> dict[str, Any]:
    current = evaluate(model, images, labels, new_bank, cfg, CPU, fixed)
    original = evaluate(model, images, labels, original_bank, cfg, CPU)
    require_equal(current["direct"], original["direct"], "evaluation.direct")
    comparison = {"direct": current["direct"], "accuracy_guided": original["adaptive_hopfield"],
                  "all_direct_correct": current["adaptive_hopfield"]}
    if fixed is not None:
        comparison["fixed_kmeans_reference"] = current["fixed_kmeans_reference"]
    return comparison


def train_seed(parent: Path, folder: Path, cfg: dict[str, Any], seed: int,
               images: torch.Tensor, labels: torch.Tensor, source: dict[str, str], *,
               resume: bool = False, stop_after_epoch: int | None = None) -> dict[str, Any] | None:
    warmup = cfg["warmup_epochs"]
    warm = torch.load(parent / f"seed_{seed}/checkpoints/epoch_{warmup}.pt", weights_only=False, map_location="cpu")
    if warm["config"] != cfg or warm["bank"]["_extra_state"]["version"] != 0:
        raise ValueError("Parent warm-up config or empty-bank state is incompatible.")
    train_ids = torch.as_tensor(warm["split_ids"]["train"], dtype=torch.long)
    val_ids = torch.as_tensor(warm["split_ids"]["val"], dtype=torch.long)
    all_ids = torch.cat([train_ids, val_ids])
    if (len(torch.unique(all_ids)) != len(all_ids) or int(all_ids.min()) < 0
            or int(all_ids.max()) >= len(images)):
        raise ValueError("Invalid or overlapping original training/validation IDs.")
    counts = torch.bincount(labels[train_ids], minlength=cfg["num_classes"])
    if len(counts) != cfg["num_classes"] or (counts < cfg["fixed_reference"]["prototypes_per_class"]).any():
        raise ValueError("Missing training classes or insufficient fixed-reference coverage.")
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / "latest.json"
    state = warm
    if resume and latest.exists():
        state = torch.load(folder / json.loads(latest.read_text())["checkpoint"], weights_only=False, map_location="cpu")
        if state["source"] != source or state["config"] != cfg or state["variant_id"] != VARIANT_ID:
            raise ValueError("Resume checkpoint source, config, or variant differs.")
        require_equal(state["split_ids"], {"train": train_ids, "val": val_ids}, "resume.splits")
    model = build_baseline_net(seed)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    optimizer.load_state_dict(state["optimizer"])
    bank = make_bank(cfg, model.head.fc.in_features)
    bank.load_state_dict(state["bank"])
    if not bank.center_initialized:
        raise ValueError("Parent checkpoint has no warm-up center.")
    shuffle = torch.Generator()
    shuffle.set_state(state["shuffle_rng"])
    restore_rng(state["rng"])
    is_continuation = state is not warm
    if is_continuation:
        original_resume = torch.load(parent / f"seed_{seed}/checkpoints/epoch_{state['epoch']}.pt",
                                     weights_only=False, map_location="cpu")
        require_equal(model.state_dict(), original_resume["model"], "resume.CNN")
        require_equal(optimizer.state_dict(), original_resume["optimizer"], "resume.Adam")
        require_equal(shuffle.get_state(), original_resume["shuffle_rng"], "resume.shuffle")
    history = list(state["history"]) if is_continuation else []
    costs = dict(state["costs"]) if is_continuation else {}
    elapsed_before = state.get("elapsed_seconds", 0.) if is_continuation else 0.
    writer = EventWriter(folder, cfg["event_chunk_size"], cfg["write_chunk_size"],
                         state["log_cursor"] if is_continuation else None)
    if not is_continuation:
        writer.event({"type": "center", "center": bank.center.tolist(), "origin": "parent_warmup_train_mean"})
        np.savez_compressed(folder / "split_ids.npz", train=train_ids.numpy(), val=val_ids.numpy())
    step = state["optimizer_step"]
    started = time.perf_counter()
    fixed_state = torch.load(parent / f"seed_{seed}/fixed_reference.pt", weights_only=False, map_location="cpu")
    fixed = (fixed_state["values"], fixed_state["center"])
    try:
        for epoch in range(state["epoch"] + 1, cfg["total_epochs"] + 1):
            print(f"all-direct-correct seed {seed}: epoch {epoch}/{cfg['total_epochs']}", flush=True)
            model.train()
            permutation = torch.randperm(len(train_ids), generator=shuffle)
            totals: Counter[str] = Counter()
            actions: Counter[str] = Counter()
            weighted_loss = 0.
            with timed(costs, "training_wall", CPU):
                for batch, start in enumerate(range(0, len(train_ids), cfg["batch_size"])):
                    source_ids = train_ids[permutation[start:start + cfg["batch_size"]]]
                    result = train_step(model, bank, optimizer, images[source_ids], labels[source_ids], source_ids,
                                        epoch=epoch, batch=batch, step=step, seed=seed, run_id=folder.parent.name,
                                        cfg=cfg, writer=writer, costs=costs, admission_policy=POLICY)
                    step += 1
                    weighted_loss += result["loss"] * result["examples"]
                    actions.update(result["counts"])
                    for name in ("examples", "direct_correct", "compared", "hop_correct", "fallback_count",
                                 "entropy_sum", "top1_mass_sum", "local_before_correct", "local_after_correct",
                                 "local_compared_writes", "in_sample_post_write_correct"):
                        totals[name] += result.get(name, 0)
                    if (batch + 1) % 50 == 0:
                        print(f"  seed {seed} epoch {epoch} batch {batch+1}/"
                              f"{(len(train_ids)+cfg['batch_size']-1)//cfg['batch_size']}: "
                              f"writes={bank.version}, entries={len(bank.metadata)}, "
                              f"seconds={time.perf_counter()-started:.1f}", flush=True)
            original = torch.load(parent / f"seed_{seed}/checkpoints/epoch_{epoch}.pt", weights_only=False, map_location="cpu")
            require_equal(model.state_dict(), original["model"], "CNN")
            require_equal(optimizer.state_dict(), original["optimizer"], "Adam")
            require_equal(shuffle.get_state(), original["shuffle_rng"], "shuffle")
            if step != original["optimizer_step"]:
                raise AssertionError("Optimizer-step count differs from the primary run.")
            original_bank = make_bank(cfg, model.head.fc.in_features)
            original_bank.load_state_dict(original["bank"])
            with timed(costs, "clean_validation", CPU):
                validation = evaluate_comparison(model, bank, original_bank,
                                                fixed if epoch == cfg["total_epochs"] else None,
                                                images[val_ids], labels[val_ids], cfg)
            compared = totals["compared"]
            metrics = {"epoch": epoch, "optimizer_step": step, "loss": weighted_loss / totals["examples"],
                       "direct_training_accuracy": totals["direct_correct"] / totals["examples"],
                       "adaptive_training_accuracy": totals["hop_correct"] / compared if compared else None,
                       "paired_delta": (actions["helped"]-actions["harmed"]) / compared if compared else None,
                       "counts": dict(actions), "totals": dict(totals), "validation": validation,
                       "bank": bank.storage(step), "cnn_exact": True, "adam_exact": True, "shuffle_exact": True}
            history.append(metrics)
            cursor = writer.flush()
            relative = f"checkpoints/epoch_{epoch}.pt"
            save_checkpoint(folder / relative, {"variant_id": VARIANT_ID, "source": source, "config": cfg,
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "bank": bank.state_dict(),
                            "shuffle_rng": shuffle.get_state(), "rng": rng_state(), "epoch": epoch,
                            "optimizer_step": step, "log_cursor": cursor, "history": history, "costs": costs,
                            "split_ids": {"train": train_ids, "val": val_ids},
                            "elapsed_seconds": elapsed_before + time.perf_counter()-started})
            atomic_json(latest, {"checkpoint": relative, "epoch": epoch, "log_cursor": cursor})
            atomic_json(folder / "progress.json", {"history": history})
            print(f"  seed {seed} epoch {epoch}: exact CNN/Adam/shuffle parity; "
                  f"validation direct={validation['direct']['accuracy']:.4f}, "
                  f"original={validation['accuracy_guided']['accuracy']}, "
                  f"all-correct={validation[POLICY]['accuracy']}", flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < cfg["total_epochs"]:
                return None
        cursor = writer.flush()
        final_original = torch.load(parent / f"seed_{seed}/checkpoints/epoch_{cfg['total_epochs']}.pt", weights_only=False, map_location="cpu")
        require_equal(model.state_dict(), final_original["model"], "final.CNN")
        require_equal(optimizer.state_dict(), final_original["optimizer"], "final.Adam")
        original_bank = make_bank(cfg, model.head.fc.in_features)
        original_bank.load_state_dict(final_original["bank"])
        rows, aucs = [], {}
        with timed(costs, "validation_corruption_evaluation", CPU):
            for kind, key in (("gaussian", "gaussian_sigmas"), ("salt_pepper", "salt_pepper_probs"), ("occlusion", "occlusion_sizes")):
                for level in cfg[key]:
                    corrupted = make_corrupted_test_set(images[val_ids], seed=seed, kind=kind, level=level, stream=1)
                    scored = evaluate_comparison(model, bank, original_bank, fixed, corrupted, labels[val_ids], cfg)
                    rows.extend({"seed": seed, "kind": kind, "level": level, "method": method, **value}
                                for method, value in scored.items())
                for method in ("direct", "accuracy_guided", POLICY, "fixed_kmeans_reference"):
                    values = [r["accuracy"] for r in rows if r["kind"] == kind and r["method"] == method]
                    aucs[f"{method}:{kind}"] = robust_auc(cfg[key], values) if all(v is not None for v in values) else None
        with timed(costs, "mutation_audit", CPU):
            audit = audit_log(folder, cursor, cfg, model.head.fc.in_features, bank, train_ids, labels)
        expected_examples = len(train_ids) * (cfg["total_epochs"]-warmup)
        if audit["examples"] != expected_examples:
            raise AssertionError("Post-warm-up exposure log is incomplete.")
        snapshot = bank.snapshot()
        health = memory_health(model.head, snapshot.values, snapshot.labels) if snapshot.ready else None
        result = {"seed": seed, "history": history, "validation_results": rows, "robust_auc": aucs,
                  "bank": bank.storage(step), "memory_health": health, "audit": audit, "costs": costs,
                  "elapsed_seconds": elapsed_before+time.perf_counter()-started, "log_cursor": cursor,
                  "cnn_exact_every_epoch": all(h["cnn_exact"] for h in history),
                  "adam_exact_every_epoch": all(h["adam_exact"] for h in history),
                  "disk_storage": {"checkpoints_bytes": sum(p.stat().st_size for p in (folder/"checkpoints").glob('*.pt')),
                                   "committed_logs_bytes": sum((folder/name).stat().st_size for key in ("event_files", "vector_files") for name in cursor[key])}}
        atomic_json(folder / "results.json", result)
        return result
    except Exception as error:
        atomic_json(folder / "failure.json", {"error": type(error).__name__, "message": str(error),
                    "optimizer_step": step, "uncommitted_log_cursor": writer.flush()})
        raise


def report(root: Path, results: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    methods = ("direct", "accuracy_guided", POLICY, "fixed_kmeans_reference")
    cfg = manifest["config"]
    rows = [row for result in results for row in result["validation_results"]]
    atomic_json(root / "results.json", {"manifest": manifest, "seeds": results})
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["seed", "kind", "level", "method", "accuracy", "delta_vs_direct", "evaluated",
                  "helped", "harmed", "both_correct", "both_wrong", "mcnemar_p"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Step 1: online admission ablation", "", "Validation only; the official test set was not opened.",
             "Encoder and head continued training from the original empty-bank warm-up checkpoints.",
             "CNN, Adam, and shuffle states matched the original trajectory exactly at every epoch.",
             f"Alpha={cfg['ema_alpha']:g}, capacity={cfg['max_memories_per_class']}/class, beta={cfg['hopfield_beta']:g}, "
             f"{cfg['total_epochs']} total epochs, frozen warm-up center, and all write rules were held fixed.",
             "The sole admission change permits both-correct examples as well as direct-correct/Hopfield-wrong examples.",
             "The fixed reference is reused from the identical original final CNN.", "",
             "## Final clean validation", "", "| Seed | Direct | Original accuracy-guided | All-direct-correct | Fixed K-means | New minus original (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    def clean(result, method):
        return next(r["accuracy"] for r in result["validation_results"] if r["kind"] == "gaussian" and r["level"] == 0 and r["method"] == method)
    def percent(value):
        return "unavailable" if value is None else f"{100*value:.3f}%"
    for result in results:
        values = [clean(result, method) for method in methods]
        delta = None if any(v is None for v in values[1:3]) else values[2]-values[1]
        lines.append(f"| {result['seed']} | " + " | ".join(percent(v) for v in values) + f" | {'unavailable' if delta is None else f'{100*delta:+.3f}'} |")
    available = [[clean(r, method) for r in results if clean(r, method) is not None] for method in methods]
    means = [statistics.mean(values) if values else None for values in available]
    paired = [clean(r, POLICY)-clean(r, 'accuracy_guided') for r in results
              if clean(r, POLICY) is not None and clean(r, 'accuracy_guided') is not None]
    lines.append("| Mean | " + " | ".join(percent(v)+f" (n={len(a)})" for v, a in zip(means, available)) + " | " +
                 ("unavailable" if not paired else f"{100*statistics.mean(paired):+.3f} (n={len(paired)})") + " |")
    lines += ["", "## Normalized validation robustness AUC", "", "| Method | Gaussian | Salt-and-pepper | Occlusion |", "|---|---:|---:|---:|"]
    for method in methods:
        values = []
        for kind in ("gaussian", "salt_pepper", "occlusion"):
            available = [r["robust_auc"][f"{method}:{kind}"] for r in results if r["robust_auc"][f"{method}:{kind}"] is not None]
            values.append(f"{statistics.mean(available):.6f} (n={len(available)})" if available else "unavailable")
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines += ["", "AUC is normalized accuracy integrated over the same corruption grids. These are validation measurements, not test results.",
              f"Reported seeds: {[r['seed'] for r in results]}. Per-image tests and seed differences remain exploratory.",
              "Missing-bank values are excluded, never replaced by direct fallback; n gives the available seed count.", "", "## Bank and audit", "",
              "| Seed | Entries | Seeds | Adds | Merges | Examples | Replay | Seconds |", "|---|---:|---:|---:|---:|---:|---|---:|"]
    for result in results:
        audit = result["audit"]
        lines.append(f"| {result['seed']} | {sum(result['bank']['occupancy'])} | {audit.get('seed',0)} | {audit.get('add',0)} | "
                     f"{audit.get('merge',0)} | {audit['examples']} | exact | {result['elapsed_seconds']:.1f} |")
    lines += ["", "Counts are exposures; repeated images are not distinct examples. Every example decision and mutation is logged.",
              "More admissions also change update frequency; this comparison does not isolate semantic coverage from update rate.",
              "No alpha, capacity, beta, epoch, or checkpoint selection was performed here. Incomplete-bank metrics remain unavailable.",
              "Parent artifact hashes were checked again after training. Source, settings, environment, and split provenance are retained.",
              "", "[Structured metrics](results.json) | [Per-condition CSV](results.csv) | [Manifest](manifest.json)"]
    (root / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, kind in zip(axes, ("gaussian", "salt_pepper", "occlusion")):
        for method in methods:
            levels = sorted({r["level"] for r in rows if r["kind"] == kind})
            values = []
            for level in levels:
                available = [r["accuracy"] for r in rows if r["method"] == method and r["kind"] == kind and r["level"] == level and r["accuracy"] is not None]
                values.append(statistics.mean(available)*100 if available else float('nan'))
            ax.plot(levels, values, marker="o", label=method)
        ax.set(title="Validation: "+kind, xlabel="Corruption level", ylabel="Accuracy (%)")
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root/"validation_corruptions.png", dpi=140)
    plt.close(fig)


def run_ablation(parent: Path | None = None, *, seeds: list[int] | None = None,
                 output: Path | None = None, resume: bool = False,
                 stop_after_epoch: int | None = None) -> Path:
    prior = None
    if resume:
        if output is None:
            raise ValueError("Resume requires the existing output directory.")
        prior = json.loads((output/"manifest.json").read_text())
        if prior["variant_id"] != VARIANT_ID:
            raise ValueError("Wrong resume variant.")
        if parent is not None or seeds is not None:
            raise ValueError("Resume uses its saved parent and seeds.")
        parent, seeds = Path(prior["parent_run"]), prior["seeds"]
    if parent is None:
        raise ValueError("A completed primary run directory is required.")
    parent = parent.resolve()
    parent_manifest = json.loads((parent/"manifest.json").read_text())
    cfg = validate_config(parent_manifest["config"])
    if parent_manifest.get("status") != "complete" or cfg["warmup_epochs"] < 1:
        raise ValueError("Require a completed primary run with at least one warm-up epoch.")
    seeds = cfg["seeds"] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s not in cfg["seeds"] for s in seeds):
        raise ValueError("Seeds must be a nonempty unique subset of the parent seeds.")
    if stop_after_epoch is not None and not cfg["warmup_epochs"] < stop_after_epoch <= cfg["total_epochs"]:
        raise ValueError("Invalid stopping epoch.")
    torch.set_num_threads(cfg["num_threads"])
    source = source_identity()
    fingerprints = parent_fingerprints(parent, cfg, seeds)
    images, labels = load_mnist_tensors(cfg["data_dir"], train=True)
    fingerprint = training_fingerprint(images, labels)
    if prior is not None:
        for key, value in (("source", source), ("parent_fingerprints", fingerprints), ("training_dataset_sha256", fingerprint), ("config", cfg)):
            if prior[key] != value:
                raise ValueError(f"Resume {key} is incompatible.")
    root = output or EXPERIMENT_ROOT/"results"/"variants"/f"admission_{uuid.uuid4().hex}"
    root = root.resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    manifest = {"variant_id": VARIANT_ID, "admission_policy": POLICY, "source": source,
                "parent_run": str(parent), "parent_fingerprints": fingerprints, "config": cfg, "seeds": seeds,
                "training_dataset_sha256": fingerprint, "validation_only": True, "test_set_opened": False,
                "encoder_frozen": False, "environment": environment_info(CPU), "status": "running"}
    atomic_json(root/"manifest.json", manifest)
    print(f"Admission ablation output: {root}", flush=True)
    completed = []
    try:
        for seed in seeds:
            folder = root/f"seed_{seed}"
            result_file = folder/"results.json"
            result = json.loads(result_file.read_text()) if resume and result_file.exists() else train_seed(
                parent, folder, cfg, seed, images, labels, source, resume=resume, stop_after_epoch=stop_after_epoch)
            if result is None:
                atomic_json(root/"manifest.json", manifest | {"status": "incomplete", "last_seed": seed})
                return root
            completed.append(result)
        if fingerprints != parent_fingerprints(parent, cfg, seeds):
            raise AssertionError("An original artifact changed during the control.")
        manifest |= {"status": "complete", "parent_artifacts_unchanged": True,
                     "cnn_and_adam_exact_every_epoch": all(r["cnn_exact_every_epoch"] and r["adam_exact_every_epoch"] for r in completed)}
        report(root, completed, manifest)
        atomic_json(root/"manifest.json", manifest)
    except Exception as error:
        atomic_json(root/"manifest.json", manifest | {"status": "failed", "error": type(error).__name__, "message": str(error)})
        raise
    print(f"Admission ablation complete: {root}", flush=True)
    return root


def aggregate_runs(runs: list[Path], *, output: Path | None = None) -> Path:
    """Pool completed, audited seeds from compatible serial or independent runs.

    An incomplete worker may contribute completed seeds. Uncommitted tails are
    never read, duplicated seeds are rejected, and original files are rehashed.
    The aggregate is a report with worker pointers, not a resumable checkpoint.
    """
    if not runs:
        raise ValueError("At least one worker run is required.")
    runs = [path.resolve() for path in runs]
    manifests = [json.loads((path/"manifest.json").read_text()) for path in runs]
    first = manifests[0]
    if any(m.get("variant_id") != VARIANT_ID for m in manifests):
        raise ValueError("Only admission-ablation worker runs can be aggregated.")
    for manifest in manifests[1:]:
        for key in ("source", "config", "training_dataset_sha256", "parent_run", "admission_policy"):
            if manifest[key] != first[key]:
                raise ValueError(f"Workers have incompatible {key}.")
    parent = Path(first["parent_run"]).resolve()
    fingerprints, results, origins, required = {}, {}, [], set()
    for folder, manifest in zip(runs, manifests):
        required.update(manifest["seeds"])
        for name, digest in manifest["parent_fingerprints"].items():
            if name in fingerprints and fingerprints[name] != digest:
                raise ValueError("Workers have incompatible parent fingerprints.")
            fingerprints[name] = digest
        origin = {"run_directory": str(folder), "manifest_sha256": file_hash(folder/"manifest.json"),
                  "worker_status": manifest["status"], "completed_seeds": []}
        for seed in manifest["seeds"]:
            file = folder/f"seed_{seed}/results.json"
            if not file.is_file():
                continue
            if seed in results:
                raise ValueError(f"Duplicate completed seed: {seed}.")
            result = json.loads(file.read_text())
            if (result["seed"] != seed or not result["audit"]["replay_exact"]
                    or not result["audit"]["eligibility_and_mutations_verified"]
                    or not result["cnn_exact_every_epoch"] or not result["adam_exact_every_epoch"]):
                raise ValueError(f"Seed {seed} did not pass all required audits.")
            cfg = first["config"]
            if [h["epoch"] for h in result["history"]] != list(range(cfg["warmup_epochs"]+1, cfg["total_epochs"]+1)):
                raise ValueError(f"Seed {seed} has an incomplete epoch history.")
            with np.load(folder/f"seed_{seed}/split_ids.npz", allow_pickle=False) as ids:
                expected = len(ids["train"]) * (cfg["total_epochs"]-cfg["warmup_epochs"])
            if result["audit"]["examples"] != expected:
                raise ValueError(f"Seed {seed} has incomplete example logs.")
            result["worker_seed_directory"] = str(file.parent)
            results[seed] = result
            origin["completed_seeds"].append({"seed": seed, "result_sha256": file_hash(file)})
        origins.append(origin)
    if set(results) != required:
        raise ValueError(f"Missing completed seeds: {sorted(required-set(results))}.")
    for name, digest in fingerprints.items():
        if file_hash(parent/name) != digest:
            raise ValueError(f"Original parent artifact changed: {name}.")
    root = output or EXPERIMENT_ROOT/"results"/"variants"/f"admission_comparison_{uuid.uuid4().hex}"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"variant_id": VARIANT_ID+"_aggregate", "artifact_type": "completed_seed_aggregate",
                "source": first["source"], "aggregation_source": source_identity(), "config": first["config"],
                "training_dataset_sha256": first["training_dataset_sha256"], "parent_run": str(parent),
                "parent_fingerprints": fingerprints, "source_runs": origins, "seeds": sorted(results),
                "admission_policy": POLICY, "validation_only": True, "test_set_opened": False,
                "selection_priority": "clean_accuracy_first_then_noise_robustness",
                "parent_artifacts_unchanged": True, "cnn_and_adam_exact_every_epoch": True,
                "status": "complete", "note": "Only completed audited seed results are included; worker crash/stop tails are excluded."}
    for folder in runs:
        if (folder/"training_source.zip").is_file():
            manifest["training_source_archive"] = {"path": str(folder/"training_source.zip"),
                                                   "sha256": file_hash(folder/"training_source.zip")}
            break
    atomic_json(root/"manifest.json", manifest)
    report(root, [results[seed] for seed in sorted(results)], manifest)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--run-dir", type=Path)
    source_group.add_argument("--resume", type=Path)
    source_group.add_argument("--aggregate", nargs="+", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if args.resume and args.output:
        parser.error("--resume already identifies the output directory")
    if args.aggregate:
        if args.seeds or args.stop_after_epoch is not None:
            parser.error("--aggregate uses completed worker seeds and does not train")
        print(aggregate_runs(args.aggregate, output=args.output))
    else:
        run_ablation(args.run_dir, seeds=args.seeds, output=args.resume or args.output,
                     resume=bool(args.resume), stop_after_epoch=args.stop_after_epoch)
