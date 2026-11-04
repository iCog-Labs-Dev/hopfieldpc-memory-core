"""Validation-only diagnosis of saved Experiment 1 banks; no original artifact writes.

Beta sweeps are readout ablations, not retrained online methods. Refreshed-value
replays hold admission/assignment history fixed; they are posthoc counterfactuals.
Only the official TRAIN dataset is opened. The held-out test set is not used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from mnist_online_memory import (
    EXPERIMENT_ROOT, build_baseline_net, checked_logits, extract, make_bank,
    make_corrupted_test_set,
)
from benchmarks.mnist_baseline import environment_info, load_mnist_tensors
from online_memory import centered_retrieve
from online_memory_artifacts import atomic_json, diagnostic_mode, read_events, source_identity

VARIANT_ID = "exp001_validation_readout_and_coordinate_diagnosis"
SELECTION_SETS = (("clean", 0.), ("gaussian", .4), ("salt_pepper", .15))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def independent_retrieve(query: torch.Tensor, memory: torch.Tensor, center: torch.Tensor,
                         beta: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 oracle used only to audit, never in the experiment training path."""
    q = F.normalize(query.double() - center.double(), dim=-1, eps=eps)
    m = F.normalize(memory.double() - center.double(), dim=-1, eps=eps)
    weights = (beta * q @ m.T).softmax(dim=-1)
    return weights @ memory.double(), weights


def audit_and_history(folder: Path, state: dict[str, Any], train_labels: torch.Tensor
                      ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit actual writes, assignment, class budgets, source IDs and policy flags."""
    cfg = state["config"]
    center = state["bank"]["center"]
    values: list[torch.Tensor] = []
    labels: list[int] = []
    sources = set(map(int, state["split_ids"]["train"]))
    histories: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    pending: dict[int, dict[str, Any]] = {}
    version = 0
    cached_name = None
    cached = None
    snapshots: dict[tuple[int, int], tuple[int, bool]] = {}
    for event in read_events(folder, state["log_cursor"]):
        if event["type"] == "write":
            assert event["version_before"] == version and event["version_after"] == version + 1
            source = event["example_id"]
            assert source in sources and int(train_labels[source]) == event["label"]
            reference = event["vectors"]
            if reference["file"] != cached_name:
                with np.load(folder / reference["file"], allow_pickle=False) as vectors:
                    cached = {k: vectors[k].copy() for k in ("candidate", "old", "new")}
                cached_name = reference["file"]
            row = reference["row"]
            candidate = torch.from_numpy(cached["candidate"][row])
            new = torch.from_numpy(cached["new"][row])
            entry = event["entry_id"]
            same = [i for i, c in enumerate(labels) if c == event["label"]]
            if event["action"] == "seed":
                assert not same and entry == len(values)
            else:
                keys = F.normalize(torch.stack([values[i] for i in same]) - center,
                                   dim=-1, eps=cfg["normalization_epsilon"])
                key = F.normalize(candidate - center, dim=-1, eps=cfg["normalization_epsilon"])
                distances = (1 - keys @ key).clamp(0, 2)
                chosen = same[int(distances.argmin())]
                distance = float(distances.min())
                assert abs(distance - event["distance"]) < 2e-6
                if event["action"] == "merge":
                    assert entry == chosen and distance <= event["threshold"] + 2e-6
                    torch.testing.assert_close(values[entry], torch.from_numpy(cached["old"][row]), rtol=0, atol=0)
                    expected = (1 - event["alpha"]) * values[entry] + event["alpha"] * candidate
                    torch.testing.assert_close(new, expected, rtol=1e-6, atol=1e-6)
                else:
                    assert event["action"] == "add" and distance > event["threshold"] - 2e-6
                    assert len(same) < cfg["max_memories_per_class"] and entry == len(values)
            if event["action"] != "merge":
                torch.testing.assert_close(new, candidate, rtol=0, atol=0)
                values.append(new.clone())
                labels.append(event["label"])
            else:
                values[entry] = new.clone()
            histories.append({k: event[k] for k in ("action", "entry_id", "label", "example_id", "alpha")})
            pending[event["seq"]] = event
            counts[event["action"]] += 1
            counts["local_compared"] += event["genuine_local_comparison"]
            counts["local_correct_after"] += bool(event["local_after_correct"])
            version += 1
        elif event["type"] == "example":
            counts["examples"] += 1
            counts["capacity_skips"] += event["reason"] == "class_capacity_reached"
            assert event["example_id"] in sources
            assert event["label"] == int(train_labels[event["example_id"]])
            assert event["direct_correct"] == (event["direct_prediction"] == event["label"])
            batch = (event["epoch"], event["batch"])
            snapshot = (event["snapshot_version"], event["ready"])
            assert snapshots.setdefault(batch, snapshot) == snapshot
            if event["ready"]:
                assert event["hopfield_correct"] == (event["hopfield_prediction"] == event["label"])
                category = ("harmed" if event["direct_correct"] and not event["hopfield_correct"] else
                            "helped" if event["hopfield_correct"] and not event["direct_correct"] else
                            "both_correct" if event["direct_correct"] else "both_wrong")
                assert category == event["category"]
                counts[category] += 1
                if event["action"] != "skip":
                    assert category == "harmed"
            else:
                assert event["hopfield_prediction"] is None and event["hopfield_correct"] is None
                counts["fallbacks"] += 1
            if event["write_seq"] is not None:
                write = pending.pop(event["write_seq"])
                assert write["example_id"] == event["example_id"] and write["label"] == event["label"]
                assert write["decision_snapshot_version"] == event["snapshot_version"]
                assert event["direct_correct"]
    assert not pending
    active = state["bank"]["entry_ids"] >= 0
    assert version == state["bank"]["_extra_state"]["version"]
    torch.testing.assert_close(torch.stack(values), state["bank"]["values"][active], rtol=0, atol=0)
    assert labels == state["bank"]["labels"][active].tolist()
    return dict(checks_passed=True, counts=dict(counts), writes=version), histories


@torch.no_grad()
def refreshed_history(history: list[dict[str, Any]], embeddings: dict[int, torch.Tensor]) -> torch.Tensor:
    """Same accepted sources/entries/alpha, re-encoded by ONE final encoder.

    No K-means, new admissions, new assignment or feedback decisions occur here.
    """
    values: list[torch.Tensor] = []
    for event in history:
        candidate = embeddings[event["example_id"]].detach().clone()
        entry = event["entry_id"]
        if event["action"] in ("seed", "add"):
            if entry != len(values):
                raise ValueError("Nonsequential history IDs.")
            values.append(candidate)
        else:
            values[entry] = (1 - event["alpha"]) * values[entry] + event["alpha"] * candidate
    return torch.stack(values)


@torch.no_grad()
def memory_health(head: torch.nn.Module, memory: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    logits = head(memory)
    pred = checked_logits(logits)
    margins = logits.gather(1, labels[:, None]).squeeze(1) - logits.masked_fill(
        F.one_hot(labels, logits.shape[1]).bool(), -torch.inf).max(dim=1).values
    return dict(entries=len(memory), classified_as_own_class=int((pred == labels).sum()),
                misclassified_entry_ids=(pred != labels).nonzero().flatten().tolist(),
                mean_own_class_margin=float(margins.mean()), min_own_class_margin=float(margins.min()),
                mean_raw_norm=float(memory.norm(dim=-1).mean()))


@torch.no_grad()
def readout_metrics(head: torch.nn.Module, z: torch.Tensor, y: torch.Tensor, memory: torch.Tensor,
                    memory_labels: torch.Tensor, center: torch.Tensor, beta: float,
                    eps: float, *, hard: bool = False) -> dict[str, Any]:
    if hard:
        similarities = F.normalize(z - center, dim=-1, eps=eps) @ F.normalize(memory - center, dim=-1, eps=eps).T
        weights = F.one_hot(similarities.argmax(dim=1), len(memory)).to(z.dtype)
        retrieved = weights @ memory
    else:
        retrieved, weights = centered_retrieve(z, memory, center, beta=beta, normalization_epsilon=eps)
    logits = head(retrieved)
    # An affine classifier must commute with convex averaging, including bias.
    identity_error = float((logits - weights @ head(memory)).abs().max())
    direct = checked_logits(head(z))
    pred = checked_logits(logits)
    correct = pred == y
    true_mass = (weights * (memory_labels[None] == y[:, None])).sum(dim=1)
    class_mass = weights @ F.one_hot(memory_labels, logits.shape[1]).to(weights.dtype)
    return dict(accuracy=float(correct.float().mean()), direct_accuracy=float((direct == y).float().mean()),
                helped=int(((direct != y) & correct).sum()), harmed=int(((direct == y) & ~correct).sum()),
                correct_class_mass=float(true_mass.mean()),
                class_mass_vote_accuracy=float((class_mass.argmax(1) == y).float().mean()),
                top1_mass=float(weights.max(1).values.mean()),
                entropy=float(-(weights * weights.clamp_min(1e-12).log()).sum(1).mean()),
                classifier_convex_identity_max_error=identity_error)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row["condition"], row["beta"]), []).append(row)
    return [dict(variant=variant, condition=condition, beta=beta,
                 accuracy_mean=statistics.mean(r["accuracy"] for r in values),
                 accuracy_std=statistics.stdev(r["accuracy"] for r in values) if len(values) > 1 else 0.,
                 direct_mean=statistics.mean(r["direct_accuracy"] for r in values),
                 class_mass_mean=statistics.mean(r["correct_class_mass"] for r in values),
                 seed_count=len(values)) for (variant, condition, beta), values in groups.items()]


def run_diagnosis(run_dir: Path, betas: list[float], *, seeds: list[int] | None = None) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    torch.set_num_threads(cfg["num_threads"])
    # Deliberately do not call load_mnist(), which would also open the test set.
    images, targets = load_mnist_tensors(cfg["data_dir"], train=True)
    chosen_seeds = cfg["seeds"] if seeds is None else seeds
    out = EXPERIMENT_ROOT / "results" / "diagnostics" / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = dict(variant_id=VARIANT_ID, parent_run=str(run_dir.resolve()),
                      parent_manifest_sha256=file_hash(run_dir / "manifest.json"), source=source_identity(),
                      conditions=SELECTION_SETS, beta_values=betas, seeds=chosen_seeds,
                      validation_only=True, online_retraining=False,
                      environment=environment_info(torch.device("cpu")))
    atomic_json(out / "manifest.json", provenance | {"status": "running"})
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for seed in chosen_seeds:
        folder = run_dir / f"seed_{seed}"
        latest = json.loads((folder / "latest.json").read_text())
        path = folder / latest["checkpoint"]
        before_hash = file_hash(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = build_baseline_net(seed).eval()
        model.load_state_dict(state["model"])
        active = state["bank"]["entry_ids"] >= 0
        memory, labels = state["bank"]["values"][active], state["bank"]["labels"][active]
        center = state["bank"]["center"]
        print(f"seed {seed}: auditing accepted writes and policy", flush=True)
        audit, history = audit_and_history(folder, state, targets)
        source_ids = sorted({event["example_id"] for event in history})
        current = extract(model, images[source_ids], cfg, torch.device("cpu"))
        refreshed = refreshed_history(history, dict(zip(source_ids, current)))
        reference = torch.load(folder / "fixed_reference.pt", weights_only=False, map_location="cpu")
        final_center = reference["center"]
        audit.update(seed=seed, checkpoint_sha256=before_hash,
                     original_memory_health=memory_health(model.head, memory, labels),
                     refreshed_memory_health=memory_health(model.head, refreshed, labels),
                     fixed_memory_health=memory_health(model.head, reference["values"], reference["labels"]),
                     center_shift_l2=float((center - final_center).norm()),
                     memory_refresh_l2=float((memory - refreshed).norm(dim=1).mean()))
        variants = dict(
            adaptive_saved=(memory, labels, center),
            adaptive_center_only=(memory, labels, final_center),
            adaptive_values_only=(refreshed, labels, center),
            adaptive_values_and_center=(refreshed, labels, final_center),
            fixed_kmeans=(reference["values"], reference["labels"], final_center),
        )
        val_ids = torch.as_tensor(state["split_ids"]["val"], dtype=torch.long)
        assert not set(val_ids.tolist()) & set(source_ids)
        for kind, level in SELECTION_SETS:
            condition = f"{kind}:{level:g}"
            print(f"seed {seed}: validation {condition}", flush=True)
            x = images[val_ids]
            if kind != "clean":
                x = make_corrupted_test_set(x, seed=seed, kind=kind, level=level, stream=1)
            z = extract(model, x, cfg, torch.device("cpu"))
            y = targets[val_ids]
            for name, (values, memory_labels, offset) in variants.items():
                for beta in betas:
                    metrics = readout_metrics(model.head, z, y, values, memory_labels, offset,
                                              beta, cfg["normalization_epsilon"])
                    rows.append(dict(seed=seed, variant=name, condition=condition, beta=beta, **metrics))
                hard = readout_metrics(model.head, z, y, values, memory_labels, offset,
                                       betas[0], cfg["normalization_epsilon"], hard=True)
                rows.append(dict(seed=seed, variant=name + "_hard", condition=condition, beta=-1., **hard))
            # Independent expression and classifier-affinity audits on real validation vectors.
            original, weights = centered_retrieve(z[:128], memory, center, beta=cfg["hopfield_beta"],
                                                  normalization_epsilon=cfg["normalization_epsilon"])
            oracle, oracle_weights = independent_retrieve(z[:128], memory, center,
                                                         cfg["hopfield_beta"], cfg["normalization_epsilon"])
            torch.testing.assert_close(original.double(), oracle, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(weights.double(), oracle_weights, rtol=1e-5, atol=1e-7)
        assert before_hash == file_hash(path), "Parent checkpoint was modified."
        audits.append(audit)
        atomic_json(out / "progress.json", dict(rows=rows, audits=audits))
    grouped = aggregate(rows)
    atomic_json(out / "results.json", dict(manifest=provenance, rows=rows, aggregate=grouped, audits=audits))
    lines = ["# Validation-only diagnosis of Experiment 1", "",
             "Original checkpoints, banks and test results are unchanged. No test images were opened.",
             "Beta changes below are post-training readout ablations, not retrained online experiments.",
             "Value refresh replays accepted source IDs through the final encoder with the original entry IDs/EMA history.",
             "It neither reselects candidates nor recreates online training with refresh enabled.", "",
             "| Variant | Validation condition | Beta | Mean accuracy | Sample std | Correct-class mass |",
             "|---|---|---:|---:|---:|---:|"]
    for row in grouped:
        lines.append(f"| {row['variant']} | {row['condition']} | {row['beta']:g} | "
                     f"{100*row['accuracy_mean']:.3f}% | {100*row['accuracy_std']:.3f} pp | {row['class_mass_mean']:.3f} |")
    lines += ["", "## Actual training-log audit", "", "```json", json.dumps(audits, indent=2), "```", "",
              "Changing beta in the actual online algorithm requires rerunning its admission/write history.",
              "A frozen-encoder control is a different training variant, not demonstrated by this posthoc replay."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(out / "manifest.json", provenance | {"status": "complete", "elapsed_seconds": time.perf_counter() - started})
    print(f"Diagnosis: {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--betas", nargs="+", type=float, default=[5., 20., 50., 100.])
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if not args.betas or any(not np.isfinite(beta) or beta <= 0 for beta in args.betas):
        parser.error("Betas must be positive finite numbers.")
    run_diagnosis(args.run_dir, args.betas, seeds=args.seeds)


if __name__ == "__main__":
    main()
