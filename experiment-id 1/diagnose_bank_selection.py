"""Posthoc bank-construction controls in ONE frozen coordinate system.

Same frozen CNN/head, train/validation IDs, center, per-class slot counts and beta.
This is not the during-training all-correct admission ablation: the static stream
uses the endpoint classifier and never optimizes CNN weights or reroutes history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import uuid

import torch

from mnist_online_memory import EXPERIMENT_ROOT, build_baseline_net, extract, make_bank
from benchmarks.mnist_baseline import load_mnist_tensors, make_corrupted_test_set
from benchmarks.mnist_prototype_retrieval import kmeans
from diagnose_memory import SELECTION_SETS, file_hash, memory_health, readout_metrics
from online_memory_artifacts import EventWriter, atomic_json, save_checkpoint, source_identity

VARIANT_ID = "exp001_posthoc_frozen_space_matched_budget_bank_selection"


def cluster_with_quotas(z: torch.Tensor, labels: torch.Tensor, quotas: list[int], seed: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    memories, memory_labels = [], []
    for label, count in enumerate(quotas):
        candidates = z[labels == label]
        block = candidates.mean(0, keepdim=True) if count == 1 else kmeans(candidates, count, generator=generator)
        memories.append(block)
        memory_labels.append(torch.full((count,), label, dtype=torch.long))
    return torch.cat(memories), torch.cat(memory_labels)


def run_comparison(control: Path) -> Path:
    manifest = json.loads((control / "manifest.json").read_text())
    cfg, seed = manifest["config"], manifest["seed"]
    torch.set_num_threads(cfg["num_threads"])
    checkpoint = control / "checkpoints" / f"epoch_{cfg['total_epochs']}.pt"
    checkpoint_hash = file_hash(checkpoint)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model = build_baseline_net(seed).eval()
    model.load_state_dict(state["model"])
    bank = make_bank(cfg, model.head.fc.in_features)
    bank.load_state_dict(state["bank"])
    saved = bank.snapshot()
    quotas = bank.occupancy()
    if not saved.ready:
        raise ValueError("The parent bank is incomplete.")
    images, targets = load_mnist_tensors(cfg["data_dir"], train=True)
    train_ids, val_ids = state["split_ids"]["train"], state["split_ids"]["val"]
    assert not set(train_ids.tolist()) & set(val_ids.tolist())
    out = EXPERIMENT_ROOT / "results" / "diagnostics" / f"bank_selection_{uuid.uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(variant_id=VARIANT_ID, parent_control=str(control.resolve()),
                      parent_checkpoint_sha256=checkpoint_hash, source=source_identity(),
                      quotas=quotas, seed=seed, validation_only=True, online_training=False,
                      note="Static endpoint construction; not a retrained all-correct online policy.")
    atomic_json(out / "manifest.json", provenance | {"status": "running"})
    print("Matched-bank control: extracting frozen training features", flush=True)
    z = extract(model, images[train_ids], cfg, torch.device("cpu"))
    fresh_center = z.double().mean(0).float()
    center_difference = float((saved.center - fresh_center).norm())
    torch.testing.assert_close(saved.center, fresh_center, rtol=1e-5, atol=1e-5)
    print("Matched-bank control: K-means with identical per-class slot counts", flush=True)
    clustered, cluster_labels = cluster_with_quotas(z, targets[train_ids], quotas, seed)
    print("Matched-bank control: static stream of every directly correct candidate", flush=True)
    stream = make_bank(cfg, model.head.fc.in_features)
    stream.initialize_center(saved.center)
    writer = EventWriter(out, cfg["event_chunk_size"], cfg["write_chunk_size"])
    writer.event(dict(type="center", center=saved.center.tolist()))
    with torch.no_grad():
        direct_correct = model.head(z).argmax(1) == targets[train_ids]
    order = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))
    counts = Counter()
    for start in range(0, len(order), cfg["batch_size"]):
        ready = stream.ready
        for position, index in enumerate(order[start:start + cfg["batch_size"]].tolist()):
            if not direct_correct[index]:
                counts["direct_incorrect"] += 1
                continue
            label = int(targets[train_ids[index]])
            proposal = stream.plan_candidate(z[index], label, bootstrap=not ready)
            if proposal.action in ("seed", "add") and stream.occupancy()[label] >= quotas[label]:
                counts["matched_quota_reached"] += 1
            elif proposal.is_write:
                event = stream.commit(proposal, example_id=int(train_ids[index]), step=start + position, epoch=0)
                event.update(example_id=int(train_ids[index]), construction_only=True)
                writer.write(event)
                counts[proposal.action] += 1
            else:
                counts[proposal.reason] += 1
    cursor = writer.flush()
    actual = stream.snapshot()
    assert stream.occupancy() == quotas
    variants = dict(adaptive_during_training=(saved.values, saved.labels),
                    kmeans_matched_slots=(clustered, cluster_labels),
                    all_correct_static_stream=(actual.values, actual.labels))
    health = {name: memory_health(model.head, values, labels) for name, (values, labels) in variants.items()}
    rows = []
    for kind, level in SELECTION_SETS:
        print(f"Matched-bank validation: {kind}:{level:g}", flush=True)
        x = images[val_ids]
        if kind != "clean":
            x = make_corrupted_test_set(x, seed=seed, kind=kind, level=level, stream=1)
        query = extract(model, x, cfg, torch.device("cpu"))
        for name, (values, labels) in variants.items():
            for beta in (5., 20., 50., 100.):
                metrics = readout_metrics(model.head, query, targets[val_ids], values, labels,
                                          saved.center, beta, cfg["normalization_epsilon"])
                rows.append(dict(variant=name, condition=f"{kind}:{level:g}", beta=beta, **metrics))
    assert checkpoint_hash == file_hash(checkpoint)
    save_checkpoint(out / "constructed_banks.pt", dict(stream=stream.state_dict(), kmeans=clustered,
                                                       kmeans_labels=cluster_labels, center=saved.center))
    atomic_json(out / "results.json", dict(manifest=provenance, rows=rows, health=health,
                                           stream_counts=dict(counts), log_cursor=cursor,
                                           center_difference_l2=center_difference))
    lines = ["# Matched bank-construction diagnosis in frozen space", "",
             "Same final frozen-control CNN, classifier, train/validation data, center and per-class slot counts.",
             "This is an endpoint construction diagnostic, not an online all-correct retraining experiment.",
             "No test images were opened. No parent checkpoint or bank was modified.", "",
             "| Bank | Validation condition | Beta | Accuracy | Direct accuracy |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['condition']} | {row['beta']:g} | {100*row['accuracy']:.3f}% | {100*row['direct_accuracy']:.3f}% |")
    lines += ["", "## Stored-entry health", "", "```json", json.dumps(health, indent=2), "```"]
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    atomic_json(out / "manifest.json", provenance | {"status": "complete"})
    print(f"Matched-bank comparison: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-control", required=True, type=Path)
    run_comparison(parser.parse_args().frozen_control)
