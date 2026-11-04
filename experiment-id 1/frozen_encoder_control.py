"""Separate validation-only control: continue from warm-up with encoder frozen.

The same classifier continues learning by direct CE; memory uses the ORIGINAL
accuracy-gated rules. No primary runner/library edits or test-set evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
import uuid

import torch

from mnist_online_memory import (
    EXPERIMENT_ROOT, build_baseline_net, extract, make_bank, train_step,
)
from benchmarks.mnist_baseline import environment_info, load_mnist_tensors, make_corrupted_test_set
from diagnose_memory import SELECTION_SETS, file_hash, memory_health, readout_metrics
from online_memory_artifacts import (
    EventWriter, atomic_json, restore_rng, rng_state, save_checkpoint, source_identity,
)

VARIANT_ID = "exp001_control_frozen_encoder_after_warmup"


def freeze_encoder(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return {k: value.detach().cpu().clone() for k, value in model.encoder.state_dict().items()}


def verify_frozen_encoder(model: torch.nn.Module, original: dict[str, torch.Tensor]) -> None:
    for key, value in model.encoder.state_dict().items():
        if not torch.equal(value.detach().cpu(), original[key]):
            raise AssertionError(f"Frozen encoder changed: {key}")


@torch.no_grad()
def validation_conditions(model, bank, cfg, images, targets, val_ids, seed, conditions):
    results = []
    snapshot = bank.snapshot()
    for kind, level in conditions:
        x = images[val_ids]
        if kind != "clean":
            x = make_corrupted_test_set(x, seed=seed, kind=kind, level=level, stream=1)
        z = extract(model, x, cfg, torch.device("cpu"))
        y = targets[val_ids]
        if snapshot.ready:
            metrics = readout_metrics(model.head, z, y, snapshot.values, snapshot.labels,
                                      snapshot.center, cfg["hopfield_beta"], cfg["normalization_epsilon"])
        else:
            metrics = dict(accuracy=None, direct_accuracy=float((model.head(z).argmax(1) == y).float().mean()))
        results.append(dict(condition=f"{kind}:{level:g}", beta=cfg["hopfield_beta"], **metrics))
    return results


def run_control(run_dir: Path, seed: int) -> Path:
    parent_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = parent_manifest["config"]
    torch.set_num_threads(cfg["num_threads"])
    warmup = cfg["warmup_epochs"]
    if warmup < 1:
        raise ValueError("This saved-checkpoint control requires a completed warm-up epoch.")
    parent_path = run_dir / f"seed_{seed}" / "checkpoints" / f"epoch_{warmup}.pt"
    parent_hash = file_hash(parent_path)
    state = torch.load(parent_path, weights_only=False, map_location="cpu")
    if state["bank"]["_extra_state"]["version"] != 0:
        raise ValueError("Warm-up checkpoint already contains active memory writes.")
    images, targets = load_mnist_tensors(cfg["data_dir"], train=True)
    train_ids = torch.as_tensor(state["split_ids"]["train"], dtype=torch.long)
    val_ids = torch.as_tensor(state["split_ids"]["val"], dtype=torch.long)
    assert not set(train_ids.tolist()) & set(val_ids.tolist())
    model = build_baseline_net(seed)
    model.load_state_dict(state["model"])
    frozen_state = freeze_encoder(model)
    initial_head = copy.deepcopy(model.head.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    optimizer.load_state_dict(state["optimizer"])
    bank = make_bank(cfg, model.head.fc.in_features)
    bank.load_state_dict(state["bank"])
    shuffle = torch.Generator()
    shuffle.set_state(state["shuffle_rng"])
    restore_rng(state["rng"])
    out = EXPERIMENT_ROOT / "results" / "controls" / f"frozen_seed{seed}_{uuid.uuid4().hex}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(variant_id=VARIANT_ID, parent_run=str(run_dir.resolve()), seed=seed,
                    parent_checkpoint_sha256=parent_hash, source=source_identity(),
                    config=cfg, encoder_frozen_after_epoch=warmup, classifier_continues_training=True,
                    optimizer_and_shuffle_restored=True, validation_only=True, test_set_opened=False,
                    environment=environment_info(torch.device("cpu")), status="running")
    atomic_json(out / "manifest.json", manifest)
    writer = EventWriter(out, cfg["event_chunk_size"], cfg["write_chunk_size"])
    writer.event(dict(type="center", center=bank.center.tolist(), parent_checkpoint_sha256=parent_hash))
    step = state["optimizer_step"]
    history, costs = [], {}
    started = time.perf_counter()
    try:
        for epoch in range(warmup + 1, cfg["total_epochs"] + 1):
            model.train()
            permutation = torch.randperm(len(train_ids), generator=shuffle)
            counts = Counter()
            correct = compared = hop_correct = examples = 0
            weighted_loss = 0.
            print(f"frozen encoder seed {seed}: epoch {epoch}/{cfg['total_epochs']}", flush=True)
            for batch, start in enumerate(range(0, len(permutation), cfg["batch_size"])):
                source_ids = train_ids[permutation[start:start + cfg["batch_size"]]]
                result = train_step(model, bank, optimizer, images[source_ids], targets[source_ids], source_ids,
                                    epoch=epoch, batch=batch, step=step, seed=seed, run_id=out.name,
                                    cfg=cfg, writer=writer, costs=costs)
                step += 1
                examples += result["examples"]
                correct += result["direct_correct"]
                compared += result["compared"]
                hop_correct += result["hop_correct"]
                weighted_loss += result["loss"] * result["examples"]
                counts.update(result["counts"])
                if (batch + 1) % 100 == 0:
                    print(f"  batch {batch+1}, entries={len(bank.metadata)}, writes={bank.version}", flush=True)
            verify_frozen_encoder(model, frozen_state)
            final_epoch = epoch == cfg["total_epochs"]
            validation = validation_conditions(model, bank, cfg, images, targets, val_ids, seed,
                                                 SELECTION_SETS if final_epoch else SELECTION_SETS[:1])
            metrics = dict(epoch=epoch, direct_training_accuracy=correct/examples,
                           adaptive_training_accuracy=hop_correct/compared if compared else None,
                           compared=compared, examples=examples, counts=dict(counts), loss=weighted_loss/examples,
                           encoder_unchanged=True, bank=bank.storage(step), validation=validation)
            history.append(metrics)
            cursor = writer.flush()
            save_checkpoint(out / "checkpoints" / f"epoch_{epoch}.pt",
                            dict(variant_id=VARIANT_ID, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 bank=bank.state_dict(), config=cfg, epoch=epoch, optimizer_step=step,
                                 shuffle_rng=shuffle.get_state(), rng=rng_state(), log_cursor=cursor,
                                 split_ids={"train": train_ids, "val": val_ids}, history=history))
            atomic_json(out / "progress.json", dict(history=history))
        assert parent_hash == file_hash(parent_path)
        snapshot = bank.snapshot()
        health = memory_health(model.head, snapshot.values, snapshot.labels) if snapshot.ready else None
        # Optional endpoint beta sweep is explicitly a READOUT diagnostic for this control.
        beta_rows = []
        for kind, level in SELECTION_SETS:
            x = images[val_ids] if kind == "clean" else make_corrupted_test_set(
                images[val_ids], seed=seed, kind=kind, level=level, stream=1)
            z = extract(model, x, cfg, torch.device("cpu"))
            if snapshot.ready:
                for beta in (5., 20., 50., 100.):
                    row = readout_metrics(model.head, z, targets[val_ids], snapshot.values,
                                          snapshot.labels, snapshot.center, beta, cfg["normalization_epsilon"])
                    beta_rows.append(dict(condition=f"{kind}:{level:g}", beta=beta, **row))
        head_changed = any(not torch.equal(v, initial_head[k]) for k, v in model.head.state_dict().items())
        result = dict(manifest=manifest, history=history, final_validation=history[-1]["validation"],
                      final_bank_health=health, endpoint_readout_sweep=beta_rows,
                      encoder_unchanged=True, head_changed=head_changed, costs=costs,
                      elapsed_seconds=time.perf_counter()-started, log_cursor=cursor)
        atomic_json(out / "results.json", result)
        lines = ["# Frozen-encoder control (validation only)", "",
                 "Same warm-up model, optimizer, train/validation IDs and shuffle state as the primary run.",
                 "Encoder frozen after warm-up; the shared classifier continues direct-path CE training.",
                 "Bank admission/assignment/EMA/beta/capacity rules are unchanged. No test set was opened.",
                 "This is a separate control, not a replacement for Experiment 1 and not evidence across multiple seeds.", "",
                 "| Validation condition | Direct accuracy | Adaptive accuracy at training beta=5 |", "|---|---:|---:|"]
        for row in result["final_validation"]:
            acc = "unavailable" if row["accuracy"] is None else f"{row['accuracy']*100:.3f}%"
            lines.append(f"| {row['condition']} | {row['direct_accuracy']*100:.3f}% | {acc} |")
        lines += ["", f"Encoder unchanged: {True}. Classifier updated: {head_changed}.",
                  f"Bank health: `{health}`.", "", "The endpoint beta sweep in results.json is a post-training readout ablation only."]
        (out / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        atomic_json(out / "manifest.json", manifest | {"status": "complete", "elapsed_seconds": result["elapsed_seconds"]})
    except Exception as error:
        writer.flush()
        atomic_json(out / "manifest.json", manifest | {"status": "failed", "message": str(error)})
        raise
    print(f"Frozen control: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    run_control(arguments.run_dir, arguments.seed)
