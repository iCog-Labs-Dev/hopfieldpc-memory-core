"""Streaming provenance, epoch checkpoints and reports for Experiment 1."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterator
import uuid

import numpy as np
import torch

from online_memory import OnlineHopfieldBank


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()
    # Include new implementation files as well as tracked edits. Output/data
    # directories are gitignored and do not affect resumability.
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode().split("\0")
    digest = hashlib.sha256()
    for name in sorted(set(files) - {""}):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return {"commit": git("rev-parse", "HEAD"), "source_sha256": digest.hexdigest()}


def rng_state() -> dict[str, Any]:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def diagnostic_mode(model: torch.nn.Module) -> Iterator[None]:
    """Restore all module modes, buffers and RNG streams after read-only work."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    state = rng_state()
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, previous in buffers:
                buffer.copy_(previous)
        for module, training in modes:
            module.training = training
        restore_rng(state)


@contextmanager
def timed(costs: dict[str, float], name: str, device: torch.device) -> Iterator[None]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        costs[name] = costs.get(name, 0.0) + time.perf_counter() - start


class EventWriter:
    """Bounded compressed chunks. Only checkpoint-listed chunks are committed.

    New sessions use unique names; uncommitted crash tails are never overwritten
    or mixed into resumed logs. Tensor chunks retain float32 values losslessly.
    """
    def __init__(self, root: Path, event_limit: int, write_limit: int,
                 cursor: dict[str, Any] | None = None):
        self.root = root
        for directory in ("events", "writes"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        self.event_limit, self.write_limit = event_limit, write_limit
        self.session = uuid.uuid4().hex
        self.events: list[dict[str, Any]] = []
        self.vectors: list[dict[str, np.ndarray]] = []
        self.event_files = list((cursor or {}).get("event_files", []))
        self.vector_files = list((cursor or {}).get("vector_files", []))
        self.seq = int((cursor or {}).get("seq", 0))
        self.event_chunk = self.vector_chunk = 0

    def vector_name(self) -> str:
        return f"writes/chunk_{self.session}_{self.vector_chunk:06d}.npz"

    def event(self, value: dict[str, Any]) -> int:
        seq = self.seq
        self.seq += 1
        self.events.append({"seq": seq, **value})
        if len(self.events) >= self.event_limit:
            self.flush_events()
        return seq

    def write(self, value: dict[str, Any]) -> int:
        value = dict(value)
        candidate, old, new = (value.pop(k) for k in ("candidate", "old", "new"))
        reference = {"file": self.vector_name(), "row": len(self.vectors), "old_present": old is not None}
        self.vectors.append(dict(candidate=candidate.numpy().copy(), new=new.numpy().copy(),
                                 old=old.numpy().copy() if old is not None else np.zeros_like(new.numpy())))
        seq = self.event({"type": "write", **value, "vectors": reference})
        if len(self.vectors) >= self.write_limit:
            self.flush_vectors()
        return seq

    def flush_events(self) -> None:
        if not self.events:
            return
        name = f"events/chunk_{self.session}_{self.event_chunk:06d}.jsonl.gz"
        path = self.root / name
        with gzip.open(path.with_suffix(".tmp"), "wt", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        path.with_suffix(".tmp").replace(path)
        self.event_files.append(name)
        self.events.clear()
        self.event_chunk += 1

    def flush_vectors(self) -> None:
        if not self.vectors:
            return
        name = self.vector_name()
        path = self.root / name
        with open(path.with_suffix(".tmp"), "wb") as stream:
            np.savez_compressed(stream, **{k: np.stack([v[k] for v in self.vectors])
                                          for k in ("candidate", "old", "new")})
        path.with_suffix(".tmp").replace(path)
        self.vector_files.append(name)
        self.vectors.clear()
        self.vector_chunk += 1

    def flush(self) -> dict[str, Any]:
        self.flush_vectors()
        self.flush_events()
        return dict(seq=self.seq, event_files=list(self.event_files), vector_files=list(self.vector_files))


def read_events(root: Path, cursor: dict[str, Any]) -> Iterator[dict[str, Any]]:
    expected = 0
    for name in cursor["event_files"]:
        with gzip.open(root / name, "rt", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["seq"] != expected:
                    raise ValueError("Noncontiguous event sequence.")
                expected += 1
                yield event
    if expected != cursor["seq"]:
        raise ValueError("Checkpoint event cursor does not match log.")


@torch.no_grad()
def replay_bank(bank: OnlineHopfieldBank, root: Path, cursor: dict[str, Any]) -> OnlineHopfieldBank:
    """Replay losslessly from an empty bank, validating versions and old values."""
    for event in read_events(root, cursor):
        if event["type"] == "center":
            bank.initialize_center(torch.tensor(event["center"]))
        elif event["type"] == "write":
            if event["version_before"] != bank.version or event["version_after"] != bank.version + 1:
                raise ValueError("Replay bank version mismatch.")
            with np.load(root / event["vectors"]["file"], allow_pickle=False) as data:
                row = event["vectors"]["row"]
                old, new = torch.from_numpy(data["old"][row]), torch.from_numpy(data["new"][row])
                slot = event["entry_id"]
                if event["action"] == "merge":
                    if not torch.equal(bank.values[slot].cpu(), old):
                        raise ValueError("Replay old value mismatch.")
                else:
                    if slot != bank.next_id or bank.active[slot]:
                        raise ValueError("Replay entry order mismatch.")
                    bank.next_id += 1
                    bank.metadata.append({})
                    bank.labels[slot] = event["label"]
                    bank.entry_ids[slot] = slot
                if bank._invalid_key(new.to(bank.values)):
                    raise ValueError("Replay invalid key.")
                bank.values[slot].copy_(new)
                bank.metadata[slot] = dict(event["metadata"])
                bank.version = event["version_after"]
    return bank


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def reports(root: Path, cfg: dict[str, Any], seeds: list[dict[str, Any]]) -> None:
    """Preserve negative results; held-out metrics and training diagnostics differ."""
    import csv
    rows = [dict(seed=s["seed"], **row) for s in seeds for row in s["test_results"]]
    atomic_json(root / "results.json", {"config": cfg, "seeds": seeds})
    with open(root / "results.csv", "w", newline="", encoding="utf-8") as stream:
        fields = ["seed", "method", "kind", "level", "accuracy", "evaluated", "fallback_count",
                  "delta_vs_direct", "mcnemar_p", "helped", "harmed", "both_correct", "both_wrong"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# " + cfg["display_name"], "", f"Experiment ID: `{cfg['experiment_id']}`.", "",
             "**Quick smoke test; not performance evidence.**" if cfg["quick"] else "Full fixed-config run.",
             "", "## Held-out test results", "",
             "| Seed | Method | Corruption | Level | Accuracy | Δ vs direct | Evaluated | Fallback |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        acc = "unavailable" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
        delta = "—" if r["delta_vs_direct"] is None else f"{r['delta_vs_direct']:+.4f}"
        lines.append(f"| {r['seed']} | {r['method']} | {r['kind']} | {r['level']:g} | {acc} | {delta} |"
                     f" {r['evaluated']} | {r['fallback_count']} |")
    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        if row["accuracy"] is not None:
            grouped[(row["method"], row["kind"], row["level"])].append(row["accuracy"])
    lines += ["", "## Across-seed descriptive accuracy", "",
              "| Method | Corruption | Level | Mean | Sample std | Available seeds |",
              "|---|---|---:|---:|---:|---:|"]
    for (method, kind, level), values in grouped.items():
        a = np.asarray(values)
        std = a.std(ddof=1) if len(a) > 1 else 0.0
        lines.append(f"| {method} | {kind} | {level:g} | {a.mean():.4f} | {std:.4f} | {len(a)} |")
    lines += ["", "## Bank state, costs and interpretation", "",
              "Both banks use the same fixed beta, update count and epsilon. The adaptive center is frozen at"
              " warm-up; the fixed reference uses its post-training train mean. No validation tuning is run.", ""]
    for seed in seeds:
        lines += [f"### Seed {seed['seed']}", "", f"Adaptive bank: `{seed['bank_storage']}`.", "",
                  f"Centering policies: `{seed['centering']}`.", "",
                  f"Fixed bank entries: {seed['fixed_bank_count']}.", "",
                  f"Fixed bank tensor storage: `{seed['fixed_bank_storage']}`.", "",
                  f"Robustness AUC: `{seed['robust_auc']}`.", "",
                  f"Wall costs (seconds): `{seed['costs']}`.", "",
                  f"Storage: `{seed['disk_storage']}`.", ""]
    lines += ["Stored embeddings and the adaptive center can become stale as the CNN changes."
              " Occupancy can differ between banks. Training write diagnostics are in-sample and do not"
              " establish generalization. Per-image McNemar results and per-seed deltas are descriptive;"
              " three seeds do not establish strong across-seed evidence. Negative deltas are retained.", "",
              "## Training/validation diagnostics", "",
              "Per-epoch metrics include direct training accuracy, genuine pre-write Hopfield accuracy,"
              " fallback denominators, local updates, validation accuracy, occupancy and drift."
              " See each seed's metrics.jsonl and plots; validation results are not test results."]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_plots(root, seeds)


def make_plots(root: Path, seeds: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = root / "plots"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for seed in seeds:
        metrics = seed["epochs"]
        xs = [m["epoch"] for m in metrics]
        label = f"seed {seed['seed']}"
        axes[0].plot(xs, [m["validation"]["direct"]["accuracy"] for m in metrics], label=label + " direct")
        axes[0].plot(xs, [m["validation"]["adaptive_hopfield"]["accuracy"] for m in metrics],
                     linestyle="--", label=label + " adaptive")
        axes[1].plot(xs, [m["counts"].get("helped", 0) for m in metrics], label=label + " helped")
        axes[1].plot(xs, [m["counts"].get("harmed", 0) for m in metrics], linestyle="--", label=label + " harmed")
        axes[2].plot(xs, [sum(m["bank"]["occupancy"]) for m in metrics], label=label)
    for ax, title in zip(axes, ("Held-out validation accuracy", "In-sample pre-write counts", "Adaptive bank growth")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, kind in zip(axes, ("gaussian", "salt_pepper", "occlusion")):
        for method in ("direct", "adaptive_hopfield", "fixed_kmeans_reference"):
            points = [(key[2], np.mean([r["accuracy"] for s in seeds for r in s["test_results"]
                                       if r["method"] == method and r["kind"] == kind and r["level"] == key[2]
                                       and r["accuracy"] is not None]))
                      for key in sorted({(r["method"], r["kind"], r["level"]) for s in seeds
                                         for r in s["test_results"] if r["method"] == method
                                         and r["kind"] == kind and r["accuracy"] is not None})]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=method)
        ax.set_title("Test: " + kind)
        ax.set_xlabel("corruption level")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "held_out.png")
    plt.close(fig)
