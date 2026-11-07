"""Experiment 1: online accuracy-guided memory, isolated from Task 13 caches.

Eligibility uses a pre-write batch snapshot; ordered commits recompute assignment.
CNN gradients come exclusively from the direct path. See README.md in this folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# The experiment folder intentionally has the user-requested name with a space.
# Support direct execution from a checkout without changing the library package.
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.mnist_baseline import (
    MnistNet, MnistSplits, build_baseline_net, encode, environment_info, load_mnist,
    make_corrupted_test_set, make_splits as baseline_make_splits, resolve_device, robust_auc,
)
from benchmarks.mnist_prototype_retrieval import build_memory, mcnemar_test
from online_memory_artifacts import (
    EventWriter, atomic_json, diagnostic_mode, reports, restore_rng, rng_state,
    save_checkpoint, source_identity, timed,
)
from hopfieldpc.api import validate_beta
from hopfieldpc.diagnostics import attention_entropy
from online_memory import (OnlineHopfieldBank, centered_retrieve, correctness_categories,
                           positive_int, validate_normalization_epsilon)

EXPERIMENT_ID = "exp_001_online_accuracy_guided_memory"
DISPLAY_NAME = "Experiment 1 - Online Accuracy-Guided Hopfield Memory"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config.yaml"
DEFAULTS: dict[str, Any] = dict(
    experiment_id=EXPERIMENT_ID, display_name=DISPLAY_NAME, seeds=[0, 1, 2], device="auto",
    data_dir="data", reports_dir=str(EXPERIMENT_ROOT / "results"), num_classes=10,
    total_epochs=5, warmup_epochs=1, batch_size=128, learning_rate=0.001,
    val_size=5000, train_subset=None, test_subset=None, eval_batch_size=256, num_threads=4,
    max_memories_per_class=8, ema_alpha=0.1, hopfield_beta=5.0, hopfield_num_updates=1,
    score_mode="centered_cosine", write_distance="centered_cosine_distance",
    merge_distance_threshold=0.2, normalization_epsilon=1e-8, memory_refresh="none",
    bank_full_policy="skip_write", drift_probe_per_class=16, batch_post_write_diagnostics=True,
    event_chunk_size=4096, write_chunk_size=1024,
    fixed_reference=dict(prototypes_per_class=8, center_policy="post_training_train_mean",
                         beta_policy="same_fixed_beta"),
    gaussian_sigmas=[0., .1, .2, .3, .5, .7, 1.], salt_pepper_probs=[0., .05, .1, .2, .3, .4],
    occlusion_sizes=[0, 7, 10, 14],
)


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS) - {"quick"}:
        raise ValueError("Configuration must be a mapping with known Experiment 1 keys.")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(raw)
    if not isinstance(cfg.get("quick", False), bool):
        raise ValueError("Resolved quick must be boolean.")
    cfg["quick"] = cfg.get("quick", False)
    for key in ("experiment_id", "display_name", "score_mode", "write_distance", "memory_refresh", "bank_full_policy"):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError(f"Unsupported {key}: {cfg[key]!r}.")
    for key in ("num_classes", "total_epochs", "batch_size", "val_size", "eval_batch_size", "num_threads",
                "max_memories_per_class", "hopfield_num_updates", "drift_probe_per_class",
                "event_chunk_size", "write_chunk_size"):
        positive_int(key, cfg[key])
    if (isinstance(cfg["warmup_epochs"], bool) or not isinstance(cfg["warmup_epochs"], int)
            or not 0 <= cfg["warmup_epochs"] < cfg["total_epochs"]):
        raise ValueError("Require 0 <= warmup_epochs < total_epochs.")
    for key in ("train_subset", "test_subset"):
        if cfg[key] is not None:
            positive_int(key, cfg[key])
    for key in ("learning_rate", "ema_alpha", "hopfield_beta"):
        cfg[key] = validate_beta(cfg[key])
    cfg["normalization_epsilon"] = validate_normalization_epsilon(cfg["normalization_epsilon"])
    if cfg["ema_alpha"] > 1:
        raise ValueError("ema_alpha must be <= 1.")
    threshold = cfg["merge_distance_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 2:
        raise ValueError("merge_distance_threshold must be in [0,2].")
    seeds = cfg["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int)
                                                     or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative integers.")
    if not isinstance(cfg["batch_post_write_diagnostics"], bool):
        raise ValueError("batch_post_write_diagnostics must be boolean.")
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda.")
    for key in ("reports_dir", "data_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty path.")
    reference = cfg["fixed_reference"]
    if not isinstance(reference, dict) or set(reference) - set(DEFAULTS["fixed_reference"]):
        raise ValueError("Unknown fixed_reference setting.")
    reference = {**DEFAULTS["fixed_reference"], **reference}
    positive_int("fixed_reference.prototypes_per_class", reference["prototypes_per_class"])
    if (reference["center_policy"] != "post_training_train_mean"
            or reference["beta_policy"] != "same_fixed_beta"):
        raise ValueError("Only the matched-beta fixed reference is supported.")
    cfg["fixed_reference"] = reference
    for key, maximum, integer in (("gaussian_sigmas", float("inf"), False),
                                  ("salt_pepper_probs", 1, False), ("occlusion_sizes", 28, True)):
        grid = cfg[key]
        if (not isinstance(grid, list) or len(grid) < 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                       or not 0 <= v <= maximum or (integer and not isinstance(v, int)) for v in grid)
                or grid[0] != 0 or any(b <= a for a, b in zip(grid, grid[1:]))):
            raise ValueError(f"{key} must strictly increase from 0 with valid levels.")
    return cfg


def load_config(path: Path, *, quick: bool = False) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML must be a mapping.")
    overrides = raw.pop("quick", {})
    if not isinstance(overrides, dict):
        raise ValueError("quick overrides must be a mapping.")
    if quick:
        raw.update(overrides)
    raw["quick"] = quick
    return validate_config(raw)


def make_bank(cfg: dict[str, Any], dim: int, device: torch.device | str = "cpu") -> OnlineHopfieldBank:
    return OnlineHopfieldBank(dim, cfg["num_classes"], cfg["max_memories_per_class"],
                              ema_alpha=cfg["ema_alpha"], beta=cfg["hopfield_beta"],
                              num_updates=cfg["hopfield_num_updates"],
                              merge_distance_threshold=cfg["merge_distance_threshold"],
                              normalization_epsilon=cfg["normalization_epsilon"]).to(device)


def checked_logits(logits: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite classifier logits.")
    return logits.argmax(dim=-1)


def extract(model: MnistNet, images: torch.Tensor, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    with diagnostic_mode(model):
        z = encode(model.encoder, images, batch_size=cfg["eval_batch_size"], device=device)
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite extracted embeddings.")
    return z


def _local_prediction(model: MnistNet, bank: OnlineHopfieldBank, query: torch.Tensor,
                      available: bool) -> int | None:
    if not available:
        return None
    return int(checked_logits(model.head(bank.retrieve(query[None])[0]))[0].item())


def train_step(model: MnistNet, bank: OnlineHopfieldBank, optimizer: torch.optim.Optimizer,
               images: torch.Tensor, labels: torch.Tensor, example_ids: torch.Tensor, *,
               epoch: int, batch: int, step: int, seed: int, run_id: str,
               cfg: dict[str, Any], writer: EventWriter | None = None,
               costs: dict[str, float] | None = None, memory_enabled: bool = True,
               admission_policy: str = "accuracy_guided") -> dict[str, Any]:
    """One direct optimizer step, after fixed batch decisions and ordered commits."""
    if admission_policy not in ("accuracy_guided", "all_direct_correct"):
        raise ValueError(f"Unsupported admission policy: {admission_policy!r}.")
    device = next(model.parameters()).device
    costs = costs if costs is not None else {}
    optimizer.zero_grad(set_to_none=True)
    with timed(costs, "cnn_forward", device):
        z = model.encoder(images)
        direct_logits = model.head(z)
        direct = checked_logits(direct_logits).detach()
        loss = F.cross_entropy(direct_logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite direct loss or embedding.")
    counts: Counter[str] = Counter()
    class_total = torch.bincount(labels.cpu(), minlength=cfg["num_classes"])
    class_direct = torch.bincount(labels[direct == labels].cpu(), minlength=cfg["num_classes"])
    class_hop = torch.zeros(cfg["num_classes"], dtype=torch.long)
    class_compared = torch.zeros_like(class_hop)
    ready = memory_enabled and bank.ready
    stats: dict[str, Any] = dict(loss=float(loss.detach().item()), direct_correct=int((direct == labels).sum()),
                                 examples=len(labels), compared=0, hop_correct=0, fallback_count=0,
                                 entropy_sum=0., top1_mass_sum=0., local_before_correct=0,
                                 local_after_correct=0, local_compared_writes=0,
                                 class_total=class_total.tolist(), class_direct_correct=class_direct.tolist())
    if memory_enabled and bank.center_initialized:
        queries = z.detach().clone()
        with diagnostic_mode(model):
            with timed(costs, "snapshot_retrieval", device):
                snapshot = bank.snapshot()
                hop = None
                if snapshot.ready:
                    retrieved, weights = bank.retrieve(queries, snapshot=snapshot)
                    hop = checked_logits(model.head(retrieved))
                    categories = correctness_categories(direct, hop, labels)
                    counts.update(categories)
                    stats.update(compared=len(labels), hop_correct=int((hop == labels).sum()),
                                 entropy_sum=float(attention_entropy(weights).sum()),
                                 top1_mass_sum=float(weights.max(dim=-1).values.sum()))
                    class_compared = class_total.clone()
                    class_hop = torch.bincount(labels[hop == labels].cpu(), minlength=cfg["num_classes"])
                else:
                    categories = ["bootstrap"] * len(labels)
                    stats["fallback_count"] = len(labels)
            for i in range(len(labels)):
                direct_correct = bool(direct[i] == labels[i])
                common = dict(run_id=run_id, seed=seed, epoch=epoch, batch=batch,
                              optimizer_step=step, example_id=int(example_ids[i]), batch_position=i)
                action, reason, write_seq = "skip", categories[i], None
                local_before_version = bank.version
                distance = None
                eligible = (direct_correct if not snapshot.ready or admission_policy == "all_direct_correct"
                            else categories[i] == "harmed")
                if eligible:
                    with timed(costs, "assignment", device):
                        proposal = bank.plan_candidate(queries[i], int(labels[i]), bootstrap=not snapshot.ready)
                    action, reason, distance = proposal.action, proposal.reason, proposal.distance
                    if proposal.is_write:
                        with timed(costs, "local_diagnostics", device):
                            before = _local_prediction(model, bank, queries[i], snapshot.ready)
                        with timed(costs, "writes", device):
                            write = bank.commit(proposal, example_id=int(example_ids[i]), step=step, epoch=epoch)
                        with timed(costs, "local_diagnostics", device):
                            after = _local_prediction(model, bank, queries[i], snapshot.ready)
                        write.update(common, decision_snapshot_version=snapshot.version,
                                     admission_policy=admission_policy, snapshot_category=categories[i],
                                     local_before_prediction=before, local_after_prediction=after,
                                     local_direct_prediction=int(direct[i]),
                                     local_before_correct=None if before is None else before == int(labels[i]),
                                     local_after_correct=None if after is None else after == int(labels[i]),
                                     genuine_local_comparison=snapshot.ready, fallback=not snapshot.ready)
                        if snapshot.ready:
                            stats["local_compared_writes"] += 1
                            stats["local_before_correct"] += before == int(labels[i])
                            stats["local_after_correct"] += after == int(labels[i])
                        if writer:
                            with timed(costs, "logging", device):
                                write_seq = writer.write(write)
                elif not snapshot.ready:
                    reason = "direct_incorrect"
                counts[action] += 1
                counts["reason:" + reason] += 1
                if writer:
                    with timed(costs, "logging", device):
                        writer.event(dict(type="example", **common, label=int(labels[i]),
                                          ready=snapshot.ready, phase="active" if snapshot.ready else "bootstrap",
                                          direct_prediction=int(direct[i]), direct_correct=direct_correct,
                                          hopfield_prediction=None if hop is None else int(hop[i]),
                                          hopfield_correct=None if hop is None else bool(hop[i] == labels[i]),
                                          fallback_prediction=int(direct[i]) if hop is None else None,
                                          category=categories[i], action=action, reason=reason, distance=distance,
                                          admission_policy=admission_policy, allows_consideration=eligible,
                                          snapshot_version=snapshot.version,
                                          working_version_before=local_before_version,
                                          working_version_after=bank.version, write_seq=write_seq))
            if snapshot.ready and cfg["batch_post_write_diagnostics"]:
                with timed(costs, "batch_post_write_diagnostics", device):
                    post = checked_logits(model.head(bank.retrieve(queries)[0]))
                    stats["in_sample_post_write_correct"] = int((post == labels).sum())
    # No memory work consumes RNG or changes CNN tensors; all images train once.
    with timed(costs, "cnn_backward_optimizer", device):
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite CNN gradients.")
        optimizer.step()
    stats.update(counts=dict(counts), class_hop_correct=class_hop.tolist(),
                 class_compared=class_compared.tolist(), ready_at_batch_start=bool(ready))
    return stats


@torch.no_grad()
def evaluate(model: MnistNet, images: torch.Tensor, labels: torch.Tensor, bank: OnlineHopfieldBank,
             cfg: dict[str, Any], device: torch.device,
             fixed: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, Any]:
    """Read-only all-class inference. Labels enter only after predictions exist."""
    z = extract(model, images, cfg, device)
    predictions: dict[str, list[torch.Tensor]] = defaultdict(list)
    diagnostics: dict[str, list[torch.Tensor]] = defaultdict(list)
    with diagnostic_mode(model):
        snapshot = bank.snapshot()
        for start in range(0, len(z), cfg["eval_batch_size"]):
            query = z[start:start + cfg["eval_batch_size"]].to(device)
            predictions["direct"].append(checked_logits(model.head(query)).cpu())
            if snapshot.ready:
                result, weights = bank.retrieve(query, snapshot=snapshot)
                predictions["adaptive_hopfield"].append(checked_logits(model.head(result)).cpu())
                diagnostics["entropy"].append(attention_entropy(weights).cpu())
                diagnostics["top1_mass"].append(weights.max(dim=-1).values.cpu())
            if fixed is not None:
                values, center = fixed
                result, _ = centered_retrieve(query, values.to(device), center.to(device),
                                             beta=cfg["hopfield_beta"], num_updates=cfg["hopfield_num_updates"],
                                             normalization_epsilon=cfg["normalization_epsilon"])
                predictions["fixed_kmeans_reference"].append(checked_logits(model.head(result)).cpu())
    joined = {name: torch.cat(chunks) for name, chunks in predictions.items()}
    results: dict[str, Any] = {}
    for method in ("direct", "adaptive_hopfield", "fixed_kmeans_reference"):
        if method == "fixed_kmeans_reference" and fixed is None:
            continue
        pred = joined.get(method)
        if pred is None:
            results[method] = dict(accuracy=None, evaluated=0, fallback_count=len(labels),
                                   delta_vs_direct=None, mcnemar_p=None, helped=0, harmed=0,
                                   both_correct=0, both_wrong=0, per_class_accuracy=[None] * cfg["num_classes"],
                                   per_class_denominator=[0] * cfg["num_classes"], confusion=None,
                                   status="incomplete_bank")
            continue
        correct = pred == labels
        counts = Counter(correctness_categories(joined["direct"], pred, labels))
        denom = torch.bincount(labels, minlength=cfg["num_classes"])
        numer = torch.bincount(labels[correct], minlength=cfg["num_classes"])
        confusion = torch.bincount(labels * cfg["num_classes"] + pred,
                                   minlength=cfg["num_classes"] ** 2).view(cfg["num_classes"], -1)
        direct_accuracy = (joined["direct"] == labels).float().mean().item()
        accuracy = correct.float().mean().item()
        results[method] = dict(accuracy=accuracy, evaluated=len(labels), fallback_count=0,
                               delta_vs_direct=accuracy - direct_accuracy,
                               mcnemar_p=mcnemar_test(joined["direct"] == labels, correct)["p"],
                               **{key: counts[key] for key in ("helped", "harmed", "both_correct", "both_wrong")},
                               per_class_accuracy=[int(n) / int(d) if d else None for n, d in zip(numer, denom)],
                               per_class_denominator=denom.tolist(), confusion=confusion.tolist(), status="evaluated")
    if diagnostics:
        results["adaptive_hopfield"].update({k: torch.cat(v).mean().item() for k, v in diagnostics.items()})
    return results


@dataclass
class IndexedSplits(MnistSplits):
    train_indices: torch.Tensor
    val_indices: torch.Tensor
    test_indices: torch.Tensor


def splits_for(cfg: dict[str, Any], data: tuple[torch.Tensor, ...], seed: int) -> IndexedSplits:
    base = baseline_make_splits(data, seed=seed, val_size=cfg["val_size"], train_subset=cfg["train_subset"],
                                test_subset=cfg["test_subset"])
    # Recover the original IDs using the existing helper's seeded permutation.
    # The images/splits themselves still come from that unmodified helper.
    permutation = torch.randperm(len(data[1]), generator=torch.Generator().manual_seed(seed))
    train_ids = permutation[cfg["val_size"]:][:len(base.train_labels)]
    val_ids = permutation[:cfg["val_size"]]
    splits = IndexedSplits(**vars(base), train_indices=train_ids, val_indices=val_ids,
                            test_indices=torch.arange(len(base.test_labels)))
    if not torch.equal(data[0][train_ids], base.train_images) or not torch.equal(data[0][val_ids], base.val_images):
        raise ValueError("Dataset IDs do not match the library split helper.")
    for name in ("train", "val", "test"):
        images, labels = getattr(splits, name + "_images"), getattr(splits, name + "_labels")
        if (not len(labels) or len(images) != len(labels) or labels.dtype != torch.long
                or labels.min() < 0 or labels.max() >= cfg["num_classes"]
                or images.ndim != 4 or images.shape[1:] != (1, 28, 28)
                or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1):
            raise ValueError(f"Invalid {name} split.")
    counts = torch.bincount(splits.train_labels, minlength=cfg["num_classes"])
    if (counts == 0).any():
        raise ValueError("Training split is missing configured classes.")
    k = cfg["fixed_reference"]["prototypes_per_class"]
    if k > 1 and (counts <= k).any():
        raise ValueError("Fixed K-means reference needs more than k training examples per class.")
    return splits


def data_fingerprint(data: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in data:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(memoryview(tensor.contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def _center_and_probe(model: MnistNet, bank: OnlineHopfieldBank, splits: MnistSplits,
                       cfg: dict[str, Any], device: torch.device, writer: EventWriter,
                       costs: dict[str, float], step: int) -> tuple[torch.Tensor, torch.Tensor]:
    with timed(costs, "center_extraction", device):
        z = extract(model, splits.train_images, cfg, device)
        center = z.double().mean(dim=0).float()
        bank.initialize_center(center.to(device))
        probe = torch.cat([(splits.train_labels == c).nonzero().flatten()[:cfg["drift_probe_per_class"]]
                           for c in range(cfg["num_classes"])])
    with timed(costs, "logging", device):
        writer.event(dict(type="center", center=center.tolist(), optimizer_step=step,
                          probe_source_ids=splits.train_indices[probe].tolist()))
    return probe, z[probe].clone()


def train_seed(cfg: dict[str, Any], data: tuple[torch.Tensor, ...], seed: int, seed_root: Path,
               source: dict[str, str], dataset_hash: str, *, resume: bool = False,
               stop_after_epoch: int | None = None,
               model_factory: Callable[[int], MnistNet] = build_baseline_net) -> dict[str, Any] | None:
    device = resolve_device(cfg["device"])
    environment = environment_info(device)
    splits = splits_for(cfg, data, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model_factory(seed).to(device)
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
           for m in model.modules()):
        raise ValueError("Experiment 1 requires an encoder/head without Dropout or BatchNorm.")
    if model.head.fc.out_features != cfg["num_classes"]:
        raise ValueError("Classifier output size differs from configured class count.")
    bank = make_bank(cfg, model.head.fc.in_features, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    shuffle = torch.Generator().manual_seed(seed)
    costs: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    probe = reference = previous_probe = None
    epoch_start, step, cursor = 1, 0, None
    seed_root.mkdir(parents=True, exist_ok=True)
    split_ids = {name: getattr(splits, name + "_indices").numpy() for name in ("train", "val", "test")}
    latest = seed_root / "latest.json"
    if resume and latest.exists():
        path = seed_root / json.loads(latest.read_text())["checkpoint"]
        # Checkpoints are local, trusted artifacts containing Python/NumPy RNG state.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state["source"] != source or state["dataset_hash"] != dataset_hash:
            raise ValueError("Resume config/source/data fingerprint mismatch.")
        if any(state["environment"][key] != environment[key] for key in ("device", "torch_version", "num_threads")):
            raise ValueError("Resume device/PyTorch/thread environment mismatch.")
        for name, ids in split_ids.items():
            if not np.array_equal(ids, state["split_ids"][name]):
                raise ValueError("Resume split IDs mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        bank.load_state_dict(state["bank"])
        shuffle.set_state(state["shuffle_rng"])
        restore_rng(state["rng"])
        epoch_start, step = state["epoch"] + 1, state["optimizer_step"]
        probe, reference, previous_probe = state["probe"], state["probe_reference"], state["previous_probe"]
        history, costs, cursor = state["history"], state["costs"], state["log_cursor"]
    else:
        np.savez_compressed(seed_root / "split_ids.npz", **split_ids)
    writer = EventWriter(seed_root, cfg["event_chunk_size"], cfg["write_chunk_size"], cursor)
    try:
        if cfg["warmup_epochs"] == 0 and not bank.center_initialized:
            probe, reference = _center_and_probe(model, bank, splits, cfg, device, writer, costs, step)
            previous_probe = reference.clone()
        for epoch in range(epoch_start, cfg["total_epochs"] + 1):
            print(f"seed {seed}: epoch {epoch}/{cfg['total_epochs']} ({'active' if bank.ready else 'bootstrap' if bank.center_initialized else 'warmup'})", flush=True)
            model.train()
            permutation = torch.randperm(len(splits.train_labels), generator=shuffle)
            # Only scalar/per-class aggregates are retained, never event vectors.
            totals: Counter[str] = Counter()
            categories: Counter[str] = Counter()
            class_totals = {key: np.zeros(cfg["num_classes"], dtype=np.int64)
                            for key in ("class_total", "class_direct_correct", "class_hop_correct", "class_compared")}
            weighted_loss = 0.
            start_epoch = time.perf_counter()
            for batch, start in enumerate(range(0, len(permutation), cfg["batch_size"])):
                idx = permutation[start:start + cfg["batch_size"]]
                result = train_step(model, bank, optimizer, splits.train_images[idx].to(device),
                                    splits.train_labels[idx].to(device), splits.train_indices[idx],
                                    epoch=epoch, batch=batch, step=step, seed=seed, run_id=seed_root.parent.name,
                                    cfg=cfg, writer=writer, costs=costs)
                step += 1
                weighted_loss += result["loss"] * result["examples"]
                categories.update(result["counts"])
                for key in class_totals:
                    class_totals[key] += result[key]
                for key in ("examples", "direct_correct", "compared", "hop_correct", "fallback_count",
                            "entropy_sum", "top1_mass_sum", "local_before_correct", "local_after_correct",
                            "local_compared_writes", "in_sample_post_write_correct"):
                    totals[key] += result.get(key, 0)
                if (batch + 1) % 50 == 0:
                    batch_count = (len(permutation) + cfg["batch_size"] - 1) // cfg["batch_size"]
                    print(f"  seed {seed}, epoch {epoch}: batch {batch + 1}/{batch_count}, "
                          f"loss={weighted_loss / totals['examples']:.4f}, "
                          f"memories={len(bank.metadata)}, ready={bank.ready}", flush=True)
            costs["training_wall"] = costs.get("training_wall", 0.) + time.perf_counter() - start_epoch
            if epoch == cfg["warmup_epochs"]:
                probe, reference = _center_and_probe(model, bank, splits, cfg, device, writer, costs, step)
                previous_probe = reference.clone()
            drift = None
            if probe is not None:
                with timed(costs, "drift_probe", device):
                    current = extract(model, splits.train_images[probe], cfg, device)
                    drift = dict(from_warmup_l2=(current - reference).norm(dim=-1).mean().item(),
                                 from_previous_epoch_l2=(current - previous_probe).norm(dim=-1).mean().item(),
                                 source_ids=splits.train_indices[probe].tolist())
                    previous_probe = current.clone()
            with timed(costs, "validation", device):
                validation = evaluate(model, splits.val_images, splits.val_labels, bank, cfg, device)
            compared = totals["compared"]
            metrics = dict(epoch=epoch, optimizer_step=step,
                           phase="active" if bank.ready else "bootstrap" if bank.center_initialized else "warmup",
                           loss=weighted_loss / totals["examples"], counts=dict(categories),
                           totals=dict(totals), direct_accuracy=totals["direct_correct"] / totals["examples"],
                           hopfield_accuracy=totals["hop_correct"] / compared if compared else None,
                           paired_direct_accuracy=(categories["harmed"] + categories["both_correct"]) / compared if compared else None,
                           accuracy_delta=(categories["helped"] - categories["harmed"]) / compared if compared else None,
                           entropy=totals["entropy_sum"] / compared if compared else None,
                           top1_mass=totals["top1_mass_sum"] / compared if compared else None,
                           per_class_counts={k: v.tolist() for k, v in class_totals.items()},
                           per_class_direct_accuracy=[int(n) / int(d) if d else None for n, d in
                                                      zip(class_totals["class_direct_correct"], class_totals["class_total"])],
                           per_class_hopfield_accuracy=[int(n) / int(d) if d else None for n, d in
                                                        zip(class_totals["class_hop_correct"], class_totals["class_compared"])],
                           bank=bank.storage(step), drift=drift, validation=validation)
            history.append(metrics)
            with timed(costs, "logging", device):
                cursor = writer.flush()
            checkpoint = dict(config=cfg, source=source, environment=environment, dataset_hash=dataset_hash,
                              split_ids=split_ids, epoch=epoch, optimizer_step=step,
                              model=model.state_dict(), optimizer=optimizer.state_dict(), bank=bank.state_dict(),
                              rng=rng_state(), shuffle_rng=shuffle.get_state(),
                              probe=probe, probe_reference=reference, previous_probe=previous_probe,
                              history=history, costs=costs, log_cursor=cursor,
                              phase="active" if bank.ready else "bootstrap" if bank.center_initialized else "warmup")
            relative = f"checkpoints/epoch_{epoch}.pt"
            save_checkpoint(seed_root / relative, checkpoint)
            atomic_json(latest, dict(checkpoint=relative, epoch=epoch, log_cursor=cursor))
            (seed_root / "metrics.jsonl").write_text("".join(json.dumps(m, allow_nan=False) + "\n" for m in history), encoding="utf-8")
            if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < cfg["total_epochs"]:
                return None
        # A post-training construction, never a replacement for online history.
        with timed(costs, "fixed_reference_extraction", device):
            z_train = extract(model, splits.train_images, cfg, device)
        with timed(costs, "fixed_reference_construction", device):
            try:
                values, labels, center = build_memory(
                    z_train, splits.train_labels, per_class=cfg["fixed_reference"]["prototypes_per_class"],
                    center=True, seed=seed, num_classes=cfg["num_classes"],
                )
            except (ValueError, RuntimeError) as error:
                raise FloatingPointError(f"Fixed reference construction failed: {error}") from error
            # Validate keys before test evaluation; report failures rather than repairing memory.
            centered_retrieve(z_train[:1], values, center, beta=cfg["hopfield_beta"],
                              num_updates=cfg["hopfield_num_updates"],
                              normalization_epsilon=cfg["normalization_epsilon"])
            save_checkpoint(seed_root / "fixed_reference.pt", dict(values=values, labels=labels, center=center,
                                                                   center_policy="post_training_train_mean"))
        rows = []
        auc: dict[str, Any] = {}
        with timed(costs, "test_evaluation", device):
            for kind, key in (("gaussian", "gaussian_sigmas"), ("salt_pepper", "salt_pepper_probs"),
                              ("occlusion", "occlusion_sizes")):
                for level in cfg[key]:
                    images = make_corrupted_test_set(splits.test_images, seed=seed, kind=kind, level=level)
                    evaluated = evaluate(model, images, splits.test_labels, bank, cfg, device, (values, center))
                    rows += [dict(method=method, kind=kind, level=level, **metrics)
                             for method, metrics in evaluated.items()]
                for method in ("direct", "adaptive_hopfield", "fixed_kmeans_reference"):
                    points = [r for r in rows if r["kind"] == kind and r["method"] == method]
                    auc[f"{method}:{kind}"] = (robust_auc(cfg[key], [r["accuracy"] for r in points])
                                                if all(r["accuracy"] is not None for r in points) else None)
        def disk_size(pattern: str) -> int:
            return sum(p.stat().st_size for p in seed_root.glob(pattern) if p.is_file())
        summary = dict(seed=seed, epochs=history, test_results=rows, robust_auc=auc,
                       bank_storage=bank.storage(step), fixed_bank_count=len(values),
                       fixed_bank_storage=dict(vector_bytes=values.numel() * values.element_size(),
                                               label_bytes=labels.numel() * labels.element_size(),
                                               center_bytes=center.numel() * center.element_size()),
                       centering=dict(adaptive="warmup_train_mean_frozen", fixed="post_training_train_mean"),
                       costs=costs, log_cursor=cursor,
                       disk_storage=dict(checkpoints_bytes=disk_size("checkpoints/*"),
                                         logs_bytes=disk_size("events/*") + disk_size("writes/*"),
                                         fixed_reference_bytes=(seed_root / "fixed_reference.pt").stat().st_size))
        atomic_json(seed_root / "seed_results.json", summary)
        return summary
    except Exception as error:
        failure_cursor = writer.flush()
        atomic_json(seed_root / "failure.json", dict(error=type(error).__name__, message=str(error),
                                                    optimizer_step=step, numerical_failure=isinstance(error, FloatingPointError),
                                                    uncommitted_log_cursor=failure_cursor))
        raise


def run_experiment(cfg: dict[str, Any], *, resume_root: Path | None = None,
                   data: tuple[torch.Tensor, ...] | None = None,
                   stop_after_epoch: int | None = None,
                   model_factory: Callable[[int], MnistNet] = build_baseline_net) -> Path:
    cfg = validate_config(cfg)
    torch.set_num_threads(cfg["num_threads"])
    source = source_identity()
    dataset = data if data is not None else load_mnist(cfg["data_dir"])
    dataset_hash = data_fingerprint(dataset)
    # Preflight every split before any seed consumes training resources.
    for seed in cfg["seeds"]:
        splits_for(cfg, dataset, seed)
    if resume_root is None:
        root = Path(cfg["reports_dir"]) / ("quick" if cfg["quick"] else "full") / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        (root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        manifest = dict(source=source, dataset_hash=dataset_hash, config=cfg,
                        environment=environment_info(resolve_device(cfg["device"])), status="training")
        atomic_json(root / "manifest.json", manifest)
    else:
        root = resume_root
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["source"] != source or manifest["dataset_hash"] != dataset_hash or manifest["config"] != cfg:
            raise ValueError("Resume source/config/data mismatch.")
        current_environment = environment_info(resolve_device(cfg["device"]))
        if any(manifest["environment"][key] != current_environment[key]
               for key in ("device", "torch_version", "num_threads")):
            raise ValueError("Resume device/PyTorch/thread environment mismatch.")
        manifest["status"] = "training"
        manifest.pop("error", None)
        manifest.pop("message", None)
        atomic_json(root / "manifest.json", manifest)
    print(f"{DISPLAY_NAME}\nOutput: {root}", flush=True)
    summaries = []
    try:
        for seed in cfg["seeds"]:
            seed_root = root / f"seed_{seed}"
            completed = seed_root / "seed_results.json"
            if resume_root is not None and completed.exists():
                summary = json.loads(completed.read_text())
            else:
                summary = train_seed(cfg, dataset, seed, seed_root, source, dataset_hash,
                                     resume=resume_root is not None, stop_after_epoch=stop_after_epoch,
                                     model_factory=model_factory)
            if summary is None:
                manifest["status"] = "paused_at_epoch_boundary"
                atomic_json(root / "manifest.json", manifest)
                return root
            summaries.append(summary)
        report_start = time.perf_counter()
        reports(root, cfg, summaries)
        atomic_json(root / "report_cost.json", {"report_generation_seconds": time.perf_counter() - report_start})
        manifest["status"] = "complete"
        atomic_json(root / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=type(error).__name__, message=str(error))
        atomic_json(root / "manifest.json", manifest)
        raise
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=DISPLAY_NAME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume a trusted local run at its committed epoch boundary.")
    args = parser.parse_args(argv)
    if args.resume:
        if args.quick or args.config != DEFAULT_CONFIG:
            parser.error("--resume loads its own resolved config; do not combine with --quick/--config.")
        cfg = yaml.safe_load((args.resume / "resolved_config.yaml").read_text(encoding="utf-8"))
    else:
        cfg = load_config(args.config, quick=args.quick)
    root = run_experiment(cfg, resume_root=args.resume)
    print(f"Reports: {root}")


if __name__ == "__main__":
    main()
