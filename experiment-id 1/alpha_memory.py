"""Step 2 memory operations and an independent, bounded NumPy replay audit.

The only bank optimization caches occupancy, which cannot change on a merge.
All assignment, EMA, validation, retrieval, and commit math remains inherited.
"""
from __future__ import annotations

from collections import Counter
from typing import Any
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from online_memory import OnlineHopfieldBank, correctness_categories
from online_memory_artifacts import EventWriter, read_events, timed
from mnist_online_memory import checked_logits
from hopfieldpc.diagnostics import attention_entropy

VECTOR_ENCODING = 'fp32_xor_ema_v1'


class ResidualEventWriter(EventWriter):
    """Lossless state-based float32 vector coding with per-vector CRC checks.

    Old values predict from the preceding committed entry. New values predict
    from its declared EMA. XOR residuals preserve every bit, including any
    discrepancy, rather than assuming that either predictor is correct.
    """

    def __init__(self, root, event_limit, write_limit, cursor=None, *, initial_bank=None):
        super().__init__(root,event_limit,write_limit,cursor)
        self.previous = {}
        if initial_bank is not None:
            for slot in initial_bank.active.nonzero().flatten().tolist():
                entry = int(initial_bank.entry_ids[slot])
                self.previous[entry] = initial_bank.values[slot].cpu().numpy().copy()

    def write(self, value):
        value = dict(value)
        candidate, old, new = [value.pop(k) for k in ('candidate','old','new')]
        candidate, new = candidate.numpy().copy(), new.numpy().copy()
        entry, alpha = value['entry_id'], value['alpha']
        if old is None:
            if entry in self.previous:
                raise ValueError('Addition reuses an existing logged entry.')
            old_xor, old_crc, expected = np.zeros_like(candidate,dtype=np.uint32), np.uint32(0), candidate
        else:
            if entry not in self.previous:
                raise ValueError('Merge requires its previous vector or initial checkpoint bank.')
            old = old.numpy().copy()
            old_xor = old.view(np.uint32) ^ self.previous[entry].view(np.uint32)
            old_crc = np.uint32(zlib.crc32(old.tobytes()))
            expected = old*np.float32(1-alpha)+candidate*np.float32(alpha)
        new_xor = new.view(np.uint32) ^ expected.view(np.uint32)
        reference = dict(file=self.vector_name(),row=len(self.vectors),old_present=old is not None,encoding=VECTOR_ENCODING)
        self.vectors.append(dict(candidate=candidate,old_xor=old_xor,new_xor=new_xor,
                                 old_crc32=old_crc,new_crc32=np.uint32(zlib.crc32(new.tobytes()))))
        self.previous[entry] = new.copy()
        seq = self.event(dict(type='write',**value,vectors=reference))
        if len(self.vectors) >= self.write_limit:
            self.flush_vectors()
        return seq

    def flush_vectors(self):
        if not self.vectors:
            return
        name = self.vector_name()
        path = self.root/name
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,**{k:np.stack([row[k] for row in self.vectors])
                                        for k in ('candidate','old_xor','new_xor','old_crc32','new_crc32')})
        path.with_suffix('.tmp').replace(path)
        self.vector_files.append(name)
        self.vectors.clear()
        self.vector_chunk += 1


def decode_vectors(event, vectors, previous):
    """Recover actual candidate/old/new vectors and reject reconstruction errors."""
    ref, row = event['vectors'], event['vectors']['row']
    candidate = vectors['candidate'][row]
    if ref.get('encoding') is None:
        return candidate, vectors['old'][row] if ref['old_present'] else None, vectors['new'][row]
    if ref['encoding'] != VECTOR_ENCODING:
        raise ValueError('Unsupported vector encoding.')
    old = None
    if ref['old_present']:
        old = (previous.view(np.uint32) ^ vectors['old_xor'][row]).view(np.float32)
        if zlib.crc32(old.tobytes()) != int(vectors['old_crc32'][row]):
            raise ValueError('Old-vector reconstruction checksum differs.')
    elif np.any(vectors['old_xor'][row]) or int(vectors['old_crc32'][row]):
        raise ValueError('Absent old vector has a nonzero residual.')
    expected = candidate if old is None else old*np.float32(1-event['alpha'])+candidate*np.float32(event['alpha'])
    new = (expected.view(np.uint32) ^ vectors['new_xor'][row]).view(np.float32)
    if zlib.crc32(new.tobytes()) != int(vectors['new_crc32'][row]):
        raise ValueError('New-vector reconstruction checksum differs.')
    return candidate, old, new


class CachedOccupancyBank(OnlineHopfieldBank):
    """Cache derived counts; rebuild after insertion, replay, or state loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count_size = -1
        self._counts: list[int] = []

    def occupancy(self) -> list[int]:
        if self._count_size != len(self.metadata):
            self._counts = super().occupancy()
            self._count_size = len(self.metadata)
        return list(self._counts)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        super().set_extra_state(state)
        self._count_size = -1


def make_alpha_bank(cfg, dim: int, alpha: float) -> CachedOccupancyBank:
    return CachedOccupancyBank(dim, cfg["num_classes"], cfg["max_memories_per_class"],
                               ema_alpha=alpha, beta=cfg["hopfield_beta"],
                               num_updates=cfg["hopfield_num_updates"],
                               merge_distance_threshold=cfg["merge_distance_threshold"],
                               normalization_epsilon=cfg["normalization_epsilon"])


@torch.no_grad()
def apply_bank_batch(model, bank, queries, direct, labels, example_ids, *, cfg,
                     epoch: int, batch: int, step: int, seed: int, run_id: str,
                     writer: EventWriter | None = None, costs: dict | None = None) -> dict:
    """All-direct-correct admission with pre-write decisions and ordered writes.

    Caller supplies a mode/RNG-preserving diagnostic context. Immediate local
    prediction diagnostics are explicitly unavailable; they never govern writes.
    """
    costs = costs if costs is not None else {}
    device = queries.device
    with timed(costs, "snapshot_retrieval", device):
        snapshot = bank.snapshot()
        hop = None
        if snapshot.ready:
            retrieved, weights = bank.retrieve(queries, snapshot=snapshot)
            hop = checked_logits(model.head(retrieved))
            categories = correctness_categories(direct, hop, labels)
            entropy = float(attention_entropy(weights).sum())
            top1 = float(weights.max(-1).values.sum())
        else:
            categories = ["bootstrap"] * len(labels)
            entropy = top1 = 0.
    counts = Counter(categories) if snapshot.ready else Counter()
    targets, predictions, sources = labels.tolist(), direct.tolist(), example_ids.tolist()
    hop_predictions = hop.tolist() if hop is not None else None
    for i, (label, prediction, source) in enumerate(zip(targets, predictions, sources)):
        correct = prediction == label
        common = dict(run_id=run_id, seed=seed, epoch=epoch, batch=batch,
                      optimizer_step=step, example_id=source, batch_position=i)
        action, reason, write_seq, distance = "skip", categories[i], None, None
        before_version = bank.version
        if correct:
            with timed(costs, "assignment", device):
                proposal = bank.plan_candidate(queries[i], label, bootstrap=not snapshot.ready)
            action, reason, distance = proposal.action, proposal.reason, proposal.distance
            if proposal.is_write:
                with timed(costs, "writes", device):
                    write = bank.commit(proposal, example_id=source, step=step, epoch=epoch)
                write.update(common, decision_snapshot_version=snapshot.version,
                             admission_policy="all_direct_correct", snapshot_category=categories[i],
                             local_before_prediction=None, local_after_prediction=None,
                             local_direct_prediction=prediction, local_before_correct=None,
                             local_after_correct=None, genuine_local_comparison=False,
                             local_diagnostics_available=False, fallback=not snapshot.ready)
                if writer:
                    with timed(costs, "logging", device):
                        write_seq = writer.write(write)
        elif not snapshot.ready:
            reason = "direct_incorrect"
        counts[action] += 1
        counts["reason:"+reason] += 1
        if writer:
            with timed(costs, "logging", device):
                writer.event(dict(type="example", **common, label=label, alpha=bank.alpha,
                                  ready=snapshot.ready, phase="active" if snapshot.ready else "bootstrap",
                                  direct_prediction=prediction, direct_correct=correct,
                                  hopfield_prediction=None if hop is None else hop_predictions[i],
                                  hopfield_correct=None if hop is None else hop_predictions[i] == label,
                                  fallback_prediction=prediction if hop is None else None,
                                  category=categories[i], action=action, reason=reason, distance=distance,
                                  admission_policy="all_direct_correct", allows_consideration=correct,
                                  snapshot_version=snapshot.version, working_version_before=before_version,
                                  working_version_after=bank.version, write_seq=write_seq))
    post = None
    if snapshot.ready and cfg["batch_post_write_diagnostics"]:
        with timed(costs, "batch_post_write_diagnostics", device):
            post = int((checked_logits(model.head(bank.retrieve(queries)[0])) == labels).sum())
    return dict(counts=dict(counts), examples=len(labels), compared=len(labels) if snapshot.ready else 0,
                direct_correct=int((direct == labels).sum()), hop_correct=int((hop == labels).sum()) if hop is not None else 0,
                fallback_count=0 if snapshot.ready else len(labels), entropy_sum=entropy, top1_mass_sum=top1,
                in_sample_post_write_correct=post or 0, local_diagnostics_available=False)


def audit_alpha_log(folder, cursor, cfg, expected_bank, train_ids, targets) -> dict:
    """Independent float64 assignment oracle; exact float32 EMA/vector replay.

    Near ties or threshold boundaries are checked with the original float32
    Torch expression. Only one compressed vector chunk is held in memory.
    """
    dim = expected_bank.dim
    capacity, classes = cfg["max_memories_per_class"], cfg["num_classes"]
    values = np.zeros((capacity*classes, dim), dtype=np.float32)
    labels = np.full(capacity*classes, -1, dtype=np.int64)
    keys = np.zeros_like(values, dtype=np.float64)
    center, meta, counts, n, version = None, [], Counter(), 0, 0
    allowed = set(train_ids.tolist())
    cached_name, vectors, pending, current_batch, snapshot = None, None, None, None, None
    alpha, threshold, eps = expected_bank.alpha, expected_bank.threshold, expected_bank.eps
    alpha32, old32 = np.float32(alpha), np.float32(1-alpha)
    target_array = targets.numpy()
    fallback_checks = 0
    for event in read_events(folder, cursor):
        if event["type"] == "center":
            assert center is None
            center = np.asarray(event["center"], dtype=np.float32)
            assert center.shape == (dim,) and np.isfinite(center).all()
            continue
        key = (event["epoch"], event["batch"])
        if key != current_batch:
            current_batch = key
            snapshot = (version, bool(np.all(np.bincount(labels[:n], minlength=classes) > 0)))
        if event["type"] == "write":
            assert pending is None and center is not None
            assert event["admission_policy"] == "all_direct_correct"
            assert event["decision_snapshot_version"] == snapshot[0]
            assert event["fallback"] == (not snapshot[1])
            assert event["snapshot_category"] in ("bootstrap", "harmed", "both_correct")
            source, label, slot = event["example_id"], event["label"], event["entry_id"]
            assert source in allowed and int(target_array[source]) == label
            assert 0 <= label < classes and 0 <= slot < len(labels)
            assert event["alpha"] == alpha and event["threshold"] == threshold
            assert event["version_before"] == version and event["version_after"] == version+1
            assert event["size_before"] == n
            ref = event["vectors"]
            if ref["file"] != cached_name:
                with np.load(folder/ref["file"], allow_pickle=False) as chunk:
                    names = ('candidate','old','new') if ref.get('encoding') is None else ('candidate','old_xor','new_xor','old_crc32','new_crc32')
                    vectors = {k: chunk[k].copy() for k in names}
                cached_name = ref["file"]
            row = ref["row"]
            candidate, old_vector, new = decode_vectors(event,vectors,values[slot])
            for vector in (candidate, new):
                assert vector.shape == (dim,) and vector.dtype == np.float32 and np.isfinite(vector).all()
                norm32 = np.linalg.norm(vector-center)
                assert np.isfinite(norm32) and norm32 > eps
            same = np.flatnonzero(labels[:n] == label)
            action = event["action"]
            if event["fallback"]:
                assert action == "seed" and len(same) == 0
            else:
                assert snapshot[1] and len(same)
                q = candidate.astype(np.float64)-center.astype(np.float64)
                q /= max(float(np.linalg.norm(q)), eps)
                distances = np.clip(1-keys[same] @ q, 0, 2)
                order = np.argsort(distances, kind="stable")
                nearest, distance = int(same[order[0]]), float(distances[order[0]])
                ambiguous = (len(order) > 1 and distances[order[1]]-distance < 1e-5) or abs(distance-threshold) < 1e-5
                if ambiguous:
                    native_keys = F.normalize(torch.from_numpy(values[same].copy())-torch.from_numpy(center), dim=-1, eps=eps)
                    native_q = F.normalize(torch.from_numpy(candidate)-torch.from_numpy(center), dim=-1, eps=eps)
                    native = (1-native_keys @ native_q).clamp(0, 2)
                    nearest, distance = int(same[int(native.argmin())]), float(native.min())
                    fallback_checks += 1
                assert abs(distance-event["distance"]) < 3e-6
                assert (distance <= threshold) == (action == "merge")
                if action == "merge":
                    assert slot == nearest and event["distance"] <= threshold
                else:
                    assert action == "add" and event["distance"] > threshold and len(same) < capacity
            if action == "merge":
                assert labels[slot] == label and ref["old_present"]
                assert np.array_equal(values[slot], old_vector)
                expected = values[slot]*old32 + candidate*alpha32
                assert np.array_equal(expected, new), "EMA vector differs."
                metadata = dict(meta[slot])
            else:
                assert slot == n and not ref["old_present"] and len(same) < capacity
                assert np.array_equal(candidate, new)
                metadata = dict(entry_id=slot, label=label, created_step=event["optimizer_step"],
                                created_epoch=event["epoch"], first_source_id=source, accepted_exposure_count=0)
                labels[slot] = label
                meta.append({})
                n += 1
            metadata.update(updated_step=event["optimizer_step"], updated_epoch=event["epoch"],
                            latest_source_id=source, accepted_exposure_count=metadata["accepted_exposure_count"]+1,
                            bank_version=version+1)
            assert metadata == event["metadata"]
            meta[slot] = metadata
            values[slot] = new
            centered = new.astype(np.float64)-center.astype(np.float64)
            keys[slot] = centered/max(float(np.linalg.norm(centered)), eps)
            version += 1
            assert event["size_after"] == n
            counts[action] += 1
            pending = event
        elif event["type"] == "example":
            source, label = event["example_id"], event["label"]
            assert source in allowed and int(target_array[source]) == label
            assert (event["snapshot_version"], event["ready"]) == snapshot
            assert event["direct_correct"] == (event["direct_prediction"] == label)
            assert event["allows_consideration"] == event["direct_correct"]
            assert event["working_version_after"] == version
            if event["action"] != "skip":
                assert pending is not None and event["write_seq"] == pending["seq"]
                assert pending["example_id"] == source and event["direct_correct"]
                assert pending["snapshot_category"] == event["category"]
                assert event["working_version_before"] == version-1
                pending = None
            else:
                assert pending is None and event["write_seq"] is None
                assert event["working_version_before"] == version
                if event["reason"] == "class_capacity_reached":
                    assert np.count_nonzero(labels[:n] == label) == capacity and event["distance"] > threshold
            if event["ready"]:
                direct, hop = event["direct_correct"], event["hopfield_prediction"] == label
                assert hop == event["hopfield_correct"]
                category = "both_correct" if direct and hop else "harmed" if direct else "helped" if hop else "both_wrong"
                assert event["category"] == category
                counts[category] += 1
            else:
                assert event["hopfield_prediction"] is None and event["fallback_prediction"] == event["direct_prediction"]
            counts["examples"] += 1
            counts["eligible"] += event["direct_correct"]
            counts["fallback"] += not event["ready"]
        else:
            raise ValueError("Unexpected event type.")
    assert pending is None and center is not None
    assert version == expected_bank.version and n == expected_bank.next_id
    assert np.array_equal(values, expected_bank.values.cpu().numpy())
    assert np.array_equal(labels, expected_bank.labels.cpu().numpy())
    assert np.array_equal(center, expected_bank.center.cpu().numpy())
    assert np.array_equal(expected_bank.entry_ids.cpu().numpy(), np.r_[np.arange(n), np.full(len(labels)-n, -1)])
    assert meta == expected_bank.metadata
    assert counts["examples"] == counts["fallback"] + sum(counts[k] for k in ("both_correct", "both_wrong", "helped", "harmed"))
    return dict(replay_exact=True, independent_assignment_and_ema_verified=True,
                native_boundary_checks=fallback_checks, **counts)
