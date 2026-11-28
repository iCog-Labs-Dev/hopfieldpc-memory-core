"""Reference GASS writes, fixed-capacity storage and structural proposals."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from hopfieldpc.diagnostics import kish_effective_sample_size
from hopfieldpc.ema import ema_step
from hopfieldpc.functional import read_from_logits, squared_distance
from hopfieldpc.kalman import condition_position_velocity, fractional_measurement_factors, predict_position_velocity

from .config import ReferenceConfig
from .geometry import KeyGeometry
from .loss import RetrievalLoss


@dataclass
class Components:
    """A structural proposal owns its arrays. Positive mixture mass marks active slots."""

    keys: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor

    def clone(self):
        return Components(self.keys.clone(), self.values.clone(), self.weights.clone())

    def clear(self, slot):
        self.keys[slot].zero_()
        self.values[slot].zero_()
        self.weights[slot] = 0


@dataclass
class Edit:
    name: str
    layout: Components
    changed: tuple[int, ...]


class ReferenceBank(nn.Module):
    """Continuous associative memory with model-supplied keys and retrieval loss."""

    def __init__(self, dim, initial, minimum, maximum, config=None, *, key_dim=None,
                 geometry: KeyGeometry, loss: RetrievalLoss):
        super().__init__()
        if not 1 <= minimum <= initial <= maximum:
            raise ValueError("Require 1 <= minimum <= initial <= maximum.")
        self.cfg = config or ReferenceConfig()
        self.initial, self.minimum = initial, minimum
        self.dim = dim
        self.geometry = geometry
        self.retrieval_loss = loss
        self.register_buffer("keys", torch.zeros(maximum, dim if key_dim is None else key_dim))
        self.register_buffer("values", torch.zeros(maximum, dim))
        self.register_buffer("weights", torch.zeros(maximum))
        self.register_buffer("uncertainty", torch.zeros(maximum))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("velocity", torch.zeros(maximum, dim) if self.cfg.momentum else None)
        self.register_buffer("covariance", torch.zeros(maximum, 2, 2) if self.cfg.momentum else None)
        self.register_buffer("smoothed_mass", torch.zeros(maximum))
        self.register_buffer("smoothed_square_mass", torch.zeros(maximum))
        self.register_buffer("seen", torch.zeros(maximum, dtype=torch.bool))
        self.register_buffer("weak_runs", torch.zeros(maximum, dtype=torch.long))
        self.register_buffer("overlap_runs", torch.zeros(maximum, maximum, dtype=torch.long))
        self.last_structure = {}

    @property
    def mask(self):
        return self.weights > 0

    @property
    def active_count(self):
        return int(self.mask.sum().item())

    @property
    def ready(self):
        return self.active_count > 0

    @property
    def layout(self):
        return Components(self.keys, self.values, self.weights)

    def _observations(self, queries, targets=None):
        raw = queries.detach().to(self.keys)
        v = raw if targets is None else targets.detach().to(self.values)
        if raw.ndim != 2 or raw.shape[1] != self.keys.shape[1] or v.shape != (len(raw), self.dim):
            raise ValueError("Expected one query direction and one raw value per row.")
        valid = self.geometry.valid_queries(raw, fitting=True) & v.isfinite().all(-1)
        return raw[valid], v[valid]

    def _feasible_keys(self, keys):
        if not self.geometry.feasible(keys).all():
            return False
        if self.cfg.separation is None or len(keys) < 2:
            return True
        affinity = self.geometry.redundancy(keys)
        pairs = torch.triu_indices(len(keys), len(keys), 1, device=keys.device)
        return bool((affinity[pairs[0], pairs[1]] <= self.cfg.separation).all())

    def _reset_filters(self, slots):
        active = self.mask[slots]
        p0 = self.scale * self.cfg.initial_variance
        self.uncertainty[slots] = active.to(self.values.dtype) * p0
        if self.cfg.momentum:
            self.velocity[slots] = 0
            matrix = self._state_noise(self.cfg.initial_covariance, self.cfg.initial_variance)
            self.covariance[slots] = active[:, None, None] * matrix
            self.uncertainty[slots] = self.covariance[slots, 0, 0]
        self.smoothed_mass[slots] = 0
        self.smoothed_square_mass[slots] = 0
        self.seen[slots] = False
        self.weak_runs[slots] = 0
        self.overlap_runs[slots, :] = 0
        self.overlap_runs[:, slots] = 0

    def _state_noise(self, packed, default):
        pp, pv, vv = (default, 0., default) if packed is None else packed
        return self.scale * self.scale.new_tensor([[pp, pv], [pv, vv]])

    @torch.no_grad()
    def initialize_from_embeddings(self, queries, targets=None, *, seed=0):
        u, v = self._observations(queries, targets)
        if len(u) < self.initial:
            return False
        # Farthest-first seeds in the write geometry; fit any metric on training data only.
        if not self.ready:
            self.geometry.fit_metric(u)
        candidates = self.geometry.seed_keys(u)
        feasible = self.geometry.feasible(candidates)
        if not feasible.all():
            candidates, u, v = candidates[feasible], u[feasible], v[feasible]
            if len(u) < self.initial:
                return False
        first = int(torch.randint(len(u), (1,), generator=torch.Generator().manual_seed(seed)))
        seeds = [first]
        distances = self.geometry.distance(candidates, candidates[first:first+1]).flatten()
        for _ in range(1, self.initial):
            if self.cfg.separation is not None:
                # Check only against selected seeds, avoiding a window-by-window Gram matrix.
                eligible = self.geometry.pair_affinity(candidates, candidates[seeds]).amax(1) <= self.cfg.separation
                distances.masked_fill_(~eligible, -torch.inf)
            distances.masked_fill_((candidates == candidates[seeds[-1]]).all(-1), -torch.inf)
            distances[seeds] = -torch.inf
            index = int(distances.argmax())
            if distances[index] <= torch.finfo(u.dtype).eps ** 2:
                return False
            seeds.append(index)
            distances = torch.minimum(distances, self.geometry.distance(candidates, candidates[index:index+1]).flatten())
        keys = candidates[seeds]
        if not self._feasible_keys(keys):
            return False
        assignment = self.geometry.distance(u, keys).argmin(1)
        assignment[seeds] = torch.arange(self.initial, device=u.device)
        counts = torch.bincount(assignment, minlength=self.initial)
        means = v.new_zeros(self.initial, self.dim).index_add_(0, assignment, v) / counts[:, None]
        scale = v.var(0, correction=1).mean() if len(v) > 1 else v.new_zeros(())
        if scale <= 0:
            scale = v.square().mean()
        self.scale.copy_(scale if scale > 0 else self.scale.new_ones(()))
        self.keys.zero_()
        self.values.zero_()
        self.weights.zero_()
        self.keys[:self.initial] = keys
        self.values[:self.initial] = means
        self.weights[:self.initial] = 1/self.initial if self.cfg.uniform_weights else counts.to(v) / len(v)
        self._reset_filters(torch.arange(len(self.weights), device=u.device))
        self.last_structure = {}
        return True

    @torch.no_grad()
    def read(self, queries, fallback=None):
        raw = queries.detach().to(self.keys)
        if raw.ndim != 2 or raw.shape[1] != self.keys.shape[1]:
            raise ValueError("Expected a matrix with the configured query dimension.")
        active = self.mask
        if fallback is None and raw.shape[1] == self.dim:
            fallback = raw.nan_to_num(nan=0., posinf=0., neginf=0.)
        if fallback is None:
            # A target-free fallback for separate query and value spaces.
            fallback = (self.weights[active] @ self.values[active]).expand(len(raw), -1)
        if not self.ready:
            return fallback
        result = fallback.clone()
        valid = self.geometry.valid_queries(raw)
        logits = self._read_logits(raw[valid], self.keys[active], self.weights[active])
        if not logits.isfinite().all():
            raise ValueError("Read scores and temperature must produce finite logits.")
        result[valid], _ = read_from_logits(logits, self.values[active])
        return result

    def _read_logits(self, queries, keys, weights):
        return weights.log() + self.cfg.beta_r * self.geometry.read_similarity(queries, keys)

    def _scores(self, u, v, keys, means, weights):
        read = self._read_logits(u, keys, weights)
        write = weights.log() + self.cfg.beta_w * self.geometry.write_similarity(u, keys)
        if self.cfg.value_aware:
            write = write - squared_distance(v, means) / (2 * self.cfg.observation_noise * self.scale)
        if not (read.isfinite().all() and write.isfinite().all()):
            raise ValueError("The configured scores, temperatures or Gaussian residual exceeded the finite range.")
        return read, write

    @torch.no_grad()
    def empirical_covariance(self, slot, queries, targets=None):
        """Eq. 28 on a frozen window. Undefined centroids/covariances return None."""
        if not self.mask[slot]:
            raise ValueError("Select an active slot.")
        u, v = self._observations(queries, targets)
        slots = self.mask.nonzero().flatten()
        v = v.double()
        _, scores = self._scores(u.double(), v, self.keys[slots].double(), self.values[slots].double(), self.weights[slots].double())
        weights = scores.softmax(1)[:, (slots == slot).nonzero().item()]
        if weights.sum() == 0:
            return None
        weights /= weights.sum()
        mean = weights @ v
        centered = v - mean
        return mean, centered.T @ (weights[:, None] * centered)

    @torch.no_grad()
    def online_update(self, queries, targets=None, *, dt=None):
        """Update the bank with observation variance fixed at r0*s0."""
        dt = self.cfg.dt if dt is None else dt
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite.")
        if not self.ready:
            return
        u, v = self._observations(queries, targets)
        active = self.mask
        means = self.values[active]
        q = self.cfg.process_noise * self.scale * dt
        r = self.cfg.observation_noise * self.scale
        if self.cfg.momentum:
            velocity = self.velocity[active]
            means = means + dt * velocity
            velocity = self.cfg.damping * velocity
            process = self._state_noise(self.cfg.process_covariance, self.cfg.process_noise)
            prior_cov = predict_position_velocity(self.covariance[active], process, dt, self.cfg.damping)
        else:
            prior = self.uncertainty[active] + q
        _, scores = self._scores(u, v, self.keys[active], means, self.weights[active])
        responsibilities = scores.softmax(1)
        mass = responsibilities.sum(0)
        # The summed innovation equals N*(v_bar - m_minus), including when N=0.
        innovation = responsibilities.T @ v - mass[:, None] * means
        if self.cfg.momentum:
            factors, new_covariance = condition_position_velocity(prior_cov, mass, r)
            position_factor = factors[:, 0]
            new_velocity = velocity + factors[:, 1, None] * innovation
            new_uncertainty = new_covariance[:, 0, 0]
        else:
            position_factor, retention = fractional_measurement_factors(prior, r, mass)
            new_uncertainty = prior * retention
        new_values = means + position_factor[:, None] * innovation
        new_keys = self.geometry.key_candidates(u, responsibilities, self.keys[active])
        mixture = self.weights[active]
        if len(u) and not self.cfg.uniform_weights:
            mixture = ema_step(mixture, mass / len(u), alpha=self.cfg.mixture_rate)
            mixture.clamp_min_(torch.finfo(mixture.dtype).tiny)
            mixture = mixture / mixture.sum()
        if not self._feasible_keys(new_keys):
            return
        if len(u) and self.cfg.guard_writes:
            old = self._evaluate(self.layout, u.double(), v.double())[0]
            proposed = Components(new_keys, new_values, mixture)
            new = self._evaluate(proposed, u.double(), v.double())[0]
            if not math.isfinite(new) or new > old:
                return
        self.values[active], self.keys[active], self.weights[active] = new_values, new_keys, mixture
        self.uncertainty[active] = new_uncertainty
        if self.cfg.momentum:
            self.velocity[active], self.covariance[active] = new_velocity, new_covariance

    def _evaluate(self, layout, u, v, *, details=False):
        """One fixed-window evaluation in float64, including the actual query-only read."""
        slots = (layout.weights > 0).nonzero().flatten()
        keys, means, weights = layout.keys[slots].double(), layout.values[slots].double(), layout.weights[slots].double()
        read, joint = self._scores(u, v, keys, means, weights)
        fit = -joint.logsumexp(1)
        retrieved, _ = read_from_logits(read, means)
        read_error = self.retrieval_loss(v, retrieved, self.cfg.observation_noise * self.scale)
        loss = fit + self.cfg.read_weight * read_error
        result = float(loss.mean()) + self.cfg.capacity_cost * len(slots), joint.softmax(1), loss
        return (*result, float(read_error.mean())) if details else result

    @staticmethod
    def _statistics(responsibilities, v):
        mass = responsibilities.sum(0)
        squared = responsibilities.square().sum(0)
        safe = torch.where(mass > 0, mass, 1)
        conditional = responsibilities / safe
        centroid = conditional.T @ v
        distance = torch.cdist(v, centroid, compute_mode="donot_use_mm_for_euclid_dist")
        return {
            "mass": mass, "squared": squared,
            "kish": kish_effective_sample_size(mass, torch.where(squared > 0, squared, 1), eps=0.),
            # Trace of the empirical within-slot value covariance, distinct from filter P.
            "spread": (conditional * distance.square()).sum(0),
        }

    def _advance_history(self, slots, stats, window_length):
        mass, squared = stats["mass"].to(self.values), stats["squared"].to(self.values)
        seen = self.seen[slots]
        for buffer, current in ((self.smoothed_mass, mass), (self.smoothed_square_mass, squared)):
            buffer[slots] = torch.where(seen, ema_step(buffer[slots], current, alpha=.5), current)
        self.seen[slots] = True
        threshold = self.cfg.support_fraction * window_length / len(slots)
        weak = (self.smoothed_mass[slots] < threshold) | (self.smoothed_square_mass[slots] < threshold / len(slots))
        self.weak_runs[slots] = torch.where(weak, self.weak_runs[slots] + 1, 0)
        gram = self.geometry.redundancy(self.keys[slots])
        pair = (slots[:, None], slots[None, :])
        self.overlap_runs[pair] = torch.where(gram >= self.cfg.merge_cosine, self.overlap_runs[pair] + 1, 0)

    def _prune(self, source, slot):
        layout = source.clone()
        layout.clear(slot)
        layout.weights /= layout.weights.sum()
        return Edit("prune", layout, (slot,))

    def _merge(self, source, left, right):
        layout = source.clone()
        total = source.weights[left] + source.weights[right]
        fractions = source.weights[[left, right]] / total
        key = self.geometry.merged_key(source.keys[[left, right]], fractions)
        if key is None:
            return None
        layout.keys[left] = key
        # Preserve exact duplicates even when the normalized fractions round in float32.
        values = source.values[[left, right]]
        layout.values[left] = values[0] if torch.equal(values[0], values[1]) else fractions @ values
        layout.weights[left] = total
        layout.clear(right)
        return Edit("merge", layout, (left, right))

    def _splits(self, source, parent, free, u, v, r):
        for keys in self.geometry.split_keys(u, r, source.keys[parent].double()):
            allocation = self.cfg.beta_w * self.geometry.write_similarity(u, keys)
            portions = r[:, None] * allocation.softmax(1)
            masses = portions.sum(0)
            if (masses < self.cfg.min_child_mass).any():
                continue
            layout = source.clone()
            children = [parent, free]
            layout.keys[children] = keys.to(layout.keys)
            layout.values[children] = ((portions.T @ v) / masses[:, None]).to(layout.values)
            layout.weights[children] = (source.weights[parent] * (masses / masses.sum())).to(layout.weights)
            yield Edit("split", layout, (parent, free))

    def _birth(self, source, free, u, v, losses):
        count = max(math.ceil(self.cfg.min_child_mass), math.ceil(0.05 * len(u)))
        if count > len(u):
            return None
        selected = losses.topk(count).indices
        key = self.geometry.birth_key(u[selected])
        if key is None:
            return None
        layout = source.clone()
        weight = self.cfg.birth_weight or min(0.5, count / len(u))
        if not (source.weights > 0).any():
            weight = 1.
        layout.weights *= 1 - weight
        layout.weights[free] = weight
        layout.keys[free] = key
        layout.values[free] = v[selected].mean(0)
        return Edit("birth", layout, (free,))

    def _candidates(self, u, v, responsibilities, losses, stats):
        source = self.layout
        slots = self.mask.nonzero().flatten()
        k = len(slots)
        mass = stats["mass"]
        # Screen a finite pool; history advances once per window, even if several edits pass.
        threshold = self.cfg.support_fraction * len(u) / k
        weak = (self.weak_runs[slots] >= self.cfg.patience) & (
            (mass < threshold) | (stats["squared"] < threshold / k)
        )
        for col in mass.argsort().tolist():
            if k > self.minimum and weak[col]:
                yield self._prune(source, int(slots[col]))
        keys = self.keys[slots]
        gram = self.geometry.redundancy(keys)
        pair_history = self.overlap_runs[slots[:, None], slots[None, :]]
        pairs = ((gram >= self.cfg.merge_cosine) & (pair_history >= self.cfg.patience)).triu(1).nonzero()
        if k > self.minimum and len(pairs):
            order = gram[pairs[:, 0], pairs[:, 1]].argsort(descending=True)[:4]
            for left, right in pairs[order].tolist():
                candidate = self._merge(source, int(slots[left]), int(slots[right]))
                if candidate:
                    yield candidate
        free_slots = (~self.mask).nonzero().flatten()
        if len(free_slots):
            free = int(free_slots[0])
            supported = (mass >= 2 * self.cfg.min_child_mass) & (stats["kish"] >= 2)
            ranked = supported.nonzero().flatten()
            ranked = ranked[stats["spread"][ranked].argsort(descending=True)[:4]]
            for col in ranked.tolist():
                yield from self._splits(source, int(slots[col]), free, u, v, responsibilities[:, col])
            candidate = self._birth(source, free, u, v, losses)
            if candidate:
                yield candidate
        elif k == 1:
            slot = int(slots[0])
            empty = source.clone()
            empty.clear(slot)
            birth = self._birth(empty, slot, u, v, losses)
            if birth:
                yield Edit("prune+birth", birth.layout, (slot,))
        else:
            # Compare only the final capacity-neutral layout; neither part commits on its own.
            reductions = [self._prune(source, int(slots[mass.argmin()]))]
            closest = gram.masked_fill(torch.eye(k, device=gram.device, dtype=torch.bool), -torch.inf).argmax().item()
            merged = self._merge(source, int(slots[closest // k]), int(slots[closest % k]))
            if merged:
                reductions.append(merged)
            for reduction in reductions:
                free = reduction.changed[-1]
                _, _, residuals = self._evaluate(reduction.layout, u, v)
                birth = self._birth(reduction.layout, free, u, v, residuals)
                if birth:
                    yield Edit(reduction.name + "+birth", birth.layout, reduction.changed)

    @torch.no_grad()
    def adapt_structure(self, queries, targets=None, *, epoch=None):
        """Fit candidates on a frozen training window and commit only objective improvements."""
        limit = self.cfg.window_size
        u, v = self._observations(queries[:limit], None if targets is None else targets[:limit])
        report = {"epoch": epoch, "pruned": 0, "coalesced": 0, "spawned": 0, "trials": []}
        if not self.ready or not len(u):
            self.last_structure = report
            return report
        u, v = u.double(), v.double()
        before, responses, losses, before_read = self._evaluate(self.layout, u, v, details=True)
        report["before"] = before
        slots = self.mask.nonzero().flatten()
        stats = self._statistics(responses, v)
        self._advance_history(slots, stats, len(u))
        for _ in range(0 if self.cfg.fixed_capacity else self.cfg.edit_budget):
            for edit in self._candidates(u, v, responses, losses, stats):
                layout = edit.layout
                active = layout.weights > 0
                if self.cfg.uniform_weights:
                    layout.weights[active] = 1. / int(active.sum())
                expected = self.active_count + {"prune": -1, "merge": -1, "split": 1, "birth": 1}.get(edit.name, 0)
                if (int(active.sum()) != expected or expected < self.minimum or
                    not torch.isclose(layout.weights.sum(), layout.weights.new_ones(())) or
                    not self._feasible_keys(layout.keys[active])):
                    continue
                after, new_responses, new_losses, after_read = self._evaluate(layout, u, v, details=True)
                improves = after < before and after <= before - self.cfg.margin
                protect_read = self.cfg.guard_read_loss or getattr(self.cfg, "retrieval_margin", None) is not None
                accepted = (math.isfinite(after) and (improves or self.cfg.heuristic_edits)
                            and (not protect_read or after_read <= before_read))
                report["trials"].append(dict(operation=edit.name, slots=edit.changed,
                                             before=before, after=after, read_before=before_read,
                                             read_after=after_read, accepted=accepted))
                if not accepted:
                    continue
                self.keys.copy_(layout.keys)
                self.values.copy_(layout.values)
                self.weights.copy_(layout.weights)
                self._reset_filters(list(edit.changed))
                report["pruned"] += int("prune" in edit.name)
                report["coalesced"] += int("merge" in edit.name)
                report["spawned"] += int(edit.name == "split" or "birth" in edit.name)
                before, responses, losses = after, new_responses, new_losses
                before_read = after_read
                stats = self._statistics(responses, v)
                break
            else:
                break
        report["after"] = before
        self.last_structure = report
        return report
