"""Bounded variance, safeguarded GASS writes and optional stream movement budget."""
from __future__ import annotations

import math

import torch

from hopfieldpc.functional import read_from_logits, squared_distance
from hopfieldpc.kalman import condition_position_velocity, fractional_measurement_factors, predict_position_velocity

from .config import GASSConfig
from .geometry import KeyGeometry
from .loss import RetrievalLoss
from .reference import Components, ReferenceBank


def project_mixture(weights, floor):
    """Euclidean projection onto sum(pi)=1, pi>=floor."""
    shifted = weights - floor
    ordered = shifted.sort(descending=True).values
    ranks = torch.arange(1, len(weights) + 1, device=weights.device, dtype=weights.dtype)
    offsets = (ordered.cumsum(0) - (1 - len(weights) * floor)) / ranks
    threshold = offsets[(ordered > offsets).sum() - 1]
    return (shifted - threshold).clamp_min(0) + floor


class GASSBank(ReferenceBank):
    """Safeguarded continuous writes, frozen-window replay and structural edits."""

    def __init__(self, dim, initial, minimum, maximum, config=None, *, key_dim=None,
                 geometry: KeyGeometry, loss: RetrievalLoss):
        super().__init__(dim, initial, minimum, maximum, config or GASSConfig(),
                         key_dim=key_dim, geometry=geometry, loss=loss)
        if self.cfg.mixture_floor * maximum >= 1:
            raise ValueError("mixture_floor must be strictly smaller than 1 / max_memories.")
        self.register_buffer("variance", torch.ones(()))
        self.register_buffer("budget_step", torch.zeros((), dtype=torch.long))
        self.last_write = {}

    @property
    def retrieval_scale(self):
        return self.scale  # Fixed s_ref = R_ref = warm-up scale.

    def _clip_trust(self, trust):
        return trust.clamp(self.cfg.trust_min * self.scale, self.cfg.trust_max * self.scale)

    def _clip_covariance(self, covariance):
        eigenvalues, basis = torch.linalg.eigh((covariance + covariance.mT) / 2)
        return (basis * self._clip_trust(eigenvalues).unsqueeze(-2)) @ basis.mT

    def _reset_filters(self, slots):
        super()._reset_filters(slots)
        active = self.mask[slots]
        if self.cfg.momentum:
            covariance = self._clip_covariance(self.covariance[slots].double()).to(self.covariance)
            self.covariance[slots] = active[:, None, None] * covariance
            self.uncertainty[slots] = self.covariance[slots, 0, 0]
        else:
            self.uncertainty[slots] = active * self._clip_trust(self.uncertainty[slots])

    @torch.no_grad()
    def initialize_from_embeddings(self, queries, targets=None, *, seed=0):
        if self.ready and self.cfg.stream_budget is not None:
            raise ValueError("A budgeted stream starts with one initialization. Create a new bank to restart.")
        if not super().initialize_from_embeddings(queries, targets, seed=seed):
            return False
        active = self.mask
        self.values.clamp_(-self.cfg.logit_bound, self.cfg.logit_bound)
        self.weights[active] = project_mixture(self.weights[active].double(), self.cfg.mixture_floor).to(self.weights)
        self.variance.copy_(self.scale * min(max(self.cfg.observation_noise, self.cfg.variance_min), self.cfg.variance_max))
        self.last_write = {}
        return True

    def _read_logits(self, queries, keys, weights):
        if not self.cfg.recall_steps:
            return super()._read_logits(queries, keys, weights)
        return self.geometry.refined_read_logits(queries, keys, weights, self.cfg.beta_r, self.cfg.recall_steps)

    def _scores_at(self, u, v, keys, means, weights, variance):
        read = self._read_logits(u, keys, weights)
        joint = weights.log() + self.cfg.beta_w * self.geometry.write_similarity(u, keys)
        if self.cfg.value_aware:
            joint = joint - squared_distance(v, means) / (2 * variance)
        return read, joint

    def _scores(self, u, v, keys, means, weights):
        return self._scores_at(u, v, keys, means, weights, self.variance)

    def _objective(self, layout, u, v, variance):
        read, joint = self._scores_at(u, v, layout.keys, layout.values, layout.weights, variance)
        retrieved, _ = read_from_logits(read, layout.values)
        error = self.retrieval_loss(v, retrieved, self.retrieval_scale)
        rows = -joint.logsumexp(1) + self.cfg.read_weight * error
        if self.cfg.value_aware:
            rows = rows + (self.dim / 2) * (variance / self.scale).log()
        return rows.mean() + self.cfg.capacity_cost * len(layout.weights), joint, rows, error

    @staticmethod
    def _active_layout(layout):
        active = layout.weights > 0
        return Components(layout.keys[active].double(), layout.values[active].double(), layout.weights[active].double())

    def _evaluate(self, layout, u, v, *, details=False):
        cost, joint, rows, errors = self._objective(self._active_layout(layout), u.double(), v.double(), self.variance.double())
        result = float(cost), joint.softmax(1), rows
        return (*result, float(errors.mean())) if details else result

    def _candidate(self, state, field, proposal):
        # Score the exact coordinates representable in the bank storage dtype.
        trial = Components(state.keys, state.values, state.weights)
        setattr(trial, field, proposal.to(self.values.dtype).double())
        return trial

    @staticmethod
    def _decreases(before, after, penalty):
        scale = torch.maximum(before.abs(), after.abs()).clamp_min(1)
        slack = 16 * torch.finfo(before.dtype).eps * scale
        return bool(torch.isfinite(after) & (before - after >= penalty + slack))

    def _variance_step(self, state, u, v, before):
        current = self.variance.double()
        if not self.cfg.value_aware:
            return before
        _, joint = self._scores_at(u, v, state.keys, state.values, state.weights, current)
        estimate = (joint.softmax(1) * squared_distance(v, state.values)).sum() / v.numel()
        candidate = estimate.clamp(self.scale * self.cfg.variance_min, self.scale * self.cfg.variance_max).to(self.variance).double()
        after = self._objective(state, u, v, candidate)[0]
        coefficient = self.dim / 4 * self.cfg.variance_min / self.cfg.variance_max
        if self._decreases(before, after, coefficient * (candidate / current).log().square()):
            self.variance.copy_(candidate)
            return after
        return before

    def _value_step(self, state, u, v, prior, before):
        cfg, variance = self.cfg, self.variance.double()
        read, joint = self._scores_at(u, v, state.keys, state.values, state.weights, variance)
        h, w = read_from_logits(read, state.values)
        r = joint.softmax(1)
        grad, curvature = self.retrieval_loss.local_model(v, h, self.retrieval_scale)
        mass = r.sum(0)
        evidence = int(cfg.value_aware) / variance
        direction = evidence * (r.T @ v - mass[:, None] * state.values) - cfg.read_weight * (w.T @ grad)
        precision = prior.reciprocal() + evidence * mass + cfg.read_weight * curvature * w.sum(0)
        step = direction / precision[:, None]
        attempts = 1 if self.retrieval_loss.gradient_curvature is not None else cfg.line_search_steps
        for _ in range(attempts):
            candidate = self._candidate(state, "values", (state.values + step).clamp(-cfg.logit_bound, cfg.logit_bound))
            delta = candidate.values - state.values
            if not delta.any():
                break
            penalty = (delta.square().sum(1) / prior).sum() / (2 * len(u))
            after = self._objective(candidate, u, v, variance)[0]
            if self._decreases(before, after, penalty):
                return candidate, after
            step = step / 2
        return state, before

    def _gradient_step(self, state, field, u, v, before):
        if field == "weights" and (self.cfg.uniform_weights or len(state.weights) == 1):
            return state, before, "fixed"
        with torch.enable_grad():
            block = getattr(state, field).detach().requires_grad_()
            trial = Components(state.keys, state.values, state.weights)
            setattr(trial, field, block)
            objective = self._objective(trial, u, v, self.variance.double())[0]
            grad = torch.autograd.grad(objective, block, allow_unused=True)[0] if objective.requires_grad else None
        if field == "keys" and self.geometry.use_exemplar_updates:
            # A thresholded write score is locally constant almost everywhere.
            # Try feasible exemplars and check their actual retrieval and fit cost.
            _, joint = self._scores(u, v, state.keys, state.values, state.weights)
            keys = self.geometry.key_candidates(u, joint.softmax(1), state.keys)
            candidate = self._candidate(state, field, self.geometry.project_keys(keys))
            if self._feasible_keys(candidate.keys):
                after = self._objective(candidate, u, v, self.variance.double())[0]
                penalty = self.cfg.descent_margin * (candidate.keys - state.keys).square().sum() / 2
                if self._decreases(before, after, penalty):
                    return candidate, after, "accepted exemplar"
        if grad is None:
            return state, before, "no gradient"
        rate = min(self.cfg.mixture_rate if field == "weights" else 1., 1 / self.cfg.descent_margin)
        for _ in range(self.cfg.line_search_steps):
            proposed = block.detach() - rate * grad
            proposed = (project_mixture(proposed, self.cfg.mixture_floor) if field == "weights"
                        else self.geometry.project_keys(proposed))
            candidate = self._candidate(state, field, proposed)
            delta = getattr(candidate, field) - getattr(state, field)
            if not delta.any():
                return state, before, "unchanged"
            if field != "keys" or self._feasible_keys(candidate.keys):
                after = self._objective(candidate, u, v, self.variance.double())[0]
                half_norm = delta.square().sum() / 2
                penalty = self.cfg.descent_margin * half_norm
                upper = before + (grad * delta).sum() + (1 / rate - self.cfg.descent_margin) * half_norm
                if after <= upper and self._decreases(before, after, penalty):
                    return candidate, after, "accepted"
            rate /= 2
        return state, before, "budget exhausted"

    def _sweep(self, u, v, prior):
        state = self._active_layout(self.layout)
        before = self._objective(state, u, v, self.variance.double())[0]
        variance_cost = self._variance_step(state, u, v, before)
        state, value_cost = self._value_step(state, u, v, prior, variance_cost)
        state, key_cost, key_status = self._gradient_step(state, "keys", u, v, value_cost)
        state, after, weight_status = self._gradient_step(state, "weights", u, v, key_cost)
        active = self.mask
        self.keys[active] = state.keys.to(self.keys)
        self.values[active] = state.values.to(self.values)
        self.weights[active] = state.weights.to(self.weights)
        return dict(before=float(before), variance=float(variance_cost), values=float(value_cost),
                    keys=float(key_cost), after=float(after), key_step=key_status, weight_step=weight_status)

    def _advance_trust(self, state, u, v, dt):
        cfg, variance = self.cfg, self.variance.double()
        active = self.mask
        _, joint = self._scores_at(u, v, state.keys, state.values, state.weights, variance)
        r = joint.softmax(1)
        mass = r.sum(0)
        if not cfg.momentum:
            prior = self._clip_trust(self.uncertainty[active].double() + dt * cfg.process_noise * self.scale)
            _, retention = fractional_measurement_factors(prior, variance, mass)
            self.uncertainty[active] = (prior * retention).to(self.uncertainty)
            return prior

        process = self._state_noise(cfg.process_covariance, cfg.process_noise)
        predicted = predict_position_velocity(self.covariance[active].double(), process, dt, cfg.damping)
        covariance = self._clip_covariance(predicted)
        prior = covariance[:, 0, 0]
        factors, posterior = condition_position_velocity(covariance, mass, variance)
        self.covariance[active] = posterior.to(self.covariance)
        self.uncertainty[active] = self.covariance[active, 0, 0]

        velocity = self.velocity[active].double()
        prediction = state.values + dt * velocity
        innovation = r.T @ v - mass[:, None] * prediction
        position = prediction + factors[:, 0, None] * innovation
        candidate = self._candidate(state, "values", position.clamp(-cfg.logit_bound, cfg.logit_bound))
        before = self._objective(state, u, v, variance)[0]
        after = self._objective(candidate, u, v, variance)[0]
        penalty = cfg.descent_margin * (candidate.values - state.values).square().sum()
        if self._decreases(before, after, penalty):
            self.values[active] = candidate.values.to(self.values)
            velocity = cfg.damping * velocity + factors[:, 1, None] * innovation
            self.velocity[active] = velocity.clamp(-cfg.logit_bound, cfg.logit_bound).to(self.velocity)
        else:
            self.velocity[active] = 0
        return prior

    def _budget_snapshot(self, *, structure=False):
        if self.cfg.stream_budget is None:
            return None
        names = ["keys", "values", "weights", "variance"]
        if structure:
            # A rejected edit must also undo filter resets and history changes.
            names.extend(("uncertainty", "velocity", "covariance", "seen", "smoothed_mass",
                          "smoothed_square_mass", "weak_runs", "overlap_runs"))
        return {name: getattr(self, name).clone() for name in names if getattr(self, name) is not None}

    def _apply_budget(self, snapshot):
        """One transaction of StreamStability.commit, using the sup distance on (k,m,pi,R)."""
        if snapshot is None:
            return {}
        step = int(self.budget_step)
        allowance = math.ldexp(self.cfg.stream_budget, -min(step + 1, 4096))
        if step < 4096:  # Beyond this point the allowance is zero for every finite float64 budget.
            self.budget_step.add_(1)
        distances = [(getattr(self, name).double() - snapshot[name].double()).abs().max()
                     for name in ("keys", "values", "weights", "variance")]
        movement = float(torch.stack(distances).amax())
        accepted = math.isfinite(movement) and movement <= allowance
        if not accepted:
            for name, value in snapshot.items():
                getattr(self, name).copy_(value)
        return dict(budget_accepted=accepted, allowance=allowance, proposed_movement=movement)

    @staticmethod
    def _write_report(report, budget, start):
        if budget and not budget["budget_accepted"]:
            report = dict(report, proposed_after=report["after"], after=start)
        return dict(report, **budget)

    @torch.no_grad()
    def online_update(self, queries, targets=None, *, dt=None):
        dt = self.cfg.dt if dt is None else dt
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite.")
        u, v = self._observations(queries, targets)
        if not self.ready or not len(u):
            return
        u, v = u.double(), v.double()
        before = self._evaluate(self.layout, u, v)[0]
        snapshot = self._budget_snapshot()
        prior = self._advance_trust(self._active_layout(self.layout), u, v, dt)
        report = self._sweep(u, v, prior)
        budget = self._apply_budget(snapshot)
        if self.cfg.momentum and budget and not budget["budget_accepted"]:
            self.velocity.zero_()
        report = self._write_report(report, budget, before)
        self.last_write = dict(report, fresh_before=before, observation_variance=float(self.variance))
        return self.last_write

    @torch.no_grad()
    def refine_window(self, queries, targets=None, *, sweeps=1):
        """Optimize one detached window with constant evidence, trust and capacity."""
        if type(sweeps) is not int or sweeps < 0:
            raise ValueError("sweeps must be a nonnegative integer.")
        u, v = self._observations(queries, targets)
        if not self.ready or not len(u):
            return []
        u, v = u.double(), v.double()
        prior = self._clip_trust(self.uncertainty[self.mask].double())
        reports = []
        for _ in range(sweeps):
            snapshot = self._budget_snapshot()
            report = self._sweep(u, v, prior)
            reports.append(self._write_report(report, self._apply_budget(snapshot), report["before"]))
        return reports

    def _candidates(self, u, v, responsibilities, losses, stats):
        for edit in super()._candidates(u, v, responsibilities, losses, stats):
            layout = edit.layout
            active = layout.weights > 0
            if not active.any() or not layout.values[active].isfinite().all():
                continue
            layout.keys[active] = self.geometry.project_keys(layout.keys[active])
            layout.values.clamp_(-self.cfg.logit_bound, self.cfg.logit_bound)
            if self.cfg.uniform_weights or (layout.weights[active] >= self.cfg.mixture_floor).all():
                yield edit

    @torch.no_grad()
    def adapt_structure(self, queries, targets=None, *, epoch=None):
        limit = self.cfg.window_size
        refinement = self.refine_window(queries[:limit], None if targets is None else targets[:limit],
                                        sweeps=self.cfg.refinement_sweeps) if self.cfg.refinement_sweeps else []
        edits_enabled = not self.cfg.fixed_capacity and self.cfg.edit_budget > 0
        snapshot = self._budget_snapshot(structure=True) if edits_enabled else None
        report = super().adapt_structure(queries, targets, epoch=epoch)
        if snapshot is not None and "after" in report:
            budget = self._apply_budget(snapshot)
            report = dict(report, **budget)
            if not budget["budget_accepted"]:
                report.update(after=report["before"], pruned=0, coalesced=0, spawned=0)
                for trial in report["trials"]:
                    if trial["accepted"]:
                        trial.update(accepted=False, budget_rejected=True)
        self.last_structure = dict(report, refinement=refinement)
        return self.last_structure
