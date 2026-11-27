"""Association functions for continuous MNIST decoder logits (proposal §§2, 6, 9–10).

Queries and candidate keys retain logits for mixed profiles. Each score prepares its
own view: unit directions, raw features, bounded dot-product keys, sigmoid masks,
strictly positive softmax distributions, or thresholded binary codes. Values never
pass through these transforms on a Kalman write.
"""
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from hopfieldpc.functional import multi_step_retrieve, sample_space_fission, squared_distance, unit_rows as unit, weighted_spectral_fission
from hopfieldpc.gass import KeyGeometry, RetrievalLoss

SCORES = ("cosine", "euclidean", "mahalanobis", "dot", "dice", "kl", "jaccard", "hamming")
LOSSES = ("squared", "absolute", "angular", "dice", "kl", "bernoulli_kl")

@dataclass(frozen=True)
class ScoreAdapter:
    """A fixed, slot-local score on decoder logits, with an explicit upper bound.

    The caller provides pairwise score and dissimilarity matrices [B,K], a finite
    upper bound per query (or a scalar), and optionally the raw-key domain. The
    generic learner evaluates feasible exemplars, including the previous key.
    """
    score: Callable
    distance: Callable
    upper_bound: Callable
    seed_keys: Callable = torch.clone
    valid_queries: Callable = lambda x: x.isfinite().all(-1)
    feasible_keys: Callable = lambda x: x.isfinite().all(-1)


_EXTRA_SCORES = {}
_EXTRA_LOSSES = {}


def register_score(name, adapter):
    if name in score_names() or not isinstance(adapter, ScoreAdapter):
        raise ValueError("Provide a new name and a ScoreAdapter.")
    _EXTRA_SCORES[name] = adapter


def register_loss(name, loss):
    if name in loss_names() or not callable(loss):
        raise ValueError("Provide a new loss name and a callable (target, retrieved, variance).")
    _EXTRA_LOSSES[name] = loss


def score_names():
    return (*SCORES, *_EXTRA_SCORES)


def loss_names():
    return (*LOSSES, *_EXTRA_LOSSES)


def probabilities(logits):
    # Reserve a total probability of 1e-7 for uniform smoothing, in every dtype.
    return (1 - 1e-7) * logits.softmax(-1) + 1e-7 / logits.shape[-1]


def dice_loss_parts(product, magnitude):
    # Treat two empty masks as a perfect match.
    return torch.where(magnitude > 0, 2 * product / (magnitude + 1e-8), 1.0).clamp(0, 1)


def association_loss(name, target, retrieved, variance):
    if name in _EXTRA_LOSSES:
        loss = _EXTRA_LOSSES[name](target, retrieved, variance)
        if loss.shape != target.shape[:-1] or not (loss.isfinite() & (loss >= 0)).all():
            raise ValueError("A custom associative loss must return one finite nonnegative value per row.")
        return loss
    if name == "squared":
        return (target - retrieved).square().sum(-1) / (2 * variance)
    if name == "bernoulli_kl":
        # KL(Bernoulli(sigmoid(target)) || Bernoulli(sigmoid(retrieved))), per pixel.
        p = target.sigmoid()
        softplus = torch.nn.functional.softplus
        return (softplus(retrieved) - softplus(target) - p * (retrieved - target)).mean(-1).clamp_min(0)
    if name == "absolute":
        return (target - retrieved).abs().mean(-1)
    if name == "angular":
        cosine = (unit(target) * unit(retrieved)).sum(-1).clamp(-1, 1)
        both_zero = (target.norm(dim=-1) == 0) & (retrieved.norm(dim=-1) == 0)
        return torch.where(both_zero, 0., 1 - cosine)
    if name == "dice":
        a, b = target.sigmoid(), retrieved.sigmoid()
        return 1 - dice_loss_parts((a*b).sum(-1), (a.square() + b.square()).sum(-1))
    if name == "kl":
        a, b = probabilities(target), probabilities(retrieved)
        return (a * (a.log() - b.log())).sum(-1).clamp_min(0)
    raise ValueError(f"Unknown associative loss {name!r}")


def _squared_local_model(target, retrieved, variance):
    return (retrieved - target) / variance, 1 / variance


def _bernoulli_local_model(target, retrieved, variance):
    dim = retrieved.shape[-1]
    return (retrieved.sigmoid() - target.sigmoid()) / dim, 1 / (4 * dim)


def retrieval_loss(name):
    models = {"squared": _squared_local_model, "bernoulli_kl": _bernoulli_local_model}
    return RetrievalLoss(partial(association_loss, name), models.get(name))


class AssociationGeometry(KeyGeometry):
    def __init__(self, config, dim, metric=None):
        super().__init__()
        self.cfg = config
        self.read_score, self.write_score = config.read_score, config.write_score
        self.unit_keys = self.read_score == self.write_score == "cosine"
        self.metric_supplied = metric is not None
        self.register_buffer("whitener", None)
        if "mahalanobis" in (self.read_score, self.write_score):
            if metric is None:
                self.whitener = torch.eye(dim)
            else:
                metric = torch.as_tensor(metric, dtype=torch.float64)
                if metric.shape != (dim, dim) or not torch.isfinite(metric).all() or not torch.allclose(metric, metric.T):
                    raise ValueError("The Mahalanobis metric must be a finite symmetric key_dim by key_dim matrix.")
                factor, info = torch.linalg.cholesky_ex(metric)
                if info.item():
                    raise ValueError("The Mahalanobis metric must be positive definite.")
                self.whitener = factor
        elif metric is not None:
            raise ValueError("A Mahalanobis metric requires a Mahalanobis read or write profile.")

    def fit_metric(self, u):
        if self.whitener is None or self.metric_supplied:
            return
        # Estimate once at warm-up. Shrink the training covariance toward isotropy.
        centered = u.double() - u.double().mean(0)
        cov = centered.T @ centered / max(len(u) - 1, 1)
        scale = cov.diagonal().mean()
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        cov = .9 * cov + .1 * torch.where(scale > 0, scale, 1.) * eye
        chol = torch.linalg.cholesky(cov)
        self.whitener.copy_(torch.linalg.solve_triangular(chol, eye, upper=False).T)

    def valid_queries(self, raw, *, fitting=False):
        valid = raw.isfinite().all(-1)
        if self.read_score == "cosine" or (fitting and self.write_score == "cosine"):
            valid &= raw.norm(dim=-1) > 0
        for name in {self.read_score, self.write_score} if fitting else (self.read_score,):
            if name in _EXTRA_SCORES:
                valid &= _EXTRA_SCORES[name].valid_queries(raw)
        return valid

    def bounded(self, keys):
        return keys / (keys.norm(dim=-1, keepdim=True) / self.cfg.key_radius).clamp_min(1.)

    def project_keys(self, raw):
        if self.write_score in _EXTRA_SCORES:
            return _EXTRA_SCORES[self.write_score].seed_keys(raw)
        if self.unit_keys:
            return unit(raw)
        if self.write_score == "dot":
            return self.bounded(raw)
        return raw.clone()

    def feasible(self, keys):
        valid = keys.isfinite().all(-1)
        norm = keys.norm(dim=-1)
        if self.unit_keys:
            valid &= torch.isclose(norm, torch.ones_like(norm))
        elif "cosine" in (self.read_score, self.write_score):
            valid &= norm > 0
        if self.write_score == "dot":
            valid &= norm <= self.cfg.key_radius * (1 + 1e-6)
        for name in {self.read_score, self.write_score}:
            if name in _EXTRA_SCORES:
                valid &= _EXTRA_SCORES[name].feasible_keys(keys)
        return valid

    def read_similarity(self, queries, keys):
        return self.score(self.read_score, queries, keys)

    def write_similarity(self, queries, keys):
        return self.score(self.write_score, queries, keys)

    def tied_recall(self, queries, keys, weights, beta, steps):
        cosine = self.read_score == "cosine"
        patterns = unit(keys) if cosine else self.bounded(keys)
        state = unit(queries) if cosine else queries
        beta = beta if cosine else beta / math.sqrt(keys.shape[1])
        if steps and len(state):
            state, _ = multi_step_retrieve(state, patterns, beta=beta, num_updates=steps,
                                           log_prior=weights.log(), unit_state=cosine)
        return state, patterns, beta

    def refined_read_logits(self, queries, keys, weights, beta, steps):
        state, patterns, beta = self.tied_recall(queries, keys, weights, beta, steps)
        return weights.log() + beta * (state @ patterns.T)

    def score(self, name, a, b):
        if name in _EXTRA_SCORES:
            adapter = _EXTRA_SCORES[name]
            result = adapter.score(a, b)
            bound = torch.as_tensor(adapter.upper_bound(a), device=a.device, dtype=a.dtype).reshape(-1, 1)
            if (result.shape != (len(a),len(b)) or len(bound) not in (1,len(a)) or not bound.isfinite().all()
                    or not result.isfinite().all() or (result > bound + 8*torch.finfo(a.dtype).eps*(1+bound.abs())).any()):
                raise ValueError(f"Invalid pairwise scores or upper bound from {name!r}.")
            return result
        if name == "cosine":
            return (unit(a) @ unit(b).T).clamp(-1, 1)
        if name == "mahalanobis":
            transform = self.whitener.to(a)
            return -squared_distance(a @ transform, b @ transform)
        if name == "euclidean":
            return -squared_distance(a, b)
        if name == "dot":
            return a @ self.bounded(b).T / math.sqrt(a.shape[-1])
        if name == "kl":
            p, q = probabilities(a), probabilities(b)
            return (p @ q.log().T - (p * p.log()).sum(-1, keepdim=True)).clamp_max(0)
        if name == "dice":
            a, b = a.sigmoid(), b.sigmoid()
            return dice_loss_parts(a @ b.T, a.square().sum(-1, keepdim=True) + b.square().sum(-1))
        if name in ("jaccard", "hamming"):
            a, b = (a > 0).to(a), (b > 0).to(b)
            overlap = a @ b.T
            total = a.sum(-1, keepdim=True) + b.sum(-1)
            if name == "hamming":
                return -(total - 2 * overlap)
            union = total - overlap
            return torch.where(union > 0, overlap / union.clamp_min(1), 1.)
        raise ValueError(f"Unknown score {name!r}")

    def distance(self, a, b):
        """Dissimilarity for seeding and exemplar proposals, with zero for identical views."""
        name = self.write_score
        if name in _EXTRA_SCORES:
            d = _EXTRA_SCORES[name].distance(a, b)
            if d.shape != (len(a), len(b)) or not (d.isfinite() & (d >= 0)).all():
                raise ValueError("A custom dissimilarity must be finite and nonnegative.")
            return d
        if name == "dot":
            return squared_distance(self.bounded(a), self.bounded(b))
        if name == "cosine":
            return .5 * squared_distance(unit(a), unit(b))
        score = self.score(name, a, b)
        return (1 - score).clamp_min(0) if name in ("dice", "jaccard") else -score

    def pair_affinity(self, a, b):
        name = self.write_score
        if name in ("cosine", "dice", "jaccard"):
            return self.score(name, a, b)
        distance = self.distance(a, b)
        # Symmetric, bounded affinity; 1 denotes identical score representations.
        if name == "kl" or name in _EXTRA_SCORES:
            return (-.5 * (distance + self.distance(b, a).T)).exp()
        return (-distance / a.shape[-1]).exp()

    def propose_keys(self, u, responsibility, conditional, old):
        name = self.write_score
        if name in ("euclidean", "mahalanobis"):
            candidate = conditional.T @ u
        elif name in ("cosine", "dot"):
            resultant = conditional.T @ (unit(u) if name == "cosine" else u)
            radius = self.cfg.key_radius if name == "dot" else (1. if self.unit_keys else old.norm(dim=-1, keepdim=True))
            candidate = unit(resultant) * radius
            candidate = torch.where(resultant.norm(dim=-1, keepdim=True) > 0, candidate, old)
        elif name == "kl":
            mean_p = conditional.T @ probabilities(u)
            # Invert the smoothed softmax, retaining the old logit offset for mixed reads.
            p = ((mean_p - 1e-7 / u.shape[-1]) / (1 - 1e-7)).clamp_min(torch.finfo(u.dtype).tiny)
            candidate = (p / p.sum(-1, keepdim=True)).log() + old.logsumexp(-1, keepdim=True)
        else:
            return super().propose_keys(u, responsibility, conditional, old)
        return candidate

    def merged_key(self, pair, fractions):
        if torch.equal(pair[0], pair[1]):
            return pair[0]
        if self.write_score in ("dice", "kl", "jaccard", "hamming") or self.write_score in _EXTRA_SCORES:
            return super().merged_key(pair, fractions)
        mean = fractions @ pair
        if self.write_score == "cosine":
            direction = fractions @ unit(pair)
            if direction.norm() == 0:
                return None
            radius = 1. if self.unit_keys else fractions @ pair.norm(dim=-1)
            return unit(direction) * radius
        return self.seed_keys(mean)

    def birth_key(self, group):
        name = self.write_score
        if name in ("dice", "kl", "jaccard", "hamming", "dot") or name in _EXTRA_SCORES:
            return super().birth_key(group)
        if name == "cosine":
            direction = unit(group).mean(0)
            if direction.norm() <= 1e-12:
                return None
            return unit(direction) * (1. if self.unit_keys else group.norm(dim=-1).mean())
        return group.mean(0)

    def split_keys(self, u, r, key):
        name = self.write_score
        if name not in ("cosine", "euclidean", "mahalanobis"):
            yield from super().split_keys(u, r, key)
            return

        x = unit(u) if name == "cosine" else u
        if name == "mahalanobis":
            x = u @ self.whitener.to(u)
        direction = unit(key)
        if name == "cosine":
            # Project onto the key's tangent plane before computing the shared principal split.
            x = x - (x @ direction)[:, None] * direction
        fission = sample_space_fission if len(x) < x.shape[1] else weighted_spectral_fission
        eps = torch.finfo(x.dtype).eps
        split = fission(x, r, split_scale=self.cfg.split_scale, eps=eps)
        if split["sigma1"] <= eps:
            return
        if name == "cosine":
            axis = split["u1"]
            axis = axis - axis.dot(direction) * direction
            if axis.norm() <= eps:
                return
            axis = unit(axis)
            radius = 1. if self.unit_keys else key.norm()
            for angle in self.cfg.split_angles or (0.5 * math.acos(self.cfg.merge_cosine), math.pi / 6):
                offset = math.sin(angle) * axis
                mid = math.cos(angle) * direction
                yield radius * torch.stack((mid + offset, mid - offset))
        else:
            children = torch.stack((split["child_a"], split["child_b"]))
            if name == "mahalanobis":
                children = torch.linalg.solve(self.whitener.to(u).T, children.T).T
            yield children


class SafeguardedGeometry(AssociationGeometry):
    @property
    def use_exemplar_updates(self):
        return self.write_score in ("hamming", "jaccard")

    def score(self, name, a, b):
        if name == "dice":
            p, q = a.sigmoid(), b.sigmoid()
            return (2 * (p @ q.T) + 1e-8) / (p.square().sum(1, keepdim=True) + q.square().sum(1) + 1e-8)
        return super().score(name, a, b)

    def project_keys(self, keys):
        if self.unit_keys:
            return unit(keys)
        if self.write_score == "dot":
            return self.bounded(keys)
        return keys.clamp(-self.cfg.logit_bound, self.cfg.logit_bound)

    def seed_keys(self, raw):
        return self.project_keys(super().project_keys(raw))
