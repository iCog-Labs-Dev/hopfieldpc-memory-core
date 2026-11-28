"""Model-supplied key spaces for continuous GASS memory."""
from abc import ABC, abstractmethod

import torch
from torch import nn

from ..functional import squared_distance


class KeyGeometry(nn.Module, ABC):
    """Define a slot-local similarity and a projection onto the feasible key domain.

    Scores and distances have shape (queries, slots). The default structural
    proposals use exemplars and squared Euclidean distance in key coordinates.
    Override distance and pair_affinity for another geometry. Merge/separation
    thresholds operate on symmetric affinities in [-1, 1].
    """

    use_exemplar_updates = False
    # Starting step for projected key descent. Backtracking can shorten it.
    initial_key_step = 1.0

    @abstractmethod
    def read_similarity(self, queries, keys):
        """Return finite scores on valid queries and feasible keys."""

    def write_similarity(self, queries, keys):
        return self.read_similarity(queries, keys)

    @abstractmethod
    def project_keys(self, keys):
        """Project each row onto the model's bounded key domain."""

    def refined_read_logits(self, queries, keys, weights, beta, steps):
        raise ValueError("This geometry does not support iterative recall.")

    def valid_queries(self, raw, *, fitting=False):
        return raw.isfinite().all(-1)

    def fit_metric(self, queries):
        """Optionally fit the geometry when the bank initializes."""

    def feasible(self, keys):
        return keys.isfinite().all(-1) & torch.isclose(keys, self.project_keys(keys)).all(-1)

    def seed_keys(self, queries):
        return self.project_keys(queries)

    def distance(self, queries, keys):
        return squared_distance(queries, keys)

    def pair_affinity(self, a, b):
        return (-0.5 * (self.distance(a, b) + self.distance(b, a).T)).exp()

    def redundancy(self, keys):
        return self.pair_affinity(keys, keys)

    def propose_keys(self, u, responsibility, conditional, old):
        high = responsibility.topk(min(3, len(u)), dim=0).indices.flatten()
        spaced = torch.linspace(0, len(u) - 1, min(len(u), 12), device=u.device).long()
        exemplars = self.seed_keys(u[torch.cat((high, spaced)).unique()])
        exemplars = exemplars[self.feasible(exemplars)]
        if not len(exemplars):
            return old
        scores = conditional.T @ self.write_similarity(u, exemplars)
        return exemplars[scores.argmax(1)]

    def key_candidates(self, u, responsibility, old):
        if not len(u):
            return old
        mass = responsibility.sum(0)
        conditional = responsibility / torch.where(mass > 0, mass, 1)
        candidate = self.propose_keys(u, responsibility, conditional, old)
        old_score = (conditional * self.write_similarity(u, old)).sum(0)
        new_score = (conditional * self.write_similarity(u, candidate)).sum(0)
        take = (mass > 0) & self.feasible(candidate) & new_score.isfinite() & (new_score >= old_score)
        return torch.where(take[:, None], candidate, old)

    def merged_key(self, pair, fractions):
        return pair[fractions.argmax()]

    def birth_key(self, group):
        candidates = self.seed_keys(group)
        candidates = candidates[self.feasible(candidates)]
        if len(candidates):
            return candidates[self.distance(group, candidates).sum(0).argmin()]
        return None

    def split_keys(self, u, r, key):
        support = u[r > 0]
        if len(support) >= 2:
            first = self.seed_keys(u[r.argmax()].unsqueeze(0))
            distances = self.distance(support, first).flatten()
            if distances.max() > 1e-12:
                second = self.seed_keys(support[distances.argmax()].unsqueeze(0))
                yield torch.cat((first, second))
