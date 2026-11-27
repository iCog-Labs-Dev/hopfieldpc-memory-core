"""Bind Experiment 5's key and loss profiles to the shared reference bank."""
import torch

from hopfieldpc.gass import ReferenceBank as CoreReferenceBank

from .config import ReferenceConfig
from .geometry import AssociationGeometry, retrieval_loss


class MNISTBankProfile:
    def __init__(self, dim, initial, minimum, maximum, config=None, *, key_dim=None, metric=None):
        cfg = config or self.config_class()
        geometry = self.geometry_class(cfg, dim if key_dim is None else key_dim, metric)
        super().__init__(dim, initial, minimum, maximum, cfg, key_dim=key_dim,
                         geometry=geometry, loss=retrieval_loss(cfg.retrieval_loss))

    def online_update(self, queries, targets=None, *, dt=None, task_score=None):
        # The reconstruction UI supplies a display score alongside each batch.
        return super().online_update(queries, targets, dt=dt)

    @torch.no_grad()
    def refine_direction(self, queries, steps=1):
        if self.cfg.read_score != "cosine" or not self.ready or steps < 0:
            raise ValueError("Directional refinement requires a ready cosine read and nonnegative step count.")
        raw = queries.to(self.keys)
        if not self.geometry.valid_queries(raw).all():
            raise ValueError("Directional refinement needs finite, nonzero queries.")
        state, _, _ = self.geometry.tied_recall(raw, self.keys[self.mask], self.weights[self.mask],
                                                self.cfg.beta_r, steps)
        return state


class ReferenceBank(MNISTBankProfile, CoreReferenceBank):
    config_class = ReferenceConfig
    geometry_class = AssociationGeometry
