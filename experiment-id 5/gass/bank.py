"""Bind Experiment 5's key and loss profiles to the shared safeguarded bank."""
from hopfieldpc.gass import GASSBank as CoreGASSBank

from .config import GASSConfig
from .geometry import SafeguardedGeometry
from .reference import MNISTBankProfile


class GASSBank(MNISTBankProfile, CoreGASSBank):
    config_class = GASSConfig
    geometry_class = SafeguardedGeometry

    def online_update(self, queries, targets=None, *, dt=None, task_score=None, retrieval_reference=None):
        if getattr(self.cfg, "marc_window", None) == "reference" and retrieval_reference is None:
            raise ValueError("Reference-window MARC needs the training reference for each write.")
        return CoreGASSBank.online_update(self, queries, targets, dt=dt, retrieval_reference=retrieval_reference)
