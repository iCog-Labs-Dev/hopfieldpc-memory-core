"""Experiment 5 score profiles, key geometry, and training schedule."""
import math
from dataclasses import dataclass

from hopfieldpc.gass import GASSConfig as CoreGASSConfig, ReferenceConfig as CoreReferenceConfig

from .geometry import loss_names, score_names


@dataclass(frozen=True)
class ReferenceConfig(CoreReferenceConfig):
    read_score: str = "cosine"
    write_score: str = "cosine"
    retrieval_loss: str = "squared"
    key_radius: float = 1.
    split_scale: float = 1.
    split_angles: tuple[float, ...] | None = None
    warmup_epochs: int = 1
    structural_interval: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.read_score not in score_names() or self.write_score not in score_names() or self.retrieval_loss not in loss_names():
            raise ValueError("Unknown GASS association score or retrieval loss.")
        if self.split_angles is not None and (not self.split_angles or not all(0 < x < math.pi/2 for x in self.split_angles)):
            raise ValueError("Split angles must be in (0, pi/2) radians.")
        if min(self.key_radius, self.split_scale) <= 0:
            raise ValueError("Key radius and split scale must be positive.")
        if type(self.warmup_epochs) is not int or type(self.structural_interval) is not int:
            raise ValueError("Epoch counts must be integers.")
        if self.warmup_epochs < 0 or self.structural_interval < 1:
            raise ValueError("Invalid warm-up or structural interval.")


@dataclass(frozen=True)
class GASSConfig(CoreGASSConfig, ReferenceConfig):
    read_score: str = "dice"
    write_score: str = "dice"
    retrieval_loss: str = "bernoulli_kl"

    def __post_init__(self):
        super().__post_init__()
        if self.recall_steps and self.read_score not in ("cosine", "dot"):
            raise ValueError("Frozen key iterations require a cosine or dot read score.")
