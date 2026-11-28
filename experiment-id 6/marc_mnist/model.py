"""Experiment 6 memory initialization, reference-window MARC and bottleneck readout."""
from dataclasses import dataclass, replace

import torch

from gass import GASSConfig
from gass_autoencoder import GASSVariant as SafeguardedVariant


GASS_BANK_INIT_CHOICES = ("warm-up", "random")


@dataclass(frozen=True)
class ReconstructionConfig(GASSConfig):
    window_size: int = 256
    edit_budget: int = 5
    merge_cosine: float = 0.90
    support_fraction: float = 0.10
    min_child_mass: float = 1.0
    read_beta: float | None = 80.0
    bank_init: str = "warm-up"
    marc_window: str = "minibatch"

    def __post_init__(self):
        super().__post_init__()
        if self.bank_init not in GASS_BANK_INIT_CHOICES:
            raise ValueError("Choose warm-up or random for GASS initialization.")
        if self.marc_window not in ("minibatch", "reference"):
            raise ValueError("Choose minibatch or reference for the MARC window.")
        if self.marc_window == "reference" and self.retrieval_margin is None:
            object.__setattr__(self, "retrieval_margin", 0.)


class GASSVariant(SafeguardedVariant):
    @property
    def uses_reference_for_marc(self):
        return getattr(self.settings, "marc_window", "minibatch") == "reference"

    def train_batch_metrics(self, images, device, *, retrieval_reference=None):
        if self.uses_reference_for_marc and self.bank.ready and retrieval_reference is None:
            if self.reference_images is None:
                raise ValueError("Supply the structural training images for reference-window MARC.")
            features = self.training_embeddings(self.reference_images, self.training_config, device)
            retrieval_reference = features, features
        return super().train_batch_metrics(images, device, retrieval_reference=retrieval_reference)

    def __init__(self, cfg, seed, device, settings, metric=None):
        random_init = getattr(settings, "bank_init", "warm-up") == "random"
        if random_init:
            settings = replace(settings, warmup_epochs=0)
        super().__init__(cfg, seed, device, settings, metric)
        self.training_config = cfg
        self.reference_images = None
        if random_init:
            if settings.read_score == settings.write_score == "dice":
                # Dice normalizes over all pixels. Scale the first trial step so a
                # random key can learn the stroke pattern of its stored value.
                self.bank.geometry.initial_key_step = self.output_dim
            # Draw tied key/value logits without changing the training RNG stream.
            generator = torch.Generator().manual_seed(70123 + seed)
            logits = torch.randn(self.bank.initial, self.output_dim, generator=generator)
            if not self.bank.initialize_from_embeddings(logits, seed=seed):
                raise ValueError("Random prototypes could not satisfy the configured key constraints.")


class BottleneckGASSVariant(GASSVariant):
    """Store encoder embeddings and decode recalled values into image logits."""

    def __init__(self, cfg, seed, device, settings, metric=None):
        self.output_dim = cfg.memory.dim
        super().__init__(cfg, seed, device, settings, metric)

    def memory_features(self, embedding):
        return embedding

    def features_to_logits(self, features):
        return self.net.output_logits(features)
