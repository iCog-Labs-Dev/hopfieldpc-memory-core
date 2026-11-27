"""Autoencoder adapters for the reference and safeguarded GASS banks."""
from importlib import import_module

from gass import GASSBank, ReferenceBank
from mnist_reconstruction.model import AutoencoderVariant


class ReferenceVariant(AutoencoderVariant):
    memory_type = ReferenceBank

    def __init__(self, cfg, seed, device, settings, metric=None):
        self.settings = settings
        self.metric = metric
        super().__init__("GASS", cfg, seed, device)

    def bank_class(self, memory_cfg, *, online_update_mode):
        return self.memory_type(memory_cfg.dim, memory_cfg.initial_memories, memory_cfg.min_active_memories,
                                memory_cfg.max_memories, self.settings, metric=self.metric)

    def retrieve_logits(self, logits):
        return self.bank.read(logits.flatten(1)).reshape_as(logits)

    def finish_bank_epoch(self, images, cfg, device, epoch, score):
        if epoch < self.settings.warmup_epochs:
            return {}
        if self.bank.ready and (epoch - self.settings.warmup_epochs) % self.settings.structural_interval:
            return {}
        features = self.training_embeddings(images, cfg, device)
        if not self.bank.ready:
            self.bank.initialize_from_embeddings(features, seed=self.seed)
            return {}
        return self.bank.adapt_structure(features, epoch=epoch)


class GASSVariant(ReferenceVariant):
    memory_type = GASSBank

    @staticmethod
    def benchmark_ablations(variants):
        variants = {name: dict(overrides, recall_steps=0) for name, overrides in variants.items() if name != "GASS heuristic"}
        variants["GASS cosine squared"] = dict(read_score="cosine", write_score="cosine", retrieval_loss="squared", recall_steps=0)
        return variants

    @staticmethod
    def paired_logit_rms(clean, noisy, clean_read, noisy_read):
        return {"cue_rms": float((clean - noisy).square().mean().sqrt()),
                "recall_rms": float((clean_read - noisy_read).square().mean().sqrt())}


def comparison_config(args, quick, overrides):
    memory = dict(overrides)
    for attr in ("initial_memories", "min_active_memories", "max_memories"):
        if getattr(args, attr) is not None:
            memory[attr] = getattr(args, attr)
    # Use the same baseline preset as Experiment 4. The experiment folder contains a space.
    get_config = import_module("experiment-id 4.config").get_config
    return get_config(quick=quick, memory_overrides=memory)
