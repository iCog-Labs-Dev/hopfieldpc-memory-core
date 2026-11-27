"""Tk controls for the two GASS reconstruction experiments."""
from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import ttk

import torch

from mnist_reconstruction.model import AutoencoderVariant
from mnist_reconstruction.ui import RealtimeAutoencoderUI
from gass_autoencoder import GASSVariant, ReferenceVariant, comparison_config
from gass.geometry import loss_names, score_names


class ReferenceComparisonUI(RealtimeAutoencoderUI):
    variant_names = ("Direct", "EMA", "Dynamic", "GASS")
    variant_class = ReferenceVariant

    def __init__(self, root, args):
        self.settings = args.gass
        self.gass_bank = None
        self.displayed_report = None
        super().__init__(root, args)
        self.gass_summary = ttk.Label(self.preview_controls)
        self.gass_summary.pack(side=tk.LEFT, padx=12)
        panel = ttk.Frame(self.preview_controls.master)
        panel.pack(fill=tk.X, after=self.preview_controls)
        self.selectors = {}
        for field, caption, choices in (("read_score", "Read score", score_names()), ("write_score", "Write score", score_names()),
                                        ("retrieval_loss", "Retrieval loss", loss_names())):
            ttk.Label(panel, text=caption).pack(side=tk.LEFT, padx=(8, 3))
            var = tk.StringVar(value=getattr(self.settings, field))
            self.selectors[field] = var
            ttk.Combobox(panel, textvariable=var, values=choices, width=12, state="readonly").pack(side=tk.LEFT)
        for field, caption in (("value_aware", "Value-aware writes"), ("momentum", "Velocity filter")):
            var = tk.BooleanVar(value=getattr(self.settings, field))
            self.selectors[field] = var
            ttk.Checkbutton(panel, text=caption, variable=var).pack(side=tk.LEFT, padx=6)
        self.temperatures = {}
        for field, caption in (("read_beta", "Read β"), ("write_beta", "Write β")):
            value = getattr(self.settings, field)
            var = tk.StringVar(value=f"{self.settings.beta if value is None else value:g}")
            self.temperatures[field] = var
            ttk.Label(panel, text=caption).pack(side=tk.LEFT, padx=(6,2))
            ttk.Entry(panel, textvariable=var, width=5).pack(side=tk.LEFT)
        root.title("MNIST autoencoder comparison — Direct, EMA, Dynamic and GASS")

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            settings = {key: var.get() for key, var in self.selectors.items()}
            settings.update({key: float(var.get()) for key,var in self.temperatures.items()})
            self.settings = replace(self.args.gass, **settings)
        except ValueError as exc:
            self.status_text.set(str(exc))
            return
        super().start()

    def config_factory(self, *, quick, memory_overrides):
        return comparison_config(self.args, quick, memory_overrides)

    def load_images(self, cfg):
        train, test, _ = super().load_images(cfg)
        gen = torch.Generator().manual_seed(self.args.seed)
        indices = torch.randperm(len(train), generator=gen)[:self.settings.window_size]
        if len(indices) <= cfg.memory.initial_memories:
            raise ValueError("The training reference window must contain more images than initial slots.")
        self.reference = train[indices]
        return train, test, self.reference

    def create_variant(self, name, cfg, device):
        if name == "GASS":
            metric = self.args.metric if "mahalanobis" in (self.settings.read_score, self.settings.write_score) else None
            variant = self.variant_class(cfg, self.args.seed, device, self.settings, metric)
            self.gass_bank = variant.bank
            if self.settings.warmup_epochs == 0:
                variant.initialize_bank(self.reference, cfg, device)
            return variant
        return AutoencoderVariant(name, cfg, self.args.seed, device)

    def poll_queue(self):
        super().poll_queue()
        # The training thread publishes the report only after the frozen-window pass.
        report = self.gass_bank.last_structure if self.gass_bank is not None else None
        if report is self.displayed_report:
            return
        self.displayed_report = report
        summary = ""
        if report and "after" in report:
            tried = len(report["trials"])
            accepted = sum(trial["accepted"] for trial in report["trials"])
            summary = (f"GASS epoch {report['epoch']}    Objective {report['before']:.4f} → {report['after']:.4f}"
                       f"    {accepted}/{tried} edits accepted    {report['pruned']} pruned,"
                       f" {report['coalesced']} merged, {report['spawned']} added")
        self.gass_summary.configure(text=summary)


class GASSComparisonUI(ReferenceComparisonUI):
    variant_class = GASSVariant

    def __init__(self, root, args):
        self.preview_stability = ""
        super().__init__(root, args)
        root.title("MNIST autoencoder comparison — safeguarded GASS")
        recall_panel = ttk.Frame(self.preview_controls.master)
        recall_panel.pack(fill=tk.X, after=self.preview_controls)
        ttk.Label(recall_panel, text="Frozen key iterations (cosine or dot)").pack(side=tk.LEFT, padx=(8, 3))
        self.selectors["recall_steps"] = tk.IntVar(value=self.settings.recall_steps)
        ttk.Combobox(recall_panel, textvariable=self.selectors["recall_steps"], width=4,
                     values=sorted({0, 1, 3, 5, 10, self.settings.recall_steps}), state="readonly").pack(side=tk.LEFT)
        self.stability_label = ttk.Label(recall_panel)
        self.stability_label.pack(side=tk.LEFT, padx=12)

    @torch.no_grad()
    def preview_outputs(self, variant, clean, noisy, device):
        if not isinstance(variant, GASSVariant):
            return super().preview_outputs(variant, clean, noisy, device)
        variant.net.eval()
        pair = torch.cat([clean, noisy]).to(device)
        cue = variant.net.output_logits(variant.encode(pair))
        recalled = variant.retrieve_logits(cue)
        clean_cue, noisy_cue = cue.chunk(2)
        clean_recall, noisy_recall = recalled.chunk(2)
        rms = variant.paired_logit_rms(clean_cue, noisy_cue, clean_recall, noisy_recall)
        self.preview_stability = (f"Clean vs. corrupted preview · cue logit RMS {rms['cue_rms']:.4f} · "
                                  f"recalled logit RMS {rms['recall_rms']:.4f}" if variant.bank.ready else "GASS warming up")
        return clean_recall.sigmoid().cpu(), noisy_recall.sigmoid().cpu()

    def poll_queue(self):
        super().poll_queue()
        if hasattr(self, "stability_label"):
            self.stability_label.configure(text=self.preview_stability)
