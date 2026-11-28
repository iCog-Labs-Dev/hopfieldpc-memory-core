"""Experiment 6 controls for MARC, readout placement and live prototype views."""
import tkinter as tk
from dataclasses import replace
from tkinter import ttk

from gass_ui import GASSComparisonUI as SafeguardedUI
from .model import GASS_BANK_INIT_CHOICES, BottleneckGASSVariant, GASSVariant


class GASSComparisonUI(SafeguardedUI):
    variant_class = GASSVariant
    readout_variants = {"Decoder output": GASSVariant, "Encoder bottleneck": BottleneckGASSVariant}
    prototype_description = "Bottleneck values are arranged by feature index. Decoded images show pixel logits."

    def __init__(self, root, args):
        super().__init__(root, args)
        root.title("Experiment 6 — GASS memory readout on MNIST")
        gallery_controls = ttk.Frame(self.bank_header)
        gallery_controls.pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(gallery_controls, text="GASS prototype view").pack(side=tk.LEFT, padx=(0, 8))
        self.prototype_view = tk.StringVar(value="decoded")
        for value, caption in (("stored", "Stored values"), ("decoded", "Decoded image")):
            ttk.Radiobutton(gallery_controls, text=caption, variable=self.prototype_view, value=value,
                            command=self.change_prototype_view).pack(side=tk.LEFT, padx=(0, 8))
        bank_panel = ttk.Frame(self.preview_controls.master)
        bank_panel.pack(fill=tk.X, after=self.stability_label.master)
        ttk.Label(bank_panel, text="GASS readout layer").pack(side=tk.LEFT, padx=(8, 3))
        self.readout_layer = tk.StringVar(value="Decoder output")
        readout = ttk.Combobox(bank_panel, textvariable=self.readout_layer, values=tuple(self.readout_variants),
                               width=18, state="readonly")
        readout.pack(side=tk.LEFT)
        self.layer_profiles = {"Encoder bottleneck": ("cosine", "cosine", "squared", 0)}
        self.displayed_layer = self.readout_layer.get()
        readout.bind("<<ComboboxSelected>>", self.change_readout_layer)
        ttk.Label(bank_panel, text="GASS initialization").pack(side=tk.LEFT, padx=(8, 3))
        self.selectors["bank_init"] = tk.StringVar(value=self.settings.bank_init)
        ttk.Combobox(bank_panel, textvariable=self.selectors["bank_init"], values=GASS_BANK_INIT_CHOICES,
                     width=10, state="readonly").pack(side=tk.LEFT)
        ttk.Label(bank_panel, text="Window size (training images)").pack(side=tk.LEFT, padx=(12, 3))
        self.window_size = tk.StringVar(value=str(self.settings.window_size))
        ttk.Entry(bank_panel, textvariable=self.window_size, width=7).pack(side=tk.LEFT)
        ttk.Label(bank_panel, text="MARC checks").pack(side=tk.LEFT, padx=(12, 3))
        self.marc_checks = tk.StringVar(value="off" if self.settings.retrieval_margin is None else self.settings.marc_window)
        ttk.Combobox(bank_panel, textvariable=self.marc_checks, values=("off", "minibatch", "reference"),
                     width=10, state="readonly").pack(side=tk.LEFT)

    def bank_preview(self, variants):
        preview = super().bank_preview(variants)
        gass = variants["GASS"]
        if isinstance(gass, BottleneckGASSVariant):
            preview["stored_banks"] = {"GASS": gass.prototype_images(stored=True)}
        return preview

    def change_prototype_view(self):
        gallery = self.galleries["GASS"]
        gallery.show_stored = self.prototype_view.get() == "stored"
        gallery.render()

    def change_readout_layer(self, _event):
        layer = self.readout_layer.get()
        fields = ("read_score", "write_score", "retrieval_loss", "recall_steps")
        self.layer_profiles[self.displayed_layer] = tuple(self.selectors[field].get() for field in fields)
        for field, value in zip(fields, self.layer_profiles[layer]):
            self.selectors[field].set(value)
        self.displayed_layer = layer

    def settings_from_controls(self):
        try:
            window_size = int(self.window_size.get())
        except ValueError:
            raise ValueError("Enter a whole number for the window size.") from None
        initial = self.config_factory(quick=self.args.quick, memory_overrides={}).memory.initial_memories
        if window_size <= initial:
            raise ValueError(f"Window size must be greater than the initial memory count ({initial}).")
        checks = self.marc_checks.get()
        settings = replace(super().settings_from_controls(), window_size=window_size,
                           marc_window="minibatch" if checks == "off" else checks,
                           retrieval_margin=None if checks == "off" else (self.args.gass.retrieval_margin or 0.))
        self.variant_class = self.readout_variants[self.readout_layer.get()]
        return settings
