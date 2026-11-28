"""Shared Tk controls and prototype gallery for the MNIST autoencoder comparisons."""
from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from abc import ABC, abstractmethod
from dataclasses import replace
from tkinter import ttk

import torch
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageTk

from .model import AutoencoderVariant, build_sample_corruption
from benchmarks.mnist_baseline import configure_threads, load_mnist, resolve_device

TILE_WIDTH, TILE_HEIGHT = 68, 76


def draw_prototype(sheet: Image.Image, slot: int, image: Image.Image, x: int, y: int) -> None:
    height = max(1, round(56 * image.height / image.width))
    sheet.paste(image.resize((56, height), Image.Resampling.NEAREST), (x + 6, y + 2))
    ImageDraw.Draw(sheet).text((x + 6, y + 59), f"#{slot}", fill="black")


class PrototypeGallery(ttk.LabelFrame):
    def __init__(self, parent: tk.Widget, name: str) -> None:
        super().__init__(parent, text=name)
        self.name = name
        self.images: dict[int, Image.Image] = {}
        self.stored_images: dict[int, Image.Image] | None = None
        self.show_stored = False
        self.photo: ImageTk.PhotoImage | None = None
        self.canvas = tk.Canvas(self, highlightthickness=0, background="white")
        scroll = ttk.Scrollbar(self, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.canvas.bind("<Map>", lambda _: self.render())

    def render(self) -> None:
        if not self.canvas.winfo_ismapped():
            return
        images = self.stored_images if self.show_stored and self.stored_images is not None else self.images
        self.canvas.delete("all")
        self.configure(text=f"{self.name}  ({len(images)} active)")
        if not images:
            self.canvas.create_text(10, 12, anchor="nw", text="Waiting for epoch 1 initialization")
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.photo = None
            return
        columns = max(1, self.canvas.winfo_width() // TILE_WIDTH)
        rows = max(images) // columns + 1
        sheet = Image.new("RGB", (columns * TILE_WIDTH, rows * TILE_HEIGHT), "white")
        for slot, image in images.items():
            draw_prototype(sheet, slot, image, slot % columns * TILE_WIDTH, slot // columns * TILE_HEIGHT)
        self.photo = ImageTk.PhotoImage(sheet, master=self.canvas)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class RealtimeAutoencoderUI(ABC):
    dataset_name = "MNIST"
    variant_names = ("Direct", "EMA", "Fixed", "Dynamic")
    variant_class = AutoencoderVariant
    axis_titles = ("Clean input", "Noisy input", "From clean input", "From noisy input")
    prototype_description = "28×28 prototype logits, one value per output pixel."

    @abstractmethod
    def config_factory(self, *, quick, memory_overrides):
        """Supply the experiment's training and memory settings."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.queue: queue.Queue[dict[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        root.title(f"Real-time {self.dataset_name} Autoencoder")
        root.geometry("1180x760")

        control = ttk.Frame(root, padding=8)
        control.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control, text="Start Training", command=self.start).pack(side=tk.LEFT)
        ttk.Button(control, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(control, text="Epochs").pack(side=tk.LEFT)
        self.epochs = tk.StringVar(value=str(args.epochs))
        ttk.Entry(control, textvariable=self.epochs, width=6).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(control, text="Learning rate").pack(side=tk.LEFT)
        self.learning_rate = tk.StringVar(value=str(args.learning_rate))
        ttk.Entry(control, textvariable=self.learning_rate, width=9).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(control, text="Noise").pack(side=tk.LEFT)
        self.noise_kind = tk.StringVar(value=args.noise)
        ttk.Combobox(
            control,
            textvariable=self.noise_kind,
            values=("gaussian", "salt_pepper", "occlusion"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(control, text="Noise level").pack(side=tk.LEFT)
        self.noise_level = tk.StringVar(value=str(args.noise_level))
        ttk.Entry(control, textvariable=self.noise_level, width=7).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(control, text="Apply", command=self.request_redraw).pack(side=tk.LEFT)

        status = ttk.Frame(root, padding=(8, 0, 8, 4))
        status.pack(side=tk.TOP, fill=tk.X)
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_text).pack(side=tk.LEFT)
        tabs = ttk.Notebook(root)
        tabs.pack(fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(tabs)
        banks_frame = ttk.Frame(tabs, padding=6)
        tabs.add(plot_frame, text="Reconstructions")
        tabs.add(banks_frame, text="Memory banks")
        preview_controls = ttk.Frame(plot_frame, padding=(8, 4))
        self.preview_controls = preview_controls
        preview_controls.pack(fill=tk.X)
        ttk.Label(preview_controls, text="Preview image").pack(side=tk.LEFT)
        self.preview_image = ttk.Combobox(preview_controls, width=4, state="disabled")
        self.preview_image.pack(side=tk.LEFT, padx=6)
        self.preview_image.bind("<<ComboboxSelected>>", lambda _: self.update_images())
        self.bank_header = ttk.Frame(banks_frame)
        self.bank_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(self.bank_header, text=self.prototype_description + "  "
                  "Negative = dark    Zero = gray    Positive = light    "
                  "Shared scale  x / (1 + |x|)").pack(anchor=tk.W)
        galleries = ttk.Frame(banks_frame)
        galleries.pack(fill=tk.BOTH, expand=True)
        self.galleries = {}
        for column, name in enumerate(self.variant_names[1:]):
            gallery = PrototypeGallery(galleries, name)
            galleries.columnconfigure(column, weight=1, uniform="bank")
            gallery.grid(row=0, column=column, sticky="nsew")
            self.galleries[name] = gallery
        galleries.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(11.5, 7.2), constrained_layout=True)
        self.axes = self.figure.subplots(len(self.variant_names), len(self.axis_titles))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        for column, title in enumerate(self.axis_titles):
            self.axes[0, column].set_title(title)
        for row, name in enumerate(self.variant_names):
            self.axes[row, 0].set_ylabel(name, fontsize=11)
        for axis in self.axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        self.canvas.draw()

        self.display_clean: torch.Tensor | None = None
        self.display_noisy: torch.Tensor | None = None
        self.display_generations: dict[str, tuple[torch.Tensor, ...]] = {}
        self.root.after(50, self.poll_queue)

    def request_redraw(self) -> None:
        if self.display_clean is not None:
            # A newer training preview may already be queued.
            self.queue.put({"redraw": True})

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.queue = queue.Queue()
        for gallery in self.galleries.values():
            gallery.images = {}
            gallery.stored_images = None
            gallery.render()
        self.stop_event.clear()
        self.worker = threading.Thread(target=self.train_worker, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status_text.set("Stopping after the current batch...")

    def load_images(self, cfg):
        train, _, test, _ = load_mnist(cfg.data_dir)
        train = train[:cfg.train_subset]
        return train, test[:64], train

    def create_variant(self, name, cfg, device):
        return self.variant_class(name, cfg, seed=0, device=device)

    def train_worker(self) -> None:
        try:
            memory_overrides = {"warmup_epochs": 1}
            if getattr(self.args, "similarity_beta", None) is not None:
                memory_overrides["beta_read"] = self.args.similarity_beta
            cfg = self.config_factory(quick=self.args.quick, memory_overrides=memory_overrides)
            cfg = replace(
                cfg,
                device=self.args.device,
                data_dir=self.args.data_dir,
                epochs=int(self.epochs.get()),
                batch_size=self.args.batch_size,
                eval_batch_size=max(self.args.batch_size, 64),
                learning_rate=float(self.learning_rate.get()),
                train_subset=(2000 if self.args.quick else None),
                test_subset=64,
            )
            configure_threads(cfg.num_threads)
            train_images, test_images, reference_images = self.load_images(cfg)

            device = resolve_device(cfg.device)
            variants = {
                name: self.create_variant(name, cfg, device)
                for name in self.variant_names
            }
            shuffle = torch.Generator().manual_seed(getattr(self.args, "seed", 0))

            for epoch in range(1, cfg.epochs + 1):
                if self.stop_event.is_set():
                    break
                order = torch.randperm(len(train_images), generator=shuffle)
                loss_sums = {name: 0.0 for name in variants}
                score_sums = {name: 0.0 for name in variants}

                for start in range(0, len(order), cfg.batch_size):
                    if self.stop_event.is_set():
                        break
                    batch = train_images[order[start : start + cfg.batch_size]]
                    for name, variant in variants.items():
                        mse, score = variant.train_batch_metrics(batch, device)
                        loss_sums[name] += len(batch) * mse
                        score_sums[name] += len(batch) * score

                    if start % (cfg.batch_size * 4) == 0:
                        self.publish_preview(epoch, variants, test_images, device)

                if self.stop_event.is_set():
                    break
                mean_losses = {name: total / len(order) for name, total in loss_sums.items()}
                changes = {}
                for name, variant in variants.items():
                    if variant.bank is None:
                        continue
                    stats = variant.finish_bank_epoch(
                        reference_images, cfg, device, epoch, score_sums[name] / len(order),
                    )
                    if name == "Dynamic" and stats:
                        changes = {key: stats[key] for key in ("pruned", "coalesced", "spawned")}

                active = {
                    name: int(variant.bank.active_count)
                    if variant.bank is not None and variant.bank.ready
                    else 0
                    for name, variant in variants.items()
                }
                self.publish_preview(epoch, variants, test_images, device)
                self.queue.put({
                    "epoch": epoch,
                    "losses": mean_losses,
                    "active": active,
                    "changes": changes,
                })

        except Exception as exc:
            self.queue.put({"error": f"{type(exc).__name__}: {exc}"})
        else:
            self.queue.put({"status": "Stopped" if self.stop_event.is_set() else "Finished"})

    def publish_preview(
        self,
        epoch: int,
        variants: dict[str, AutoencoderVariant],
        test_images: torch.Tensor,
        device: torch.device,
    ) -> None:
        clean = test_images[:8].clone()
        try:
            noise_level = float(self.noise_level.get())
        except ValueError:
            noise_level = 0.3
        kind = self.noise_kind.get()
        noisy = build_sample_corruption(clean, kind, noise_level, seed=epoch + getattr(self.args, "seed", 0))
        generations = {
            name: self.preview_outputs(variant, clean, noisy, device)
            for name, variant in variants.items()
        }
        self.queue.put({
            "epoch": epoch,
            "clean": clean,
            "noisy": noisy,
            "generations": generations,
            **self.bank_preview(variants),
            "status": f"Training epoch {epoch}",
        })

    def bank_preview(self, variants):
        return {"banks": {name: variant.prototype_images() for name, variant in variants.items()
                          if variant.bank is not None}}

    def preview_outputs(self, variant, clean, noisy, device):
        clean_recon, _ = variant.generate(clean, device)
        noisy_recon, _ = variant.generate(noisy, device)
        return clean_recon, noisy_recon

    def poll_queue(self) -> None:
        dirty_galleries = set()
        redraw = False
        try:
            while True:
                event = self.queue.get_nowait()
                if "error" in event:
                    self.status_text.set(str(event["error"]))
                    continue
                if event.get("status"):
                    self.status_text.set(str(event["status"]))
                if "epoch" in event:
                    losses = event.get("losses")
                    active = event.get("active")
                    parts = [f"Epoch {event['epoch']}"]
                    if losses:
                        parts.append(
                            " | ".join(
                                f"{name} MSE {float(value):.4f}"
                                for name, value in losses.items()
                            )
                        )
                    if active:
                        parts.append(
                            " | ".join(
                                f"{name} {int(value)} active"
                                for name, value in active.items()
                                if name != "Direct"
                            )
                        )
                    if event.get("changes"):
                        parts.append(" | ".join(f"{count} {kind}" for kind, count in event["changes"].items()))
                    self.status_text.set("  -  ".join(parts))

                if "clean" in event:
                    self.display_clean = event["clean"]  # type: ignore[assignment]
                    self.display_noisy = event["noisy"]  # type: ignore[assignment]
                    self.display_generations = event["generations"]  # type: ignore[assignment]
                if "generations" in event or event.get("redraw"):
                    redraw = True
                for name, images in event.get("banks", {}).items():
                    self.galleries[name].images = images
                    self.galleries[name].stored_images = event.get("stored_banks", {}).get(name)
                    dirty_galleries.add(name)
        except queue.Empty:
            pass
        for name in dirty_galleries:
            self.galleries[name].render()
        if redraw:
            self.update_images()
        self.root.after(50, self.poll_queue)

    def reconstruction_panels(self, name, index):
        target = self.display_clean[index]
        return [(images[index], f"MSE {F.mse_loss(images[index], target).item():.4f}")
                for images in self.display_generations[name]]

    def update_images(self) -> None:
        if self.display_clean is None or self.display_noisy is None or not len(self.display_clean):
            return
        clean = self.display_clean
        noisy = self.display_noisy
        if len(self.preview_image["values"]) != len(clean):
            previous = self.preview_image.current()
            self.preview_image.configure(values=tuple(range(1, len(clean) + 1)), state="readonly")
            self.preview_image.current(max(0, min(previous, len(clean) - 1)))
        index = self.preview_image.current()
        for row, name in enumerate(self.variant_names):
            panels = [(clean[index], ""), (noisy[index], ""), *self.reconstruction_panels(name, index)]
            for column, (image, caption) in enumerate(panels):
                axis = self.axes[row, column]
                axis.clear()
                image = image.detach()
                if image.shape[0] == 1:
                    axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
                else:
                    axis.imshow(image.permute(1, 2, 0).numpy())
                axis.set_xticks([])
                axis.set_yticks([])
                if caption:
                    axis.set_xlabel(caption, fontsize=8)
                if row == 0:
                    axis.set_title(self.axis_titles[column])
                if column == 0:
                    axis.set_ylabel(name, fontsize=11)
        self.canvas.draw_idle()
