"""Compare direct, EMA, fixed, and dynamic MNIST reconstruction with the original bank."""
from __future__ import annotations

import tkinter as tk

from mnist_autoencoder_ui_revised import (
    AutoencoderVariant as ReconstructionVariant,
    MnistAutoencoder,
    PrototypeGallery,
    RealtimeAutoencoderUI as ComparisonUI,
    argument_parser,
    build_sample_corruption,
    digit_overlap,
    draw_prototype,
    get_config,
    parse_args,
    vector_images,
)
from hopfieldpc import GlobalDynamicHopfieldBank

__all__ = [
    "AutoencoderVariant", "MnistAutoencoder", "PrototypeGallery", "RealtimeAutoencoderUI",
    "argument_parser", "build_sample_corruption", "digit_overlap", "draw_prototype",
    "get_config", "main", "parse_args", "vector_images",
]


class AutoencoderVariant(ReconstructionVariant):
    """Read and update decoder logits with the original Experiment 4 noise model."""

    bank_class = GlobalDynamicHopfieldBank

    def train_batch(self, images, device):
        mse, _ = self.train_batch_metrics(images, device)
        return mse


class RealtimeAutoencoderUI(ComparisonUI):
    variant_class = AutoencoderVariant


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    RealtimeAutoencoderUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
