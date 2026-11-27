"""Compare Direct, EMA, Fixed, and Dynamic MNIST reconstructions on clean and noisy digits.

All four autoencoders train on clean images with direct reconstruction MSE.
Direct displays the decoder output. After one warm-up epoch, the memory rows
read softmax(beta * Q @ M.T) @ M, where Q contains the raw decoder logits
and M contains the active prototype logits. Sigmoid maps the result to pixels.
"""

from __future__ import annotations
import argparse
import math
import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiment-id 5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mnist_reconstruction.model import (
    AutoencoderVariant, MnistAutoencoder, RevisedHopfieldBank, argument_parser,
    build_sample_corruption, digit_overlap, vector_images,
)
from mnist_reconstruction.ui import PrototypeGallery, RealtimeAutoencoderUI as ComparisonUI, draw_prototype
from config import get_config

__all__ = [
    "AutoencoderVariant", "MnistAutoencoder", "PrototypeGallery", "RealtimeAutoencoderUI",
    "RevisedHopfieldBank", "argument_parser", "build_sample_corruption", "digit_overlap",
    "draw_prototype", "main", "parse_args", "vector_images",
]


class RealtimeAutoencoderUI(ComparisonUI):
    config_factory = staticmethod(get_config)


def parse_args() -> argparse.Namespace:
    parser = argument_parser(__doc__)
    parser.add_argument("--beta-read", dest="similarity_beta", type=float,
                        help="Set the softmax inverse temperature for all three banks (default: bank configuration)")
    args = parser.parse_args()
    if args.similarity_beta is not None and not (math.isfinite(args.similarity_beta) and args.similarity_beta > 0):
        parser.error("--beta-read must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    RealtimeAutoencoderUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
