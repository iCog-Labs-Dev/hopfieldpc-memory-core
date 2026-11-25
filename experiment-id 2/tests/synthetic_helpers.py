"""Shared synthetic dataset factory for offline experiment-2 tests."""
from __future__ import annotations

import torch


def synthetic_block_dataset(
    n_train: int,
    n_test: int,
    channels: int,
    image_size: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """10-class synthetic images with distinct spatial block locations in [0, 1].

    Grayscale datasets get bright spatial patches per class.
    RGB datasets get distinct per-class color fills so global average pooling
    can separate all 10 classes in a few training epochs.
    """

    def _make(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        labels = torch.arange(n) % num_classes
        x = torch.rand(n, channels, image_size, image_size, generator=gen) * 0.15
        for cls_id in range(num_classes):
            r = (cls_id // 4) * 8 + 2
            col = (cls_id % 4) * 6 + 2
            if channels == 1:
                x[labels == cls_id, :, r : r + 6, col : col + 6] += 0.75
            else:
                # Distinct 10-class RGB color & stripe signature.
                r_val = 0.2 + 0.7 * ((cls_id * 3 % 10) / 9.0)
                g_val = 0.2 + 0.7 * (((cls_id * 7 + 2) % 10) / 9.0)
                b_val = 0.2 + 0.7 * (((cls_id * 5 + 4) % 10) / 9.0)
                color = torch.tensor([r_val, g_val, b_val], dtype=torch.float32).view(1, 3, 1, 1)
                x[labels == cls_id] = 0.85 * color + 0.10 * x[labels == cls_id]
        return x.clamp(0.0, 1.0), labels

    train_x, train_y = _make(n_train, 0)
    test_x, test_y = _make(n_test, 1)
    return train_x, train_y, test_x, test_y

