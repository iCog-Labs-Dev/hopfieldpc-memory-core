"""Dataset-adaptive CNN backbones keeping Hopfield at the pre-last layer.

All architectures preserve the exact Experiment 1/2 interface:
    images [B, C, H, W] -> encoder -> z [B, embed_dim]
    -> bridge (Identity for direct CNN, Hopfield retrieval for memory methods)
    -> ClassifierHead(embed_dim, num_classes) -> logits [B, num_classes]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DatasetSpec

from benchmarks.mnist_baseline import ClassifierHead


class GrayscaleCnnEncoder(nn.Module):
    """Regularized 2-stage double-conv encoder for 28x28 grayscale datasets.

    Uses BatchNorm2d and spatial/dense dropout so Fashion-MNIST and KMNIST
    surpass 80% test accuracy with a tight train-validation generalization gap.
    """

    def __init__(
        self,
        *,
        mean: tuple[float, ...] = (0.2860,),
        std: tuple[float, ...] = (0.3530,),
        c1: int = 32,
        c2: int = 64,
        hidden: int = 128,
        embed_dim: int = 64,
        image_size: int = 28,
        conv_dropout: float = 0.10,
        dense_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError(f"image_size must be divisible by 4, got {image_size}.")
        if len(mean) != 1 or len(std) != 1:
            raise ValueError("GrayscaleCnnEncoder expects 1-channel mean and std.")

        self.embed_dim = embed_dim
        self.image_size = image_size
        self.register_buffer(
            "pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 1)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(conv_dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(conv_dropout + 0.05),
        )
        flat_dim = c2 * (image_size // 4) ** 2
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dense_dropout),
            nn.Linear(hidden, embed_dim),  # linear signed z for centered cosine Hopfield
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.pixel_mean) / self.pixel_std
        x = self.block1(x)
        x = self.block2(x)
        return self.proj(x)


class ResidualBlock(nn.Module):
    """Compact 3x3 residual block with optional stride downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class RgbResNetEncoder(nn.Module):
    """Compact 3-stage residual CNN with Global Average Pooling for 32x32 RGB.

    Designed for CIFAR-10 and SVHN: residual BatchNorm blocks capture rich
    natural-image features while Global Average Pooling avoids large dense layers
    that cause train-validation overfitting gaps.
    """

    def __init__(
        self,
        *,
        mean: tuple[float, ...] = (0.4914, 0.4822, 0.4465),
        std: tuple[float, ...] = (0.2470, 0.2435, 0.2616),
        channels: tuple[int, int, int] = (32, 64, 128),
        embed_dim: int = 64,
        block_dropout: float = 0.10,
        dense_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("RgbResNetEncoder expects 3-channel mean and std.")
        c1, c2, c3 = channels
        self.embed_dim = embed_dim
        self.register_buffer(
            "pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c1, stride=1, dropout=block_dropout),
            ResidualBlock(c1, c1, stride=1, dropout=0.0),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, dropout=block_dropout),
            ResidualBlock(c2, c2, stride=1, dropout=0.0),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, dropout=block_dropout),
            ResidualBlock(c3, c3, stride=1, dropout=0.0),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dense_dropout),
            nn.Linear(c3, embed_dim),  # linear signed z for centered cosine Hopfield
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.pixel_mean) / self.pixel_std
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.proj(x)


class BackboneNet(nn.Module):
    """Unified encoder -> bridge -> head container compatible with MnistNet."""

    def __init__(
        self,
        encoder: nn.Module,
        bridge: nn.Module,
        head: ClassifierHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.head = head

    def forward_from_embedding(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        out = self.bridge(z)
        h, aux = out if isinstance(out, tuple) else (out, {})
        return self.head(h), z, h, aux

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.forward_from_embedding(self.encoder(images))


def build_encoder(
    spec: DatasetSpec,
    *,
    embed_dim: int = 64,
    rgb_channels: tuple[int, int, int] = (32, 64, 128),
) -> nn.Module:
    """Build the dataset-appropriate CNN encoder producing [B, embed_dim]."""
    if spec.in_channels == 1:
        return GrayscaleCnnEncoder(
            mean=spec.mean,
            std=spec.std,
            embed_dim=embed_dim,
            image_size=spec.image_size,
        )
    if spec.in_channels == 3:
        return RgbResNetEncoder(
            mean=spec.mean,
            std=spec.std,
            channels=rgb_channels,
            embed_dim=embed_dim,
        )
    raise ValueError(f"Unsupported channel count {spec.in_channels} for {spec.name}.")


def build_backbone(
    spec: DatasetSpec,
    *,
    embed_dim: int = 64,
    seed: int = 0,
    rgb_channels: tuple[int, int, int] = (32, 64, 128),
) -> BackboneNet:
    """Construct a seeded BackboneNet with Identity bridge at the pre-last layer."""
    torch.manual_seed(seed)
    encoder = build_encoder(spec, embed_dim=embed_dim, rgb_channels=rgb_channels)
    head = ClassifierHead(embed_dim=embed_dim, num_classes=spec.num_classes)
    return BackboneNet(encoder=encoder, bridge=nn.Identity(), head=head)
