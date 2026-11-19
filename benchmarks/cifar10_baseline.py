"""Task 13 non-Hopfield CIFAR-10 baseline: 3-block RGB CNN with an Identity bridge.

Mirrors the exact modular interface of ``benchmarks.mnist_baseline`` and matches
the repository's CIFAR-10 Task 13 specification:
- ``load_cifar10`` / ``make_cifar10_splits`` (images in [0, 1] with shape [N, 3, 32, 32])
- Reuses ``corrupt_images`` / ``make_corrupted_test_set`` (gaussian, salt_pepper, occlusion)
- ``Cifar10Encoder`` -> ``bridge`` (Identity here) -> ``ClassifierHead``
- ``build_cifar10_baseline_net``, ``train_cifar10_baseline``, ``get_or_train_baseline``, ``encode``
- Reuses ``resolve_device``, ``configure_threads``, ``environment_info``, ``ClassifierHead``, ``encode``

Architecture (bridge slot between ``z`` and ``head`` matches ``MnistNet``):
    Block 1: Conv(3->32, 3x3)+ReLU -> Conv(32->32, 3x3)+ReLU -> MaxPool2 -> Dropout(0.2)
    Block 2: Conv(32->64, 3x3)+ReLU -> Conv(64->64, 3x3)+ReLU -> MaxPool2 -> Dropout(0.2)
    Block 3: Conv(64->128, 3x3)+ReLU -> Conv(128->128, 3x3)+ReLU -> MaxPool2 -> Dropout(0.2)
    FC:      Flatten(2048) -> Linear(2048, 256)+ReLU+Dropout(0.3) -> Linear(256, 128) = z (signed)
    Bridge:  Identity(z) = h (replaced by Hopfield retrieval in memory experiments)
    Head:    Linear(128, 10)

Run standalone from the repository root:
    python -m benchmarks.cifar10_baseline --quick
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.mnist_baseline import (
    NOISE_TYPES,
    ClassifierHead,
    configure_threads,
    corrupt_images,
    corruption_seed,
    encode,
    environment_info,
    make_corrupted_test_set,
    resolve_device,
)

CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
CIFAR10_NUM_CLASSES: int = 10
CIFAR10_IMAGE_SIZE: int = 32
CIFAR10_HIDDEN_DIM: int = 256
CIFAR10_EMBED_DIM: int = 128
NUM_CLASSES: int = CIFAR10_NUM_CLASSES
IMAGE_SIZE: int = CIFAR10_IMAGE_SIZE


@dataclass
class Cifar10Splits:
    """Train/validation/test tensors for one seed. Images are [N, 3, 32, 32] in [0, 1]."""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def load_cifar10_tensors(
    data_dir: str | Path,
    *,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one CIFAR-10 split directly as [N, 3, 32, 32] float32 in [0, 1]."""

    extract_dir = Path(data_dir) / "cifar-10-batches-py"
    if extract_dir.exists():
        batch_names = (
            [extract_dir / f"data_batch_{i}" for i in range(1, 6)]
            if train
            else [extract_dir / "test_batch"]
        )
        records = []
        for batch_path in batch_names:
            with batch_path.open("rb") as f:
                records.append(pickle.load(f, encoding="bytes"))
        pixels = np.concatenate([r[b"data"] for r in records]).reshape(
            -1, 3, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE
        )
        labels_np = np.concatenate([r[b"labels"] for r in records]).astype(np.int64)
        images = torch.from_numpy(pixels).float().div_(255.0)
        labels = torch.from_numpy(labels_np)
        return images, labels

    import torchvision

    dataset = torchvision.datasets.CIFAR10(root=str(data_dir), train=train, download=True)
    images = (
        torch.from_numpy(dataset.data)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .div_(255.0)
    )
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    return images, labels


def load_cifar10(data_dir: str | Path) -> tuple[torch.Tensor, ...]:
    """Load (train_images, train_labels, test_images, test_labels) for CIFAR-10."""

    train_images, train_labels = load_cifar10_tensors(data_dir, train=True)
    test_images, test_labels = load_cifar10_tensors(data_dir, train=False)
    return train_images, train_labels, test_images, test_labels


def make_cifar10_splits(
    data: Sequence[torch.Tensor],
    *,
    seed: int,
    val_size: int = 5000,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> Cifar10Splits:
    """Split official 50k CIFAR-10 train set into seeded train/validation tensors."""

    train_images, train_labels, test_images, test_labels = data
    num_train = train_images.shape[0]
    if val_size <= 0 or val_size >= num_train:
        raise ValueError(f"val_size ({val_size}) must be in (0, {num_train}).")

    perm = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    num_test = (
        test_images.shape[0]
        if test_subset is None
        else min(test_subset, test_images.shape[0])
    )
    return Cifar10Splits(
        train_images=train_images[train_idx],
        train_labels=train_labels[train_idx],
        val_images=train_images[val_idx],
        val_labels=train_labels[val_idx],
        test_images=test_images[:num_test],
        test_labels=test_labels[:num_test],
    )


def corrupt_cifar10_images(
    images: torch.Tensor,
    kind: str,
    level: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply pixel-space [0, 1] corruptions to RGB images [N, 3, H, W]."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape [N, 3, H, W], got {tuple(images.shape)}.")
    return corrupt_images(images, kind, level, generator)


def make_corrupted_cifar10_test_set(
    images: torch.Tensor,
    *,
    seed: int,
    kind: str,
    level: float,
    stream: int = 0,
) -> torch.Tensor:
    """Corrupt a CIFAR-10 test set deterministically from (seed, kind, level, stream)."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape [N, 3, H, W], got {tuple(images.shape)}.")
    return make_corrupted_test_set(images, seed=seed, kind=kind, level=level, stream=stream)


class Cifar10Encoder(nn.Module):
    """Task 13 CIFAR-10 CNN encoder: [B, 3, 32, 32] in [0, 1] -> signed z [B, embed_dim].

    Matches the 3-block (32/32, 64/64, 128/128) + 256 -> 128 architecture with
    dropout regularization and in-encoder CIFAR-10 RGB normalization.
    """

    def __init__(
        self,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        hidden: int = CIFAR10_HIDDEN_DIM,
        embed_dim: int = CIFAR10_EMBED_DIM,
        image_size: int = CIFAR10_IMAGE_SIZE,
        conv_dropout: float = 0.2,
        dense_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}.")
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.conv_dropout = conv_dropout
        self.dense_dropout = dense_dropout

        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        flat = c3 * (image_size // 8) ** 2
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(conv_dropout)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(conv_dropout)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(conv_dropout)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat, hidden)
        self.drop_fc = nn.Dropout(dense_dropout)
        self.fc2 = nn.Linear(hidden, embed_dim)  # L7: no activation -> signed z

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.pixel_mean) / self.pixel_std
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2b(x))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3b(x))
        x = self.drop3(self.pool3(x))

        x = self.drop_fc(F.relu(self.fc1(self.flatten(x))))
        return self.fc2(x)


class Cifar10Net(nn.Module):
    """encoder -> bridge -> head container matching ``MnistNet``."""

    def __init__(
        self,
        encoder: Cifar10Encoder,
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


def build_cifar10_baseline_net(
    seed: int,
    *,
    c1: int = 32,
    c2: int = 64,
    c3: int = 128,
    hidden: int = CIFAR10_HIDDEN_DIM,
    embed_dim: int = CIFAR10_EMBED_DIM,
    conv_dropout: float = 0.2,
    dense_dropout: float = 0.3,
    num_classes: int = CIFAR10_NUM_CLASSES,
) -> Cifar10Net:
    """Build seeded CIFAR-10 CNN baseline with Identity bridge."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = Cifar10Encoder(
            c1=c1,
            c2=c2,
            c3=c3,
            hidden=hidden,
            embed_dim=embed_dim,
            conv_dropout=conv_dropout,
            dense_dropout=dense_dropout,
        )
        head = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes)
    return Cifar10Net(encoder, nn.Identity(), head)


@torch.no_grad()
def evaluate_cifar10_baseline(
    net: Cifar10Net,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 256,
    device: torch.device,
) -> float:
    """Evaluate classification accuracy of ``net`` on CIFAR-10 images."""

    net.eval()
    correct = 0
    total = 0
    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].to(device)
        y = labels[start : start + batch_size]
        logits, _, _, _ = net(x)
        correct += int((logits.argmax(dim=-1).cpu() == y).sum().item())
        total += len(y)
    return correct / max(total, 1)


def train_cifar10_baseline(
    net: Cifar10Net,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device,
    on_epoch_end: Callable[[int], None] | None = None,
) -> float:
    """Train the CIFAR-10 baseline with Adam + L2 weight decay; return wall seconds."""

    net.to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(images.shape[0], generator=generator)
        total_loss = 0.0
        total = 0
        for start in range(0, images.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            x = images[idx].to(device)
            y = labels[idx].to(device)
            logits, _, _, _ = net(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * int(idx.numel())
            total += int(idx.numel())
        print(f"    epoch {epoch}/{epochs}  train loss {total_loss / max(total, 1):.4f}")
        if on_epoch_end is not None:
            net.eval()
            on_epoch_end(epoch)
            net.train()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    net.eval()
    return time.perf_counter() - t0


def checkpoint_fingerprint(
    cfg: dict[str, Any],
    seed: int,
    num_train: int,
) -> dict[str, Any]:
    """Settings a cached CIFAR-10 backbone must match to be reused."""

    return {
        "seed": seed,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg.get("weight_decay", 1e-4),
        "val_size": cfg["val_size"],
        "num_train": num_train,
        "architecture": [32, 32, 64, 64, 128, 128, 256, 128],
    }


def get_or_train_baseline(
    cfg: dict[str, Any],
    splits: Cifar10Splits,
    seed: int,
    device: torch.device,
    *,
    retrain: bool = False,
) -> tuple[Cifar10Net, float, bool]:
    """Load ``backbone_seed{seed}.pt`` if matching, else train and cache it."""

    path = Path(cfg["reports_dir"]) / f"backbone_seed{seed}.pt"
    fingerprint = checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0])
    net = build_cifar10_baseline_net(seed)

    if path.exists() and not retrain:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") == fingerprint:
            net.load_state_dict(payload["state_dict"])
            print(f"  seed {seed}: loaded cached backbone {path}")
            return net.to(device).eval(), float(payload["train_time_s"]), True
        print(f"  seed {seed}: cached backbone {path} has different settings; retraining")

    print(f"  seed {seed}: training CIFAR-10 baseline on {splits.train_images.shape[0]} clean images")
    train_time_s = train_cifar10_baseline(
        net,
        splits.train_images,
        splits.train_labels,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
        seed=seed,
        device=device,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "fingerprint": fingerprint,
            "train_time_s": train_time_s,
        },
        path,
    )
    return net.eval(), train_time_s, False


# Standard module-level aliases matching ``benchmarks.mnist_baseline``
make_splits = make_cifar10_splits
build_baseline_net = build_cifar10_baseline_net
evaluate_baseline = evaluate_cifar10_baseline
train_baseline = train_cifar10_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cifar10_baseline/results.json"),
    )
    args = parser.parse_args()

    configure_threads(4)
    device = resolve_device(args.device)
    data = load_cifar10(args.data_dir)
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 10)
    corruptions: dict[str, list[float | int]] = (
        {
            "gaussian": [0.0, 0.3, 0.5, 1.0],
            "salt_pepper": [0.0, 0.1, 0.3],
            "occlusion": [0, 7, 14],
        }
        if args.quick
        else {
            "gaussian": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "salt_pepper": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
            "occlusion": [0, 7, 10, 14],
        }
    )

    seed_results = []
    for seed in args.seeds:
        split = make_cifar10_splits(
            data,
            seed=seed,
            val_size=500 if args.quick else 5000,
            train_subset=2000 if args.quick else None,
            test_subset=1000 if args.quick else None,
        )
        net = build_cifar10_baseline_net(seed).to(device)
        wall_s = train_cifar10_baseline(
            net,
            split.train_images,
            split.train_labels,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        clean_train = evaluate_cifar10_baseline(
            net, split.train_images, split.train_labels, device=device
        )
        clean_val = evaluate_cifar10_baseline(
            net, split.val_images, split.val_labels, device=device
        )
        clean_test = evaluate_cifar10_baseline(
            net, split.test_images, split.test_labels, device=device
        )
        corr_acc: dict[str, dict[str, float]] = {}
        for kind, levels in corruptions.items():
            corr_acc[kind] = {}
            for lvl in levels:
                c_imgs = make_corrupted_cifar10_test_set(
                    split.test_images, seed=seed, kind=kind, level=float(lvl)
                )
                corr_acc[kind][str(lvl)] = evaluate_cifar10_baseline(
                    net, c_imgs, split.test_labels, device=device
                )
        print(
            f"seed={seed}: train={clean_train:.4f} val={clean_val:.4f} "
            f"test={clean_test:.4f} ({wall_s:.1f}s)",
            flush=True,
        )
        seed_results.append(
            {
                "seed": seed,
                "wall_seconds": wall_s,
                "train_accuracy": clean_train,
                "validation_accuracy": clean_val,
                "test_accuracy": clean_test,
                "corruptions": corr_acc,
            }
        )

    payload = {
        "environment": environment_info(device),
        "epochs": epochs,
        "quick": args.quick,
        "seeds": seed_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved CIFAR-10 baseline results to {args.output}")


if __name__ == "__main__":
    main()
