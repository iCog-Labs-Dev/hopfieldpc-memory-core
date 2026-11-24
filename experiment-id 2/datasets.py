"""Dataset specifications, loaders, splits, and augmentation for Experiment 2."""
from __future__ import annotations

import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and training defaults for one supported image dataset."""

    name: str
    display_name: str
    in_channels: int
    image_size: int
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    val_size: int
    horizontal_flip: bool
    crop_padding: int
    recommended_epochs: int
    recommended_lr: float
    recommended_weight_decay: float
    occlusion_sizes: tuple[int, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        display_name="Fashion-MNIST",
        in_channels=1,
        image_size=28,
        num_classes=10,
        mean=(0.2860,),
        std=(0.3530,),
        val_size=5000,
        horizontal_flip=True,
        crop_padding=2,
        recommended_epochs=10,
        recommended_lr=1e-3,
        recommended_weight_decay=1e-4,
        occlusion_sizes=(0, 7, 10, 14),
    ),
    "kmnist": DatasetSpec(
        name="kmnist",
        display_name="KMNIST",
        in_channels=1,
        image_size=28,
        num_classes=10,
        mean=(0.1918,),
        std=(0.3483,),
        val_size=5000,
        # Cursive Hiragana characters are handedness-sensitive; never flip.
        horizontal_flip=False,
        crop_padding=2,
        recommended_epochs=8,
        recommended_lr=1e-3,
        recommended_weight_decay=1e-4,
        occlusion_sizes=(0, 7, 10, 14),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        display_name="CIFAR-10",
        in_channels=3,
        image_size=32,
        num_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        val_size=5000,
        horizontal_flip=True,
        crop_padding=4,
        recommended_epochs=16,
        recommended_lr=1.5e-3,
        recommended_weight_decay=5e-4,
        occlusion_sizes=(0, 8, 12, 16),
    ),
    "svhn": DatasetSpec(
        name="svhn",
        display_name="SVHN",
        in_channels=3,
        image_size=32,
        num_classes=10,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
        val_size=5000,
        # Street digits (e.g., 6, 9, 2, 5) are not flip-invariant.
        horizontal_flip=False,
        crop_padding=2,
        recommended_epochs=8,
        recommended_lr=1e-3,
        recommended_weight_decay=2e-4,
        occlusion_sizes=(0, 8, 12, 16),
    ),
    "mnist": DatasetSpec(
        name="mnist",
        display_name="MNIST",
        in_channels=1,
        image_size=28,
        num_classes=10,
        mean=(0.1307,),
        std=(0.3081,),
        val_size=5000,
        horizontal_flip=False,
        crop_padding=2,
        recommended_epochs=5,
        recommended_lr=1e-3,
        recommended_weight_decay=1e-4,
        occlusion_sizes=(0, 7, 10, 14),
    ),
}

DEFAULT_MULTI_DATASETS: tuple[str, ...] = (
    "fashion_mnist",
    "kmnist",
    "svhn",
    "cifar10",
)


@dataclass
class DatasetSplits:
    """Seeded train/validation/test tensors in [0, 1] pixel space."""

    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def get_dataset_spec(name: str) -> DatasetSpec:
    """Look up a registered dataset specification by canonical name."""
    key = name.strip().lower()
    if key not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset {name!r}; expected one of: {supported}.")
    return DATASET_SPECS[key]


def _to_nchw_float_and_labels(
    raw_data: Any,
    raw_targets: Any,
    *,
    in_channels: int,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw torchvision dataset buffers into [N, C, H, W] float32 in [0, 1]."""
    images = torch.as_tensor(raw_data)
    if in_channels == 1:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        elif images.ndim == 4 and images.shape[-1] == 1:
            images = images.permute(0, 3, 1, 2)
    else:
        if images.ndim == 4 and images.shape[-1] == in_channels:
            images = images.permute(0, 3, 1, 2)
    if images.ndim != 4 or images.shape[1] != in_channels or images.shape[2] != image_size:
        raise ValueError(
            f"Unexpected image tensor shape {tuple(images.shape)}; "
            f"expected [N, {in_channels}, {image_size}, {image_size}]."
        )
    if images.dtype == torch.uint8:
        images = images.float().div(255.0)
    else:
        images = images.float()
        if images.numel() > 0 and float(images.max()) > 1.0:
            images = images.div(255.0)
    labels = torch.as_tensor(raw_targets, dtype=torch.long).reshape(-1)
    if labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}."
        )
    return images.contiguous(), labels.contiguous()


KMNIST_CSV_MIRROR = "https://huggingface.co/datasets/vincent-espitalier/K-MNIST-CSV/resolve/main"


def _load_kmnist_from_csv_mirror(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback KMNIST loader when codh.rois.ac.jp returns HTTP 403 Forbidden."""
    cache_dir = data_dir / "KMNIST" / "csv_mirror"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_pt = cache_dir / "kmnist_tensors.pt"
    if cache_pt.exists():
        cached = torch.load(cache_pt, map_location="cpu", weights_only=True)
        return tuple(cached)  # type: ignore[return-value]

    def _fetch_and_parse(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        dest = cache_dir / filename
        if not dest.exists():
            req = urllib.request.Request(
                f"{KMNIST_CSV_MIRROR}/{filename}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fout:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fout.write(chunk)
        arr = np.loadtxt(dest, delimiter=",", skiprows=1, dtype=np.uint8)
        labels = torch.from_numpy(arr[:, 0].astype(np.int64)).contiguous()
        images = torch.from_numpy(arr[:, 1:].reshape(-1, 1, 28, 28)).float().div(255.0).contiguous()
        return images, labels

    train_x, train_y = _fetch_and_parse("kmnist_train.csv")
    test_x, test_y = _fetch_and_parse("kmnist_test.csv")
    torch.save((train_x, train_y, test_x, test_y), cache_pt)
    return train_x, train_y, test_x, test_y


def load_dataset_tensors(
    name: str,
    data_dir: str | Path,
    *,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load (train_images, train_labels, test_images, test_labels) in [0, 1]."""
    import torchvision

    spec = get_dataset_spec(name)
    root = str(data_dir)
    if spec.name in ("fashion_mnist", "kmnist", "mnist"):
        cls_map = {
            "fashion_mnist": torchvision.datasets.FashionMNIST,
            "kmnist": torchvision.datasets.KMNIST,
            "mnist": torchvision.datasets.MNIST,
        }
        cls = cls_map[spec.name]
        try:
            train_ds = cls(root=root, train=True, download=download)
            test_ds = cls(root=root, train=False, download=download)
        except RuntimeError:
            if spec.name == "kmnist" and download:
                return _load_kmnist_from_csv_mirror(Path(data_dir))
            raise
        train_x, train_y = _to_nchw_float_and_labels(
            train_ds.data, train_ds.targets,
            in_channels=spec.in_channels, image_size=spec.image_size,
        )
        test_x, test_y = _to_nchw_float_and_labels(
            test_ds.data, test_ds.targets,
            in_channels=spec.in_channels, image_size=spec.image_size,
        )
        return train_x, train_y, test_x, test_y

    if spec.name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=download)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=download)
        train_x, train_y = _to_nchw_float_and_labels(
            train_ds.data, train_ds.targets,
            in_channels=spec.in_channels, image_size=spec.image_size,
        )
        test_x, test_y = _to_nchw_float_and_labels(
            test_ds.data, test_ds.targets,
            in_channels=spec.in_channels, image_size=spec.image_size,
        )
        return train_x, train_y, test_x, test_y

    if spec.name == "svhn":
        svhn_root = str(Path(root) / "svhn")
        try:
            train_ds = torchvision.datasets.SVHN(root=svhn_root, split="train", download=download)
            test_ds = torchvision.datasets.SVHN(root=svhn_root, split="test", download=download)
        except ModuleNotFoundError as exc:
            if exc.name == "scipy":
                raise ImportError(
                    "Loading SVHN via torchvision requires 'scipy'. "
                    "Install it with `pip install scipy` in your virtual environment."
                ) from exc
            raise
        train_x, train_y = _to_nchw_float_and_labels(
            train_ds.data, train_ds.labels,
            in_channels=spec.in_channels, image_size=spec.image_size,
        )
        test_x, test_y = _to_nchw_float_and_labels(
            test_ds.data, test_ds.labels,
            in_channels=spec.in_channels, image_size=spec.image_size,
        )
        return train_x, train_y, test_x, test_y

    raise ValueError(f"Loader not implemented for dataset {spec.name!r}.")


def make_dataset_splits(
    data: Sequence[torch.Tensor],
    *,
    seed: int,
    val_size: int,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> DatasetSplits:
    """Split train data deterministically into train/val and slice optional subsets."""
    train_images, train_labels, test_images, test_labels = data
    num_train = int(train_images.shape[0])
    if val_size <= 0 or val_size >= num_train:
        raise ValueError(
            f"val_size ({val_size}) must be in (0, {num_train})."
        )
    if train_subset is not None and train_subset <= 0:
        raise ValueError(f"train_subset must be positive, got {train_subset}.")
    if test_subset is not None and test_subset <= 0:
        raise ValueError(f"test_subset must be positive, got {test_subset}.")

    perm = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    num_test = (
        int(test_images.shape[0])
        if test_subset is None
        else min(test_subset, int(test_images.shape[0]))
    )
    return DatasetSplits(
        train_images=train_images[train_idx],
        train_labels=train_labels[train_idx],
        val_images=train_images[val_idx],
        val_labels=train_labels[val_idx],
        test_images=test_images[:num_test],
        test_labels=test_labels[:num_test],
    )


def augment_batch(
    images: torch.Tensor,
    *,
    random_crop: bool = True,
    horizontal_flip: bool = False,
    padding: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Vectorized training-time batch augmentation preserving [0, 1] range.

    Applies per-example reflection-padded random cropping and optional horizontal
    flipping without leaving PyTorch tensor space.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D [B, C, H, W] tensor, got {tuple(images.shape)}.")
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}.")

    out = images
    batch, channels, height, width = out.shape

    if random_crop and padding > 0:
        padded = F.pad(out, (padding, padding, padding, padding), mode="reflect")
        top = torch.randint(
            0, 2 * padding + 1, (batch,), generator=generator, device=out.device
        )
        left = torch.randint(
            0, 2 * padding + 1, (batch,), generator=generator, device=out.device
        )
        b_idx = torch.arange(batch, device=out.device)[:, None, None, None]
        c_idx = torch.arange(channels, device=out.device)[None, :, None, None]
        r_idx = (top[:, None] + torch.arange(height, device=out.device)[None, :])[:, None, :, None]
        k_idx = (left[:, None] + torch.arange(width, device=out.device)[None, :])[:, None, None, :]
        out = padded[b_idx, c_idx, r_idx, k_idx]

    if horizontal_flip:
        flip_mask = (
            torch.rand(batch, generator=generator, device=out.device) < 0.5
        )[:, None, None, None]
        out = torch.where(flip_mask, out.flip(-1), out)

    return out

