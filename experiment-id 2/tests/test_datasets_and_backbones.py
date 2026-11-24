"""Offline tests for Milestone 1: multi-dataset loaders, augmentation, and backbones."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXPERIMENT_2 = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_2.parent
for path in (str(EXPERIMENT_2), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from backbones import build_backbone
from datasets import (
    DEFAULT_MULTI_DATASETS,
    augment_batch,
    get_dataset_spec,
    load_dataset_tensors,
    make_dataset_splits,
)

from hopfieldpc import HybridOnlineBank


def test_dataset_specs_preserve_digit_orientation() -> None:
    """Digits and Japanese cursive characters must never be horizontally flipped."""
    assert get_dataset_spec("fashion_mnist").horizontal_flip is True
    assert get_dataset_spec("cifar10").horizontal_flip is True
    assert get_dataset_spec("kmnist").horizontal_flip is False
    assert get_dataset_spec("svhn").horizontal_flip is False
    assert set(DEFAULT_MULTI_DATASETS) == {"fashion_mnist", "kmnist", "svhn", "cifar10"}


def test_make_dataset_splits_is_seeded_and_disjoint() -> None:
    n_train, n_test = 100, 30
    train_x = torch.arange(n_train, dtype=torch.float32).view(n_train, 1, 1, 1).expand(-1, 1, 28, 28) / n_train
    train_y = torch.arange(n_train) % 10
    test_x = torch.zeros(n_test, 1, 28, 28)
    test_y = torch.arange(n_test) % 10

    s1 = make_dataset_splits((train_x, train_y, test_x, test_y), seed=7, val_size=20, train_subset=40, test_subset=15)
    s2 = make_dataset_splits((train_x, train_y, test_x, test_y), seed=7, val_size=20, train_subset=40, test_subset=15)

    assert torch.equal(s1.train_images, s2.train_images)
    assert torch.equal(s1.val_images, s2.val_images)
    assert s1.train_images.shape == (40, 1, 28, 28)
    assert s1.val_images.shape == (20, 1, 28, 28)
    assert s1.test_images.shape == (15, 1, 28, 28)

    train_ids = set(s1.train_images[:, 0, 0, 0].tolist())
    val_ids = set(s1.val_images[:, 0, 0, 0].tolist())
    assert train_ids.isdisjoint(val_ids)


def test_augment_batch_preserves_shape_range_and_determinism() -> None:
    x = torch.rand(12, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    y1 = augment_batch(x, random_crop=True, horizontal_flip=True, padding=4, generator=g1)
    y2 = augment_batch(x, random_crop=True, horizontal_flip=True, padding=4, generator=g2)

    assert y1.shape == x.shape
    assert torch.equal(y1, y2)
    assert float(y1.min()) >= 0.0 and float(y1.max()) <= 1.0

    # Zero-padding + no flip returns exact input.
    y_noop = augment_batch(x, random_crop=False, horizontal_flip=False, padding=0)
    assert torch.equal(y_noop, x)


def test_load_dataset_tensors_mocked_for_all_four_datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torchvision

    class FakeGrayDataset:
        def __init__(self, root: str, train: bool, download: bool) -> None:
            n = 20 if train else 8
            self.data = torch.randint(0, 256, (n, 28, 28), dtype=torch.uint8)
            self.targets = torch.randint(0, 10, (n,), dtype=torch.long)

    class FakeCifarDataset:
        def __init__(self, root: str, train: bool, download: bool) -> None:
            n = 20 if train else 8
            self.data = torch.randint(0, 256, (n, 32, 32, 3), dtype=torch.uint8).numpy()
            self.targets = list(range(n))

    class FakeSvhnDataset:
        def __init__(self, root: str, split: str, download: bool) -> None:
            n = 20 if split == "train" else 8
            self.data = torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8).numpy()
            self.labels = torch.randint(0, 10, (n,), dtype=torch.long).numpy()

    monkeypatch.setattr(torchvision.datasets, "FashionMNIST", FakeGrayDataset)
    monkeypatch.setattr(torchvision.datasets, "KMNIST", FakeGrayDataset)
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", FakeCifarDataset)
    monkeypatch.setattr(torchvision.datasets, "SVHN", FakeSvhnDataset)

    for name in DEFAULT_MULTI_DATASETS:
        spec = get_dataset_spec(name)
        train_x, train_y, test_x, test_y = load_dataset_tensors(name, tmp_path, download=False)
        assert train_x.shape == (20, spec.in_channels, spec.image_size, spec.image_size)
        assert test_x.shape == (8, spec.in_channels, spec.image_size, spec.image_size)
        assert train_x.dtype == torch.float32
        assert 0.0 <= float(train_x.min()) and float(train_x.max()) <= 1.0
        assert train_y.dtype == torch.long and test_y.dtype == torch.long


def test_kmnist_csv_fallback_parses_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When torchvision.datasets.KMNIST hits HTTP 403, fallback parses CSV and caches .pt."""
    import io
    import urllib.request

    import torchvision

    def failing_kmnist(*args, **kwargs):
        raise RuntimeError("HTTP Error 403: Forbidden")

    header = "label," + ",".join(f"pixel{i}" for i in range(1, 785))
    row0 = "3," + ",".join(["255"] * 784)
    row1 = "7," + ",".join(["0"] * 784)
    csv_bytes = f"{header}\n{row0}\n{row1}\n".encode()

    def fake_urlopen(req, timeout=120):
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                self.close()
        return _Resp(csv_bytes)

    monkeypatch.setattr(torchvision.datasets, "KMNIST", failing_kmnist)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    tr_x, tr_y, te_x, te_y = load_dataset_tensors("kmnist", tmp_path, download=True)
    assert tr_x.shape == (2, 1, 28, 28)
    assert tr_y.tolist() == [3, 7]
    assert te_x.shape == (2, 1, 28, 28)
    assert te_y.tolist() == [3, 7]
    assert float(tr_x[0].mean()) == pytest.approx(1.0)
    assert float(tr_x[1].mean()) == pytest.approx(0.0)
    assert (tmp_path / "KMNIST" / "csv_mirror" / "kmnist_tensors.pt").exists()


@pytest.mark.parametrize("dataset_name", DEFAULT_MULTI_DATASETS)
def test_backbones_plug_into_pre_last_layer_hopfield(dataset_name: str) -> None:
    """Every dataset backbone outputs [B, 64] z compatible with HybridOnlineBank -> head."""
    spec = get_dataset_spec(dataset_name)
    net = build_backbone(spec, embed_dim=64, seed=0)
    net.eval()

    x = torch.rand(6, spec.in_channels, spec.image_size, spec.image_size)
    logits, z, h, aux = net(x)
    assert logits.shape == (6, spec.num_classes)
    assert z.shape == (6, 64)
    assert torch.equal(z, h)
    assert aux == {}

    # Verify compatibility with untouched HybridOnlineBank at the pre-last layer.
    bank = HybridOnlineBank(64, spec.num_classes, 8, method="hybrid")
    bank.initialize_center(torch.zeros(64))
    for c in range(spec.num_classes):
        vec = torch.randn(64)
        prop = bank.plan_candidate(vec, c, bootstrap=True)
        if prop.is_write:
            bank.commit(prop, example_id=c, step=0, epoch=1)
    assert bank.ready
    retrieved, weights = bank.retrieve(z)
    assert retrieved.shape == (6, 64)
    assert weights.shape[0] == 6
    hop_logits = net.head(retrieved)
    assert hop_logits.shape == (6, spec.num_classes)
