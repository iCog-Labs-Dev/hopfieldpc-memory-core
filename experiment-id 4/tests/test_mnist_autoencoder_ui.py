"""Tests for the Experiment 4 real-time MNIST autoencoder UI helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from queue import Queue
from threading import Event
from unittest.mock import Mock

import torch
import pytest

EXP4_DIR = Path(__file__).resolve().parents[1]
if str(EXP4_DIR) not in sys.path:
    sys.path.insert(0, str(EXP4_DIR))

from mnist_autoencoder_ui import (
    AutoencoderVariant,
    MnistAutoencoder,
    RealtimeAutoencoderUI,
    build_sample_corruption,
    digit_overlap,
    get_config,
    parse_args,
    vector_images,
)


def test_ui_worker_trains_all_four_arms_and_publishes_previews(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mnist_autoencoder_ui.py", "--quick", "--device", "cpu",
                                      "--epochs", "2", "--batch-size", "8", "--beta-read", "0.7"])
    ui = object.__new__(RealtimeAutoencoderUI)
    ui.args = parse_args()
    ui.epochs = Mock(get=lambda: str(ui.args.epochs))
    ui.learning_rate = Mock(get=lambda: str(ui.args.learning_rate))
    ui.noise_level = Mock(get=lambda: str(ui.args.noise_level))
    ui.noise_kind = Mock(get=lambda: ui.args.noise)
    ui.stop_event = Event()
    ui.queue = Queue()
    images = torch.rand(24, 1, 28, 28, generator=torch.Generator().manual_seed(5))
    monkeypatch.setattr(ui, "load_images", lambda cfg: (images[:16], images[16:], images[:16]))

    ui.train_worker()

    events = list(ui.queue.queue)
    assert not [event["error"] for event in events if "error" in event]
    assert events[-1] == {"status": "Finished"}
    summaries = [event for event in events if "losses" in event]
    assert [event["epoch"] for event in summaries] == [1, 2]
    assert all(set(event["losses"]) == {"Direct", "EMA", "Fixed", "Dynamic"} for event in summaries)
    assert summaries[-1]["active"]["Dynamic"] > 0
    preview = [event for event in events if "generations" in event][-1]
    for clean, noisy in preview["generations"].values():
        assert clean.shape == noisy.shape == images[16:].shape
        assert clean.isfinite().all() and noisy.isfinite().all()


def test_mnist_autoencoder_shapes_and_decode() -> None:
    torch.manual_seed(0)
    net = MnistAutoencoder(embed_dim=64)
    images = torch.rand(8, 1, 28, 28)

    z, reconstruction = net(images)

    assert z.shape == (8, 64)
    assert reconstruction.shape == images.shape
    assert float(reconstruction.detach().min()) >= 0.0
    assert float(reconstruction.detach().max()) <= 1.0
    assert net.decode(z).shape == images.shape


def test_build_sample_corruption_is_deterministic() -> None:
    images = torch.rand(8, 1, 28, 28)

    noisy_a = build_sample_corruption(images, "gaussian", 0.3, seed=7)
    noisy_b = build_sample_corruption(images, "gaussian", 0.3, seed=7)

    assert torch.equal(noisy_a, noisy_b)
    assert noisy_a.shape == images.shape
    assert float(noisy_a.min()) >= 0.0
    assert float(noisy_a.max()) <= 1.0


def test_digit_overlap_scores_strokes_even_on_a_mostly_black_canvas() -> None:
    target = torch.zeros(1, 1, 28, 28)
    target[:, :, 5:20, 10] = 0.7
    black = torch.zeros_like(target)
    assert digit_overlap(black, target).item() == 0
    assert digit_overlap(target, target).item() == pytest.approx(1)
    assert digit_overlap(black, black).item() == 1
    assert digit_overlap(target.roll(1, dims=3), target).item() == 0
    assert digit_overlap(torch.cat((black, target)), target.repeat(2, 1, 1, 1)).item() == pytest.approx(0.5)


def test_raw_coordinates_use_the_same_brightness_across_slots() -> None:
    values = torch.tensor([[-1., 0., 1., 3., -3.], [0., 1., -3., -1., 3.]])
    images = vector_images(values, (1, 7))
    assert images[1].size == (3, 2)
    assert images[1].tobytes() == bytes([64, 128, 191, 223, 32, 128])
    assert images[7].tobytes() == bytes([128, 191, 32, 64, 223, 128])


@pytest.mark.parametrize("name", ["Fixed", "Dynamic", "EMA"])
def test_comparison_reads_active_logits_with_one_dot_product_softmax(name: str) -> None:
    cfg = get_config(memory_overrides={
        "initial_memories": 2, "max_memories": 3, "beta_read": 0.7, "num_updates": 2,
    })
    ui = object.__new__(RealtimeAutoencoderUI)
    variant = ui.create_variant(name, cfg, torch.device("cpu"))
    bank = variant.bank
    assert bank is not None
    bank.initialize_from_embeddings(torch.eye(3, bank.dim))
    slots = torch.zeros(2, bank.dim)
    slots[:, :2] = torch.tensor([[2., -1.], [0.5, 1.]])
    bank.mask[:] = torch.tensor([True, False, True])
    bank.values[bank.mask] = slots
    bank.values[1].fill_(float("nan"))
    query = torch.zeros(3, bank.dim)
    query[:, :2] = torch.tensor([[1., 0.], [-0.5, 1.], [0., 0.]])
    logits = query.reshape(-1, 1, 28, 28).requires_grad_()
    weights = torch.softmax(0.7 * (query @ slots.T), dim=1)
    expected = (weights.unsqueeze(-1) * slots).sum(dim=1).reshape_as(logits)
    actual = variant._read_output(logits)
    torch.testing.assert_close(actual, expected.sigmoid())
    assert not actual.requires_grad


@pytest.mark.parametrize("name", ["Fixed", "Dynamic", "EMA"])
def test_memory_reads_and_writes_without_changing_autoencoder_training(name: str, monkeypatch) -> None:
    cfg = get_config(memory_overrides={"initial_memories": 3})
    device = torch.device("cpu")
    variant = AutoencoderVariant(name, cfg, seed=3, device=device)
    baseline = AutoencoderVariant("Direct", cfg, seed=3, device=device)
    images = torch.rand(12, 1, 28, 28)
    bank = variant.bank
    assert bank is not None and bank.dim == 784
    logits = variant.net.output_logits(variant.encode(images)).detach()
    assert torch.equal(variant.training_embeddings(images, cfg, device), logits.flatten(1))
    assert torch.equal(variant.generate(images, device)[0], logits.sigmoid())
    variant.initialize_bank(images, cfg, device)
    reconstruction, latent = variant.generate(images, device)
    assert latent.shape == (12, 64)
    assert torch.equal(reconstruction, variant._read_output(logits))
    updates = []
    owner, method = (variant, "_ema_write") if name == "EMA" else (bank, "online_update")
    original_update = getattr(owner, method)

    def record_update(*args, **kwargs):
        assert not args[0].requires_grad
        update = original_update(*args, **kwargs)
        updates.append(update)
        return update

    monkeypatch.setattr(owner, method, record_update)
    for batch in (images, 1.0 - images):
        # Force a poor memory prediction while the direct network learns normally.
        bank.values[bank.mask] = 20.0
        expected_output, _ = variant.generate(batch, device)
        measured_mse, score = variant.train_batch_metrics(batch, device)
        direct_mse, direct_score = baseline.train_batch_metrics(batch, device)
        assert measured_mse == pytest.approx((expected_output - batch).square().mean().item())
        assert measured_mse > direct_mse
        assert score == direct_score
        if name != "EMA":
            assert updates[-1]["observation_noise"] == pytest.approx(1.0 - direct_score + bank.cfg.eps_stab)
        for param, reference in zip(variant.net.parameters(), baseline.net.parameters()):
            assert torch.equal(param.grad, reference.grad)
            assert torch.equal(param, reference)
    assert len(updates) == 2
    assert bank.values.grad is None


def test_ema_writes_assigned_batch_means_and_preserves_empty_slots() -> None:
    cfg = get_config(memory_overrides={"initial_memories": 3, "max_memories": 4})
    variant = AutoencoderVariant("EMA", cfg, seed=3, device=torch.device("cpu"))
    bank = variant.bank
    bank.initialize_from_embeddings(torch.eye(4, bank.dim))
    bank.mask[:] = torch.tensor([True, False, True, True])
    bank.values.zero_()
    bank.values[:, 0] = torch.tensor([0., 1., 10., 20.])
    observations = torch.zeros(3, bank.dim)
    observations[:, 0] = torch.tensor([1., 3., 12.])
    expected = bank.values.clone()
    expected[:, 0] = torch.tensor([0.2, 1., 10.2, 20.])
    variant._ema_write(observations)
    torch.testing.assert_close(bank.values, expected)


def test_mnist_epoch_adaptation_reports_prune_merge_and_split() -> None:
    cfg = get_config(memory_overrides=dict(
        dim=4, max_memories=4, initial_memories=3, min_active_memories=1,
        prune_patience=1, coalescence_patience=1, tau_support=1.0, tau_eff=0.0,
        tau_uncert=1e-9, n_kish_min=1.0, beta_write=16.0,
    ))
    variant = AutoencoderVariant("Dynamic", cfg, seed=0, device=torch.device("cpu"))
    bank = variant.bank
    assert bank is not None
    assert variant.prototype_images() == {}
    points = torch.zeros(16, bank.dim)
    points[:, 0] = 1
    points[:, 1] = torch.tensor([0.1, -0.1] * 8)
    bank.initialize_from_embeddings(torch.eye(4, bank.dim))
    bank.center.zero_()
    bank.values.zero_()
    bank.values[:3, 0] = torch.tensor([1., 1.1, -1.])
    before = variant.prototype_images()
    assert before[2].size == (28, 28)
    assert before[2].tobytes() == bytes([64] + [128] * 783)

    stats = bank.adapt_structure(points, epoch=2, task_score=0.95, allow_spawn=True)
    assert (stats["pruned"], stats["coalesced"], stats["spawned"]) == (1, 1, 1)
    assert stats["observation_noise"] == pytest.approx(0.05 + bank.cfg.eps_stab)
    assert set(variant.prototype_images()) == {0, 1}
