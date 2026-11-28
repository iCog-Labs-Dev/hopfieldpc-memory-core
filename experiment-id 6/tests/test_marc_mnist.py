"""Experiment 6 tests for retrieval protection, readout placement and the prototype gallery."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marc_mnist.cli import parse_args
from marc_mnist.model import GASSVariant
from gass_autoencoder import comparison_config
from gass import GASSBank, GASSConfig


def example(*, initial=3, **overrides):
    rng = torch.Generator().manual_seed(930)
    u = torch.randn(21, 4, generator=rng)
    v = torch.randn(21, 3, generator=rng)
    bank = GASSBank(3, initial, 1, 5, GASSConfig(**overrides), key_dim=4)
    assert bank.initialize_from_embeddings(u, v, seed=1)
    return bank, u, v


def buffer_copy(bank):
    return {k: v.clone() for k, v in bank.named_buffers()}


def test_reconstruction_defaults():
    args = parse_args([])
    assert args.gass_variant_class is GASSVariant
    assert args.gass.bank_init == "warm-up"
    memory = comparison_config(args, False, {}).memory
    assert (args.epochs, args.gass.window_size) == (20, 256)
    assert (memory.initial_memories, memory.min_active_memories, memory.max_memories) == (20, 2, 80)
    assert (args.gass.patience, args.gass.merge_cosine, args.gass.support_fraction) == (2, 0.90, 0.10)
    assert (args.gass.min_child_mass, args.gass.edit_budget, args.gass.beta_r, args.gass.beta_w) == (1, 5, 80, 20)
    assert (args.gass.read_score, args.gass.write_score, args.gass.retrieval_loss) == ("dice", "dice", "bernoulli_kl")


def test_temperature_overrides_and_benchmark_grids(tmp_path):
    shared = parse_args(["--gass-beta", "7"]).gass
    mixed = parse_args(["--gass-beta", "7", "--gass-read-beta", "11"]).gass
    assert (shared.beta_r, shared.beta_w) == (7, 7)
    assert (mixed.beta_r, mixed.beta_w) == (11, 7)
    sweep = parse_args(["--benchmark", "--beta-grid", "3", "7"])
    assert sweep.beta_grid == [3, 7] and sweep.gass.read_beta is None
    with pytest.raises(SystemExit):
        parse_args(["--beta-grid", "3", "--gass-read-beta", "80"])

    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps([{}, {"beta": 7}, {"beta": 7, "read_beta": 11}]), encoding="utf-8")
    candidates = parse_args(["--benchmark", "--validation-grid", str(grid)]).validation_grid
    assert [(c.beta_r, c.beta_w) for c in candidates] == [(80, 20), (7, 7), (11, 7)]


@pytest.mark.parametrize("layer,dim", [("Decoder output", 784), ("Encoder bottleneck", 64)])
def test_ui_applies_reference_window_and_bank_initialization(monkeypatch, layer, dim):
    import tkinter as tk
    from marc_mnist.ui import GASSComparisonUI
    from mnist_reconstruction.ui import RealtimeAutoencoderUI

    args = parse_args(["--initial-memories", "3", "--window-size", "12", "--warmup-epochs", "2"])
    ui = GASSComparisonUI.__new__(GASSComparisonUI)
    ui.args, ui.settings, ui.worker = args, args.gass, None
    tcl = tk.Tcl()
    ui.selectors = {"bank_init": tk.StringVar(tcl, "random")}
    ui.readout_layer = tk.StringVar(tcl, layer)
    if layer == "Encoder bottleneck":
        ui.selectors.update({key: tk.StringVar(tcl, "squared" if key == "retrieval_loss" else "cosine")
                             for key in ("read_score", "write_score", "retrieval_loss")})
    ui.window_size = tk.StringVar(tcl, "8")
    ui.marc_checks = tk.StringVar(tcl, "off")
    ui.status_text = tk.StringVar(tcl)
    ui.temperatures = {}
    starts = []
    monkeypatch.setattr(RealtimeAutoencoderUI, "start", lambda self: starts.append(self.settings))
    train = torch.rand(16, 1, 28, 28, generator=torch.Generator().manual_seed(123))
    monkeypatch.setattr(RealtimeAutoencoderUI, "load_images", lambda self, cfg: (train, train[:2], train))
    cfg = ui.config_factory(quick=False, memory_overrides={})
    device = torch.device("cpu")

    ui.start()
    random_arm = ui.create_variant("GASS", cfg, device)  # No reference images have been loaded.
    assert random_arm.bank.ready and random_arm.bank.active_count == 3
    assert random_arm.bank.dim == dim
    assert random_arm.settings.warmup_epochs == 0 and random_arm.bank.cfg.window_size == 8
    repeat = ui.create_variant("GASS", cfg, device)
    torch.testing.assert_close(random_arm.bank.values, repeat.bank.values)
    _, _, reference = ui.load_images(cfg)
    assert len(reference) == 8

    ui.selectors["bank_init"].set("warm-up")
    ui.marc_checks.set("reference")
    ui.window_size.set("10")
    ui.start()
    _, _, reference = ui.load_images(cfg)
    assert len(reference) == 10
    warm_arm = ui.create_variant("GASS", cfg, device)
    assert not warm_arm.bank.ready
    waiting = ui.bank_preview({"GASS": warm_arm})
    assert waiting["banks"]["GASS"] == {}
    assert waiting.get("stored_banks", {}).get("GASS", {}) == {}
    with torch.no_grad():
        z = warm_arm.encode(train[:2])
        torch.testing.assert_close(warm_arm.generate(train[:2], device)[0], warm_arm.net.decode(z))
    warm_arm.finish_bank_epoch(reference, cfg, device, 1, 0.)
    assert not warm_arm.bank.ready
    warm_arm.finish_bank_epoch(reference, cfg, device, 2, 0.)
    assert warm_arm.bank.ready and warm_arm.bank.active_count == 3
    assert warm_arm.reference_images is reference and warm_arm.settings.retrieval_margin == 0
    assert warm_arm.training_embeddings(reference, cfg, device).shape == (10, dim)
    warm_arm.train_batch_metrics(train[:2], device)
    warm_arm.finish_bank_epoch(reference, cfg, device, 3, 0.)
    clean, noisy = train[:2], train[2:4]
    expected, _ = warm_arm.generate(clean, device)
    clean_preview, noisy_preview = ui.preview_outputs(warm_arm, clean, noisy, device)
    torch.testing.assert_close(clean_preview, expected)
    torch.testing.assert_close(noisy_preview, warm_arm.generate(noisy, device)[0])
    slot = int(warm_arm.bank.mask.nonzero()[0].item())
    tile = warm_arm.prototype_images((slot,))[slot]
    assert tile.size == (28, 28)
    snapshot = ui.bank_preview({"GASS": warm_arm})
    decoded = snapshot["banks"]["GASS"]
    stored = snapshot.get("stored_banks", snapshot["banks"])["GASS"]
    assert stored.keys() == decoded.keys() == set(warm_arm.bank.mask.nonzero().flatten().tolist())
    if layer == "Encoder bottleneck":
        from mnist_reconstruction.model import vector_images
        raw = vector_images(warm_arm.bank.values[[slot]], (slot,))[slot]
        assert stored[slot].size == (8, 8) and stored[slot].tobytes() == raw.tobytes()
        with torch.no_grad():
            retrieved = warm_arm.bank.read(warm_arm.encode(clean))
            torch.testing.assert_close(expected, warm_arm.net.decode(retrieved))
            logits = warm_arm.net.output_logits(warm_arm.bank.values[[slot]]).flatten(1)
            assert tile.tobytes() == vector_images(logits, (slot,))[slot].tobytes()
    else:
        assert "stored_banks" not in snapshot
        assert stored[slot].tobytes() == warm_arm.prototype_images((slot,), stored=True)[slot].tobytes()

    for invalid, message in (("3", "initial memory count"), ("3.5", "whole number")):
        ui.window_size.set(invalid)
        ui.start()
        assert message in ui.status_text.get()
    assert [settings.bank_init for settings in starts] == ["random", "warm-up"]


@pytest.mark.parametrize("margin,momentum", [(0., False), (.001, True)])
def test_reference_protects_old_targets_and_updates_trust_from_fresh_data(margin, momentum):
    bank, u, v = example(initial=1, retrieval_margin=margin, momentum=momentum, retrieval_loss="squared")
    regular, _, _ = example(initial=1, retrieval_margin=margin, momentum=momentum, retrieval_loss="squared")
    old = bank.values[bank.mask].clone()
    reference = u[:7], old.expand(7, -1)
    fresh = old.expand(3, -1) + 2.
    bank.online_update(u[-3:], fresh, retrieval_reference=reference)
    regular.online_update(u[-3:], fresh)
    torch.testing.assert_close(bank.values[bank.mask], old, rtol=0, atol=0)
    assert not torch.equal(regular.values[bank.mask], old)
    torch.testing.assert_close(bank.uncertainty, regular.uncertainty, rtol=0, atol=0)
    previous = buffer_copy(bank)
    with pytest.raises(ValueError, match="valid observations"):
        bank.online_update(u, v, retrieval_reference=(u[:0], v[:0]))
    assert all(torch.equal(value, previous[name]) for name, value in bank.named_buffers())


def test_reference_strict_value_step_uses_reference_size():
    bank, u, v = example(initial=1, retrieval_margin=.01, retrieval_loss="squared")
    state = bank._active_layout(bank.layout)
    fresh_u, fresh_v = u.double(), state.values.expand(len(u), -1) + 1.
    ref = fresh_u[:3], fresh_v[:3]
    before = bank._objective(state, fresh_u, fresh_v, bank.variance.double())[0]
    updated, after = bank._value_step(state, fresh_u, fresh_v, bank.uncertainty[bank.mask].double(), before, ref)
    movement = (updated.values - state.values).square().sum()
    assert movement > 0 and after < before
    old_read = bank._objective(state, *ref, bank.variance.double())[3].mean()
    new_read = bank._objective(updated, *ref, bank.variance.double())[3].mean()
    assert new_read <= old_read - bank.cfg.retrieval_margin * movement / len(ref[0])


def test_random_dice_keys_learn_to_recall_distinct_stored_strokes():
    args = parse_args(["--bank-init", "random", "--initial-memories", "2", "--max-memories", "2",
                       "--gass-read-beta", "80"])
    cfg = comparison_config(args, False, {})
    bank = GASSVariant(cfg, 0, torch.device("cpu"), args.gass).bank
    strokes = torch.full((2, 28, 28), -3.)
    strokes[0, 4:24, 7:12] = 3.
    strokes[1, 4:24, 17:22] = 3.
    cues = strokes.flatten(1)
    # The bank already stores both strokes. Give the keys a weak bias toward
    # the opposite stroke and present both patterns equally often.
    bank.values.copy_(cues)
    bank.keys.copy_(bank.keys * 0.25 + 0.05 * cues.flip(0))
    initial_error = (bank.read(cues).sigmoid() - cues.sigmoid()).square().mean()
    batch = cues.repeat(8, 1)
    for _ in range(4):
        report = bank.online_update(batch)
        assert report["after"] <= report["keys"] <= report["values"]
    recall = bank.read(cues).sigmoid()
    assert torch.equal(torch.cdist(recall, cues.sigmoid()).argmin(1), torch.arange(2))
    assert (recall - cues.sigmoid()).square().mean() < initial_error * 0.25


@pytest.mark.parametrize("margin,momentum", [(0., False), (.001, True)])
def test_protected_writes_replay_and_edits(margin, momentum):
    assert parse_args(["--retrieval-margin", str(margin)]).gass.retrieval_margin == margin
    bank, u, v = example(retrieval_margin=margin, momentum=momentum, patience=1, edit_budget=2)
    v = v + 1.
    state = bank._active_layout(bank.layout)
    u, v = u.double(), v.double()
    before, _, _, errors = bank._objective(state, u, v, bank.variance.double())
    updated, after = bank._value_step(state, u, v, bank.uncertainty[bank.mask].double(), before)
    _, _, _, new_errors = bank._objective(updated, u, v, bank.variance.double())
    movement = (updated.values - state.values).square().sum()
    assert movement > 0 and after < before
    assert new_errors.mean() <= errors.mean() - margin * movement / len(v)
    previous = float(errors.mean())
    for update in (bank.online_update, bank.refine_window, bank.adapt_structure):
        update(u, v)
        loss = bank._evaluate(bank.layout, u, v, details=True)[3]
        assert loss <= previous + 1e-12
        previous = loss
    assert all(t["read_after"] <= t["read_before"] for t in bank.last_structure["trials"] if t["accepted"])
