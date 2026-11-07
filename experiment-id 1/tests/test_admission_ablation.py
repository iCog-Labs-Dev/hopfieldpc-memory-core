"""Causal admission checks and a download-free epoch-resume integration fixture."""
import copy
import json

import pytest
import torch
from torch import nn

import admission_ablation as ablation
from benchmarks.mnist_baseline import ClassifierHead, MnistNet
from mnist_online_memory import extract, make_bank, train_step, validate_config
from online_memory_artifacts import EventWriter, atomic_json, read_events, rng_state, save_checkpoint


@pytest.fixture(autouse=True)
def deterministic_cpu(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(ablation, "source_identity", lambda: {"commit": "fixture", "source_sha256": "fixture"})


def toy_model():
    model = MnistNet(nn.Identity(), nn.Identity(), ClassifierHead(embed_dim=2, num_classes=2))
    with torch.no_grad():
        model.head.fc.weight.copy_(torch.eye(2))
        model.head.fc.bias.zero_()
    return model


def toy_bank():
    cfg = validate_config({"num_classes": 2, "ema_alpha": 1., "merge_distance_threshold": 2.})
    bank = make_bank(cfg, 2)
    bank.initialize_center(torch.zeros(2))
    for c, value in enumerate(([-1., 2.], [-1., 3.])):
        bank.commit(bank.plan_candidate(torch.tensor(value), c, bootstrap=True), example_id=c, step=0, epoch=1)
    return cfg, bank


def step(model, bank, cfg, x, y, **kwargs):
    optimizer = kwargs.pop("optimizer", torch.optim.Adam(model.parameters(), lr=.001))
    return train_step(model, bank, optimizer, x, y, torch.arange(len(y)), epoch=2,
                      batch=0, step=1, seed=0, run_id="fixture", cfg=cfg, **kwargs)


def test_all_correct_admission_adds_both_correct_and_skips_both_direct_wrong_cases(tmp_path):
    cfg, bank = toy_bank()
    writer = EventWriter(tmp_path, 3, 1)
    result = step(toy_model(), bank, cfg, torch.tensor([[3., 0.], [0., 3.], [3., 0.], [0., 3.]]),
                  torch.tensor([0, 1, 1, 0]), writer=writer, admission_policy=ablation.POLICY)
    events = list(read_events(tmp_path, writer.flush()))
    examples = [e for e in events if e["type"] == "example"]
    assert [e["category"] for e in examples] == ["harmed", "both_correct", "helped", "both_wrong"]
    assert [e["allows_consideration"] for e in examples] == [True, True, False, False]
    assert [e["action"] for e in examples] == ["merge", "merge", "skip", "skip"]
    assert result["counts"]["merge"] == result["counts"]["skip"] == 2
    assert all(e["snapshot_version"] == 2 for e in examples)
    assert bank.version == 4


def test_default_policy_remains_identical_to_explicit_original_policy():
    cfg, bank = toy_bank()
    other_bank, model = copy.deepcopy(bank), toy_model()
    other_model = copy.deepcopy(model)
    x, y = torch.tensor([[3., 0.], [0., 3.]]), torch.tensor([0, 1])
    first = step(model, bank, cfg, x, y)
    second = step(other_model, other_bank, cfg, x, y, admission_policy="accuracy_guided")
    ablation.require_equal(first, second)
    ablation.require_equal(model.state_dict(), other_model.state_dict())
    ablation.require_equal(bank.state_dict(), other_bank.state_dict())
    assert first["counts"]["merge"] == 1


def test_all_correct_policy_preserves_direct_losses_gradients_rng_and_adam_trajectory():
    cfg, bank = toy_bank()
    model, disabled = toy_model(), toy_model()
    original = copy.deepcopy(bank)
    first, second = torch.optim.Adam(model.parameters(), lr=.01), torch.optim.Adam(disabled.parameters(), lr=.01)
    x = torch.tensor([[3., 0.], [0., 3.], [3., 0.], [0., 3.]])
    y = torch.tensor([0, 1, 1, 0])
    for _ in range(4):
        state = torch.get_rng_state().clone()
        actual = step(model, bank, cfg, x, y, optimizer=first, admission_policy=ablation.POLICY)
        assert torch.equal(torch.get_rng_state(), state)
        expected = step(disabled, original, cfg, x, y, optimizer=second, memory_enabled=False)
        assert actual["loss"] == expected["loss"]
        for a, b in zip(model.parameters(), disabled.parameters()):
            assert torch.equal(a.grad, b.grad) and torch.equal(a, b)
        ablation.require_equal(first.state_dict(), second.state_dict())


def test_new_policy_keeps_bootstrap_batch_fallback_and_seeds_existing_classes_only_once(tmp_path):
    cfg, _ = toy_bank()
    bank = make_bank(cfg, 2)
    bank.initialize_center(torch.zeros(2))
    writer = EventWriter(tmp_path, 3, 2)
    result = step(toy_model(), bank, cfg, torch.tensor([[3., 0.], [2., 0.], [0., 3.], [0., 2.]]),
                  torch.tensor([0, 0, 1, 1]), admission_policy=ablation.POLICY, writer=writer)
    examples = [e for e in read_events(tmp_path, writer.flush()) if e["type"] == "example"]
    assert result["compared"] == 0 and result["fallback_count"] == 4
    assert bank.version == 2 and bank.occupancy() == [1, 1]
    assert all(e["hopfield_prediction"] is None for e in examples)
    assert [e["action"] for e in examples] == ["seed", "skip", "seed", "skip"]


def test_invalid_policy_fails_before_any_state_mutation():
    cfg, bank = toy_bank()
    model = toy_model()
    before = copy.deepcopy((model.state_dict(), bank.state_dict()))
    with pytest.raises(ValueError, match="admission policy"):
        step(model, bank, cfg, torch.eye(2), torch.tensor([0, 1]), admission_policy="unknown")
    ablation.require_equal((model.state_dict(), bank.state_dict()), before)


def fixture_model(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 2))
        model = MnistNet(encoder, nn.Identity(), ClassifierHead(embed_dim=2, num_classes=2))
    with torch.no_grad():
        encoder[1].weight.zero_()
        encoder[1].bias.zero_()
        encoder[1].weight[0, 4*28+4] = 1.
        encoder[1].weight[1, 20*28+20] = 1.
        model.head.fc.weight.copy_(torch.eye(2))
        model.head.fc.bias.zero_()
    return model


def parent_fixture(tmp_path, monkeypatch, model_factory=fixture_model):
    parent = tmp_path/"parent"
    cfg = validate_config({"seeds": [0], "num_classes": 2, "total_epochs": 3, "warmup_epochs": 1,
                           "batch_size": 8, "val_size": 8, "num_threads": 1, "eval_batch_size": 8,
                           "learning_rate": .01, "max_memories_per_class": 2,
                           "fixed_reference": {"prototypes_per_class": 1},
                           "gaussian_sigmas": [0., .1], "salt_pepper_probs": [0., .1],
                           "occlusion_sizes": [0, 2], "event_chunk_size": 11, "write_chunk_size": 3})
    generator = torch.Generator().manual_seed(17)
    images = torch.rand(48, 1, 28, 28, generator=generator)*.02
    labels = torch.arange(48) % 2
    images[labels == 0, :, 3:8, 3:8] += .8
    images[labels == 1, :, 18:23, 18:23] += .8
    train_ids, val_ids = torch.arange(40), torch.arange(40, 48)
    model = model_factory(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    bank = make_bank(cfg, 2)
    shuffle = torch.Generator().manual_seed(13)
    step_count = 0
    for epoch in range(1, 4):
        model.train()
        order = torch.randperm(len(train_ids), generator=shuffle)
        for batch, start in enumerate(range(0, len(order), 8)):
            ids = train_ids[order[start:start+8]]
            train_step(model, bank, optimizer, images[ids], labels[ids], ids, epoch=epoch, batch=batch,
                       step=step_count, seed=0, run_id="parent", cfg=cfg)
            step_count += 1
        if epoch == 1:
            bank.initialize_center(extract(model, images[train_ids], cfg, torch.device('cpu')).double().mean(0).float())
        save_checkpoint(parent/f"seed_0/checkpoints/epoch_{epoch}.pt",
                        {"config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "bank": bank.state_dict(), "shuffle_rng": shuffle.get_state(), "rng": rng_state(),
                         "epoch": epoch, "optimizer_step": step_count,
                         "split_ids": {"train": train_ids, "val": val_ids}})
    z = extract(model, images[train_ids], cfg, torch.device('cpu'))
    save_checkpoint(parent/"seed_0/fixed_reference.pt", {"values": torch.stack([z[labels[train_ids] == c].mean(0) for c in range(2)]),
                    "center": z.double().mean(0).float(), "labels": torch.tensor([0, 1])})
    atomic_json(parent/"manifest.json", {"config": cfg, "status": "complete"})
    def train_only_loader(data_dir, *, train):
        assert train is True, "The test dataset must never be opened by this control."
        return images, labels
    monkeypatch.setattr(ablation, "load_mnist_tensors", train_only_loader)
    monkeypatch.setattr(ablation, "build_baseline_net", model_factory)
    return parent


def test_synthetic_online_control_reaches_active_writes_four_comparators_and_exact_replay(tmp_path, monkeypatch):
    parent = parent_fixture(tmp_path, monkeypatch)
    output = ablation.run_ablation(parent, output=tmp_path/"control")
    result = json.loads((output/"results.json").read_text())
    seed = result["seeds"][0]
    assert result["manifest"]["parent_artifacts_unchanged"]
    assert result["manifest"]["validation_only"] and not result["manifest"]["test_set_opened"]
    assert seed["cnn_exact_every_epoch"] and seed["adam_exact_every_epoch"]
    assert seed["audit"]["replay_exact"] and seed["audit"]["examples"] == 80
    assert seed["audit"]["merge"] > 0 and seed["audit"]["both_correct"] > 0
    assert {r["method"] for r in seed["validation_results"]} == {"direct", "accuracy_guided", ablation.POLICY, "fixed_kmeans_reference"}
    assert "fixed_kmeans_reference" not in seed["history"][0]["validation"]
    assert (output/"report.md").is_file() and (output/"validation_corruptions.png").is_file()


def test_epoch_resume_matches_uninterrupted_bank_optimizer_and_history(tmp_path, monkeypatch):
    parent = parent_fixture(tmp_path, monkeypatch)
    full = ablation.run_ablation(parent, output=tmp_path/"full")
    resumed = ablation.run_ablation(parent, output=tmp_path/"resumed", stop_after_epoch=2)
    assert json.loads((resumed/"manifest.json").read_text())["status"] == "incomplete"
    ablation.run_ablation(output=resumed, resume=True)
    a = torch.load(full/"seed_0/checkpoints/epoch_3.pt", weights_only=False)
    b = torch.load(resumed/"seed_0/checkpoints/epoch_3.pt", weights_only=False)
    for key in ("model", "optimizer", "bank", "history", "shuffle_rng", "split_ids", "optimizer_step"):
        ablation.require_equal(a[key], b[key], key)
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    with pytest.raises(ValueError, match="source"):
        monkeypatch.setattr(ablation, "source_identity", lambda: {"commit": "different", "source_sha256": "different"})
        ablation.run_ablation(output=resumed, resume=True)


def test_invalid_seed_choices_are_rejected(tmp_path, monkeypatch):
    parent = parent_fixture(tmp_path, monkeypatch)
    for seeds in ([], [0, 0], [True], [0.0], [1]):
        with pytest.raises(ValueError, match="Seeds"):
            ablation.run_ablation(parent, seeds=seeds, output=tmp_path/"bad")


def test_incomplete_final_bank_keeps_genuine_accuracy_unavailable(tmp_path, monkeypatch):
    def single_prediction_model(seed):
        model = fixture_model(seed)
        with torch.no_grad():
            model.head.fc.bias[0] = 100.
        return model
    parent = parent_fixture(tmp_path, monkeypatch, single_prediction_model)
    output = ablation.run_ablation(parent, output=tmp_path/"incomplete")
    result = json.loads((output/"results.json").read_text())["seeds"][0]
    assert not result["bank"]["ready"] and result["memory_health"] is None
    assert result["audit"]["fallback"] == result["audit"]["examples"] == 80
    for row in result["validation_results"]:
        if row["method"] == ablation.POLICY:
            assert row["accuracy"] is None and row["evaluated"] == 0 and row["fallback_count"] == 8
    assert "unavailable" in (output/"report.md").read_text()


def test_aggregation_requires_unique_audited_seeds_and_unchanged_parent(tmp_path, monkeypatch):
    parent = parent_fixture(tmp_path, monkeypatch)
    worker = ablation.run_ablation(parent, output=tmp_path/"worker")
    combined = ablation.aggregate_runs([worker], output=tmp_path/"combined")
    result = json.loads((combined/"results.json").read_text())
    assert result["manifest"]["artifact_type"] == "completed_seed_aggregate"
    assert result["manifest"]["parent_artifacts_unchanged"] and result["manifest"]["seeds"] == [0]
    assert result["seeds"][0]["worker_seed_directory"] == str(worker/"seed_0")
    with pytest.raises(ValueError, match="Duplicate"):
        ablation.aggregate_runs([worker, worker], output=tmp_path/"duplicate")
    assert not (tmp_path/"duplicate").exists()
    file = worker/"seed_0/results.json"
    data = json.loads(file.read_text())
    data["audit"]["replay_exact"] = False
    atomic_json(file, data)
    with pytest.raises(ValueError, match="audits"):
        ablation.aggregate_runs([worker], output=tmp_path/"unaudited")
    data["audit"]["replay_exact"] = True
    atomic_json(file, data)
    (parent/"manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="parent artifact changed"):
        ablation.aggregate_runs([worker], output=tmp_path/"changed_parent")
