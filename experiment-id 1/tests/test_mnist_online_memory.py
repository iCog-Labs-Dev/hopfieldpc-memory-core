"""Ordering, parity, streaming/replay and CPU resume tests for Experiment 1."""
import copy
import json
import pytest
import torch
from torch import nn

import mnist_online_memory as experiment
from benchmarks.mnist_baseline import ClassifierHead, MnistEncoder, MnistNet, make_splits
from online_memory_artifacts import EventWriter, diagnostic_mode, read_events, replay_bank
from online_memory import OnlineHopfieldBank


@pytest.fixture(autouse=True)
def deterministic_cpu(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(experiment, "source_identity", lambda: dict(commit="fixture", source_sha256="fixture"))


def config(tmp_path, **extra):
    return experiment.validate_config(dict(
        reports_dir=str(tmp_path), seeds=[0], device="cpu", num_threads=1,
        total_epochs=3, warmup_epochs=1, train_subset=800, val_size=100, test_subset=100,
        learning_rate=.01, batch_size=64, eval_batch_size=100,
        fixed_reference=dict(prototypes_per_class=2), drift_probe_per_class=2,
        event_chunk_size=31, write_chunk_size=7,
        gaussian_sigmas=[0., .3], salt_pepper_probs=[0., .2], occlusion_sizes=[0, 7],
    ) | extra)


def tiny_model(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return MnistNet(MnistEncoder(c1=3, c2=6, hidden=32, embed_dim=16),
                        nn.Identity(), ClassifierHead(embed_dim=16))


def synthetic_data():
    def split(n, seed):
        generator = torch.Generator().manual_seed(seed)
        y = torch.arange(n) % 10
        x = torch.rand(n, 1, 28, 28, generator=generator) * .05
        for c in range(10):
            r, col = (c // 5) * 12 + 2, (c % 5) * 5 + 1
            x[y == c, :, r:r + 8, col:col + 4] += .9
        return x, y
    return (*split(1200, 0), *split(100, 1))


def toy_model():
    model = MnistNet(nn.Identity(), nn.Identity(), ClassifierHead(embed_dim=2, num_classes=2))
    with torch.no_grad():
        model.head.fc.weight.copy_(torch.eye(2))
        model.head.fc.bias.zero_()
    return model


def toy_bank():
    bank = OnlineHopfieldBank(2, 2, ema_alpha=1., merge_distance_threshold=2)
    bank.initialize_center(torch.zeros(2))
    for c, value in enumerate(([-1., 2.], [-1., 3.])):
        p = bank.plan_candidate(torch.tensor(value), c, bootstrap=True)
        bank.commit(p, example_id=c, step=0, epoch=1)
    return bank


def step(model, bank, cfg, x, y, writer=None, enabled=True, optimizer=None):
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=.001)
    return experiment.train_step(model, bank, optimizer, x, y, torch.arange(len(y)), epoch=2,
                                 batch=0, step=1, seed=0, run_id="fixture", cfg=cfg,
                                 writer=writer, memory_enabled=enabled)


def test_ordered_commits_keep_snapshot_eligibility_and_reassign(tmp_path):
    model, bank = toy_model(), toy_bank()
    cfg = config(tmp_path, num_classes=2)
    writer = EventWriter(tmp_path, 2, 1)
    result = step(model, bank, cfg, torch.tensor([[3., 0.], [3., 0.]]), torch.zeros(2, dtype=torch.long), writer)
    events = list(read_events(tmp_path, writer.flush()))
    writes = [e for e in events if e["type"] == "write"]
    examples = [e for e in events if e["type"] == "example"]
    assert result["counts"]["harmed"] == result["counts"]["merge"] == 2
    assert [e["hopfield_prediction"] for e in examples] == [1, 1]
    assert [w["local_before_prediction"] for w in writes] == [1, 0]
    assert [w["local_after_prediction"] for w in writes] == [0, 0]
    assert writes[1]["distance"] == pytest.approx(0.)
    assert [w["version_before"] for w in writes] == [2, 3]
    assert all(w["decision_snapshot_version"] == 2 for w in writes)
    assert bank.version == 4 and len(writes) == 2  # no extra diagnostic admissions
    assert all(e["working_version_after"] == e["working_version_before"] + 1 for e in examples)


def test_bootstrap_only_seeds_missing_classes_and_delays_comparisons(tmp_path):
    bank = OnlineHopfieldBank(2, 2)
    bank.initialize_center(torch.zeros(2))
    model = toy_model()
    x = torch.tensor([[3., 0.], [2., 0.], [0., 3.], [0., 2.]])
    y = torch.tensor([0, 0, 1, 1])
    writer = EventWriter(tmp_path, 10, 2)
    cfg = config(tmp_path, num_classes=2)
    result = step(model, bank, cfg, x, y, writer)
    assert result["compared"] == 0 and result["fallback_count"] == 4
    assert result["counts"]["seed"] == 2 and bank.occupancy() == [1, 1]
    assert all(e["hopfield_prediction"] is None for e in read_events(tmp_path, writer.flush()) if e["type"] == "example")
    assert step(model, bank, cfg, x, y)["compared"] == 4


def test_bootstrap_rejects_directly_wrong_and_warmup_never_writes(tmp_path):
    bank = OnlineHopfieldBank(2, 2)
    model = toy_model()
    cfg = config(tmp_path, num_classes=2)
    x, y = torch.tensor([[3., 0.], [0., 3.]]), torch.tensor([1, 0])
    step(model, bank, cfg, x, y)
    assert bank.version == 0
    bank.initialize_center(torch.zeros(2))
    result = step(model, bank, cfg, x, y)
    assert bank.version == 0 and result["counts"]["reason:direct_incorrect"] == 2


def test_no_write_for_helped_both_correct_or_both_wrong(tmp_path):
    cfg = config(tmp_path, num_classes=2)
    model, bank = toy_model(), toy_bank()
    before = copy.deepcopy(bank.state_dict())
    result = step(model, bank, cfg, torch.tensor([[3., 0.], [0., 3.], [0., 3.]]), torch.tensor([1, 1, 0]))
    assert result["counts"]["helped"] == result["counts"]["both_correct"] == result["counts"]["both_wrong"] == 1
    assert result["counts"]["skip"] == 3
    assert_state_equal(before, bank.state_dict())


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_direct_gradient_and_training_trajectory_parity(tmp_path):
    enabled = toy_model()
    disabled = copy.deepcopy(enabled)
    bank, other = toy_bank(), toy_bank()
    cfg = config(tmp_path, num_classes=2)
    a = torch.optim.Adam(enabled.parameters(), lr=.01)
    b = torch.optim.Adam(disabled.parameters(), lr=.01)
    rng = torch.get_rng_state().clone()
    for _ in range(3):
        x, y = torch.tensor([[3., 0.], [0., 3.], [2., 0.]]), torch.tensor([0, 1, 1])
        result_a = step(enabled, bank, cfg, x, y, optimizer=a)
        result_b = step(disabled, other, cfg, x, y, enabled=False, optimizer=b)
        assert result_a["loss"] == result_b["loss"]
        for p, q in zip(enabled.parameters(), disabled.parameters()):
            assert torch.equal(p.grad, q.grad) and torch.equal(p, q)
        assert_state_equal(a.state_dict(), b.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p is not bank.values for group in a.param_groups for p in group["params"])


def test_mode_rng_buffers_restore_and_evaluation_is_read_only(tmp_path):
    cfg = config(tmp_path)
    model = tiny_model(0).train()
    bank = experiment.make_bank(cfg, 16)
    bank.initialize_center(torch.zeros(16))
    saved = copy.deepcopy(bank.state_dict())
    state = torch.get_rng_state().clone()
    with diagnostic_mode(model):
        assert not model.training
        torch.rand(10)
    assert model.training and torch.equal(state, torch.get_rng_state())
    x, y = synthetic_data()[2:]
    result = experiment.evaluate(model, x, y, bank, cfg, torch.device("cpu"))
    assert result["adaptive_hopfield"]["accuracy"] is None
    assert result["adaptive_hopfield"]["fallback_count"] == len(y)
    assert result["adaptive_hopfield"]["evaluated"] == 0
    assert model.training
    assert_state_equal(saved, bank.state_dict())


def checkpoint(root):
    seed_root = root / "seed_0"
    latest = json.loads((seed_root / "latest.json").read_text())
    return torch.load(seed_root / latest["checkpoint"], weights_only=False)


def test_synthetic_end_to_end_replay_and_epoch_resume_equivalence(tmp_path):
    # Deliberately diffuse retrieval reliably exercises harmed/active writes;
    # this fixture is not an attempt to optimize MNIST performance.
    cfg = config(tmp_path, hopfield_beta=.01)
    data = synthetic_data()
    full = experiment.run_experiment(cfg, data=data, model_factory=tiny_model)
    first = checkpoint(full)
    partial = experiment.run_experiment(cfg, data=data, model_factory=tiny_model, stop_after_epoch=1)
    # Uncommitted crash tail must never become part of resumed events.
    (partial / "seed_0/events/orphan.jsonl.gz").write_bytes(b"invalid orphan")
    resumed = experiment.run_experiment(cfg, data=data, model_factory=tiny_model, resume_root=partial)
    second = checkpoint(resumed)
    for key in ("model", "optimizer", "bank", "shuffle_rng", "probe_reference", "previous_probe", "history"):
        assert_state_equal(first[key], second[key])
    assert first["optimizer_step"] == second["optimizer_step"]
    restored = experiment.make_bank(cfg, 16)
    restored.load_state_dict(first["bank"])
    replayed = replay_bank(experiment.make_bank(cfg, 16), full / "seed_0", first["log_cursor"])
    assert_state_equal(restored.state_dict(), replayed.state_dict())
    assert restored.ready and restored.version > 10
    events = list(read_events(full / "seed_0", first["log_cursor"]))
    examples = [e for e in events if e["type"] == "example"]
    writes = [e for e in events if e["type"] == "write"]
    assert len(examples) == 1600  # two post-warmup epochs
    assert len(writes) == restored.version
    assert len({w["seq"] for w in writes}) == len(writes)
    assert all(e["epoch"] > 1 for e in examples)
    summary = json.loads((full / "results.json").read_text())["seeds"][0]
    assert summary["fixed_bank_count"] == 20
    assert {r["method"] for r in summary["test_results"]} == {"direct", "adaptive_hopfield", "fixed_kmeans_reference"}
    assert all(r["evaluated"] == 100 for r in summary["test_results"])
    for row in summary["test_results"]:
        assert row["helped"] + row["harmed"] + row["both_correct"] + row["both_wrong"] == row["evaluated"]
        assert row["delta_vs_direct"] == pytest.approx((row["helped"] - row["harmed"]) / row["evaluated"], abs=1e-7)
    for path in ("summary.md", "results.csv", "plots/training.png", "plots/held_out.png"):
        assert (full / path).is_file()
    assert json.loads((full / "manifest.json").read_text())["status"] == "complete"
    # Different experimental settings cannot reuse the bank history.
    with pytest.raises(ValueError, match="mismatch"):
        experiment.run_experiment(cfg | {"hopfield_beta": 10.}, data=data,
                                  model_factory=tiny_model, resume_root=full)


@pytest.mark.parametrize("override", [dict(warmup_epochs=5), dict(ema_alpha=0), dict(hopfield_beta=True),
    dict(merge_distance_threshold=2.1), dict(normalization_epsilon=float("nan")), dict(seeds=[0, 0]),
    dict(memory_refresh="periodic"), dict(bank_full_policy="evict"), dict(write_distance="euclidean"),
    dict(learning_rate=float("inf")), dict(batch_size=True), dict(unknown=1),
    dict(gaussian_sigmas=[.1, .2]), dict(salt_pepper_probs=[0., 2.]), dict(occlusion_sizes=[0, 1.5]),
    dict(fixed_reference={"beta_policy": "validation_tuned"}), dict(fixed_reference={"unknown": True})])
def test_invalid_config(override):
    with pytest.raises(ValueError):
        experiment.validate_config(override)


def test_config_and_quick_have_postwarmup_epochs():
    full = experiment.load_config(experiment.DEFAULT_CONFIG)
    quick = experiment.load_config(experiment.DEFAULT_CONFIG, quick=True)
    assert full["total_epochs"] == 5 and full["seeds"] == [0, 1, 2]
    assert quick["total_epochs"] > quick["warmup_epochs"] >= 1
    assert quick["hopfield_beta"] == full["hopfield_beta"] == 5


def test_split_ids_reuse_existing_split_and_missing_class_preflight(tmp_path):
    data = synthetic_data()
    cfg = config(tmp_path)
    splits = experiment.splits_for(cfg, data, 0)
    legacy = make_splits(data, seed=0, val_size=100, train_subset=800, test_subset=100)
    assert not hasattr(legacy, "train_indices")
    assert torch.equal(splits.train_images, legacy.train_images)
    assert torch.equal(data[0][splits.train_indices], splits.train_images)
    assert not set(splits.train_indices.tolist()) & set(splits.val_indices.tolist())
    bad = (data[0], data[1].clamp_max(8), data[2], data[3])
    with pytest.raises(ValueError, match="missing"):
        experiment.splits_for(cfg, bad, 0)
    with pytest.raises(ValueError, match="more than k"):
        experiment.splits_for(cfg | {"fixed_reference": {"prototypes_per_class": 100}}, data, 0)


def test_incomplete_final_bank_never_reports_fallback_as_success(tmp_path):
    cfg = config(tmp_path, learning_rate=1e-30, total_epochs=2)
    def factory(seed):
        model = tiny_model(seed)
        with torch.no_grad():
            model.head.fc.weight.zero_()
            model.head.fc.bias.zero_()
            model.head.fc.bias[0] = 10
        return model
    root = experiment.run_experiment(cfg, data=synthetic_data(), model_factory=factory)
    state = checkpoint(root)
    assert state["history"][-1]["validation"]["adaptive_hopfield"]["accuracy"] is None
    assert json.loads((root / "manifest.json").read_text())["status"] == "complete"
    result = json.loads((root / "results.json").read_text())["seeds"][0]
    adaptive = [r for r in result["test_results"] if r["method"] == "adaptive_hopfield"]
    assert all(r["accuracy"] is None and r["evaluated"] == 0 and r["fallback_count"] == 100 for r in adaptive)


def test_nonfinite_training_is_not_silent_fallback(tmp_path):
    cfg = config(tmp_path, num_classes=2)
    model, bank = toy_model(), toy_bank()
    with pytest.raises(FloatingPointError):
        step(model, bank, cfg, torch.tensor([[float("nan"), 0.]]), torch.tensor([0]))
    assert bank.version == 2


def test_zero_warmup_center_precedes_training_examples(tmp_path):
    cfg = config(tmp_path, warmup_epochs=0, total_epochs=1, train_subset=100,
                 test_subset=20, fixed_reference=dict(prototypes_per_class=1))
    root = experiment.run_experiment(cfg, data=synthetic_data(), model_factory=tiny_model)
    state = checkpoint(root)
    events = list(read_events(root / "seed_0", state["log_cursor"]))
    assert events[0]["type"] == "center" and events[0]["optimizer_step"] == 0
    assert len([e for e in events if e["type"] == "example"]) == 100
    assert state["environment"]["device"] == "cpu"


def test_fixed_reference_failure_preserves_failed_status_and_checkpoint(tmp_path, monkeypatch):
    cfg = config(tmp_path, total_epochs=2, train_subset=200, test_subset=20)
    def fail(*args, **kwargs):
        raise ValueError("degenerate fixed fixture")
    monkeypatch.setattr(experiment, "build_memory", fail)
    with pytest.raises(FloatingPointError, match="Fixed reference construction failed"):
        experiment.run_experiment(cfg, data=synthetic_data(), model_factory=tiny_model)
    root = next((tmp_path / "full").iterdir())
    assert checkpoint(root)["epoch"] == 2
    failure = json.loads((root / "seed_0/failure.json").read_text())
    assert failure["numerical_failure"] and "degenerate fixed fixture" in failure["message"]
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"
    assert not (root / "results.csv").exists()
