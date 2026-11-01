"""Deterministic policy/geometry tests; no MNIST downloads."""
import copy

import pytest
import torch

from hopfieldpc.functional import hopfield_retrieve
from online_memory import OnlineHopfieldBank, centered_retrieve, correctness_categories


def seed(bank, value, label=0, step=0):
    proposal = bank.plan_candidate(torch.tensor(value, dtype=torch.float32), label, bootstrap=True)
    return bank.commit(proposal, example_id=10 + label, step=step, epoch=1)


def ready_bank(**kwargs):
    bank = OnlineHopfieldBank(2, num_classes=1, **kwargs)
    bank.initialize_center(torch.zeros(2))
    seed(bank, [0.7, 0.3])
    return bank


def test_empty_warmup_bootstrap_snapshots_and_detachment():
    bank = OnlineHopfieldBank(2, 2)
    assert not bank.ready and bank.occupancy() == [0, 0]
    assert bank.plan_candidate(torch.ones(2), 0, bootstrap=True).reason == "warmup"
    with pytest.raises(ValueError, match="incomplete"):
        bank.retrieve(torch.ones(1, 2))
    bank.initialize_center(torch.zeros(2))
    snapshot = bank.snapshot()
    source = torch.tensor([1., 0.], requires_grad=True)
    proposal = bank.plan_candidate(source, 0, bootstrap=True)
    bank.commit(proposal, example_id=7, step=2, epoch=1)
    source.data.fill_(9)
    assert torch.equal(bank.values[0], torch.tensor([1., 0.]))
    assert not snapshot.ready and snapshot.values.shape == (0, 2)
    assert bank.plan_candidate(torch.tensor([2., 0.]), 0, bootstrap=True).reason == "class_already_seeded"
    seed(bank, [0., 1.], 1)
    assert bank.ready and bank.occupancy() == [1, 1]
    assert not list(bank.parameters())
    assert bank.values.grad_fn is None and not bank.values.requires_grad
    with pytest.raises(ValueError, match="frozen"):
        bank.initialize_center(torch.ones(2))


def test_correctness_table_not_confidence_or_loss():
    direct_logits = torch.tensor([[.01, 0.], [0., 100.], [.1, 0.], [0., .01]])
    hop_logits = torch.tensor([[0., .01], [100., 0.], [100., 0.], [0., 100.]])
    assert correctness_categories(direct_logits.argmax(1), hop_logits.argmax(1), torch.zeros(4, dtype=torch.long)) == [
        "harmed", "helped", "both_correct", "both_wrong"]
    with pytest.raises(ValueError):
        correctness_categories(torch.zeros(3), torch.zeros(4), torch.zeros(4))


def test_exact_ema_stable_id_contributions_and_size():
    bank = ready_bank(merge_distance_threshold=1, ema_alpha=.1)
    proposal = bank.plan_candidate(torch.tensor([.9, .1]), 0)
    assert proposal.action == "merge"
    event = bank.commit(proposal, example_id=10, step=3, epoch=2)
    torch.testing.assert_close(bank.values[0], torch.tensor([.72, .28]))
    assert event["entry_id"] == 0 and event["size_before"] == event["size_after"] == 1
    assert bank.metadata[0]["accepted_exposure_count"] == 2  # repeat source is another exposure
    assert bank.metadata[0]["first_source_id"] == bank.metadata[0]["latest_source_id"] == 10
    assert bank.metadata[0]["created_step"] == 0 and bank.metadata[0]["updated_step"] == 3
    assert event["version_after"] == 2


def test_threshold_equality_add_capacity_and_existing_entry_protection():
    bank = OnlineHopfieldBank(2, 1, 2, merge_distance_threshold=1)
    bank.initialize_center(torch.zeros(2))
    seed(bank, [1., 0.])
    equality = bank.plan_candidate(torch.tensor([0., 1.]), 0)
    assert equality.distance == 1 and equality.action == "merge"
    bank.threshold = .1
    old = bank.values.clone()
    add = bank.plan_candidate(torch.tensor([0., 1.]), 0)
    bank.commit(add, example_id=2, step=1, epoch=1)
    assert bank.occupancy() == [2] and torch.equal(bank.values[0], old[0])
    full = bank.plan_candidate(torch.tensor([-1., -1.]), 0)
    assert full.action == "skip" and full.reason == "class_capacity_reached"
    assert bank.next_id == 2


def test_nearest_same_class_tie_and_other_classes_unchanged():
    bank = OnlineHopfieldBank(2, 2, 3, merge_distance_threshold=.01)
    bank.initialize_center(torch.zeros(2))
    seed(bank, [1., 0.], 0)
    seed(bank, [1., 1.], 1)
    proposal = bank.plan_candidate(torch.tensor([0., 1.]), 0)
    bank.commit(proposal, example_id=8, step=1, epoch=1)
    before = bank.values.clone()
    bank.threshold = 2
    tied = bank.plan_candidate(torch.tensor([1., 1.]), 0)
    assert tied.slot == 0  # class 1 is closer but cannot receive a class-0 write
    bank.commit(tied, example_id=9, step=2, epoch=1)
    assert torch.equal(before[1:], bank.values[1:])
    with pytest.raises(ValueError, match="Stale"):
        bank.commit(tied, example_id=9, step=2, epoch=1)


@pytest.mark.parametrize("vector, reason", [([0., 0.], "degenerate_centered_key"),
                                            ([float("nan"), 1.], "nonfinite_candidate"),
                                            ([float("inf"), 1.], "nonfinite_candidate")])
def test_invalid_candidates_never_mutate(vector, reason):
    bank = ready_bank()
    before = copy.deepcopy(bank.state_dict())
    assert bank.plan_candidate(torch.tensor(vector), 0).reason == reason
    assert torch.equal(before["values"], bank.values)


def test_degenerate_ema_does_not_corrupt_bank():
    bank = OnlineHopfieldBank(2, 1, ema_alpha=.5, merge_distance_threshold=2)
    bank.initialize_center(torch.zeros(2))
    seed(bank, [1., 0.])
    proposal = bank.plan_candidate(torch.tensor([-1., 0.]), 0)
    assert proposal.reason == "invalid_proposed_update" and not proposal.is_write
    assert bank.version == 1 and torch.equal(bank.values[0], torch.tensor([1., 0.]))


def test_all_class_retrieval_inactive_exclusion_raw_values_and_center():
    bank = OnlineHopfieldBank(2, 2)
    center = torch.tensor([2., -3.])
    bank.initialize_center(center)
    seed(bank, [5., -3.], 0)
    seed(bank, [2., -2.], 1)
    bank.values[2:].fill_(float("nan"))  # must never enter retrieval
    query = torch.tensor([[2., -2.], [2., -3.]])
    result, weights = bank.retrieve(query)
    active = bank.snapshot().values
    expected, expected_weights = hopfield_retrieve(query - center, active - center, beta=5, normalize=True)
    torch.testing.assert_close(result, weights @ active)
    torch.testing.assert_close(result, expected + center)
    torch.testing.assert_close(weights, expected_weights)
    assert weights.shape == (2, 2) and weights[0, 1] > weights[0, 0]
    torch.testing.assert_close(weights[1], torch.tensor([.5, .5]))
    # Relabeling entries does not change all-class retrieval.
    bank.labels[:2] = bank.labels[:2].flip(0)
    torch.testing.assert_close(bank.retrieve(query)[0], result)
    with pytest.raises(FloatingPointError):
        bank.retrieve(torch.tensor([[float("nan"), 0.]]))


def test_experiment_epsilon_and_repeated_retrieval_without_core_changes():
    query = torch.tensor([[1e-6, 0.]])
    memory = torch.eye(2)
    default = centered_retrieve(query, memory, torch.zeros(2), beta=1)[1]
    custom = centered_retrieve(query, memory, torch.zeros(2), beta=1, normalization_epsilon=.1)[1]
    assert default[0, 0] > custom[0, 0]
    state = query
    for _ in range(3):
        state, expected_weights = centered_retrieve(state, memory, torch.zeros(2), beta=1,
                                                    normalization_epsilon=.1)
    result, actual_weights = centered_retrieve(query, memory, torch.zeros(2), beta=1,
                                               normalization_epsilon=.1, num_updates=3)
    torch.testing.assert_close(result, state)
    torch.testing.assert_close(actual_weights, expected_weights)
    raw, weights = centered_retrieve(torch.zeros(1, 2), memory, torch.zeros(2))
    torch.testing.assert_close(raw, weights @ memory)


@pytest.mark.parametrize("kwargs", [dict(dim=True), dict(num_classes=0), dict(max_memories_per_class=0),
                                     dict(ema_alpha=0), dict(ema_alpha=1.1), dict(beta=float("inf")),
                                     dict(num_updates=True), dict(merge_distance_threshold=2.1),
                                     dict(merge_distance_threshold=float("nan")), dict(normalization_epsilon=0)])
def test_bank_config_errors(kwargs):
    with pytest.raises(ValueError):
        OnlineHopfieldBank(**({"dim": 2} | kwargs))


@pytest.mark.parametrize("epsilon", [0, -1, True, float("inf"), float("nan")])
def test_invalid_epsilon_rejected_by_experiment_interfaces(epsilon):
    with pytest.raises(ValueError):
        centered_retrieve(torch.ones(1, 2), torch.eye(2), torch.zeros(2), normalization_epsilon=epsilon)
    with pytest.raises(ValueError):
        OnlineHopfieldBank(2, normalization_epsilon=epsilon)


def test_candidate_shapes_labels_and_snapshots_are_independent():
    bank = ready_bank()
    snapshot = bank.snapshot()
    snapshot.values.zero_()
    assert bank.values[0].abs().sum() > 0
    for query, label in [(torch.ones(1, 2), 0), (torch.ones(2), -1), (torch.ones(2), True)]:
        with pytest.raises(ValueError):
            bank.plan_candidate(query, label)
    restored = ready_bank()
    restored.load_state_dict(copy.deepcopy(bank.state_dict()))
    assert restored.get_extra_state() == bank.get_extra_state()
    assert restored.storage(5)["entry_age_steps"] == [5]
