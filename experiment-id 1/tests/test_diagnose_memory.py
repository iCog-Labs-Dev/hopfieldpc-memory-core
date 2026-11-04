"""Independent mathematical and provenance fixtures for read-only diagnosis."""
import copy

import pytest
import torch

from diagnose_memory import independent_retrieve, memory_health, readout_metrics, refreshed_history
from online_memory import centered_retrieve
from benchmarks.mnist_baseline import ClassifierHead
from frozen_encoder_control import freeze_encoder, verify_frozen_encoder
from mnist_online_memory import train_step, validate_config
from online_memory import OnlineHopfieldBank
from diagnose_bank_selection import cluster_with_quotas


def test_float64_oracle_agrees_with_existing_core_on_nonzero_center():
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(19, 7, generator=generator)
    memory = torch.randn(23, 7, generator=generator)
    center = torch.randn(7, generator=generator)
    actual, weights = centered_retrieve(query, memory, center, beta=20)
    expected, expected_weights = independent_retrieve(query, memory, center, 20, 1e-8)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(weights.double(), expected_weights, rtol=2e-5, atol=2e-6)


def test_reencoding_preserves_accepted_history_ema_and_sources_without_aliasing():
    history = [dict(action="seed", entry_id=0, example_id=10, alpha=.1),
               dict(action="seed", entry_id=1, example_id=20, alpha=.1),
               dict(action="merge", entry_id=0, example_id=11, alpha=.1),
               dict(action="merge", entry_id=0, example_id=10, alpha=.1)]
    embeddings = {10: torch.tensor([1., 0.]), 11: torch.tensor([0., 2.]), 20: torch.tensor([-1., 0.])}
    old_history, old_embeddings = copy.deepcopy(history), copy.deepcopy(embeddings)
    actual = refreshed_history(history, embeddings)
    torch.testing.assert_close(actual[0], torch.tensor([.91, .18]))
    assert torch.equal(actual[1], embeddings[20]) and history == old_history
    actual.zero_()
    assert all(torch.equal(embeddings[k], v) for k, v in old_embeddings.items())
    with pytest.raises(ValueError, match="Nonsequential"):
        refreshed_history([dict(action="add", entry_id=9, example_id=10, alpha=.1)], embeddings)


def test_classifier_convex_identity_and_class_mass_are_separate_from_head_prediction():
    head = ClassifierHead(embed_dim=2, num_classes=2)
    with torch.no_grad():
        head.fc.weight.copy_(torch.eye(2))
        head.fc.bias.copy_(torch.tensor([.3, -.2]))
    memory = torch.tensor([[4., 0.], [0., 4.]])
    labels = torch.tensor([0, 1])
    z = torch.tensor([[4., 0.], [0., 4.], [0., 0.]])
    y = torch.tensor([0, 1, 0])
    metrics = readout_metrics(head, z, y, memory, labels, torch.zeros(2), 5., 1e-8)
    assert metrics["classifier_convex_identity_max_error"] < 1e-6
    assert metrics["accuracy"] == 1
    assert 0 <= metrics["correct_class_mass"] <= 1
    assert metrics["helped"] == metrics["harmed"] == 0
    healthy = memory_health(head, memory, labels)
    assert healthy["classified_as_own_class"] == 2
    wrong = memory_health(head, memory.flip(0), labels)
    assert wrong["classified_as_own_class"] == 0


def test_hard_readout_uses_shared_classifier_not_stored_labels():
    head = ClassifierHead(embed_dim=2, num_classes=2)
    with torch.no_grad():
        head.fc.weight.copy_(torch.eye(2))
        head.fc.bias.zero_()
    memory = torch.tensor([[2., 0.], [0., 2.]])
    # Deliberately incorrect labels: hard retrieval still predicts through head.
    labels = torch.tensor([1, 0])
    z = torch.eye(2)
    result = readout_metrics(head, z, torch.tensor([0, 1]), memory, labels,
                             torch.zeros(2), 5, 1e-8, hard=True)
    assert result["accuracy"] == 1 and result["class_mass_vote_accuracy"] == 0


def test_frozen_control_preserves_encoder_and_matches_direct_only_head_updates():
    from benchmarks.mnist_baseline import MnistNet
    torch.manual_seed(8)
    model = MnistNet(torch.nn.Linear(2, 2), torch.nn.Identity(), ClassifierHead(embed_dim=2, num_classes=2))
    expected = copy.deepcopy(model)
    original = freeze_encoder(model)
    freeze_encoder(expected)
    before_head = copy.deepcopy(model.head.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    head_optimizer = torch.optim.Adam(expected.head.parameters(), lr=.01)
    bank = OnlineHopfieldBank(2, 2)
    bank.initialize_center(torch.zeros(2))
    cfg = validate_config(dict(num_classes=2))
    x, y = torch.tensor([[1., 0.], [0., 1.], [2., 1.]]), torch.tensor([0, 1, 0])
    for step in range(4):
        train_step(model, bank, optimizer, x, y, torch.arange(3), epoch=2, batch=step, step=step,
                   seed=0, run_id="frozen-fixture", cfg=cfg)
        head_optimizer.zero_grad()
        torch.nn.functional.cross_entropy(expected.head(expected.encoder(x)), y).backward()
        head_optimizer.step()
        verify_frozen_encoder(model, original)
        for p, q in zip(model.head.parameters(), expected.head.parameters()):
            assert torch.equal(p, q)
    assert any(not torch.equal(v, before_head[k]) for k, v in model.head.state_dict().items())
    assert all(p.grad is None for p in model.encoder.parameters())


def test_matched_cluster_quotas_reuse_exact_budget_and_training_labels():
    generator = torch.Generator().manual_seed(19)
    features = torch.cat([torch.randn(30, 2, generator=generator) * .1 + torch.tensor([3., 0.]),
                          torch.randn(30, 2, generator=generator) * .1 + torch.tensor([-3., 0.])])
    labels = torch.tensor([0] * 30 + [1] * 30)
    memory, memory_labels = cluster_with_quotas(features, labels, [2, 3], seed=1)
    assert memory.shape == (5, 2) and memory_labels.tolist() == [0, 0, 1, 1, 1]
    assert torch.all(memory[:2, 0] > 2.) and torch.all(memory[2:, 0] < -2.)


def test_cosine_lookup_cannot_preserve_every_affine_classifier_decision():
    # Center is genuinely the mean of these stored training vectors. Two
    # queries differ in radius but have identical centered direction. The
    # classifier uses radius; score-only cosine retrieval cannot distinguish it.
    memory = torch.tensor([[.5, 0.], [2., 0.], [-2.5, 0.]])
    center = memory.mean(0)
    query = memory[:2]
    head = ClassifierHead(embed_dim=2, num_classes=2)
    with torch.no_grad():
        head.fc.weight.copy_(torch.tensor([[0., 0.], [1., 0.]]))
        head.fc.bias.copy_(torch.tensor([1., 0.]))
    direct = head(query).argmax(1)
    assert direct.tolist() == [0, 1]
    retrieved, weights = centered_retrieve(query, memory, center, beta=5.)
    assert torch.equal(weights[0], weights[1])
    assert torch.equal(retrieved[0], retrieved[1])
    assert not torch.equal(head(retrieved).argmax(1), direct)
