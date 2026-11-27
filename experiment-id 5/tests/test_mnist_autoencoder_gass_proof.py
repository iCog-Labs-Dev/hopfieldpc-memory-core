"""Checks for the continuous safeguarded MNIST memory and its optional stream bound."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnist_autoencoder_ui_gass_proof import parse_args
from gass_autoencoder import GASSVariant
from gass import GASSBank, GASSConfig
from gass.geometry import association_loss


def example(*, initial=3, **overrides):
    rng = torch.Generator().manual_seed(930)
    u = torch.randn(21, 4, generator=rng)
    v = torch.randn(21, 3, generator=rng)
    bank = GASSBank(3, initial, 1, 5, GASSConfig(**overrides), key_dim=4)
    assert bank.initialize_from_embeddings(u, v, seed=1)
    return bank, u, v


def buffer_copy(bank):
    return {k: v.clone() for k, v in bank.named_buffers()}


def test_cli_defaults_and_stable_pixel_kl():
    args = parse_args([])
    assert args.gass_variant_class is GASSVariant
    assert (args.gass.read_score, args.gass.write_score, args.gass.retrieval_loss) == ("dice", "dice", "bernoulli_kl")
    v = torch.tensor([[1000., -1000.]], dtype=torch.double)
    h = (-v).requires_grad_()
    error = association_loss("bernoulli_kl", v, h, 1.)
    torch.testing.assert_close(error, v.new_tensor([1000.]))
    torch.testing.assert_close(torch.autograd.grad(error.sum(), h)[0], (h.sigmoid() - v.sigmoid()) / 2)
    with pytest.raises(SystemExit):
        parse_args(["--recall-steps", "2"])
    with pytest.raises(ValueError):
        GASSBank(2, 2, 1, 5, GASSConfig(mixture_floor=.2))


@pytest.mark.parametrize("score,iterations", [("dice", 0), ("cosine", 3), ("dot", 3)])
def test_frozen_continuous_read_matches_explicit_iteration(score, iterations):
    bank, u, v = example(read_score=score, write_score=score, recall_steps=iterations, beta=2.)
    snapshot = buffer_copy(bank)
    keys, means, pi = bank.keys[bank.mask], bank.values[bank.mask], bank.weights[bank.mask]
    state = u
    if score == "dice":
        p, q = u.sigmoid(), keys.sigmoid()
        logits = pi.log() + 2 * (2 * (p @ q.T) + 1e-8) / (p.square().sum(1, keepdim=True) + q.square().sum(1) + 1e-8)
    else:
        beta = 2. if score == "cosine" else 2 / math.sqrt(u.shape[1])
        if score == "cosine":
            state = torch.nn.functional.normalize(state)
        for _ in range(iterations):
            state = (pi.log() + beta * (state @ keys.T)).softmax(1) @ keys
            if score == "cosine":
                state = torch.nn.functional.normalize(state)
        logits = pi.log() + beta * (state @ keys.T)
    expected = logits.softmax(1) @ means
    torch.testing.assert_close(bank.read(u), expected)
    _, _, _, read_loss = bank._evaluate(bank.layout, u, v, details=True)
    assert read_loss == pytest.approx(float(association_loss("bernoulli_kl", v, expected, 1.).mean()), abs=1e-7)
    for name, buf in bank.named_buffers():
        assert torch.equal(buf, snapshot[name])


@pytest.mark.parametrize("loss", ["bernoulli_kl", "squared"])
@pytest.mark.parametrize("aware", [False, True])
def test_arku_variance_trust_and_cyclic_blocks(loss, aware):
    bank, u, v = example(retrieval_loss=loss, value_aware=aware, logit_bound=.75, mixture_floor=.05)
    v = v + 2.
    active = bank.mask
    old_r = bank.variance.double().clone()
    state = bank._active_layout(bank.layout)
    _, joint = bank._scores(u.double(), v.double(), state.keys, state.values, state.weights)
    mass = joint.softmax(1).sum(0)
    prior = bank._clip_trust(bank.uncertainty[active].double() + bank.cfg.process_noise * bank.scale)
    residual = (v.double()[:, None] - state.values).square().sum(-1)
    expected_r = ((residual * joint.softmax(1)).sum() / v.numel()).clamp(
        bank.cfg.variance_min * bank.scale, bank.cfg.variance_max * bank.scale) if aware else old_r
    old_cost = bank._evaluate(bank.layout, u, v)[0]
    report = bank.online_update(u, v)
    torch.testing.assert_close(bank.uncertainty[active].double(), prior / (1 + prior * mass / old_r))
    torch.testing.assert_close(bank.variance.double(), expected_r)
    assert report["before"] == pytest.approx(old_cost)
    costs = [report[key] for key in ("before", "variance", "values", "keys", "after")]
    assert all(b <= a for a, b in zip(costs, costs[1:]))
    penalty = ((bank.values[active].double() - state.values).square().sum(1) / prior).sum() / (2 * len(u))
    assert costs[2] + float(penalty) <= costs[1] + 1e-10
    assert bank.values.abs().max() <= .75 and bank.keys.abs().max() <= .75
    assert (bank.weights[active] >= .05).all()
    torch.testing.assert_close(bank.weights.sum(), torch.tensor(1.))
    trust = bank.uncertainty.clone()
    history = bank.refine_window(u, v, sweeps=3)
    assert history[-1]["after"] <= costs[-1] and torch.equal(bank.uncertainty, trust)
    assert history[-1]["after"] == pytest.approx(bank._evaluate(bank.layout, u, v)[0])


def test_variance_normalization_fixed_scale_and_closed_form_value_step():
    bank, u, v = example(initial=1, retrieval_loss="squared", variance_min=1., variance_max=1.)
    state = bank._active_layout(bank.layout)
    v = v.double() + .4
    r = bank.variance.double().clone()
    prior = bank._clip_trust(bank.uncertainty[bank.mask].double() + bank.cfg.process_noise * bank.scale)
    strength = (1 / r + bank.cfg.read_weight / bank.retrieval_scale)
    expected = state.values + strength * (v - state.values).sum(0) / (1 / prior + len(v) * strength)
    old, _, _, read_old = bank._objective(state, u.double(), v, r)
    new, _, _, read_new = bank._objective(state, u.double(), v, 2 * r)
    torch.testing.assert_close(read_old, read_new)
    delta = (bank.dim / 2) * math.log(2) - (v - state.values).square().sum(1).mean() / (4 * r)
    torch.testing.assert_close(new - old, delta)
    bank.online_update(u, v)
    torch.testing.assert_close(bank.values[bank.mask].double(), expected)


@pytest.mark.parametrize("momentum", [False, True])
def test_zero_stream_budget_restores_parameters_and_advances_only_fresh_trust(momentum):
    bank, u, v = example(momentum=momentum, stream_budget=0., refinement_sweeps=1)
    original = buffer_copy(bank)
    report = bank.online_update(u, v + 3)
    assert not report["budget_accepted"] and report["after"] == report["fresh_before"]
    assert not torch.equal(bank.uncertainty, original["uncertainty"])
    trust = bank.uncertainty.clone()
    edit_report = bank.adapt_structure(u, v, epoch=1)
    assert edit_report["after"] == edit_report["before"]
    assert int(bank.budget_step) == 3
    for name in ("keys", "values", "weights", "variance"):
        assert torch.equal(getattr(bank, name), original[name])
    assert torch.equal(bank.uncertainty, trust)
    if momentum:
        assert (torch.linalg.eigvalsh(bank.covariance[bank.mask]) > 0).all()
    with pytest.raises(ValueError):
        bank.initialize_from_embeddings(u, v)


def test_geometric_budget_on_alternating_stream_survives_state_dict():
    bank, u, v = example(stream_budget=3.)
    total = 0.
    for step in range(10):
        snapshot = buffer_copy(bank)
        report = bank.online_update(u, v + (-1.) ** step)
        movement = max(float((getattr(bank, name) - snapshot[name]).abs().max())
                       for name in ("keys", "values", "weights", "variance"))
        assert movement <= report["allowance"] == 3. / 2 ** (step + 1)
        total += movement
    assert 0 < total <= 3.
    restored, _, _ = example(stream_budget=3.)
    restored.load_state_dict(bank.state_dict())
    assert restored.online_update(u, v)["allowance"] == 3. / 2 ** 11


@pytest.mark.parametrize("budget", [None, 0.])
def test_duplicate_merge_uses_same_window_and_restores_rejected_filters(budget):
    bank, u, v = example(initial=2, capacity_cost=2., margin=.25, patience=1, edit_budget=1,
                         min_child_mass=1000., stream_budget=budget, momentum=True)
    bank.keys[1].copy_(bank.keys[0])
    bank.values[1].copy_(bank.values[0])
    bank.weights[:2] = .5
    bank.uncertainty[:2].mul_(.7)
    bank.covariance[:2].mul_(.7)
    before = buffer_copy(bank)
    report = bank.adapt_structure(u, v, epoch=3)
    assert torch.equal(bank.variance, before["variance"])
    if budget is None:
        assert report["coalesced"] == 1 and bank.active_count == 1
        assert report["before"] - report["after"] == pytest.approx(2.)
        assert not bank.covariance[~bank.mask].any()
    else:
        assert not report["budget_accepted"] and report["coalesced"] == 0
        assert report["trials"][0]["budget_rejected"]
        for name, buf in bank.named_buffers():
            if name != "budget_step":
                assert torch.equal(buf, before[name])
