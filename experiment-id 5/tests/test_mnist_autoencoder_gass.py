"""Regression checks for the GASS equations and four-arm MNIST comparison."""

import math
import queue
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnist_autoencoder_ui_gass import parse_args
from gass_ui import ReferenceComparisonUI as GASSComparisonUI
from gass import ReferenceBank as GASSBank, ReferenceConfig as GASSConfig
from gass import geometry
from gass.geometry import SCORES, LOSSES, association_loss, probabilities


def tensor(rows):
    return torch.tensor(rows, dtype=torch.float64)


def test_apply_redraw_uses_latest_pending_preview():
    ui = GASSComparisonUI.__new__(GASSComparisonUI)
    ui.queue = queue.Queue()
    ui.root = Mock()
    ui.gass_bank = ui.displayed_report = None
    images = torch.arange(8.).view(8, 1, 1, 1)
    ui.display_clean = ui.display_noisy = images
    ui.display_generations = {"GASS": (images, images)}
    latest = {"GASS": (images + 1, images + 2)}
    noisy = images + .25
    rendered = []
    ui.update_images = lambda: rendered.append(ui.display_generations)

    ui.queue.put({"clean": images, "noisy": noisy, "generations": latest})
    ui.request_redraw()
    ui.poll_queue()

    assert len(rendered) == 1 and rendered[0] is latest
    assert ui.display_noisy is noisy


@pytest.mark.parametrize("momentum", [False, True])
def test_frozen_responsibilities_and_filter_match_information_form(momentum):
    cfg = GASSConfig(beta=2., observation_noise=2., process_noise=.2, momentum=momentum)
    bank = GASSBank(2, 2, 1, 4, cfg, key_dim=3).double()
    warm = tensor([[1, 0, 0], [0, 1, 0], [.8, .2, 0], [.1, 1, 0]])
    values = tensor([[1, 2], [-1, 3], [2, 1], [-2, 1]])
    assert bank.initialize_from_embeddings(warm, values, seed=3)
    u = warm[:2]
    v = tensor([[1.1, 1.3], [-.3, 2]])
    active = bank.mask
    m, k, weights = bank.values[active], bank.keys[active], bank.weights[active]
    dt = 1.5
    q, r = cfg.process_noise * bank.scale * dt, cfg.observation_noise * bank.scale
    if momentum:
        velocity = tensor([[.1, .2], [-.1, .05]])
        bank.velocity[active] = velocity
        transition = tensor([[1, dt], [0, cfg.damping]])
        p = transition @ bank.covariance[active] @ transition.T + q * torch.eye(2)
        predicted = m + dt * velocity
    else:
        p = bank.uncertainty[active] + q
        predicted = m
    angular = weights.log() + cfg.beta * (u @ k.T)
    torch.testing.assert_close(bank.read(u), angular.softmax(1) @ m)
    resp = (angular - (v[:, None] - predicted).square().sum(2) / (2*r)).softmax(1)
    mass = resp.sum(0)
    measurement = resp.T @ v / mass[:, None]
    bank.online_update(u, v, dt=dt, task_score=0.)
    if momentum:
        extra = torch.zeros_like(p)
        extra[:, 0, 0] = mass / r
        post = torch.linalg.inv(torch.linalg.inv(p) + extra)
        state = torch.stack((predicted, cfg.damping*velocity), dim=1)
        information = torch.linalg.solve(p, state)
        information[:, 0] += (mass / r)[:, None] * measurement
        expected = post @ information
        torch.testing.assert_close(bank.covariance[active], post)
        torch.testing.assert_close(bank.velocity[active], expected[:, 1])
        torch.testing.assert_close(bank.values[active], expected[:, 0])
        assert (torch.linalg.eigvalsh(post) > 0).all()
    else:
        post = (1/p + mass/r).reciprocal()
        torch.testing.assert_close(bank.uncertainty[active], post)
        torch.testing.assert_close(bank.values[active], post[:, None] * (predicted/p[:, None] + resp.T @ v/r))
    torch.testing.assert_close(bank.keys[active], F.normalize(resp.T @ u))
    torch.testing.assert_close(bank.weights[active], weights.lerp(mass/len(u), cfg.mixture_rate))
    assert (bank.weights[active] > 0).all() and bank.weights.sum().item() == pytest.approx(1.)


@pytest.mark.parametrize("momentum", [False, True])
def test_zero_queries_use_fallback_and_only_predict(momentum):
    bank = GASSBank(2, 2, 1, 3, GASSConfig(momentum=momentum)).double()
    zero = tensor([[0, 0], [0, 0]])
    assert not bank.initialize_from_embeddings(zero)
    assert bank.initialize_from_embeddings(tensor([[1, 0], [0, 1]]))
    before = {name: value.clone() for name, value in bank.named_buffers()}
    fallback = tensor([[3, 7], [4, 1]])
    torch.testing.assert_close(bank.read(zero, fallback), fallback)
    bank.online_update(zero)
    for name in ("values", "keys", "weights"):
        torch.testing.assert_close(getattr(bank, name), before[name])
    assert (bank.uncertainty[bank.mask] > before["uncertainty"][bank.mask]).all()
    assert not bank.adapt_structure(zero)["trials"]


def test_duplicate_merge_and_pruning_identities():
    cfg = GASSConfig(beta=3., capacity_cost=.12, margin=.01, patience=1, min_child_mass=100, edit_budget=1)
    bank = GASSBank(2, 3, 1, 4, cfg).double()
    v = tensor([[1, 0], [0, 1], [1, 1], [-1, 0]])
    u = F.normalize(v)
    assert bank.initialize_from_embeddings(v)
    bank.keys[:3] = tensor([[1, 0], [1, 0], [0, 1]])
    bank.values[:3] = tensor([[2, .5], [2, .5], [-1, 2]])
    bank.weights[:3] = tensor([.3, .2, .5])
    bank.uncertainty[:3] /= 3
    initial, resp, _ = bank._evaluate(bank.layout, u, v)
    read = bank.read(u)
    angular = cfg.beta * (u @ bank.keys[:3].T) + bank.weights[:3].log()
    r = cfg.observation_noise * bank.scale
    joint_score = angular - (v[:, None] - bank.values[:3]).square().sum(2) / (2*r)
    expected = (-joint_score.logsumexp(1).mean()
                + cfg.read_weight * (v - read).square().sum(1).mean() / (2*r)
                + cfg.capacity_cost * 3)
    assert initial == pytest.approx(expected.item())
    removal = bank._prune(bank.layout, 1).layout
    remaining = removal.weights > 0
    pruned_read = (cfg.beta * (u @ removal.keys[remaining].T) + removal.weights[remaining].log()).softmax(1) @ removal.values[remaining]
    w = angular.softmax(1)[:, 1:2]
    torch.testing.assert_close(pruned_read, (read - w * bank.values[1]) / (1-w))
    _, joint = bank._scores(u, v, bank.keys[:3], bank.values[:3], bank.weights[:3])
    _, joint2 = bank._scores(u, v, removal.keys[remaining], removal.values[remaining], removal.weights[remaining])
    torch.testing.assert_close(joint.logsumexp(1) - joint2.logsumexp(1),
                               torch.log1p(-bank.weights[1]) - torch.log1p(-resp[:, 1]))
    survivor = bank.uncertainty[2].clone()
    report = bank.adapt_structure(v)
    assert report["coalesced"] == 1 and bank.active_count == 2
    assert report["after"] == pytest.approx(initial - cfg.capacity_cost)
    torch.testing.assert_close(bank.read(u), read)
    torch.testing.assert_close(bank.uncertainty[0], bank.scale * cfg.initial_variance)
    torch.testing.assert_close(bank.uncertainty[2], survivor)
    assert not bank.keys[1].any() and not bank.values[1].any() and not bank.uncertainty[1]


def test_supported_tangent_split_scales_and_rejection_is_transactional():
    modes = tensor([[1, .55], [1, -.55]]).repeat(12, 1)
    runs = []
    for scale, margin in [(1., .001), (100., .001), (1., 1e8)]:
        cfg = GASSConfig(beta=12., observation_noise=.25, capacity_cost=.01, margin=margin, edit_budget=1)
        bank = GASSBank(2, 1, 1, 2, cfg).double()
        z = scale*modes
        assert bank.initialize_from_embeddings(z)
        bank.online_update(z)
        old_value, old_p = bank.values.clone(), bank.uncertainty.clone()
        report = bank.adapt_structure(z)
        if margin > 1:
            assert report["trials"] and not any(t["accepted"] for t in report["trials"])
            torch.testing.assert_close(bank.values, old_value)
            torch.testing.assert_close(bank.uncertainty, old_p)
            continue
        assert [t["operation"] for t in report["trials"] if t["accepted"]] == ["split"]
        assert report["after"] <= report["before"] - cfg.margin
        assert float(bank.keys[0] @ bank.keys[1]) == pytest.approx(cfg.merge_cosine)
        torch.testing.assert_close(bank.keys.norm(dim=1), torch.ones_like(bank.weights))
        torch.testing.assert_close(bank.weights @ bank.values, z.mean(0))
        runs.append((bank.read(z)/scale, bank.weights.clone(), bank.uncertainty/bank.scale, report))
    for a, b in zip(runs[0][:3], runs[1][:3]):
        torch.testing.assert_close(a, b)
    assert runs[0][-1]["after"] == pytest.approx(runs[1][-1]["after"])


def test_full_bank_can_accept_a_joint_replacement_at_the_capacity_floor():
    cfg = GASSConfig(observation_noise=.1, edit_budget=1)
    bank = GASSBank(2, 2, 2, 2, cfg).double()
    z = tensor([[1, 0], [-1, 0]]).repeat(12, 1)
    assert bank.initialize_from_embeddings(z)
    bank.keys[:] = bank.values[:] = tensor([[1, 0], [1, 0]])
    bank.weights[:] = .5
    report = bank.adapt_structure(z)
    assert bank.active_count == 2
    accepted = [t for t in report["trials"] if t["accepted"]]
    assert len(accepted) == 1 and accepted[0]["operation"].endswith("+birth")
    assert report["after"] <= report["before"] - cfg.margin
    assert bank.weights.sum().item() == pytest.approx(1)
    assert set(GASSComparisonUI.variant_names) == {"Direct", "EMA", "Dynamic", "GASS"}


@pytest.mark.parametrize("momentum", [False, True])
def test_100x_rescaling_preserves_multislot_reads_writes_and_decisions(momentum):
    gen = torch.Generator().manual_seed(38)
    warm, batch, window = (torch.randn(size, 4, generator=gen, dtype=torch.float64) for size in (35, 17, 32))
    cfg = GASSConfig(beta=5., observation_noise=.9, edit_budget=2, momentum=momentum)
    runs = []
    for factor in (1., 100.):
        bank = GASSBank(4, 3, 2, 6, cfg).double()
        bank.initialize_from_embeddings(factor*warm, seed=3)
        bank.online_update(factor*batch)
        bank.online_update(factor*batch.flip(1))
        report = bank.adapt_structure(factor*window)
        response = bank._evaluate(bank.layout, F.normalize(window), factor*window)[1]
        outputs = [bank.mask, bank.keys, bank.weights, response, bank.values/factor,
                   bank.read(factor*window)/factor, bank.uncertainty/bank.scale]
        if momentum:
            outputs.extend([bank.velocity/factor, bank.covariance/bank.scale])
        runs.append((outputs, report))
    for a, b in zip(runs[0][0], runs[1][0]):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    first, second = runs[0][1], runs[1][1]
    for report in (first, second):
        for trial in report["trials"]:
            assert trial["accepted"] == (trial["after"] <= trial["before"] - cfg.margin)
    assert [(t["operation"], t["accepted"]) for t in first["trials"]] == [
        (t["operation"], t["accepted"]) for t in second["trials"]
    ]
    assert first["after"] == pytest.approx(second["after"])


def test_prune_patience_advances_on_distinct_windows_and_birth_can_discover_a_mode():
    cfg = GASSConfig(edit_budget=1, patience=2, min_child_mass=100)
    bank = GASSBank(2, 2, 1, 3, cfg).double()
    points = tensor([[1, 0], [-1, 0]])
    bank.initialize_from_embeddings(points)
    bank.keys[:2] = bank.values[:2] = points
    bank.weights[:2] = tensor([.999999, .000001])
    samples = points[:1].repeat(8, 1)
    assert bank.adapt_structure(samples)["pruned"] == 0
    assert bank.adapt_structure(samples)["pruned"] == 1
    assert bank.weights.tolist() == [1., 0., 0.]

    # Opposite directions have zero tangent variation about the current key.
    # Only a residual-driven birth can discover the missing mode in this pool.
    bank = GASSBank(2, 1, 1, 2, GASSConfig(edit_budget=1)).double()
    z = points.repeat(12, 1)
    bank.initialize_from_embeddings(z)
    bank.keys[0] = bank.values[0] = points[0]
    report = bank.adapt_structure(z)
    assert [t["operation"] for t in report["trials"] if t["accepted"]] == ["birth"]
    assert bank.active_count == 2 and bank.weights.sum().item() == pytest.approx(1.)

def expected_score(name, a, b, radius, metric):
    """Independent scalar definitions on two raw MNIST-logit vectors."""
    if name == "cosine":
        return F.cosine_similarity(a[None], b[None])[0]
    if name == "euclidean":
        return -(a-b).square().sum()
    if name == "mahalanobis":
        return -(a-b) @ metric @ (a-b)
    if name == "dot":
        return a.dot(b / max(1., b.norm().item()/radius)) / len(a)**.5
    if name == "kl":
        p, q = probabilities(a), probabilities(b)
        return -(p*(p.log()-q.log())).sum()
    if name == "dice":
        p, q = a.sigmoid(), b.sigmoid()
        return 2*p.dot(q)/(p.square().sum()+q.square().sum()+1e-8)
    bits, other = a>0, b>0
    if name == "hamming":
        return -(bits != other).sum().double()
    union = (bits | other).sum()
    return (bits & other).sum()/union.double() if union else a.new_tensor(1.)


@pytest.mark.parametrize("read", SCORES)
@pytest.mark.parametrize("write", SCORES)
def test_mixed_adapters_are_used_for_read_write_and_frozen_objective(read, write):
    cfg = GASSConfig(beta=5., read_beta=2.5, write_beta=1.5, read_score=read, write_score=write,
                     value_aware=False, key_radius=1.4, retrieval_loss="absolute")
    metric = tensor([[2, .2], [.2, .6]])
    bank = GASSBank(2, 2, 1, 3, cfg, metric=metric if "mahalanobis" in (read,write) else None).double()
    raw = tensor([[1.7, -.4], [-.1, 1.1], [.8, .7], [-1, -1.2]])
    values = tensor([[3,2], [1,5], [-1, 3], [.1,-.3]])
    assert bank.initialize_from_embeddings(raw, values)
    active = bank.mask
    keys, means, mix = bank.keys[active], bank.values[active], bank.weights[active]
    def pair_scores(name):
        return torch.stack([torch.stack([expected_score(name, a, b, cfg.key_radius, metric) for b in keys]) for a in raw])
    read_logits = mix.log() + cfg.read_beta * pair_scores(read)
    write_logits = mix.log() + cfg.write_beta * pair_scores(write)
    expected_read = read_logits.softmax(1) @ means
    resp = write_logits.softmax(1)
    torch.testing.assert_close(bank.read(raw), expected_read)
    objective, actual, _ = bank._evaluate(bank.layout, raw, values)
    torch.testing.assert_close(actual, resp)
    expected_j = -write_logits.logsumexp(1).mean() + (expected_read-values).abs().mean() + cfg.capacity_cost*2
    assert objective == pytest.approx(float(expected_j))
    prior = bank.uncertainty[active] + bank.scale*cfg.process_noise
    variance = bank.scale*cfg.observation_noise
    post = (1/prior + resp.sum(0)/variance).reciprocal()
    bank.online_update(raw, values)
    torch.testing.assert_close(bank.uncertainty[active], post)
    torch.testing.assert_close(bank.values[active], post[:,None]*(means/prior[:,None] + resp.T @ values/variance))
    proposed = bank.geometry.score(write, raw, bank.keys[active])
    assert ((resp*(proposed-pair_scores(write))).sum(0) >= -1e-12).all()
    assert bank.geometry.feasible(bank.keys[active]).all()
    assert bank.weights.sum().item() == pytest.approx(1.)


@pytest.mark.parametrize("profile", SCORES)
@pytest.mark.parametrize("value_aware", [True, False])
def test_duplicate_merge_and_pruning_across_adapters(profile, value_aware):
    bank = GASSBank(2, 3, 1, 3, GASSConfig(read_score=profile, write_score=profile, beta=2.,
                       value_aware=value_aware, retrieval_loss="dice")).double()
    z = tensor([[1.,-.2], [-1.,.5], [.4,1.], [-1.,-1.]])
    assert bank.initialize_from_embeddings(z)
    bank.keys[1] = bank.keys[0]
    bank.values[1] = bank.values[0]
    bank.weights[:] = tensor([.25,.25,.5])
    score, response, _ = bank._evaluate(bank.layout,z,z)
    edited = bank._merge(bank.layout, 0,1).layout
    assert bank._evaluate(edited,z,z)[0] == pytest.approx(score-bank.cfg.capacity_cost)
    read, joint = bank._scores(z,z,bank.keys,bank.values,bank.weights)
    if not value_aware:
        torch.testing.assert_close(response, read.softmax(1))
    expected_read = read.softmax(1) @ bank.values
    w = read.softmax(1)[:,1:2]
    cut = bank._prune(bank.layout,1).layout
    active = cut.weights > 0
    rd, wr = bank._scores(z,z,cut.keys[active],cut.values[active],cut.weights[active])
    torch.testing.assert_close(rd.softmax(1) @ cut.values[active], (expected_read-w*bank.values[1])/(1-w))
    torch.testing.assert_close(joint.logsumexp(1)-wr.logsumexp(1), torch.log1p(-bank.weights[1])-torch.log1p(-response[:,1]))


@pytest.mark.parametrize("name", LOSSES)
def test_associative_losses_and_zero_conventions(name):
    a = tensor([[0, 0], [1, -1], [2., .5]])
    b = tensor([[0, 0], [0, 0], [-1., .2]])
    v = .4
    actual = association_loss(name,a,b,v)
    if name == "squared":
        expected = (a-b).square().sum(1)/(2*v)
    elif name == "absolute":
        expected = (a-b).abs().mean(1)
    elif name == "angular":
        expected = 1-F.cosine_similarity(a,b)
        expected[0]=0
    elif name == "dice":
        expected = 1 - torch.stack([expected_score("dice", x,y,1,None) for x,y in zip(a,b)])
    elif name == "bernoulli_kl":
        expected = torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=a),
                                                     torch.distributions.Bernoulli(logits=b)).mean(1)
    else:
        expected = torch.stack([-expected_score("kl",x,y,1,None) for x,y in zip(a,b)])
    torch.testing.assert_close(actual,expected)
    assert actual.isfinite().all() and (actual>=0).all()
    assert (probabilities(tensor([[10000,-10000]]))>0).all()


@pytest.mark.parametrize("name", ["euclidean","mahalanobis"])
def test_unconstrained_spectral_split_separation_in_its_metric(name):
    cfg = GASSConfig(read_score="dice", write_score=name, split_scale=1.3)
    metric = tensor([[2.,.5],[.5,1.]])
    bank = GASSBank(2,1,1,2,cfg,metric=metric if name=="mahalanobis" else None).double()
    z = tensor([[.2,.8], [-.7, .6], [1.8,-.1],[-2,-1.]])
    assert bank.initialize_from_embeddings(z)
    weights = tensor([.1,.2,.4,.3])
    keys, = bank.geometry.split_keys(z,weights,bank.keys[0])
    transform = torch.linalg.cholesky(metric) if name=="mahalanobis" else torch.eye(2,dtype=z.dtype)
    x = z @ transform
    centered = x - weights @ x
    eig = torch.linalg.eigvalsh(centered.T @ (weights[:,None]*centered))[-1]
    assert ((keys[0]-keys[1]) @ transform).norm().item() == pytest.approx(2*cfg.split_scale*eig.sqrt().item())
    torch.testing.assert_close(keys.mean(0),weights @ z)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA split regression")
@pytest.mark.parametrize("rows,dim", [(512, 784), (96, 32)])
def test_cuda_cosine_split_with_rank_one_support(rows, dim, monkeypatch):
    # A concentrated slot sees just two distinct directions in its window.
    # The projected covariance has one positive eigenvalue and a repeated zero.
    gen = torch.Generator().manual_seed(41)
    basis, _ = torch.linalg.qr(torch.randn(dim, 2, dtype=torch.float64, generator=gen))
    key, tangent = basis.T.cuda()
    u = torch.zeros(rows, dim, dtype=key.dtype, device=key.device)
    u[::2] = key + .5 * tangent
    u[1::2] = key - .5 * tangent
    responsibility = u.new_zeros(rows)
    responsibility[:8] = 1
    angle = .3
    cfg = GASSConfig(read_score="dice", retrieval_loss="dice", split_angles=(angle,))
    bank = GASSBank(dim, 1, 1, 2, cfg).double().cuda()

    eigh = torch.linalg.eigh

    def checked_eigh(matrix):
        # Stop before launching the affected solver if the regression returns.
        assert not matrix.is_cuda
        return eigh(matrix)

    monkeypatch.setattr(torch.linalg, "eigh", checked_eigh)
    children, = bank.geometry.split_keys(u, responsibility, key)
    torch.testing.assert_close(children @ key, key.new_full((2,), math.cos(angle)))
    offsets = (children @ tangent).sort().values
    torch.testing.assert_close(offsets, key.new_tensor([-math.sin(angle), math.sin(angle)]))
    torch.testing.assert_close(children.norm(dim=1), key.new_ones(2))


def test_validity_depends_on_profile_and_strict_separation_guards_writes():
    z = tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,0.]])
    bank = GASSBank(2,2,1,3,GASSConfig(beta=.1,separation=.5)).double()
    assert bank.initialize_from_embeddings(z)
    before = {name: value.clone() for name,value in bank.named_buffers()}
    bank.online_update(tensor([[1.,.4]]).repeat(4,1))
    for name,value in bank.named_buffers():
        torch.testing.assert_close(value,before[name])
    bank = GASSBank(2,2,1,3,GASSConfig(read_score="euclidean",write_score="kl")).double()
    assert bank.initialize_from_embeddings(z)
    u, _ = bank._observations(torch.cat((z, tensor([[float("nan"),1.]]))))
    assert len(u)==4
    assert bank.read(z).isfinite().all()
    with pytest.raises(ValueError,match="positive definite"):
        GASSBank(2,2,1,3,GASSConfig(write_score="mahalanobis"),metric=tensor([[1,2],[2,1]]))


def test_similarity_only_leaves_gaussian_noise_in_filter_and_gui_cli_selects_profiles():
    args = parse_args(["--read-score","dice","--write-score","euclidean","--gass-read-beta","4",
                       "--gass-write-beta","2","--similarity-only","--retrieval-loss","kl","--momentum"])
    assert (args.gass.read_score,args.gass.write_score,args.gass.beta_r,args.gass.beta_w)==("dice","euclidean",4,2)
    assert not args.gass.value_aware and args.gass.retrieval_loss=="kl" and args.gass.momentum
    a,b = (GASSBank(2,2,1,3,replace(args.gass,observation_noise=r)).double() for r in (.1,10.))
    z = tensor([[1,2],[-1,-2],[2,-1],[.5,1.]])
    for bank in (a,b):
        assert bank.initialize_from_embeddings(z)
    def weights(bank):
        return bank._scores(z,z,bank.keys[bank.mask],bank.values[bank.mask],bank.weights[bank.mask])[1].softmax(1)
    torch.testing.assert_close(weights(a),weights(b))
    for bank in (a,b):
        bank.online_update(z)
    assert (a.uncertainty[a.mask] < b.uncertainty[b.mask]).all()


@pytest.mark.parametrize("profile", SCORES)
def test_structural_edits_on_each_geometry_respect_actual_mixed_objective(profile):
    g = torch.Generator().manual_seed(81)
    z = torch.randn(48, 4, generator=g, dtype=torch.float64)
    cfg = GASSConfig(read_score="dice", write_score=profile, retrieval_loss="angular",
                     value_aware=False, edit_budget=3, beta=3., min_child_mass=1., patience=1)
    bank = GASSBank(4, 2, 1, 4, cfg).double()
    assert bank.initialize_from_embeddings(z)
    bank.online_update(z)
    for _ in range(2):
        before = bank._evaluate(bank.layout,z,z)[0]
        report = bank.adapt_structure(z)
        assert report["trials"] and report["before"] == pytest.approx(before)
        for trial in report["trials"]:
            assert trial["accepted"] == (trial["after"] <= trial["before"] - cfg.margin)
        assert report["after"] == pytest.approx(bank._evaluate(bank.layout,z,z)[0])
        assert 1 <= bank.active_count <= 4 and bank.weights.sum().item() == pytest.approx(1.)
        assert bank.geometry.feasible(bank.keys[bank.mask]).all()
        assert (bank.uncertainty[bank.mask] > 0).all()


def test_custom_score_bound_checked_medoid_and_user_defined_loss(monkeypatch):
    monkeypatch.setattr(geometry,"_EXTRA_SCORES",{})
    monkeypatch.setattr(geometry,"_EXTRA_LOSSES",{})
    adapter = geometry.ScoreAdapter(score=lambda a,b: -torch.cdist(a,b,p=1),
                                    distance=lambda a,b: torch.cdist(a,b,p=1), upper_bound=lambda u: 0.)
    geometry.register_score("negative-l1", adapter)
    geometry.register_loss("max-error", lambda v,h,r: (v-h).abs().amax(1))
    settings = GASSConfig(read_score="cosine",write_score="negative-l1",retrieval_loss="max-error")
    z = tensor([[1,0],[-1,1],[2,.7],[1,-2]])
    bank = GASSBank(2,2,1,3,settings).double()
    assert bank.initialize_from_embeddings(z)
    old = bank.keys[bank.mask].clone()
    _,resp,_ = bank._evaluate(bank.layout,z,z)
    bank.online_update(z)
    new = bank.keys[bank.mask]
    assert ((resp*(adapter.score(z,new)-adapter.score(z,old))).sum(0)>=-1e-12).all()
    assert bank.adapt_structure(z)["after"] <= bank.last_structure["before"]
    geometry.register_score("invalid-bound", replace(adapter,upper_bound=lambda u:-100.))
    with pytest.raises(ValueError,match="upper bound"):
        bank.geometry.score("invalid-bound",z,new)


def test_stationary_scalar_recurrence_empirical_covariance_and_directional_iteration():
    cfg = GASSConfig(process_noise=.1,observation_noise=.5)
    bank = GASSBank(2,1,1,2,cfg).double()
    x = tensor([[1,1],[1,-1]]).repeat(3,1)
    assert bank.initialize_from_embeddings(x)
    for _ in range(60):
        bank.online_update(x)
    q = cfg.process_noise * bank.scale
    rb = cfg.observation_noise * bank.scale / len(x)
    expected = ((q*q+4*q*rb).sqrt()-q)/2
    torch.testing.assert_close(bank.uncertainty[0],expected)
    mean,cov = bank.empirical_covariance(0,x)
    torch.testing.assert_close(mean,x.mean(0))
    torch.testing.assert_close(cov,torch.cov(x.T,correction=0))
    keys = bank.keys[bank.mask]
    old = (cfg.beta * (F.normalize(x) @ keys.T) + bank.weights[bank.mask].log()).logsumexp(1)
    new = (cfg.beta * (bank.refine_direction(x,2) @ keys.T) + bank.weights[bank.mask].log()).logsumexp(1)
    assert (new>=old-1e-12).all()


def test_full_objective_write_guard_and_packed_velocity_covariance():
    cfg = GASSConfig(momentum=True,initial_covariance=(2.,.25,.7),process_covariance=(.2,.01,.15),
                     guard_writes=True,guard_read_loss=True,read_score="dice",write_score="cosine",
                     retrieval_loss="absolute",value_aware=False,dt=.7)
    b = GASSBank(2,2,1,4,cfg).double()
    z = tensor([[1,-2],[-2,2],[.4,1],[2,.2]])
    assert b.initialize_from_embeddings(z)
    torch.testing.assert_close(b.covariance[0],b.scale*tensor([[2,.25],[.25,.7]]))
    for batch in (z,z.flip(1),-z):
        before = b._evaluate(b.layout,batch,batch)[0]
        b.online_update(batch)
        assert b._evaluate(b.layout,batch,batch)[0] <= before + 1e-12
        assert torch.linalg.eigvalsh(b.covariance[b.mask]).min()>0
    report = b.adapt_structure(z)
    assert all(t["read_after"]<=t["read_before"] for t in report["trials"] if t["accepted"])
