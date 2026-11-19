"""Unit tests for Experiment 4 global responsibility-weighted dynamic prototype modules."""

from __future__ import annotations

import math

import pytest
import torch

from hopfieldpc import (
    GLOBAL_ABLATION_MODES,
    GlobalDynamicConfig,
    GlobalDynamicHopfieldBank,
    auto_hopfield_betas,
    auto_spawning_uncertainty_threshold,
    auto_split_scale,
    auto_support_thresholds,
    coalesce_uncertainty_total_variance,
    fission_uncertainty_inverse_total_variance,
    john_effective_noise,
    john_sharpness_index,
    kish_effective_sample_size,
    masked_global_responsibilities,
    online_similarity_penalty_step,
    responsibility_kalman_gain,
    responsibility_kalman_momentum_step,
    responsibility_momentum_velocity_update,
    riccati_steady_state_uncertainty,
    structural_uncertainty_update,
    weighted_spectral_fission,
)


def test_john_factor_limiting_behavior() -> None:
    p = torch.tensor([0.8, 0.8])
    r = torch.tensor([1.0, 0.0])
    r_eff = john_effective_noise(r, observation_noise=0.2, gamma_john=1.0, eps=1e-8)
    k = responsibility_kalman_gain(p, r_eff, eps=1e-8)

    # r -> 1 recovers P / (P + R_n), r -> 0 approaches 0.
    assert k[0].item() == pytest.approx(0.8 / (0.8 + 0.2), abs=1e-5)
    assert k[1].item() < 1e-6

    current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidate = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    velocity = torch.zeros_like(current)
    out = responsibility_kalman_momentum_step(
        current,
        candidate,
        velocity,
        p,
        r,
        observation_noise=0.2,
        gamma_mom=0.9,
        gamma_john=1.0,
        process_noise=0.01,
        eps=1e-8,
    )
    # Unassociated prototype (r=0) preserves its prior uncertainty and position.
    assert out.uncertainty[1].item() == pytest.approx(0.8, abs=1e-5)
    assert torch.allclose(out.new_value[1], current[1], atol=1e-5)
    assert out.uncertainty[0].item() < 0.8


def test_responsibility_momentum_velocity_update_limits() -> None:
    vel = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
    innov = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    r = torch.tensor([0.0, 1.0])
    v_next = responsibility_momentum_velocity_update(vel, innov, r, gamma_mom=0.9)
    assert torch.allclose(v_next[0], vel[0], atol=1e-6)
    expected_1 = 0.9 * vel[1] + 0.1 * innov[1]
    assert torch.allclose(v_next[1], expected_1, atol=1e-6)


def test_diffuse_vs_concentrated_support_and_sharpness() -> None:
    m_count = 100
    # Prototype 0: diffuse epsilon=0.05 across 100 samples; Prototype 1: concentrated 1.0 across 5 samples.
    n_mass = torch.tensor([m_count * 0.05, 5.0])
    s_mass = torch.tensor([m_count * (0.05**2), 5.0 * (1.0**2)])
    kish = kish_effective_sample_size(n_mass, s_mass)
    sharp = john_sharpness_index(n_mass, s_mass)

    assert n_mass[0].item() == pytest.approx(n_mass[1].item())
    assert kish[0].item() > kish[1].item()
    assert s_mass[0].item() < 0.3 and s_mass[1].item() == pytest.approx(5.0)
    assert sharp[0].item() == pytest.approx(0.05, abs=1e-5)
    assert sharp[1].item() == pytest.approx(1.0, abs=1e-5)


def test_gram_coalescence_and_law_of_total_variance() -> None:
    cfg = GlobalDynamicConfig(
        dim=4,
        max_memories=6,
        initial_memories=2,
        min_active_memories=1,
        tau_coal=0.9,
        coalescence_patience=1,
        tau_support=0.0,
        tau_eff=0.0,
        tau_uncert=100.0,  # disable spawning
        lambda_sim=0.0,
    )
    bank = GlobalDynamicHopfieldBank(cfg)
    bank.center.zero_()
    bank.center_initialized = True
    bank._activate_slot(0, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    bank._activate_slot(1, torch.tensor([2.0, 0.02, 0.0, 0.0]))
    bank.uncertainty[0] = 0.2
    bank.uncertainty[1] = 0.4
    bank.velocity[0] = torch.tensor([0.5, 0.0, 0.0, 0.0])
    bank.velocity[1] = torch.tensor([-0.5, 0.0, 0.0, 0.0])
    bank.initialized = True

    # Embeddings near the collinear direction so both prototypes get mass and remain redundant.
    features = torch.tensor([[1.5, 0.01, 0.0, 0.0]] * 8)
    stats = bank.adapt_structure(features, epoch=2, allow_spawn=False)

    assert stats["coalesced"] == 1
    assert bank.active_count == 1
    assert bank.total_coalesced == 1
    # Verify total-variance formula preserves separation spread.
    p_check = coalesce_uncertainty_total_variance(
        torch.tensor(0.2),
        torch.tensor(0.4),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.5, 0.0, 0.0, 0.0]),
        0.5,
        0.5,
    )
    assert p_check.item() == pytest.approx(0.3 + 0.25 / 4.0, abs=1e-6)


def test_weighted_spectral_fission_preserves_centroid() -> None:
    torch.manual_seed(0)
    c1 = torch.randn(20, 3) * 0.1 + torch.tensor([-2.0, 0.0, 0.0])
    c2 = torch.randn(20, 3) * 0.1 + torch.tensor([2.0, 0.0, 0.0])
    emb = torch.cat([c1, c2], dim=0)
    w = torch.full((40,), 1.0 / 40.0)
    res = weighted_spectral_fission(emb, w, split_scale=0.5)

    assert abs(res["u1"][0].item()) > 0.95
    assert torch.allclose(0.5 * (res["child_a"] + res["child_b"]), res["mean"], atol=1e-5)


def test_starvation_pruning_respects_global_min_and_deactivates_diffuse_slot() -> None:
    cfg = GlobalDynamicConfig(
        dim=4,
        max_memories=6,
        initial_memories=3,
        min_active_memories=1,
        tau_support=2.0,
        tau_eff=1.5,
        prune_patience=2,
        tau_uncert=100.0,
        tau_coal=0.999,
        beta_write=8.0,
        lambda_sim=0.0,
    )
    bank = GlobalDynamicHopfieldBank(cfg)
    bank.center.zero_()
    bank.center_initialized = True
    bank._activate_slot(0, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    bank._activate_slot(1, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    bank._activate_slot(2, torch.tensor([-1.0, -1.0, 0.0, 0.0]))
    bank.initialized = True

    # Features exclusively support slots 0 and 1; slot 2 gets starved.
    features = torch.cat(
        [
            torch.tensor([[1.0, 0.01, 0.0, 0.0]] * 10),
            torch.tensor([[0.01, 1.0, 0.0, 0.0]] * 10),
        ],
        dim=0,
    )
    s1 = bank.adapt_structure(features, epoch=2, allow_spawn=False)
    assert s1["pruned"] == 0
    s2 = bank.adapt_structure(features, epoch=3, allow_spawn=False)
    assert s2["pruned"] == 1
    assert bank.active_count == 2
    assert not bank.mask[2].item()


def test_fixed_capacity_and_dynamic_adaptive_modes() -> None:
    assert GLOBAL_ABLATION_MODES == ("fixed_capacity", "dynamic_adaptive")
    torch.manual_seed(42)
    emb = torch.cat(
        [
            torch.randn(16, 4) * 0.2 + torch.tensor([2.0, 0.0, 0.0, 0.0]),
            torch.randn(16, 4) * 0.2 + torch.tensor([-2.0, 0.0, 0.0, 0.0]),
        ],
        dim=0,
    )

    for mode in GLOBAL_ABLATION_MODES:
        cfg = GlobalDynamicConfig(
            dim=4,
            max_memories=8,
            initial_memories=3,
            min_active_memories=2,
            ablation_mode=mode,
            tau_support=1.0,
            tau_eff=0.5,
            n_kish_min=2.0,
            tau_uncert=0.3,
        )
        bank = GlobalDynamicHopfieldBank(cfg, online_update_mode="batch")
        bank.initialize_from_embeddings(emb, seed=0)
        up = bank.online_update(emb[:8])
        assert up["updated"] is True
        st = bank.adapt_structure(emb, epoch=2)
        assert st["active_total"] >= 2
        ret, w = bank.retrieve(emb[:4])
        assert ret.shape == (4, 4)
        assert w.shape == (4, 8)
        assert torch.isfinite(ret).all()


def test_online_similarity_penalty() -> None:
    values = torch.tensor([[1.0, 0.01], [1.0, -0.01], [-1.0, 0.0]])
    center = torch.zeros(2)
    mask = torch.tensor([True, True, False])
    updated, loss = online_similarity_penalty_step(
        values, center, mask, tau_sim=0.5, lambda_sim=0.1
    )
    assert loss > 0.0
    # Inactive slot 2 is untouched; active slots 0 and 1 repel each other.
    assert torch.allclose(updated[2], values[2])
    r, _ = masked_global_responsibilities(
        torch.tensor([[1.0, 0.0]]), updated, center, mask, beta_write=5.0
    )
    assert r[0, 2].item() == pytest.approx(0.0, abs=1e-7)


def test_coalescence_preserves_unmerged_smoothed_support_and_resets_persistence() -> None:
    cfg = GlobalDynamicConfig(
        dim=3,
        max_memories=5,
        initial_memories=3,
        min_active_memories=1,
        tau_coal=0.95,
        coalescence_patience=1,
        alpha_n=0.8,
        alpha_s=0.8,
        tau_support=0.0,
        tau_eff=0.0,
        tau_uncert=100.0,
        lambda_sim=0.0,
    )
    bank = GlobalDynamicHopfieldBank(cfg)
    bank.center.zero_()
    bank.center_initialized = True
    bank._activate_slot(0, torch.tensor([1.0, 0.0, 0.0]))
    bank._activate_slot(1, torch.tensor([1.0, 0.01, 0.0]))
    bank._activate_slot(2, torch.tensor([0.0, 0.0, 1.0]))
    bank.initialized = True

    # Pre-seed smoothed history on unmerged slot 2.
    bank.smoothed_mass_first_order[2] = 100.0
    bank.smoothed_mass_second_order[2] = 90.0
    bank.has_smoothed_history[2] = True

    features = torch.cat(
        [
            torch.tensor([[1.0, 0.005, 0.0]] * 6),
            torch.tensor([[0.0, 0.0, 1.0]] * 4),
        ],
        dim=0,
    )
    expected_slot2_raw = float(bank.compute_responsibilities(features)[0][:, 2].sum().item())
    expected_slot2_smoothed = 0.8 * 100.0 + 0.2 * expected_slot2_raw

    stats = bank.adapt_structure(features, epoch=2, allow_spawn=False)
    assert stats["coalesced"] == 1
    assert bank.smoothed_mass_first_order[2].item() == pytest.approx(
        expected_slot2_smoothed, rel=1e-4
    )
    assert int(bank.coal_persistence.sum().item()) == 0


def test_sequential_online_update_matches_repeated_step_observation() -> None:
    cfg = GlobalDynamicConfig(
        dim=3,
        max_memories=4,
        initial_memories=2,
        min_active_memories=1,
        lambda_sim=0.0,
    )
    emb = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [0.9, -0.1, 0.1],
            [-1.0, 0.3, -0.2],
        ]
    )
    init_pts = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.1, -0.1, 0.0],
        ]
    )
    b_seq = GlobalDynamicHopfieldBank(cfg, online_update_mode="sequential")
    b_obs = GlobalDynamicHopfieldBank(cfg, online_update_mode="sequential")
    b_seq.initialize_from_embeddings(init_pts, seed=0)
    b_obs.initialize_from_embeddings(init_pts, seed=0)

    b_seq.online_update(emb, apply_similarity_penalty=False)
    for i in range(emb.shape[0]):
        b_obs.step_observation(emb[i], apply_similarity_penalty=False)

    torch.testing.assert_close(b_seq.values, b_obs.values)
    torch.testing.assert_close(b_seq.velocity, b_obs.velocity)
    torch.testing.assert_close(b_seq.uncertainty, b_obs.uncertainty)


def test_fixed_capacity_does_not_mutate_uncertainty_in_adapt_structure() -> None:
    cfg = GlobalDynamicConfig(
        dim=3,
        max_memories=4,
        initial_memories=2,
        min_active_memories=1,
        ablation_mode="fixed_capacity",
    )
    bank = GlobalDynamicHopfieldBank(cfg)
    pts = torch.tensor(
        [[1.0, 0.0, 0.0], [1.2, 0.1, 0.0], [-1.0, 0.0, 0.0], [-1.1, -0.2, 0.0]]
    )
    bank.initialize_from_embeddings(pts, seed=0)
    bank.online_update(pts)
    p_before = bank.uncertainty.clone()
    bank.adapt_structure(pts, epoch=2)
    torch.testing.assert_close(bank.uncertainty, p_before)


def test_large_offset_covariance_precision() -> None:
    # Large mean offset (1e4) with small spread along dim 0 (0.05) tests float64 moment precision.
    offset = torch.tensor([10000.0, -10000.0, 5000.0])
    pts = torch.stack(
        [
            offset + torch.tensor([0.05, 0.0, 0.0]),
            offset + torch.tensor([-0.05, 0.0, 0.0]),
            offset + torch.tensor([0.04, 0.001, 0.0]),
            offset + torch.tensor([-0.04, -0.001, 0.0]),
        ]
    )
    cfg = GlobalDynamicConfig(
        dim=3,
        max_memories=4,
        initial_memories=1,
        min_active_memories=1,
        tau_uncert=0.01,
        n_kish_min=1.0,
        tau_support=0.0,
        tau_eff=0.0,
        ablation_mode="dynamic_adaptive",
    )
    bank = GlobalDynamicHopfieldBank(cfg, online_update_mode="batch")
    bank.center.copy_(offset - torch.tensor([1.0, 0.0, 0.0]))
    bank.center_initialized = True
    bank._activate_slot(0, offset)
    bank.uncertainty[0] = 0.8
    bank.initialized = True

    up = bank.online_update(pts)
    assert up["updated"] is True

    bank.uncertainty[0] = 0.8
    st = bank.adapt_structure(pts, epoch=2, allow_spawn=True)
    assert st["spawned"] == 1
    assert st["spawn_events"][0]["lambda1"] == pytest.approx(0.00205, rel=0.1)


def test_analytical_auto_calibration_formulas_and_auto_bank() -> None:
    # 1. Riccati fixed point satisfies P_inf = (1 - P_inf / (P_inf + R_n)) P_inf + Q.
    q, rn = 0.01, 0.05
    p_inf = riccati_steady_state_uncertainty(q, rn)
    rhs = (1.0 - p_inf / (p_inf + rn)) * p_inf + q
    assert p_inf == pytest.approx(rhs, abs=1e-10)

    # 2. Spawning threshold tracks changes in Q and R_n automatically.
    tau_low = auto_spawning_uncertainty_threshold(0.01, 0.05, lambda_disp=2.0, tau_coal=0.95)
    tau_high = auto_spawning_uncertainty_threshold(0.10, 0.20, lambda_disp=2.0, tau_coal=0.95)
    assert tau_low == pytest.approx(0.0441, abs=1e-3)
    assert tau_high > tau_low

    # 3. Betas, support thresholds, and split scale obey geometric invariants.
    br, bw = auto_hopfield_betas(dim=64, active_count=10)
    assert bw > br > 8.0
    t_sup, t_eff, n_kish = auto_support_thresholds(num_samples=55000, active_count=20, dim=64)
    assert t_eff < t_sup
    assert n_kish >= 2.0
    s_scale = auto_split_scale(centered_parent_norm=4.0, sigma1=2.0, tau_sim=0.85)
    assert 0.1 <= s_scale <= 2.0

    # 4. GlobalDynamicConfig.auto runs full Phase 1 + Phase 2 with zero manual thresholds.
    torch.manual_seed(0)
    emb = torch.cat(
        [
            torch.randn(20, 8) * 0.3 + torch.tensor([3.0] + [0.0] * 7),
            torch.randn(20, 8) * 0.3 + torch.tensor([-3.0] + [0.0] * 7),
        ],
        dim=0,
    )
    cfg = GlobalDynamicConfig.auto(dim=8, max_memories=10, initial_memories=2, min_active_memories=2)
    bank = GlobalDynamicHopfieldBank(cfg)
    bank.initialize_from_embeddings(emb, seed=0)
    bank.online_update(emb[:16])
    st = bank.adapt_structure(emb, epoch=2)
    assert "resolved_thresholds" in st
    assert st["resolved_thresholds"]["tau_uncert"] > 0.0
    ret, w = bank.retrieve(emb[:4])
    assert ret.shape == (4, 8) and w.shape == (4, 10)


def test_inverse_total_variance_fission_coalescence_duality() -> None:
    mean = torch.tensor([2.0, -1.0, 0.5, 1.0])
    delta = torch.tensor([0.2, -0.1, 0.0, 0.1])
    child_a = mean + delta
    child_b = mean - delta

    # 1. Exact inverse identity above the Riccati floor.
    p_scalar = torch.tensor(0.50)
    c_scalar = fission_uncertainty_inverse_total_variance(
        p_scalar, delta, process_noise=0.01, observation_noise=0.05
    )
    remerged_scalar = coalesce_uncertainty_total_variance(
        c_scalar, c_scalar, child_a, child_b, mean, 0.5, 0.5
    )
    assert remerged_scalar.item() == pytest.approx(p_scalar.item(), abs=1e-6)

    # 2. Floors cleanly at Riccati steady state P_inf when split explains >= parent variance.
    p_inf = riccati_steady_state_uncertainty(0.01, 0.05)
    c_floored = fission_uncertainty_inverse_total_variance(
        torch.tensor(0.05), torch.tensor([1.0, 1.0, 1.0, 1.0]), process_noise=0.01, observation_noise=0.05
    )
    assert c_floored.item() == pytest.approx(p_inf, abs=1e-6)


def test_auto_calibrated_bank_is_scale_invariant_across_orders_of_magnitude() -> None:
    torch.manual_seed(7)
    base_emb = torch.cat(
        [
            torch.randn(20, 8) * 0.4 + torch.tensor([2.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.randn(20, 8) * 0.4 + torch.tensor([-2.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
        dim=0,
    )
    cfg = GlobalDynamicConfig.auto(dim=8, max_memories=10, initial_memories=2, min_active_memories=2)

    banks = {}
    outputs = {}
    for scale in (1e-2, 1.0, 1e2):
        scaled = base_emb * scale
        b = GlobalDynamicHopfieldBank(cfg)
        b.initialize_from_embeddings(scaled, seed=0)
        b.online_update(scaled[:16])
        st = b.adapt_structure(scaled, epoch=2)
        ret, w = b.retrieve(scaled[:5])
        banks[scale] = (b, st)
        outputs[scale] = (ret, w)

    # Identical structural counts, identical attention weights, and equivariant retrieved vectors!
    assert banks[1e-2][1]["active_total"] == banks[1.0][1]["active_total"] == banks[1e2][1]["active_total"]
    assert banks[1e-2][1]["spawned"] == banks[1.0][1]["spawned"] == banks[1e2][1]["spawned"]
    torch.testing.assert_close(outputs[1e-2][1], outputs[1.0][1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(outputs[1e2][1], outputs[1.0][1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(outputs[1e2][0] / 1e2, outputs[1.0][0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(outputs[1e-2][0] / 1e-2, outputs[1.0][0], atol=1e-4, rtol=1e-4)


def test_structural_uncertainty_zero_sharpness_boundary_has_no_drift() -> None:
    p = torch.tensor([0.25, 0.25])
    sharp = torch.tensor([0.0, 1.0])
    disp = torch.tensor([0.15, 0.15])
    p_next, k_struct, _ = structural_uncertainty_update(p, sharp, disp)
    # Zero sharpness => zero structural gain and zero dispersion drift.
    assert k_struct[0].item() == pytest.approx(0.0, abs=1e-8)
    assert p_next[0].item() == pytest.approx(0.25, abs=1e-8)
    assert k_struct[1].item() > 0.0
