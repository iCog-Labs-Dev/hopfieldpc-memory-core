"""Global responsibility-weighted dynamic prototype memory bank (Experiment 4).

Implements:
- Masked pre-allocated buffer state up to N_max slots (no dynamic tensor resizing).
- Phase 1: Intra-epoch global soft responsibilities (beta_write), John Factor effective
  observation noise, responsibility-aware Kalman-momentum updates, and masked online
  soft similarity repulsion.
- Phase 2: Epoch-end structural adaptation (temporally smoothed support, Kish ESS,
  John Sharpness Index, structural uncertainty, starvation/diffuse pruning,
  persistent Gram-matrix coalescence with Law-of-Total-Variance fusion, and
  responsibility-weighted spectral fission with Inverse Law-of-Total-Variance child P).
- Supports ``fixed_capacity`` (control baseline) and ``dynamic_adaptive`` modes.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .api import (
    GLOBAL_ABLATION_MODES,
    GlobalDynamicConfig,
    validate_global_ablation_mode,
)
from .diagnostics import (
    gram_off_diagonal_stats,
    john_sharpness_index,
    kish_effective_sample_size,
    support_diagnostics_summary,
)
from .functional import (
    auto_hopfield_betas,
    auto_smoothing_alpha,
    auto_support_thresholds,
    kmeans_prototypes,
    masked_centered_retrieve,
    masked_global_responsibilities,
    masked_gram_matrix,
    online_similarity_penalty_step,
    weighted_spectral_fission,
)
from .kalman import (
    auto_spawning_uncertainty_threshold,
    coalesce_uncertainty_total_variance,
    fission_uncertainty_inverse_total_variance,
    john_effective_noise,
    responsibility_kalman_momentum_step,
    riccati_steady_state_uncertainty,
    structural_uncertainty_update,
)


class GlobalDynamicHopfieldBank(nn.Module):
    """Class-agnostic global soft-responsibility dynamic Hopfield memory bank."""

    def __init__(
        self,
        config: GlobalDynamicConfig,
        *,
        online_update_mode: str = "batch",
    ) -> None:
        super().__init__()
        if online_update_mode not in ("batch", "sequential"):
            raise ValueError("online_update_mode must be 'batch' or 'sequential'.")
        self.cfg = config
        self.online_update_mode = online_update_mode
        validate_global_ablation_mode(config.ablation_mode)

        n = config.max_memories
        d = config.dim
        self.dim = d
        self.max_memories = n

        # Pre-allocated masked buffer state (Sec. 3.2).
        p0_buf = 1.0 if config.initial_uncertainty is None else float(config.initial_uncertainty)
        self.register_buffer("values", torch.zeros(n, d))
        self.register_buffer("velocity", torch.zeros(n, d))
        self.register_buffer("uncertainty", torch.full((n,), p0_buf))
        self.register_buffer("mask", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("center", torch.zeros(d))
        self.register_buffer("variance_scale", torch.tensor(p0_buf))

        # Support, sharpness, starvation, coalescence persistence, and lifetime tracking.
        self.register_buffer("mass_first_order", torch.zeros(n))
        self.register_buffer("mass_second_order", torch.zeros(n))
        self.register_buffer("smoothed_mass_first_order", torch.zeros(n))
        self.register_buffer("smoothed_mass_second_order", torch.zeros(n))
        self.register_buffer("kish_ess", torch.zeros(n))
        self.register_buffer("effective_support", torch.zeros(n))
        self.register_buffer("sharpness", torch.zeros(n))
        self.register_buffer("starvation_streak", torch.zeros(n, dtype=torch.long))
        self.register_buffer("last_dispersion", torch.zeros(n))
        self.register_buffer("age_epochs", torch.zeros(n, dtype=torch.long))
        self.register_buffer("coal_persistence", torch.zeros(n, n, dtype=torch.long))
        self.register_buffer("has_smoothed_history", torch.zeros(n, dtype=torch.bool))

        self.center_initialized = False
        self.initialized = False
        self.version = 0
        self.total_pruned = 0
        self.total_spawned = 0
        self.total_coalesced = 0
        self.total_pruned_low_mass = 0
        self.total_pruned_diffuse = 0
        self.completed_lifetimes: list[int] = []
        self.recent_spawned_slots: set[int] = set()
        self.spawned_subsequently_removed = 0

    @property
    def active(self) -> torch.Tensor:
        return self.mask

    @property
    def active_count(self) -> int:
        return int(self.mask.sum().item())

    @property
    def ready(self) -> bool:
        return self.center_initialized and self.initialized and self.active_count > 0

    @property
    def effective_ablation(self) -> str:
        return self.cfg.ablation_mode

    @property
    def _is_fixed(self) -> bool:
        return self.cfg.ablation_mode == "fixed_capacity"

    def resolve_variance_scale(self) -> float:
        """Per-dimension embedding variance scale (1.0 when manual P_0 is supplied)."""

        if self.cfg.initial_uncertainty is None:
            return float(self.variance_scale.item())
        return 1.0

    def resolve_initial_uncertainty(self) -> float:
        """Resolve P_0 (warm-up per-dimension variance when None)."""

        if self.cfg.initial_uncertainty is not None:
            return float(self.cfg.initial_uncertainty)
        return float(self.variance_scale.item())

    def resolve_uncertainty_bounds(self) -> tuple[float, float]:
        """Resolve (P_min, P_max), scaling with P_0 when None."""

        p0 = self.resolve_initial_uncertainty()
        p_min = 1e-4 * p0 if self.cfg.uncertainty_min is None else float(self.cfg.uncertainty_min)
        p_max = 10.0 * p0 if self.cfg.uncertainty_max is None else float(self.cfg.uncertainty_max)
        return p_min, p_max

    def resolve_noises(self) -> tuple[float, float, float]:
        """Resolve (process_noise Q, observation_noise R_n, structural_base_noise R_0) in variance units."""

        s = self.resolve_variance_scale()
        r0_raw = (
            float(self.cfg.observation_noise)
            if self.cfg.structural_base_noise is None
            else float(self.cfg.structural_base_noise)
        )
        return float(self.cfg.process_noise) * s, float(self.cfg.observation_noise) * s, r0_raw * s

    def resolve_alphas(self) -> tuple[float, float]:
        """Resolve (alpha_n, alpha_s), coupling to 1 - 1 / prune_patience when None."""

        auto_a = auto_smoothing_alpha(self.cfg.prune_patience)
        a_n = auto_a if self.cfg.alpha_n is None else float(self.cfg.alpha_n)
        a_s = auto_a if self.cfg.alpha_s is None else float(self.cfg.alpha_s)
        return a_n, a_s

    def resolve_lambda_sim(self) -> tuple[float, bool]:
        """Resolve (lambda_sim, scale_by_norm) for sphere-norm-invariant repulsion."""

        if self.cfg.lambda_sim is None:
            return 2e-4, True
        return float(self.cfg.lambda_sim), bool(self.cfg.auto_calibrate)

    def resolve_betas(self, active_override: int | None = None) -> tuple[float, float]:
        """Resolve (beta_read, beta_write), computing from sqrt(D) + ln(K_active) when None."""

        k = self.active_count if active_override is None else active_override
        auto_r, auto_w = auto_hopfield_betas(self.dim, k)
        b_read = auto_r if self.cfg.beta_read is None else float(self.cfg.beta_read)
        b_write = auto_w if self.cfg.beta_write is None else float(self.cfg.beta_write)
        return b_read, b_write

    def resolve_tau_sim(self) -> float:
        """Resolve online similarity threshold tau_sim (defaults to tau_coal - 0.10 when None)."""

        if self.cfg.tau_sim is not None:
            return float(self.cfg.tau_sim)
        return max(0.0, float(self.cfg.tau_coal) - 0.10)

    def resolve_structural_base_noise(self) -> float:
        """Resolve unscaled R_0 (couples to observation_noise R_n when None)."""

        if self.cfg.structural_base_noise is not None:
            return float(self.cfg.structural_base_noise)
        return float(self.cfg.observation_noise)

    def resolve_support_thresholds(self, num_samples: int) -> tuple[float, float, float]:
        """Resolve (tau_support, tau_eff, n_kish_min), scaling with M / K_active when None."""

        auto_sup, auto_eff, auto_kish = auto_support_thresholds(
            num_samples, self.active_count, self.dim
        )
        t_sup = auto_sup if self.cfg.tau_support is None else float(self.cfg.tau_support)
        t_eff = auto_eff if self.cfg.tau_eff is None else float(self.cfg.tau_eff)
        n_kish = auto_kish if self.cfg.n_kish_min is None else float(self.cfg.n_kish_min)
        return t_sup, t_eff, n_kish

    def resolve_tau_uncert(self) -> float:
        """Resolve spawning threshold tau_uncert from DARE P_inf and tau_coal when None."""

        if self.cfg.tau_uncert is not None:
            return float(self.cfg.tau_uncert)
        return auto_spawning_uncertainty_threshold(
            self.cfg.process_noise,
            self.cfg.observation_noise,
            structural_base_noise=self.resolve_structural_base_noise(),
            lambda_disp=self.cfg.lambda_disp,
            tau_coal=self.cfg.tau_coal,
            variance_scale=self.resolve_variance_scale(),
        )

    def _reset_slot_state(self, slot: int) -> None:
        self.mask[slot] = False
        self.values[slot].zero_()
        self.velocity[slot].zero_()
        self.uncertainty[slot] = self.resolve_initial_uncertainty()
        self.mass_first_order[slot] = 0.0
        self.mass_second_order[slot] = 0.0
        self.smoothed_mass_first_order[slot] = 0.0
        self.smoothed_mass_second_order[slot] = 0.0
        self.kish_ess[slot] = 0.0
        self.effective_support[slot] = 0.0
        self.sharpness[slot] = 0.0
        self.starvation_streak[slot] = 0
        self.last_dispersion[slot] = 0.0
        self.age_epochs[slot] = 0
        self.has_smoothed_history[slot] = False
        self.coal_persistence[slot, :] = 0
        self.coal_persistence[:, slot] = 0
        if slot in self.recent_spawned_slots:
            self.recent_spawned_slots.discard(slot)
            self.spawned_subsequently_removed += 1

    def _activate_slot(
        self,
        slot: int,
        key_value: torch.Tensor,
        *,
        uncertainty_override: torch.Tensor | None = None,
    ) -> None:
        self.mask[slot] = True
        self.values[slot].copy_(key_value)
        self.velocity[slot].zero_()
        if uncertainty_override is not None:
            self.uncertainty[slot].copy_(uncertainty_override)
        else:
            self.uncertainty[slot] = self.resolve_initial_uncertainty()
        self.mass_first_order[slot] = 0.0
        self.mass_second_order[slot] = 0.0
        self.smoothed_mass_first_order[slot] = 0.0
        self.smoothed_mass_second_order[slot] = 0.0
        self.kish_ess[slot] = 0.0
        self.effective_support[slot] = 0.0
        self.sharpness[slot] = 0.0
        self.starvation_streak[slot] = 0
        self.last_dispersion[slot] = 0.0
        self.age_epochs[slot] = 0
        self.has_smoothed_history[slot] = False
        self.coal_persistence[slot, :] = 0
        self.coal_persistence[:, slot] = 0

    @torch.no_grad()
    def initialize_from_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize frozen global center mu and initial K_init active prototypes."""

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings must have shape [N, {self.dim}].")
        if embeddings.shape[0] <= self.cfg.initial_memories:
            raise ValueError("Need more embeddings than initial_memories to seed k-means.")
        emb = embeddings.detach().to(self.values.device, dtype=self.values.dtype)
        center = emb.mean(dim=0)
        if not torch.isfinite(center).all():
            raise FloatingPointError("Nonfinite embedding center.")
        self.center.copy_(center)
        self.center_initialized = True

        var_per_dim = float((emb - center).pow(2).mean().item())
        if self.cfg.initial_uncertainty is None:
            self.variance_scale.fill_(max(var_per_dim, self.cfg.eps_stab))
        else:
            self.variance_scale.fill_(float(self.cfg.initial_uncertainty))
        self.uncertainty.fill_(self.resolve_initial_uncertainty())

        gen = torch.Generator(device="cpu").manual_seed(seed)
        centroids = kmeans_prototypes(emb.cpu(), self.cfg.initial_memories, generator=gen).to(
            self.values.device, dtype=self.values.dtype
        )
        for j in range(self.cfg.initial_memories):
            key = centroids[j]
            if (key - self.center).norm().item() <= self.cfg.eps_norm:
                jitter = torch.zeros_like(key)
                jitter[0] = math.sqrt(self.resolve_initial_uncertainty()) * 1e-2
                key = key + jitter
            self._activate_slot(j, key)
        self.initialized = True
        self.version += 1

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked centered-cosine Hopfield retrieval using beta_read (Sec. 5.1)."""

        if not self.ready:
            raise ValueError("Memory bank is not initialized and ready.")
        q = query.detach().to(self.values.device, dtype=self.values.dtype)
        beta_read, _ = self.resolve_betas()
        return masked_centered_retrieve(
            q,
            self.values,
            self.center,
            self.mask,
            beta=beta_read,
            num_updates=self.cfg.num_updates,
            normalization_epsilon=self.cfg.eps_norm,
        )

    @torch.no_grad()
    def compute_responsibilities(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute global write responsibilities r_ij over all active slots (Sec. 5.2)."""

        if not self.ready:
            raise ValueError("Memory bank is not initialized and ready.")
        z = embeddings.detach().to(self.values.device, dtype=self.values.dtype)
        _, beta_write = self.resolve_betas()
        return masked_global_responsibilities(
            z,
            self.values,
            self.center,
            self.mask,
            beta_write=beta_write,
            normalization_epsilon=self.cfg.eps_norm,
        )

    @torch.no_grad()
    def step_observation(
        self,
        embedding: torch.Tensor,
        *,
        apply_similarity_penalty: bool = True,
    ) -> dict[str, Any]:
        """Exact single-observation Phase 1 update (Sections 5.2–5.7)."""

        if embedding.ndim != 1 or embedding.shape[0] != self.dim:
            raise ValueError(f"embedding must have shape [{self.dim}].")
        return self.online_update(
            embedding.unsqueeze(0),
            force_sequential=True,
            apply_similarity_penalty=apply_similarity_penalty,
        )

    @torch.no_grad()
    def online_update(
        self,
        embeddings: torch.Tensor,
        *,
        force_sequential: bool = False,
        apply_similarity_penalty: bool = True,
    ) -> dict[str, Any]:
        """Phase 1: Intra-epoch global responsibility-weighted memory updates."""

        if not self.ready:
            return {"updated": False, "sim_loss": 0.0, "mean_gain": 0.0, "mean_uncertainty": 0.0}

        z = embeddings.detach().to(self.values.device, dtype=self.values.dtype)
        active = self.mask.clone()
        m_act = self.values[active]
        v_act = self.velocity[active]
        p_act = self.uncertainty[active]

        use_seq = force_sequential or self.online_update_mode == "sequential" or z.shape[0] == 1
        gains_accum = []
        r_eff_accum = []
        eff_q, eff_rn, _ = self.resolve_noises()
        eff_p_min, eff_p_max = self.resolve_uncertainty_bounds()

        if use_seq:
            full_active_mask = torch.ones(m_act.shape[0], dtype=torch.bool, device=m_act.device)
            _, beta_write_seq = self.resolve_betas(active_override=m_act.shape[0])
            for i in range(z.shape[0]):
                ri_2d, _ = masked_global_responsibilities(
                    z[i : i + 1],
                    m_act,
                    self.center,
                    full_active_mask,
                    beta_write=beta_write_seq,
                    normalization_epsilon=self.cfg.eps_norm,
                )
                ri = ri_2d[0]
                zi = z[i].unsqueeze(0).expand_as(m_act)
                out = responsibility_kalman_momentum_step(
                    m_act,
                    zi,
                    v_act,
                    p_act,
                    ri,
                    observation_noise=eff_rn,
                    gamma_mom=self.cfg.gamma_mom,
                    gamma_john=self.cfg.gamma_john,
                    process_noise=eff_q,
                    eps=self.cfg.eps_stab,
                    uncertainty_min=eff_p_min,
                    uncertainty_max=eff_p_max,
                )
                m_act, v_act, p_act = out.new_value, out.velocity, out.uncertainty
                gains_accum.append(out.gain.mean())
                r_eff_accum.append(
                    john_effective_noise(
                        ri,
                        eff_rn,
                        gamma_john=self.cfg.gamma_john,
                        eps=self.cfg.eps_stab,
                    ).mean()
                )
        else:
            r_mat, _ = self.compute_responsibilities(z)
            r_act = r_mat[:, active]
            r_sum = r_act.sum(dim=0)
            s_sum = r_act.pow(2).sum(dim=0)
            r_batch = (s_sum / (r_sum + self.cfg.eps_stab)).clamp(0.0, 1.0)
            has_mass = r_sum > self.cfg.eps_stab
            z_bar = torch.where(
                has_mass.unsqueeze(-1),
                (r_act.T @ z) / (r_sum.unsqueeze(-1) + self.cfg.eps_stab),
                m_act,
            )
            out = responsibility_kalman_momentum_step(
                m_act,
                z_bar,
                v_act,
                p_act,
                r_batch,
                observation_noise=eff_rn,
                gamma_mom=self.cfg.gamma_mom,
                gamma_john=self.cfg.gamma_john,
                process_noise=eff_q,
                eps=self.cfg.eps_stab,
                uncertainty_min=eff_p_min,
                uncertainty_max=eff_p_max,
            )
            m_act, v_act, p_act = out.new_value, out.velocity, out.uncertainty
            gains_accum.append(out.gain.mean())
            r_eff_accum.append(
                john_effective_noise(
                    r_batch,
                    eff_rn,
                    gamma_john=self.cfg.gamma_john,
                    eps=self.cfg.eps_stab,
                ).mean()
            )

        self.values[active] = m_act
        self.velocity[active] = v_act
        self.uncertainty[active] = p_act

        sim_loss = 0.0
        eff_lambda_sim, scale_sim_by_norm = self.resolve_lambda_sim()
        if apply_similarity_penalty and eff_lambda_sim > 0.0:
            updated_vals, sim_loss = online_similarity_penalty_step(
                self.values,
                self.center,
                self.mask,
                tau_sim=self.resolve_tau_sim(),
                lambda_sim=eff_lambda_sim,
                normalization_epsilon=self.cfg.eps_norm,
                scale_by_norm=scale_sim_by_norm,
            )
            self.values.copy_(updated_vals)

        self.version += 1
        return {
            "updated": True,
            "sim_loss": sim_loss,
            "mean_gain": float(torch.stack(gains_accum).mean().item()) if gains_accum else 0.0,
            "mean_effective_noise": float(torch.stack(r_eff_accum).mean().item()) if r_eff_accum else 0.0,
            "mean_uncertainty": float(self.uncertainty[active].mean().item()),
        }

    @torch.no_grad()
    def _accumulate_epoch_statistics(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Chunked Sec. 7.2 float64 accumulation of N_j, S_j, moments, and dispersion."""

        n_slots = self.max_memories
        d = self.dim
        device = self.values.device
        dtype = self.values.dtype

        n_mass = torch.zeros(n_slots, dtype=torch.float64, device=device)
        s_mass = torch.zeros(n_slots, dtype=torch.float64, device=device)
        sum_rz = torch.zeros(n_slots, d, dtype=torch.float64, device=device)
        sum_rzz = torch.zeros(n_slots, d, d, dtype=torch.float64, device=device)
        sum_rd = torch.zeros(n_slots, dtype=torch.float64, device=device)
        sum_rd2 = torch.zeros(n_slots, dtype=torch.float64, device=device)

        active = self.mask
        chunk = self.cfg.stat_chunk_size
        total = features.shape[0]

        for start in range(0, total, chunk):
            zb = features[start : start + chunk].detach().to(device=device, dtype=dtype)
            rb, sim = self.compute_responsibilities(zb)
            rb64 = rb.to(torch.float64)
            zb64 = zb.to(torch.float64)
            sim64 = sim.to(torch.float64)
            n_mass += rb64.sum(dim=0)
            s_mass += rb64.pow(2).sum(dim=0)
            sum_rz += rb64.T @ zb64
            act_idx = active.nonzero().flatten()
            if act_idx.numel() > 0:
                rb_act = rb64[:, act_idx]
                sum_rzz[act_idx] += torch.einsum("bi,bj,bk->ijk", rb_act, zb64, zb64)
            dist = (1.0 - sim64).clamp(0.0, 2.0)
            sum_rd += (rb64 * dist).sum(dim=0)
            sum_rd2 += (rb64 * dist.pow(2)).sum(dim=0)

        denom = n_mass.clamp_min(self.cfg.eps_stab)
        mu_d = sum_rd / denom
        var_d = (sum_rd2 / denom - mu_d.pow(2)).clamp_min(0.0)
        dispersion = torch.sqrt(var_d + self.cfg.eps_stab).to(dtype)
        dispersion = torch.where(
            active & (n_mass > self.cfg.eps_stab),
            dispersion,
            torch.zeros_like(dispersion),
        )

        return {
            "n_mass": n_mass.to(dtype),
            "s_mass": s_mass.to(dtype),
            "n_mass64": n_mass,
            "sum_rz64": sum_rz,
            "sum_rzz64": sum_rzz,
            "dispersion": dispersion,
        }

    @torch.no_grad()
    def adapt_structure(
        self,
        features: torch.Tensor,
        *,
        epoch: int,
        allow_spawn: bool = True,
    ) -> dict[str, Any]:
        """Phase 2: Exact 10-step epoch-end structural adaptation (Sec. 6 & 7.3)."""

        if not self.ready:
            return {
                "active_total": self.active_count,
                "pruned": 0,
                "coalesced": 0,
                "spawned": 0,
            }

        self.age_epochs[self.mask] += 1

        # Steps 1 & 2: Recompute global responsibilities & accumulate N_j, S_j, moments.
        stats = self._accumulate_epoch_statistics(features)
        n_mass = stats["n_mass"]
        s_mass = stats["s_mass"]
        dispersion = stats["dispersion"]
        active = self.mask.clone()

        self.mass_first_order.copy_(n_mass)
        self.mass_second_order.copy_(s_mass)
        self.last_dispersion.copy_(dispersion)

        # Step 3: Temporally smoothed support, Kish ESS, effective support, John Sharpness.
        eff_alpha_n, eff_alpha_s = self.resolve_alphas()
        first_seen = (~self.has_smoothed_history) & active
        self.smoothed_mass_first_order = torch.where(
            first_seen,
            n_mass,
            eff_alpha_n * self.smoothed_mass_first_order
            + (1.0 - eff_alpha_n) * n_mass,
        )
        self.smoothed_mass_second_order = torch.where(
            first_seen,
            s_mass,
            eff_alpha_s * self.smoothed_mass_second_order
            + (1.0 - eff_alpha_s) * s_mass,
        )
        self.has_smoothed_history |= active

        n_bar = self.smoothed_mass_first_order
        s_bar = self.smoothed_mass_second_order
        kish = kish_effective_sample_size(n_bar, s_bar, eps=self.cfg.eps_stab)
        sharp = john_sharpness_index(n_bar, s_bar, eps=self.cfg.eps_stab)
        self.kish_ess.copy_(torch.where(active, kish, torch.zeros_like(kish)))
        self.effective_support.copy_(torch.where(active, s_bar, torch.zeros_like(s_bar)))
        self.sharpness.copy_(torch.where(active, sharp, torch.zeros_like(sharp)))

        # Step 4: Structural dispersion and uncertainty update (skipped for fixed_capacity).
        eff_r0_unscaled = self.resolve_structural_base_noise()
        eff_p_min, eff_p_max = self.resolve_uncertainty_bounds()
        eff_var_scale = self.resolve_variance_scale()
        eff_q, eff_rn, _ = self.resolve_noises()
        if not self._is_fixed and active.any():
            updated_p, _, _ = structural_uncertainty_update(
                self.uncertainty[active],
                self.sharpness[active],
                self.last_dispersion[active],
                structural_base_noise=eff_r0_unscaled,
                lambda_disp=self.cfg.lambda_disp,
                lambda_sharp=self.cfg.lambda_sharp,
                process_noise=self.cfg.process_noise,
                variance_scale=eff_var_scale,
                uncertainty_min=eff_p_min,
                uncertainty_max=eff_p_max,
                eps=self.cfg.eps_stab,
            )
            self.uncertainty[active] = updated_p

        pruned_count = 0
        pruned_low_mass = 0
        pruned_diffuse = 0
        coalesced_count = 0
        coalesce_candidates_count = 0
        merged_surviving_slots: set[int] = set()
        spawned_count = 0
        spawn_events: list[dict[str, Any]] = []

        eff_tau_support, eff_tau_eff, _ = self.resolve_support_thresholds(features.shape[0])

        # Step 5: Starvation & diffuse-prototype pruning (Sec. 6.3).
        if not self._is_fixed:
            active_slots = self.mask.nonzero().flatten().tolist()
            starved_reasons: dict[int, str] = {}
            for slot in active_slots:
                low_mass = float(n_bar[slot].item()) < eff_tau_support
                low_eff = float(s_bar[slot].item()) < eff_tau_eff
                if low_mass or low_eff:
                    self.starvation_streak[slot] += 1
                    if low_mass and low_eff:
                        starved_reasons[slot] = "both"
                    elif low_mass:
                        starved_reasons[slot] = "low_mass"
                    else:
                        starved_reasons[slot] = "diffuse"
                else:
                    self.starvation_streak[slot] = 0

            prunable = [
                s
                for s in active_slots
                if int(self.starvation_streak[s].item()) >= self.cfg.prune_patience
            ]
            prunable.sort(key=lambda s: float(s_bar[s].item()))
            for slot in prunable:
                if self.active_count <= self.cfg.min_active_memories:
                    break
                reason = starved_reasons.get(slot, "low_mass")
                if reason in ("low_mass", "both"):
                    pruned_low_mass += 1
                if reason in ("diffuse", "both"):
                    pruned_diffuse += 1
                self.completed_lifetimes.append(int(self.age_epochs[slot].item()))
                self._reset_slot_state(slot)
                pruned_count += 1
                self.total_pruned += 1
                self.version += 1

        self.total_pruned_low_mass += pruned_low_mass
        self.total_pruned_diffuse += pruned_diffuse

        # Steps 6 & 7: Active Gram matrix & persistent redundancy coalescence (Sec. 6.4).
        _, gram = masked_gram_matrix(
            self.values, self.center, self.mask, normalization_epsilon=self.cfg.eps_norm
        )
        gram_before_stats = gram_off_diagonal_stats(gram, self.mask)

        if not self._is_fixed and self.active_count > self.cfg.min_active_memories:
            active_pair_mask = self.mask.unsqueeze(1) & self.mask.unsqueeze(0)
            upper_tri = torch.triu(
                torch.ones(self.max_memories, self.max_memories, dtype=torch.bool, device=gram.device),
                diagonal=1,
            )
            candidate_pairs = (gram > self.cfg.tau_coal) & active_pair_mask & upper_tri
            coalesce_candidates_count = int(candidate_pairs.sum().item())

            self.coal_persistence = torch.where(
                candidate_pairs,
                self.coal_persistence + 1,
                torch.zeros_like(self.coal_persistence),
            )

            while self.active_count > self.cfg.min_active_memories:
                ready_pairs = (
                    (self.coal_persistence >= self.cfg.coalescence_patience)
                    & (gram > self.cfg.tau_coal)
                    & self.mask.unsqueeze(1)
                    & self.mask.unsqueeze(0)
                    & upper_tri
                )
                if not ready_pairs.any():
                    break
                masked_scores = gram.masked_fill(~ready_pairs, float("-inf"))
                flat_best = int(masked_scores.argmax().item())
                j_slot = flat_best // self.max_memories
                k_slot = flat_best % self.max_memories

                n_j = float(self.smoothed_mass_first_order[j_slot].item())
                n_k = float(self.smoothed_mass_first_order[k_slot].item())
                denom_w = n_j + n_k
                if denom_w <= self.cfg.eps_stab:
                    omega_j = omega_k = 0.5
                else:
                    omega_j = n_j / (denom_w + self.cfg.eps_stab)
                    omega_k = n_k / (denom_w + self.cfg.eps_stab)

                m_j = self.values[j_slot].clone()
                m_k = self.values[k_slot].clone()
                m_merged = omega_j * m_j + omega_k * m_k
                v_merged = omega_j * self.velocity[j_slot] + omega_k * self.velocity[k_slot]
                p_merged = coalesce_uncertainty_total_variance(
                    self.uncertainty[j_slot],
                    self.uncertainty[k_slot],
                    m_j,
                    m_k,
                    m_merged,
                    omega_j,
                    omega_k,
                    uncertainty_min=eff_p_min,
                    uncertainty_max=eff_p_max,
                )

                self.values[j_slot].copy_(m_merged)
                self.velocity[j_slot].copy_(v_merged)
                self.uncertainty[j_slot].copy_(p_merged)
                self.smoothed_mass_first_order[j_slot] = n_j + n_k
                self.mass_first_order[j_slot] = (
                    self.mass_first_order[j_slot] + self.mass_first_order[k_slot]
                )
                self.starvation_streak[j_slot] = 0
                self.coal_persistence[j_slot, :] = 0
                self.coal_persistence[:, j_slot] = 0
                merged_surviving_slots.add(j_slot)
                self.completed_lifetimes.append(int(self.age_epochs[k_slot].item()))
                self._reset_slot_state(k_slot)
                coalesced_count += 1
                self.total_coalesced += 1
                self.version += 1

                # Recompute active Gram matrix before selecting the next pair (Sec. 6.4.4).
                _, gram = masked_gram_matrix(
                    self.values,
                    self.center,
                    self.mask,
                    normalization_epsilon=self.cfg.eps_norm,
                )

        # Step 8: Refresh responsibility statistics if any merges occurred (Sec. 6.4.4 & 7.3).
        if coalesced_count > 0:
            stats = self._accumulate_epoch_statistics(features)
            self.mass_first_order.copy_(stats["n_mass"])
            self.mass_second_order.copy_(stats["s_mass"])
            self.last_dispersion.copy_(stats["dispersion"])
            for m_slot in merged_surviving_slots:
                if self.mask[m_slot]:
                    self.smoothed_mass_second_order[m_slot] = stats["s_mass"][m_slot]
            n_bar = self.smoothed_mass_first_order
            s_bar = self.smoothed_mass_second_order
            kish = kish_effective_sample_size(n_bar, s_bar, eps=self.cfg.eps_stab)
            sharp = john_sharpness_index(n_bar, s_bar, eps=self.cfg.eps_stab)
            self.kish_ess.copy_(torch.where(self.mask, kish, torch.zeros_like(kish)))
            self.effective_support.copy_(torch.where(self.mask, s_bar, torch.zeros_like(s_bar)))
            self.sharpness.copy_(torch.where(self.mask, sharp, torch.zeros_like(sharp)))

        # Step 9 & 10: Responsibility-weighted spectral fission (Sec. 6.5).
        eff_tau_uncert = self.resolve_tau_uncert()
        _, _, eff_n_kish_min = self.resolve_support_thresholds(features.shape[0])
        eff_tau_sim = self.resolve_tau_sim()

        if not self._is_fixed and allow_spawn:
            surviving = self.mask.nonzero().flatten().tolist()
            surviving.sort(key=lambda s: float(self.uncertainty[s].item()), reverse=True)

            n_mass64 = stats["n_mass64"]
            sum_rz64 = stats["sum_rz64"]
            sum_rzz64 = stats["sum_rzz64"]

            for slot in surviving:
                free_slots = (~self.mask).nonzero().flatten()
                if free_slots.numel() == 0:
                    break
                p_val = float(self.uncertainty[slot].item())
                kish_val = float(self.kish_ess[slot].item())
                if p_val <= eff_tau_uncert or kish_val < eff_n_kish_min:
                    continue
                n_j64 = float(n_mass64[slot].item())
                if n_j64 <= self.cfg.eps_stab:
                    continue

                mean_j64 = sum_rz64[slot] / n_j64
                cov_j64 = sum_rzz64[slot] / n_j64 - torch.outer(mean_j64, mean_j64)
                mean_j = mean_j64.to(self.values.dtype)
                cov_j = cov_j64.to(self.values.dtype)

                fission = weighted_spectral_fission(
                    split_scale=self.cfg.split_scale,
                    center=self.center,
                    tau_sim=eff_tau_sim,
                    eps=self.cfg.eps_stab,
                    weighted_mean=mean_j,
                    cov_matrix=cov_j,
                )

                child_a = fission["child_a"]
                child_b = fission["child_b"]
                if (
                    (child_a - self.center).norm().item() <= self.cfg.eps_norm
                    or (child_b - self.center).norm().item() <= self.cfg.eps_norm
                ):
                    continue

                new_slot = int(free_slots[0].item())
                child_p: torch.Tensor | None = None
                if self.cfg.auto_calibrate:
                    child_p = fission_uncertainty_inverse_total_variance(
                        self.uncertainty[slot],
                        fission["delta"],
                        process_noise=eff_q,
                        observation_noise=eff_rn,
                        uncertainty_min=eff_p_min,
                        uncertainty_max=eff_p_max,
                    )

                self._activate_slot(slot, child_a, uncertainty_override=child_p)
                self._activate_slot(new_slot, child_b, uncertainty_override=child_p)
                self.recent_spawned_slots.add(new_slot)
                spawned_count += 1
                self.total_spawned += 1
                self.version += 1
                spawn_events.append(
                    {
                        "epoch": epoch,
                        "parent_slot": slot,
                        "child_slot": new_slot,
                        "uncertainty": p_val,
                        "child_uncertainty": float(self.uncertainty[new_slot].item()),
                        "lambda1": fission["lambda1"],
                        "lambda2": fission["lambda2"],
                        "anisotropy": fission["anisotropy"],
                        "kish_ess": kish_val,
                        "split_scale": fission["split_scale"],
                        "displacement_norm": fission["displacement_norm"],
                    }
                )

        # Final Gram geometry and Section 9 diagnostics.
        _, gram_after = masked_gram_matrix(
            self.values, self.center, self.mask, normalization_epsilon=self.cfg.eps_norm
        )
        gram_after_stats = gram_off_diagonal_stats(gram_after, self.mask)
        support_summary = support_diagnostics_summary(
            self.smoothed_mass_first_order,
            self.effective_support,
            self.kish_ess,
            self.sharpness,
            self.mask,
        )
        p_active = self.uncertainty[self.mask]
        res_beta_r, res_beta_w = self.resolve_betas()
        return {
            "epoch": epoch,
            "active_total": self.active_count,
            "buffer_utilization": self.active_count / self.max_memories,
            "pruned": pruned_count,
            "pruned_low_mass": pruned_low_mass,
            "pruned_diffuse": pruned_diffuse,
            "coalesced": coalesced_count,
            "coalesce_candidates": coalesce_candidates_count,
            "spawned": spawned_count,
            "spawn_events": spawn_events,
            "mean_uncertainty": float(p_active.mean().item()) if p_active.numel() else 0.0,
            "mean_dispersion": float(self.last_dispersion[self.mask].mean().item())
            if self.mask.any()
            else 0.0,
            "resolved_thresholds": {
                "beta_read": res_beta_r,
                "beta_write": res_beta_w,
                "initial_uncertainty": self.resolve_initial_uncertainty(),
                "uncertainty_min": eff_p_min,
                "uncertainty_max": eff_p_max,
                "alpha_n": eff_alpha_n,
                "alpha_s": eff_alpha_s,
                "tau_sim": eff_tau_sim,
                "tau_support": eff_tau_support,
                "tau_eff": eff_tau_eff,
                "n_kish_min": eff_n_kish_min,
                "tau_uncert": eff_tau_uncert,
                "riccati_p_inf": riccati_steady_state_uncertainty(eff_q, eff_rn),
            },
            "gram_before": gram_before_stats,
            "gram_after": gram_after_stats,
            "support_summary": support_summary,
        }


__all__ = [
    "GLOBAL_ABLATION_MODES",
    "GlobalDynamicHopfieldBank",
]
