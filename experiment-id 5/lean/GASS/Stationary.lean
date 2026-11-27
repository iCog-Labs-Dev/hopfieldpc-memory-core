import GASS.Descent
import Mathlib.Topology.Algebra.InfiniteSum.Real

namespace GASS
open Filter
open scoped Topology InnerProductSpace
noncomputable section
variable {E : Type*} [NormedAddCommGroup E]

/-- A block version of the criticality argument, directly from minimization of a
    touching model with bounded quadratic remainder. At a cluster point p,
    -grad p is a proximal normal of C (the displayed supporting inequality).
    Apply to each block of the product domain. -/
theorem critical_block_of_sweeps {S : Type*} [PseudoMetricSpace S] [InnerProductSpace ℝ E]
    (coord grad : S → E) (x pre : ℕ → S) (d : ℕ → ℝ) (C : Set E) (M : ℝ)
    (hc : IsClosed C) (feas : ∀ n, coord (x n) ∈ C) (cs : Continuous coord)
    (cg : Continuous grad) (small : Tendsto d atTop (𝓝 0))
    (near : ∀ n, dist (x n) (pre n) ≤ d n)
    (model : ∀ n z, z ∈ C →
      ⟪grad (pre n),coord (x n)-coord (pre n)⟫_ℝ ≤
        ⟪grad (pre n),z-coord (pre n)⟫_ℝ + M*‖z-coord (pre n)‖^2)
    (s : ℕ → ℕ) (p : S) (hs : StrictMono s) (hx : Tendsto (x ∘ s) atTop (𝓝 p)) :
    coord p ∈ C ∧ ∀ z ∈ C, ⟪-grad p,z-coord p⟫_ℝ ≤ M*‖z-coord p‖^2 := by
  have hpre := hx.congr_dist ((squeeze_zero (fun _ => dist_nonneg) near small).comp hs.tendsto_atTop)
  have hc₁ := (cs.tendsto p).comp hx
  have hc₂ := (cs.tendsto p).comp hpre
  have hg := (cg.tendsto p).comp hpre
  refine ⟨hc.mem_of_tendsto hc₁ (Eventually.of_forall fun n => feas (s n)), ?_⟩
  intro z hz
  have hz' := (tendsto_const_nhds.sub hc₂ : Tendsto (fun n => z-coord (pre (s n))) atTop _)
  have h := le_of_tendsto_of_tendsto (hg.inner (hc₁.sub hc₂))
    ((hg.inner hz').add ((hz'.norm.pow 2).const_mul M))
    (Eventually.of_forall fun n => model (s n) z hz)
  simp only [sub_self, inner_zero_right] at h
  rw [inner_neg_left]; linarith

/-- Theorem 15 after the sweep estimates have been established. G is the graph
    of the constrained subdifferential on the feasible domain. The graph and
    the endpoint residual certificate are explicit inputs. -/
theorem stationary_endpoints (x g : ℕ → E) (F d : ℕ → ℝ) (G : Set (E × E))
    {a b floor : ℝ} (ha : 0 < a) (hd : ∀ n, 0 ≤ d n)
    (decrease : ∀ n, a*(d n)^2 ≤ F n-F (n+1)) (lower : ∀ n, floor ≤ F n)
    (closed : IsClosed G) (cert : ∀ n, (x n,g n) ∈ G)
    (relative : ∀ n, ‖g n‖ ≤ b*d n) :
    Tendsto F atTop (𝓝 (⨅ n, F n)) ∧ Summable (fun n => (d n)^2) ∧
    ∀ (s : ℕ → ℕ) (p : E), StrictMono s → Tendsto (x ∘ s) atTop (𝓝 p) → (p,0) ∈ G := by
  have hsum := summable_of_sum_range_le (fun n => sq_nonneg (d n))
    (squared_steps_bound F d ha decrease lower)
  have hzero : Tendsto d atTop (𝓝 0) := by
    simpa only [Real.sqrt_sq (hd _), Real.sqrt_zero, Function.comp_def] using
      (Real.continuous_sqrt.tendsto (0:ℝ)).comp hsum.tendsto_atTop_zero
  have hg : Tendsto g atTop (𝓝 0) := by
    apply tendsto_zero_iff_norm_tendsto_zero.mpr
    exact squeeze_zero (fun n => norm_nonneg (g n)) relative (by simpa using hzero.const_mul b)
  have mono : Antitone F := antitone_nat_of_succ_le fun n =>
    by linarith [decrease n, mul_nonneg ha.le (sq_nonneg (d n))]
  refine ⟨tendsto_atTop_ciInf mono ⟨floor, Set.forall_mem_range.mpr lower⟩, hsum, ?_⟩
  intro s p hs hx
  exact closed.mem_of_tendsto (hx.prodMk_nhds (hg.comp hs.tendsto_atTop))
    (Eventually.of_forall fun n => cert (s n))

/-- Theorem 16 for a tail on which the uniform KL inequality holds. `φ n` is
    the desingularized objective gap, `slope n` its scalar derivative, and
    `res n` the stationarity residual. A zero gap forces the next step to vanish.
    The index convention takes d n to be the length of the sweep ending at n. -/
theorem kl_endpoint_limit [CompleteSpace E] (x : ℕ → E) (gap d φ slope res : ℕ → ℝ)
    {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hd : ∀ n, 0 ≤ d n)
    (hgap : ∀ n, 0 ≤ gap n) (hφ : ∀ n, 0 ≤ φ n) (hslope : ∀ n, 0 ≤ slope n)
    (step : ∀ n, dist (x n) (x (n+1)) ≤ d (n+1))
    (decrease : ∀ n, a*(d (n+1))^2 ≤ gap n-gap (n+1))
    (relative : ∀ n, res n ≤ b*d n)
    (tangent : ∀ n, slope n*(gap n-gap (n+1)) ≤ φ n-φ (n+1))
    (kl : ∀ n, 0 < gap n → 1 ≤ slope n*res n) :
    Summable d ∧ ∃ p, Tendsto x atTop (𝓝 p) := by
  let ψ := fun n => b/a*φ n
  have scale : 0 < b/a := div_pos hb ha
  have drop (n : ℕ) : 0 ≤ gap n-gap (n+1) :=
    (mul_nonneg ha.le (sq_nonneg _)).trans (decrease n)
  have mono (n : ℕ) : φ (n+1) ≤ φ n := by
    linarith [tangent n, mul_nonneg (hslope n) (drop n)]
  have chain (n : ℕ) : (d (n+1))^2 ≤ d n*(ψ n-ψ (n+1)) := by
    by_cases hz : gap n = 0
    · have h := decrease n
      rw [hz] at h
      have hd0 : d (n+1) = 0 := by
        by_contra hn
        linarith [hgap (n+1), mul_pos ha (sq_pos_of_ne_zero hn)]
      rw [hd0, zero_pow (by decide : 2 ≠ 0)]
      exact mul_nonneg (hd n) (sub_nonneg.mpr (mul_le_mul_of_nonneg_left (mono n) scale.le))
    · have one : 1 ≤ slope n*(b*d n) :=
        (kl n (lt_of_le_of_ne (hgap n) (Ne.symm hz))).trans
          (mul_le_mul_of_nonneg_left (relative n) (hslope n))
      have h1 := mul_le_mul_of_nonneg_right one (drop n)
      have h2 := mul_le_mul_of_nonneg_left (tangent n) (mul_nonneg hb.le (hd n))
      have he : a*(d (n+1))^2 ≤ b*d n*(φ n-φ (n+1)) := by nlinarith [decrease n]
      calc
        (d (n+1))^2 ≤ (b*d n*(φ n-φ (n+1)))/a := (le_div_iff₀ ha).2 (by linarith)
        _ = _ := by dsimp [ψ]; ring
  have hsum := summable_of_sum_range_le (fun n => hd (n+1))
    (kl_length_bound d ψ hd (fun n => mul_nonneg scale.le (hφ n))
      (fun n => mul_le_mul_of_nonneg_left (mono n) scale.le) chain)
  exact ⟨(summable_nat_add_iff 1).mp hsum,
    cauchySeq_tendsto_of_complete (cauchySeq_of_dist_le_of_summable _ step hsum)⟩

end
end GASS
