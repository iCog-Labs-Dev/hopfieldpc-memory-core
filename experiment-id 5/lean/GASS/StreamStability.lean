import GASS.Kalman

/-! What can be guaranteed for arbitrary data streams? The first results concern
    the original single-slot write. The remaining results define a different
    commit rule for applications that require the stored state to settle. -/
namespace GASS.StreamStability
open Filter
open scoped Topology BigOperators NNReal
noncomputable section

def alternatingTarget (n : ℕ) : ℝ := (-1)^n

theorem alternating_has_no_limit (p : ℝ) :
    ¬ Tendsto alternatingTarget atTop (𝓝 p) := by
  intro h
  have even : StrictMono (fun n : ℕ => 2*n) := strictMono_nat_of_lt_succ fun n => by omega
  have odd : StrictMono (fun n : ℕ => 2*n+1) := strictMono_nat_of_lt_succ fun n => by omega
  have h₁ : Tendsto (fun _ : ℕ => (1:ℝ)) atTop (𝓝 p) := by
    simpa [Function.comp_def, alternatingTarget, pow_mul] using h.comp even.tendsto_atTop
  have h₂ : Tendsto (fun _ : ℕ => (-1:ℝ)) atTop (𝓝 p) := by
    simpa [Function.comp_def, alternatingTarget, pow_succ, pow_mul] using h.comp odd.tendsto_atTop
  have e₁ : p = 1 := tendsto_nhds_unique h₁ tendsto_const_nhds
  have e₂ : p = -1 := tendsto_nhds_unique h₂ tendsto_const_nhds
  linarith

/-- For the same query with alternating targets, no scalar response can both
    converge and have its tracking error tend to zero. -/
theorem convergence_excludes_exact_tracking (x : ℕ → ℝ) (p : ℝ)
    (hx : Tendsto x atTop (𝓝 p)) :
    ¬ Tendsto (fun n => dist (x n) (alternatingTarget n)) atTop (𝓝 0) :=
  fun he => alternating_has_no_limit p (hx.congr_dist he)

/-- An exact two-cycle of the original write. P⁻=R=N=1 gives gain 1/2.
    With process noise 1/2 the next predicted covariance is again 1. -/
theorem kalman_two_cycle :
    let x := fun n => -alternatingTarget n / 3
    GASS.posterior 1 1 1 + 1/2 = 1 ∧
    (∀ n, x (n+1) = x n + GASS.gain 1 1 1 * (alternatingTarget n - x n)) ∧
    (∀ p, ¬ Tendsto x atTop (𝓝 p)) := by
  intro x
  refine ⟨by norm_num [GASS.posterior], ?_, ?_⟩
  · intro n
    simp only [x, alternatingTarget, pow_succ, GASS.gain]
    ring
  · intro p h
    apply alternating_has_no_limit (-3*p)
    have eq : (fun n => -3 * x n) = alternatingTarget := by funext n; dsimp [x]; ring
    rw [← eq]
    exact h.const_mul (-3)

section Guard
variable {S : Type*} [MetricSpace S]

/-- B is the total distance available from initialization. Time starts at zero. -/
def allowance (B : ℝ≥0) (n : ℕ) : ℝ := (B : ℝ) / 2 / 2^n

/-- A feasible candidate passes the additional whole-state movement test.
    The caller can supply the result of its usual retrieval/objective checks. -/
def commit (C : Set S) (B : ℝ≥0) (n : ℕ) (old candidate : S) : S := by
  classical
  exact if candidate ∈ C ∧ dist old candidate ≤ allowance B n then candidate else old

/-- The proposal may depend arbitrarily on time, observations, and current state. -/
def run (C : Set S) (B : ℝ≥0) (propose : ℕ → S → S) (initial : S) : ℕ → S
  | 0 => initial
  | n+1 =>
    let old := run C B propose initial n
    commit C B n old (propose n old)

theorem step_bound (C : Set S) (B : ℝ≥0) (propose : ℕ → S → S) (initial : S) (n : ℕ) :
    dist (run C B propose initial n) (run C B propose initial (n+1)) ≤ allowance B n := by
  rw [run]
  unfold commit
  split_ifs with h
  · exact h.2
  · rw [dist_self]; unfold allowance; positivity

theorem run_feasible {C : Set S} (B : ℝ≥0) (propose : ℕ → S → S) {initial : S}
    (hi : initial ∈ C) (n : ℕ) : run C B propose initial n ∈ C := by
  induction n with
  | zero => exact hi
  | succ n ih =>
    rw [run]; unfold commit
    split_ifs with h
    · exact h.1
    · exact ih

/-- Even an adversarial continuation can move the bank by at most the unused
    budget. This bounds total path length over every finite future interval. -/
theorem future_travel (C : Set S) (B : ℝ≥0) (propose : ℕ → S → S) (initial : S) (n m : ℕ) :
    let x := run C B propose initial
    (∑ k ∈ Finset.range m, dist (x (n+k)) (x (n+k+1))) ≤
      (B:ℝ)/2^n - (B:ℝ)/2^(n+m) := by
  intro x
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Finset.sum_range_succ, ← Nat.add_assoc n m 1, pow_succ (2:ℝ) (n+m)]
    have h := step_bound C B propose initial (n+m)
    change dist (x (n+m)) (x (n+m+1)) ≤ (B:ℝ)/2/2^(n+m) at h
    linarith only [ih, h,
      (show (B:ℝ)/2/2^(n+m) + (B:ℝ)/(2^(n+m)*2) = (B:ℝ)/2^(n+m) by ring)]

/-- Input-unconditional convergence of the modified rule in a complete state
    space. The same limit is feasible whenever the feasible set is closed and
    initialization was feasible. The rate holds at every index for every stream. -/
theorem converges_for_every_stream [CompleteSpace S] (C : Set S) (B : ℝ≥0)
    (propose : ℕ → S → S) (initial : S) :
    let x := run C B propose initial
    ∃ p, Tendsto x atTop (𝓝 p) ∧ (∀ n, dist (x n) p ≤ (B:ℝ)/2^n) ∧
      (IsClosed C → initial ∈ C → p ∈ C) := by
  intro x
  have h (n : ℕ) : dist (x n) (x (n+1)) ≤ (B:ℝ)/2/2^n := step_bound C B propose initial n
  obtain ⟨p,hp⟩ := cauchySeq_tendsto_of_complete (cauchySeq_of_le_geometric_two h)
  exact ⟨p,hp,dist_le_of_le_geometric_two_of_tendsto h hp,
    fun closed hi => closed.mem_of_tendsto hp (Eventually.of_forall (run_feasible B propose hi))⟩

/-- Sensitivity to all future proposals: if two runs agree at time n, even
    entirely different continuations give limits at distance at most 2B/2^n. -/
theorem continuation_stability (C : Set S) (B : ℝ≥0) (f g : ℕ → S → S)
    (u v p q : S) (n : ℕ) (same : run C B f u n = run C B g v n)
    (hp : Tendsto (run C B f u) atTop (𝓝 p)) (hq : Tendsto (run C B g v) atTop (𝓝 q)) :
    dist p q ≤ 2*((B:ℝ)/2^n) := by
  have h₁ := dist_le_of_le_geometric_two_of_tendsto (step_bound C B f u) hp n
  have h₂ := dist_le_of_le_geometric_two_of_tendsto (step_bound C B g v) hq n
  rw [← same] at h₂
  linarith [dist_triangle p (run C B f u n) q, dist_comm p (run C B f u n)]

/-- In a fixed-capacity encoding, charge at least ε in the whole-state metric
    whenever the active-slot mask changes. All sufficiently late masks then agree. -/
theorem structure_eventually_constant {Label : Type*} (label : S → Label) {ε : ℝ}
    (hε : 0 < ε) (separated : ∀ s t, label s ≠ label t → ε ≤ dist s t)
    (C : Set S) (B : ℝ≥0) (f : ℕ → S → S) (u : S) :
    let x := run C B f u
    ∃ N, ∀ k, label (x (N+k)) = label (x N) := by
  intro x
  have small : ∀ᶠ n in atTop, allowance B n < ε :=
    (summable_geometric_two' (B:ℝ)).tendsto_atTop_zero.eventually_lt_const hε
  obtain ⟨N,hN⟩ := eventually_atTop.1 small
  refine ⟨N,fun k => ?_⟩
  induction k with
  | zero => simp
  | succ k ih =>
    have smallStep := (step_bound C B f u (N+k)).trans_lt (hN (N+k) (Nat.le_add_right ..))
    have eq : label (x (N+k)) = label (x (N+k+1)) :=
      Classical.byContradiction fun h => (not_le_of_gt smallStep) (separated _ _ h)
    exact eq.symm.trans ih

end Guard
end
end GASS.StreamStability
