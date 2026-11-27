import GASS.Common

namespace GASS
open Finset Filter
open scoped BigOperators Topology
noncomputable section

/-- Eq. 66. Each index here counts an accepted edit on the same window. -/
theorem fixed_edit_bound {J : ℕ → ℝ} {δ floor : ℝ} (hδ : 0 < δ) (s : ℕ)
    (step : ∀ t < s, J (t+1) ≤ J t - δ) (lower : floor ≤ J s) :
    J s ≤ J 0 - s*δ ∧ (s : ℝ) ≤ (J 0-floor)/δ := by
  have h (t : ℕ) (ht : t ≤ s) : J t ≤ J 0 - t*δ := by
    induction t with
    | zero => simp
    | succ t ih => push_cast; linarith [step t (by omega), ih (by omega)]
  exact ⟨h s le_rfl, (le_div_iff₀ hδ).2 (by linarith [h s le_rfl])⟩

theorem no_edit_cycle {S : Type*} {s : ℕ → S} {J : S → ℝ} {δ : ℝ}
    (hδ : 0 < δ) {i k : ℕ} (hk : 0 < k)
    (step : ∀ t < k, J (s (i+t+1)) ≤ J (s (i+t)) - δ) : s i ≠ s (i+k) := by
  have h := (fixed_edit_bound (J := fun t => J (s (i+t))) hδ k step le_rfl).1
  simp only [Nat.add_zero] at h
  have hpos := mul_pos (Nat.cast_pos.mpr hk : 0 < (k : ℝ)) hδ
  intro e
  rw [← e] at h
  linarith

/-- A finite list of transactions within a time step can contain zero-edit continuous steps. -/
theorem batch_descent {S : Type*} (J : S → ℝ) (state : ℕ → S) (edits : ℕ → ℕ) (δ : ℝ)
    (step : ∀ i, J (state (i+1)) ≤ J (state i) - δ * edits i) (n : ℕ) :
    J (state n) + δ * (∑ i ∈ range n, edits i : ℕ) ≤ J (state 0) := by
  induction n with
  | zero => simp
  | succ n ih => rw [sum_range_succ, Nat.cast_add, mul_add]; linarith [step n]

/-- Theorem 14, Eq. 67. For `T=n+1` windows there are `n` changes of objective.
    `ω` can be any uniform upper bound on the upward drift on the feasible state space. -/
theorem drift_edit_bound {S : Type*} (J : ℕ → S → ℝ) (s : ℕ → S) (edits : ℕ → ℕ)
    (ω : ℕ → ℝ) {δ floor : ℝ}
    (step : ∀ t, J t (s (t+1)) ≤ J t (s t) - δ * edits t)
    (drift : ∀ t x, J (t+1) x ≤ J t x + ω t)
    (lower : ∀ t x, floor ≤ J t x) (n : ℕ) :
    δ * (∑ t ∈ range (n+1), edits t : ℕ) ≤ J 0 (s 0) - floor + ∑ t ∈ range n, ω t := by
  have h (k : ℕ) : J k (s (k+1)) + δ * (∑ t ∈ range (k+1), edits t : ℕ) ≤
      J 0 (s 0) + ∑ t ∈ range k, ω t := by
    induction k with
    | zero => simpa using (show J 0 (s 1) + δ * edits 0 ≤ J 0 (s 0) by linarith [step 0])
    | succ k ih =>
      rw [sum_range_succ (n := k+1), sum_range_succ (n := k) (f := ω)]
      rw [Nat.cast_add, mul_add]
      linarith [step (k+1), drift k (s (k+1))]
  linarith [h n, lower n (s (n+1))]

private theorem bounded_count_stops (f : ℕ → ℕ) (mono : Monotone f)
    (bound : ∃ b, ∀ n, f n ≤ b) : ∃ t, ∀ n ≥ t, f n = f t := by
  classical
  let b := Nat.find bound
  have hb : ∀ n, f n ≤ b := Nat.find_spec bound
  have hit : ∃ t, f t = b := by
    by_contra! miss
    have hlt (n : ℕ) : f n < b := lt_of_le_of_ne (hb n) (miss n)
    have smaller : ∀ n, f n ≤ b-1 := fun n => by have := hlt n; omega
    have hmin := Nat.find_min' bound smaller
    have hpos := hlt 0
    change b ≤ b-1 at hmin
    omega
  obtain ⟨t, ht⟩ := hit
  exact ⟨t, fun n hn => le_antisymm ((hb n).trans_eq ht.symm) (mono hn)⟩

/-- Summable external drift implies that all sufficiently late windows have zero edits. -/
theorem finite_churn {S : Type*} (J : ℕ → S → ℝ) (s : ℕ → S) (edits : ℕ → ℕ)
    (ω : ℕ → ℝ) {δ floor : ℝ} (hδ : 0 < δ)
    (step : ∀ t, J t (s (t+1)) ≤ J t (s t) - δ * edits t)
    (drift : ∀ t x, J (t+1) x ≤ J t x + ω t) (lower : ∀ t x, floor ≤ J t x)
    (hw : ∀ t, 0 ≤ ω t) (hs : Summable ω) : ∃ T, ∀ t ≥ T, edits t = 0 := by
  let count (n : ℕ) := ∑ t ∈ range n, edits t
  have hcount (n : ℕ) : count (n+1) = count n + edits n := sum_range_succ ..
  have mono : Monotone count := monotone_nat_of_le_succ fun n => by rw [hcount]; omega
  obtain ⟨B, hB⟩ := exists_nat_ge ((J 0 (s 0)-floor+∑' t, ω t)/δ)
  have bound : ∀ n, count n ≤ B := by
    intro n
    cases n with
    | zero => simp [count]
    | succ n =>
      have h := drift_edit_bound J s edits ω step drift lower n
      have hsum := hs.sum_le_tsum (range n) (fun i _ => hw i)
      have hc : (count (n+1) : ℝ) ≤ (J 0 (s 0)-floor+∑' t, ω t)/δ := by
        apply (le_div_iff₀ hδ).2
        dsimp only [count]; linarith
      exact_mod_cast hc.trans hB
  obtain ⟨T, ht⟩ := bounded_count_stops count mono ⟨B,bound⟩
  exact ⟨T, fun t h => by have := ht t h; have := ht (t+1) (by omega); have := hcount t; omega⟩

/-- Eq. 47 for any block whose quadratic model is bounded above by its value at zero. -/
theorem certified_block {old new linear L γ μ d₂ : ℝ} (hd : 0 ≤ d₂) (hγ : L+μ ≤ γ)
    (smooth : new-old ≤ linear + L/2*d₂) (minimizes : linear + γ/2*d₂ ≤ 0) :
    μ/2*d₂ ≤ old-new := by nlinarith [mul_nonneg (sub_nonneg.mpr hγ) hd]

/-- The sum of the decrease inequalities in Theorem 15; uniform in the horizon. -/
theorem squared_steps_bound (F d : ℕ → ℝ) {a floor : ℝ} (ha : 0 < a)
    (decrease : ∀ n, a*(d n)^2 ≤ F n - F (n+1)) (lower : ∀ n, floor ≤ F n) (n : ℕ) :
    (∑ t ∈ range n, (d t)^2) ≤ (F 0-floor)/a := by
  have h (k : ℕ) : a*(∑ t ∈ range k, (d t)^2) ≤ F 0-F k := by
    induction k with
    | zero => simp
    | succ k ih => rw [sum_range_succ, mul_add]; linarith [decrease k]
  exact (le_div_iff₀ ha).2 (by linarith [h n, lower n])

/-- Theorem 16's finite-length argument once the uniformized KL inequality yields
    `d[n+1]^2 ≤ d[n] * (φ[n]-φ[n+1])`, with the constant absorbed into `φ`. -/
theorem kl_length_bound (d φ : ℕ → ℝ) (hd : ∀ n, 0 ≤ d n) (hp : ∀ n, 0 ≤ φ n)
    (hm : ∀ n, φ (n+1) ≤ φ n) (kl : ∀ n, (d (n+1))^2 ≤ d n * (φ n-φ (n+1))) (n : ℕ) :
    (∑ t ∈ range n, d (t+1)) ≤ d 0+φ 0 := by
  have hstep (i : ℕ) : 2*d (i+1) ≤ d i + (φ i-φ (i+1)) := by
    apply (sq_le_sq₀ (by linarith [hd (i+1)]) (by linarith [hd i, hm i])).1
    nlinarith [kl i, sq_nonneg (d i - (φ i-φ (i+1)))]
  have h (k : ℕ) : (∑ t ∈ range k, d (t+1)) + d k + φ k ≤ d 0+φ 0 := by
    induction k with
    | zero => simp
    | succ k ih => rw [sum_range_succ]; linarith [hstep k]
  linarith [h n, hd n, hp n]

end
end GASS
