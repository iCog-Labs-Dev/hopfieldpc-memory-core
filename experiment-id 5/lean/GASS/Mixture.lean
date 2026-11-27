import GASS.Common

namespace GASS
open scoped BigOperators
noncomputable section
variable {ι : Type*} [Fintype ι] [Nonempty ι]

def partition (a : ι → ℝ) := ∑ j, a j
def weight (a : ι → ℝ) (j : ι) := a j / partition a

theorem partition_pos {a : ι → ℝ} (ha : ∀ j, 0 < a j) : 0 < partition a :=
  Finset.sum_pos (fun i _ => ha i) Finset.univ_nonempty

theorem weights_sum {a : ι → ℝ} (ha : ∀ j, 0 < a j) : ∑ j, weight a j = 1 := by
  simp only [weight, ← Finset.sum_div]
  exact div_self (partition_pos ha).ne'

/-- Eq. 19 in difference form. Tilting the current positive masses by `exp d`
    gives the exact candidate partition for any change in log-score. -/
theorem mixture_change {a : ι → ℝ} (ha : ∀ j, 0 < a j) (d : ι → ℝ) :
    -Real.log (partition fun j => a j * Real.exp (d j)) + Real.log (partition a) ≤
      -(∑ j, weight a j * d j) := by
  have hZ := partition_pos ha
  have hZ' := partition_pos (fun j => mul_pos (ha j) (Real.exp_pos (d j)))
  have hj := convexOn_exp.map_sum_le (t := Finset.univ) (p := d)
    (fun j _ => (div_nonneg (ha j).le hZ.le : 0 ≤ weight a j)) (weights_sum ha)
    (fun _ _ => Set.mem_univ _)
  simp only [smul_eq_mul] at hj
  change Real.exp (∑ j, weight a j * d j) ≤ ∑ j, weight a j * Real.exp (d j) at hj
  have eq : (∑ j, weight a j * Real.exp (d j)) =
      partition (fun j => a j * Real.exp (d j)) / partition a := by
    unfold weight partition
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun _ _ => by ring
  rw [eq] at hj
  have := Real.log_le_log (Real.exp_pos _) hj
  rw [Real.log_exp, Real.log_div hZ'.ne' hZ.ne'] at this
  linarith

/-- Proposition 9, with any finite set of slots and any real normed value space. -/
theorem read_mem {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {C : Set E}
    (hc : Convex ℝ C) {a : ι → ℝ} (ha : ∀ i, 0 < a i) (m : ι → E) (hm : ∀ i, m i ∈ C) :
    (∑ j, weight a j • m j) ∈ C := by
  exact hc.sum_mem (fun j _ => div_nonneg (ha j).le (partition_pos ha).le)
    (weights_sum ha) (fun j _ => hm j)

theorem feasible_run {S : Type*} (C : S → Prop) (s : ℕ → S) (init : C (s 0))
    (commit : ∀ n, C (s n) → C (s (n+1))) : ∀ n, C (s n) :=
  fun n => Nat.recOn n init commit

/-- Proposition 10. Removed mass `a`, survivor mass `s`, survivor numerator `v`.
    Applied coordinatewise, the first two equalities give the vector read identities. -/
theorem prune_read {a s : ℝ} (m v : ℝ) (ha : 0 < a) (hs : 0 < s) :
    let w := a/(a+s)
    let h := (a*m+v)/(a+s)
    v/s = (h-w*m)/(1-w) ∧ v/s-h = w*(v/s-m) := by
  intro w h
  have hZ : 0 < a+s := by positivity
  have hw : 0 < 1-w := sub_pos.mpr ((div_lt_one hZ).2 (by linarith))
  constructor
  · apply (eq_div_iff hw.ne').2
    dsimp [h,w]; field_simp
  · dsimp [h,w]; field_simp; ring

theorem prune_norm {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {h h' m : E} {B w : ℝ} (hw : 0 ≤ w) (hh : ‖h'‖ ≤ B) (hm : ‖m‖ ≤ B)
    (eq : h' - h = w • (h' - m)) : ‖h' - h‖ ≤ 2*B*w := by
  rw [eq, norm_smul, Real.norm_eq_abs, abs_of_nonneg hw]
  nlinarith [mul_le_mul_of_nonneg_left ((norm_sub_le h' m).trans (add_le_add hh hm)) hw]

/-- Eq. 57. Renormalizing priors multiplies every surviving write term by `1/(1-π)`. -/
theorem prune_log {a s π : ℝ} (ha : 0 < a) (hs : 0 < s) (hπ : π < 1) :
    -Real.log (s/(1-π)) + Real.log (a+s) =
      Real.log (1-π) - Real.log (1-a/(a+s)) := by
  have hz : 0 < a+s := by positivity
  have hr : 1-a/(a+s) = s/(a+s) := by field_simp
  rw [hr, Real.log_div hs.ne' (sub_pos.mpr hπ).ne', Real.log_div hs.ne' hz.ne']
  ring

/-- Theorem 11. The two slot-local exponential factors agree for equal keys.
    All other slots are represented by their unchanged numerator and denominator. -/
theorem merge_read {p q : ℝ} (hp : 0 < p) (hq : 0 < q) (e m n v z : ℝ) :
    (v + p*e*m + q*e*n) / (z+p*e+q*e) =
      (v + (p+q)*e*((p*m+q*n)/(p+q))) / (z+(p+q)*e) := by
  have h : p+q ≠ 0 := ne_of_gt (add_pos hp hq)
  congr 1
  · field_simp; ring
  · ring

theorem duplicate_cost (z p q e loss cost : ℝ) (k : ℕ) :
    -Real.log (z+(p+q)*e) + loss + cost*k =
      (-Real.log (z+p*e+q*e) + loss + cost*(k+1)) - cost := by
  rw [show z+(p+q)*e = z+p*e+q*e by ring]
  ring

/-- Eq. 10, the per-observation lower bound before adding the normalizer and slot cost. -/
theorem mixture_floor {π score : ι → ℝ} {U : ℝ} (hp : ∀ j, 0 < π j) (hs : ∑ j, π j = 1)
    (hU : ∀ j, score j ≤ U) : -U ≤ -Real.log (partition fun j => π j * Real.exp (score j)) := by
  have h := Finset.sum_le_sum (s := Finset.univ) fun j _ =>
    mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (hU j)) (hp j).le
  rw [← Finset.sum_mul, hs, one_mul] at h
  have := Real.log_le_log (partition_pos fun j => mul_pos (hp j) (Real.exp_pos _)) h
  simpa using neg_le_neg this

/-- Eq. 10 on the full finite window. `c = χD/2`, and the slot score includes
    the nonpositive Gaussian residual. This supplies a common floor for Eq. 67. -/
theorem objective_floor {κ : Type*} [Fintype κ] [Nonempty κ] (π : ι → ℝ) (score : κ → ι → ℝ)
    (loss : κ → ℝ) {R Rmin ref c η cost U : ℝ} {Kmin K : ℕ}
    (hp : ∀ j, 0 < π j) (hs : ∑ j, π j = 1) (hU : ∀ i j, score i j ≤ U)
    (hl : ∀ i, 0 ≤ loss i) (hm : 0 < Rmin) (hr : Rmin ≤ R) (href : 0 < ref)
    (hc : 0 ≤ c) (hη : 0 ≤ η) (hcost : 0 ≤ cost) (hk : Kmin ≤ K) :
    let L : ℝ := Fintype.card κ
    c*Real.log (Rmin/ref)-U+cost*Kmin ≤ c*Real.log (R/ref) +
      (∑ i, (-Real.log (partition fun j => π j*Real.exp (score i j)) + η*loss i))/L + cost*K := by
  intro L
  have hL : 0 < L := Nat.cast_pos.mpr Fintype.card_pos
  have hrow (i : κ) : -U ≤ -Real.log (partition fun j => π j*Real.exp (score i j)) + η*loss i :=
    (mixture_floor hp hs (hU i)).trans (le_add_of_nonneg_right (mul_nonneg hη (hl i)))
  have hsum := Finset.sum_le_sum (s := Finset.univ) fun i _ => hrow i
  simp only [Finset.sum_const,Finset.card_univ,nsmul_eq_mul] at hsum
  have hav := (le_div_iff₀ hL).2 ((mul_comm (-U) L).le.trans hsum)
  have hlog := Real.log_le_log (div_pos hm href) (div_le_div_of_nonneg_right hr href.le)
  linarith [mul_le_mul_of_nonneg_left hlog hc, mul_le_mul_of_nonneg_left (Nat.cast_le.mpr hk : (Kmin:ℝ) ≤ K) hcost]

end
end GASS
