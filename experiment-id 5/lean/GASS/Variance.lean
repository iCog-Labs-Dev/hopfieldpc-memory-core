import GASS.Mixture
import Mathlib.Analysis.Convex.Deriv

namespace GASS
open scoped BigOperators
noncomputable section

/-- Taylor's lower bound on the entire feasible log-variance chord. -/
private theorem exp_quadratic {t H : ℝ} (hH : 0 ≤ H) (ht : -H ≤ t) :
    Real.exp (-H) * t^2 / 2 ≤ Real.exp t - 1 - t := by
  let c := Real.exp (-H) * t^2
  let f := fun u : ℝ => Real.exp (t*u) - t*u - c/2 * u^2
  let g := fun u : ℝ => t * Real.exp (t*u) - t - c*u
  have he (u : ℝ) : HasDerivAt (fun z : ℝ => Real.exp (t*z)) (Real.exp (t*u)*t) u := by
    simpa using (((hasDerivAt_id u).const_mul t).exp)
  have hf (u : ℝ) : HasDerivAt f (g u) u := by
    convert ((he u).sub ((hasDerivAt_id u).const_mul t)).sub
      (((hasDerivAt_pow 2 u).const_mul (c/2))) using 1
    simp [f,g]; ring
  have hg (u : ℝ) : HasDerivAt g (t^2 * Real.exp (t*u) - c) u := by
    convert (((he u).const_mul t).sub_const t).sub ((hasDerivAt_id u).const_mul c) using 1
    simp [g]; ring
  have hc : ConvexOn ℝ (Set.Icc (0:ℝ) 1) f := by
    apply convexOn_of_hasDerivWithinAt2_nonneg (convex_Icc ..)
      (fun u _ => (hf u).continuousAt.continuousWithinAt)
      (fun u _ => (hf u).hasDerivWithinAt) (fun u _ => (hg u).hasDerivWithinAt)
    intro u hu
    have hu' := interior_subset hu
    have hx : -H ≤ t*u := by
      nlinarith [mul_nonneg (by linarith : 0 ≤ t+H) hu'.1, mul_nonneg hH (sub_nonneg.mpr hu'.2)]
    dsimp [c]
    nlinarith [mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr hx) (sq_nonneg t)]
  have h := hc.le_slope_of_hasDerivAt (x := 0) (y := 1) (by simp) (by simp) zero_lt_one (hf 0)
  simp [slope_def_field, f, g, c] at h
  linarith

def clip (lo hi y : ℝ) := max lo (min y hi)
def varianceModel (D y r : ℝ) := D/2 * (Real.log r + y/r)

/-- Theorem 4, Eqs. 20–21. Variances are measured in units of Rref,
    so `y = C/(L D Rref)`. This also covers C=0 and lo=hi. -/
theorem variance_gap {D y lo hi r : ℝ} (hD : 0 < D) (hl : 0 < lo)
    (hr : lo ≤ r) (hu : r ≤ hi) :
    let p := clip lo hi y
    D/4 * Real.exp (-(Real.log hi-Real.log lo)) * (Real.log r-Real.log p)^2 ≤
      varianceModel D y r - varianceModel D y p := by
  intro p
  have hlh := hr.trans hu
  have hlp : lo ≤ p := le_max_left ..
  have hph : p ≤ hi := max_le hlh (min_le_right ..)
  have hp := hl.trans_le hlp
  have hR := hl.trans_le hr
  let H := Real.log hi - Real.log lo
  let t := Real.log p - Real.log r
  have hH : 0 ≤ H := sub_nonneg.mpr (Real.log_le_log hl hlh)
  have ht : -H ≤ t := by dsimp [H,t]; linarith [Real.log_le_log hl hlp, Real.log_le_log hR hu]
  have he : Real.exp t = p/r := by dsimp [t]; rw [Real.exp_sub, Real.exp_log hp, Real.exp_log hR]
  have hq := exp_quadratic hH ht
  rw [he] at hq
  have corr : 0 ≤ (y/p-1)*(p/r-1) := by
    by_cases low : y ≤ lo
    · have heq : p = lo := max_eq_left ((min_le_left ..).trans low)
      rw [heq]
      exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr ((div_le_one hl).2 low))
        (sub_nonpos.mpr ((div_le_one hR).2 hr))
    by_cases high : hi ≤ y
    · have heq : p = hi := by dsimp [p,clip]; rw [min_eq_right high, max_eq_right hlh]
      rw [heq]
      exact mul_nonneg (sub_nonneg.mpr ((one_le_div (hl.trans_le hlh)).2 high))
        (sub_nonneg.mpr ((one_le_div hR).2 hu))
    · have heq : p = y := by dsimp [p,clip]; rw [min_eq_left (le_of_not_ge high),
            max_eq_right (le_of_not_ge low)]
      rw [← heq, div_self hp.ne']; simp
  have hid : y/r-y/p = (p/r-1) + (y/p-1)*(p/r-1) := by field_simp; ring
  have h := mul_le_mul_of_nonneg_left hq hD.le
  dsimp [varianceModel,t,H] at *
  nlinarith [mul_nonneg hD.le corr, congrArg (fun z : ℝ => D*z) hid]

theorem variance_unique {D y lo hi r : ℝ} (hD : 0 < D) (hl : 0 < lo) (hr : lo ≤ r)
    (hu : r ≤ hi) (hne : r ≠ clip lo hi y) :
    varianceModel D y (clip lo hi y) < varianceModel D y r := by
  have hp : 0 < clip lo hi y := hl.trans_le (le_max_left ..)
  have hn : Real.log r ≠ Real.log (clip lo hi y) := fun h =>
    hne (Real.log_injOn_pos (hl.trans_le hr) hp h)
  have h := variance_gap (y := y) hD hl hr hu
  have := mul_pos (mul_pos (by positivity : 0 < D/4) (Real.exp_pos (-(Real.log hi-Real.log lo))))
    (sq_pos_of_ne_zero (sub_ne_zero.mpr hn))
  linarith

/-- Eq. 19 specialized to a Gaussian variance change on one observation.
    `res` is the squared residual divided by 2Rref. -/
theorem variance_row {ι : Type*} [Fintype ι] [Nonempty ι] (base res : ι → ℝ)
    (hb : ∀ j, 0 < base j) (r p : ℝ) :
    let mass := fun s j => base j * Real.exp (-res j/s);
    -Real.log (partition (mass p)) + Real.log (partition (mass r)) ≤
      (∑ j, weight (mass r) j * res j) * (1/p-1/r) := by
  intro mass
  let d := fun j => -res j/p - (-res j/r)
  have h := mixture_change (a := mass r) (fun j => mul_pos (hb j) (Real.exp_pos _)) d
  have heq : (fun j => mass r j * Real.exp (d j)) = mass p := by
    funext j
    dsimp [mass,d]
    rw [mul_assoc, ← Real.exp_add]; congr 2; ring
  rw [heq] at h
  refine h.trans_eq ?_
  rw [Finset.sum_mul, ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun _ _ => by dsimp [d]; ring

/-- Eq. 21 for the actual finite-window log-mixture objective, including the
    Gaussian normalizer. All terms independent of variance are collected in `fixed`. -/
theorem variance_objective {ι κ : Type*} [Fintype ι] [Nonempty ι] [Fintype κ] [Nonempty κ]
    (base res : κ → ι → ℝ) (hb : ∀ i j, 0 < base i j) (fixed : ℝ)
    {D lo hi r : ℝ} (hD : 0 < D) (hl : 0 < lo) (hr : lo ≤ r) (hu : r ≤ hi) :
    let L : ℝ := Fintype.card κ
    let mass := fun s i j => base i j * Real.exp (-res i j/s)
    let y := 2 * (∑ i, ∑ j, weight (mass r i) j * res i j) / (L*D)
    let p := clip lo hi y
    let J := fun s => D/2*Real.log s - (∑ i, Real.log (partition (mass s i))) / L + fixed
    D/4 * Real.exp (-(Real.log hi-Real.log lo)) * (Real.log r-Real.log p)^2 ≤ J r-J p := by
  intro L mass y p J
  have hL : 0 < L := Nat.cast_pos.mpr Fintype.card_pos
  have h := Finset.sum_le_sum (s := Finset.univ) fun i _ => variance_row (base i) (res i) (hb i) r p
  change (∑ i, (-Real.log (partition (mass p i)) + Real.log (partition (mass r i)))) ≤
    ∑ i, (∑ j, weight (mass r i) j * res i j) * (1/p-1/r) at h
  rw [Finset.sum_add_distrib, Finset.sum_neg_distrib, ← Finset.sum_mul] at h
  refine (variance_gap (y := y) hD hl hr hu).trans ?_
  dsimp only [varianceModel,J]
  have hy : y*(L*D) = 2 * (∑ i, ∑ j, weight (mass r i) j * res i j) :=
    div_mul_cancel₀ _ (mul_pos hL hD).ne'
  apply (le_of_mul_le_mul_left _ hL : _)
  simp only [div_eq_mul_inv] at *
  nlinarith [mul_inv_cancel₀ hL.ne', congrArg (fun z : ℝ => z * (p⁻¹-r⁻¹)) hy,
    congrArg (fun z : ℝ => z * (∑ i, Real.log (partition (mass p i)) -
      ∑ i, Real.log (partition (mass r i)) )) (mul_inv_cancel₀ hL.ne')]

end
end GASS
