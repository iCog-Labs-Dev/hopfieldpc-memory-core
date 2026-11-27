import GASS.Mixture
import GASS.Kalman
import Mathlib.Analysis.Convex.Mul

namespace GASS
open scoped BigOperators InnerProductSpace
noncomputable section
variable {E ι κ : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Fintype ι]

/-- Jensen's squared-norm bound on a change in readout with frozen read weights. -/
theorem read_step_sq {w : ι → ℝ} (hw : ∀ j, 0 ≤ w j) (hs : ∑ j, w j = 1) (d : ι → E) :
    ‖∑ j, w j • d j‖^2 ≤ ∑ j, w j * ‖d j‖^2 := by
  have h := norm_sum_le Finset.univ (fun j => w j • d j)
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw _)] at h
  have hj := (convexOn_pow (𝕜 := ℝ) 2).map_sum_le (p := fun j => ‖d j‖)
    (fun j _ => hw j) hs (fun j _ => norm_nonneg _)
  simp only [smul_eq_mul] at hj
  exact (pow_le_pow_left₀ (norm_nonneg _) h 2).trans hj

variable [Nonempty ι]

def drive (q η r w : ℝ) (v m grad : E) : E := (q*r) • (v-m) - (η*w) • grad
def curvature (q η L r w : ℝ) := q*r + η*L*w

/-- Eqs. 44–45 before summing rows. `q = χ/R`, `base` is the key-only positive mass.
    The loss hypothesis is the smooth upper bound along this particular readout chord. -/
theorem arku_row (base w : ι → ℝ) (m d : ι → E) (v grad : E) (q η L lossChange : ℝ)
    (hb : ∀ j, 0 < base j) (hw : ∀ j, 0 ≤ w j) (hs : ∑ j, w j = 1)
    (hη : 0 ≤ η) (hL : 0 ≤ L)
    (smooth : lossChange ≤ ⟪grad, ∑ j, w j • d j⟫_ℝ + L/2*‖∑ j, w j • d j‖^2) :
    let old := fun j => base j * Real.exp (-q/2 * ‖v-m j‖^2)
    let new := fun j => base j * Real.exp (-q/2 * ‖v-(m j+d j)‖^2);
    -Real.log (partition new) + Real.log (partition old) + η*lossChange ≤
      ∑ j, (-⟪drive q η (weight old j) (w j) v (m j) grad, d j⟫_ℝ +
        curvature q η L (weight old j) (w j)/2 * ‖d j‖^2) := by
  intro old new
  let t := fun j => q*⟪v-m j,d j⟫_ℝ - q/2*‖d j‖^2
  have hz := mixture_change (a := old) (fun j => mul_pos (hb j) (Real.exp_pos _)) t
  have he : (fun j => old j * Real.exp (t j)) = new := by
    funext j
    dsimp [old,new,t]
    rw [mul_assoc, ← Real.exp_add, ← sub_sub, norm_sub_sq_real (v-m j) (d j)]
    congr 2; ring
  rw [he] at hz
  have hloss : lossChange ≤ (∑ j, w j * ⟪grad,d j⟫_ℝ) + L/2 * (∑ j, w j*‖d j‖^2) := by
    rw [inner_sum] at smooth
    simp only [real_inner_smul_right] at smooth
    exact smooth.trans (add_le_add_left (mul_le_mul_of_nonneg_left (read_step_sq hw hs d)
      (by positivity : 0 ≤ L/2)) _)
  refine (add_le_add hz (mul_le_mul_of_nonneg_left hloss hη)).trans_eq ?_
  rw [mul_add, ← Finset.sum_neg_distrib, Finset.mul_sum, ← mul_assoc, Finset.mul_sum,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _
  simp only [drive, curvature, inner_sub_left, real_inner_smul_left, t]
  ring

/-- Projection in displacement coordinates (the feasible set is Cv minus the old value).
    Every other feasible displacement pays at least a/2 times its squared distance from d. -/
theorem projected_quadratic_gap {C : Set E} (hc : Convex ℝ C) {a : ℝ} (ha : 0 < a)
    (b d : E) (hd : d ∈ C)
    (proj : ‖a⁻¹ • b-d‖ = ⨅ z : C, ‖a⁻¹ • b-z‖) (z : E) (hz : z ∈ C) :
    let U := fun x : E => -⟪b,x⟫_ℝ + a/2*‖x‖^2
    a/2*‖z-d‖^2 ≤ U z-U d := by
  intro U
  have h := (norm_eq_iInf_iff_real_inner_le_zero hc hd).1 proj z hz
  simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_self_eq_norm_sq] at h
  rw [norm_sub_sq_real, ← real_inner_comm z d]
  dsimp [U]
  have hh := mul_nonpos_of_nonneg_of_nonpos ha.le h
  simp only [mul_sub, ← mul_assoc, mul_inv_cancel₀ ha.ne', one_mul] at hh
  linarith

/-- Eq. 42 has a unique minimizer on every nonempty closed convex displacement
    domain in a complete real inner product space. -/
theorem projected_minimum [CompleteSpace E] {C : Set E} (hc : Convex ℝ C) (hne : C.Nonempty)
    (closed : IsClosed C) {a : ℝ} (ha : 0 < a) (b : E) :
    ∃ d ∈ C, ‖a⁻¹ • b-d‖ = (⨅ z : C, ‖a⁻¹ • b-z‖) ∧
      ∀ z ∈ C, z ≠ d → -⟪b,d⟫_ℝ+a/2*‖d‖^2 < -⟪b,z⟫_ℝ+a/2*‖z‖^2 := by
  obtain ⟨d,hd,proj⟩ := exists_norm_eq_iInf_of_complete_convex hne closed.isComplete hc (a⁻¹ • b)
  refine ⟨d,hd,proj,?_⟩
  intro z hz hzd
  have gap := projected_quadratic_gap hc ha b d hd proj z hz
  have pos := mul_pos (half_pos ha) (sq_pos_of_ne_zero (norm_ne_zero_iff.mpr (sub_ne_zero.mpr hzd)))
  linarith

omit [Nonempty ι] in
/-- Eq. 43 from the per-row quadratic bounds and Eq. 42. `before` and `after` are
    row costs (negative log-mixture plus weighted retrieval loss); fixed terms cancel.
    `b i j` and `c i j` are the per-row drive and curvature in `arku_row`. -/
theorem arku_descent [Fintype κ] [Nonempty κ] (before after : κ → ℝ) (b : κ → ι → E)
    (c : κ → ι → ℝ) (P : ι → ℝ) (d : ι → E) (C : ι → Set E)
    (hp : ∀ j, 0 < P j) (hc : ∀ i j, 0 ≤ c i j) (conv : ∀ j, Convex ℝ (C j))
    (hzero : ∀ j, 0 ∈ C j) (hfeas : ∀ j, d j ∈ C j)
    (row : ∀ i, after i-before i ≤ ∑ j, (-⟪b i j,d j⟫_ℝ + c i j/2*‖d j‖^2))
    (proj : ∀ j, let a := (P j)⁻¹ + ∑ i, c i j
                 let center := a⁻¹ • ∑ i, b i j
                 ‖center-d j‖ = ⨅ z : C j, ‖center-z‖) :
    let L : ℝ := Fintype.card κ
    (∑ i, after i)/L + (∑ j, ‖d j‖^2 / P j)/(2*L) ≤ (∑ i, before i)/L := by
  intro L
  have hL : 0 < L := Nat.cast_pos.mpr Fintype.card_pos
  have hj (j : ι) : (∑ i, (-⟪b i j,d j⟫_ℝ + c i j/2*‖d j‖^2)) + ‖d j‖^2 / P j / 2 ≤ 0 := by
    have hsum : 0 ≤ ∑ i, c i j := Finset.sum_nonneg fun i _ => hc i j
    have ha : 0 < (P j)⁻¹ + ∑ i, c i j := add_pos_of_pos_of_nonneg (inv_pos.mpr (hp j)) hsum
    have h := projected_quadratic_gap (conv j) ha (∑ i, b i j) (d j) (hfeas j) (proj j) 0 (hzero j)
    simp only [inner_zero_right, neg_zero, norm_zero, zero_pow (by decide : 2 ≠ 0),
      mul_zero, add_zero, zero_sub, norm_neg, sum_inner] at h
    rw [Finset.sum_add_distrib, Finset.sum_neg_distrib]
    simp_rw [div_mul_eq_mul_div, ← Finset.sum_div, ← Finset.sum_mul]
    have hn := mul_nonneg (half_pos ha).le (sq_nonneg ‖d j‖)
    simp only [div_eq_mul_inv] at *
    linarith
  have hr := Finset.sum_le_sum (s := Finset.univ) (fun i _ => row i)
  have hs := Finset.sum_nonpos (s := Finset.univ) (fun j _ => hj j)
  rw [Finset.sum_sub_distrib, Finset.sum_comm] at hr
  rw [Finset.sum_add_distrib, ← Finset.sum_div] at hs
  rw [← div_div, ← add_div]
  exact (div_le_div_iff_of_pos_right hL).2 (by linarith)

/-- Theorem 7 on the full fixed window. Keys, weights, variances and capacity
    are held fixed, so the omitted additive terms in J cancel. `q` is χ/R. -/
theorem arku_window [Fintype κ] [Nonempty κ]
    (base w : κ → ι → ℝ) (m d : ι → E) (v grad : κ → E) (lossOld lossNew : κ → ℝ)
    (q η L : ℝ) (P : ι → ℝ) (C : ι → Set E)
    (hb : ∀ i j, 0 < base i j) (hw : ∀ i j, 0 ≤ w i j) (hs : ∀ i, ∑ j, w i j = 1)
    (hq : 0 ≤ q) (hη : 0 ≤ η) (hL : 0 ≤ L) (hp : ∀ j, 0 < P j)
    (conv : ∀ j, Convex ℝ (C j)) (hzero : ∀ j, 0 ∈ C j) (hf : ∀ j, d j ∈ C j)
    (smooth : ∀ i, lossNew i-lossOld i ≤ ⟪grad i,∑ j, w i j • d j⟫_ℝ +
      L/2*‖∑ j, w i j • d j‖^2) :
    let old := fun i j => base i j * Real.exp (-q/2 * ‖v i-m j‖^2)
    let new := fun i j => base i j * Real.exp (-q/2 * ‖v i-(m j+d j)‖^2)
    let r := fun i => weight (old i)
    let b := fun i j => drive q η (r i j) (w i j) (v i) (m j) (grad i)
    let c := fun i j => curvature q η L (r i j) (w i j)
    let J := fun (mass : κ → ι → ℝ) (loss : κ → ℝ) =>
      (∑ i, (-Real.log (partition (mass i)) + η*loss i))/(Fintype.card κ : ℝ)
    (∀ j, let a := (P j)⁻¹ + ∑ i, c i j
          let center := a⁻¹ • ∑ i, b i j
          ‖center-d j‖ = ⨅ z : C j, ‖center-z‖) →
    J new lossNew + (∑ j, ‖d j‖^2 / P j)/(2*(Fintype.card κ : ℝ)) ≤ J old lossOld := by
  intro old new r b c J proj
  apply arku_descent _ _ b c P d C hp _ conv hzero hf _ proj
  · intro i j
    have hr : 0 ≤ r i j := div_nonneg (mul_pos (hb i j) (Real.exp_pos _)).le
      (partition_pos fun k => mul_pos (hb i k) (Real.exp_pos _)).le
    exact add_nonneg (mul_nonneg hq hr) (mul_nonneg (mul_nonneg hη hL) (hw i j))
  · intro i
    have h := arku_row (base i) (w i) m d (v i) (grad i) q η L (lossNew i-lossOld i)
      (hb i) (hw i) (hs i) hη hL (smooth i)
    change -Real.log (partition (new i)) + Real.log (partition (old i)) + η*(lossNew i-lossOld i)
      ≤ ∑ j, (-⟪b i j,d j⟫_ℝ + c i j/2*‖d j‖^2) at h
    linarith

/-- The unconstrained value-aware update with zero retrieval weight in Eq. 42. -/
theorem arku_kalman_reduction {P R N : ℝ} (hp : 0 < P) (hr : 0 < R) (hn : 0 ≤ N)
    (μ centroid : ℝ) :
    μ + (N/R*(centroid-μ))/(1/P+N/R) = μ + gain P R N*(centroid-μ) := by
  have h : 0 < R+P*N := by positivity
  have h' : 0 < 1/P+N/R := by positivity
  unfold gain
  field_simp [hp.ne',hr.ne',h.ne',h'.ne']; ring

end
end GASS
