import GASS.Mixture
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

namespace GASS
open scoped BigOperators InnerProductSpace
noncomputable section
variable {E ι : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Proposition 12. A nonzero eigenvalue and symmetry put the fission direction
    in the tangent space when the parent key is in the covariance kernel. -/
theorem tangent_eigenvector (A : E → E) (k e : E) {l : ℝ} (hl : l ≠ 0)
    (symm : ∀ x y, ⟪A x,y⟫_ℝ = ⟪x,A y⟫_ℝ) (hk : A k = 0) (he : A e = l • e) :
    ⟪k,e⟫_ℝ = 0 := by
  have h := symm k e
  rw [hk, he, inner_zero_left, real_inner_smul_right] at h
  exact (mul_eq_zero.mp h.symm).resolve_left hl

theorem angular_split (k e : E) (θ : ℝ) (hk : ‖k‖ = 1) (he : ‖e‖ = 1)
    (orth : ⟪k,e⟫_ℝ = 0) :
    let a := Real.cos θ • k + Real.sin θ • e
    let b := Real.cos θ • k - Real.sin θ • e
    ‖a‖ = 1 ∧ ‖b‖ = 1 ∧ ⟪a,b⟫_ℝ = Real.cos (2*θ) := by
  intro a b
  have ho : ⟪e,k⟫_ℝ = 0 := (real_inner_comm _ _).trans orth
  have htrig := Real.sin_sq_add_cos_sq θ
  have cosNorm : ‖Real.cos θ • k‖^2 = Real.cos θ^2 := by simp [norm_smul,hk,Real.norm_eq_abs]
  have sinNorm : ‖Real.sin θ • e‖^2 = Real.sin θ^2 := by simp [norm_smul,he,Real.norm_eq_abs]
  have ha : ‖a‖^2 = 1 := by
    rw [← real_inner_self_eq_norm_sq]; dsimp [a]
    simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right,
      real_inner_self_eq_norm_sq, hk, he, orth, ho, cosNorm, sinNorm]; nlinarith
  have hb : ‖b‖^2 = 1 := by
    rw [← real_inner_self_eq_norm_sq]; dsimp [b]
    simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right,
      real_inner_self_eq_norm_sq, hk, he, orth, ho, cosNorm, sinNorm]; nlinarith
  refine ⟨by nlinarith [norm_nonneg a], by nlinarith [norm_nonneg b], ?_⟩
  dsimp [a,b]
  simp only [inner_add_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right,
    real_inner_self_eq_norm_sq, hk, he, orth, ho, cosNorm, sinNorm, Real.cos_two_mul']; ring

theorem angular_threshold {τ θ : ℝ} (h₀ : -1 ≤ τ) (h₁ : τ ≤ 1)
    (target : Real.arccos τ/2 ≤ θ) (hθ : θ ≤ Real.pi/2) :
    Real.cos (2*θ) ≤ τ := by
  have h := Real.cos_le_cos_of_nonneg_of_le_pi (Real.arccos_nonneg τ)
    (by linarith : 2*θ ≤ Real.pi) (by linarith : Real.arccos τ ≤ 2*θ)
  simpa only [Real.cos_arccos h₀ h₁] using h

variable [Fintype ι] [Nonempty ι]
def directionalMass (π : ι → ℝ) (k : ι → E) (β : ℝ) (u : E) (j : ι) :=
  π j * Real.exp (β*⟪k j,u⟫_ℝ)

/-- Proposition 17. The candidate unit direction improves the frozen-bank score.
    The zero-resultant branch retains u, so its score change is exactly zero. -/
theorem directional_read (π : ι → ℝ) (k : ι → E) {β : ℝ} (u : E)
    (hp : ∀ j, 0 < π j) (hs : ∑ j, π j = 1) (hk : ∀ j, ‖k j‖ = 1)
    (hb : 0 ≤ β) (hu : ‖u‖ = 1) :
    let a := directionalMass π k β u
    let g := ∑ j, weight a j • k j
    let v := if ‖g‖ = 0 then u else ‖g‖⁻¹ • g
    let f := fun x => Real.log (partition (directionalMass π k β x))
    f u ≤ f v ∧ f v ≤ β ∧ ‖v‖ = 1 := by
  classical
  intro a g v f
  have hv : ‖v‖ = 1 := by
    by_cases h : g = 0
    · simpa [v,h] using hu
    · simp [v,h,norm_smul,Real.norm_eq_abs,abs_of_nonneg (inv_nonneg.mpr (norm_nonneg g)),
        norm_ne_zero_iff.mpr h]
  have hmax : ⟪g,u⟫_ℝ ≤ ⟪g,v⟫_ℝ := by
    by_cases h : g = 0
    · simp [h]
    have hn := norm_ne_zero_iff.mpr h
    have he : ⟪g,v⟫_ℝ = ‖g‖ := by
      simp [v,h,real_inner_smul_right,real_inner_self_eq_norm_sq, sq, hn]
    rw [he]; simpa [hu] using real_inner_le_norm g u
  let d := fun j => β*⟪k j,v⟫_ℝ - β*⟪k j,u⟫_ℝ
  have h := mixture_change (a := a) (fun j => mul_pos (hp j) (Real.exp_pos _)) d
  have change : (fun j => a j * Real.exp (d j)) = directionalMass π k β v := by
    funext j; dsimp [a,d,directionalMass]; rw [mul_assoc, ← Real.exp_add]; congr 2; ring
  rw [change] at h
  have improve : 0 ≤ ∑ j, weight a j * d j := by
    have eq : (∑ j, weight a j * d j) = β*(⟪g,v⟫_ℝ-⟪g,u⟫_ℝ) := by
      dsimp [g]; simp only [sum_inner, real_inner_smul_left, ← Finset.sum_sub_distrib, Finset.mul_sum]
      exact Finset.sum_congr rfl fun _ _ => by dsimp [d]; ring
    rw [eq]; exact mul_nonneg hb (sub_nonneg.mpr hmax)
  refine ⟨by dsimp [f]; change -f v + f u ≤ _ at h; linarith, ?_, hv⟩
  have hupp (j : ι) : β*⟪k j,v⟫_ℝ ≤ β := by
    have := real_inner_le_norm (k j) v
    rw [hk,hv,one_mul] at this
    simpa using mul_le_mul_of_nonneg_left this hb
  exact neg_le_neg_iff.mp (mixture_floor hp hs hupp)

/-- Iterating the update of Proposition 17 gives a convergent sequence of scores. -/
theorem directional_score_limit (π : ι → ℝ) (k : ι → E) {β : ℝ} (u : ℕ → E)
    (hp : ∀ j, 0 < π j) (hs : ∑ j, π j = 1) (hk : ∀ j, ‖k j‖ = 1)
    (hb : 0 ≤ β) (hu : ‖u 0‖ = 1)
    (step : ∀ n,
      let g := ∑ j, weight (directionalMass π k β (u n)) j • k j
      u (n+1) = if ‖g‖ = 0 then u n else ‖g‖⁻¹ • g) :
    let f := fun n => Real.log (partition (directionalMass π k β (u n)))
    Filter.Tendsto f Filter.atTop (nhds (⨆ n, f n)) := by
  intro f
  have unit (n : ℕ) : ‖u n‖ = 1 := by
    induction n with
    | zero => exact hu
    | succ n ih => rw [step]; exact (directional_read π k (u n) hp hs hk hb ih).2.2
  have mono : Monotone f := monotone_nat_of_le_succ fun n => by
    dsimp [f]; rw [step]; exact (directional_read π k (u n) hp hs hk hb (unit n)).1
  have bound (n : ℕ) : f n ≤ β := by
    apply neg_le_neg_iff.mp (mixture_floor hp hs _)
    intro j
    have h := real_inner_le_norm (k j) (u n)
    rw [hk,unit,one_mul] at h
    simpa using mul_le_mul_of_nonneg_left h hb
  exact tendsto_atTop_ciSup mono ⟨β,Set.forall_mem_range.mpr bound⟩

end
end GASS
