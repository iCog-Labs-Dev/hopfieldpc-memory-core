import GASS.Common
import Mathlib.Analysis.Convex.Deriv

namespace GASS
noncomputable section

def sigmoid (z : ℝ) := Real.exp z / (1+Real.exp z)
def softplus (z : ℝ) := Real.log (1+Real.exp z)

/-- Softplus form of Eq. 35 for a single coordinate. -/
def pixelLoss (v h : ℝ) := softplus h - sigmoid v*h - softplus v + sigmoid v*v

theorem sigmoid_deriv (z : ℝ) : HasDerivAt sigmoid (sigmoid z*(1-sigmoid z)) z := by
  have h : 0 < 1+Real.exp z := by positivity
  convert (Real.hasDerivAt_exp z).div ((Real.hasDerivAt_exp z).const_add 1) h.ne' using 1
  dsimp [sigmoid]; field_simp; ring

/-- Eqs. 36–37 before taking the mean over coordinates. -/
theorem pixel_gradient (v h : ℝ) : HasDerivAt (pixelLoss v) (sigmoid h-sigmoid v) h := by
  have hd : 0 < 1+Real.exp h := by positivity
  have hs : HasDerivAt softplus (sigmoid h) h := by
    convert ((Real.hasDerivAt_exp h).const_add 1).log hd.ne' using 1
  change HasDerivAt (fun z => softplus z - sigmoid v*z - softplus v + sigmoid v*v) _ _
  simpa only [id_eq, mul_one] using (((hs.sub ((hasDerivAt_id h).const_mul (sigmoid v))).sub_const
    (softplus v)).add_const (sigmoid v*v))

theorem pixel_curvature (z : ℝ) : 0 ≤ sigmoid z*(1-sigmoid z) ∧ sigmoid z*(1-sigmoid z) ≤ 1/4 := by
  have h : 0 < 1+Real.exp z := by positivity
  have h₀ : 0 ≤ sigmoid z := div_nonneg (Real.exp_pos _).le h.le
  have h₁ : sigmoid z ≤ 1 := (div_le_one h).2 (by linarith)
  exact ⟨mul_nonneg h₀ (sub_nonneg.mpr h₁), by nlinarith [sq_nonneg (sigmoid z-1/2)]⟩

/-- The concrete global descent-lemma hypothesis needed by ARKU, Eq. 37.
    Summing over D pixels and dividing by D gives the coefficient 1/(8D). -/
theorem pixel_chord (v h q : ℝ) :
    pixelLoss v (h+q)-pixelLoss v h ≤ (sigmoid h-sigmoid v)*q + q^2/8 := by
  let f := fun z : ℝ => z^2/8 - pixelLoss v (h+z) + (sigmoid h-sigmoid v)*z
  let g := fun z : ℝ => z/4 - (sigmoid (h+z)-sigmoid v) + (sigmoid h-sigmoid v)
  have hf (z : ℝ) : HasDerivAt f (g z) z := by
    convert ((((hasDerivAt_pow 2 z).div_const 8).sub
      ((pixel_gradient v (h+z)).comp z ((hasDerivAt_id z).const_add h))).add
      ((hasDerivAt_id z).const_mul (sigmoid h-sigmoid v))) using 1
    simp [f,g]; ring
  have hg (z : ℝ) : HasDerivAt g (1/4 - sigmoid (h+z)*(1-sigmoid (h+z))) z := by
    have hσ := (sigmoid_deriv (h+z)).comp z ((hasDerivAt_id z).const_add h)
    convert (((hasDerivAt_id z).div_const 4).sub (hσ.sub_const (sigmoid v))).add_const
      (sigmoid h-sigmoid v) using 1
    simp [g]
  have cv : ConvexOn ℝ Set.univ f := convexOn_of_hasDerivWithinAt2_nonneg convex_univ
    (fun z _ => (hf z).continuousAt.continuousWithinAt) (fun z _ => (hf z).hasDerivWithinAt)
    (fun z _ => (hg z).hasDerivWithinAt) (fun z _ => sub_nonneg.mpr (pixel_curvature _).2)
  have hzero : HasDerivAt f 0 0 := by simpa [g] using hf 0
  have hmin := cv.isMinOn_of_rightDeriv_eq_zero (x := 0) (by simp)
    (hzero.hasDerivWithinAt.derivWithin (uniqueDiffWithinAt_Ioi 0))
  have hq := hmin (show q ∈ Set.univ by trivial)
  simp [f] at hq
  linarith

end
end GASS
