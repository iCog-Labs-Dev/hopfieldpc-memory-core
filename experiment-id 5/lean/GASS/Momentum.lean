import GASS.Common

namespace GASS
noncomputable section

/-- Theorem 5, per coordinate of the position–velocity filter. The prior has
    entries (p,c;c,s). Positive leading minor and determinant characterize
    positive definiteness for a symmetric 2×2 matrix. R here is the batch variance R/N. -/
theorem momentum_covariance {p c s R : ℝ} (hp : 0 < p) (hdet : 0 < p*s-c^2) (hR : 0 < R) :
    let g := p/(p+R)
    let h := c/(p+R)
    let pp := p*R/(p+R)
    let cc := c*R/(p+R)
    let ss := s-c^2/(p+R)
    -- Three independent entries of the Joseph form, Eq. 30.
    (1-g)^2*p + R*g^2 = pp ∧
    (1-g)*(c-h*p) + R*g*h = cc ∧
    s-2*h*c+h^2*p+R*h^2 = ss ∧
    0 < pp ∧ 0 < pp*ss-cc^2 ∧
    -- Left multiplication by prior precision plus diag(1/R,0) gives identity.
    (s/(p*s-c^2)+1/R)*pp - c/(p*s-c^2)*cc = 1 ∧
    (s/(p*s-c^2)+1/R)*cc - c/(p*s-c^2)*ss = 0 ∧
    -c/(p*s-c^2)*pp+p/(p*s-c^2)*cc = 0 ∧
    -c/(p*s-c^2)*cc+p/(p*s-c^2)*ss = 1 := by
  intro g h pp cc ss
  have hd : 0 < p+R := by positivity
  have det : pp*ss-cc^2 = (p*s-c^2)*R/(p+R) := by
    dsimp [pp,cc,ss]; field_simp; ring
  refine ⟨?_, ?_, ?_, by dsimp [pp]; positivity, det ▸ div_pos (mul_pos hdet hR) hd,
    ?_, ?_, ?_, ?_⟩ <;> dsimp [g,h,pp,cc,ss] <;> field_simp [hdet.ne',hR.ne',hd.ne'] <;> ring

/-- Completing the square at Eq. 29. Here `x,y` are displacements from the prior
    position/velocity and `v` is the observed position minus the predicted position. -/
theorem momentum_gap {p c s R : ℝ} (hp : 0 < p) (hdet : 0 < p*s-c^2) (hR : 0 < R)
    (v x y : ℝ) :
    let m := p/(p+R)*v
    let b := c/(p+R)*v
    let Q := fun z w : ℝ => (s*z^2-2*c*z*w+p*w^2)/(2*(p*s-c^2)) + (z-v)^2/(2*R)
    Q x y-Q m b = ((x-m)^2 / p + ((p*(y-b)-c*(x-m))^2)/(p*(p*s-c^2)) + (x-m)^2/R)/2 ∧
    ((x,y) ≠ (m,b) → Q m b < Q x y) := by
  intro m b Q
  have hd : 0 < p+R := by positivity
  have eq : Q x y-Q m b =
      ((x-m)^2 / p + ((p*(y-b)-c*(x-m))^2)/(p*(p*s-c^2)) + (x-m)^2/R)/2 := by
    dsimp [Q,m,b]
    field_simp [hd.ne',hp.ne',hdet.ne',hR.ne']; ring
  refine ⟨eq, fun h => sub_pos.mp ?_⟩
  rw [eq]
  by_cases hx : x = m
  · have hy : y ≠ b := fun e => h (Prod.ext hx e)
    rw [hx,sub_self]; simp only [zero_pow (by decide : 2 ≠ 0), zero_div, mul_zero, sub_zero,
      zero_add, add_zero]
    exact half_pos (div_pos (sq_pos_of_ne_zero (mul_ne_zero hp.ne' (sub_ne_zero.mpr hy)))
      (mul_pos hp hdet))
  · exact half_pos (add_pos_of_pos_of_nonneg
      (add_pos_of_pos_of_nonneg (div_pos (sq_pos_of_ne_zero (sub_ne_zero.mpr hx)) hp)
        (div_nonneg (sq_nonneg _) (mul_pos hp hdet).le)) (div_nonneg (sq_nonneg _) hR.le))

end
end GASS
