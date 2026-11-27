import GASS.Mixture

namespace GASS
open scoped BigOperators
noncomputable section
variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- The normalized responsibility masses maximize the mixture-weight term.
    Interpolating from the old mixture toward them also improves that term. -/
theorem mixture_interpolation (p u : ι → ℝ) (hp : ∀ j, 0 < p j) (hu : ∀ j, 0 < u j)
    (sp : ∑ j, p j = 1) (su : ∑ j, u j = 1) {ζ : ℝ} (hz : 0 ≤ ζ) (hz' : ζ ≤ 1) :
    0 ≤ ∑ j, u j*(Real.log ((1-ζ)*p j+ζ*u j)-Real.log (p j)) := by
  have kl (j : ι) : u j*(Real.log (p j)-Real.log (u j)) ≤ p j-u j := by
    have h := Real.log_le_sub_one_of_pos (div_pos (hp j) (hu j))
    rw [Real.log_div (hp j).ne' (hu j).ne'] at h
    have e : u j*(p j/u j-1) = p j-u j := by field_simp [(hu j).ne']
    exact (mul_le_mul_of_nonneg_left h (hu j).le).trans_eq e
  have hk := Finset.sum_le_sum (s := Finset.univ) (fun j _ => kl j)
  rw [Finset.sum_sub_distrib,sp,su,sub_self] at hk
  have hrow (j : ι) : -ζ*(u j*(Real.log (p j)-Real.log (u j))) ≤
      u j*(Real.log ((1-ζ)*p j+ζ*u j)-Real.log (p j)) := by
    have h := strictConcaveOn_log_Ioi.concaveOn.2 (hp j) (hu j)
      (sub_nonneg.mpr hz') hz (by ring : 1-ζ+ζ=1)
    simp only [smul_eq_mul] at h
    nlinarith [mul_le_mul_of_nonneg_left h (hu j).le]
  have h := Finset.sum_le_sum (s := Finset.univ) (fun j _ => hrow j)
  rw [← Finset.mul_sum] at h
  exact (mul_nonneg_of_nonpos_of_nonpos (neg_nonpos.mpr hz) hk).trans h

/-- Theorem 6. Changes in key log-scores, value log-scores and log mixture
    weights add in each logit. The inequalities are the frozen block checks
    (the Gaussian prediction penalty is included on the value side). -/
theorem regularized_write [Nonempty ι] (a key value : κ → ι → ℝ) (priorChange : ι → ℝ) (penalty : ℝ)
    (ha : ∀ i j, 0 < a i j)
    (hk : ∀ j, 0 ≤ ∑ i, weight (a i) j * key i j)
    (hv : penalty ≤ ∑ i, ∑ j, weight (a i) j * value i j)
    (hm : 0 ≤ ∑ j, (∑ i, weight (a i) j)*priorChange j) :
    let d := fun i j => key i j + value i j + priorChange j
    (∑ i, -Real.log (partition fun j => a i j * Real.exp (d i j))) + penalty ≤
      ∑ i, -Real.log (partition (a i)) := by
  intro d
  have h := Finset.sum_le_sum (s := Finset.univ) fun i _ => mixture_change (ha i) (d i)
  rw [Finset.sum_add_distrib, Finset.sum_neg_distrib, Finset.sum_neg_distrib] at h
  have keypos : 0 ≤ ∑ i, ∑ j, weight (a i) j * key i j := by
    rw [Finset.sum_comm]; exact Finset.sum_nonneg fun j _ => hk j
  have mixpos : 0 ≤ ∑ i, ∑ j, weight (a i) j * priorChange j := by
    rw [Finset.sum_comm]; simpa only [Finset.sum_mul] using hm
  dsimp only [d] at h
  simp only [mul_add,Finset.sum_add_distrib] at h
  rw [Finset.sum_neg_distrib,Finset.sum_neg_distrib]
  linarith

end
end GASS
