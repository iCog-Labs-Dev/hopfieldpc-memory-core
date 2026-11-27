import GASS.Common

/-! Scalar coordinates of the isotropic filter (Sections 4, 6 and 9). -/
namespace GASS
open scoped BigOperators
open Filter
noncomputable section

def gain (P R N : ℝ) := P * N / (R + P * N)
def posterior (P R N : ℝ) := 1 / (1 / P + N / R)

/-- Theorem 1, Eq. 14. The objective gap is an exact square, for any finite batch.
`m` is one coordinate and the full isotropic objective sums these gaps. -/
theorem kalman_gap {ι : Type*} [Fintype ι] (r v : ι → ℝ) (p R μ x : ℝ)
    (hp : 0 < p) (hR : 0 < R) (hr : ∀ i, 0 ≤ r i) :
    let A := 1 / p + (∑ i, r i) / R
    let m := (μ / p + (∑ i, r i * v i) / R) / A
    let Φ := fun z : ℝ => (z - μ)^2 / (2*p) + (∑ i, r i * (v i - z)^2) / (2*R)
    Φ x - Φ m = A / 2 * (x - m)^2 := by
  intro A m Φ
  have hmass : 0 ≤ ∑ i, r i := Finset.sum_nonneg fun i _ => hr i
  have hA : 0 < A := add_pos_of_pos_of_nonneg (by positivity) (div_nonneg hmass hR.le)
  have hm : m * A = μ / p + (∑ i, r i * v i) / R := div_mul_cancel₀ _ hA.ne'
  have hex (z : ℝ) : (∑ i, r i * (v i - z)^2) =
      (∑ i, r i * (v i)^2) - (∑ i, r i * v i) * (2*z) + (∑ i, r i) * z^2 := by
    rw [Finset.sum_mul, Finset.sum_mul, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  dsimp only [Φ]
  rw [hex x, hex m]
  dsimp only [A] at hm ⊢
  simp only [div_eq_mul_inv, mul_inv_rev] at hm ⊢
  linear_combination (x-m) * hm

/-- The finite-dimensional vector minimizer of Eq. 14, including batches of
    zero total mass. The left side compares arbitrary distinct prototype vectors. -/
theorem kalman_vector_minimum {ι κ : Type*} [Fintype ι] [Fintype κ] (r : ι → ℝ)
    (v : ι → κ → ℝ) (μ : κ → ℝ) {P R : ℝ} (hp : 0 < P) (hr : 0 < R) (hw : ∀ i, 0 ≤ r i) :
    let A := 1/P + (∑ i, r i)/R
    let m := fun k => (μ k/P + (∑ i, r i * v i k)/R)/A
    let Φ := fun (x : κ → ℝ) => ∑ k, ((x k-μ k)^2/(2*P) + (∑ i, r i*(v i k-x k)^2)/(2*R))
    ∀ x, x ≠ m → Φ m < Φ x := by
  intro A m Φ x hx
  classical
  have ha : 0 < A := add_pos_of_pos_of_nonneg (by positivity)
    (div_nonneg (Finset.sum_nonneg fun i _ => hw i) hr.le)
  obtain ⟨k,hk⟩ : ∃ k, x k ≠ m k := by
    by_contra! h; exact hx (funext h)
  have h := Finset.sum_congr (rfl : (Finset.univ : Finset κ) = Finset.univ)
    (fun j _ => kalman_gap r (fun i => v i j) P R (μ j) (x j) hp hr hw)
  change (∑ j, (((x j-μ j)^2/(2*P) + (∑ i, r i*(v i j-x j)^2)/(2*R)) -
    ((m j-μ j)^2/(2*P) + (∑ i, r i*(v i j-m j)^2)/(2*R)))) =
    ∑ j, A/2*(x j-m j)^2 at h
  rw [Finset.sum_sub_distrib] at h
  apply sub_pos.mp
  rw [show Φ x-Φ m = _ from h]
  exact Finset.sum_pos' (fun j _ => mul_nonneg (half_pos ha).le (sq_nonneg _))
    ⟨k,Finset.mem_univ _,mul_pos (half_pos ha) (sq_pos_of_ne_zero (sub_ne_zero.mpr hk))⟩

/-- Strict uniqueness follows coordinatewise from the gap, including empty batches. -/
theorem quadratic_strict_min {A m x : ℝ} {Φ : ℝ → ℝ} (hA : 0 < A) (hx : x ≠ m)
    (gap : Φ x - Φ m = A / 2 * (x-m)^2) : Φ m < Φ x := by
  have := mul_pos (half_pos hA) (sq_pos_of_ne_zero (sub_ne_zero.mpr hx))
  linarith

/-- The unnormalized density has Gaussian kernel centered at m with reciprocal
    precision as its variance parameter. The prefactor is independent of x. -/
theorem gaussian_kernel {A m : ℝ} {Φ : ℝ → ℝ} (_hA : 0 < A)
    (gap : ∀ x, Φ x - Φ m = A/2*(x-m)^2) (x : ℝ) :
    Real.exp (-Φ x) = Real.exp (-Φ m) * Real.exp (-(x-m)^2 / (2*(1/A))) := by
  rw [← Real.exp_add]
  congr 1
  have h : -(x-m)^2/(2*(1/A)) = -(A/2*(x-m)^2) := by field_simp; ring
  rw [h]; linarith [gap x]

/-- The mean in information form equals the centroid/gain formula at positive mass. -/
theorem kalman_mean {P R N : ℝ} (μ total : ℝ) (hp : 0 < P) (hr : 0 < R) (hn : 0 < N) :
    (μ / P + total / R) / (1 / P + N / R) = μ + gain P R N * (total / N - μ) := by
  have h : 0 < R + P*N := by positivity
  have h' : 0 < 1/P + N/R := by positivity
  unfold gain
  field_simp [hp.ne', hr.ne', hn.ne', h.ne', h'.ne']; ring

theorem posterior_eq {P R N : ℝ} (hp : 0 < P) (hr : 0 < R) (hn : 0 ≤ N) :
    posterior P R N = (1 - gain P R N) * P := by
  have h : 0 < R + P*N := by positivity
  have h' : 0 < 1/P + N/R := by positivity
  unfold posterior gain
  field_simp [hp.ne', hr.ne', h.ne', h'.ne']; ring

@[simp] theorem zero_evidence (P R : ℝ) : gain P R 0 = 0 ∧ posterior P R 0 = P := by
  simp [gain, posterior]

/-- Proposition 2. -/
theorem gain_bounds {P R N : ℝ} (hp : 0 < P) (hr : 0 < R) (hn : 0 ≤ N) :
    0 ≤ gain P R N ∧ gain P R N < 1 ∧ 0 < posterior P R N ∧ posterior P R N ≤ P := by
  have hd : 0 < R + P*N := by positivity
  have h₀ : 0 ≤ gain P R N := div_nonneg (mul_nonneg hp.le hn) hd.le
  have h₁ : gain P R N < 1 := (div_lt_iff₀ hd).2 (by linarith)
  rw [posterior_eq hp hr hn]
  exact ⟨h₀, h₁, mul_pos (sub_pos.mpr h₁) hp, by nlinarith [mul_nonneg h₀ hp.le]⟩

theorem gain_derivative {P R N : ℝ} (hp : 0 < P) (hr : 0 < R) (hn : 0 ≤ N) :
    HasDerivAt (gain P R) (P*R / (R+P*N)^2) N := by
  have hd : 0 < R + P*N := by positivity
  unfold gain
  convert ((hasDerivAt_id N).const_mul P).div
    (((hasDerivAt_id N).const_mul P).const_add R) hd.ne' using 1
  simp; ring

theorem posterior_le_noise {P R N : ℝ} (hp : 0 < P) (hr : 0 < R) (hn : 0 < N) :
    posterior P R N ≤ R/N := by
  have h : N/R ≤ 1/P + N/R := by linarith [one_div_pos.mpr hp]
  simpa [posterior] using one_div_le_one_div_of_le (div_pos hn hr) h

/-- Theorem 8. A clipped prediction and bounded fresh evidence imply Eq. 49. -/
theorem trust_bounds {lo hi P Rmin R N B : ℝ} (hl : 0 < lo) (hlp : lo ≤ P)
    (hph : P ≤ hi) (hr : 0 < Rmin) (hrr : Rmin ≤ R) (hn : 0 ≤ N) (hnb : N ≤ B) :
    1 / (1/lo + B/Rmin) ≤ posterior P R N ∧ posterior P R N ≤ hi := by
  have hp := hl.trans_le hlp
  have hR := hr.trans_le hrr
  have h : 0 < 1/P + N/R := by positivity
  exact ⟨one_div_le_one_div_of_le h (add_le_add (one_div_le_one_div_of_le hl hlp)
    (div_le_div₀ (hn.trans hnb) hnb hr hrr)), (gain_bounds hp hR hn).2.2.2.trans hph⟩

def riccati (Q B p : ℝ) := B * (p+Q) / (p+Q+B)

/-- Proposition 3. The fixed point and a geometric error estimate for every iterate. -/
theorem riccati_contraction {Q B x y : ℝ} (hq : 0 < Q) (hb : 0 < B) (hx : 0 ≤ x)
    (hy : 0 ≤ y) : |riccati Q B x - riccati Q B y| ≤ (B/(Q+B))^2 * |x-y| := by
  have hd₁ : 0 < x+Q+B := by positivity
  have hd₂ : 0 < y+Q+B := by positivity
  have hqb : 0 < Q+B := by positivity
  have hid : riccati Q B x - riccati Q B y = B^2 / ((x+Q+B)*(y+Q+B)) * (x-y) := by
    unfold riccati
    field_simp; ring
  rw [hid, abs_mul, abs_of_nonneg (by positivity : 0 ≤ B^2 / ((x+Q+B)*(y+Q+B)))]
  apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
  rw [div_pow, sq (Q+B)]
  exact div_le_div_of_nonneg_left (sq_nonneg _) (mul_pos hqb hqb)
    (mul_le_mul (by linarith) (by linarith) hqb.le hd₁.le)

theorem riccati_limit {Q B : ℝ} (hq : 0 < Q) (hb : 0 < B) (p : ℕ → ℝ)
    (h₀ : 0 ≤ p 0) (step : ∀ n, p (n+1) = riccati Q B (p n)) :
    let s := (Real.sqrt (Q^2 + 4*Q*B) - Q)/2
    0 < s ∧ riccati Q B s = s ∧ (∀ x ≥ 0, riccati Q B x = x → x = s) ∧
      Tendsto p atTop (nhds s) := by
  intro s
  have hsqrt := Real.sq_sqrt (by positivity : 0 ≤ Q^2 + 4*Q*B)
  have hs : 0 < s := by dsimp [s]; nlinarith [Real.sqrt_nonneg (Q^2 + 4*Q*B), mul_pos hq hb]
  have hpoly : s^2 + Q*s = Q*B := by dsimp [s]; nlinarith
  have hfix : riccati Q B s = s := (div_eq_iff (by positivity : s+Q+B ≠ 0)).2 (by nlinarith)
  have hnon : ∀ n, 0 ≤ p n := by
    intro n; induction n with
    | zero => exact h₀
    | succ n ih => rw [step]; unfold riccati; positivity
  let k := (B/(Q+B))^2
  have hk₀ : 0 ≤ k := sq_nonneg _
  have hk₁ : k < 1 := by
    have h : B/(Q+B) < 1 := (div_lt_one (by positivity)).2 (by linarith)
    dsimp [k]; nlinarith [div_nonneg hb.le (by positivity : 0 ≤ Q+B)]
  have bound (n : ℕ) : |p n - s| ≤ k^n * |p 0-s| := by
    induction n with
    | zero => simp
    | succ n ih =>
      rw [step]
      calc
        _ ≤ k * |p n-s| := by simpa only [hfix] using riccati_contraction hq hb (hnon n) hs.le
        _ ≤ _ := by rw [pow_succ]; nlinarith [mul_le_mul_of_nonneg_left ih hk₀]
  refine ⟨hs, hfix, ?_, ?_⟩
  · intro x hx h
    have hxq := (div_eq_iff (by positivity : x+Q+B ≠ 0)).1 h
    have : (x-s)*(x+s+Q) = 0 := by nlinarith
    exact sub_eq_zero.mp ((mul_eq_zero.mp this).resolve_right (by positivity))
  · apply tendsto_iff_dist_tendsto_zero.mpr
    apply squeeze_zero (fun n => dist_nonneg) (fun n => by simpa [Real.dist_eq] using bound n)
    simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one hk₀ hk₁).mul_const |p 0-s|

end
end GASS
