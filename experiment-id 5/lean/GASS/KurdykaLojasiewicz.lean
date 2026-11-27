import GASS.Stationary
import Mathlib.Analysis.Convex.Deriv

namespace GASS
open Filter Set
open scoped BigOperators Topology
noncomputable section
variable {E : Type*} [NormedAddCommGroup E]

/-- Local KL data restricted to the iterates. The gap is the objective minus its
    limiting value. A KL inequality for every feasible point in the neighborhood
    implies the last field when res bounds the distance to the subdifferential. -/
structure LocalKL (x : ℕ → E) (gap res : ℕ → ℝ) (p : E) where
  region : Set E
  isOpen_region : IsOpen region
  mem_region : p ∈ region
  width : ℝ
  width_pos : 0 < width
  φ : ℝ → ℝ
  slope : ℝ → ℝ
  concave : ConcaveOn ℝ (Ico 0 width) φ
  nonneg : ∀ z ∈ Ico 0 width, 0 ≤ φ z
  deriv : ∀ z ∈ Ioo 0 width, HasDerivAt φ (slope z) z
  slope_nonneg : ∀ z ∈ Ioo 0 width, 0 ≤ slope z
  kl : ∀ n, x n ∈ region → gap n ∈ Ioo 0 width → 1 ≤ slope (gap n)*res n

/-- The slope at a positive gap times the next decrease is bounded by the
    decrease of the desingularizing function. At zero use the zero slope. -/
private theorem local_tangent {x : ℕ → E} {gap res : ℕ → ℝ} {p : E} (c : LocalKL x gap res p)
    {v u : ℝ} (hv : 0 ≤ v) (hvu : v ≤ u) (hu : u < c.width) :
    (if 0 < u then c.slope u else 0)*(u-v) ≤ c.φ u-c.φ v := by
  rcases eq_or_lt_of_le hvu with e | e
  · simp [e]
  have hpos := hv.trans_lt e
  rw [if_pos hpos]
  have h := c.concave.le_slope_of_hasDerivAt ⟨hv,e.trans hu⟩ ⟨hpos.le,hu⟩ e
    (c.deriv u ⟨hpos,hu⟩)
  rw [slope_def_field] at h
  exact (le_div_iff₀ (sub_pos.mpr e)).mp h

/-- Theorem 16. Pointwise KL neighborhoods at cluster points suffice. Compactness
    gives a finite cover; summing its desingularizers gives one valid on a tail. -/
theorem kl_of_cluster_points [CompleteSpace E] (K : Set E) (compact : IsCompact K)
    (x : ℕ → E) (gap d res : ℕ → ℝ) (feas : ∀ n, x n ∈ K)
    {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hd : ∀ n, 0 ≤ d n) (hr : ∀ n, 0 ≤ res n)
    (hgap : ∀ n, 0 ≤ gap n) (lim : Tendsto gap atTop (𝓝 0))
    (step : ∀ n, dist (x n) (x (n+1)) ≤ d (n+1))
    (decrease : ∀ n, a*(d (n+1))^2 ≤ gap n-gap (n+1))
    (rel : ∀ n, res n ≤ b*d n)
    (localKL : ∀ p ∈ K, MapClusterPt p atTop x → Nonempty (LocalKL x gap res p)) :
    Summable d ∧ ∃ p ∈ K, Tendsto x atTop (𝓝 p) := by
  classical
  let cluster := K ∩ {p | MapClusterPt p atTop x}
  have hC : IsCompact cluster := compact.inter_right isClosed_setOf_clusterPt
  let chart (p : cluster) := Classical.choice (localKL p p.property.1 p.property.2)
  obtain ⟨T, cover⟩ := hC.elim_finite_subcover (fun p => (chart p).region)
    (fun p => (chart p).isOpen_region) (fun p hp => mem_iUnion.mpr ⟨⟨p,hp⟩,(chart ⟨p,hp⟩).mem_region⟩)
  let V := ⋃ p ∈ T, (chart p).region
  have hopen : IsOpen V := isOpen_biUnion fun p _ => (chart p).isOpen_region
  have inside : ∀ᶠ n in atTop, x n ∈ V := by
    by_contra h
    have outside : ∃ᶠ n in atTop, x n ∈ K ∩ Vᶜ :=
      (not_eventually.mp h).mono fun n hn => ⟨feas n,hn⟩
    obtain ⟨p,hp,cl⟩ := (compact.inter_right hopen.isClosed_compl).exists_mapClusterPt_of_frequently outside
    exact hp.2 (cover (show p ∈ cluster from ⟨hp.1,cl⟩))
  have narrow : ∀ᶠ n in atTop, ∀ p ∈ T, gap n < (chart p).width :=
    (eventually_all_finset T).2 fun p _ => lim.eventually_lt_const (chart p).width_pos
  obtain ⟨N,hN⟩ := eventually_atTop.1 (inside.and narrow)
  let φ := fun n => ∑ p ∈ T, (chart p).φ (gap (N+n))
  let s := fun n p => if 0 < gap (N+n) then (chart p).slope (gap (N+n)) else 0
  let slope := fun n => ∑ p ∈ T, s n p
  have narrow' (n) := (hN (N+n) (Nat.le_add_right ..)).2
  have hs (n p) (hp : p ∈ T) : 0 ≤ s n p := by
    dsimp [s]; split_ifs with hz
    · exact (chart p).slope_nonneg _ ⟨hz,narrow' n p hp⟩
    · rfl
  have mono (n : ℕ) : gap (n+1) ≤ gap n := by
    linarith [decrease n, mul_nonneg ha.le (sq_nonneg (d (n+1)))]
  have tangent (n) : slope n*(gap (N+n)-gap (N+(n+1))) ≤ φ n-φ (n+1) := by
    dsimp only [slope,φ]
    rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_le_sum fun p hp => local_tangent (chart p) (hgap _) (mono _) (narrow' n p hp)
  have kl (n) (hg : 0 < gap (N+n)) : 1 ≤ slope n*res (N+n) := by
    obtain ⟨p,hp,hx⟩ := mem_iUnion₂.mp (show x (N+n) ∈ V from (hN _ (Nat.le_add_right ..)).1)
    have h := (chart p).kl _ hx ⟨hg,narrow' n p hp⟩
    have hle := Finset.single_le_sum (f := s n) (hs n) hp
    exact h.trans (mul_le_mul_of_nonneg_right (by simpa [s,hg,slope] using hle) (hr _))
  obtain ⟨sum,p,conv⟩ := kl_endpoint_limit (fun n => x (N+n)) (fun n => gap (N+n))
    (fun n => d (N+n)) φ slope (fun n => res (N+n)) ha hb (fun _ => hd _) (fun _ => hgap _)
    (fun n => Finset.sum_nonneg fun p hp => (chart p).nonneg _ ⟨hgap _,narrow' n p hp⟩)
    (fun n => Finset.sum_nonneg (hs n)) (fun n => step (N+n)) (fun n => decrease (N+n))
    (fun _ => rel _) tangent kl
  have hx : Tendsto x atTop (𝓝 p) := (tendsto_add_atTop_iff_nat N).1 (by simpa [Nat.add_comm] using conv)
  exact ⟨(summable_nat_add_iff N).1 (by simpa [Nat.add_comm] using sum),
    p,compact.isClosed.mem_of_tendsto hx (Eventually.of_forall feas),hx⟩

/-- Full critical-limit conclusion from the endpoint certificates. The graph G is
    the constrained subdifferential on K; the squared decrease uses the length of
    the sweep ending at n+1. Each subgradient belongs to the endpoint it certifies. -/
theorem critical_limit [CompleteSpace E] (K : Set E) (compact : IsCompact K)
    (x g : ℕ → E) (F d : ℕ → ℝ) (G : Set (E × E))
    (closed : IsClosed G) (cert : ∀ n, (x n,g n) ∈ G) (feas : ∀ n, x n ∈ K)
    {a b floor : ℝ} (ha : 0 < a) (hb : 0 < b) (hd : ∀ n, 0 ≤ d n)
    (lower : ∀ n, floor ≤ F n) (step : ∀ n, dist (x n) (x (n+1)) ≤ d (n+1))
    (decrease : ∀ n, a*(d (n+1))^2 ≤ F n-F (n+1)) (rel : ∀ n, ‖g n‖ ≤ b*d n)
    (localKL : ∀ p ∈ K, MapClusterPt p atTop x →
      Nonempty (LocalKL x (fun n => F n-⨅ t, F t) (fun n => ‖g n‖) p)) :
    Summable d ∧ ∃ p ∈ K, (p,0) ∈ G ∧ Tendsto x atTop (𝓝 p) := by
  let Fstar := ⨅ n, F n
  have hfloor : BddBelow (Set.range F) := ⟨floor,Set.forall_mem_range.mpr lower⟩
  have mono : Antitone F := antitone_nat_of_succ_le fun n => by
    linarith [decrease n, mul_nonneg ha.le (sq_nonneg (d (n+1)))]
  have lim : Tendsto (fun n => F n-Fstar) atTop (𝓝 0) := by
    simpa [Fstar] using (tendsto_atTop_ciInf mono hfloor).sub_const Fstar
  obtain ⟨sum,p,hp,hx⟩ := kl_of_cluster_points K compact x _ d (fun n => ‖g n‖) feas ha hb hd
    (fun _ => norm_nonneg _) (fun n => sub_nonneg.mpr (ciInf_le hfloor n)) lim step
    (fun n => by simpa only [sub_sub_sub_cancel_right] using decrease n) rel localKL
  have hg : Tendsto g atTop (𝓝 0) := tendsto_zero_iff_norm_tendsto_zero.mpr
    (squeeze_zero (fun _ => norm_nonneg _) rel (by simpa using sum.tendsto_atTop_zero.const_mul b))
  exact ⟨sum,p,hp,closed.mem_of_tendsto (hx.prodMk_nhds hg) (Eventually.of_forall cert),hx⟩

end
end GASS
