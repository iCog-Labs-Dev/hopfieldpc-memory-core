# Lean proofs for safeguarded GASS

Open this directory in VS Code with **File → Open Folder**. Open `GASS.lean` and Ctrl+click an import to jump to the proofs. Select a theorem to see its hypotheses in the Lean Infoview. The project pins Lean and mathlib to version 4.19.0.

From this directory, run `lake build` to check the project. On Windows, PowerShell can also invoke Lake with `& "$env:USERPROFILE\.elan\bin\lake.exe" build`. On a fresh checkout, `lake exe cache get` downloads precompiled mathlib dependencies before the build.

## Correspondence with the proposal

The numbers below follow *GASS: A Generalized Self-Adaptive Associative Memory*, safeguarded revision. Every inequality uses exact real arithmetic.

| PDF result | Lean module | What the statement checks |
| --- | --- | --- |
| 1–3 | `GASS.Kalman` | Finite-batch completion of the square, a unique vector minimizer, the Gaussian kernel, gain derivative and bounds, posterior covariance, and convergence to the explicit positive Riccati fixed point. |
| 4 | `GASS.Variance` | Uniqueness of the clipped variance minimizer and Eq. 21 for the finite-window log-mixture objective. Variances are expressed in units of the fixed reference variance. The proof includes zero residuals and singleton feasible intervals. |
| 5 | `GASS.Momentum` | The two-dimensional Gaussian quadratic has its unique minimum at the position–velocity update. The Joseph covariance is positive definite and inverts the updated precision. |
| 6 | `GASS.Regularized` | The normalized mixture-weight interpolation improves the frozen surrogate. The key and value block inequalities then give descent of the actual penalized log-mixture score. |
| 7 | `GASS.ARKU`, `GASS.ImageLoss` | The finite-window read and write equations give the quadratic upper bound. Nearest-point projection gives the minimizer and Eq. 43. The softplus form of the Bernoulli loss satisfies the required chord bound with curvature 1/4 per pixel, before averaging. |
| 8–9 | `GASS.Kalman`, `GASS.Mixture` | The uniform scalar trust interval, preservation of feasibility by checked commits, and membership of a readout in the convex value domain. |
| 10–12 | `GASS.Mixture`, `GASS.Geometry` | Pruning read and log-score identities, the read perturbation bound, exact coalescence of slot-local masses and numerators, and the unit norms and cosine of a directional split. |
| 13–14 | `GASS.Descent` | The edit budget on a finite fixed window, exclusion of accepted cycles, the cumulative-drift bound, and eventual cessation of edits under summable drift. The changing objectives share one feasible state type. |
| 15–16 | `GASS.Stationary`, `GASS.KurdykaLojasiewicz`, `GASS.Descent` | Objective convergence and square summability from sufficient decrease, block criticality from uniform model comparisons, and criticality from a closed graph and relative error. Local KL inequalities at cluster points give finite length and convergence to one feasible endpoint. Lean derives the uniform inequality on a tail by compactness. |
| 17 | `GASS.Geometry` | A frozen-bank directional update increases or preserves the log-partition score, preserves unit norm, and keeps the score below the inverse temperature. Iterating this update gives a convergent sequence of scores. |

## Scope of the certificates

For Theorems 15–16, the sweep decrease, block model or endpoint subgradient certificate, and local Kurdyka–Łojasiewicz property are explicit hypotheses at the points where they are used. Instantiating the full cyclic solver and proving that each named adapter has KL regularity are additional proof obligations. For Theorems 1 and 5, the Gaussian conclusions are expressed through the quadratic kernel and precision. Integration against a normalized multivariate probability measure is outside these statements.

The certificates describe the equations in the PDF. Custom adapters must supply their curvature and score bounds. The optional physical-noise estimators, performance hypotheses, and correspondence to the floating-point Python implementation require separate validation.

## Convergence for arbitrary input streams

`GASS.StreamStability` studies a stronger requirement separately from the proposal's stationary optimizer. In `kalman_two_cycle`, one slot has predicted covariance, observation variance, and evidence mass all equal to 1. Its gain is 1/2. Starting at −1/3 with targets +1, −1, +1, −1, …, its stored value alternates between −1/3 and +1/3 forever. More generally, `convergence_excludes_exact_tracking` proves that a convergent scalar response to the same query cannot have vanishing error on those alternating targets.

The module also defines an optional commit rule. Choose a total movement budget B ≥ 0. At update n, starting at n = 0, commit a feasible proposed bank only when its distance from the current bank is at most B/2^(n+1). Otherwise retain the current bank. Apply the rule to the entire state covered by the guarantee. A fixed-capacity encoding can include the active mask and all stored parameters.

| Checked result | Guarantee for every stream of proposed updates |
| --- | --- |
| `converges_for_every_stream` and `future_travel` | In a complete metric space the stored state has a limit. After n updates the total remaining travel is at most B/2^n. A closed feasible set contains the limit when it contains the initial state. |
| `continuation_stability` | Two runs at the same state at time n can receive completely different later proposals. Their limits lie within 2B/2^n of each other. |
| `structure_eventually_constant` | If changing the active mask costs at least a fixed positive distance in the chosen metric, the mask eventually stays fixed. |

These bounds concern stored-state motion in exact arithmetic. The limit can depend on the early data and can have high loss. Large late proposals can all be rejected, even when they would improve retrieval. The Python training loop has its own commit policy. Applying these guarantees to that loop requires implementing the rule in every write path and accounting for the state representation and arithmetic.
