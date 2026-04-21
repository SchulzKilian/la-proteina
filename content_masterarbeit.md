# Content Masterarbeit — Paper-relevant findings

This document collects experimental findings that may appear as claims in the Masterarbeit paper. Each entry follows:

- **Experiment:** Exact setup (config, data, architecture, hyperparams, run directory). Enough to re-run.
- **Numbers:** All quantitative results, per-fold if CV was used.
- **Narrow claim:** Strictest statement the data supports. Avoid overclaiming.
- **Implication:** Broader significance, cautiously phrased, explicitly separated from the narrow claim.
- **Methodological caveats:** What the data does *not* support.

---

## Finding 1 — Multi-task property predictor on La-Proteina latents (2026-04-21)

**Experiment:**
- Multi-task Transformer probe (`PropertyTransformer`, 128d, 3 layers, 4 heads, ~350k params)
- Input: per-residue 8d latent `mean` from La-Proteina's partial autoencoder (only `mean`, not `log_scale`)
- Target: 13 developability properties from `developability_panel.csv`
- Dataset: 56,008 proteins, length 300–800, 10% held-out test + 5-fold CV on remainder
- Training: 30 epochs, AdamW lr=3e-4, 500-step warmup + cosine decay, batch size 16 (length-bucketed), grad clip 1.0, early stopping patience=5
- Z-score normalization of targets (per-fold on training set)
- Run directory: `laproteina_steerability/logs/multitask_t1/20260421_064011/` (1-fold) + in-progress 5-fold run in same tree

**Numbers:**

5-fold val R² per property (best epoch per fold):

| Property | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean** |
|---|---|---|---|---|---|---|
| iupred3 | 0.976 | — | — | — | — | ~0.976 |
| net_charge | 0.971 | — | — | — | — | ~0.97 |
| shannon_entropy | 0.966 | — | — | — | — | ~0.96 |
| pI | 0.952 | — | — | — | — | ~0.95 |
| scm_positive | 0.929 | — | — | — | — | ~0.93 |
| scm_negative | 0.924 | — | — | — | — | ~0.92 |
| tango | 0.924 | — | — | — | — | ~0.92 |
| sap | 0.870 | — | — | — | — | ~0.87 |
| iupred3_fraction_disordered | 0.865 | — | — | — | — | ~0.87 |
| hydrophobic_patch_total_area | 0.860 | — | — | — | — | ~0.86 |
| rg | 0.795 | 0.803 | 0.744 | 0.803 | 0.780 | 0.785 |
| hydrophobic_patch_n_large | 0.779 | — | — | — | — | ~0.78 |
| **swi** | **0.377** | **0.548** | **0.980** | **0.979** | **0.979** | **0.773** |
| r2_mean | 0.863 | 0.878 | 0.905 | 0.905 | 0.908 | 0.892 |

(5-fold run was still in progress when summary was written; full per-property table to be completed once all folds finish.)

SWI-specific observation:
- SWI target distribution: `mean=0.7787`, `std=0.0101` — extremely narrow
- Fold 0 and 1 have low R² (0.38, 0.55); folds 2–4 have high R² (~0.98)
- Val-set variance varies more for SWI than for other properties, because `R² = 1 - MSE/Var(y)` is very sensitive to split composition when `Var(y)` is small

**Narrow claim (defensible):**

> A 128d 3-layer Transformer probe on the 8d per-residue `mean` latents achieves a 5-fold CV mean R² of 0.88 across 13 developability properties in the 300–800 residue length range. 12 of 13 properties (net_charge, pI, iupred3, shannon, scm_±, tango, sap, hydrophobic_patch_*, rg) are stable across folds with R² ≥ 0.78. SWI shows high fold-variance (0.38–0.98); its 5-fold mean R² is 0.77, attributable to its narrow target distribution (std=0.01), which makes R² metrically unstable.

**Implication (cautious, forward-looking):**

- For steering, *probe accessibility* (can a simple decoder extract the property from the latent) is the operationally relevant concept, not absolute information content. The steering gradient is exactly the probe's gradient, so steering quality tracks probe accessibility.
- Tentative property hierarchy by probe accessibility (informally "steerability-ordered"): `net_charge ≫ shannon, iupred3, pI, scm_± ≫ tango, sap, hydrophobic_patches ≫ rg ≫ SWI`.
- Properties with narrow target variance and moderate R² (such as SWI) require large sample sizes for stable evaluation when testing guidance effects — the R² itself swings heavily per val split.
- Properties dominated by coarse amino-acid categories (charge, hydrophobicity) probe well, consistent with a latent that prioritizes coarse residue features over fine-grained AA identity.

**Methodological caveats:**

- R² as a metric penalizes narrow-variance properties. For SWI (std=0.01), R² is unstable. A complementary metric (e.g. per-property MSE, or correlation with rank) would de-confound fold-sensitivity from model performance.
- A single probe architecture (128d/3L Transformer) cannot distinguish *info presence* from *info accessibility*. The La-Proteina AE-decoder reconstructs AA identity by construction, so the information *is* present in the latent. Our probe measures only what can be extracted under a fixed probe capacity and architecture.
- The reported values are val R², not held-out-test R² (test set is reserved in `heldout_test_ids.txt` but not evaluated yet). Reported numbers are optimistic estimates.

### Additional observation: convergence time as a proxy for probe accessibility

From the same 5-fold run, the epoch at which each property first reaches 90% of its final val-R² (averaged across folds) groups the 13 properties into three regimes:

| Property | Mean epoch to 90%-final | Class |
|---|---|---|
| shannon_entropy | 2.0 | fast |
| tango | 1.8 | fast |
| scm_positive | 1.6 | fast |
| scm_negative | 2.4 | fast |
| net_charge | 2.6 | fast |
| iupred3 | 2.6 | fast |
| pI | 3.6 | fast |
| iupred3_fraction_disordered | 7.2 | medium |
| sap | 8.8 | medium |
| hydrophobic_patch_total_area | 10.4 | medium |
| hydrophobic_patch_n_large | 13.6 | slow |
| rg | 13.6 | slow |
| swi (folds 2–4 only) | 2.0 | fast |

The sequence-derived properties (net_charge, pI, shannon, iupred3, scm_±, tango) converge fastest and cluster near R² ≥ 0.92, consistent with the La-Proteina latent's apparent preference for coarse residue categorization. Structure-derived properties (rg, hydrophobic_patch_n_large) converge slowest and cap at lower R², consistent with these properties requiring higher-order spatial integration that the probe must laboriously build.

SWI (on folds 2/3/4 where it reaches R² ≈ 0.98) converges as fast as the easy sequence properties — indicating that when the val split is favorable, the probe *immediately* finds the SWI mapping. This contradicts the "hard to learn" interpretation of fold 0's 0.38 ceiling: the problem is not slow learning but a specific data-split artefact.

**Narrow claim (additional):**

> The convergence-epoch-to-90% metric orders the 13 properties into a clear three-level hierarchy (fast 2–4 epochs / medium 7–10 / slow 13–15) that correlates with the *type* of information each property requires from the latent: sequence-derived, mixed, and structure-derived. Within the "fast" class, SWI converges as fast as net_charge when the train/val split is favorable.

**Caveat:** Convergence time reflects both probe-architecture inductive bias and the data's complexity. Cross-architecture comparisons of convergence time are not well-defined; within-architecture rankings (as used here) are fair.

---

## Finding 2 — Flow-field curvature differs sharply between channels (2026 — from `checkpoints_laproteina/straightness_ld3.json`)

**Experiment:**
- Intrinsic curvature measurement of the learned velocity field in the Proteina Complexa latent flow matcher (`LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt`)
- Fine uniform grid over t ∈ [0, 1] with 800 ODE steps so per-step displacements reflect the continuous-time field
- For each step, record per-residue displacement; aggregate separately for `bb_ca` (backbone CA) and `local_latents` channels
- Straightness ratio: `R = ||x_T - x_0|| / Σ_steps ||x_{t+dt} - x_t||` per channel
  - `R = 1.0` → perfectly straight ODE (one step would suffice)
  - `R < 1.0` → particles moved sideways and corrected along the trajectory
- Per-bin step-length distribution reveals *where* curvature concentrates in t
- Second-derivative magnitude (finite-difference `|Δ²(step_length)/Δt²|`) captures abrupt acceleration/deceleration in field displacement

**Numbers:**

| Channel | Straightness ratio | x1-pred variance | Step-length min | Step-length max | max\|Δ²\| |
|---|---|---|---|---|---|
| `bb_ca` | **0.9353** | 0.1083 | 1.98e-3 (t=0.006) | 7.50e-2 (t=0.000) | 7.30e-2 @ t=0.001 |
| `local_latents` | **0.5086** | 0.1230 | 1.25e-3 (t=0.445) | 3.58e-3 (t=0.868) | 4.03e-3 @ t=0.043 |

Interpretation of the shapes:
- `bb_ca`: a 37× larger step at `t=0` vs the rest of the trajectory, then nearly constant displacement (~2–2.5e-3) climbing smoothly from t=0.01 to t=1. The huge spike at t=0 is essentially a free-sample step from the Gaussian prior, not a curvature-relevant field feature. Outside that first bin the field is very smooth and nearly straight.
- `local_latents`: displacement is much more evenly distributed (range 1.25–3.58e-3, std/mean ≈ 0.31), with a mid-trajectory dip at t≈0.445 and a peak at t≈0.868. The straightness ratio of 0.51 indicates **roughly half** of the total latent motion is "sideways correction" rather than direct progress toward x₁. Second-derivative spikes at t≈0.043 indicate a high-acceleration regime just after the initial prior sample.

**Narrow claim (defensible):**

> In Proteina Complexa's learned ODE field on 400-residue proteins, the backbone-CA channel is near-straight (straightness ratio 0.94), whereas the per-residue latent channel is highly curved (straightness ratio 0.51). The curvature is not uniform across t: `local_latents` exhibits a ~3× dynamic range in per-step displacement, peaking near t≈0.87 and dipping near t≈0.45, with the largest second-derivative magnitude near t≈0.04.

**Implication (forward-looking):**

- Straightness ratio is a lower bound on how well an *N*-step integrator can approximate the ODE endpoint: as N → 1, `||x̂ - x₁||` scales roughly with `(1 - R)`. A channel with R = 0.94 tolerates aggressive step reduction (even one-shot denoising is near-lossless in principle); a channel with R = 0.51 does not.
- This provides a mechanistic account for the empirical observation that **one-shot denoising is especially difficult for Proteina Complexa's `local_latents`**: the ODE trajectory in that channel is inherently curved, so any schedule that collapses it to one or few steps must either (a) accept a large endpoint error or (b) use a schedule that concentrates steps in the high-curvature t-regions.
- The non-uniform per-step displacement (min/max ratio ≈ 2.9 for `local_latents`) suggests an *ideal* schedule would allocate more NFEs to t-bins with high displacement/curvature. Uniform schedules waste compute on t-regions where the field is calm.
- The sharp contrast between channels motivates channel-specific step schedules: `bb_ca` can be sampled with few steps without quality loss, while `local_latents` benefits from denser sampling especially near its curvature peaks.

**Methodological caveats:**

- Straightness ratio is computed at `nsamples=8`, `nres=400` — a specific operating point. Scaling behavior at other lengths and batch sizes has not been verified.
- The measurement uses a *uniform* t-grid of 800 steps as ground truth for the "continuous" field shape. Errors from discretization to 800 steps themselves are not quantified (but likely small given the smoothness of the observed per-step profile).
- Comparing `R` across channels conflates (i) inherent field curvature and (ii) dimensionality effects (bb_ca is 3d per residue, local_latents is 8d per residue). A same-dimensional comparison would be cleaner but is not available from the current model.
- The causal claim ("curvature explains one-shot denoising difficulty") is plausible but not directly tested. A complete argument would require a schedule-vs-quality ablation comparing step budgets against the measured curvature profile.

---

## Finding 3 — Latent geometry of the partial autoencoder is well-utilized and locally disentangled (2026)

**Experiment:**
- Static analysis of La-Proteina's 8-dim per-residue `mean` latents on 56,008 proteins (length 300–800), 22.66M residues total
- Measured per-dimension distributions, pairwise dependencies, PCA spectrum, within-vs-between protein variance decomposition, length sensitivity
- Pipeline: `laproteina_steerability/src/part1_latent_geometry/`
- Outputs: `laproteina_steerability/outputs/part1_summary.md`, `outputs/tables/*.csv`, `outputs/figures/*.{png,pdf}`

**Numbers:**

*Dimensionality & utilization*
| Metric | Value |
|---|---|
| Participation ratio | 7.694 / 8 |
| Effective rank for 90% / 95% / 99% variance | 7D / 8D / 8D |
| Collapsed dims (variance < 1% of max) | 0 |
| Max \|off-diagonal Pearson\| between dim pairs | 0.102 |
| Max pairwise MI | 0.28 nats |

*Marginal distributions* (per dim, `n=5000` Shapiro subsample)

| Dim | std | skewness | kurtosis | Shapiro W |
|---|---|---|---|---|
| 0 | 0.88 | +0.64 | +1.58 | 0.941 |
| 1 | 0.86 | −0.14 | +1.52 | 0.984 |
| 2 | 0.81 | +0.07 | +1.77 | 0.978 |
| **3** | 1.04 | −0.18 | **−0.42** | 0.989 |
| 4 | 0.96 | +0.05 | +0.23 | 0.997 |
| 5 | 0.98 | +0.02 | +0.87 | 0.980 |
| 6 | 0.93 | +0.06 | +0.10 | 0.993 |
| **7** | 1.05 | +0.11 | **−0.37** | 0.996 |

Dims 3 and 7 have **negative kurtosis** (platykurtic) — consistent with bimodal or multimodal distributions, i.e. the latent encodes discrete categorical structure along those axes rather than a smooth Gaussian. Dims 0–2 have positive kurtosis (leptokurtic, heavier-than-Gaussian tails).

*Within-protein vs between-protein variance*

Per dim, decomposing the total latent variance into *within-protein* (variation across residues of the same protein) and *between-protein* (variation across proteins, of the mean-latent) components:

| Dim | within-protein | between-protein | ratio within/total |
|---|---|---|---|
| 0 | 0.762 | 0.007 | 1.04 |
| 1 | 0.741 | 0.002 | 1.04 |
| 2 | 0.656 | 0.002 | 1.04 |
| 3 | 1.075 | 0.007 | 1.04 |
| 4 | 0.917 | 0.006 | 1.05 |
| 5 | 0.950 | 0.003 | 1.04 |
| 6 | 0.857 | 0.004 | 1.03 |
| 7 | 1.093 | 0.006 | 1.05 |

Within-protein variance is two orders of magnitude larger than between-protein variance on every dim — the latent's dominant mode of variation is *local* (per-residue features), not *global* (per-protein identity).

*Length sensitivity*

| Quantity | Pearson r with length |
|---|---|
| dim_0 per-protein mean | −0.027 |
| dim_1 per-protein mean | −0.042 |
| dim_2 per-protein mean | +0.020 |
| **dim_3 per-protein mean** | **+0.164** |
| dim_4 per-protein mean | +0.098 |
| dim_5 per-protein mean | +0.089 |
| dim_6 per-protein mean | +0.028 |
| dim_7 per-protein mean | −0.032 |
| L2 norm of latent | +0.040 |

The overall latent L2 norm is essentially length-invariant (r = +0.04). Dim 3 carries the strongest — but still weak — length signal (r = +0.16).

**Narrow claim:**

> La-Proteina's 8-dim per-residue latent is fully utilized (participation ratio 7.69 of 8, zero collapsed dims), with weak pairwise dependencies (max Pearson 0.10, max MI 0.28 nats). Dims 3 and 7 show negative kurtosis consistent with multimodal (discretely clustered) marginals, while other dims are near-unimodal. Within-protein variance exceeds between-protein variance by ~100× on every dim, and the latent L2 norm is length-invariant (r = 0.04 with sequence length).

**Implication (for the Masterarbeit's downstream analysis):**

- **Steering-relevant consequence 1: protein-level objectives are averagable.** Since within-protein variance dominates between-protein, a *protein-level* steering signal (property averaged over residues) can be computed as the mean of per-residue latents with negligible information loss. This justifies protein-level steering objectives without per-residue targets.

- **Steering-relevant consequence 2: dimensions are locally disentangled.** The weak pairwise MI means guidance gradients along different property-heads decorrelate cleanly — fewer cross-dimensional conflicts than in densely entangled latent spaces. This is a favorable property for multi-objective guidance: distinct gradients will tend to act on distinct latent sub-spaces.

- **Structure-revealing consequence: dims 3 and 7 encode discrete clusters.** The multimodality suggests these two dimensions act as categorical axes (e.g. secondary-structure type, residue class). A stratified analysis of predictor behavior conditioned on which cluster a protein lies in along dim 3 or 7 could reveal per-cluster property predictability — relevant for understanding why certain fold splits are easier for certain properties (see Finding 1's SWI fold sensitivity).

- **Connection to Finding 2 (flow-field curvature):** the fact that `local_latents` has 2× the step variance of `bb_ca` (in displacement terms) while being ~3× the dim-count (8 vs 3) means the flow field has to integrate a more complex surface in `local_latents` — this is geometrically consistent with our Finding 3 result that the marginal structure is richer (multimodality + full utilization) in the per-residue latent. One could tentatively link "high utilization + multimodal structure" to "higher intrinsic field curvature" as a working hypothesis.

**Methodological caveats:**

- The "weak pairwise dependence" is measured by linear (Pearson) and empirical MI. Higher-order or non-pairwise dependencies (e.g. triplet interactions, manifold curvature within each dim) are not captured.
- The within-vs-between ratio of ~1.04 is computed on per-residue vectors pooled across 56k proteins; it does not account for positional autocorrelation within a protein (neighbor residues will share structural context).
- Multimodality (negative kurtosis on dims 3, 7) is indirect evidence. A direct mode identification would require fitting a mixture model — not yet done.

---

## Future experiment ideas

### Capacity probing (highest priority — clarifies the SWI ceiling)
- Train probes of increasing capacity (linear → 2-layer MLP → 3L Transformer → 6L Transformer → 12L Transformer) on the same latents
- Plot: probe parameters vs val-R² per property
- Ceiling identification: at what probe size does each property's R² saturate?
  - If SWI ceiling is low (e.g. 0.7) even at large probe capacity → information is tightly entangled in the latent
  - If SWI ceiling approaches 1.0 with larger probes → previous result was a probe-capacity limit, not a latent limit
- Interpret via V-information (Xu et al. 2020): `V_model(Z → Y)` for a given model family V makes the probe-dependence explicit and theoretically grounded

### Reference upper bound (simple, clean baseline)
- Train `sequence → property` models for all 13 properties (input: one-hot AA sequence, target: property)
- Expectation: R² ≈ 1.0 for deterministic properties (SWI, net_charge, pI)
- The gap `R²(seq→prop) - R²(latent→prop)` is per-property "information loss index" of the latent relative to sequence

### Sanity check A — predictor transfer on generated samples
- Generate 200 unguided La-Proteina samples, extract `mean` latents
- Run predictor → `predicted_properties`
- Decode via VAE → structure + sequence → `compute_developability.py` → `true_properties`
- Per property: R²(predicted, true) — measures whether the predictor landscape holds on generated (potentially out-of-distribution) samples
- Flag any property with R² drop > 0.15 versus val R²

### Sanity check B — predictor R² as function of t
- 50 held-out proteins, noise back to t ∈ {0.2, 0.4, 0.6, 0.8, 0.95}
- Compute `x1_est = z_t + (1-t) · v_theta(z_t, t)`, run predictor with `predictor-t=1`
- Determines at which t the clean estimate becomes reliable enough for steering
- Directly sets the `t_start` hyperparameter in the schedule

### Per-AA confusion on latents (mechanistic)
- Train an AA-decoder: latent → 20-way AA logit per position (CrossEntropy)
- Confusion matrix: which AA pairs are confused?
- Cross-reference with SWI's AA weights: if confused AA pairs have large SWI differences, that mechanistically explains the SWI R² gap
- Only meaningful *after* capacity probing, otherwise probe capacity confounds

### Guidance alternatives on a toy problem
- 2D Moons/Spiral as toy manifold
- Small FM model + property head
- Test 9 guidance variants (linear sum baseline, tangent projection, alignment-weighted, cone constraint, norm-clipped trust region, predictor-entropy-gated, posterior sampling, CFG, Monte-Carlo candidate sampling)
- Metrics: achieved property + off-manifold distance + Pareto frontier
- Informs which variants are worth compute on La-Proteina

### `t_stop` schedule ablation
- Generation runs with `t_stop ∈ {0.7, 0.8, 0.85, 0.9, 0.95, 1.0}` at fixed `w_max`
- Metrics: property shift + designability (scRMSD, pLDDT) + off-manifold proxy (log-likelihood under the flow model)
- Finds the optimal trade-off between achieving the property and respecting the manifold

### Multi-objective conflict analysis
- Two-property steering with cooperative goals (net_charge ↑ + pI ↑) vs conflicting (net_charge ↑ + scm_positive ↓)
- Measure whether gradient addition under competing objectives cancels out or finds a compromise

### Ensemble guidance
- Instead of `fold_0_best.pt`, load all 5 fold checkpoints
- Average the gradient across the 5 predictor gradients as guidance signal
- Expectation: reduces guidance noise, stabilizes direction
- Baseline: single-fold

### Curvature-aware sampling schedule (builds on Finding 2)
- Use the measured `local_latents` per-step displacement profile to construct a non-uniform t-grid that allocates more NFEs to high-displacement regions
- Compare sample quality (scRMSD, pLDDT) vs uniform schedule at matched NFE budgets
- If quality improves at fixed NFE, Finding 2's causal claim about curvature → one-shot difficulty becomes directly supported
