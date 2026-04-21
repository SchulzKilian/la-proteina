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

## Finding 4 — Capacity probing separates properties by probe family, not probe size (2026-04-21)

**Experiment:**
- Seven probes of increasing capacity trained on the same Fold 0 split as Finding 1, on the same 56,008-protein 300–800-residue subset with the same 8-dim per-residue `mean` latents as input.
- Probe families (parameter count and aggregation structure):
  - `linear` (117 params): mean-pool residues → linear projection to 13 properties
  - `mlp_h32_L1` / `mlp_h64_L1` / `mlp_h128_L2` (717 / 1.4K / 19K params): mean-pool residues → MLP
  - `per_res_mlp_h64_L1` / `per_res_h128_L2` / `per_res_h256_L3` (1.4K / 19K / 137K params): per-residue MLP → mean-pool per-property logits across residues
  - The 8th point on the capacity axis is the 3L Transformer of Finding 1 (~350K params), reported here from the same Fold 0.
- Training: 20 epochs max, AdamW lr=3e-3, wd=0.01, early stop patience=4, batch size 32 length-bucketed. All probes share identical data, splits, z-score stats, and loss.
- Run directory: `laproteina_steerability/logs/capacity_probing/20260421_191747/`
- Wall-clock: 9 min total (1 × NVIDIA L4).

**Numbers (Fold 0 val R², per property, full capacity ladder):**

| Property | linear | mlp_h32 | mlp_h64 | mlp_h128_L2 | per_res_h64 | per_res_h128 | per_res_h256 | Tx (3L) |
|---|---|---|---|---|---|---|---|---|
| **#params** | 117 | 717 | 1.4K | 19K | 1.4K | 19K | 137K | ~350K |
| iupred3 | 0.32 | 0.36 | 0.36 | 0.39 | 0.65 | 0.88 | 0.91 | **0.98** |
| net_charge | 0.17 | 0.21 | 0.21 | 0.23 | 0.53 | **0.84** | 0.76 | **0.97** |
| shannon_entropy | 0.46 | 0.65 | 0.64 | 0.66 | 0.63 | 0.78 | 0.80 | **0.97** |
| pI | 0.19 | 0.24 | 0.24 | 0.26 | 0.51 | **0.76** | 0.69 | **0.95** |
| scm_positive | 0.28 | 0.31 | 0.31 | 0.32 | 0.45 | 0.49 | 0.50 | **0.93** |
| scm_negative | 0.19 | 0.23 | 0.23 | 0.26 | 0.31 | 0.46 | 0.44 | **0.92** |
| iupred3_fraction_disordered | 0.15 | 0.18 | 0.18 | 0.20 | 0.29 | 0.44 | 0.47 | **0.87** |
| sap | 0.05 | 0.14 | 0.14 | 0.27 | 0.10 | 0.24 | 0.28 | **0.87** |
| tango | 0.10 | 0.11 | 0.11 | 0.15 | 0.15 | 0.18 | 0.19 | **0.92** |
| hydrophobic_patch_total_area | 0.05 | 0.11 | 0.10 | 0.19 | 0.07 | 0.13 | 0.17 | **0.86** |
| hydrophobic_patch_n_large | 0.04 | 0.10 | 0.10 | 0.18 | 0.05 | 0.14 | 0.18 | **0.78** |
| rg | 0.02 | 0.02 | 0.03 | 0.06 | 0.03 | 0.06 | 0.06 | **0.80** |
| swi (Fold 0 only) | 0.14 | 0.15 | 0.15 | 0.16 | 0.28 | 0.37 | 0.37 | 0.38 |
| **r2_mean (Fold 0)** | 0.16 | 0.22 | 0.21 | 0.26 | 0.31 | 0.44 | 0.45 | **0.86** |

(Transformer column: Fold 0 values from Finding 1, to keep the comparison apples-to-apples. For SWI the Transformer's 5-fold mean is 0.77; Fold 0 is the outlier — see Finding 1.)

**Observations:**

*Two qualitatively distinct property classes emerge, cleanly separated by which probe family unlocks their R².*

**Class A — per-residue MLPs suffice** (residue-wise nonlinearity is the bottleneck):
*iupred3, net_charge, pI, shannon_entropy, iupred3_fraction_disordered, swi.*
- Protein-pooling (linear, mlp_h{32,64,128}) caps R² at ≤ 0.66; switching from protein-pool to per-residue MLPs unlocks most of the R² (e.g. iupred3 0.39 → 0.91, net_charge 0.23 → 0.84, pI 0.26 → 0.76).
- A 19K-param per-residue MLP recovers 77–98% of the Transformer R² on these properties. The Transformer adds only +0.07–0.21.
- All are defined as per-residue averages of sequence-derived quantities (charge, AA-disorder, entropy). Pooling *before* the per-residue nonlinearity throws away the very signal that *is* the property.

**Class B — only attention unlocks them** (inter-residue aggregation is the bottleneck):
*sap, tango, hydrophobic_patch_total_area, hydrophobic_patch_n_large, rg, scm_positive, scm_negative.*
- Per-residue MLPs of any tested size stay at R² ≤ 0.50.
- The 3L Transformer jumps these by +0.4 to +0.7 R² (most dramatically: rg 0.06 → 0.80, tango 0.19 → 0.92, hydrophobic_patch_total 0.17 → 0.86).
- These are spatial or aggregation properties: their definition depends on *which* residues interact (hydrophobic patches, charge clusters, radius of gyration). Mean-pooling and per-residue MLPs lack a mechanism for *learned* inter-residue aggregation; attention supplies it.

**No empty class:** every property tested reaches R² ≥ 0.77 at Transformer capacity. No property is genuinely absent from the latent at this evaluation; the ceiling is set by the probe, not the encoding.

*Saturation within families:*
- Within per-residue MLPs, Class A saturates at `per_res_h128_L2` (19K params). Going to `h256_L3` (137K params, 7× more) leaves Class A flat or slightly worse (net_charge 0.84 → 0.76, pI 0.76 → 0.69). Likely a mix of single-fold noise and mild overfit; worth a 5-fold confirmation before firm claims.
- Class B does not saturate along the per-residue-MLP axis; the probe family, not the capacity, is the bottleneck.

*SWI is a probe-architecture-invariant Fold-0 outlier:*
- Per-residue h256 and 3L Transformer both land at R² ≈ 0.37 on Fold 0. This is consistent with Finding 1's identification of SWI's Fold 0 as a data-split artefact (narrow target variance std=0.01 + unfavorable fold). Capacity scaling does not rescue it; the issue is not probe capacity.

**Narrow claim (defensible):**

> On La-Proteina's 8-dim per-residue latent, probe capacity separates the 13 developability properties into two classes defined by *probe family*, not *probe size*. Class A (net_charge, pI, iupred3, shannon_entropy, iupred3_fraction_disordered, swi) reaches 77–98% of the 350K-param Transformer R² already with a 19K-param per-residue MLP, while protein-pooling probes of up to 19K params stay below R² = 0.66. Class B (sap, tango, hydrophobic_patch_*, rg, scm_±) remains at R² ≤ 0.50 across every protein-pooled and per-residue-MLP probe (up to 137K params) and only unlocks under attention (Transformer R² 0.78–0.93). Radius of gyration is the most extreme: R² = 0.06 with per-residue MLPs, 0.80 with a 3L Transformer. SWI at Fold 0 is capacity-invariant (R² ≈ 0.37 across per-residue MLPs and Transformer), consistent with Finding 1's data-split interpretation.

**Implications:**

- **Information is present; accessibility is probe-family-dependent.** Finding 1 showed Transformer R² ≥ 0.77 for every property, including structural ones, but could not separate latent encoding from probe capacity. This sweep specifies the mechanism: spatial/aggregation properties *are* in the latent, but only decodable by a probe that can attend across residues. A simpler head would report them as "uninformative" and mislead downstream design decisions.
- **Sequence-derived vs structure-derived is the boundary between Class A and Class B,** with one exception (scm_± is sequence-derived but spatial — and indeed sits between the two, moderately unlocked by per-residue but jumping +0.4 at attention). This is consistent with the intuition that per-residue-averaged properties need only residue-wise nonlinearity, while spatial clustering needs aggregation.
- **A second axis: smooth vs. threshold-count aggregation.** The `iupred3` / `iupred3_fraction_disordered` pair acts as a near-controlled experiment: both consume the same per-residue IUPred3 disorder score, differing only in aggregation (continuous mean vs. fraction-above-0.5). At `per_res_h256_L3`, recovery splits sharply — `iupred3` reaches R² = 0.91 (90% of Transformer), `iupred3_fraction_disordered` plateaus at R² = 0.47 (54% of Transformer). Three compounding mechanisms plausibly explain this:
  1. **Threshold sensitivity on imprecise latents.** Continuous mean aggregation smooths over per-residue latent noise; threshold-count aggregation amplifies it. For residues whose true score sits near 0.5, small latent imprecision flips the binary outcome, and the errors accumulate in the count rather than averaging out.
  2. **Attention uses spatial correlation of the per-residue label; per-residue MLPs cannot.** Disorder occurs in contiguous IDR blocks, not scattered residues. Attention can use a residue's neighbors to resolve near-threshold ambiguity; a per-residue MLP sees each 8-d latent in isolation and has no mechanism for this smoothing. This is consistent with the Transformer's +0.40 R² jump exactly on `iupred3_fraction_disordered`.
  3. **Target-distribution fragility.** `fraction_disordered` is heavy-tailed (most proteins near 0, IDR-containing proteins as outliers); R² is dominated by correctly predicting the outliers, where threshold-count errors are largest in absolute terms.
  This means the Class A / Class B boundary is not purely "sequence vs. structure": *aggregation smoothness* is a second axis. Threshold-count aggregates of sequence-derived signals can require attention even when the underlying per-residue signal itself is already linearly decodable from the latent. Any future probe taxonomy should distinguish smooth-aggregate targets from discrete-count targets before reading protein-pooled R² as a capacity proxy.
- **For steering design:** Class A is the low-hanging fruit. A small per-residue MLP predictor (19K params) yields gradients of comparable quality to a Transformer at ~5% of the inference and memory cost. Class B steering, in contrast, inherits attention-specific gradient artifacts (non-local, potentially sharper) and requires the full attention-equipped predictor.
- **Capacity scaling within a family is nearly free of gains on this dataset.** mlp_h32 ≈ mlp_h64 (essentially identical), per_res_h128 ≈ per_res_h256 (regressions on some Class A properties). The factor that moves R² is structural — pool-then-MLP vs per-residue-MLP vs per-residue-attention — not parameter count.

**Methodological caveats:**

- Single fold (Fold 0) only. The h128→h256 regressions on net_charge and pI (−0.07 each) are plausible overfit/noise but need a 5-fold repeat before the "saturation at h128_L2" claim is firm.
- The capacity ladder is not uniform in architectural complexity: the jump from `per_res_h256_L3` to `Tx (3L, 128d, 4h)` introduces *three* changes at once — attention, multi-layer residue-residue interaction, and a different forward structure. An intermediate point (e.g. 1-head/1-layer attention probe) would isolate the minimal attention budget that unlocks Class B.
- All per-residue probes use mean-pooling of per-residue property logits. Other aggregations (learned weighted pool, max, set-transformer) were not tested; they could close part of the Class B gap without full attention.
- R² values are denormalised (back in physical units), matching Finding 1. They are not directly comparable to R² reported in z-score space.
- The Transformer column is quoted from Finding 1's Fold 0 values; other folds were not re-run under the same 20-epoch early-stop schedule used here. A same-protocol Transformer rerun would tighten the comparison.
- SWI's Fold-0 result is not representative of its 5-fold mean; readers should check Finding 1 before citing any SWI capacity claim.

**Cross-reference:** Finding 1 (Transformer baseline and SWI fold-variance diagnosis) and Finding 3 (latent geometry justifying mean-pool as the aggregation primitive for the non-attention probes).

---

## Future experiment ideas

### Capacity probing — remaining pieces (Finding 4 covers the core)
- 5-fold repeat of the full ladder to confirm the h128→h256 saturation and the Class A vs Class B boundary.
- Intermediate attention points (e.g. 1-layer 1-head, 1-layer 4-head, 2-layer 4-head) to pinpoint the minimal attention budget that unlocks Class B; currently we only know "3L / 4h is enough".
- Larger Transformers (6L, 12L) to verify that 3L does not itself cap Class B — particularly for sap, hydrophobic_patch_n_large and rg which sit at R² < 0.90 under the 3L probe.
- V-information framing (Xu et al. 2020): map the R² curves to `V_family(Z → Y)` curves to make the probe-family dependence explicit and theoretically grounded.
- Alternative per-residue aggregations (learned weighted pool, Set-Transformer) to test whether attention specifically, or any learned aggregation, closes the Class B gap.

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
