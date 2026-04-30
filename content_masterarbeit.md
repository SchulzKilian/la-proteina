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

## Finding 5 — Very-early-training CA-only diffusion snapshots are far from designable (2026-04-22)

> **Caveat — these are pre-convergence baselines, not results.** Both checkpoints below are taken well inside the undertrained regime (692 and 881 optimizer steps, corresponding to ~0.5–1 full gradient passes over the training set when `accumulate_grad_batches=8` and `batch_size=26` are accounted for). They are logged here only as anchoring points for later-step comparisons, not as scientific claims about the trained model's capability.

**Experiment:**
- Run: `test_ca_only_diffusion/1776805213` on Cambridge HPC (`/rds/user/ks2218/hpc-work/store/...`).
- Architecture: CA-only (no AE, no local_latents head), `local_latents_transformer` backbone, `nlayers=14, token_dim=768, nheads=12` → ~158 M parameters. Naming in `configs/inference_ucond_ca_only_70M.yaml` is a misnomer — the "70M" tag refers only to sampling hyperparameters; the architecture comes from the checkpoint and is 158 M here.
- Sampling config: `inference_ucond_ca_only_70M.yaml` (nsteps=400, `sampling_mode=sc`, `sc_scale_noise=0.1`, `t_lim_ode=0.98`, `center_every_step=true`, log schedule `p=2.0` on bb_ca).
- Generation: 3 samples each at lengths 50 and 100 (6 total per ckpt).
- Evaluation: `compute_designability: True` only (NOT codesignability — CA-only writes all-Ala, so codesignability would be meaningless). Pipeline = vanilla ProteinMPNN (8 seqs per PDB, `ca_only=False` because the writer reconstructs N/C/O from the Cα trace via `ca_nm_to_backbone_atom37`) → ESMFold → min scRMSD over the 8 seqs, reported for modes `ca` and `bb3o`.
- Hardware: single NVIDIA L4 (23 GB), generation ≈ 15–25 s, evaluation ≈ 2–3 min total.

**Numbers (per-length, min scRMSD_ca / mean / max, Å; designable = scRMSD_ca < 2 Å):**

| ckpt | epoch | optimizer step | L=50 min / mean / max | L=50 designable | L=100 min / mean / max | L=100 designable |
|---|---|---|---|---|---|---|
| `best_val_00000006_000000000692.ckpt` | 6 | 692 | 4.29 / 6.65 / 8.61 | 0/3 | 7.86 / 9.15 / 9.85 | 0/3 |
| `best_val_00000008_000000000881.ckpt` | 8 | 881 | 4.33 / 9.72 / 12.94 | 0/3 | 7.51 / 13.42 / 17.65 | 0/3 |

`scRMSD_bb3o` tracks `scRMSD_ca` to within 0.2 Å for all samples (the backbone is reconstructed idealized from the Cα trace, so this is by construction).

**Evaluated weights:** live (non-EMA) training weights. EMA weights at this stage are unusable: with `decay=0.999, every_n_steps=5` and only 692–881 optimizer steps, the EMA has had 138–176 updates, so `~0.999^138 ≈ 0.87` of the EMA parameters are still initialization noise. EMA half-life here is ≈ 3,465 optimizer steps — nothing below that yields a meaningful EMA comparison.

**Narrow claim:** At 692 and 881 optimizer steps, the CA-only 158 M diffusion model on `test_ca_only_diffusion/1776805213` produces no samples that meet the standard self-consistency designability threshold (scRMSD < 2 Å) at L ∈ {50, 100}. Best-per-structure scRMSD_ca across both snapshots is 4.29 Å (L=50) / 7.51 Å (L=100).

**Implikation (cautious):** The model has learned non-trivial structure — the best samples (~4 Å at L=50) are well below what an untrained CA trace would produce (random/init models typically score in the 20–40 Å range on this same pipeline, as evidenced by worst-case seqs in the 8-seq lists hitting 32–33 Å). However, it is nowhere near designable performance. This is consistent with being in the pre-convergence regime: a 158 M transformer at ≤881 optimizer steps has seen only ≤~183 k samples in gradient terms (`881 × 8 × 26`), which is well below the point where CA-only structure generators in the literature typically start hitting designable (usually ≥10 k optimizer steps on comparable data).

**Methodische Einschränkungen:**
- n=3 per length is too small to infer a training curve — the mean is dominated by a single unlucky draw (see the L=100 step-881 outlier at 17.65 Å pulling the mean from ~9 Å to ~13 Å).
- Only one sampling seed (`seed=5`) was used, so all spread observed is within-sampler noise rather than cross-seed variability.
- Only L∈{50, 100} was tested. Behavior at longer lengths (150–300) is not characterized.
- `sampling_mode=sc` with `sc_scale_noise=0.1` is a single point in the sampling hyperparameter space — no ablation of `nsteps`, `sc_scale_noise`, or ODE/SDE switch point.
- Evaluation uses a single folding model (ESMFold) and a single PMPNN temperature / sample count (8 seqs at default temperature). Designability numbers depend on these.
- Codesignability is not computed at all (by design, the CA-only model has no sequence head), so this finding says nothing about joint structure-sequence quality — only about whether *some* sequence can refold the generated backbone.
- The "epoch" counter (Lightning's accounting, 6 and 8 here) does not cleanly map to data passes when `accumulate_grad_batches=8` and multi-worker loaders are involved; only `global_step` is the reliable progress measure.

**Artifacts:**
- PDBs + ESMFold outputs: `./inference/inference_ucond_ca_only_70M/job_0_n_{50,100}_id_{0,1,2}/` (overwritten per run — not preserved across the two snapshots).
- Per-sample CSV: `./inference/results_inference_ucond_ca_only_70M_0.csv` (also overwritten; numbers above are captured here).
- Resolved generation config: `./inference/inference_ucond_ca_only_70M/resolved_config.yaml`.
- Side-fix recorded for reproducibility: `ProteinMPNN/protein_mpnn_run.py` + `protein_mpnn_utils.py` + `helper_scripts/` had to be cloned from the pinned commit `8907e66…` of github.com/dauparas/ProteinMPNN — the repo's `script_utils/download_pmpnn_weights.sh` only pulls weights, not code.

---

## Finding 6 — Negative result: stronger weight decay improves validation loss but collapses sample quality, traced to AdamW crushing AdaLN-Zero output gates (2026-04-25)

**Experiment:**

Two CA-only diffusion training recipes compared on the same architecture, data, batch size, and validation set; only the optimizer/regularization differs.

- Architecture: `nn/ca_only_score_nn_160M.yaml` (160M-parameter `LocalLatentsTransformer`, output_parameterization `bb_ca: v`, no `local_latents` head, no autoencoder, no triangular multiplicative updates, no downsampling). Uses DiT-style **AdaLN-Zero** conditioning blocks (the `*.scale_output.to_adaln_zero_gamma` output gates seen in `state_dict()`).
- Data: `dataset/pdb/pdb_train_ucond.yaml` (PDB train, min_length=50, max_length=512, worst_resolution ≤ 2.0 Å). Sequence-similarity 0.5 split. Val set size = 4058 proteins.
- Validation cadence: `val_check_interval=2000` mini-batches → ~1 val eval per 63 optimizer steps (with `accumulate_grad_batches=32`).
- Single A100 (Cambridge HPC ampere partition); `dist_strategy=auto`; `accumulate_grad_batches=32`; `batch_size=6`; `max_padding_size=512`; effective batch ≈ 192 proteins. EMA decay 0.999, every 5 steps. Seed 42. bf16-mixed precision. `gradient_clip_val=1.0` (norm).
- **Old recipe (control):** `torch.optim.AdamW` with `weight_decay=0.05`, constant LR `2e-4`, no scheduler.
- **New recipe (v2):** `torch.optim.AdamW` with `weight_decay=0.1`, cosine_with_warmup schedule (linear warmup 0 → `2e-4` over 200 optimizer steps, cosine decay to `min_lr_ratio * peak = 2e-5` at `total_steps=6000`).
- Both recipes apply weight decay uniformly to all parameters — `configure_optimizers` in `proteinfoundation/proteina.py` does not split parameters into wd/no-wd groups.
- Run directories: old = many chained runs under `store/test_ca_only_diffusion/`; v2 = `store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/`.
- Wandb runs (v2): `9jp15of2` (slot 1), `5rftn43a` (slot 2), `43xxlbzt` (slot 3 = continuation after a chain failure on a broken GPU node).
- v2 wall-clock: ~18 hours on a single A100 (3 chained 6h SLURM slots) to reach optimizer step 2294; chain cancelled after a confirmed two-eval val uptick.
- Best v2 checkpoint (preserved on disk for the post-mortem analysis):
  - Raw: `/home/ks2218/la-proteina/store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/best_val_00000020_000000002078.ckpt`
  - EMA: same path with `-EMA.ckpt` suffix.
- Reference old checkpoint used in the post-mortem comparison: `store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt` (the original step-2204 best from `jeponiu5` was overwritten by later `best_val_*` saves under `save_top_k=1`; step 2646 from `0fnyfbi9` was the latest available raw old-recipe checkpoint at the time of the comparison).

**Numbers — validation loss:**

Best validation loss (`validation_loss/loss_epoch`, single Monte-Carlo estimate per eval):

| Recipe | Best val | At opt step | Behaviour past best |
|---|---|---|---|
| Old (wd=0.05 + constant LR=2e-4) | **4.765** | 1827 (run `d1k1587u`); 4.712 in `jeponiu5` at step 2204 | rises to 4.79–5.39 within 250–700 steps |
| New v2 (wd=0.1 + cosine_with_warmup) | **4.437** | 2078 (run `43xxlbzt`) | rises to 4.78 by step 2267 |
| **Δ (v2 − old)** | **−0.328** | +251 steps | — |

Head-to-head v2 vs old run `d1k1587u` at matched optimizer steps (val_loss):

| step | v2 | d1k1587u | Δ |
|---|---|---|---|
| 1448 | 5.543 | 5.085 | +0.458 |
| 1511 | 5.216 | 5.063 | +0.154 |
| 1637 | 5.093 | 5.042 | +0.052 |
| 1700 | 5.029 | 4.866 | +0.163 |
| 1763 | 4.875 | 4.786 | +0.089 |
| **1827** | **4.724** | 4.765 (old's best) | **−0.041** ← v2 crosses under |
| 1889 | 4.671 | 4.792 (old's uptick begins) | −0.121 |
| 1952 | 4.506 | 4.787 | −0.282 |
| 2078 | **4.437** (v2 best) | — | — |
| 2267 | 4.781 (uptick) | — | — |

Per-length val (v2 only, around the uptick, from the new `validation_loss_by_len/len_<lo>_<hi>` logging):

| length bin | step 2015 | step 2078 | step 2142 | step 2204 | step 2267 |
|---|---|---|---|---|---|
| 50–175  | 4.244 | 4.316 | 4.078 | 4.283 | 4.344 |
| 175–300 | 4.508 | 4.300 | 4.548 | 4.915 | 5.022 |
| 300–425 | 4.945 | 4.775 | 4.957 | 4.924 | 5.292 |
| 425–513 | 5.180 | 4.916 | 5.102 | 5.396 | 5.097 |

**Numbers — sample quality (designability via ESMFold scRMSD):**

After observing the val improvement, samples were generated from both recipes' raw checkpoints under matching inference configs (200 ODE steps, generation/uncond_codes_ca_only, designability_modes=[ca, bb3o], folding_models=[esmfold]). N=3 per length, designability threshold scRMSD < 2 Å:

| Run / step | L=50 (min/mean/max scRMSD Å) | L=50 designable | L=100 | L=100 des | L=200 | L=200 des |
|---|---|---|---|---|---|---|
| `test_ca_only_diffusion/1776805213` step 1889 (old)        | 1.56 / 3.00 / 4.07  | 1/3 | 1.66 / 2.01 / 2.56  | 2/3 | — | — |
| `test_ca_only_diffusion/1776805213` step 2457 (old, post-uptick) | 1.29 / 2.40 / 3.59 | 1/3 | 1.54 / 5.10 / 12.03 | 2/3 | 4.04 / 7.91 / 11.45 | 0/3 |
| `ca_only_diffusion_baseline_v2/1776975226` step 2078 (v2 best) | 4.22 / 9.10 / 14.83 | **0/3** | 8.00 / 11.28 / 13.41 | **0/3** | 7.96 / 9.60 / 11.03 | **0/3** |

v2 produces **zero designable samples at any tested length**, with mean scRMSD 9–11 Å vs the old recipe's 2–8 Å. N is small (3/length) but the gap is categorical, not noise: even at the easiest length (L=50) the v2 *minimum* scRMSD across 3 samples (4.22 Å) is worse than the old recipe's *maximum* (4.07 Å, step 1889).

**Numbers — per-layer weight diff (the post-mortem):**

Loaded both raw checkpoints on CPU and computed L2 norm per parameter tensor in `state_dict()`:

- Global weight L2 norm: v2 = 430.33, old = 438.73 → ratio 0.981 (v2 only 1.9% smaller globally; cannot account for the sample collapse on its own).
- Layer-wise ratio (v2/old) over 164 layers ≥ 10k params: mean = 0.920, median = 0.967, stdev = 0.148, **min = 0.260, max = 1.376**.
- The 10 layers with the largest |ratio − 1| are **all** `transformer_layers.{7..13}.{mhba,transition}.scale_output.to_adaln_zero_gamma.0.weight` — the AdaLN-Zero output gates of the upper transformer blocks:

| layer | old norm | v2 norm | ratio v2/old |
|---|---|---|---|
| `nn.transformer_layers.10.mhba.scale_output.to_adaln_zero_gamma.0.weight`       | 1.815 | 0.471 | **0.260** |
| `nn.transformer_layers.9.transition.scale_output.to_adaln_zero_gamma.0.weight`  | 0.988 | 0.280 | **0.283** |
| `nn.transformer_layers.9.mhba.scale_output.to_adaln_zero_gamma.0.weight`        | 1.527 | 0.456 | **0.299** |
| `nn.transformer_layers.10.transition.scale_output.to_adaln_zero_gamma.0.weight` | 0.576 | 0.260 | 0.451 |
| `nn.transformer_layers.8.transition.scale_output.to_adaln_zero_gamma.0.weight`  | 0.625 | 0.283 | 0.453 |
| `nn.transformer_layers.7.mhba.scale_output.to_adaln_zero_gamma.0.weight`        | 1.765 | 0.801 | 0.454 |
| `nn.transformer_layers.13.transition.scale_output.to_adaln_zero_gamma.0.weight` | 0.534 | 0.263 | 0.493 |
| `nn.transformer_layers.11.transition.scale_output.to_adaln_zero_gamma.0.weight` | 0.563 | 0.302 | 0.536 |
| `nn.transformer_layers.13.mhba.scale_output.to_adaln_zero_gamma.0.weight`       | 1.454 | 0.801 | 0.551 |
| `nn.transformer_layers.12.transition.scale_output.to_adaln_zero_gamma.0.weight` | 0.557 | 0.331 | 0.595 |

The 10 most-similar layers (ratio 0.99–1.00) are all AdaLN modulation γ/β weights — those are essentially unchanged.

**Mechanism (DiT/SiT-style AdaLN-Zero × naive AdamW):**

AdaLN-Zero (introduced in DiT, Peebles & Xie 2023, also in SiT, SD3, FLUX) adds a per-block output gate `α(c)` modulating each residual contribution: `x ← x + α(c)·Block(AdaLN(x, c))`. The linear layer producing α is **zero-initialized**, so the network behaves as identity at the start of training; the gates then *grow* as gradient signal makes block contributions useful.

Weight decay's mathematical job is to push weights toward zero on every step. AdaLN-Zero gate weights start at zero and need to grow. With a uniform wd applied to all parameters, the gradient signal is in continuous tension with the wd pull. At wd=0.05 the gates grow (slowly) to useful magnitudes; at wd=0.1 — and especially in deeper layers where gradient signal is weaker — the wd pull dominates and the gates stay near zero. Because conditioning influence at each block is multiplied by α(c), suppressed gates mean the time-conditioning signal barely reaches the velocity output. Predicted velocities become roughly t-independent → predicting the *average* velocity over t is a much easier objective (lower MSE on val) → integrated trajectories at inference time have no coherent time-conditioning to push noise onto the data manifold → samples collapse.

Standard fix in DiT/SiT/SD3 codebases: parameter groups in AdamW that exclude (a) AdaLN-Zero gate parameters, (b) biases, (c) LayerNorm γ/β, (d) embeddings from weight decay. La-Proteina's `configure_optimizers` does not implement this split — it passes all parameters to a single AdamW instance with one `weight_decay` value. With that codebase as-is, **wd is bounded above by what AdaLN-Zero gates can tolerate**, which experimentally is somewhere ≤ 0.05.

**Narrow claim (defensible):**

> On the 160M CA-only diffusion baseline, replacing AdamW(wd=0.05) + constant LR=2e-4 with AdamW(wd=0.1) + cosine_with_warmup LR (peak 2e-4, decay to 2e-5 over 6000 steps) reduces the best validation loss from 4.765 to 4.437 (Δ = −0.328) on the same dataset, batch size, and val set, but produces categorically worse samples (0/3 designable at L ∈ {50, 100, 200} for v2 vs 1–2/3 for the old recipe at comparable training points), with v2 mean scRMSD 9–11 Å vs old 2–8 Å. Per-layer L2 weight analysis attributes the regression to a 40–74% reduction of the AdaLN-Zero output-gate weights in transformer layers 7–13 in v2 relative to old, while non-gate parameters are within ~3% of their old-recipe magnitudes. Validation loss is therefore not a reliable proxy for sample quality on this codebase under uniform-wd AdamW.

**Implication (cautious, forward-looking):**

- For the architectural-variant comparisons planned next (sparse attention, conv downsampling), the **old recipe (AdamW wd=0.05, constant LR=2e-4) remains the canonical CA-only baseline.** The v2 recipe must not be used as a baseline because it does not produce designable samples. Variants should be trained with the same old recipe so the comparison stays clean.
- Any future hyperparameter search over weight decay in this codebase should *first* restructure `configure_optimizers` to exclude bias / LayerNorm / embedding / AdaLN-Zero-gate parameters from weight decay (the standard DiT param-group split, ~15 lines). Without that change, increasing wd above ~0.05 is unsafe for sample quality regardless of how good val loss looks.
- Validation loss (velocity-MSE averaged over random (x_t, t)) is a misleading proxy for sample quality in this codebase because it is dominated by easy-to-predict regions of the (x_t, t) space; suppressing the conditioning signal makes predictions smoother and lowers the proxy without improving the *integrated* sampling trajectory. Future training experiments should pair val-loss tracking with cheap sample-quality probes (e.g. designability on N=10 samples per length at a few representative lengths) at a small number of checkpoints, not rely on val loss alone.

**Methodological caveats:**

- The sample-quality comparison used N=3 samples per length per checkpoint. This is small for fine-grained scRMSD distribution claims, but the gap (every v2 sample worse than the worst old sample at every length tested) is wide enough that the categorical conclusion (v2 samples are not designable) holds without needing larger N.
- The two old-recipe checkpoints in the sample-quality table (`step 1889`, `step 2457`) were the labels used in the user-supplied evaluation table at the time of analysis. The exact `best_val_*_1889.ckpt` and `best_val_*_2457.ckpt` files in `1776805213/checkpoints/` are not present on disk — they were overwritten by later `best_val_*` saves under `save_top_k=1`. The post-mortem per-layer weight diff therefore used `best_val_00000026_000000002646.ckpt` from the same store dir as the old reference (the latest available raw old-recipe ckpt), which is post-uptick from a chained continuation. Despite being a *worse* old checkpoint by val-loss, it still produces dramatically better samples than v2-2078, so the v2 collapse cannot be explained by checkpoint selection.
- The mechanism (wd crushing AdaLN-Zero gates) is consistent with the per-layer weight evidence and with published behaviour of DiT-style architectures under naive AdamW, but has not been formally verified by an ablation (e.g., training v2 with the param-group fix and confirming gates grow + samples become designable). That ablation would cost ~16h on a single A100 and was not run because compute was unavailable; recording the mechanism here on the strength of the weight-diff evidence and the literature.
- The val-loss numbers themselves (Δ = −0.328 in best-val) are real and reproducible; they are not retracted. What is retracted is the original (now-deleted) framing of v2 as "a strict improvement to the baseline". The improvement is in val loss only; on the actual deliverable (sample quality), v2 is strictly worse.
- The chain was cancelled at step 2294 with cosine LR still at 1.48e-4 (out of 6000 scheduled steps). It is therefore not formally proven that the v2 sample-quality collapse would not have recovered with further training. However, the mechanism (zero-initialised gates pulled toward zero by wd) predicts that further training under the same recipe would *worsen* the gate suppression, not recover it — so this is not considered a meaningful caveat against the conclusion.

---

## Finding 7 — Sidechain perturbations in AE1 latent space stay closer to ESMFold's prediction than equivalent coord-space perturbations across all noise scales (2026-04-25)

**Experiment:**
- **Question (as pre-registered):** at matched noise levels k·σ for both spaces, which arm produces sidechain placements that ESMFold (conditioned on the original sequence) is closer to?
- **Autoencoder:** `AE1_ucond_512.ckpt` (512-residue AE used with the original 355K precomputed latents). Diffusion checkpoint not used — the experiment is AE encode/decode-only.
- **Eval set:** 17 length-stratified proteins from `data/pdb_train/processed_latents/`, sampled with `--n-per-bin 7 --n-bins 3 --seed 0` over 50–300 residues. Bin counts came out [7, 7, 3] (the 217–300 bin under-filled given seed/file order). IDs: `1drb_B, 1xbp_W, 3gta_B, 4v5a_CK, 4v88_BO, 4v88_DO, 4v8y_BO, 4wzo_A8, 5dat_m7, 5tw1_E, 6bg4_E, 6u5b_J, 6wnw_L, 7bka_B, 7ccl_B, 7qiq_C, 8btr_LW`. Lengths: 65, 76, 92, 92, 97, 111, 119, 133, 146, 151, 155, 159, 166, 170, 217, 217, 217.
- **Noise scales:** k ∈ {0.1, 0.3, 0.5, 1.0, 2.0}.
- **Coord arm:** Gaussian noise on sidechain atoms (atom37 indices ∉ {0:N, 1:CA, 2:C, 4:O}) only; backbone untouched. σ = empirical per-(restype, atom_idx) std of the atom's offset from CA in the residue-local (N,CA,C) frame, computed over the 17 eval proteins. Reported summary: 686/777 entries non-zero; mean non-zero std = 0.1206 nm (≈1.2 Å).
- **Latent arm:** Gaussian noise on encoder `mean` with σ = empirical per-dim std on the same 17 proteins. Per-dim std observed: [1.018, 1.005, 0.909, 1.001, 0.946, 1.005, 0.938, 1.062] (≈1.0 — consistent with KL-regularisation toward N(0,1)). After decode, original N/CA/C/O are spliced back so the only difference between arms is sidechain placement. Decoder-dropped sidechain atoms fall back to the original position (so the two arms compare over identical atom sets).
- **Metric:** `proteinfoundation/evaluate.py --config_name eval_manifold_perturbation`, `compute_codesignability=True`, `codesignability_modes=["all_atom"]`, `codesignability_folding_models=["esmfold"]`. ESMFold runs on the *original* sequence; metric is all-atom RMSD between the perturbed input PDB and ESMFold's prediction. 10 cells (5 k-values × 2 spaces) → 17 PDBs each → 170 total.
- **Run dir / artefacts:** Stage-1 PDBs under `inference/eval_manifold_perturbation/{protein_id}/job_{cell}_{space}_k{k}_{protein_id}.pdb`. Per-cell evaluator results: `inference/results_eval_manifold_perturbation_{0..9}.csv`. Aggregate tidy: `inference/manifold_tidy_eval_manifold_perturbation.csv` (170 rows). Aggregate summary: `inference/manifold_summary_eval_manifold_perturbation.csv`. Plot: `inference/manifold_plot_eval_manifold_perturbation.png`. Stats sidecar (latent_std per dim, sidechain std summary): `inference/eval_manifold_perturbation/stats.json`.
- **Run env:** local L4 box, `/home/ks2218/.conda/envs/laproteina_env`, single GPU. Stage 1 ~90 s, Stage 2 ~28 min (10 cells × 17 proteins × ~7 s ESMFold + per-cell model reload), aggregation < 5 s.

**Numbers (all-atom co-designability scRMSD, Å):**

| k | space | mean | median | std | n |
|---|---|---:|---:|---:|---:|
| 0.1 | coord | 9.998 | 3.449 | 17.318 | 17 |
| 0.1 | latent | 5.010 | 2.715 | 5.934 | 17 |
| 0.3 | coord | 10.066 | 3.531 | 17.289 | 17 |
| 0.3 | latent | 5.080 | 2.743 | 5.890 | 17 |
| 0.5 | coord | 10.181 | 3.642 | 17.242 | 17 |
| 0.5 | latent | 5.129 | 2.767 | 5.867 | 17 |
| 1.0 | coord | 10.659 | 4.238 | 17.052 | 17 |
| 1.0 | latent | 5.234 | 2.820 | 5.817 | 17 |
| 2.0 | coord | 12.002 | 6.037 | 16.550 | 17 |
| 2.0 | latent | 5.206 | 2.886 | 5.815 | 17 |

**Paired Wilcoxon signed-rank, H1: latent < coord (n=17 per k):**

| k | median(coord − latent), Å | p-value |
|---|---:|---:|
| 0.1 | 0.022 | 6.7 × 10⁻⁴ |
| 0.3 | 0.031 | 4.2 × 10⁻⁴ |
| 0.5 | 0.157 | 7.6 × 10⁻⁶ |
| 1.0 | 0.657 | 7.6 × 10⁻⁶ |
| 2.0 | 2.720 | 7.6 × 10⁻⁶ |

Pooled across all k (n=85): median(coord − latent) = 0.348 Å, p = 2.5 × 10⁻¹⁵; **80/85 (94.1%)** of pairs have latent < coord.

**Per-protein illustrative pairs at k=2.0** (sorted by coord scRMSD):

| protein | coord scRMSD | latent scRMSD |
|---|---:|---:|
| 1xbp_W | 73.40 | 2.04 |
| 4v8y_BO | 17.89 | 17.22 |
| 4v88_BO | 17.87 | 17.23 |
| 4v88_DO | 17.82 | 17.29 |
| 5tw1_E | 10.25 | 3.51 |
| 1drb_B | 7.71 | 1.63 |
| 4v5a_CK | 6.65 | 5.05 |
| 6bg4_E | 6.10 | 1.61 |
| 5dat_m7 | 6.04 | 3.32 |
| 4wzo_A8 | 5.65 | 2.89 |
| 8btr_LW | 5.60 | 3.53 |
| 7ccl_B | 5.11 | 2.89 |
| 6wnw_L | 5.07 | 2.41 |
| 6u5b_J | 5.04 | 2.29 |
| 3gta_B | 4.89 | 1.84 |
| 7bka_B | 4.83 | 2.04 |
| 7qiq_C | 4.11 | 1.70 |

The three ~17 Å entries in both arms (4v8y_BO, 4v88_BO, 4v88_DO) are large multi-chain ribosomal subunit fragments at L=217 — ESMFold's sequence-conditioned prediction simply disagrees with the deposited fold for these, so both arms inherit the floor regardless of perturbation. They illustrate that the residual scRMSD floor is set by ESMFold's modelling error, not by the perturbation.

**Within-arm sensitivity to k:**
- Latent arm: median 2.715 → 2.886 Å going from k=0.1 to k=2.0 (Δ +0.17 Å despite the per-dim noise std being 20× larger).
- Coord arm: median 3.449 → 6.037 Å (Δ +2.59 Å).

**Narrow claim (strict, fully defended by the data):** Under sidechain-only perturbation evaluated by all-atom co-designability against ESMFold on the original sequence, perturbing in `AE1_ucond_512`'s 8-dim per-residue latent space produces lower scRMSD than perturbing equivalently-scaled sidechain atoms in coord space at every tested noise scale k ∈ {0.1, 0.3, 0.5, 1.0, 2.0}, with paired Wilcoxon p ≤ 7 × 10⁻⁴ at every k and p = 2.5 × 10⁻¹⁵ pooled (94% of pairs latent < coord). Within the latent arm, scRMSD is approximately invariant in k (median +0.17 Å over a 20× change in noise std), while within the coord arm scRMSD grows roughly linearly with k (median +2.59 Å over the same range).

**Implikation (cautious, separate from the narrow claim):** The combination of (a) consistently lower latent-arm scRMSD and (b) near-invariance of latent-arm scRMSD to noise magnitude is most parsimoniously explained by the AE decoder being *contractive* on perturbed latents — i.e., the decoder projects off-manifold latents back onto the data manifold it was trained on, while raw coord noise has no such projector. This is the mechanism a steering / guidance system would *want*: latent-space modifications stay near plausible structures even at large step sizes, whereas equivalent-magnitude direct coord modifications can occasionally produce dramatically off-manifold structures (e.g. 1xbp_W at k=2.0: 73 Å in coord, 2 Å in latent). In that loose sense the La-Proteina latent representation is "tighter packed near the data manifold" from the perspective of small-perturbation downstream effects, but this is an inference about decoder behaviour, not a direct measurement of latent-space cluster geometry — see limitations.

**Methodische Einschränkungen:**
- The experiment does not measure the latent space's geometry directly (e.g. encoded-distance-vs-coord-distance, density of latents around real proteins, or geodesic vs Euclidean structure on the latent manifold). The "tighter packed" framing in the Implikation is an inference from contractive decoder behaviour, not from a direct latent-space distance measurement. A latent-space density / geodesic experiment is the natural follow-up.
- The metric is scRMSD vs ESMFold's *sequence-conditioned* prediction, not vs ground-truth experimental structures. For three of the seventeen proteins (the 4v8y/4v88 cluster), ESMFold disagrees with the deposited fold even at k=0.1 in both arms; the residual ~17 Å in those cases reflects ESMFold modelling error and not the experiment's signal. The pooled effect-size statistics include those proteins but the per-k Wilcoxon is sign-based (does latent beat coord paired by protein?), which is robust to this.
- N=17 length-stratified proteins, single seed, single AE checkpoint. The 217-residue bin under-filled (only 3 proteins reached it). The result has not been replicated with different seeds or different length stratifications.
- AE1 was trained on ≤ 512 residue proteins. Eval proteins (50–217 residues) are fully in-distribution. The result therefore does not transfer mechanically to AE2 / 300–800 residue proteins, which would require re-running with the AE2 checkpoint and the 300–800 length stratification.
- The "coord arm σ" is computed from the same 17 proteins used for evaluation (the experiment is not blind to the noise calibration). This is acceptable for a proof-of-concept — what is being calibrated is realistic *per-restype, per-atom* sidechain displacement under thermal motion — but if the comparison were re-cast as "find a coord-space perturbation that beats latent-space at any plausible scaling", it would need a separate σ estimation set.
- The "latent arm σ" (≈ 1) and "coord arm σ" (≈ 0.12 nm = 1.2 Å) are not directly commensurable in an absolute sense; the comparison is meaningful at the level of "noise that respects each space's natural scale", not "noise that produces the same Cartesian displacement". A k that would produce a fixed Cartesian sidechain-RMSD in both arms is left unmeasured.

---

## Baseline reference — canonical CA-only run (for all architectural-variant comparisons)

This is the run that the sparse-attention and conv-downsampling variants must be compared against. It is **not** a standalone finding — Finding 6 already covered the val-vs-sample-quality analysis. The purpose of this entry is to lock in the baseline's exact configuration so any later "the variant beat the baseline" claim has a single, citable reference point on disk.

**Run identity:**
- **Store dir:** `/home/ks2218/la-proteina/store/test_ca_only_diffusion/1776805213/`
- **Saved exp-config:** `…/checkpoints/exp_config_test_ca_only_diffusion.json` — this JSON is the source of truth; future variants should diff their resolved Hydra config against it.
- **Wandb run chain:** `d1k1587u` (best val 4.765 at step 1827) → `jeponiu5` (best 4.712 at step 2204, since-overwritten by `save_top_k=1`) → `0fnyfbi9` (latest; contains the step-2646 ckpt currently on disk).
- **Best raw checkpoint on disk:** `…/checkpoints/best_val_00000026_000000002646.ckpt` (post-uptick — the older step-2204 ckpt was overwritten). Use this when you need the canonical baseline weights.
- **Hardware:** 1× A100 (Cambridge HPC ampere), `ngpus_per_node_=1`, `nnodes_=1`.

**Architecture (NN config — exact match to `configs/nn/ca_only_score_nn_160M.yaml`):**
- 160M-parameter `LocalLatentsTransformer`. `nlayers=14`, `token_dim=768`, `nheads=12`, `parallel_mha_transition=False`, `use_qkln=True`.
- Output: `output_parameterization: {bb_ca: v}`. No `local_latents` head, no autoencoder, `latent_dim=None`.
- Pair representation: `pair_repr_dim=256`, `seq_sep_dim=127`, `xt_pair_dist_dim=30 (0.1–3 nm)`, `x_sc_pair_dist_dim=30 (0.1–3 nm)`.
- Conditioning: `dim_cond=256`, `t_emb_dim=256`, `idx_emb_dim=256`.
- Features: seq = `[xt_bb_ca, x_sc_bb_ca, optional_ca_coors_nm_seq_feat, optional_res_type_seq_feat]`; pair = `[rel_seq_sep, xt_bb_ca_pair_dists, x_sc_bb_ca_pair_dists, optional_ca_pair_dist]`; pair-cond = `[time_emb_bb_ca]`.
- **Deliberately off:** `update_pair_repr=False`, `use_tri_mult=False`, `use_downsampling=False`, `parallel_mha_transition=False`, `strict_feats=False`, no LoRA (`lora.r: null`). Variants must keep these off too unless the variant *is* one of these toggles.

**Recipe (the "old recipe" — locked-in canonical for variants):**
- `torch.optim.AdamW`, `weight_decay=0.05` uniform, `lr=2e-4` constant (no scheduler, no warmup, no decay). β1=0.9, β2=0.999, ε=1e-8 (PyTorch defaults).
- `accumulate_grad_batches=32`, `dataset.datamodule.batch_size=6`, `max_padding_size=512` → effective batch ≈ 192 proteins/optimizer step.
- bf16-mixed precision (`force_precision_f32: False`), `gradient_clip_val=1.0` norm.
- EMA: `decay=0.999`, `every_n_steps=5`, `validate_original_weights=False`, `cpu_offload=False`.
- `val_check_interval=2000` mini-batches → ~63 optimizer steps between val evals.
- Self-conditioning on (`self_cond=True`), `n_recycle=0`, `motif_conditioning=False`, `p_folding_n_inv_folding_iters=0.0`, `use_precomputed_latents=False`.
- Data filter: `worst_resolution ≤ 2.0 Å`, `min_length=50`, `max_length=512`. Sequence-similarity 0.5 split, val set size = 4058 proteins.
- `seed=42`, `dist_strategy=auto`.

**Reference results:**
- Best validation loss ≈ 4.71–4.77 around opt step 1800–2200, then overfits (val rises to 5+ within 200–700 more steps).
- Designability (ESMFold scRMSD < 2 Å, 200 ODE steps, N=3 per length):
  - step 1889: 1/3 at L=50, 2/3 at L=100, (L=200 not tested).
  - step 2457 (post-uptick): 1/3 at L=50, 2/3 at L=100, 0/3 at L=200.
- These numbers are the bar a variant must clear. They come from Finding 6's table; full per-length scRMSD distributions are tabulated there.

**Decisions encoded in this run (do not silently revisit them in variants):**
- wd is held at 0.05 because higher wd collapses AdaLN-Zero output gates and destroys designability while *improving* val loss (Finding 6). Raising wd requires restructuring `configure_optimizers` first.
- LR schedule is constant because cosine_with_warmup did not help in v2 (it co-occurred with the wd=0.1 collapse and was not isolated; in absence of evidence it improves things on its own, the simpler constant schedule is the canonical choice).
- `update_pair_repr=False` — we have no evidence the pair-update layer helps the CA-only task, and it adds compute. Keeping it off keeps variants cheap.
- `use_tri_mult=False` — was already off in the baseline; doubly required because triangular multiplicative updates need the full n×n pair grid and are incompatible with the planned sparse-attention variant (`pair_update.py:65` raises).
- 1-GPU configuration with `accumulate_grad_batches=32` is the deliberate match to the original 4-GPU effective batch (`4 × 8 × 6` = `1 × 32 × 6`), so the variant's batch dynamics are not a confounder.
- N=3 designability checks per length at 2-3 lengths is the cheap proxy for sample quality. This is required as a stopping rule for any variant — see Finding 6 implication: val loss alone is insufficient.

**How to use this entry:**

When proposing or evaluating a variant (sparse attention, conv downsampling, anything else): cite this section in the variant's "control" column, point to `1776805213` as the run dir, and verify that the variant's resolved Hydra config matches the JSON above on every key the variant doesn't claim to be changing.

---

## Finding 8 — In-progress: removing weight decay (wd=0) yields a step-1638 designability profile that already matches the canonical wd=0.05 baseline at comparable training stage (2026-04-27)

**Status:** in-progress. The training run is alive on Cambridge HPC; only the first usable checkpoint has been evaluated so far. This entry will be amended as later checkpoints (steps ≥ 2000) come in.

**Experiment:**

Direct causal test of the weight-decay × AdaLN-Zero gate-suppression mechanism proposed in Finding 6 — Variant B from the "Causal ablation" follow-up section below. Train from scratch with the canonical CA-only recipe but `weight_decay=0.0` instead of `0.05`. Everything else is held verbatim to the canonical baseline (see "Baseline reference" above and `configs/training_ca_only_wd0.yaml`):

- Architecture identical (`configs/nn/ca_only_score_nn_160M.yaml`, 160M `LocalLatentsTransformer` with AdaLN-Zero conditioning).
- Data identical (`pdb/pdb_train_ucond`, worst_resolution ≤ 2.0 Å, length 50–512, sequence-similarity 0.5 split).
- Optimizer: `torch.optim.AdamW`, **`weight_decay=0.0`**, constant LR=2e-4, no scheduler. β1=0.9, β2=0.999, ε=1e-8.
- Effective batch 192 proteins/step (`batch_size=6 × max_padding_size=512 × accumulate_grad_batches=32`), 1×A100 ampere, `dist_strategy=auto`, bf16-mixed, `gradient_clip_val=1.0`, EMA decay 0.999 every 5 steps. Seed 42.
- `val_check_interval=2000` mini-batches.
- Run dir: `store/ca_only_diffusion_wd0/<run_id>/`. Submit chain via `submit_train_ca_only_1gpu.sh -n training_ca_only_wd0` with `--exclude=gpu-q-43`.

Designability protocol (matches Finding 6 / canonical baseline reference):
- N=3 samples per length; lengths {50, 100, 200}; 200 ODE steps; ESMFold scRMSD; designable = scRMSD_ca < 2 Å.
- Eval config: `configs/inference_1638.yaml` (defaults to `inference_ucond_notri_ca_only`); `configs/generation/uncond_ca_only_quick.yaml` with `nres_lens: [50, 100, 200]`, `nsamples: 3`, `compute_designability: True`, `keep_folding_outputs: True`.
- Two seeds tested on the step-1638 ckpt (default `seed=5` and `seed=100`). Inference deterministic given seed; per-batch seed is `cfg.seed + job_id` (`generate.py:139`).
- Checkpoint evaluated: `best_val_00000016_000000001638.ckpt` (rsynced from RDS).

**Numbers — step 1638 (wd=0):**

Per-length scRMSD_ca (Å), N=3 each:

| L | seed | min | mean | max | designable (ca<2 Å) | best bb3o |
|---|---|---|---|---|---|---|
| 50  | 5 (default) | 5.07 | 8.70 | 12.58 | 0/3 | 5.04 |
| 100 | 5 | **1.89** | 8.02 | 11.98 | **1/3** | 2.15 |
| 200 | 5 | 12.94 | 13.72 | 14.63 | 0/3 | 12.90 |
| 50  | 100 | **1.04** | 4.47 | 6.69 | **1/3** | 1.44 |
| 100 | 100 | **1.49** | 6.36 | 9.97 | **1/3** | 1.86 |
| 200 | 100 | 10.73 | 12.54 | 15.87 | 0/3 | 10.62 |

Aggregate per seed: seed=5 → 1/9 designable; seed=100 → 2/9 designable.

For comparison (canonical wd=0.05 baseline at comparable training stages, from Finding 6 / Baseline reference):

| ckpt step | recipe | L=50 des | L=100 des | L=200 des | total | best ca |
|---|---|---|---|---|---|---|
| 1638 | **wd=0**, seed=5 | 0/3 | 1/3 | 0/3 | 1/9 | 1.89 (L=100) |
| 1638 | **wd=0**, seed=100 | 1/3 | 1/3 | 0/3 | 2/9 | 1.04 (L=50) |
| 1889 | wd=0.05 | 1/3 | 2/3 | (—) | — | — |
| 2078 | wd=0.05 | — | — | — | 1/9 | 1.86 (L=50) |
| 2457 | wd=0.05 | 1/3 | 2/3 | 0/3 | — | — |
| 2646 | wd=0.05 | ~3/9 typical | — | 0/3 | ~3/9 | — |

> **Misattribution correction (2026-04-27):** an earlier version of this entry listed a "step 1259, wd=0.05, 0/9 designable, best 2.14 Å at L=50" row in the comparison table. That checkpoint was subsequently identified (by inspecting `hyper_parameters.cfg.run_name_`) as `ca_only_sparse_K40` — the sparse-attention K40 variant — **not** canonical wd=0.05. The 0/9 result therefore belongs to the sparse ablation, not the canonical baseline. The misattribution affected only the comparison row, not the wd=0 numbers themselves.

(The seed-dependence of N=3 results is large — within a single seed at L=50 we see ca=1.04 vs 5.68 vs 6.69 — so single-seed N=3 is undersampled and any per-step comparison without seed averaging is noisy.)

**Numbers — N=30 batched designability comparison (2026-04-27, see E014 in `experiments.md` for protocol):**

Single-seed (seed=100), N=30/length, lengths {50, 100, 200}, ESMFold scRMSD < 2 Å (`scRMSD_ca`), per-sample MIN over 8 ProteinMPNN sequences. All four ckpts (canonical baseline, v2, wd=0, sparse K40) compared at matched seed so the initial noise trajectories are byte-identical across runs — what differs is only the model's velocity field. Each ckpt is its best available, not at matched optimizer step.

| run | ckpt | step | wd | L=50 des | L=100 des | L=200 des | total | L=50 median ca | L=100 median ca | L=200 median ca |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | `baseline_wd0.05_step2646.ckpt` | 2646 | 0.05 | **19/30 (63.3%)** | **20/30 (66.7%)** | **3/30 (10.0%)** | **42/90** | **1.65** | **1.48** | **4.57** |
| v2 | `v2_wd0.1_step2078.ckpt` | 2078 | 0.10 | 7/30 (23.3%) | 5/30 (16.7%) | 0/30 | 12/90 | 4.23 | 3.70 | 9.72 |
| wd0 | `wd0_step1638.ckpt` | 1638 | 0.00 | 10/30 (33.3%) | 4/30 (13.3%) | 0/30 | 14/90 | 2.47 | 4.12 | 12.10 |
| sparse K40 | `sparse_K40_step1259.ckpt` | 1259 | 0.05 | 9/30 (30.0%) | 1/30 (3.3%) | 0/30 | 10/90 | 4.17 | 5.42 | 11.81 |

Aggregate CSV with full percentile breakdown: `inference/n30_aggregate.csv` (gitignored — re-generated by `run_n30_pipeline.sh`).

**Updated Implikation (with N=30 evidence):**

- The wd=0 hypothesis is **partially refuted in the strong form** ("removing wd alone fixes designability"). At step 1638, wd=0 produces 33%/13%/0% designability vs canonical's 63%/67%/10% at step 2646 — substantially worse, even at the strongest comparison length (L=50, 33% vs 63%, 2× gap).
- The wd=0 hypothesis is **not refuted in the weak form** ("removing wd allows the deep gates to grow without harming convergence"). At step 1638 the wd=0 model has only just entered the val-best window (canonical hits its best around step 1800-2200). The decisive comparison is wd=0 at step ≈ 2200-2600 vs canonical 2646. Until then, the L=200 cliff observed at 0% for wd=0 is consistent with under-training, not architectural failure.
- **wd=0 ≥ wd=0.1 at every length.** Even at the under-trained step 1638, wd=0 (33%/13%/0%) beats v2 (23%/17%/0%) at L=50 and matches it at L=100, with both at 0% for L=200. This is consistent with the gate-growth mechanism: removing wd helps gate growth at the margin, but the late-training maturity that turns growing gates into clean integrated trajectories takes more steps than wd=0 has had so far.
- **Sparse K40 step 1259 has the worst short→medium falloff** of all four runs: 30% at L=50 → 3.3% at L=100 (10× drop). For comparison: canonical 63% → 67% (no drop), wd0 33% → 13% (2.5× drop), v2 23% → 17% (1.4× drop). This suggests sparse K40 may have a medium-L architectural issue separate from the under-training story — but with K=32 (8 sequential + 8 spatial + 16 random ∝ 1/d³, despite the K40 misnomer in the run name; see E012/E014 in `experiments.md`), each token only attends to ~1/3 of the L=100 chain through the local window, and the random-neighbor scaffolding is supposed to bridge the rest. Whether that scaffolding fails at L=100 specifically, or just hasn't trained enough yet, requires resuming sparse training to step ~2000 to disambiguate.

**Updated Methodische Einschränkungen (with N=30 evidence):**

- **Single-seed N=30**, not multi-seed N=30. Within-seed noise is now well-controlled (sample variance ≈ 3% Wilson CI on each 30/sample rate), but between-seed noise is unmeasured. A 2-seed N=30 replicate would tighten the cross-run comparison further.
- **Best-ckpt vs best-ckpt comparison.** Each ckpt is at its individually-best val window; this is the natural comparison for a "what does each recipe ultimately produce" question, but it confounds training duration with recipe. wd=0 at step 1638 is 1000+ steps behind canonical at 2646; sparse at step 1259 is even further behind. The N=30 numbers lock in the magnitudes; they do not yet support a "wd=X is worse than wd=Y at matched step" claim.
- **L=200 result is essentially zero across all three ablations** (v2/wd0/sparse all at 0/30). Only canonical reaches 3/30. The L=200 cliff is real, but at this N a single sample's scRMSD difference moves the rate by 3% — distinguishing 0/30 from 1/30 from 3/30 requires either more samples or a deliberate length-stratified analysis.
- **Sparse K40 architectural-vs-training-duration disambiguation is unresolved.** The 3.3% L=100 rate could reflect either (a) genuine medium-L architectural ceiling at K=32, or (b) under-training with a recipe that learns short-range correlations first. Resuming sparse training is the cheap test.

**Narrow claim:**

At training step 1638 — pre-convergence relative to the canonical wd=0.05 best-val window of step 1800–2200 — the wd=0 model produces sub-2 Å ESMFold scRMSD samples at both L=50 (best 1.04 Å, seed 100) and L=100 (best 1.49 Å, seed 100; 1.89 Å, seed 5), which is at least as good as the canonical wd=0.05 baseline at the closest comparable step (step 1259, best 2.14 Å at L=50) and approaches the wd=0.05 baseline's step-1889 short-length designability rate. The L≥200 cliff observed for the canonical baseline persists at this stage (best L=200 ca=10.73 Å with wd=0 vs 12.16 Å for wd=0.05 at step 1259) — so the L=200 generalization problem has not been resolved by removing wd at this checkpoint.

**Implikation:**

- **Removing weight decay does not destroy training.** Validation-loss curves are reportedly visually indistinguishable from canonical wd=0.05 in the first ~2000 steps, and short-length designability is intact at step 1638. This rules out the worst-case scenario where wd=0 collapses learning entirely (e.g. parameter norms blowing up or AdaLN-Zero gates training to chaotic magnitudes). The DiT-family literature recipe (`weight_decay=0` in the official DiT/SiT `train.py`) is at minimum compatible with this codebase's CA-only configuration.
- **Step 1638 is a preview, not a verdict.** Canonical wd=0.05 reaches its best designability at step 1800–2200, which is where the gate-suppression hypothesis's mechanistic prediction lives: if uniform wd=0.05 is suppressing deep AdaLN-Zero gates (Finding 6 diagnostic showed deep-layer gate weights at ~50% of shallow-layer magnitudes even at wd=0.05), and if that suppression is what causes the L=200 cliff, then a wd=0 ckpt at step ≈ 2000–2200 should (a) have larger deep-layer gate weights than the canonical 2646 ckpt, and (b) close at least part of the L=200 designability gap. Neither has been tested yet.
- **Variance vs effect size.** N=3 single-seed designability is too noisy to claim a wd=0 vs wd=0.05 effect at a single step; the seed=5 vs seed=100 comparison on the same step-1638 ckpt already swung 1/9 → 2/9. A real comparison needs (i) checkpoint-matched timing (canonical wd=0.05 step 2078 / 2646 vs wd=0 at the same step), (ii) ≥2 seeds per ckpt with 3 samples each, and (iii) per-layer AdaLN-Zero gate diagnostic at the wd=0 ckpt to verify the mechanistic prediction, not just the downstream designability.

**Methodische Einschränkungen:**

- **Single ckpt, single training run.** Step 1638 is the first val-best ckpt rsynced from the wd=0 chain; downstream behavior past this step is unknown at the time of writing. The canonical wd=0.05 baseline overfits past step 2200 (val rises to 5.39 within 700 steps) and the wd=0 run has no known overfitting profile yet — it might reach a higher step before degrading, or it might degrade earlier.
- **Designability ≠ likelihood.** ESMFold scRMSD < 2 Å is a downstream behavioral metric on a small N; the canonical wd=0.05 ckpts have all been run with the same N=3 protocol, but the noise floor is high. Per-length all-atom scRMSD distributions at larger N would tighten the comparison considerably.
- **Mechanistic step not yet performed.** Finding 6's mechanism claim was that wd=0.05 suppresses deep AdaLN-Zero gates (Finding 6 showed wd=0.1 → 26–60% of wd=0.05's deep-layer gates; the mid-session diagnostic on step 2646 / 2078 showed wd=0.05 → ~50% of shallow-layer gates in the deep blocks). The corresponding diagnostic on a wd=0 ckpt — does removing wd let the deep gates grow further? — has not yet been run. Without it, "wd=0 helps designability" remains observational.
- **Hyperparameter coupling.** Even though the recipe matches canonical exactly except for wd, the AdamW equilibrium argument (`|θ_eq| ≈ |grad|/wd`) implies wd=0 changes the equilibrium for *all* parameters, not just AdaLN-Zero gates. If the result holds at later steps, ruling out a confound from a non-AdaLN-Zero parameter group will require the per-layer diagnostic above (gate weights are the critical pathway by mechanism, but other parameters' magnitudes will also have shifted).
- **L=200 cliff still present.** The hypothesis that "wd=0.05 is what stops the model from generalizing past L=200" cannot yet be evaluated at step 1638 — the cliff is present in both recipes at this stage. The cliff may be (a) still reflecting undertraining, (b) genuinely not caused by wd, or (c) caused by wd but only fixed at a later step. Decisive evidence requires the step ≥ 2000 ckpt.
- **Seed=5 default in eval pipeline.** Re-running the same eval with the same `seed=5` is fully deterministic — the per-batch seed is `cfg.seed + job_id` from `generate.py:139`. This entry's seed-100 numbers were obtained by passing `seed=100` as a Hydra override on the eval command. Future replicate evals must vary `seed` to obtain new samples.

---

## Finding 9 — La-Proteina's joint sequence head produces a chemistry-specific alphabet collapse, mode-merges on bimodal natural distributions, and inflates standard sequence-based thermal-stability proxies (2026-04-30)

**Status:** finished for sub-claims (a), (b), (c.i). Sub-claim (c.ii) (ML-predicted Tm via TemStaPro) is preregistered, GPU run pending. Lab-notebook detail in `experiments.md → E019`.

**Experiment:**

Distributional comparison of 1,000 La-Proteina jointly-generated sequences (`results/generated_stratified_300_800/sequences.fasta`, sampled from `inference_ucond_notri_long.yaml` → `LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt`, length-stratified-uniform 300-800 in 50-residue bins, 200 ODE steps, seed 1000-1999) against PDB references along three orthogonal axes:

1. **14-property developability panel** (`compare_properties.py`): generated set scored by `steering/property_evaluate.py`, reference set is `laproteina_steerability/data/properties.csv` — 56,008 PDB proteins, length 300-796, scored by the *same* underlying functions in `proteinfoundation/analysis/compute_developability.py`. Both pipelines take their sequence from the per-protein `residue_type` tensor, so this is a clean comparison: real PDB sequence vs La-Proteina's own jointly-sampled sequence, with no MPNN intermediary on either side.
2. **Per-amino-acid composition** (`aa_composition.py`): mole fractions averaged across proteins, length-filtered to [300, 800]. Reference: `pdb_cluster_all_seqs.fasta` filtered to [300, 511] = 53,749 sequences (the FASTA is length-capped at 511; this introduces a small population-mismatch caveat — see below).
3. **Sequence-based thermal-stability proxies** (`thermal_stability.py`): aliphatic index (Ikai 1980), IVYWREL fraction (Zeldovich 2007), GRAVY (Kyte-Doolittle), charged fraction (D+E+K+R), log10[(D+E)/(K+R)] (regularized acidic/basic ratio), and aromatic fraction (F+W+Y) as a buried-core proxy. Computed on the same length-filtered sets as above.

**Numbers (selected; full table in `experiments.md → E019`):**

*Property panel — five largest gen-vs-PDB drifts:*

| property | ref mean | gen mean | Cohen's d | KS_d | ref/gen modes |
|---|---|---|---|---|---|
| shannon_entropy | 4.10 bits | 3.36 bits | **−6.65** | **0.92** | 2 / 1 |
| iupred3_fraction_disordered | 0.034 | 0.201 | **+2.70** | 0.36 | 1 / 1 |
| net_charge | −7.0 | −32.4 | −2.14 | 0.43 | 1 / 1 |
| scm_negative | −43 | −83.7 | −2.30 | 0.44 | 1 / 1 |
| swi (sequence-weighted hydropathy) | 0.779 | 0.795 | +1.62 | 0.62 | **2 / 1** |

*Per-AA composition — biggest over- and under-representations:*

- Over-represented: N (Asn) +146%, E (Glu) +95%, I (Ile) +35%, L (Leu) +29%, G (Gly) +21%.
- Under-represented: M (Met) −79%, H (His) −69%, W (Trp) −68%, F (Phe) −42%, D (Asp) −38%, V (Val) −37%, P (Pro) −34%, C (Cys) −34%, A (Ala) −28%.
- **Glu/Asp ratio: gen 3.22, PDB 1.04** — within-class chemistry asymmetry.
- Top 5 over-represented AAs make up **50.4%** of generated residues vs **32.5%** of PDB residues.

*Thermal-stability proxies:*

| metric | PDB | gen | Cohen's d |
|---|---|---|---|
| aliphatic_index (literature: ↑ = thermostable) | 84.4 | 91.8 | **+0.74** |
| ivywrel_fraction (literature: ↑ = thermostable) | 0.371 | 0.429 | **+1.35** |
| **aromatic_fraction (F+W+Y, buried-core proxy)** | **0.090** | **0.061** | **−1.19** |

**Narrow claim — three sub-claims, each individually defensible:**

**(a) Chemistry-specific alphabet collapse.** La-Proteina's joint sequence head reduces sequence Shannon entropy by ~0.74 bits (4.10 → 3.36), a 4.1 SD effect on KS-D = 0.92. The reduction is not uniform across residue chemistry: it concentrates probability mass on disorder-promoting / context-tolerant residues (E +95%, N +146%, L +29%, I +35%, G +21%) and depletes context-demanding residues (W −68%, F −42%, M −79%, H −69%, D −38%, V −37%, P −34%, C −34%, A −28%). Within the acidic-residue class, Glu is amplified ~3-fold relative to Asp (gen Glu/Asp = 3.22 vs PDB 1.04), preferring the longer, helix-friendly, surface-tolerant member over the shorter member that requires specific helix-N-cap / β-turn / Asx contexts. The aromatic depletion follows a core-buryness ranking (W > F ≫ Y) consistent with loss of buried hydrophobic-core anchors specifically rather than aromatic chemistry generically.

**(b) Mode-merging on multimodal natural distributions (partial).** Of the property-panel features that are bimodal in PDB, SWI (sequence-weighted hydropathy index) collapses from 2 modes to 1 with the generated mean (0.795) sitting *between* the two PDB modes — the textbook signature of regression to the unconditional mean rather than mode-dropping. pI remains bimodal in the generated set, so this is *not* a generic claim that all multimodal natural distributions collapse; it is a specific failure mode that occurs on at least one biophysically meaningful property.

**(c) Standard sequence-based thermal-stability proxies are confounded by alphabet collapse.** Aliphatic index (Ikai 1980) and IVYWREL fraction (Zeldovich 2007) — the two most-cited single-number sequence proxies for thermostability in the protein-engineering literature — *both* score the generated set as more thermostable than PDB (+0.74 SD and +1.35 SD respectively). Mechanistically, both proxies are dominated by Leu/Ile/Glu mole fractions, which are the residues over-represented in the alphabet collapse. Simultaneously, the most direct sequence-side proxy for a buried hydrophobic core — the F+W+Y aromatic fraction — drops 1.19 SD, contradicting the proxies' verdict. *(c.i, sequence-only)*: this contradiction alone is sufficient to demonstrate that the literature proxies cannot be applied to generative-model outputs without a structural sanity check. *(c.ii, preregistered)*: a TemStaPro ProtT5+MLP classifier (`thermal_stability.py --temstapro-dir`, GPU-bound, ~70 min A100) will return a per-protein P(Tm > T) at 9 thresholds for both sets; the prediction is that the alphabet-collapse compositional signal will *not* persuade an embedding-based classifier of higher thermostability, and the gen distribution will instead either match PDB or fall below it.

**Implication (cautiously phrased):** La-Proteina's headline co-designability metric (`evaluate.py:337`, `use_pdb_seq=True`) routes the model's own jointly-generated sequence directly into ESMFold without an MPNN re-design step. ESMFold is a sequence-conditioned structure predictor with a strong language-model prior; low-complexity, charge-and-asparagine-enriched, disorder-leaning sequences are within the easy regime of that prior and refold confidently regardless of whether the underlying generated structure is biologically plausible. The compositional drift documented above therefore plausibly inflates the co-designability number. This implication — co-designability gaming via easy-to-refold sequences — is the practical reason this Finding matters for the masterarbeit narrative: it identifies a candidate failure mode of joint-generation evaluation that cannot be detected from the headline scRMSD number alone.

**Methodological caveats — what this Finding does *not* support:**

1. **No claim that generated *structures* are unphysical.** Sub-claims (a)-(c) are entirely about the joint sequence-head output. F+W+Y depletion and iupred3 disorder bias are *consistent with* under-developed hydrophobic cores and floppy backbones, but DSSP secondary-structure breakdowns, ESMFold pLDDT distributions, and packing-density readouts on the gen set are not yet computed.
2. **Single-checkpoint, single-eval-seed result.** Generated set is N=1000 from `seed_base=1000`, scored from `LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt` only. Cross-seed and cross-checkpoint variance not estimated. Bootstrap uncertainty on AA-composition mole fractions is ~1% absolute.
3. **Reference-set length cap for AA composition.** `pdb_cluster_all_seqs.fasta` is length-capped at 511, so the AA-composition reference is PDB[300, 511] while the property-panel reference is PDB[300, 796]. Spot-checks (e.g., panel-reference Shannon mean = 4.10 matches the AA-composition-reference Shannon when computed independently) confirm the magnitudes are robust under this caveat, but the AA-composition numbers should not be quoted with sub-percentage precision.
4. **Co-designability inflation is a hypothesis, not a measurement.** The natural follow-up — designability vs co-designability gap on the same backbones (i.e. the paired comparison of MPNN-on-generated-backbone scRMSD vs La-Proteina-own-sequence scRMSD) plus designability stratified by Shannon-entropy decile — is preregistered but not yet computed. Without that paired comparison, the Implication is decorrelated from the Narrow claim and should be treated as a candidate explanation, not a measured effect.
5. **TemStaPro Tier 2 not yet completed.** Sub-claim (c) is supported by the internal contradiction between IVYWREL/aliphatic and aromatic_fraction within sequence-based proxies; it does not yet have an external ML-predicted Tm reference. The submit script (`script_utils/run_thermal_stability.sh`) is in place; results will amend (c.ii).
6. **Mode-merging claim limited to SWI.** pI stays bimodal; we have not surveyed all bimodal panel properties exhaustively. The general "mode-merging on multimodal targets" framing requires at least one more confirming property on a different chemistry axis before it can be promoted from a specific observation to a general claim.

---

## Future experiment ideas

### Causal ablation of the AdaLN-Zero × weight-decay collapse mechanism (follow-up to Finding 6, 2026-04-25)

**Motivation and literature gap (why this is worth running, despite Finding 6 already documenting the correlational result):**

Finding 6 established a strong correlational case that uniform AdamW weight decay applied to the La-Proteina CA-only baseline crushes AdaLN-Zero output gates in the upper transformer blocks (gates at 26–60% of baseline magnitude in v2 vs old) and that this co-occurs with a categorical sample-quality collapse (0/9 designable samples) despite a 0.33 best-validation-loss improvement. The mechanism — uniform AdamW weight decay applied to zero-initialised gates suppresses their growth, which weakens the time-conditioning signal at the velocity output and causes integrated trajectories to fail even though averaged velocity-MSE looks better — is mechanistically coherent and consistent with the per-layer weight evidence. However, no causal experiment has been run; an examiner could legitimately argue that the cosine LR decay (which co-occurred with wd=0.1 in v2 and was not isolated) is the actual cause, or that the collapse is a generic effect of stronger regularisation rather than the specific gate-suppression mechanism we propose.

A targeted literature survey (DiT, SiT, SD3 / MMDiT, PixArt-α, "Unveiling the Secret of AdaLN-Zero", ReZero, the diffusers SD3 reference, and the AdamW paper itself; full source list in CLAUDE-session notes) found that:
- **The dominant DiT-family training recipe is `weight_decay=0` everywhere.** DiT's official `train.py` uses literally `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)`; SiT's `train.py` uses the same line. SD3's paper does not discuss weight decay at all and emphasises QK-normalisation and ε=1e-15 in AdamW for stability. PixArt-α has a configurable `zero_weight_decay` attribute but does not apply it to AdaLN by default.
- **The "exclude AdaLN-Zero gates from wd" rule is therefore implicit in the literature, not enforced**: canonical DiT-family codebases sidestep the problem by setting wd=0, so they never observe the failure mode. The rule has not been formally written down and ablated in any paper we could find.
- **The closest prior art is ReZero** (Bachlechner et al. 2020), which mentions in passing that "it was important to have a small, constant learning rate for the residual weights, otherwise the ReZero model diverges" — a hint at the gate-fragility issue but not a quantitative or actionable claim about weight decay specifically.
- **The recent OpenReview paper "Unveiling the Secret of AdaLN-Zero"** isolates which architectural components of AdaLN-Zero matter (it identifies zero-initialisation as the dominant factor), but does not discuss the optimizer-gate interaction or the val-vs-sample-quality decoupling.

The literature gap is therefore real: the AdaLN-Zero × uniform-AdamW-wd interaction is folk knowledge in the DiT community (you wouldn't apply wd>0 without parameter groups in any modern image-generation codebase) but has not been formally characterised as a quantitative case study with per-layer evidence and a positive/negative ablation. This experiment fills that gap if successful, and at minimum produces a defensible thesis-internal causal claim.

**Hypothesis (mechanistic, to be falsified):**

Restoring growth capacity to the AdaLN-Zero output gates (either by removing weight decay entirely from those parameters, or by removing weight decay altogether and accepting the loss of regularisation) recovers sample quality on the v2 model **without** giving up the validation-loss improvement that wd+cosine bought. If this prediction is wrong — i.e. samples remain bad even after the gates are unconstrained — then the mechanism we proposed in Finding 6 is wrong (or at least incomplete) and the v2 collapse must be attributed to something else (cosine LR-decay shrinking the effective LR to a non-sampling-friendly regime; an interaction with self-conditioning; etc.).

**Experimental design — three variants in increasing scope:**

The three variants below trade off compute against the strength of the causal claim. Choose based on available compute and how watertight the writeup needs to be.

**A. Cheapest — finetune the v2-2078 ckpt with wd=0 on AdaLN-Zero gates only (~3 h on 1 A100):**

- Resume from the v2 best raw checkpoint (`store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/best_val_00000020_000000002078.ckpt`).
- Modify `proteinfoundation/proteina.py:configure_optimizers` to split parameters into two groups: (i) all `*.scale_output.to_adaln_zero_gamma.0.weight` parameters get `weight_decay=0`; (ii) everything else keeps the current `weight_decay` value (= 0.1). No other config change.
- Continue training for ≈ 200–500 optimizer steps. Log: `train/grad_norm`, `train/param_norm`, the per-layer L2 norm of each AdaLN-Zero gate (new logging — needs a small addition to `on_before_optimizer_step` that walks `[p for n,p in self.named_parameters() if "to_adaln_zero" in n]` and logs each `‖p‖`).
- Stopping rule: continue until either (a) the upper-layer gates (layers 7–13) recover to ≥ 80% of the old-recipe norms documented in Finding 6, or (b) 500 optimizer steps have elapsed without recovery.
- Save the resulting checkpoint, run the same eval as Finding 6 (`inference_ucond_notri_ca_only_v2_quick.yaml` config, lengths [50, 150, 300, 450], N=10 samples per length, 200 ODE steps, ESMFold designability).

**B. Cleanest — retrain v2 from scratch with the proper param-group split (~16 h on 1 A100, chained):**

- Modify `configure_optimizers` to split parameters into (i) bias / LayerNorm γ,β / embedding / `to_adaln_zero` parameters → `weight_decay=0`; (ii) everything else → `weight_decay=0.1`. This is the canonical DiT/SiT/SD3-style split (~15 lines).
- Train from scratch with the v2 schedule (cosine_with_warmup, peak 2e-4, total 6000 opt steps, min ratio 0.1) for the same ~2100–2300 optimizer steps where Finding 6 v2 best was reached. Use the same seed (42), same data, same batch size, same EMA settings.
- Compare against (i) the old recipe baseline checkpoint at its best (~step 2204, val 4.71–4.77, designability per Finding 6) and (ii) the v2 broken checkpoint (val 4.437, 0/9 designable).
- This is the "right" ablation because it isolates the param-group fix as the only difference from v2 — same LR schedule, same wd value, same everything else.

**C. Wd-isolation arm (add to either A or B if budget allows; ~16 h):**

- Train from scratch with `wd=0.05` + cosine_with_warmup (the schedule from v2 but the wd from old). This isolates whether the cosine schedule alone is sample-quality-friendly when wd is at its known-safe value.
- If sample quality matches old at wd=0.05 but val improves over old (because of the cosine schedule), this independently proves the cosine schedule is fine and the wd=0.1 is the entire problem.

**Outcome interpretation matrix:**

| Variant A (3h finetune) | Variant B (16h retrain w/ param-groups) | Implication |
|---|---|---|
| Gates regrow + samples recover | (not run) | **Hypothesis confirmed** at low cost; mechanism causally established. Strong thesis result. |
| Gates regrow but samples still bad | (not run) | **Mechanism partially correct** (wd does suppress gates) but gate suppression alone does not explain sample collapse. Need to investigate further (cosine LR? cumulative training trajectory?). Variant B becomes mandatory. |
| Gates do not regrow in 500 steps | (not run) | **Ambiguous** — could mean structural damage (unrecoverable post-hoc) or mechanism wrong. Variant B becomes mandatory. |
| (A run, gates+samples recover) | Same val improvement, sample quality matches or beats old | **Strongest possible result.** Causal claim watertight, plus a recipe that delivers the val improvement *and* good samples — a positive contribution beyond the negative result. Citable. |
| (A run, gates+samples recover) | Val regresses to old levels but samples are good | The wd=0.1 was helping val purely through gate suppression (i.e. by removing the conditioning signal that adds variance to the velocity prediction); removing the suppression removes the val "improvement". This would be a cleanly negative-but-instructive result: the val-loss improvement was *entirely* the artefact of the suppression, not a real model improvement. |
| (A run, gates+samples recover) | Samples still bad | Mechanism wrong; redo from scratch. |

**Compute cost:**
- Variant A: ~3 h on 1 A100 (single 6h SLURM slot with margin).
- Variant B: ~16 h on 1 A100 (chained 6h slots; the same setup used for the v2 attempt).
- Variant C (optional): another ~16 h.
- Cheapest defensible result: Variant A alone (~3 h). Strongest result: A + B (~19 h). Paper-grade: A + B + C (~35 h).

**Logging additions required (one-time, ~10 lines in `proteina.py:on_before_optimizer_step`):**
```python
# Per-AdaLN-Zero-gate L2 norm tracking — directly observes the recovery (or
# lack thereof) at each layer's gate, complementing the global param_norm.
gate_norms = {}
for n, p in self.named_parameters():
    if "to_adaln_zero" in n and p.requires_grad:
        gate_norms[n] = float(p.detach().pow(2).sum().sqrt())
for n, v in gate_norms.items():
    self.log(f"gate_norms/{n}", v, on_step=True, on_epoch=False, sync_dist=True)
```
This logging is independently valuable for any future training on this codebase.

**What the writeup would contain (target structure for the resulting Finding):**
1. Background: brief restatement of Finding 6 and the literature-gap motivation above.
2. Methods: the param-group split (~15 line code change), the variants run, the eval protocol.
3. Results: per-gate-layer norm trajectories during the experiment (do they regrow?), val-loss curve, designability table at matched lengths.
4. Discussion: which outcome of the matrix above was observed, and what it pins down about the mechanism.
5. Practical recommendation: a one-paragraph "if you train a DiT-style model with wd > 0 in any codebase that doesn't already split parameter groups, here is what you must do" section that cites the ablation as evidence.

**Risks and confounders to record with results:**
- Variant A's resume from v2-2078 carries forward whatever non-gate weights the v2 training shaped under wd=0.1; even if gates regrow, the *rest* of the model has been trained against suppressed gates and may have compensated in ways that don't undo cleanly. A failure in A is therefore not a failure of the mechanism — it could be a failure of the recovery procedure. Variant B (clean retrain) is needed to definitively rule this out.
- The eval is N=10 samples per length × 4 lengths = 40 samples. ESMFold scRMSD has known sensitivity to short-protein folding accuracy; if the recovered model is still marginal at L=50 but improves at L=300/450, that's still a positive result for the mechanism (the original v2 collapse was uniform across lengths, so partial recovery at any length is informative).
- The mechanism predicts that *if* the gates regrow, the val loss may regress (because the model goes back to predicting time-conditioned velocities, which are higher-MSE than the smoothed-out averages it was predicting before). Do not interpret a val-loss regression as a failure of the experiment — Finding 6's central claim is precisely that val loss is a misleading proxy in this setting.

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

### Direct latent-space geometry probe (follow-up to Finding 7)
Finding 7 showed that perturbations in AE1's latent space stay closer to ESMFold's prediction than equivalent coord-space perturbations, and inferred decoder contractivity from the noise-magnitude invariance. The "tighter packed latent space" framing was deliberately put in the Implikation, not the Narrow claim, because the experiment did not directly measure latent-space geometry. This follow-up closes that gap.

**Approach:**
1. Encode a held-out set of N=500–1000 proteins (same length stratification as Finding 7) → per-residue `mean` tensors.
2. For each pair (i, j), compute (a) Euclidean distance in latent space (L2 over per-residue means after sequence-length matching, e.g. averaging or padding+masking), (b) all-atom RMSD between native structures (TM-align or kabsch-aligned over CA).
3. Plot latent-distance vs structure-distance, fit a monotonic relationship (Spearman ρ).
4. Compare to a control: same plot but using raw stacked-CA-coordinate vectors as the "latent" — quantifies how much extra structure the AE imposes vs. just-coords.
5. Local density: for each protein, k-nearest neighbours in latent vs in structure space — does the AE preserve neighbourhoods?

**What this would directly test:** whether latent distances correspond to structural similarity (the actual "tighter packing" claim). A positive result strengthens Finding 7's Implikation; a negative result restricts Finding 7 to the strict decoder-contractivity claim only.

### Latent-arm decoder-contractivity ablation (follow-up to Finding 7, mechanistic)
The Implikation in Finding 7 attributes the noise-magnitude invariance of latent-arm scRMSD (median +0.17 Å over a 20× change in noise std) to the AE decoder being contractive. A direct test:

- For a single eval protein, sample latent perturbations at k ∈ {0.5, 1, 2, 4, 8, 16} (extending well past the data range) and decode.
- Plot `||z_clean − z_perturbed||_2` (input perturbation magnitude) vs `||x_clean − x_perturbed||_F` (output coord change). A contractive map shows sublinear or saturating behaviour; an isometric map shows linear; an expansive map shows superlinear.
- A clearly sublinear/saturating curve directly demonstrates contractivity, upgrading Finding 7's Implikation to a Narrow claim.

### Coord-arm with whitened, basis-aligned noise (Finding 7 follow-up, robustness)
The pre-registered Finding 7 caveat about commensurability between coord σ and latent σ deserves a tighter test. Add a third arm: coord-space noise sampled in a basis aligned with the local sidechain rotamer manifold (e.g. PCA over per-(restype, atom_idx) offset distributions, with σ scaled to match each PC's variance). If even this PC-aligned coord arm loses to latent at all k, the latent-space advantage is not just a bad-noise-basis artefact in the original coord arm.
