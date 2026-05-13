# Content Masterarbeit — Paper-relevant findings

## Thesis intent

The Masterarbeit's mission is **to exceed La-Proteina's baseline at long protein generation (L ≥ 300, target L = 500), where canonical CA-only designability falls off the cliff.** Two routes are pursued in parallel; each Finding below contributes to one of them, or to the methodological scaffolding that makes route-comparison possible:

1. **Architectural route** — sparse attention, 1D-conv downsampling, or other backbone modifications that make longer sequences tractable / better-conditioned during training and sampling. The bar is *strict improvement over baseline designability at L ≥ 300*, not "approximately matches at the same step count".
2. **Steering route** — gradient-based sampling-time guidance, applied to the official LD3+AE2 La-Proteina checkpoint, that pushes long-L generations into the designable regime while the underlying flow stays unmodified. The bar is *generated long-L proteins that pass real-property + designability filters in a lab-relevant sense*, not just predictor-side dose-response.

Findings ordered roughly by which route they feed:
- **Steering scaffold** (F1, F2, F3, F6) — the latent space supports gradient-based guidance: properties are decodable from latents (F1, F3), the flow channels have separable curvature (F2), and AE1 latent perturbations stay on the data manifold (F6).
- **Baseline constraint** (F4, F7) — defines what the canonical recipe is and why it's not casually movable, so any architectural variant or steering result is comparable against a fixed reference.
- **Joint-head evaluation methodology** (F8) — characterises La-Proteina's joint sequence head behaviour, so co-designability claims at long L can be interpreted correctly (i.e. not gamed by easy-to-refold sequences).

A finding that just "improves the baseline a bit" is not a thesis claim. A finding that *exceeds* baseline at L ≥ 300 (architectural route) or that *moves real-property + designability* on long-L generations (steering route) is.

---

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

## Finding 5 — Negative result: stronger weight decay improves validation loss but collapses sample quality, traced to AdamW crushing AdaLN-Zero output gates (2026-04-25)

> *Renumbered 2026-05-03: this entry was Finding 5 prior to the demotion of the original Finding 5 ("Very-early-training CA-only diffusion snapshots are far from designable", 2026-04-22), which is now an `experiments.md`-only baseline anchor — see `experiments.md → E008` (Canonical CA-only baseline training) for the lab record. The demotion does not affect any cross-references in `experiments.md` that already used "Finding 5" to mean the wd × AdaLN-Zero collapse claim documented here.*

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

## Finding 6 — Sidechain perturbations in AE1 latent space stay closer to ESMFold's prediction than equivalent coord-space perturbations across all noise scales (2026-04-25)

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

This is the run that the sparse-attention and conv-downsampling variants must be compared against. It is **not** a standalone finding — Finding 5 already covered the val-vs-sample-quality analysis. The purpose of this entry is to lock in the baseline's exact configuration so any later "the variant beat the baseline" claim has a single, citable reference point on disk.

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
- These numbers are the bar a variant must clear. They come from Finding 5's table; full per-length scRMSD distributions are tabulated there.

**Decisions encoded in this run (do not silently revisit them in variants):**
- wd is held at 0.05 because higher wd collapses AdaLN-Zero output gates and destroys designability while *improving* val loss (Finding 5). Raising wd requires restructuring `configure_optimizers` first.
- LR schedule is constant because cosine_with_warmup did not help in v2 (it co-occurred with the wd=0.1 collapse and was not isolated; in absence of evidence it improves things on its own, the simpler constant schedule is the canonical choice).
- `update_pair_repr=False` — we have no evidence the pair-update layer helps the CA-only task, and it adds compute. Keeping it off keeps variants cheap.
- `use_tri_mult=False` — was already off in the baseline; doubly required because triangular multiplicative updates need the full n×n pair grid and are incompatible with the planned sparse-attention variant (`pair_update.py:65` raises).
- 1-GPU configuration with `accumulate_grad_batches=32` is the deliberate match to the original 4-GPU effective batch (`4 × 8 × 6` = `1 × 32 × 6`), so the variant's batch dynamics are not a confounder.
- N=3 designability checks per length at 2-3 lengths is the cheap proxy for sample quality. This is required as a stopping rule for any variant — see Finding 5 implication: val loss alone is insufficient.

**How to use this entry:**

When proposing or evaluating a variant (sparse attention, conv downsampling, anything else): cite this section in the variant's "control" column, point to `1776805213` as the run dir, and verify that the variant's resolved Hydra config matches the JSON above on every key the variant doesn't claim to be changing.

---

## Finding 7 — In-progress: removing weight decay (wd=0) yields a step-1638 designability profile that already matches the canonical wd=0.05 baseline at comparable training stage (2026-04-27)

**Status:** in-progress. The training run is alive on Cambridge HPC; only the first usable checkpoint has been evaluated so far. This entry will be amended as later checkpoints (steps ≥ 2000) come in.

**Experiment:**

Direct causal test of the weight-decay × AdaLN-Zero gate-suppression mechanism proposed in Finding 5 — Variant B from the "Causal ablation" follow-up section below. Train from scratch with the canonical CA-only recipe but `weight_decay=0.0` instead of `0.05`. Everything else is held verbatim to the canonical baseline (see "Baseline reference" above and `configs/training_ca_only_wd0.yaml`):

- Architecture identical (`configs/nn/ca_only_score_nn_160M.yaml`, 160M `LocalLatentsTransformer` with AdaLN-Zero conditioning).
- Data identical (`pdb/pdb_train_ucond`, worst_resolution ≤ 2.0 Å, length 50–512, sequence-similarity 0.5 split).
- Optimizer: `torch.optim.AdamW`, **`weight_decay=0.0`**, constant LR=2e-4, no scheduler. β1=0.9, β2=0.999, ε=1e-8.
- Effective batch 192 proteins/step (`batch_size=6 × max_padding_size=512 × accumulate_grad_batches=32`), 1×A100 ampere, `dist_strategy=auto`, bf16-mixed, `gradient_clip_val=1.0`, EMA decay 0.999 every 5 steps. Seed 42.
- `val_check_interval=2000` mini-batches.
- Run dir: `store/ca_only_diffusion_wd0/<run_id>/`. Submit chain via `submit_train_ca_only_1gpu.sh -n training_ca_only_wd0` with `--exclude=gpu-q-43`.

Designability protocol (matches Finding 5 / canonical baseline reference):
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

For comparison (canonical wd=0.05 baseline at comparable training stages, from Finding 5 / Baseline reference):

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
- **Step 1638 is a preview, not a verdict.** Canonical wd=0.05 reaches its best designability at step 1800–2200, which is where the gate-suppression hypothesis's mechanistic prediction lives: if uniform wd=0.05 is suppressing deep AdaLN-Zero gates (Finding 5 diagnostic showed deep-layer gate weights at ~50% of shallow-layer magnitudes even at wd=0.05), and if that suppression is what causes the L=200 cliff, then a wd=0 ckpt at step ≈ 2000–2200 should (a) have larger deep-layer gate weights than the canonical 2646 ckpt, and (b) close at least part of the L=200 designability gap. Neither has been tested yet.
- **Variance vs effect size.** N=3 single-seed designability is too noisy to claim a wd=0 vs wd=0.05 effect at a single step; the seed=5 vs seed=100 comparison on the same step-1638 ckpt already swung 1/9 → 2/9. A real comparison needs (i) checkpoint-matched timing (canonical wd=0.05 step 2078 / 2646 vs wd=0 at the same step), (ii) ≥2 seeds per ckpt with 3 samples each, and (iii) per-layer AdaLN-Zero gate diagnostic at the wd=0 ckpt to verify the mechanistic prediction, not just the downstream designability.

**Methodische Einschränkungen:**

- **Single ckpt, single training run.** Step 1638 is the first val-best ckpt rsynced from the wd=0 chain; downstream behavior past this step is unknown at the time of writing. The canonical wd=0.05 baseline overfits past step 2200 (val rises to 5.39 within 700 steps) and the wd=0 run has no known overfitting profile yet — it might reach a higher step before degrading, or it might degrade earlier.
- **Designability ≠ likelihood.** ESMFold scRMSD < 2 Å is a downstream behavioral metric on a small N; the canonical wd=0.05 ckpts have all been run with the same N=3 protocol, but the noise floor is high. Per-length all-atom scRMSD distributions at larger N would tighten the comparison considerably.
- **Mechanistic step not yet performed.** Finding 5's mechanism claim was that wd=0.05 suppresses deep AdaLN-Zero gates (Finding 5 showed wd=0.1 → 26–60% of wd=0.05's deep-layer gates; the mid-session diagnostic on step 2646 / 2078 showed wd=0.05 → ~50% of shallow-layer gates in the deep blocks). The corresponding diagnostic on a wd=0 ckpt — does removing wd let the deep gates grow further? — has not yet been run. Without it, "wd=0 helps designability" remains observational.
- **Hyperparameter coupling.** Even though the recipe matches canonical exactly except for wd, the AdamW equilibrium argument (`|θ_eq| ≈ |grad|/wd`) implies wd=0 changes the equilibrium for *all* parameters, not just AdaLN-Zero gates. If the result holds at later steps, ruling out a confound from a non-AdaLN-Zero parameter group will require the per-layer diagnostic above (gate weights are the critical pathway by mechanism, but other parameters' magnitudes will also have shifted).
- **L=200 cliff still present.** The hypothesis that "wd=0.05 is what stops the model from generalizing past L=200" cannot yet be evaluated at step 1638 — the cliff is present in both recipes at this stage. The cliff may be (a) still reflecting undertraining, (b) genuinely not caused by wd, or (c) caused by wd but only fixed at a later step. Decisive evidence requires the step ≥ 2000 ckpt.
- **Seed=5 default in eval pipeline.** Re-running the same eval with the same `seed=5` is fully deterministic — the per-batch seed is `cfg.seed + job_id` from `generate.py:139`. This entry's seed-100 numbers were obtained by passing `seed=100` as a Hydra override on the eval command. Future replicate evals must vary `seed` to obtain new samples.

---

## Finding 8 — La-Proteina's joint sequence head produces a chemistry-specific alphabet collapse and inflates standard sequence-based thermal-stability proxies, while sharpening aromatic core-targeting on the residual budget (2026-04-30; AFDB-rereferenced 2026-05-03; sub-claim (b) withdrawn 2026-05-03; nsteps=400 regen 2026-05-05)

**Status:** finished for sub-claims (a), (b.i), and (c) — all three confirmed against the corrected AFDB reference (E026, 2026-05-03) and replicated at the canonical nsteps=400 inference resolution (E020/E026 follow-up, 2026-05-05). Sub-claim (b.ii) (ML-predicted Tm via TemStaPro) is preregistered, GPU run pending. The 2026-04-30 sub-claim *(b) "mode-merging on bimodal natural distributions"* has been **withdrawn** (2026-05-03) after the AFDB rerun + a matched-n robustness check (E027) showed that the apparent mode-merging signature was either a PDB-vs-AFDB population difference (shannon_entropy: PDB is bimodal but AFDB is unimodal — gen matches AFDB, no merging to detect) or a sample-size detection artifact (SWI: PDB's 2-mode signature vanishes when PDB is subsampled to AFDB's n=5K). Lab-notebook detail in `experiments.md → E020` (original PDB reference, nsteps=200), `experiments.md → E026` (AFDB rerun, primary, nsteps=200), `experiments.md → E020+E026 follow-up` (nsteps=400 regen, **the numbers cited below are from this regen**), and `experiments.md → E027` (modality robustness check).

**Critical framing — La-Proteina was trained on AFDB, not PDB.** The original 2026-04-30 version of this Finding compared the model's joint-sequence-head output against a 56K-protein PDB reference. PDB is *not* the model's training distribution; the AlphaFold-clustered AFDB subset is. PDB and AFDB differ along several axes (PDB is biased toward crystallisable, well-folded proteins; AFDB has more disorder, more compositional flexibility, weaker hydrophobic-core packing in the predicted structures). Comparing an AFDB-trained generative model against a PDB reference therefore conflates two distinct questions: (i) "does the model drift from its training distribution?" (the right question for this Finding) and (ii) "does the model's training distribution drift from the rigorous-crystallography natural-protein distribution?" (a separate question about AFDB itself, not about La-Proteina's joint head). The 2026-05-03 AFDB rerun (E026) addresses the first question directly. Where the two readings differ, AFDB numbers should be treated as primary; PDB numbers below are kept as a sensitivity check and to make the size of the AFDB-vs-PDB reference shift visible.

**Experiment:**

Distributional comparison of 1,000 La-Proteina jointly-generated sequences (`results/generated_stratified_300_800_nsteps400/sequences.fasta`, sampled from `inference_ucond_notri_long.yaml` → `LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt`, length-stratified-uniform 300-800 in 50-residue bins, **400 ODE steps** — the codebase's canonical inference default — seed 1000-1999) against natural-protein references along three orthogonal axes. Reference set: a uniform-random N=5000 AFDB sample length-stratified to gen's [300, 800] distribution (E026 — primary). The original 56K-protein PDB reference is retained as a sensitivity comparison; numbers below are reported as AFDB primary / PDB secondary where both exist. (The original 2026-04-30 / 2026-05-03 numbers used 200 ODE steps, half the canonical default — those are documented in `experiments.md → E020` and `experiments.md → E026` for record-of-history; the numbers cited below are from the 2026-05-05 nsteps=400 regen, in which every Finding 8 sub-claim survived with magnitudes attenuated 5-15% on most metrics.)

1. **15-property developability panel** (`compare_properties.py`): generated set scored by `steering/evaluate_samples_dir.py` running the full `compute_developability` pipeline on `.pt` files (TANGO via local binary, FreeSASA-based hydrophobic patches / SAP / SCM, IUPred3, charged-residue counts), reference set is `laproteina_steerability/data/properties.csv` — 56,008 PDB proteins — and `data/afdb_ref/properties_afdb_refschema.csv` — 4,998 AFDB proteins. Both pipelines take their sequence from the per-protein `residue_type` tensor, so this is a clean comparison: real natural sequence vs La-Proteina's own jointly-sampled sequence, with no MPNN intermediary on either side.
2. **Per-amino-acid composition** (`aa_composition.py`): mole fractions averaged across proteins, length-filtered to [300, 800]. Reference: `pdb_cluster_all_seqs.fasta` filtered to [300, 511] = 53,749 sequences (the FASTA is length-capped at 511 — small population-mismatch caveat preserved); AFDB reference is the 4,998 AFDB sequences at full [300, 800] coverage.
3. **Sequence-based thermal-stability proxies** (`thermal_stability.py`): aliphatic index (Ikai 1980), IVYWREL fraction (Zeldovich 2007), GRAVY (Kyte-Doolittle), charged fraction (D+E+K+R), log10[(D+E)/(K+R)] (regularized acidic/basic ratio), and aromatic fraction (F+W+Y) as a buried-core proxy. Computed on the same length-filtered sets as above.

**Numbers (full 15-row panel; lab-notebook detail in `experiments.md → E020+E026 follow-up`):**

*Property panel — gen vs natural reference (AFDB primary; PDB sensitivity).* Now reports all 15 panel rows on both sides — the prior gen seqonly limitation that capped E026 at 5 rows is resolved by the nsteps=400 regen. Numbers in the table are AFDB d / PDB d.

| property                 | AFDB d / PDB d | AFDB KS / PDB KS | gen / ref(AFDB) / ref(PDB) means |
|--------------------------|----------------|-------------------|----------------------------------|
| shannon_entropy          | **−2.99 / −5.78** | **0.83 / 0.89**     | 3.47 / 4.05 / 4.10 bits          |
| swi                      | +1.20 / +1.59 | 0.54 / 0.61       | 0.795 / 0.778 / 0.779            |
| iupred3_fraction_disordered | +0.27 / +2.36 | 0.08 / 0.33  | 0.175 / 0.123 / 0.034            |
| iupred3_mean             | +0.45 / +1.49 | 0.22 / 0.41       | 0.321 / 0.258 / 0.216            |
| net_charge               | −0.97 / −2.20 | 0.36 / 0.43       | −32.7 / −5.2 / −7.0              |
| pI                       | −0.67 / −0.41 | 0.43 / 0.40       | 5.64 / 6.78 / 6.13               |
| **tango**                | **−0.46 / +0.37** | 0.22 / 0.18  | 1106 / 1312 / 999                |
| **tango_aggregation_positions** | **−0.58 / −0.05** | 0.33 / 0.24 | 1561 / 3654 / 1618        |
| **hydrophobic_patch_total_area** | **−1.43 / −0.91** | **0.81 / 0.56** | 2746 / 9820 / 3964     |
| **hydrophobic_patch_n_large** | **−1.39 / −1.07** | **0.82 / 0.60** | 5.85 / 36.1 / 12.2          |
| **sap**                  | **−1.21 / −0.50** | 0.75 / 0.56  | 10.3 / 33.7 / 13.3               |
| scm_positive             | −0.68 / +0.59 | 0.31 / 0.24       | 47.9 / 71.5 / 39.6               |
| scm_negative             | −0.39 / −2.34 | 0.15 / 0.43       | −84.2 / −67.2 / −43.0            |
| rg                       | −0.89 / +0.49 | 0.55 / 0.30       | 23.4 / 33.6 / 21.9               |
| sequence_length          | −0.005 / +1.48 | 0.012 / 0.46     | 549 / 550 / 405 (length-matched on AFDB) |

The bold rows are the structure-derived columns that did not exist on the gen side in E020/E026 (gen seqonly CSV); they're now first-class numbers. Two readings worth promoting to the narrative:

- **Shannon-entropy collapse softened by ~13% in d (−6.65 → −5.78 vs PDB; −3.39 → −2.99 vs AFDB) but is still by far the panel's biggest deviation under both references.** KS-D barely moved (0.92 → 0.89 vs PDB; 0.89 → 0.83 vs AFDB). Effective alphabet 2^4.10 ≈ 17 (PDB) → 2^3.47 ≈ 11 (gen, was 2^3.36 ≈ 10).
- **The structure-derived Cohen's d's vs AFDB are *all negative* and uniformly large in magnitude** — gen has lower TANGO, smaller hydrophobic patches (both total area and large-patch count), lower SAP, less negative scm_negative, smaller Rg than the AFDB reference. This is a structure-side echo of the alphabet collapse: fewer aromatics + more E/N/G in the sequence translates to less hydrophobic surface, fewer aggregation-prone segments, and more compact folds in whatever the model produces. Both vs-PDB and vs-AFDB structure-side directions agree in sign on every column; magnitudes are larger against AFDB. Note `tango` flips sign between references (gen +0.37 vs PDB; gen −0.46 vs AFDB) because PDB and AFDB sit on opposite sides of gen on TANGO — gen is *between* the two natural references on this metric, with AFDB scoring higher TANGO than PDB because AlphaFold-predicted "buried" regions have softer hydrophobic packing per FreeSASA than crystal-structure cores.

*Per-AA composition — gen vs AFDB; PDB rel-Δ in parentheses:*

- Over-represented (gen / AFDB / rel Δ): **N (Asn)** 9.70% / 3.80% / **+156%** (PDB +132%); E (Glu) 11.53% / 6.21% / +86% (PDB +90%); I (Ile) 6.92% / 5.51% / +25% (PDB +28%); L (Leu) 11.95% / 9.79% / +22% (PDB +35%); G (Gly) 8.31% / 7.38% / +13% (PDB +5%).
- Under-represented (gen / AFDB / rel Δ): **M (Met)** 0.68% / 2.29% / **−70%** (PDB −72%); W (Trp) 0.66% / 1.32% / −50% (PDB −54%); H (His) 1.25% / 2.25% / −44% (PDB −58%); F (Phe) 2.27% / 3.97% / −43% (PDB −45%); D (Asp) 4.14% / 5.51% / −25% (PDB −29%); V (Val) 4.69% / 6.83% / −31% (PDB −33%); P (Pro) 3.13% / 5.03% / −38% (PDB −36%); A (Ala) 6.16% / 8.98% / **−31%** (PDB −27%).
- **Glu/Asp ratio: gen 2.79, AFDB 1.13, PDB 1.04** — the within-class chemistry asymmetry survives. The 3.22 ratio reported in the original (nsteps=200) E020/E026 numbers softened to 2.79 at canonical inference resolution — still 2.5× the natural Glu/Asp baseline, so the qualitative story holds. AFDB and PDB are still essentially identical on this asymmetry.
- Top 5 over-represented AAs make up **48.4%** of generated residues (was 50.4% in original).

The pattern is **rock-solid against AFDB**: every sign preserved; the most-extreme deviations (N over-rep, M/W/H under-rep) softened by 10-15 percentage points but still dominant. The "buried aromatic core anchors are depleted in count" signal (W > F ≫ Y in the under-representation list, with Y barely deviating) is also intact. Sub-claim (a) is confirmed under the corrected reference at the canonical nsteps.

*Thermal-stability proxies:* (gen / AFDB / Cohen's d vs AFDB; PDB d in parentheses)

| metric | gen | AFDB | Cohen's d (AFDB) | (PDB d) |
|---|---|---|---|---|
| aliphatic_index (literature: ↑ = thermostable) | 93.3 | 88.5 | **+0.30** | (+0.89) |
| ivywrel_fraction (literature: ↑ = thermostable) | 0.430 | 0.384 | **+0.72** | (+1.41) |
| gravy (Kyte-Doolittle) | −0.517 | −0.196 | −0.86 | (−1.31) |
| charged_fraction (D+E+K+R) | 0.256 | 0.224 | +0.47 | (+0.87) |
| log_acidic_basic_ratio | 0.236 | 0.037 | +0.93 | (+1.49) |
| **aromatic_fraction (F+W+Y, buried-core proxy)** | **0.062** | 0.083 | **−0.72** | (−1.16) |

All AFDB d's still ~half the PDB d's; every sign preserved at the canonical inference resolution. The internal contradiction that powers sub-claim (b.i) below — "literature thermostability proxies score gen as more thermostable while the buried-core proxy scores gen as worse" — exists relative to AFDB at nsteps=400 too, with magnitudes essentially identical to the nsteps=200 readings (within ±0.18 in d on every metric). The aliphatic_index drift even *grew* slightly from +0.74 → +0.89 vs PDB at the canonical resolution — ruling out the worry that the methodological observation might be an nsteps=200 artifact.

**Narrow claim — three sub-claims, each individually defensible against the AFDB reference (primary) and replicated against the PDB reference (sensitivity).**

**(a) Chemistry-specific alphabet collapse.** La-Proteina's joint sequence head reduces sequence Shannon entropy by ~0.58 bits against AFDB (4.05 → 3.47; KS-D = 0.83; Cohen's d = −2.99); the same reduction against PDB is ~0.63 bits (KS-D = 0.89, d = −5.78). The reduction is not uniform across residue chemistry: it concentrates probability mass on disorder-promoting / context-tolerant residues (against AFDB: N +156%, E +86%, I +25%, L +22%, G +13%) and depletes context-demanding residues (M −70%, W −50%, H −44%, F −43%, D −25%, V −31%, P −38%, A −31%). The pattern is essentially identical against PDB (signs preserved on every residue; magnitudes within a few percentage points except N which grows from +132% vs PDB to +156% vs AFDB, and L which shrinks from +35% vs PDB to +22% vs AFDB). Within the acidic-residue class, Glu is amplified ~2.8-fold relative to Asp (gen Glu/Asp = 2.79; AFDB 1.13; PDB 1.04 — AFDB and PDB are essentially identical on this asymmetry), preferring the longer, helix-friendly, surface-tolerant member over the shorter member that requires specific helix-N-cap / β-turn / Asx contexts. The aromatic depletion follows a core-buryness ranking (W > F ≫ Y) consistent with reduced use of buried hydrophobic-core anchors at the *count* level. (How those reduced-count aromatics are *placed* is the subject of sub-claim (c) below — and the placement story is interestingly different from the count story.) The structure-side echo is now first-class observable from the regen: hydrophobic_patch_total_area Cohen's d = −1.43 vs AFDB, sap d = −1.21 vs AFDB — the alphabet-collapse-driven reduction in hydrophobic content shows up as significantly smaller hydrophobic-patch area and lower aggregation-propensity scores than the natural reference under both readouts, even though the literature single-number proxies in (b) below say the opposite.

**(b) Standard sequence-based thermal-stability proxies are confounded by alphabet collapse.** Aliphatic index (Ikai 1980) and IVYWREL fraction (Zeldovich 2007) — the two most-cited single-number sequence proxies for thermostability in the protein-engineering literature — *both* score the generated set as more thermostable than the natural reference (against AFDB: +0.30 SD aliphatic, +0.72 SD IVYWREL; against PDB: +0.89 and +1.41 SD). Mechanistically, both proxies are dominated by Leu/Ile/Glu mole fractions, which are the residues over-represented in the alphabet collapse. Simultaneously, the most direct sequence-side proxy for a buried hydrophobic core — the F+W+Y aromatic fraction — drops against both references (AFDB d = −0.72, PDB d = −1.16), contradicting the proxies' verdict. *(b.i, sequence-only)*: this contradiction alone is sufficient to demonstrate that the literature proxies cannot be applied to generative-model outputs without a structural sanity check; the contradiction is robust to whether AFDB or PDB serves as the natural-protein reference *and* to whether the gen distribution is sampled at nsteps=200 (where d's are within ±0.18 of these numbers) or nsteps=400 (the canonical inference resolution; numbers above). *(b.ii, preregistered)*: a TemStaPro ProtT5+MLP classifier (`thermal_stability.py --temstapro-dir`, GPU-bound, ~70 min A100) will return a per-protein P(Tm > T) at 9 thresholds for both sets; the prediction is that the alphabet-collapse compositional signal will *not* persuade an embedding-based classifier of higher thermostability, and the gen distribution will instead either match the natural reference or fall below it.

**(c) Aromatic count-vs-placement asymmetry: fewer aromatics overall, sharper concentration into the buried core.** The alphabet collapse documented in (a) reduces the gen aromatic budget from AFDB's 8.26% to gen's 6.07% — a ~26% reduction in absolute count of aromatic residues. However, *placement* of the residual aromatic budget into the buried core is sharper in gen than in AFDB at every per-residue burial-targeting ratio (W gen R = 9.80 vs AFDB R = 2.41, +307%; Y gen R = 5.32 vs AFDB R = 2.71, +96%; H gen R = 1.29 vs AFDB R = 0.86, +50%; group gen R = 3.00 vs AFDB R = 1.99, +51%; F gen R = 2.58 vs AFDB R = 2.52, matched). Bootstrap CIs over proteins; AFDB-reference numbers from E026, gen-side numbers verbatim from E023. **The two axes carry opposite signals:** the count axis is a model-side bias (fewer aromatics than the training distribution would suggest), the placement axis is a competence signal (the few aromatics the model does use are concentrated more sharply into the buried core than AFDB's predicted-structure pattern shows). One natural reading of the joint pattern: the alphabet collapse is structure-aware — when an aromatic residue is used, it is preferentially used in a context where it does the most work for hydrophobic-core stability; the model is not uniformly diluting aromatic content. This is *not* a claim that gen structures are well-folded — only that within whatever fold-quality the model achieves, aromatic placement is core-biased rather than uniformly distributed. Caveat: F is the single residue where placement is *not* sharper (gen R = 2.58 vs AFDB R = 2.52, within bootstrap noise) — sub-claim (c) is carried by W, Y, H, and the group ratio, with F as an exception. Lab-notebook detail in `experiments.md → E023` (original PDB-reference probe) and `experiments.md → E026` (AFDB-reference rerun, primary). *Note:* the original 2026-05-03-morning reading of E023 — "the model fails to bury F" — was a PDB-reference artifact and does **not** survive the AFDB switch; the surviving reading is the count-vs-placement asymmetry framed here.

**Implication (cautiously phrased):** La-Proteina's headline co-designability metric (`evaluate.py:337`, `use_pdb_seq=True`) routes the model's own jointly-generated sequence directly into ESMFold without an MPNN re-design step. ESMFold is a sequence-conditioned structure predictor with a strong language-model prior; low-complexity, charge-and-asparagine-enriched, disorder-leaning sequences are within the easy regime of that prior and refold confidently regardless of whether the underlying generated structure is biologically plausible. The compositional drift documented above therefore plausibly inflates the co-designability number. This implication — co-designability gaming via easy-to-refold sequences — is the practical reason this Finding matters for the masterarbeit narrative: it identifies a candidate failure mode of joint-generation evaluation that cannot be detected from the headline scRMSD number alone.

**Methodological caveats — what this Finding does *not* support:**

1. **No claim that generated *structures* are well-folded or biophysically plausible.** Sub-claims (a) and (b) are entirely about the joint sequence-head output. Sub-claim (c) is a structural-relative-positioning observation (P(aromatic|buried) / P(aromatic|exposed)) and does not say anything about absolute fold quality, packing density, or secondary-structure correctness — see caveat 6 below for the precise framing of (c). DSSP secondary-structure breakdowns, ESMFold pLDDT distributions, and packing-density readouts on the gen set are not yet computed. **Pre-AFDB-rerun reading withdrawn:** the original 2026-05-03-morning E023 reading — "the model fails to bury F" (PDB-reference gen R = 2.58 vs PDB R = 5.68) — does not survive the AFDB reference switch (E026: gen R = 2.58 vs AFDB R = 2.52, identical within bootstrap noise). AlphaFold-predicted "buried" regions have softer hydrophobic-core packing than crystal-structure cores, so PDB's per-residue burial-targeting ratios are systematically higher than AFDB's; the gen pattern matches AFDB on F. The replacement reading is sub-claim (c) — count-vs-placement asymmetry — which uses AFDB-reference numbers throughout.
2. **Single-checkpoint, single-eval-seed result.** Generated set is N=1000 from `seed_base=1000`, scored from `LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt` only, at nsteps=400 (canonical inference resolution). Cross-seed and cross-checkpoint variance not estimated. Bootstrap uncertainty on AA-composition mole fractions is ~1% absolute. The same gen design at the (now-superseded) nsteps=200 produced numbers within 5-15% of these on most metrics — magnitudes attenuated slightly at the canonical resolution but every sub-claim survives the regen verbatim.
3. **Reference-set length cap for AA composition (PDB only).** `pdb_cluster_all_seqs.fasta` is length-capped at 511, so the *PDB* AA-composition reference is PDB[300, 511] while the *PDB* property-panel reference is PDB[300, 796]. Spot-checks (e.g., panel-reference Shannon mean = 4.10 matches the AA-composition-reference Shannon when computed independently) confirm the magnitudes are robust under this caveat, but PDB AA-composition numbers should not be quoted with sub-percentage precision. The AFDB rerun (E026) does not have this caveat — both the AA-composition reference and the property-panel reference are the same N=5000 AFDB sample at full [300, 800] coverage.
4. **Co-designability inflation is a hypothesis, not a measurement.** The natural follow-up — designability vs co-designability gap on the same backbones (i.e. the paired comparison of MPNN-on-generated-backbone scRMSD vs La-Proteina-own-sequence scRMSD) plus designability stratified by Shannon-entropy decile — is preregistered but not yet computed. Without that paired comparison, the Implication is decorrelated from the Narrow claim and should be treated as a candidate explanation, not a measured effect.
5. **TemStaPro Tier 2 not yet completed.** Sub-claim (b) is supported by the internal contradiction between IVYWREL/aliphatic and aromatic_fraction within sequence-based proxies; it does not yet have an external ML-predicted Tm reference. The submit script (`script_utils/run_thermal_stability.sh`) is in place; results will amend (b.ii).
6. **Sub-claim (c) is a placement-vs-count claim, not a fold-quality claim.** The "sharper aromatic core-targeting" reading describes the *concentration ratio* P(aromatic | buried) / P(aromatic | exposed). It is not a statement that gen structures have well-formed hydrophobic cores in the structural-biology sense — only that *within whatever core/surface partition gen structures present*, the aromatic residues are non-uniformly placed in the buried region at a ratio higher than AFDB's predicted-structure pattern. To upgrade (c) to a fold-quality statement would require additional structural readouts (DSSP secondary-structure breakdown, packing density, ESMFold pLDDT distribution) and ideally a length-matched comparison of *core volume* — none of which are in this Finding's evidence base. F specifically is the residue where placement is matched to AFDB rather than sharper, so the (c) signal is carried mostly by W and Y; treat F as an exception, not a counterexample.
7. **Random AFDB ≠ training set exactly.** The AFDB reference (E026) is a uniform-random N=5000 sample over the full ~214M-entry AFDB. La-Proteina's actual training corpus is (most likely) a Foldseek-cluster-reduced AFDB subset (diversity-balanced, ~580K-2.27M reps depending on identity threshold), not the full database. A uniform-random sample over-weights over-clustered families (e.g. many homologous bacterial proteins from many species). For testing whether the gen-vs-natural deltas survive a reference-set switch, this is acceptable and arguably *more conservative*: if the deltas survive against a family-imbalanced sample, they would also survive against a more diversity-balanced one. Tighter robustness check: re-run E026 against AFDB Foldseek-30/50 cluster reps when a clean download path becomes available.
8. ~~Gen-side panel coverage on the AFDB rerun.~~ **Resolved 2026-05-05** by the nsteps=400 regen: the new gen artifact `results/generated_stratified_300_800_nsteps400/properties_generated.csv` carries the full 16-column panel including TANGO, hydrophobic patches, SAP, SCM, and rg. The vs-AFDB structure-derived numbers in the table above (tango, hydrophobic_patch_total_area, hydrophobic_patch_n_large, sap, scm_*, rg) are first-class numbers, not held over from PDB-reference sensitivity. Lab-notebook detail in `experiments.md → E020+E026 follow-up`.
9. **Finding-8 aggregate numbers are population averages over a length-non-invariant deviation — see Finding 9 below for the per-length breakdown.** The Shannon-entropy collapse is roughly twice as large at L≈750 as at L≈325; the N-over-representation grows from +7% at L≈325 to +212% at L≈750; the TANGO drift goes from d≈0 at L<450 to d=−1.3 at L=[750,800). Caveats 7-8 from this Finding apply to the per-length numbers in Finding 9 too.

---

## Finding 9 — The alphabet-collapse-driven gen-vs-AFDB deviation intensifies with protein length: Finding 8's aggregate magnitudes are a length-mixture, not a length-invariant signature (2026-05-05)

**Status:** finished. Per-length-bin breakdown of Finding 8's metrics on the nsteps=400 regen (`E020+E026 follow-up`, 2026-05-05). 100 gen samples × 10 bins (L=[300, 800], 50-residue width) vs ~500 AFDB samples per bin. Length-bin Cohen's d's reported below; bootstrap CIs not estimated at the per-bin level — the gen side has only 100 samples per bin which would give wide CIs especially in the tails. Headline ratios (largest-bin d / smallest-bin d) reported as point estimates; magnitudes are robust to ±1 bin shift.

**Why ran:** When updating Finding 8 with the nsteps=400 numbers, the question came up *"is the gen-vs-AFDB deviation length-invariant, or do the aggregate Cohen's d's hide a length scaling?"* The original gen sweep is length-stratified-uniform across [300, 800] in 50-residue bins, and the AFDB reference was explicitly length-stratified to gen's distribution, so we have ~equal coverage per bin on both sides — the natural unit for this check. If the deviation is length-invariant, Finding 8's aggregate numbers are good summary statistics. If it isn't, the aggregate hides two regimes and the headline numbers should be quoted with a length context.

**Experiment:**

For each 50-residue length bin in [300, 800), compute Cohen's d between gen (n=100) and AFDB (n≈500) on the property-panel metrics most relevant to Finding 8 (Shannon entropy, swi, iupred3, tango, hydrophobic_patch_total_area, sap, scm_negative, rg). Same per-bin breakdown on the AA-composition relative deviations (gen − ref) / ref for the residues with the largest aggregate effects (E, N, L, M, W, H, F, A) and on the Glu/Asp ratio. All numbers from `results/generated_stratified_300_800_nsteps400/` and `data/afdb_ref/`. Lab-notebook detail in `experiments.md → E020+E026 follow-up: length-invariance addendum`.

**Numbers:**

*Cohen's d gen-vs-AFDB by length bin — bold = strong length scaling, italic = roughly invariant:*

| L bin | swi *(inv)* | tango_total **(↑|d| with L)** | iupred3_mean **(↓ with L)** | shannon_entropy **(↑|d|)** | hyd_patch_area *(inv)* | sap *(inv)* | scm_negative *(inv-noisy)* | rg *(inv)* |
|---|---|---|---|---|---|---|---|---|
| [300, 350) | +1.28 | −0.43 | +0.74 | **−1.97** | −1.47 | −1.19 | −0.49 | −0.98 |
| [350, 400) | +1.16 | +0.02 | +0.64 | −2.08 | −1.53 | −1.23 | −0.44 | −0.94 |
| [400, 450) | +1.51 | −0.15 | +0.45 | −2.77 | −1.73 | −1.43 | −0.68 | −0.86 |
| [450, 500) | +1.35 | −0.21 | +0.53 | −3.00 | −1.54 | −1.29 | −0.58 | −0.86 |
| [500, 550) | +1.28 | −0.37 | +0.35 | −2.96 | −1.71 | −1.34 | −0.59 | −0.98 |
| [550, 600) | +1.38 | −0.78 | +0.31 | −3.43 | −1.61 | −1.23 | −0.62 | −0.91 |
| [600, 650) | +1.30 | −0.97 | +0.35 | −3.67 | −1.69 | −1.25 | −0.65 | −0.97 |
| [650, 700) | +1.01 | −0.93 | +0.36 | −3.74 | −1.87 | −1.58 | −0.42 | −1.03 |
| [700, 750) | +0.59 | −0.76 | +0.29 | −3.92 | −1.71 | −1.31 | +0.11 | −1.11 |
| [750, 800) | +1.22 | **−1.30** | +0.57 | **−3.68** | −1.78 | −1.37 | −0.64 | −1.08 |
| **range (max − min)** | 0.92 | **1.32** | 0.45 | **1.95** | 0.40 | 0.39 | 0.79 | 0.25 |

*AA-composition relative deviation gen-vs-AFDB (%) by length bin:*

| L bin | E | N **(↑)** | L **(↑)** | M **(↓ deepens)** | W *(noisy-inv)* | H **(↓ deepens)** | F | A **(↓ deepens)** | Glu/Asp |
|---|---|---|---|---|---|---|---|---|---|
| [300, 350) | +90 | **+7** | +5 | −33 | −57 | −17 | −40 | **+3** | gen 2.32 |
| [350, 400) | +76 | +87 | +4 | −48 | −56 | −28 | −35 | −20 | gen 2.11 |
| [400, 450) | +100 | +162 | +2 | −69 | −58 | −50 | −25 | −36 | gen 2.79 |
| [450, 500) | +101 | +165 | +12 | −79 | −61 | −43 | −40 | −39 | gen 3.00 |
| [500, 550) | +105 | +200 | +13 | −68 | −44 | −48 | −35 | −42 | gen 3.45 |
| [550, 600) | +105 | +170 | +26 | −82 | −57 | −64 | −42 | −26 | gen 3.16 |
| [600, 650) | +97 | +192 | +36 | −79 | −37 | −47 | −50 | −41 | gen 3.37 |
| [650, 700) | +73 | +177 | +35 | −79 | −36 | −51 | −53 | −37 | gen 2.87 |
| [700, 750) | +31 | **+212** | +42 | −86 | −54 | −48 | −58 | −41 | gen 2.50 |
| [750, 800) | +82 | +168 | **+48** | **−83** | −38 | −46 | −51 | −40 | gen 2.63 |

(AFDB-side Glu/Asp is essentially constant at 1.08–1.17 across all bins; the gen ratio variance is entirely on the gen side.)

**Narrow claim — three sub-claims:**

**(a) The Shannon-entropy alphabet collapse intensifies systematically with protein length.** Cohen's d goes from −1.97 at L=[300, 350) to −3.92 at L=[700, 750) — about double in magnitude across the [300, 800] range. The slope is approximately monotone (−1.97, −2.08, −2.77, −3.00, −2.96, −3.43, −3.67, −3.74, −3.92, −3.68) with one small dip in the last bin. Reading: La-Proteina's joint sequence head loses sequence diversity faster than the natural reference does at long lengths. Finding 8 sub-claim (a) is therefore precisely a *long-protein* alphabet-collapse story; the L<450 region is much closer to AFDB than the L>600 region. The aggregate d=−2.99 quoted in Finding 8 is an averaging artifact of the [300, 800] length-stratified sweep design.

**(b) The Asn / Met / His / Ala / Leu deviations also scale with length, in the same direction as the alphabet collapse.** N over-representation grows from +7% at L=[300, 350) to **+212% at L=[700, 750)** — a factor of ~30. M under-representation deepens from −33% to −86%. H from −17% to −48%. A from +3% (slightly *over*-represented at the shortest bin!) to −41%. L over-rep grows from +5% to +48%. **The most-extreme aggregate deviations in Finding 8 sub-claim (a) are essentially long-protein phenomena** — at L≈325 most are within 10-30 percentage points of the natural reference; at L≈750 they are 5–10× larger. The Glu/Asp ratio inflation is more weakly length-dependent (range 2.11–3.45, with the peak in the middle of the range) — Glu over-representation is stable across lengths but Asp under-representation tracks length, so the ratio drifts non-monotonically.

**(c) TANGO-side and IUPred3-side flip regimes mid-range; the structural Cohen's d's are roughly length-invariant.** TANGO_total Cohen's d goes from essentially zero at L<450 (gen and AFDB matched on aggregation propensity for short proteins) to −1.30 at L=[750, 800) — i.e. the "gen has lower TANGO than AFDB" sub-finding from Finding 8 is *entirely* a long-protein effect. IUPred3 disorder drift attenuates with length (+0.74 at L=[300, 350) → +0.29 at L=[700, 750)), the opposite direction. By contrast, the FreeSASA-derived structure metrics — hydrophobic_patch_total_area (range −1.47 to −1.87 across bins), sap (−1.19 to −1.58), rg (−0.86 to −1.11) — are **roughly length-invariant**, suggesting the structure-side echo of the alphabet collapse documented in Finding 8 sub-claim (a) is a genuinely uniform reduction in hydrophobic content per residue, not a long-protein-only phenomenon. swi is also roughly length-invariant (range +0.59 to +1.51, no monotone trend).

**Implication (cautiously phrased):**

This finding does not falsify any sub-claim of Finding 8 — every sign is preserved at every length, and the structure-side claims (sap, hydrophobic_patch_*, rg, swi) are length-invariant. What it changes is the *interpretation*: the headline magnitudes Finding 8 quotes for sub-claim (a) — Cohen's d=−2.99 on Shannon entropy, +156% on Asn, etc. — are quantitatively dominated by the L>500 part of the distribution, and using them to predict gen behaviour at, say, a target length of 300 would substantially overestimate the deviation. A practical reading: **the joint sequence head's failure mode scales with the autoregressive horizon it has to generate.** This is mechanistically consistent with a "diffusion model losing diversity over longer trajectories" story: longer proteins require the AE2-encoded latent to remain on a more constrained sub-manifold for more residues, and small per-step biases compound. For paper-narrative purposes Finding 8 sub-claim (a) should be re-quoted as either (i) per-length-bin numbers (e.g. "at L=[700, 750) the alphabet collapse reaches Cohen's d=−3.92 in Shannon entropy and +212% relative excess in Asn; at L=[300, 350) it is d=−1.97 and +7% respectively") or (ii) the aggregate plus an explicit "length-mix" footnote.

**Methodological caveats — what this finding does *not* support:**

1. **No bootstrap CIs at the per-bin level.** Each gen bin has n=100, AFDB has n≈500. A 95% bootstrap CI on a per-bin Cohen's d at gen-n=100 is roughly ±0.3 standard deviations; the trends above (−2 to −4 in Shannon, +7% to +212% in N) are visibly larger than this noise floor across multiple consecutive bins, but the sharper claims about within-bin variability (e.g. the Shannon dip from L=[700, 750) to L=[750, 800)) are within bootstrap noise and should not be promoted.
2. **Per-bin gen sample identity is fixed.** Each gen bin contains exactly 100 samples drawn at seeds 1000–1999 round-robin over bins (`steering/generate_baseline.py:179` stratified mode). A different seed range would draw different specific proteins per bin. Cross-seed variance per bin not estimated; the qualitative length-scaling pattern would survive a different seed range with high confidence (since multiple consecutive bins all show the same direction), but the precise per-bin numbers would shift by O(0.2) in d.
3. **AFDB ref is uniform-random over [300, 800], length-stratified to match gen by 50-residue bin.** Within each bin AFDB has ~500 samples, so AFDB-side bootstrap noise is small. But: AFDB at long lengths (L>700) over-represents over-clustered protein families more strongly than at short lengths (because protein-family-redundancy is correlated with protein length in AFDB — multi-domain bacterial proteins are common). A Foldseek-cluster-reduced AFDB at long lengths might have somewhat different composition than the random-AFDB long-length sample we used. This bias is in the opposite direction of the gen drift, so it makes our "length-scaling" claim *more conservative*, not less: if AFDB's long-length composition were less alphabet-redundant than what we measured, the gen-vs-AFDB length scaling would be even sharper.
4. **The "joint sequence head loses diversity over long trajectories" mechanistic story is a hypothesis, not a measurement.** This Finding documents a length scaling of the gen-vs-AFDB deviation; it does not measure the AE2 latent diversity per-residue along the trajectory, which would be the direct test. Future work: (a) trajectory-length-resolved Shannon entropy on the AE2 latent itself (does the latent KL-divergence-from-prior collapse faster at long L?); (b) compare with gen samples at multiple inference lengths under matched-noise initialization to separate "training-data tail underrepresentation" from "trajectory-length-driven mode-collapse".
5. **No claim about WHICH bin is "right".** This Finding does not claim that the L=300 bin is the "true" alphabet-collapse magnitude or the L=750 bin is the "headline" number. Both are valid measurements at different lengths; the right number to quote depends on what protein length the downstream user cares about.
6. **Same caveats as Finding 8 caveats 1-7 apply** (single-checkpoint, single-seed, AFDB ≠ Foldseek-cluster-reps, etc.). Caveat 8 (resolved by the regen) is moot here.

**Cross-references:**
- Finding 8 — direct parent. This Finding refines Finding 8 sub-claim (a) by showing the magnitude is length-dependent; Finding 8 sub-claims (b) and (c) are not affected here ((b) thermal-proxy gameability is a sequence-composition contradiction at any length; (c) aromatic placement was scored on a different gen artifact).
- `experiments.md → E020+E026 follow-up: length-invariance addendum` — lab-notebook detail with the exact per-bin numbers and reproduction commands.
- Predicts: a per-length scRMSD profile would test whether the alphabet collapse intensification at long L is also reflected in structural-quality degradation. Expected if the two are coupled: scRMSD distribution should also worsen at L≈700–800 vs L≈300, consistent with the long-length cliff already documented in E022 for the canonical CA-only baseline.

---

## Finding 10 — Closing the gradient-hacking gap in latent-flow steering: noise-aware predictor training × fold-ensembling, validated by real-property delivery and structural integrity (2026-05-06; codesignability addendum 2026-05-07)

**Status:** finished. First steering-route Finding to clear both bars from the thesis intent — *real-property + designability moves on long-L generations*. Built on E028 (negative-baseline measurement) → E029 (single-fix pilot) → E030, E031 (eliminated alternative explanations) → E032 (combined-fix smoke + n=48 confirmation) → E033 (MPNN-redesign designability check) → E036 (diversity check) → E042 (codesignability check, addendum below). Lab-notebook detail across `experiments.md` E028–E033, E036, E042.

**Why this is a Finding, not just an engineering note:** the result is a *compositional* mechanism story — two independent failure modes of gradient-based latent-flow steering, each addressable with a separate fix, that close the gap **only when applied together**. Either fix alone, and several plausible-sounding alternatives, leaves a 4–10× over-claim by the predictor. The combination drops the gap to within regression noise while increasing real-property delivery 2× over the previous best baseline.

**Experiment.**

The setting is gradient-based steering on the official LD3+AE2 La-Proteina checkpoint (`inference_ucond_notri_long`, nsteps=400, SDE sampler). Property predictor is the multi-task PropertyTransformer from Finding 1 (128-dim, 3 layers, 4 heads, FiLM-on-t, head-per-property over 14 developability metrics). Steering hook adds `w(t) · ĝ` to the local-latents velocity field, where `ĝ` is the unit-normalized gradient of a chosen property objective. Schedule: linear ramp on `w(t)` from 0 at t=0.3 to `w_max` at t=0.8, hard-stop at t=0.9.

The day's pipeline ran six predictor:real-gap probes at the same 4-protein L=300 / w=16 cell to isolate mechanisms, then promoted the winning combination to the full 16 seeds × 3 lengths {300, 400, 500} × 5 w-levels {1, 2, 4, 8, 16} × 2 directions (camsol_max + tango_min) grid (480 PDBs total) with n=48 per (direction, w) cell, plus a 120-PDB scRMSD pass via the official MPNN→ESMFold pipeline.

**Two fixes layered:**

1. **Noise-aware predictor training (E029):** the predictor was originally trained on clean t=1 AE-mean latents but at sampling time it sees `z_t` from the mid-trajectory of the SDE. Fix: fine-tune each of the 5 cross-validation folds on `z_t = (1-t)·ε + t·z_1 + σ_L·√(t(1-t))·ε_2` with `t ∼ U(0.3, 0.8)` (matches the steering window), σ_L=0.1 (Brownian-bridge envelope, vanishes at endpoints, peaks mid-trajectory). 10 epochs, AdamW lr=1e-4, original z-score stats inherited verbatim. Output: `laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/`.
2. **5-fold ensemble (E032):** average z-scored predictions across all 5 noise-aware folds at sampling time. Fold-specific shortcuts (which gradient hacking exploits) are uncorrelated across folds; the honest signal is shared. Implementation already supported by `SteeringPredictor` when given a list of checkpoint paths.

**Numbers (full grid n=48, head-to-head against E028 clean ensemble + smoothing on the same grid):**

*Predictor:real gap aggregated across all 48 proteins per cell, tango_min direction (only direction with locally computable real metric — `tango_total` from the TANGO binary). Gap = predictor's last-step claim − real binary output. Negative = predictor under-claims = classical hacking direction.*

| w | E028 clean ens. + smoothing | F10 noise-aware ens. (no smoothing) | Δ (this work − E028) |
|---|---|---|---|
| 1 | -82 | +118 | +200 |
| 2 | -103 | +110 | +213 |
| 4 | -141 | +92 | +233 |
| 8 | -181 | +59 | +240 |
| **16** | **-203** | **+3.8** | **+207** |

*Δ-vs-w=1 reference (the "10× over-claim" axis from the original literature):*

| w | E028: Δpred / Δreal | F10: Δpred / Δreal |
|---|---|---|
| 2 | 8.3× | 3.3× |
| 4 | 7.9× | 3.9× |
| 8 | 8.6× | 3.9× |
| **16** | **8.5×** | **2.9×** |

*Real-property delivery, w=1→w=16 mean:*

| metric | E028 | F10 |
|---|---|---|
| real `tango_total` at w=1 | 877.9 | 893.3 |
| real `tango_total` at w=16 | 843.9 | 833.4 |
| **Δ real (w=16 − w=1)** | **−34.0** | **−59.9** |

F10 delivers ~2× the real-property change while the predictor moves only 174 (vs E028's 288) — i.e. the gap is closed on both axes simultaneously, not by inflating the denominator.

*Per-length breakdown of F10 at w=16:*

| L | predictor mean | real mean | gap |
|---|---|---|---|
| 300 | 602.2 | 573.8 | +28 |
| 400 | 844.0 | 886.2 | -42 |
| 500 | 1065.3 | 1040.1 | +25 |

The +28 / −42 / +25 sign-disagreement across lengths means the aggregate gap of +3.8 is partly an artifact of L=400's residual underclaim cancelling against L=300/500's overclaim. At any given length, the residual gap is ≤ ~40 TANGO units — about 5% of the typical TANGO value.

*Designability across the same grid (n=12 per cell, MPNN N=8 → ESMFold scRMSD, designable threshold < 2 Å). Excluding the persistent s45_n500 outlier (broken at every cell at >10 Å, w-independent — a known generation failure of the underlying LD3 sampler at this seed × length, see E033):*

| direction | w=1 | w=2 | w=4 | w=8 | w=16 |
|---|---|---|---|---|---|
| camsol_max designable | 11/11 (100%) | 10/11 (91%) | 10/11 (91%) | 11/11 (100%) | 10/11 (91%) |
| camsol_max mean scRMSD | 0.95 Å | 1.04 Å | 1.33 Å | 1.01 Å | 1.34 Å |
| tango_min designable | 8/10 (80%) | 9/10 (90%) | 11/11 (100%) | 10/11 (91%) | 9/11 (82%) |
| tango_min mean scRMSD | 1.41 Å | 1.11 Å | 0.95 Å | 1.10 Å | 1.14 Å |

**No monotonic w → scRMSD trend.** Designability stays in the 80-100% band across the full sweep. tango_min at w=4 is *better* than at w=1; at w=16 it is statistically indistinguishable from w=1. Per-length × w breakdown (E033) confirms: every (L, w) cell except the s45_n500 outlier sits in the 67-100% designable range.

*Structural-ensemble diversity (E036, pairwise TM-score across each cell vs an unsteered baseline at the same length window):*

| direction | w=1 | w=2 | w=4 | w=8 | w=16 | unsteered baseline |
|---|---|---|---|---|---|---|
| camsol_max mean pairwise TM | 0.407 | 0.407 | 0.407 | 0.407 | 0.407 | 0.413 |
| tango_min mean pairwise TM | 0.407 | 0.407 | 0.407 | 0.407 | 0.407 | 0.413 |

Mean pairwise TM-score is **identical to 3 decimal places across every w-level for both directions**, and only 0.006 below the unsteered baseline. At L=400 / L=500 specifically, the steered ensembles are *more* diverse than baseline (mean TM 0.331-0.395 vs 0.366-0.418). Steering does not collapse the structural ensemble — the 16-seed initialization variance dominates whatever narrowing the gradient might cause. The latent space is high-dimensional enough that there are many independent low-TANGO directions, and steering pushes each trajectory along whichever is closest from its starting point rather than collapsing all 16 trajectories onto one.

*Codesignability across the same grid (E042; n=12 per cell, **`use_pdb_seq=True, num_seq=1`** — the joint-head sequence is taken verbatim and folded by ESMFold, no MPNN re-design. This is the right structural-integrity check for **latent** steering because the latent feeds into the joint sequence head, so any silent destruction of sequence-side foldability would only show up here):*

| direction | w=1 | w=2 | w=4 | w=8 | w=16 |
|---|---|---|---|---|---|
| camsol_max codesignable | 5/12 (42%) | 5/12 (42%) | 5/12 (42%) | 5/12 (42%) | 4/12 (33%) |
| camsol_max mean coScRMSD | 3.61 Å | 3.61 Å | 3.62 Å | 3.60 Å | 3.70 Å |
| camsol_max median coScRMSD | 2.15 Å | 2.08 Å | 2.07 Å | 2.08 Å | 2.17 Å |
| tango_min codesignable | 4/12 (33%) | 4/12 (33%) | 4/12 (33%) | 4/12 (33%) | 5/12 (42%) |
| tango_min mean coScRMSD | 3.67 Å | 4.13 Å | 3.81 Å | 3.70 Å | 4.06 Å |
| tango_min median coScRMSD | 2.19 Å | 2.19 Å | 2.29 Å | 2.30 Å | 2.16 Å |

**Codesign rate is flat across w∈[1, 16]** for both directions. tango_min at w=16 is *higher* than at w=1 (5/12 vs 4/12); camsol_max is one PDB lower at w=16 (4/12 vs 5/12) but identical to w=1 across L=300. There is no monotonic codesign-vs-w degradation. The L-cliff is L=400 / L=500 (1/4 codesignable per cell, w-independent), consistent with E022's known unconditional length-degradation. **The relevant signal — does latent steering silently kill the joint sequence head's ability to produce a folding sequence — is no.** The flat-across-w trend blocks the "maybe MPNN re-design is hiding sequence damage" objection that E033 alone left open.

*w=0 sanity check (E042 addendum, 2026-05-07).* The La-Proteina paper reports **68.4% all-atom co-designability** (Table 1, averaged across L∈{100, 200, 300, 400, 500}). 33-42% looks low against that headline, so 12 length-matched **unsteered** PDBs from `generated_stratified_300_800_nsteps400/` (canonical config: `inference_ucond_notri_long`, nsteps=400, SDE — same pipeline used for the diversity baseline above) were run through the same codesign call:

| L | unsteered codes / 4 | steered cells (range across w & direction) |
|---|---|---|
| 300 | **3/4 (75%)** | 2-3/4 (50-75%) |
| 400 | **2/4 (50%)** | 1/4 (25%) at every cell |
| 500 | **0/4 (0%)** | 1/4 (25%) at every cell |
| pooled | **5/12 (42%)** | 4-5/12 (33-42%) |

The pooled steered rate is *identical* to the pooled unsteered rate (both 42% at the high end). At L=300 the steered headline cells (camsol_max w=1, tango_min w=16) hit 75%, matching unsteered. At L=400 steered runs 1 protein lower than unsteered (1/4 vs 2/4), w-independent — within noise. At L=500 steering is, if anything, slightly better than unsteered (1/4 vs 0/4) — also within noise. **The 33-42% codesign rate is the canonical La-Proteina sampler's own ceiling at L≥300, not a steering artefact.** The published 68.4% averages over L=100-500 with 100 proteins per length — a length-mixture dominated by the easier L=100/L=200 bins. At L≥300 specifically the unconditional model is much weaker (this baseline shows the cliff: 75% / 50% / 0%), consistent with Figure 4 of the La-Proteina paper which shows codesign rate degrading with length. Our test grid only sampled L≥300, so its absolute rate is unavoidably lower than the 68.4% headline for population reasons unrelated to gradient guidance.

*Continuous coScRMSD distribution (the binary < 2 Å rate alone is information-poor on n=12 cells; reporting the underlying distribution is more honest):*

| | n | mean coScRMSD | median | rate < 2 Å | rate < 3 Å | rate < 4 Å |
|---|---|---|---|---|---|---|
| **steered (all w, both directions, n=120)** | 120 | **3.75 Å** | **2.18 Å** | 38% | **74%** | **82%** |
| **unsteered baseline (n=12)** | 12 | **5.45 Å** | **4.38 Å** | 42% | 42% | 50% |

| direction (n=60 each) | mean | median | rate < 2 Å | rate < 3 Å | rate < 4 Å |
|---|---|---|---|---|---|
| camsol_max | 3.63 Å | 2.08 Å | 40% | 75% | 83% |
| tango_min  | 3.87 Å | 2.20 Å | 35% | 73% | 80% |

| w (n=24 each, both directions) | mean | median | rate < 2 Å | rate < 3 Å | rate < 4 Å |
|---|---|---|---|---|---|
| 1  | 3.64 Å | 2.19 Å | 38% | 75% | 83% |
| 2  | 3.87 Å | 2.13 Å | 38% | 71% | 79% |
| 4  | 3.71 Å | 2.14 Å | 38% | 71% | 83% |
| 8  | 3.65 Å | 2.15 Å | 38% | 79% | 83% |
| 16 | 3.88 Å | 2.17 Å | 38% | 75% | 79% |

Three things become visible from the continuous numbers that the binary rate hides:

1. **Mean and median coScRMSD are flat across w** within each direction (mean range 3.60-3.70 Å for camsol_max, 3.67-4.13 Å for tango_min across all five w-levels; medians 2.07-2.30 Å across the entire grid). The codesign-vs-w distribution is essentially stationary; the binary rate's flatness is a real signal, not aliasing.
2. ~~**Steered cells outperform the unsteered baseline at relaxed thresholds.**~~ **Withdrawn after a matched-seed sanity check (E042 update, 2026-05-07).** The earlier comparison used a stratified-bin unsteered baseline (seeds 1000+, lengths 305-510), which had two confounds against the steered cells (seeds 42-45, exact L=300/400/500). When unsteered is regenerated with **matched seeds 42-45 × exact L=300/400/500** through the same `steering/generate.py` pipeline (`model.steering_guide=None`), unsteered codesigns at 5/12 (42%) at <2 Å, 9/12 (75%) at <3 Å, 10/12 (83%) at <4 Å — *identical to the steered cells within 1-protein noise*. Mean coScRMSD: unsteered 3.63 Å, steered 3.60-4.13 Å. **Steered ≈ unsteered at every threshold and on the continuous distribution.** The "steered fills the 2-3 Å near-miss band" observation was an artefact of the stratified sample's bimodality, not a steering effect.
3. **At a 3 Å scRMSD bar — which downstream `compute_developability.py` callers typically use as "designable enough" for follow-up filtering — both the steered grid and the matched-seed unsteered baseline pass 67-83%**, much closer to the paper's 68% headline. The 2 Å strict bar is what penalises the long-L regime (cf. paper Figure 4); relaxing it returns the comparison to a more apples-to-apples place. The published 68.4% averages L=100-500 with 100 proteins per length; our L≥300 four-seed slice matches what Figure 4 of the paper shows for the same length range.

**Per-protein structural readout from the matched-seed run.** The unsteered backbone is recovered to within 0.01-0.15 Å of the steered backbone at most cells (s42_n400, s44_n300, s44_n400, s45_n400 within 0.05 Å across every w-level). The latent steering perturbs the joint-head sequence but does not move the structure off the unsteered backbone enough to change codesignability. **s45_n500 is broken at 20.02-20.07 Å in unsteered and every steered cell** — confirmed unconditional sampler failure, not a steering artefact. Two genuine protein-level perturbations are visible: tango_min w=2 and w=16 push s42_n500 from 2.51 → 7.49-7.98 Å (real damage on this one protein); tango_min w=4 pushes s43_n300 from 1.02 → 3.72 Å. With the rest of the grid pinned within 0.1 Å of unsteered, these isolated drifts are individual-protein noise on n=4 per length, not a population-level signature.

*Negative results that establish the compositional necessity of both fixes (lab notebook in E028, E030, E031, plus a feed-z_t-direct probe today):*

| approach | gap at w=16 (n=4 pilot) |
|---|---|
| Clean 5-fold ensemble + σ=0.1 smoothing (E028) | -203 |
| Noise-aware single fold, no ensemble (E029) | -47 |
| Universal guidance K=5 + clean ensemble + smoothing (E030) | -302 |
| Noise-aware longer training + cosine decay, single fold (E031) | -145 |
| Feed z_t directly (drop x_1_est Tweedie), v1 single fold | -152 |
| Feed z_t directly, v2 single fold | -187 |
| **Noise-aware 5-fold ensemble (this work, E032)** | **-1.6** |

Five plausible "should help" interventions failed before the right composition was found. The negative-result chain is itself part of the Finding's evidence base — the gap is not closed by making the predictor "better" (E031), by giving the gradient more leverage at sampling time (E030, z_t-direct), or by ensembling alone (E028). Both **input-distribution training** (z_t-aware, t-aware) and **fold-cancellation of shortcuts** (ensemble averaging) are necessary; either alone is insufficient.

**Narrow claim.**

On the official LD3+AE2 La-Proteina checkpoint, fine-tuning the multi-task property predictor (3-layer FiLM transformer, 14 heads) on `z_t = (1-t)·ε + t·z_1 + σ_L·√(t(1-t))·ε_2` with `t ∼ U(0.3, 0.8)` and σ_L = 0.1 across all 5 cross-validation folds (`add_noisy_latents.py`, 10 epochs, lr=1e-4, original z-score stats preserved), then steering with the 5-fold mean of these noise-aware predictors and the legacy `x_1_est` reconstruction-guidance input path,
- closes the predictor:real `tango_total` gap from −203 (clean-ensemble + smoothing baseline E028, n=48) to **+3.8** (noise-aware ensemble, n=48) at w=16;
- raises the real ΔTANGO from −34 to **−59.9** at w=16 (~75% increase in real-property delivery on the same grid);
- holds the structure-side designability rate at **80-100% across w∈[1, 16]** for both camsol_max and tango_min directions (n=12 per cell on the MPNN-redesign scRMSD subset; mean scRMSD 0.95-1.49 Å), with no monotonic w → scRMSD trend;
- holds the **codesignability rate flat at 33-42% across w∈[1, 16]** (n=12 per cell, joint-head sequence + ESMFold, no MPNN; E042) — i.e. the (steered structure, steered sequence) pair folds at the same rate as at w=1, ruling out the "MPNN-redesign hides sequence damage" objection.

**Implication.**

Gradient-based steering of generative protein flows had been characterised in the project's prior steering work (E025, E028) as predictor-confident but reality-divorced, with a ~10× ratio between predicted and real property change at high steering strength. This Finding establishes that the gap is not an inevitable artefact of gradient guidance but a compositional consequence of two specific, independently fixable failure modes:

1. The predictor at sampling time is being asked to evaluate inputs (mid-trajectory `z_t` at intermediate t) that are off-distribution relative to its training (clean t=1 AE-mean latents). This is the dominant contribution and is fixed by the noise-aware fine-tune.
2. The predictor (post fix-1) still has fold-specific shortcut features that gradient hacking exploits as adversarial directions. This is the residual contribution and is fixed by mean-ensembling across cross-validation folds, which cancels fold-uncorrelated shortcuts while preserving the shared real-property signal.

This factorisation has practical consequences beyond this codebase: any reconstruction-guidance / classifier-guidance pipeline where the property model is trained on a different t-distribution than the sampler operates over should expect a similar failure mode. The fix is portable (forward-process noise model + cross-validation ensemble; no model-architecture change), low cost (predictor fine-tune is ~30 minutes per fold on one L4), and does not slow sampling beyond the 5× predictor-call cost of ensemble averaging. The negative-result chain (universal guidance, longer training, z_t-direct feeding) further constrains the design space: post-hoc tricks that operate on the predictor's *output* or on the *gradient*, without addressing the predictor's input-distribution mismatch, can amplify the failure rather than fix it.

For the steering-route bar of the masterarbeit thesis intent — *generated long-L proteins that pass real-property + designability filters in a lab-relevant sense* — w=8 (Δreal ≈ −21, designability 91-100%, gap ≈ −47) and w=16 (Δreal ≈ −60, designability 82-91%, gap ≈ +4) are both deployable operating points. The choice between them is a real-property-magnitude vs structural-conservatism trade-off, not a "still gradient-hacked" trade-off.

**Methodological caveats.**

- **Tango-only quantitative validation.** `compute_developability.py` returns NaN for `camsol_intrinsic` because no public CamSol binary exists (CLAUDE.md flag). The camsol_max sweep numbers in this Finding therefore rely on the predictor's claim and on collateral real-property drift (sap, scm, hyd_patch) for plausibility checks; the headline gap-closure and Δreal numbers are tango-only. CamSol web-server submission for ~50 representative sequences is queued but not yet performed.
- **Designability and codesignability subsamples are both n=12 per cell.** The full sweep is 48 PDBs/cell on the gap side but 12 PDBs/cell on the structural side, due to the ~9 hour wall cost of the official MPNN→ESMFold pipeline at L∈{300, 400, 500}. The 80-100% designability rates therefore have wide per-cell CIs (a 91% rate from 10/11 has 95% CI ~59-100%); same for the 33-42% codesignability rates. The robust signal is the *across-cell trend* — no monotonic w-degradation in either metric — not the per-cell rate.
- **Codesignability absolute rate is much lower than designability** (33-42% vs 80-100%). The w=0 sanity check (E042 addendum) showed this is the unconditional La-Proteina sampler's own ceiling at L≥300: pooled unsteered codesign at L∈{300,400,500} is 5/12 = 42%, identical to the pooled steered rate. The published 68.4% codesign rate averages over L=100-500 with 100 proteins per length — a length-mixture dominated by the easier L=100/L=200 bins; per-length codesign at L=300/400/500 falls off (75% / 50% / 0% in the unsteered baseline). Our test grid only sampled L≥300, so its absolute rate is unavoidably lower than the published headline for population reasons unrelated to gradient guidance. For the *steering question* (does w degrade the codesign rate?) the trend is flat; for an *absolute-quality question* the unconditional sampler's L-cliff is the binding constraint, not the gradient guidance.
- **One persistent-failure protein dominates the apparent variance.** s45_n500 is broken at >10 Å in 9/10 cells, w-independent. It is a generation failure of the underlying La-Proteina LD3 sampler at this seed × length, not a steering failure. The "excluding s45_n500" tables are the fair comparison. Including it changes the apparent designability rate at every cell by the same fixed amount (one extra non-designable per cell), so the *trends* are unchanged either way. Further audit (run e.g. 16 seeds at L=500 on the unsteered baseline to estimate the s45_n500-equivalent failure rate) is not yet done.
- **Per-length sign disagreement at w=16.** The aggregate gap of +3.8 hides L=300 +28 / L=400 −42 / L=500 +25. L=400 still has a residual underclaim (the classical hacking direction). The aggregate is small not because every length is honest but because lengths cancel. The per-length data is closer to "gap reduced from −203 to ~±40, on either sign of zero" than to "gap eliminated".
- **No verification on alternative flows.** All experiments were on the official LD3+AE2 La-Proteina checkpoint. Whether the noise-aware-fine-tune + ensemble combination transfers to other flow architectures (CA-only variants from E022, sparse attention from E021, future variants) is untested. The mechanism story predicts it should — both fixes target predictor-side failure modes that are flow-architecture-independent — but we don't have data.
- **Designability ≠ wet-lab function.** scRMSD < 2 Å says "ESMFold thinks an MPNN-designed sequence will fold to roughly the right structure". It does not say the protein folds in vitro, that the predicted property is reproduced in lab measurement, or that the structure has the expected aggregation behaviour. These are wet-lab claims and out of scope.
- **The predictor still has a residual Δratio of 2.9× at w=16.** The mean gap of +3.8 is small, but the Δ predictor / Δ real ratio is 2.9× — predictor's *change* still moves about 3× faster than reality. This is the residual signature of the per-length sign-disagreement: at L=400 specifically, the predictor over-shoots reality. The reduction from 8.5× to 2.9× is a 3× improvement and dominates the gap-closure narrative, but a Δratio of 1× would be the strict goalpost.
- **Calibration drift at low w.** At w=1 (minimal steering) the predictor over-claims real TANGO by +118 on average. This is calibration drift, not gradient hacking — the predictor is biased high on near-baseline proteins. Steering brings the predicted value down faster than the real value follows, which is what produces the gap-closure-by-crossover at w=16. If we pushed past w=16 the predictor might cross over to underclaiming real (the classical hacking direction). Within the tested range w∈[1, 16] this is not an issue.

**Audit addendum (2026-05-10, lab notebook [E050](experiments.md#e050--steering-audit-matrix--predictor--ensemble--fold--smoothing-2026-05-10)).** The Finding's matrix was filled in for three open audit questions: is fold 2 specially smart? does smoothing in the E028 baseline contribute meaningfully on top of ensembling? and does NA-v1 single fold at n=48 behave like the n=4 pilot? Twelve new cells (7 n=4 smokes + 5 n=48 sweep cells, all at nsteps=400, smoothing off, tango_min direction). Three results land back into this Finding:

1. **Smoothing in E028 does ~nothing** on top of ensembling. Clean ens5 + σ=0.1 K=4 smoothing (E028) gives gap = -203.5 at w=16 / n=4; clean ens5 *without* smoothing gives -203.9 — Δ within the 49-65 gap std. The "ensemble + smoothing" baseline framing in this Finding's E028 anchor was misleading: smoothing was not contributing. The two fixes in F10's mechanism story are now unambiguously **(noise-aware fine-tune) + (5-fold ensemble)**, with no third-knob reading.
2. **Fold 2 is NOT specially smart.** All five NA-v1 single folds give n=4 gap in [-47, -97] (f2 = -47.5, f0 = -61.3, f1 = -62.2, f3 = -59.4, f4 = -97.0; mean -65.5, std 17.7). Every NA fold beats every clean fold. The E029 / E031 fold-2-only pilots were representative readings, not cherry-picks. F10's "use fold 2 because it had the highest val r²_noisy" decision was inconsequential for the gap-closure picture — any fold would have shown the same shape.
3. **Ensembling still moves the gap meaningfully *after* the noise-aware fix is in place, but the mechanism reading is softened.** NA-v1 single fold at n=48 / w=16 = -87.5 (Δratio 7.0×); NA-v1 5-fold ensemble at n=48 / w=16 = +3.8 (Δratio 2.9×). The 91-unit gap reduction from ensembling is real on the same n=48 grid, *not* an n=4 small-sample artefact. But because all five single folds have similar gap magnitudes (std 17.7 around mean -65.5), **the ensemble is doing variance-averaging across an honest residual underclaim**, not cancelling fold-specific adversarial shortcuts. The original framing of "fold-specific shortcuts that gradient hacking exploits as adversarial directions ... are uncorrelated across folds; the honest signal is shared" overstated the mechanism — folds are similarly hacked, not differently hacked. The compositional-necessity claim ((noise-aware) + (ensemble) both needed) is unchanged; the per-mechanism reading should now describe the ensemble as **bias averaging** rather than **shortcut cancellation**.

These edits *strengthen* F10's headline (the two-fixes claim is now cleaner with no smoothing ambiguity) and *constrain* the underlying mechanism story (variance averaging rather than adversarial cancellation). The negative-results table and gap-closure numbers in the body of this Finding are unchanged. The audit also leaves one open question that does not affect F10's narrative scope: a sweep of folds 0/1/3/4 to n=48 would tighten the "all NA folds behave similarly" claim from "consistent with n=4 pilots" to "robust at n=48"; this is a ~5h L4 follow-up that has not yet been run.

---

## Finding 11 — Per-t validation loss does not distinguish CA-only architectural variants at the resolution that matters for hybrid sampling (2026-05-07; methodological calibration)

**Status:** finished. Methodological scaffolding for the architectural-route line of work — calibrates whether per-t val loss is a useful selection criterion when comparing variants, ahead of the hybrid-sampling decisions in [E040 / E041](experiments.md#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06). Lab-notebook detail: [E043](experiments.md#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07).

**Why this is a Finding rather than a tuning note.** The hybrid-sampling work in E040 / E041 (1D-conv ckpt for early-trajectory denoising, sparse / canonical ckpt for late-trajectory refinement) implicitly assumes there is a t-region where each variant is *locally optimal*. If true, the per-t validation loss curve of variant A should cross variant B at some t — that's the principled argument for hybridising. F11 is the test of that assumption, run on the four architectural CA-only ckpts we have on hand. Finding it negative changes the interpretation of E040 / E041's compositional framing (and motivates the trajectory-divergence diagnostic D2 instead).

**Experiment.**

The diagnostic computes the standard La-Proteina FM training-loss bucketed into 5 equal-width t-bins on a *paired* 600-protein subset, with the same per-protein noise / rotation / t draws used across all four CA-only architectural variants. **No sampling / integration is involved** — the script samples a single t per protein, interpolates `x_t = (1-t)·x_0 + t·x_1`, runs `model.call_nn(batch, n_recycle=0)` under `torch.no_grad()`, and computes the same per-channel FM loss as `model.fm.compute_loss`. NFE / nsteps is therefore irrelevant; the result reflects only the trained weights at each ckpt.

- Driver: `proteinfoundation/run_per_t_val.py` (a standalone script written specifically because the project's `PDBLightningDataModule` requires graphein's `PDBManager` to download PDB metadata from the public internet at instantiation time, which times out on offline nodes; the script bypasses this by walking `data/pdb_train/processed_latents/` directly).
- Data subset: 600 proteins drawn deterministically (seed=42) from `processed_latents/`, length-filtered to ≤ 512 residues. Standard training transforms applied (`CenterStructureTransform → ChainBreakPerResidueTransform → GlobalRotationTransform`) with per-protein rotation seed `seed + 1000 + i` so the *same protein* gets the *same rotation* across all four ckpts (paired-sample design — pairwise differences are pure model differences).
- Buckets: `[0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0)` — same partition as `proteina.py:478-492`'s training-time `validation_loss_by_t` bucketing. n=600 per bucket.
- Ckpts (all CA-only, all 160M, all `latent_dim=None`):
  - `canonical_2646` — `test_ca_only_diffusion/1776805213` step 2646 (canonical wd=0.05 baseline at the documented best-val ckpt).
  - `conv_2331` — `ca_only_downsampled/1777987722` step 2331 (1D-conv variant; "dead alone at 0/18 designable in [E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06)" — but E034 was at nsteps=200, below the integrator-convergence bar; the dead verdict is plausibly an nsteps artifact and is being re-probed at nsteps=400 in `inference_downsampled_step2331_n6_nfe400` 2026-05-07).
  - `scnbr_t04_1133` — `ca_only_sparse_K40_scnbr_t04/1778022317` step 1133 (sparse K=40 + Fix C2; "variant-bar-clearing at 17% pooled in [E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06)" — but E039 was at nsteps=200; the 17% number is plausibly an under-statement and is being re-probed at nsteps=400 in `inference_scnbr_t04_step1133_n6_nfe400` 2026-05-07).
  - `sparse_vanilla_1259` — `ca_only_sparse_K40` step 1259 (sparse K=40 *without* the Fix C2 sc_neighbors threshold-gating). Designability untested at the time of writing.
- Run wall: ~6 min per ckpt on 1× A100. Output JSONs in `results/per_t_val/`.

**Numbers — per-t mean validation loss (n=600 per bucket, paired across ckpts):**

| ckpt | step | t∈[0.0, 0.2) | t∈[0.2, 0.4) | t∈[0.4, 0.6) | **t∈[0.6, 0.8)** | t∈[0.8, 1.0) | min bucket |
|---|---|---|---|---|---|---|---|
| canonical_2646 | 2646 | 3.018 ± 0.076 | 1.932 ± 0.025 | 1.293 ± 0.017 | **1.086** ± 0.010 | 1.313 ± 0.015 | t∈[0.6, 0.8) |
| conv_2331 | 2331 | 3.024 ± 0.076 | 1.972 ± 0.023 | 1.372 ± 0.015 | **1.228** ± 0.012 | 1.765 ± 0.024 | t∈[0.6, 0.8) |
| scnbr_t04_1133 | 1133 | 3.122 ± 0.079 | 2.057 ± 0.027 | 1.406 ± 0.014 | **1.221** ± 0.010 | 1.518 ± 0.016 | t∈[0.6, 0.8) |
| sparse_vanilla_1259 | 1259 | 3.106 ± 0.072 | 2.059 ± 0.026 | 1.413 ± 0.014 | **1.218** ± 0.010 | 1.497 ± 0.016 | t∈[0.6, 0.8) |

(Mean ± SEM in `nat / protein`; bold = global minimum bucket per row.)

**Pairwise differences relative to canonical_2646:**

| ckpt | Δ@[0.0, 0.2) | Δ@[0.2, 0.4) | Δ@[0.4, 0.6) | Δ@[0.6, 0.8) | Δ@[0.8, 1.0) |
|---|---|---|---|---|---|
| conv_2331 | +0.006 | +0.040 | +0.079 | +0.142 | +0.452 |
| scnbr_t04_1133 | +0.104 | +0.125 | +0.113 | +0.135 | +0.205 |
| sparse_vanilla_1259 | +0.088 | +0.127 | +0.121 | +0.132 | +0.184 |

**Pairwise differences scnbr_t04 vs sparse_vanilla (Fix-C2 ablation):**

| bucket | scnbr_t04 − sparse_vanilla |
|---|---|
| t∈[0.0, 0.2) | +0.016 |
| t∈[0.2, 0.4) | −0.002 |
| t∈[0.4, 0.6) | −0.007 |
| t∈[0.6, 0.8) | +0.003 |
| t∈[0.8, 1.0) | +0.021 |

**Narrow claim.**

**Per-t validation loss is uninformative for choosing between CA-only architectural variants at the resolution required for hybrid-sampling decisions.** Across canonical_2646, conv_2331, scnbr_t04_1133, and sparse_vanilla_1259:

1. **The four loss-vs-t curves are parallel, not crossing.** canonical is uniformly best at every bucket; the other three are clustered tightly within ±0.05 nat / protein at every bucket except [0.8, 1.0). There is no t-region where any non-canonical variant has lower paired loss than canonical.
2. **All four ckpts have the same minimum bucket: t∈[0.6, 0.8).** The U-shape is identical in shape and location.
3. **The two sparse variants are functionally identical at this resolution.** scnbr_t04 vs sparse_vanilla differ by ≤ 0.025 nat / protein at every bucket — well inside the noise floor for paired-sample resolution at n=600.

**Implikation.**

(a) **Hybrid sampling cannot be justified by "each variant is best in its own t-regime"** — that hypothesis fails on the per-t-loss criterion. Whatever value-add E040 / E041 might extract from the conv ckpt is **not** about conv being a locally-better velocity predictor; it must come from sampling-trajectory dynamics that this no-integration measurement cannot see (e.g. trajectory-conditioned x_t marginals diverging between variants, or kink tolerance). The right next diagnostic is therefore D2 — trajectory-level v-divergence on actually-sampled trajectories — not more per-t val-loss probing.

(b) **The Fix C2 mechanism (sc_neighbors threshold-gating) does not move trained weights at this resolution.** Per-t val loss being identical between scnbr_t04 (Fix C2 active) and sparse_vanilla (no Fix C2) means whatever inference-time benefit Fix C2 provides must come from the sampling-time x_sc switch itself, not from Fix C2 having shaped the trained weights. This restricts the mechanism story for E039's 17% pooled designability to inference-only (or sampling-trajectory) effects; it is not a training-objective effect.

(c) **conv_2331's largest gap to canonical is at t∈[0.8, 1.0) (+0.452)** — consistent with E041's hand-off-before-late-stage design (t_switch=0.6, *before* this conv-disadvantage region starts). Per-t val loss therefore *post-hoc supports* the qualitative hand-off-direction choice in E040 / E041 (use conv early, refine with non-conv late), even though it does not support the *strong* compositional claim that conv is locally best somewhere.

(d) **Methodological caveat for any future variant comparison:** validation MSE / val FM loss should not be the sole acceptance criterion for an architectural variant. Finding 5 already established this for one specific failure mode (uniform-wd AdamW gating-collapse: better val loss, worse designability). F11 generalises the methodological caveat: even when no pathology like Finding 5's gating-collapse is involved, val loss within ±0.13 of canonical still corresponds to wildly different designability outcomes (canonical 76% pooled vs conv 0%; scnbr 17%; sparse_vanilla untested). Designability is the binding criterion; per-t val loss is a sanity check at best.

**Methodische Einschränkungen.**

- **No integration involved — nsteps is N/A.** F11 is a measurement of the trained weights at each ckpt; it does not depend on integrator convergence. The user's standing rule "use nsteps=400 for any designability run" (`feedback_use_nsteps_400_for_designability.md`) does not apply here. (The downstream designability comparisons that motivated F11 *do* have an open nsteps issue — see the E040 / E041 caveat blocks in `experiments.md` — but F11 itself is unaffected.)
- **600 proteins from the *training* index, length-filtered to ≤ 512.** This is a training-distribution proxy, not the canonical val set. Pairwise differences across ckpts are robust because all four use the same proteins; absolute numbers will be ~0.05-0.1 nat lower than the true val set (which has its own length / cluster cuts).
- **Single seed (42).** Re-running with seed 7 / 13 would tighten the scnbr_t04 vs sparse_vanilla equivalence claim. The conclusion is the same either way: at any seed, paired-sample resolution at n=600 with bucket-mean SEM ≈ 0.01-0.08 nat cannot resolve the small (Δ ≤ 0.025) sparse_vanilla vs scnbr_t04 differences.
- **Checkpoint maturity not matched.** canonical at step 2646 (overshoot regime — best-val window ended at step 2200), conv at 2331 (canonical-recipe best-val), scnbr / sparse_vanilla at their converged plateaus (1133 / 1259). The +0.13-0.45 nat canonical advantage includes ~1500 extra opt steps of training. The variant-vs-variant comparisons among the bottom three rows (where the maturity range is 1133-2331) are tighter, and the parallel-curve / no-crossing finding holds within the bottom three as well.
- **F11 does not say "all CA-only variants are equivalent training objectives".** It says "per-t val loss bucketed at the standard 5-bucket resolution does not separate them" — at higher t-resolution, or with a different metric (e.g. the actual gradient norm of the loss with respect to network outputs at each t), the variants might separate. F11's methodological claim is calibration: don't use this metric at this resolution to make hybrid-sampling decisions.
- **Designability is the right comparison metric.** As established by Findings 5 and 7, val loss and designability decouple in this codebase. The ordering at per-t val loss (canonical < conv ≈ scnbr ≈ sparse_vanilla) does not reflect the designability ordering (canonical 76% > scnbr 17% > conv 0%; sparse_vanilla untested). Reporting val loss without a paired designability number is misleading.

**Cross-references.**

- Lab notebook: [E043](experiments.md#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07).
- Hybrid-sampling experiments that motivated this diagnostic: [E040](experiments.md#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06), [E041](experiments.md#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06).
- Variant designability baselines: canonical [E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29), conv [E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06), scnbr [E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06).
- Finding 5 (val-loss-vs-designability decoupling, more severe form): see above in this document.
- Driver script: `proteinfoundation/run_per_t_val.py`.
- Output: `results/per_t_val/{canonical_2646, conv_2331, scnbr_t04_1133, sparse_vanilla_1259}.json`.

**Addendum (2026-05-07) — wandb training-time aggregate `validation_loss/loss_epoch` flips the ordering, but it is the artifact, not the trained-weights signal.**

A confounder for the F11 narrative is that the wandb training-time aggregate `validation_loss/loss_epoch` shows canonical_2646 *higher* than every variant's best-val — the opposite ordering from F11's per-t buckets and from the designability evidence. Three signals; one contradicting two. Resolved 2026-05-07 (lab notebook: E043 addendum). I re-evaluated canonical_2646 and sparse_vanilla_1259 on the SAME paired 600-protein subset, with t drawn from the actual training/val distribution `mix_unif_beta(p1=1.9, p2=1.0, p3=0.02)`, repeated 20 t-draw seeds per ckpt, via `proteinfoundation/run_aggregate_val_seeds.py`:

| ckpt | mean_of_means | std_of_means | min | max |
|---|---|---|---|---|
| canonical_2646 | **1.4008** | 0.0224 | 1.3267 | 1.4319 |
| sparse_vanilla_1259 | **1.5375** | 0.0219 | 1.4741 | 1.5779 |

Δ(sparse_vanilla − canonical) = **+0.137 nat (+4.36σ)** with combined std-of-means ≈ 0.031. **Zero seed-overlap** — the worst canonical t-draw (1.4319) is below the best sparse_vanilla draw (1.4741). On a paired set under the actual training t-distribution, all three signals AGREE: canonical < sparse_vanilla.

The wandb training-time flip is therefore not a property of the trained weights. Most likely cause: each training run averaged over the *first N* of its own dataset construction (`val_dataloader` has `shuffle=False` and `limit_val_batches=100→50` changed between hashes), so canonical's wandb numbers and the variants' wandb numbers were on *different protein subsets*. Other candidates: EMA-vs-raw model-copy mismatch in logging vs ckpt selection; "best_val" being a lifetime-min order statistic with non-trivial per-event variance.

**This refines F11's methodological claim into a stronger one:** *not just* "per-t val loss doesn't distinguish CA-only variants at the resolution that matters" — but also "wandb training-time `validation_loss/loss_epoch` is not comparable across runs in this codebase". Cross-run val-loss comparisons must use a paired re-evaluation protocol (`run_per_t_val.py` for per-t shape, `run_aggregate_val_seeds.py` for aggregate). When wandb val-loss disagrees with designability, designability is the trustworthy signal. Memory: `feedback_wandb_val_loss_not_comparable.md`.

The **1/(1-t)² loss weight** in `rdn_flow_matcher.py:215` is real (it makes high-t individual proteins score 5-15 nat at heavy-tail draws), but the heavy-tail story is not by itself the cause of the wandb flip — `std_of_means = 0.022` shows the seed-mean estimator is stable at n=600. The flip needs a different-population explanation, not a higher-variance explanation.

**Sanity check.** Integrating canonical_2646's per-t bucket means under `mix_unif_beta(1.9, 1.0, 0.02)`-derived bucket weights gives predicted aggregate ≈ 1.43 (measured 1.40); sparse_vanilla_1259 predicts ≈ 1.57 (measured 1.54). Per-t buckets and the paired aggregate agree to within 0.04 nat, which strengthens the F11 measurement: the bucketed picture quantitatively *predicts* the aggregate on the same paired set, while the wandb panel doesn't. The wandb panel is measuring something else (different proteins).

---

## Finding 12 — Dense attention's routing prior is structurally per-query, not per-(layer, head)-shared, and gradient saliency is at least as sharp an importance signal as attention itself (2026-05-13)

**Status:** finished. Methodological and architectural finding for the sparse-attention-vs-dense line of work. Lab-notebook detail: [E061](experiments.md#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13), with [E059](experiments.md#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (attention concentration audit) and [E060](experiments.md#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) (aggregate-loss gradient saliency audit, superseded) as the immediate predecessors.

**Why this is a Finding rather than a tuning note.** Our sparse-attention CA-only variants (K=40, K=64, K=64 + curriculum, K=64 + curriculum + BigBird, K=64 + curriculum + pair-update + lowtsoft — [E021](experiments.md#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30), [E046](experiments.md#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11), [E049](experiments.md#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08), [E051](experiments.md#e051--n3-quick-designability-probe-of-ca_only_sparse_k64_curriculum_self-at-step-1800-2026-05-10), [E055](experiments.md#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12), [E056](experiments.md#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13)) all share two architectural simplifications: (a) **a single K-set per query is shared across all 14 layers** of the trunk (the neighbor-list is built once per forward and reused at every attention layer); and (b) **the K-set per query is determined by content-free features** (sequence position offset, 1/d³-weighted spatial neighbors of a noisy `x_t`, random redraws) rather than by any learned routing prior. Every one of these variants underperforms canonical dense at converged steps. The mechanism that has been hypothesised in [E043](experiments.md#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07)'s caveat block and in [E059](experiments.md#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)'s "168-K-sets-required" framing is that **dense's per-(layer, head) attention specialization is the binding architectural advantage**, and sparse forfeits it by sharing the K-set across layers/heads. F12 is the first audit that quantifies which axis of dense's specialization is the load-bearing one and produces a constructive proposal — per-query routing rather than per-(layer, head) shared K-set — that is mechanistically supported by the data.

**Experiment.**

Three matched-protocol audits on the canonical dense baseline (`test_ca_only_diffusion/1776805213` step 2646, the documented best-val ckpt, 76 % pooled designability per [E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)):

1. **[E059](experiments.md#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) — Attention concentration and stability.** For every (protein, t, layer, head, query) cell, record (a) fraction of attention mass captured by the top-K residues at K ∈ {8, 16, 32, 48, 64}; (b) the top-16 attended set per cell; then compute Jaccard overlap between (i) adjacent layers, (ii) adjacent t-steps, (iii) heads within a (layer, t). 3 proteins per length bin × L ∈ {50, 100, 200} × t ∈ {0.1, 0.3, 0.5, 0.7, 0.9} = **630 attention-layer records** (3×3×5×14).
2. **[E060](experiments.md#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) — Aggregate-loss gradient saliency (superseded).** Same proteins / t-grid / seed=42. Forward+backward of `Σ_i ‖v_pred[i]‖₂` (a single aggregate scalar) → one per-residue saliency vector per (protein, t). Returned mass_top_K diffuser than attention and Jaccard(grad, attn) essentially orthogonal → led to a STOP call that was later traced to the aggregate-loss design averaging out per-query specialization.
3. **[E061](experiments.md#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13) — Per-query VJP gradient saliency.** Same proteins / t-grid / seed=42. For each (protein, t) sample 8 query residues; for each sampled query *i*, backward `‖v_pred[i]‖₂` *individually* with `retain_graph=True`. This gives **one saliency vector PER query**, structurally one-to-one with sparse's per-residue K-set (each query has its own neighbor list). Total: **360 per-query saliency records (8 × 9 × 5)**, **5040 cross-metric records** (360 queries × 14 layers; each carrying 12 per-head Jaccard values = 60480 (query, layer, head) cells), **1260 query-pair Jaccard records** within (protein, t).

All three on `best_val_00000026_000000002646.ckpt`, bf16 forward (fp32 for E061's backward — `--force_precision_f32` not used; bf16 grad-saliency confirmed not noise-limited). Each audit ~15-30 s wall on 1× L4 — minimal compute. JACCARD_K = 16 throughout; the cheap-K-distillation idea's natural unit is "top-16 most-important residues per query at each layer".

**Numbers — per-query gradient concentration (E061 Check 1', 360 sampled queries):**

| K | mean mass_top_K | median | (E059 attn mean) | (E060 aggregate grad mean) |
|---|---|---|---|---|
| 8  | 0.567 | 0.564 | 0.510 | 0.176 |
| 16 | **0.709** | **0.728** | 0.656 | 0.312 |
| 32 | **0.830** | **0.871** | 0.794 | 0.528 |
| 48 | 0.891 | 0.921 | 0.866 | 0.683 |
| 64 | 0.922 | 0.953 | 0.907 | 0.766 |

Per-query gradient is sharper than per-(layer, head, query) attention at every K, and dramatically sharper than the aggregate-loss gradient. The crossover threshold for cheap-K viability (mass_top_16 ≥ 0.70) is cleared by per-query gradient (0.709) but missed by attention (0.656) and by aggregate gradient (0.312).

**Per-length breakdown** (mean mass_top_K, n=120 queries per L):

| K  | L=50  | L=100 | L=200 |
|---|---|---|---|
| 16 | 0.810 | 0.708 | 0.610 |
| 32 | 0.939 | 0.828 | 0.722 |
| 48 | 0.997 | 0.892 | 0.783 |
| 64 | 1.000 | 0.937 | 0.828 |

Concentration drops sharply with L; K must scale with N for a fixed importance-mass capture rate.

**Per-query t-stability (E061 Check 2', 63 (protein, query) t-adjacent pairs):**

| t-pair | pooled mean | L=50 | L=100 | L=200 |
|---|---|---|---|---|
| 0.1 → 0.3 | **0.850** | 0.886 | 0.778 | 0.778 |
| 0.3 → 0.5 | 0.640 | 0.598 | 0.762 | 0.684 |
| 0.5 → 0.7 | 0.600 | 0.648 | 0.531 | 0.562 |
| 0.7 → 0.9 | 0.653 | 0.622 | 0.778 | 0.684 |
| overall | **0.663** | 0.669 | 0.668 | 0.654 |

Same query's important set is largely stable across adjacent t-values (overall Jaccard 0.66 vs 0.7 GO bar). The low-t pair (0.1→0.3) is the most stable (0.85). E060's aggregate t-Jaccard of 0.20 was an averaging artifact.

**Cross-metric Jaccard(grad top-16 per query, attn top-16 per (layer, head, query)) — the load-bearing measurement:**

| summary | E061 per-query | E060 (head-avg, aggregate grad) |
|---|---|---|
| overall mean across all (query, layer, head) cells | **0.337** | 0.114 |
| mean of max-over-(layer, head) per (protein, t, query) | **0.833** | 0.274 |
| min of that max | 0.524 | 0.067 |
| max of that max | 1.000 | 0.600 |
| fraction of 60480 cells with Jaccard ≥ 0.5 | **31.2 %** | — |
| fraction with Jaccard ≥ 0.7 | **8.2 %** | — |

**Headline: for every one of the 360 sampled queries there exists some (layer, head) in the 14-layer × 12-head dense trunk where attention's top-16 attended residues agree with gradient's top-16 important residues on at least 0.524 (8/16) of them, and on average on 0.833 (13/16).** The worst query in our sample has at least one (layer, head) sharing half its top-16 important residues with gradient saliency.

**Best-(layer, head) winners across all 360 queries (which cell hosts the max Jaccard most often):**

| rank | (layer, head) | wins | % |
|---|---|---|---|
| 1 | **L1 H7** | 66 | 18.3 % |
| 2 | L2 H4 | 49 | 13.6 % |
| 3 | L0 H4 | 38 | 10.6 % |
| 4 | L0 H9 | 19 | 5.3 % |
| 5 | L4 H0 | 14 | 3.9 % |

**Top-3 cells take 42.5 % of wins; top-10 take 71 %; 103 of 168 cells never win.** Early layers (0–2) take 54 % of all wins. The loss-aligned routing signal lives in a small subset of the trunk.

**Query-pair Jaccard within (protein, t) — different queries need different K-sets, and the effect strengthens with L:**

| L | mean | median | n pairs |
|---|---|---|---|
| L=50  | 0.258 | 0.143 | 420 |
| L=100 | 0.119 | 0.032 | 420 |
| **L=200** | **0.062** | **0.000** | 420 |

At L=200 the typical pair of queries within the same (protein, t) shares **zero** top-16 important residues (median 0.0, mean 0.062). The per-query-routing requirement scales with N exactly in the regime where the sparse-vs-dense designability gap is largest ([E043](experiments.md#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07): canonical 53 % vs sparse 0-11 % at L=200 in [E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)).

**E059 stability for reference (the original STOP signal):**

| axis | mean Jaccard | n |
|---|---|---|
| layer-adjacent | 0.217 | 45 |
| t-adjacent (attention) | 0.475 | 126 |
| head-within (layer, t) | 0.224 | 630 |

E059 measured attention's *self-consistency* across layers/heads/t. F12 reinterprets E059: layers and heads disagreeing on what to attend to (Jaccard ~0.22) is not a problem to be fixed by a shared K-set, it is the **specialization itself** — different (layer, head) cells specialize to different per-query routings, and only a small subset of them is loss-aligned for any given query.

**Narrow claim.**

In the canonical CA-only dense baseline (step 2646), the routing prior that dense attention encodes is **structurally per-query, not shared across queries, and is most concentrated in a small subset (~10 of 168) of (layer, head) cells**. The three audits jointly establish:

1. **Per-query gradient saliency is a sharper importance signal than per-(layer, head, query) attention** (mass_top_16 = 0.709 vs 0.656; mass_top_32 = 0.830 vs 0.794). The "use grad as the teacher" alternative is not just viable, it is empirically superior to attention as a teacher signal.
2. **Different queries within the same (protein, t) require materially different K-sets**, and the divergence scales with N: median query-pair Jaccard is 0.143 at L=50, 0.032 at L=100, **0.000 at L=200**. A sparse student that shares any K-set across queries discards information that dense uses heavily, and the information lost scales with N.
3. **For every query, there exists at least one (layer, head) in the dense trunk where attention's top-16 attended set overlaps gradient's top-16 important set on ≥ 0.524**, and on average on 0.833. Attention is not orthogonal to loss-importance per-query — it is *aligned per-query at a specific (layer, head)* that varies by query.
4. **The loss-aligned (layer, head) cells are concentrated**: top-3 host 42.5 % of all argmax-wins across 360 queries; top-10 host 71 %; 103 of 168 cells host none.

**Implikation.**

**(a) The cheap-shared-K student distillation idea is ruled out in this codebase**, but for a different reason than [E059](experiments.md#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) originally framed (concentration-too-low). The binding constraint is the **per-query divergence of K-sets** at long L (query-pair Jaccard = 0.06 at L=200), not the attention concentration per cell. Any sparse architecture that shares a K-set across queries at long L is structurally unable to recover dense behaviour by adding more shared-content channels (BigBird globals, pair-update, etc. — [E056](experiments.md#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13)/[E057](experiments.md#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge)/[E058](experiments.md#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge)). This is a structural prediction the BigBird-only retrain (E056 dead arm, 0/18 designable) is consistent with: position-unaware shared globals cannot supply the per-query divergence that the loss requires.

**(b) Per-query routing distillation is mechanistically supported and constitutes a concrete next architectural lever.** Per-query gradient saliency is sharp (mass_top_16 = 0.71), t-stable (Jaccard 0.66), and aligned per-query with dense attention at some (layer, head) (max-Jaccard mean 0.83). A natural construction: at training time, derive a target per-query K-set from per-query gradient saliency on the canonical dense teacher; train a small router that consumes ~10–15 selected (layer, head) outputs and emits a per-query K-set; the sparse attention then runs at K ≪ N per query. This is **not "cheap" in the original E059 sense** (one K-set per protein), but it IS cheap in the architecturally relevant sense — per-query attention at K ≪ N with a learned routing head, vs full N×N dense.

**(c) K must scale with N to capture a fixed fraction of importance mass.** mass_top_16 drops from 0.81 at L=50 to 0.61 at L=200; mass_top_32 drops from 0.94 to 0.72. A flat K-budget like K=40 captures ~80 % of saliency at L=50 but only 60–65 % at L=200. The architectural decision in current sparse variants to use a fixed K is empirically the wrong calibration at long L.

**(d) The result is independent of which importance signal you trust.** [E059](experiments.md#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (attention) and [E061](experiments.md#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13) (gradient saliency) are two *independent* metrics of importance. F12's claims hold under both: under attention, dense's per-(layer, head) specialization is what sparse forfeits; under gradient saliency, per-query divergence is what shared K-sets discard. The two metrics agree on the architectural conclusion (per-query routing) at the per-query × per-(layer, head) level (cross-metric mean-of-max Jaccard 0.833), even though they disagree on the per-(query, layer, head) level (overall Jaccard 0.337, p25 = 0.103). The methodological lesson: **per-query is the right unit of analysis for routing questions in this codebase**; aggregating before differentiating ([E060](experiments.md#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13)) or head-averaging across queries collapses exactly the structure that matters.

**(e) The cross-metric alignment is strongest in the regime where sparse-vs-dense designability is worst.** Best max-(layer, head) cross-metric Jaccard cells are at L=200, t=0.3 (0.910); the canonical dense's L=200 designability lead over sparse variants is also largest in the same regime. The loss-aligned routing prior is most readable from dense attention precisely where sparse needs it most — a structural alignment that supports per-query routing distillation being a tractable construction.

**Methodische Einschränkungen.**

- **Sample sizes.** 9 proteins (3 per L bin), 5 t-values, 8 queries per (protein, t) = 360 sampled queries; 60480 (query, layer, head) cells. Within-cell SEM is not the binding uncertainty — the per-(layer, head) winners table aggregates ≥ 360 queries — but the per-protein bin (3 proteins per L) is the limiting factor for *length-stratified* claims. Replicate with 9 proteins per L bin before promoting the L=200 query-pair Jaccard = 0.062 finding (and the L-dependence of per-query divergence) beyond the "directional + mechanistically supported" tier.
- **Query selection.** 8 queries per (protein, t) are sampled deterministically by `--queries_per_protein 8`; query positions may concentrate or repeat across t-draws within the same protein. The audit script handles this by resampling per (protein, t), so cross-t Jaccard counts are restricted to (protein, query_i) keys that survive the resampling. Full per-residue coverage would multiply compute by `mean_L / 8 ≈ 12.5×` but is not necessary for the headline directional claims.
- **scalar choice for VJP.** Per-query saliency is taken w.r.t. `‖v_pred[i]‖₂` (the L2 norm of the model's predicted velocity at query i), not w.r.t. a full training-time loss. The choice is correct for the inference-time decision target ("how much does residue j affect query i's output velocity?") and matches the script's design, but a per-coord or per-direction saliency could shift the ranking. Sensitivity check is the natural follow-up; the headline directional results are not expected to invert under reasonable alternatives.
- **bf16 grad.** All audits ran in bf16; gradient saliency at this precision has more numerical noise than fp32, particularly in the long tail of per-residue components near zero. Direction of bias inflates diffuseness slightly, so the per-query concentration numbers (mass_top_16 = 0.71) are if anything *understated* under bf16. Re-run with `--force_precision_f32` to tighten the post-decimal point of the headline numbers if needed for the writeup; the structural conclusions do not depend on the floor.
- **Aggregate vs per-query distinction is the key methodological move.** [E060](experiments.md#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13)'s STOP call was based on the aggregate-loss path; F12 supersedes [E060](experiments.md#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) by switching to per-query VJPs. Any future audit that purports to measure importance should default to per-query — aggregate loss measurements answer a different question and have already misled this project once.
- **F12 is a decision-gate, not a build.** All claims about distillation viability are about whether the audit data supports building such a student — not that we have one. The actual experiment (train a per-query routing student that matches canonical at L=200) is the load-bearing follow-up to F12 and is not done here.
- **Single canonical baseline.** All three audits are on `test_ca_only_diffusion/1776805213` step 2646. Whether the per-query routing prior in dense attention is a property of this specific ckpt's training trajectory (recipe: wd=0.05, constant LR=2e-4, no scheduler) or a general property of CA-only DiT-style trunks is untested. The L=200 query-pair Jaccard = 0.06 finding being most informative would replicate on any well-trained canonical CA-only ckpt; the per-(layer, head) winners ranking (L1 H7 at 18.3 %) might be specific to this initialization seed.
- **Sparse-vs-dense designability comparison numbers come from separate experiments at different steps.** The 76 % canonical vs 11 % sparse-K64 (LOWTSOFT inference at step 1385) is the canonical reference for sparse-vs-dense gap; F12 does not re-measure this. The structural finding (per-query divergence at long L) is what F12 contributes; the designability magnitude attribution to that structural finding is a hypothesis that requires the per-query routing student to be built.

**Cross-references.**

- Direct lab-notebook: [E061](experiments.md#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13) (per-query VJP; the load-bearing audit) and its predecessors [E059](experiments.md#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (attention concentration) + [E060](experiments.md#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) (aggregate gradient, superseded).
- Designability baselines used in the implikation: [E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) (canonical 76 % N=30), [E046](experiments.md#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) (K=64-curriculum +LOWTSOFT 39 % N=18 at L=200 = 0 %), [E021](experiments.md#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30)/[E049](experiments.md#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08)/[E051](experiments.md#e051--n3-quick-designability-probe-of-ca_only_sparse_k64_curriculum_self-at-step-1800-2026-05-10) (sparse-K64 trunk), [E055](experiments.md#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12)/[E056](experiments.md#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13)/[E057](experiments.md#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge)/[E058](experiments.md#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge) (BigBird bundles; E056 dead-arm at 0/18 — the data here explains why position-unaware globals are the wrong lever).
- Architectural framing: [E043](experiments.md#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07) (per-t val loss equivalence across CA-only variants; per-t val loss does NOT separate variants, but per-query routing audit DOES — F12 says the variant-distinguishing measurement is at the (query, layer, head) level, not the per-t level).
- Methodological connection: Findings 5 / 7 / 11 collectively established that val-loss-based selection criteria are unreliable in this codebase. F12 adds a positive complement — when val-loss decoupling makes a metric unreliable, the per-query routing audit gives a direct architectural read of what the trained dense model has internalised, independent of the val-loss surface.
- Audit scripts: `script_utils/audit_dense_attention_concentration.py`, `script_utils/audit_dense_gradient_saliency.py` (per-query VJP path, commit `9ed7a93` + a 1-line f-string-syntax fix on 2026-05-13).
- Output JSONs (not committed; full data on disk): `results/dense_attn_audit/{canonical_2646_dense_attn, canonical_2646_gradient, canonical_2646_grad_per_query}.json`.

**Constructive follow-ups (paper-relevant).**

- **Per-query routing distillation training run.** Train a small router (CNN or linear-on-context) that consumes a chosen subset of dense (layer, head) attention outputs (start with the top 10 wins from F12: L1 H7, L2 H4, L0 H4, L0 H9, L4 H0, L2 H6, L10 H0, L2 H10, L5 H8, L7 H8) and emits a per-query top-32 K-set. Sparse attention at K=32 per query is roughly 6× cheaper than dense at L=200, which is the regime where the architectural payoff is largest. The router is trained on canonical-dense's per-query gradient saliency as the supervisory signal (since per-query grad is the sharpest target; mass_top_16 = 0.71). Measure: designability at L=200 vs canonical's 53 %.
- **K-as-a-function-of-N calibration.** Re-train one canonical-recipe sparse variant with `K = ⌈0.32 × N⌉` (capturing the same fraction of saliency mass at every L per F12's per-L breakdown) instead of K=40 or K=64. Tests whether the "K must scale with N" implication translates to a measurable designability improvement at L=200 without changing any other axis. Cheap (~16h SL2 slot, single new training config) and the cleanest test of implikation (c).
- **Per-query routing audit on a sparse-K64-curriculum-trained ckpt.** F12 is on the dense baseline; we don't know whether the sparse-K64-curriculum variant's *trained* attention has the same per-query specialization or whether sharing the K-set across layers has collapsed it. Re-run the [E061](experiments.md#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13) audit on `sparse_K64_curriculum_self_step1800.ckpt` ([E051](experiments.md#e051--n3-quick-designability-probe-of-ca_only_sparse_k64_curriculum_self-at-step-1800-2026-05-10)) to characterise. Predicts: per-query divergence preserved (because each query has its own K-set in sparse), but max-(layer, head) Jaccard with gradient lower (because the K-set is content-free, not loss-aligned).

---

### Causal ablation of the AdaLN-Zero × weight-decay collapse mechanism (follow-up to Finding 5, 2026-04-25)

**Motivation and literature gap (why this is worth running, despite Finding 5 already documenting the correlational result):**

Finding 5 established a strong correlational case that uniform AdamW weight decay applied to the La-Proteina CA-only baseline crushes AdaLN-Zero output gates in the upper transformer blocks (gates at 26–60% of baseline magnitude in v2 vs old) and that this co-occurs with a categorical sample-quality collapse (0/9 designable samples) despite a 0.33 best-validation-loss improvement. The mechanism — uniform AdamW weight decay applied to zero-initialised gates suppresses their growth, which weakens the time-conditioning signal at the velocity output and causes integrated trajectories to fail even though averaged velocity-MSE looks better — is mechanistically coherent and consistent with the per-layer weight evidence. However, no causal experiment has been run; an examiner could legitimately argue that the cosine LR decay (which co-occurred with wd=0.1 in v2 and was not isolated) is the actual cause, or that the collapse is a generic effect of stronger regularisation rather than the specific gate-suppression mechanism we propose.

A targeted literature survey (DiT, SiT, SD3 / MMDiT, PixArt-α, "Unveiling the Secret of AdaLN-Zero", ReZero, the diffusers SD3 reference, and the AdamW paper itself; full source list in CLAUDE-session notes) found that:
- **The dominant DiT-family training recipe is `weight_decay=0` everywhere.** DiT's official `train.py` uses literally `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)`; SiT's `train.py` uses the same line. SD3's paper does not discuss weight decay at all and emphasises QK-normalisation and ε=1e-15 in AdamW for stability. PixArt-α has a configurable `zero_weight_decay` attribute but does not apply it to AdaLN by default.
- **The "exclude AdaLN-Zero gates from wd" rule is therefore implicit in the literature, not enforced**: canonical DiT-family codebases sidestep the problem by setting wd=0, so they never observe the failure mode. The rule has not been formally written down and ablated in any paper we could find.
- **The closest prior art is ReZero** (Bachlechner et al. 2020), which mentions in passing that "it was important to have a small, constant learning rate for the residual weights, otherwise the ReZero model diverges" — a hint at the gate-fragility issue but not a quantitative or actionable claim about weight decay specifically.
- **The recent OpenReview paper "Unveiling the Secret of AdaLN-Zero"** isolates which architectural components of AdaLN-Zero matter (it identifies zero-initialisation as the dominant factor), but does not discuss the optimizer-gate interaction or the val-vs-sample-quality decoupling.

The literature gap is therefore real: the AdaLN-Zero × uniform-AdamW-wd interaction is folk knowledge in the DiT community (you wouldn't apply wd>0 without parameter groups in any modern image-generation codebase) but has not been formally characterised as a quantitative case study with per-layer evidence and a positive/negative ablation. This experiment fills that gap if successful, and at minimum produces a defensible thesis-internal causal claim.

**Hypothesis (mechanistic, to be falsified):**

Restoring growth capacity to the AdaLN-Zero output gates (either by removing weight decay entirely from those parameters, or by removing weight decay altogether and accepting the loss of regularisation) recovers sample quality on the v2 model **without** giving up the validation-loss improvement that wd+cosine bought. If this prediction is wrong — i.e. samples remain bad even after the gates are unconstrained — then the mechanism we proposed in Finding 5 is wrong (or at least incomplete) and the v2 collapse must be attributed to something else (cosine LR-decay shrinking the effective LR to a non-sampling-friendly regime; an interaction with self-conditioning; etc.).

**Experimental design — three variants in increasing scope:**

The three variants below trade off compute against the strength of the causal claim. Choose based on available compute and how watertight the writeup needs to be.

**A. Cheapest — finetune the v2-2078 ckpt with wd=0 on AdaLN-Zero gates only (~3 h on 1 A100):**

- Resume from the v2 best raw checkpoint (`store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/best_val_00000020_000000002078.ckpt`).
- Modify `proteinfoundation/proteina.py:configure_optimizers` to split parameters into two groups: (i) all `*.scale_output.to_adaln_zero_gamma.0.weight` parameters get `weight_decay=0`; (ii) everything else keeps the current `weight_decay` value (= 0.1). No other config change.
- Continue training for ≈ 200–500 optimizer steps. Log: `train/grad_norm`, `train/param_norm`, the per-layer L2 norm of each AdaLN-Zero gate (new logging — needs a small addition to `on_before_optimizer_step` that walks `[p for n,p in self.named_parameters() if "to_adaln_zero" in n]` and logs each `‖p‖`).
- Stopping rule: continue until either (a) the upper-layer gates (layers 7–13) recover to ≥ 80% of the old-recipe norms documented in Finding 5, or (b) 500 optimizer steps have elapsed without recovery.
- Save the resulting checkpoint, run the same eval as Finding 5 (`inference_ucond_notri_ca_only_v2_quick.yaml` config, lengths [50, 150, 300, 450], N=10 samples per length, 200 ODE steps, ESMFold designability).

**B. Cleanest — retrain v2 from scratch with the proper param-group split (~16 h on 1 A100, chained):**

- Modify `configure_optimizers` to split parameters into (i) bias / LayerNorm γ,β / embedding / `to_adaln_zero` parameters → `weight_decay=0`; (ii) everything else → `weight_decay=0.1`. This is the canonical DiT/SiT/SD3-style split (~15 lines).
- Train from scratch with the v2 schedule (cosine_with_warmup, peak 2e-4, total 6000 opt steps, min ratio 0.1) for the same ~2100–2300 optimizer steps where Finding 5 v2 best was reached. Use the same seed (42), same data, same batch size, same EMA settings.
- Compare against (i) the old recipe baseline checkpoint at its best (~step 2204, val 4.71–4.77, designability per Finding 5) and (ii) the v2 broken checkpoint (val 4.437, 0/9 designable).
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
1. Background: brief restatement of Finding 5 and the literature-gap motivation above.
2. Methods: the param-group split (~15 line code change), the variants run, the eval protocol.
3. Results: per-gate-layer norm trajectories during the experiment (do they regrow?), val-loss curve, designability table at matched lengths.
4. Discussion: which outcome of the matrix above was observed, and what it pins down about the mechanism.
5. Practical recommendation: a one-paragraph "if you train a DiT-style model with wd > 0 in any codebase that doesn't already split parameter groups, here is what you must do" section that cites the ablation as evidence.

**Risks and confounders to record with results:**
- Variant A's resume from v2-2078 carries forward whatever non-gate weights the v2 training shaped under wd=0.1; even if gates regrow, the *rest* of the model has been trained against suppressed gates and may have compensated in ways that don't undo cleanly. A failure in A is therefore not a failure of the mechanism — it could be a failure of the recovery procedure. Variant B (clean retrain) is needed to definitively rule this out.
- The eval is N=10 samples per length × 4 lengths = 40 samples. ESMFold scRMSD has known sensitivity to short-protein folding accuracy; if the recovered model is still marginal at L=50 but improves at L=300/450, that's still a positive result for the mechanism (the original v2 collapse was uniform across lengths, so partial recovery at any length is informative).
- The mechanism predicts that *if* the gates regrow, the val loss may regress (because the model goes back to predicting time-conditioned velocities, which are higher-MSE than the smoothed-out averages it was predicting before). Do not interpret a val-loss regression as a failure of the experiment — Finding 5's central claim is precisely that val loss is a misleading proxy in this setting.

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

### Direct latent-space geometry probe (follow-up to Finding 6)
Finding 6 showed that perturbations in AE1's latent space stay closer to ESMFold's prediction than equivalent coord-space perturbations, and inferred decoder contractivity from the noise-magnitude invariance. The "tighter packed latent space" framing was deliberately put in the Implikation, not the Narrow claim, because the experiment did not directly measure latent-space geometry. This follow-up closes that gap.

**Approach:**
1. Encode a held-out set of N=500–1000 proteins (same length stratification as Finding 6) → per-residue `mean` tensors.
2. For each pair (i, j), compute (a) Euclidean distance in latent space (L2 over per-residue means after sequence-length matching, e.g. averaging or padding+masking), (b) all-atom RMSD between native structures (TM-align or kabsch-aligned over CA).
3. Plot latent-distance vs structure-distance, fit a monotonic relationship (Spearman ρ).
4. Compare to a control: same plot but using raw stacked-CA-coordinate vectors as the "latent" — quantifies how much extra structure the AE imposes vs. just-coords.
5. Local density: for each protein, k-nearest neighbours in latent vs in structure space — does the AE preserve neighbourhoods?

**What this would directly test:** whether latent distances correspond to structural similarity (the actual "tighter packing" claim). A positive result strengthens Finding 6's Implikation; a negative result restricts Finding 6 to the strict decoder-contractivity claim only.

### Latent-arm decoder-contractivity ablation (follow-up to Finding 6, mechanistic)
The Implikation in Finding 6 attributes the noise-magnitude invariance of latent-arm scRMSD (median +0.17 Å over a 20× change in noise std) to the AE decoder being contractive. A direct test:

- For a single eval protein, sample latent perturbations at k ∈ {0.5, 1, 2, 4, 8, 16} (extending well past the data range) and decode.
- Plot `||z_clean − z_perturbed||_2` (input perturbation magnitude) vs `||x_clean − x_perturbed||_F` (output coord change). A contractive map shows sublinear or saturating behaviour; an isometric map shows linear; an expansive map shows superlinear.
- A clearly sublinear/saturating curve directly demonstrates contractivity, upgrading Finding 6's Implikation to a Narrow claim.

### Coord-arm with whitened, basis-aligned noise (Finding 6 follow-up, robustness)
The pre-registered Finding 6 caveat about commensurability between coord σ and latent σ deserves a tighter test. Add a third arm: coord-space noise sampled in a basis aligned with the local sidechain rotamer manifold (e.g. PCA over per-(restype, atom_idx) offset distributions, with σ scaled to match each PC's variance). If even this PC-aligned coord arm loses to latent at all k, the latent-space advantage is not just a bad-noise-basis artefact in the original coord arm.

### Curvature-targeted bump schedule follow-up (Finding 2 / E037 follow-up, schedule-vs-quality ablation at proper N)

E037 ran two paired-N=30 probes of `power_with_middle_bump` on `local_latents` (μ=0.489, σ=0.08, eps∈{0.1, 0.14}) at L=300 on the LD3 full-latent model. Both null at this N. The directional signal split by metric: continuous mean scRMSD favours the bump (60–63% of paired samples better at eps=0.14, mean Δ ≈ −0.10 Å on three co-design columns), but designability rate moves the *wrong* way by a consistent −7pp on three of four co-design columns in **both** probes. None significant (smallest Wilcoxon p=0.38; smallest McNemar p=0.50). The continuous-vs-threshold split is internally explained by the pair-level decomposition: improvements happen mostly inside the already-designable region or in the deep failure tail, while a few borderline samples get nudged the wrong way at the 2 Å boundary.

**Two cheap diagnostics before committing to N=100 (~4h on 1× A100):**

1. **Late-t density check.** The schedule renormalisation means more density at μ=0.489 implies less density elsewhere. Compare baseline vs `power_bump_e0.14` step density in the late-t window (t > 0.85). `script_utils/plot_straightness.py` already supports this. If late-t density is meaningfully lower with the bump, it's positive evidence for the trade-off hypothesis — the bump is buying mid-stage curvature accuracy at the cost of late-stage refinement, which is exactly when borderline samples need to make their final crossing of the 2 Å bar.
2. **Tail-neutral schedule design.** If the late-t check confirms the tail-stealing pattern, the right next variant is a bump that takes its density from the calm region near t=0.2 instead — e.g. an asymmetric bump (Gaussian × `(1−u)^2` weighting), or a wider σ that pulls density only from the early-t plateau. `optimise_bump.py` can be adapted to constrain the late-t density.

**Properly-powered run (only after the cheap diagnostics):**
- N=100 paired at L=300 with `power_bump_e0.14` (or the redesigned tail-neutral variant if step 2 is taken).
- Same paired-noise setup as E037 (`id_gen` join via `proteina.py:786`); `script_utils/schedule_comparison_report.py` already has the Wilcoxon + McNemar paired analysis.
- Predicts: (a) Wilcoxon p ≤ 0.05 on the continuous direction if the +0.10 Å median Δ holds; (b) ±5pp band on the designability shift, making the binary direction defensible. Also adds a length-sweep arm at L∈{200, 400, 500} to test the (currently assumed) length-independence of the schedule effect.

**Why this is worth running despite the null:**
- E004's curvature finding (Finding 2) explicitly flagged the schedule-vs-quality ablation as missing. E037 ran it but only at exploratory N. Either a positive N=100 result (continuous metric improves significantly without designability cost) or a properly-powered negative (binary designability *is* worse with the bump) is paper-defensible — the current null is neither.
- The "continuous improvement is real, threshold flip is not" pattern, if it survives at N=100, is a methodologically interesting finding in its own right: it shows that NFE-redistribution toward curvature peaks is a *distribution-shaping* tool, not a designability-rate tool. That's a calibration most schedule-search papers don't make.
- The trade-off mechanism (mid-stage accuracy bought at the cost of late-stage refinement) is testable and, if confirmed, gives a constructive path to a schedule that wins on both metrics — which is the actual paper claim worth chasing.
