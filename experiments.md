# Experiments — comprehensive lab notebook

Companion file to `content_masterarbeit.md`. This file is the **complete, chronological log of every experiment run**, regardless of whether the result feeds the thesis narrative. `content_masterarbeit.md` keeps the curated paper-facing findings; `experiments.md` keeps the lab record from which findings are extracted.

**Logging policy (also locked into `CLAUDE.md`):** every experimental run — training, eval, probe, smoke test, diagnostic, ablation, sweep — gets appended here automatically and without asking, no matter how small. The bar for entry is "did we run code that produced a number". The bar for `content_masterarbeit.md` is "is this defensible enough to write into the paper".

Each entry has:

- **ID + date** — assignable handle for cross-references.
- **Status** — finished / in progress / cancelled / failed.
- **Why ran** — the question the experiment was supposed to answer, and what decision or claim it feeds.
- **Configs** — exact setup (config files, recipe, hardware, run dir, wandb run IDs, checkpoint paths). Enough that someone could re-run.
- **Results** — every quantitative output. Tables, per-fold numbers, per-length numbers, weight-norm diffs, designability counts. Not just the headline.
- **Possible narrative** — does this become a Finding? If yes, link to the `content_masterarbeit.md` section. If no, note "non-narrative — kept for tuning/decision-making" and explain what decision it informs.
- **Methodological caveats** — what the data does *not* support. Single-seed, narrow N, confounded variables, etc.

When a finding is later promoted from this file into `content_masterarbeit.md`, leave the experiment entry here unchanged (do not delete) and add a back-link to the Finding section. The lab record is append-only.

---

## Index

| ID | Date | Status | Topic | Narrative? |
|---|---|---|---|---|
| [E001](#e001--multi-task-property-predictor-on-la-proteina-latents-2026-04-21) | 2026-04-21 | finished | Multi-task property predictor on AE latents | → Finding 1 |
| [E002](#e002--capacity-probing-of-property-decoders-2026-04-21) | 2026-04-21 | finished | Capacity probing (linear / MLP / per-residue MLP / Tx) | → Finding 4 |
| [E003](#e003--latent-geometry-of-the-partial-autoencoder-2026) | 2026 (Apr) | finished | Latent geometry (Part 1 of steerability pipeline) | → Finding 3 |
| [E004](#e004--flow-field-curvature-on-proteina-complexa-2026) | 2026 | finished | Flow-field straightness ratio per channel | → Finding 2 |
| [E005](#e005--cheap-diagnostics-pdb-vs-generated-property-correlations-2026-04) | 2026-04 | finished | PDB vs generated property correlations + length KS | non-narrative |
| [E006](#e006--steering-smoke-test-pre-round1-2026-04) | 2026-04 | finished | Standalone steering smoke test (pre round1) | non-narrative (engineering) |
| [E007](#e007--steering-round-1-net_charge-up-2026-04) | 2026-04 | finished | Steering eval: net_charge ↑, 5 proteins, all 13 properties | potential narrative |
| [E008](#e008--canonical-ca-only-baseline-training-old-recipe-2026-04-21--ongoing-chain) | 2026-04-21+ | finished (chain) | Canonical CA-only diffusion baseline | reference run for variants |
| [E009](#e009--v2-recipe-attempt-wd01--cosine_with_warmup-2026-04-23--2026-04-25) | 2026-04-23 → 2026-04-25 | finished, cancelled mid-chain | Stronger wd + cosine LR retraining attempt + post-mortem | → Finding 5 |
| [E010](#e010--sparse-attention-variant-k32-training-2026-04-25-in-progress) | 2026-04-25 → ongoing | in progress | SALAD-style K=32 sparse attention training | pending |
| [E011](#e011--sidechain-manifold-experiment-preregistered-2026-04-25) | 2026-04-25 → ongoing | preregistered / in progress | Coord-space vs latent-space sidechain perturbation | preregistered |
| [E012](#e012--three-run-comparison-baseline--v2--sparse-side-by-side-2026-04-26) | 2026-04-26 | finished | Side-by-side config + result diff of E008 / E009 / E010 | reference table |
| [E013](#e013--wd0-ablation-training-canonical-recipe-with-weight_decay00-2026-04-26--ongoing) | 2026-04-26 → ongoing | in progress | wd=0 ablation training on canonical CA-only recipe | → Finding 8 |
| [E014](#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27) | 2026-04-27 | finished | N=30/length matched-seed designability across baseline/v2/wd0/sparse | → Finding 8 (N=30 update) |
| [E015](#e015--three-wd-weight-norm-comparison--feasibility-of-param-group-fix-2026-04-27) | 2026-04-27 | finished | wd ∈ {0, 0.05, 0.1} per-layer gate + non-gate norm diff; pre-registration check for AdaLN-Zero param-group-fix experiment | non-narrative — disconfirmed a planned experiment's premise |
| [E016](#e016--ca-only-eval-pipeline-audit-reconstructed-bb-vs-ca-only-mpnn-2026-04-28) | 2026-04-28 | in progress | Audit of designability eval for CA-only generations: backbone-reconstruction geometry + SLURM probe on real natives | non-narrative — decides whether CA-only designability numbers need re-computing |

---

## E001 — Multi-task property predictor on La-Proteina latents (2026-04-21)

**Status:** finished (1-fold complete; 5-fold sweep completed shortly after).

**Why ran:** Decide whether a small multi-task head can read 13 developability properties out of the 8d per-residue AE latent. The output (per-property R²) doubles as the upper bound on guidance quality — the steering gradient *is* the predictor's gradient, so probe accessibility ≈ steerability for that property. Direct decision input for which properties to steer.

**Configs:**
- Architecture: `PropertyTransformer`, 128d, 3 layers, 4 heads, ~350k params.
- Input: per-residue 8d latent `mean` (only `mean`, not `log_scale`) from La-Proteina's partial autoencoder.
- Targets: 13 developability properties from `developability_panel.csv` (swi, tango, net_charge, pI, iupred3, iupred3_fraction_disordered, shannon_entropy, hydrophobic_patch_total_area, hydrophobic_patch_n_large, sap, scm_positive, scm_negative, rg).
- Dataset: 56,008 proteins, length 300–800. 10% held-out test (`heldout_test_ids.txt`), 5-fold CV on remainder.
- Training: 30 epochs, AdamW lr=3e-4, 500-step linear warmup + cosine decay, batch 16 (length-bucketed), grad-clip 1.0, early stopping patience=5. Z-score normalization of targets per-fold.
- Run dir: `laproteina_steerability/logs/multitask_t1/20260421_064011/`. Checkpoint root: `checkpoints_multitask_predictor/20260421_064011/` and `…/20260421_081025/`. `latest` symlink in `checkpoints_multitask_predictor/`.

**Results:**

5-fold val R² per property (best epoch per fold; some folds shown here are still being filled in as the sweep finished):

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

SWI specifics: target distribution `mean=0.7787, std=0.0101`. Folds 0/1 R² = 0.38/0.55, folds 2–4 R² ≈ 0.98. The instability is metric-driven (`R² = 1 - MSE/Var(y)` with very small `Var(y)` is split-sensitive), not learning-driven (see convergence-time table below).

Convergence-time table (epoch at which each property first reaches 90% of its final val R², averaged across folds):

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
| swi (folds 2–4) | 2.0 | fast |

Three-level hierarchy: sequence-derived ≪ mixed ≪ structure-derived in terms of how many epochs the probe needs.

**Possible narrative:** **Yes — this is Finding 1** (`content_masterarbeit.md → ## Finding 1`). The narrow-claim there is: "5-fold mean R² 0.88 across 13 properties; 12/13 stable across folds at R² ≥ 0.78; SWI fold-variance attributable to narrow target std=0.01."

**Methodological caveats:**
- R² is a metrically poor choice for narrow-variance properties (SWI). Should be paired with per-property MSE or rank correlation.
- A single probe architecture cannot distinguish "info present in latent" from "info accessible to this probe". E002 addresses the second half by sweeping probe family.
- Reported numbers are val R², not held-out test R² (held-out test reserved in `heldout_test_ids.txt`, not yet evaluated). Optimistic estimate.

---

## E002 — Capacity probing of property decoders (2026-04-21)

**Status:** finished (Fold 0 only; 5-fold repeat is on the future-experiments list).

**Why ran:** E001 told us R² across 13 properties at one probe architecture. It could not separate "info is in the latent" from "info is accessible to a 350K-param Transformer". This sweep adds 7 simpler probes (linear → per-residue MLPs of growing capacity) to E001's Transformer to draw the boundary between probe-family-bottlenecked and probe-size-bottlenecked properties. Decision input for steering predictor sizing.

**Configs:**
- Probes (parameter count + aggregation):
  1. `linear` (117 params): mean-pool residues → linear → 13 properties.
  2. `mlp_h32_L1` (717 params): mean-pool → MLP (hidden 32, 1 layer) → 13.
  3. `mlp_h64_L1` (1.4K params): mean-pool → MLP (hidden 64, 1 layer) → 13.
  4. `mlp_h128_L2` (19K params): mean-pool → MLP (hidden 128, 2 layers) → 13.
  5. `per_res_mlp_h64_L1` (1.4K params): per-residue MLP (h64, 1L) → mean-pool per-property logits.
  6. `per_res_h128_L2` (19K params): per-residue MLP (h128, 2L) → mean-pool.
  7. `per_res_h256_L3` (137K params): per-residue MLP (h256, 3L) → mean-pool.
  8. `Tx (3L, 128d, 4h)` (~350K params): same as E001's Transformer; Fold 0 values reused.
- Identical data, splits, z-score stats, loss across all probes.
- Training: 20 epochs max, AdamW lr=3e-3, wd=0.01, early-stop patience=4, batch 32 length-bucketed.
- Hardware: 1× NVIDIA L4. Total wall-clock 9 min.
- Run dir: `laproteina_steerability/logs/capacity_probing/20260421_191747/`. Checkpoints at `checkpoints_capacity_probing/20260421_191747/`.

**Results — Fold 0 val R², full ladder:**

| Property | linear | mlp_h32 | mlp_h64 | mlp_h128_L2 | per_res_h64 | per_res_h128 | per_res_h256 | Tx (3L) |
|---|---|---|---|---|---|---|---|---|
| #params | 117 | 717 | 1.4K | 19K | 1.4K | 19K | 137K | ~350K |
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

Class A (per-residue MLPs already unlock most of the R²): iupred3, net_charge, pI, shannon_entropy, iupred3_fraction_disordered, swi.
Class B (only attention unlocks): sap, tango, hydrophobic_patch_*, rg, scm_±.

`iupred3` (smooth aggregate) → R²=0.91 at `per_res_h256`; `iupred3_fraction_disordered` (threshold-count of the same per-residue signal) → R²=0.47 at `per_res_h256`. The Class A/B boundary is therefore not purely sequence-vs-structure but also smooth-vs-threshold-count.

**Possible narrative:** **Yes — this is Finding 4** (`content_masterarbeit.md → ## Finding 4`). Cross-referenced from Finding 1.

**Methodological caveats:**
- Single fold (Fold 0). h128→h256 regressions on net_charge (0.84→0.76) and pI (0.76→0.69) need a 5-fold repeat before "saturation at h128" is firm.
- Capacity ladder is non-uniform in architectural complexity at the per-residue-MLP → Tx step (3 changes at once: attention, multi-layer residue-residue, different aggregation). An intermediate 1-head/1-layer attention probe would isolate the minimum attention budget.
- All per-residue probes use mean-pool. Other aggregations (learned weighted pool, max, set-transformer) untested.
- Tx column is from E001's Fold 0 — uses 30-epoch budget vs the other probes' 20-epoch early-stop. Same-protocol Tx rerun would tighten the comparison.

---

## E003 — Latent geometry of the partial autoencoder (2026)

**Status:** finished.

**Why ran:** Before training property probes (E001/E002) and before designing steering objectives, characterise the AE latent itself. Concrete decisions this informs: (a) is the latent posterior-collapsed? (no → predictors have signal to work with); (b) are dims locally disentangled? (yes → multi-objective steering gradients won't always conflict); (c) does within-protein variance exceed between-protein? (yes → protein-level steering objectives can be averaged from per-residue latents without information loss).

**Configs:**
- Pipeline: `laproteina_steerability/src/part1_latent_geometry/` (Part 1 of the steerability analysis pipeline, see CLAUDE.md → Steerability Analysis Pipeline).
- Data: 56,008 proteins, length 300–800; 22.66M residues total; only `mean` field (8d per-residue) loaded.
- Outputs: `laproteina_steerability/outputs/part1_summary.md`, `outputs/tables/*.csv`, `outputs/figures/*.{png,pdf}`.

**Results:**

Dimensionality and utilization:

| Metric | Value |
|---|---|
| Participation ratio | 7.694 / 8 |
| Effective rank for 90% / 95% / 99% variance | 7D / 8D / 8D |
| Collapsed dims (variance < 1% of max) | 0 |
| Max \|off-diagonal Pearson\| between dim pairs | 0.102 |
| Max pairwise MI | 0.28 nats |

Marginal distributions (Shapiro on n=5000 subsample per dim):

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

Dims 3 and 7 are platykurtic (negative kurtosis), consistent with discretely clustered (multimodal) marginals.

Within-protein vs between-protein variance:

| Dim | within-protein | between-protein | within/total |
|---|---|---|---|
| 0 | 0.762 | 0.007 | 1.04 |
| 1 | 0.741 | 0.002 | 1.04 |
| 2 | 0.656 | 0.002 | 1.04 |
| 3 | 1.075 | 0.007 | 1.04 |
| 4 | 0.917 | 0.006 | 1.05 |
| 5 | 0.950 | 0.003 | 1.04 |
| 6 | 0.857 | 0.004 | 1.03 |
| 7 | 1.093 | 0.006 | 1.05 |

Within-protein variance is 100× between-protein on every dim.

Length sensitivity (Pearson r of per-protein-mean of each dim vs sequence length):

| Quantity | Pearson r |
|---|---|
| dim_0 mean | −0.027 |
| dim_1 mean | −0.042 |
| dim_2 mean | +0.020 |
| **dim_3 mean** | **+0.164** |
| dim_4 mean | +0.098 |
| dim_5 mean | +0.089 |
| dim_6 mean | +0.028 |
| dim_7 mean | −0.032 |
| L2 norm of latent | +0.040 |

**Possible narrative:** **Yes — this is Finding 3** (`content_masterarbeit.md → ## Finding 3`).

**Methodological caveats:**
- Pairwise dependencies measured with Pearson + empirical MI; higher-order or manifold dependencies not captured.
- Within/between ratio of ~1.04 ignores positional autocorrelation along the chain.
- Multimodality on dims 3 and 7 inferred from negative kurtosis only; no mixture model fit.

---

## E004 — Flow-field curvature on Proteina Complexa (2026)

**Status:** finished.

**Why ran:** Quantify how curved the learned ODE field is, separately for `bb_ca` and `local_latents`. Decision input for whether one-shot or few-step denoising is feasible per-channel (it is for `bb_ca`, not for `local_latents`), and for whether non-uniform t-grids (more NFEs near curvature peaks) could improve sample quality at fixed budget.

**Configs:**
- Checkpoints: `LD3_ucond_notri_800.ckpt` (flow model) + `AE2_ucond_800.ckpt` (autoencoder).
- Setup: 800-step uniform t-grid as proxy for continuous-time field; record per-residue per-step displacement; aggregate per channel.
- Operating point: `nsamples=8`, `nres=400`.
- Output: `checkpoints_laproteina/straightness_ld3.json`.

**Results:**

| Channel | Straightness ratio R | x1-pred variance | Step-length min | Step-length max | max\|Δ²\| |
|---|---|---|---|---|---|
| `bb_ca` | **0.9353** | 0.1083 | 1.98e-3 (t=0.006) | 7.50e-2 (t=0.000) | 7.30e-2 @ t=0.001 |
| `local_latents` | **0.5086** | 0.1230 | 1.25e-3 (t=0.445) | 3.58e-3 (t=0.868) | 4.03e-3 @ t=0.043 |

`bb_ca`: 37× larger first-step displacement (essentially a free Gaussian-prior sample) then near-constant ~2–2.5e-3 climbing smoothly through the trajectory. Field is very straight outside t=0.

`local_latents`: per-step displacement spans 1.25–3.58e-3 (std/mean ≈ 0.31). Mid-trajectory dip at t≈0.445, peak near t≈0.868. Half of total motion is "sideways correction".

**Possible narrative:** **Yes — this is Finding 2** (`content_masterarbeit.md → ## Finding 2`).

**Methodological caveats:**
- R is computed at one operating point (`nsamples=8`, `nres=400`); not verified at other lengths/batches.
- Discretization error of 800-step grid not quantified.
- Channel-R comparison conflates field curvature with dimensionality (3d bb_ca vs 8d local_latents).
- Causal claim ("curvature explains one-shot denoising difficulty") is plausible but not tested via schedule-vs-quality ablation.

---

## E005 — Cheap diagnostics: PDB vs generated property correlations (2026-04)

**Status:** finished.

**Why ran:** Before designing the steering experiment, sanity-check that (a) generated samples (unguided) have a property distribution similar enough to PDB that the steering predictor (trained on PDB) will be in-distribution at inference time, and (b) which property pairs are correlated in nature — to avoid setting a steering objective that's actually trying to fight a strong native correlation. Decision input for steering objective selection and for interpreting collateral effects in E007.

**Configs:**
- Inputs:
  - PDB property file: `laproteina_steerability/data/properties.csv` (56,008 rows, length 300–800).
  - Generated property file: `results/generated_baseline_300_800/properties_generated.csv` (100 unguided samples).
- Code: `analysis_cheap_diagnostics/run_cheap_diagnostics.py`.
- Outputs: `analysis_cheap_diagnostics/summary.md`, `length_bin_counts.csv`, `pdb_pearson_corr.csv`, `pdb_spearman_corr.csv`, `li_ji_per_bin.csv`, plus chained training-loss plots `train_loss_chained.png` / `val_loss_chained.png` and pulled wandb history `wandb_history_chained.csv`.

**Results:**

Length distribution match (300–800 only): `KS D=0.0769, p=5.69e-1`. PDB n=56,008, generated n=100. **No detectable length-distribution mismatch.**

Effective number of independent properties (Li-Ji M_eff): Pearson 9.0 / Spearman 10.0 (out of 14). Properties cluster into ~9–10 effective groups.

Top 10 |Pearson| pairs on PDB:

| prop A | prop B | r |
|---|---|---|
| hydrophobic_patch_total_area | hydrophobic_patch_n_large | +0.908 |
| hydrophobic_patch_total_area | sap | +0.901 |
| net_charge | pI | +0.855 |
| hydrophobic_patch_n_large | sap | +0.837 |
| iupred3 | iupred3_fraction_disordered | +0.741 |
| hydrophobic_patch_total_area | rg | +0.632 |
| tango | rg | +0.570 |
| tango_aggregation_positions | sap | +0.540 |
| net_charge | scm_negative | +0.524 |
| scm_positive | scm_negative | -0.518 |

Top 10 |Spearman| pairs:

| prop A | prop B | rho |
|---|---|---|
| net_charge | pI | +0.941 |
| hydrophobic_patch_total_area | sap | +0.878 |
| hydrophobic_patch_total_area | hydrophobic_patch_n_large | +0.843 |
| iupred3 | iupred3_fraction_disordered | +0.766 |
| hydrophobic_patch_n_large | sap | +0.752 |
| hydrophobic_patch_total_area | rg | +0.703 |
| tango | rg | +0.603 |
| swi | iupred3 | +0.600 |
| sap | rg | +0.564 |
| scm_negative | rg | -0.560 |

Bonferroni thresholds at α=0.05: naive (14 tests) 0.00357; Li-Ji Pearson (9.00 tests) 0.00556; Li-Ji Spearman (10.00 tests) 0.00500.

Per-bin Li-Ji M_eff (Spearman, PDB) is 9–10 across all 50-residue bins from [300,800), so the property-clustering structure is stable across length.

**Possible narrative:** **Non-narrative — kept for tuning/decision-making.** Direct downstream uses:
- The strong native correlation between (net_charge ↔ pI) is **why E007's net_charge-up steering also moved pI upwards** by +0.79 even though pI was not steered — collateral on natively-correlated properties is expected.
- The (scm_+ ↔ scm_−) negative correlation predicts that steering scm_+ up will pull scm_− down. Worth noting if a scm experiment is ever designed.
- M_eff ≈ 9 means "reporting 13 separate p-values is overcounting"; multiple-testing thresholds in any future steering eval should be Li-Ji-corrected.

**Methodological caveats:**
- Generated n=100 vs PDB n=56,008. Length-KS at this sample size has limited power to detect mid-tail mismatches.
- Generated samples are from a single unguided checkpoint — does not test property-distribution drift across sampling configurations.
- Li-Ji M_eff assumes Gaussian-like marginals; some properties (fraction_disordered, hydrophobic_patch_n_large) are heavy-tailed and the M_eff estimate is approximate.

---

## E006 — Steering smoke test (pre-round1, 2026-04)

**Status:** finished.

**Why ran:** End-to-end engineering check before running the real steering eval (E007). Confirms (a) the predictor checkpoint loads, (b) gradients propagate through `z_1_est = z_t + (1-t) v` without numerical issues, (c) guided + unguided runs both produce property CSVs in the expected schema, (d) `comparison.csv` and `summary.csv` get written. No claim attached.

**Configs:**
- Run dir: `results/steering_eval/smoke_test/`.
- Outputs present: `run_config.yaml`, `guided_properties.csv`, `unguided_properties.csv`, `comparison.csv`, `summary.csv`, `diagnostics/`. (No `report.txt`, this was a pre-flight only.)

**Results:** all expected files written; pipeline shape OK. No quantitative claim recorded for this run.

**Possible narrative:** **Non-narrative — engineering smoke.** Logged here only so the existence of `results/steering_eval/smoke_test/` is traceable.

**Methodological caveats:** N/A (smoke test).

---

## E007 — Steering round 1: net_charge ↑ (2026-04)

**Status:** finished.

**Why ran:** First real steering evaluation on La-Proteina. net_charge was chosen because (a) E001 ranked it as one of the most probe-accessible properties (Class A, R² ≈ 0.97 at Tx, ~0.84 already at per-residue MLP h128), so the gradient signal is expected to be reliable, (b) net_charge has well-defined sign (no symmetry issue) and a wide PDB range, so a "successful steer" produces a large, easy-to-detect shift, (c) the predictor was trained at z-score scale and unit-normalised gradients are used, so this is also a test that `w_max` is the only knob needed to control magnitude.

**Configs:**
- Objective: `[{"direction": "maximize", "property": "net_charge", "weight": 1.0}]`.
- Sample N: 5 guided + 5 unguided proteins (paired).
- ODE: 200 steps; `inference_ucond_notri` family; backbone-only (`bb_ca`) + latent steering (latent channel only).
- Predictor: same checkpoint as E001 (`PropertyTransformer`, 128d, 3L, 4h).
- Run dir: `results/steering_eval/round1_net_charge_up/`. Files: `run_config.yaml`, `report.txt`, `summary.csv`, `comparison.csv`, `guided/`, `unguided/`, `diagnostics/`.

**Results:**

Steered property (intended target):

| property | mean Δ | std Δ | p-value | frac correct direction |
|---|---|---|---|---|
| net_charge_ph7 | **+23.45** | 4.61 | **0.0003**\*\*\* | **1.00** |

Collateral effects on the other 13 properties (non-steered):

| property | mean Δ | std Δ | p-value |
|---|---|---|---|
| swi | -0.0058 | 0.0027 | 0.0086\* |
| tango_total | -0.347 | 27.12 | 0.979 |
| tango_aggregation_positions | -0.20 | 0.45 | 0.374 |
| pI | +0.794 | 0.929 | 0.129 |
| iupred3_mean | -0.040 | 0.038 | 0.078 |
| iupred3_fraction_disordered | -0.059 | 0.053 | 0.068 |
| shannon_entropy | -0.027 | 0.025 | 0.076 |
| hydrophobic_patch_total_area | +182.5 | 338.5 | 0.294 |
| hydrophobic_patch_n_large | +0.6 | 1.14 | 0.305 |
| sap_total | -0.313 | 0.453 | 0.198 |
| scm_positive | +15.63 | 14.57 | 0.075 |
| scm_negative | **+12.92** | 3.88 | **0.0017**\* |
| radius_of_gyration | +0.0004 | 0.0024 | 0.735 |

Designability (ESMFold scRMSD): **not computed in this run** ("ESMFold not available, skipping designability evaluation"). Verdict from `report.txt`: "STEERING WORKS: all steered properties shifted significantly in the correct direction."

Notable collateral analysis:
- `pI` rises by +0.79 (not significant, p=0.129) — directionally consistent with E005's strong native correlation `(net_charge, pI) Pearson +0.855 / Spearman +0.941`. Steering net_charge up *should* drag pI up; the small N=5 means it didn't reach significance but the sign is right.
- `scm_negative` rises by +12.9 (p=0.002, significant) — also consistent with the E005-observed `(net_charge, scm_negative) Pearson +0.524`.
- `scm_positive` rises by +15.6 (p=0.075, marginal) — adding positive charges naturally pulls SCM-positive up; expected.
- `swi` drops by 0.006 (p=0.009, significant). SWI std=0.01, so this is a 0.6-σ drop. Sign is consistent with hydrophobicity dropping when net_charge rises.

**Possible narrative:** **Potential narrative.** Could become a Finding ("steering works for the most probe-accessible Class A property; collateral effects on the strongly natively-correlated properties (pI, scm_±) are expected from E005 and observed; designability not yet measured"), but **N=5 is too small to write into the paper** without scaling up. The natural follow-up is N=30–50 with ESMFold designability included. Logged here so the result is recoverable; not yet promoted to `content_masterarbeit.md`.

**Methodological caveats:**
- N=5 is below standard significance thresholds for collateral-effect inference. The "significant" entries (swi, scm_negative) survive Bonferroni-13 (threshold 0.0038) for swi but not for scm_negative; under Li-Ji-9 from E005 (threshold 0.0056) only swi survives.
- Designability not computed → cannot tell whether the +23 net_charge shift came at the cost of going off-manifold. Until ESMFold is wired in for this eval, the verdict "STEERING WORKS" is provisional.
- Steering was applied via the latent channel only; no backbone-channel guidance was tested in this round.

---

## E008 — Canonical CA-only baseline training (old recipe, 2026-04-21 → ongoing chain)

**Status:** finished (chain). Best raw checkpoint preserved on disk.

**Why ran:** Reference baseline against which all CA-only architectural variants (E010 sparse attention, future conv-downsampling) are compared. Goal: lock in a single, citable run with a documented config, val curve, and designability table. **Decisions encoded by this run** are listed below; future variants should not silently revisit them.

**Configs:**
- Run name: `test_ca_only_diffusion`. Store dir: `/home/ks2218/la-proteina/store/test_ca_only_diffusion/1776805213/`.
- Saved exp-config (source of truth): `…/checkpoints/exp_config_test_ca_only_diffusion.json`.
- Wandb chain: `d1k1587u` → `jeponiu5` → `0fnyfbi9`.
- Best raw checkpoint on disk: `…/checkpoints/best_val_00000026_000000002646.ckpt`. (The original step-2204 best from `jeponiu5` was overwritten by later `best_val_*` saves under `save_top_k=1`.)
- Hardware: 1× A100 (Cambridge HPC ampere); `ngpus_per_node_=1`, `nnodes_=1`.

Architecture (NN config — exact match to `configs/nn/ca_only_score_nn_160M.yaml`):
- 160M-parameter `LocalLatentsTransformer`. `nlayers=14`, `token_dim=768`, `nheads=12`, `parallel_mha_transition=False`, `use_qkln=True`.
- Output: `output_parameterization: {bb_ca: v}`. No `local_latents` head, no autoencoder, `latent_dim=None`.
- Pair representation: `pair_repr_dim=256`, `seq_sep_dim=127`, `xt_pair_dist_dim=30 (0.1–3 nm)`, `x_sc_pair_dist_dim=30 (0.1–3 nm)`.
- Conditioning: `dim_cond=256`, `t_emb_dim=256`, `idx_emb_dim=256`.
- Features: seq = `[xt_bb_ca, x_sc_bb_ca, optional_ca_coors_nm_seq_feat, optional_res_type_seq_feat]`; pair = `[rel_seq_sep, xt_bb_ca_pair_dists, x_sc_bb_ca_pair_dists, optional_ca_pair_dist]`; pair-cond = `[time_emb_bb_ca]`.
- Deliberately off: `update_pair_repr=False`, `use_tri_mult=False`, `use_downsampling=False`, `parallel_mha_transition=False`, `strict_feats=False`, no LoRA (`lora.r: null`).

Recipe (the "old recipe" — locked-in canonical for variants):
- `torch.optim.AdamW`, `weight_decay=0.05` uniform, `lr=2e-4` constant (no scheduler, no warmup, no decay). β1=0.9, β2=0.999, ε=1e-8 (PyTorch defaults).
- `accumulate_grad_batches=32`, `dataset.datamodule.batch_size=6`, `max_padding_size=512` → effective batch ≈ 192 proteins/optimizer step.
- bf16-mixed precision (`force_precision_f32: False`), `gradient_clip_val=1.0` norm.
- EMA: `decay=0.999`, `every_n_steps=5`, `validate_original_weights=False`, `cpu_offload=False`.
- `val_check_interval=2000` mini-batches → ~63 optimizer steps between val evals.
- Self-conditioning on (`self_cond=True`), `n_recycle=0`, `motif_conditioning=False`, `p_folding_n_inv_folding_iters=0.0`, `use_precomputed_latents=False`.
- Data filter: `worst_resolution ≤ 2.0 Å`, `min_length=50`, `max_length=512`. Sequence-similarity 0.5 split, val set size = 4058 proteins.
- `seed=42`, `dist_strategy=auto`.

**Results:**

Validation:
- Best val ≈ 4.71–4.77 around opt step 1800–2200. `d1k1587u` best 4.765 at step 1827; `jeponiu5` best 4.712 at step 2204 (ckpt overwritten).
- Past best, val rises to 5+ within 200–700 more steps (overfit).

Designability (ESMFold scRMSD < 2 Å, 200 ODE steps, N=3 per length):

| step | L=50 (min/mean/max scRMSD Å) | L=50 des | L=100 | L=100 des | L=200 | L=200 des |
|---|---|---|---|---|---|---|
| 1889 | 1.56 / 3.00 / 4.07 | 1/3 | 1.66 / 2.01 / 2.56 | 2/3 | — | — |
| 2457 (post-uptick) | 1.29 / 2.40 / 3.59 | 1/3 | 1.54 / 5.10 / 12.03 | 2/3 | 4.04 / 7.91 / 11.45 | 0/3 |

These numbers are the bar a variant must clear.

**Decisions encoded in this run (do NOT silently revisit them in variants):**
- wd held at 0.05 because higher wd collapses AdaLN-Zero output gates and destroys designability while improving val loss (E009 / Finding 5). Raising wd requires restructuring `configure_optimizers` first.
- LR schedule constant because cosine_with_warmup did not help in v2 (it co-occurred with the wd=0.1 collapse and was not isolated).
- `update_pair_repr=False` — no evidence the pair-update layer helps the CA-only task, and it adds compute.
- `use_tri_mult=False` — incompatible with the planned sparse-attention variant (`pair_update.py:65` raises) and unnecessary in baseline.
- 1-GPU configuration with `accumulate_grad_batches=32` is the deliberate match to the original 4-GPU effective batch (`4×8×6 = 1×32×6`).
- N=3 designability checks per length at 2–3 lengths is the cheap proxy for sample quality. Required as a stopping rule for any variant — val loss alone is insufficient (see E009).

**Possible narrative:** **Yes — this is the "Baseline reference" anchor in `content_masterarbeit.md`** (`## Baseline reference — canonical CA-only run`), and is referenced by Findings 5 and the run-comparison entries.

**Methodological caveats:**
- N=3 designability per length is small for fine-grained scRMSD distribution claims; sufficient for "designable vs not" gating but not for headline numbers.
- Step-1889 and step-2457 designability was measured at the time those checkpoints existed; the original ckpts were overwritten under `save_top_k=1` (the file currently on disk is step 2646). Per-step designability claims can no longer be re-run from disk for those exact steps.
- Wall-clock per opt-step is ~131 steps/hour with the v2-era `on_before_optimizer_step` logging in place (~300 steps/hour without); the two full-parameter L2 traversals per step are the bottleneck. Throughput was higher during the original training.

---

## E009 — v2 recipe attempt: wd=0.1 + cosine_with_warmup (2026-04-23 → 2026-04-25)

**Status:** finished, cancelled at step 2294 after a confirmed two-eval val uptick. Best raw + EMA checkpoints preserved.

**Why ran:** Test whether the standard "modern" recipe (wd=0.1 + cosine_with_warmup LR) improves on the old recipe (wd=0.05, constant LR=2e-4) on the canonical CA-only baseline. Hypothesis was that this would deliver a strict improvement to the baseline. **Result: it did not — see post-mortem below; this experiment is the basis of Finding 5.**

**Configs:**
- Run name: `ca_only_diffusion_baseline_v2`. Store dir: `store/ca_only_diffusion_baseline_v2/1776975226/`.
- Wandb chain: `9jp15of2` (slot 1) → `5rftn43a` (slot 2) → `43xxlbzt` (slot 3, after a chain failure on broken GPU node `gpu-q-43`).
- Best raw checkpoint (preserved): `…/checkpoints/best_val_00000020_000000002078.ckpt`. EMA companion at the same path with `-EMA.ckpt` suffix.
- Hardware: 1× A100 (Cambridge HPC ampere), 3 chained 6h SLURM slots, ~18h wall-clock total to step 2294.
- Architecture: identical to E008.
- Recipe diff vs E008:
  - `torch.optim.AdamW`, `weight_decay=0.10` (vs 0.05).
  - LR: `cosine_with_warmup` (linear warmup 0 → 2e-4 over 200 opt steps, cosine decay to `min_lr_ratio × peak = 2e-5` at `total_steps=6000`) (vs constant 2e-4).
  - Both versions apply weight decay uniformly to all parameters (`configure_optimizers` does not split into wd/no-wd groups).
- Reference old checkpoint used in the post-mortem comparison: `store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt`.

**Results — validation:**

Best validation loss (`validation_loss/loss_epoch`, single MC estimate per eval):

| Recipe | Best val | At opt step | Behaviour past best |
|---|---|---|---|
| Old (E008) | **4.765** | 1827 | rises to 4.79–5.39 within 250–700 steps |
| New v2 | **4.437** | 2078 | rises to 4.78 by step 2267 |
| **Δ (v2 − old)** | **−0.328** | +251 steps | — |

Head-to-head v2 vs `d1k1587u` at matched optimizer steps (val_loss):

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

Per-length val (v2 only, around the uptick):

| length bin | step 2015 | step 2078 | step 2142 | step 2204 | step 2267 |
|---|---|---|---|---|---|
| 50–175  | 4.244 | 4.316 | 4.078 | 4.283 | 4.344 |
| 175–300 | 4.508 | 4.300 | 4.548 | 4.915 | 5.022 |
| 300–425 | 4.945 | 4.775 | 4.957 | 4.924 | 5.292 |
| 425–513 | 5.180 | 4.916 | 5.102 | 5.396 | 5.097 |

**Results — sample quality (designability via ESMFold scRMSD):**

After observing the val improvement, samples were generated under matching inference (`generation/uncond_codes_ca_only`, 200 ODE steps, `designability_modes=[ca, bb3o]`, `folding_models=[esmfold]`). N=3 per length, threshold scRMSD < 2 Å:

| Run / step | L=50 (min/mean/max) | L=50 des | L=100 | L=100 des | L=200 | L=200 des |
|---|---|---|---|---|---|---|
| Old, step 1889 | 1.56 / 3.00 / 4.07 | 1/3 | 1.66 / 2.01 / 2.56 | 2/3 | — | — |
| Old, step 2457 (post-uptick) | 1.29 / 2.40 / 3.59 | 1/3 | 1.54 / 5.10 / 12.03 | 2/3 | 4.04 / 7.91 / 11.45 | 0/3 |
| **v2, step 2078 (best val)** | **4.22 / 9.10 / 14.83** | **0/3** | **8.00 / 11.28 / 13.41** | **0/3** | **7.96 / 9.60 / 11.03** | **0/3** |

v2 produces **zero designable samples at any tested length**. Even the v2 *minimum* scRMSD at L=50 (4.22 Å) is worse than the old *maximum* (4.07 Å, step 1889).

**Results — per-layer weight diff (post-mortem):**

Loaded both raw checkpoints on CPU and computed L2 norm per parameter tensor in `state_dict()`:

- Global weight L2 norm: v2 = 430.33, old = 438.73 → ratio 0.981 (v2 only 1.9% smaller globally; cannot account for the sample collapse on its own).
- Layer-wise ratio (v2/old) over 164 layers ≥ 10k params: mean = 0.920, median = 0.967, stdev = 0.148, **min = 0.260, max = 1.376**.
- Top-10 most-changed layers (largest |ratio − 1|) are **all** AdaLN-Zero output gates of upper transformer blocks:

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

The 10 most-similar layers (ratio 0.99–1.00) are all AdaLN modulation γ/β weights — essentially unchanged.

**Mechanism (DiT/SiT-style AdaLN-Zero × naive uniform-AdamW-wd):**

AdaLN-Zero (DiT, Peebles & Xie 2023) adds a per-block output gate `α(c)` modulating each residual contribution: `x ← x + α(c)·Block(AdaLN(x, c))`. The linear layer producing α is **zero-initialized**, so the network behaves as identity at init; the gates need to *grow* under gradient signal. Weight decay's job is to push weights toward zero. With uniform wd applied to all parameters including the gates, the gradient signal is in continuous tension with the wd pull. At wd=0.05 the gates grow (slowly) to useful magnitudes; at wd=0.1 — especially in deeper layers where gradient signal is weaker — wd pull dominates, gates stay small. Suppressed gates → conditioning barely reaches velocity output → predicted velocities ≈ time-averaged velocity (smoother, lower-variance MSE → lower val loss) → integrated trajectories at inference have no coherent time-conditioning → samples collapse.

Standard fix in DiT/SiT/SD3: parameter groups in AdamW that exclude (a) AdaLN-Zero gate parameters, (b) biases, (c) LayerNorm γ/β, (d) embeddings from weight decay. La-Proteina's `configure_optimizers` does not implement this split. With the codebase as-is, **wd is bounded above by what AdaLN-Zero gates can tolerate**, experimentally ≤ 0.05.

**Possible narrative:** **Yes — this is Finding 5** (`content_masterarbeit.md → ## Finding 5`). The narrow claim there: "wd=0.1+cosine reduces best val by 0.328 but produces 0/3 designable at every L; per-layer weights show 40–74% gate-norm reduction in upper transformer layers; val loss is therefore not a reliable proxy on this codebase under uniform-wd AdamW."

A causal ablation isolating the wd from the LR schedule (and confirming gate-recovery via param-group fix recovers samples) is registered in `content_masterarbeit.md → Future experiments → Causal ablation of the AdaLN-Zero × weight-decay collapse mechanism`.

**Methodological caveats:**
- N=3 designability per length is the same low-N gate as E008; the categorical gap (every v2 sample worse than every old sample at every length) holds regardless.
- Step-1889 / step-2457 old-recipe ckpts were overwritten under `save_top_k=1`; the per-layer post-mortem used step-2646 (post-uptick from chained continuation). Despite being *worse* by val-loss, step-2646 still produces dramatically better samples than v2-2078, so the v2 collapse cannot be explained by old-checkpoint selection.
- The mechanism is consistent with the per-layer evidence and DiT-family literature, but has not been formally verified by an ablation. That ablation (~16h on 1 A100) is registered as future work.
- v2 had two confounded variables (wd 0.05→0.10 + scheduler constant→cosine_with_warmup). The mechanism is wd-specific, not LR-schedule-specific (LR decay slows gate growth but does not pull weights toward zero), so the wd is the load-bearing cause on mechanistic grounds. A causal ablation that varies them independently would settle the residual ambiguity.
- The val-loss numbers themselves (Δ = −0.328 in best-val) are real and reproducible; the framing of v2 as "an improvement" is what is retracted, not the val number.
- Chain was cancelled at step 2294 with cosine LR still at 1.48e-4 (out of 6000 scheduled). The collapse is therefore not formally proven to not recover with further training, but the mechanism predicts further training would *worsen* gate suppression, not recover it.

---

## E010 — Sparse-attention variant K=32 training (2026-04-25, in progress)

**Status:** in progress (training; ≥ step 1259 as of 2026-04-26). Designability eval pending.

**Why ran:** Architectural variant of the CA-only baseline (E008). Replaces dense `[B,N,N,d]` pair representation + dense attention with a per-residue neighbor list. The thesis question is two-fold: (a) does sparse attention preserve sample quality at matched recipe and matched per-step training budget? (architectural axis), and (b) does the implementation realise the FLOP savings as wall-clock at n=512? (throughput axis). Defensible negative throughput finding already observed at smoke-test time (see below).

**Configs:**
- Run name: `ca_only_sparse_K40` (**misnomer — actual K=32, not 40**; see below). Store dir: `store/ca_only_sparse_K40/1777125234/`.
- Saved exp-config: `…/checkpoints/exp_config_ca_only_sparse_K40.json`.
- Wandb chain: `c60iiywv` → `pgdo2dw3` (training in progress).
- Architecture (sparse arm): `configs/nn/ca_only_sparse_160M.yaml` — byte-equivalent to `ca_only_score_nn_160M.yaml` (E008's NN config) except for four added keys:
  - `sparse_attention=True`
  - `n_seq_neighbors=8` (NOT 16 as the run name suggests)
  - `n_spatial_neighbors=8`
  - `n_random_neighbors=16` (∝ 1/d³)
  - ⇒ K = 8 + 8 + 16 = **32** (not 40).
  - Verified 2026-04-26 from saved exp_config and runtime `cfg_exp.nn` log.
  - The original design intent had been 16/8/16=K=40; the YAML committed and run is 8/8/16=K=32. The run name is preserved to keep the store-dir and wandb history valid; an actual K=40 run would be a separate variant.
- Architecture (dense control = E008): `store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt`. Not retrained.
- Recipe: identical to E008 (canonical OLD recipe — wd=0.05, constant LR=2e-4, no scheduler, accumulate_grad_batches=32, batch_size=6, EMA decay=0.999 every 5 steps, seed=42, bf16-mixed, val_check_interval=2000, data filter ≤2.0 Å resolution + length 50–512, 1× A100). Verified by structural diff of the three `exp_config_*.json` (E008/E009/E010) on 2026-04-26.
- Eval configs (created 2026-04-25):
  - `configs/inference_ucond_notri_ca_only_baseline_quick.yaml` — control, 4 lengths × 10 samples × 200 steps, points at E008's best ckpt.
  - `configs/inference_ucond_notri_ca_only_sparse_quick.yaml` — variant; ckpt path is `PLACEHOLDER_best_val.ckpt`, must be filled in after training (see note below on `evaluate.py` argparse strictness).

Implementation files (worth knowing if anything in the sparse path is touched):
- `proteinfoundation/nn/modules/sparse_neighbors.py` — neighbor list builder (`@torch.no_grad`, recomputed each forward from `x_t["bb_ca"]`).
- `proteinfoundation/nn/modules/pair_bias_attn.py:_attn_sparse` — actual sparse attention. Switched on by presence of `neighbor_idx` argument.
- `proteinfoundation/nn/modules/pair_update.py` — sparse pair update; **raises if `use_tri_mult=True`** (line 65).
- `proteinfoundation/nn/modules/pair_rep_initial.py` — sparse-aware pair builder.
- `proteinfoundation/nn/feature_factory.py:130` — `_gather_sparse_pairs` fallback for any pair feature without `supports_sparse=True`. All current pair features have the fast path.
- `proteinfoundation/nn/local_latents_transformer.py:228-242` — wires sparse_attention from kwargs.

**Results (training):**
- Best-val ckpt at step 1259 (training in progress; final step pending).
- val curve and per-length val being logged via the same `validation_loss/loss_epoch` and `validation_loss_by_len/len_<lo>_<hi>` channels as E008/E009.

**Results (throughput, smoke-test, 2026-04-25):**
At n=512, K=32, B=6, H=12, D=64 on a single A100 (bf16-mixed, 160M model), **the sparse-attention variant runs SLOWER per optimizer step than the dense baseline** despite reducing the pair representation from `[B,N,N,d_pair]` (≈ 803 MB) to `[B,N,K,d_pair]` (≈ 50 MB) and reducing attention scores from `[B,H,N,N]` to `[B,H,N,K]` (a 16× reduction).

Mechanism (identified by code-level inspection of `_attn_sparse`):
- Sparse path materialises two `[B*H, N, K, D]` tensors per layer via `torch.gather` along the N dimension on a non-contiguous index pattern. At our shapes that's ≈ 150 MB × 2 × 14 layers ≈ 4 GB of memory-bound traffic per forward, with random N-axis access.
- Dense path has zero gathers — Q, K, V are already laid out contiguously for matmul; the dense attention kernel is bandwidth-friendly.
- Both paths use plain `einsum + softmax + einsum` (no flash/SDPA fusion), so dense does not get a kernel-fusion advantage. The throughput gap is entirely memory-access-pattern.

Crossover with dense is hypothesised at n ≥ 1024 but not measured.

**Possible narrative:** **Two axes**, both tracked here, both feed `content_masterarbeit.md → Future experiments → Sparse-attention variant vs dense baseline (pre-registered, 2026-04-25)`:

1. *Architectural axis (pending):* val-loss-vs-step + per-length val + designability at matched optimizer step. Headline claim form: *"At matched recipe and matched per-step budget on the 160M CA-only baseline, K=32 SALAD-style sparse attention (8 seq + 8 spatial + 16 random ∝ 1/d³) reaches val=X / designability=Y vs dense val=4.71-4.77 / designability per Finding 5."*

2. *Throughput axis (already defensible — negative result):* Sparse is slower per opt step than dense at n=512 in this implementation. **Defensible narrow claim already today:** *"Replacing the CA-only baseline's dense pair representation and attention with a SALAD-style K=32 sparse neighbor-list attention reduces the pair-representation memory footprint by ≈ 16× at n=512 but does not realise the FLOP savings as per-step wall-clock; the gather-based kernel is memory-bandwidth-bound."* This is the honest framing the thesis should adopt regardless of the architectural-axis outcome.

**Methodological caveats:**
- Single seed on the architectural axis. N=10 designability per length × 4 lengths is gating, not a definitive headline number.
- "Random" neighbors are ∝ 1/d³, not uniform — closer to "extra spatial neighbors with stochastic exploration" than to BigBird-style global tokens. Long-range information transport relies entirely on multi-layer composition.
- Self is excluded from each query's neighbor list (`eye` added to `base_invalid` in `sparse_neighbors.py:44`); self-info propagates only via the residual.
- Padding-slot guard (`slot_valid` in `sparse_neighbors.py:121-127`) prevents short proteins (<K=32 residues) from double-counting residue 0 in attention. Critical and untouchable.
- Neighbor list rebuilt every forward from the noisy `x_t`. At t≈0 the spatial+random groups are essentially random subsets and only sequential neighbors carry useful info — the model is implicitly trained on a connectivity-noisy-early curriculum. Whether that hurts low-t sample quality in a way that doesn't show in val loss is a known unknown.
- Throughput numbers from a single A100 in bf16-mixed; absolute steps/hour will differ on other hardware, but the relative dense-vs-sparse ordering at n=512 is structural.
- `evaluate.py` does NOT honour Hydra CLI overrides for `ckpt_path`/`ckpt_name` (its argparse is strict — see `gen_n_eval_ca_only.sh:97-102`). The eval-step YAML must be edited with the actual ckpt before running. `generate.py` does honour CLI overrides.
- Throughput-axis allocator tweak occasionally helps gather-heavy bf16: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (numerically no-op).

**Things deliberately NOT combined with sparse attention on this first variant run** (so the comparison stays clean): no `update_pair_repr=True`, no `use_downsampling=True`, no non-default K (K=32 only), no LoRA. Those become separate variants if K=32 produces a working result.

---

## E011 — Sidechain manifold experiment (preregistered, 2026-04-25)

**Status:** preregistered / in progress (initial commit `e4ba5a6` "Sidechain manifold experiment + in-progress source/configs"). Result not yet recorded.

**Why ran:** Test whether the AE latent is more "manifold-aligned" than raw sidechain coordinates — i.e. at matched percentile-scaled noise levels (k · σ for both spaces), which space produces sidechain placements that ESMFold (conditioned on the original sequence) is closer to. If the AE latent is more manifold-aligned, that empirically justifies steering in latent space rather than coordinate space (the implicit assumption of every steering experiment so far). If it is not, steering in coord space becomes a viable alternative the thesis should at least mention.

**Configs (locked in):**
- AE checkpoint: `AE1_ucond_512.ckpt` (the 512-residue AE used for the original 355K precomputed latents, paired with LD1 in the original release).
- LD checkpoint: *not used.* This experiment touches only the AE encode/decode round-trip; the flow model is irrelevant.
- Eval set: length-stratified subset of 50–300 residue proteins from `/rds/user/ks2218/hpc-work/processed/`, seed-fixed.
- Noise levels: k ∈ {0.1, 0.3, 0.5, 1.0, 2.0}.
- Coord arm: Gaussian noise added to sidechain atoms (atom37 indices ∉ {0:N, 1:CA, 2:C, 4:O}) only; backbone untouched. σ = empirical per-(residue_type, atom_idx) std of the atom's offset from CA in the residue-local (N,CA,C) frame, computed across the eval set.
- Latent arm: Gaussian noise added to encoder `mean` with σ = empirical per-dim std on the eval set (≈ 1, since latents are KL-regularised toward N(0,1)). Decode with original CA coords; splice the original N/CA/C/O back so the *only* difference between conditions is sidechain placement.
- Metric: `proteinfoundation/evaluate.py` with `compute_codesignability=True`, `codesignability_modes=["all_atom"]`, `codesignability_folding_models=["esmfold"]`. ESMFold on original sequence vs perturbed structure, all-atom RMSD. Lower = closer to ESMFold's manifold.
- Code: `analysis_manifold/perturbation_experiment.py`, `analysis_manifold/aggregate_and_plot.py`.

**Why short proteins (50–300) — explicit compute-saving choice:** Sidechain conformational constraints are predominantly local (rotamer preferences, immediate neighbour packing, ~5–8 Å context). Restricting to 50–300 residues loses no power on the central claim and makes the experiment tractable on a single A100. If positive on short proteins, scale to 300–800 with AE2/LD3 as the natural follow-up; if negative, the result already disconfirms the hypothesis at the regime where it is most likely to hold.

**Caveat to record with results:** AE1 was trained on ≤512 residue proteins, so 50–300 is fully in-distribution for the encoder. A positive result for AE1 in 50–300 does not transfer mechanically to AE2 / 300–800.

**Results:** *not yet collected.* Update this entry once aggregate plots (`analysis_manifold/aggregate_and_plot.py` outputs) are produced.

**Possible narrative:** **Yes, intended to be a Finding** if the result is clean. Cross-reference: `content_masterarbeit.md → Future experiments → Sidechain manifold comparison`.

**Methodological caveats:**
- ESMFold's all-atom predictions are themselves a model output, not ground truth. The RMSD-to-ESMFold metric measures *distance to ESMFold's manifold*, not necessarily distance to the true protein manifold. A mismatch with crystal structure is therefore confounded with ESMFold's own bias.
- Noise scaling at "k · σ" is per-axis-per-modality, not equivalent in information-theoretic terms. Comparing latent-arm and coord-arm at matched k assumes equal informativeness of the σ-units across spaces — a working assumption that the experiment is partially testing.
- Splicing original N/CA/C/O back from the unperturbed structure into the latent-arm decode gives the latent arm a small backbone-fidelity advantage (CA placement is exactly preserved) that the coord-arm doesn't get (sidechains are perturbed in a frame defined by the unperturbed N/CA/C, but the local frame interaction with the perturbation isn't trivially equivalent).

---

## E012 — Three-run comparison: baseline / v2 / sparse side-by-side (2026-04-26)

**Status:** finished (at-time-of-comparison snapshot; sparse arm still training).

**Why ran:** Single citable record of the three CA-only training runs whose configs and outcomes are referenced elsewhere in `content_masterarbeit.md` (E008 baseline, E009 v2, E010 sparse). Confirmed by structural diff of the three saved `exp_config_*.json` files that everything except the per-run differing keys is byte-identical — so any "the variant beat the baseline" claim resolves to a row of one of these tables.

**Configs (only differing keys shown — everything else byte-identical):**

| key | baseline (E008) | v2 (E009) | sparse (E010) |
|---|---|---|---|
| `opt.weight_decay` | **0.05** | **0.10** | 0.05 |
| `opt.scheduler` | *(absent — constant LR)* | `cosine_with_warmup`, warmup=200, total=6000, min_lr_ratio=0.1 | *(absent — constant LR)* |
| `nn.sparse_attention` | *(absent → False)* | *(absent → False)* | **True** |
| `nn.n_seq_neighbors` | — | — | 8 |
| `nn.n_spatial_neighbors` | — | — | 8 |
| `nn.n_random_neighbors` | — | — | 16 |

Common to all three: `opt.lr=2e-4` constant, `opt.accumulate_grad_batches=32`, `opt.dist_strategy=auto`, `opt.val_check_interval=2000`, `hardware.ngpus_per_node_=1`, `hardware.nnodes_=1`, EMA(decay=0.999, every_n_steps=5), `seed=42`, `force_precision_f32=False`, `training.self_cond=True`, `training.n_recycle=0`, `training.p_folding_n_inv_folding_iters=0.0`, `training.use_precomputed_latents=False`, dataset filter `worst_resolution≤2.0Å, min_length=50, max_length=512`, NN backbone `nlayers=14, token_dim=768, nheads=12, pair_repr_dim=256, dim_cond=256, update_pair_repr=False, use_tri_mult=False, use_downsampling=False`.

**Identity:**

| | baseline | v2 | sparse |
|---|---|---|---|
| `run_name_` | `test_ca_only_diffusion` | `ca_only_diffusion_baseline_v2` | `ca_only_sparse_K40` (misnomer — actual K=32) |
| store dir | `store/test_ca_only_diffusion/1776805213/` | `store/ca_only_diffusion_baseline_v2/1776975226/` | `store/ca_only_sparse_K40/1777125234/` |
| wandb chain | `d1k1587u → jeponiu5 → 0fnyfbi9` | `9jp15of2 → 5rftn43a → 43xxlbzt` | `c60iiywv → pgdo2dw3` (training in progress) |
| training started | 2026-04-21 | 2026-04-23 | 2026-04-25 14:53 BST |
| status | finished | finished, cancelled at step 2294 | in progress (≥ step 1259 as of 2026-04-26) |

**Mini-eval results (designability via ESMFold scRMSD < 2 Å, 200 ODE steps, N=3 per length):**

| Run / step | L=50 (min/mean/max) | L=50 des | L=100 | L=100 des | L=200 | L=200 des |
|---|---|---|---|---|---|---|
| baseline @ step 1889 | 1.56 / 3.00 / 4.07 | 1/3 | 1.66 / 2.01 / 2.56 | 2/3 | — | — |
| baseline @ step 2457 (post-uptick) | 1.29 / 2.40 / 3.59 | 1/3 | 1.54 / 5.10 / 12.03 | 2/3 | 4.04 / 7.91 / 11.45 | 0/3 |
| v2 @ step 2078 (best val) | 4.22 / 9.10 / 14.83 | **0/3** | 8.00 / 11.28 / 13.41 | **0/3** | 7.96 / 9.60 / 11.03 | **0/3** |
| sparse @ best val | *(eval not yet run)* | — | — | — | — | — |

Best validation loss (single MC estimate per eval):

| Run | best val | at opt step | behaviour past best |
|---|---|---|---|
| baseline | 4.71–4.77 (4.765 in `d1k1587u`) | 1827–2204 | rises to 4.79–5.39 within 250–700 steps |
| v2 | **4.437** | 2078 | rises to 4.78 by step 2267 |
| sparse | (training in progress; latest best-val ckpt at step 1259) | — | — |

**Diff isolation — what each row's outcome can and cannot be attributed to:**

- **baseline vs v2** differs in *exactly two* knobs: `weight_decay` (0.05→0.10) and `scheduler` (constant→cosine_with_warmup). Two confounded variables, one outcome ("better val, dead samples"). Mechanism (Finding 5 / E009) — AdaLN-Zero gate collapse in upper transformer layers (gates at 26–60% of baseline magnitude in v2) — is wd-specific, not LR-schedule-specific. On mechanistic grounds the wd=0.10 is the load-bearing cause; cosine LR plausibly compounds the suppression in late training but no known mechanism predicts gate collapse from cosine LR alone. A causal ablation that varies them independently would settle this (see Future experiments → Causal ablation in `content_masterarbeit.md`).
- **baseline vs sparse** differs in *exactly four* keys, all on the architecture axis (`sparse_attention=True` plus the three neighbor-count keys). The training recipe is byte-identical to the baseline. Therefore, when the sparse designability eval is run, the result is unambiguously attributable to architecture — there is no v2-style recipe confound. (Earlier session confusion suggested the sparse run might have inherited the v2 recipe because `configs/training_ca_only.yaml` still had v2 leftover values at the time of sparse submission. Verified on 2026-04-26 from the saved `exp_config_ca_only_sparse_K40.json`: it did not. Hydra picks one root config per `--config-name`, and `training_ca_only_sparse.yaml` was always at the canonical recipe.)

**The K=40 misnomer (sparse run):**

The sparse run is named `ca_only_sparse_K40` and earlier writeups described the architecture as "K=40 = 16 sequential + 8 spatial + 16 random". The saved `exp_config_ca_only_sparse_K40.json` and runtime `cfg_exp.nn` log both show `n_seq_neighbors=8` (not 16), `n_spatial_neighbors=8`, `n_random_neighbors=16` ⇒ **K=32 (8 seq / 8 spatial / 16 random)**. Half the sequential count claimed in the docs. The model sees ±4 residues sequentially per layer, not ±8. Long-range information transport relies even more heavily on multi-layer composition than the K=40 framing suggested. The throughput observation is unaffected (gathered tensor `[B*H,N,K,D]` is 32/40 = 0.8× the K=40 size, still firmly memory-bound). Run name kept for store-dir / wandb continuity.

**Possible narrative:** **Yes — this is the "Run comparison — baseline / v2 / sparse" entry in `content_masterarbeit.md`** (`## Run comparison — baseline / v2 / sparse (clean config, side-by-side, 2026-04-26)`). Treat that section as the citation anchor for any thesis claim about these three runs.

**Methodological caveats:**
- Sparse arm is mid-training; its row in the designability table is empty until the post-training eval runs. Comparison is therefore 2-of-3 complete.
- Diff isolation argument for baseline vs sparse holds *only if* the eval is run on the locked recipe — re-tuning anything during the sparse run would re-introduce confounds.

---

## E013 — wd=0 ablation training (canonical recipe with `weight_decay=0.0`, 2026-04-26 → ongoing)

**Status:** in progress (training; first val-best ckpt at step 1638 evaluated; chain continues).

**Why ran:** Direct causal test of the mechanism proposed in Finding 6 / E009. That finding showed that increasing wd from 0.05 → 0.10 collapses AdaLN-Zero output gates in the upper transformer blocks (gates at 26-60% of canonical magnitude in v2) and destroys designability while *improving* val loss. The mid-session diagnostic on the canonical step-2646 ckpt extended this: even at the canonical wd=0.05, deep-layer (L7-13) AdaLN-Zero gate weights are ~50% of shallow-layer (L0-5) magnitudes, suggestive of partial gate suppression even at the recipe-recommended wd. The hypothesis: "even wd=0.05 is bottlenecking deep-layer conditioning enough that it caps designability — especially long-length generalization (L≥200) — and fully removing wd lets those gates grow without harming convergence." This is "Variant B" of the Causal-ablation follow-up section in `content_masterarbeit.md`. Decision input for whether the canonical recipe should be revised to wd=0 (matching the DiT/SiT literature default) before any further architectural variants are run on top of it.

**Configs:**
- Run name: `ca_only_diffusion_wd0`. Store dir: `store/ca_only_diffusion_wd0/<run_id>/`.
- Wandb chain: pending — set per-slot via `WANDB_RUN_GROUP=ca_only_diffusion_wd0` (auto-grouped by 46fc39b).
- Training config: `configs/training_ca_only_wd0.yaml`. **Diff from canonical (`configs/training_ca_only.yaml`):** only `opt.weight_decay: 0.05 → 0.0`. Everything else byte-identical (same NN config `ca_only_score_nn_160M.yaml`, same dataset, same effective batch ≈ 192, same EMA, same seed=42, no scheduler block → constant LR=2e-4, `accumulate_grad_batches=32`, single-GPU `dist_strategy=auto`, bf16-mixed).
- Submit: `bash script_utils/submit_train_ca_only_1gpu.sh -n training_ca_only_wd0` with `--exclude=gpu-q-43`. Chain via `--dependency=afterany:$prev`.
- Hardware: 1× A100 ampere on Cambridge HPC (COMPUTERLAB-SL2-GPU), 6h slot, `--time=6:00:00`.

**Results — training (live):**
- First useful checkpoint: `best_val_00000016_000000001638.ckpt` (step 1638, epoch 16). Renamed locally to `wd0_step1638.ckpt`.
- val-loss curve at this stage is reportedly visually indistinguishable from canonical wd=0.05 in the same step range.
- Chain still alive — later checkpoints (step ≥ 2000) will be appended to this entry as they land.

**Results — eval at step 1638:**

(a) **N=3 single-seed quick probes** (used as the gating signal during training):

| seed | L=50 (min/mean) | L=50 des | L=100 (min/mean) | L=100 des | L=200 (min/mean) | L=200 des |
|---|---|---|---|---|---|---|
| 5 (default) | 5.07 / 8.70 | 0/3 | **1.89** / 8.02 | 1/3 | 12.94 / 13.72 | 0/3 |
| 100 | **1.04** / 4.47 | 1/3 | **1.49** / 6.36 | 1/3 | 10.73 / 12.54 | 0/3 |

(b) **N=30 batched eval (seed=100)** — see E014 for full protocol:

| L | min | p25 | median | mean | p75 | max | designable | rate |
|---|---|---|---|---|---|---|---|---|
| 50  | 1.24 | 1.81 | 2.47  | 4.17  | 4.17  | 18.52 | 10/30 | 33.3% |
| 100 | 1.33 | 2.35 | 4.12  | 5.29  | 8.00  | 12.29 | 4/30  | 13.3% |
| 200 | 4.53 | 9.62 | 12.10 | 11.52 | 13.51 | 16.76 | 0/30  | 0.0%  |

**Possible narrative:** **Yes — feeds Finding 8** (`content_masterarbeit.md → ## Finding 8`). Finding 8 frames the wd=0 result as currently in-progress; the cross-recipe N=30 comparison is in E014.

**Methodological caveats:**
- Single training run, single ckpt evaluated so far. Step 1638 is in the front edge of canonical's val-best window (1800-2200), so it is plausibly under-trained relative to canonical 2646.
- N=3 designability is too noisy to claim a wd=0 ↔ wd=0.05 difference on its own — the seed=5 vs seed=100 swing on the same step-1638 ckpt was 1/9 → 2/9 from a seed change alone. The N=30 eval (E014) is the gating data.
- AdamW equilibrium argument (`|θ_eq| ≈ |grad|/wd`) means wd=0 changes the equilibrium for *all* parameters, not just AdaLN-Zero gates. Without a per-layer gate-magnitude diagnostic on a wd=0 ckpt, "wd=0 helps because gates are larger" remains a mechanism inference, not a measurement. Diagnostic is owed before promoting Finding 8 to a Narrow claim.
- Canonical (E008) and v2 (E009) reached their best val at steps 1800-2200 / 2078; wd=0 may peak at a different step. Promoting Finding 8 to a "wd=X is best" claim requires comparing each recipe at *its own* peak ckpt, not at matched step.
- The pre-existing canonical ckpts at steps 692, 1638-equivalent, 1889, 2078, 2457, 2646 give a partial within-recipe designability trajectory under wd=0.05. wd=0 has only step 1638 so far. Without more wd=0 ckpts, the wd=0 trajectory cannot be drawn.

---

## E014 — Four-run N=30 designability comparison (baseline / v2 / wd0 / sparse, 2026-04-27)

**Status:** finished (one matched-seed N=30 batch per run; multi-seed replicates not yet collected).

**Why ran:** Previous side-by-side comparisons (E012, the N=3 runs in E008/E009/E010, and the N=3 single-seed probes in E013) had per-rate Wilson confidence intervals so wide that the rate-comparison between any two runs was nearly always overlapping at single-digit sample counts. Within a single seed, N=3 designability rates swing 0/3 ↔ 2/3 (0% ↔ 67%) just from the choice of three initial-noise samples. This was empirically observed on the wd=0 step-1638 ckpt (seed 5: 1/9 designable; seed 100: 2/9). Decision input that the multi-seed N≥30 batched comparison is required to make any "recipe X is better than recipe Y" claim about CA-only designability.

The natural minimum scope is "the four most important CA-only ckpts" — canonical baseline (the bar all variants must clear), v2 (the Finding 6 negative), wd0 (the Finding 8 in-progress causal ablation), sparse K40 (the architectural variant from E010). All compared at matched seed=100 so initial noise is byte-identical across runs (the ODE trajectory differs only by the model's velocity field).

**Configs:**
- Generation config: `configs/generation/uncond_ca_only_n30.yaml` — `nsamples: 30`, `max_nsamples_per_batch: 10`, `nres_lens: [50, 100, 200]`. Otherwise byte-identical to `uncond_ca_only_quick.yaml` (the N=3 default).
- Per-run inference stub configs: `configs/inference_baseline_n30.yaml`, `configs/inference_v2_n30.yaml`, `configs/inference_wd0_n30.yaml`, `configs/inference_sparse_n30.yaml`. Each is two lines: `defaults: [inference_ucond_notri_ca_only]`. Per-run differences passed as Hydra CLI overrides (`ckpt_name=…`, `seed=100`, `generation=uncond_ca_only_n30`).
- Pipeline: `run_n30_pipeline.sh` — sequential generate → eval → next, four runs, single tmux session `n30`. Idempotent: `rm -rf` of any prior `inference/inference_<run>_n30/` and the always-overwritten `inference/inference_base/` before each run.
- ESMFold patch: `proteinfoundation/metrics/folding_models.py` already had the L>250 batch_size=1 patch from the earlier 24GB-L4 work; no-op for L≤200 (which dominates this experiment), so it does not affect any of the L=50/100/200 numbers.
- Hardware: single L4 (24 GB), local machine (NOT Cambridge HPC). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` exported by the pipeline script.
- Wall-clock: 3h11min total (11:46 → 14:57 BST). Per-run breakdown: baseline 48 min, v2 48 min, wd0 48 min, sparse 47 min. Generation per run ~5-6 min; ESMFold + ProteinMPNN per run ~42-43 min (L=200 dominates with batch_size=4 in the unpatched-for-L≤200 path).

**Identity of the four ckpts:**

| run | ckpt filename (current) | from-run | step | wd | scheduler | sparse? |
|---|---|---|---|---|---|---|
| baseline | `baseline_wd0.05_step2646.ckpt` | `test_ca_only_diffusion` (E008) | 2646 | 0.05 | none (constant LR) | no |
| v2 | `v2_wd0.1_step2078.ckpt` (+ `-EMA`) | `ca_only_diffusion_baseline_v2` (E009) | 2078 | 0.10 | cosine_with_warmup (warmup=200, total=6000, min=0.1) | no |
| wd0 | `wd0_step1638.ckpt` | `ca_only_diffusion_wd0` (E013) | 1638 | 0.00 | none | no |
| sparse K40 | `sparse_K40_step1259.ckpt` | `ca_only_sparse_K40` (E010) | 1259 | 0.05 | none | yes (K=32 — see E012/E010 for misnomer note) |

Identification was by `torch.load(p, map_location='cpu', weights_only=False)['hyper_parameters']['cfg']['run_name_']` and `…['cfg']['opt']['weight_decay']`.

> **Identification correction (2026-04-27):** during pre-pipeline ckpt survey, `best_val_00000012_000000001259.ckpt` was initially mistaken for canonical wd=0.05 because (a) it was the second-most-recently rsynced and (b) the filename gives no recipe info. Loading hyper_parameters revealed `run_name_=ca_only_sparse_K40, sparse_attention=True, wd=0.05` — i.e. it is the sparse run's best-val ckpt, not canonical. All four important ckpts were then renamed to recipe-bearing names (above) so this kind of misidentification cannot recur. The N=3 step-1259 result reported earlier as "canonical wd=0.05, 0/9 designable" was therefore the sparse arm, not canonical; the misattribution was corrected in `content_masterarbeit.md → Finding 8`.

**Results — full per-length percentile tables:**

baseline (canonical, step 2646, wd=0.05):

| L | N | min | p25 | median | mean | p75 | max | designable (<2 Å) | rate |
|---|---|---|---|---|---|---|---|---|---|
| 50  | 30 | 0.76 | 1.14 | 1.65 | 2.89 | 3.39 | 11.48 | **19/30** | **63.3%** |
| 100 | 30 | 0.86 | 1.20 | 1.48 | 2.21 | 2.40 |  9.64 | **20/30** | **66.7%** |
| 200 | 30 | 1.50 | 2.81 | 4.57 | 5.87 | 9.60 | 12.26 | **3/30**  | **10.0%** |

v2 (step 2078, wd=0.10, cosine):

| L | N | min | p25 | median | mean | p75 | max | designable | rate |
|---|---|---|---|---|---|---|---|---|---|
| 50  | 30 | 1.08 | 2.13 | 4.23 | 6.14 | 9.70 | 19.05 | 7/30 | 23.3% |
| 100 | 30 | 1.41 | 2.20 | 3.70 | 5.30 | 7.29 | 17.58 | 5/30 | 16.7% |
| 200 | 30 | 3.58 | 5.20 | 9.72 | 8.92 | 11.13 | 14.66 | 0/30 |  0.0% |

wd0 (step 1638, wd=0.00):

| L | N | min | p25 | median | mean | p75 | max | designable | rate |
|---|---|---|---|---|---|---|---|---|---|
| 50  | 30 | 1.24 | 1.81 | 2.47  | 4.17  | 4.17  | 18.52 | 10/30 | 33.3% |
| 100 | 30 | 1.33 | 2.35 | 4.12  | 5.29  | 8.00  | 12.29 | 4/30  | 13.3% |
| 200 | 30 | 4.53 | 9.62 | 12.10 | 11.52 | 13.51 | 16.76 | 0/30  |  0.0% |

sparse K40 (step 1259, wd=0.05, sparse_attention=True / K=32):

| L | N | min | p25 | median | mean | p75 | max | designable | rate |
|---|---|---|---|---|---|---|---|---|---|
| 50  | 30 | 1.05 | 1.78 | 4.17  | 5.67  | 9.31  | 13.88 | 9/30 | 30.0% |
| 100 | 30 | 1.21 | 3.56 | 5.42  | 6.04  | 7.74  | 12.02 | 1/30 |  3.3% |
| 200 | 30 | 3.34 | 9.60 | 11.81 | 11.08 | 13.02 | 14.91 | 0/30 |  0.0% |

Cross-run designability rate matrix (already in Finding 8 in `content_masterarbeit.md`):

| | L=50 | L=100 | L=200 |
|---|---|---|---|
| **baseline (2646)** | **63.3%** | **66.7%** | **10.0%** |
| v2 (2078) | 23.3% | 16.7% | 0% |
| wd0 (1638) | 33.3% | 13.3% | 0% |
| sparse K40 (1259) | 30.0% |  3.3% | 0% |

Aggregate CSV with min/p25/median/mean/p75/max columns per (run, L): `inference/n30_aggregate.csv` (gitignored — re-generated by the pipeline script's tail block).

**Observations not in `content_masterarbeit.md` (kept here for completeness):**
- baseline mean is dragged up substantially by a few outliers at every length (mean 2.89 vs median 1.65 at L=50; mean 5.87 vs median 4.57 at L=200). The rate metric is the right summary, not the mean.
- baseline p75 at L=100 (2.40) is lower than at L=50 (3.39) — i.e. baseline's *typical* sample quality is actually slightly *better* at L=100 than at L=50 in this batch. This is the opposite of the L=200 cliff and worth flagging: the cliff is at L=200, not at "all L > 50".
- All three ablations (v2, wd0, sparse) have p75 ≥ 7.7 at L=100. The "tails" of the bad runs go far further than the baseline's tails.
- L=200 minima for baseline (1.50) and wd0 (4.53) differ by 3 Å — meaningful for "what's the best this recipe can do at L=200" — but the corresponding rate gap (10% vs 0%) at N=30 is exactly 3 samples. Distinguishing 0/30 from 3/30 reliably requires a second seed.

**Possible narrative:** **Yes — feeds Finding 8** (`content_masterarbeit.md → ## Finding 8`). The N=30 numbers above are the load-bearing evidence in Finding 8's "Numbers" section.

**Methodological caveats:**
- **Single seed (seed=100), N=30 per length, per run.** Within-seed binomial CI is now ~±9% on the rate, which is enough to separate baseline's 63% from v2's 23%, but not yet enough to pin sparse's 30% vs wd0's 33% at L=50 as different. A second seed × N=30 (≈3 more hours of L4 wall-clock) would tighten that.
- **Best-of-each-run snapshot, not matched-step.** baseline 2646 vs v2 2078 vs wd0 1638 vs sparse 1259 — comparing each ckpt at its individually-best val. This is the right comparison if the question is "what does each recipe ultimately produce on this codebase given the training runtime that was actually invested", but it confounds training duration with recipe. wd0 needs a step ≥ 2200 ckpt to make a duration-matched comparison against baseline 2646; sparse needs a step ≥ 2000 ckpt for the same.
- **L4 GPU vs A100 numerics.** Generation and ESMFold both ran on a single L4 24GB. bf16-mixed numerics on L4 vs A100 are not bit-exact; but the difference is well below the per-sample scRMSD noise floor (~0.5 Å between equivalent-seed re-runs on the same machine), so no meaningful confound here.
- **scRMSD < 2 Å is a coarse summary.** It collapses an 8-sequence × 1-fold ensemble into a single bit per sample. Using *any* of "min over 8 sequences", "mean", or "median" changes the rates somewhat — this report uses min (matches `_res_scRMSD_ca_esmfold` in the CSV, which is the per-sample best-of-8 used in all prior CA-only designability work in this repo).
- **L=200 across three of four runs is a 0/30 floor**, so individual ckpts cannot be ordered there. The cliff is well-established but the cliff *position* (does L=150 also collapse? L=180?) is unmeasured.
- **No sparse-with-more-training run yet.** If sparse is resumed from `last.ckpt` for one more 6h slot (~step 1850-1900), the L=100 = 3% finding either holds (architectural) or rises substantially (under-training). Until then, sparse's headline is a mid-training result and labeling it "the architecture is broken at L=100" would be premature.

---

## E015 — Three-wd weight-norm comparison + feasibility check for param-group-fix experiment (2026-04-27)

**Status:** finished. Pre-registration check, not a training run.

**Why ran:** The author observed that the wd=0 chain (`ca_only_diffusion_wd0`) qualitatively held up better at long protein lengths than the wd=0.05 baseline, and conjectured a "best of both worlds" experiment: split AdamW parameter groups so AdaLN-Zero gates (and biases / LN γβ / embeddings) get wd=0 while attention/MLP weights get wd≥0.05. The hypothesised mechanism: wd=0 preserves gate magnitudes and therefore long-range conditioning strength, while wd>0 on the rest of the model still buys regularization. Before committing a 16h training slot to test this, weight norms across the existing wd ∈ {0, 0.05, 0.1} runs were measured to check whether the mechanism's premises were even satisfied at the current "safe" recipe.

**Configs:**
- Three best-val raw checkpoints (NON-EMA — EMA decay/every_n_steps not tuned on this project, see operator preference):
  - wd=0:    `store/ca_only_diffusion_wd0/1777225343/checkpoints/best_val_00000021_000000002142.ckpt` (epoch 21, step 2142)
  - wd=0.05: `store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt` (epoch 26, step 2646)
  - wd=0.1:  `store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/best_val_00000020_000000002078.ckpt` (epoch 20, step 2078)
- Analysis: per-tensor L2 norm via `torch.linalg.vector_norm` on every floating-point parameter in `state_dict`; AdaLN-Zero gate weights identified by regex `nn\.transformer_layers\.\d+\.(mhba|transition)\.scale_output\.to_adaln_zero_gamma\.0\.weight$`. Script: `/tmp/wd_compare.py`; raw output: `/tmp/wd_compare_results.json`.
- 14 transformer layers × 2 sub-blocks (mhba + transition) = 28 gate weight tensors per ckpt.

**Results — AdaLN-Zero gate weight L2 norms (geom-mean per depth band, mhba+transition):**

| layer band | wd=0 / wd=0.05 | wd=0.1 / wd=0.05 |
|---|---|---|
| early (0–4)   | 0.85 | 0.88 |
| mid (5–9)     | 0.68 | 0.63 |
| upper (10–13) | 0.75 | **0.54** |

E009's per-layer collapse story is replicated: wd=0.1 upper-layer gates at 54% of canonical baseline; the most-collapsed individual gate is `transformer_layers.10.mhba` at 26% (matches E009's 0.260 ratio exactly — same comparison ckpts).

**Surprise finding:** The naive prediction was wd=0 → no decay pulling on gates → gates *grow larger* than wd=0.05. The data shows the opposite — **wd=0 gates are smaller than wd=0.05 gates across every depth band**. Most plausible mechanism: training-step confound. The wd=0 best-val snapshot is at step 2142 because wd=0 overfits sooner (no regularization → val curve turns up earlier), while the wd=0.05 best-val snapshot is at step 2646 — gates at wd=0.05 had ~500 extra optimizer steps to grow. With `save_top_k=1` overwriting earlier ckpts, no same-step ckpt pair is available to disentangle the step confound from a possible direct wd-on-gates effect.

**Direct wd=0 vs wd=0.1 comparison (`wd=0 / wd=0.1` per-gate ratio):**

| layer band | wd=0 / wd=0.1 |
|---|---|
| early (0–4)   | 1.00 — no recovery |
| mid (5–9)     | 1.24 — partial recovery |
| upper (10–13) | **1.38** — biggest recovery |

Removing wd from wd=0.1 → wd=0 recovers gate magnitudes specifically in the depth band where wd=0.1's collapse was most severe (upper layers). This is the depth-dependent gradient-signal mechanism made visible: in deep layers gradient signal is weaker → uniform wd dominates more → removing wd has the biggest effect there. Per-gate, **18 of 28** got closer to the wd=0.05 baseline when wd was removed; the recovery is a noisy two-thirds majority, not categorical (some gates, e.g. layer 11 mhba, went *further* from baseline at wd=0 than at wd=0.1).

**Heterogeneity of the wd=0.1 effect (the part E009 obscured by aggregating into a band):**

| stat | wd=0.1 / wd=0.05 | wd=0 / wd=0.05 |
|---|---|---|
| min   | **0.259** (layer 10 mhba, 74% collapse) | 0.318 (layer 10 mhba) |
| max   | **1.376** (layer 3 mhba, 38% GROWTH) | 1.314 (layer 11 mhba) |
| range | 1.117 | 0.996 |
| mean  | 0.692 | 0.763 |
| stdev | 0.252 | 0.211 |

Two gates *grew* under wd=0.1 — both at layer 3 (`mhba` at 1.376×, `transition` at 1.033×). Same depth, both subblocks. Two non-mutually-exclusive interpretations: (a) compensation — when adjacent gates collapse, layer 3 absorbs the slack to maintain conditioning throughput; (b) single-seed noise. The single-seed framing of all three runs makes (a) impossible to distinguish from (b) without replicate runs. wd=0.1 is also *more* chaotic than wd=0 (stdev 0.25 vs 0.21) — gate suppression is not just shifted-toward-zero but more variable layer-to-layer.

**Non-gate weights at wd=0.1 — heavy-tailed shrinkage:**

| metric | wd=0.05 vs wd=0 | wd=0.1 vs wd=0 |
|---|---|---|
| global L2 (sum-of-squares, sqrt) | 0.979 (2.1% smaller) | 0.960 (4.0% smaller) |
| per-tensor median ratio | 0.980 (2.0% smaller) | 0.953 (4.7% smaller) |
| per-tensor mean ratio   | 1.029 | 0.923 (7.7% smaller) |

The big median-vs-mean gap at wd=0.1 (0.95 vs 0.92) reveals heavy-tailed shrinkage. **An earlier draft of this entry conflated "per-tensor mean ratio" with "global L2 ratio" and quoted a misleading 10% figure for non-gate shrinkage** — the actual global non-gate L2 shrinkage from wd=0.05 to wd=0.1 is ~1.9% (438.65 → 430.27), not 10%. The 10% was the per-tensor-mean, inflated by small tensors that shrunk dramatically but contribute negligibly to global L2.

**Where the shrinkage actually concentrates (decomposition of total L2² drop wd=0.05 → wd=0.1, total = 7299.6):**

| component | L2² drop | share |
|---|---|---|
| AdaLN-Zero gate weights | 20.55 | **0.3%** |
| non-gate weights | 7279.05 | **99.7%** |

Gate weights have a global L2 of only 8.04 at wd=0.05 (vs non-gate global 438.65) — they are **1.83% of the model's weight magnitude**. So in pure weight-magnitude terms, the wd=0.1 regularization was already ~entirely happening on non-gate weights even with uniform wd. The gates dominate the sample-quality damage (Finding 5) despite being a tiny fraction of the L2 budget — they are functionally load-bearing far out of proportion to their numerical mass.

**Top-15 non-gate tensors by per-tensor shrinkage at wd=0.1 — all are biases of LayerNorm or AdaLN normalization layers** (each 256–768 params, shrunk to 13–50% of wd=0.05 baseline). Tiny in absolute magnitude, dominate the per-tensor mean. Examples:

| tensor | n_params | wd=0.05 norm | wd=0.1 norm | ratio |
|---|---|---|---|---|
| `transformer_layers.5.transition.adaln.norm_cond.bias` | 256 | 0.292 | 0.038 | 0.130 |
| `transformer_layers.6.transition.adaln.norm_cond.bias` | 256 | 0.240 | 0.040 | 0.167 |
| `transformer_layers.9.mhba.mha.pair_norm.bias` | 256 | 0.155 | 0.036 | 0.234 |
| `transformer_layers.5.mhba.mha.q_layer_norm.bias` | 768 | 0.202 | 0.060 | 0.299 |
| `transformer_layers.8.mhba.adaln.norm_cond.bias` | 256 | 0.118 | 0.036 | 0.307 |

**Top-15 non-gate tensors by absolute L2² contribution to global drop — all are large dense weights** (4.7M-param `transition.swish_linear.0.weight` and 1.8M-param `mhba.mha.to_qkv.weight`), each shrunk by only 5–7%:

| tensor | n_params | wd=0.05 norm | wd=0.1 norm | L2² drop |
|---|---|---|---|---|
| `transformer_layers.6.transition.transition.swish_linear.0.weight` | 4,718,592 | 47.234 | 44.074 | 288.5 |
| `transformer_layers.5.transition.transition.swish_linear.0.weight` | 4,718,592 | 47.059 | 44.186 | 262.2 |
| `transformer_layers.4.transition.transition.swish_linear.0.weight` | 4,718,592 | 47.415 | 44.948 | 227.8 |
| `transformer_layers.4.mhba.mha.to_qkv.weight` | 1,769,472 | 29.026 | 27.220 | 101.6 |
| `transformer_layers.7.mhba.mha.to_qkv.weight` | 1,769,472 | 28.867 | 27.099 | 98.9 |

Two-population picture: bias/LN parameters take massive *relative* hits (≤30% of baseline) but contribute negligibly to the actual regularization mass; the few large dense matrices take small *relative* hits (5-7%) but dominate the L2 budget. **Uniform wd is poorly targeted** — biases/LN params don't carry overfitting capacity (biases are constant offsets, LN scale/shift are bounded magnitude transformations) but soak up most of wd's per-tensor effect. The standard DiT/SiT/SD3 recipe excludes biases, LayerNorm γ/β, embeddings, and AdaLN-Zero gates from wd for exactly this reason; this codebase applies wd uniformly to all parameters.

**Implication for the param-group-fix scope:**
A "minimal" param-group fix (exclude only AdaLN-Zero gates) protects the functional gate-collapse failure mode but leaves the bias/LN over-regularization in place. The "full" param-group fix (exclude biases + LN γβ + embeddings + AdaLN-Zero gates — the standard DiT/SiT pattern) corrects both. The full fix is the same code complexity (~15 lines in `configure_optimizers`); no reason to do a partial version.

**Bias trajectory across wd values (asymmetric β-vs-γ collapse):**

| group | wd=0 / wd=0.05 (median) | wd=0.1 / wd=0.05 (median) |
|---|---|---|
| Q/K/node/pair LayerNorm biases (attention path) | 0.74 | **0.62** |
| AdaLN `norm_cond.bias` (conditioning path) | 0.77 | **0.51** |
| LayerNorm γ (scale parameters) | 1.03 | 0.99 — **essentially unchanged** |
| other biases | 1.02 | 0.99 |
| other weights | 1.02 | 0.97 |

Clean asymmetry: LayerNorm γ (initialized at 1, kept near 1 by the loss) is unaffected by any wd setting; LayerNorm β + AdaLN `norm_cond.bias` (initialized at 0, grows under loss pressure, opposed by wd) shrink in the same dynamic as the AdaLN-Zero gates. The "gate-collapse" mechanism generalizes: it's a **β-collapse mechanism** affecting any parameter family that needs to grow from zero against wd. AdaLN-Zero gates are the most functionally consequential instance, but Q/K LayerNorm biases (length-calibration of attention softmax) and AdaLN normalization biases (conditioning baseline) collapse by the same mechanism at a similar wd threshold.

**Hypothesis raised after this analysis (untested):** Q/K LayerNorm bias collapse may contribute specifically to long-protein performance degradation. The mechanistic link is the well-known softmax-diffusion-with-length effect: as N grows, more keys compete in the softmax and attention diffuses unless calibration (γ + β) compensates. wd=0.1 cuts those calibration biases to ~62% of baseline. The hypothesis is *not* directly supported by the wd=0 vs wd=0.05 comparison (wd=0 has *smaller* Q/K biases than wd=0.05 in this snapshot, due to the step confound, but wd=0 is qualitatively reported to hold up better at long lengths). It is mechanistically distinct from the gate-collapse failure and should be testable by isolating the bias contribution.

**Refined experiment recommendation:** A single discriminating run — *bias-only* param-group fix at wd=0.1 (exclude biases + LN γβ from wd, keep AdaLN-Zero gates *in* the wd budget) — would separate the bias-collapse contribution from the gate-collapse contribution. Outcomes:
- If long-protein designability recovers but short-protein still collapses (gates still suffer) → biases were a length-dependent contributor independent of gate collapse.
- If both recover → biases + gates both load-bearing in different regimes.
- If neither recovers → biases weren't doing real work; the gate-collapse story is sufficient.

This is more discriminating than the full DiT-style param-group fix (which is what you'd ship) because it isolates which excluded class actually matters. Same training cost (~16h slot).

**Results — non-gate weight norms:**

| | wd=0 | wd=0.05 | wd=0.1 |
|---|---|---|---|
| global L2 (sum of squares, sqrt) | 448.15 | 438.65 | 430.27 |
| per-tensor median ratio vs wd=0.05 | 1.021 | 1.000 | 0.973 |
| per-tensor mean ratio vs wd=0.05   | 0.972 | 1.000 | 0.897 |

Non-gate weights at wd=0.05 are only ~2% smaller than at wd=0 (per-tensor median). **Weight decay at the canonical recipe is barely doing any regularization on non-gate weights.** wd=0.1 produces ~10% global shrinkage, ~3% per-tensor median.

**Decision criteria for the param-group-fix experiment (pre-registered before the analysis):**

| condition | required | observed | satisfied? |
|---|---|---|---|
| C1 — gate-magnitude headroom at wd=0 vs wd=0.05 (≥20% larger in upper layers) | required | wd=0 upper gates are *smaller* (0.75×) | **NO** |
| C2 — non-gate wd biting at 0.05 (≥5% per-tensor shrinkage vs wd=0) | required | only ~2% shrinkage | **NO** |
| C3 — long-protein effect localises to upper-layer gate magnitude | required | premise (gate magnitude) not observed; mechanism cannot be the proposed one | **untested but premise broken** |

**Implications for the planned param-group-fix experiment:**

The "best of both worlds" framing does **not** survive contact with the data. There is no wd=0-specific gate-magnitude advantage to preserve, and almost no wd=0.05-specific regularization to preserve either. The mechanistically-defensible reframing — informed by the wd=0.1 non-gate shrinkage data — is: *"With param-groups in place, applying wd=0.1 to non-gate weights only would produce ~10% non-gate weight shrinkage (real regularization, far more than wd=0.05's ~2%) without the gate-collapse tax. Does that level of non-gate-only regularization help sample quality?"* That is a legitimate causal-ablation question (turn param-group fix ON vs OFF at non-gate wd=0.1, otherwise identical), but it is a narrow exploration of a previously-unsafe wd region rather than a recovery of a hypothesised wd=0 advantage.

The author's observation that wd=0 holds up better at long protein lengths is **not** explained by gate magnitudes (which are smaller at wd=0). The mechanism is therefore something else — possibly the small non-gate weight-norm increase, possibly inference-dynamics differences not summarized by norm, possibly an artefact of the step-confounded comparison. **Recommended next step before running the param-group-fix training:** measure designability (N≥10) at L=200, 300+ on both wd=0 and wd=0.05 best-val ckpts to verify the long-protein observation is real once N is large enough to be meaningful. If it is, identify the actual mechanism before committing training time to a fix targeting the wrong one.

**Possible narrative:** Non-narrative — kept for tuning/decision-making. Specifically, this entry's purpose is to record *why a planned experiment was downgraded in priority* and to leave the disconfirmation visible so a future re-attempt doesn't repeat the same flawed framing.

**Methodological caveats:**
- Best-val ckpts compared are at different optimizer steps (2142 vs 2646 vs 2078); confounded with what each recipe's val curve permits as a "best" snapshot. A clean comparison would require either same-step ckpts (`save_top_k=1` overwrote them) or fresh runs with explicit interim checkpointing.
- L2 norms are a coarse summary. Two weight tensors can have identical L2 but very different singular-value spectra / effective rank / activation behaviour. The gate-norm story is a *necessary* condition for the mechanism, not a sufficient one — even if gate magnitudes lined up, the experiment might still fail on inference-dynamics grounds.
- The wd=0 best-val ckpt is from a chain still in progress (training job 28492667 was still running at the time of this analysis); a later best-val snapshot may shift the numbers.
- E009's per-layer ratios are reproduced byte-equivalent for wd=0.1 vs wd=0.05, which validates the analysis script against an existing post-mortem.
- All checkpoints are RAW best_val (not `-EMA`). EMA hyperparameters on this project are inherited defaults, untuned — using EMA ckpts for cross-run comparison would mix EMA-schedule artefacts into the result.

---

## E016 — CA-only eval pipeline audit: reconstructed BB vs CA-only MPNN (2026-04-28)

**Status:** in progress (geometry diagnostic finished on login node; SLURM probe submitted as job 28551152, pending in queue at time of writing).

**Why ran:** The CA-only designability eval (`proteinfoundation/metrics/designability.py`) calls `run_proteinmpnn(..., ca_only=False)` at lines 375 and 560 — i.e., it uses the **vanilla full-backbone ProteinMPNN model** on a PDB whose N/C/O atoms were **reconstructed** from the generated Cα trace by `ca_to_backbone_atom37` (`proteinfoundation/utils/coors_utils.py:140`). The model itself only generates Cα (`_ca_only_mode = "local_latents" not in cfg_exp.product_flowmatcher`); N/C/O are synthesised post-hoc by extrapolating along the trace. Question: is this fake-backbone-vanilla-MPNN recipe (call it path A) producing materially different designability numbers from the canonical path B = bare-CA PDB + CA-only MPNN weights? If yes, all CA-only designability numbers in E008–E012 (baseline, v2, sparse) are computed under a suboptimal eval and may need re-running.

The current eval recipe is a workaround from commit `b1afbc4` ("eval for ca only fixed", 2026-04-07) — pre-commit, the call defaulted to `ca_only=True` and PDBs were CA-only. The CA-only ProteinMPNN weights (`ProteinMPNN/ca_model_weights/v_48_*.pt`) have file mtimes of **2026-04-17**, ten days *after* the commit, so the most plausible reason for the workaround was that those weights weren't on disk when the commit happened, not a deliberate choice between recipes.

**Configs:**
- Geometry diagnostic: `script_utils/probe_ca_eval/diagnose_geometry.py`. Loads two real native PDBs (5L33 109aa, 6MRR 71aa from `ProteinMPNN/inputs/PDB_monomers/pdbs/`); for each, computes |N-CA|, |CA-C|, and N-CA-C angle on the **native** backbone (filtered to residues with all three BB atoms) and on the **reconstructed** backbone produced by `ca_to_backbone_atom37` from CA-only input. No GPU.
- SLURM probe: `script_utils/probe_ca_eval/run_ca_eval_probe.py` driven by `submit_probe.sh` (-A COMPUTERLAB-SL3-GPU, ampere, 1×A100, --exclude=gpu-q-43, 1h walltime). Three conditions per native:
  - **A** RECON-vanillaMPNN: native → strip to CA → `ca_to_backbone_atom37` → save → ProteinMPNN `ca_only=False` → ESMFold (8 seqs) → CA-RMSD vs input PDB. Mirrors current eval.
  - **B** BARECA-caMPNN: native → strip to CA only (N/C/O zeroed) → save → ProteinMPNN `ca_only=True` → ESMFold (8 seqs) → CA-RMSD. Canonical CA-only design path.
  - **C** NATIVE-vanillaMPNN: native unchanged → ProteinMPNN `ca_only=False` → ESMFold (8 seqs) → CA-RMSD. Sanity check that the rest of the pipeline (ProteinMPNN/ESMFold/RMSD) is healthy on real backbones.
- Job ID: 28551152. Output: `slurm_ca_eval_probe_28551152.out`; summary JSON at `script_utils/probe_ca_eval/outputs/summary.json`.

**Results so far:**

Geometry diagnostic (no GPU). Bond lengths are pinned by construction (`ca_to_backbone_atom37` uses ideal 1.459 Å / 1.525 Å), so the live signal is the N-CA-C angle:

| Source | Protein | L | \|N-CA\| Å | \|CA-C\| Å | **N-CA-C deg (mean ± std)** |
|---|---|---|---|---|---|
| Native PDB | 5L33 | 106 (BB-complete) | 1.457 ± 0.006 | 1.522 ± 0.007 | **111.03 ± 1.88** |
| Reconstructed | 5L33 | 106 | 1.459 ± 0.000 | 1.525 ± 0.000 | **109.90 ± 19.38** |
| Native PDB | 6MRR | 68 (BB-complete) | 1.456 ± 0.007 | 1.521 ± 0.005 | **110.45 ± 1.98** |
| Reconstructed | 6MRR | 68 | 1.459 ± 0.000 | 1.525 ± 0.000 | **106.03 ± 13.92** |

Mean of the reconstructed angle is roughly correct (110° vs native 111°) but per-residue variance is **~10× the native variance**. Algebraic reason: the reconstruction places N along Caᵢ₋₁→Caᵢ and C along Caᵢ→Caᵢ₊₁, so the reconstructed N-CA-C angle equals the **virtual bond angle** Caᵢ₋₁-Caᵢ-Caᵢ₊₁, not the internal residue geometry. Real proteins have virtual bond angles ~88° in helix and ~120° in sheet, with sharp deviations at turns — that range is what the reconstructed N-CA-C inherits. Vanilla ProteinMPNN was trained on the tight ±2° native distribution, so a fraction of residues (turns, loops, breakpoints) sit OOD.

SLURM probe results: not yet available; will be appended to this entry once the job runs.

**Possible narrative:** Non-narrative — diagnostic, decides whether to re-run all CA-only designability numbers (E008–E012). If A vs B gap is ≥ 0.5 Å on real natives, re-evaluation of all CA-only checkpoints under path B becomes mandatory before any thesis claim about variant ordering is final. If gap is < 0.2 Å, current numbers stand and we move on.

**Methodological caveats:**
- N=2 native proteins, both short (71aa, 109aa). Probe cannot distinguish "is the eval recipe biased at all" (which it should answer cleanly) from "is the bias length-dependent" (which would require a 200-aa and 400-aa native added). Length-dependence question deferred until short-protein result lands.
- `ca_to_backbone_atom37` boundary residues use a duplicated direction (`forward[0]` for residue 0; `forward[-1]` for residue N-1). Effect dominates at small L; should be largely invisible at L > 100. Not a confound for the test as set up.
- "Designable" decision threshold (CA-RMSD < 2 Å) is the field-standard cutoff; numbers reported here are min/mean over 8 ProteinMPNN sequences as the literature standard. No multi-seed.
- ESMFold call cost dominates the wall-clock; no re-runs planned in the same job.

---
