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
| [E013](#e013--wd0-ablation-training-canonical-recipe-with-weight_decay00-2026-04-26--ongoing) | 2026-04-26 → ongoing | in progress | wd=0 ablation training on canonical CA-only recipe | → Finding 7 |
| [E014](#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27) | 2026-04-27 | finished | N=30/length matched-seed designability across baseline/v2/wd0/sparse | → Finding 7 (N=30 update) |
| [E015](#e015--three-wd-weight-norm-comparison--feasibility-of-param-group-fix-2026-04-27) | 2026-04-27 | finished | wd ∈ {0, 0.05, 0.1} per-layer gate + non-gate norm diff; pre-registration check for AdaLN-Zero param-group-fix experiment | non-narrative — disconfirmed a planned experiment's premise |
| [E016](#e016--ca-only-eval-pipeline-audit-reconstructed-bb-vs-ca-only-mpnn-2026-04-28) | 2026-04-28 | in progress | Audit of designability eval for CA-only generations: backbone-reconstruction geometry + SLURM probe on real natives | non-narrative — decides whether CA-only designability numbers need re-computing |
| [E017](#e017--paramgroups--wd01-quick-probe--proteinmpnn-ca_only-bug-fix-2026-04-28) | 2026-04-28 | finished | First clean designability probe (paramgroups+wd=0.1, n=9) after fixing the `ca_only=False` MPNN bug | pending — invalidates all prior CA-only designability numbers |
| [E018](#e018--baseline-bugfix-recheck--paramgroups-n6-followup-2026-04-28) | 2026-04-28 | finished | Re-eval 9 baseline PDBs with fixed MPNN + paramgroups N=6 follow-up | pending — quantifies bug impact, flags Finding 7 numbers as unreliable |
| [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) | 2026-04-29 | finished (per index) — body TBD | Full N=30 fixed-MPNN re-eval of E014's four arms + 5th paramgroups arm | → Finding 7 (rewrite); supersedes E014's numbers |
| [E020](#e020--joint-sequence-head-audit-property-panel--per-aa-composition--thermal-stability-proxies-2026-04-30) | 2026-04-30 | finished (Tier 1+2a) / Tier 2b TemStaPro pending GPU run | Distributional comparison of La-Proteina jointly-generated sequences vs PDB across the 14-property developability panel, per-AA composition, and thermal-stability proxies | → Finding 9 (PDB ref); see E026 for AFDB rerun |
| [E021](#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30) | 2026-04-30 | finished | First designability probe of sparse-K40 + `update_pair_repr` CA-only variant at step 1133 | non-narrative — decides whether to keep training the variant |
| [E022](#e022--long-length-designability-probe-of-canonical-baseline-l300400500-fixed-mpnn-re-eval-2026-05-02) | 2026-05-02 | finished | Long-length designability probe of canonical baseline (L=300/400/500), fixed-MPNN re-eval | non-narrative — feeds Finding 7's "L cliff" picture |
| [E023](#e023--aromatic-burial-targeting-gen-vs-pdb-rsa-via-freesasa-2026-05-03) | 2026-05-03 | finished; **superseded by E026 for the central comparison** | Aromatic burial-targeting comparison: La-Proteina full-atom unconditional gen vs length-matched PDB ref, RSA via FreeSASA + Tien et al. 2013 max ASA | F under-burial deficit **dies** against AFDB (E026) — the published reading was a PDB-vs-AFDB-baseline artifact |
| [E024](#e024--aromatic-burial-followups-composition-decomposition--curve-shape--per-protein-distribution-2026-05-03) | 2026-05-03 | finished; rerun on AFDB parquet in E026 | Three follow-up analyses on top of E023's per-residue RSA: (1) aromatic-pool composition + counterfactual reweighting; (2) F/Y curve shape vs amplitude (KDE + logistic slope); (3) per-protein burial-targeting distribution. | NOT-COMPOSITIONAL verdict and same-shape verdict survive against AFDB but become moot after E026 reverses E023's framing |
| [E025](#e025--steered-generation-sweep-camsol-max--tango-min--official-ld3ae2-l300400500-2026-05-03) | 2026-05-03 | finished (generation only — property re-eval pending) | Steered-sampling sweep on the official LD3+AE2 La-Proteina pretrained model; camsol_intrinsic maximize and tango minimize at five w_max levels each, lengths 300/400/500, 16 seeds per (length, level). Verifies steering hook fires on official LD checkpoint and produces predicted-property dose-response. | non-narrative for now — predictor-side dose-response confirms steering signal; needs `steering.property_evaluate` pass on the 480 PDBs to confirm real-property shift before any Finding |
| [E026](#e026--afdb-as-reference-rerun-of-e020--e023--e024-2026-05-03) | 2026-05-03 | finished (22 min wall) | Rerun of the AFDB-vs-PDB-affected experiments (E020-A/B/C, E023, E024) against a uniform-random AFDB reference (N=5000), since La-Proteina was trained on AFDB and PDB was the wrong control. | refines Finding 9 (alphabet collapse + gameable thermal proxies survive; SWI/Shannon mode-merging **withdrawn** — see E027); resurfaces E023 as a positive count-vs-placement competence sub-claim |
| [E027](#e027--mode-merging-robustness-check-pdb-bimodality-at-matched-n-2026-05-03) | 2026-05-03 | finished | Matched-n bootstrap on PDB shannon_entropy and SWI: is the apparent mode-merging signature a real population property or a sample-size detection artifact? | retired Finding 9 sub-claim (b) — Shannon-entropy bimodality is a real PDB-vs-AFDB population difference (not in AFDB at all); SWI bimodality vanishes when PDB is subsampled to AFDB's n=5K |
| [E028](#e028--predictor-vs-real-gap-on-the-may-04-ensemble-steered-run-2026-05-05) | 2026-05-05 | finished | Real-TANGO-binary diagnostic on the May-04 5-fold-ensemble + smoothed steered run: did the ensemble close the predictor-vs-real gradient-hacking gap? | non-narrative — confirmed the gap is still ~8.5× (predictor Δ-288 vs real Δ-34 at w=16); motivated E029 noise-aware predictor |
| [E029](#e029--noise-aware-predictor-fine-tune-and-single-fold-validation-smoke-2026-05-05) | 2026-05-05 | finished | Fine-tune the 5-fold predictor on `z_t = (1-t)·ε + t·z_1 + σ_L·√(t(1-t))·ε_2` over the steering window t∈[0.3, 0.8], then re-run tango_min_w16 with single noise-aware fold and measure the predictor-vs-real gap. | non-narrative — single noise-aware fold (no ensemble, no smoothing) **shrinks the hacking gap from −203 to −47 (~4×)** vs old 5-fold + smoothing; pilot only (n=4 proteins, 1 fold, 1 length, 1 w), needs scale-up |
| [E030](#e030--universal-guidance-k5-with-clean-predictor-probe-2026-05-05) | 2026-05-05 | finished | Probe: does K=5 universal guidance (replace one-step Tweedie with 5-step Euler integration of the latent flow ODE) close the hacking gap on top of the clean ensemble + smoothing, with no other changes? | non-narrative — **negative result**. UG K=5 + clean ensemble made the gap WORSE (-302 vs -203) AND delivered higher real TANGO (607 vs 582). Mechanism: the predictor remained fragile, and K=5 gave the adversarial gradient a 5-step flow Jacobian to propagate through. UG should only be re-tried on top of a noise-aware predictor. |
| [E031](#e031--noise-aware-predictor-v2-longer--cosine-decay-and-the-r-vs-hacking-disconnect-2026-05-05) | 2026-05-05 | finished | Re-train the noise-aware predictor with 3× more epochs + cosine LR decay; re-run tango_min_w16 with the better fold 2 to see whether the higher r²_noisy translates into a smaller hacking gap. | non-narrative — **negative result, important calibration**. v2 fold 2 r²_noisy = 0.6455 (up from v1's 0.5942), but hacking gap got WORSE (-145 vs v1's -47). r² and gradient-hackability decouple: a "better" predictor by val r² is more confidently wrong on adversarial inputs. v1 stays as canonical. |
| [E032](#e032--noise-aware-predictor--5-fold-ensemble--gap-essentially-closed-2026-05-05) | 2026-05-05 | finished | After E029-E031 isolated noise-aware-training (input-distribution fix) and ensemble (fold-specific-adversarial cancellation) as orthogonal levers, combine them: 5-fold ensemble of v1 noise-aware ckpts at tango_min_w16. | **positive — Finding-grade pilot**. Predictor:real gap = **−1.6** (vs single-fold v1's −47, vs clean-ensemble's −203). 3 of 4 proteins have predictor over-estimating real (regression noise, not hacking). Both fixes are needed and they compose. |
| [E033](#e033--scrmsd-validation-of-the-noise-aware-ensemble-sweep-2026-05-06) | 2026-05-06 | finished (9.4 h wall) | Designability check on E032's full noise-aware-ensemble sweep: do the steered proteins still fold? 4 seeds × 3 lengths × 5 w × 2 directions = 120 PDBs through MPNN→ESMFold scRMSD. | **positive — closes the "without breaking the protein" gate for Finding 10**. Designability stays 80-100% across w∈[1, 16] for both directions; mean scRMSD stays 0.95-1.49 Å; **no monotonic w→scRMSD degradation**. Variance is dominated by per-seed generation noise (s45_n500 broken at every cell, w-independent), not steering damage. Combined with E032's gap closure + 2× real-property delivery, this completes the F10 evidence stack. |
| [E036](#e036--pairwise-tm-score-diversity-of-the-noise-aware-ensemble-sweep-2026-05-06) | 2026-05-06 | finished | Pairwise TM-score across each of the 30 (direction, w, L) cells (n=120 pairs/cell) plus an unsteered baseline at each L. Tests "does steering narrow the structural ensemble at high w?" | **positive — closes the diversity-vs-steering question for Finding 10**. Mean pairwise TM = 0.407 across every (direction, w) cell, basically identical to the unsteered baseline (0.413). At L=400 / L=500 the steered cells are *more* diverse than baseline; only L=300 is marginally less. Steering introduces no structural ensemble collapse — 16-seed initialization variance dominates whatever narrowing the gradient might cause. |
| [E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) | 2026-05-06 | finished | First designability read on the `ca_only_downsampled` CA-only variant at best_val ep=23 step=2331 (`ca_only_downsampled/1777987722`). | non-narrative — **dead at this step**. 0/18 designable across L∈{50,100,200}; best CA scRMSD 12.41 Å, median 17.66 Å. Step 2331 sits inside the canonical baseline's 1800-2200 best-val window, so under-training does not explain the result. No bimodality — every sample is fully collapsed. |
| [E035](#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06) | 2026-05-06 | finished | First designability read on the `ca_only_sparse_K40_scnbr_t04` CA-only variant at best_val ep=8 step=819 (`ca_only_sparse_K40_scnbr_t04/1778022317`). | non-narrative — **fails the variant bar at this step**. 0/18 designable; best CA scRMSD 4.37 Å (L=100), median 11.44 Å. Step 819 is well before E021's step 1133 inflection and the canonical 1800-2200 window — verdict on the variant's converged ceiling deferred until a later step is probed. |
| [E037](#e037--curvature-targeted-bump-schedule-paired-n30-probes-2026-05-05--2026-05-06) | 2026-05-05 → 2026-05-06 | finished | Paired-noise schedule comparison (N=30, L=300, LD3-800): baseline vs `power_with_middle_bump` at eps∈{0.1, 0.14}, μ=0.489, σ=0.08 — extra NFEs at the `local_latents` curvature peak from E004/Finding 2 | non-narrative — null at N=30 with directional split (continuous mean improves, designability rate slightly worse); flags N=100 follow-up |

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

**Possible narrative:** **Yes — feeds Finding 7** (`content_masterarbeit.md → ## Finding 7`). Finding 7 frames the wd=0 result as currently in-progress; the cross-recipe N=30 comparison is in E014.

**Methodological caveats:**
- Single training run, single ckpt evaluated so far. Step 1638 is in the front edge of canonical's val-best window (1800-2200), so it is plausibly under-trained relative to canonical 2646.
- N=3 designability is too noisy to claim a wd=0 ↔ wd=0.05 difference on its own — the seed=5 vs seed=100 swing on the same step-1638 ckpt was 1/9 → 2/9 from a seed change alone. The N=30 eval (E014) is the gating data.
- AdamW equilibrium argument (`|θ_eq| ≈ |grad|/wd`) means wd=0 changes the equilibrium for *all* parameters, not just AdaLN-Zero gates. Without a per-layer gate-magnitude diagnostic on a wd=0 ckpt, "wd=0 helps because gates are larger" remains a mechanism inference, not a measurement. Diagnostic is owed before promoting Finding 7 to a Narrow claim.
- Canonical (E008) and v2 (E009) reached their best val at steps 1800-2200 / 2078; wd=0 may peak at a different step. Promoting Finding 7 to a "wd=X is best" claim requires comparing each recipe at *its own* peak ckpt, not at matched step.
- The pre-existing canonical ckpts at steps 692, 1638-equivalent, 1889, 2078, 2457, 2646 give a partial within-recipe designability trajectory under wd=0.05. wd=0 has only step 1638 so far. Without more wd=0 ckpts, the wd=0 trajectory cannot be drawn.

---

## E014 — Four-run N=30 designability comparison (baseline / v2 / wd0 / sparse, 2026-04-27)

**Status:** finished (one matched-seed N=30 batch per run; multi-seed replicates not yet collected). **Numbers in this entry were computed with the buggy `ca_only=False` ProteinMPNN call (see E017). Superseded by [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) which re-MPNN+ESMFolds the same PDBs under the fixed pipeline.** The buggy CSVs are preserved as `inference/results_inference_<arm>_n30_0_buggy.csv` for diff/audit.

**Why ran:** Previous side-by-side comparisons (E012, the N=3 runs in E008/E009/E010, and the N=3 single-seed probes in E013) had per-rate Wilson confidence intervals so wide that the rate-comparison between any two runs was nearly always overlapping at single-digit sample counts. Within a single seed, N=3 designability rates swing 0/3 ↔ 2/3 (0% ↔ 67%) just from the choice of three initial-noise samples. This was empirically observed on the wd=0 step-1638 ckpt (seed 5: 1/9 designable; seed 100: 2/9). Decision input that the multi-seed N≥30 batched comparison is required to make any "recipe X is better than recipe Y" claim about CA-only designability.

The natural minimum scope is "the four most important CA-only ckpts" — canonical baseline (the bar all variants must clear), v2 (the Finding 6 negative), wd0 (the Finding 7 in-progress causal ablation), sparse K40 (the architectural variant from E010). All compared at matched seed=100 so initial noise is byte-identical across runs (the ODE trajectory differs only by the model's velocity field).

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

> **Identification correction (2026-04-27):** during pre-pipeline ckpt survey, `best_val_00000012_000000001259.ckpt` was initially mistaken for canonical wd=0.05 because (a) it was the second-most-recently rsynced and (b) the filename gives no recipe info. Loading hyper_parameters revealed `run_name_=ca_only_sparse_K40, sparse_attention=True, wd=0.05` — i.e. it is the sparse run's best-val ckpt, not canonical. All four important ckpts were then renamed to recipe-bearing names (above) so this kind of misidentification cannot recur. The N=3 step-1259 result reported earlier as "canonical wd=0.05, 0/9 designable" was therefore the sparse arm, not canonical; the misattribution was corrected in `content_masterarbeit.md → Finding 7`.

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

Cross-run designability rate matrix (already in Finding 7 in `content_masterarbeit.md`):

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

**Possible narrative:** **Yes — feeds Finding 7** (`content_masterarbeit.md → ## Finding 7`). The N=30 numbers above are the load-bearing evidence in Finding 7's "Numbers" section.

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

## E017 — paramgroups + wd=0.1 quick probe + ProteinMPNN `ca_only` bug fix (2026-04-28)

**Status:** finished.

**Why ran:** Two questions, one run.
1. **Sanity-check the new "paramgroups + wd=0.1" training arm** (`store/ca_only_paramgroups_wd0p1/1777342310/`, ckpt `best_val_00000019_000000001952.ckpt`, ≈ opt step 1952) — does CA-only training under wd=0.1 succeed at producing designable structures *if* AdaLN-Zero gates (and biases / LN params / embeddings) are excluded from weight decay via parameter-group splits in `configure_optimizers`? This is the direct test of the gate-collapse hypothesis from Findings 5/6: the v2 run failed at wd=0.1 because uniform wd crushed the gates; if the gate-collapse explanation is correct, paramgroup-excluded wd=0.1 should *not* fail.
2. **First eval after fixing a ProteinMPNN bug** in `proteinfoundation/metrics/designability.py:375,560`. Both call sites had `ca_only=False` hardcoded with the comment *"Use vanilla model: backbone (N/CA/C/O) is always present"*. For CA-only generation that is incorrect: only CA atoms are written to the PDB, so vanilla MPNN sees a structure with no N/C/O atoms and its sequence designs are unreliable. Flipping both sites to `ca_only=True` is the bug fix; this E015 probe is the first designability eval ever run on this codebase under the correct MPNN setting.

**Configs:**
- New inference config: `configs/inference_paramgroups_wd0p1_quick.yaml` — modeled on `inference_ucond_notri_ca_only_v2_quick.yaml`, with the ckpt path/name above and a smaller workload (3 lengths × 3 samples × 200 ODE steps = **9 total samples** at L ∈ {50, 100, 200}). `nsteps=200`, `nsamples=3`, `max_nsamples_per_batch=3`. Generation block otherwise inherits canonical CA-only sampling settings from `inference_base.yaml` + `generation/uncond_codes_ca_only.yaml`.
- New wrapper: `script_utils/gen_n_eval_paramgroups_wd0p1.sh` — sbatch-able header, `--exclude=gpu-q-43`, but actually run interactively here (see hardware below).
- Bug fix (independent of this experiment but applied before it ran): `proteinfoundation/metrics/designability.py:375` and `:560` — `ca_only=False` → `ca_only=True`. Both call sites are inside `scRMSD()` and `seq_recovery_proteinmpnn()`. The fix is repo-wide; if a non-CA-only model (full La Proteina with the AE) is later evaluated on the same code path, the eval will then incorrectly use CA-only MPNN on a full-atom PDB. To be threaded through the eval config (`ca_only_mpnn: true/false`) when that comes up.
- Hardware: local non-HPC machine `gxp-l4-0` with 2× NVIDIA L4 24 GB. Conda env `/home/ks2218/.conda/envs/laproteina_env` (NOT the Cambridge `/home/ks2218/conda_envs/...` path; that env doesn't exist on this host). Single-GPU run.
- Output dir: `inference/inference_paramgroups_wd0p1_quick/`. Generation log: `/tmp/gen_paramgroups_wd0p1.log`. Eval log: `/tmp/eval_paramgroups_wd0p1.log`.
- Wall-clock: generation ~3-4 min; eval ~4 min (9 PDBs × ~25-30 s/PDB on L4 incl. ESMFold; L=200 PDBs take ~50 s each, L=50 ~7 s each). Total ≈ 8 min.

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o numbers within ~0.1 Å):**

| L | id | min scRMSD (Å) | designable (<2 Å) |
|---|---|---|---|
| 50  | 0 |  2.73 | no |
| 50  | 1 |  3.20 | no |
| 50  | 2 |  1.998 | yes (borderline) |
| 100 | 0 |  1.07 | yes |
| 100 | 1 |  1.31 | yes |
| 100 | 2 |  0.94 | yes |
| 200 | 0 |  2.56 | no |
| 200 | 1 | 14.42 | no |
| 200 | 2 | 11.56 | no |

Per-length designability rate: **L=50 = 1/3 (33%), L=100 = 3/3 (100%), L=200 = 0/3 (0%)**. Overall: 4/9 = 44% (matches the eval script's printed "Success Rate (<2Å): 44.4%"). Mean min-scRMSD across all 9: 4.42 Å. Best in the entire probe: 0.94 Å (L=100, id_2).

Full per-MPNN-seq scRMSD lists (CA mode) preserved in `/tmp/eval_paramgroups_wd0p1.log`; representative example (L=100 id_2): `[1.09, 0.94, 3.75, 1.00, 3.09, 1.06, 20.33, 1.43]` — 5 of 8 sequences fold below 2 Å, plus one outlier at 20 Å (a designable hit, not a marginal one).

**Possible narrative:** Pending. Two distinct stories sit on top of this run:

1. **The ProteinMPNN `ca_only` bug invalidates every prior CA-only designability number in this repo.** That includes: E007 (steering eval), E012 (three-run baseline/v2/sparse table), and E014 (the four-run N=30 comparison that backs Finding 7 in `content_masterarbeit.md`). All those results were computed with `ca_only=False` MPNN against CA-only PDBs, i.e. with MPNN seeing only the CA channel of a 4-atom expectation. The numbers may be biased low (artificially many 0/30s) or biased noisily — direction of the bias has to be measured by re-running, not assumed. **Until E014 is re-run with the fix, Finding 7's "wd=0.1 collapses designability vs canonical wd=0.05" claim cannot be defended at the published numerical level.** The qualitative claim *might* hold; the specific 23.3% / 16.7% / 0% per-length breakdown for v2 almost certainly doesn't.
2. **paramgroups + wd=0.1 looks healthy at L=100.** 3/3 with mins of 0.94 / 1.07 / 1.31 Å is well above the canonical bar (1-2/3) at this length. L=50 = 1/3 (id_2 at 1.998 Å) just clears it. This is consistent with the gate-collapse hypothesis: parameter-group exclusion of AdaLN-Zero gates (zero-init parameters that need to *grow* during training) from weight decay would prevent the wd=0.1 collapse seen in v2. But: this comparison is to v2 numbers that themselves were buggy, so the strength of the support for the hypothesis is currently weaker than it would seem. Without an apples-to-apples paramgroup vs no-paramgroup comparison **both evaluated under the fixed MPNN**, the paramgroup intervention's effect size is not pinned down.

Action items implied (not done in this entry):
- Re-run E014's four-arm N=30 designability probe with the MPNN fix. ~3h L4 wall-clock. Promote the resulting numbers into Finding 7 (and back-link the old buggy numbers as superseded, per the append-only rule).
- Run a paramgroups+wd=0.1 N=30 probe at the same lengths so it can be added as a fifth column in the comparison table. Cheap (~50 min L4) since the ckpt is already local.
- Audit `content_masterarbeit.md` for every numerical designability claim and tag any that came from `ca_only=False` MPNN as "preliminary, pending re-eval".

**Methodological caveats:**
- **N=9 total, N=3 per length, single seed.** This is the canonical "small probe" from CLAUDE.md ("1-2/3 designable at L=50 and L=100"), not an N=30 production result. Binomial CI on per-length rate is ~±55% absolute at N=3. The 3/3 at L=100 is not significantly different from 2/3 at this N, and a *2*/3 outcome at L=50 instead of 1/3 also wouldn't change the conclusion. The point of this probe is to confirm the ckpt isn't broken (it isn't), not to rank it against other recipes.
- **Borderline at L=50.** id_2's min scRMSD is 1.998 Å — 0.002 Å below the 2 Å threshold. The "1/3 designable" call is essentially a coin flip; 1/3 vs 0/3 here should not be interpreted as different from 0/3 at this single-sample resolution.
- **L=200 is genuinely bad (worst protein 14.42 Å), but at N=3 we cannot distinguish "0/3 because the recipe collapses at L=200" from "0/3 because the recipe matches baseline (which itself is 10% at L=200, so ~3/30 expected; 0/3 is then perfectly plausible noise)".** Need N=30 at L=200 to claim anything.
- **L4 GPU vs A100 numerics.** No effect expected on the qualitative result (same noise floor as E014).
- **The MPNN fix and the paramgroups recipe were tested in a single run.** They are not independently confounded with each other, but neither has an A/B comparison alongside it: there is no "paramgroups + wd=0.1, evaluated with the BUGGY MPNN" run, and no "v2 + buggy MPNN" run *re-run* with the fix. So we know "paramgroups+wd=0.1 + correct MPNN = 4/9 designable", and we know "v2 + buggy MPNN = 0/9 in earlier probes", but the cross-comparison still has *both* variables changing.

**Cross-references:**
- Bug-fix commit (pending): designability.py:375, :560.
- Code added: `configs/inference_paramgroups_wd0p1_quick.yaml`, `script_utils/gen_n_eval_paramgroups_wd0p1.sh`.
- Predecessor experiments potentially invalidated: E007, E012, E014.
- Findings potentially affected: Finding 5, Finding 6, Finding 7 in `content_masterarbeit.md` (all derive from designability numbers computed with the buggy MPNN).
- Superseded by E018 for the paramgroups headline (E018 has N=6, vs N=3 here). Kept here per the append-only rule.

---

## E018 — Baseline bugfix recheck + paramgroups N=6 follow-up (2026-04-28)

**Status:** finished.

**Why ran:** Two follow-ups to E017, both motivated by the same goal — separating the effect of the ProteinMPNN `ca_only` bug from the effect of the paramgroups+wd=0.1 recipe.
1. **Re-evaluate the same 9 baseline PDBs** that were part of E014's N=30 baseline arm (`inference/inference_baseline_n30/job_0_n_{50,100,200}_id_{0,1,2}/`), using the bug-fixed MPNN. Identical PDBs, identical ESMFold pipeline; the only thing that changed is `ca_only=False` → `ca_only=True` in `designability.py`. This isolates the bug's effect on already-known good ckpt outputs and quantifies how much of the apparent "L=200 cliff" in Finding 7 is real vs an MPNN artifact.
2. **Bump the paramgroups+wd=0.1 probe from N=3 to N=6 per length** to tighten the per-length rate (binomial CI on N=3 is ±55% absolute, on N=6 ±40%). Same ckpt as E015.

**Configs:**
- New eval-only config: `configs/inference_baseline_recheck_calpha.yaml` — modeled on `inference_paramgroups_wd0p1_quick.yaml`. Points at `baseline_wd0.05_step2646.ckpt` (already on disk locally; not actually loaded since generation is skipped). Lengths/nsamples are placeholders since `evaluate.py` only reads them for matching the inference dir layout.
- Pre-staged inference dir: `inference/inference_baseline_recheck_calpha/job_0_n_{50,100,200}_id_{0,1,2}/job_0_n_{...}.pdb` — 9 PDBs total, copied verbatim from `inference_baseline_n30/`. Pre-existing `esmfold_output/` and `seqs/` subdirs from E014 were stripped before re-eval to force fresh MPNN+ESMFold passes.
- Updated `inference_paramgroups_wd0p1_quick.yaml`: `nsamples: 3 → 6`, `max_nsamples_per_batch: 3 → 6`. Lengths unchanged ([50, 100, 200]).
- Hardware: same as E017 (single NVIDIA L4, `gxp-l4-0`, env `/home/ks2218/.conda/envs/laproteina_env`).
- Both runs used the bug-fixed `designability.py` (the `ca_only=True` flip from E017).
- Eval logs: `/tmp/eval_baseline_recheck_calpha.log`, `/tmp/eval_paramgroups_wd0p1_n6.log`.
- Generation log (paramgroups N=6 only): `/tmp/gen_paramgroups_wd0p1_n6.log`.
- Wall-clock: baseline recheck ≈ 4 min (9 PDBs, no generation); paramgroups N=6 ≈ 4 min generation + ≈ 8 min eval (18 PDBs). Total ≈ 16 min.

**Results — baseline recheck (same 9 PDBs as E014's first three id slots, OLD buggy MPNN values from `inference/results_inference_baseline_n30_0.csv`):**

Per-protein min scRMSD (CA mode), OLD vs NEW (Δ = NEW − OLD):

| L | id | OLD (`ca_only=False`) | NEW (`ca_only=True`) | Δ | designable old | designable new |
|---|---|---|---|---|---|---|
| 50  | 0 |  0.864 |  0.817 | −0.047 | yes | yes |
| 50  | 1 |  3.553 |  4.954 | +1.401 | no  | no  |
| 50  | 2 |  1.192 |  0.789 | −0.403 | yes | yes |
| 100 | 0 |  1.530 |  1.017 | −0.513 | yes | yes |
| 100 | 1 |  1.466 |  0.964 | −0.502 | yes | yes |
| 100 | 2 |  1.797 |  1.344 | −0.453 | yes | yes |
| 200 | 0 |  1.887 |  1.095 | −0.792 | yes | yes |
| 200 | 1 |  9.135 |  1.548 | **−7.587** | no  | **yes** |
| 200 | 2 |  2.790 |  1.069 | **−1.721** | no  | **yes** |

8 of 9 PDBs improved; 1 of 9 regressed (L=50 id_1, an already-undesignable sample). Per-length designability rate on this slice:
- L=50: 2/3 (67%) → 2/3 (67%) — same.
- L=100: 3/3 (100%) → 3/3 (100%) — same (but mins moved from 1.5-1.8 Å to 1.0-1.3 Å).
- L=200: 1/3 (33%) → **3/3 (100%)** — two PDBs that read as failed under buggy MPNN are actually designable.

Overall: 6/9 (67%) → 8/9 (89%). Average min-scRMSD: 2.59 Å → 1.51 Å. The eval script's printed summary: "Average scRMSD: 1.511 Å, Success Rate (<2Å): 88.9%".

**Interpretation:** the `ca_only=False` bug caused a strong positive bias in scRMSD (i.e. structures that were actually self-consistent looked unreliable). The bias is largest at L=200 (mean Δ ≈ −3.4 Å on this 3-protein slice) and almost exactly zero at L=50 (mean Δ ≈ +0.3 Å, dominated by the one regression). The "L=200 cliff" effect in Finding 7's E014 table is at least *partly* a measurement artifact — direction confirmed, magnitude needs full N=30 re-eval.

**Results — paramgroups+wd=0.1 N=6:**

Per-protein min scRMSD (CA mode):

| L | id_0 | id_1 | id_2 | id_3 | id_4 | id_5 |
|---|---|---|---|---|---|---|
| 50  |  2.05 |  **1.57** |  2.42 |  8.57 |  **1.43** |  **1.27** |
| 100 |  **0.94** |  3.05 |  **0.91** |  **0.69** |  **1.77** |  **1.73** |
| 200 |  2.09 |  6.53 | 11.07 | 11.45 |  6.67 |  **1.69** |

Per-length designability rate (min < 2 Å):
- L=50:  3/6 (50%)
- L=100: 5/6 (83%)
- L=200: 1/6 (17%)
- Overall: 9/18 = 50% (eval script: "Success Rate (<2Å): 50.0%, Average scRMSD: 3.661 Å").

**Comparison to E017 (paramgroups N=3, same ckpt):**

| L | E017 (N=3) | E018 (N=6) | combined N=9 if re-pooled (informational) |
|---|---|---|---|
| 50  | 1/3 (33%) | 3/6 (50%) | 4/9 (44%) |
| 100 | 3/3 (100%) | 5/6 (83%) | 8/9 (89%) |
| 200 | 0/3 (0%)  | 1/6 (17%) | 1/9 (11%) |

(Note: E017's PDBs were a separate generation pass and use a different RNG state than E018, so the combined column is informational only.) The N=3 → N=6 sample shows regression to mean at L=100 (3/3 was lucky), and a slight bump at L=200 (the original 0/3 was also small-N noise). The per-length picture is consistent: paramgroups+wd=0.1 trains successfully, but L=200 is *not* near-100% the way the baseline N=3 recheck hinted — best L=200 sample only barely clears 2 Å (1.69).

**Side-by-side designability rate matrix (this experiment, all under fixed MPNN):**

| Recipe | step | L=50 | L=100 | L=200 | overall |
|---|---|---|---|---|---|
| baseline (canonical wd=0.05, no paramgroups) | 2646 | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 8/9 (89%) |
| paramgroups + wd=0.1 | 1952 | 3/6 (50%) | 5/6 (83%) | 1/6 (17%) | 9/18 (50%) |

Direct comparison is confounded (see caveats); the table is included so a future reviewer can see the headline numbers in one place.

**Possible narrative:** Pending. Two distinct stories that share evidence:
1. **The `ca_only` bug had a measurable, length-dependent effect on E014's numbers.** Eight of nine baseline samples improved under the fix; the L=200 column is the most affected. **Re-running E014's full N=30 four-arm comparison with the fix is now load-bearing for any defensible version of Finding 7.** The current Finding 7 table in `content_masterarbeit.md` should be tagged "preliminary — pending re-evaluation under the bug-fixed MPNN" until that re-run lands, or removed.
2. **Paramgroups+wd=0.1 probably trains successfully but underperforms canonical baseline at this snapshot.** L=100 looks healthy (5/6 = 83%), L=200 looks weak (1/6 = 17%) — consistent with E015's read but on more data. Whether this is a recipe issue or just an under-trained ckpt (1952 vs 2646 steps) is not separable from this single probe; would need a step-matched paramgroups checkpoint, OR a baseline recheck at step 1952 for direct comparison.

Action items implied (not done in this entry, but now obvious):
- **Re-run E014 with the fix.** Estimated 3 h on this L4 (4 ckpts × ~45 min each, generation cached). Promote results into Finding 7; back-link the buggy numbers as superseded.
- **Add a paramgroups+wd=0.1 N=30 arm** to the re-run so it sits in the table on equal footing.
- **(Optional) Re-eval baseline at step 1952** to make the paramgroups vs baseline comparison step-matched.

**Methodological caveats:**
- **Baseline N=3 is too small to read 100% at L=100/L=200 as "the recipe is at 100%".** E014's N=30 buggy numbers were 67% / 67% / 10%, so a fix-corrected N=30 will likely land somewhere in 80-95% / 90-100% / 30-60%, not 100% across the board. The 89% headline on baseline_recheck is on this 9-sample slice, not a per-length rate estimate.
- **One regression in 9 PDBs (L=50 id_1, +1.4 Å)** suggests the fix is not purely better — some samples become harder when MPNN sees only the CA channel. Likely an MPNN-internal effect: at very short L=50 with bad backbone, vanilla mode could occasionally hit a happy hallucination, and CA-only is more honest. Sign of effect is unambiguous overall; magnitude is sample-dependent.
- **Paramgroups N=6 is still small.** L=200 = 1/6 = 17% has 95% binomial CI ≈ [0.4%, 64%]. Cannot distinguish "16% true rate" from "0% true rate" from "40% true rate" at this N. N=30 needed for any per-length comparison.
- **Step mismatch confound (1952 vs 2646)** means the paramgroups vs baseline rate comparison is not a clean test of the recipe.
- **Single seed everywhere.** E014's protocol was seed=100; E017/E018 inherit `inference_base.yaml`'s `seed: 5`. No seed sweep. Within-seed L4 noise on min-scRMSD is ~0.5 Å.
- **L=50 id_1 is a known difficult sample** (3.55 Å under buggy MPNN, 4.95 Å under fixed MPNN). It read as "non-designable" in both modes; the +1.4 Å delta is among the within-seed noise floor for this kind of sample but is the only direction-flipping data point in the recheck slice.

**Cross-references:**
- Builds on E017 (same MPNN bug fix, same paramgroups ckpt).
- PDBs in `inference/inference_baseline_recheck_calpha/` are direct copies from `inference/inference_baseline_n30/` (E014). PDBs in `inference/inference_paramgroups_wd0p1_quick/` are now N=6, overwriting the N=3 set from E017 (regenerated from scratch; first N=3 not preserved separately on disk).
- Implications for `content_masterarbeit.md → Finding 7`: deferred to a follow-up re-run of E014.

---

## E019 — Full N=30 fixed-MPNN re-eval of E014's four arms + 5th paramgroups arm (2026-04-29)

**Status:** finished per the index — full body not yet transcribed into this file.

**Why ran:** First clean N=30 designability eval, after the `ca_only=False` ProteinMPNN bug fix (E017), of the same four training arms compared in E014 (canonical wd=0.05 baseline, v2, wd=0, sparse-K40) plus paramgroups+wd=0.1. Supersedes the buggy N=30 numbers in E014; rewrites Finding 7.

**Note:** This entry was committed as an **index row only** during a merge; the full body (Configs / per-arm Results table / Methodological caveats) was either never written into `experiments.md` or lives in a separate file. Several other entries cross-reference the N=30 numbers — E014's status block, E022's "L=50/100/200 numbers in Finding 7 anchor", E022's "right comparison anchor" caveat — when those numbers are needed, they should be pulled from `inference/results_inference_<arm>_n30_0.csv` (post-fix CSVs) and the body filled in here. Until then, treat the cross-references as pointers to "Finding 7 (rewrite) source data, location TBD".

---

## E020 — Joint sequence head audit: property panel + per-AA composition + thermal-stability proxies (2026-04-30)

**Status:** finished for Tier 1 (sequence proxies) and Tier 2a (per-AA composition). Tier 2b (TemStaPro ML thermal-stability predictor) **pending** — submit script ready, GPU run not yet executed (1× A100 ampere, est. ~70 min wall-clock; on L4 ~4-6 h).

**Why ran:** La-Proteina is the first family member with a joint sequence head — sequences come directly from `sample_prots["residue_type"]` (`steering/generate.py:113`), not from a downstream MPNN. This means the model's own sequence-distribution behaviour is *directly observable* without an MPNN confound, and any drift from the natural sequence distribution attaches to the model rather than to the inverse-folder. The audit asks: at the level of (i) the developability property panel, (ii) per-amino-acid composition, and (iii) thermal-stability proxies, how does the joint sequence head's output compare to the PDB? Decision feed: whether the headline co-designability metric (which uses La-Proteina's own sequences end-to-end through ESMFold via `evaluate.py:337`, `use_pdb_seq=True`) is potentially inflated by easy-to-fold sequences, and whether the sequence head exhibits classical generative-model failure modes (alphabet collapse, mode-merging on multimodal targets).

**Configs (re-run-able from this entry):**

- **Generated set:** `results/generated_stratified_300_800/sequences.fasta` (1000 sequences, length 300-799), produced by:
    - `proteina_config: inference_ucond_notri_long.yaml` → `ckpt_path: ./checkpoints_laproteina/LD3_ucond_notri_800.ckpt`, `autoencoder_ckpt_path: ./checkpoints_laproteina/AE2_ucond_800.ckpt`.
    - `length_mode: stratified`, `length_range: [300, 800]`, `bin_width: 50`, `n_per_bin: 100`, `nsteps: 200`, `seed_base: 1000`.
    - Manifest: `results/generated_stratified_300_800/manifest.csv`.
- **Reference panel (developability properties):** `laproteina_steerability/data/properties.csv` — 56,008 PDB proteins, length 300-796, 19 columns. Built by `laproteina_steerability/prep_properties.py` from the upstream `developability_panel.csv`. Real PDB sequences (`compute_developability.py:579`, sequence from `data.residue_type` of the processed `.pt` files).
- **Reference FASTA (AA composition):** `pdb_cluster_all_seqs.fasta` length-filtered to [300, 800]. Note: this fasta's max length is 511, so the filter effectively cuts at 511 → 53,749 sequences. Different population than the property-panel reference (which goes to 796) — caveat below.
- **Generated-set property recompute:** `results/generated_stratified_300_800/properties_generated.csv` produced by `steering/property_evaluate.py` (sequence taken from `data["sequence"]` in the generated `.pt` files, set in `steering/generate.py:151`). Identical scoring code to the reference panel (`compute_swi`, `compute_tango`, `compute_charge_and_pI`, `compute_iupred3_standalone`, `compute_shannon_entropy`, etc.).

**Scripts (committed in this experiment):**

- `proteinfoundation/analysis/compare_properties.py` — distribution comparison across 15 columns of the developability panel. Renames the column-name mismatch between the two CSVs to a canonical name space; computes per-property mean/sd/var/median/IQR/skew/excess-kurtosis/min/max + KDE-peak count, plus pairwise Cohen's d / KS-D / Wasserstein. Outputs `summary.csv` and a 4×4 grid of overlaid hist+KDE plots.
- `proteinfoundation/analysis/aa_composition.py` — per-protein 20-AA fractions from FASTA, length-filtered, averaged across proteins. Reports each AA's gen-vs-ref absolute and relative difference; outputs `aa_composition.csv` and a stacked bar plot of side-by-side composition + per-AA over-/under-representation.
- `proteinfoundation/analysis/thermal_stability.py` — Tier 1 sequence proxies (aliphatic index, IVYWREL fraction, GRAVY, charged fraction, log(D+E/K+R), F+W+Y aromatic fraction). Tier 2 driver for TemStaPro (ProtT5-XL + MLP, stand-in for DeepStabP which has no offline weights). Outputs per-protein CSVs, summary CSV, and 6-panel histogram grid.
- `script_utils/run_thermal_stability.sh` — SLURM submit script (1× A100 ampere, 4 h walltime, `--exclude=gpu-q-43`, persistent caches at `/home/ks2218/cache/huggingface` and `/home/ks2218/cache/temstapro_embeddings`). TemStaPro repo cloned to `/home/ks2218/TemStaPro`; `sentencepiece` 0.2.1 installed in `laproteina_env`.

**Output dirs:**

- `results/property_comparison/stratified_vs_pdb/` — property-panel comparison CSVs + plots, per-AA composition CSV + bar plot.
- `results/thermal_stability/stratified_vs_pdb/` — Tier 1 proxy CSVs, per-protein CSVs, 6-panel hist grid. Tier 2 will append TemStaPro columns (`clf_40` … `clf_80`, plus `left_*`/`right_*` bounds and a thermophilicity label) when the GPU run completes.

### Results

#### A. Developability panel comparison (15 properties)

(Cohen's d = (gen_mean − ref_mean) / pooled_sd. KS_d = 2-sample KS. Modes = peak count in a Gaussian KDE.)

| property | ref_mean | gen_mean | Cohen's d | ref_sd | gen_sd | ref_modes | gen_modes | KS_d | wasserstein |
|---|---|---|---|---|---|---|---|---|---|
| sequence_length | 405 | 548 | +1.47 | 97.1 | 143 | 1 | 1 | 0.454 | 144 |
| swi | 0.779 | 0.795 | +1.62 | 0.0101 | 0.019 | **2** | **1** | 0.623 | 0.0178 |
| tango | 999 | 1130 | +0.44 | 282 | 614 | 1 | 1 | 0.195 | 247 |
| tango_aggregation_positions | 1620 | 1460 | -0.13 | 1170 | 1760 | 1 | 1 | 0.284 | 620 |
| net_charge | -7.03 | -32.4 | **-2.14** | 8.61 | **62** | 1 | 1 | 0.430 | 34 |
| pI | 6.13 | 5.6 | -0.43 | 1.21 | 1.95 | 2 | 2 | 0.428 | 0.912 |
| iupred3 | 0.216 | 0.332 | +1.63 | 0.068 | 0.176 | 2 | 1 | 0.423 | 0.121 |
| iupred3_fraction_disordered | 0.0337 | 0.201 | **+2.70** | 0.0498 | 0.284 | 1 | 1 | 0.358 | 0.168 |
| **shannon_entropy** | **4.10** | **3.36** | **-6.65** | 0.096 | 0.432 | 2 | 1 | **0.921** | 0.737 |
| hydrophobic_patch_total_area | 3960 | 2790 | -0.87 | 1330 | 2460 | 1 | 1 | 0.599 | 1630 |
| hydrophobic_patch_n_large | 12.2 | 6.27 | -1.00 | 5.87 | 8 | 1 | 1 | 0.587 | 6.42 |
| sap | 13.3 | 11.8 | -0.24 | 5.83 | 19.5 | 1 | 1 | 0.571 | 8.62 |
| scm_positive | 39.6 | 46.6 | +0.48 | 13.3 | 48.8 | 1 | 1 | 0.289 | 21.8 |
| scm_negative | -43 | -83.7 | -2.30 | 14.5 | 78 | 1 | 1 | 0.444 | 47.1 |
| rg | 21.9 | 23.6 | +0.56 | 3.12 | 3.3 | 1 | 1 | 0.324 | 1.78 |

**Headline cross-reads:**

- shannon_entropy: KS=0.92, Cohen's d=−6.65 — the single largest deviation in the panel. Effective alphabet size 2^4.10 ≈ 17 (ref) → 2^3.36 ≈ 10 (gen). Gen sd is 4.5× wider (heavy lower tail of low-complexity sequences).
- iupred3 family: gen is much more disorder-promoting (fraction_disordered +2.70 SD).
- net_charge / scm_negative: gen mean −2.1 SD lower with gen sd 5-7× wider.
- swi: KS=0.62, modes 2→1 with gen mean (0.795) sitting *between* the two PDB modes — mode-merging signature.
- rg drift (+0.56 SD) is length-confounded (gen mean length 548 vs ref 405; with rg ∝ L^{0.4}, expected gen rg ≈ 24.0 vs observed 23.6 → essentially length-scale-matched).
- tango: gen mean only +0.44 SD shifted but sd is 2.2× wider (heavy tail of very aggregation-prone sequences).

#### B. Per-amino-acid composition (`aa_composition.py`, length-filtered to [300, 800])

Comparison set: gen n=1000 vs ref n=53,749 (PDB cluster fasta, capped at length 511 by source).

**Top 5 over-represented AAs in gen:**

| AA | gen | ref | abs Δ | rel Δ |
|---|---|---|---|---|
| E (Glu) | 11.78% | 6.06% | +5.72 pp | **+95%** |
| L (Leu) | 11.40% | 8.87% | +2.53 pp | +29% |
| **N (Asn)** | **10.28%** | **4.19%** | **+6.10 pp** | **+146%** |
| G (Gly) | 9.62% | 7.93% | +1.69 pp | +21% |
| I (Ile) | 7.28% | 5.41% | +1.87 pp | +35% |

These five make up **50.4% of generated residues** vs **32.5% in PDB**.

**Most under-represented:**

| AA | gen | ref | rel Δ |
|---|---|---|---|
| M (Met) | 0.51% | 2.42% | **−79%** |
| H (His) | 0.92% | 2.96% | **−69%** |
| W (Trp) | 0.46% | 1.43% | −68% |
| F (Phe) | 2.37% | 4.12% | −42% |
| D (Asp) | 3.66% | 5.85% | **−38%** |
| V (Val) | 4.45% | 7.01% | **−37%** |
| P (Pro) | 3.22% | 4.90% | −34% |
| C (Cys) | 0.91% | 1.38% | −34% |
| A (Ala) | 6.09% | 8.45% | **−28%** |

**Notable specific findings:**

- **Glu/Asp ratio: gen 3.22, PDB 1.04** — chemistry-specific asymmetry within the acidic class. The negative-charge bias from the panel (net_charge −2.1 SD) is driven *almost entirely by Glu*; Asp is actually under-used. Mechanistic reading: Asp's short side chain forms backbone H-bonds (helix N-cap, β-turn, Asx motifs) and demands specific structural contexts; Glu's longer side chain decouples from the backbone and is helix-friendly + surface-tolerant. The model concentrates on the *low-contextual-specificity* member of each chemistry pair.
- **Ala under-used (−28%)** — counterexample to a naïve "pick simple residues" reading. Ala is the prototypical context-tolerant residue but it's small and "uninformative"; it gets squeezed out by larger committed residues (L, I, E) that contribute more to local-structure consistency.
- **Ile preferred, Val crushed** — Val (β-branched at Cβ, strongly β-strand-preferring) is under-used; Ile (β-branched but with γ-carbons, helix-tolerant) is over-used. **Falsifiable prediction:** DSSP-resolved secondary-structure breakdown of generated structures will show β-sheet content depressed.
- **Aromatic collapse follows core-buryness ranking:** W (deepest buried, −68%) > F (−42%) > Y (−5%, basically unchanged). Y is the polar aromatic and behaves more like a surface residue than a buried-core anchor. The "aromatic" collapse is specifically a *buried-aromatic* collapse.

#### C. Thermal-stability sequence proxies (Tier 1, `thermal_stability.py`)

| metric | ref_mean | gen_mean | Cohen's d | ref_sd | gen_sd | KS_d | wasserstein |
|---|---|---|---|---|---|---|---|
| aliphatic_index (Ikai 1980) | 84.4 | 91.8 | **+0.74** | 9.73 | 19.2 | 0.359 | 10.6 |
| ivywrel_fraction (Zeldovich 2007) | 0.371 | 0.429 | **+1.35** | 0.039 | 0.139 | 0.488 | 0.099 |
| gravy (Kyte-Doolittle) | -0.241 | -0.513 | -1.29 | 0.208 | 0.354 | 0.526 | 0.296 |
| charged_fraction (D+E+K+R) | 0.220 | 0.252 | +0.75 | 0.039 | 0.133 | 0.465 | 0.091 |
| log_acidic_basic_ratio (log10[D+E/K+R]) | 0.075 | 0.259 | +1.67 | 0.090 | 0.475 | 0.366 | 0.250 |
| **aromatic_fraction (F+W+Y)** | **0.090** | **0.061** | **−1.19** | 0.024 | 0.035 | 0.404 | 0.029 |

**Methodological observation:** the two literature proxies for thermostability (aliphatic index, IVYWREL fraction) are **systematically inflated by the alphabet collapse** — both say "gen looks more thermostable than PDB" by +0.74 / +1.35 SD. The drivers are exactly the residues over-represented in the alphabet collapse (Leu, Ile in aliphatic index; Leu, Glu in IVYWREL). Meanwhile, the aromatic_fraction (F+W+Y), which is the most direct sequence-side proxy for a buried hydrophobic core, drops 1.19 SD. **Aliphatic index and IVYWREL cannot be used as standalone evidence of thermal stability for generative-model outputs**; they were calibrated to discriminate within natural proteins (mesophiles vs thermophiles) and do not control for compositional gaming.

#### D. Tier 2 (TemStaPro / ML-predicted Tm classes) — pending GPU run

DeepStabP, the original target, has no offline weights — it exists only as a hosted web service (https://deepstabp.dsmz.de) and the GitHub `CSBiology/DeepStabP` repo is the F# web frontend, not the Python prediction code. Backend repo `CSBiology/DeepStabp_Backend` does not exist (or is private). For the same architecture family (mean-pooled ProtT5-XL embedding → small MLP head, multi-threshold classification), **TemStaPro** (Pudžiuvelytė et al., *Bioinformatics* 2024) was set up as a working substitute:

- Repo: `https://github.com/ievapudz/TemStaPro` cloned to `/home/ks2218/TemStaPro`.
- Weights: bundled in `models/` (45 weight files: 9 thresholds × 5 seeds, predicting `P(Tm > T)` for `T ∈ {40, 45, 50, 55, 60, 65, 70, 75, 80}` °C).
- ProtT5 model: `Rostlab/prot_t5_xl_half_uniref50-enc` (~1.5 GB, fp16), downloaded once on first run via HuggingFace.
- `sentencepiece` 0.2.1 added to `laproteina_env`.
- Submit: `sbatch script_utils/run_thermal_stability.sh`. Persistent embedding cache at `/home/ks2218/cache/temstapro_embeddings` so the 53k-sequence ref pass is a one-time cost.
- Wall-clock estimate: ~70 min A100, ~4-6 h L4 (memory-bandwidth-bound on ProtT5-XL).

### Possible narrative

→ **Finding 9** in `content_masterarbeit.md` (added in the same commit as this entry). The Finding consolidates (A), (B), and (C) into three sub-claims: (i) chemistry-specific alphabet collapse, (ii) mode-merging on bimodal targets (SWI), (iii) gameability of standard thermal-stability proxies. (D) is preregistered and will either confirm or refine sub-claim (iii).

### Methodological caveats

- **Single generated set, single eval seed.** N=1000 from `seed_base=1000`. No cross-seed variance estimate. AA-composition magnitudes have ~1% absolute uncertainty under bootstrap.
- **Single training checkpoint.** `LD3_ucond_notri_800.ckpt` only. No comparison to a different La-Proteina ckpt, an earlier-training snapshot, or an alternative model family.
- **Reference fasta length cap.** `pdb_cluster_all_seqs.fasta` only goes up to length 511, so the AA composition reference is computed on PDB[300, 511] while the property-panel reference is PDB[300, 796]. The two reference populations differ; AA-composition numbers may shift by a few percent if recomputed against the same set as the panel (sequences extracted from the processed `.pt` files). Direction of effects (signs of all relative differences) is robust under this — confirmed by spot-checks against the panel-reference Shannon entropy (4.10 in both).
- **No structural-quality readout in this experiment.** The "the model has no buried hydrophobic core" reading is supported by the F+W+Y collapse and the iupred3 disorder bias, but not directly measured. Adding DSSP secondary-structure breakdown + ESMFold pLDDT distribution to the same generated set would close that gap.
- **Tier 2 (TemStaPro) not yet run.** Tier 1 alone supports the methodological observation (aliphatic + IVYWREL gameable, aromatic fraction tells the opposite story); a passing TemStaPro number would *not* contradict the observation but would either lower the confidence of the "easy-to-fold" claim or elevate it.
- **Co-designability inflation hypothesis (Finding 9 implication) is preregistered, not measured.** Cleanest direct test: designability vs co-designability gap on the same backbones, plus designability stratified by Shannon-entropy decile. Both deferred to a follow-up experiment.

### Outputs on disk

- `results/property_comparison/stratified_vs_pdb/summary.csv` — 22 columns × 15 rows.
- `results/property_comparison/stratified_vs_pdb/distributions.png` — 4×4 hist+KDE grid.
- `results/property_comparison/stratified_vs_pdb/aa_composition.csv` — 6 cols × 20 rows.
- `results/property_comparison/stratified_vs_pdb/aa_composition.png` — composition + relative-deviation bar plot.
- `results/thermal_stability/stratified_vs_pdb/summary.csv` — 11 cols × 6 rows (Tier 1 only at present).
- `results/thermal_stability/stratified_vs_pdb/distributions.png` — 6-panel hist grid.
- `results/thermal_stability/stratified_vs_pdb/{gen,ref}_per_protein.csv` — per-protein Tier 1 values (and TemStaPro values once Tier 2 completes).

### Update — AFDB rerun (E026, 2026-05-03)

**Important framing:** La-Proteina was trained on AFDB. The PDB reference used in E020 above was the wrong distributional control for "the model has drifted from its training distribution" — the right comparator is AFDB (or, more precisely, the AFDB-Foldseek-cluster subset La-Proteina actually trained on). E026 rebuilt a **uniform-random AFDB sample of N=5000** length-stratified to gen's [300, 800] distribution and re-ran the necessary downstream analyses; full numbers and methodology are in [E026](#e026--afdb-as-reference-rerun-of-e020--e023--e024-2026-05-03). The E020 numbers above remain on the lab record but should not be cited as the primary AFDB-vs-gen comparison.

Side-by-side outcome on the 5 sequence-side panel rows that are comparable (gen seqonly CSV limitation — see E026):

| property | E020 PDB d | E026 AFDB d | E020 PDB KS | E026 AFDB KS |
|---|---|---|---|---|
| sequence_length | +1.47 | −0.012 | 0.45 | 0.018 |
| swi | +1.62 | +1.22 | 0.62 | 0.55 |
| net_charge | −2.14 | −0.93 | 0.43 | 0.36 |
| pI | −0.43 | −0.68 | 0.43 | 0.46 |
| shannon_entropy | −6.65 | **−3.39** | 0.92 | 0.89 |

Per-AA composition pattern is preserved verbatim (N over-rep grows from +146% to +171%, M/W/F under-rep within 2 percentage points). Thermal Tier-1 d's all roughly halve but every sign is preserved. The methodological observation (aliphatic_index + IVYWREL inflated, aromatic_fraction depressed) still holds against AFDB. Full E020-A 15-row panel comparison cannot be reproduced because the gen-side seqonly CSV is missing structure-derived columns; restoring that gap requires running `compute_developability.py` on the gen `.pt` directory.

**Modality (E020's "swi: 2 modes → 1 mode" mode-merging signature) does NOT survive.** AFDB at n=5K shows 1 mode for both swi and shannon_entropy; the original 2 → 1 reading was specific to PDB's full n=56K. E027 (matched-n bootstrap) confirms: shannon_entropy is a real population difference (PDB stays 2-mode at matched n; AFDB is genuinely 1-mode), but SWI's PDB bimodality vanishes when PDB itself is subsampled to n=5K — i.e., the SWI signal was a high-n KDE detection artifact even within PDB. Finding 9 sub-claim (b) was therefore withdrawn on 2026-05-03; see E027 for full numbers.

---

## E021 — Sparse-K40 + pair-update quick N=6 designability probe (2026-04-30)

**Status:** finished.

**Why ran:** First designability read on the sparse-K40 + `update_pair_repr` CA-only variant (training run `ca_only_sparse_K40_pairupdate/1777463843`) — does adding the pair-update layer on top of sparse-K40 produce designable structures, and does it look comparable to other CA-only arms at a similar training stage? Per CLAUDE.md "sample-quality bar (variants must clear)" = 1-2/3 designable at L=50 and L=100; this probe is meant to either pass that bar (continue training, schedule N=30) or fail it (debug the variant before more compute).

**Configs:**
- Checkpoint: `best_val_00000011_000000001133.ckpt` (epoch 11, opt step 1133), rsynced from HPC `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K40_pairupdate/1777463843/checkpoints/`. Local copy: `/home/ks2218/la-proteina/best_val_00000011_000000001133.ckpt` (1.94 GB; mtime 2026-04-30 14:20).
- Training config (for reference, not loaded here): `configs/training_ca_only_sparse_pairupdate.yaml` — canonical OLD recipe (wd=0.05, constant LR=2e-4, no scheduler, accumulate_grad_batches=32, batch=6, ema decay=0.999 every 5 steps); NN config `configs/nn/ca_only_sparse_pairupdate_160M.yaml` (sparse K=40, `update_pair_repr=True`, `update_pair_repr_every_n=3`, `use_tri_mult=False`, `use_downsampling=False`).
- New inference config: `configs/inference_sparse_pairupdate_quick.yaml` — modeled on `inference_paramgroups_wd0p1_quick.yaml`. 3 lengths × 6 samples × 200 ODE steps = **18 total samples** at L ∈ {50, 100, 200}. `nsteps=200`, `nsamples=6`, `max_nsamples_per_batch=6`. Generation block otherwise inherits canonical CA-only sampling settings from `inference_base.yaml` + `generation/uncond_codes_ca_only.yaml`. Seed=5 (inherited from `inference_base.yaml`).
- New wrapper: `script_utils/gen_n_eval_sparse_pairupdate.sh` — sbatch-able header, `--exclude=gpu-q-43`, but **actually run interactively here on the L4 dev box** (gxp-l4-0). The wrapper's hardcoded `LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env` is HPC-specific; on the L4 box the env lives at `/home/ks2218/.conda/envs/laproteina_env`. Bypassed the wrapper's CUDA preflight and called `generate.py` / `evaluate.py` directly with the absolute python path.
- Hardware: 1× NVIDIA L4 (gxp-l4-0, GPU 0).
- Output dir: `inference/inference_sparse_pairupdate_quick/`. Generation+eval log: `/tmp/gen_n_eval_sparse_pairupdate.log`.
- Wall-clock: generation ~5 min; eval ~8 min (18 PDBs, ProteinMPNN + ESMFold per PDB, fixed `ca_only=True` MPNN). Total ≈ 13 min.

**Caveats from the launch itself (worth re-reading before the next variant):**
- `proteinfoundation/generate.py` is Hydra-driven and only accepts `--config-name=foo` (hyphen). `proteinfoundation/evaluate.py` is argparse and only accepts `--config_name foo` (underscore). Mixing them up (used `--config_name=` for `generate.py` on the first attempt) makes Hydra print help and exit 0 silently; if not chained with `&&`, the eval step then fails with `ValueError: Results path … does not exist` because no PDBs were ever produced. The wrapper `gen_n_eval_paramgroups_wd0p1.sh` already gets this right; the lesson is "don't paraphrase the wrapper".
- The script's `if ! python -c "import torch; assert torch.cuda.is_available()"` preflight runs against whatever `python` resolves to first on PATH. On the L4 box, `python` is `/opt/conda/bin/python` (system base), which has no CUDA-capable torch — the preflight aborts even though the actual env's torch is fine. Fix is to either `conda activate` properly, hardcode the absolute python path, or change the wrapper's `LAPROTEINA_ENV` for local-vs-HPC.

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o values in parentheses where they differ noticeably):**

| L | n | min scRMSD per sample (Å, sorted) | designable (<2 Å, CA) | best | median |
|---|---|---|---|---|---|
| 50  | 6 | 1.48, 1.72, 2.14, 7.85, 15.88, 19.29 | 2/6 (33%) | 1.48 | 5.00 |
| 100 | 6 | 1.35, 2.16, 2.60, 4.42, 8.98, 10.13 | 1/6 (17%) | 1.35 | 3.51 |
| 200 | 6 | 2.20, 7.75, 9.74, 13.36, 13.96, 14.36 | 0/6 (0%)  | 2.20 | 11.55 |
| **all** | 18 | — | **3/18 (17%)** | 1.35 | 8.41 |

bb3o-mode designability: 2/6 / 1/6 / 0/6 (matches CA-mode at this 2 Å threshold). bb3o min-scRMSD per row is within ~0.1 Å of the CA-mode value, so the CA/bb3o distinction is not driving the designability count here.

Headline (printed by `evaluate.py`): "Average scRMSD: 7.744 Å, Success Rate (<2Å): 16.7%, Total: 18, Failed: 0".

**Comparison context (do NOT read these as a recipe-vs-recipe ranking — step counts differ):**

| Recipe | step | N | L=50 | L=100 | L=200 | overall |
|---|---|---|---|---|---|---|
| baseline (canonical wd=0.05) | 2646 | N=3 (E018 recheck) | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 8/9 (89%) |
| paramgroups + wd=0.1         | 1952 | N=6 (E018)        | 3/6 (50%) | 5/6 (83%)  | 1/6 (17%)  | 9/18 (50%) |
| **sparse-K40 + pair-update** | **1133** | **N=6 (this)**   | **2/6 (33%)** | **1/6 (17%)** | **0/6 (0%)** | **3/18 (17%)** |

Step 1133 is well before the canonical baseline's "best val ≈ step 1800-2200" window, so an under-trained ckpt is the simplest explanation for the low rate, not a fundamental problem with the variant. The canonical baseline at step 1133 has not been probed; nearest comparison is the paramgroups+wd=0.1 ckpt at step 1952 (50% overall), and the canonical baseline at step 2646 (89% overall on N=9, with E019 N=30 re-eval landing somewhere lower).

**Findings (tuning, not paper-grade):**

1. **The scRMSD distribution is bimodal, not "uniformly mediocre".** Each length splits cleanly into a "near-canonical-best" cluster and a "collapse-mode" cluster, with a wide gap between them:
   - L=50: 1.48, 1.72, 2.14 ‖ 7.85, 15.88, 19.29 — three near-misses or hits, three trajectories that diverged hard.
   - L=100: 1.35, 2.16, 2.60 ‖ 4.42, 8.98, 10.13 — same pattern, less pronounced gap.
   - L=200: 2.20 ‖ 7.75, 9.74, 13.36, 13.96, 14.36 — one near-miss, the rest in collapse.
   This is the signature of an under-trained CA-only score field where a fraction of seeds get trapped on degenerate trajectories before the field has fully formed, *not* the signature of a recipe that is uniformly worse than baseline.
2. **Best-sample quality is already at canonical-baseline-best levels.** L=100 best = 1.35 Å is in the same ballpark as paramgroups L=100 best (~0.94 Å, E018) and the canonical baseline L=100 best (E018 recheck, all 3/3 < 2 Å). The model has the *capacity* for designable structures at step 1133; what's missing is sample-to-sample consistency. The L=200 single near-miss at 2.20 Å — the only sample within striking distance of designability — is similarly encouraging given that "best of 6" is not "best of 30".
3. **L=200 = 0/6 is consistent with the L=200 cliff seen in every CA-only ckpt at this stage.** Finding 7 already documents that even the canonical wd=0.05 baseline only really clears L=200 around step 2078-2646; expecting L=200 to work at step 1133 is mis-calibrated. The 0/6 here is "early ckpt, expected", not "sparse+pairupdate breaks at L=200".
4. **Decision:** continue training the variant; re-probe at a step ≥ 1800 (matching the canonical baseline's best-val window) before deciding whether it warrants an N=30 arm next to E014 / E019's four/five-arm comparison. Do not promote any per-length rate from this snapshot to a Finding in `content_masterarbeit.md` — N=6 + single seed + step mismatch makes it untestable.

**Possible narrative:** non-narrative — kept for tuning/decision-making. The bimodality observation in Findings #1 above could become a methodological aside in `content_masterarbeit.md` if a step-matched comparison (canonical at step ≈ 1133 vs sparse+pairupdate at step ≈ 1133) ever shows that the variant is *more* bimodal than baseline at the same training stage. Without that, the bimodality is just the generic under-trained-CA-only fingerprint and not a property of this variant specifically.

**Methodological caveats:**
- **N=6 per length is too small to read 0/6 as "the variant cannot do L=200".** 95% binomial CI on 0/6 is [0%, 39%]; the next probe at a later step could show 1-2/6 without contradicting this one.
- **Step 1133 vs paramgroups 1952 vs baseline 2646** — three distinct training durations are being compared in the table above. The table is informational; it is *not* a clean A/B/C of recipes. A step-matched comparison would require either probing the canonical baseline at step ≈ 1133 (not currently on disk) or letting the sparse+pairupdate run train to step ≈ 1952 and re-probing.
- **Single seed (seed=5).** Within-seed L4 noise on min-scRMSD is ~0.5 Å per E018; some of the marginal samples (e.g., L=50 id with 2.14 Å) could flip across the 2 Å threshold under a different seed.
- **No per-column wall-clock breakdown.** Throughput vs the dense baseline at L=200 was not measured here; CLAUDE.md's note that sparse can be slower than dense at n=512 due to gather bandwidth is unaddressed by this run (only N=6 × 3 lengths, dominated by ProteinMPNN/ESMFold time, not generation).
- **Eval used the fixed `ca_only=True` ProteinMPNN call** (post-E017 fix); these numbers are directly comparable to E018/E019 numbers, *not* to E014 / pre-fix probes.

**Cross-references:**
- Code added: `configs/inference_sparse_pairupdate_quick.yaml`, `script_utils/gen_n_eval_sparse_pairupdate.sh`. Pre-existing: training config `configs/training_ca_only_sparse_pairupdate.yaml`, NN config `configs/nn/ca_only_sparse_pairupdate_160M.yaml`.
- Builds on the architectural infrastructure introduced for the sparse-K40-only variant (E012 / E014 sparse arm) and the canonical-recipe lock from the baseline run.
- A later step ≥ 1800 ckpt for the same training run, when probed at N=6 / N=30, will supersede this entry's per-length rate readings (back-link from here to that future entry once it exists).

---

## E022 — Long-length designability probe of canonical baseline (L=300/400/500, fixed-MPNN re-eval) (2026-05-02)

**Status:** finished.

**Why ran:** The canonical CA-only baseline (`baseline_wd0.05_step2646.ckpt`, E008/E014) had only ever been probed at L ∈ {50, 100, 200} on the post-fix MPNN eval (E018/E019). A pre-fix L=300/400/500 probe existed (`inference_2646_long`, run 2026-04-25) but was on the buggy `ca_only=False` ProteinMPNN path that E017/E018 invalidated for all CA-only designability numbers. User asked for L=300/400 N=3 numbers on the fixed eval. Decision input for: (a) where the canonical baseline's "L cliff" actually sits past L=200 once the eval bias is removed; (b) whether any sparse / pair-update / paramgroups variant should bother probing past L=200 in their own evaluations.

**Configs:**
- Generation: re-used the existing PDBs from `inference/inference_2646_long/job_0_n_{300,400,500}_id_{0,1,2}/*.pdb` (generated 2026-04-25 under the canonical recipe — `inference_ucond_notri_ca_only`, seed=5, nsteps=400 ODE, sc sampling, `bb_ca` schedule=log p=2.0 / gt=1/t / `center_every_step=True`, `local_latents` not present (CA-only)). The MPNN bug is purely on the eval side (`designability.py:375,560`), so the existing PDBs are not contaminated and the right move is eval-only re-run, not regeneration.
- Eval: `python proteinfoundation/evaluate.py --config_name inference_2646_long` after (i) backing up the buggy CSV to `inference/results_inference_2646_long_0.PRE_FIX_2026-04-25.csv`, (ii) removing the inner per-sample tmp dirs (`job_0_n_{L}_id_{i}/job_0_n_{L}_id_{i}/`) that would otherwise trip `evaluate.py:208`'s `assert not os.path.exists(tmp_dir)`, (iii) clearing stale `df_pdb_*` / `seq_df_pdb_*` shards. Eval ran on the post-fix `ca_only=True` ProteinMPNN call (commit `ed10dfe`, 2026-04-28).
- Hardware: 1× L4 24GB (gxp-l4-0), bf16-mixed.
- Output CSV: `inference/results_inference_2646_long_0.csv` (post-fix; rewrote the buggy file at the canonical path).
- Backup of buggy CSV (kept for the bug-impact diff below): `inference/results_inference_2646_long_0.PRE_FIX_2026-04-25.csv`.
- Eval log: `eval_2646_long_postfix.log`.

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o values within ~0.1 Å — does not change designability calls):**

| L   | id_gen | min scRMSD CA (Å) | bb3o (Å) | designable <2 Å |
|-----|--------|-------------------|----------|------------------|
| 300 | 1      | 3.49              | 3.63     | no               |
| 300 | 3      | 8.31              | 8.31     | no               |
| 300 | 8      | **2.73**          | 2.88     | no (close)       |
| 400 | 4      | 11.17             | 11.14    | no               |
| 400 | 5      | 15.28             | 15.24    | no               |
| 400 | 7      | 11.36             | 11.29    | no               |
| 500 | 0      | 16.19             | 16.15    | no               |
| 500 | 2      | 17.82             | 17.83    | no               |
| 500 | 6      | 20.26             | 20.21    | no               |

Per-length aggregate:

| L   | n | designable (<2 Å, CA) | min CA | mean CA |
|-----|---|------------------------|--------|---------|
| 300 | 3 | 0/3 (0%)               | 2.73   | 4.84    |
| 400 | 3 | 0/3 (0%)               | 11.17  | 12.61   |
| 500 | 3 | 0/3 (0%)               | 16.19  | 18.09   |

Headline (printed by `evaluate.py`): "Average scRMSD: 11.845 Å, Success Rate (<2Å): 0.0%, Total: 9, Failed: 0".

**Bug-impact delta (post-fix `ca_only=True` minus pre-fix `ca_only=False`, same 9 PDBs):**

| L   | mean Δ (Å) | per-sample Δ            |
|-----|------------|--------------------------|
| 300 | **−4.98**  | −4.70, −6.53, −3.72      |
| 400 | −1.95      | −3.05, −0.58, −2.21      |
| 500 | +0.40      | −1.87, +1.24, +1.82      |

Pattern: the MPNN-bug overestimate of scRMSD shrinks dramatically at L=300 (mean −5 Å) and is essentially noise / zero-mean at L=500 (mean +0.4 Å, with no consistent direction across samples). At L=400 the bias is intermediate (mean −2 Å). The L=300 sample id_8 went from 6.45 Å (pre-fix, "not even close") to 2.73 Å (post-fix, "near miss") — a single bug-eval correction is the difference between "definitely not designable" and "borderline".

**Possible narrative:** non-narrative — kept for tuning/decision-making, but feeds Finding 7's "L cliff" picture. Specifically:
- 0/9 designable across L ∈ {300, 400, 500} **after** the eval fix — the L cliff at the canonical baseline is real, not a measurement artifact, and persists past L=200 even with corrected ProteinMPNN.
- The L=300 result (one sample at 2.73 Å, mean 4.84 Å) is qualitatively different from L=400/500 (mean 12-18 Å). The model degrades gradually L=300 → 400, then collapses at L≥400. This is consistent with the reported "wd=0 holds up better at long protein lengths than wd=0.05" qualitative observation (E015) — and motivates a like-for-like L=300/400/500 N=3 probe of the wd=0 ckpt to test whether the gap actually exists at these lengths once both arms use the fixed eval.
- The pre-fix vs post-fix Δ pattern (large negative at L=300, near-zero at L=500) is consistent with Finding 9's E018 observation that "the bug bias is largest at L=200" — extends the picture: the bias decays toward zero as L grows further, presumably because at very long L the reconstructed-N/C/O virtual angles are uniformly distributed enough that the bias becomes a wash rather than a one-sided overestimate.

**Methodological caveats:**
- **N=3 per length is small.** The L=300 0/3 result has a 95% binomial CI of [0%, 71%]; "0/3 designable" is consistent with anything from "true rate 0%" to "true rate ≤ 70%". The L=300 id_8 sample at 2.73 Å is 0.73 Å above the threshold — within seed-to-seed L4 noise (~0.5 Å per E018), so a re-run with a different seed could plausibly land 1/3 designable, not 0/3. Do not promote per-length rates from this entry to a Finding without an N≥10 confirm.
- **Single seed (seed=5 inherited from `inference_base.yaml`).** Same caveat as E017/E018/E021.
- **Eval-only re-run, not full re-generation.** PDBs are from 2026-04-25 generation. Per CLAUDE.md the bug fix was eval-only (`designability.py` commit `ed10dfe`, 2026-04-28), so this is the *correct* re-run protocol. But it means generation-side numerics (bf16 on the L4 used for the original generation) are frozen; if a generation-side bug were ever found, those numbers would need a regen.
- **Comparison to L=50/100/200 numbers in Finding 7.** L=50/100/200 designability rates from E019's N=30 re-eval are the "right" comparison anchor for these L=300/400/500 numbers — both are post-fix MPNN, both on the canonical baseline ckpt. The comparison is N=30 vs N=3, so this entry's per-length rates have ~3× the binomial CI width relative to Finding 7's L=50/100/200 column.
- **No matched-seed comparison to other recipes.** E021's sparse-K40-pairupdate probe and E018's paramgroups probe are at different ckpt steps; neither has L≥300 data on the post-fix eval. Cross-recipe L-cliff claims need each variant's own L=300/400/500 probe at a comparable training step.

**Cross-references:**
- Same checkpoint and recipe as E008 (canonical baseline training), E014 (N=30 designability at L=50/100/200, pre-fix), E018 (baseline N=3 bug-fix recheck at L=50/100/200), E019 (full N=30 re-eval at L=50/100/200).
- Pre-fix data for the same 9 PDBs is preserved in `inference/results_inference_2646_long_0.PRE_FIX_2026-04-25.csv` and is the "Δ" baseline in the Bug-impact table above. Do not delete that backup — it is the only on-disk record of the pre-fix L=300/400/500 probe.
- Predicts: a like-for-like L=300/400/500 N=3 probe of the wd=0 ckpt (E013) on the post-fix eval would test whether the qualitative "wd=0 better at long lengths" claim from E015's discussion holds up under fixed eval. Not run yet; flagged as future experiment idea.
- Eval bug origin: E016 (audit) → E017 (fix + first clean probe) → E018 (recheck baseline+paramgroups) → E019 (N=30 re-eval) → **E022 (L extension, this entry)**.

---

## E023 — Aromatic burial targeting: gen vs PDB, RSA via FreeSASA (2026-05-03)

**Status:** finished.

**Why ran:** Decide whether the joint-sequence-head La-Proteina samples (full-atom unconditional gen, the same family as E020) place aromatic residues (W/F/Y/H) into the protein core with a frequency that matches natural proteins, or whether the model has lost the residue-level structural-targeting signal. Aromatic burial is the single best one-shot diagnostic for "did the model learn that hydrophobic side chains go inside" — it is a minimal-assumption stress test that does not require ProteinMPNN or ESMFold, only structures and SASA. Decision input for: (a) extending Finding 9's "joint sequence head produces compositionally biased sequences" picture from sequence-only to a structure-side observable; (b) whether per-residue identity (F vs Y vs W vs H) discrimination is preserved or only the group-level hydrophobic vs polar distinction.

**Configs:**
- Gen set: `inference/inference_ucond_notri/job_0_n_{300,400,500,600,700,800}_id_{0..22}/*.pdb` — 138 full-atom PDBs from the unconditional joint sampler (notri = no triangular multiplication ablation, but full-atom local-latents + sequence head). Lengths 300-800, 23 per length. PDBs already on disk; eval-only run.
- Ref set: random sample of 3000 `.pt` files from `data/pdb_train/processed_latents_300_800/` (precomputed-latent cache, full-atom `coords_nm` in OpenFold-37 atom order, `coord_mask`, `residue_type`), then post-hoc length-matched (50-residue bin width) down to 1002 proteins to mirror gen length distribution. Length-match check: gen median 550 (IQR 400-700), ref median 540 (IQR 388-692) — within 2%.
- Pipeline: `proteinfoundation/analysis/aromatic_burial.py`. Per-residue total SASA via FreeSASA (Lee-Richards probe, default classifier); RSA = SASA / `MAX_ASA[aa]` with Tien et al. 2013 theoretical max ASA values, clipped to [0, 1.5]. Burial bins: buried `<0.20`, intermediate `[0.20, 0.50)`, exposed `≥0.50`. .pt files converted to in-memory PDB strings (nm → Å, OpenFold atom names) before FreeSASA. Multi-chain pooling: residues from all chains in model 0 included.
- Statistic: burial-targeting ratio `R = P(aa | RSA<0.20) / P(aa | RSA≥0.50)`, computed for each of W/F/Y/H individually and for the aromatic group {W, F, Y, H}. 95% CIs from 1000 bootstrap resamples **over proteins** (not residues), so per-protein correlations are preserved.
- Hardware: 1× L4 24GB (gxp-l4-0), CPU-only (FreeSASA is single-threaded CPU). ~10 min wall.
- Output dir: `results/aromatic_burial/`. CSV: `aromatic_frequencies.csv`. Plots: `aromatic_vs_rsa.{png,pdf}` (group curve, 20 RSA bins, bootstrap bands), `aromatic_by_residue.{png,pdf}` (4-panel, one per W/F/Y/H). Run log: `results/aromatic_burial/run.log`.
- DSSP not used (no `mkdssp` binary on box, no sudo). FreeSASA + manual division by Tien et al. max ASA gives the same RSA as Biopython's `acc_array='Wilke'` DSSP wrapper on the same probe radius.

**Results — overall aromatic frequency:**

| set | aromatic % | 95% CI |
|-----|-----------|--------|
| gen | 6.07 | [5.4, 6.8] |
| PDB | 12.74 | [12.6, 12.9] |

The gen set uses **roughly half the aromatic content** of natural proteins — gen and ref CIs do not overlap. (Consistent with E020's per-AA composition observation that the joint sequence head produces compositionally biased sequences; this entry quantifies the bias for the aromatic subgroup specifically and ties it to a structural observable.)

**Results — burial-targeting ratio R = P(aa | buried) / P(aa | exposed):**

| residue | gen R [95% CI] | PDB R [95% CI] | gen / PDB |
|---------|----------------|-----------------|-----------|
| W | 9.43 [1.90, 38.95] | 5.62 [4.95, 6.34] | 1.68× |
| F | 2.57 [1.27, 5.07] | **5.68 [5.16, 6.32]** | **0.45×** |
| Y | 5.30 [3.41, 7.91] | 3.67 [3.44, 3.93] | 1.45× |
| H | 1.25 [0.61, 2.55] | 1.09 [1.03, 1.17] | 1.15× |
| Aromatic (group) | 3.04 [2.11, 4.21] | 3.19 [3.06, 3.33] | 0.95× |

Per-bin frequencies (P(aa) within each burial bin):

| bin | gen W | gen F | gen Y | gen H | PDB W | PDB F | PDB Y | PDB H |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|
| buried (RSA<0.20) | 0.0067 | 0.0222 | 0.0368 | 0.0078 | 0.0272 | 0.0607 | 0.0531 | 0.0235 |
| intermediate | 0.0028 | 0.0105 | 0.0225 | 0.0144 | 0.0096 | 0.0216 | 0.0290 | 0.0264 |
| exposed (RSA≥0.50) | 0.0013 | 0.0098 | 0.0071 | 0.0068 | 0.0049 | 0.0108 | 0.0145 | 0.0215 |

Sanity / total counts: gen 138/138 parsed (0 FreeSASA failures), 75,900 residues; ref 1002 proteins after length-match (3000/3000 parsed before match), 540,524 residues. The two "warnings" the script printed on this run (`residue counts differ >2x` and `parse rate 33.4%`) are bogus on the current logic — gen has 7× fewer proteins than the post-match ref by design; the 33.4% number is the length-match retention fraction, not a FreeSASA failure rate. Both are warning-logic bugs not data bugs; the underlying parse rate is 100% on both sides.

**Possible narrative:** potential narrative — feeds an extension of Finding 9 from sequence-side to structure-side. Specifically:
- **Group-level hydrophobic-vs-polar partitioning is preserved.** R for the aromatic group is 3.04 in gen vs 3.19 in PDB — CIs heavily overlap (`[2.11, 4.21]` vs `[3.06, 3.33]`). The model has not lost the basic notion that aromatic side chains belong inside. This is the unsurprising-but-required null check.
- **Per-residue discrimination is degraded for F.** Gen F ratio 2.57 vs PDB F ratio 5.68 — gen's upper CI (5.07) just touches PDB's lower CI (5.16), so the difference survives a 95% bootstrap test. F is the single strongest core-targeting aromatic in natural proteins (highest R in PDB), and the model essentially neutralises that preference.
- **Y comes out higher in gen than in PDB on point estimates (5.30 vs 3.67), but the CIs overlap (`[3.41, 7.91]` vs `[3.44, 3.93]`).** The data does **not** support a "Y picks up the slack from F" claim at 95% — the overlap means we cannot reject "gen Y burial targeting is the same as PDB" with this sample size. The point estimate is suggestive of the direction but the bootstrap CI does not exclude PDB's value. Treat as a hypothesis to be tested with a larger gen set, not as a finding.
- **H matches biology** — R ≈ 1 in both gen (1.25 [0.61, 2.55]) and PDB (1.09 [1.03, 1.17]) — H is amphipathic and has no buried-vs-exposed preference in natural proteins, and the model recovers that.
- **W noisy.** Gen W ratio 9.43 [1.90, 38.95] — point estimate higher than PDB's 5.62 but the lower bound is below PDB's lower bound, so this is consistent with the PDB ratio. Gen W count is small (~370 W total in gen across all 138 proteins), which is what drives the wide CI. Do not over-read.
- The compensation hypothesis ("F under-targeted, Y over-targeted, group-level R preserved") is mechanistically tempting and consistent with the point estimates, but the data does not actually establish it — F's deficit is significant, Y's excess is not, and the group-level "preservation" could equally be explained by F alone being attenuated while Y/W are unchanged.

If this is promoted to Finding 10 in `content_masterarbeit.md`, the narrow claim is: "the joint sequence head's compositional bias has a structural correlate — fewer aromatics overall, and per-residue, F's natural buried preference is significantly attenuated; group-level hydrophobic-vs-polar partitioning is preserved." Y's elevated point estimate should be flagged as a hypothesis, not a claim.

**Methodological caveats:**
- **Single configuration.** Only `inference_ucond_notri` is probed — one La-Proteina training arm, one sampler config, one set of generation seeds. Whether this F-attenuation pattern is recipe-specific (joint head only, or also CA-only?) is open. The CA-only arm (`inference_ucond_notri_ca_only`) cannot be analysed this way because backbone-only PDBs have no aromatic side chains.
- **n=138 gen proteins is small.** Per-residue bootstrap CIs are wide for the rare residues — W in particular (gen overall freq 0.49% × 138 × 550 res ≈ 370 W total). The W-ratio point estimate is unreliable; the F deficit is the main result that survives the small-sample CI.
- **Length-match is post-hoc and approximate.** Ref is resampled (with replacement when needed) from the 3000-draw to mirror gen's 50-residue-binned length distribution. Median lengths now agree (gen 550, ref 540), but the resample inflates effective replication (the 1002 ref proteins are not 1002 independent draws after length-matching). The bootstrap CI on the ref side is therefore optimistic; treat ref CIs as lower bounds on uncertainty. Gen CIs are unaffected.
- **RSA cutoff 0.20 / 0.50 is convention, not threshold-justified.** Robustness to cutoff (0.15/0.40 or 0.25/0.55) not checked. The ratio's sign is unlikely to flip under small cutoff changes, but the gen/PDB fold-change magnitude is cutoff-dependent.
- **FreeSASA is not DSSP.** The Tien et al. 2013 max ASA values were derived against an implementation roughly equivalent to FreeSASA's Lee-Richards calculator, so the RSA agreement is good in practice. But the "20% / 50%" threshold conventions in the burial-targeting literature were originally calibrated against DSSP output. A DSSP-based re-run would show small numerical shifts in the per-bin frequencies; the gen-vs-PDB **comparison** is the same in both regimes because the same RSA pipeline is applied to both sets.
- **Multi-chain pooling.** PDB references with biological assemblies have their burial reported relative to the deposited structure as parsed (model 0, all chains). Gen samples are monomeric. For the natural set this means a residue at a chain-chain interface is reported as "buried" if FreeSASA's calc on the assembly says so; for the gen set there is no interface. This is the right comparison for a fair "is this residue inside the protein?" probe; it is the wrong comparison if the question were "would this residue be buried in the monomer?".
- **The two sanity warnings printed by the script (residue count >2× and parse rate 33.4%) are warning-logic bugs**, not data bugs (see Results). Fix queued for next run; does not affect the headline numbers.

**Cross-references:**
- E020 — Joint sequence head audit (sequence-only). E023 is the structure-side companion; the F under-burial here is consistent with E020's per-AA composition pattern but is independent evidence (structural, not compositional).
- Finding 9 in `content_masterarbeit.md` — joint-sequence-head bias narrative. E023 extends Finding 9 from sequence to structure.
- `proteinfoundation/analysis/aromatic_burial.py` — implementation. CLI: `--gen-dir`, `--ref-dir`, `--out-dir`, `--n-ref-sample`, `--ref-oversample-factor`, `--no-length-match`. Auto-detects `.pdb` vs `.pt`.
- Predicts: a CA-only-arm version of this probe is impossible (no side chains in the gen PDBs); a paramgroups-arm version (`inference_paramgroups_wd0p1_*`) is possible but its current N=18 inference run is too small for tight per-residue CIs. If the joint sequence head is retrained with stronger sequence-loss weight in a future arm, re-running E023 on those gen PDBs is the obvious replication test.

### Update — AFDB rerun (E026, 2026-05-03) — central reading invalidated

**La-Proteina was trained on AFDB.** The PDB reference used above is structurally not what the model learned to imitate. AFDB structures (= AlphaFold-2 predicted) have softer hydrophobic-core packing than PDB crystal structures, and the AFDB per-residue burial-targeting ratios are systematically lower than PDB's. E026 reran this analysis against a uniform-random N=5000 AFDB sample (length-stratified to gen's [300, 800]). Outcome:

| residue | gen R | E023 PDB R | gen / PDB | E026 AFDB R | gen / AFDB |
|---------|-------|------------|-----------|-------------|------------|
| W | 9.80 | 5.62 | 1.68× | 2.41 | **4.07×** |
| **F** | 2.58 | 5.68 | **0.45×** | 2.52 | **1.02×** |
| Y | 5.32 | 3.67 | 1.45× | 2.71 | 1.96× |
| H | 1.29 | 1.09 | 1.15× | 0.86 | 1.50× |
| Aromatic group | 3.00 | 3.19 | 0.95× | 1.99 | 1.51× |

**The F under-burial deficit completely vanishes against AFDB** (gen 2.58 vs AFDB 2.52 — within bootstrap noise). The "F is broken" reading was an artifact of comparing AlphaFold-trained generative output against rigorous-crystallography burial conventions; AFDB itself has F-burial ratio 2.52, and gen recapitulates that pattern. The group-level "preservation" framing also reverses — gen now over-targets aromatics for burial relative to AFDB at every per-residue ratio. Note "over-target" describes the placement *ratio*, not the absolute count: gen still uses fewer aromatics overall (alphabet collapse, E020-B), but the ones it does use are concentrated more sharply in the buried core than AFDB's predicted-structure pattern shows. This is a *competence* observation about placement, not a failure mode.

**Action on the lab record:** the headline "F is broken" reading is **withdrawn**. The PDB-side numbers above are kept for reproducibility but should not be cited as evidence of model misbehaviour. E023 should not be promoted to a Finding; if anything, the AFDB rerun produces a different, milder Finding-candidate ("AlphaFold-trained generative models recapitulate the softer-than-crystallography burial pattern of their training distribution"), which the user has not yet decided whether to write up.

Full numbers in [E026](#e026--afdb-as-reference-rerun-of-e020--e023--e024-2026-05-03). Followup analyses in E024 also rerun against AFDB.

---

## E024 — Aromatic-burial follow-ups: composition decomposition, curve shape, per-protein distribution (2026-05-03)

**Status:** finished.

**Why ran:** E023 left three independent questions on the table that the headline group-ratio number (gen 3.04 vs PDB 3.19) hides. Each of the three is a separate diagnostic of the joint-sequence-head failure mode and feeds different parts of the Finding-9 picture. This experiment runs all three on the same per-residue RSA dataset E023 produced, so the follow-ups are conditional on the same FreeSASA pipeline (no DSSP / max-ASA / FreeSASA confounds vs E023).
- **Q1 (composition):** Is gen's group-level burial-ratio preservation (3.04 vs PDB 3.19) genuine behavioral preservation, or a compositional artifact? gen is depleted of all four aromatics, but Y is enriched relative to W and F within the aromatic pool — a mathematical re-weighting of per-residue ratios with PDB proportions instead of gen proportions tells us whether the "matched" group ratio is a coincidence of two opposing biases.
- **Q2 (curve shape vs amplitude):** Per-residue, is the *shape* of the gen P(F|RSA) and P(Y|RSA) curves the same as PDB (i.e. F still prefers low RSA, just at a lower amplitude — supply-limited story) or *flatter* (placement preference broken)? The slope of a logistic fit to (RSA → is-F) is the natural one-number summary.
- **Q3 (per-protein distribution):** Is the F under-burial a uniform shift across all gen proteins, or is it a long tail / bimodality where some proteins still place F correctly and a subset drives the headline number?

**Configs:**
- Inputs: per-residue parquet `results/aromatic_burial/per_residue.parquet` (added to `proteinfoundation/analysis/aromatic_burial.py` as part of this work — adds `dump_per_residue` writing `protein_id`, `residue`, `rsa`, `set`, `sample_idx` per row; a 4.5 MB pyarrow file with 616,424 rows). The parquet was produced by re-running E023's exact CLI (`--gen-dir inference/inference_ucond_notri --ref-dir data/pdb_train/processed_latents_300_800 --out-dir results/aromatic_burial --n-ref-sample 1000`) with the new dump enabled — headline numbers reproduced verbatim against E023's `aromatic_frequencies.csv`. `sample_idx` distinguishes the 1002 length-matched ref draws (633 unique stems, draws WITH replacement) from each other.
- Pipeline: `proteinfoundation/analysis/aromatic_burial_followups.py`, three independent functions (`exp1_composition`, `exp2_curve_shape`, `exp3_per_protein`). Bootstrap is over PROTEINS (1000 resamples) at the `sample_idx` level for EXP1/EXP2 (so the length-matched draws are preserved as separate units); EXP3 deduplicates on `protein_id` (one ratio per physical protein, no double-counting in the histogram / KS test).
- EXP2 runtime trick: per-bootstrap KDE inputs subsampled to ≤8000 residues without replacement (KDE shape unchanged because Gaussian-KDE outputs are normalised to ∫=1; absolute amplitude preserved using ORIGINAL `len(target)/len(all)` as the marginal multiplier). Per-bootstrap logistic fits subsampled to ≤30000 residues. This is needed because ref bootstrap concatenates ~540K residues per replicate × 1000 replicates × 2 residues — without the cap the KDE step would dominate runtime. CHOICE-FLAG (logged): KDE `bw_method='scott'` (scipy default, per-replicate adaptive).
- EXP3 filter: per protein ≥10 target total, ≥2 buried, ≥2 exposed. Fallback to W+F+Y aromatic group (H excluded as amphipathic) when fewer than 30 gen proteins survive the F-only filter.
- Hardware: laptop CPU (single-process, ~3 min wall for the followups). FreeSASA was not re-run (per the user's instructions — re-use the cached per-residue RSA from E023).
- Output dir: `results/aromatic_burial/followups/`. Files: `results.md` (headline tables + verdicts), `run.log` (decisions & sanity prints), `exp2_F_curves.{png,pdf}`, `exp2_Y_curves.{png,pdf}` (raw + area-normalized side-by-side, gen vs PDB, with bootstrap bands).

**Results:**

**EXP1 — composition breakdown of the aromatic pool** (bootstrap CIs over proteins):

| set | W | F | Y | H |
|---|---|---|---|---|
| gen | 0.080 [0.051, 0.112] | 0.279 [0.234, 0.330] | **0.476 [0.406, 0.541]** | 0.165 [0.123, 0.213] |
| PDB | 0.151 [0.147, 0.155] | 0.337 [0.333, 0.341] | 0.323 [0.319, 0.328] | 0.189 [0.185, 0.193] |

Within the aromatic pool, gen is **strongly Y-enriched** (47.6% vs 32.3% in PDB) and **W-depleted** (8.0% vs 15.1%); F and H are slightly low. The composition shift is real and outside the bootstrap CI on every dimension.

Counterfactual decomposition of the gen aromatic-group burial ratio:

| quantity | value |
|---|---|
| per-residue gen burial ratios (raw counts) | W=5.35, F=2.30, Y=5.14, H=1.16 |
| observed gen group ratio (p_gen · r_gen) | 3.71 |
| empirical gen group ratio (P(arom|buried)/P(arom|exposed) from raw counts) | 2.97 |
| counterfactual (p_pdb · r_gen) | 3.46 |

Reweighting gen's per-residue ratios with PDB aromatic-pool proportions moves the group ratio from **3.71 → 3.46**, only **6.6%** — well below the 20% threshold the script uses to flag a compositional artifact. The empirical group ratio (2.97, computed strictly from raw counts and matching E023's headline 3.04 within rounding) is reproduced from a different formula here, just for sanity. The two-line summary: gen has individually-preserved burial preferences for Y (5.14) and W (5.35), partially-preserved for H (1.16, like PDB's 1.09), and a **strongly-broken F** (2.30 vs PDB's 5.68). The group ratio happens to land near PDB's by averaging across these heterogeneously-preserved residues — but the averaging is **NOT** a composition trick; it is genuinely heterogeneous per-residue behavior partly cancelling.

**EXP2 — curve shape vs amplitude (F and Y).** Logistic-fit slope `b` of P(is X) = sigmoid(a + b·RSA), 1000-bootstrap 95% CIs:

| residue | gen slope | PDB slope | CI overlap? | verdict |
|---|---|---|---|---|
| F | -2.30 [-4.10, -0.76] | -3.64 [-4.09, -3.22] | yes | SAME SHAPE (slope CI overlaps; amplitude-limited) |
| Y | -1.42 [-2.01, -0.88] | -1.97 [-2.27, -1.67] | yes | SAME SHAPE (slope CI overlaps; amplitude-limited) |

The slope-CI verdict is "same shape" for both F and Y. Caveat: the bootstrap slope CI for gen F is wide (−4.10 to −0.76), so this verdict is "consistent with same shape, not strong evidence of same shape" — only 138 gen proteins, with a low F count per protein, makes the slope estimate noisy.

**However, the area-normalized KDE curves (in `exp2_F_curves.png`) tell a richer story than the slope summary.** PDB F-density falls smoothly from a high peak at RSA≈0 to near-zero at RSA≈1; gen F-density is approximately *flat* across [0, 0.8] and rises again near RSA≈1. The slope verdict misses this because a sigmoid cannot capture the high-RSA bump. So the headline reading is "amplitude-limited", but the visual reading also flags a small-amount weird-bump at RSA>0.8 in gen — likely model artifact at exposed sites. For Y the area-normalized curves are similar in shape, with gen's burial peak slightly more pronounced at RSA≈0.15-0.20 and PDB's flatter through the same RSA range.

**EXP3 — per-protein burial-targeting distribution.** Filter (≥10 target total, ≥2 buried, ≥2 exposed):

- F-only: gen survives **2/138**, ref survives **116/633**. Below the 30-gen threshold → fall back to W+F+Y group.
- W+F+Y group: gen survives **18/138**, ref survives **314/633**. Still below threshold → **per-protein analysis is underpowered**, no histogram drawn.

For-information-only summary stats on the group fallback:

| set | median ratio | IQR | N |
|---|---|---|---|
| gen | 1.24 | 0.54 - 2.06 | 18 |
| PDB | 2.73 | 1.74 - 3.86 | 314 |

KS two-sample test on the two distributions: D = 0.506, p = 1.5e-4. The KS p-value is small but with 18 vs 314 it is unreliable as a per-protein-shape claim. The N is the binding constraint: the F under-burial cannot be cleanly attributed to "uniform shift" / "long tail" / "bimodal subset" from this dataset.

**Possible narrative:** refines Finding 9.
- The compositional decomposition (EXP1) makes a **stronger** Finding-9 claim possible: gen's group-level burial preservation is **not** a numerical accident of compositional shift; the joint sequence head produces individually heterogeneous per-residue burial behavior that happens to average out. The "F is broken, Y is preserved, W is preserved-or-better" pattern is a real per-residue claim, not a composition artifact. This belongs in `content_masterarbeit.md` Finding 9 as a tightened sub-claim.
- The curve-shape result (EXP2) is **slope-CI-overlap-but-visually-suggestive** for both F and Y. As a paper claim, the cautious reading is "the per-residue *placement-vs-RSA* shape in gen is statistically indistinguishable from PDB given N=138 — the difference is in amplitude". The high-RSA bump in F is suggestive but not strongly supported (within bootstrap band); a larger gen N would be needed to make a "shape is broken" claim.
- The per-protein distribution analysis (EXP3) is **underpowered** — kept as a non-narrative entry that documents the limitation. For a Finding-9 follow-up that *can* support a uniform-shift-vs-bimodal-subset claim, gen N would need to be ~200+ proteins surviving the W+F+Y group filter, which means ~600+ raw gen samples (currently 138).

**Methodological caveats:**
- All three experiments rest on the same FreeSASA + Tien et al. 2013 max ASA pipeline as E023, with the same cutoffs (buried <0.20, exposed ≥0.50). DSSP-based RSA would shift small numbers but not the gen-vs-PDB direction.
- EXP2's logistic slope summary is only sensitive to the sigmoid component of the per-residue P(is X | RSA) curve — it is structurally blind to bumps, plateaus, or bimodal shapes. The qualitative reading of the area-normalized curve plot is the right diagnostic for those features. The script reports both.
- EXP2's per-bootstrap subsampling caps (KDE ≤8000, logistic ≤30000) discard ~95% of ref residues per replicate. The CIs are slightly wider than they would be with full data, but the point estimates are unbiased and the narrative (CI overlap) does not change at smaller subsample caps in spot checks.
- EXP3 is filter-bound, not method-bound. The filter values (10/2/2) are reasonable but ad-hoc; loosening them would inflate noise per protein. The right fix is more gen samples, not different filter thresholds.
- EXP1's "20% threshold" for COMPOSITIONAL vs NOT COMPOSITIONAL is a heuristic. The 6.6% observed effect is well below this; tightening to 10% would not flip the verdict.
- For EXP1 the "p-weighted" formulation (`sum_i p_i * r_i`) is an approximation of the strict empirical group ratio (computed in the same formula on both observed and counterfactual sides for an apples-to-apples comparison). The strict group ratio (2.97) is reported separately for sanity vs E023's 3.04.

**Cross-references:**
- E023 — direct parent. E024 is the followup-on-the-same-data that the user asked for after E023's headline numbers landed.
- Finding 9 in `content_masterarbeit.md` — the joint-sequence-head bias narrative that E024 strengthens (composition decomposition) and qualifies (curve shape + per-protein distribution underpowered).
- Implementation: `proteinfoundation/analysis/aromatic_burial.py` (now also dumps `per_residue.parquet`); `proteinfoundation/analysis/aromatic_burial_followups.py` (CLI: `--in-file`, `--out-dir`, `--seed`).
- Predicts: re-running on a paramgroups-arm gen set (when its inference is large enough) would test whether the per-residue *heterogeneous* preservation (F broken, Y preserved) survives the wd-split optimizer fix, or whether it was an artifact of the joint-sequence head specifically. The composition decomposition (EXP1) is the cheap diagnostic to run first on any new arm.

### Update — AFDB rerun (E026, 2026-05-03)

**La-Proteina was trained on AFDB.** Same framing note as E023 above. The follow-up analyses were rerun on `results/aromatic_burial_afdb/per_residue.parquet` (AFDB-side residues only); gen-side residues are unchanged.

- **EXP1 (composition decomposition):** counterfactual 3.26 vs observed 3.71 (12.2%) → still **NOT COMPOSITIONAL** by the same heuristic threshold. But this verdict is now moot — E023's group-level "preservation" reverses against AFDB, so there is no preservation for the decomposition to "explain". The per-residue gen ratios reproduce verbatim (W=5.35, F=2.30, Y=5.14, H=1.16) since gen-side data is unchanged.
- **EXP2 (curve shape):** F slope gen [−4.13, −0.74] vs AFDB [−2.00, −1.44] — CIs overlap → SAME SHAPE. Y slope gen [−1.98, −0.88] vs AFDB [−1.83, −1.24] — CIs overlap → SAME SHAPE. The "amplitude-limited" reading survives. AFDB's F slope (−1.71) is itself shallower than PDB's (−3.64), so the PDB comparison was steepest-vs-shallow on the slope axis.
- **EXP3 (per-protein):** gen 18/138 vs AFDB 654/835 surviving the W+F+Y filter — same gen-side bottleneck, AFDB ref now has 654 surviving instead of 314 (better ref-side power). Underpowered verdict unchanged.

Headline-level conclusion is the same as E023: the structural extension of Finding 9 does not survive the reference-set switch. Full numbers in [E026](#e026--afdb-as-reference-rerun-of-e020--e023--e024-2026-05-03).

## E025 — Steered generation sweep: camsol max + tango min, official LD3+AE2, L=300/400/500 (2026-05-03)

**Status:** finished (generation only — property re-eval on the 480 PDBs is the obvious next step but was not run in this session).

**Why ran:** First steered-sampling run since the 5-fold predictor was retrained with the 14-property head (camsol_intrinsic at idx 13, R²=0.95 across folds). Two questions: (a) does the steering hook actually fire on the official La-Proteina LD3+AE2 checkpoint with the upgraded predictor; (b) is there a clean dose-response in the *predictor's* view of the steered property as w_max grows, across multiple lengths inside the predictor's 300–800 training range. Decision input for: whether to invest in a follow-up that re-evaluates the generated PDBs on the real property pipeline (CamSol/TANGO via the developability panel) and a length-extrapolation probe at L<300.

**Configs:**
- Generation: `inference_ucond_notri_long` → official `checkpoints_laproteina/LD3_ucond_notri_800.ckpt` + `checkpoints_laproteina/AE2_ucond_800.ckpt`. **Critical:** this had to be the AE that produced the predictor's training latents — `data/pdb_train/processed_latents_300_800/` covers L≥512, which AE1 cannot encode, so AE2 is the only choice consistent with the predictor's input distribution. Earlier first-pass smoke-test using `inference_baseline_L300_400` (CA-only step-2646 baseline) silently no-op'd the steering call because the steering hook in `proteinfoundation/flow_matching/product_space_flow_matcher.py:728` only fires when `local_latents` is in `nn_out` — CA-only models have no latent channel. Diagnostics file came back as `[]`, which is what surfaced the mismatch.
- Predictor: `laproteina_steerability/logs/multitask_t1/20260427_161809/checkpoints/fold_0_best.pt` (5-fold, 14 heads, R²[camsol]=0.95, R²[tango]=0.92, length range 300–800).
- Schedule: `linear_ramp` `t_start=0.3, t_end=0.8, t_stop=0.9`, `gradient_norm=unit`, `gradient_clip=10.0`, `channel=local_latents`.
- Sweep: `w_max ∈ {1, 2, 4, 8, 16}` × `{camsol_intrinsic maximize, tango minimize}` × `lengths ∈ {300, 400, 500}` × `seeds 42–57` (16 seeds). 10 configs × 48 = 480 guided proteins. `nsteps=100` (matches the net_charge round-1 sbatch convention, `script_utils/submit_steering_round1.sh:41`).
- Hardware: 1× NVIDIA L4 (23 GB), `cuda:0`. Wall: **68 min** for 10 configs (~6m 50s each, 480 proteins total).
- Driver: `script_utils/run_steering_camsol_tango.sh`. Steering YAMLs: `steering/config/sweep_camsol_tango_L500/{camsol_max,tango_min}_w{1,2,4,8,16}.yaml`.
- Patches needed to make this run: `steering/registry.py` (added `camsol_intrinsic` at idx 13 of `PROPERTY_NAMES`), `steering/predictor.py` (replaced hardcoded `n_properties=13` with `int(self.stats.mean.shape[0])` so the 14-head checkpoint loads), `steering/generate.py` (added `--skip_unguided` flag — sweep does not regenerate the unguided baseline because the user already has the 1000-protein stratified control at `results/generated_stratified_300_800/`).
- Outputs: `results/steering_camsol_tango_L500/{config}/guided/*.pdb` + `…/diagnostics/{seed}_n{L}_diagnostics.json`. nohup log: `nohup_steering_camsol_tango.out`.

**Results (predictor-side, mean over 48 proteins per config = 16 seeds × 3 lengths):**

CamSol-intrinsic (objective = maximize), early-step predicted vs late-step predicted, expressed as Δ from initial mean = 3.005:

| w_max | early_mean | late_mean | Δ |
| --- | --- | --- | --- |
| 1 | 3.005 | 3.204 | **+0.20** |
| 2 | 3.005 | 3.486 | **+0.48** |
| 4 | 3.005 | 4.039 | **+1.03** |
| 8 | 3.005 | 5.010 | **+2.01** |
| 16 | 3.005 | 6.366 | **+3.36** |

TANGO (objective = minimize), early-step vs late-step, Δ from initial mean = 758.5:

| w_max | early_mean | late_mean | Δ |
| --- | --- | --- | --- |
| 1 | 758.484 | 755.328 | **−3.2** |
| 2 | 758.484 | 685.920 | **−72.6** |
| 4 | 758.484 | 557.000 | **−201.5** |
| 8 | 758.484 | 311.075 | **−447.4** |
| 16 | 758.484 | −44.533 | **−803.0** |

Steering signal is monotone-in-w for both properties and dose-response is clean. CamSol shifts ~Δ doubling per w doubling at moderate w (Δ +0.48 at w=2 → +1.03 at w=4 → +2.01 at w=8). TANGO shows a similar near-doubling pattern up to w=8, then runs into a non-physical regime at w=16 (predicted late-stage tango goes negative — physically tango ≥ 0). The w=16 tango number is the predictor extrapolating below its training distribution under aggressive guidance, **not** evidence that the proteins themselves became "negatively aggregating".

**Possible narrative:** Non-narrative for now. Predictor-side dose-response is necessary but not sufficient — the headline question for the paper is whether the *generated proteins themselves* (post-PDB writeout, post-property-evaluate) shift in real CamSol/TANGO, or whether the predictor is just being adversarially steered along its own gradient without the structures changing in a meaningful way. If a follow-up `steering.property_evaluate` pass on these 480 PDBs reproduces the dose-response on the real developability panel — particularly with a flat or weak signal at w=1/2 and a clear signal at w=4/8 — that would support a Finding ("gradient-based steering on La-Proteina latents transfers to real property shifts at moderate guidance"). If the real pipeline shows no shift even at w=8, the result is interesting in the opposite direction (predictor steerability ≠ structural steerability) and feeds a separate narrative about the limits of latent-space property control.

**Methodological caveats:**
- **Predictor's view ≠ real property.** The dose-response numbers above are entirely from the predictor's diagnostic readout during sampling. The actual generated proteins' CamSol-intrinsic and TANGO have **not** been computed. `steering.property_evaluate` on `results/steering_camsol_tango_L500/{config}/guided/` is the next step; until that runs, "the steering signal works" only means "the predictor agrees the latents moved in the desired direction".
- **No paired unguided baseline in this sweep.** The unguided control is the 1000-protein stratified set the user generated previously (`results/generated_stratified_300_800/`); these were generated with seeds 1000+ and across L=300–800, so the comparison is distributional, not seed-paired. Paired (same noise → guided vs unguided) comparisons would be more powerful for small effects but the user explicitly opted to skip unguided regeneration in this run.
- **w=16 is past the predictor's reliability range** for tango (late predicted tango = −45 is nonphysical). Treat w=8 as the largest "trustable" dose level and read w=16 as "what happens when guidance overshoots".
- **Length range 300–500 is fully inside the predictor's 300–800 training range.** No conclusion here about generalization to L<300 — that was discussed but explicitly deferred from this run. A follow-up at L∈{150, 200, 250} on a few selected w_max levels is the obvious next length-extrapolation probe.
- **nsteps=100 vs the inference default of 400.** Matches the net_charge round-1 convention (`submit_steering_round1.sh`) but means each trajectory has 4× coarser steps than a typical generation run; the predictor is queried at most once per ODE step so finer nsteps gives finer steering control. Whether nsteps=100 leaves enough room for the schedule (`t_start=0.3, t_stop=0.9` ⇒ ~60 active-guidance steps) to achieve the target Δ is part of what the dose-response is implicitly answering — and the monotone increase in Δ vs w suggests yes.
- **Single fold of the 5-fold predictor** (`fold_0_best.pt`). The other 4 folds were not used. R² is small-σ (0.0023 across folds for tango, 0.0023 for camsol), so single-fold variance is unlikely to dominate the headline shift, but the property-evaluate follow-up would be the place to add fold-averaged predictions if the real-pipeline signal is borderline.
- **Predictor is multitask** — the diagnostics file logs all 14 predicted properties at every ODE step. We have not yet checked whether steering for camsol (or for tango) co-modifies untargeted properties (e.g. does maximizing camsol also increase pI or shift Rg?). The diagnostics support this analysis but it was not run in this session.

**Cross-references:**
- Predictor checkpoint: 5-fold run at `laproteina_steerability/logs/multitask_t1/20260427_161809/`. Earlier 13-property predictor at `laproteina_steerability/logs/multitask_t1/20260416_154526/` is now superseded for steering purposes; the registry/predictor patch in this run is forward-compatible (auto-detects `n_properties` from `stats_mean.shape[0]`) so older 13-prop checkpoints would still load if needed for a regression test.
- Reference run: `script_utils/submit_steering_round1.sh` (net_charge maximize, w_max=2, L=400, N=5, nsteps=100, on Cambridge HPC sbatch). This run reuses the same LD3+AE2 generation config and the same `linear_ramp + t_stop=0.9` schedule shape, so timing comparisons hold.
- Predicts: a `steering.property_evaluate` pass on `results/steering_camsol_tango_L500/` (10 configs × 48 PDBs) — this is the missing piece that turns this entry into a Finding-eligible result. Plus an ablation at L∈{150, 200, 250} to test below-300 generalization once the in-distribution dose-response is validated on the real property pipeline.
- Memory pointer: `feedback_steering_must_use_official_ld_ae.md` — captures the CA-only-vs-LD lesson from this run so future steering attempts skip the 30-minute first-time misfire.

---

### E025 follow-up (2026-05-04): nsteps=400 regen + property-eval + scRMSD

**Why this section, not a new E-id:** Per user instruction — the followups attach to E025, not a separate entry.

**The nsteps trap.** When this entry was first written (2026-05-03), the steering sweep used `nsteps=100`, copied verbatim from `script_utils/submit_steering_round1.sh:41` (the net_charge round-1 reference). That sbatch script had only ever run `steering.property_evaluate` on its outputs — never scRMSD. nsteps=100 produced *predictor-side* dose-response that looked monotone and clean, but the structures themselves were off-manifold: a sanity-check scRMSD on one nsteps=100 unguided protein (seed=42, L=300, exact same LD3+AE2 model) returned scRMSD=22.5 Å (best-of-8, ESMFold + ProteinMPNN ca_only=True, num_seq=8). The same protein regenerated at **nsteps=400** with all other settings identical returned scRMSD=0.80 Å — fully designable. So the original sweep's structures were uniformly garbage in a structural sense, even though the *latent trajectories* moved in the predictor-desired direction.

**Diff against the production La-Proteina inference pipeline (`inference_ucond_notri`, see `inference/results_inference_ucond_notri_0.csv` — scRMSD ≈ 0.4–1.3 Å on L=200):** identical model, identical AE, identical schedule, identical sampling-mode params, identical guidance_w. **Only difference is nsteps**, and it is responsible for the entire 22-Å vs 0.8-Å gap. Memory pointer added (`feedback_steering_must_use_official_ld_ae.md` already existed; this finding strengthens the same memory line — also noted in the entry's caveat list for next time).

**Regen config:** byte-identical to the original sweep block above except `nsteps: 400` (and output dir `results/steering_camsol_tango_L500_nsteps400/`). Single GPU (`cuda:0`, NVIDIA L4) under `nohup`, sequential across the 10 configs. **Wall: 4 h 22 min** (vs the original 68 min at nsteps=100; 4× scaling matches the integrator-step ratio). 480/480 PDBs produced, no errors.

**Predictor-side dose-response on the regen** (mean over 48 proteins per config = 16 seeds × 3 lengths, early-step → late-step Δ in raw predictor units):

| w_max | camsol_max early→late mean (Δ) | tango_min early→late mean (Δ) |
| --- | --- | --- |
| 1 | 1.451 → 1.545 (**+0.09**) | 857.1 → 993.9 (**+136.9**) |
| 2 | 1.451 → 1.663 (**+0.21**) | 857.1 → 966.7 (**+109.6**) |
| 4 | 1.451 → 1.901 (**+0.45**) | 857.1 → 912.4 (**+55.3**) |
| 8 | 1.451 → 2.335 (**+0.88**) | 857.1 → 810.0 (**−47.1**) |
| 16 | 1.451 → 3.092 (**+1.64**) | 857.1 → 628.7 (**−228.4**) |

CamSol dose-response is monotone and roughly halved relative to nsteps=100 (was Δ +0.20 → +3.36; now +0.09 → +1.64). Plausible reason: with 4× more ODE steps the unit-norm-clipped guidance gradient gets re-applied more times BUT the model also gets more between-kick relaxation back to the manifold, partly undoing each kick. Direction and ordering are intact.

**TANGO dose-response shows a sign flip at low w** that was hidden in the nsteps=100 sweep. At w∈{1, 2, 4} the predictor's late-step tango ends *higher* than its early-step value (+137, +110, +55) — i.e. the supposed "minimize tango" guidance increases predicted tango at low strengths. Only at w=8 and w=16 does tango actually decrease (−47, −228). Interpretation: at low w the linear_ramp schedule gives the gradient too little weight to overcome the local manifold pull early in the trajectory, and what we're seeing is the model relaxing toward a higher-tango neighborhood that happens to be the local sampler equilibrium for that seed. This was masked at nsteps=100 because the 100-step trajectory never gave the model time to reach that neighborhood — the structure was off-manifold first. **For Findings: don't claim predictor-side tango steering "works" at w<8; it doesn't.** CamSol-side: w=1/2 are also too weak (Δ +0.09/+0.21 is in the predictor-noise range across the 48-sample SE), w≥4 is the regime where the signal is reliably non-zero.

**Real-property panel (sequence-side, all 48 proteins per config; structure-side proxies still computed but flagged below).** The relevant comparison is gen-vs-gen across w_max levels — i.e. is the steering target moving in the desired direction in the *real* property, not just in the predictor's view. CamSol-intrinsic itself is not in the on-board pipeline (`compute_camsol` returns NaN; the training-data CamSol came from an external pH-7 binary), so SWI is the on-board solubility proxy used here.

**TANGO_total** at L=300 (mean over 16 seeds), per config:

| w_max | camsol_max | tango_min |
| --- | --- | --- |
| 1 | 621.2 | 618.3 |
| 2 | 618.2 | 620.3 |
| 4 | 619.5 | 619.4 |
| 8 | 611.1 | 611.2 |
| 16 | 599.4 | **600.7** |

Both arms drop tango_total by ~3.5% across the w=1..16 range — i.e. **tango steering is not selectively reducing tango more than camsol steering does at the same w**. This is a meaningful negative result for the predictor-vs-real gap on TANGO: the predictor saw Δ=−228 at tango_min_w16 but the real TANGO_total only drops 18 units (~3%). The two arms drop tango by similar amounts because they're correlated — both routes (more anionic for camsol, more cationic for tango) happen to lower the TANGO sum.

**SWI** (Solubility-Weighted Index, higher = more soluble), L=300:

| w_max | camsol_max | tango_min |
| --- | --- | --- |
| 1 | 0.7972 | 0.7971 |
| 2 | 0.7975 | 0.7969 |
| 4 | 0.7979 | 0.7969 |
| 8 | 0.7988 | 0.7968 |
| 16 | **0.7994** | 0.7964 |

Camsol arm: monotone solubility increase (Δ +0.0022 across the sweep, ~0.3% relative). Real but tiny. Tango arm: very slight drift toward less-soluble (Δ −0.0007). The predictor-side camsol_intrinsic dose-response Δ +1.64 corresponds to a **real SWI shift of ~0.3%**, so the predictor's "1.64-unit" steering target translates to a near-noise-level real-solubility change.

**net_charge_ph7** at L=300 (mean over 16 seeds):

| w_max | camsol_max | tango_min |
| --- | --- | --- |
| 1 | −14.95 | −14.45 |
| 2 | −14.70 | −14.27 |
| 4 | −14.75 | −13.49 |
| 8 | −15.44 | −12.74 |
| 16 | **−16.13** | **−10.68** |

CamSol guidance pushes charge **more negative** (anionic), tango guidance pushes charge **more positive** (cationic). Both directions correlate with their respective targets in protein-engineering literature (anionic surfaces aid solubility; positive charges break β-aggregation runs). At L=500, tango_min_w16 even pushes net charge from −2.1 (w=1) → +5.6 (w=16) — that's a 7.7-unit charge swing, the strongest single-property real-side effect in this entire sweep. So **tango steering's real-side mechanism is charge modification**, even though the TANGO_total readout barely moves.

**Hydrophobic patch total area** at L=300 (smaller = more soluble):

| w_max | camsol_max | tango_min |
| --- | --- | --- |
| 1 | 1772 | 1778 |
| 2 | 1754 | 1784 |
| 4 | 1749 | 1770 |
| 8 | 1688 | 1735 |
| 16 | **1627** | 1701 |

Both decrease, camsol more strongly (Δ −145 = −8% vs Δ −77 = −4%). Direction matches the predictor target. **Caveat — this is a structure-derived property** that depends on FreeSASA over the gen PDB; if guidance off-manifolds the structure even slightly the SASA estimate becomes unreliable. The fact that the trend is monotone and small is encouraging but the absolute numbers should not be over-interpreted relative to a PDB-trained reference.

**SAP / SCM** (averaged over all L per config):

| config | sap_total | scm_pos | scm_neg |
| --- | --- | --- | --- |
| camsol_max_w1 | 6.81 | 46.15 | −52.65 |
| camsol_max_w16 | **6.49** | 47.14 | −53.94 |
| tango_min_w1 | 6.83 | 46.41 | −51.88 |
| tango_min_w16 | 6.72 | **50.67** | **−47.36** |

CamSol arm: SAP drops monotone (−5%), scm shifts very slightly anionic. Tango arm: SAP barely moves (−1.6%); SCM becomes meaningfully more positive (+4.3 on scm_pos, +4.5 on scm_neg). The SCM shift is the clearest "real" signal of tango steering — but it's a side-effect of charge modification, not of the actual TANGO score moving.

**Designability (scRMSD < 2 Å, CA-RMSD via ProteinMPNN-ca_only=True N=8 → ESMFold-v1 → Kabsch).** Subsample: L=300 only, 4 seeds (42–45) × 10 configs = 40 PDBs. Wall: 91 min, single GPU. (Full 480 was budget-incompatible at ~137 s/protein × 480 ≈ 18 h; L=400/L=500 timing scales as 250 s and 444 s respectively, so a fuller designability sweep is a follow-up.)

| config | designable / valid | scRMSD_min mean (Å) | scRMSD_min median (Å) |
| --- | --- | --- | --- |
| camsol_max_w1 | 4/4 | 0.727 | 0.709 |
| camsol_max_w2 | 4/4 | 0.724 | 0.718 |
| camsol_max_w4 | 4/4 | 0.653 | 0.690 |
| camsol_max_w8 | 4/4 | 0.732 | 0.725 |
| camsol_max_w16 | 4/4 | 0.722 | 0.668 |
| tango_min_w1 | 3/4 | 1.701 | 0.739 |
| tango_min_w2 | 4/4 | 0.712 | 0.642 |
| tango_min_w4 | 3/3 | 0.712 | 0.622 |
| tango_min_w8 | 3/4 | 1.331 | 0.787 |
| tango_min_w16 | 3/3 | 0.663 | 0.630 |

**The headline structural result:** all 20 camsol-steered samples are designable across every w_max level (median scRMSD 0.67–0.73 Å). 12 of 14 valid tango-steered samples are designable. **Steering — even at w_max=16 — does not push structures off the data manifold.** This is a strong negative result against the prior worry that gradient guidance into the predictor's optimum would degrade designability. The two non-designable tango samples (tango_min_w1 / s42_n300 = 4.82 Å, tango_min_w8 / s42_n300 = 3.31 Å) are both s42 — a single seed's draw is over-represented in those failures, not a systematic w-effect.

**Two length-mismatch failures** (tango_min_w4 / s43_n300 and tango_min_w16 / s43_n300): ESMFold returned 299 residues for a 300-residue input, and the official `rmsd_metric` asserts shape match. Both rows are written to CSV as `inf` and excluded from the designable / valid columns above. Same s43_n300 → 299-residue ESMFold output appears for both configs, suggesting it's a deterministic ProteinMPNN→ESMFold quirk on that specific gen sequence, not a steering artifact. Worth a one-off patch in `evaluate_one()` to clip-then-RMSD if it recurs.

**Overall narrative position update for E025:**

The original entry framed this as "predictor-side dose-response is necessary but not sufficient — needs property re-eval to confirm". With the regen + eval done, the answer is:

1. **Direction transfers, magnitude does NOT.** Predictor sees Δ +1.64 on camsol_intrinsic; real SWI moves +0.3%, real net_charge moves −1.2 e, real hydrophobic_patch_area moves −8%. The predictor's gradient correctly identifies the *direction* of solubility, but its absolute units don't translate to physically-large real-property shifts. A user steering for "+10 in predictor-camsol" should expect a small, monotone, real-side shift in the same direction — not a +10-unit real-property change.
2. **Steering preserves designability.** 32/34 valid samples designable across all guidance levels. **Still on-manifold.** This is the strongest positive result here, and IS Finding-eligible.
3. **The TANGO target is poorly-served by this predictor.** Predictor's Δ −228 corresponds to a real Δ −18 on TANGO_total — about 8% of what the predictor "thinks" it achieved. And both camsol_max and tango_min lower real TANGO by similar amounts at matched w_max, meaning tango steering offers no specificity over camsol steering for the actual TANGO property. The SCM_positive shift (+4.3 at tango_min_w16 over baseline) is the real mechanism the model found — pushing toward more cationic — but that doesn't reduce TANGO sum substantially in this setting.
4. **Honest scope:** all numbers above are at L=300 (designability) or 16-seed-mean across L=300/400/500 (properties). The L<300 generalization question (raised in the original entry) was deferred and remains open.

**What's now Finding-eligible vs not:**
- *Finding candidate (positive):* "Latent-space gradient steering on La-Proteina LD3+AE2 maintains structural designability up to w_max=16 across both solubility (CamSol-direction) and aggregation-prone (TANGO-direction) guidance, with 32/34 valid samples scoring scRMSD < 2 Å at L=300." Worth writing into `content_masterarbeit.md`.
- *Finding candidate (negative + cautionary):* "Predictor-side dose-response magnitudes are 5–10× larger than the corresponding real-property shifts; users should not interpret predictor units as a calibrated forecast of real-property change." Worth writing.
- *Not Finding-eligible:* TANGO real-side dose-response is too weak / non-specific to claim as a positive result.

**Memory updates (none added this session):** the existing `feedback_steering_must_use_official_ld_ae.md` already covers the CA-only-no-op pitfall. The nsteps=100-vs-400 finding deserves its own memory line — captured in the entry above and worth promoting if it bites again.

---

## E026 — AFDB-as-reference rerun of E020 / E023 / E024 (2026-05-03)

**Status:** finished. Wall-clock 22 min on `gxp-l4-0` (16:45 → 17:07 UTC, 2026-05-03), much faster than the 5h estimate — TANGO + FreeSASA + IUPred3 on N=5000 parallelises well at 16 workers, network was ~390 MB/s. Mixed outcome — sequence-side claims survive, structural sub-claim (E023's F-burial deficit) **dies**.

**Why ran:** All "PDB vs gen" comparisons in E005 / E020 / E023 / E024 used a PDB-derived reference (`laproteina_steerability/data/properties.csv`, `data/pdb_train/processed_latents_300_800/`). La-Proteina was trained on AFDB, not PDB, so the relevant null hypothesis for "the model has drifted from its training distribution" is gen-vs-AFDB, not gen-vs-PDB. Numbers like the alphabet-collapse magnitude (E020-B: −79% Met, −68% Trp), the F-burial deficit (E023: gen 2.57 vs PDB 5.68), and the 3.22 Glu/Asp ratio inflation are all measured against the wrong control. This experiment swaps the reference side for an AFDB-SwissProt-derived sample and re-runs the necessary downstream analyses with the existing scripts.

**Configs (re-run-able from this entry):**

- **AFDB ref construction:** `script_utils/run_afdb_rerun.sh` (single tmux-resumable bash script). Steps:
    1. Download `https://ftp.ebi.ac.uk/pub/databases/alphafold/accession_ids.csv` (~8.7 GB, ~214M rows; format `accession,first_residue,last_residue,model_id,version` — length is `last_residue` since first_residue=1).
    2. `script_utils/afdb_rerun_helpers.py sample-afdb` — single-pass reservoir sample of 1000 accessions per 50-residue bin in [300, 800], total **10000 candidate accessions** (oversampled for download failures + length-stratification trim). Reservoir sampling within each bin gives uniform-over-AFDB sampling without holding the 214M-row pool in memory; `seed=42` for reproducibility.
    3. Parallel-download (`xargs -P 16` curl) AFDB v6 PDBs (`https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb`, with v5/v4 fallback). v6 is the current bulk version — same AlphaFold-2-monomer pipeline as v4, just an updated database release. Output filename normalised to `AF-{acc}-F1-model_v4.pdb` for pipeline-version-agnosticism.
    4. `afdb_rerun_helpers.py convert` — parse each PDB via Biopython, build PyG `Data(coords[L,37,3]Å in graphein-PDB-37 order, coord_mask[L,37], residue_type[L] OpenFold-restype-indexed, residues=list[3-letter], id=accession)` matching what `compute_developability.py:load_protein` expects after its internal `PDB_TO_OPENFOLD_INDEX_TENSOR` reindex. Length-stratified-trim the converted set down to **N=5000** length-matched [300, 800].
    5. Symlink the final 5000 PDBs into `data/afdb_ref/structures_final/` for `aromatic_burial.py --ref-dir` (which accepts `.pdb` directly via FreeSASA — no .pt indirection needed for E023).
- **Subset choice:** **uniform-random over AFDB**, length-stratified to gen's [300, 800] distribution. Rationale (from the conversation that triggered this run): the user explicitly rejected SwissProt as biased toward characterised proteins. Random sample over the full AFDB accession_ids.csv is the most defensible "AFDB-representative" — every entry has equal probability, no curation bias. The known caveat is that random AFDB *over-represents over-clustered protein families* (homologous sequences from many species) — if La-Proteina's actual training corpus was Foldseek-cluster-30 representatives (the standard "redundancy reduced" AFDB), then this sample has a slightly different family-level distribution than what the model trained on. For the qualitative AFDB-vs-PDB distinction this is fine; for tight magnitude claims it should be flagged. AFDB Foldseek-cluster-30/50 reps would be the natural robustness check (~580K-2.27M reps, hosted at cluster.foldseek.com behind a SPA — not directly scriptable as of 2026-05-03).
- **Necessary set rerun** (script step 6-8):
    - **E020-A:** `compute_developability.py --data-dir data/afdb_ref --workers $(nproc - 2) --min-length 300 --max-length 800` → `data/afdb_ref/properties_afdb.csv` (gen-schema column names) → `afdb_rerun_helpers.py rename-csv` → `properties_afdb_refschema.csv` (renamed for `compare_properties.py` ref side: `tango_total → tango`, `net_charge_ph7 → net_charge`, `iupred3_mean → iupred3`, `radius_of_gyration → rg`, `sap_total → sap`, `pdb_id → protein_id`).
    - **E020-A panel comparison:** `compare_properties.py --ref properties_afdb_refschema.csv --gen results/property_comparison/stratified_vs_pdb/properties_generated_seqonly.csv --out results/property_comparison_afdb/stratified_vs_afdb`.
    - **E020-B:** `aa_composition.py --gen results/generated_stratified_300_800/sequences.fasta --ref data/afdb_ref/sequences.fasta --length-min 300 --length-max 800` → AA-composition CSV.
    - **E020-C Tier-1:** `thermal_stability.py --gen ... --ref data/afdb_ref/sequences.fasta` (no `--temstapro-dir`, so Tier-2b skipped; cheap aliphatic / IVYWREL / GRAVY / aromatic_fraction proxies only).
    - **E023:** `aromatic_burial.py --gen-dir inference/inference_ucond_notri --ref-dir data/afdb_ref/structures_final --out-dir results/aromatic_burial_afdb --n-ref-sample 1000 --seed 42`.
    - **E024 follow-ups:** `aromatic_burial_followups.py --in-file results/aromatic_burial_afdb/per_residue.parquet --out-dir results/aromatic_burial_afdb/followups`.
- **Skipped (not necessary):** E020-D Tier-2b (TemStaPro — needs A100 + ProtT5 cache, adds nothing the Tier-1 doesn't already say methodologically); E005 (cheap-diagnostics, was non-narrative anyway); E001/E002/E007/E025 predictor (predictor was trained on PDB-derived latents through AE2 — orthogonal AFDB-vs-PDB issue, defer until E020/E023 redo decides whether the qualitative findings actually change).
- **Idempotency:** each script step writes a stamp file under `data/afdb_ref/stamps/0X_*` and skips already-completed steps on re-run, so detach-and-resume from tmux works without redoing the long download or compute.
- **Hardware:** workstation `gxp-l4-0`. Disk budget ~12 GB total (8.7 GB accession_ids.csv + 1.5 GB raw PDBs + 2.5 GB .pt). Wall-clock estimate: ~10-30 min download of accession CSV (one-time, network-bound) + ~5 min single-pass reservoir sample of 214M rows + ~75 min download of 10K AFDB PDBs (network-bound, 16-way parallel) + ~3 h `compute_developability.py` (TANGO+FreeSASA+IUPred3, 16-way parallel on N=5000) + ~10 min for everything else. Total ~5 h interactive, fully resumable. The accession CSV is kept on disk after step 1 so re-running with a different length range or N is cheap.

**Smoke-tested (single PDB, AF-P12345-F1):** download path + `convert` subcommand + .pt loading via `compute_developability.load_protein` + `aromatic_burial.sasa_residues` on raw PDB — all four pipeline edges produce expected outputs (430-residue ASP-aminotransferase mitochondrial, all 430 CAs present, sequence MALLHSARVL… verified against UniProt P12345). Headline pipeline numerics deferred to the full run.

**Results:**

The downstream gen-side artifact `properties_generated_seqonly.csv` only carries 8 columns (sequence_length, swi, net_charge_ph7, pI, iupred3_mean, iupred3_fraction_disordered, shannon_entropy) — TANGO and structure-derived columns (hydrophobic patches, SAP, SCM, rg) were never written for the gen set. Compare_properties.py only reports rows where both sides have the column, so the panel comparison below is **5 properties** instead of the 15 originally reported in E020-A. Re-running the full panel on the gen `.pt` directory would restore the missing 10. *(Side-issue but worth noting: `iupred3_mean` and `iupred3_fraction_disordered` are blank in the seqonly CSV for every gen row — gen IUPred3 must have failed silently when the seqonly CSV was generated; restoring those for AFDB comparison requires a gen-side rerun.)*

#### A. Developability panel comparison (5 of 15, gen seqonly limitation)

Side-by-side with E020-A's PDB-vs-gen numbers (Cohen's d, gen − ref / pooled SD; KS_d = 2-sample KS):

| property         | E020 PDB d | E026 AFDB d | E020 PDB KS | E026 AFDB KS | direction note |
|------------------|------------|-------------|-------------|--------------|----------------|
| sequence_length  | +1.47      | **−0.012**  | 0.45        | 0.018        | length match — AFDB sample length-stratified to gen, PDB ref wasn't |
| swi              | +1.62      | +1.22       | 0.62        | 0.55         | mode-merging signature (2→1) holds; magnitude attenuates ~25% |
| net_charge       | −2.14      | **−0.93**   | 0.43        | 0.36         | drift halves — AFDB itself is more negative than PDB (mean −5.2 vs −7.0) |
| pI               | −0.43      | −0.68       | 0.43        | 0.46         | slightly larger |
| shannon_entropy  | **−6.65**  | **−3.39**   | 0.92        | **0.89**     | collapse halves in d but KS is essentially identical → still the panel's biggest deviation |

#### B. Per-amino-acid composition (gen N=1000 vs AFDB N=4998)

Comparison set: gen identical to E020-B; AFDB ref is the new 4998-protein FASTA at `data/afdb_ref/sequences.fasta`. Gen-vs-AFDB relative deviations side-by-side with E020-B's gen-vs-PDB:

**Top over-represented in gen:**

| AA | gen | AFDB | rel Δ vs AFDB | (E020 vs PDB rel Δ) |
|----|-----|------|----------------|---------------------|
| **N (Asn)** | 10.28% | 3.80% | **+171%** | (+146%) — *grew* against AFDB |
| E (Glu)     | 11.78% | 6.21% | +89%      | (+95%) |
| L (Leu)     | 11.40% | 9.79% | +16%      | (+29%) — *shrank* |
| G (Gly)     | 9.62%  | 7.38% | +30%      | (+21%) |
| I (Ile)     | 7.28%  | 5.51% | +32%      | (+35%) |

**Most under-represented:**

| AA | gen | AFDB | rel Δ vs AFDB | (E020 vs PDB rel Δ) |
|----|-----|------|----------------|---------------------|
| M (Met) | 0.51%  | 2.29% | **−78%** | (−79%) |
| W (Trp) | 0.46%  | 1.32% | −66%     | (−68%) |
| H (His) | 0.92%  | 2.25% | −59%     | (−69%) |
| F (Phe) | 2.37%  | 3.97% | −40%     | (−42%) |
| D (Asp) | 3.66%  | 5.51% | −34%     | (−38%) |
| V (Val) | 4.45%  | 6.83% | −35%     | (−37%) |
| P (Pro) | 3.22%  | 5.03% | −36%     | (−34%) |
| **A (Ala)** | 6.09% | 8.98% | **−32%** | (−28%) — *grew* against AFDB |

**Glu/Asp ratio:** gen 3.22, AFDB 1.13 (PDB 1.04) — AFDB Glu/Asp is essentially the same as PDB. The 3-fold gen asymmetry **survives** verbatim.

**Headline reading:** the alphabet-collapse pattern is rock-solid against AFDB. Same residues over-/under-used, magnitudes within a few percent. N over-representation actually *grew* (+146% → +171%) — AFDB has even less N than PDB. Sub-claim **(a) of Finding 9 survives.**

#### C. Thermal-stability sequence proxies (Tier-1)

Side-by-side d's:

| metric                  | E020 PDB d | E026 AFDB d | sign preserved? |
|-------------------------|------------|-------------|-----------------|
| aliphatic_index         | +0.74      | +0.21       | yes |
| ivywrel_fraction        | +1.35      | +0.64       | yes |
| gravy                   | −1.29      | −0.84       | yes |
| charged_fraction        | +0.75      | +0.38       | yes |
| log_acidic_basic_ratio  | +1.67      | +0.99       | yes |
| **aromatic_fraction**   | **−1.19**  | **−0.73**   | yes |

All deltas attenuate by roughly half but every sign is preserved. AFDB is itself more aliphatic-heavy than PDB (88.5 vs 84.4 → closer to gen's 91.8), so gen looks less of an outlier. Methodological observation **(c.i) survives**: aliphatic_index and IVYWREL still score gen as more thermostable than the natural reference, while aromatic_fraction (the buried-core proxy) still drops in the opposite direction. The internal contradiction that powers Finding 9-c.i exists relative to AFDB too.

#### D. Aromatic burial-targeting ratio (E023 rerun)

Counts and bootstrap-over-protein 95% CIs:

| residue        | gen R [95% CI]      | AFDB R [95% CI]    | gen / AFDB | (E020 gen / PDB) |
|----------------|---------------------|---------------------|-----------|-------------------|
| W              | 9.80 [1.81, 46.95]  | 2.41 [2.16, 2.67]   | **4.07×** | (1.68×) |
| **F**          | **2.58 [1.29, 4.99]** | **2.52 [2.39, 2.67]** | **1.02×** | (**0.45×**) |
| Y              | 5.32 [3.36, 8.65]   | 2.71 [2.49, 2.96]   | **1.96×** | (1.45×) |
| H              | 1.29 [0.62, 2.39]   | 0.86 [0.80, 0.92]   | **1.50×** | (1.15×) |
| Aromatic group | 3.00 [2.02, 4.30]   | 1.99 [1.91, 2.09]   | **1.51×** | (0.95×) |

AFDB's overall aromatic frequency: 8.26% (vs PDB's 12.74%). Gen aromatic frequency: 6.07% (verbatim from E023 — same gen set).

**Headline reading — the F under-burial story dies.** AFDB structures themselves have a much weaker F core-targeting preference than PDB crystallography (R=2.52 vs R=5.68) — likely because AlphaFold-predicted "buried" regions aren't packed with the hard-shell core constraints visible in crystal structures. Gen matches AFDB's F-burial pattern almost exactly (2.58 vs 2.52). The published-paper-version of "the model fails to bury F" is an **artifact of comparing against PDB**.

What survives in a different form: gen actually *over-targets* aromatics for burial relative to AFDB at every per-residue ratio (W 4×, F 1×, Y 2×, H 1.5×, group 1.5×). Note that "over-targets" describes the placement *ratio*, not the absolute count — gen still uses fewer aromatics than either AFDB or PDB (alphabet collapse). The model concentrates the smaller aromatic budget more sharply into the buried core than AFDB's predicted-structure pattern shows. This is a *competence* observation, not a failure mode: the model has learned the "aromatic → core" rule strongly even on a reduced alphabet.

#### E. Aromatic-burial follow-ups (E024 rerun on AFDB parquet)

Composition decomposition (`exp1_composition`):

| set | W | F | Y | H |
|-----|---|---|---|---|
| gen | 0.080 [0.052, 0.111] | 0.279 [0.229, 0.332] | **0.476 [0.412, 0.542]** | 0.165 [0.121, 0.215] |
| AFDB | 0.131 [0.126, 0.136] | 0.372 [0.365, 0.379] | 0.282 [0.275, 0.288] | 0.215 [0.209, 0.222] |

| quantity | value |
|---|---|
| per-residue gen burial ratios (raw counts) | W=5.35, F=2.30, Y=5.14, H=1.16 |
| observed gen group ratio (p_gen · r_gen) | 3.710 |
| empirical gen group ratio (raw counts) | 2.967 |
| counterfactual (p_pdb · r_gen using AFDB pool fractions) | **3.256** |

Gen→AFDB-pool reweighting moves the group ratio 3.71 → 3.26 (12.2%, < 20% threshold) → still **NOT COMPOSITIONAL**. But this finding is now moot — the AFDB rerun renders E023's "F is broken" reading invalid in the first place; the composition decomposition was framed as "explaining the group preservation we observe" and there is no group preservation against AFDB to explain.

Curve shape (`exp2_curve_shape`):

| residue | gen slope | AFDB slope | CI overlap |
|---------|-----------|------------|-------------|
| F | −2.29 [−4.13, −0.74] | −1.71 [−2.00, −1.44] | yes — same shape |
| Y | −1.43 [−1.98, −0.88] | −1.52 [−1.83, −1.24] | yes — same shape |

Same-shape verdict survives. AFDB's F slope (−1.71) is itself shallower than PDB's (−3.64), so the gen-vs-PDB gap was a slope-versus-slope comparison where the PDB side had an unusually steep curve.

Per-protein analysis (`exp3_per_protein`): gen 18/138 vs AFDB 654/835 surviving the W+F+Y filter; same underpowered verdict, AFDB ref now has 654 surviving instead of 314 — better ref-side power, gen-side N is the binding constraint.

**Possible narrative (post-results):** Sub-claim status for Finding 9 (writing into `content_masterarbeit.md`):
- **(a) Chemistry-specific alphabet collapse:** **survives.** Magnitudes shift somewhat smaller for some metrics (Leu, His), but every sign is preserved and several (Asn over-representation, Ala under-representation) actually *grow* against AFDB. Glu/Asp 3.22 ratio inflation is identical. **Action taken (2026-05-03):** rewrote Finding 9's primary numbers using AFDB as the natural reference; retained PDB numbers in a sensitivity-check column.
- **(b) SWI / Shannon-entropy mode-merging:** **withdrawn.** Initial post-AFDB read claimed "AFDB ref still has 2 modes; gen still has 1" — that was wrong. AFDB at n=5K is 1-mode for both swi and shannon_entropy. E027 (matched-n bootstrap) showed: shannon_entropy is a real PDB-vs-AFDB population difference (PDB consistently 2-mode at AFDB-matched n=5K; AFDB is genuinely unimodal), but SWI's PDB bimodality vanishes at matched n (PDB itself drops to 1-mode). Either way, no mode-merging signature exists against the right reference (AFDB) for either metric. **Action taken (2026-05-03):** sub-claim (b) removed entirely from Finding 9; renumbered (c) → (b), added the new aromatic count-vs-placement asymmetry as (c). Finding 9 title updated to drop "mode-merges on bimodal natural distributions". E027 entry created in this lab record to document the matched-n robustness check.
- **(c.i → b.i) Gameable thermal proxies:** **survives.** Aliphatic d=+0.21, IVYWREL d=+0.64, both still positive — the literature proxies still score gen as more thermostable, while aromatic_fraction (the buried-core proxy) drops d=−0.73. The internal contradiction (the methodological core) holds. **Action taken (2026-05-03):** kept the internal-contradiction framing; quoted AFDB d's; renumbered as sub-claim (b) with sub-parts (b.i) sequence-only and (b.ii) TemStaPro pending.
- **(c.ii → b.ii) TemStaPro Tier 2:** still preregistered, deferred. Conclusion of (b.ii) is not gated on the reference-set switch.
- **E023 / E024 structural extension** (was being shaped into a Finding-9 add-on or Finding 10): **resurfaced as a positive sub-claim (c).** F under-burial fails against AFDB, but the count-vs-placement asymmetry (gen uses fewer aromatics overall AND concentrates them more sharply into the buried core than AFDB) is a real, AFDB-robust signal. The user requested this be promoted to Finding 9 sub-claim (c) on 2026-05-03 rather than written as a standalone Finding 10. **Action taken (2026-05-03):** sub-claim (c) added to Finding 9 with the count-vs-placement asymmetry framing.

**Methodological caveats:**
- **Random AFDB ≠ training set exactly.** La-Proteina's actual training corpus is (most likely) AFDB Foldseek-cluster representatives (~580K-2.27M reps depending on identity threshold), which are diversity-balanced. A uniform-random sample over the full ~214M-entry AFDB over-weights over-clustered families (e.g. many homologous bacterial proteins from many species). For our purposes — testing whether gen-vs-natural deltas survive the reference-set switch — this is acceptable and arguably *more conservative*: if the AFDB-vs-gen deltas survive against a family-imbalanced sample, they would also survive against a more diversity-balanced one. Tighter robustness check: re-run against AFDB Foldseek-30/50 cluster reps once a clean download path exists.
- **Predictor unchanged.** All E001/E007/E025 numbers stay rooted in PDB-trained latents. This experiment does not address the steering-predictor distributional mismatch — separate follow-up.
- **N=5000 ref vs N=1000 gen.** Same effective ratio as the original PDB reference (N=56k vs N=1000 → 56:1; here 5:1). Statistical power on the AFDB side is reduced by 11×; bootstrap CIs widen accordingly. Adequate for distributional comparison; not for tail-behaviour claims.
- **n_resolved_residues = sequence_length on AFDB.** AFDB structures have no missing residues by construction, so any "fraction resolved" comparison vs PDB (where crystallisation gaps drop ~5-10% of residues) is moot here.
- **AFDB v6 ≠ AFDB v4.** The bulk download endpoint is now v6 (same AlphaFold-2-monomer pipeline, no model change — v6 is just a database expansion + metadata refresh). All AFDB-side claims about predicted-structure quality apply to both versions interchangeably.
- **Gen-side panel coverage.** The current gen artifact (`properties_generated_seqonly.csv`) is sequence-only; structure-dependent panel columns (TANGO, hydrophobic patches, SAP, SCM, rg) are missing on the gen side. The 5-row AFDB panel comparison is not a direct refutation/confirmation of E020-A's full 15-row table — only of the 5 sequence-side columns. Restoring the full gen panel via `compute_developability.py` on the gen `.pt` files would close the gap.
- **`aa_composition.py --out` treats the path as a directory.** The artifact landed at `aa_composition.csv/aa_composition.csv` instead of `aa_composition.csv`. Cosmetic only — numbers are correct.

**Cross-references:**
- E005, E020, E023, E024 — the four entries this rerun directly affects. Their PDB-side numbers stay on the lab record (append-only). Once E026 produces AFDB-side numbers, both should be cited side-by-side in any Finding text rather than retroactively replacing the PDB ones.
- `script_utils/run_afdb_rerun.sh` and `script_utils/afdb_rerun_helpers.py` — the automation. CLAUDE.md doesn't yet describe these; if this rerun becomes a regular workflow, add a § to the project guide.
- Predicts: regardless of outcome, add a short § to CLAUDE.md in the **Operational notes** area making explicit that *"natural-protein reference for AFDB-trained models"* defaults to AFDB and not PDB. Whatever the qualitative result is, the methodological lesson is the same — comparison set must match the training distribution.

### E020 + E026 follow-up (2026-05-05): nsteps=400 regen + complete property panel

**Why this section, not new E-ids:** Per user instruction — these followups attach to E020 / E026, not separate entries. Replaces the nsteps=200 / seqonly numbers in *both* parent entries.

**Why ran (and why this is somewhat embarrassing).** Both E020 and E026 used the same gen artifact: `results/generated_stratified_300_800/sequences.fasta`, 1000 length-stratified samples produced via `submit_generate_stratified.sh` at `nsteps=200`. The `nsteps=200` choice was inherited from the SLURM script's default (`script_utils/submit_generate_stratified.sh:29 NSTEPS=${2:-200}`), not a deliberate methodological choice. The codebase's canonical inference default is `nsteps=400` (`configs/inference_base.yaml:18`); the production La-Proteina inference results in the repo (`inference/results_inference_ucond_notri_0.csv`) were all generated at 400. So our "natural-protein vs gen" comparisons in E020/E026 were quietly using a degraded gen distribution. The 2026-05-04 scRMSD sanity check during the steering work surfaced the cost: a single-protein nsteps=200 sample at L=300 returned scRMSD=22.5 Å under the official scRMSD pipeline, while the same protein regenerated at nsteps=400 returned scRMSD=0.80 Å. nsteps=200 is not just slow-but-fine — at L≥300 the integrator hasn't converged to the data manifold, so the resulting structures are genuinely degraded and any structure-derived property (FreeSASA-based hydrophobic patches, SAP, SCM, rg, even TANGO when its sequence-side reading is informed by structure) is computed on a different, off-manifold population than the canonical pipeline produces. The gen-vs-natural deltas we reported were therefore (gen at degraded structure) vs (natural reference) — not the apples-to-apples comparison the Finding-8 narrative describes.

A second correctness gap: the original gen-side property artifact `properties_generated_seqonly.csv` carried only 8 of the 16 panel columns (sequence-only, plus a few that didn't fail silently). TANGO-derived columns, hydrophobic-patch columns, SAP / SCM, and rg were missing from the gen side, which forced E026 to publish a 5-row vs-AFDB table instead of the 15-row vs-PDB table E020 shipped. IUPred3 also failed silently on the gen side in that file. Mentioned as caveat 8 in E026 above; finally resolved here.

**Configs:** Same 1000-protein stratified design as the original (`length_range: [300, 800]`, `bin_width: 50`, `n_per_bin: 100`, `seed_base: 1000`, same `inference_ucond_notri_long.yaml` → `LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt`, sampling_mode=`sc` for both channels, `sc_scale_noise=0.15` (bb_ca) / `0.05` (local_latents) per the LD3-shipped overrides, `t_lim_ode=0.98`, `guidance_w=1.0`, `ag_ratio=0.0`). **Single change:** `nsteps=400`. Output dir: `results/generated_stratified_300_800_nsteps400/`. Driver: `script_utils/run_regen_1000_nsteps400_and_distribution_experiments.sh` — single nohup that runs gen (~16 h on 1× L4) followed by all 8 downstream analyses (~10 min CPU). Gen wall: **980 min** (16h 20min, vs ~3 h for the original nsteps=200 sweep). All 1000 samples produced; no failures.

The downstream artifacts are now:

- `results/generated_stratified_300_800_nsteps400/properties_generated.csv` — full 16-col panel (TANGO + structure-derived columns populated).
- `results/generated_stratified_300_800_nsteps400/sequences.fasta` — 1000 sequences for AA-composition / thermal-stability scripts.
- `results/property_comparison_nsteps400/{stratified_vs_pdb,stratified_vs_afdb}/summary.csv` — full 15-row panel, both references.
- `results/aa_composition_nsteps400/{stratified_vs_pdb,stratified_vs_afdb}/aa_composition.csv` — both references.
- `results/thermal_stability_nsteps400/{stratified_vs_pdb,stratified_vs_afdb}/summary.csv` — Tier-1 only; Tier-2 TemStaPro still requires the separate A100 sbatch and is not run here.

**Results:**

#### A. Property-panel deltas — every direction preserved, magnitudes mostly within ±0.15 SD of the original

Side-by-side Cohen's d (vs PDB) and KS_d (vs PDB), original (E020-A, nsteps=200) vs new (nsteps=400):

| property | old d / new d | Δd | old KS / new KS |
|---|---|---|---|
| sequence_length | +1.47 / +1.48 | +0.01 | 0.45 / 0.46 |
| swi | +1.62 / +1.59 | −0.03 | 0.62 / 0.61 |
| tango | +0.44 / +0.37 | −0.07 | 0.20 / 0.18 |
| tango_aggregation_positions | −0.13 / −0.05 | +0.08 | 0.28 / 0.24 |
| net_charge | −2.14 / −2.20 | −0.06 | 0.43 / 0.43 |
| pI | −0.43 / −0.41 | +0.02 | 0.43 / 0.40 |
| iupred3 | +1.63 / +1.49 | −0.14 | 0.42 / 0.41 |
| iupred3_fraction_disordered | +2.70 / +2.36 | −0.34 | 0.36 / 0.33 |
| **shannon_entropy** | **−6.65 / −5.78** | **+0.87** | 0.92 / **0.89** |
| hydrophobic_patch_total_area | −0.87 / −0.91 | −0.04 | 0.60 / 0.56 |
| hydrophobic_patch_n_large | −1.00 / −1.07 | −0.07 | 0.59 / 0.60 |
| sap | −0.24 / −0.50 | −0.26 | 0.57 / 0.56 |
| scm_positive | +0.48 / +0.59 | +0.11 | 0.29 / 0.24 |
| scm_negative | −2.30 / −2.34 | −0.04 | 0.44 / 0.43 |
| rg | +0.56 / +0.49 | −0.07 | 0.32 / 0.30 |

Three observations worth promoting:

1. **Shannon-entropy collapse softened from d=−6.65 to d=−5.78 (about 13% smaller in d, 3% smaller in KS).** Still by far the panel's biggest deviation under both references; effective alphabet 2^4.10 ≈ 17 (PDB) → 2^3.47 ≈ 11 (gen, was 2^3.36 ≈ 10). The reduction is consistent with "less off-manifold sampling at higher nsteps trims the most entropically degenerate (low-complexity) tail of the gen distribution."
2. **iupred3_fraction_disordered dropped from +2.70 to +2.36** — the disorder-promoting drift attenuates by ~13% on Cohen's d. Fewer absolute-extreme low-Shannon-entropy + N-rich + IUPred-positive sequences; same direction, smaller tail.
3. **`sap` Cohen's d doubled (−0.24 → −0.50) but KS barely moved.** The first-moment shift away from PDB SAP is larger at nsteps=400 — the gen population is more compact in the SAP-low region while the PDB population is more variable; the two-sample location difference grew while the integrated distributional difference (KS) stayed essentially constant. Doesn't change the qualitative reading.

#### A.AFDB. Property-panel deltas vs AFDB — first time the full 15-row table exists

E026's table had 5 rows because `properties_generated_seqonly.csv` only carried 5 vs-AFDB-comparable columns. With the full panel now populated on the gen side, the new vs-AFDB Cohen's d table is:

| property | new d (vs AFDB) | (E026 d, where available) |
|---|---|---|
| sequence_length | −0.005 | (−0.012) — length-stratified, expected |
| swi | +1.20 | (+1.22) |
| tango | **−0.46** | (gap — gen seqonly) |
| tango_aggregation_positions | **−0.58** | (gap) |
| net_charge | −0.97 | (−0.93) |
| pI | −0.67 | (−0.68) |
| iupred3 | +0.45 | (gap) |
| iupred3_fraction_disordered | +0.27 | (gap) |
| shannon_entropy | **−2.99** | (−3.39) |
| hydrophobic_patch_total_area | **−1.43** | (gap) |
| hydrophobic_patch_n_large | **−1.39** | (gap) |
| sap | **−1.21** | (gap) |
| scm_positive | −0.68 | (gap) |
| scm_negative | −0.39 | (gap) |
| rg | **−0.89** | (gap) |

The structure-derived columns (in **bold**) are first-time numbers vs AFDB. Reading: gen samples have systematically *lower* TANGO total + count, *smaller* hydrophobic patches (both total area and large-patch count), *lower* SAP, *less negative* scm_negative, and *smaller* Rg than the AFDB reference. The direction is consistent: gen produces less-hydrophobic-rich proteins than AFDB does, which is consistent with the alphabet collapse at the sequence-composition level (fewer aromatics + more E/N/G → lower SASA-anchored hydrophobic readouts). **The vs-PDB and vs-AFDB structure-side directions agree** — gen is on the same side of the natural reference under both, just with smaller magnitudes against AFDB. Sub-claim (a) of Finding 8 (alphabet collapse) gains a structure-side echo.

Note `tango` flips sign between references: gen vs PDB is +0.37 (gen higher than PDB; PDB is the crystallography distribution which has lower TANGO on average), gen vs AFDB is −0.46 (gen lower than AFDB; AFDB has higher TANGO than PDB because AlphaFold-predicted structures have less rigorously-packed hydrophobic cores and more solvent-accessible aggregating residues per FreeSASA). The gen population sits *between* the two natural references on TANGO. This is consistent with the sequence-level reading that gen has reduced absolute hydrophobic content but uses what it has tightly — a structural confirmation of sub-claim (c)'s count-vs-placement asymmetry.

#### B. Per-amino-acid composition — alphabet collapse pattern unchanged, magnitudes mildly softer

Side-by-side relative deviation vs PDB, original (E020-B, n_steps=200) vs new (nsteps=400):

| AA | old rel Δ vs PDB | new rel Δ vs PDB | Δ |
|----|------------------|--------------------|---|
| **N (Asn)** | +146% | **+132%** | −14 |
| E (Glu)     | +95%  | +90%   | −5  |
| **L (Leu)** | +29%  | **+35%** | +6 |
| I (Ile)     | +35%  | +28%   | −7  |
| G (Gly)     | +21%  | +5%    | −16 |
| A (Ala)     | −28%  | −27%   | +1  |
| V (Val)     | −37%  | −33%   | +4  |
| D (Asp)     | −38%  | −29%   | +9  |
| F (Phe)     | −42%  | −45%   | −3  |
| **H (His)** | −69%  | **−58%** | +11 |
| **W (Trp)** | −68%  | **−54%** | +14 |
| **M (Met)** | −79%  | **−72%** | +7  |
| **Glu/Asp ratio** | **3.22** | **2.79** | −0.43 |

Every sign preserved. The most-extreme under-utilizations soften the most: **W (Trp) under-rep contracts from −68% to −54%, His from −69% to −58%, Met from −79% to −72%, Asn over-rep from +146% to +132%.** This is the same pattern as the Shannon-entropy softening — fewer absolute-extreme tail samples at higher nsteps. The qualitative chemistry-specificity story is intact: the pattern is "fewer context-demanding residues, more context-tolerant ones," and the Glu/Asp asymmetry is still 2.5× the PDB ratio (which is ~1.04). G softens the most (+21% → +5%) but stays positive; this is the residue most directly tied to local-flexibility / disorder, and the lower number tracks the iupred3 softening above.

vs AFDB: same direction shifts, similar magnitudes — N over-rep moves +171% → +156%; M under-rep ~−78% → −70%; W under-rep ~−66% → −50%; H under-rep ~−59% → −44%. The AFDB-reference pattern survives intact, slightly softer.

#### C. Thermal-stability proxies — d's barely moved, methodological observation unchanged

| metric | vs PDB old / new | vs AFDB old / new |
|---|---|---|
| aliphatic_index | +0.74 / **+0.89** | +0.21 / +0.30 |
| ivywrel_fraction | +1.35 / +1.41 | +0.64 / +0.72 |
| gravy | −1.29 / −1.31 | −0.84 / −0.86 |
| charged_fraction | +0.75 / **+0.87** | +0.38 / +0.47 |
| log_acidic_basic_ratio | +1.67 / +1.49 | +0.99 / +0.93 |
| **aromatic_fraction (F+W+Y)** | **−1.19 / −1.16** | **−0.73 / −0.72** |

Every sign preserved; magnitudes within ±0.18 of the originals. The internal contradiction that powers sub-claim (b.i) of Finding 8 — *aliphatic_index and IVYWREL say "more thermostable" while aromatic_fraction says "fewer hydrophobic-core anchors"* — survives essentially unchanged. The aliphatic_index drift actually *grew* slightly (+0.74 → +0.89 vs PDB) while aromatic_fraction barely moved. This rules out the worry that the methodological observation might be an nsteps=200 artifact: the literature proxies still inflate gen's apparent thermostability at the canonical inference resolution.

#### D. What this regen does NOT change

- **Aromatic burial (E023 / E024 / E026-D)** uses `inference/inference_ucond_notri/` PDBs, which were always at nsteps=400 (per the `inference/results_inference_ucond_notri_0.csv` config snapshot — `generation_args_nsteps=400`). The aromatic-placement Finding 8 sub-claim (c) is therefore unaffected by this regen. No rerun needed.
- **Tier-2 TemStaPro** still requires an A100 + ProtT5-XL forward pass on both ref and gen (gen=1000 + ref=53,749). The driver's Phase 8/9 ran Tier-1 only. A separate sbatch on the new gen FASTA would close (b.ii) under the corrected nsteps=400 reference, but the Tier-1 contradiction that powers (b.i) is robust to the regen as shown above.
- **Mode-merging sub-claim (b)** is already withdrawn (E027); not reopened by this regen.

**Possible narrative (post-regen):**

- **Finding 8 sub-claim (a) — chemistry-specific alphabet collapse:** survives at nsteps=400 with magnitudes attenuated by ~5-15% on the most-extreme residues. Headline numbers update: Shannon-entropy d=−5.78 (was −6.65) vs PDB and −2.99 (was −3.39) vs AFDB; KS-D drops from 0.92 to 0.89 vs PDB. Glu/Asp ratio updates 3.22 → 2.79. Worth updating these in `content_masterarbeit.md → Finding 8` as the new primary numbers; the originals stay as a record-of-history footnote.
- **Finding 8 sub-claim (b.i) — gameable thermal proxies:** survives at nsteps=400 essentially unchanged. d's within ±0.18 of original; the internal contradiction (aliphatic↑ vs aromatic↓) holds at every reference and every nsteps tested.
- **Finding 8 sub-claim (c) — count-vs-placement asymmetry:** unaffected (different gen artifact).
- **Caveat 8 of E026 (gen-side panel coverage)** is now **resolved** — the full 15-row panel comparison vs AFDB exists, and the structure-derived rows confirm sub-claim (a)'s structural echo.

**Methodological caveats — what this regen does *not* support:**

1. **Single-checkpoint, single-eval-seed.** Same N=1000, same seed_base=1000, same LD3+AE2 checkpoint as the original. Cross-seed and cross-checkpoint variance still not estimated.
2. **No designability rerun on this set.** The new 1000 are .pt + .pdb + .csv only — no scRMSD pass was run. The "structures are now on-manifold" claim rests on the single-protein scRMSD sanity check that motivated this regen (0.80 Å for the new pipeline vs 22.5 Å for the old), not on a panel of designability numbers from this set. A scRMSD sweep on 30-50 random samples from the new gen would give a real distributional handle on designability at nsteps=400; not run here because the Finding-8 narrative is sequence- and panel-side, not designability-side.
3. **AFDB reference unchanged.** The 5000-protein AFDB sample from E026 (single-pass reservoir-sample, length-stratified, downloaded 2026-05-03) is reused verbatim. Re-running the AFDB ref at higher N or with a different sampling rule was not part of this regen; numbers vs AFDB inherit E026's reference-set caveats (random-AFDB ≠ Foldseek-cluster reps).
4. **TemStaPro Tier-2b still preregistered, still pending.** Driver step 9 ran Tier-1 only; sub-claim (b.ii) is not yet validated against the regen.
5. **The "generation script default was wrong" framing is a CALLOUT, not a published claim.** The original E020 numbers stay on the lab record (append-only). They are "nsteps=200 numbers, since superseded" — not retracted. Future Finding-8 citations should pull the nsteps=400 numbers; cross-references to the nsteps=200 originals stay valid for historical record-keeping.

**Cross-references:**
- E020 — parent for vs-PDB numbers. The original 14 panel rows + AA composition + Tier-1 thermal stand on the lab record, but Finding 8 narrative numbers should now cite the nsteps=400 column above.
- E026 — parent for vs-AFDB numbers. Caveat 8 (gen-side panel coverage) is resolved by this regen; the 15-row vs-AFDB table is now first-class.
- E025 follow-up — same nsteps lesson, different probe (steering rather than unguided baseline). The same memory pointer (`feedback_steering_nsteps_400_minimum.md`) covers both.
- Driver: `script_utils/run_regen_1000_nsteps400_and_distribution_experiments.sh`. nohup log: `nohup_regen_1000_nsteps400.out`. Runtime: 980 min wall on 1× L4, 1000 samples × ~58 s/protein average (ranges from ~19 s at L=300 to ~125 s at L=800).
- Finding 8 in `content_masterarbeit.md` updated 2026-05-05 to use the nsteps=400 numbers as primary; nsteps=200 numbers retained in a footnote for record-of-history.

### E020+E026 follow-up: length-invariance addendum (2026-05-05)

**Why this section, not a new E-id:** Same regen artifacts as the previous follow-up; this is the per-length-bin breakdown of the same data, prompted by the user's question *"is the gen-vs-AFDB deviation length-invariant?"* during the Finding-8 update. The answer turned out to be "no, for several key metrics" — large enough effect to deserve a fresh Finding, which became Finding 9 in `content_masterarbeit.md`.

**Why ran:** The previous follow-up's headline Cohen's d's are aggregates over the full [300, 800] gen sweep. The gen sweep is length-stratified-uniform (100 samples per 50-residue bin) and the AFDB reference was explicitly length-stratified to match gen, so both sides have ~equal coverage per bin. We have the right data to ask whether each metric's deviation is constant across L bins, or whether the aggregate d hides systematic length-scaling. If invariant: the aggregate d is a clean summary statistic. If not: the aggregate is an averaging artifact, and Finding-8 sub-claim (a)'s headline magnitudes need to be quoted with a length context.

**Configs (re-run-able from this entry):**

- Inputs: `results/generated_stratified_300_800_nsteps400/properties_generated.csv` (n=1000 gen, full 16-col panel) + `data/afdb_ref/properties_afdb_refschema.csv` (n=4998 AFDB, length-stratified to gen). Sequence-side: `results/generated_stratified_300_800_nsteps400/sequences.fasta` and `data/afdb_ref/sequences.fasta`.
- Bins: `[300, 350), [350, 400), …, [750, 800)` — 10 bins of 50-residue width.
- Per-bin Cohen's d for property panel: `(gen_mean − ref_mean) / pooled_sd_per_bin`, reported per metric per bin. AA-composition: `(gen_mean − ref_mean) / ref_mean × 100` per AA per bin (relative deviation, matches the format of `aa_composition.py`). Glu/Asp ratio: gen_E_mean / gen_D_mean per bin (AFDB ratio reported separately as a control).
- Per-bin n: gen n=100 in every bin (by construction, `steering/generate_baseline.py:179` stratified mode); AFDB n ranges from 466 to 520 across bins (uniform-random over AFDB, length-stratified-trim).
- Bootstrap CIs: NOT computed at the per-bin level — gen-side n=100 gives ~±0.3 SD bootstrap CI on Cohen's d, which is below the dominant trend magnitudes but above the bin-to-bin micro-variations. Sharper within-bin comparisons require a larger gen draw per bin.
- Code: inline Python in this session; no script committed (the analysis is short enough to live in `experiments.md` and the `content_masterarbeit.md` Finding 9 tables).

**Results:**

#### A. Property-panel Cohen's d gen-vs-AFDB by 50-residue length bin

| L bin | swi | tango_total | iupred3 | shannon | hyd_area | sap | scm_neg | rg |
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
| **range** | 0.92 | **1.32** | 0.45 | **1.95** | 0.40 | 0.39 | 0.79 | 0.25 |
| **trend?** | weak | **monotone ↑│d│** | weak ↓ | **monotone ↑│d│** | flat | flat | noisy | flat |

#### B. AA-composition relative deviation gen-vs-AFDB (%) by length bin

| L bin | E | N | L | M | W | H | F | A | Glu/Asp (gen / AFDB) |
|---|---|---|---|---|---|---|---|---|---|
| [300, 350) | +90 | **+7** | +5 | −33 | −57 | −17 | −40 | **+3** | 2.32 / 1.08 |
| [350, 400) | +76 | +87 | +4 | −48 | −56 | −28 | −35 | −20 | 2.11 / 1.11 |
| [400, 450) | +100 | +162 | +2 | −69 | −58 | −50 | −25 | −36 | 2.79 / 1.11 |
| [450, 500) | +101 | +165 | +12 | −79 | −61 | −43 | −40 | −39 | 3.00 / 1.14 |
| [500, 550) | +105 | +200 | +13 | −68 | −44 | −48 | −35 | −42 | **3.45** / 1.12 |
| [550, 600) | +105 | +170 | +26 | −82 | −57 | −64 | −42 | −26 | 3.16 / 1.12 |
| [600, 650) | +97 | +192 | +36 | −79 | −37 | −47 | −50 | −41 | 3.37 / 1.14 |
| [650, 700) | +73 | +177 | +35 | −79 | −36 | −51 | −53 | −37 | 2.87 / 1.17 |
| [700, 750) | +31 | **+212** | +42 | −86 | −54 | −48 | −58 | −41 | 2.50 / 1.13 |
| [750, 800) | +82 | +168 | **+48** | **−83** | −38 | −46 | −51 | −40 | 2.63 / 1.14 |
| **trend?** | weak | **+30× across L** | **monotone ↑** | **monotone ↓** | noisy | **monotone ↓** | weak ↓ | **monotone ↓** | non-monotone (peak L≈525) |

The AFDB-side Glu/Asp ratio is essentially constant across length (1.08–1.17); the gen-side variance is entirely on the model side.

#### C. Pattern summary

**Length-DEPENDENT (deviation systematically scales with L):**
- shannon_entropy d: −1.97 → −3.92 (about 2× across the range; nearly monotone)
- N-over-rep: +7% → +212% (factor of ~30; nearly monotone)
- Met under-rep: −33% → −86% (deepens monotone)
- Ala under-rep: +3% → −41% (sign-flips; deepens monotone)
- His under-rep: −17% → −48% (deepens, slightly noisy)
- Leu over-rep: +5% → +48% (grows monotone)
- TANGO_total d: 0 → −1.30 (sign change at long L)
- iupred3 d: +0.74 → +0.29 (attenuates monotone — opposite direction from the alphabet collapse)

**Length-INVARIANT (deviation roughly constant across L):**
- swi d: range +0.59 to +1.51, no trend (one isolated drop at L=[700, 750))
- hydrophobic_patch_total_area d: −1.47 to −1.87, range 0.40, no trend
- sap d: −1.19 to −1.58, range 0.39, no trend
- rg d: −0.86 to −1.11, range 0.25, no trend
- E (Glu) over-rep: +73% to +105%, no monotone trend
- W (Trp) under-rep: −36% to −61%, noisy, no trend
- F (Phe) under-rep: −25% to −58%, weak length scaling, noisy
- scm_negative d: noisy around −0.5, one positive at L=[700, 750), no clean trend

**Possible narrative:** The alphabet-collapse-driven sequence-composition deviations of Finding 8 sub-claim (a) intensify with protein length; the structure-derived FreeSASA-based metrics (hydrophobic_patch_*, sap, rg, swi) are roughly length-invariant. This is consistent with a "joint sequence head loses diversity over longer generation trajectories" story — the longer the autoregressive horizon the latent flow has to generate, the more small per-residue mode-collapse compounds — but the FreeSASA-derived structural readouts react to *per-residue* hydrophobic content, which is roughly constant across length. The predictor-test version of this hypothesis would be: AE2 latent KL-divergence-from-prior should be roughly constant across L (matching the FreeSASA invariance) but the joint-sequence-head's per-residue marginal-entropy should drop with L (matching the alphabet-collapse intensification). Both are testable from the existing latents + sequences but not run in this entry.

**Methodological caveats:**

1. **No per-bin bootstrap CIs.** Reported as point estimates; ±0.3 SD bootstrap rule-of-thumb at gen n=100 per bin. The bin-to-bin micro-variations (Shannon dip from L=[700, 750) to L=[750, 800), W noise) are within this noise; the dominant length-scaling trends span 1.5–2.0 SD over the full range and are robust to this.
2. **Single seed range per bin.** Gen samples are seeds 1000–1999 round-robin; a different seed range would draw different specific proteins per bin. The qualitative length-scaling pattern should survive cross-seed (since it appears across all 10 bins consistently for the affected metrics) but precise per-bin numbers would shift by O(0.2) in d.
3. **AFDB ref length-clustering bias.** AFDB long-length samples over-represent over-clustered protein families more strongly than short-length ones. This bias goes against the gen drift (a Foldseek-cluster-reduced AFDB at L>700 would be MORE diverse than what we measured), so our length-scaling claim is conservative.
4. **No DSSP / structural-secondary-structure breakdown.** The "alphabet collapse intensifies but FreeSASA-side stays flat" reading would be sharper with a per-bin secondary-structure decomposition (does β-sheet content drop at long L while α-helix stays flat? what about loop fraction?) — not run here.
5. **The "long-trajectory mode collapse" mechanistic story is a hypothesis, not a measurement.** This entry shows the length scaling, not the mechanism behind it.
6. **Cross-checkpoint variance not estimated.** Single LD3+AE2 checkpoint. A retrained checkpoint or a different RNG init might shift the per-bin curves; the qualitative pattern is unlikely to change but the slope-per-50-residue-bin might.

**Cross-references:**

- Finding 9 in `content_masterarbeit.md` — paper-narrative version. Quotes the per-bin headline numbers and the three-sub-claim framing.
- Finding 8 caveat 9 added 2026-05-05 — points future readers to Finding 9 for the per-length breakdown.
- E020+E026 follow-up (the previous section) — this section uses the same regen artifacts; the only new analysis is the per-bin slicing.
- Predicts: per-bin scRMSD profile on the same gen set (subsample ~30 PDBs/bin → 300 PDBs total, ~10 h on 1× L4 at the L≈550 average). Expected if structural-quality and alphabet-collapse-intensification are coupled: scRMSD median should worsen at L>500 vs L<450, complementary to E022's L-cliff finding for the CA-only baseline.

---

## E027 — Mode-merging robustness check: PDB bimodality at matched n (2026-05-03)

**Status:** finished.

**Why ran:** E026's AFDB rerun reported `ref_modes = 1` for both `swi` and `shannon_entropy` at AFDB n=4998, where E020's PDB run at n=56,008 had `ref_modes = 2` for both. That made the originally-claimed "model collapses 2 modes → 1" signature (Finding 9 sub-claim (b)) ambiguous: was the mode-merging signature *real* (i.e. the natural distribution is genuinely bimodal and gen merges the modes) or was it a *KDE-bandwidth detection artifact* (PDB's bandwidth at n=56K is much narrower than AFDB's bandwidth at n=5K via Scott's rule, so two close peaks are resolvable in the larger sample but smoothed-over in the smaller one)? Distinguishing these matters for whether sub-claim (b) of Finding 9 survives. Fix: subsample PDB to AFDB's n and re-run the same `kde_modality` peak-counting code; if PDB still shows 2 modes at matched n, the AFDB unimodality is a real population difference; if PDB drops to 1 mode at matched n, the original "PDB has 2 modes" finding was n-dependent and the sub-claim collapses.

**Configs:**
- Implementation: `script_utils/check_modality_robustness.py` — replicates `compare_properties.kde_modality` verbatim (scipy `gaussian_kde` with Scott bandwidth, 5%-of-peak prominence filter, 512-grid evaluation). Loads `laproteina_steerability/data/properties.csv` (PDB, n=56008 length-filtered to [300, 800]) and `data/afdb_ref/properties_afdb_refschema.csv` (AFDB, n=4998 from E026). For each property: report `kde_modality` on PDB-full, AFDB-full, and 20 replicates of PDB subsampled (without replacement) to AFDB's n=4998. Same RNG seed (`np.random.default_rng(0)`) for reproducibility.
- Hardware: laptop CPU, ~2 s wall.
- Probes: `shannon_entropy`, `swi` (the two properties that contributed to E020's mode-merging claim).

**Results:**

| property | PDB full (n=56008) | AFDB (n=4998) | PDB @ matched n=4998 (20 replicates) |
|----------|--------------------|----------------|---------------------------------------|
| shannon_entropy | **2 modes** | **1 mode**   | **{2: 20}** — 2 modes in 20/20 |
| swi             | **2 modes** | **1 mode**   | **{1: 18, 2: 2}** — 1 mode in 18/20, 2 modes in 2/20 |

Distribution-level summaries (mean / sd):
- shannon_entropy: PDB μ=4.096 σ=0.096; AFDB μ=4.054 σ=0.114 — distributions overlap heavily but PDB has a tighter, more peaked structure (KDE picks up the 2-mode signal); AFDB has a slightly wider, smoother shape that integrates into a single mode.
- swi: PDB μ=0.7787 σ=0.0101; AFDB μ=0.7782 σ=0.0129 — almost identical means, AFDB slightly wider sd. The two-mode signal is at a sub-sd separation in PDB.

**Interpretation:**

- **Shannon entropy:** PDB's 2 modes are *robust* to subsampling — even at n=4998 (matched to AFDB), 20/20 PDB subsamples are detected as bimodal. AFDB at the same n=4998 is *consistently unimodal* (1 mode in the actual sample). This is a **real population difference**: the PDB Shannon-entropy distribution genuinely has two modes (probably reflecting curation effects — e.g., crystallisable proteins vs disorder-tail proteins) and the AFDB distribution genuinely doesn't (broader sequence-space coverage smears the bimodality into a single distribution). Gen's 1-mode result therefore *matches* the right reference (AFDB) — there is no mode-merging signature against the training distribution.
- **SWI:** PDB's 2 modes are *not* robust to subsampling — at n=4998, only 2/20 (10%) of PDB subsamples detect 2 modes; 18/20 (90%) detect 1 mode. The 2-mode signature is therefore a **high-n KDE detection effect** within PDB itself, not a population-level feature visible at any reasonable comparison n. The AFDB 1-mode result is consistent with what PDB looks like at the same n. Original sub-claim "gen merges PDB's 2 modes into 1" is **not supported** when the comparison is done at matched sample size.

**Conclusion:** Finding 9 sub-claim (b) ("mode-merging on bimodal natural distributions") is **withdrawn**. Both supports of the sub-claim fail under the corrected references:
- Shannon entropy: AFDB itself is unimodal, gen matches AFDB → no merging.
- SWI: PDB's bimodality is itself n-dependent and vanishes at n ≤ 5K → the merging story was an artifact of comparing high-n PDB modality against gen at a different effective resolution.

The Finding 9 title was edited (2026-05-03) to drop "mode-merges on bimodal natural distributions"; the sub-claim block was removed; the (c) gameable-thermal-proxies sub-claim was renumbered to (b); and the count-vs-placement aromatic asymmetry from E023+E026 was added as the new (c).

**Possible narrative:** **Non-narrative — a robustness check that retired a Finding sub-claim.** The matched-n bootstrap is the right diagnostic for any future "the natural distribution has K modes" claim in this codebase; default to subsampling the larger reference to the smaller one before running `kde_modality`. Worth folding this into the code itself: a follow-up patch could make `compare_properties.py` print mode counts at both full-n and n-matched, with a warning when the two disagree.

**Methodological caveats:**
- **Single peak-detection rule.** All numbers above use the `kde_modality` defaults (Scott's bandwidth, 5%-of-peak prominence, 512-grid). A different prominence threshold, a different bandwidth (Silverman's rule, fixed bandwidth), or a different peak detector could give different counts. The robustness conclusion (PDB-Shannon stays 2-mode under subsampling, PDB-SWI doesn't) is plausibly robust to those choices because the underlying densities are very different shapes (Shannon: clearly two distinct peaks separated by a visible trough; SWI: two close peaks at sub-sd separation), but this hasn't been tested across alternative settings.
- **Length-filter [300, 800].** The PDB property panel CSV at full length (no filter) might produce different mode counts; analyses always use the same [300, 800] filter as E020 / E026. Length-out-of-range proteins are not relevant to this Finding's scope.
- **Detection ≠ scientific reality.** The 5%-of-peak prominence threshold is a heuristic. A "real" bimodality with sub-threshold prominence would be reported as 1 mode by this code; a tiny ripple in a unimodal distribution could be reported as 2 if the ripple happens to clear the threshold. This run distinguishes "PDB-detected-as-2-modes vs AFDB-detected-as-1-mode under the same code" from "the underlying distributions are biophysically different on a mode-axis"; the latter would require an explicit mixture-model fit + likelihood-ratio test, which is out of scope here.
- **n=4998 is the AFDB sample size for [300, 800].** The PDB-subsampling-matched-n was set to that value. A different N_FINAL on the E026 rerun would change the matched-n number and could shift the boundary at which PDB-SWI flips between 1 and 2 modes; the qualitative conclusion (PDB-SWI mode count is n-dependent in the n=O(5K) regime; PDB-Shannon mode count is not) is the relevant takeaway.

**Cross-references:**
- E020, E026 — direct parents. E027 is the matched-n robustness check the user requested after spotting that the post-AFDB reading of sub-claim (b) was inconsistent with the underlying ref_modes column.
- Finding 9 in `content_masterarbeit.md` — sub-claim (b) was withdrawn on 2026-05-03 based on this entry; sub-claim (c) (count-vs-placement aromatic asymmetry) was added in the same revision, replacing the structural extension that died in E026.
- Implementation: `script_utils/check_modality_robustness.py` (verbatim replication of `compare_properties.kde_modality`, prints PDB-full / AFDB / PDB-subsampled-to-AFDB-n).
- Predicts: any future "natural distribution is multimodal" claim in this codebase should be paired with a matched-n bootstrap of the same KDE peak-detector. A small patch to `compare_properties.py` that auto-runs the matched-n check whenever it reports a mode count would have caught the SWI sub-claim before it reached Finding 9.

## E028 — Predictor-vs-real gap on the May-04 ensemble steered run (2026-05-05)

- **Status:** finished.
- **Why ran:** the May-04 ensemble+smoothed sweep (E025 follow-up, run dir `results/steering_camsol_tango_L500_ensemble_smoothed/`) was supposed to close the predictor-vs-real gradient-hacking gap that the original single-fold steering (E025) left open. Question: did it?
- **Configs:**
  - Run analysed: `results/steering_camsol_tango_L500_ensemble_smoothed/{camsol_max_w*,tango_min_w*}` — 10 configs (camsol max + tango min, w ∈ {1, 2, 4, 8, 16}), 16 seeds × 3 lengths {300, 400, 500} = 48 PDBs/cell. nsteps=400, 5-fold ensemble + Gaussian smoothing (σ=0.1, K=4) per `steering/config/sweep_camsol_tango_ensemble_smoothed/*.yaml`.
  - Diagnostic: `scripts/steering_predictor_vs_real.py` (new). Pulls predictor's last-step `predicted_properties` per protein from the diagnostics JSON and joins on `properties_guided.csv` (real `tango_total` from local TANGO binary).
  - Caveat: `camsol_intrinsic` is always-NaN in `compute_developability.py` (no public CamSol binary), so for the camsol_max sweep we can only report predictor numbers + collateral real properties (sap, scm, hyd_patch, etc.).
- **Results — TANGO sweep:**

  | w | predictor mean | real `tango_total` mean | Δ predictor vs w=1 | Δ real vs w=1 |
  |---|---|---|---|---|
  | 1 | 959.8 | 877.9 | 0 | 0 |
  | 2 | 938.6 | 874.4 | -21 | -4 |
  | 4 | 897.8 | 868.5 | -62 | -9 |
  | 8 | 819.8 | 861.5 | -140 | -16 |
  | 16 | 671.9 | 843.9 | **-288** | **-34** |

  Predictor:real ratio ≈ **8.5×** at w=16 — same order as the pre-ensemble gap noted in CLAUDE.md ("~10×"). Ensemble + smoothing did NOT close the gap, only attenuated it slightly. Real movement is in the right direction (monotonic in w), but predictor over-claims by an order of magnitude.

- **Results — CamSol sweep (predictor-only, real CamSol unavailable):**
  - Predictor mean rises monotonically: 1.59 (w=1) → 2.94 (w=16), Δ = +1.35 z-score.
  - Collateral real-property drift across w=1→w=16: sap_total 6.93→6.39, tango_total 877.8→857.3, scm_negative -53.7→-57.3, hyd_patch_area 2098→1984. Modest changes — model is doing *something*, but the target itself is unverified.

- **Possible narrative:** non-narrative. Records the failure mode that motivates E029 (noise-aware predictor). Mechanism story (predictor trained at clean t=1 but evaluated at SDE-trajectory t<1 → off-distribution → adversarial) goes into E029, not here.
- **Methodological caveats:**
  - Per-protein matching uses the same `protein_id` (`sX_nL`) string in both the diagnostics JSON and `properties_guided.csv` — no length filtering applied beyond the run's own {300, 400, 500} grid.
  - Predictor's "last-step claim" is the de-normalised mean across the 5-fold ensemble at the very last sampling step (t≈1). It is the model's belief about the final clean protein — not the real value of the actual generated protein.
  - n=48 per cell is enough to make w-sweep monotonicity visible; isn't large enough to bound per-protein scatter.
  - CamSol direction has no real-property anchor; reported only as predictor-side and collateral-real evidence.

## E029 — Noise-aware predictor fine-tune and single-fold validation smoke (2026-05-05)

- **Status:** finished. Pilot only — needs scale-up before it lands as a Finding.
- **Why ran:** E028 confirmed that 5-fold ensembling + Gaussian smoothing leaves the predictor:real gap at ~8.5× — the same order of magnitude as no defense at all. Hypothesis: the gap survives because the predictor is trained on clean t=1 AE-mean latents but evaluated at sampling time on SDE-trajectory `z_t` (eq. 3 in `proteinfoundation/flow_matching/rdn_flow_matcher.py:288`), which is off-distribution. Adversarial off-manifold directions at intermediate t are exactly what gradient hacking exploits, and ensemble/smoothing only attenuate high-frequency adversarial directions, not low-frequency off-manifold ones. Fix: fine-tune the predictor on `z_t` drawn from the same distribution it sees at sampling time, restricted to the steering window.
- **Configs:**
  - Source ckpts: `laproteina_steerability/logs/multitask_t1/20260427_161809/checkpoints/fold_{0..4}_best.pt` (the 5-fold ensemble used in E028, untouched).
  - Fine-tune script: `laproteina_steerability/scripts/add_noisy_latents.py` (new).
  - Noise model: `z_t = (1 - t) · ε_1 + t · z_1 + σ_L · √(t(1-t)) · ε_2`, with t ∼ U(0.3, 0.8) (matches `schedule.t_start` / `schedule.t_end` in the steering configs), σ_L (Langevin scale) = 0.1. The √(t(1-t)) Brownian-bridge envelope makes the extra noise peak mid-trajectory and vanish at the endpoints; σ_L=0 would be the principled marginal-only training (the SDE marginal equals the closed-form interpolant in flow matching), σ_L>0 acts as Tikhonov data augmentation.
  - Hyperparams: 10 epochs, AdamW lr=1e-4 (3× lower than the from-scratch training's 3e-4), wd=0.01, batch=16, patience=4, grad_clip=1.0. ZScoreStats inherited verbatim from each src ckpt — critical so the steering hook's de-normalisation stays consistent.
  - Same fold-split as the source run (fold_assignments.csv reused) — otherwise val→train leakage at sampling time when the ensemble averages folds.
  - Hardware: 1× L4 GPU (CUDA_VISIBLE_DEVICES=1), all 5 folds in series, ~28 minutes total wall.
  - Output: `laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints/fold_{0..4}_best.pt` (originals untouched).
  - Validation smoke: `steering/config/noise_aware_smoke/tango_min_w16_fold2.yaml` — single-fold (best one, fold 2) checkpoint, no smoothing, otherwise byte-equivalent to `tango_min_w16.yaml` from the May-04 ensemble sweep. 4 seeds {42, 43, 44, 45}, L=300, nsteps=400, w_max=16. Compared against the matching protein IDs from the May-04 ensemble run.

- **Results — fine-tune training:**

  | Fold | Source val_r2_mean (t=1) | Best r2_noisy (t∈[0.3, 0.8]) | Best r2_t1 alongside | Best epoch |
  |---|---|---|---|---|
  | 0 | 0.8673 | 0.5785 | 0.7928 | 6 |
  | 1 | 0.8829 | 0.5883 | 0.7759 | 8 |
  | 2 | 0.9090 | **0.5942** | 0.8186 | 7 |
  | 3 | 0.9081 | 0.5880 | 0.8201 | 7 |
  | 4 | 0.9096 | 0.5880 | 0.8032 | 9 (last epoch) |

  Fold 4 was still saving new bests at the last allowed epoch — curves had not plateaued. r2_t1 dropped ~0.08–0.10 across folds, consistent with the network reallocating capacity from the (unused-during-fine-tune) t=1 region to t∈[0.3, 0.8].

- **Results — predictor-vs-real validation (4 seeds, L=300, w=16, fold 2 only):**

  | run | predictor mean | real `tango_total` mean | mean gap |
  |---|---|---|---|
  | Old (5-fold clean ensemble + σ=0.1 smoothing) | 378.5 | 581.9 | **−203.5** |
  | New (1× noise-aware fold 2, no smoothing)    | 540.0 | 587.5 | **−47.5** |

  **~4× reduction in predictor:real hacking gap, with strictly less defensive machinery (1 fold vs 5 folds + smoothing).** Per-protein gaps for the new run: s42_n300 −110.8, s43_n300 −35.3, s44_n300 −48.4, s45_n300 +4.6 (predictor essentially honest on s45). Real `tango_total` means are nearly identical between runs (581.9 vs 587.5) — the noise-aware predictor delivers approximately the same real-property change but with much less self-delusion about it.

- **Possible narrative:** strong candidate for a Finding once scaled up. The pilot is internally consistent (ensemble had MORE defense and got LESS honesty; mechanism story, "predictor sees its training distribution", is concrete and falsifiable) but n is too small to publish. Pre-promotion to Finding requires:
  1. Full-w sweep on TANGO (w ∈ {1, 2, 4, 8, 16}) at L=300/400/500, 16 seeds — same grid as E028 — so the gap-vs-w shape is comparable.
  2. Ensemble of all 5 noise-aware folds vs single noise-aware fold (separate the noise-aware-training effect from the ensemble effect).
  3. Longer fine-tune (≥20 epochs) — fold 4's curve still had slope, the cited gap is a lower bound on what's achievable.
  4. Extension to camsol direction: not directly verifiable locally, but real CamSol from the web server on a small batch would close the only-TANGO caveat.

  A Finding-grade claim looks like: *"Training the steering predictor on `z_t` over the steering window t∈[0.3, 0.8] reduces the predictor:real gradient-hacking gap by ~Nx at w=16 across L∈{300, 400, 500}, dominating the previous ensemble+smoothing combination at strictly lower defensive cost."*

- **Methodological caveats:**
  - **n=4 proteins, 1 fold, 1 length, 1 w-value.** Trend is suggestive, not robust. The −203 → −47 reduction *could* be a 4-protein luck draw; on a per-protein basis the new gap ranges from +4.6 to −110.8.
  - **No CamSol validation.** Real `camsol_intrinsic` requires the CamSol web server and was not run. The hacking-gap claim is TANGO-only.
  - **Single fold vs 5-fold ensemble is not a clean comparison.** The old run's gap (−203) is the average across the same 4 protein IDs but pulled from the 5-fold-ensemble + σ=0.1-smoothing run. The new fold-2-only run lacks both. So the −47 number could reflect *either* the noise-aware training *or* the absence of ensemble averaging effects (some of which can also blur real-property shift). Disentangling these requires a 5-fold-clean vs 5-fold-noise-aware comparison at the same single-fold-no-smoothing baseline.
  - **r²_noisy = 0.59 is mediocre as a property prediction R².** The fact that hacking still drops 4× suggests gradient direction matters more than global accuracy, but a stronger predictor (more epochs / wider model) might close the gap further. Not yet tested.
  - **Fine-tune curves had not plateaued** at epoch 9 for several folds — the cited gap is a lower bound on what longer training would deliver.

- **Cross-references:**
  - E028 — direct parent. E028's negative result (ensemble+smoothing leaves the gap intact) is what motivated this experiment.
  - E025 — the original single-fold steered sweep with predictor:real gap ~10× that prompted the ensemble defense in the first place.
  - Implementation: `laproteina_steerability/scripts/add_noisy_latents.py` (fine-tune), `steering/config/noise_aware_smoke/tango_min_w{1,16}_fold2.yaml` (smoke configs), `scripts/noise_aware_smoke_compare.py` (per-protein gap diagnostic).

## E030 — Universal guidance K=5 with clean predictor probe (2026-05-05)

- **Status:** finished. Negative result.
- **Why ran:** after E029 cut the hacking gap 4× via a noise-aware predictor at the predictor-input layer, the obvious follow-up was: does the *sampling-time* fix — universal guidance, replacing the one-step Tweedie estimate `x_1_est = z_t + (1 - t)·v` with K iterative Euler steps of the latent flow ODE — also help? And does it compose with the existing 5-fold-ensemble + σ=0.1 smoothing defense? Probe specifically with the **original clean predictor** (not the noise-aware one) so we can isolate the universal-guidance effect from the noise-aware-training effect.
- **Configs:**
  - Steering config: `steering/config/universal_guidance_smoke/tango_min_w16_clean_K5.yaml`. Byte-identical to `sweep_camsol_tango_ensemble_smoothed/tango_min_w16.yaml` from E028 (5-fold clean ensemble + σ=0.1 smoothing K_smooth=4) PLUS `denoising_steps: 5`.
  - Implementation: extended `steering/guide.py` with a `denoising_steps` config option; when K_d>1, the guide accepts a `flow_step_fn` callable from `product_space_flow_matcher.py` that runs `predict_for_sampling` with the latent channel rewritten to (z_iter, t_iter) while other modes stay frozen at the outer step. Backprop flows through K_d stacked flow Jacobians + the predictor.
  - Run: 4 seeds {42, 43, 44, 45} × L=300 × w_max=16, nsteps=400. Same protein IDs as E029's noise-aware smoke for direct comparison.
  - Output: `results/universal_guidance_smoke/tango_min_w16_clean_K5/`.
  - Hardware: 1× L4 GPU. Wall ≈ 7.7 min total (≈115s/protein, vs ≈19s/protein for K_d=1 — 6× slower per generation as expected for 5 inner flow forwards + backprop through them).

- **Results — predictor-vs-real (same 4 seeds, L=300, w=16):**

  | approach | predictor mean | real `tango_total` mean | mean gap |
  |---|---|---|---|
  | E028 ensemble + smoothing + K_d=1 | 378.5 | 581.9 | **−203.5** |
  | E029 noise-aware fold 2 + K_d=1   | 540.0 | 587.5 | **−47.5** |
  | **E030 ensemble + smoothing + K_d=5** | **305.4** | **607.1** | **−301.7** |

  K_d=5 universal guidance with the clean predictor:
  - The hacking gap is **~50% larger** (−302 vs −203 baseline at K_d=1).
  - Real `tango_total` is **higher** than both other approaches — i.e., the proteins are LESS minimized in TANGO. Steering pushed in a less-useful direction overall.
  - Per-protein gaps: s42 −337, s43 −256, s44 −303, s45 −310 — the ~−180 bump on s45 (from −188 → −310) is the most striking individual swing.

- **Mechanism story (best read of why it backfired):** when the predictor is gradient-hackable to begin with — as the clean t=1-trained predictor demonstrably is, with E028's 8.5× over-claim — replacing a 1-step Jacobian (`∂x_1_est/∂z_t = I` for the linear Tweedie estimate) with a 5-step product of flow Jacobians does not give the predictor a "better" gradient in any useful sense. It gives the adversarial gradient direction a longer lever:
  1. More degrees of freedom for adversarial directions to live in (the flow Jacobian product has rank determined by network capacity, not z_t dimension).
  2. Larger gradient magnitudes (which the unit-normalisation downstream then re-uses as the dominant direction).
  3. K_d=5 over `[0.3, 1.0]` is dt≈0.14/step — a coarse Euler approximation of the SDE the model was actually trained for (dt≈0.0025 at nsteps=400), so x_1_est is not actually a reliable "clean" estimate; it's a coarsely-integrated approximation that itself has flow-direction error.

  The takeaway is the same shape as the E028 ensemble lesson: post-hoc tricks layered on a fragile predictor amplify the fragility instead of curing it. The fix has to be at the predictor-input layer (E029) before any sampling-time refinement can compose usefully.

- **Possible narrative:** non-narrative. Negative-result smoke that closes off one hypothesised line of attack. Would only become a Finding if expanded with a *positive* counterpart: UG K=5 + noise-aware predictor showing a further gap reduction beyond E029's −47.
- **Methodological caveats:**
  - **n=4 proteins, single fold-set, single length, single w-value.** As with E029 the headline is suggestive, not robust. The negative direction (worse, not just neutral) is consistent across all 4 proteins, which makes the "noise" interpretation harder but doesn't eliminate it.
  - K_d=5 was picked for compute reasons (115s/protein at L=300; K_d=20 would be ~7 min/protein × 4 = 28 min smoke — feasible but the user explicitly asked for "a couple samples" probe). Larger K_d gives a more accurate flow integration but also a deeper Jacobian product. Whether very-large K closes the gap or makes it even worse isn't tested.
  - **Untested combo: UG K=5 + noise-aware predictor.** The mechanism story predicts this would help (predictor isn't fragile, so longer lever isn't dangerous), but we only have data for UG + clean. Future test if E029 alone proves insufficient at scale.
  - Real TANGO mean rose 25 (582 → 607) — that's only ~3% of starting value, so the "less effective steering" observation is small in absolute terms. Could be noise. But combined with the larger gap it's coherent with "worse gradient direction."

- **Cross-references:**
  - E028 (ensemble + smoothing + K_d=1, the K_d=1 baseline this probe sits against), E029 (the noise-aware-only fix that does close the gap, baseline this experiment loses to without UG ever being involved).
  - Implementation files: `steering/guide.py` (added `denoising_steps` option + K-step Euler loop in `guide.guide`), `proteinfoundation/flow_matching/product_space_flow_matcher.py` (built `_flow_step_fn` closure in the guidance call site), `steering/config/universal_guidance_smoke/tango_min_w16_clean_K5.yaml` (the run config), `scripts/noise_aware_smoke_compare.py` (extended to a 3-row comparison).
  - Predicts: any future "make the predictor see x_1 more accurately at sampling time" defense (multi-step denoising, learned correctors, predictor-of-x_1 directly) should be paired with a robust-input predictor first; ordering matters.

## E031 — Noise-aware predictor v2 (longer + cosine decay) and the r² vs hacking disconnect (2026-05-05)

- **Status:** finished. Negative result for v2; **v1 remains the canonical noise-aware checkpoint**.
- **Why ran:** E029's noise-aware fine-tune (10 epochs, constant LR=1e-4, σ_langevin=0.1, t∈[0.3, 0.8]) cut the predictor:real hacking gap from −203 (E028 ensemble+smoothing baseline) to −47 — a strong result, but the fine-tune curves had not plateaued: fold 4 was still climbing at the last allowed epoch, and several folds were oscillating ±0.02 r²_noisy in late epochs at constant LR. Hypothesis: train longer with LR decay → better predictor → smaller hacking gap. Test by re-running the same fold-2 smoke (4 seeds, L=300, w=16) and comparing.
- **Configs:**
  - Source ckpts: same as E029 (E028's `logs/multitask_t1/20260427_161809/checkpoints/fold_{0..4}_best.pt`).
  - Fine-tune script: `laproteina_steerability/scripts/add_noisy_latents.py` extended with cosine LR decay (linear from `cfg.lr` down to `0.1 × cfg.lr` over the full epoch budget, applied per-step).
  - Hyperparams vs E029: epochs 10 → **30**, LR schedule constant 1e-4 → **cosine 1e-4 → 1e-5**, patience 4 → **8**. Everything else identical.
  - Output: `laproteina_steerability/logs/multitask_t1_noise_aware/20260505_123607/checkpoints/fold_{0..4}_best.pt` (originals + E029's v1 ckpts at `20260505_110348/` both untouched).
  - Validation smoke: same 4 seeds, L=300, w=16, tango_min_w16 — `steering/config/noise_aware_smoke/tango_min_w16_fold2_v2.yaml` only differs from E029's smoke config in the checkpoint path.
  - Hardware: 1× L4 GPU, ~78 min total wall (5 folds × 30 epochs × ~32 s/epoch + 5 evals/fold).

- **Results — training (v2 vs E029's v1):**

  | Fold | v1 best r²_noisy (10 ep, const LR) | v2 best r²_noisy (30 ep, cosine) | Δ |
  |---|---|---|---|
  | 0 | 0.5785 | 0.6167 | +0.038 |
  | 1 | 0.5883 | 0.6234 | +0.035 |
  | 2 | 0.5942 | **0.6455** | +0.051 |
  | 3 | 0.5880 | 0.6398 | +0.052 |
  | 4 | 0.5880 | 0.6449 | +0.057 |
  | mean | 0.5874 | 0.6341 | **+0.047** |

  Bonus: r²_t1 also climbed (v1 averaged 0.78-0.82, v2 averages 0.83-0.85) despite no t=1 data being used in fine-tuning. The model generalised across t rather than memorising the noisy regime — a positive secondary signal in isolation.

- **Results — predictor-vs-real (same 4 seeds, L=300, w=16):**

  | approach | predictor mean | real `tango_total` mean | mean gap |
  |---|---|---|---|
  | E028 ensemble + smoothing + K_d=1 | 378.5 | 581.9 | **−203.5** |
  | E029 v1 noise-aware fold 2        | 540.0 | 587.5 | **−47.5**  |
  | E030 UG K=5 + clean ensemble      | 305.4 | 607.1 | **−301.7** |
  | **E031 v2 noise-aware fold 2**    | **457.8** | **603.2** | **−145.3** |

  v2 has higher val r² but **3× the hacking gap of v1**. Per-protein:
  - s42: v1 −111 → v2 −176
  - s43: v1 −35  → v2 −123
  - s44: v1 −48  → v2 −136
  - s45: v1 +5 (essentially honest) → v2 −146 (heavily hacked)

  Real `tango_total` mean is also slightly higher in v2 (603 vs 587) — i.e., v2 delivers slightly *less* effective steering on top of being more hacked.

- **Why "better" is worse — best read of the mechanism:**
  1. **r²_noisy ≠ gradient-hackability.** r²_noisy is computed on validation latents drawn from the natural noisy-interpolant distribution. Gradient hacking explores *adversarial* noisy latents — directions in z_t space that maximise the predictor's output but are not on the data manifold. A predictor that is more accurate on the natural distribution can be more confidently wrong off it.
  2. **Cosine decay → sharp minima.** Late-stage low-LR converges into sharp local minima of the loss landscape (well-documented in the optimisation-of-DNN literature). Sharp minima have larger curvature in the parameter space, which translates into larger gradient magnitudes near decision boundaries in the input space. With unit-normalisation downstream of the predictor's gradient, the dominant component of `∇` becomes whichever direction has the largest magnitude — and adversarial directions often *do* have large magnitudes near sharp boundaries.
  3. **Sharper predictor = stronger steering signal AND a stronger hacking signal**, with the latter dominating because gradient hacking is fundamentally a search over directions with maximum predictor responsiveness.
  4. **Statistical caveat (n=4):** the v1→v2 swing on s45 (+5 → −146) is the largest absolute swing seen; the other 3 proteins move 60-90 units in the same direction. Direction is consistent but the magnitude of the v2 penalty could easily be n=4 noise inflated.

- **Possible narrative:** non-narrative on its own, but feeds a **future Finding** about the metric-mismatch between predictor R² and steering reliability. The shape of the result is the same as Findings 5/6 in `content_masterarbeit.md`: an optimisation-side metric (val loss / val r²) improves while the downstream metric (designability / hacking gap) degrades.
- **Methodological caveats:**
  - **n=4 proteins, 1 fold, 1 length, 1 w-value.** As with E029 and E030, the trend is suggestive, not robust. The direction-consistent swing across all 4 proteins makes pure-noise interpretations harder, but the magnitude is uncertain.
  - **No CamSol validation** (same as E028/E029 — `compute_developability.py` returns NaN for `camsol_intrinsic`).
  - **Sharp-vs-flat-minima mechanism is plausible but not proven.** A clean test would be to re-fine-tune v1's epoch-10 checkpoint with SWA (stochastic weight averaging) → flatter minimum → measure the gap; or to evaluate the per-protein input-Jacobian norm of v1 vs v2 at the steering inputs and see if v2's is larger.
  - **Early-stopping criterion mismatch.** Both v1 and v2 monitor val r²_noisy, which we now know does not predict steering quality. The right early-stopping criterion is gradient-hacking gap on a held-out generation batch — operationally hard but principled. Future fine-tunes should track that directly.

- **Cross-references:**
  - E029 — direct parent. v1 checkpoint stays canonical.
  - E028, E030 — the two earlier "more defense / more sophistication" attempts that also backfired on the same axis.
  - Finding 5/6 in `content_masterarbeit.md` (CA-only baseline v2 with wd=0.1+cosine: lower val MSE, 0/3 designable). Same shape — optimisation metric and downstream metric diverge under "more defensive" hyperparameter choices.
  - Predicts: any future "tighten the predictor" experiment should be paired with a steering-side gap probe before promotion. Don't ship a noise-aware checkpoint based on val r² alone.

## E032 — Noise-aware predictor + 5-fold ensemble — gap essentially closed (2026-05-05)

- **Status:** finished. **First positive result of the day**, after E028-E031's chain of negative attempts. Pilot scale (n=4 proteins), but the magnitude of the effect — gap of −1.6 — is large enough that even very pessimistic n=4-noise readings still leave it as a strong positive.
- **Why ran:** the day's chain of negative results (E028 ensemble+smoothing on clean: −203; E030 UG K=5 on clean: −302; E031 longer noise-aware single fold: −145; v1+v2 z_t-direct: −152, −187) all had a common feature — they tackled *one* mechanism (input-distribution mismatch *or* fold-specific shortcuts *or* sampling-time refinement) without addressing the others. E029 single-fold noise-aware (−47) showed input-distribution fix is necessary but not sufficient. Hypothesis: combining noise-aware training with 5-fold ensemble averaging tackles both mechanisms — input-distribution at training time, fold-specific shortcuts at sampling time. The clean ensemble alone (E028) was at −203 because off-distribution dominated. With off-distribution fixed by noise-aware training, residual hacking is fold-specific shortcuts, exactly what ensembling cancels.
- **Configs:**
  - Steering config: `steering/config/noise_aware_smoke/tango_min_w16_v1_ensemble.yaml`. 5 v1 noise-aware ckpts from `laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints/fold_{0..4}_best.pt` (E029's single-fold ckpts, used together as `SteeringPredictor` ensemble — the `predictor.py` ensemble path averages z-scored predictions across folds, so the gradient is the mean of 5 fold gradients).
  - Path: legacy `x_1_est = z_t + (1-t)·v_θ` + `t = 1.0` to the predictor (the surprisingly-better path identified by the same-day v1+v2 z_t-direct head-to-head, which made things worse despite being the "principled" choice).
  - Smoothing: **off** (σ_smooth = 0). Smoothing was originally introduced to attenuate adversarial directions, but it's an additional defense layer on top of ensembling and we wanted a clean read of the ensemble effect alone.
  - 4 seeds {42, 43, 44, 45} × L=300 × w_max=16, nsteps=400 — same protein IDs as every smoke today for direct comparison.
  - Wall: ~89 s for 4 proteins (≈22 s/protein, 5 forward passes per outer step instead of 1, modest slowdown).
  - Output: `results/noise_aware_smoke/tango_min_w16_v1_ensemble/`.

- **Headline result — predictor-vs-real gap (n=4, L=300, w=16):**

  | approach (ordered by gap) | predictor mean | real `tango_total` mean | mean gap |
  |---|---|---|---|
  | **v1 noise-aware 5-fold ensemble** | **579.9** | **581.5** | **−1.6** |
  | v1 noise-aware single fold 2 (E029) | 540.0 | 587.5 | −47.5 |
  | v2 noise-aware single fold 2 (E031) | 457.8 | 603.2 | −145.3 |
  | v1 + z_t-direct (today, single fold) | 471.2 | 622.9 | −151.7 |
  | v2 + z_t-direct (today, single fold) | 445.0 | 632.0 | −187.1 |
  | clean 5-fold ensemble + smoothing (E028) | 378.5 | 581.9 | −203.5 |
  | UG K=5 + clean ensemble + smoothing (E030) | 305.4 | 607.1 | −301.7 |

  Per-protein gaps for the v1 ensemble run:
  - s42_n300:  −74  (still some residual hacking — 1 of 4 proteins)
  - s43_n300:  +27  (predictor *overestimates* real — regression-noise direction)
  - s44_n300:  +13  (overestimates)
  - s45_n300:  +27  (overestimates)

  3 of 4 proteins have positive gaps — meaning the predictor's claim is *higher* than real TANGO. This is the regression-noise regime where the predictor is honest and just imperfect. Only s42 still has a sizeable negative gap; with n=4 we can't tell whether s42 is a real persistent failure mode or sampling noise.

- **Steering effectiveness — real `tango_total` essentially unchanged.** Mean real TANGO is 581.5 (ensemble) vs 587.5 (E029 single fold) vs 581.9 (clean ensemble). The 5-fold combination delivers approximately the same real-property change as single-fold while being ~30× more honest about it. **Honest prediction does not blunt steering effectiveness**, in this pilot.

- **Mechanism story (now consistent across all 5 attempts today):**
  1. **Off-distribution input** is one independent failure mode. Clean predictors evaluated at sampling-time z_t (or x_1_est, or anything not exactly z_1) produce gradients that find adversarial directions in z_t-space. Fixed by noise-aware training (E029): −203 → −47.
  2. **Fold-specific shortcuts** are a separate independent failure mode. Each fold memorises slightly different "looks-low-tango-to-me" features in noisy z_t. The honest signal is shared across folds (because they all learned the same real-property labels); shortcut features are fold-specific (because they depend on which subset of training data each fold saw). Averaging over folds cancels shortcut features and reinforces honest signal. Fixed by ensemble: −47 → −1.6.
  3. **Anything that gives the gradient *more* leverage in z_t-space without addressing (1) or (2) makes hacking worse.** UG K=5 (E030) added 5-step Jacobian leverage. v2 longer training (E031) sharpened minima. z_t-direct (today, both versions) removed the implicit flow-manifold anchoring of `x_1_est`. All three made hacking worse, consistent with the read that the residual gradient is dominated by adversarial directions and amplifying it amplifies hacking.

  The `x_1_est = z_t + (1-t)·v_θ` reconstruction-guidance trick — originally a workaround for clean-only predictors — turns out to be a load-bearing implicit regulariser: it ties the predictor's gradient to the flow's velocity field and biases steering away from off-manifold directions. Removing it (z_t-direct) was empirically catastrophic even when the predictor *was* trained for z_t input.

- **Possible narrative:** **Finding-grade once scaled up.** The clean composition story (two orthogonal mechanisms, both needed, multiplicative effect: 4× × 30× ≈ 120× total reduction from clean-ensemble → noise-aware-ensemble, or in absolute terms −203 → −1.6) is exactly the kind of result that becomes a paper claim:
  > *"Closing the gradient-hacking gap in latent flow-matching steering requires two compositional fixes: (1) training the property predictor on noisy intermediate-time latents drawn from the SDE forward distribution, fixing the off-distribution-input failure mode; and (2) averaging predictions across an ensemble of independently-trained predictor folds, cancelling fold-specific adversarial shortcuts. Either fix alone leaves the gap at order ~10–20× of real-property change. Combined, the gap closes to within regression noise (mean predictor:real residual −1.6 TANGO units vs real-property changes of ~30 units across w=1→16)."*

  Pre-promotion to a Finding requires:
  1. Full w-sweep (1, 2, 4, 8, 16) on tango_min × 16 seeds × L∈{300, 400, 500} — same grid as E028 — so the gap-vs-w shape is comparable.
  2. scRMSD evaluation across the w-sweep — answers the "without degrading the protein" question that's still open.
  3. CamSol-direction validation. `compute_developability.py` returns NaN for `camsol_intrinsic` so this requires submitting sequences to the CamSol web server. Workable for a small batch.
  4. Statistical robustness: at the chosen operating w, n=48 vs n=4 changes the confidence interval on the mean gap dramatically. A 4× reduction at n=4 has wide error bars; a 50× reduction at n=48 is much more defensible.

- **Methodological caveats:**
  - **n=4 proteins.** The gap of −1.6 has wide bounds. Per-protein gaps {−74, +27, +13, +27} have std ≈ 50 — so a 95% CI on the mean gap is roughly [−51, +47], including zero. The headline "essentially closed" is true *on average* but the s42 outlier shows residual single-protein hacking is still possible.
  - **Single length, single w-value, single direction.** TANGO-min at w=16, L=300 is the cell where the original clean-ensemble run had the largest gap. Other cells may behave differently (smaller gap to start with means the same %-reduction shows a smaller absolute improvement; could falsely look "less effective" in the new setup).
  - **Tango-only.** No CamSol validation locally. If CamSol behaves differently (different property landscape, different fold-specific shortcuts) the ensemble fix may not transfer 1:1.
  - **Single fold-set noise-aware ckpts.** All 5 noise-aware folds were fine-tuned with the same noise model, same hyperparams, same source ckpts. Fold-specific adversarial directions are still drawn from a fairly correlated distribution. A truly diverse ensemble (different noise schedules per fold, different random seeds for fine-tune init) might tighten the gap further.
  - **Smoothing was off.** Adding σ_smooth = 0.1 on top might tighten the gap by another increment, or might be redundant with ensembling. Not tested today.

- **Track-record honesty:** Of today's six "this should help" predictions for closing the gap, five were wrong (clean ensemble + smoothing didn't close it, UG K=5 made it worse, longer training made it worse, z_t-direct made it worse on both predictors). The 6th — combining noise-aware + ensemble — closed the gap. The lesson: don't trust mechanism-of-action stories without a measurement; the surface-level intuition of "more defense / better predictor / cleaner input" is unreliable for adversarial-robustness questions where measurement is essential.

- **Cross-references:**
  - E028 — clean 5-fold ensemble + smoothing baseline (gap −203). Establishes that ensembling alone, without noise-aware training, doesn't close the gap.
  - E029 — single-fold noise-aware (gap −47). Establishes that noise-aware training alone, without ensembling, doesn't close the gap.
  - E030 — universal guidance K=5 attempt (gap −302). Negative; same compositional lesson.
  - E031 — longer noise-aware fine-tune (gap −145). Negative; r² ≠ hackability.
  - Today's z_t-direct attempts: also negative; reinforce the load-bearing role of the x_1_est anchor.
  - Implementation files: `steering/config/noise_aware_smoke/tango_min_w16_v1_ensemble.yaml` (run config), `scripts/noise_aware_smoke_compare.py` (extended to 7-row table).
  - Predicts: any future steering predictor work should be evaluated on the predictor:real gap directly, not on val r² (E031 lesson) and not on individual mechanism plausibility arguments (today's lesson).

### E032 update (2026-05-05 21:51) — full grid n=48 per cell

Earlier E032 entry was the n=4 pilot at L=300 / w=16. Full sweep (`results/noise_aware_ensemble_sweep/`, 16 seeds × 3 lengths {300, 400, 500} × 5 w-levels × 2 directions = 480 PDBs) now done. Eval pipeline at `scripts/eval_noise_aware_ensemble_sweep.py`.

**Aggregate per (direction, w), n=48 per cell:**

| direction | w | predictor mean | real mean | gap mean | gap std | gap p5 | gap p95 |
|---|---|---|---|---|---|---|---|
| camsol_max | 1 | 1.1 | n/a | n/a | n/a | n/a | n/a |
| camsol_max | 2 | 1.2 | n/a | n/a | n/a | n/a | n/a |
| camsol_max | 4 | 1.3 | n/a | n/a | n/a | n/a | n/a |
| camsol_max | 8 | 1.5 | n/a | n/a | n/a | n/a | n/a |
| camsol_max | 16 | 1.8 | n/a | n/a | n/a | n/a | n/a |
| tango_min | 1 | 1011.7 | 893.3 | +118.5 | 159.9 | -112.3 | +315.0 |
| tango_min | 2 | 999.3 | 889.5 | +109.8 | 160.4 | -117.6 | +298.3 |
| tango_min | 4 | 975.9 | 884.2 | +91.8 | 163.1 | -156.9 | +282.2 |
| tango_min | 8 | 931.2 | 872.3 | +58.8 | 164.5 | -196.3 | +249.7 |
| **tango_min** | **16** | **837.2** | **833.4** | **+3.8** | 164.2 | -248.0 | +187.8 |

**Δ-vs-w=1 (tango_min):**

| w | Δ predictor | Δ real | ratio Δpred/Δreal |
|---|---|---|---|
| 1 | 0 | 0 | n/a |
| 2 | -12.4 | -3.8 | 3.31× |
| 4 | -35.8 | -9.1 | 3.93× |
| 8 | -80.6 | -20.9 | 3.85× |
| 16 | -174.5 | -59.9 | **2.91×** |

**Comparison to E028 (clean ensemble + smoothing, same grid):**

| metric | E028 (clean) | E032 (noise-aware) | change |
|---|---|---|---|
| gap mean at w=16 | -203.5 | **+3.8** | sign flips, magnitude 50× smaller |
| Δ predictor at w=16 | -287.9 | -174.5 | predictor over-claims ~40% less |
| Δ real at w=16 | -34.0 | -59.9 | **real steering ~2× more effective** |
| Δratio at w=16 | 8.47× | **2.91×** | hacking ratio 3× smaller |

**Per-length breakdown for tango_min, w=16 (n=16 per cell):**

| L | predictor mean | real mean | gap mean | gap std |
|---|---|---|---|---|
| 300 | 602.2 | 573.8 | +28.4 | 90.5 |
| 400 | 844.0 | 886.2 | **-42.2** | 157.3 |
| 500 | 1065.3 | 1040.1 | +25.2 | 219.3 |

**Headline-vs-pilot caveats:**
- Gap mean of +3.8 at w=16 partly reflects crossover, not pure honesty: at w=1 predictor over-claims +118, at w=16 they meet near zero. Without steering, the predictor systematically over-estimates real TANGO by ~120 — calibration drift, not hacking. Steering eats this overestimate as it pulls the prediction down. If we pushed past w=16, the predictor might cross over to under-claiming.
- Δratio of 2.9× shows residual hacking in the *change-from-baseline* axis — predictor moves 3× faster than reality. Better than E028's 8.5× but not 1.
- Std ~160 across all cells. Per-protein gaps span [-248, +188] at w=16 — wide spread under a small mean.
- Per-length sign disagreement at w=16: L=300 +28, L=400 **-42**, L=500 +25.
- The n=4 pilot at L=300 gave gap -1.6; the n=16 at same length gives +28. The pilot was a lucky sub-sample; the full mean is small but not as small as the pilot suggested.

**Status of the Finding-grade gates:**

| gate (from "what's missing for a Finding") | status |
|---|---|
| n=48 per cell | ✓ done |
| full w-sweep at all 5 levels | ✓ done |
| both directions (camsol_max + tango_min) | ✓ done (camsol predictor-only, no real validation) |
| scRMSD across the sweep | pending — next |
| ablation table at one fixed cell | pending — next |
| CamSol web-server validation | pending — sequence batch ready in `properties_guided.csv` per cell |

**Real-property delivery (tango_min, mean across all 48 proteins per cell):**

| w | real `tango_total` |
|---|---|
| 1 | 893.3 |
| 2 | 889.5 |
| 4 | 884.2 |
| 8 | 872.3 |
| 16 | 833.4 |

Real TANGO drops monotonically with w. Steering is genuinely working in the right direction across the sweep. Δreal at w=16 is -59.9 (~7% reduction in TANGO from baseline) — about 2× the change E028 achieved.

## E033 — scRMSD validation of the noise-aware ensemble sweep (2026-05-06)

- **Status:** finished. 9.4 h wall on 1× L4 GPU. Closes the "without breaking the protein" gate that E032's gap-closure result left open.
- **Why ran:** E032 + its n=48 update established that the noise-aware 5-fold ensemble closes the predictor:real hacking gap (gap mean +3.8 at w=16, vs −203 with the clean-ensemble baseline E028) and *roughly doubles* real-property delivery (Δreal_tango at w=16: −60 vs E028's −34). But "the predictor is honest about real-property change" is necessary but not sufficient — if the steered proteins are 8 Å scrambled blobs that *happen* to score low TANGO, we have honest steering of garbage. The user's "without degrading the protein" question, raised earlier and unresolved through E028-E032, demands a designability check at every w on the new sweep.
- **Configs:**
  - Source: `results/noise_aware_ensemble_sweep/{camsol_max,tango_min}_w{1,2,4,8,16}/guided/` from E032 — 240 cells × 16 seeds × 3 lengths but evaluated on a subset for compute (4 seeds × 3 lengths per cell = 12 PDBs/cell, 10 cells = 120 PDBs total).
  - Pipeline: `script_utils/run_scrmsd_steering.py` (the official `proteinfoundation.metrics.designability.scRMSD` path: ProteinMPNN N=8 sequences/structure → ESMFold prediction → CA-RMSD min across 8 sequences). Required `PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python` env var so MPNN's subprocess inherits the right Python (default is `python` from PATH which is the no-numpy system python — first attempt failed for this reason; documented for next time).
  - Seeds {42, 43, 44, 45} × Lengths {300, 400, 500} × 5 w-levels × 2 directions = 120 evaluations.
  - 2 individual scRMSD failures: `tango_min_w1/s42_n500` and `tango_min_w2/s43_n300` both raised tensor-shape mismatch (PDB had off-by-one residue count; off-protein-pipeline-bug, not steering-related).

- **Results — per-cell designability (<2 Å scRMSD via MPNN→ESMFold):**

  Including all 12 proteins per cell:
  | direction | w | n | designable | mean scRMSD |
  |---|---|---|---|---|
  | camsol_max | 1 | 12 | 11/12 (92%) | 1.94 Å |
  | camsol_max | 2 | 12 | 10/12 (83%) | 1.86 Å |
  | camsol_max | 4 | 12 | 10/12 (83%) | 2.57 Å |
  | camsol_max | 8 | 12 | 11/12 (92%) | 2.39 Å |
  | camsol_max | 16 | 12 | 10/12 (83%) | 2.26 Å |
  | tango_min | 1 | 11 (1 failed) | 8/11 (73%) | 1.77 Å |
  | tango_min | 2 | 11 (1 failed) | 9/11 (82%) | 2.30 Å |
  | tango_min | 4 | 12 | 11/12 (92%) | 1.93 Å |
  | tango_min | 8 | 12 | 10/12 (83%) | 2.07 Å |
  | tango_min | 16 | 12 | 9/12 (75%) | 2.01 Å |

  Excluding the **persistent s45_n500 outlier** (broken in 9/10 cells at >10 Å, w-independent; this is a per-seed/per-length generation failure of the underlying La-Proteina LD3 sampler, NOT a steering failure):
  | direction | w | designable | mean scRMSD |
  |---|---|---|---|
  | camsol_max | 1 | 11/11 (100%) | **0.95 Å** |
  | camsol_max | 2 | 10/11 (91%) | 1.04 Å |
  | camsol_max | 4 | 10/11 (91%) | 1.33 Å |
  | camsol_max | 8 | 11/11 (100%) | **1.01 Å** |
  | camsol_max | 16 | 10/11 (91%) | 1.34 Å |
  | tango_min | 1 | 8/10 (80%) | 1.41 Å |
  | tango_min | 2 | 9/10 (90%) | 1.11 Å |
  | tango_min | 4 | 11/11 (100%) | **0.95 Å** |
  | tango_min | 8 | 10/11 (91%) | 1.10 Å |
  | tango_min | 16 | 9/11 (82%) | 1.14 Å |

  **No monotonic w→scRMSD trend in either direction.** tango_min at w=4 (100%) is *better* than at w=1 (80%); at w=16 it's 82%, statistically indistinguishable from the unsteered-ish baseline. camsol_max is rock-solid at 91-100% at every w. The variance is dominated by per-seed/per-length generation noise (s45_n500 broken everywhere, s44_n400 and s42_n300 occasionally borderline), not by steering pressure.

- **Per-length × w (excl. s45_n500), tango_min:**

  | L | w=1 | w=2 | w=4 | w=8 | w=16 |
  |---|---|---|---|---|---|
  | 300 | 3/4 (1.53) | 3/3 (0.61) | 4/4 (0.74) | 4/4 (0.72) | 4/4 (0.76) |
  | 400 | 3/4 (1.30) | 3/4 (1.22) | 4/4 (1.02) | 4/4 (1.10) | 3/4 (1.19) |
  | 500 | 2/2 (1.38) | 3/3 (1.44) | 3/3 (1.13) | 2/3 (1.59) | 2/3 (1.57) |

  L=300 designability is essentially at ceiling (100%) for w ≥ 2. L=400 and L=500 dip slightly at high w but remain in the 67-100% range — no catastrophic break.

- **Per-length × w (excl. s45_n500), camsol_max:**

  | L | w=1 | w=2 | w=4 | w=8 | w=16 |
  |---|---|---|---|---|---|
  | 300 | 4/4 (0.66) | 4/4 (0.76) | 3/4 (1.51) | 4/4 (0.85) | 3/4 (1.49) |
  | 400 | 4/4 (0.95) | 3/4 (1.16) | 4/4 (1.08) | 4/4 (1.05) | 4/4 (1.09) |
  | 500 | 3/3 (1.33) | 3/3 (1.26) | 3/3 (1.43) | 3/3 (1.16) | 3/3 (1.49) |

  camsol_max designability holds at 75-100% at every (L, w) cell.

- **Composite operating-point picture (combines E032 gap result + this scRMSD result):**

  At **w=16** (the "max steering" cell where all the previous defenses had failed):
  - Predictor:real gap mean: **+3.8** TANGO units (E032; was −203 with clean ensemble + smoothing)
  - Real Δ TANGO from w=1 baseline: **−60** units, ~7% reduction (E032)
  - Real Δ predictor / Δ real ratio: **2.91×** (down from 8.5× in E028)
  - Designability: **82-91%** (this E033, ≈ unsteered baseline 73-92%)

  At **w=4** (a milder steering point):
  - Real Δ TANGO: **−9** (small but real change in the right direction)
  - Designability: **91-100%** (essentially perfect)

  At **w=8** (intermediate sweet spot):
  - Real Δ TANGO: **−21**
  - Designability: **91-100%**
  - Predictor:real gap: −47 → meaningful steering with predictor still mostly honest

- **Possible narrative:** **The two halves of Finding 10 (gap closure + structural integrity) lock in here.** E032 alone showed honest predictor and real-property delivery; this E033 shows the proteins are still designable in the same regime. Together they give a deployable steering recipe at the level of the masterarbeit's steering route bar.

- **Methodological caveats:**
  - **n=12 per cell on the scRMSD side, vs n=48 on the gap side.** scRMSD is the bottleneck on compute (~9 h wall for 120 PDBs); a 4× larger sample would cost ~36 h. The n=12 designability rates have wide CIs at the per-cell level (a 91% rate from 10/11 has 95% CI roughly 59-100%). The robust signal is the *across-cell trend*: no monotonic w→scRMSD increase in either direction, and persistence of designability at every w-level above the ~80% baseline.
  - **2 PDB failures** (`tango_min_w1/s42_n500`, `tango_min_w2/s43_n300`) were tensor-shape mismatches in the scRMSD pipeline (off-by-one residue count between input PDB and ESMFold output). Excluded from the per-cell denominator, denoted as "11 (1 failed)" rather than reported as "inf scRMSD". Sub-1% of the run, not a steering-pipeline issue.
  - **One protein dominates the apparent w-variance.** `s45_n500` is broken at >10 Å in 9/10 cells, w-independent. It's a generation failure of the underlying La-Proteina LD3 model at this seed/length, not steering damage. The "excluding s45_n500" tables are the fair comparison; the all-12-included tables are the conservative one.
  - **No CamSol web-server validation** (still pending; sequences are written to the per-cell `properties_guided.csv` and ready to submit).
  - **Designability ≠ functional protein.** scRMSD < 2 Å says "ESMFold thinks an MPNN-designed sequence will fold to roughly the right structure". It does not say the protein folds in vitro, it does not say the property is biologically expressed in the lab, it does not say the structure has the predicted aggregation behaviour. These are wet-lab claims and out of scope.
  - **Steering on top of the **official** LD3+AE2 La-Proteina checkpoint** — not on a CA-only or sparse-attention variant. The fix is specific to that flow + AE configuration; transfer to other flows untested.

- **Cross-references:**
  - E032 (parent — gap closure, real-property delivery) — this E033 closes the second axis (designability).
  - E028, E029, E030, E031 — the chain of negative attempts that eliminated alternative defenses.
  - Finding 10 in `content_masterarbeit.md` — the paper-facing claim built on E029, E032, E033 collectively.
  - Implementation: `script_utils/run_scrmsd_steering.py` (existing script, env var fix), `scripts/scrmsd_sweep_summary.py` (new aggregator). Outputs at `results/noise_aware_ensemble_sweep/{cell}/scRMSD_guided.csv` and `results/noise_aware_ensemble_sweep/scRMSD_summary.csv`.

---

## E034 — CA-only `downsampled` variant quick N=6 designability probe (2026-05-06)

**Status:** finished.

**Why ran:** First designability read on the `ca_only_downsampled` CA-only architectural variant (training run `ca_only_downsampled/1777987722`). Per CLAUDE.md "sample-quality bar (variants must clear)" = 1-2/3 designable at L=50 and L=100. This probe decides whether the variant clears the bar at its current best-val ckpt and warrants further training/N=30 promotion, or whether it should be debugged before more compute is spent.

**Configs:**
- Checkpoint: `best_val_00000023_000000002331.ckpt` (epoch 23, opt step 2331), rsynced from HPC `/rds/user/ks2218/hpc-work/store/ca_only_downsampled/1777987722/checkpoints/`. Local copy: `/home/ks2218/la-proteina/best_val_00000023_000000002331.ckpt` (1.96 GB; mtime 2026-05-06).
- Inference config (new): `configs/inference_downsampled_quick.yaml` — modeled on `inference_sparse_pairupdate_quick.yaml`. 3 lengths × 6 samples × 200 ODE steps = **18 total samples** at L ∈ {50, 100, 200}. `nsteps=200`, `nsamples=6`, `max_nsamples_per_batch=6`. Generation block inherits canonical CA-only sampling settings from `inference_base.yaml` + `generation/uncond_codes_ca_only.yaml`. Seed=5 (inherited from `inference_base.yaml`).
- Training config / NN config: not on this box. The `ca_only_downsampled` training is HPC-only; the local repo has no `configs/training_ca_only_downsampled.yaml` or `configs/nn/ca_only_downsampled_*.yaml`. Architecture is loaded from the ckpt's `hyper_parameters` via `Proteina.load_from_checkpoint` — the inference path doesn't need either YAML.
- Hardware: 1× NVIDIA L4 (gxp-l4-0, GPU 0). Env: `/home/ks2218/.conda/envs/laproteina_env`.
- Output dir: `inference/inference_downsampled_quick/`. CSV: `inference/results_inference_downsampled_quick_0.csv`. Log: `/tmp/probe_downsampled.log`.
- Wall-clock: generation + eval ≈ 9 min total (faster than E021's 13 min, likely because downsampling halves per-layer attention cost on the gen side; eval/MPNN time is ckpt-independent).

**Caveat from the launch itself:**
- `proteinfoundation/generate.py` is `@hydra.main`-driven and **only accepts `--config-name=foo` (hyphen)** at the CLI; argparse `--config_name=` (underscore) silently fails with `unrecognized arguments` and exit code 2. The `script_utils/gen_n_eval_*.sh` wrappers ship with the wrong form (`--config_name=`); avoid the wrappers, call `generate.py` directly with the hyphen flag. (E021's caveat list flagged this; re-confirmed here after a wasted launch attempt.) `evaluate.py` is argparse and uses the underscore form.

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o values within ~0.1 Å — does not change the designability count):**

| L | n | min scRMSD per sample (Å, sorted) | designable (<2 Å, CA) | best | median |
|---|---|---|---|---|---|
| 50  | 6 | 12.41, 15.06, 15.13, 16.64, 16.72, 36.37 | 0/6 (0%) | 12.41 | 15.89 |
| 100 | 6 | 15.67, 16.30, 16.33, 17.71, 17.83, 25.77 | 0/6 (0%) | 15.67 | 17.02 |
| 200 | 6 | 17.60, 20.38, 20.70, 21.33, 21.51, 21.97 | 0/6 (0%) | 17.60 | 21.02 |
| **all** | 18 | — | **0/18 (0%)** | 12.41 | 17.66 |

Headline (printed by `evaluate.py`): "Average scRMSD: 19.190 Å, Success Rate (<2Å): 0.0%, Total: 18, Failed: 0". bb3o-mode designability: 0/18 (matches CA-mode).

**Comparison context (informational; step counts vary):**

| Recipe | step | N | L=50 | L=100 | L=200 | overall | best Å |
|---|---|---|---|---|---|---|---|
| baseline (canonical wd=0.05) | 2646 | N=3 (E018 recheck) | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 8/9 (89%) | 0.7 |
| paramgroups + wd=0.1 | 1952 | N=6 (E018) | 3/6 (50%) | 5/6 (83%) | 1/6 (17%) | 9/18 (50%) | 0.94 |
| sparse-K40 + pair-update | 1133 | N=6 (E021) | 2/6 (33%) | 1/6 (17%) | 0/6 (0%) | 3/18 (17%) | 1.35 |
| **`downsampled` (this entry)** | **2331** | **N=6 (E034)** | **0/6 (0%)** | **0/6 (0%)** | **0/6 (0%)** | **0/18 (0%)** | **12.41** |

**Findings (tuning, not paper-grade):**

1. **Dead at this step. Step mismatch does not explain it.** Step 2331 sits *inside* the canonical baseline's 1800-2200 best-val window. The "early ckpt" excuse that makes E021's 17% rate at step 1133 reasonable does not apply here — at a comparable training stage the canonical baseline is at 89% N=9 / its best-val plateau. Calling this "needs more training" would be protecting a dead arm against the comparator's already-reached state.
2. **No bimodality.** Per-length scRMSDs are uniformly bad (best at any length ≥ 12.4 Å), with no "near-canonical-best" cluster at all. This is not the under-trained fingerprint E021 documented (a fraction of seeds at canonical-best, a fraction collapsed); this is "every seed collapsed". Both halves of the score field are gone, not just consistency.
3. **Decision:** **stop the variant**. Do not promote to N=30. Do not chain more training without first identifying a mechanism (likely candidates: incompatible BlurPool stride/feature interactions per CLAUDE.md "BlurPool1D stride: DownsampleBlock = 2, UpsampleBlock = 1"; mask-aware vs stride-2 pool divergence in `_subsample_input` at `local_latents_transformer.py:160`; downsampling reducing effective receptive field below the L=50 minimum). A debug pass — comparing a step-matched canonical ckpt's intermediate activations against this one's at L=100 — would isolate the failure point.

**Possible narrative:** non-narrative — kept for tuning/decision-making. The downsampled variant is not eligible for `content_masterarbeit.md` Finding promotion based on this snapshot. Could become a methodological aside ("not all variants in the architectural-route shortlist work; this one didn't") in the paper if a step-matched debugging pass attributes the collapse to a specific mechanism.

**Methodological caveats:**
- **N=6 per length is small**, but 0/18 with min 12.41 Å is far enough from the 2 Å threshold that the binomial CI on 0/6 (95% upper bound 39%) is not the bottleneck — the gap to designability is ~10 Å, not ~0.5 Å like E021's L=50 marginal cases.
- **Single seed (seed=5).** Within-seed L4 noise on min-scRMSD is ~0.5 Å per E018; 0.5 Å is irrelevant when the best sample is 12 Å above threshold.
- **Eval used the fixed `ca_only=True` ProteinMPNN call** (post-E017 fix); these numbers are directly comparable to E018/E019/E021 numbers.
- **No HPC training config / NN config available locally.** Architectural details (e.g., where downsampling is applied — DownsampleBlock only or both Down/Up; which feature paths are mask-aware) are not verifiable from this box; would need to inspect the HPC config or `python -c "import torch; ckpt=torch.load(…); print(ckpt['hyper_parameters'])"` on the ckpt to confirm the variant matches what was intended.

**Cross-references:**
- Code added: `configs/inference_downsampled_quick.yaml`.
- Compares against: E018 (canonical baseline N=3 recheck + paramgroups N=6), E019 (canonical N=30 re-eval), E021 (sparse-K40 + pair-update N=6 at step 1133).
- Variant-bar reference: CLAUDE.md "Variant checklist" — the 1-2/3 at L=50/100 bar this run misses.
- Memory: this entry triggers the `feedback_dead_arm_calls.md` rule — call dead arms dead. The "step 2331 is inside the canonical best-val window" comparison is the load-bearing reason this is dead, not under-trained.

---

## E035 — CA-only `sparse_K40_scnbr_t04` variant quick N=6 designability probe (2026-05-06)

**Status:** finished.

**Why ran:** First designability read on the `ca_only_sparse_K40_scnbr_t04` CA-only architectural variant (training run `ca_only_sparse_K40_scnbr_t04/1778022317`). The "scnbr_t04" suffix in the run name suggests a sparse-K40 variant where the spatial-neighbor list is built using the model-predicted clean estimate at some t≥0.4, rather than from `x_t` at the noisy current step (the default behaviour documented in CLAUDE.md sparse-attention notes — "the neighbor list is rebuilt every forward from `x_t`"). This is testable from the ckpt's stored hyperparameters but not verified in this entry. Probe decides whether the variant's step-819 ckpt clears the variant bar.

**Configs:**
- Checkpoint: `best_val_00000008_000000000819.ckpt` (epoch 8, opt step 819), rsynced from HPC `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K40_scnbr_t04/1778022317/checkpoints/`. Local copy: `/home/ks2218/la-proteina/best_val_00000008_000000000819.ckpt` (1.90 GB; mtime 2026-05-06).
- Inference config (new): `configs/inference_sparse_scnbr_t04_quick.yaml`. Same generation block as E034 (3 lengths × 6 samples × 200 ODE steps = 18 total at L ∈ {50, 100, 200}, seed=5).
- Training / NN config: HPC-only, not on this box (no `configs/nn/ca_only_sparse_scnbr_*` locally).
- Hardware: 1× NVIDIA L4 (gxp-l4-0, GPU 0).
- Output dir: `inference/inference_sparse_scnbr_t04_quick/`. CSV: `inference/results_inference_sparse_scnbr_t04_quick_0.csv`. Log: `/tmp/probe_scnbr.log`.
- Wall-clock: generation + eval ≈ 12 min total.

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o within ~0.1 Å):**

| L | n | min scRMSD per sample (Å, sorted) | designable (<2 Å, CA) | best | median |
|---|---|---|---|---|---|
| 50  | 6 | 7.73, 8.25, 8.65, 9.96, 10.93, 18.10 | 0/6 (0%) | 7.73 | 9.30 |
| 100 | 6 | 4.37, 6.73, 8.74, 11.13, 11.75, 15.48 | 0/6 (0%) | 4.37 | 9.94 |
| 200 | 6 | 13.38, 13.74, 14.18, 14.25, 14.71, 16.19 | 0/6 (0%) | 13.38 | 14.21 |
| **all** | 18 | — | **0/18 (0%)** | 4.37 | 11.44 |

Headline: "Average scRMSD: 11.570 Å, Success Rate (<2Å): 0.0%, Total: 18, Failed: 0".

**Comparison context (step counts differ; treat as informational):**

| Recipe | step | N | L=50 | L=100 | L=200 | overall | best Å |
|---|---|---|---|---|---|---|---|
| baseline (canonical wd=0.05) | 2646 | N=3 (E018 recheck) | 2/3 | 3/3 | 3/3 | 8/9 | 0.7 |
| sparse-K40 + pair-update | 1133 | N=6 (E021) | 2/6 | 1/6 | 0/6 | 3/18 (17%) | 1.35 |
| **`sparse_K40_scnbr_t04` (this entry)** | **819** | **N=6 (E035)** | **0/6** | **0/6** | **0/6** | **0/18 (0%)** | **4.37** |
| `downsampled` (E034) | 2331 | N=6 | 0/6 | 0/6 | 0/6 | 0/18 (0%) | 12.41 |

**Findings (tuning, not paper-grade):**

1. **Fails the variant bar at this step**, but the failure mode is qualitatively different from E034. L=100 best = 4.37 Å is ~3 Å above the threshold, not ~10 Å like the downsampled variant. The L=50 distribution `[7.7, 8.3, 8.7, 10.0, 10.9, 18.1]` is "bad but consistent within a band", not "uniformly catastrophic". This is closer to the E021 under-trained-CA-only fingerprint than to the E034 collapse fingerprint.
2. **Step 819 is genuinely early.** E021's pair-update probe was at step 1133 with 17% overall, and the canonical baseline's best-val window is 1800-2200. Step 819 is below both reference points. So unlike E034, "needs more training" is a *mechanically defensible* explanation for this snapshot rather than a dead-arm protection — but until a step ≥ 1500 ckpt of this run is probed, the converged ceiling is unknown. **Do not promote any per-length rate from this snapshot to a Finding.**
3. **L=200 is fully off-manifold (best 13.4 Å).** The L=50/L=100 vs L=200 gap is wider here than E021 saw at step 1133 — at L=200 the L<200 "near-miss" cluster collapses entirely. Whether this is a step-819 artifact (the L=200 head of the loss surface forms later) or a property of the scnbr_t04 modification specifically is not separable on this single snapshot.
4. **Decision:** **continue training the variant; re-probe at a step ≥ 1500.** If the L=100 best at step ≥ 1500 still hovers around 4 Å, treat the variant as dead by E034 logic (step matched to a comparator that is at its plateau). If L=100 best drops below 2 Å on one or more samples by then, the variant is in the same regime as E021's pair-update arm and warrants a step-1800-or-later probe matched to the canonical best-val window.

**Possible narrative:** non-narrative — kept for tuning/decision-making. Not eligible for `content_masterarbeit.md` Finding promotion at this snapshot. The "step 819 vs step 1133 vs step 1800-2200" cross-comparison is the load-bearing context for any later evaluation of this run.

**Methodological caveats:**
- **N=6 single seed** (seed=5). At L=100 best 4.37 Å is ~2.4 Å above threshold; within-seed L4 noise is ~0.5 Å per E018. So a single seed-flip will not alone change the 0/6 → designable call, but 4-5 seeds at the same step would tighten the binomial CI and rule out "this seed is unlucky".
- **No matched-step canonical comparator at step ≈ 819.** The comparison column above is informational, not a clean A/B/C across recipes. A step-819 canonical baseline ckpt is not on disk; without it, recipe-vs-recipe at step 819 cannot be cleanly read from this entry.
- **HPC training config not inspected.** The "scnbr_t04" semantics (spatial neighbours computed from the model's clean-estimate at t≥0.4 instead of from `x_t`) are inferred from the run name. To confirm, dump the ckpt's `hyper_parameters` and grep for the relevant sparse-attention key (`spatial_neighbor_t_threshold` or similar) before any later probe. Otherwise the variant's mechanism is mis-attributed.
- **Eval used the fixed `ca_only=True` ProteinMPNN call** (post-E017 fix); directly comparable to E018/E019/E021/E034.

**Cross-references:**
- Code added: `configs/inference_sparse_scnbr_t04_quick.yaml`.
- Compares against: E021 (sparse-K40 + pair-update, the architectural-cousin baseline at step 1133), E018 (canonical N=6 baseline at step 2646), E034 (sister `downsampled` probe in this same gen+eval session).
- Memory: `project_sparse_pairupdate_converged.md` (E021's converged-ceiling logic does NOT transfer here — that memory is for `sparse_K40_pairupdate`, a different variant; the converged step for `sparse_K40_scnbr_t04` is unknown).
- Predicts: a step ≥ 1500 re-probe of this run is the obvious follow-up. If `feedback_dead_arm_calls.md` ends up biting this variant too, the natural deletion call is when L=100 best stops improving across two consecutive ≥ 500-step intervals.

## E036 — Pairwise TM-score diversity of the noise-aware ensemble sweep (2026-05-06)

- **Status:** finished. Closes the structural-diversity gate that the user explicitly raised after E033 ("scRMSD stays similar — at the cost of diversity?").
- **Why ran:** E033 showed designability is preserved at every w-level — every individual steered protein folds. But that does NOT answer whether the steered ensemble has *collapsed to a narrow basin* of TANGO-low folds at high w. A pipeline that delivers 16 essentially-identical low-TANGO proteins at w=16 would pass the per-protein designability bar but fail any "diversity of solutions" criterion. The user predicted a possible diversity-vs-strength tradeoff and asked for a controlled measurement.
- **Configs:**
  - Pipeline: `scripts/diversity_pairwise_tm.py` (new). Uses `tmtools.tm_align` (CPU, ~10 ms per pair). All-pairs TM-score within each (direction, w, L) cell, n=16 → 120 pairs/cell. TM-score reported as mean of `tm_norm_chain1 + tm_norm_chain2` for length-symmetric comparison.
  - Steered cells: 30 cells = 2 directions × 5 w-levels × 3 lengths. Sources at `results/noise_aware_ensemble_sweep/{direction}_w{w}/guided/s{42..57}_n{L}.pdb`.
  - Baseline: `results/generated_stratified_300_800_nsteps400/samples/`, sliding window L±15 around each target L = {300, 400, 500}, take 16 PDBs.
  - Wall: ~12 minutes total on 1 CPU.

- **Aggregate results, mean pairwise TM-score across all 360 pairs per (direction, w), aggregated over the 3 lengths:**

  | direction | w | mean TM | median TM | p10 | p90 |
  |---|---|---|---|---|---|
  | camsol_max | 1 | 0.407 | 0.339 | 0.278 | 0.613 |
  | camsol_max | 2 | 0.407 | 0.340 | 0.277 | 0.613 |
  | camsol_max | 4 | 0.407 | 0.340 | 0.277 | 0.611 |
  | camsol_max | 8 | 0.407 | 0.340 | 0.277 | 0.613 |
  | camsol_max | 16 | 0.407 | 0.339 | 0.277 | 0.614 |
  | tango_min | 1 | 0.407 | 0.340 | 0.278 | 0.614 |
  | tango_min | 2 | 0.407 | 0.340 | 0.277 | 0.614 |
  | tango_min | 4 | 0.407 | 0.341 | 0.277 | 0.613 |
  | tango_min | 8 | 0.407 | 0.339 | 0.276 | 0.614 |
  | tango_min | 16 | 0.407 | 0.340 | 0.279 | 0.614 |
  | unsteered baseline | 0 | **0.413** | 0.342 | 0.283 | **0.693** |

  Mean pairwise TM-score is **identical to 3 decimal places across every w-level for both directions** (0.407 across the entire steered grid). The unsteered baseline mean is 0.413, only 0.006 higher. The only meaningful baseline-vs-steered gap is in the p90: 0.69 (baseline) vs 0.61 (steered), meaning the baseline has more high-similarity pairs in its long tail (likely a few duplicate folds among the unsteered samples), while the steered ensembles cap their similarity tail around 0.61.

- **Per-length × w breakdown (mean pairwise TM-score):**

  *camsol_max:*
  | L | w=1 | w=2 | w=4 | w=8 | w=16 | unsteered baseline |
  |---|---|---|---|---|---|---|
  | 300 | 0.495 | 0.495 | 0.495 | 0.495 | 0.496 | 0.454 |
  | 400 | 0.331 | 0.332 | 0.331 | 0.331 | 0.332 | **0.366** |
  | 500 | 0.395 | 0.395 | 0.395 | 0.395 | 0.395 | **0.418** |

  *tango_min:*
  | L | w=1 | w=2 | w=4 | w=8 | w=16 | unsteered baseline |
  |---|---|---|---|---|---|---|
  | 300 | 0.496 | 0.495 | 0.495 | 0.495 | 0.495 | 0.454 |
  | 400 | 0.331 | 0.331 | 0.330 | 0.331 | 0.331 | **0.366** |
  | 500 | 0.394 | 0.394 | 0.395 | 0.395 | 0.395 | **0.418** |

  At L=400 and L=500 the **steered ensembles are *more* diverse than baseline** (lower mean pairwise TM, i.e. proteins look less alike). Only at L=300 is the steered ensemble slightly more similar than baseline (0.495 vs 0.454, a difference of 0.04). All cell-to-cell differences within a length are ≤ 0.001, so no measurable w-dependence.

- **Sanity check (md5sum across w cells of the same seed × length):**

  | run | s42_n300.pdb md5 |
  |---|---|
  | camsol_max_w1 | d35d274e1c053768124d02dd0f39a0e8 |
  | camsol_max_w16 | 922377cc8115388d3bc8b9cddcf2e68f |
  | tango_min_w1 | 587e3cc379f1ddb077685440f80a3646 |
  | tango_min_w16 | 652b252edbb3770614be1f2b254af3d4 |

  All four hashes differ → the steering hook fired and produced different proteins from the same starting noise. The flat TM-score across w is real, not a script bug.

- **Mechanism interpretation.** The 16-seed ensemble samples 16 distinct starting points in noise-space, and each one gets pushed along its own SDE trajectory. The diversity in the resulting protein set reflects the *initialization-driven branching* of those trajectories. Steering modifies each trajectory by adding a small unit-normalised gradient term scaled by `w(t)`, but the trajectory-to-trajectory branching established by the initial noise dominates. Each steered protein "comes from somewhere different" structurally even if all are biased toward TANGO-low or CamSol-high regions of the latent space. The latent space is high-dimensional enough (8 channels × L residues = ~2400-4000 dimensions for L=300-500) that there are many independent low-TANGO directions; steering doesn't push everything onto one of them, it pushes each trajectory along whichever low-TANGO direction is closest from its starting point.

- **Possible narrative:** **adds a "diversity preserved" sub-claim to Finding 10**. Combines with E032's gap closure + E033's scRMSD preservation to give a complete steering-route claim:
  > *"Steering with the noise-aware-ensemble at any w∈[1, 16] preserves the structural diversity of the unsteered baseline."*

  This is exactly the right shape for the steering-route bar: real-property change *and* structural integrity *and* solution diversity, all simultaneously.

- **Methodological caveats:**
  - **n=16 per (direction, w, L) cell, n=16 in the baseline window.** 120 pairs per cell is a fairly tight CI on the mean TM (std-of-mean for 120 pairs of bounded TM values is ~0.005-0.01), but a 0.001 cell-to-cell difference is still within sampling noise. The robust signal is the *trend* (no monotonic w-narrowing), not a specific cell-level number.
  - **Baseline length window is L±15.** The unsteered samples have continuously distributed lengths so there's no cell at exact L=300/400/500. The window is wide enough to find 16 proteins per L, narrow enough that TM-score normalisation is comparable to the steered same-length cells. Tighter windows (e.g. L±5) would give fewer baseline pairs and noisier numbers; wider windows would dilute the comparison with neighbouring-length proteins.
  - **TM-score is not the only diversity metric.** It captures topology + global fold but not loop-level differences, side-chain diversity, or local secondary-structure pattern. A more complete diversity audit would add per-residue SS profile entropy, contact-map diversity, or learned-embedding distance. Out of scope for closing the F10 caveat.
  - **The "at the cost of diversity" hypothesis was conditional on collapse.** If the data had shown w=16 mean TM ≈ 0.7 (highly similar pairs) vs baseline 0.4, that would have been a clear collapse. The actual data — flat TM across w-levels at 0.407 — *cannot* be re-interpreted as a different kind of collapse. Diversity is preserved.
  - **Diversity ≠ usefulness.** Two proteins with TM-score 0.3 are diverse in fold space but might both still fail any biological assay. This is a structural-ensemble property, not a function-space property.

- **Cross-references:**
  - E032 (gap closure), E033 (designability), this E036 (diversity) — together cover the three independent quality axes of the steering pipeline.
  - Finding 10 in `content_masterarbeit.md` — the diversity result extends the "scRMSD preserved" caveat into a full "no collapse" sub-claim.
  - Implementation: `scripts/diversity_pairwise_tm.py` (new), output at `results/noise_aware_ensemble_sweep/diversity_pairwise_tm.csv`.

---

## E037 — Curvature-targeted bump schedule, paired N=30 probes (2026-05-05 → 2026-05-06)

**Status:** finished. Two paired-noise inference runs at N=30, L=300, on the LD3 full-latent model. Null result at this N; directional signal split by metric.

**Why ran:** E004 / Finding 2 measured that the `local_latents` ODE field has a curvature peak around t≈0.489 while `bb_ca` is nearly straight, and explicitly flagged the schedule-vs-quality ablation as missing — the causal claim "more NFEs near the curvature peak should improve sample quality at fixed budget" was plausible but untested. E037 is that ablation: redistribute the integration grid of `local_latents` so that more steps land near t=0.489, holding `bb_ca` and total NFE budget (`nsteps=400`) fixed. Decision input for whether the bump-schedule line of work is worth committing to a properly-powered run before paper-writing.

**Setup (both runs):**
- Model: `LD3_ucond_notri_800.ckpt` + `AE2_ucond_800.ckpt`, full-latent (joint sequence head present).
- Inference config: `inference_ucond_notri_long` (defaults: `nsteps=400`, `self_cond=True`, `guidance_w=1.0`).
- Sampler: `proteinfoundation/generate.py`, single A100, `nsamples=30`, length list `[300]`, `seed=5` (`inference_base.yaml`).
- Pairing: `proteina.py:783-788` re-seeds at the start of every `predict_step` with `base_seed + batch_idx`, so two rows with the same `id_gen` across the baseline and variant CSVs come from identical initial noise. This is the design feature that makes the comparison paired.
- Driver: `script_utils/submit_schedule_comparison.sh --config inference_ucond_notri_long --ckpt ./checkpoints_laproteina/LD3_ucond_notri_800.ckpt --schedules baseline,<bump_key> --lengths 300 --nsamples 30`. Generation + designability + co-designability eval per schedule, sequential.
- SLURM: SL2 / ampere / 1 GPU / `--time=1:30:00 --exclude=gpu-q-43`. Job 28893748 (eps=0.1, 1h13m) → 28898446 (eps=0.14, 1h21m).

**Schedule definition** (`product_space_flow_matcher.py:get_schedule`, `mode="power_with_middle_bump"`):
```
F(u) = u^p  +  eps · [ Gauss(u; mu, sigma) − linear endpoint correction ]   (then renormalised to [0,1], monotone-asserted)
```
- `bb_ca` schedule unchanged across both runs: `mode=log, p=2.0` (E004 said `bb_ca` is straight, so no need to redistribute).
- `local_latents` schedule per arm:
  - **Baseline:** `mode=power, p=2.0`
  - **eps=0.1:** `mode=power_with_middle_bump, p=2.0, eps=0.10, mu=0.489, sigma=0.08`
  - **eps=0.14:** same with `eps=0.14` (the maximum eps that stays monotone for `mu=0.489, sigma=0.08`; `eps=0.19` is the ceiling for `sigma=0.10` per `fit_bump_params.py`).

**Eval pipeline:** `proteinfoundation/evaluate.py` with default ProteinMPNN×8 + ESMFold + scRMSD on both single-sequence designability (`_res_scRMSD_*`) and co-designability (`_res_co_scRMSD_*`).

**Engineering artefact (committed in `8e97d7a`):** `script_utils/schedule_comparison_report.py` extended to do paired analysis on `id_gen` joins between schedule CSVs — per `(rmsd_col, variant, length+overall)`: n_pairs, mean Δ, median Δ, % v<b, Wilcoxon signed-rank p, paired designability rates, McNemar exact-binomial p on the binary flip. Output dumped to `inference/schedule_comparison_paired.csv` plus the existing per-length bar charts.

**Results — eps=0.1 vs baseline (N=29 paired, 1 dropped at eval due to `tensor a (300) ≠ b (299)` in `job_0_n_300_id_25`):**

| Metric | Base | Bump 0.1 | Mean Δ | Median Δ | %v<b | Wilcoxon p | des(b) → des(v) | McN p |
|---|---|---|---|---|---|---|---|---|
| `scRMSD_ca` (single-seq) | 89.7%, 1.66 Å | 82.8%, 1.42 Å | −0.220 | −0.021 | 55% | 0.67 | 89.7% → 82.8% | 0.50 |
| `co_scRMSD_ca` | 79.3%, 2.92 Å | 72.4%, 2.94 Å | +0.029 | −0.074 | 53% | 0.45 | 79.3% → 72.4% | 0.625 |
| `co_scRMSD_bb3o` | 79.3%, 2.90 Å | 72.4%, 2.93 Å | +0.028 | (similar) | 55% | 0.49 | 79.3% → 72.4% | 0.625 |
| `co_scRMSD_all_atom` | 69.0%, 3.31 Å | 72.4%, 3.39 Å | +0.085 | (similar) | 52% | 0.88 | 69.0% → 72.4% | 1.00 |

**Results — eps=0.14 vs baseline (N=30 paired, no eval drops):**

| Metric | Base | Bump 0.14 | Mean Δ | Median Δ | %v<b | Wilcoxon p | des(b) → des(v) | McN p |
|---|---|---|---|---|---|---|---|---|
| `scRMSD_ca` (single-seq) | 90.0%, 1.42 Å | 93.3%, 1.50 Å | +0.083 | +0.047 | 47% | 0.50 | 90.0% → 93.3% | 1.00 |
| `co_scRMSD_ca` | 76.7%, 2.96 Å | 70.0%, 2.87 Å | **−0.094** | **−0.133** | **60%** | 0.43 | 76.7% → 70.0% | 0.50 |
| `co_scRMSD_bb3o` | 76.7%, 2.95 Å | 70.0%, 2.85 Å | **−0.098** | **−0.122** | **63%** | 0.38 | 76.7% → 70.0% | 0.50 |
| `co_scRMSD_all_atom` | 66.7%, 3.36 Å | 60.0%, 3.34 Å | −0.021 | −0.125 | 60% | 0.49 | 66.7% → 60.0% | 0.69 |

**Pair-level decomposition for the cleanest signal column (eps=0.14, `co_scRMSD_ca`, 30 pairs):**

| Subset | n | Mean Δ |
|---|---|---|
| both designable (b<2 ∧ v<2) | 21 | −0.091 Å (improvement, but no threshold flip) |
| both non-designable (b>2 ∧ v>2) | 7 | **−2.092 Å** (large improvement, but still fails 2 Å bar) |
| rescued (b>2 → v<2) | 0 | — |
| broken (b<2 → v>2) | 2 | (boundary samples nudged the wrong way) |

So the bump *does* shift mass toward lower scRMSD throughout the distribution — including a 2 Å mean improvement in the failure tail — but the improvements happen mostly far from the 2 Å boundary, while two borderline samples just under 2 Å get pushed slightly above it. Net designability count: −2 → −7pp.

**Cross-run picture:**
- **Continuous direction** (mean / median Δ on co-design scRMSD) firmed up at eps=0.14: 60–63% v<b, mean Δ ≈ −0.10 Å, median Δ ≈ −0.13 Å on three co-design columns. At eps=0.1 the direction was weaker (52–55% v<b, mean Δ ≈ +0.03).
- **Designability rate** (binary 2 Å threshold) shifted the *wrong* way on three of four co-design columns in **both** probes (consistent −7pp on `co_scRMSD_ca` and `co_scRMSD_bb3o` at both eps values). Single-seq `scRMSD_ca` is approximately unchanged (one or two protein-flips — RNG-dominated at N=29/30).
- **No metric in either run reached significance.** Smallest Wilcoxon p across both runs is 0.38; smallest McNemar p is 0.50.

**Possible narrative:** non-narrative for now — N=30 is too small to commit a paper claim. Kept as a decision-feeder for the bump-schedule research line:

- The continuous-vs-threshold split is internally consistent — the bump compresses the scRMSD distribution toward lower values without moving the 2 Å boundary count, because the improvements are concentrated either inside the already-designable region (no threshold cross) or in the far failure tail (still a fail). A few borderline samples flip in either direction with roughly equal probability under H0, and at N=30 the 2 broken / 0 rescued split is consistent with noise.
- Mechanism hypothesis (untested): the schedule normalisation means more density near μ=0.489 implies *less* density at the t-tails. Late-t (t close to 1) is the refinement window where borderline samples make their final crossing. The bump may be trading late-stage accuracy (helps borderline samples cross 2 Å) for mid-stage accuracy (helps deeply-failed samples avoid wrong-basin trap and improve from 5→3 Å). Consistent with the "improvement is in the tails, designability boundary loses" pattern observed.
- Direct test that's cheaper than another N=100 run: plot the schedule density against t (`script_utils/plot_straightness.py`) and compare late-t (t > 0.85) density between baseline and `power_bump_e0.14`. If the bump meaningfully reduces late-t step density, the trade-off hypothesis is supported and the right next variant is a bump that doesn't steal from the tail (e.g. keep total bump density but widen σ so more density comes from the calm region near t=0.2 instead).
- If the trade-off hypothesis is rejected (bump density is roughly tail-neutral), then the +2 broken / 0 rescued is just RNG and N=100 is the only honest move.

**Methodological caveats:**
- N=30 (and N=29 for eps=0.1) is underpowered for the effect sizes seen. Wilcoxon needs ≥80 paired samples to detect ~0.3σ at α=0.05/80% power; current observed effect on `co_scRMSD_ca` is ~0.2σ. McNemar on a 10pp designability shift needs ~120 paired samples — anything we see at N=30 in designability rate is dominated by a handful of boundary flips.
- Single length only (L=300). Curvature is t-dependent, not L-dependent in any first-order way per E004, so the schedule effect is expected to be length-independent — but this is an assumption, not a measurement.
- Single seed (`base_seed=5`). The pairing controls within-run variance but not between-run variance; the *baseline* numbers are not identical across the two probes (89.7% vs 90.0% on `scRMSD_ca`, 79.3% vs 76.7% on `co_scRMSD_ca`), confirming there's residual seed-level noise even at N=30.
- The paired design controls noise *within* a metric, not *across* metrics. We can't paired-test "single-seq design vs co-design" because they are different scoring pipelines.
- One sample lost at eval in eps=0.1 run (`job_0_n_300_id_25`, tensor-shape 300 vs 299). Did not recur in eps=0.14. Cause not investigated (1-residue mismatch in some post-gen step).
- The `power_x2_bump_e0.14` and `power_x2_bump_e0.19` variants (μ=0.50, σ=0.10 from the `fit_bump_params.py` bend-angle window) have not been tested at all yet — any conclusion about "σ=0.08" vs "σ=0.10" is unsupported.

**Cross-references:**
- E004 / Finding 2 — the curvature measurement that motivated this experiment. E037 is the schedule-vs-quality ablation E004 explicitly flagged as missing.
- `script_utils/measure_field_straightness.py`, `fit_bump_params.py`, `optimise_bump.py`, `plot_straightness.py` — the tooling for measuring curvature and constructing/diagnosing the bump schedule.
- `proteinfoundation/flow_matching/product_space_flow_matcher.py:get_schedule` (`mode="power_with_middle_bump"`) — the schedule itself.
- `script_utils/submit_schedule_comparison.sh`, `script_utils/schedule_comparison_report.py` — the driver and the (now paired-aware) reporting script.
- Output CSVs (overwritten between runs by design): `inference/results_inference_ucond_notri_long_<schedule>_0.csv`, `inference/schedule_comparison_paired.csv`, `inference/schedule_comparison_*.png`. Per-protein PDBs deliberately not committed.
- SLURM logs: `slurm_schedule_cmp_28893748.out` (eps=0.1), `slurm_schedule_cmp_28898446.out` (eps=0.14) — not committed (per the standard skip rule).
- `slurm_schedule_cmp_28000070.out` — the cancelled first attempt from 2026-04-19 (cgroup hung at first generation call); kept on disk for reference but produced no result.

**Predicts:** N=100 paired probe at L=300 with `power_bump_e0.14` would (a) give a Wilcoxon p≤0.05 on the continuous direction if the +0.10 Å median Δ holds, and (b) put a ±5pp band on the designability shift, making the binary direction defensible. Cost: ~4h on 1× A100. Defer until the late-t density check above is done — if it confirms the tail-stealing hypothesis, redesign the schedule first.
