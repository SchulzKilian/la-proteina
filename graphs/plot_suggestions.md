# Plot & Figure Suggestions for the Masterarbeit

Curated visual content proposals derived from `content_masterarbeit.md` (Findings F1–F13) and `experiments.md` (E001–E079). Plot types are starting suggestions only; agent 3 will finalize the exact chart form.

Ordering follows the proposed thesis flow:
1. Thesis-overview & methodological scaffolding (P01–P03)
2. Latent-space characterization (P04–P09)
3. CA-only baseline & recipe failures (P10–P14)
4. Architectural-route variants & negative results (P15–P19)
5. Steering route — predictor diagnostics, dose-response, real properties (P20–P28)
6. Diagnostic/audit figures (P29–P32)

---

## Index

| ID | Title | Visualizes | Plot type |
|---|---|---|---|
| P01 | Two-route thesis overview map | Thesis intent (architectural vs steering) | Schematic / diagram |
| P02 | Designability vs. nsteps cliff (LD3+AE2) | nsteps=200 vs nsteps=400 (22 Å → 0.8 Å) | Paired scatter / bar |
| P03 | Master experiment timeline | E001–E079 chronology | Gantt / dotplot |
| P04 | Multi-task property predictor: 5-fold R² panel | F1 / E001 | Horizontal bar with error |
| P05 | Probe convergence-epoch hierarchy | F1 additional observation | Lollipop / grouped bar |
| P06 | Capacity probing R² matrix (Class A vs Class B) | F4 / E002 | Heatmap |
| P07 | Latent geometry summary (utilization, multimodality, length sensitivity) | F3 / E003 | 4-panel composite |
| P08 | Flow-field curvature per channel | F2 / E004 | Dual-line + inset |
| P09 | Sidechain manifold perturbation: latent vs coord | F6 / E011 | Paired box / strip |
| P10 | CA-only baseline canonical val-loss trajectory | F5 / E008 / E009 | Multi-line over opt steps |
| P11 | AdaLN-Zero gate-norm collapse table | F5 / E009 / E015 | Table |
| P12 | wd × design quality (N=30 four-arm) | F5 / F7 / E014 / E019 | Grouped bar |
| P13 | Per-layer weight-norm ratio (v2 / canonical) | F5 / E009 | Sorted bar with highlight |
| P14 | Val loss vs designability decoupling | F5 / F7 / F11 / E054 | Scatter w/ annotations |
| P15 | Variant designability ladder | E019 / E021 / E034 / E039 / E053 / E055 / E056 / E077 | Grouped bar by L |
| P16 | Per-t validation loss across CA-only variants | F11 / E043 | Multi-line |
| P17 | Hybrid handover kink figure | E040 / E041 | Twin scatter + bar |
| P18 | Sparse-vs-dense compute crossover | E073 / E074 | Line + crossover marker |
| P19 | Dead-arm gallery table | E034 / E053 / E055 / E056 / E058 | Table |
| P20 | Predictor:real TANGO gap dose-response | F10 / E028 / E032 / E050 | Dual-line with shaded gap |
| P21 | Negative-results ladder for predictor fixes | F10 / E028 / E029 / E030 / E031 / E032 | Horizontal lollipop |
| P22 | Codesign vs Δ-real Pareto frontier | F10 / F13 / E066 / E076 | Pareto scatter |
| P23 | Real CamSol per-length effect | F13 / E076 | Strip / violin by L |
| P24 | Steering schedule + diagnostic anatomy | F10 / steering hook design | Anatomy diagram |
| P25 | Per-length steering response heterogeneity | E066 / E067 / E072 / F13 | Heatmap |
| P26 | Multi-objective cocktail comparison | E068 / E072 | Grouped bar |
| P27 | Structural ensemble diversity (TM-score) | F10 / E036 | Box / strip per cell |
| P28 | Steering walltime overhead | E069 | Single bar / log line |
| P29 | Alphabet collapse: AA composition gen vs natural | F8 / E020 / E026 | Diverging bar |
| P30 | Length scaling of alphabet collapse (F9) | F9 / E020+E026 follow-up | Multi-line / heatmap |
| P31 | Aromatic count vs placement asymmetry | F8 (c) / E023 / E026 | Slope chart |
| P32 | Thermal-stability proxy contradiction | F8 (b.i) / E020 / E026 | Diverging bar |
| P33 | Dense attention routing audit | F12 / E059 / E060 / E061 | 3-panel composite |
| P34 | Variant ablation summary table | F5 / F7 / F11 / F12 + variant E-IDs | Table |
| P35 | Finding-to-experiment cross-reference | All Findings + E-IDs | Mapping table |

---

## Per-plot details

### P01 — Two-route thesis overview map

- **Visualizes:** Thesis intent (architectural route vs steering route, with the L ≥ 300 designability bar).
- **Plot type idea:** Schematic / block diagram (two arrows from baseline; nodes = Findings F1–F13 grouped by route).
- **Data:** No empirical data — synthesized from `content_masterarbeit.md` "Thesis intent" section. Group F1/F2/F3/F6 → steering scaffold; F4 → probe taxonomy; F5/F7 → baseline constraint; F8/F9 → joint-head methodology; F10/F13 → steering route deliverable; F11/F12 → architectural diagnostics.
- **Why it matters:** Frames the whole thesis so the reader can map every figure to one of the two routes.
- **Caveats:** Editorial/synthetic figure. Make sure to call out which Findings remain "scaffold" vs which clear the L ≥ 300 bar.

### P02 — Designability vs nsteps cliff (LD3+AE2)

- **Visualizes:** The hard-rule nsteps=400 phenomenon (CLAUDE.md: 22.5 Å @ nsteps=200 → 0.80 Å @ nsteps=400 at identical seed/model/L=300).
- **Plot type idea:** Paired-seed scatter (x = nsteps, y = scRMSD on log scale) or before/after bars.
- **Data:** From CLAUDE.md and the E028 → E033 → F10 chain (`results/generated_stratified_300_800_nsteps400/` and the equivalent pre-flip generations). Also `results/property_comparison_nsteps400/` vs `results/property_comparison/`.
- **Why it matters:** Justifies why every downstream evaluation uses nsteps=400 and why E020/E026/E021/E034/E035/E038/E039/E040/E041 had to be regen'd or flagged.
- **Caveats:** Cliff originally shown on a single-protein single-seed L=300 case (per CLAUDE.md). If a wider replication exists, prefer the full distribution; otherwise label as single-protein illustrative.

### P03 — Master experiment timeline

- **Visualizes:** The whole E001–E079 lab-record arc grouped by route (steering scaffold / baseline / variants / steering deliverable / diagnostics).
- **Plot type idea:** Gantt/dotplot — x = date, y = experiment ID, color = route/finding tag.
- **Data:** Index table at the top of `experiments.md`.
- **Why it matters:** Orients the reader through the lab notebook without forcing them to read the index in full.
- **Caveats:** Some entries (E010, E013, E047, E058) are in-progress; mark accordingly. Dates span 2026-04-21 → 2026-05-23.

### P04 — Multi-task property predictor: 5-fold R² panel

- **Visualizes:** F1 / E001 — 5-fold CV val R² across the 13 developability properties (mean R² ≈ 0.88).
- **Plot type idea:** Horizontal bar chart sorted by mean R², with per-fold dots overlaid (especially SWI 0.38–0.98 spread).
- **Data:** Table in F1 (`content_masterarbeit.md` lines 44–59). Source run: `laproteina_steerability/logs/multitask_t1/20260421_064011/`.
- **Why it matters:** Establishes that the latent space is information-rich and the steering route is mechanistically grounded.
- **Caveats:** Most properties have only Fold 0 in the curated table (5-fold run was in progress when written). SWI's fold-variance must be displayed honestly (narrow target std=0.01 → metrically unstable R²).

### P05 — Probe convergence-epoch hierarchy

- **Visualizes:** F1 additional observation — epoch-to-90%-final R² grouped into fast/medium/slow regimes (sequence-derived vs structure-derived).
- **Plot type idea:** Lollipop chart sorted by epoch-to-90%, colored by regime (fast/medium/slow).
- **Data:** Table in F1 (`content_masterarbeit.md` lines 89–103).
- **Why it matters:** Visually argues that the latent encodes coarse residue chemistry quickly and spatial properties slowly — the basis for F4's Class A/B split.
- **Caveats:** Single-fold convergence speeds; SWI's "fast on folds 2–4 only" is an outlier and should be annotated.

### P06 — Capacity probing R² matrix (Class A vs Class B)

- **Visualizes:** F4 / E002 — 7-probe capacity ladder × 13 properties; Class A (per-residue MLP suffices) vs Class B (only attention unlocks).
- **Plot type idea:** Heatmap (rows = properties grouped by class, columns = probes ordered by param count); cell colored by R²; bold cells where Tx clears 0.80.
- **Data:** Table in F4 (`content_masterarbeit.md` lines 264–280). Run dir: `laproteina_steerability/logs/capacity_probing/20260421_191747/`.
- **Why it matters:** Visually proves the probe-family ≠ probe-size distinction; supports the steering design choice to use small per-residue heads for Class A and attention for Class B.
- **Caveats:** Single fold (Fold 0); SWI's Fold 0 = 0.38 should carry an asterisk pointing at F1's 5-fold mean (0.77).

### P07 — Latent geometry summary

- **Visualizes:** F3 / E003 — utilization (PR=7.69/8), kurtosis (dim 3, 7 multimodal), within-vs-between variance (~100× larger), length sensitivity (r≤0.16).
- **Plot type idea:** 4-panel composite — (a) PCA spectrum bars; (b) per-dim KDE highlighting dims 3 and 7; (c) bar of within/between variance ratio per dim; (d) scatter of dim-3 protein-mean vs length.
- **Data:** Tables in F3 (`content_masterarbeit.md` lines 168–222); CSVs at `laproteina_steerability/outputs/tables/`.
- **Why it matters:** Foundational for the steering route — proves the latent supports protein-level averaging and is locally disentangled.
- **Caveats:** N=56K proteins. The "categorical clusters on dims 3/7" reading is indirect (negative kurtosis only); no direct mixture-fit yet.

### P08 — Flow-field curvature per channel

- **Visualizes:** F2 / E004 — straightness ratio R = 0.94 (`bb_ca`) vs 0.51 (`local_latents`), and step-length vs t.
- **Plot type idea:** Twin-axis line plot (per-step displacement vs t), one curve per channel; inset showing R values as bars.
- **Data:** `checkpoints_laproteina/straightness_ld3.json`. F2 table (lines 131–134).
- **Why it matters:** Mechanistic basis for the curvature-aware sampling schedule (E037, F2 follow-up) and the nsteps cliff at low N.
- **Caveats:** Computed at L=400, nsamples=8, nsteps=800 — single operating point. Don't overgeneralize to other L.

### P09 — Sidechain manifold perturbation: latent vs coord

- **Visualizes:** F6 / E011 — per-protein paired scRMSD (latent < coord at every k); near-invariance of latent arm vs growth of coord arm.
- **Plot type idea:** Paired box/strip — x = k, y = scRMSD; one stripe per arm; lines connecting paired proteins.
- **Data:** F6 numbers (lines 471–504). `inference/manifold_tidy_eval_manifold_perturbation.csv` (170 rows).
- **Why it matters:** Direct evidence the AE decoder is contractive — supports the steering route's "latent perturbations stay on-manifold" premise.
- **Caveats:** N=17 length-stratified proteins, single seed, AE1 (512-res). Does not transfer mechanically to AE2 / 300–800. 3 outlier proteins (4v8y_BO/4v88_BO/4v88_DO) at the 17 Å floor in both arms (ESMFold modeling error, not perturbation signal).

### P10 — CA-only baseline canonical val-loss trajectory

- **Visualizes:** F5 / E008 / E009 — val loss vs optimizer step for canonical (wd=0.05) and v2 (wd=0.1 + cosine).
- **Plot type idea:** Multi-line plot; x = opt step (1000–3000), y = val loss; canonical, v2, wd=0 curves overlaid; mark best-val points with vertical dashed lines.
- **Data:** F5 head-to-head table (lines 365–384). Wandb runs: `d1k1587u` / `jeponiu5` / `0fnyfbi9` (canonical chain), `9jp15of2`/`5rftn43a`/`43xxlbzt` (v2). Memory: wandb numbers are not directly comparable across runs — rely on the per-t paired re-eval where possible (E054 / `results/per_t_val/`, `results/aggregate_val_seeds/`).
- **Why it matters:** Sets up the "v2 looks better by val" deception that F5 then unmasks.
- **Caveats:** Wandb val_loss/loss_epoch is misleading across runs (E054 / `feedback_wandb_val_loss_not_comparable.md`); say so in the caption or use the re-evaluated paired curves from `run_per_t_val.py`.

### P11 — AdaLN-Zero gate-norm collapse table

- **Visualizes:** F5 / E009 — 10 worst-affected layers' L2 norm ratios (v2/old) for `transformer_layers.{7..13}.{mhba,transition}.scale_output.to_adaln_zero_gamma.0.weight`.
- **Plot type idea:** Table (clean numerical readout); optionally a per-layer sorted bar plot as a companion.
- **Data:** F5 table at lines 415–426.
- **Why it matters:** The mechanism evidence for F5's central claim — uniform-wd AdamW crushes the conditioning gates.
- **Caveats:** Correlational evidence; the causal ablation (Variant A/B in F5 follow-up) was planned but not yet run as of the writing of `content_masterarbeit.md`.

### P12 — wd × design quality (N=30 four-arm)

- **Visualizes:** F5 / F7 / E014 / E019 — designability rate per length × recipe (canonical wd=0.05 vs v2 wd=0.1 vs wd=0 vs sparse K40).
- **Plot type idea:** Grouped bar (one group per L ∈ {50, 100, 200}, four bars per group); annotate with N=30 rates and total /90.
- **Data:** F7 N=30 table at lines 645–650 of `content_masterarbeit.md`. Generated PDBs under `results/` (paths in E019 entry).
- **Why it matters:** Shows the "val-loss-improving recipe destroys samples" headline at a glance.
- **Caveats:** Single-seed N=30. Each arm at its own best ckpt (not matched opt step). v2 / wd0 / sparse-K40 at step 1638/1259 are under-trained relative to canonical step 2646.

### P13 — Per-layer weight-norm ratio (v2 / canonical)

- **Visualizes:** F5 / E009 — distribution of layer-wise norm ratios over 164 layers ≥10k params; mean 0.92, min 0.26 at AdaLN-Zero gates.
- **Plot type idea:** Sorted bar (one bar per layer, x = sorted rank, y = ratio v2/old) with the 10 AdaLN-Zero gate layers highlighted in a distinct color.
- **Data:** F5 prose at lines 411–428.
- **Why it matters:** Companion to P11 — shows that the collapse is localized to gate layers, not a global weight shrink.
- **Caveats:** Single-checkpoint comparison (v2 step 2078 vs canonical step 2646 post-uptick).

### P14 — Val loss vs designability decoupling

- **Visualizes:** F5 / F7 / F11 / E054 — y = best val_loss/loss_epoch (wandb), x = pooled designability rate; or alternatively per-length scRMSD vs val loss.
- **Plot type idea:** Scatter; one point per ckpt; annotate canonical_2646, v2_2078, wd0_1638, sparse_K40_1259, lastv2_1952, scnbr_t04_1133, downsampled steps 2331/2961/3716, K64 step 944/1800.
- **Data:** Best val loss + designability per ckpt — assembled from F5 / F7 / F11 / E054 and the variant E-entries (E021, E034, E039, E053, E055, E056, E077).
- **Why it matters:** Motivates the F11 methodological claim and the project-wide rule "don't trust val loss alone".
- **Caveats:** Cross-run wandb val loss is misleading (E054). If possible, use paired re-evaluated numbers from `run_per_t_val.py` / `run_aggregate_val_seeds.py` instead.

### P15 — Variant designability ladder

- **Visualizes:** All CA-only architectural variants at their probed steps — pooled designability rates per L ∈ {50, 100, 200}.
- **Plot type idea:** Grouped bar; one cluster per variant (canonical, sparse-K40, sparse-K40+pair-update, sparse-K40+scnbr_t04+Fix C2, downsampled, K64-curric, K64-curric+BigBird, K64-curric+BigBird+pair-update+lowtsoft, K40+curric+self).
- **Data:** E019 (canonical N=30 76% pooled), E021, E034, E039, E053, E055, E056, E077 — pulled from the relevant entries in `experiments.md`. Many at N=6 only.
- **Why it matters:** Shows that no variant strictly exceeds canonical at L ≥ 100 — the architectural route's empirical state.
- **Caveats:** Variants are at non-matched opt steps and mostly N=6; canonical N=30. Annotate the step/N for each bar.

### P16 — Per-t validation loss across CA-only variants

- **Visualizes:** F11 / E043 — paired-protein per-t val loss for canonical_2646, conv_2331, scnbr_t04_1133, sparse_vanilla_1259 across the 5 t-buckets.
- **Plot type idea:** Multi-line (4 curves, one per ckpt); x = bucket midpoint, y = mean loss; SEM error bars (~0.01–0.08).
- **Data:** F11 table (lines 1058–1062 of `content_masterarbeit.md`). `results/per_t_val/{canonical_2646, conv_2331, scnbr_t04_1133, sparse_vanilla_1259}.json`.
- **Why it matters:** Visual proof of "curves are parallel, not crossing" → hybrid-sampling justification collapses.
- **Caveats:** 600-protein train-subset, not the canonical val set. Single seed=42. Ckpts at unmatched maturity (1133–2646).

### P17 — Hybrid handover kink figure

- **Visualizes:** E040 / E041 — `‖v_A − v_B‖ / ‖v_A‖` and cos(v_A, v_B) at the conv→scnbr (E040) and conv→canonical (E041) handover, plus designability outcome.
- **Plot type idea:** Twin scatter — x = t_switch, y_left = relative magnitude disagreement, y_right = cos similarity; bar overlay for pooled designability per setting.
- **Data:** E040 prose (kink at 0.79–0.86, cos 0.52–0.61 → 1/9 designable); E041 prose (kink 0.76–0.81, cos 0.59–0.66 → 5/9 designable).
- **Why it matters:** Hybrid section's main empirical figure; supports the architectural-similarity-determines-handoff-quality claim.
- **Caveats:** N=3 per length; small. E040 t=0.75 partial because of L=200 OOM.

### P18 — Sparse-vs-dense compute crossover

- **Visualizes:** E073 / E074 — per-step wall and memory of sparse-K40 vs canonical dense across L; the absence of crossover at L=512 and (hypothesized) crossover at L≥1024.
- **Plot type idea:** Line plot (x = L, y = ms/step); dense and sparse curves; mark sparse-K40 as "slower per step" annotation.
- **Data:** E073 (wandb compute efficiency audit) + E074 (inference-time compute benchmark). Output paths in those entries.
- **Why it matters:** Honest accounting of when sparse is and isn't a throughput win — supports CLAUDE.md's "do not propose sparse attention as throughput optimisation at n=512".
- **Caveats:** Sparse is slower per step at L=512 (the canonical training length). Don't quote sparse as a speedup without the long-L caveat.

### P19 — Dead-arm gallery table

- **Visualizes:** All called-dead architectural arms with their kill reason.
- **Plot type idea:** Table — columns: Variant, Best step, Pooled des @ N, L=50/100/200 rates, Reason called dead.
- **Data:** E034 (downsampled at step 2331/2961/3716), E056 (BigBird-only at step 819 — 0/18, position-unaware globals), E055 (5-axis bundle at 944, undecided), E058 (BigBird-only training), E077 (K40+curric+self at 1133 with N=6 NOCURRIC), E063 (attn-routing hybrid dead).
- **Why it matters:** Honest record of negative results; supports the thesis claim that several architectural axes were tried and failed.
- **Caveats:** Some arms (E055) were "dead at the probed step" but compute was deferred rather than verdicted; distinguish "dead at converged step" vs "dead at this checkpoint".

### P20 — Predictor:real TANGO gap dose-response

- **Visualizes:** F10 / E028 / E032 / E050 — predictor mean vs real mean TANGO at w ∈ {1, 2, 4, 8, 16} for E028 (clean ens) and E032 (noise-aware ens); the gap closing from −203 to +3.8.
- **Plot type idea:** Two-line plot per recipe (predictor solid, real dashed) over w (log scale); shaded gap band per recipe; highlight crossover at w=16 for F10.
- **Data:** F10 tables (lines 870–905). `experiments.md` E028/E032 full sweeps (lines 4616–4625 of `experiments.md`).
- **Why it matters:** F10's headline figure — gradient hacking gap closed by the two-fix composition.
- **Caveats:** TANGO-only (camsol_intrinsic is NaN in `compute_developability`). n=48 per cell. Per-length sign-disagreement (+28 / −42 / +25 at L=300/400/500) hides inside the aggregate.

### P21 — Negative-results ladder for predictor fixes

- **Visualizes:** F10 / E028 / E029 / E030 / E031 / E032 — gap at w=16 for each predictor variant in the "5 plausible fixes failed before the right composition" chain.
- **Plot type idea:** Horizontal lollipop / bar — y-axis labels = approach; x-axis = gap at w=16 (n=4 pilot); positive vs negative on either side of zero.
- **Data:** F10 negative-results table (lines 981–989).
- **Why it matters:** Demonstrates the compositional necessity of (noise-aware) + (ensemble); illustrates that "plausible-sounding" interventions can amplify, not fix, the failure mode.
- **Caveats:** n=4 pilot for most rows; the headline NA-v1 5-fold ensemble is also reported at n=48 (gap +3.8) and that should be on the figure.

### P22 — Codesign vs Δ-real Pareto frontier

- **Visualizes:** F10 / F13 / E066 / E070 / E076 — production-knee analysis: codesign rate vs real-property delivery (TANGO or CamSol) across w.
- **Plot type idea:** Pareto scatter — x = real-property delta (TANGO or CamSol), y = codesign rate; one point per (w, direction, predictor recipe); highlight the production-knee at w=32.
- **Data:** F13 cross-reference table (lines 1518–1521): w=0 codesign 47.9%, w=32 ΔCamSol +0.65 codesign 41.7%, w=128 +5.00 codesign 2.1%. Also E066's full Pareto sweep, E072's combo cell.
- **Why it matters:** Headline thesis-deliverable plot — the steering route's defensible operating point (+91% P(soluble) at 41.7% codesign).
- **Caveats:** n=48 steered vs n=200 unsteered. Population-level (not paired-by-seed). L=500 contributes Δ ≈ 0 at w=32 — break out by length in companion or footnote.

### P23 — Real CamSol per-length effect

- **Visualizes:** F13 / E076 — per-length Cohen's d at w=32 (+1.55 / +1.20 / +0.05 at L=300/400/500); also full distribution per group.
- **Plot type idea:** Strip + violin per (L, w) cell; mark unsteered and steered (w=32, w=128) groups; annotate Cohen's d above each L.
- **Data:** F13 table (lines 1490–1503). `CamSolpH_results.txt` (in repo root) + `camsol_submission_296.fasta`.
- **Why it matters:** The first real-property measurement on the predictor's actual training target; visualizes the L-heterogeneity that the aggregate hides.
- **Caveats:** Population-level (steered seeds 42–57, unsteered seeds 1000+). Public web-server CamSol, not Sormanni-lab direct run (assumed but not verified to match). L=500 zero-effect not separable from a ceiling effect.

### P24 — Steering schedule + diagnostic anatomy

- **Visualizes:** F10 / `steering/` design — the schedule `w(t)` and the gradient-flow path.
- **Plot type idea:** Three-panel anatomy — (a) `w(t)` linear ramp t∈[0.3,0.8] + hard stop at 0.9; (b) data-flow diagram (z_t → x_1_est → predictor → ∂L/∂z_t → unit-normed); (c) example per-step grad_norm trace from `additional_info["steering_diagnostics"]`.
- **Data:** F10 prose (steering hook design, lines 858–863); `steering/registry.py`; diagnostics output (when `log_diagnostics: true`).
- **Why it matters:** Methodological figure that introduces the steering machinery before the dose-response plots.
- **Caveats:** Diagnostics log batch element 0 only — a single-protein trajectory, not a population average.

### P25 — Per-length steering response heterogeneity

- **Visualizes:** F13 / E066 / E067 / E072 — Cohen's d of real property × L × w for camsol_max, tango_min, iupred3_max, and the 4-objective combo.
- **Plot type idea:** Heatmap (rows = (direction, w), columns = L ∈ {300, 400, 500}); cell = Cohen's d vs unsteered (use diverging colormap centered at 0).
- **Data:** F13 per-length numbers + E066 / E067 / E072 results dirs (`results/noise_aware_high_w_scout/`, `results/iupred_max_scout/`, `results/combo_devel4_scout/`).
- **Why it matters:** Shows the "under-steering at long L" pattern that motivates the combo cocktail (E072).
- **Caveats:** Per-length n=16 (E066) or n=20 unsteered + 16 steered (F13). Wide CIs at L=500.

### P26 — Multi-objective cocktail comparison

- **Visualizes:** E068 / E072 — single-objective camsol_max vs 2-objective (camsol_max + tango_min) vs 4-objective (camsol + tango + sap + scm+) at matched codesign budget.
- **Plot type idea:** Grouped bar — y = SWI Δσ delivery, codesign rate; one group per recipe at w=32.
- **Data:** E068 (combo_camsol_tango_scout), E072 (combo_devel4_scout). Result dirs in `results/combo_camsol_tango_scout/` and `results/combo_devel4_scout/`.
- **Why it matters:** Supports the F13 implication that combo at w=32 is the right tool for long-L codesign-preserving solubility.
- **Caveats:** Single seed range, single predictor recipe; codesign rates have wide CIs at n=12 per cell.

### P27 — Structural ensemble diversity (TM-score)

- **Visualizes:** F10 / E036 — pairwise TM-score per (direction, w, L) cell vs unsteered baseline.
- **Plot type idea:** Box / strip plot — x = w (or cell), y = mean pairwise TM; horizontal baseline at unsteered = 0.413.
- **Data:** F10 table (lines 921–924). `results/noise_aware_ensemble_sweep/` per-cell TM-score outputs.
- **Why it matters:** Rules out the "steering collapses the ensemble" objection; thesis-defending figure for the steering route.
- **Caveats:** n=120 pairs per cell. Mean TM essentially flat across w — almost too flat for a chart; report distribution rather than just the mean.

### P28 — Steering walltime overhead

- **Visualizes:** E069 — wall-time per ODE step (or per-protein) of unsteered vs steered (1×, 5× ensemble predictor calls).
- **Plot type idea:** Single bar plot — unsteered / steered single fold / steered 5-fold ensemble.
- **Data:** E069 (steering walltime overhead measurement). `results/unsteered_timing_test/`.
- **Why it matters:** Honest cost accounting for the steering recipe; shows the 5× predictor-call cost.
- **Caveats:** Hardware-specific (likely L4 or A100); declare which.

### P29 — Alphabet collapse: AA composition gen vs natural

- **Visualizes:** F8 (a) / E020 / E026 — per-AA mole-fraction deviation gen vs AFDB and PDB.
- **Plot type idea:** Diverging bar — y = AA, x = (gen − ref) / ref %; two series side by side (AFDB-ref and PDB-ref).
- **Data:** F8 numbers (lines 731–736). `results/aa_composition_nsteps400/` and `results/property_comparison_afdb/`.
- **Why it matters:** Headline number for the joint-head failure mode (N +156%, M −70%, W −50%, Glu/Asp = 2.79).
- **Caveats:** nsteps=400 numbers; the older nsteps=200 versions are in `results/property_comparison/`. PDB AA-composition ref capped at L=511.

### P30 — Length scaling of alphabet collapse (F9)

- **Visualizes:** F9 / E020+E026 follow-up — per-50-residue-bin Cohen's d for Shannon entropy, TANGO, IUPred3; and relative % deviation for E/N/L/M/W/H/F/A.
- **Plot type idea:** Multi-line (one line per metric) over L bin; or heatmap (rows = metric, cols = L bin).
- **Data:** F9 tables (lines 791–818). `results/property_comparison_afdb/` + the length-binned addendum CSVs.
- **Why it matters:** Refines F8's aggregate claim — the failure mode scales with the autoregressive horizon.
- **Caveats:** n=100 per gen bin (wide CI). Don't promote sub-bin variability claims (e.g. the Shannon dip at L=[700,750) → [750,800)) — within bootstrap noise.

### P31 — Aromatic count vs placement asymmetry

- **Visualizes:** F8 (c) / E023 / E026 — per-aromatic burial-targeting ratio P(aromatic | buried) / P(aromatic | exposed) for gen, AFDB, PDB.
- **Plot type idea:** Slope chart — one line per AA (W, F, Y, H); x = {AFDB, gen}; y = burial ratio. Or grouped bar with three references (gen/AFDB/PDB).
- **Data:** F8 (c) prose (`content_masterarbeit.md` line 758). `results/aromatic_burial_afdb/`.
- **Why it matters:** The competence signal — sub-claim (c) — that survives the alphabet collapse story.
- **Caveats:** F is the exception (placement matched, not sharper). N=1000 gen samples vs N=5000 AFDB sample. Bootstrap CIs needed on the figure.

### P32 — Thermal-stability proxy contradiction

- **Visualizes:** F8 (b.i) / E020 / E026 — aliphatic index, IVYWREL, GRAVY, charged_fraction, log_acidic_basic_ratio, aromatic_fraction Cohen's d (gen vs AFDB, gen vs PDB).
- **Plot type idea:** Diverging bar — y = metric, x = Cohen's d; two series (vs AFDB, vs PDB); highlight that aromatic_fraction goes one way and the rest go the other way.
- **Data:** F8 (b.i) table (lines 741–748). `results/thermal_stability_nsteps400/` and `results/thermal_stability_afdb/`.
- **Why it matters:** Visualizes the "gen looks thermostable by composition but lacks buried-core anchors" contradiction.
- **Caveats:** TemStaPro (b.ii) ML classifier pending. Magnitudes against AFDB are ~half of PDB — show both.

### P33 — Dense attention routing audit (per-query VJP)

- **Visualizes:** F12 / E059 / E060 / E061 — three-panel: (a) mass_top_K vs K for per-query gradient, per-(layer,head,query) attention, aggregate gradient; (b) cross-metric Jaccard distribution per-query; (c) per-query-pair Jaccard vs L showing collapse to 0.06 at L=200.
- **Plot type idea:** 3-panel composite line + histogram + box-by-L.
- **Data:** F12 tables (lines 1159–1233). `results/dense_attn_audit/canonical_2646_grad_per_query.json`, `canonical_2646_gradient.json`, `canonical_2646_dense_attn.json`.
- **Why it matters:** Mechanism story for why content-free shared-K-set sparse can't recover dense behavior at long L; constructive lever for per-query routing distillation.
- **Caveats:** 3 proteins per L bin; 8 queries per (protein, t). bf16 grad slightly understates concentration. F12 is decision-gate, not build — note the inference-only K-swap follow-up failed (E062/E063).

### P34 — Variant ablation summary table

- **Visualizes:** Master table — for each architectural / recipe variant, the (config delta from canonical, opt step probed, N, designability per L, total /, val-loss reading, dead-arm verdict).
- **Plot type idea:** Table.
- **Data:** F5/F7/F11/F12 + E008/E009/E010/E013/E014/E019/E021/E034/E039/E046/E047/E053/E055/E056/E077/E078/E079.
- **Why it matters:** Single reference exhibit covering everything in the architectural route — the reader can scan and see "no variant strictly beats canonical".
- **Caveats:** Variants are at heterogeneous maturity. Annotate the step + N + whether the verdict is "at this step" vs "at the converged plateau".

### P35 — Finding-to-experiment cross-reference

- **Visualizes:** Mapping between paper Findings F1–F13 and the lab-notebook E-IDs they draw on.
- **Plot type idea:** Table (Finding | Primary E-IDs | Supporting E-IDs | Status).
- **Data:** Cross-reference sections inside each Finding in `content_masterarbeit.md`.
- **Why it matters:** Lets the examiner traverse from a paper claim back to the lab record in one step.
- **Caveats:** Some findings have multiple supporting experiments; pick the primary one for the headline column.

---

## Notes for the plot maker (agent 3)

- **nsteps=400 hard rule.** Any plot whose y-axis is a structure-derived metric (scRMSD, designability rate, scRMSD-conditional codesign) must be computed at nsteps=400. Plots from old nsteps=200 generations (E020/E021/E034/E035/E038/E039/E040/E041 originals) should either be regenerated or carry an explicit nsteps caveat in the caption.
- **Length binning for property metrics.** TANGO, SAP, SCM, hydrophobic-patch, and rg scale with N. Per the user memory `feedback_length_bin_property_sigmas.md`, pool-then-z-score under-reports steering effects. Bin by 50- or 100-residue width before computing Cohen's d / σ-deltas.
- **N small in many cells.** Several designability sweeps are N=3 or N=6 per length per ckpt. Show per-protein points wherever possible; binary rates from N≤6 carry wide Wilson CIs.
- **Wandb val_loss is not cross-run comparable.** When showing val curves across runs (P10, P14), prefer paired re-evaluations from `proteinfoundation/run_per_t_val.py` (`results/per_t_val/`) or `run_aggregate_val_seeds.py` (`results/aggregate_val_seeds/`); note this in any wandb-sourced curve's caption.
- **Codesign vs designability.** For the steering plots, the right structural-integrity metric is **codesign** (`use_pdb_seq=True`, joint-head sequence), not MPNN-redesigned designability. F10/F13 lean on this distinction.
- **AFDB is the right natural-protein reference**, not PDB. La-Proteina was trained on AFDB. Where both numbers are available (F8/F9), AFDB is primary, PDB is sensitivity.
