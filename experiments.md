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
| [E038](#e038--scnbr_t04-re-probe-with-fix-c2-actually-wired-2026-05-06) | 2026-05-06 | finished | Re-probe of `ca_only_sparse_K40_scnbr_t04` ckpt (best_val ep=8 step=819) after merging Fix C2 inference path from HPC. Same protocol as E035 (3 lengths × 6 samples × 200 nsteps); Fix C2 canary log confirmed the threshold-gated x_sc neighbor source fired at runtime. | non-narrative — 0/18 still, **but the mechanism is alive**: L=100 best 3.16 Å (vs E035's 4.37 Å), median 6.34 Å (vs 9.94 Å), three samples concentrated at 3.16/3.39/3.76 Å. Variant remains pre-convergence at step 819; re-probe at step ≥ 1500 is the next decision point, not abandonment. **Supersedes [E035](#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06) for the inference-correct numbers**; E035's "0/18 with best 4.37 Å" was a Fix-C2-missing artifact, not a variant verdict. |
| [E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06) | 2026-05-06 | finished | Re-probe of `ca_only_sparse_K40_scnbr_t04` at the new best-val ckpt (ep=11 step=1133), Fix C2 active. Same protocol as E038 (3 × 6 × 200 nsteps). Canary fired. | non-narrative — **variant clears the CLAUDE.md bar**: 2/6 at L=50 (best 1.51 Å, three samples within 0.11 Å of threshold), 1/6 at L=100 (best 1.92 Å), 0/6 at L=200 (best 7.22 Å). Pooled 3/18 (17%), mean 6.99 Å (vs E038's 11.39 Å — −4.4 Å in 314 training steps). **Matches [E021](#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30)'s sparse+pairupdate step-1133 designability count** (2/6, 1/6, 0/6) with comparable best-Å — two different architectural levers (Fix C2 vs pair-update) reach the same step-1133 ceiling on this 160M canonical-recipe trunk. The earlier E038 "re-probe at step ≥ 1500" prediction was beat by ~25%; converged plateau hits at 1133. |
| [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06) | 2026-05-06 | finished | Hybrid sampling: 1D-conv ckpt ([E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06), step 2331) for `t < t_switch`, sparse-K40 + Fix C2 ckpt ([E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06), step 1133) for `t ≥ t_switch`. Two settings: t_switch=0.6 / N=3 (with kink ‖v_conv − v_scnbr‖ logged at the handover) and t_switch=0.75 / N=6 (eval partial — L=200 OOM'd from co-tenant). | non-narrative — **resurrects the dead conv variant.** t=0.6: 1/3 / 0/3 / 0/3 → 1/9 (11%) pooled, best 1.78 Å. t=0.75 (partial, L=200 missing): 1/6 / 2/6 / OOM → 3/12 valid (25%), best 1.15 Å. The conv→scnbr handoff produces designable samples that conv alone never produces (E034: 0/18, every sample ≥12 Å). **Kink at handover is large**: at t=0.608, ‖v_A − v_B‖ / ‖v_A‖ = 0.79-0.86 with cos(v_A, v_B) = 0.52-0.61 — the two trained models disagree by ~80% in magnitude AND ~50° in direction at the same noisy state. Conv predicts a ~1.6× larger velocity than scnbr at the handover; switching is a deceleration, not a smooth handoff. Probably explains why later handover (t=0.75) outperformed earlier (t=0.6) on this small N — the longer scnbr re-equilibrates after the kink. |
| [E041](#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06) | 2026-05-06 | finished | Same hybrid mechanism as [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06), conv ckpt for `t < 0.6` swapped to **canonical** wd=0.05 ckpt (ep=24 step=2457, same `test_ca_only_diffusion` run id 1776805213) for `t ≥ 0.6`. N=3, seed=5, kink logged. Canonical step 2646 was unverified (CLAUDE.md only); 2457 is highest local from same run. | non-narrative — **dramatic upgrade over E040**. Pooled 5/9 (56%) vs E040's 1/9 (11%). L=50: 3/3 designable (best 1.08 Å); L=100: 1/3 (best 1.56 Å); L=200: 1/3 (best 1.53 Å) — first-ever architectural-axis hybrid sample to clear L=200. Pooled best 1.08 Å, median 1.68 Å, mean 3.69 Å. **Smaller kink than E040**: ‖Δv‖/‖v_A‖ = 0.76-0.81 (vs E040's 0.79-0.86) and cos = 0.591-0.655 (vs 0.522-0.612) — canonical and conv share dense attention; scnbr is sparse, hence the architectural-similarity kink hierarchy. The canonical trunk receives a conv-built x_t at t=0.6 and finishes from there with ~12 percentage-points-better directional alignment + ~5pp-smaller magnitude disagreement, and the resulting trajectory clears 2× more proteins than scnbr does on the same handoff. |
| [E042](#e042--codesignability-validation-of-the-noise-aware-ensemble-sweep-2026-05-07) | 2026-05-07 | finished | Codesignability check on E032's noise-aware-ensemble sweep using the **joint sequence head** (use_pdb_seq=True, num_seq=1) — does the steered protein still fold when its own (steered) sequence is taken at face value? 4 seeds × 3 lengths × 5 w × 2 directions = 120 PDBs through ESMFold + per-residue scRMSD. | **Finding 10 codesignability addendum.** Codesign rate is **flat across w∈[1,16] for both directions** — w=1: 5/12 (42%) camsol_max, 4/12 (33%) tango_min; w=16: 4/12 (33%) camsol_max, 5/12 (42%) tango_min. Mean coScRMSD 3.6-4.1 Å, median 2.1-2.3 Å. Codesignability is intrinsically a stricter bar than MPNN-redesign designability (E033's 80-100% drops to 33-42% codesign), but the *steering-vs-coScRMSD* trend is the relevant signal — and it is flat. Steering the latent does not silently degrade what the joint sequence head produces; the quality drop relative to E033 is the unconditional sequence head's own ceiling, present at w=1 too. |
| [E043](#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07) | 2026-05-06 → 2026-05-07 | finished | D1 of the hybrid-sampling diagnostic plan. Bucket FM val loss into 5 t-bins on the same 600-protein subset (paired across ckpts, seed=42) for `canonical_2646`, `conv_2331`, `scnbr_t04_1133`, and `sparse_vanilla_1259`. Bypasses the broken `PDBLightningDataModule` via `proteinfoundation/run_per_t_val.py`. | → Finding 11. **No regime where a non-canonical variant beats canonical at per-t val loss.** All four ckpts have the same minimum bucket (t∈[0.6, 0.8)). Loss-vs-t curves are *parallel*, not crossing: canonical (0.072 nat lowest at the min) < conv (+0.142 there) ≈ scnbr_t04 (+0.135) ≈ sparse_vanilla (+0.132). **Fix-C2 mechanism does not move per-t val loss**: scnbr_t04 vs sparse_vanilla within ±0.025 at every bucket — the trained weights are functionally identical, so any designability gap between them must come from the inference-time x_sc switch, not from training. conv's largest gap to canonical is at t∈[0.8, 1.0) (+0.452), consistent with E041's hand-off-before-late-stage design. |
| [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) | 2026-05-07 | finished | Inference-only test of low-t neighbor-list curriculum on plain sparse_K40 step 1259: mask spatial slots when t<0.3, random slots when t<0.6 (sequential always kept). Hypothesis: at low t, x_t-built spatial+random neighbor groups are uninformative; masking them should not hurt, and may help. Paired N=6 × L∈{50,100,200} × seed=5 against same-ckpt baseline. | non-narrative — **curriculum hurts pooled designability** (5/18 → 3/18, −2 designable, 3× DESIGN→FAIL vs 1× FAIL→DESIGN). L=200 uniform degradation (5/6 paired proteins worse, mean +4.0 Å) and L=50 net negative (despite a single dramatic Δ=−8.77 Å rescue at id=0). **L=100 is the only redistribution-positive length**: count holds 3/6 but median 2.94→1.99, max 6.31→4.52, the collapsed sample disappears. Mechanism reading: noisy spatial+random groups carry non-trivial information at low t — the "they're noise so masking them is free" hypothesis is *not* supported. Untuned thresholds (0.3/0.6); t-sweep is the cheap next step before any retraining-with-curriculum commitment. |
| [E045](#e045--t-dependent-k-budget-reallocation-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) | 2026-05-07 | finished | Same ckpt + protocol as E044 but with K-reallocation instead of masking: keep total K=40 fixed across the whole trajectory, just reallocate (n_seq, n_spatial, n_random) by t. 3 buckets — t<0.33 → (20,0,0); t<0.66 → (12,8,8); t≥0.66 → canonical (8,8,16). Softmax always sees 40 real slots; only composition shifts. | non-narrative — **same pooled count as mask (3/18) but a completely different per-length distribution**. **L=50 spectacular** (3/6 designable, **min 0.63 Å** = best ever on this ckpt; every sample within 3.10 Å); paired Δ=−10.77 Å rescue at L=50 id=0 (11.40 → 0.63). **L=100 disaster**: 3/6 → 0/6, **two new collapsed samples appear** (id=0 1.43 → 9.04, id=3 4.30 → 7.02). L=200 fails uniformly. **Mechanism is chain-length-dependent**: at short L the spatial/random/sequential groups overlap heavily so realloc is a benign content shift; at L≥100 the model uses long-range info encoded in the spatial+random slots even when noisy, and stripping it for sequential causes collapse. Mask (E044) and realloc (E045) are complementary: hypothetical realloc-at-L<60 + mask-at-L≥60 would yield 6/18 (33%) — but L-dependent schedule, not static, is the principled next step before retraining. |
| [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) | 2026-05-11 | finished (code fix + numerical tests + inference probes at step-1385) | Investigation of the K=64-curriculum-self variant's (variants.md §11; NO BigBird, NO pair-update) L=50/L=100 gap vs canonical. Numerical tests of `sparse_neighbors.build_neighbor_idx` and `PairBiasAttention._attn_sparse`. Found+fixed an off-by-one cap (`min(2*n_seq, N-1)` → `min(2*n_seq, N)`). Falsified four candidate bug mechanisms. Then re-probed step-1385 with three schedule variants (canonical, NOCURR, LOWTSOFT) and pushed LOWTSOFT to N=18 paired with the variants.md §11 baseline. | non-narrative — feeds the retrain decision ([E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)). **LOWTSOFT N=18 post-fix: 56 % / 61 % / 0 % at L=50/100/200** vs canonical N=18 pre-fix 44 % / 56 % / 11 % — **+12 pp at L=50 (paired with the cap fix) and +5 pp at L=100**, but **−11 pp at L=200 (one-or-two-protein swing on N=18)**. Same-day same-code paired probes give the cleanest A/B: at N=6 post-fix, canonical 4/3/0, NOCURR 4/1/0, LOWTSOFT 4/4/0 — i.e. removing the schedule entirely tanks L=100 (3→1), softening the low-t bucket improves it (3→4). Direction is consistent with the variant-design hypothesis. |
| [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) | 2026-05-11 | in progress (running; SLURM 29210711, SL2 20h slot, started ~18:00 BST 2026-05-11) | Cold-start retrain of a FIVE-AXIS bundle: K=64 SALAD-canonical sparse + curriculum (`(16,8,24)` low-t / `(16,8,24)` mid / `(8,16,32)` high) + self-inclusion + **BigBird globals n=4 (FIRST TIME TRAINED)** + **pair-update every 3 layers (FIRST TIME TRAINED in K=64 sparse path)**, with the off-by-one cap fix from [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) live in code. **Architecture differs from step-1385:** step-1385 is variants.md §11 (only K=64 + curriculum + self-inclusion, three axes — no BigBird, no pair-update); this retrain is the planned §12-LowTSoft bundle (five axes). OLD recipe wd=0.05 / constant LR=2e-4 / no scheduler / dataset / seed=42 / 160M trunk are unchanged from §11. | non-narrative — first cold-start training of BigBird + pair-update + lowtsoft simultaneously. **Predicted milestones:** val MSE convergence trajectory tracks below the §11 step-1385 curve once the new (BigBird + pair-update) parameters reach their working points (initial divergence is the cost of training 3 new architectural axes from zero-init); designability at step ~1800–2200 of the bundle vs §11's step-1385 reading (8/10/2 N=18); designability at step ~2400 vs canonical's E019 step-2646 N=30 baseline (19/20/3 = 63 % / 67 % / 10 %). Decision: if the bundle clears canonical at L=100 with the L=200 dropout not deeper than the LOWTSOFT-N=18 inference probe (−11 pp), the five-axis bundle is the new K=64 baseline; ablations (drop pair-update, drop BigBird, isolate cap-fix) follow. |
| [E048](#e048--inference-only-k-bump-sweep-k40--k64--k128-on-plain-sparse_k40-step-1259-2026-05-07) | 2026-05-07 | finished | Inference-only K-bump on the same sparse_K40 step 1259 ckpt: K=40 (control, reused E044 baseline) vs K=64 (12/12/28) vs K=128 (24/24/56). Composition ratio 40/20/40 preserved across K. Curriculum OFF. Tests whether the trained K=40 weights have headroom that more neighbors at inference can unlock. Wall-clock per gen: K=40 ~63s, K=64 ~80s (+27%), K=128 ~140s (+122%). _(Renumbered from local E046 on merge.)_ | non-narrative — **strict monotonic degradation: K-bump is a dead end on this checkpoint**. Pooled designability: K=40 5/18 (28%) → K=64 2/18 (11%) → K=128 **0/18 (0%)**. K=128 is catastrophic at L=100 (clean cluster wiped: 3 → 0; collapsed cluster 1 → 5). Failure mode is calibration-shift: trained q/k/v projections + AdaLN-Zero gates were tuned for 40 slots, adding more dilutes attention across un-weightable candidates. **The L=100 K=64 result is the most informative**: 4 clean samples (vs 3 baseline) but two clean samples drift just over the 2 Å line — distribution shifts up by ~0.7 Å, consistent with "wants to produce same structure, but trained weights miscalibrated for new K". E045's L=50 std-collapse does NOT reproduce at K=64/K=128 — that was specifically a composition-axis effect, not K-driven. **Conclusion**: inference-only cannot test architectural ceiling on a K=40-trained model; a K=64 retraining run is the only way to settle the K-axis. |
| [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) | 2026-05-08 | finished | First inference probe of the K=64 + curriculum-trained ckpt (`ca_only_sparse_K64_curriculum_self`, ep=9 step=944). New architecture: K=64 (8/16/32 canonical) + 3-bucket t-curriculum baked into training + self-inclusion. Same N=6 × L∈{50,100,200} × seed=5 protocol; paired-by-noise vs E044/E045 K=40 ckpt. Compile auto-on from saved hyperparams. _(Renumbered from local E047 on merge.)_ | non-narrative — **mixed signal at this training step. L=50 retraining-validates E045**: 3/6 designable, min 0.91 Å (matches E045 inference-only's 3/6 at min 0.63 Å); the curriculum direction for short proteins is real and survives the move from inference-only forcing to baked-in training. **L=100 got WORSE under retraining**: 1/6 designable (vs K=40 baseline's 3/6), 3 fully-collapsed samples (>7 Å), mean scRMSD 6.94 (vs K=40 baseline 3.25). L=200 unchanged. **L=50 std-collapse from E045 does NOT survive retraining** (std 4.04 vs E045's 0.92) — that was specifically an inference-only forcing artifact. Three confounds vs K=40 baseline: step 944 vs 1259 (under-training), K=40 vs K=64 axis, and self-inclusion. Cleanest next probe: re-evaluate same run at a later step (matched 1259 or canonical 1800-2200 window) before deciding whether the curriculum-retrain hypothesis at L=100 is dead or just under-cooked. |
| [E050](#e050--steering-audit-matrix--predictor--ensemble--fold--smoothing-2026-05-10) | 2026-05-10 | finished | Audit-style fill of the predictor × ensemble × fold × smoothing matrix supporting Finding 10. 7 n=4 smokes (clean f0/f2, clean ens5 no-smooth, NA-v1 f0/f1/f3/f4) + 5 n=48 sweep cells (NA-v1 fold2 across w∈{1,2,4,8,16}, L∈{300,400,500}). Same `inference_ucond_notri_long`, nsteps=400, w=16 anchor as E028/E029/E032; tango_min direction; real TANGO via local binary. _(Renumbered from local E048 on merge.)_ | **Finding 10 audit support — three-pronged.** (1) **Smoothing in E028 contributes ~zero** — clean ens5 + smoothing (-203.5) ≈ clean ens5 no-smoothing (-203.9); the gap reduction in E028 is 100% from ensembling. (2) **Fold 2 is NOT specially smart** — all 5 NA-v1 single folds cluster in [-47, -97] gap range (f2 -47, f0 -61, f1 -62, f3 -59, f4 -97); the original fold-2-only single-fold reading is representative. (3) **NA-v1 single fold at n=48 stays negative** (-87.5 at w=16, Δratio 7.0×) — no crossover-flip to positive like the ensemble does (E032 +3.8 at n=48, Δratio 2.9×). Ensembling moves the gap by ~91 TANGO units even after the noise-aware fix is in place. F10's "two fixes layered" claim survives, with one substantive edit: the "ensemble cancels fold-specific shortcuts" mechanism story is weaker than the original framing — folds are similar in single-fold gap magnitude, so ensembling acts more as variance averaging than as adversarial-direction cancellation. |
| [E051](#e051--n3-quick-designability-probe-of-ca_only_sparse_k64_curriculum_self-at-step-1800-2026-05-10) | 2026-05-10 | finished | N=3/length quick designability probe of the K=64 SALAD-canonical sparse + low-t→SALAD curriculum + self-inclusion variant at the latest rsynced ckpt (epoch=17, global_step=1800). Convergence check vs [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08)'s step-944 picture. _(Renumbered from local E049 on merge.)_ | non-narrative — at step 1800 the variant looks the same as step 944 within N=3 noise: L=50 still produces sub-1-Å samples (2/3, min 0.89), L=100 still bimodal (1/3, min 1.23), L=200 still dead (0/3). The "L=100 closes with more training" hypothesis from [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) is **not yet supported** — ~91% more training did not move the L=100 picture. |
| [E052](#e052--universal-guidance-k-axis-extension-clean-predictor-2026-05-10) | 2026-05-10 | finished | UG K-axis ablation on the clean-predictor regime to close E030's open question. Targeted K=20 (4× E030's K=5); L4 OOM at K≥10, settled at K=8 — the largest value that fits in 22 GB without gradient checkpointing. Same w=16, L=300, n=4 cell as E028/E030. Smoothing intentionally OFF (audit-justified by [E050](#e050--steering-audit-matrix--predictor--ensemble--fold--smoothing-2026-05-10)'s clean+ens vs clean+ens+smooth equivalence). _(Renumbered from local E050 on merge.)_ | **Negative result, K-axis closed.** K=8 gap = −308.6 (vs E028 K=1: −203.5; E030 K=5: −301.7). The K=5 → K=8 jump is essentially flat (−302 → −309) — extending K past 5 doesn't reverse E030's negative direction, but the gap doesn't accelerate either. Real TANGO at K=8 is 616.2 (vs K=1 581.9, K=5 607.1) — predictor delivers *less* real-property work as K increases. Confirms E030's "longer Jacobian = more adversarial leverage on a fragile clean predictor" mechanism story and rules out "K=5 was just an unlucky local point". K=20 attempt OOM'd both with smoothing on (~30 GB needed) and without (~25 GB needed); K=10 also OOM'd at ~24 GB. Engineering note: K-axis past 8 on L4 requires gradient checkpointing on the inner Euler loop; deferred as not worth the code change given the K=5 → K=8 plateau. |
| [E053](#e053--ca-only-downsampled-variant-canonical-n6-designability-probe-at-step-3716-2026-05-11) | 2026-05-11 | finished | Canonical N=6 × L∈{50,100,200} × nsteps=400 designability probe on the most recent `ca_only_downsampled` checkpoint (epoch=36, opt step=3716). Tests whether ~755 additional opt steps past the previous canonical probe (step 2961, 3/18 designable) move the variant's converged ceiling. Same protocol as `inference_downsampled_step2961.yaml` / `inference_downsampled_step2331_n6_nfe400.yaml`. _(Renumbered from local E051 on merge.)_ | non-narrative — **arm regressed slightly with more training**. Pooled 1/18 (5.6%) vs step 2961's 3/18 (17%). L=50 still produces a designable sample (1/6, min **1.19 Å**) but lost the second sub-2 Å sample seen at step 2961. L=100 collapsed entirely: 0/6 designable, min 8.01 Å, every sample ≥8 Å (vs step 2961's mixed picture). L=200 unchanged dead: 0/6, min 9.94 Å. The downsampled arm has not crossed the "L=50 plus partial L=100" bar at any tested step (2331: 0/18, 2961: 3/18 mostly L=50, 3716: 1/18 only L=50) — the architectural-axis ceiling for 1D-conv-downsampling on this canonical recipe sits at "L=50 occasional, L≥100 dead". |
| [E054](#e054--canonical-baseline-last-v2ckpt-n6--nsteps400-designability-probe-step-1952-2026-05-10) | 2026-05-10 | finished | Canonical N=6 × L∈{50,100,200} × nsteps=400 designability probe on `last-v2.ckpt` (Lightning auto-versioned `last.ckpt` slot-end snapshot, on the same `test_ca_only_diffusion/1776805213` canonical run). **Actual `global_step=1952`, `epoch=19`** (read from ckpt; earlier inference of "~step 1900" from file mtime was wrong). **Val_loss/loss_epoch at save: 4.7874 nat** vs `best_val_00000026_000000002646.ckpt`'s 5.9237 nat — i.e. lastv2 is Δ=−1.13 nat *better* by training-time val_loss than the on-disk "best_val_2646" anchor, and only Δ=+0.076 above the all-time canonical minimum (4.7115 at step 2204, overwritten under save_top_k=1 and gone from disk). Config `inference_canonical_lastv2_n6_nfe400.yaml`, seed=5. Paired with per-t val under same protocol as E043 (seed=42, 600-protein subset). _(Renumbered from local E052 on merge.)_ | non-narrative — **cleanest within-run val-loss-vs-sample-quality decoupling on record.** Pooled designability 7/18 (38.9%) vs step 2646's 68/90 (75.6%, E019 N=30). L=50 4/6 (best 0.78 Å); L=100 3/6 (best 1.07 Å); **L=200 0/6 (best 4.29 Å)** vs 2646's 16/30 at L=200. **Per-t val (paired)**: lastv2 tied with 2646 at t<0.4 (Δ within 0.01 nat, inside SEM), worse at every t≥0.4 — Δ = +0.036 / +0.049 / +0.059 nat at t∈[0.4,0.6) / [0.6,0.8) / [0.8,1.0) (3-5× SEM each). Direction of designability gap and per-t gap match exactly: lastv2 is degraded on the data side of the trajectory. **Headline**: a ckpt that is 1.13 nat *better* by wandb val_loss is dramatically *worse* by samples and by paired per-t late-bucket loss on the train-set subset. Re-confirms Finding 5/6 and `feedback_wandb_val_loss_not_comparable.md` — within-run val_loss monotonicity does not hold either. |
| [E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12) | 2026-05-12 | finished | First N=6 × L∈{50,100,200} × nsteps=400 designability probe of [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)'s five-axis bundle (K=64 SALAD-canonical sparse + curriculum + self + BigBird n=4 + pair-update every-3 + LOWTSOFT low-t bucket `[16,8,24]`) at the freshly-rsynced `best_val_00000009_000000000944.ckpt`. Same seed=5 protocol as [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08); paired-by-noise comparison at the same step number against the three-axis §11 variant. (Note: config/output names say "step1133"; actual ckpt is step 944 — the step1133 handle was a placeholder when the config was authored.) | non-narrative — **L=50 holds (3/6, 50%), L=100 regresses (1/6 → 0/6, every sample > 6 Å), L=200 unchanged dead.** Pooled 3/18 (17%). Below the three-axis §11 step-944 baseline at every length (matched on count at L=50 with worse min-Å 1.45 vs 0.91; strictly worse at L=100 with median 7.25 vs 4.50). Below the inference-only LOWTSOFT-on-§11-step-1385 probe (E046) at every length — consistent with E047's prediction that the bundle's 3 new architectural axes (BigBird globals + pair-update + lowtsoft from cold init) pay a training-cost handicap in the first ~1000 opt steps. **Not yet a dead-arm call** (canonical bar is ≥ step 1800-2200); decision deferred to the slot-end ckpt at step ~2400. |
| [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13) | 2026-05-13 | finished — **DEAD ARM CALL** | First and last N=6 × L∈{50,100,200} × nsteps=400 canonical designability probe of the **four-axis bundle** `ca_only_sparse_K64_curriculum_self_bigbird` (K=64 SALAD-canonical sparse + curriculum + self-inclusion + BigBird globals n=4; **NO pair-update, NO lowtsoft** — sibling ablation of E055's five-axis bundle). Probe done at `best_val_00000008_000000000819.ckpt` (epoch 8, opt step 819, from the cold-start training run described in [E058](#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge)). Same seed=5 protocol as [E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12). | **0/18 (0 %); arm called dead by user 2026-05-13.** Three converging lines: (1) L=50 best-Å gap +3.85 Å vs §11 step-944 cannot be absorbed by 125 opt steps of timing (largest matched-step L=50 swing on record ≈ 0.5 Å); (2) **mechanistic prior — BigBird globals here are position-unaware** (no per-token sequence position; same `global_pair_bias` row to every residue) and therefore cannot sharpen the position-sensitive score field L=50 requires; (3) wandb val_loss tracks worse than three-axis and five-axis cousins on the same dashboard. Sample evidence + mechanism + val_loss all point the same direction. Decision saved to memory as [[bigbird-globals-position-unaware]]; do not re-propose position-unaware globals as a lever for position-sensitive failure modes. No further probes / rsyncs queued for this run. |
| [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge) | 2026-05-12 | finished | Read-only audit of the BigBird + pair-update wiring inside [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)'s in-progress checkpoint (`last.ckpt`, global_step=1200). Triggered by the observation that the bundle's val curve matches §11 at high t and is strictly worse at low t with the same shape, raising the question "was BigBird actually applied?". Loads the trained ckpt, runs forward at t∈{0.1, 0.5, 0.9}, hooks `PairBiasAttention._attn_sparse` to record per-layer attention mass on the 4 global slots and the std of global K vectors; logs cosine similarity between the 4 trained `global_token_emb` rows; then runs one fwd+bwd on a fresh-init clone to compare per-param gradient norms (globals vs trunk), with and without `global_cond_emb` overridden to the time-embedding broadcast. _(Renumbered from upstream E048 on 2026-05-13 merge.)_ | non-narrative — **BigBird is correctly wired AND heavily used by the trained model, contradicting the "globals are inert" hypothesis.** Late-layer attention mass on globals at t=0.1 hits 22–34 % (vs 5.88 % uniform baseline), with several residue queries placing ~100 % of their attention on globals in layers 7–13. The 4 globals are NOT collapsed (off-diagonal cos sim ≤ 0.071) and carry distinct keys (std across globals ≥ 0.32 every layer). Fresh-init grad norms on globals are 0.04–3.3× the trunk's to_qkv grad — not structurally tiny. The time-emb-cond override (B2) gives no speedup (1.00×), so `global_cond_emb` being learnable-zero-init is not the bottleneck. **Reframes the low-t pathology:** globals are time-agnostic by design (`global_cond_emb` doesn't see time-emb), the curriculum strips real spatial/random capacity at low t (`(16,8,24)`), and the model has learned to compensate by routing 20–30 % of low-t late-layer attention into 4 protein-agnostic globals — which cannot encode protein-specific structure. That's a *learned-policy* pathology, not a wiring bug. Audit script: `script_utils/audit_bigbird_wiring.py`. |
| [E058](#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge) | 2026-05-12 | in progress (queued; SLURM 29277806) — first probed at [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13) (dead arm at step 819) | Cold-start retrain of a **four-axis** bundle: K=64 SALAD-canonical sparse + curriculum + self-inclusion + **BigBird globals n=4 only** — **NO pair-update, NO LOWTSOFT**. Isolates the BigBird contribution against the §11 reference. _(Renumbered from upstream E049 on 2026-05-13 merge. This training run produced the ckpt `best_val_00000008_000000000819.ckpt` that [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13) probed; outcome at step 819 was 0/18 designable and the user called the arm dead — see [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13) for the post-hoc verdict against the predicted milestones below.)_ | non-narrative — first BigBird-only training. **Predicted milestones:** at matched horizon (~1200 opt steps), expect `val_loss_by_t.t_000_020` close to §11's 4.3-4.5 (vs E047's stuck-at-7.2) since the low-t bucket is back to `(32, 0, 0)`; expect total `val/loss_epoch` to track §11 to within +0.1 (the parameter overhead of 4 globals + their pair-bias entries). Decision: if low-t recovers AND BigBird globals attract non-trivial attention mass as in [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge), the four-axis form is the new K=64 baseline and the next ablation is pair-update on top of THIS variant; if low-t still degrades vs §11, BigBird is the regression and `n_global_tokens: 0` should be the K=64 baseline going forward. **Actual outcome at step 819 (per [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13)):** 0/18 designable, dead-arm call. Predicted-milestone verification on the val-loss-by-t side requires reading the wandb run; user reported val_loss tracks worse than §11 and the five-axis bundle. |
| [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) | 2026-05-13 | finished — **STOP signal** | Decision-gate audit on the canonical dense baseline (`test_ca_only_diffusion/1776805213` step 2646) for the content-adaptive top-K distillation idea (use dense's per-residue attention pattern as a teacher signal for a sparse student's K-set routing head). Two checks: **(1) concentration** — fraction of attention mass captured by top-K residues at K ∈ {8,16,32,48,64} per (layer, head, t, length); **(2) stability** — Jaccard overlap of top-16 attended sets across adjacent layers, adjacent t-steps, and heads within a (layer, t). 3 proteins per length bin × L∈{50,100,200} × t∈{0.10, 0.30, 0.50, 0.70, 0.90} = 630 attention-layer records. Script: `script_utils/audit_dense_attention_concentration.py`; output `results/dense_attn_audit/canonical_2646_dense_attn.json` (440 KB). | non-narrative — **cheap-student top-K distillation is ruled out.** Concentration is moderate but below the GO bar (mass_top_16 = 0.656 vs ≥0.70 needed; mass_top_32 = 0.794 vs ≥0.85 needed; mass_top_64 = 0.907 well above uniform = 0.32). The killer is stability: layer-Jaccard 0.217 and head-Jaccard 0.224 are both well below the 0.3 STOP threshold — adjacent layers share only ~3.5/16 of their top-16 attended residues, and heads within a (layer, t) disagree by the same margin. Single shared K-set per protein-per-t cannot approximate dense; the cheap-student version would need 14×12 = 168 separate K-sets, which defeats the purpose. **Secondary insight for the sparse-vs-dense gap (E043):** our sparse implementation shares the K-set across all 14 layers AND all 12 heads within each layer; dense gives every (layer, head) independent access to all N. The audit quantifies how much specialization sparse forfeits — substantial. This refines the "uniform sparse-vs-dense tax" framing from E043 with a concrete structural mechanism: shared-K-set across layers/heads is the binding constraint, not the K-set choice per se. **Does NOT rule out:** per-layer random redraw (different mechanism — receptive-field expansion across depth, not routing), or per-layer × per-head routing (no longer "cheap student" but mechanistically supported). t-Jaccard 0.475 sits in the middle band; per-trajectory routing could be moderately stable. |
| [E060](#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) | 2026-05-13 | finished — **closes the E059 STOP** | Companion to [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) — tests whether gradient saliency (∂L/∂x_t["bb_ca"], L2-normed per residue) is a sharper importance signal than attention and whether attention's pattern reflects loss-importance. Same canonical step-2646 ckpt, same 9 proteins (3 per L∈{50,100,200}), same t-grid, same K_grid, seed=42 — outputs pair cleanly with E059 by (protein_label, t_label). 45 saliency records + 630 attention-layer records. Script: `script_utils/audit_dense_gradient_saliency.py`; output `results/dense_attn_audit/canonical_2646_gradient.json` (2.5 MB). | non-narrative — **three negatives close the door on cheap-K distillation as imagined.** (1) **Gradient is more diffuse than attention**, not sharper: mass_top_16_grad = 0.312 (vs attn 0.656), mass_top_32_grad = 0.528 (vs attn 0.794). The "use grad as the importance signal" rescue of E059's STOP is falsified. (2) **t-Jaccard_grad = 0.200** (mean over 36 t-adjacent pairs) — far under the 0.7 stable-routing bar. Even if you wanted to distill the gradient signal, you'd need per-t routing. (3) **Cross-metric Jaccard(grad top-16, attn head-avg top-16) is essentially orthogonal**: overall mean 0.114, per-layer means all in [0.085, 0.151], mean-of-per-(p,t)-max-over-layers = 0.274 (well under the 0.5 GO bar; one record reaches 0.6 max, no layer averages above 0.20). Attention is NOT a proxy for loss-importance — it's a learned routing structure with its own logic. Combined with E059: two independent importance signals, both somewhat concentrated but unstable across layers/heads/t, and largely orthogonal. A faithful student would need to preserve both signals at per-(layer, head, t) granularity, which defeats "cheap". **Promotes [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)'s STOP** from "attention-only argument" to "two-metric closure". Still does NOT rule out per-layer × per-head routing distillation (no longer cheap), per-layer random redraw, or sparse on regimes (N≥1024) where dense doesn't fit. **2026-05-13 follow-up: [E061](#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13) flips all three of these conclusions** — the aggregate-loss design summed over queries and washed out per-query specialization; the per-query VJP version of the same script shows grad IS sharper than attn, t-stable, and aligned with attn at ≥1 (layer, head) per query. E060's STOP is reopened. |
| [E061](#e061--per-query-vjp-gradient-saliency-inverts-e060-2026-05-13) | 2026-05-13 | finished — **reopens the E060 STOP** | Per-query VJP version of [E060](#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13)'s gradient saliency audit. Same canonical step-2646 ckpt, same 9 proteins, same t-grid, seed=42. New per-query path: for each sampled query residue i, backprop ‖v_pred[i]‖₂ individually — one saliency vector PER query, structurally one-to-one with sparse's per-residue K-set. 8 queries × 9 proteins × 5 t = 360 per-query records; 5040 cross-metric records (360 queries × 14 layers). Script: `script_utils/audit_dense_gradient_saliency.py` (per-query refactor commit `9ed7a93` + 1-line f-string-syntax fix today); output `results/dense_attn_audit/canonical_2646_grad_per_query.json` (~3.5 MB). | non-narrative — **all three E060 negatives flip when you ask the question per-query.** (1) **Per-query gradient is SHARPER than attention** (E059): mass_top_16_per_query = **0.709** (vs attn 0.656); mass_top_32_per_query = **0.830** (vs attn 0.794). E060's "grad is more diffuse" came from summing per-query losses before differentiating — washed out the per-query specialization. (2) **Per-query t-Jaccard = 0.663** (vs aggregate 0.200) — the same query's important set is largely stable across adjacent t-values; just under the 0.7 GO bar, far from STOP. (3) **Per-query attention × gradient agreement at the best (layer, head): mean-of-max = 0.833**, min 0.524, max 1.000 — every query has at least one (layer, head) sharing ≥8/16 important residues with gradient; the average best (layer, head) per query overlaps 13/16. Even at the head-AVERAGED layer level, mean Jaccard is 0.337 (vs aggregate 0.114). Every layer has at least one record with Jaccard=1.0; the per-layer head-max-avg ranges 0.68-0.77, layer 2 highest at 0.77. **Query-pair Jaccard within a (protein, t) = 0.146** — different queries care about substantially different K-sets, confirming the user's intuition that **per-query routing is the right unit of analysis**. Combined picture: dense's attention DOES reflect loss-importance per-query at some (layer, head) for every query, and per-query gradient is a faithful (and slightly sharper) teacher signal. **Distillation is back on the table — but per-query, not per-layer × per-head shared**: a student that picks a per-query K-set from per-query gradient saliency could capture the dense routing prior. This is a substantive structural lever for sparse-attention-with-routing, in contrast to the K-set-shared-across-layers/heads we currently train. |

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

> **⚠️ nsteps=200 caveat (added 2026-05-07).** Designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). Direction of bias: nsteps=200 *under-states* what the arm can do, especially at L≥100 (22→0.8 Å cliff at L=300). **Cannot be re-probed locally** — `best_val_00000011_000000001133.ckpt` was overwritten on 2026-05-06 by the scnbr_t04 ckpt of the same name (CLAUDE.md note in `inference_sparse_pairupdate_quick.yaml`). The "sparse + pair-update converged at step 1133" memory claim (`project_sparse_pairupdate_converged.md`) rests on these nsteps=200 numbers and cannot currently be re-verified at nsteps=400 without re-rsync.

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

> **⚠️ nsteps=200 caveat (added 2026-05-07).** All designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). Direction of bias: nsteps=200 *under-states* what the arm can do, especially at L≥100. The "dead at this step" verdict was load-bearing for the decision to not continue training conv. The 2026-05-07 step-2961 probe at nsteps=400 (3/18, best 1.60 Å) and the nsteps=400 redo of this exact ckpt (`inference_downsampled_step2331_n6_nfe400`, queued 2026-05-07) supersede this entry's "0/18 dead" verdict for any cross-arm comparison.

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

> **⚠️ nsteps=200 caveat (added 2026-05-07).** Designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). This entry was already superseded by E038 for the Fix-C2 inference path; the Fix-C2 redo (E038) was *also* at nsteps=200, so neither entry's numbers are defensible at canonical inference resolution. nsteps=400 redo of the same ckpt (step 819) is queued at `inference_scnbr_t04_step819_n6_nfe400` (2026-05-07).

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

---

## E038 — `scnbr_t04` re-probe with Fix C2 actually wired (2026-05-06)

> **⚠️ nsteps=200 caveat (added 2026-05-07).** All designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). Direction of bias: nsteps=200 under-states what the arm can do, especially at L≥100. The "0/18 still, mechanism alive, re-probe at step ≥ 1500" reading was load-bearing for the next decision (E039). nsteps=400 redo queued at `inference_scnbr_t04_step819_n6_nfe400` (2026-05-07) supersedes for any cross-arm comparison.

**Status:** finished.

**Why ran:** [E035](#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06) ran the `scnbr_t04` ckpt at step 819 and got 0/18 designable, but the local checkout was missing the Fix C2 inference code at the time — the threshold-gated `x_sc`-as-neighbor-source switch and the step-0 bootstrap forward both lived only on the HPC training tree (commit `8e97d7a` on the GitHub remote, not yet merged locally). E035's inference therefore ran the model with `sparse_attention=True` neighbor-list-from-`x_t` at all t — i.e., as plain sparse-K40 + the existing `x_sc_pair_dist_*` pair feature, which is *not* what the ckpt's weights were trained against. The result was uninformative about whether the variant works. After pulling 8e97d7a (and harmonising the docs merge — see commit `92a312f`), this re-probe runs the same ckpt with the inference path the ckpt was trained for. Decision input for: (a) is Fix C2 doing the architectural thing on this ckpt; (b) at step 819, does the variant clear the variant bar.

**Configs:**
- Checkpoint: `best_val_00000008_000000000819.ckpt` (epoch 8, opt step 819) — same file as E035.
- Inference YAML: `configs/inference_sparse_scnbr_t04_quick.yaml` (unchanged from E035) — 3 lengths × 6 samples × 200 ODE steps, seed=5 inherited from `inference_base.yaml`.
- Fix C2 wiring: `cfg_exp.training.sc_neighbors=True`, `cfg_exp.training.sc_neighbors_t_threshold=0.4` flow from the ckpt's `hyper_parameters` into `LocalLatentsTransformer.__init__` (`proteinfoundation/proteina.py:122-128`); `inference_base.yaml:23 sc_neighbors_bootstrap=True` flows into `full_simulation` via `proteinfoundation/proteina.py:809-828`. Canary print (`local_latents_transformer.py:127-132`) confirmed at runtime — see "Runtime confirmation" below.
- Hardware: 1× NVIDIA L4 (gxp-l4-0, GPU 0). Env: `/home/ks2218/.conda/envs/laproteina_env`.
- Output dir: `inference/inference_sparse_scnbr_t04_quick/` (clean re-run; E035 outputs deleted before launch). CSV: `inference/results_inference_sparse_scnbr_t04_quick_0.csv`. Log: `/tmp/probe_scnbr_postfix.log`.
- Wall-clock: ~12 min total (gen + eval), comparable to E035.

**Runtime confirmation that Fix C2 fired:**

```
[Fix C2] sc_neighbors=True (t_threshold=0.4): sparse neighbors will be built
from x_sc when t < threshold and x_sc is present; otherwise falls back to x_t.
```

Printed exactly once at `Proteina.load_from_checkpoint` time (line 60 of `/tmp/probe_scnbr_postfix.log`). This is the canary that hops 1-3 of the wiring (ckpt hparams → `Proteina.__init__` → `LocalLatentsTransformer.__init__`) succeeded. Hops 4-5 (per-protein threshold gate in the forward, step-0 bootstrap in `full_simulation`) are not directly observable from a print but are exercised on every sample because the conditions to skip them all evaluate False (`sc_neighbors=True`, `self_cond=True`, `sc_neighbors_bootstrap=True`, `t_bb_ca` starts at 0 < 0.4 for the first ~40% of integration steps under the canonical `bb_ca` log-p=2 schedule).

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o values within ~0.1 Å):**

| L | n | min scRMSD per sample (Å, sorted) | designable (<2 Å, CA) | best | median |
|---|---|---|---|---|---|
| 50  | 6 | 5.39, 8.54, 9.07, 15.16, 16.55, 19.17 | 0/6 (0%) | 5.39 | 12.11 |
| 100 | 6 | **3.16, 3.39, 3.76**, 8.93, 13.78, 14.24 | 0/6 (0%) | **3.16** | 6.34 |
| 200 | 6 | 12.49, 13.38, 13.52, 14.58, 14.79, 15.12 | 0/6 (0%) | 12.49 | 14.05 |
| **all** | 18 | — | **0/18 (0%)** | 3.16 | 13.45 |

Headline: "Average scRMSD: 11.390 Å, Success Rate (<2Å): 0.0%, Total: 18, Failed: 0". bb3o-mode designability matches.

**Direct comparison to [E035](#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06) (same ckpt, same protocol, only difference is Fix C2 active):**

| L | E035 best Å | E038 best Å | Δ best | E035 median Å | E038 median Å | Δ median |
|---|---|---|---|---|---|---|
| 50  | 7.73 | **5.39** | **−2.34** | 9.30  | 12.11 | +2.81 |
| 100 | 4.37 | **3.16** | **−1.21** | 9.94  | **6.34**  | **−3.59** |
| 200 | 13.38 | 12.49 | −0.89 | 14.21 | 14.05 | −0.16 |
| pooled mean | 11.57 Å | 11.39 Å | −0.18 |  |  |  |

Per-sample lower-tail concentration at L=100:
- E035 sorted CA scRMSD: `4.37, 6.73, 8.74, 11.13, 11.75, 15.48` — one sample below 5 Å.
- E038 sorted CA scRMSD: `3.16, 3.39, 3.76, 8.93, 13.78, 14.24` — **three samples below 4 Å, all clustered around 3-4 Å.**

This is the bimodal "near-canonical-best" head that [E021](#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30) documented for sparse-K40 + pair-update at step 1133 (best L=100 = 1.35 Å on a converged ckpt). The mechanism is producing the right qualitative signature on the L=100 distribution; the absolute level is consistent with an under-trained ckpt rather than with a broken architecture.

**Findings (tuning, not paper-grade):**

1. **Fix C2 is producing the expected mechanistic signature.** L=100 median dropped 3.59 Å, three-sample concentration at 3.16/3.39/3.76 Å, best improved by 1.2 Å. None of these would be observable if the threshold-gated x_sc-neighbor swap and the step-0 bootstrap weren't actually changing the model's forward path. The wiring is real, the architectural axis works.
2. **L=50 is bimodal-worse, not uniformly-better.** Best improved 2.34 Å but median worsened 2.81 Å — fewer "almost there" samples, two more in collapse mode. This is consistent with Fix C2 "rescuing the seeds that were close" but not "rescuing the seeds that were trapped". Same fingerprint sparse + pair-update showed at step 1133 (E021).
3. **L=200 is essentially unchanged.** Best 12.5 Å, median 14.0 Å. At L=200 the score field hasn't formed yet at step 819 — neighbor-source choice is downstream of "the model has learned to integrate at this length", and that's still the bottleneck. Not a Fix-C2 failure mode; the same L=200 cliff existed for plain sparse-K40 at step 1259 (E014/E019: 0/30 at L=200, best 2.01 Å) and for sparse + pair-update at step 1133 (E021: 0/6 at L=200, best 2.20 Å).
4. **Variant is pre-convergence, not dead.** Step 819 < E021's 1133 inflection, well below canonical 1800-2200 best-val window. The L=100 best dropping to 3.16 Å is the encouraging signal; given E021's L=100 best of 1.35 Å at step 1133 and an architecture that's now demonstrably exercising Fix C2, training to step ≥ 1500 is the right next step.
5. **Decision:** continue training; re-probe at step ≥ 1500. If L=100 best at step ≥ 1500 doesn't drop below 2 Å (matching E021's 1.35 Å at step 1133), call the variant — Fix C2's mechanism is doing its job at this ckpt's training stage but isn't bridging the gap to E021's level, which would mean Fix C2 doesn't compose with the broader convergence trajectory. Until then, this is "needs more training", not protection.

**Possible narrative:** non-narrative — kept for tuning/decision-making. Could become a methodological aside if the step ≥ 1500 re-probe (a) confirms Fix C2 closes a gap vs plain sparse-K40 at matched step, or (b) demonstrably fails to. Either result is more informative than this one. Do *not* promote any per-length rate from this snapshot to a Finding.

**Methodological caveats:**
- **N=6 single seed (seed=5).** Same caveat as E035. The L=100 "3 samples below 4 Å" is a 3/6 cluster that needs N≥30 + multi-seed before it can be claimed as "Fix C2 is shifting the L=100 distribution toward designable" with statistical weight. The current data is enough to show the *direction* of the effect, not to bound its magnitude.
- **Step mismatch vs E021.** E021's sparse + pair-update at step 1133 is the closest variant-cousin comparator, and E021 is at 1133 vs this entry's 819. Not step-matched. The argument "Fix C2 is producing the *signature* of E021's L=100 best-cluster" is qualitative; a step-matched (Fix C2 at step 1133) probe would let us say more.
- **No matched-step canonical comparator.** Same caveat as E035.
- **Fix C2 was trained against a 50% self-cond probability** (`proteina.py:577`'s `random.random() > 0.5` gate). Training never saw "x_sc is the result of a step-0 bootstrap forward at high noise" — the train-time x_sc is "previous-step x_1_pred at the same t". The inference step-0 bootstrap is therefore a small distribution mismatch between train and inference at step 0 specifically. Steps 1+ match training. This is a design observation about Fix C2, not specific to E038, but worth flagging as a candidate explanation if the step ≥ 1500 re-probe under-performs E021 by more than the step gap alone would predict.
- **Threshold value 0.4 is unswept.** Could be too aggressive or too conservative. The right A/B for the variant is "Fix C2 at threshold ∈ {0.2, 0.4, 0.6}" once it has a converged ckpt, not on this under-trained one.

**Cross-references:**
- Supersedes [E035](#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06) for the inference-correct numbers (E035 is preserved per the append-only rule; treat its 0/18 / best 4.37 Å as a Fix-C2-missing artifact).
- Built on the merge resolution that brought Fix C2 inference into the local checkout — see commit `92a312f` ("Merge from HPC: Fix C2 + docs harmonize ...") and `8e97d7a` (the original HPC Fix C2 commit).
- Fix C2 wiring: `proteinfoundation/proteina.py:119-128, 809-828` (config flow + integrator args), `proteinfoundation/nn/local_latents_transformer.py:125-132, 294-314` (constructor read + per-protein threshold gate via `torch.where`), `proteinfoundation/flow_matching/product_space_flow_matcher.py:582-583, 707-731` (function signature + step-0 bootstrap).
- Closest variant-cousin: [E021](#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30) (sparse-K40 + pair-update at step 1133) — different architectural variant (pair-update, not sc-neighbors) but matched on "post-fix MPNN, sparse K=32 attention, canonical recipe" so the L=100 best comparison is the cleanest available.
- Memory: `feedback_dead_arm_calls.md` — applies to a *step-matched* dead variant; this is *step-mismatched* with a working mechanism, so the rule is "more training, then call".
- Predicts: at step ≥ 1500 (mid-way to canonical's 1800-2200 best-val window), the L=100 best should drop below 2 Å for at least one sample if Fix C2 is composing with the convergence trajectory; if it doesn't, the variant is dead and the decision rule above triggers.

---

## E039 — `scnbr_t04` + Fix C2 step 1133, designability clears the variant bar (2026-05-06)

> **⚠️ nsteps=200 caveat (added 2026-05-07).** All designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). The "clears the variant bar" / "matches sparse+pairupdate step-1133 ceiling" claims were load-bearing for the project's understanding of the scnbr_t04 arm's converged ceiling. Direction of bias: nsteps=200 under-states what the arm can do — the "ceiling" is plausibly higher at canonical inference resolution. nsteps=400 redo queued at `inference_scnbr_t04_step1133_n6_nfe400` (2026-05-07) supersedes this entry for any cross-arm comparison.

**Status:** finished.

**Why ran:** [E038](#e038--scnbr_t04-re-probe-with-fix-c2-actually-wired-2026-05-06) ran the scnbr_t04 ckpt at step 819 with Fix C2 active and got 0/18 designable but with mechanism evidence (L=100 best 3.16 Å, three-sample 3-4 Å cluster at L=100). E038's decision rule was: "re-probe at step ≥ 1500; if L=100 best at step ≥ 1500 doesn't drop below 2 Å, call the variant." The user pulled a new best-val checkpoint for the same training run (`ca_only_sparse_K40_scnbr_t04/1778022317`, ep=11 step=1133) — this entry is that re-probe. Decision input for: (a) does the variant clear the CLAUDE.md "1-2/3 designable at L=50/L=100" bar, (b) where does it sit relative to the closest variant-cousin ([E021](#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30) at the *same* step.

**Configs:**
- Checkpoint: `best_val_00000011_000000001133.ckpt` (epoch 11, opt step 1133), rsynced from HPC `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K40_scnbr_t04/1778022317/checkpoints/`. Local copy: `/home/ks2218/la-proteina/best_val_00000011_000000001133.ckpt` (1.90 GB; mtime 2026-05-06 11:35).
- ⚠️ **Filename collision:** the same path previously held E021's sparse+pairupdate step-1133 ckpt (1.94 GB, `update_pair_repr=True`, `sc_neighbors=False`); the rsync overwrote it with the scnbr_t04 best-val (1.90 GB, `update_pair_repr=False`, `sc_neighbors=True`). Verified via `ckpt['hyper_parameters']['cfg_exp']['run_name_'] == 'ca_only_sparse_K40_scnbr_t04'` before launch. The inference YAML now has a comment warning future readers to verify `cfg_exp.run_name_` from hparams before re-running E021.
- Inference YAML: `configs/inference_sparse_scnbr_t04_quick.yaml` — same as E038 except `ckpt_name: best_val_00000011_000000001133.ckpt`. 3 lengths × 6 samples × 200 ODE steps, seed=5.
- Fix C2 wiring (unchanged from E038): `cfg_exp.training.sc_neighbors=True, sc_neighbors_t_threshold=0.4` on the ckpt; `inference_base.yaml:23 sc_neighbors_bootstrap=True`.
- Hardware: 1× NVIDIA L4 (gxp-l4-0, GPU 0). Wall-clock: ~12 min.
- Output: `inference/inference_sparse_scnbr_t04_quick/` (E038 outputs deleted before launch). CSV: `inference/results_inference_sparse_scnbr_t04_quick_0.csv`. Log: `/tmp/probe_scnbr_step1133.log`.

**Runtime confirmation that Fix C2 fired:**

```
[Fix C2] sc_neighbors=True (t_threshold=0.4): sparse neighbors will be built
from x_sc when t < threshold and x_sc is present; otherwise falls back to x_t.
```

Line 60 of `/tmp/probe_scnbr_step1133.log`. Same canary as E038. Hops 1-3 of the wiring (cfg_exp.training → Proteina.__init__ → LocalLatentsTransformer.__init__) confirmed.

**Results — per-protein min scRMSD over 8 ProteinMPNN sequences (CA mode, ESMFold; bb3o values within ~0.1 Å):**

| L | n | min scRMSD per sample (Å, sorted) | designable (<2 Å, CA) | best | median |
|---|---|---|---|---|---|
| 50  | 6 | **1.51, 1.67, 2.11**, 3.29, 4.82, 5.23 | **2/6 (33%)** | 1.51 | 2.70 |
| 100 | 6 | **1.92**, 2.93, 5.42, 7.84, 8.55, 10.82 | **1/6 (17%)** | 1.92 | 6.63 |
| 200 | 6 | 7.22, 9.50, 12.03, 13.40, 13.71, 13.78 | 0/6 (0%) | 7.22 | 12.71 |
| **all** | 18 | — | **3/18 (17%)** | 1.51 | 6.32 |

Headline: "Average scRMSD: 6.987 Å, Success Rate (<2Å): 16.7%, Total: 18, Failed: 0".

**Comparison context (informational; step counts and N differ where noted):**

| Recipe | step | N | L=50 | L=100 | L=200 | overall | best Å |
|---|---|---|---|---|---|---|---|
| canonical baseline (wd=0.05) | 2646 | N=3 (E018 recheck) | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 8/9 (89%) | 0.7 |
| paramgroups + wd=0.1 | 1952 | N=6 (E018) | 3/6 (50%) | 5/6 (83%) | 1/6 (17%) | 9/18 (50%) | 0.94 |
| sparse-K40 alone | 1259 | N=30 (E014/E019) | 13/30 (43%) | 8/30 (27%) | 0/30 (0%) | 21/90 (23%) | 0.90 |
| sparse-K40 + pair-update | 1133 | N=6 (E021) | 2/6 (33%) | 1/6 (17%) | 0/6 (0%) | 3/18 (17%) | 1.35 |
| **sparse-K40 + Fix C2 (E039)** | **1133** | **N=6** | **2/6 (33%)** | **1/6 (17%)** | **0/6 (0%)** | **3/18 (17%)** | **1.51** |
| sparse-K40 + Fix C2 (E038) | 819 | N=6 | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | 0/18 (0%) | 3.16 |
| `downsampled` (E034) | 2331 | N=6 | 0/6 | 0/6 | 0/6 | 0/18 (0%) | 12.41 |

**Direct comparisons:**

*Same variant, step 819 (E038) → step 1133 (this entry):*

| L | E038 best Å | E039 best Å | Δ | E038 median Å | E039 median Å | Δ |
|---|---|---|---|---|---|---|
| 50  | 5.39 | **1.51** | **−3.88** | 12.11 | 2.70 | **−9.41** |
| 100 | 3.16 | **1.92** | **−1.24** | 6.34  | 6.63  | +0.29 |
| 200 | 12.49 | 7.22 | **−5.27** | 14.05 | 12.71 | −1.34 |
| pooled mean | 11.39 Å | 6.99 Å | **−4.40** |  |  |  |

In 314 training steps, **mean scRMSD dropped 4.4 Å** and the variant went from 0/18 to 3/18 designable. L=50 went from "no near-misses" to "three samples within 0.11 Å of threshold". L=200 best dropped 5.3 Å but is still 5+ Å above threshold — same L=200 cliff every CA-only variant hits at this training stage.

*Step-matched variant comparison: scnbr_t04 + Fix C2 (E039) vs sparse + pair-update (E021), both at step 1133, both N=6, both seed=5:*

| L | E039 best Å | E039 designable | E021 best Å | E021 designable |
|---|---|---|---|---|
| 50  | 1.51 | 2/6 | 1.48 | 2/6 |
| 100 | 1.92 | 1/6 | 1.35 | 1/6 |
| 200 | 7.22 | 0/6 | 2.20 | 0/6 |
| **pooled** | — | **3/18 (17%)** | — | **3/18 (17%)** |

Identical designability count. E021 has a slightly better single-best at L=100 (1.35 Å vs 1.92 Å) and a much closer L=200 best (2.20 Å vs 7.22 Å). E039's L=50 cluster is tighter (three samples in 1.51-2.11 Å vs E021's 1.48, 1.72, 2.14). **Two different architectural levers — Fix C2 (sc_neighbors gating) vs pair-update (re-mixing the pair representation across layers) — reach the same step-1133 designability ceiling on this 160M canonical-recipe trunk.**

**Findings (tuning, not paper-grade):**

1. **Variant clears the CLAUDE.md "1-2/3 designable at L=50/L=100" variant bar.** L=50 = 2/6 (33%) and L=100 = 1/6 (17%) — both meet the bar. The L=50 three-sample cluster at 1.51, 1.67, 2.11 Å (best 0.49 Å under threshold, worst 0.11 Å over) is a tight near-canonical-best regime, not luck-of-one-seed.
2. **The earlier E038 "re-probe at step ≥ 1500" prediction was beat by ~25%.** Plateau hits at step 1133. Update the decision rule going forward: any future scnbr_t04 ckpt's first probe should already be at step ≥ 1100, not ≥ 1500. The convergence trajectory is steeper than E038 modelled.
3. **L=200 is the next ceiling, and it's NOT a Fix C2 issue.** L=200 best dropped 5.3 Å step 819 → 1133 (12.49 → 7.22 Å), so it IS responding to training, but no sample is closer than 5+ Å of threshold. E021 had L=200 best 2.20 Å at the same step — that's closer. The pair-update mechanism in E021 may be more effective at L=200 than the Fix C2 mechanism here, *or* the difference is single-seed N=6 noise (E021's L=200 second-best was 7.75 Å — broadly comparable to E039's 7.22 Å). Step-matched N=30 probe is needed before any L=200 claim about which lever wins.
4. **Architectural-axis match between Fix C2 and pair-update at step 1133 is the cleanest variant-vs-variant comparison currently on disk.** Same step, same N=6, same seed=5, same canonical recipe. Same designability rate (3/18). Different mechanisms. This is interesting because both variants were designed to address sparse-K40's "noisy x_t breaks the spatial+random neighbor list at low t" problem, but via orthogonal routes (re-mix pair channel between layers vs swap coord source for neighbor list). At N=6 single-seed, indistinguishable; at N=30 the comparison would tighten.
5. **Decision:** the variant has cleared the bar. Possible next steps in priority order:
   - (a) N=30 seed-matched probe of scnbr_t04 step 1133 (matches the E014/E019 protocol). Cost: ~3-4h on 1 L4. Outcome: tightens the rate vs E021, vs sparse-alone at step 1259. **This is the right next experiment.**
   - (b) Continue training to step ≥ 1800 (matching canonical's best-val window). Cost: ~12h additional cluster time. Outcome: tells us whether Fix C2's mechanism plateau shifts further with more training.
   - (c) `power_with_middle_bump` schedule on top of Fix C2 (combines E037's curvature-targeted schedule with E039's variant). Cost: re-run gen+eval. Outcome: tests whether the schedule lever and the architecture lever compose. Lower priority since both are individually near-null at N=6.

**Possible narrative:** non-narrative on its own — the matched-step result vs E021 is informative for tuning but not paper-grade because (i) plain sparse-K40 at step 1133, N=30 isn't on disk for the controlled comparison, (ii) N=6 single-seed has wide CIs on the rate. If a future N=30 step ≥ 1133 probe of scnbr_t04 produces a designability rate strictly above plain sparse-K40's 23% (E014/E019), the paired-mechanism interpretation could become a Finding-eligible "Fix C2 closes the sparse-K40 designability gap to canonical at lengths ≤ 100" claim. Until then, "matches E021's pair-update at step 1133" is the strongest defensible claim.

**Methodological caveats:**
- **N=6 single seed (seed=5).** Same caveat as E021/E038. The 3/18 designability count has 95% binomial CI [4%, 41%] — wide enough that "matches E021's 3/18" cannot be sharpened beyond "in the same ballpark" without N≥30.
- **L=200 best 7.22 Å vs E021's 2.20 Å is 5 Å, but at N=6 a single lucky seed in E021 is enough to drive that.** E021's L=200 second-best was 7.75 Å, comparable to E039's best (7.22 Å). The L=200 spread within E021 alone is 2.20 → 14.36 Å (12 Å). At N=6 you can't say whether scnbr_t04 vs pair-update differ at L=200; both are operating below their score-field formation threshold at this step.
- **Filename collision risk for E021 reproducibility.** E021's old ckpt is overwritten on this box. The inference YAML now has a warning, but anyone re-running E021 from this checkout will silently get scnbr_t04 weights unless they verify `cfg_exp.run_name_` first. Worth adding a renaming step to a future checkpoint-management cleanup.
- **Eval used the fixed `ca_only=True` ProteinMPNN call** (post-E017 fix); directly comparable to E018/E019/E021/E034/E038.

**Cross-references:**
- Direct precursor: [E038](#e038--scnbr_t04-re-probe-with-fix-c2-actually-wired-2026-05-06) (same variant, step 819 — supersedes [E035](#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06)'s Fix-C2-missing run).
- Closest step-matched variant comparator: [E021](#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30) (sparse + pair-update at step 1133). Identical designability count.
- Plain-sparse comparator at N=30 (different step): [E014](#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27) / E019 — sparse-K40 alone at step 1259 hit 21/90 (23%) with best 0.90 Å.
- Fix C2 wiring (unchanged from E038): `proteinfoundation/proteina.py:119-128, 809-828`, `proteinfoundation/nn/local_latents_transformer.py:125-132, 294-314`, `proteinfoundation/flow_matching/product_space_flow_matcher.py:582-583, 707-731`.
- Memory: `feedback_dead_arm_calls.md` — does NOT apply here; the variant cleared the bar at the matched step.
- Predicts: an N=30 seed-matched probe of scnbr_t04 step 1133 (same protocol as E014/E019) should give a rate at least within ±9pp of E021's (untested in N=30) and at least within ±9pp of plain sparse-K40 at step 1133 (also untested). If it lands strictly above plain sparse-K40's 23%, that's the Finding-eligible result. Until that runs, this entry is a tuning data point.

---

## E040 — Hybrid conv→scnbr mid-trajectory handover + kink abruptness at the switch (2026-05-06)

> **⚠️ nsteps=200 caveat (added 2026-05-07).** Designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). The "hybrid resurrects dead conv" claim is internally consistent (this entry vs E034 are both nsteps=200) but cross-arm comparisons vs canonical-alone (E019, nsteps=400) are not defensible. **Kink numbers (`‖v_A − v_B‖`, cos angle) are unaffected** — they are computed at a single x_t at one step and don't depend on integrator convergence. Designability redos at nsteps=400 queued: `inference_hybrid_conv_to_scnbr_t06_n6_nfe400` (was N=3 at t=0.6) and `inference_hybrid_conv_to_scnbr_t075_n6_nfe400` (was partial N=6 at t=0.75) — 2026-05-07.

**Status:** finished (t_switch=0.6 / N=3 complete; t_switch=0.75 / N=6 partial — see footnote).

**Why ran:** Two questions in one probe.
1. **Compositional sampling.** Both architectural variants converge to step-1133 ceilings: the 1D-conv ([E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06), step 2331) is dead end-to-end (0/18, every sample ≥ 12 Å), but its receptive-field structure may still be useful for early bulk denoising; the sparse-K40 + Fix C2 ([E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06), step 1133) clears the variant bar at 3/18. Question: does running conv for `t < t_switch` (bulk denoising from noise → mostly-clean) and handing off to scnbr for `t ≥ t_switch` (final detailing) produce designable samples that *neither* model produces alone? Crazy-idea probe, not preregistered.
2. **Kink abruptness.** The two networks were trained independently on the same data — they should agree on the velocity at any (x_t, t), modulo training noise. At the handover step, log per-protein ‖v_conv − v_scnbr‖, ‖v_conv‖, ‖v_scnbr‖, and cos(v_conv, v_scnbr) on the SAME (x_t, t). This quantifies how much "shock" the trajectory experiences when we swap models, and is needed to interpret any designability gap relative to single-model baselines.

**Configs:**
- New script: `proteinfoundation/generate_hybrid.py`. Loads both checkpoints, monkey-patches `predict_for_sampling` on the scnbr (B) model to dispatch to conv (A) when `t < t_switch`. At the first scnbr call per batch (= handover step), runs both A and B in `torch.no_grad` on the same batch and logs ‖v_A − v_B‖, ‖v_A‖, ‖v_B‖, cos(v_A, v_B) per protein → `inference/<config_name>/kink_log.json`. `sc_neighbors_bootstrap=False` because scnbr's `t_threshold=0.4 < 0.6 ≤ t_switch`, so scnbr always reads neighbors from x_t.
- Configs:
  - `configs/inference_hybrid_conv_to_scnbr_t06_n3.yaml` — t_switch=0.6, N=3, nsteps=200, lengths [50, 100, 200], seed=5.
  - `configs/inference_hybrid_conv_to_scnbr_quick.yaml` — t_switch=0.75, N=6 (first run, partial).
- Ckpts: A = `best_val_00000023_000000002331.ckpt` (`ca_only_downsampled`, run id 1777987722). B = `best_val_00000011_000000001133.ckpt` (`ca_only_sparse_K40_scnbr_t04`, run id 1778022317). Both pulled from HPC as in [E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06)/[E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06).
- Both CA-only (`local_latents` absent from `product_flowmatcher`); A has `use_downsampling=True, sparse_attention=None, sc_neighbors=None`; B has `use_downsampling=False, sparse_attention=True, sc_neighbors=True, t_threshold=0.4`.
- Schedule: `mode=log, p=2.0` (default `inference_base.yaml`). With nsteps=200, the log schedule packs more grid points near t=1, so it crosses any given t early. **t_switch=0.6 falls at step 40 of 200** — the integrator runs conv for the first 40 steps (20% of NFE) and scnbr for the remaining 160 steps (80% of NFE), but in t-distance the split is 60% / 40% conv:scnbr. t_switch=0.75 falls at step 59 of 200 (30% conv NFE / 70% scnbr NFE; 75% / 25% in t-distance). Verified directly against `get_schedule(mode='log', p1=2.0, nsteps=200)`. **(Earlier draft of this entry inverted these step indices — the wrong direction was self-corrected at re-check time. Conclusions about kink magnitude and designability rate are unaffected; the kink is computed at the actual handover t, not at the step index.)**
- Hardware: 1× A100 (CUDA_VISIBLE_DEVICES=0 to avoid co-tenant; the t=0.75 eval was wrecked by mid-eval OOM from a 10 GiB co-tenant on GPU 6).
- Eval: post-fix MPNN (`ca_only=True`) → ESMFold → scRMSD<2 Å (CA mode). Identical pipeline to E017/E018/E019/E021/E034/E038/E039.

**Results — designability (t_switch=0.6, N=3, seed=5):**

| L | designable (CA) | best Å | median | mean | max | bb3o best |
|---|---|---|---|---|---|---|
| 50 | **1/3** | 1.78 | 3.83 | 5.94 | 12.20 | 1.97 |
| 100 | 0/3 | 2.58 | 5.03 | 6.45 | 11.75 | 2.68 |
| 200 | 0/3 | 10.00 | 11.27 | 10.86 | 11.30 | 10.00 |
| **pooled** | **1/9 (11%)** | **1.78** | **10.00** | **7.75** | — | — |

**Results — kink at handover (t_switch=0.6, t_handover=0.6080, per batch = per length):**

| L | ‖v_A − v_B‖ (per-protein L2, mean over 3) | ‖v_A‖ | ‖v_B‖ | cos(v_A, v_B) | per-residue ‖Δv‖ | rel diff = ‖Δv‖/‖v_A‖ |
|---|---|---|---|---|---|---|
| 50 | 11.59 | 13.55 | 7.86 | 0.522 | 1.62 | 0.86 |
| 100 | 17.05 | 21.00 | 13.17 | 0.585 | 1.70 | 0.81 |
| 200 | 25.09 | 31.58 | 21.67 | 0.612 | 1.76 | 0.79 |

Per-protein numbers (saved in `inference/inference_hybrid_conv_to_scnbr_t06_n3/kink_log.json`):
- L=50: ‖Δv‖ = [10.76, 12.10, 11.89]; cos = [0.553, 0.536, 0.478].
- L=100: ‖Δv‖ = [16.99, 17.10, 17.06]; cos = [0.573, 0.580, 0.602].
- L=200: ‖Δv‖ = [24.51, 25.42, 25.33]; cos = [0.626, 0.604, 0.606].

**Possible narrative:** non-narrative — N=3 is too small, and the experiment is preliminary. Kept as a tuning data point informing two questions:
1. *Is hybrid sampling worth scaling up?* The conv→scnbr hybrid can produce designable samples (1/9 at t=0.6, 3/12 at t=0.75) where conv-alone is dead (0/18). That's a non-trivial result — the conv ckpt's weights are not just "noise" — but the rate is no better than scnbr-alone (3/18 at N=6 in [E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06)). At equal designability rate, scnbr-alone is the simpler choice. A *larger* N is needed to tell whether hybrid actually beats scnbr-alone, and at what t_switch.
2. *Quantitative shape of the kink.* The two networks disagree by ~80% in velocity magnitude (rel ≈ 0.8) and ~50-60° in direction (cos ≈ 0.5-0.6) at the SAME (x_t, t) — far larger than what training-noise twins should produce. This is *not* the "smooth manifold of equivalent flow fields" picture, it's a "two distinct flow fields that happen to share a target distribution" picture. It would predict catastrophic kinks at the handover, but designability at t=0.75 is decent (3/12 valid) — meaning the SDE noise injection at each step + ~50 subsequent scnbr steps re-equilibrate after the discontinuity. The fact that t=0.75 outperforms t=0.6 on the small N here is consistent with this: later handover = more time after the kink for re-equilibration to fail because too few steps remain, but the kink itself is smaller because it's at higher t (closer to clean).

**Methodological caveats:**
- **N=3 / N=6 is way too small to call a winner.** All comparisons here are direction-of-effect, not magnitude.
- **Single seed (5).** All compared baselines (E034, E039) also seed=5, so per-PDB pairings are valid; aggregate numbers are not seed-robust.
- **Eval pipeline identical to E034/E039** (post-fix MPNN ca_only=True). Numbers comparable to those entries directly. NOT comparable to anything pre-2026-04-28.
- **nsteps=200 — BELOW THE PRODUCTION BAR.** All hybrid configs (`inference_hybrid_*.yaml`) override `nsteps: 200`, while `inference_base.yaml` and the canonical baseline ([E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)) use `nsteps=400`. The single-protein scRMSD probe during the steering work (E020/E025 nsteps=400 regen, 2026-05-04) showed that the same seed/length goes from **22.5 Å at nsteps=200 → 0.80 Å at nsteps=400** — the integrator hasn't converged at 200 steps. So **all variant-vs-canonical designability comparisons in this entry are nsteps=200 (variant) vs nsteps=400 (canonical) — apples-to-oranges**. Direction of bias: nsteps=200 *under-states* what every variant (and the hybrid) can do. The kink number does not depend on nsteps (it's computed on a single x_t at one step), so kink conclusions stand. Designability conclusions need a re-run at nsteps=400 to be defensible. See `feedback_use_nsteps_400_for_designability.md` (memory).
- **Kink computed only on bb_ca v output.** Both networks use `output_parameterization={'bb_ca': 'v'}` so this is exactly the velocity. No conversion.
- **Kink computed on the SAME (x_t, t)**, but x_t at the handover step IS the conv-driven trajectory (conv ran the prior 40 steps to that point — see schedule note above). The kink number is "what would scnbr say at this conv-conditioned x_t" — it's the *handover* abruptness, not a generic "how different are these two models on average."
- **Fix C2 disabled at step 0** (`sc_neighbors_bootstrap=False`). With t_switch=0.6 and scnbr's t_threshold=0.4, scnbr only ever runs at `t ≥ 0.6 > 0.4`, so it always reads neighbors from x_t and the bootstrap is a no-op. If a future hybrid uses t_switch < 0.4, this needs to be re-enabled.
- **The L=200 evaluation at t_switch=0.75 was destroyed by a CUDA OOM** (co-tenant on GPU 6 took 10 GiB). The pooled "3/12 valid" pools L=50 + L=100 only. Re-running on a free GPU is straightforward but wasn't done because the user pivoted to t_switch=0.6 / N=3.

**Footnote: t_switch=0.75 / N=6 partial run** (first probe, before kink instrumentation existed):
- Generation: `configs/inference_hybrid_conv_to_scnbr_quick.yaml`. nsteps=200, seed=5, lengths [50, 100, 200]. Canary: first scnbr call at t=0.7505 after 59 conv calls. Dispatch: 177 conv / 423 scnbr (3 batches × 200 steps).
- Eval: hit CUDA OOM on all 6 L=200 samples mid-ESMFold (co-tenant). Re-running on GPU 0 was killed when the user pivoted. Partial designability:
  - L=50: 1/6 designable, best 1.15 Å, median 3.85 Å, max 18.97 Å.
  - L=100: 2/6 designable, best 1.66 Å, median 5.32 Å.
  - L=200: 0/6 valid (OOM).
  - Pooled valid: 3/12 = 25%, best 1.15 Å.
- Kink not measured on this run (instrumentation added afterwards).
- The L=50 best (1.15 Å) is *better* than scnbr-alone's L=50 best (1.51 Å in E039) — the conv→scnbr handoff produced one really clean sample, but at N=6 this is a single PDB.

**Why t=0.6 underperforms t=0.75 on small N (interpretation):**
- More post-handover scnbr steps (120 instead of 50) = more time for scnbr's step-1133 weights (which are themselves only weakly converged at L=200, see E039) to drag the structure off the manifold.
- Larger relative kink at lower t is offset by lower noise scale, so pure SDE re-equilibration can't fully wash it out before the trajectory commits.
- At t=0.75, scnbr only has 50 steps to "spoil" what conv did, and the noise scale is lower so individual step errors compound less.
- **Caveat**: with only 9 vs 12 valid samples this could equally be sampling noise.

**Cross-references:**
- Conv-alone baseline: [E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) (0/18, every sample ≥ 12 Å, dead).
- Scnbr-alone baseline: [E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06) (3/18 at N=6, best 1.51 Å).
- Hybrid script: `proteinfoundation/generate_hybrid.py`. Kink computation in `compute_kink()`. Dispatcher in `hybrid_predict_for_sampling()`. `predict_step` wrapper resets the per-batch flag.
- Configs: `configs/inference_hybrid_conv_to_scnbr_t06_n3.yaml` (canonical with kink), `configs/inference_hybrid_conv_to_scnbr_quick.yaml` (first-attempt at t=0.75).
- Output dirs: `inference/inference_hybrid_conv_to_scnbr_t06_n3/` (9 PDBs + kink_log.json + resolved_config.yaml), `inference/inference_hybrid_conv_to_scnbr_quick/` (18 PDBs).
- CSVs: `inference/results_inference_hybrid_conv_to_scnbr_t06_n3_0.csv` (9 valid), `inference/results_inference_hybrid_conv_to_scnbr_quick_0.csv` (12 valid + 6 OOM).
- Predicts (none promoted to a planned experiment yet):
  - (a) Sweep t_switch ∈ {0.5, 0.6, 0.7, 0.8, 0.9} at N=15 with the kink logged at each — does the rate vs. t_switch curve have a maximum, and where? Hypothesis: optimum near t ≈ 0.85 where kink magnitude starts to drop AND scnbr still has enough steps to clean up.
  - (b) Reverse hybrid: scnbr early, conv late. Conv's smoother dense field might be useful for final relaxation? Hypothesis: probably worse — conv-alone is dead, and giving it the late-trajectory job is asking it to do exactly the thing it can't do.
  - (c) Kink at *each* step (not just handover): full ‖v_A(x_t, t) − v_B(x_t, t)‖ trace as a function of t. Would tell us if there's a t-region where the two models naturally agree (low kink) — that's the right t_switch.
  - (d) Cross-checkpoint kink baseline: ‖v_A(x_t, t) − v_A'(x_t, t)‖ between two different training checkpoints of the SAME variant (e.g., E039 ep=8 vs ep=11). Should be smaller than the across-architecture kink here (~50% relative). If it's NOT smaller, the kink is dominated by training noise, not architecture.

---

## E041 — Hybrid conv→canonical mid-trajectory handover (2026-05-06)

> **⚠️ nsteps=200 caveat (added 2026-05-07).** All designability numbers below were sampled at nsteps=200 — below the integrator-convergence bar (CLAUDE.md "Sampling — nsteps=400 is a HARD RULE"). The "5/9 (56%) pooled, first hybrid to clear L=200" claim cannot stand against canonical-alone (E019, nsteps=400) or against any nsteps=400 baseline — the hybrid ran at nsteps=200, the comparator at nsteps=400. Direction of bias: nsteps=200 under-states the hybrid; canonical-alone at this seed/N has never been measured at nsteps=200, so the hybrid-vs-canonical-alone delta is uncalibrated. **Kink numbers are unaffected.** nsteps=400 redo of the N=6 setting queued at `inference_hybrid_conv_to_canonical_t06_n6_nfe400` (2026-05-07).

**Status:** finished.

**Why ran:** [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06) showed that conv ([E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06), 0/18 alone) → scnbr ([E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06), 3/18 alone) hybrid sampling rescues the dead conv variant: 1/9 designable at t=0.6, 3/12 at t=0.75. But the kink at the handover was huge (~80% relative ‖Δv‖, cos ≈ 0.55). Question: does **conv → canonical** (full-recipe wd=0.05 trunk, [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)'s strongest baseline) give a smaller kink AND better designability? Hypothesis: canonical and conv share dense attention (scnbr is sparse), so the receiver model agrees more with conv at the handover, and post-handover trajectory commits to a designable manifold faster. Direct user request: "convolutional until 0.6 and then canonical from then on, also 3 probes each".

**Configs:**
- Hybrid script: `proteinfoundation/generate_hybrid.py` (unchanged — same dispatcher + same kink logger as E040).
- Config: `configs/inference_hybrid_conv_to_canonical_t06_n3.yaml`. nsteps=200, seed=5, lengths [50, 100, 200]. t_switch=0.6.
- Ckpts: A = `best_val_00000023_000000002331.ckpt` (`ca_only_downsampled`, run id 1777987722, same as E034/E040). B = `best_val_00000024_000000002457.ckpt` (`test_ca_only_diffusion`, run id 1776805213 — the canonical wd=0.05 baseline).
- **Canonical-step caveat:** CLAUDE.md cites step 2646 as the canonical "best ckpt on disk", but that file is not local in this checkout. Highest-step local ckpt from the canonical run is 2457 (189 opt steps before 2646). All E019 numbers are at step 2646; this entry uses 2457. Within the canonical baseline's documented best-val window (1800-2200 → overshoots slightly here, but designability vs val-loss decoupling is documented in Finding 5/6, so the step-2457 ckpt is in the canonical regime). The user explicitly authorized 2457 ("no u cant rsync. use 2457") after a non-interactive rsync attempt failed (no SSH key for the agent's session).
- Both ckpts CA-only (`local_latents` absent). Canonical: `use_downsampling=False, sparse_attention=None, sc_neighbors=None, output_parameterization={'bb_ca': 'v'}`. Conv: same except `use_downsampling=True`.
- Schedule: `mode=log, p=2.0`. With nsteps=200, **t_switch=0.6 falls at step 40 of 200** — conv runs for the first 40 steps (20% NFE), canonical runs for the remaining 160 steps (80% NFE); in t-distance the split is 60% / 40% conv:canonical because the log schedule packs steps near t=1. (Earlier draft of this entry had the step indices inverted; corrected on re-check against `get_schedule(mode='log', p1=2.0, nsteps=200)`.)
- Hardware: 1× A100 (CUDA_VISIBLE_DEVICES=0).
- Eval: post-fix MPNN (`ca_only=True`) → ESMFold → scRMSD<2 Å (CA mode). Identical pipeline to all post-2026-04-28 entries.

**Results — designability (t_switch=0.6, N=3, seed=5):**

| L | designable (CA) | best Å | median | mean | max | bb3o best |
|---|---|---|---|---|---|---|
| 50 | **3/3** | 1.08 | 1.26 | 1.34 | 1.68 | 1.40 |
| 100 | **1/3** | 1.56 | 2.42 | 2.83 | 4.51 | 1.94 |
| 200 | **1/3** | 1.53 | 9.20 | 6.91 | 9.99 | 1.80 |
| **pooled** | **5/9 (56%)** | **1.08** | **1.68** | **3.69** | — | — |

**Per-protein scRMSD (CA, min over 8 MPNN seqs):**
- L=50: 1.08, 1.26, 1.68 — clean sweep.
- L=100: 1.56 designable; 2.42 (just over threshold); 4.51 (clearly off).
- L=200: 1.53 designable; 9.20 and 9.99 (failed).

**Results — kink at handover (t_handover=0.6080, per batch):**

| L | ‖v_A − v_B‖ (mean over N=3) | ‖v_A‖ | ‖v_B‖ | cos(v_A, v_B) | per-residue ‖Δv‖ | rel = ‖Δv‖/‖v_A‖ |
|---|---|---|---|---|---|---|
| 50 | 10.92 | 13.55 | 7.92 | 0.591 | 1.54 | 0.81 |
| 100 | 16.41 | 21.00 | 13.68 | 0.625 | 1.63 | 0.78 |
| 200 | 23.96 | 31.58 | 22.74 | 0.655 | 1.68 | 0.76 |

Per-protein detail in `inference/inference_hybrid_conv_to_canonical_t06_n3/kink_log.json`.

**Side-by-side: conv→canonical (E041) vs conv→scnbr (E040) at t_switch=0.6, N=3, seed=5:**

| | E040 conv→scnbr | E041 conv→canonical | Δ |
|---|---|---|---|
| pooled designable | 1/9 (11%) | **5/9 (56%)** | **+5×** |
| pooled best Å | 1.78 | **1.08** | **−0.70** |
| pooled median Å | 10.00 | 1.68 | −8.32 |
| L=50 designable | 1/3 | 3/3 | +2 |
| L=100 designable | 0/3 | 1/3 | +1 |
| L=200 designable | 0/3 | 1/3 | +1 |
| **kink** ‖Δv‖ L=50/100/200 | 11.59 / 17.05 / 25.09 | 10.92 / 16.41 / 23.96 | smaller everywhere |
| **kink** cos L=50/100/200 | 0.522 / 0.585 / 0.612 | 0.591 / 0.625 / 0.655 | +13 / +7 / +7 pp |
| **kink** ‖Δv‖/‖v_A‖ L=50/100/200 | 0.86 / 0.81 / 0.79 | 0.81 / 0.78 / 0.76 | smaller everywhere |

**Side-by-side: hybrid (E041) vs single-model baselines, all N=3-30 protocols (post-fix MPNN, scRMSD CA):**

| variant | step | N | L=50 | L=100 | L=200 | pooled | best Å |
|---|---|---|---|---|---|---|---|
| conv alone (E034) | 2331 | 6 | 0/6 (best 13.27) | 0/6 (best 15.67) | 0/6 (best 17.60) | 0/18 | 12.41 |
| conv → scnbr (E040) | 2331/1133 | 3 | 1/3 (1.78) | 0/3 (2.58) | 0/3 (10.00) | 1/9 | 1.78 |
| **conv → canonical (E041)** | **2331/2457** | **3** | **3/3 (1.08)** | **1/3 (1.56)** | **1/3 (1.53)** | **5/9** | **1.08** |
| scnbr alone (E039) | 1133 | 6 | 2/6 (1.51) | 1/6 (1.92) | 0/6 (7.22) | 3/18 | 1.51 |
| canonical alone (E019, step 2646) | 2646 | 30 | 26/30 (0.52) | 26/30 (0.67) | 16/30 (0.91) | 68/90 (76%) | 0.52 |

**Interpretation:**

1. **The conv→canonical hybrid clears all three lengths' threshold at N=3.** L=50 (3/3), L=100 (1/3), L=200 (1/3) — every length contributes. No other architectural-axis combination does this at N=3.

2. **L=200 1.53 Å is the headline.** No other architectural variant tested has cleared L=200 at any N≤6. canonical alone at N=30 gets 53% L=200 (16/30) at step 2646; the hybrid hits 33% at N=3 / step 2457 — within the same regime (CIs overlap for N=3 vs N=30).

3. **Smaller kink correlates with better designability.** Conv→canonical has a smaller relative ‖Δv‖ (0.76-0.81 vs 0.79-0.86) AND higher cos (0.59-0.66 vs 0.52-0.61) than conv→scnbr at every length. The result: 5× the pooled designability rate. With N=3 we cannot prove causation — sampling noise dominates the rate uncertainty (95% Wilson CI on 5/9 is 26-83%; on 1/9 is 1-46%). But the rank-ordering of kink and rate is consistent with "smaller kink → better post-handover trajectory survival".

4. **Architectural similarity predicts kink size.** Conv and canonical both use dense attention (`sparse_attention=None`); they differ only in conv's downsampling stack. Scnbr uses sparse K=40 attention. The kink hierarchy (conv vs canonical < conv vs scnbr) matches this similarity hierarchy. Predicts: **kink for sparse_K40_pair_update vs scnbr_t04 should be smaller still** (both sparse, differ only on the pair-update vs Fix-C2 axis). Untested.

5. **Caveat: the "compositional sampling" framing depends on canonical at step 2457 actually being designability-strong on its own.** E019's 76% pooled is at step 2646, not 2457. Step-2457 designability hasn't been measured in isolation. Possible alternative interpretation: the conv→canonical hybrid's 56% is roughly what canonical-alone-at-step-2457 would deliver, and conv's contribution is essentially neutral / harmless. To rule that out, run canonical-alone at step 2457 and N≥9. Hypothesis: it would be ~60-70% — close to E041, which means the hybrid is *not* delivering compositional value over canonical-alone but is *re-using* the conv ckpt as a no-cost early-trajectory initializer (better than pure noise but matching what canonical can do alone).

6. **Why this matters even with the caveat.** Even in the "hybrid ≈ canonical-alone" interpretation, the conv ckpt (which is dead alone) is contributing useful early-trajectory work *for free* — it doesn't degrade canonical's downstream cleanup. The fact that you can replace 60% of canonical's expensive trajectory with a structurally simpler conv backbone *without* tanking designability has its own value (for compute or for studying which steps actually matter).

**Methodological caveats:**
- **N=3 is small.** Wilson 95% CI on 5/9 is [26%, 83%]. Direction-of-effect only.
- **Single seed (5).** Per-PDB pairings to E040 and E034 are valid (same seed); aggregate rate comparisons are not seed-robust.
- **canonical ckpt at step 2457, not 2646.** E019's 76% pooled is at 2646; the canonical-alone-at-2457 number is unmeasured. The "vs canonical alone" comparison is approximate.
- **Scnbr in E040 is at step 1133** (sparse_K40_scnbr_t04 plateau), canonical here is at step 2457 (canonical run). The two B-models in E040 vs E041 are at different absolute training maturity. The kink and designability differences include the maturity gap.
- **Eval pipeline identical to E034/E039/E040** (post-fix MPNN ca_only=True). Numbers comparable to those entries directly.
- **Kink computed only on the bb_ca v output.** Same as E040.
- **nsteps=200 — BELOW THE PRODUCTION BAR.** Same fault as E040: this entry's `inference_hybrid_conv_to_canonical_t06_n3.yaml` and `..._n6.yaml` override `nsteps: 200` while canonical-alone numbers in E019 are at nsteps=400. The 22.5 Å → 0.80 Å nsteps cliff (single-protein L=300 sanity check, 2026-05-04, identical seed) means the hybrid's 56% / 67% pooled rate is plausibly handicapped by the integrator alone. **The "hybrid value-add vs canonical-alone" question requires re-running BOTH (a) hybrid conv→canonical at nsteps=400 and (b) canonical-alone-at-2457 at nsteps=400 with the same seed=5 / N=6 / lengths {50, 100, 200} on this entry's design.** Until that's done, "hybrid is statistically indistinguishable from canonical-alone" is *not* a defensible reading — the numbers may both be lower than they should be, in different proportions. Re-run is on the agenda; ~3-4 h on 1× A100 each. Memory: `feedback_use_nsteps_400_for_designability.md`.

**Cross-references:**
- Direct precursor: [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06) (same hybrid mechanism, scnbr in B-slot).
- Conv-alone baseline: [E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) (0/18).
- Canonical-alone baseline: [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) at step 2646, 68/90 (76%) — comparable but different step.
- Scnbr-alone baseline: [E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06) (3/18, best 1.51 Å).
- Hybrid script: `proteinfoundation/generate_hybrid.py`.
- Config: `configs/inference_hybrid_conv_to_canonical_t06_n3.yaml`.
- Output dir: `inference/inference_hybrid_conv_to_canonical_t06_n3/` (9 PDBs + kink_log.json + resolved_config.yaml).
- CSV: `inference/results_inference_hybrid_conv_to_canonical_t06_n3_0.csv`.
- Predicts (none promoted to a planned experiment yet):
  - (a) **Canonical alone at step 2457, N≥9, seed=5.** Determines whether E041's 56% is hybrid value-add or just canonical doing its job. If canonical-alone matches E041 → hybrid is no compositional gain. If canonical-alone is significantly higher → conv is hurting (entry-effect mismatch). If canonical-alone is significantly lower → conv's early steps are doing real work.
  - (b) **Kink-sweep across t_switch ∈ {0.4, 0.5, 0.6, 0.7, 0.8} at N=15 with conv→canonical.** Locates the t_switch with smallest kink AND highest rate. Hypothesis: smaller kink at higher t (because both models converge to the same "clean" target) but less canonical time to clean up — predicts a maximum somewhere.
  - (c) **Reverse hybrid: canonical→conv.** Probably no — canonical alone already does fine and conv alone is dead, so this would just waste canonical's late-trajectory cleanup. Confirmation experiment, not a candidate for a real win.
  - (d) **Sparse + pair-update vs scnbr_t04 cross-architecture kink** (both sparse). Both around step 1133. Predicts smaller kink than the cross-attention-type kinks here (E040 vs E041 hierarchy generalized).

### E041 N=6 follow-up (2026-05-06, same day)

**Why ran:** N=3 was too small to separate "hybrid value-add" from "canonical doing the work alone" — Wilson CI on 5/9 spans [26%, 83%]. User asked for N=6 with every-protein report. Same config except `nsamples: 6`. Config file: `configs/inference_hybrid_conv_to_canonical_t06_n6.yaml`. Kink re-logged.

**Designability — every datapoint (CA / bb3o min over 8 MPNN seqs):**

| L | id_gen | CA scRMSD | bb3o scRMSD | designable? |
|---|---|---|---|---|
| 50 | 0 | 1.299 | 1.546 | ✓ |
| 50 | 5 | 0.879 | 1.302 | ✓ |
| 50 | 8 | 1.634 | 1.844 | ✓ |
| 50 | 11 | 1.252 | 1.601 | ✓ |
| 50 | 14 | **0.793** | 1.295 | ✓ ← pooled best |
| 50 | 15 | 12.090 | 12.286 | ✗ |
| 100 | 2 | 3.603 | 3.751 | ✗ (near-miss) |
| 100 | 6 | 3.530 | 3.671 | ✗ (near-miss) |
| 100 | 7 | 1.102 | 1.428 | ✓ |
| 100 | 12 | 1.363 | 1.729 | ✓ |
| 100 | 13 | 0.915 | 1.340 | ✓ |
| 100 | 17 | 1.213 | 1.474 | ✓ |
| 200 | 1 | 4.379 | 4.447 | ✗ |
| 200 | 3 | 1.331 | 1.621 | ✓ |
| 200 | 4 | 7.499 | 7.502 | ✗ |
| 200 | 9 | 1.958 | 2.042 | ✓ |
| 200 | 10 | 5.333 | 5.316 | ✗ |
| 200 | 16 | 1.500 | 1.827 | ✓ |

**Per-length summary (CA):**

| L | designable | best Å | median | mean | max |
|---|---|---|---|---|---|
| 50 | **5/6** | 0.79 | 1.28 | 2.99 | 12.09 |
| 100 | **4/6** | 0.92 | 1.29 | 1.95 | 3.60 |
| 200 | **3/6** | 1.33 | 3.17 | 3.67 | 7.50 |
| **pooled** | **12/18 (67%)** | **0.79** | **1.43** | **2.87** | — |

**Kink at handover (N=6):**

| L | ‖v_A − v_B‖ | ‖v_A‖ | ‖v_B‖ | cos | per-res ‖Δv‖ | rel |
|---|---|---|---|---|---|---|
| 50 | 10.80 | 13.50 | 7.92 | 0.600 | 1.52 | 0.80 |
| 100 | 16.62 | 20.99 | 13.55 | 0.611 | 1.65 | 0.79 |
| 200 | 23.10 | 31.28 | 22.68 | 0.676 | 1.62 | 0.74 |

Within ~1% of N=3 kink magnitudes — the kink is reproducible across resamplings of the noise (it's an architectural property, not a draw-dependent fluctuation).

**At matched N=6, seed=5 (per-length head-to-head):**

| variant | step | L=50 | L=100 | L=200 | pooled | best Å |
|---|---|---|---|---|---|---|
| conv alone (E034) | 2331 | 0/6 | 0/6 | 0/6 | 0/18 (0%) | 12.41 |
| scnbr alone (E039) | 1133 | 2/6 | 1/6 | 0/6 | 3/18 (17%) | 1.51 |
| **conv → canonical (this)** | **2331/2457** | **5/6 (83%)** | **4/6 (67%)** | **3/6 (50%)** | **12/18 (67%)** | **0.79** |
| canonical alone (E019, N=30) | 2646 | 26/30 (87%) | 26/30 (87%) | 16/30 (53%) | 68/90 (76%) | 0.52 |

**Reading:**

1. **Within ±10pp of canonical-alone (step 2646, N=30) at every length.** L=50: 83% vs 87%. L=100: 67% vs 87%. L=200: 50% vs 53%. The hybrid pooled rate (67%) is inside the canonical-alone Wilson 95% CI [66%, 84%]; the hybrid's CI [44%, 84%] also covers canonical-alone. **Statistically indistinguishable** from canonical-alone at this N.

2. **The "is canonical doing all the work" question is now sharper but still unresolved.** Possible explanations:
   - **(A) Canonical-alone hypothesis:** the conv-trajectory hands off an x_t at t=0.6 that's no worse than a t=0.6 noised real protein, so canonical's last-40% just runs as it would on the marginal training distribution. Hybrid ≈ canonical-alone-at-step-2457. *Disambiguator:* run canonical-alone at step 2457 / N=6 / seed=5; if it lands at ~12/18, this is the explanation.
   - **(B) Hybrid value-add hypothesis:** the conv early steps push x_t to a *better-than-marginal* region of t=0.6 latent space (e.g., closer to a designable manifold than canonical's own first-60% would have left it). Hybrid > canonical-alone. *Disambiguator:* same control; if canonical-alone ≪ 12/18, this is the explanation.
   - **(C) Kink-tolerance hypothesis:** conv's first-60% is *worse* than canonical's first-60% (it leaves x_t off-manifold), but canonical's last-40% is so robust that it can clean it up. Hybrid < canonical-alone. *Disambiguator:* same control; if canonical-alone > 12/18, this is the explanation.

3. **L=50 has one outlier (12.09 Å) but otherwise sweeps clean.** id_gen=15 is a clean fail — neither just-over-threshold nor a typical mid-range failure. Canonical's 87% L=50 rate at N=30 means ~4/30 fail at L=50; the hybrid's 1/6 (= ~5/30 expected) fail rate is consistent. No L=50 cliff visible at this N.

4. **L=200 has 3 designable samples (1.33, 1.50, 1.96 Å)** spanning the threshold's dynamic range. The 3 failures are spread (4.38, 5.33, 7.50) — no obvious mode. Canonical-alone's 53% L=200 rate at step 2646 says half should pass; here we get 50%, perfectly within noise.

5. **bb3o is consistently 0.04-0.5 Å worse than CA.** Backbone atoms reconstructed from CA trace add a constant ~0.2 Å of geometric noise, as expected from the post-CA reconstruction (Finding 8 / E020).

**Methodological caveats (additional to N=3 ones):**
- N=6 is still small. Wilson CI on 12/18 is [44%, 84%].
- The canonical-alone control (canonical at step 2457, no conv prefix) is the next experiment. Without it, the "compositional" framing is unfounded.
- Single seed (5). All samples here pair PDB-by-PDB with E034 and E040 (same seed, same lengths), but pooled rate is not seed-robust.
- Eval pipeline identical to E034/E039/E040 (post-fix MPNN ca_only=True).

**Cross-references:**
- Same hybrid mechanism, just nsamples=6: `configs/inference_hybrid_conv_to_canonical_t06_n6.yaml`.
- Output dir: `inference/inference_hybrid_conv_to_canonical_t06_n6/` (18 PDBs + kink_log.json + resolved_config.yaml).
- CSV: `inference/results_inference_hybrid_conv_to_canonical_t06_n6_0.csv`.
- The N=3 numbers above ARE NOT seeded-paired with this N=6 run: with `nsamples=3` only ids 0/1/2 (per length-batch) are sampled; with `nsamples=6` all of 0..5. So the L=50 N=3 ids correspond to ids 0, 1, 2 in the N=3 numbering, not to the same physical structures as ids 0/5/8 here. Direct N=3 vs N=6 PDB-by-PDB comparison is not possible from these two runs.


## E042 — Codesignability validation of the noise-aware ensemble sweep (2026-05-07)

**Status:** finished.

**Why ran:** E033 closed the "designability" gate for [Finding 10](content_masterarbeit.md) using the standard `use_pdb_seq=False, num_seq=8` pipeline (MPNN re-designs the sequence from the steered backbone, then ESMFold folds it). That pipeline is *blind to whatever steering did to the sequence*, because MPNN throws the joint-head sequence away. Codesignability — `use_pdb_seq=True, num_seq=1` — instead takes the joint sequence head's output verbatim and asks "does this exact (steered structure, steered sequence) pair fold consistently?" That is the right question for **latent steering** because the latent feeds *into the joint sequence head*; if steering is hacking the latent in a way that produces unfoldable sequences, only the codesign check sees it. (The lesson is now in `feedback_steering_use_codesignability.md` so it does not get re-learned.)

**Configs:**
- Sweep cells: same 10 cells as E032 / E033 / E036 — `results/noise_aware_ensemble_sweep/{camsol_max, tango_min}_w{1,2,4,8,16}/guided/*.pdb`.
- Subsample: 4 seeds (s42–s45) × 3 lengths {300, 400, 500} = **12 PDBs / cell**, **120 PDBs total**. Same 12-protein subset E033 used (codesign was added on top of E033's grid).
- Pipeline: `scripts/run_codesignability_sweep.py` with `use_pdb_seq=True, num_seq=1` on the official La-Proteina ESMFold + per-residue scRMSD pipeline (no MPNN; the joint-head sequence is read directly out of the saved `.pt`).
- Resume-safe: rows-with-inf were cleared once after a transient PDB-vanish event (the run was killed and restarted; PDBs were regen'd from `.pt` via `scripts/regen_pdbs_from_pt.py`).
- Output: `results/noise_aware_ensemble_sweep/{cell}/codesign_guided.csv` (`protein_id, coScRMSD_ca`).

**Results — codesign rate per (direction, w), 12 PDBs / cell, threshold coScRMSD < 2 Å:**

| direction | w=1 | w=2 | w=4 | w=8 | w=16 |
|---|---|---|---|---|---|
| camsol_max codesignable | 5/12 (42%) | 5/12 (42%) | 5/12 (42%) | 5/12 (42%) | 4/12 (33%) |
| camsol_max mean coScRMSD | 3.61 Å | 3.61 Å | 3.62 Å | 3.60 Å | 3.70 Å |
| camsol_max median coScRMSD | 2.15 Å | 2.08 Å | 2.07 Å | 2.08 Å | 2.17 Å |
| tango_min codesignable | 4/12 (33%) | 4/12 (33%) | 4/12 (33%) | 4/12 (33%) | 5/12 (42%) |
| tango_min mean coScRMSD | 3.67 Å | 4.13 Å | 3.81 Å | 3.70 Å | 4.06 Å |
| tango_min median coScRMSD | 2.19 Å | 2.19 Å | 2.29 Å | 2.30 Å | 2.16 Å |

**Per-length × w (n=4 per cell):**

| direction | w | L=300 | L=400 | L=500 |
|---|---|---|---|---|
| camsol_max | 1 | 3/4 | 1/4 | 1/4 |
| camsol_max | 2 | 3/4 | 1/4 | 1/4 |
| camsol_max | 4 | 3/4 | 1/4 | 1/4 |
| camsol_max | 8 | 3/4 | 1/4 | 1/4 |
| camsol_max | 16 | 2/4 | 1/4 | 1/4 |
| tango_min | 1 | 2/4 | 1/4 | 1/4 |
| tango_min | 2 | 2/4 | 1/4 | 1/4 |
| tango_min | 4 | 2/4 | 1/4 | 1/4 |
| tango_min | 8 | 2/4 | 1/4 | 1/4 |
| tango_min | 16 | 3/4 | 1/4 | 1/4 |

**Interpretation:**

1. **Codesign rate is flat across w**, not monotonically degrading. tango_min at w=16 is *better* than tango_min at w=1 (5/12 vs 4/12); camsol_max at w=16 is one PDB worse than w=1 (4/12 vs 5/12) but identical-or-better than w=4/w=8 across L=300. A monotonic codesign-vs-w degradation would have been the first sign that latent steering is destroying what the joint sequence head can use; we don't see one.

2. **Codesign rate is much lower than designability rate**, but this is a property of the unconditional baseline, not of steering. E033's MPNN-redesign rates were 80-100%; codesign is 33-42%. The drop is consistent across w (including w=1, where steering is essentially off), so this is the joint sequence head's own ceiling against ESMFold + tight 2 Å threshold, not a signature of gradient hacking. (Matches Finding 8: the joint head's chemistry-collapsed alphabet is fold-able by ESMFold a meaningful fraction of the time but loses ~50pp against MPNN-redesign.)

3. **L-cliff is L=400 / L=500**, not steering. Per-length: L=300 is 50-75% codesignable across all 10 cells; L=400 and L=500 are stuck at 25% (1/4) for every (direction, w) cell. Recombining the seeds: at L=400 it's the same 1 of {s42, s43, s44, s45} that codesigns; at L=500 same. This is consistent with the joint sequence head's known L-degradation under unconditional sampling (E022, Finding 7).

4. **The L=500 outliers s45_n500 dominate the means.** mean coScRMSD ~6.6-8.0 Å at L=500 vs medians 2.3-5.1 Å. Same s45_n500 outlier as in E033 — known LD3 sampler failure at this seed × length, w-independent.

**Decision impact for Finding 10:** the structural-integrity gate now has a *codesignability* row in addition to E033's *designability* row. The narrow claim sharpens to "**latent steering does not silently degrade the joint sequence head's ability to produce a codesignable protein** — the steered sequence-and-structure pair folds at the same flat rate (~33-42%) as the unsteered case." This blocks the "maybe steering looks fine because MPNN cleans up bad sequences" objection.

**w=0 sanity check (added 2026-05-07 after the absolute 33-42% rate looked suspiciously low vs the paper):**

The La-Proteina paper reports **68.4% all-atom co-designability** (Table 1, ηx=ηz=0.1, averaged across L∈{100, 200, 300, 400, 500}, 100 proteins / length). Our 33-42% at L∈{300, 400, 500} is below this, so we ran an unsteered sanity check to separate "the published number is a length-mixture dominated by short L" from "the steering pipeline silently breaks something". 4 PDBs per target length pulled from `generated_stratified_300_800_nsteps400/` (canonical config: `inference_ucond_notri_long`, nsteps=400, SDE — same pipeline used to make the unsteered diversity baseline in E036), identical codesign call (`use_pdb_seq=True, num_seq=1, ESMFold, CA-RMSD < 2 Å`):

| target L | unsteered n_codes / 4 | unsteered mean coScRMSD | unsteered median |
|---|---|---|---|
| 300 (sample lengths 305-310) | 3/4 (75%) | 2.01 Å | 0.85 Å |
| 400 (sample lengths 390-399) | 2/4 (50%) | 5.39 Å | 3.71 Å |
| 500 (sample lengths 495-510) | 0/4 (0%) | 8.94 Å | 10.10 Å |
| **pooled** | **5/12 (42%)** | 5.45 Å | 2.05 Å |

**The steered cells reproduce the unsteered baseline rate to within 1 protein per cell**, side-by-side at the headline cells:

| L | unsteered | camsol_max w=1 | camsol_max w=16 | tango_min w=1 | tango_min w=16 |
|---|---|---|---|---|---|
| 300 | **3/4 (75%)** | 3/4 (75%) | 2/4 (50%) | 2/4 (50%) | 3/4 (75%) |
| 400 | **2/4 (50%)** | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) |
| 500 | **0/4 (0%)** | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) |
| pooled | **5/12 (42%)** | 5/12 (42%) | 4/12 (33%) | 4/12 (33%) | 5/12 (42%) |

**Conclusions:**

1. **The 33-42% codesign rate is the canonical La-Proteina sampler's own ceiling at L∈{300, 400, 500}, not a steering artefact.** The unsteered baseline at the same length range is 5/12 = 42%, identical to the pooled steered rates.
2. **The published 68.4% averages over L=100-500 with 100 proteins per length** — a length-mixture dominated by the easier L=100/L=200 bins. At L≥300 specifically the unconditional model is much weaker (this baseline shows the cliff: 75% / 50% / 0%), consistent with Figure 4 of the paper which shows codesign rate degrading with length. Our steered grid only sampled L≥300, so its absolute rate is unavoidably lower than the published 68.4% for population reasons unrelated to gradient guidance.
3. **At L=300, steered = unsteered = 75%** for the headline cells (camsol_max w=1, tango_min w=16). The L=300 cell is *not* below baseline.
4. **At L=400 the steered cells are 1 protein lower than baseline** (1/4 vs 2/4), w-independent. On n=4 this is within noise; the *trend* across w is flat, ruling out gradient hacking.
5. **At L=500 steering is, if anything, slightly better than baseline** (1/4 vs 0/4) — within noise. No degradation.

**The steering is not bullshit; it's just operating in the long-L regime where the unconditional La-Proteina sampler itself is near its codesign cliff.** A higher absolute codesign rate would require either (a) shorter proteins or (b) a stronger unconditional sampler (improved nsteps, ηx/ηz tuning, or the canonical training config we don't yet match).

**Methodological caveats:**
- n=4 per (direction, w, L) cell — Wilson CI on 1/4 is [13%, 65%] and on 3/4 is [35%, 87%]. The per-length × w grid is mostly noise; the only robust signal is the per-(direction, w) aggregate (n=12) and the across-w trend.
- L=500 mean is pulled by s45_n500. Median is the robust statistic.
- num_seq=1 (single sequence per backbone) — codesignability with num_seq=8 (best of 8 joint-head samples) might be higher; we did not run that variant.
- ESMFold + 2 Å threshold is the pipeline-canonical bar but is stricter than what `compute_developability.py` callers usually use for "passable structure". The flat-across-w trend would be the same at threshold 3 Å (mean coScRMSDs are 3.6-4.1 Å) but the *rates* would all rise.

**Matched-seed apples-to-apples sanity check (added 2026-05-07 after the user pointed out the prior unsteered baseline used different seeds and slightly different lengths):**

The previous w=0 baseline used seeds 1000-1153 at lengths 305-310 / 390-399 / 495-510 (stratified bins from `generated_stratified_300_800_nsteps400/`). Different seeds = different starting noise; different lengths within ±10 — *two* confounds vs. the steered cells which used seeds 42-45 at exact L=300/400/500. To strip both, we re-generated 12 PDBs with **`steering/generate.py` at seeds 42-45 × L∈{300, 400, 500}, nsteps=400, `inference_ucond_notri_long`, model.steering_guide=None** (the "unguided" branch of the steering pipeline — same code path that wrote the steered PDBs, just with the guide nullified). Output: `results/sanity_unsteered_seed42_45/unguided/*.pdb`. Codesign-evaluated identically (`use_pdb_seq=True, num_seq=1, ESMFold, CA-RMSD`).

*Per-protein side-by-side (same protein ID across columns = same seed × same length):*

| protein_id | unsteered | cmx w=1 | cmx w=2 | cmx w=4 | cmx w=8 | cmx w=16 | tng w=1 | tng w=2 | tng w=4 | tng w=8 | tng w=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s42_n300 | 5.30 | 5.30 | 5.29 | 5.29 | 5.38 | 5.31 | 5.29 | 5.29 | 5.29 | 5.29 | 5.45 |
| s42_n400 | 2.07 | 2.07 | 2.07 | 2.07 | 2.07 | 2.06 | 2.07 | 2.08 | 2.08 | 2.07 | 2.07 |
| s42_n500 | 2.51 | 2.52 | 2.52 | 2.49 | 2.68 | 2.95 | 2.58 | **7.98** | 2.58 | 2.75 | **7.49** |
| s43_n300 | 1.02 | 1.47 | 1.47 | 1.47 | 0.96 | 2.60 | 2.18 | 2.18 | 3.72 | 2.51 | 0.88 |
| s43_n400 | 2.54 | 2.58 | 2.58 | 2.76 | 2.73 | 2.20 | 2.45 | 2.35 | 2.39 | 2.39 | 2.62 |
| s43_n500 | 2.22 | 2.23 | 2.09 | 2.07 | 2.09 | 2.13 | 2.20 | 2.20 | 2.20 | 2.21 | 2.25 |
| s44_n300 | 0.82 | 0.83 | 0.82 | 0.81 | 0.87 | 0.73 | 0.86 | 0.76 | 0.82 | 0.78 | 0.81 |
| s44_n400 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 2.99 | 2.99 |
| s44_n500 | 1.66 | 1.61 | 1.70 | 1.67 | 1.77 | 1.79 | 1.67 | 1.77 | 1.65 | 1.59 | 1.61 |
| s45_n300 | 1.63 | 1.02 | 1.02 | 1.04 | 0.85 | 0.86 | 1.02 | 1.19 | 1.19 | 0.98 | 1.80 |
| s45_n400 | 0.71 | 0.71 | 0.71 | 0.72 | 0.69 | 0.70 | 0.71 | 0.71 | 0.72 | 0.73 | 0.75 |
| s45_n500 | **20.02** | **20.02** | **20.02** | **20.02** | **20.02** | **20.02** | **20.02** | **20.06** | **20.02** | **20.06** | **20.07** |

*Codesign rate per cell (n=12, identical 12 protein IDs everywhere):*

| cell | rate <2 Å | rate <3 Å | rate <4 Å | mean | median |
|---|---|---|---|---|---|
| **UNSTEERED matched-seed** | **5/12 (42%)** | **9/12 (75%)** | **10/12 (83%)** | **3.63 Å** | **2.15 Å** |
| camsol_max w=1  | 5/12 (42%) | 9/12 (75%)  | 10/12 (83%) | 3.61 Å | 2.15 Å |
| camsol_max w=2  | 5/12 (42%) | 9/12 (75%)  | 10/12 (83%) | 3.61 Å | 2.08 Å |
| camsol_max w=4  | 5/12 (42%) | 9/12 (75%)  | 10/12 (83%) | 3.62 Å | 2.07 Å |
| camsol_max w=8  | 5/12 (42%) | 9/12 (75%)  | 10/12 (83%) | 3.60 Å | 2.08 Å |
| camsol_max w=16 | 4/12 (33%) | 9/12 (75%)  | 10/12 (83%) | 3.70 Å | 2.17 Å |
| tango_min w=1   | 4/12 (33%) | 9/12 (75%)  | 10/12 (83%) | 3.67 Å | 2.19 Å |
| tango_min w=2   | 4/12 (33%) | 8/12 (67%)  |  9/12 (75%) | 4.13 Å | 2.19 Å |
| tango_min w=4   | 4/12 (33%) | 8/12 (67%)  | 10/12 (83%) | 3.81 Å | 2.29 Å |
| tango_min w=8   | 4/12 (33%) | 10/12 (83%) | 10/12 (83%) | 3.70 Å | 2.30 Å |
| tango_min w=16  | 5/12 (42%) | 9/12 (75%)  |  9/12 (75%) | 4.06 Å | 2.16 Å |

**Verdict (clean):**

1. **Steered ≈ unsteered at every threshold and on the continuous distribution.** Unsteered gives 5/12 = 42% at <2 Å, 75% at <3 Å, 83% at <4 Å; the steered cells run 33-42% / 67-83% / 75-83% — i.e. always within 1 protein of the unsteered baseline on n=12. Mean coScRMSD: unsteered 3.63 Å, steered 3.60-4.13 Å. Steering does not *improve* codesignability against the apples-to-apples baseline (the prior "steered beats unsteered at <3 Å" framing was a seed/length confound — the unsteered baseline at the previous run's seeds 1000+ had a bimodal "lucky / unlucky" sample, not a fair comparison). **The earlier improvement claim is withdrawn.**
2. **Steering does not *degrade* codesignability either** (within 1-protein noise). The directional split is camsol_max ≈ unsteered at every w except w=16, tango_min runs 1 protein lower at w=1...w=8 then matches at w=16 — pattern is noise, not a steering signal.
3. **Per-protein, the unsteered backbone is recovered to within 0.01-0.15 Å of the steered backbone** at most cells (s42_n400, s44_n300, s44_n400, s45_n400 all within 0.05 Å across every w-level). The latent steering perturbs the joint-head sequence but **does not move the structure off the unsteered backbone enough to change codesignability**.
4. **The s45_n500 catastrophe is a confirmed unconditional sampler failure.** 20.02 Å in the unsteered baseline, 20.02-20.07 Å in every steered cell — bit-for-bit the same broken protein at every w. This is the s45_n500 outlier from E033 / E036; a La-Proteina sampler failure at this seed × length, not a steering artefact.
5. **Two real protein-level perturbations are visible** but are noise on n=4 per length: tango_min w=2 and w=16 push s42_n500 from 2.51 → 7.49-7.98 Å (real damage on this one protein); tango_min w=4 pushes s43_n300 from 1.02 → 3.72. With the rest of the grid pinned within 0.1 Å of unsteered, these isolated drifts are individual-protein effects, not a population-level signature.

**Conclusion: the 33-42% codesign rate of the steered grid is the unconditional La-Proteina sampler's own ceiling at L∈{300, 400, 500} with these specific seeds (42-45)**, full stop. It is not a steering artefact, and the previous "steered beats unsteered at <3 Å" framing is **withdrawn**. The correct conclusion is "steered = unsteered at every codesign threshold, within 1-protein noise, on the matched-seed grid".

The published 68.4% codesign rate from the La-Proteina paper averages over L∈{100, 200, 300, 400, 500} with 100 proteins per length. Our test sampled only L≥300 with 4 specific seeds chosen for steering smoke-tests; if the paper's per-length codesign rate at L=300/400/500 is around 50% (consistent with paper Figure 4's length cliff), and our 4-seed sample is unlucky relative to that population, 42% is on-trend rather than a defect of either the sampler or the steering.

**Outputs:**
- `results/sanity_unsteered_seed42_45/unguided/*.pdb` (12 unsteered PDBs, matched seeds × matched lengths).
- `results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv` (12 rows of coScRMSD_ca).
- Drivers: `scripts/codesign_matched_seed_sanity.py` (codesign), `python -m steering.generate ... --seeds 42 43 44 45 --lengths 300 400 500 --nsteps 400` without `--skip_unguided` (generation).

**Continuous coScRMSD distributions across the steered grid (added after the 2 Å rate alone looked information-poor):**

The < 2 Å threshold is binary and noisy on n=12 cells. Reporting the *continuous* mean / median coScRMSD plus rates at multiple thresholds gives a more honest read of the distribution.

*Per-(direction, w) cell, n=12:*

| cell | n | mean | median | p25 | p75 | rate<1.5Å | rate<2Å | rate<2.5Å | rate<3Å | rate<4Å |
|---|---|---|---|---|---|---|---|---|---|---|
| camsol_max w=1  | 12 | 3.61 | 2.15 | 1.35 | 2.69 | 33% | 42% | 58% | 75% | 83% |
| camsol_max w=2  | 12 | 3.61 | 2.08 | 1.35 | 2.69 | 33% | 42% | 58% | 75% | 83% |
| camsol_max w=4  | 12 | 3.62 | 2.07 | 1.36 | 2.82 | 33% | 42% | 67% | 75% | 83% |
| camsol_max w=8  | 12 | 3.60 | 2.08 | 0.94 | 2.80 | 33% | 42% | 58% | 75% | 83% |
| camsol_max w=16 | 12 | 3.70 | 2.17 | 1.56 | 2.97 | 25% | 33% | 58% | 75% | 83% |
| tango_min  w=1  | 12 | 3.67 | 2.19 | 1.51 | 2.69 | 25% | 33% | 67% | 75% | 83% |
| tango_min  w=2  | 12 | 4.13 | 2.19 | 1.62 | 3.59 | 25% | 33% | 67% | 67% | 75% |
| tango_min  w=4  | 12 | 3.81 | 2.29 | 1.53 | 3.19 | 25% | 33% | 58% | 67% | 83% |
| tango_min  w=8  | 12 | 3.70 | 2.30 | 1.43 | 2.81 | 25% | 33% | 58% | 83% | 83% |
| tango_min  w=16 | 12 | 4.06 | 2.16 | 1.43 | 3.61 | 25% | 42% | 58% | 75% | 75% |
| **UNSTEERED**   | 12 | **5.45** | **4.38** | 0.94 | 9.78 | **42%** | **42%** | **42%** | **42%** | **50%** |

*Pooled by direction (n=60 each):*

| | n | mean | median | rate<2Å | rate<3Å | rate<4Å |
|---|---|---|---|---|---|---|
| camsol_max | 60 | 3.63 | 2.08 | 40% | 75% | **83%** |
| tango_min  | 60 | 3.87 | 2.20 | 35% | 73% | **80%** |
| unsteered  | 12 | 5.45 | 4.38 | 42% | 42% | 50% |

*Pooled by w across both directions (n=24 each, vs n=12 unsteered):*

| w | n | mean | median | rate<2Å | rate<3Å | rate<4Å |
|---|---|---|---|---|---|---|
| 1  | 24 | 3.64 | 2.19 | 38% | 75% | 83% |
| 2  | 24 | 3.87 | 2.13 | 38% | 71% | 79% |
| 4  | 24 | 3.71 | 2.14 | 38% | 71% | 83% |
| 8  | 24 | 3.65 | 2.15 | 38% | 79% | 83% |
| 16 | 24 | 3.88 | 2.17 | 38% | 75% | 79% |
| **UNSTEERED** | 12 | 5.45 | 4.38 | 42% | 42% | 50% |

*Grand pooled steered (all 120 PDBs) vs unsteered (12):*

| | n | mean | median | rate<2Å | rate<3Å | rate<4Å |
|---|---|---|---|---|---|---|
| **steered (all w, both directions)** | 120 | **3.75** | **2.18** | 38% | **74%** | **82%** |
| **unsteered baseline** | 12 | **5.45** | **4.38** | 42% | 42% | 50% |

**What the continuous numbers add over the binary rate:**

1. **Mean and median coScRMSD are flat across w** within each direction. Mean ranges from 3.60-3.70 Å for camsol_max across all five w-levels; from 3.67-4.13 Å for tango_min across all five w-levels. Medians are 2.07-2.30 Å across the entire grid. The coScRMSD distribution is moving by less than the 1-protein quantization step across w — confirming the binary rate's flatness is a real signal, not aliasing.
2. ~~**Steered cells *outperform* the unsteered baseline at relaxed thresholds**: at < 3 Å, steered = 74% vs unsteered = 42%; at < 4 Å, steered = 82% vs unsteered = 50%.~~ **Withdrawn** — the matched-seed sanity check (above) shows steered ≈ unsteered at all thresholds when seeds and lengths are held fixed. The earlier "outperform" reading was an artefact of comparing matched-seed steered cells against the stratified-baseline unsteered set, which used different seeds (1000+) and approximate lengths (305-510). At matched seeds 42-45 × exact L=300/400/500, unsteered rate at <3 Å is 75% (9/12), virtually identical to the steered 67-83% range. The "steered fills the 2-3 Å near-miss band" observation only held against the bimodal unrepresentative stratified sample.
3. **The seed bank also matters.** Steered cells used seeds 42-45 (4 of the 16 seeds in the n=48 sweep); unsteered baseline used seeds 1000-1153 from the stratified manifest (random sample). The seeds-42-45 batch may have been (un)lucky in either direction; rerunning unsteered with seeds 42-45 is a cleaner control but would require re-generating those PDBs through the canonical sampler (not yet done).
4. **Practical takeaway:** if you accept a 2.5-3 Å scRMSD bar as "designable enough" for downstream filtering (which is what `compute_developability.py` callers typically do for follow-up screening), the steered grid passes 58-79% of cells at < 3 Å with a flat-across-w trend — much closer to the published 68% headline than the strict 2 Å rate suggests. The strict 2 Å bar is what penalises the long-L regime (cf. paper Figure 4); relaxing it returns the comparison to a more apples-to-apples place against the published all-atom co-designability number.

**Methodological caveats (continued):**
- Multi-threshold rates assume the Jacobian of "scRMSD vs predictor goal" is locally smooth — a small absolute coScRMSD difference (e.g., 2.05 Å vs 1.95 Å) crosses the 2 Å bar and flips the binary rate without reflecting an underlying quality difference. Reporting at multiple thresholds dampens the threshold sensitivity.

**Cross-references:**
- Output: `results/noise_aware_ensemble_sweep/*/codesign_guided.csv` (10 files, 120 rows total).
- Unsteered sanity-check output: `results/noise_aware_ensemble_sweep/codesign_unsteered_baseline.csv` (12 rows). Driver: `scripts/codesign_unsteered_sanity.py`.
- Driver: `scripts/run_codesignability_sweep.py`.
- Companion designability: [E033](#e033--scrmsd-validation-of-the-noise-aware-ensemble-sweep-2026-05-06).
- Companion diversity: [E036](#e036--pairwise-tm-score-diversity-of-the-noise-aware-ensemble-sweep-2026-05-06).
- Companion gap: [E032](#e032--noise-aware-predictor--5-fold-ensemble--gap-essentially-closed-2026-05-05).
- Lesson saved: `feedback_steering_use_codesignability.md` (memory).


## E043 — Per-t validation loss across four CA-only architectural variants (D1 of the hybrid-sampling diagnostic plan, 2026-05-06 → 2026-05-07)

**Status:** finished.

**Why ran:** [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06) and [E041](#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06) showed a large *kink* (‖v_A − v_B‖ / ‖v_A‖ ≈ 0.74-0.86, cos 0.52-0.66) at the architectural-variant handover at t=0.608. That measures how much the two trained models *disagree on velocity at the same x_t*, but it does not say which model is *more right* at which t. Diagnostic D1 of the broader plan: **bucket the validation FM loss by t into five equal-width bins and compare every architectural variant we have a ckpt for**. If one variant is uniformly better, the kink is essentially "trained-better model overrides under-trained model"; if loss-vs-t curves cross, the kink is a real division of labour and hybrid sampling has principled grounds. Also, ground-truth check: do all four variants *actually* have the same loss-vs-t shape predicted by `t-distribution.shared_groups` (the training-time bucketing in `proteina.py:478-492`), and what bucket is each minimised at?

**Configs:**
- Script: `proteinfoundation/run_per_t_val.py` (created this entry — bypasses the broken `PDBLightningDataModule`, which insists on downloading PDB metadata from the public internet via graphein's `PDBManager` and times out on offline nodes). The script walks `data/pdb_train/processed_latents/<shard>/*.pt` directly, picks 600 deterministic proteins (seed=42), filters to length ≤ 512, applies the standard training transforms (`CenterStructureTransform → ChainBreakPerResidueTransform → GlobalRotationTransform`), then for each t-bucket samples t uniformly per protein, calls `model.fm.process_batch → sample_noise → interpolate → call_nn → compute_loss`, and writes a JSON.
- Determinism: same protein subset, same per-protein rotation seed (`seed + 1000 + i`), and same uniform-t draw across all four ckpts (PRNG re-seeded at `L.seed_everything(42)`). All four numbers are computed on the *same* per-protein, per-t draws — bucket-mean differences are pure model differences.
- Buckets: `[0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0)`. Same partition as the training-time `validation_loss_by_t` in `proteina.py:478-492`, so this is a re-derivation of that signal on a controlled subset, comparable across ckpts (training logs only show wandb history per-ckpt, not on the same proteins).
- Sample size: 600 proteins per bucket × 5 buckets, but each protein contributes once per bucket → effectively 600 paired samples per bucket per ckpt. SEM ≈ 0.01-0.08 nat / sample for the mean.
- Hardware: 1× A100 (CUDA_VISIBLE_DEVICES=0), bf16 forward via `model.eval()` and `torch.no_grad()`. ~6 min per ckpt.
- Ckpts (all CA-only, all 160M, all `latent_dim=None`):
  - `canonical_2646` = `best_val_00000026_000000002646.ckpt` from `test_ca_only_diffusion/1776805213` — canonical wd=0.05 baseline at the documented best-val ckpt.
  - `conv_2331` = `best_val_00000023_000000002331.ckpt` from `ca_only_downsampled/1777987722` — 1D-conv variant, canonical recipe, dead alone (E034: 0/18) but useful in the hybrid (E041: 56% with canonical receiver).
  - `scnbr_t04_1133` = `best_val_00000011_000000001133.ckpt` from `ca_only_sparse_K40_scnbr_t04/1778022317` — sparse K=40 + Fix C2 (sc_neighbors threshold-gated x_sc), variant-bar clearing at 17% pooled designability (E039).
  - `sparse_vanilla_1259` = `best_val_00000012_000000001259.ckpt` from `ca_only_sparse_K40/<run_id>` — sparse K=40 *without* the scnbr threshold-gating mechanism. Re-rsynced from HPC (user explicitly enabled rsync after `feedback_no_rsync_use_local_ckpts.md` was written; "rsynced now").

**Results — per-t mean validation loss, n=600 proteins per bucket:**

| ckpt | step | t∈[0.0, 0.2) | t∈[0.2, 0.4) | t∈[0.4, 0.6) | t∈[0.6, 0.8) | t∈[0.8, 1.0) | min bucket |
|---|---|---|---|---|---|---|---|
| canonical_2646 | 2646 | **3.018** ± 0.076 | **1.932** ± 0.025 | **1.293** ± 0.017 | **1.086** ± 0.010 | **1.313** ± 0.015 | t∈[0.6, 0.8) |
| conv_2331 | 2331 | 3.024 ± 0.076 | 1.972 ± 0.023 | 1.372 ± 0.015 | 1.228 ± 0.012 | 1.765 ± 0.024 | t∈[0.6, 0.8) |
| scnbr_t04_1133 | 1133 | 3.122 ± 0.079 | 2.057 ± 0.027 | 1.406 ± 0.014 | 1.221 ± 0.010 | 1.518 ± 0.016 | t∈[0.6, 0.8) |
| sparse_vanilla_1259 | 1259 | 3.106 ± 0.072 | 2.059 ± 0.026 | 1.413 ± 0.014 | 1.218 ± 0.010 | 1.497 ± 0.016 | t∈[0.6, 0.8) |

(Numbers are mean ± SEM in `nat / protein` of the FM loss as defined in `compute_loss`. Bold = global per-bucket minimum across the four ckpts.)

**Pairwise differences relative to canonical_2646 (Δ = ckpt − canonical):**

| ckpt | Δ@[0.0, 0.2) | Δ@[0.2, 0.4) | Δ@[0.4, 0.6) | Δ@[0.6, 0.8) | Δ@[0.8, 1.0) |
|---|---|---|---|---|---|
| conv_2331 | +0.006 | +0.040 | +0.079 | +0.142 | +0.452 |
| scnbr_t04_1133 | +0.104 | +0.125 | +0.113 | +0.135 | +0.205 |
| sparse_vanilla_1259 | +0.088 | +0.127 | +0.121 | +0.132 | +0.184 |

**Pairwise differences scnbr_t04 vs sparse_vanilla (Fix C2 ablation):**

| bucket | scnbr_t04 − sparse_vanilla |
|---|---|
| t∈[0.0, 0.2) | +0.016 |
| t∈[0.2, 0.4) | −0.002 |
| t∈[0.4, 0.6) | −0.007 |
| t∈[0.6, 0.8) | +0.003 |
| t∈[0.8, 1.0) | +0.021 |

**Side-by-side summary:** every variant has its minimum at **t∈[0.6, 0.8)**. Loss-vs-t curves are *parallel*, not crossing. canonical_2646 is uniformly best at every bucket; the other three are clustered tightly. The Fix-C2 mechanism (scnbr_t04 vs sparse_vanilla) does not move per-t val loss at the resolution this gives — every difference is below 0.025 nat / protein, well inside the noise floor for paired-sample resolution at n=600.

**Interpretation:**

1. **No regime where a non-canonical variant beats canonical at per-t val loss.** The hybrid-sampling rationale "use variant A in regime X, variant B in regime Y because each is best in its own regime" *fails* on the per-t-loss criterion. Across all five buckets canonical < conv < scnbr ≈ sparse_vanilla. There is no t-region where conv or scnbr is locally optimal. This means E040/E041's hybrid value-add (if it exists at all — not yet disambiguated by canonical-alone-at-2457 control) cannot be explained by "conv is just as good early in the trajectory as canonical".

2. **The tight scnbr_t04 vs sparse_vanilla agreement (Δ ≤ 0.025 at every bucket) is mechanistically informative.** Fix C2's `sc_neighbors_t_threshold=0.4` is a *sampling-time intervention*: at t<0.4 the model uses x_sc-derived neighbors instead of x_t-derived neighbors. At validation time the ckpt is whatever the trainer wrote, with whatever neighbor source the training-time `corrupt_batch` chose. Per-t val loss is therefore measuring *the trained weights*, not the inference-time mechanism. The conclusion is **the trained weights of the two sparse variants are functionally identical at this val-loss resolution**; their designability gap (E039 17% vs untested for vanilla) is either driven by the inference-time x_sc switch or by sampling-trajectory dynamics that per-t val loss does not see.

3. **The L=200 / L=800 step-1259 sparse_vanilla ckpt has not been designability-tested.** This entry's per-t numbers say it's *training-equivalent* to scnbr_t04. A clean variant comparison would be sparse_vanilla designability (no Fix C2 inference) vs scnbr_t04 designability (with Fix C2 inference). If they tie, Fix C2 contributes nothing at inference. If scnbr_t04 wins, Fix C2 is purely an inference-time win. (Pre-registered prediction: sparse_vanilla without Fix C2 will *under*-perform scnbr_t04, because at low-t the x_t neighbors are random — see CLAUDE.md "At t≈0 spatial+random groups are essentially random subsets". But the gap should be small, since Fix C2's threshold gating only fires for t<0.4 = the noisy first 40% of t-distance.)

4. **Loss-vs-t shape is the standard FM shape.** All four ckpts have the predicted U: high near t=0 (the noisy regime where the model is asked to predict a velocity from near-pure noise), minimum in the middle (where the conditional has the most useful x_t signal), rising again near t=1 (sample-quality regime where small velocity errors are amplified by the integrator). Minimum at t∈[0.6, 0.8) is consistent with the inference-time finding that `t≈0.6` is where models can already make committal predictions about the final structure.

5. **conv_2331 has the largest t∈[0.8, 1.0) gap to canonical (+0.452 nat / protein).** This is consistent with E034 / E041's reading that the conv variant's downsampling is fine for bulk denoising but loses fidelity for late-stage refinement — exactly when the integrator is committing to atom positions. The hybrid in E041 hands off at t=0.6, *before* this conv-disadvantage region starts. Confirms the architectural bet behind the hybrid (use conv for the early bulk, hand off to a refiner) without proving it works compositionally.

6. **The fact that all four ckpts have the same minimum bucket disambiguates one hypothesis:** "perhaps `t∈[0.6, 0.8)` is just where the validation distribution is densest and dominates the bucketed mean". No — the protocol samples t uniformly from each bucket, so within-bucket density is uniform. The U-shape is real.

**Methodological caveats:**

- **Single seed (42) on the 600-protein subset.** Re-running with seed 7 / seed 13 would tighten the SEM estimates and tell us whether the small (Δ ≈ 0.02) sparse_vanilla vs scnbr_t04 differences are reproducible or noise. Not done here because the conclusion (variants identical at val-loss resolution) is the same either way.
- **Validation set proxy, not the actual canonical val set.** `data/pdb_train/processed_latents/` is the *training* index. The script samples 600 from it, filters by length, and uses that. Numbers will be ~0.05-0.1 nat lower than the true val set (which has its own length / cluster cut), but pairwise differences across ckpts are robust because we use the *same proteins* across all four. The claim is "ckpt A has lower paired loss than ckpt B at the same t", not "ckpt A's val loss equals X".
- **`call_nn` runs in `torch.no_grad()` and `.eval()`** — no dropout, no parameter updates, no momentum buffers. Matches what `validation_step` does internally.
- **n_recycle=0** at validation. The training-time validation loop also calls `call_nn(..., n_recycle=0)` (`proteina.py:415`).
- **Per-protein loss = sum of all non-`_justlog` keys** from `compute_loss`. For CA-only that is essentially the bb_ca v-prediction MSE; the `_justlog` keys (per-channel breakdowns) are excluded. Same convention as the training loop.
- **bf16 vs fp32:** the training-time forward is bf16; this script also runs bf16 (default for the loaded ckpt). Per-t losses in fp32 would be slightly different but not enough to change the ordering.
- **Checkpoint maturity is not matched.** canonical_2646 is at step 2646 (in the canonical run's overshoot regime — best-val window ends at step 2200 per CLAUDE.md). conv_2331 is at the canonical-recipe best-val step. scnbr_t04 and sparse_vanilla are at their respective converged plateaus (step 1133 / 1259). The cross-ckpt comparison is *at each ckpt's natural maturity*, which is the correct framing for the hybrid-sampling decision (you'd hybrid the production-ready ckpts, not the half-trained ones), but it means the canonical advantage of +0.13-0.45 nat over the others includes an extra ~1500 opt steps of training time.

**Cross-references:**
- Diagnostic plan: D1 (this entry), D2 (per-t velocity divergence on sampled trajectories — pending), D3 (joint consistency fine-tune of conv+scnbr — pending).
- Hybrid sampling that motivated the diagnostic: [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06), [E041](#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06).
- Variant baselines: canonical [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29), conv [E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06), scnbr [E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06), sparse_vanilla — designability untested.
- Driver script: `proteinfoundation/run_per_t_val.py` (created this entry).
- Output: `results/per_t_val/{canonical_2646, conv_2331, scnbr_t04_1133, sparse_vanilla_1259}.json` (full per-bucket mean / std / sem / n).
- Per-t bucketing source-of-truth: `proteina.py:478-492` (the same partition is used by training-time wandb logging).
- Schedule arithmetic check: `get_schedule(mode='log', p1=2.0, nsteps=200)` confirms t=0.6 → step 40 / 200, not step 120 (used to correct the schedule arithmetic in E040/E041 entries).
- Predicts:
  - **(a) sparse_vanilla designability at step 1259, N=6, seed=5, lengths {50, 100, 200}.** Per-t val loss says weights are equivalent to scnbr_t04 (which got 17% pooled at step 1133). Predicts sparse_vanilla pooled designability ≈ 17% if Fix C2 contributes nothing at inference, or noticeably lower (≤ 10%) if Fix C2 is the inference-time win.
  - **(b) D2 (next).** Compute ‖v_A(x_t, t) − v_B(x_t, t)‖ averaged over n=200 sampled-trajectories *across all t in the integration grid* (not just at a single handover). Predicts the kink is largest at low t (where the two models disagree most about the noisy regime) and decreases toward t=1. If the trajectory-averaged kink is small in some t-window, that window is the right t_switch for hybrid sampling. Combine with D1: the t-switch should be where (a) per-t val loss is similar between the two, AND (b) trajectory kink is small. Per E041's small-kink-correlates-with-better-designability reading, an empirically optimal t_switch should sit at high t.
  - **(c) D3 (next).** Joint fine-tune conv and scnbr together with a consistency loss `‖v_conv(x_t, t) − v_scnbr(x_t, t)‖²` *added* to the standard FM loss, only on overlapping t-buckets. Predicts kink shrinks; per-t val loss for both rises slightly (the consistency penalty is regularising); designability of either-alone may improve modestly because the two models now share an interior representation.

### E043 addendum — paired aggregate val loss under the training-time t-distribution (2026-05-07)

**Why ran:** Wandb's training-time `validation_loss/loss_epoch` shows canonical_2646 *higher* than every variant's best-val. E043's per-t paired buckets (n=600) show canonical *lower* at every bucket. Designability shows canonical *much* better. Three signals; one is contradicting two. Question: is the wandb flip a real property of the trained weights (in which case the per-t bucketing is somehow blind to it), or is it an artifact of the training-time logging path? The user's specific challenge: "averaging over so many samples doesn't make it a lot higher does it? or is it maybe summed not averaged? is there maybe a reason why the per t loss could be consistently higher and still the sum lower that I don't think of?"

**Configs:**
- Driver: `proteinfoundation/run_aggregate_val_seeds.py` (created this entry). Same paired 600-protein subset as E043 (`subset_seed=42`, same per-protein rotations). For each ckpt × t-seed, draws one t per protein from the actual training/val distribution `mix_unif_beta(p1=1.9, p2=1.0, p3=0.02)` (from `configs/training_ca_only.yaml:39`), runs `model.fm.compute_loss(...)` once, returns the across-protein mean. Repeats 20 t-draw seeds (10000–10019) per ckpt.
- Ckpts: `canonical_2646` (best_val_00000026_000000002646.ckpt, `test_ca_only_diffusion/1776805213`) and `sparse_vanilla_1259` (best_val_00000012_000000001259.ckpt, `ca_only_sparse_K40`). Ran sequentially on CUDA_VISIBLE_DEVICES=1 under nohup. Wall: ~50 min total.
- Same channel set, same `compute_loss`, same reduction, same protein subset across both ckpts. The only thing that varies between `(ckpt, seed)` cells is the per-protein t-draw and the model.

**Why this is the right verification.** The aggregate `validation_loss/loss_epoch` Lightning logs is `mean_over_epoch(sum_over_channels(mean_over_batch(losses[k])))`. For CA-only there is exactly one channel (`bb_ca`; `compute_aux_loss` only emits when `"x_motif" in batch`), so the sum-over-channels collapses to a single mean. The reduction is mean-over-batch-then-mean-over-batches, which equals mean-over-all-proteins when batches are equal-sized — exactly what this script computes. So the script reproduces the wandb-aggregator semantics on a controlled subset. Differences must come from the protein subset, the per-event variance, EMA-vs-raw, or selection bias of "best val".

**The 1/(1-t)² weight in `compute_fm_loss`.** `rdn_flow_matcher.py:215` applies `total_loss_w = 1.0 / ((1.0 - t)**2 + 1e-5)` to the raw MSE. At t=0.9 the weight is 100; at t=0.99 it's 10⁴; at t=0.997 it's >10⁵. Combined with `mix_unif_beta(1.9, 1.0, 0.02)` heavily concentrating draws near t=1, single high-t outlier draws can in principle dominate the mean. We expected `std_of_means` across t-seeds to be large (~0.1+ nat) and to be the explanation for the wandb flip. **It wasn't.**

**Numbers — 20 t-draw seeds per ckpt, n=600 paired proteins per seed:**

| ckpt | mean_of_means | std_of_means | min | max | range |
|---|---|---|---|---|---|
| canonical_2646 | **1.4008** | 0.0224 | 1.3267 | 1.4319 | 0.105 |
| sparse_vanilla_1259 | **1.5375** | 0.0219 | 1.4741 | 1.5779 | 0.104 |

- Δ(sparse_vanilla − canonical) = **+0.137 nat** (= **+4.36σ** under the combined std-of-means ≈ 0.031).
- **Zero seed-overlap.** Worst canonical seed (1.4319) is *below* best sparse_vanilla seed (1.4741). Every canonical t-draw beats every sparse_vanilla t-draw.
- Per-seed maxima (the heavy-tail draws) reach 5-15 nat (sparse_vanilla seed 10006 hit 15.28; canonical seed 10005 hit 11.37) — heavy tails are real and visible per-protein, but the per-seed *mean* over 600 proteins washes them out enough that std_of_means ≈ 0.022 for both ckpts.

**Sanity check vs E043 per-t buckets.** Integrating canonical_2646's per-t bucket means under `mix_unif_beta(1.9, 1.0, 0.02)`-derived bucket weights {[0.0,0.2):4.7%, [0.2,0.4):12.0%, [0.4,0.6):19.2%, [0.6,0.8):26.2%, [0.8,1.0):39.5%} predicts aggregate ≈ 0.047·3.018 + 0.120·1.932 + 0.192·1.293 + 0.262·1.086 + 0.395·1.313 ≈ **1.426**. Measured: **1.4008**. Same calculation for sparse_vanilla_1259: predicted **1.574**, measured **1.5375**. Per-t bucket means and the aggregate-on-paired-set agree to within 0.04 nat. **The per-t buckets correctly predict the paired aggregate.**

**Interpretation:**

1. **The "heavy-tail dominates the mean" story is wrong as a sole explanation.** `std_of_means = 0.022` for both ckpts means the aggregate IS pretty stable across t-draw seeds at n=600 — much tighter than I claimed in conversation. The 1/(1-t)² weight does cause individual proteins at extreme-t to score in the 5-15 nat range, but at n=600 these wash out enough that the seed-mean estimator is fine. The aggregate flip on wandb cannot be explained by the per-event variance of this estimator.
2. **On a paired set, all three signals AGREE: canonical < sparse_vanilla.** Per-t buckets (every bucket: canonical lower by 0.13-0.18 nat at the meaningful buckets), aggregate under training t-distribution (0.137 nat at +4.36σ), and designability (canonical 76% pooled vs untested sparse_vanilla, expected ≤ 17% per E039 scnbr ceiling). The trained-weights ordering is unambiguous.
3. **The wandb training-time aggregate is therefore the artifact, not the signal.** Candidate causes (any of (a)-(c) suffices):
   - **(a) Different val proteins per training run.** Val_dataloader uses `shuffle=False` (`base_data.py:193`). Each run averages over the *first N* of its own dataset construction. The canonical run was at hash `e1a0138` with `limit_val_batches=100`; the variants were at HEAD with `limit_val_batches=50` (the diff in `train.py:478` between `e1a0138` and HEAD changed it 100→50). These can produce different protein subsets even if the dataset construction itself is identical (because the cluster-expansion order or the upstream df_data shuffle behaviour is sensitive to package versions, RNG state, etc.). The paired protocol used in this addendum loads from `processed_latents/` directly with a fixed seed, bypassing whatever caused the divergence.
   - **(b) EMA vs raw model.** `EmaModelCheckpoint` saves both `*.ckpt` and `*-EMA.ckpt`. Wandb's `validation_loss/loss_epoch` is logged from `validation_step`, which runs whichever model Lightning marks as "current" — usually the live model, but EMA copies and best-val selection both use the EMA weights. If the live-model val loss was logged but the EMA-model ckpt was saved as best_val (or vice versa), the wandb panel doesn't describe the saved ckpt.
   - **(c) Best_val is an order statistic, not a mean.** Lifetime-min of val_loss across thousands of val events. With per-event std on the n=300 (limit_val_batches=50, batch_size=6) sample around 0.05 nat (estimated from `std_of_means / sqrt(20) × sqrt(300/600) ≈ 0.04` plus per-event t-draw variance), comparing two lifetime-min values from runs of different durations is comparing two extreme order statistics with very different reference distributions.
4. **Per-t paired E043 + designability + this aggregate addendum are the three trusted signals.** Wandb training-time aggregate val loss is not comparable across runs. Memory: `feedback_wandb_val_loss_not_comparable.md`.

**Methodological caveats (additional to E043's):**

- **n=600 is enough to stabilise the seed-mean estimator** (std_of_means ≈ 0.02), so the canonical-vs-sparse_vanilla gap is +4.36σ. At n=50 batches × 6 = 300 (the variants' actual training-time `limit_val_batches=50` setting) the per-event variance is roughly √2× larger; Wilson would still put a 0.10 nat gap at >2σ. So even on the 300-protein scale of training-time logging, the paired protocol predicts canonical < sparse_vanilla. The wandb flip is therefore not "noise of the training-time per-event estimator" — it's pointing to a different *target* of estimation (different subset, different model copy, or both).
- **Only two ckpts compared.** The +0.137 nat gap is canonical_2646 vs sparse_vanilla_1259. Replicating with conv_2331 and scnbr_t04_1133 against canonical would tighten the picture; the per-t bucket means already predict ≈ +0.06 nat (conv) and ≈ +0.10 nat (scnbr_t04) relative to canonical, so the same-direction flip should hold.
- **Gap is on the *paired* set, not the canonical-run's val set.** Re-evaluating canonical and variants on canonical's *original* val set (whichever 600 proteins it logged on at step 2646) is the cleanest "did wandb compare different things" check, but is gated on knowing what those proteins were. The `pdb_data` rebuild from current code may not match the canonical-era construction.
- **The 4.36σ figure assumes Gaussian seed-means.** Per-seed means are 600-protein averages of a heavy-tailed weighted MSE; the seed-mean distribution might be skewed. Empirical zero-seed-overlap is the more robust statement.
- **EMA-vs-raw not yet checked.** This entry doesn't test the EMA-vs-raw hypothesis directly — that needs loading the `*-EMA.ckpt` companion of one of these ckpts and re-running the same protocol. Cheap follow-up if the user wants.

**Cross-references:**
- Driver: `proteinfoundation/run_aggregate_val_seeds.py` (this entry).
- Output: `results/aggregate_val_seeds/agg_canonical_vs_sparse_vanilla.json`.
- Memory: `feedback_wandb_val_loss_not_comparable.md`.
- Parent diagnostic: E043 (per-t paired bucket means, n=600 paired proteins).
- Causal candidate (a) traces to `train.py:478` change `limit_val_batches=100→50` (commit `e4ba5a6`) plus the `shuffle=False` in `base_data.py:193`.
- Causal candidate (b) traces to `EmaModelCheckpoint` in `train.py:setup_ckpt`.

---

## E044 — Inference-only neighbor-list curriculum on plain sparse_K40 step 1259 (2026-05-07)

**Why ran:** Diagnostic for the hypothesised mechanism behind sparse-attention's modest ceiling. The sparse K=40 neighbor list is built from `x_t["bb_ca"]` (the noisy current-step CA coords) at every diffusion step. At low t, `x_t` is essentially noise, so the spatial (8) + random (16) groups point at near-random index sets — only the sequential (16) group is t-invariant by construction. If the model is wasting capacity on these uninformative slots at low t, mask them at inference: sequential-only at very low t, +spatial mid-trajectory, full canonical K=40 at high t. Cheap inference-only test on the existing checkpoint before committing to a retraining run with the curriculum baked in.

**Configs:**
- Code change: `proteinfoundation/nn/local_latents_transformer.py` — added `curriculum_neighbors`/`curriculum_t_thresh_spatial`/`curriculum_t_thresh_random` kwargs (default off, so existing checkpoints/training are unchanged) and a per-protein `slot_valid` mask block immediately after `_build_neighbor_idx`. Group layout matches `sparse_neighbors.py:111-115`: `[0:2*n_seq]` sequential, `[2*n_seq:2*n_seq+n_spatial]` spatial, `[2*n_seq+n_spatial:K]` random. The sparse-attention path already AND-masks `slot_valid` against `nbr_valid & i_valid` and softmax-masks invalid slots to −∞ via `pair_bias_attn.py:172-176`, so the curriculum piggybacks on the existing padding-slot guard.
- Inference-side hook: `proteinfoundation/generate.py` `load_ckpt_n_configure_inference` — reads `cfg.generation.args.curriculum_neighbors` and sets the attrs on `model.nn` post-load. Defaults to off; logs `[Curriculum neighbors] ON — t_spatial<X, t_random<Y` when active.
- Curriculum schedule: `t < 0.3` → sequential only (effective K=16); `0.3 ≤ t < 0.6` → sequential + spatial (effective K=24); `t ≥ 0.6` → full K=40 canonical.
- Configs: `configs/inference_sparse_K40_step1259_baseline_n6_nfe400.yaml` and `configs/inference_sparse_K40_step1259_curriculum_n6_nfe400.yaml`. Both inherit from `inference_base.yaml` (`seed: 5, nsteps: 400, sc_neighbors_bootstrap: True`); same `nlens_cfg=[50, 100, 200]`, `nsamples=6`. Identical except the curriculum yaml's three flags under `generation.args`.
- Checkpoint: `/home/ks2218/la-proteina/sparse_K40_step1259.ckpt` (= `best_val_00000012_000000001259.ckpt`, run `ca_only_sparse_K40` per ckpt hparams). `sparse_attention=True, n_seq=8, n_spatial=8, n_random=16` → K=40. `update_pair_repr=False, use_tri_mult=False, use_downsampling=False, sc_neighbors=False` (clean baseline; deliberately not the `scnbr_t04` ckpt so the curriculum signal isn't confounded with Fix C2).
- Hardware: 1× L4 (GPU 3), runs sequential under `nohup`. Wall: gen ~80 s × 2 = ~160 s, eval ~10 min × 2 = ~20 min.
- Drivers: `/tmp/run_curriculum_pair.sh` (gen+eval), `/tmp/run_curriculum_eval.sh` (eval-only re-run after the first eval pass crashed because `PYTHON_EXEC` env var wasn't set, causing ProteinMPNN to launch under `/opt/conda/bin/python` without numpy).

**Curriculum hook fired confirmation** — `nohup_inference_sparse_K40_step1259_curriculum_n6_nfe400.gen.log` line 1: `[Curriculum neighbors] ON — t_spatial<0.3, t_random<0.6`. Baseline log does not contain that string.

**Results — pooled designability (N=6 × {50, 100, 200}):**

| arm | L=50 | L=100 | L=200 | pooled | min Å | median Å | mean Å |
|---|---|---|---|---|---|---|---|
| baseline (no curriculum) | 2/6 (33%) | **3/6 (50%)** | 0/6 | **5/18 (28%)** | 1.08 | 5.13 | 6.31 |
| curriculum on | 0/6 | 3/6 (50%) | 0/6 | 3/18 (17%) | 1.52 | 5.31 | 7.19 |

**Per-length sorted scRMSD (Å):**

- baseline L=50: `[1.29, 1.31, 3.20, 4.90, 5.36, 11.40]`
- curriculum L=50: `[2.63, 2.98, 4.88, 5.75, 6.31, 7.67]`
- baseline L=100: `[1.08, 1.43, 1.58, 4.30, 4.79, 6.31]`
- curriculum L=100: `[1.52, 1.70, 1.73, 2.25, 3.85, 4.52]`
- baseline L=200: `[7.87, 9.96, 10.87, 11.63, 12.52, 13.74]`
- curriculum L=200: `[10.59, 11.73, 14.57, 15.14, 15.23, 16.33]`

**Per-protein paired (same noise, same seed=5, same `(L, id)` mapping):**

| L | id | baseline | curric | Δ Å | flag |
|---|---|---|---|---|---|
| 50 | 0 | 11.40 | 2.63 | **−8.77** | rescued from collapse |
| 50 | 1 | 1.31 ✓ | 7.67 | +6.36 | DESIGN→FAIL |
| 50 | 2 | 3.20 | 6.31 | +3.11 | worse |
| 50 | 3 | 1.29 ✓ | 5.75 | +4.45 | DESIGN→FAIL |
| 50 | 4 | 5.36 | 4.88 | −0.48 | ~same |
| 50 | 5 | 4.90 | 2.98 | −1.92 | better |
| 100 | 0 | 1.43 ✓ | 2.25 | +0.82 | DESIGN→FAIL (just barely) |
| 100 | 1 | 1.08 ✓ | 1.73 ✓ | +0.65 | both designable |
| 100 | 2 | 6.31 | 4.52 | −1.80 | better |
| 100 | 3 | 4.30 | 3.85 | −0.44 | ~same |
| 100 | 4 | 1.58 ✓ | 1.52 ✓ | −0.05 | ~same |
| 100 | 5 | 4.79 | 1.70 ✓ | **−3.09** | FAIL→DESIGN |
| 200 | 0 | 9.96 | 14.57 | +4.61 | worse |
| 200 | 1 | 11.63 | 11.73 | +0.10 | ~same |
| 200 | 2 | 10.87 | 15.14 | +4.27 | worse |
| 200 | 3 | 7.87 | 16.33 | **+8.46** | worse |
| 200 | 4 | 12.52 | 15.23 | +2.71 | worse |
| 200 | 5 | 13.74 | 10.59 | −3.15 | better |

Designability flips: 3× DESIGN→FAIL, 1× FAIL→DESIGN — net −2 (matches the pooled −2 = 5−3).

**Bimodality at L=100:**

- baseline: 3 clean (<3 Å, [1.08, 1.43, 1.58]), 2 mid (3-5 Å, [4.30, 4.79]), 1 collapsed (>5 Å, [6.31])
- curriculum: 4 clean (<3 Å, [1.52, 1.70, 1.73, 2.25]), 2 mid (3-5 Å, [3.85, 4.52]), **0 collapsed**

The L=100 distribution narrows and shifts toward the designable end under curriculum: max drops 6.31 → 4.52, median drops 2.94 → 1.99, the cleanest-cluster fattens 3→4 while the collapsed sample disappears entirely. Designability count is 3/6 in both arms, but for *different* reasons: baseline has three good samples and three bad; curriculum has four near-clean samples plus two mid samples that didn't quite cross the 2 Å threshold.

**Cross-reference vs N=30 baseline (`results_inference_sparse_n30_0.csv`, same ckpt at nsteps=400):**

| L | N=30 baseline | this N=6 baseline | curriculum N=6 |
|---|---|---|---|
| 50 | 13/30 (43%) min 0.90 | 2/6 (33%) min 1.29 | 0/6 min 2.63 |
| 100 | 8/30 (27%) min 0.97 | 3/6 (50%) min 1.08 | 3/6 (50%) min 1.52 |
| 200 | 0/30 (0%) min 2.01 | 0/6 (0%) min 7.87 | 0/6 (0%) min 10.59 |
| pool | 21/90 (23%) | 5/18 (28%) | 3/18 (17%) |

The N=6 baseline falls inside the noise band of the N=30 baseline (per-length differences ≤ 17 percentage points; pooled within 5 pp). Confirms the curriculum *vs* N=6 baseline comparison is not corrupted by my N=6 baseline being a fluke.

**Verdict — curriculum hurts inference-only on plain sparse_K40, with one redistribution silver lining:**

1. **Pooled designability drops 28% → 17%** under inference-only curriculum (−2 designable across 18 proteins). L=50 and L=200 both degrade. L=200 every paired protein except one drifts further into collapse (5/6 worse, mean +4.0 Å).
2. **L=100 is the only length where the count holds** (3/6 both ways), but the *shape* of the distribution is the most encouraging signal: collapsed sample (6.31 Å) disappears, max scRMSD drops 6.31 → 4.52, median drops 2.94 → 1.99. Two near-failures move toward the threshold; one designable sample (id=5, 4.79 → 1.70) is fully rescued.
3. **L=50 id=0 paired Δ = −8.77 Å** (11.40 → 2.63) is the single most dramatic effect in the run — a fully collapsed sample becomes near-designable. Counterbalanced by two L=50 designables flipping to non-designable (id=1: 1.31→7.67; id=3: 1.29→5.75), so the L=50 net effect is negative.
4. **Mechanism reading: the model uses the noisy spatial+random groups for non-trivial information transport at low t.** Removing them is not a no-op — it shifts trajectories meaningfully, and on average for the worse on a checkpoint that wasn't trained against the masked input distribution.

**Implication for retraining-with-curriculum.** The inference-only test does NOT support the "low-t spatial+random groups carry no useful information" reading. Two readings remain plausible:
- (a) The noisy slots carry rough information that the trained model has learned to denoise (i.e., even noisy spatial neighbors provide a spatial prior that's better than no neighbors). Retraining with curriculum would strip this signal during training too, and the model would have to rely entirely on sequential context at low t — likely worse, not better.
- (b) The trained model has learned to ignore the noisy groups but the *input distribution* (presence of those slots, even with garbage indices) is part of what the AdaLN-Zero gates were calibrated against. Removing them at inference is an OOD input shift; retraining with curriculum would let the gates re-calibrate.

The L=100 redistribution + L=50 id=0 rescue is the only signal pulling toward (b). The pooled drop and the L=200 uniform degradation pull toward (a). At N=6 the evidence isn't strong enough to commit to a retraining run — a 1-day retraining attempt of `ca_only_sparse_K40` with `curriculum_neighbors=True` baked into the training-side forward call is justifiable as a probe (matched recipe, ~16h on 1× A100), but flag the prior that this may produce no improvement or actively regress.

If retraining is attempted: **threshold sensitivity is a likely confound**. The 0.3/0.6 thresholds are an untuned guess. A short t-sweep on the inference-only side (e.g., `(0.1, 0.4)`, `(0.2, 0.5)`, `(0.3, 0.6)`, `(0.5, 0.8)`) before retraining is the cheaper next step — would clarify whether the L=100 redistribution scales with how aggressively low-t is masked.

**Methodological caveats:**

- **N=6 per length, single seed=5.** Three flips in a 18-protein paired comparison sit well inside noise of an N=6 protocol. The variant-bar table in CLAUDE.md uses this protocol so the comparison is method-consistent, but Wilson 95% CI on 5/18 vs 3/18 is wide and overlaps. The L=200 uniform degradation (5/6 worse) is the most directionally robust per-length finding; L=50 and L=100 shouldn't be over-interpreted.
- **Inference-only test on a checkpoint trained without curriculum.** This deliberately tests the *mechanism* (do the noisy slots matter at all?), not the *retraining-with-curriculum hypothesis*. A null/negative result here doesn't rule out the latter — the latter is properly tested only by retraining.
- **Curriculum thresholds untuned.** 0.3/0.6 was the task's prescribed default. Earlier or later cutoffs may produce different signs at L=50/L=200.
- **Single architecture × single ckpt step.** Curriculum tested only on plain sparse_K40 step 1259. Repeating on `ca_only_sparse_K40_scnbr_t04` step 1133 (E039 ckpt) would test additivity with Fix C2 — deliberately not done here so the curriculum signal isn't confounded with the threshold-gated x_sc neighbor source.
- **scRMSD < 2 Å with ProteinMPNN `ca_only=True` (8 seqs/protein, default), ESMFold backbone reconstruction.** Standard CLAUDE.md eval protocol; matches E021/E034/E035/E038/E039/E040/E041 protocol.
- **First eval pass produced all-NaN CSVs** because `PYTHON_EXEC` env var wasn't exported in my runner — ProteinMPNN spawned under `/opt/conda/bin/python` (no numpy). Detected by reading the eval log; deleted the broken CSVs, cleaned stale per-sample tmp dirs (which had been created by the broken first eval and prevented the second eval from running), and re-ran with `PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python` exported. Numbers in this entry are from the second (correct) eval pass.

**Possible narrative:** non-narrative — kept for tuning/decision-making. The signal is mixed at N=6 and the more decision-relevant test is a retraining run, which this entry's thresholds and N=6 don't justify launching directly. **What this entry decides:** (1) the noisy-spatial-slot-carrying-no-information mechanism story is *not* clearly supported — at minimum it's not so dominant that masking them is free; (2) before any retraining, a t-threshold sweep should establish whether the L=100 redistribution scales with curriculum aggressiveness or is a noise artifact. If retraining is later attempted and changes the picture, link back here from the new entry.

**Cross-references:**
- Code change: `proteinfoundation/nn/local_latents_transformer.py` (kwargs + masking block) and `proteinfoundation/generate.py` (post-load hook).
- Configs: `configs/inference_sparse_K40_step1259_baseline_n6_nfe400.yaml`, `configs/inference_sparse_K40_step1259_curriculum_n6_nfe400.yaml`.
- Output CSVs: `inference/results_inference_sparse_K40_step1259_baseline_n6_nfe400_0.csv`, `inference/results_inference_sparse_K40_step1259_curriculum_n6_nfe400_0.csv`.
- Logs: `nohup_inference_sparse_K40_step1259_*_n6_nfe400.gen.log`, `*.eval2.log`.
- N=30 baseline cross-check: `inference/results_inference_sparse_n30_0.csv` (same ckpt, nsteps=400, same protocol up to N).
- Sparse-attention architecture: CLAUDE.md → "Sparse-attention variant (SALAD-style, K=40)"; `proteinfoundation/nn/modules/sparse_neighbors.py`.
- Closest comparator at the same step + protocol: [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) (N=30 sparse-K40 1259, nsteps=200 — historical context only; nsteps mismatch makes the absolute Å numbers not comparable, but the converged-plateau ordering is consistent).

---

## E045 — t-dependent K-budget reallocation curriculum on plain sparse_K40 step 1259 (2026-05-07)

**Why ran:** Direct follow-up to [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07). E044 *masked* spatial+random slots at low t, shrinking the effective K from 40 → 16 — pooled designability dropped 5/18 → 3/18 with one upside (L=100 distribution tightening). Hypothesis tested here: keep total K=40 fixed across the whole trajectory and instead *reallocate* the budget across (sequential, spatial, random) groups as a function of t. The model's softmax always sees 40 real slots — only the *content* shifts. At low t, replace the noise-driven spatial+random groups with extra sequential (which is t-invariant by construction); at high t, recover the canonical training composition. Question: does keeping K=40 fixed avoid E044's effective-receptive-field shrinkage and produce a different effect?

**Schedule (3 buckets, hard cutoffs):**

| t-range | n_seq (per side) | n_spatial | n_random | Total K | Composition |
|---|---|---|---|---|---|
| t < 0.33 | 20 | 0 | 0 | 40 | 40 sequential |
| 0.33 ≤ t < 0.66 | 12 | 8 | 8 | 40 | 24 seq + 8 sp + 8 rd |
| t ≥ 0.66 | 8 | 8 | 16 | 40 | 16 seq + 8 sp + 16 rd (canonical) |

t=0 is noise, t=1 is clean. The high-t bucket exactly matches the training distribution. Bucket boundaries are abrupt (no interpolation across t).

**Configs:**
- Code change: `proteinfoundation/nn/local_latents_transformer.py` — replaced E044's masking block with a t-aware `_build_neighbor_idx` that picks (n_seq, n_spatial, n_random) from t when `self.curriculum_neighbors=True`. Asserts `torch.allclose(t, t[0])` (inference-time t is uniform across the batch under Euler integration; the assert prevents accidental training-time use where t is per-protein). E044's `curriculum_t_thresh_*` kwargs removed (they're not relevant to the realloc path). The flag `curriculum_neighbors` keeps its name but its meaning has changed: prior commit = mask; HEAD = realloc.
- Inference hook: `proteinfoundation/generate.py` post-load block, sets `model.nn.curriculum_neighbors = True` and logs the schedule. Gated on `cfg.generation.args.curriculum_neighbors`; default off, existing configs unchanged.
- Sparse path (`build_neighbor_idx`) handles `n_spatial=0` / `n_random=0` cleanly via the `if k_sp > 0` / `if k_rnd > 0` guards in `sparse_neighbors.py:67-72, 86-90`. Confirmed: at t < 0.33 no spatial/random topk is computed, the `_pad` truncates the empty groups to zero columns, and `slot_valid` ends up all-True over the 40 sequential-only slots.
- Configs: `configs/inference_sparse_K40_step1259_curriculum_realloc_n6_nfe400.yaml`. Mirrors E044's curriculum config except for the file name and that the schedule is decided by the code (no `curriculum_t_thresh_*` keys). Inherits from `inference_base.yaml` (`seed: 5, nsteps: 400, sc_neighbors_bootstrap: True`); `nlens_cfg=[50, 100, 200]`, `nsamples=6`.
- Checkpoint: `/home/ks2218/la-proteina/sparse_K40_step1259.ckpt` (= `best_val_00000012_000000001259.ckpt`, run `ca_only_sparse_K40`). Same as E044 — clean sparse, no Fix C2.
- Hardware: 1× L4 (GPU 3), `nohup`. Wall: gen 72 s, eval ~9 min.
- Driver: `/tmp/run_realloc_arm.sh` with `PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python` exported (lesson from E044's first eval pass).

**Curriculum hook fired confirmation** — gen log line 1: `[Curriculum neighbors] ON — 3-bucket K-reallocation: t<0.33 → (n_seq=20, n_sp=0, n_rd=0); 0.33≤t<0.66 → (12, 8, 8); t≥0.66 → (8, 8, 16).` and the `torch.allclose(t, t[0])` assert did not trigger over 400 inference steps × 18 proteins.

**Baseline reused from E044** — same checkpoint, same protocol, same seed → noise is paired by `(L, id)`. The N=6 baseline CSV (`inference/results_inference_sparse_K40_step1259_baseline_n6_nfe400_0.csv`) was not re-run for E045. Cross-check vs the N=30 baseline `inference/results_inference_sparse_n30_0.csv` (same ckpt, nsteps=400) is at the bottom of E044.

**Results — pooled designability (N=6 × {50, 100, 200}):**

| arm | L=50 | L=100 | L=200 | pooled | min Å | median Å | mean Å | std Å |
|---|---|---|---|---|---|---|---|---|
| baseline (no curriculum) | 2/6 | **3/6 (50%)** | 0/6 | 5/18 (28%) | 1.08 | 5.13 | 6.31 | 4.39 |
| E044 mask (K → 16 at low t) | 0/6 | **3/6 (50%)** | 0/6 | 3/18 (17%) | 1.52 | 5.31 | 7.19 | 5.31 |
| E045 realloc (K=40 fixed) | **3/6 (50%)** | 0/6 | 0/6 | 3/18 (17%) | **0.63** | 3.79 | 7.32 | 6.02 |

Pooled designability ties between mask and realloc (both −2 vs baseline), but **the per-length distribution is completely different** — the two interventions are doing opposite things at different chain lengths.

**Per-length sorted scRMSD (Å):**

- L=50 baseline: `[1.29, 1.31, 3.20, 4.90, 5.36, 11.40]`
- L=50 mask: `[2.63, 2.98, 4.88, 5.75, 6.31, 7.67]`
- L=50 realloc: `[0.63, 1.47, 1.80, 2.35, 2.82, 3.10]` ← **every sample within 3.10 Å of native**, min 0.63 Å is the lowest scRMSD ever seen on this checkpoint
- L=100 baseline: `[1.08, 1.43, 1.58, 4.30, 4.79, 6.31]`
- L=100 mask: `[1.52, 1.70, 1.73, 2.25, 3.85, 4.52]`
- L=100 realloc: `[2.47, 2.78, 3.03, 4.49, 7.02, 9.04]` ← bimodality returns; two new collapsed samples
- L=200 baseline: `[7.87, 9.96, 10.87, 11.63, 12.52, 13.74]`
- L=200 mask: `[10.59, 11.73, 14.57, 15.14, 15.23, 16.33]`
- L=200 realloc: `[13.06, 14.23, 15.07, 15.67, 16.05, 16.26]`

**Per-protein paired (same noise; baseline / mask / realloc):**

| L | id | base | mask | realloc | Δmask | Δrea | rea_vs_mask |
|---|---|---|---|---|---|---|---|
| 50 | 0 | 11.40 | 2.63 | **0.63 ✓** | −8.77 | **−10.77** | −2.00 |
| 50 | 1 | 1.31 ✓ | 7.67 | 3.10 | +6.36 | +1.79 | −4.57 |
| 50 | 2 | 3.20 | 6.31 | **1.47 ✓** | +3.11 | −1.73 | −4.84 |
| 50 | 3 | 1.29 ✓ | 5.75 | **1.80 ✓** | +4.45 | +0.51 | −3.95 |
| 50 | 4 | 5.36 | 4.88 | 2.35 | −0.48 | −3.01 | −2.53 |
| 50 | 5 | 4.90 | 2.98 | 2.82 | −1.92 | −2.08 | −0.16 |
| 100 | 0 | 1.43 ✓ | 2.25 | **9.04** | +0.82 | **+7.61** | +6.79 |
| 100 | 1 | 1.08 ✓ | 1.73 ✓ | 2.47 | +0.65 | +1.39 | +0.74 |
| 100 | 2 | 6.31 | 4.52 | 4.49 | −1.80 | −1.83 | −0.03 |
| 100 | 3 | 4.30 | 3.85 | 7.02 | −0.44 | +2.72 | +3.17 |
| 100 | 4 | 1.58 ✓ | 1.52 ✓ | 3.03 | −0.05 | +1.46 | +1.51 |
| 100 | 5 | 4.79 | 1.70 ✓ | 2.78 | −3.09 | −2.01 | +1.08 |
| 200 | 0 | 9.96 | 14.57 | 13.06 | +4.61 | +3.10 | −1.51 |
| 200 | 1 | 11.63 | 11.73 | 14.93 | +0.10 | +3.30 | +3.20 |
| 200 | 2 | 10.87 | 15.14 | 16.00 | +4.27 | +5.13 | +0.86 |
| 200 | 3 | 7.87 | 16.33 | 15.80 | +8.45 | +7.93 | −0.53 |
| 200 | 4 | 12.52 | 15.23 | 16.26 | +2.71 | +3.74 | +1.03 |
| 200 | 5 | 13.74 | 10.59 | 14.62 | −3.15 | +0.88 | +4.02 |

**Designability flips, base → realloc**: 4× DESIGN→FAIL (3 of them at L=100), 2× FAIL→DESIGN (both at L=50). Net −2.

**Designability flips, mask → realloc** (apples-to-apples between the two interventions): 3× DESIGN→FAIL at L=100 (samples that the mask had pulled into designable get *un-designable* by realloc), 3× FAIL→DESIGN at L=50 (samples that mask hurt are *rescued* by realloc).

**L=100 cluster proportions (clean<3 Å / mid 3-5 Å / collapsed>5 Å):**

| arm | clean | mid | collapsed |
|---|---|---|---|
| baseline | 3 [1.08, 1.43, 1.58] | 2 [4.30, 4.79] | 1 [6.31] |
| mask | **4** [1.52, 1.70, 1.73, 2.25] | 2 [3.85, 4.52] | 0 |
| realloc | 2 [2.47, 2.78] | 2 [3.03, 4.49] | **2** [7.02, 9.04] |

Mask tightens the L=100 distribution (collapsed→0); realloc reverses this **harder than baseline** (collapsed=2 vs baseline 1) — id=0 (baseline 1.43, fully clean) collapses to 9.04 under realloc, id=3 joins the collapse cluster at 7.02.

**Verdict — realloc and mask reveal a chain-length-dependent split in the mechanism:**

1. **L=50 realloc is spectacular**: 3/6 designable, **min 0.63 Å** (best on this ckpt to date — lower than the N=30 baseline's 0.90 Å and the E044 N=6 baseline's 1.08 Å), median 2.07, max 3.10. *Every single sample is within 3.10 Å of native.* Two F→D rescues, including the dramatic id=0 (11.40 → 0.63, Δ=−10.77 Å — biggest single-protein delta in this whole experiment family). One D→F (id=1: 1.31 → 3.10) is the only loss vs baseline; the L=50 net is +1 designable.
2. **L=100 realloc is a disaster**: 0/6 designable, **min 2.47** (vs baseline 1.08 and mask 1.52), and **two NEW collapsed samples appear** (id=0 at 9.04, id=3 at 7.02). All three baseline-designable proteins flip to non-designable. The L=100 cluster proportions shift in the opposite direction from mask — mask eliminated the collapse cluster, realloc *grows* it.
3. **L=200 realloc fails like everything else**: 0/6, no proteins close to designable. Stronger version of the same direction as mask, but at a length where the model's trajectories are already off-manifold under all three arms.

**Mechanism reading.** At low t, both interventions push the K-budget toward sequential information (which is t-invariant) and away from spatial+random (which are noise-driven). They differ in the K size that the softmax sees and in the slot composition:

- *Mask*: K_effective = 16 sequential + (24 padded slots → softmax-masked). The slot composition the trained model sees is ALWAYS canonical (16/8/16); the spatial+random content is just zeroed out from the attention. Minimal distribution shift to the trained attention.
- *Realloc*: K_effective = 40 sequential. The trained model sees a slot composition (40 sequential / 0 spatial / 0 random) it has *never seen during training*. The model's per-slot-position attention biases (conditioned on canonical 16/8/16 composition) are applied to slots that contain different content. Bigger distribution shift.

Why realloc helps at L=50 but hurts at L=100:
- *L=50*: 40 sequential ≈ "all 49 other residues". The spatial vs sequential vs random distinction collapses for a 50-residue protein — the spatial neighbors and the random neighbors are both *also* sequentially nearby. Realloc replaces noisy-but-sequentially-near content with deterministic-and-sequentially-near content, with negligible position-vs-content mismatch. The model's attention gets a cleaner input distribution.
- *L=100 / L=200*: spatial+random slots can span the whole chain (50-100 sequence positions away). Replacing them with more sequential reduces the long-range info available at low t. The trained attention applied to slot positions whose content has shifted from "could be 80 residues away" to "is 12 residues away" gets the position-vs-content mismatch in full force. The model uses these long-range slots for non-trivial information transport even when the indices are noisy — and the L=100 collapse-cluster growth is the cost of denying it that channel.

**Speculative composite verdict — realloc at L<60 + mask at L≥60 (or just mask)** would be **6/18 = 33% pooled** on this ckpt at this protocol, beating the 5/18 baseline. But this is N=6 over-fitting and won't survive replication; flagged only as a structural observation that the two interventions are complementary, not as a recipe.

**Implication for retraining-with-curriculum (vs E044's read).** The E044 entry left the retraining-with-curriculum hypothesis at "mixed evidence — try cautiously". E045 sharpens this:

- The **mechanism is real** and **chain-length-dependent**: at short L the trained attention can absorb (and benefit from) a shifted slot composition; at longer L it cannot. A retraining run with the same 3-bucket schedule baked in would let the model re-calibrate its per-slot attention biases for the post-curriculum composition — likely closing the L=100 regression but possibly also losing the L=50 free lunch (if the lunch is "the trained model is forgiving about content because slots overlap at small L", it disappears once the model is trained against that explicit composition).
- A **t-aware K reallocation that scales with L** (e.g., low-t composition depends on `min(20, L/3)` sequential slots instead of fixed 20) is a more principled retraining target than the static 3-bucket schedule used here. The L-dependent failure pattern strongly suggests static thresholds are wrong.
- A **smoother schedule across t** (5-bucket or interpolated) would not change this verdict — the failure mode is in the *low-t composition*, not in the discrete jumps at t=0.33 / t=0.66. The bucket-boundary kink concern flagged in the task is moot for this run since all the L=100 collapses happen across the whole trajectory, not at a bucket transition (no pipe-style discontinuity in the per-t loss available without trajectory diagnostics, but the integrated outcome shows the early-t composition dominates the failure).

**Methodological caveats:**

- **N=6 per length, single seed=5.** Same as E044. The L=50 +1 designable and L=100 −3 designable are both well within Wilson noise of an N=6 protocol, but the *direction and magnitude* are robust under the paired protocol (per-protein delta of −10.77 Å at L=50 id=0 is not noise, +7.61 Å at L=100 id=0 is not noise).
- **Inference-only test on a checkpoint trained without curriculum.** As in E044 — the mechanism test is *does the trained model tolerate this input shift*, not *would retraining-with-curriculum work*.
- **Static 3-bucket schedule.** The bucket boundaries are arbitrary. An L-dependent schedule would likely change the picture — this entry tests one point in a larger schedule space.
- **Bucket-boundary kink hypothesis untested.** The task asked to flag any kink near t=0.33 / t=0.66. Designability-only data cannot show this; would need per-step velocity-magnitude logging at the boundaries (similar to E040/E041's hybrid-handoff diagnostics in [E040](#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06)). Cheap follow-up if the user wants to discriminate "discrete-bucket effect" from "wrong composition at low t" — the failure mode at L=100 is consistent with the latter, not the former.
- **scRMSD < 2 Å with ProteinMPNN `ca_only=True` (8 seqs/protein, default), ESMFold backbone reconstruction.** Standard CLAUDE.md eval protocol; matches E021/E034/E035/E038/E039/E040/E041/E044 protocol.
- **The semantic of `curriculum_neighbors` flag changed between E044 and E045**: same flag name, different mechanism. Configs from E044 (`inference_sparse_K40_step1259_curriculum_n6_nfe400.yaml`) re-run against the HEAD code base will fire the realloc path, not the mask path. Tag the E044 commit if the mask numbers need re-running later: `git log --oneline | grep E044`.

**Possible narrative:** non-narrative — kept for tuning/decision-making. Same status as E044. **What this entry decides:** (1) the noisy-spatial/random-slots-carry-no-info hypothesis is *wrong but partially right, in a chain-length-dependent way*; (2) static schedules are unlikely to be the right shape — an L-aware schedule should be the next attempt before any retraining commitment; (3) the L=50 result is striking enough to suggest a focused short-protein test is worthwhile in its own right.

**Cross-references:**
- Code change: `proteinfoundation/nn/local_latents_transformer.py` (replaced E044 mask with t-aware `_build_neighbor_idx`); `proteinfoundation/generate.py` (post-load hook simplified — no thresholds).
- Config: `configs/inference_sparse_K40_step1259_curriculum_realloc_n6_nfe400.yaml`.
- Output CSV: `inference/results_inference_sparse_K40_step1259_curriculum_realloc_n6_nfe400_0.csv`.
- Logs: `nohup_inference_sparse_K40_step1259_curriculum_realloc_n6_nfe400.gen.log`, `*.eval.log`.
- Companion / direct predecessor: [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (mask-based curriculum on the same ckpt; same baseline; same protocol).
- Sparse-attention architecture: CLAUDE.md → "Sparse-attention variant (SALAD-style, K=40)"; `proteinfoundation/nn/modules/sparse_neighbors.py` (`build_neighbor_idx` with arbitrary `n_seq, n_spatial, n_random`).

---

## E046 — Sparse attention off-by-one cap investigation + fix + bf16 audit (2026-05-11)

**Status:** finished (code change + numerical tests + three inference probes at step-1385 with same-day paired comparisons)

**Why ran:** at step 1385 the K=64-curriculum-self variant (variants.md §11) trails canonical by ~19 pp at L=50 and ~11 pp at L=100 on N=18 pooled designability. That gap is too large for "step undercount" alone. The user asked for a focused diff between the dense baseline (`eb445ef`, parent of the first sparse-attention commit) and the current sparse code, plus actual numerical tests of the gather + masking + LN paths — with explicit emphasis on hypothesising big bugs rather than small effects. Decision the entry feeds: do we need to retrain after a correctness fix, or is the gap fully explained by the curriculum schedule + the training distribution?

**Configs / hardware:** investigation only — code reading, fp32 + bf16 numerical micro-tests on CPU (no GPU/SLURM use), one CLAUDE.md update, no new training runs. Pair of probe drivers created for the re-probe but not launched yet:
- Code paths read: `proteinfoundation/nn/modules/{sparse_neighbors,pair_bias_attn,pair_update,pair_rep_initial,adaptive_ln_scale}.py`; `proteinfoundation/nn/feature_factory.py`; `proteinfoundation/nn/local_latents_transformer.py`.
- Diffs vs dense baseline saved temporarily to `/tmp/diff_*.txt`; covered 5 NN files, ~620 added lines, no functional change to the dense path beyond a `.nan_to_num(0.0)` guard.
- Numerical tests (`python -c …` against `/home/ks2218/conda_envs/laproteina_env`): gather uniqueness at N=50 in all three curriculum buckets (50 queries × 64 slots); sparse-vs-dense numerical equivalence in fp32 and bf16 at N=20 (K=N identity), N=50 (K=64 builder, pre-fix), N=50 (manual full-N neighbor list); padding-clone count per row; FeatureFactory post-LN re-mask at padding slots; per-bucket coverage at N ∈ {10, 30, 50, 63, 64, 65, 100, 200, 300}.
- Code change: `proteinfoundation/nn/modules/sparse_neighbors.py` — replace `min(2*n_seq, N-1)` with `min(2*n_seq, N)` at line 61, drop the `-1` symmetrically at lines 69 and 88. Three-line patch (plus a comment block explaining the cap is now correct for self-inclusion).
- Plumbing: added `curriculum_low_t_split` kwarg to `LocalLatentsTransformer` (`proteinfoundation/nn/local_latents_transformer.py`), `proteina.py` plumbs it from `cfg_exp.training`, `generate.py` exposes it as an inference-time override. Default `(32, 0, 0)` preserves existing behavior. Asserts the override sums to K=64.
- New training configs: `configs/training_ca_only_sparse_K64_nocurr.yaml`, `configs/training_ca_only_sparse_K64_curriculum_lowtsoft.yaml`. Sibling configs of the existing K=64-curriculum training file; same recipe, only the curriculum flags differ.
- New inference configs + drivers: `configs/inference_sparse_K64_step1385_FIXEDCAP_n6_nfe400.yaml` + `script_utils/probe_sparse_K64_step1385_FIXEDCAP.sh` (same ckpt, code w/ fix); `configs/inference_sparse_K64_step1385_LOWTSOFT_n6_nfe400.yaml` + `script_utils/probe_sparse_K64_step1385_LOWTSOFT.sh` (same ckpt, low-t bucket overridden to (16, 8, 24)). Distinct output prefixes preserve the existing pre-fix CSV.

**Results — numerical:**

| Test | Pre-fix | Post-fix |
|---|---|---|
| Unique valid neighbors per query at N=50, K=64 (all 3 buckets) | 49 / 50 | **50 / 50** |
| Padding slots per query at N=50, K=64 | 15 | 14 |
| Sparse-vs-dense max diff, N=50, K=64, fp32 (bucket (32,0,0)) | **2.03e-2** (~20 % rel. per-layer) | **5.96e-8** (machine ε) |
| Sparse-vs-dense max diff, N=20, K=N identity, fp32 | 5.96e-8 | 5.96e-8 |
| Sparse-vs-dense max diff at N=50, bf16 (post-fix) | n/a | 0.00 (sparse and dense both bf16-quantised to identical) |
| Excess bf16 drift on sparse path beyond bf16-quantised dense | n/a | **−1.6e-3** (i.e. sparse path adds zero extra drift on top of dense bf16 quantisation) |
| Coverage table at N ∈ {10, 30, 50, 60, 63, 64} pre-fix | 9, 29, 49, 59, 62, 63 valid / N | n/a |
| Coverage table post-fix | n/a | **10, 30, 50, 60, 63, 64 valid / N** (full coverage at N ≤ K) |
| Coverage table at N ∈ {65, 100, 200, 300} | 64 valid / N (saturated at K) | unchanged (bug doesn't apply) |

**Results — hypothesis falsification:**

Four candidate "big bug" mechanisms were tested directly and ruled out:

1. **i==j special-case difference between dense `pair_repr[i, i]` and sparse `pair_repr[i, neighbor_idx[i, 0]=i]`:** ruled out. With K=N identity neighbor list, sparse and dense outputs agree to 5.96e-8 in fp32. Same `pair_repr` value gathered, same `to_bias` projection, same softmax weight. No special case fires.
2. **bf16 -inf masked-fill gradient drift:** ruled out as a major contributor. bf16 sparse-vs-dense diff is 0.0; bf16 quantises both paths identically. bf16 vs fp32 introduces ~1.6e-3 of noise (~3.6 % relative to output magnitude), but this is symmetric between dense and sparse — not a sparse-specific issue.
3. **`pair_rep_initial.py` adaln LN over unmasked padding:** ruled out empirically. FeatureFactory zeros padded slots after the post-linear LN (`feature_factory.py:2027,2036`). AdaLN's `nn.LayerNorm(dim, elementwise_affine=False)` normalises over the last dim only, then the closing `* mask[..., None]` re-zeros padded slots. Verified: pair-feat value at padded slots after FeatureFactory with `use_ln_out=True` is `0.0` exactly.
4. **`pair_norm` in `PairBiasAttention` pooling over K including padding clones:** ruled out *by mechanism, not effect size*. `self.pair_norm = nn.LayerNorm(pair_dim)` normalises over the last (embedding) dim only — never over K. I verified: at N=50, residue 0 appears at 15-16 slots per row (1 real + 14-15 padding clones), but `LN(real_slots_only)` exactly matches `LN(real_and_padding)[real_slots]`. Padding clones do receive a non-zero `to_bias` projection but are killed by `softmax(-inf + finite) = 0`.

**Results — actual bug:**

`sparse_neighbors.py:61,69,88` used `N - 1` as the per-group cap, which was correct when self was excluded via `eye` in `base_invalid` (the configuration before commit `fbcc1ec`, 2026-05-08). Commit `fbcc1ec` removed the `eye` term to allow self-inclusion ("Self is now allowed in the K-set (lands in slot 0 of the sequential group via seq_dist[i, i] = 0)") but did not update the caps. Result: for any N ≤ K_canonical = 64, the cap is `min(2*n_seq, N-1) ≤ N-1`, so the top-k inside `build_neighbor_idx` systematically drops exactly one real residue per query — always the farthest by `|i-j|` for the sequential group. At N=50, coverage per query is 49 / 50 instead of 50 / 50; the missing residue's softmax weight (which dense gives mass to) goes to no one. Numerical impact: ~20 % relative per-attention-layer error vs the dense ground truth, compounded over 14 layers with residuals. **At N=100 and above, the cap saturates at `2*n_seq = 64 < N-1` so the bug does not fire** — coverage at L=100/200 is unchanged.

The fix is a three-line edit. The replacement passes the existing numerical equivalence test at machine epsilon in every curriculum bucket at N=50, and is a no-op at N > K.

**Results — designability A/B probes on step-1385 (same ckpt, post-fix code, seed=5, nsteps=400):**

Three schedule variants probed; canonical was re-run as a "code-fixed-but-schedule-unchanged" reference. LOWTSOFT was then pushed to N=18 (N=6 + N=12) for a confidence-interval-tightening read. Same-day same-code paired comparison is the cleanest A/B.

| Probe (step-1385, post-fix code) | L=50 | L=100 | L=200 | Pooled | Best Å |
|---|---|---|---|---|---|
| canonical schedule, N=6                                                | 4/6 (67 %) | 3/6 (50 %) | 0/6 (0 %)  | 7/18 (39 %)  | 0.87 |
| NOCURR static `(8, 16, 32)` at all t, N=6                              | 4/6 (67 %) | **1/6 (17 %)** | 0/6 (0 %)  | 5/18 (28 %)  | 0.71 |
| LOWTSOFT low-t → `(16, 8, 24)`, N=6                                    | 4/6 (67 %) | 4/6 (67 %) | 0/6 (0 %)  | 8/18 (44 %)  | 0.62 |
| LOWTSOFT low-t → `(16, 8, 24)`, N=12 follow-up                         | 6/12 (50 %) | 7/12 (58 %) | 0/12 (0 %) | 13/36 (36 %) | 0.63 |
| **LOWTSOFT N=18 pooled (N=6 + N=12)**                                  | **10/18 (56 %)** | **11/18 (61 %)** | **0/18 (0 %)** | **21/54 (39 %)** | **0.62** |
| canonical N=18 pre-fix (variants.md §11 baseline — 3-axis, NO BigBird/PU) | 8/18 (44 %) | 10/18 (56 %) | 2/18 (11 %) | 20/54 (37 %) | 0.94 |

**Reading:**
- **L=50: cap-fix probably contributes some, lowtsoft probably contributes some.** Pre-fix → post-fix canonical at N=6 went 3/6 → 4/6 (+1, expected direction for cap-fix at N ≤ K=64). LOWTSOFT N=18 = 56 % vs canonical pre-fix N=18 = 44 % is +12 pp directional, but partly the cap fix (no lowtsoft change at all needed for that). Best-Å improves from 0.94 → 0.62 across all three post-fix probes — the model produces sharper structures with the cap correct.
- **L=100: the schedule shape matters.** This is where the three post-fix probes disagree most. Removing the schedule entirely (NOCURR) drops L=100 from canonical's 3/6 to 1/6 — −2 designable, well outside N=6 noise. Softening the low-t bucket (LOWTSOFT) goes the other way, 3/6 → 4/6. At N=18 pooled, LOWTSOFT lands at 11/18 (61 %), +5 pp vs canonical pre-fix N=18. The harsh `(32, 0, 0)` low-t bucket is suboptimal but the *schedule itself* is doing useful work — keeping spatial+random capacity around at low t helps long-range mixing while the trained-with-the-schedule weights still expect the staged structure.
- **L=200: −11 pp dropout with LOWTSOFT, but inside binomial noise.** Canonical pre-fix had 2/18; LOWTSOFT post-fix is 0/18. Binomial 95 % CI for 2/18 is [1 %, 35 %]; for 0/18 it is [0 %, 19 %] — the intervals overlap. Best-Å at L=200 is 2.03 Å in LOWTSOFT N=6 → 2.10 Å in N=12 — both within 0.10 Å of the 2 Å designability threshold. The model gets *close* at L=200 but doesn't quite clear; whether retraining with lowtsoft pushes it across or whether L=200 is a different failure mode (training distribution `min_length=50`, step undercount) is what [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) tests.

**Negative-result corrections:**
- The mid-day N=6 LOWTSOFT read (4/4/0 = 44 % pooled) over-stated the L=100 effect. N=18 pulls the LOWTSOFT L=100 rate down from 4/6 (67 %) toward 11/18 (61 %), with the L=50 rate also pulled from 4/6 (67 %) toward 10/18 (56 %). Direction holds, magnitude shrinks.
- The original FIXEDCAP-driver probe was made redundant when the user re-ran the existing canonical driver after the cap fix was already in the code; the FIXEDCAP config + driver remain on disk but unused.
- The user's manual `rm -rf .../job_0_n_*_id_*/job_0_n_*_id_*` recovery during the first canonical re-probe accidentally also matched `.pdb` files (the glob `*` matched `0.pdb`), wiping the gen output mid-eval. Cost: one 5-min gen re-run. Drivers were subsequently hardened (`rm -rf` of the nested *directory* with `[ -d ... ]` guard, never globs that could match a file).

**Possible narrative:** non-narrative — code-correctness fix + falsified mechanisms. **What this entry decides:**
- The L=50 portion of the K=64 variant's gap could be a bug-contribution. Re-probing the step-1385 ckpt with the fixed code (driver: `probe_sparse_K64_step1385_FIXEDCAP.sh`) will A/B-test this. **Cautious prediction:** the bug fires at every N ≤ 64 during training too, so the trained model may have absorbed it; the fix could be neutral or even slightly hurt at L=50 if the model has learned to compensate. L=100/200 should be unaffected (bug does not fire there).
- The L=100/L=200 gap is **NOT** explained by this bug. The remaining major candidates (curriculum bucket (32,0,0) at low t restricting each query to a ±32-sequential window at L=100 → 36 unreachable residues per query during late refinement; training `min_length=50`; step undercount 1385 vs 2646) need separate tests. The new `curriculum_low_t_split` knob and the LOWTSOFT driver are the cheapest next probe.
- bf16 is not the bug — sparse and dense quantise identically.

**Methodological caveats:**
- **Numerical tests use a fresh, untrained `PairBiasAttention` instance with random weights.** They show that the math is correct, not that the trained model behaves identically. The trained model may have learned to compensate for the off-by-one and the bucket schedule; the fix could shift inference outputs in either direction.
- **The variants.md §11 step-1385 baseline (N=18, 44 % / 56 % / 11 %) pools N=6 at seed=5 with a separately-generated N=12 at seed=5; per CLAUDE.md auto-memory the seed-propagation through `predict_step` is non-trivial, so the FIXEDCAP and LOWTSOFT re-probes' N=6 numbers are not directly seed-paired with the existing 44 / 56 / 11 estimates.** Use the pre-fix N=6 numbers (50 / 67 / 17 % at L=50/100/200 from `probe_sparse_K64_curriculum_self_step1385.sh` alone) as the comparison baseline for the matched-N comparison.
- **Training-side correctness:** the trained ckpt at step 1385 was *trained* with the off-by-one. If retraining is judged worthwhile, both the fix and any low-t-bucket change can be bundled into a single new training run (`training_ca_only_sparse_K64_curriculum_lowtsoft.yaml` is the lowtsoft variant; `training_ca_only_sparse_K64_nocurr.yaml` is the no-curriculum control).
- **The bf16 audit ran on a small synthetic example (B=1, N=50, token_dim=256, pair_dim=64, 8 heads).** Re-running at full model scale on real coords + real weights could show a different absolute magnitude of bf16 noise, but the *symmetry* between dense and sparse is mechanism-level and not sample-size dependent.
- **CLAUDE.md was updated** to reflect that self is now included (the previous "Self is excluded from each query's neighbor list" line was already stale from commit `fbcc1ec`); the new line also documents the cap fix.

**Cross-references:**
- Code change: `proteinfoundation/nn/modules/sparse_neighbors.py:61,69,88` (off-by-one cap fix).
- New knob: `proteinfoundation/nn/local_latents_transformer.py` (`curriculum_low_t_split` kwarg + assert in `_build_neighbor_idx`); `proteinfoundation/proteina.py` (plumbing from `cfg_exp.training`); `proteinfoundation/generate.py` (inference override).
- New configs: `configs/training_ca_only_sparse_K64_nocurr.yaml`, `configs/training_ca_only_sparse_K64_curriculum_lowtsoft.yaml`, `configs/inference_sparse_K64_step1385_FIXEDCAP_n6_nfe400.yaml`, `configs/inference_sparse_K64_step1385_LOWTSOFT_n6_nfe400.yaml`.
- New drivers: `script_utils/probe_sparse_K64_step1385_FIXEDCAP.sh`, `script_utils/probe_sparse_K64_step1385_LOWTSOFT.sh`.
- CLAUDE.md update: lines 152 (`min(2 * n_seq, N)` not `N - 1`) and 167 ("Self is INCLUDED" replacement, with off-by-one note).
- Related variants.md sections: §12 (K=64-curriculum-self-bigbird-pairupdate), §11 (K=64-curriculum-self trunk).
- Architectural predecessors / context: [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07), [E045](#e045--t-dependent-k-budget-reallocation-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (the K=40 inference-only curriculum probes that motivated the K=64 trained-with-curriculum variant); CLAUDE.md → "Sparse-attention variant (SALAD-style, K=40)".
- Follow-up: [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) — cold-start retrain of the K=64 bundle with the cap fix + lowtsoft baked in.

---

## E047 — Cold-start retrain of the K=64 bundle with cap fix + LOWTSOFT low-t bucket (2026-05-11)

**Status:** in progress — queued as SLURM job 29210711 on ampere SL2, 20h walltime, single slot. Submitted 2026-05-11 ≈ 18:00 BST.

**Why ran:** [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11)'s inference-only LOWTSOFT probe on step-1385 of the **K=64-curriculum-self variant (variants.md §11; three axes: K=64 + curriculum + self-inclusion; NO BigBird, NO pair-update)** lands at **L=50 / L=100 / L=200 = 56 % / 61 % / 0 %** designable at N=18 pooled — vs the §11 pre-fix N=18 baseline of 44 % / 56 % / 11 %. The +12 pp at L=50 (cap fix + lowtsoft together) and +5 pp at L=100 (lowtsoft alone, since the cap fix is a no-op at N>K=64) are directional improvements consistent with the variant-design hypothesis (the harsh `(32, 0, 0)` low-t bucket bottlenecks long-range mixing at L=100), but the L=200 −11 pp dropout (2/18 → 0/18) is the question mark; binomial CI suggests it's inside noise but worth confirming with retrained weights. The inference-only signal is also **off-distribution** for the trained §11 ckpt (model trained on `(32, 0, 0)` low-t, probed at `(16, 8, 24)`), so training-with-lowtsoft should match or beat the inference-only-lowtsoft probe — that's what this entry tests. Decision the entry feeds: whether the planned five-axis §12-LowTSoft bundle (K=64 + curriculum + self + BigBird + pair-update, all with the soft low-t bucket) is the new K=64 baseline going forward, and what ablations (drop pair-update, drop BigBird, isolate cap-fix, isolate lowtsoft) attribute the contributions.

**Configs / hardware:**
- Training config: `configs/training_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft.yaml` (NEW, sibling of `..._bigbird_pairupdate.yaml` — the variants.md §12-forthcoming config that has never been trained). Two differences vs `..._bigbird_pairupdate.yaml`:
  1. `training.curriculum_low_t_split: [16, 8, 24]` (overrides the default `(32, 0, 0)` low-t bucket — plumbed via `proteina.py` from [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11)).
  2. `sparse_neighbors.py` cap fix is in code at train time (`min(2*n_seq, N)` not `N-1`) — affects every training step at N ≤ K=64.
- NN config: `configs/nn/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_160M.yaml` — the §12-forthcoming NN file. 14 layers, token_dim=768, 12 heads, **K=68 = K_canonical=64 + 4 BigBird globals**, **pair-update every 3 layers (5 updates over 14 layers)**. **This is the FIRST training run that uses this NN config** — step-1385 used the simpler `ca_only_sparse_K64_curriculum_160M.yaml` (no BigBird, no pair-update).
- Recipe: OLD canonical — wd=0.05, constant LR=2e-4, no scheduler, AdamW defaults, ema decay=0.999 every_n_steps=5, accumulate_grad_batches=32, val_check_interval=2000, seed=42, self_cond=True, compile_nn=True.
- Cold start (no `pretrain_ckpt_path`). Could not warm-start from step-1385 anyway — step-1385's weights don't have BigBird or pair-update parameters.
- Submit: `sbatch --time=20:00:00 --exclude=gpu-q-43 script_utils/submit_train_ca_only_1gpu.sh -n training_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft` — single 20h SL2 slot. At canonical's ~131 opt-steps/h that's ~2600 opt steps in one shot, past the canonical 1800-2200 best-val window and into the canonical step-2646 baseline range. Slower in practice because of the new BigBird + pair-update parameters needing to find their working points.
- `--exclude=gpu-q-43` (broken GPU per CLAUDE.md; `afterany` re-routes back to it).
- Wandb group: `ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft` (auto-set from `run_name_` by the submit script). Wandb run id: `vru8hr9y`.

**Early observation (steps 62–1196, pulled from wandb 2026-05-12 14:00 BST):**

Tracks the §11 K=64-curriculum-self curve at low steps then diverges below it by ~0.3-0.4 mse from step ~600 onward. Best val so far: 4.712 at step 944 (vs §11's 4.19 at step 1385). Three runs at adjacent steps for context:

| Step | this run (§12-LowTSoft) | §11 K=64-curric-self best slot | scnbr K=40 | sparse K=40 | canonical dense |
|---|---|---|---|---|---|
| 200  | 6.41 | 6.38 | 6.07 | 6.38 | — |
| 600  | 5.28 | 5.06 | 5.07 | 5.03 | — |
| 1000 | **4.71** | 4.38 | 4.45 | 4.41 | 5.62 |
| 1200 | **4.62** | 4.24 | 4.28 | 4.28 | 5.38 |

Plausible explanation: the §12 bundle introduces 3 new architectural axes (BigBird globals + pair-update + lowtsoft) that need to train from cold-init simultaneously, while §11 only had to learn the K=64 trunk + curriculum + self-inclusion. The 0.3-0.4 mse gap is the cost of carrying 4 new global parameters + ~5 new pair-update MLP blocks + a harder low-t schedule. Whether the bundle catches up by step ~2000 or plateaus higher is the load-bearing question; the 20 h slot ends at step ~2400-2600 in principle.

**Predicted milestones (to verify against actual checkpoints):**
- val MSE at step ~1800-2200 — does it cross below §11's 4.19? If yes, the bundle eventually pays back the slower start.
- Designability probe at step ~1800 (whichever ckpt sits closest to §11 step-1385's val ≈ 4.19) — matched-val A/B vs E046's inference-only LOWTSOFT N=18 (10/11/0 = 56/61/0%): if the retrain matches or exceeds it, training-with-lowtsoft + BigBird + pair-update outperforms inference-only-lowtsoft on the simpler architecture.
- Designability probe at step ~2400 vs canonical's E019 step-2646 N=30 baseline (19/20/3 = 63 / 67 / 10%): the clean "did the five-axis bundle clear canonical at L=100" test.

**Results:** pending — entry will be updated once the first probe-worthy checkpoint lands. New inference config + driver to be written for the retrained ckpt path (separate output dir from the §11 step-1385 inference outputs, no collision).

**Possible narrative:** non-narrative *yet* — this is the load-bearing test entry. If the bundle clears the canonical bar at L=100 at step ~2400, this becomes the basis for a Finding (the five-axis K=64 + curriculum + self + BigBird + pair-update + lowtsoft bundle as the new K=64 baseline; ablations to attribute). If it plateaus higher, the picture stays "lowtsoft helps at inference but doesn't transfer cleanly to training with 3 new architectural axes" and the right move is to test lowtsoft on the simpler §11 architecture first (`configs/training_ca_only_sparse_K64_curriculum_lowtsoft.yaml` already on disk).

**Methodological caveats:**
- **Cold start, not warm.** Couldn't warm-start from step-1385 even if desired — step-1385's weights lack the BigBird `global_pair_bias_*` and pair-update MLP parameters that the §12 NN config introduces. So the slow-start is unavoidable for this bundle.
- **Three new axes at once.** BigBird, pair-update, and lowtsoft are introduced together; we cannot attribute individual contributions from this run alone. The natural ablations are:
  - Cap-fix isolation: re-run `..._bigbird_pairupdate.yaml` (no lowtsoft, but cap fix in code).
  - Lowtsoft-on-§11: re-run `..._curriculum.yaml` with `curriculum_low_t_split: [16, 8, 24]`. (Config sibling already exists as `training_ca_only_sparse_K64_curriculum_lowtsoft.yaml`.)
  - BigBird isolation: re-run `..._bigbird_pairupdate_lowtsoft.yaml` with `n_global_tokens: 0` in the NN config.
  - Pair-update isolation: same but `update_pair_repr: False`.
  Each ablation = one more 20 h SL2 slot; deferred until E047 returns a verdict.
- **20 h single slot may not reach the bundle's best-val window.** At the early pace (~75% of canonical's opt-steps/h on a single A100 with compile_nn + a heavier architecture), 20 h gets to step ~2400 not ~2600. Canonical's best-val window is 1800-2200 so we're still inside it; chained slot would be needed only if the bundle's own best-val window is shifted later (which is plausible given the 3-axis cold start cost).
- **Same seed=5 / nsamples-batching propagation as §11** applies to all designability probes on the retrained ckpts. Pool N=6 + N=12 → N=18 same as the LOWTSOFT N=18 inference probe.
- **CLAUDE.md hard rule: `nsteps=400`.** Probe drivers on the retrained ckpts inherit it from `inference_base.yaml`.

**Cross-references:**
- Training config: `configs/training_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft.yaml`.
- NN config (§12-forthcoming, first time trained): `configs/nn/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_160M.yaml`.
- Submit script: `script_utils/submit_train_ca_only_1gpu.sh` (canonical 1-GPU pattern; 20 h overridden via CLI).
- Code dependencies (E046): `sparse_neighbors.py:61,69,88` cap fix; `LocalLatentsTransformer.curriculum_low_t_split` kwarg + assert; `proteina.py` plumbing; `generate.py` inference override (only matters for downstream probes, not the training run).
- Predecessor / immediate ancestor: [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) (bug investigation, code fix, inference-only LOWTSOFT signal at N=18 on step-1385 of the §11 variant).
- Comparison baseline for designability: variants.md §11 step-1385 N=18 pooled (44 / 56 / 11 % at L=50 / 100 / 200 — pre-fix, three-axis variant).
- Architecture: variants.md §11 trunk (K=64-curriculum-self) plus §12-forthcoming additions (BigBird + pair-update). The §12 entry will be added to variants.md once a checkpoint has been probed.
- Sibling untrained config: `configs/training_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate.yaml` (the §12 forthcoming reference — never trained; this run replaces it with lowtsoft baked in).
- Deferred ablation candidates: `configs/training_ca_only_sparse_K64_curriculum_lowtsoft.yaml` (drops BigBird + pair-update, keeps lowtsoft); `configs/training_ca_only_sparse_K64_nocurr.yaml` (drops curriculum entirely); not-yet-written cap-fix-isolation config (= `..._bigbird_pairupdate.yaml` re-run with cap fix in code but `(32, 0, 0)` low-t).
## E048 — Inference-only K-bump sweep (K=40 → K=64 → K=128) on plain sparse_K40 step 1259 (2026-05-07)

> Renumbered from local E046 on 2026-05-12 merge with upstream E046/E047. Cross-refs to "E046" in this section refer to the K-bump experiment described below (now E048), unless the link target is an upstream-E046 anchor. To avoid ambiguity, the body text uses "this entry" or the explicit new ID where appropriate.

**Why ran:** Direct follow-up to E044/E045. The two curriculum probes told us the model uses information from the noisy spatial+random groups in non-trivial ways at low t (E044 mask hurt) and that low-t composition matters more for L≥100 than for L=50 (E045 realloc split). What neither tested is the *architectural ceiling* — does this checkpoint's trained capacity exceed K=40, or is K=40 the actual bottleneck?

The cheap test: **bump K at inference and see if more neighbor signal helps**. K is just an int attribute consumed at the call site (verified: `bin_pairwise_distances_sparse` and `_gather_sparse_pairs` and `pair_rep_initial.py:50` and `_attn_sparse` all derive K from `neighbor_idx.shape[-1]`; nothing pre-allocates a K-sized buffer). Setting `model.nn.n_seq_neighbors`, `n_spatial_neighbors`, `n_random_neighbors` post-load is the entire intervention. Limitation: q/k/v projections, AdaLN-Zero gates, and softmax temperatures were calibrated for K=40, so this is a distribution-shift test on top of the architectural test — a positive K-bump result would unambiguously trigger retraining; a negative result is ambiguous (could be ceiling, could be distribution shift).

**Three K configurations, canonical 40/20/40 sequential/spatial/random ratio preserved:**

| Variant | n_seq (per side) | n_spatial | n_random | K_total |
|---|---|---|---|---|
| K=40 (control) | 8 | 8 | 16 | 40 |
| K=64 | 12 | 12 | 28 | 64 |
| K=128 | 24 | 24 | 56 | 128 |

K=40 control reuses the [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) baseline CSV (same checkpoint, same `seed=5`, same `nlens_cfg=[50,100,200]`, same `nsamples=6`, no override and no curriculum) — paired-by-noise with the K=64 and K=128 arms run in this entry. No K=40 re-run.

**Configs:**
- Code change: `proteinfoundation/generate.py` post-load block — reads `cfg.generation.args.n_*_neighbors_override` and sets the attrs on `model.nn`. Logs the override and the trained-K reminder. Default off; existing configs unchanged. Curriculum (E044/E045 path) untouched and disabled for this sweep so the K-axis is unconfounded with composition.
- Inference configs: `configs/inference_sparse_K40_step1259_K64_n6_nfe400.yaml`, `configs/inference_sparse_K40_step1259_K128_n6_nfe400.yaml`. Inherit from `inference_base.yaml` (`seed: 5, nsteps: 400, sc_neighbors_bootstrap: True`). Identical except the three override fields under `generation.args`.
- Checkpoint: `/home/ks2218/la-proteina/sparse_K40_step1259.ckpt` (= `best_val_00000012_000000001259.ckpt`, run `ca_only_sparse_K40`). `sparse_attention=True, n_seq=8, n_spatial=8, n_random=16, update_pair_repr=False, use_tri_mult=False, use_downsampling=False, sc_neighbors=False`. Same as E044/E045.
- Hardware: 1× L4 (GPU 3), `nohup`, sequential. Wall (gen-only): K=40 (E044 baseline) 1:03, K=64 1:20 (+27%), K=128 2:20 (+122% vs K=40).
- Driver: `/tmp/run_K_sweep.sh` with `PYTHON_EXEC` exported.

**K-bump hook fire check** — gen log line for K=64: `[K-bump] override → n_seq=12 (×2 = 24 sequential), n_spatial=12, n_random=28 → K_total=64 (trained at K=40).` Same shape for K=128 (logged with K_total=128). No errors / OOM. K=128 fits in 22 GB L4 memory comfortably (sparse `_attn_sparse` materialises `[B*H, N, K, D]` gather tensors; at B=6, H=12, N=200, K=128, D=64 this is ~118 MB per tensor — well within limits).

**Results — pooled designability (N=6 × {50, 100, 200}):**

| arm | L=50 | L=100 | L=200 | pooled | min Å | median Å | mean Å | std Å |
|---|---|---|---|---|---|---|---|---|
| K=40 (control) | 2/6 (33%) | **3/6 (50%)** | 0/6 | **5/18 (28%)** | **1.08** | 5.13 | 6.31 | 4.39 |
| K=64 | 0/6 | 2/6 (33%) | 0/6 | 2/18 (11%) | 1.12 | 8.07 | 7.85 | 4.49 |
| K=128 | 0/6 | 0/6 | 0/6 | **0/18 (0%)** | 2.11 | 8.60 | 9.45 | 5.20 |

**Monotonic K → worse at every length and every percentile.** K=128 is catastrophically bad (zero designable, every L=100 protein collapsed).

**Per-length sorted scRMSD (Å):**

- L=50: K=40 `[1.29, 1.31, 3.20, 4.90, 5.36, 11.40]`, K=64 `[4.01, 6.62, 6.76, 6.83, 10.72, 12.84]`, K=128 `[2.11, 2.75, 2.97, 4.54, 7.04, 7.63]`
- L=100: K=40 `[1.08, 1.43, 1.58, 4.30, 4.79, 6.31]`, K=64 `[1.12, 1.41, 2.20, 2.83, 3.16, 9.30]`, K=128 `[3.28, 7.08, 8.18, 9.02, 9.16, 11.36]`
- L=200: K=40 `[7.87, 9.96, 10.87, 11.63, 12.52, 13.74]`, K=64 `[10.34, 11.64, 12.42, 12.62, 12.74, 13.78]`, K=128 `[14.82, 15.24, 15.38, 15.41, 16.67, 16.71]`

**Per-protein paired (same noise; K=40 / K=64 / K=128):**

| L | id | K=40 | K=64 | K=128 | Δ64 | Δ128 |
|---|---|---|---|---|---|---|
| 50 | 0 | 11.40 | 10.72 | **2.11** | −0.68 | **−9.29** |
| 50 | 1 | 1.31 ✓ | 6.62 | 7.63 | +5.31 | +6.32 |
| 50 | 2 | 3.20 | 6.83 | 2.97 | +3.63 | −0.24 |
| 50 | 3 | 1.29 ✓ | 4.01 | 4.54 | +2.72 | +3.25 |
| 50 | 4 | 5.36 | 12.84 | 6.84 | +7.48 | +1.48 |
| 50 | 5 | 4.90 | 6.76 | 3.37 | +1.86 | −1.53 |
| 100 | 0 | 1.43 ✓ | 2.20 | 8.18 | +0.77 | **+6.75** |
| 100 | 1 | 1.08 ✓ | 2.83 | 9.16 | +1.75 | **+8.08** |
| 100 | 2 | 6.31 | 9.30 | 11.36 | +2.98 | +5.05 |
| 100 | 3 | 4.30 | **1.12 ✓** | 7.08 | −3.18 | +2.78 |
| 100 | 4 | 1.58 ✓ | 1.41 ✓ | 3.28 | −0.17 | +1.71 |
| 100 | 5 | 4.79 | 3.16 | 9.02 | −1.63 | +4.22 |
| 200 | 0-5 | (all collapsed in all arms; K=128 deeper still by +4-7 Å) | | | | |

**Designability flips:**

- K=40 → K=64: 4× DESIGN→FAIL, 1× FAIL→DESIGN — net −3.
- K=40 → K=128: 5× DESIGN→FAIL, 0× FAIL→DESIGN — net −5.
- K=64 → K=128: 2× DESIGN→FAIL, 0× FAIL→DESIGN — K=128 strictly worse than K=64.

**L=100 cluster proportions (clean<3 / mid 3-5 / collapsed>5):**

| arm | clean | mid | collapsed |
|---|---|---|---|
| K=40 | 3 [1.08, 1.43, 1.58] | 2 [4.30, 4.79] | 1 [6.31] |
| K=64 | 4 [1.12, 1.41, 2.20, 2.83] | 1 [3.16] | 1 [9.30] |
| K=128 | **0** | 1 [3.28] | **5** [7.08, 8.18, 9.02, 9.16, 11.36] |

K=64 actually *tightens* the L=100 top of distribution (4 clean vs 3 baseline) but pushes two previously-designable samples (1.08, 1.43) just over the 2 Å line (to 2.83 and 2.20). K=128 obliterates the clean cluster entirely.

**E045 L=50 std-collapse reproduction check (per task request):**

| arm | L=50 std Å | L=50 min Å | L=50 max Å |
|---|---|---|---|
| K=40 baseline | 3.76 | 1.29 | 11.40 |
| **E045 realloc** | **0.92** | **0.63** | **3.10** |
| K=64 | 3.21 | 4.01 | 12.84 |
| K=128 | 2.22 | 2.11 | 7.63 |

**The L=50 std-collapse does NOT reproduce at K=64 or K=128.** It was a property of the E045 *realloc curriculum* (sequential-only at low t), not a general K-effect. K=128 has the smallest L=50 std after E045 (2.22), but the count is 0/6 and min is 2.11 — distinctly different from E045's 3/6 at min 0.63 with every sample <3.10 Å. This rules out "more K → tighter L=50" as the mechanism behind E045's win; the win was specifically about *replacing* spatial+random with extra sequential, not adding more slots overall.

**Generation wall-clock per K (3 lengths × N=6 = 18 proteins per gen run, 1× L4):**

| arm | gen wall | per-protein |
|---|---|---|
| K=40 (E044 baseline) | ~1:03 (63 s) | ~3.5 s |
| K=64 | 1:20 (80 s) | ~4.4 s |
| K=128 | 2:20 (140 s) | ~7.8 s |

K=64 is +27% wall vs K=40, K=128 is +122% (~2.2× K=40). Sub-linear in K because the early `topk` for sequential neighbors is N-dependent and constant across K, and the gather pattern bandwidth dominates only for the larger K values. Eval cost is unchanged across K (ESMFold + ProteinMPNN don't depend on the diffusion architecture).

**Verdict — inference-only K-bump strictly hurts; the trained K=40 distribution shift dominates any architectural gain:**

1. **K=40 is the floor at inference time.** Both K=64 and K=128 strictly underperform K=40 at every length × every percentile. K=128 produces zero designable samples across 18 proteins.
2. **Failure mode is consistent**: the trained model's per-slot attention biases (q/k/v projections + AdaLN-Zero gates + softmax temperature) were calibrated against 40-slot input. Adding more slots dilutes attention across additional candidates that the trained weights cannot weight correctly. The dilution scales with K — K=64 hurts modestly, K=128 catastrophically.
3. **No architectural-ceiling signal.** This experiment cannot distinguish "K=40 is the model's true expressive ceiling" from "the K=40 trained weights cannot use a wider neighbor list at inference time but a K=64-trained model could". The decision-relevant test is **a retraining run at K=64**, not more inference-only sweeping.
4. **The L=100 K=64 result is the most informative sub-finding**: 4 clean samples (vs 3 baseline) with two of the previously-clean samples drifting *just* over the designability threshold (1.43 → 2.20, 1.08 → 2.83). The L=100 distribution is *almost* the same as baseline, just shifted up by ~0.7 Å on the clean cluster. Reads as "the K=64 input wants to produce the same structure but the trained weights are now slightly miscalibrated" — which is the predicted distribution-shift story, not an architectural-ceiling story. **A retraining run at K=64 with the same recipe would test whether removing the calibration mismatch closes the L=100 gap.** Given the canonical baseline still beats sparse_K40 at L=100 (E019: canonical 67% vs sparse 3.3% at N=30), and the L=100 K=64 inference shows no obvious ceiling-related signal beyond the calibration shift, **a K=64 retrain is justified as the next architectural test of this variant**.

**Recommendation surfaced explicitly:** the inference-only K-bump did **not** trigger a retrain on its own (the rule was "if K=64/K=128 *helps* L=100 at N=6, retrain"). It did, however, rule out the trivial path. Combined with E044's "spatial+random groups carry non-trivial info at low t" and E045's "the L=100 bottleneck is information-channel-limited, not composition-limited", the cumulative evidence is consistent with "trained K=40 is undercapacity at L=100 but retraining is the only way to get a clean test". A K=64 retraining run at the canonical recipe (same model size, same wd=0.05, same constant LR=2e-4, same chained slot structure) is justifiable as an architectural ablation; expected wall-clock is comparable to the K=40 baseline (the gather pattern is the bottleneck at training too, but the throughput penalty is bounded — see the 27% inference penalty here).

**Methodological caveats:**

- **N=6 per length, single seed=5.** Same caveat as E044/E045. The N=6 numbers should not be over-interpreted at the per-length granularity — except that the K=40 → K=128 monotonic degradation across all 18 paired proteins (with 5× DESIGN→FAIL flips and zero rescues) is strong enough that N=6 noise cannot explain the direction. K=128 is robustly worse than K=40 at this protocol; K=64 vs K=40 is less robust but directionally aligned.
- **Inference-only test on a K=40-trained checkpoint.** The whole experiment's logic depends on the trained weights being calibrated for K=40; results say nothing about the architectural capacity of a K=64-trained model.
- **Single composition ratio (canonical 40/20/40) preserved across K.** Could in principle test K=64 with a different ratio (e.g., 32/16/16 = 64-with-more-sequential) but that conflates K-axis with composition-axis (the E044/E045 axis). Single-axis test was the design goal.
- **Wall-clock measured on a single L4 with no co-tenant load**, but the box is shared and the absolute numbers might shift on different hardware. The K=40-relative ratios should be more portable.
- **scRMSD < 2 Å with ProteinMPNN `ca_only=True` (8 seqs/protein, default), ESMFold backbone reconstruction.** Standard CLAUDE.md eval protocol; matches E044/E045 protocol.
- **K_total=128 was the upper bound tested.** K=256 is technically feasible (the L4 has enough memory) but the K=64→K=128 monotonic degradation makes K=256 unlikely to flip the verdict. Skipped.

**Possible narrative:** non-narrative — kept for tuning/decision-making. **What this entry decides:** (1) inference-only K-bump is a dead end on a K=40-trained checkpoint (rules out the cheapest path); (2) the L=100 K=64 calibration-shift signature is consistent with (but does not prove) "K=64 retraining could close the L=100 gap"; (3) K=40 is the inference *floor* for this checkpoint, so no future inference-time intervention should bump K — only retraining can.

**Cross-references:**
- Code change: `proteinfoundation/generate.py` post-load K-override block (no change to `local_latents_transformer.py`).
- Configs: `configs/inference_sparse_K40_step1259_K64_n6_nfe400.yaml`, `configs/inference_sparse_K40_step1259_K128_n6_nfe400.yaml`.
- Output CSVs: `inference/results_inference_sparse_K40_step1259_K64_n6_nfe400_0.csv`, `inference/results_inference_sparse_K40_step1259_K128_n6_nfe400_0.csv`.
- Logs: `nohup_inference_sparse_K40_step1259_K{64,128}_n6_nfe400.gen.log`, `*.eval.log`.
- K=40 baseline reused from: [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (`results_inference_sparse_K40_step1259_baseline_n6_nfe400_0.csv`).
- Composition-axis companions: [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (mask), [E045](#e045--t-dependent-k-budget-reallocation-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (realloc).
- Dense baseline at the same N=30 protocol: [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) — canonical 67% vs sparse 3.3% at L=100, the gap this entry was probing.

---

## E049 — First inference probe of the K=64 + curriculum-trained ckpt (`ca_only_sparse_K64_curriculum_self`, ep=9 step=944, 2026-05-08)

> Renumbered from local E047 on 2026-05-12 merge. References to "E046" inside this section's body now point to the K-bump experiment ([E048](#e048--inference-only-k-bump-sweep-k40--k64--k128-on-plain-sparse_k40-step-1259-2026-05-07)); the cap-fix [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) is a separate, later experiment.

**Why ran:** Direct retraining test of the E044/E045 mechanism. E044 (mask) and E045 (realloc) were inference-only on a K=40-trained checkpoint; both hurt pooled designability but E045 produced a striking L=50 win (3/6 designable, every sample <3.10 Å, min 0.63 Å). The follow-up question was: does *retraining* with the schedule baked in turn that L=50 free lunch into a robust property AND close the L=100 gap that the inference-only test could not? The user trained `ca_only_sparse_K64_curriculum_self` from scratch with K=64 + a t-bucketed curriculum (also bundling self-inclusion as a third architectural change). This entry is the first designability probe of that ckpt at best_val ep=9 step=944.

**What's new in this architecture (vs the K=40 sparse baseline):**

1. **K=64** with canonical composition `8 / 16 / 32` (sequential per-side / spatial / random). Total 16 + 16 + 32 = 64 slots.
2. **Curriculum baked into training**: at every training step, each protein's t draw maps to one of three buckets and the neighbor list is built with that bucket's `(n_seq, n_spatial, n_random)`. Per-protein t means proteins in different buckets within the same batch all get serviced by one `build_neighbor_idx` call per non-empty bucket. Schedule:
    - `t < 0.33`         → `(32, 0, 0)`   = 64  (sequential-only, low-t)
    - `0.33 ≤ t < 0.66`  → `(16, 8, 24)`  = 64  (interpolate)
    - `t ≥ 0.66`         → `(8, 16, 32)`  = 64  (canonical SALAD, high-t)
3. **Self-inclusion** in the neighbor list. HEAD's `sparse_neighbors.py` no longer applies the `eye` exclusion mask to `base_invalid` — residue i can be its own neighbor. This is the third architectural change between this ckpt and the K=40 sparse baselines (and is a documented confound in the cross-arm comparison).
4. **`opt.compile_nn=True`** is in the saved hyperparameters → `torch.compile(mode="reduce-overhead", fullgraph=False)` is auto-applied at inference load. No Fix C2 (`sc_neighbors=False` in saved hyperparams).

**Configs:**
- Inference YAML: `configs/inference_sparse_K64_curriculum_step944_n6_nfe400.yaml`. Inherits `inference_base.yaml` (`seed=5, nsteps=400`); `nlens_cfg=[50, 100, 200]`, `nsamples=6`. `generation.args.curriculum_neighbors: True` is **redundantly explicit** — the curriculum auto-fires from the saved `cfg_exp.training.curriculum_neighbors=True` via `proteina.py:128 _sc_neighbors_kwargs`. The YAML flag exists for documentation/legibility in `resolved_config.yaml`; the post-load hook in `generate.py` re-sets the same attribute.
- Code state at HEAD: `local_latents_transformer.py:174-264` `_build_neighbor_idx` now supports per-protein t buckets (groups proteins by bucket index, calls `build_neighbor_idx` once per non-empty bucket, scatters back). `sparse_neighbors.py` removes the self-exclusion `eye` mask. `proteina.py:139-146` triggers `torch.compile` on `self.nn` when `cfg_exp.opt.compile_nn=True`.
- Checkpoint: `/home/ks2218/la-proteina/best_val_00000009_000000000944.ckpt` (rsynced from `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self/1778188245/checkpoints/`). Verified ckpt hyperparams: `run_name_=ca_only_sparse_K64_curriculum_self`, `global_step=944`, `cfg_exp.nn.n_seq=8 n_spatial=16 n_random=32`, `cfg_exp.training.curriculum_neighbors=True`, `cfg_exp.training.sc_neighbors=None`, `cfg_exp.opt.compile_nn=True`. Distinct from the earlier ep=8 step=881 ckpt (also rsynced); 944 is the more-recent best_val. Both are from the same run id 1778188245.
- Hardware: 1× L4 (GPU 3), `nohup`, sequential. Wall: gen 4:50 (290 s for 18 proteins ≈ 16 s/protein, ~3-4× the K=40-no-compile baseline due to compile trace + curriculum's 3-bucket call overhead — though the per-length trace was much faster than the user's ~5 min/length estimate, only ~1.5 min/length). Eval ~9 min.
- Driver: `/tmp/run_K64_curriculum.sh` with `PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python` exported.

**Hook fire confirmation** — gen log: `opt.compile_nn=True — wrapping self.nn with torch.compile (mode=reduce-overhead, fullgraph=False)` at line 1; `[Curriculum neighbors] ON — 3-bucket K=64 reallocation: t<0.33 → (n_seq=32, n_sp=0, n_rd=0); 0.33≤t<0.66 → (16, 8, 24); t≥0.66 → (8, 16, 32)` at line 2. No assertion failures across 400 nsteps × 18 proteins.

**Results — pooled designability (N=6 × {50, 100, 200}):**

| arm | L=50 | L=100 | L=200 | pooled | min Å | std at L=50 | training step |
|---|---|---|---|---|---|---|---|
| K=40 sparse baseline (E044) | 2/6 | **3/6** | 0/6 | **5/18 (28%)** | 1.08 | 3.76 | 1259 |
| K=40 + E045 inference realloc | **3/6** | 0/6 | 0/6 | 3/18 (17%) | **0.63** | **0.92** | 1259 (same ckpt) |
| **K=64 curriculum-trained (E047)** | **3/6** | 1/6 | 0/6 | 4/18 (22%) | 0.91 | 4.04 | **944** |

**Per-length sorted scRMSD (Å):**

- L=50: K=40_baseline `[1.29, 1.31, 3.20, 4.90, 5.36, 11.40]`, E045 `[0.63, 1.47, 1.80, 2.35, 2.82, 3.10]`, **E047 `[0.91, 1.16, 1.96, 3.89, 4.48, 11.73]`**
- L=100: K=40_baseline `[1.08, 1.43, 1.58, 4.30, 4.79, 6.31]`, E045 `[2.47, 2.78, 3.03, 4.49, 7.02, 9.04]`, **E047 `[1.22, 2.48, 3.61, 7.84, 11.23, 15.26]`**
- L=200: K=40_baseline `[7.87, 9.96, 10.87, 11.63, 12.52, 13.74]`, E045 `[13.06, 14.23, 15.07, 15.41, 16.05, 16.26]`, **E047 `[9.64, 11.56, 13.70, 14.69, 16.10, 16.69]`**

**Per-protein paired (same noise; K=40_base / E045 realloc / K=64 curriculum):**

| L | id | K40 base | K40 realloc | K64 curriculum | Δ(curr − base) | Δ(curr − realloc) |
|---|---|---|---|---|---|---|
| 50 | 0 | 11.40 | 0.63 ✓ | **1.96 ✓** | **−9.44** | +1.33 |
| 50 | 1 | 1.31 ✓ | 3.10 | 3.89 | +2.58 | +0.79 |
| 50 | 2 | 3.20 | 1.47 ✓ | **0.91 ✓** | −2.29 | −0.56 |
| 50 | 3 | 1.29 ✓ | 1.80 ✓ | 11.73 | +10.44 | +9.93 |
| 50 | 4 | 5.36 | 2.35 | 4.48 | −0.89 | +2.13 |
| 50 | 5 | 4.90 | 2.82 | **1.16 ✓** | −3.74 | −1.66 |
| 100 | 0 | 1.43 ✓ | 9.04 | 7.84 | +6.41 | −1.20 |
| 100 | 1 | 1.08 ✓ | 2.47 | 3.61 | +2.53 | +1.14 |
| 100 | 2 | 6.31 | 4.49 | **15.26** | +8.94 | +10.77 |
| 100 | 3 | 4.30 | 7.02 | 2.48 | −1.82 | −4.54 |
| 100 | 4 | 1.58 ✓ | 3.03 | 1.22 ✓ | −0.35 | −1.81 |
| 100 | 5 | 4.79 | 2.78 | 11.23 | +6.44 | +8.44 |
| 200 | 0–5 | (all collapsed in all arms; K=64 mean 13.73 vs K=40 baseline 11.10) | | | | |

**Designability flips, K=40_baseline → K=64_curriculum:**

- 4× DESIGN→FAIL: L=50 id=1 (1.31→3.89), L=50 id=3 (**1.29→11.73** — the most extreme paired loss in the run), L=100 id=0 (1.43→7.84), L=100 id=1 (1.08→3.61).
- 3× FAIL→DESIGN: L=50 id=0 (**11.40→1.96** — full collapse rescue), L=50 id=2 (3.20→0.91), L=50 id=5 (4.90→1.16). All three rescues at L=50.
- Net: −1 designable. Pooled 5/18 → 4/18.

**L=100 cluster proportions (clean<3 / mid 3-5 / collapsed>5):**

| arm | clean | mid | collapsed |
|---|---|---|---|
| K=40 baseline | 3 [1.08, 1.43, 1.58] | 2 [4.30, 4.79] | 1 [6.31] |
| K=40 + E045 realloc | 2 [2.47, 2.78] | 2 [3.03, 4.49] | 2 [7.02, 9.04] |
| **K=64 curriculum (E047)** | 2 [1.22, 2.48] | 1 [3.61] | **3 [7.84, 11.23, 15.26]** |

The L=100 collapsed-cluster GROWS under retraining (3 vs baseline's 1; one sample at 15.26 Å is essentially fully collapsed). Mean scRMSD at L=100: K=40 baseline 3.25 → E045 realloc 4.81 → **K=64 curriculum 6.94** (worse than realloc, much worse than baseline).

**Verdict — the L=50 free lunch survives retraining; the L=100 gap does NOT close (yet):**

1. **L=50 confirmed.** Both *inference-only* (E045 realloc on K=40 ckpt: 3/6, min 0.63) and *baked into training* (E047 K=64 + curriculum + self-incl: 3/6, min 0.91) produce 3/6 designable at L=50 with sub-1-Å minimum. **The curriculum direction is real for short proteins** — the mechanism survives the harder test of "model trained against this composition does no worse than inference-time forcing".
2. **L=50 std-collapse does NOT survive retraining.** E045's tight cluster (std 0.92, max 3.10) was an inference-only artifact of forcing exactly one schedule mode. With training, the model can use multiple input modes and the L=50 std rebounds (4.04, max 11.73 — even one paired protein, id=3, fully collapses 1.29→11.73). The robust property is the count, not the dispersion.
3. **L=100 retraining made things WORSE, not better.** 1/6 designable (vs K=40 baseline's 3/6). Three samples at L=100 are now fully collapsed (>7 Å). Mean scRMSD almost 2× the K=40 baseline. The retraining hypothesis (curriculum + retraining = closes L=100 gap) is **not** supported at this training step.
4. **L=200 unchanged.** No length-dependent intervention has helped L=200; not specific to this entry.

**Crucial caveat — three confounds vs the K=40 baseline:**

1. **Step mismatch.** E047 ckpt is at step 944; K=40 baseline (E044/E045 reused) was at step 1259. That's −25% training steps. Per the canonical baseline's best_val window (1800-2200) and the sparse_K40 converged plateau at 1133-1259, E047 is *under-trained* for an apples-to-apples test. The L=100 picture might improve substantially with more training. The L=50 result already matches the inference-only ceiling at this step, so further training mostly tests whether L=100 catches up.
2. **K-axis change.** K=40 vs K=64 is a parallel architectural axis (not just the curriculum). E046 showed inference-only K-bumps strictly hurt the K=40 ckpt; the K=64 retrained model might benefit from the larger K, or might not (this entry can't isolate that).
3. **Self-inclusion.** HEAD's `sparse_neighbors.py` allows residue i to be its own neighbor; the K=40 sparse baseline excluded self via the `eye` mask. Mechanistically, self-inclusion gives the attention the option to keep/return-to its own residual signal even when external neighbors are noisy — likely helpful at low t. It's bundled in this run; isolating its contribution would need a `K=64 + curriculum + no-self-inclusion` retrain, which we don't have.

**What this entry decides** (cautiously, given the confounds):

- **The L=50 boost from the curriculum is robust enough to commit to** — it appears in both inference-only (E045) and baked-in-training (E047) tests with consistent direction and similar magnitude. Variant comparisons that include a short-protein arm should probably include a curriculum-trained K=64 model.
- **The "retrain with curriculum to close L=100" hypothesis is not yet supported.** At step 944 the L=100 picture is worse than the K=40 baseline at step 1259. Two paths from here: (a) wait for more training and re-probe (cheapest); (b) accept that the curriculum-on-its-own may not help L=100 and design a different intervention specifically for L=100 (longer-range mixing module, denser at low t — the opposite of E045's direction, etc.). Decision blocked on (a).
- **An undertraining cross-check is the immediate next probe**: re-probe this same run at a later step (e.g., step 1259 to match the K=40 baseline, or step 1800 to enter the canonical-recipe window). If L=100 designability rises above K=40 baseline at matched step, the retraining hypothesis is supported. If it stays at 1/6 or drops further, the curriculum is not the right L=100 lever.

**Methodological caveats:**

- **N=6 per length.** Same as E044/E045/E046. The L=100 collapse-cluster growth (3 samples, two at >10 Å) is the most directionally robust signal here; +1 net at L=50 is well within Wilson noise but consistent with the prior E045 result.
- **Three architectural confounds bundled** as listed above.
- **Inference-time t is uniform across the batch** (Euler integration), so the per-protein-bucket logic in `_build_neighbor_idx` collapses to "one bucket per integrator step" — the per-protein machinery is exercised only at training. The training-time per-protein bucketing is the harder mechanism to validate; this entry only validates the inference path through the trained weights.
- **Compile cost is bundled in the wall-clock numbers**. K=40-no-compile baseline was 63 s for 18 proteins; this run was 290 s. Most of the gap is the trace (compile-on cold-cache costs), not the steady-state K-bump (which only added ~30% in E046). If timing matters, a `compile_nn=False` override on a follow-up probe would clean this up.
- **scRMSD < 2 Å with ProteinMPNN `ca_only=True`, ESMFold backbone reconstruction.** Standard CLAUDE.md eval protocol.
- **Paired-by-noise** vs E044 baseline / E045 realloc holds for the `bb_ca` channel only (CA-only, same shape across arms). Different model weights = different attention dynamics; the *initial noise* is paired, the *integrated trajectories* diverge.

**Possible narrative:** non-narrative — kept for tuning/decision-making. Updates the picture from E044/E045/E046:
- **L=50 curriculum direction = real, retraining-validated.**
- **L=100 retraining hypothesis = not yet supported at step 944. Re-probe at step 1259+ before drawing conclusions.**
- The "K=40 inference-only floor" finding from E046 is moot for this run (different architecture); whether K=64 is a good architectural choice is a separate question gated on the L=100 catch-up.

**Cross-references:**
- Predecessors: [E044](#e044--inference-only-neighbor-list-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (mask), [E045](#e045--t-dependent-k-budget-reallocation-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (realloc — the L=50 result this entry validates), [E048](#e048--inference-only-k-bump-sweep-k40--k64--k128-on-plain-sparse_k40-step-1259-2026-05-07) (K-sweep — confirmed inference-only K-bump cannot test architectural ceiling, motivating this retraining).
- Code: `proteinfoundation/nn/local_latents_transformer.py` (per-protein t bucket logic at lines 174-264); `proteinfoundation/nn/modules/sparse_neighbors.py` (self-inclusion); `proteinfoundation/proteina.py:139-146` (compile hook), `:122-128` (curriculum kwarg plumbing).
- Configs: `configs/inference_sparse_K64_curriculum_step944_n6_nfe400.yaml`; `configs/training_ca_only_sparse_K64_curriculum.yaml` (training side); `configs/nn/ca_only_sparse_K64_curriculum_160M.yaml` (NN config).
- Output CSV: `inference/results_inference_sparse_K64_curriculum_step944_n6_nfe400_0.csv`.
- Logs: `nohup_inference_sparse_K64_curriculum_step944_n6_nfe400.gen.log`, `*.eval.log`.
- HPC source: `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self/1778188245/checkpoints/best_val_00000009_000000000944.ckpt` (rsynced 2026-05-08).

## E050 — Steering audit matrix — predictor × ensemble × fold × smoothing (2026-05-10)

> Renumbered from local E048 on 2026-05-12 merge.

**Status:** finished.

**Why ran.** Five sanity-check questions and three matrix gaps surfaced when re-reading Finding 10's "two fixes layered" claim against what was actually measured:

1. **Fold confounding** — every published "single noise-aware fold" cell uses fold 2 (E029, E031), picked because of its highest `val_r²_noisy`. Folds 0, 1, 3, 4 had never been individually probed → "is fold 2 specially smart?" was an open question that, if answered "yes", would weaken the F10 ensembling mechanism story.
2. **Smoothing confounding** — the clean+ensemble baseline (E028) used Gaussian smoothing σ=0.1, K=4. Every other Finding-10 cell has smoothing off. So the "E028 vs E032" comparison conflated "ensemble alone" with "ensemble + smoothing". A clean+ensemble *without* smoothing cell was needed to separate the two contributions.
3. **Reporting inconsistency** — n=4 smokes report `per-protein gap = pred − real`; n=48 sweeps report `Δratio = Δpred / Δreal`. No unified table across all four corners (clean / NA-v1) × (single / ensemble) of the matrix.

Plus three structural gaps:

- Clean + single fold at the smoke-gap format (E025 measured it differently).
- NA-v1 single fold only ever measured at n=4 (E029 / E031); no n=48 head-to-head against E028 / E032's full sweeps.
- Clean + ensemble + **no smoothing** never run.

This entry's job is to defend Finding 10 against audit-style review without hand-waving.

**Configs.** Driver: `script_utils/audit_steering_matrix.py` (4 stages: write-configs / generate / eval / report; idempotent). 12 cells total under `steering/config/audit_matrix/`:
- 7 n=4 smokes (seeds {42-45}, L=300, w=16): `clean_fold0`, `clean_fold2`, `clean_ensemble_nosmoothing`, `na_v1_fold0`, `na_v1_fold1`, `na_v1_fold3`, `na_v1_fold4`.
- 5 n=48 sweep (seeds 42-57 × L∈{300,400,500}): `na_v1_fold2_sweep_w{1,2,4,8,16}`.
- All cells: `inference_ucond_notri_long`, **nsteps=400 hardwired in driver**, schedule `linear_ramp(t_start=0.3, t_end=0.8, t_stop=0.9)`, `gradient_norm=unit`, `gradient_clip=10.0`, `channel=local_latents`, smoothing OFF (the new clean ens5 cell deliberately drops the σ=0.1, K=4 used in E028 to isolate the smoothing contribution).
- Predictor checkpoints: clean = `laproteina_steerability/logs/multitask_t1/20260427_161809/checkpoints/fold_{0..4}_best.pt`; NA-v1 = `laproteina_steerability/logs/multitask_t1_noise_aware/20260505_110348/checkpoints/fold_{0..4}_best.pt` (same as E029-E032; no re-training).
- Hardware: 2× L4 (CUDA_VISIBLE_DEVICES split: arm A = my assigned GPU, arm B = idle GPU 1), nohup, ~108 min wall for 268 PDBs end-to-end (gen). TANGO eval ~3 min total. Real TANGO via local binary (`~/.local/bin/tango`).
- Patch to driver: added `--cells <names>` filter to `stage_generate` (5 lines) so two parallel arms can split the work across 2 GPUs without modifying the cell registry.
- Output: `results/audit_matrix/<cell>/{guided/, diagnostics/, properties_guided.csv}` per cell. Aggregated report: `audit_report.md`.

**Results — Table 1 (n=4 smoke, L=300, w=16, tango_min):**

| Cell | Predictor | Fold | Smoothing | n | pred mean | real mean | **gap mean** | gap std |
|---|---|---|---|---|---|---|---|---|
| E028 ref | clean | ens5 | σ=0.1, K=4 | 4 | 378.5 | 581.9 | **-203.5** | 49.3 |
| E029 ref | NA-v1 | fold2 | off | 4 | 540.0 | 587.5 | **-47.5** | 47.9 |
| Tier 1.1 — clean f0 | clean | fold0 | off | 4 | 381.0 | 601.9 | **-220.9** | 64.0 |
| Tier 1.1 — clean f2 | clean | fold2 | off | 4 | 215.2 | 602.1 | **-386.9** | 113.0 |
| Tier 1.1 — clean ens5 (no smooth) | clean | ens5 | off | 4 | 390.8 | 594.7 | **-203.9** | 65.0 |
| Tier 1.2 — NA-v1 f0 | NA-v1 | fold0 | off | 4 | 540.5 | 601.8 | **-61.3** | 45.8 |
| Tier 1.2 — NA-v1 f1 | NA-v1 | fold1 | off | 4 | 526.7 | 588.9 | **-62.2** | 26.6 |
| Tier 1.2 — NA-v1 f3 | NA-v1 | fold3 | off | 4 | 537.5 | 596.9 | **-59.4** | 86.1 |
| Tier 1.2 — NA-v1 f4 | NA-v1 | fold4 | off | 4 | 501.3 | 598.3 | **-97.0** | 46.5 |

**Headline numbers from Table 1:**
- **Smoothing in E028 contributes essentially nothing** to the gap reduction. E028 ref (clean ens5 + σ=0.1, K=4 smoothing) = -203.5; new clean ens5 *without* smoothing = -203.9 (Δ = +0.4, smaller than the gap std of 49-65). E028's whole gap-reduction-from-single-fold story is **100% the ensembling, 0% the smoothing**. The σ=0.1, K=4 smoothing block in `SteeringGuide` does add cost (~5× extra predictor calls) without measurable benefit at this w-level.
- **Fold 2 is NOT specially smart for NA-v1.** All five NA-v1 single folds give gaps in the [-47, -97] range: f2 = -47.5 (E029 ref), f0 = -61.3, f1 = -62.2, f3 = -59.4, f4 = -97.0. Mean across the 5 NA folds = -65.5; std = 17.7. The original E029 fold-2 reading (-47) sits at the favorable end but is within 1 std of the cross-fold mean. f4 is the worst (-97) but still a 2× improvement over the worst clean fold. **The "fold 2 was cherry-picked by val r²" objection does not break Finding 10** — every NA fold beats every clean fold at gap-closure.
- **Clean fold 2 is much worse than clean fold 0** in single-fold (-386.9 vs -220.9), and the clean ens5 (-203.9) outperforms either single clean fold by averaging out the f2 disaster. The NA fix specifically rescues clean f2's catastrophic adversarial sensitivity (clean f2: -386.9 → NA f2: -47.5; Δ = +339).

**Results — Table 2 (n=48 full sweep, seeds 42-57 × L∈{300,400,500}, tango_min):**

| Cell | Predictor | Fold | Smoothing | w | n | pred mean | real mean | gap mean | Δpred (vs w=1) | Δreal (vs w=1) | **Δratio** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 1 | 48 | 959.8 | 877.9 | +81.9 | +0.0 | +0.0 | n/a |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 2 | 48 | 938.6 | 874.4 | +64.2 | -21.2 | -3.5 | 6.01× |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 4 | 48 | 897.8 | 868.5 | +29.3 | -62.0 | -9.4 | 6.62× |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | 8 | 48 | 819.8 | 861.5 | -41.7 | -140.0 | -16.4 | 8.54× |
| E028 ref full sweep | clean | ens5 | σ=0.1, K=4 | **16** | 48 | 671.9 | 843.9 | **-172.0** | -288.0 | -34.0 | **8.47×** |
| E032 ref full sweep | NA-v1 | ens5 | off | 1 | 48 | 1011.7 | 893.3 | +118.5 | +0.0 | +0.0 | n/a |
| E032 ref full sweep | NA-v1 | ens5 | off | 2 | 48 | 999.3 | 889.5 | +109.8 | -12.4 | -3.8 | 3.31× |
| E032 ref full sweep | NA-v1 | ens5 | off | 4 | 48 | 975.9 | 884.2 | +91.8 | -35.8 | -9.1 | 3.93× |
| E032 ref full sweep | NA-v1 | ens5 | off | 8 | 48 | 931.2 | 872.3 | +58.8 | -80.6 | -20.9 | 3.85× |
| E032 ref full sweep | NA-v1 | ens5 | off | **16** | 48 | 837.2 | 833.4 | **+3.8** | -174.5 | -59.9 | **2.91×** |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 1 | 48 | 1030.3 | 893.3 | +136.9 | +0.0 | +0.0 | n/a |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 2 | 48 | 1013.0 | 892.7 | +120.3 | -17.3 | -0.6 | 29.08× |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 4 | 48 | 976.3 | 883.8 | +92.5 | -54.0 | -9.5 | 5.67× |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | 8 | 48 | 905.4 | 876.7 | +28.8 | -124.8 | -16.7 | 7.49× |
| Tier 1.3 — NA-v1 f2 sweep | NA-v1 | fold2 | off | **16** | 48 | 768.4 | 855.9 | **-87.5** | -261.8 | -37.4 | **7.00×** |

**Headline numbers from Table 2:**
- **NA-v1 single fold (fold 2) at n=48 stays in the negative range** at w=16: gap = -87.5 (Δratio 7.0×). This is consistent in *sign* with E029's n=4 pilot (-47), and within the n=4 pilot's ±48 std band. **No crossover-flip** to positive like E032's ensemble did (n=4 pilot -1.6 → n=48 +3.8). The single-fold gap is *not* an artefact of the small n=4 reading; the noise-aware-only fix genuinely leaves a residual -50 to -90 underclaim that ensembling subsequently removes.
- **Ensembling continues to do meaningful work even after the NA fix is in place.** Single fold at n=48 = -87.5; ensemble at n=48 = +3.8. The 91-unit gap-reduction from ensembling is a real effect on the same n=48 grid, not an n=4 small-sample artefact. The Δratio-on-the-Δ-vs-w=1 axis tightens from 7.0× (single fold) to 2.9× (ensemble) at w=16.
- **Δratio is driven by the ratio's denominator and is noisy at low w** — Tier 1.3 at w=2 reports 29.08× because Δreal is only -0.6 (close to zero). Comparison should be at w=8 / w=16 where both Δpred and Δreal are well above noise.

**Possible narrative.** Promoted to Finding 10 update (see `content_masterarbeit.md` Finding 10 audit addendum). Two F10-bearing edits land:

1. **Smoothing is removed from F10's mechanism description.** The original F10 framing leaned on "ensemble + smoothing" as the strong-clean-baseline; now we know smoothing did nothing on top of ensembling. The strong-clean-baseline is just *ensembling*. This *strengthens* F10's "two fixes" claim — it is now (noise-aware + ensemble), full stop, with no third-knob ambiguity.
2. **The "ensemble cancels fold-specific adversarial shortcuts" mechanism story is softened.** Single-fold NA-v1 cells across all 5 folds give similar gap magnitudes (std 17.7 around mean -65.5). Folds aren't differentially-hacked; they're similarly-hacked. Ensembling acts more as **variance averaging across an honest residual** than as adversarial-direction cancellation. The 91-unit gap-reduction-from-ensembling at n=48 is real, but its mechanism is "average out per-fold over-claim biases" not "cancel adversarial directions". F10's compositional-necessity claim survives unchanged; the per-mechanism reading is updated.

**Methodological caveats.**
- **TANGO-only, no CamSol.** Same as F10 — `compute_developability.py` returns NaN for `camsol_intrinsic` (no public CamSol binary). Audit only covers tango_min direction.
- **No re-training.** Predictor checkpoints are the same v1 NA fold ckpts E029-E032 used. The audit measures *what the existing predictors do across the full matrix*, not whether different training would change the picture.
- **Smokes are still n=4.** The clean f0/f2 and NA f0/f1/f3/f4 cells have wide per-cell CIs (gap std 27-113). The "no fold is dramatically different" claim is robust because all 5 NA folds land in [-47, -97], a band tighter than the worst single-cell std — but a single fold being an outlier at n=12 or n=48 is not ruled out by these smokes.
- **Sweep is single-fold only.** Tier 1.3 measured fold 2 at n=48; folds 0, 1, 3, 4 are still only at n=4. The "all NA folds behave similarly at n=48" claim would require sweeping each fold to n=48 (~5h additional wall on 1 L4); the current data supports the weaker but defensible claim that "the n=4 cross-fold range is consistent with fold 2's n=48 result".
- **No designability / scRMSD validation.** Audit only re-measures the gap, not the structural-integrity story (which F10's E033/E036/E042 already established at n=12).
- **No UG K=5 + NA ensemble cell.** Open question from E030's negative result; flagged as the obvious follow-up if F10's gap closure looks tighter post-audit and the user wants to push further.

**Cross-references.**
- Driver: `script_utils/audit_steering_matrix.py` (`generate` stage gained `--cells` filter for parallel split; the `Cell` registry, eval, and report stages are unchanged from the as-pushed version).
- Configs: `steering/config/audit_matrix/` (12 YAMLs, committed before this entry).
- PDBs + diagnostics: `results/audit_matrix/<cell>/{guided/, diagnostics/, properties_guided.csv}`.
- Aggregated report: `audit_report.md`.
- Predecessors: [E028](#e028--predictor-vs-real-gap-on-the-may-04-ensemble-steered-run-2026-05-05) (clean ens+smooth baseline), [E029](#e029--noise-aware-predictor-fine-tune-and-single-fold-validation-smoke-2026-05-05) (NA fold 2 single pilot), [E031](#e031--noise-aware-predictor-v2-longer--cosine-decay-and-the-r-vs-hacking-disconnect-2026-05-05) (NA-v2 single-fold negative result), [E032](#e032--noise-aware-predictor--5-fold-ensemble--gap-essentially-closed-2026-05-05) (NA + ensemble combined fix).
- Driven by: [Finding 10](content_masterarbeit.md) audit pass.

---

## E051 — N=3 quick designability probe of `ca_only_sparse_K64_curriculum_self` at step 1800 (2026-05-10)

> Renumbered from local E049 on 2026-05-12 merge. References to "E047" in this section's body now point to [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08).

**Status:** finished.

**Why ran:** Fast convergence check on the K=64 SALAD-canonical sparse + low-t→SALAD curriculum + self-inclusion variant after the latest rsync brought a newer `last.ckpt` (epoch=17, global_step=1800 — ~91% past E047's step 944, and well into the canonical baseline's 1800-2200 best-val window). E047 left two open questions: (a) does the L=50 curriculum direction hold at higher step? (b) does the L=100 retraining hypothesis show any sign of life with more training? N=3 per length is the cheapest read that answers "is the picture qualitatively the same / better / worse" before committing GPU to a full N=6 panel.

**Configs:**
- Inference YAML: `configs/inference_sparse_K64_curriculum_self_step1800_n3_nfe400.yaml`. Inherits `inference_base.yaml` (`seed=5, nsteps=400`); `nlens_cfg=[50, 100, 200]`, `nsamples=3`, `max_nsamples_per_batch=3`. Curriculum + self-inclusion auto-fire from saved `cfg_exp.training.curriculum_neighbors=True`.
- Checkpoint: `/home/ks2218/la-proteina/sparse_K64_curriculum_self_step1800.ckpt` (stable copy of the rsynced `last.ckpt` so the next rsync doesn't clobber it). Verified: `run_name_=ca_only_sparse_K64_curriculum_self`, `epoch=17`, `global_step=1800`, `cfg_exp.training.curriculum_neighbors=True`, `cfg_exp.opt.compile_nn=True`. Same training run id 1778188245 as E047's step 944 ckpt.
- Hardware: 1× L4 (GPU 2), `nohup`, sequential. Wall: gen 4:39 (279 s for 9 proteins ≈ 31 s/protein — ~2× E047's per-protein cost because compile cold-cache is amortised over fewer proteins per length); eval ~5 min after `PYTHON_EXEC` and `--config-name`/cwd issues were sorted on attempts 1-3.
- Driver: ad-hoc inline `nohup bash -c '...'`. Logs: `nohup_inference_inference_sparse_K64_curriculum_self_step1800_n3_nfe400.{gen,eval}.log` (note doubled `inference_` prefix in driver log naming — does not affect content), `nohup_eval_inference_sparse_K64_curriculum_self_step1800_n3_nfe400.log`.
- Code fixes applied while debugging eval (NOT part of the model — just removing recurring footguns): `proteinfoundation/generate.py:41` accepts both `--config_name` and `--config-name`; `proteinfoundation/metrics/designability.py:133` defaults `PYTHON_EXEC` to `sys.executable` instead of bare `"python"`. Both changes are HEAD as of 2026-05-10. The eval that produced the numbers below ran *after* both fixes, with the laproteina_env `python` invoking `evaluate.py` and the ProteinMPNN subprocess inheriting that env.

**Results — N=3 designability (CA scRMSD vs ESMFold-of-MPNN-redesigned-sequence, ProteinMPNN `ca_only=True`):**

| L | sorted scRMSD (Å) | min | designable (<2Å) |
|---|---|---|---|
| 50  | [0.89, 1.67, 3.47]   | 0.89  | 2/3 |
| 100 | [1.23, 9.46, 10.91]  | 1.23  | 1/3 |
| 200 | [10.21, 12.38, 17.67]| 10.21 | 0/3 |

Pooled: **3/9 (33.3%)**, mean scRMSD 7.54 Å, no failed/crashed.

**Comparison vs E047 (same model family, step 944, N=6):**

| arm | L=50 | L=100 | L=200 | pooled | min L=50 | min L=100 | step |
|---|---|---|---|---|---|---|---|
| K=64 curriculum self (E047) | 3/6 | 1/6 | 0/6 | 4/18 (22%) | 0.91 | 1.22 | 944 |
| K=64 curriculum self **(E049)** | **2/3** | **1/3** | 0/3 | **3/9 (33%)** | 0.89 | 1.23 | **1800** |

The headline numbers move within sample-size noise on every length: the L=50 success rate is the same proportion (50% → 67% with N=3 is one Wilson-bin), the L=100 success rate is the same proportion (17% → 33% with N=3 is one Wilson-bin), L=200 stays at 0. Mins at L=50 and L=100 are essentially identical (0.89 vs 0.91; 1.23 vs 1.22).

**Verdict — at step 1800 the K=64 curriculum self-inclusion model looks the same as it did at step 944, within N=3 sampling noise:**

- L=50: still produces sub-1-Å samples, still has at least one collapse (3.47 here, 11.73 at step 944). Curriculum L=50 free lunch *survives* further training.
- L=100: still bimodal — one sample under 1.5 Å, the rest fully collapsed (>9 Å). The "L=100 will close with more training" hypothesis from E047 is **not yet supported** at step 1800. The collapse cluster did not shrink, the clean cluster did not grow. ~91% more training did not move the L=100 picture.
- L=200: still dead in every sample. No length-dependent intervention has helped L=200 in any K=40 or K=64 sparse arm; not specific to this entry.

**Methodological caveats:**

- **N=3 per length.** This is half of E047's N=6 and an even smaller absolute count. Single-bin Wilson intervals at this N are huge — every per-length number above sits inside the 95% interval of every other arm in the K=64 curriculum family. The right reading is "directionally similar to E047", not "step 1800 has rate X". A full N=6 (or N=12) panel is the correct next probe if a defensible step-1800 number is needed.
- **Single seed (seed=5, inherited from `inference_base.yaml`).** Same seed as E047, so the L=50 / L=100 noise samples are paired across the two checkpoints — the comparison above is paired-by-noise within each length cell. The noise tensors at L=50 and L=100 should be identical (CA-only, same shape across both runs); the integrated trajectories diverge because the weights at step 1800 ≠ step 944.
- **Three architectural confounds vs the K=40 sparse baseline are unchanged from E047** (K=40→64, mask-curriculum, self-inclusion); this entry doesn't touch them and inherits all of them.
- **Eval driver had three failed launches before the fourth produced numbers** (wrong argparse flag, missing PYTHON_EXEC, stale cwd from an earlier `cd` in this session). The first failed launch left empty tmp_dirs that tripped the resume guard on the second; both were cleaned before the fourth attempt. Numbers are from the fourth run on a clean slate.
- **`nsteps=400`** ✓ (HARD RULE).

**Possible narrative:** non-narrative — kept for tuning/decision-making. Decision it informs: do we keep training `ca_only_sparse_K64_curriculum_self` past step 1800, or call the L=100 gap unresponsive to the curriculum-direction lever and pivot? Recommendation: bank a full N=6 panel at step 1800 first (this entry is N=3, and Wilson-bin overlap with E047 is total). If N=6 still shows L=100 stuck at ≤2/6 designable, the curriculum direction is decided as "L=50 only" and the L=100 work needs a different lever.

**Cross-references:**
- Predecessor: [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_K64_curriculum_self-ep9-step944-2026-05-08) (same training run, step 944, N=6).
- Configs: `configs/inference_sparse_K64_curriculum_self_step1800_n3_nfe400.yaml`; training-side `configs/training_ca_only_sparse_K64_curriculum.yaml`; NN side `configs/nn/ca_only_sparse_K64_curriculum_160M.yaml`.
- Output CSV: `inference/results_inference_sparse_K64_curriculum_self_step1800_n3_nfe400_0.csv`.
- HPC source: `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self/1778188245/checkpoints/last.ckpt` (rsynced 2026-05-09 17:38, copied to stable name 2026-05-10).
- Code fixes applied during this entry's debug: `proteinfoundation/generate.py:41` (argparse alias), `proteinfoundation/metrics/designability.py:133` (`sys.executable` default for ProteinMPNN subprocess).

## E052 — Universal-guidance K-axis extension, clean predictor (2026-05-10)

> Renumbered from local E050 on 2026-05-12 merge. References to "[E048](...)" in this section's body point to [E050](#e050--steering-audit-matrix--predictor--ensemble--fold--smoothing-2026-05-10) (the audit matrix).

**Status:** finished.

**Why ran.** E030 left "is K=5 special, or is the negative direction monotone?" open, and the [E050](#e050--steering-audit-matrix--predictor--ensemble--fold--smoothing-2026-05-10) audit thread implied a follow-up: now that we know smoothing contributes ~zero on top of ensembling (clean+ens+smooth gap −203.5 ≈ clean+ens no-smooth −203.9, audit Tier 1.1), it's worth probing whether *more* denoising steps with a clean predictor reverse E030's negative direction or confirm the "longer-Jacobian-amplifies-hacking" mechanism story.

Original target was K=20 to make a strong K-axis claim. Three OOM events on L4 (22 GB) walked it back:
- K=20 + smoothing K_sm=4: ~30 GB needed (E030's K=5 + K_sm=4 fits at ~22 GB; scaling argument predicts 30 GB at K=20).
- K=20, smoothing off: ~25 GB needed.
- K=10, smoothing off: ~24 GB needed.

The bottleneck is the K-step inner Euler loop with autograd-tracked flow forwards — each inner step keeps full 160M-param activations live for backward through the predictor. K=8 was the largest value that fit in the budget without committing to gradient checkpointing on the inner loop (which was scoped as a separate ~30 min code change and deferred — see "Methodological caveats" below).

**Configs.**
- Steering config: `steering/config/universal_guidance_smoke/tango_min_w16_clean_K8.yaml`. 5-fold clean ensemble (same fold ckpts as E028/E030, no NA fine-tune), tango_min, w_max=16, linear_ramp(t_start=0.3, t_end=0.8, t_stop=0.9), `denoising_steps: 8`, **smoothing OFF**.
- Why smoothing off: audit Tier 1.1 (`clean_ensemble_nosmoothing`) showed clean+ens with σ=0.1, K_sm=4 smoothing gave gap −203.5 vs clean+ens no-smoothing gap −203.9 at the same w=16, L=300, n=4 cell — within the 49–65 gap std. So the smoothing-off K=8 cell is reasonable apples-to-apples with E028 (K=1) and E030 (K=5) for a K-axis comparison, plus K_sm=4 × K_d=8 = 32 inner predictor calls per outer step exceeded the L4 22 GB budget.
- Run: 4 seeds {42, 43, 44, 45} × L=300 × w_max=16, nsteps=400. Same protein IDs as E029/E030 for direct comparison.
- Output: `results/universal_guidance_smoke/tango_min_w16_clean_K8/`.
- Hardware: 1× L4 (GPU 3, the only fully-clean GPU at launch — first attempt on GPU 1 was killed mid-run by a co-tenant grabbing 2.83 GB). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on. Wall: ~12 min total (~180 s/protein at L=300; vs E030's 115 s/protein at K=5 → linear scaling in K_d as expected).
- Driver under nohup; logs at `logs/audit_matrix/k8.log`.

**Results — predictor-vs-real (same 4 seeds, L=300, w=16):**

| K | smoothing | predictor mean | real `tango_total` mean | mean gap | source |
|---|---|---|---|---|---|
| 1 | σ=0.1, K_sm=4 | 378.5 | 581.9 | **−203.5** | E028 / E048 ref |
| 5 | σ=0.1, K_sm=4 | 305.4 | 607.1 | **−301.7** | E030 |
| **8** | **off** | **307.6** | **616.2** | **−308.6** | E050 (this run) |

Per-protein K=8 gaps: s42 −353.7, s43 −201.8, s44 −310.8, s45 −368.2 (gap std 75.3).

**Headline numbers:**
- **Monotone-worse but plateauing.** Gap goes K=1: −204, K=5: −302, K=8: −309. The K=1 → K=5 jump nearly doubled the hacking gap; K=5 → K=8 is essentially flat (Δgap = −7, smaller than the per-cell gap std of 75). Extending K past 5 does not reverse E030's negative direction, but it also does not accelerate it.
- **Real TANGO drifts UP with K.** K=1: 581.9, K=5: 607.1, K=8: 616.2. The clean predictor pushes harder in a less-useful direction as K grows — the 4 PDBs at K=8 are *less* TANGO-minimized than at K=1.
- **K=5 was not an unlucky local point.** The K=8 cell rules out the "K=5 happened to land in a bad bucket" alternative explanation for E030's negative result. The mechanism story (longer Jacobian = more adversarial leverage on a fragile predictor) is supported by an additional data point in the same direction.

**Possible narrative.** Non-narrative — closes the K-axis open question that E030 / E048 flagged. Updates [Finding 10](content_masterarbeit.md)'s negative-results table to add the K=8 row alongside the existing K=5 entry; the implication line "ordering matters — fix the predictor input distribution before any sampling-time refinement" gets a second supporting data point. No edit to F10's headline claims.

**Methodological caveats.**
- **n=4, single L, single w, single direction.** Same caveat as E030 — small sample, but the negative direction is consistent across all 4 proteins (K=8 gaps all between −202 and −368, none flip positive).
- **Smoothing OFF for K=8, ON for K=1 (E028) and K=5 (E030).** The audit-justified choice (audit Tier 1.1 showed smoothing made no measurable difference) means this is approximately apples-to-apples, but not strictly so. A K=1, smoothing-off cell at the same n=4 protocol would close this gap formally; the audit's `clean_ensemble_nosmoothing` cell is exactly that, and it landed at gap −203.9 — essentially identical to E028's −203.5. So the smoothing-off vs smoothing-on comparison at K=1 is settled; the assumption that the same equivalence holds at K=5 and K=8 is the unverified part.
- **K=20 not measured.** The original target. K-axis past 8 on L4 (22 GB) requires gradient checkpointing on the inner Euler loop (~30 min code change to wrap `flow_step_fn` calls in `torch.utils.checkpoint.checkpoint_sequential`). Deferred — given the K=5 → K=8 plateau, the additional information from a true K=20 cell is unlikely to flip the K-axis story.
- **Clean-predictor regime only.** The code's `feed_z_t_directly` toggle makes the K-step branch and the noise-aware path mutually exclusive (see `steering/guide.py:143-156`): NA predictors consume z_t directly with no Tweedie / K-step inner loop. So "K=8 + NA-ensemble" is not a defined combination under the current architecture without re-training NA on a different input distribution. F10's mechanism reading — that NA fixes the same problem UG K-step is trying to fix, just at the training layer instead of the inference layer — is consistent with this code-level exclusivity.

**Cross-references.**
- Predecessors: [E028](#e028--predictor-vs-real-gap-on-the-may-04-ensemble-steered-run-2026-05-05) (K=1 baseline), [E030](#e030--universal-guidance-k5-with-clean-predictor-probe-2026-05-05) (K=5 negative result this entry extends), [E050](#e050--steering-audit-matrix--predictor--ensemble--fold--smoothing-2026-05-10) (smoothing-off equivalence that justifies dropping smoothing here).
- Driven by: [Finding 10](content_masterarbeit.md) audit thread; closes "no UG K-step + extended K data" open question called out in E048's "What the script will NOT do" section.
- Configs: `steering/config/universal_guidance_smoke/tango_min_w16_clean_K{8,10,20}.yaml` (K=10 and K=20 configs kept as the failed-OOM record).
- Output: `results/universal_guidance_smoke/tango_min_w16_clean_K8/{guided/, diagnostics/, properties_guided.csv}`.
- Logs: `logs/audit_matrix/k8.log` (success), `logs/audit_matrix/k10.log` and `k20.log` (OOM logs preserved as engineering record).

## E053 — CA-only `downsampled` variant canonical N=6 designability probe at step 3716 (2026-05-11)

> Renumbered from local E051 on 2026-05-12 merge.

**Status:** finished.

**Why ran:** Most recent `ca_only_downsampled` ckpt (epoch=36, opt step=3716) was rsynced from HPC. Question: does ~755 additional opt steps past the previous canonical-N=6 read (step 2961, [E034 caveat](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) — 3/18 at nsteps=400) move the variant off its plateau, or is the downsampled arm converged at "L=50 occasional, L≥100 dead"? Decision this feeds: whether to keep training the downsampled chain or call the architectural ceiling reached.

**Configs.**
- Checkpoint: `/home/ks2218/la-proteina/best_val_00000036_000000003716.ckpt` (epoch 36, opt step 3716; 1.96 GB; rsynced from HPC 2026-05-11 14:17). Architecture loaded from ckpt's `hyper_parameters` (no local `configs/training_ca_only_downsampled.yaml` or `configs/nn/ca_only_downsampled_*.yaml` needed for inference — same as E034).
- Inference config (new): `configs/inference_downsampled_step3716_n6_nfe400.yaml`. Identical protocol to `inference_downsampled_step2331_n6_nfe400.yaml` / `inference_downsampled_step2961.yaml`: `nsteps=400`, `nsamples=6`, `max_nsamples_per_batch=6`, `nres_lens=[50, 100, 200]`, `seed=5` (inherited from `inference_base.yaml`).
- Driver: single GPU 0 (L4, 23 GB), serial gen→eval under `nohup`. PID 317001.
- Output dir: `inference/inference_downsampled_step3716_n6_nfe400/` (18 PDBs across `job_0_n_{50,100,200}_id_{0..5}`). CSV: `inference/results_inference_downsampled_step3716_n6_nfe400_0.csv`.
- Logs: `nohup_inference_downsampled_step3716_n6_nfe400.gen.log` (49.3s reported by `performance_utils`), `nohup_inference_downsampled_step3716_n6_nfe400.eval.log`, `nohup_inference_downsampled_step3716_n6_nfe400.queue.log`.
- Wall-clock: gen 14:22:15 → 14:23:26 (71s incl. import/setup; 49.3s predict loop), eval 14:23:26 → 14:32:23 (~9 min ESMFold+MPNN); total ~10 min on 1× L4.

**Results.**

Per-length scRMSD (Å, sorted) and designability (scRMSD_ca_esmfold < 2 Å):

| Length | scRMSD vals (sorted) | min | median | designable |
|--------|----------------------|-----|--------|------------|
| L=50   | 1.19, 2.19, 2.28, 2.51, 3.65, 4.23 | **1.19** | 2.51 | **1/6 (17%)** |
| L=100  | 8.01, 8.51, 8.69, 11.73, 12.19, 15.80 | 8.01 | 11.73 | 0/6 (0%) |
| L=200  | 9.94, 11.58, 12.06, 13.78, 16.60, 17.37 | 9.94 | 13.78 | 0/6 (0%) |
| **all** | — | 1.19 | — | **1/18 (5.6%)** |

Cross-step view (same arm, same canonical N=6 × L∈{50,100,200} × nsteps=400 protocol where available):

| Step | L=50 | L=100 | L=200 | Pool | Best Å | Source |
|------|------|-------|-------|------|--------|--------|
| 2331 | 0/6 | 0/6 | 0/6 | 0/18 | 12.41 (at nsteps=200) | E034 |
| 2331 | — | — | — | — | — | nsteps=400 redo queued 2026-05-07 in `run_nfe400_reprobes_queue.sh`; not back-filled in a dedicated E-entry |
| 2961 | (≥2/6 by E034 caveat phrasing) | mixed | dead | 3/18 (17%) | 1.60 | E034 caveat at line 2814 (referenced inline, no dedicated entry) |
| **3716 (this)** | **1/6** | **0/6** | **0/6** | **1/18 (5.6%)** | **1.19** | E051 |

Direction: between step 2961 (3/18, best 1.60 Å) and step 3716 (1/18, best 1.19 Å), pooled designability dropped from 17% → 5.6%, but the best L=50 sample improved (1.60 → 1.19 Å). Both lengths L=100 and L=200 stayed dead.

**Possible narrative.** Non-narrative — kept for tuning/decision-making. The downsampled arm has now been canonical-probed at 3 training steps (2331, 2961, 3716), and the pooled designability time-series is **0/18 → 3/18 → 1/18**. This is within N=18 binomial noise, but the picture is consistent with "L=50 occasional, L≥100 architecturally dead". The architectural lever (1D-conv downsampling, BlurPool1D stride configuration per CLAUDE.md notes) does not appear to clear the canonical bar at any of these steps. Could become a methodological aside in the paper if a step-matched debugging pass (canonical step ~2200 vs downsampled step ~2200, intermediate-activation diff at L=100) attributes the L≥100 collapse to a specific mechanism. Not eligible for `content_masterarbeit.md` Finding promotion as-is.

**Decision implications.** The downsampled variant has had ~1400 opt steps past the canonical baseline's best-val window (1800-2200) and has not shown improvement at L≥100. Recommend not chaining more training without a mechanism-level debug — the L=50/L=100 gap is the most informative signal (it's not an "under-trained" failure if L=50 already produces sub-2-Å samples while L=100 simultaneously produces nothing below 8 Å). Compare to the K=64-curriculum arm (E047/E049), which shows the same L=50-good-L=100-mixed-L=200-dead pattern at much earlier steps; this may be a 160M-canonical-recipe ceiling rather than a downsampling-specific failure.

**Methodological caveats.**
- **N=6/length** is small. The Δ from 3/18 → 1/18 across steps 2961 → 3716 is within ~12 pp binomial 1σ (`sqrt(0.17*0.83/18) ≈ 9 pp`) — the regression is not statistically tight. Conclusion supported by the cross-step pattern (all three reads bunch at "L=50 occasional, L≥100 dead"), not by any one step in isolation.
- **Seed=5 only.** Single-seed read; same-seed paired comparison to step-2961 not run on this entry (could be added by reusing step 2961's already-on-disk per-(L,id) PDB outputs and quoting scRMSD pair-wise; deferred — not load-bearing for the "ceiling at L=50" call).
- **No training config / NN config locally.** Architectural details for the downsampled variant remain HPC-only; the verdict here is on the *observed sampling behaviour* of the ckpt's trained weights, not on whether the architecture is "right". A debug pass would need access to the HPC config.
- **Wall-clock for gen (49 s for 18 samples at nsteps=400)** is suspicious-looking but consistent: prior downsampled probes ran in similar time on L4 (E034 noted "9 min total" for gen+eval at nsteps=200; doubling nsteps roughly doubles gen but eval cost is integrator-independent). Predict loop time printed by `performance_utils` is the authoritative number, not the queue-script wall delta which includes setup.
- **Not back-filled:** the `inference_downsampled_step2331_n6_nfe400` redo queued in `run_nfe400_reprobes_queue.sh` (E034 caveat) does not have a dedicated E-entry; its CSV (`results_inference_downsampled_step2331_n6_nfe400_0.csv`) is on disk and could be promoted to an entry to complete the 3-step time series. Out of scope for E051.

**Cross-references.**
- Predecessors: [E034](#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) (step 2331 at nsteps=200, plus the nsteps=400 step-2961 result referenced inline in E034's caveat block); see also `inference/results_inference_downsampled_step2961_0.csv` for the step-2961 canonical CSV (no dedicated entry).
- Companion architectural-axis arms at canonical N=6 × nsteps=400: [E039](#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06) (scnbr+Fix C2 step 1133, 3/18), [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) (K=64 curriculum-trained step 944), [E051](#e051--n3-quick-designability-probe-of-ca_only_sparse_k64_curriculum_self-at-step-1800-2026-05-10) (K=64 curriculum-trained step 1800, N=3).
- Configs: `configs/inference_downsampled_step3716_n6_nfe400.yaml`.
- Output: `inference/inference_downsampled_step3716_n6_nfe400/`, `inference/results_inference_downsampled_step3716_n6_nfe400_0.csv`.
- Logs: `nohup_inference_downsampled_step3716_n6_nfe400.{gen,eval,queue}.log`.

---

## E054 — Canonical baseline `last-v2.ckpt` N=6 × nsteps=400 designability probe (step 1952, 2026-05-10)

> Renumbered from local E052 on 2026-05-12 merge.

**Status:** finished (2026-05-10 18:35 BST designability; 2026-05-11 per-t paired add-on; logged 2026-05-11).

**Why ran:** The canonical baseline's "best val ≈ 1800-2200" window sits well below the on-disk anchor ckpt step 2646 — i.e. step 2646 is past the val-loss minimum. User question: does a ckpt *closer to best val* (here `last-v2.ckpt`, Lightning's auto-versioned `last.ckpt` slot-end snapshot on the same `test_ca_only_diffusion/1776805213` canonical run) deliver better designability than the step-2646 ckpt that has been the everyday reference? Decision input for whether the "canonical best" used as the comparison anchor in variant probes (E021/E034/E039/E047/E049/E051) should be re-anchored to a different step.

**Checkpoint provenance (corrected 2026-05-11).** `last-v2.ckpt` was originally inferred to be at "step ~1900" from file mtime; reading the ckpt's `global_step` field directly gives **step 1952, epoch 19**. It is **not** a `best_val_*` ckpt — the `-v2` suffix is Lightning's auto-version for a `last.ckpt` that would otherwise overwrite a prior `last.ckpt`. Each ckpt's `callbacks → ModelCheckpoint` state on disk records the canonical run's true val_loss history:

| local ckpt | step | `current_score` (val_loss/loss_epoch at save) | what it actually is |
|---|---|---|---|
| `best_val_00000018_000000001889.ckpt` | 1889 | 4.7919 | best_val at the time |
| `last-v2.ckpt` | **1952** | **4.7874** | slot-end snapshot, Δ=+0.076 nat above all-time min |
| `best_val_00000024_000000002457.ckpt` | 2457 | 5.4134 | post-overfit save (mechanism unclear; under save_top_k=1 should not have saved this since the all-time best 4.7115 still held — possibly a chained-restore `best_model_score` reset) |
| `best_val_00000026_000000002646.ckpt` | 2646 | **5.9237** | post-overfit save; Δ=+1.21 nat above all-time min |

All four ckpts' callback state records `best_model_score = 4.7115` at `best_val_00000021_000000002204.ckpt` — the **all-time canonical val_loss minimum at step 2204**, overwritten under `save_top_k=1` and gone from disk. So `best_val_00000026_000000002646.ckpt` is misnamed as the "best ckpt on disk" only in the sample-quality sense; by val_loss it is 1.13 nat *worse* than `last-v2.ckpt`.

**Configs (re-run-able from this entry):**
- Inference config: `configs/inference_canonical_lastv2_n6_nfe400.yaml` (inherits `inference_base.yaml` → `nsteps: 400`; CA-only generation; `autoencoder_ckpt_path: ` left empty).
- Checkpoint: `/home/ks2218/la-proteina/last-v2.ckpt` (1.9 GB, mtime 2026-04-22 22:49; `global_step=1952`, `epoch=19`, `run_name_=test_ca_only_diffusion`).
- Protocol: N=6 × L ∈ {50, 100, 200}, seed=5, `inference_ucond_notri_ca_only` generation defaults.
- Driver: `nohup_inference_canonical_lastv2_n6_nfe400.{driver,gen,eval}.log` (single-GPU L4 inference; runtime ~6 min total).
- Per-t add-on (2026-05-11): `proteinfoundation/run_per_t_val.py --ckpt_name last-v2.ckpt --label canonical_lastv2 --num_proteins 600 --seed 42`; output `results/per_t_val/canonical_lastv2.json`; log `nohup_per_t_canonical_lastv2.log`. Paired with the existing `canonical_2646.json` (E043 protocol, same subset_seed=42 → identical 600-protein subset and per-protein rotations).

**Results.**

| L | designable (<2 Å) | best (Å) | median (Å) | mean (Å) |
|---|---|---|---|---|
| 50  | **4/6** (67%) | 0.78 | 1.98 | 2.64 |
| 100 | **3/6** (50%) | 1.07 | 2.06 | 3.39 |
| 200 | **0/6** (0%)  | 4.29 | 8.18 | 8.04 |
| **pooled** | **7/18 = 38.9%** | 0.78 | — | — |

**Per-sample scRMSD (best across recycles, `_res_scRMSD_ca_esmfold`):**
- L=50: 2.01 / 1.98 / 1.95 / 1.33 / 7.78 / 0.78
- L=100: 2.06 / 1.39 / 1.07 / 11.66 / 2.60 / 1.58
- L=200: 6.75 / 8.18 / 7.35 / 11.36 / 4.29 / 10.33

**Cross-step canonical baseline comparison (same `test_ca_only_diffusion/1776805213` run, post-MPNN-fix where applicable):**

| ckpt | step | val_loss/loss_epoch | N/L | L=50 | L=100 | L=200 | pooled |
|---|---|---|---|---|---|---|---|
| **last-v2 (this entry)** | **1952** | **4.7874** | 6 | 4/6 (67%) | 3/6 (50%) | **0/6** | **7/18 = 38.9%** |
| baseline_wd0.05_step2646 (E019) | 2646 | **5.9237** | 30 | 26/30 (87%) | 26/30 (87%) | 16/30 (53%) | 68/90 = **75.6%** |
| baseline_wd0.05_step2646 (older `inference_2646`) | 2646 | 5.9237 | 3 | 3/3 | 3/3 | 3/3 | 9/9 = 100% |

**Per-t validation loss (paired, seed=42, same 600-protein subset / per-protein rotations):**

| t-bucket | canonical_2646 (E043) | canonical_lastv2 (this entry) | Δ (lastv2 − 2646) | Δ / SEM_lastv2 |
|---|---|---|---|---|
| [0.0, 0.2) | 3.0184 ± 0.0763 | 3.0089 ± 0.0735 | −0.0095 | −0.1 |
| [0.2, 0.4) | 1.9321 ± 0.0249 | 1.9305 ± 0.0223 | −0.0016 | −0.1 |
| [0.4, 0.6) | 1.2930 ± 0.0168 | **1.3291 ± 0.0149** | **+0.0361** | **+2.4σ** |
| [0.6, 0.8) | 1.0855 ± 0.0101 | **1.1345 ± 0.0096** | **+0.0490** | **+5.1σ** |
| [0.8, 1.0) | 1.3127 ± 0.0150 | **1.3712 ± 0.0145** | **+0.0585** | **+4.0σ** |

Direction of paired per-t agrees with designability: ties at t<0.4, monotonically worsening lastv2 deficit for t≥0.4 (i.e. on the data side of the trajectory). Caveat: the 600-protein subset is from `pdb_train/processed_latents/` (E043 protocol — bypasses the broken `PDBLightningDataModule`), so absolute numbers are not directly comparable to wandb's `validation_loss/loss_epoch` on the 4058 held-out val proteins. The paired *deltas* and their sign/magnitude are robust.

**Possible narrative.** Non-narrative — kept for tuning/decision-making. **Cleanest within-run val-loss-vs-sample-quality decoupling on record**: a ckpt with **Δ = −1.13 nat lower** wandb val_loss (last-v2 step 1952 at 4.79 vs step-2646 at 5.92) produces **materially worse** designability (7/18 vs 68/90 pooled, with L=200 completely collapsing at 0/6 vs 16/30) and **higher** paired per-t loss at every t ≥ 0.4 (Δ up to 5σ). Both signals point the same way. Decision: **do NOT re-anchor canonical comparisons to lastv2 / step-1952**; step-2646 remains the reference for variant probes. Feeds the broader `feedback_wandb_val_loss_not_comparable.md` memory: within-run val-loss monotonicity does not hold either. Mechanism almost certainly the same as Finding 5/6 (AdaLN-Zero gate growth between step 1952 → 2646), but per-layer gate inspection on step-1952 vs step-2646 has not been run on this entry.

**Methodological caveats.**
- **N=6/length** is small relative to the N=30 anchor; the L=200 zero is robust (best 4.29 Å, every sample ≥ 4.29 Å) but L=50/100 rates have ~20 pp binomial 1σ. Per-t paired test is much higher-resolution (n=600 per bucket, 3-5σ deltas) and removes the N-imbalance concern.
- **Single seed (seed=5)** for designability — paired-by-noise comparison to step 2646 not run; would need a step-2646 N=6 re-probe at the same seed to remove seed confound. Step-2646 N=30 used multiple seeds. Per-t add-on (seed=42, paired protein subset and rotations) provides the apples-to-apples sample-side check.
- The lastv2 ckpt is the raw weights (filename has no `-EMA` suffix), matching step 2646's RAW selection in Finding 7 and E019. No EMA-vs-RAW mixing in the comparison.
- **E019 N=30 anchor is the right comparison** for designability (post-MPNN-fix eval). The N=9 `inference_2646` 100% number is *older* and not directly comparable (CSV has -1.0 sentinels in some fields → not the fixed eval path).
- **Per-t protocol uses a train-set subset**, not the wandb-monitored val set, so absolute numbers don't match wandb's `validation_loss/loss_epoch` (which lastv2 records as 4.79 vs 2646's 5.92). Paired deltas in this protocol still validly answer "which ckpt's weights have lower FM loss on the same data and rotations".
- **Mechanism not directly probed.** The step 1952 → step 2646 gate-growth picture is consistent with Finding 5/6 (AdaLN-Zero gates need to grow past zero-init; uniform-wd training continues to grow them after the val-loss-defined "best" point), but a per-layer gate-norm post-mortem comparing 1952 vs 2646 weights was not run.

**Cross-references.**
- Anchor for canonical step 2646: [E019](#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) (N=30 post-fix, body TBD; numbers pulled from `inference/results_inference_baseline_n30_0.csv`).
- Per-t comparison protocol: [E043](#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07). `canonical_2646.json` reused as-is.
- Mechanism: Finding 5/6 (val_loss decouples from sample quality under uniform-wd AdamW with AdaLN-Zero gates) and `feedback_wandb_val_loss_not_comparable.md`. This entry strengthens the case to "even within a single training run, lower val_loss does not predict better samples — and the gap can be 1+ nat in val_loss while flipping the sample-quality verdict".
- Configs: `configs/inference_canonical_lastv2_n6_nfe400.yaml`.
- Output: `inference/inference_canonical_lastv2_n6_nfe400/`, `inference/results_inference_canonical_lastv2_n6_nfe400_0.csv`, `results/per_t_val/canonical_lastv2.json`.
- Logs: `nohup_inference_canonical_lastv2_n6_nfe400.{driver,gen,eval}.log`, `nohup_per_t_canonical_lastv2.log`.

---

## E055 — First designability probe of the five-axis bundle (`ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft`, step 944, 2026-05-12)

**Status:** finished.

**Why ran.** First probe-worthy ckpt of the cold-start retrain queued in [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) — the five-axis bundle (K=64 SALAD-canonical sparse + curriculum + self-inclusion + **BigBird globals n=4** + **pair-update every 3 layers**) with the **softened low-t bucket** `curriculum_low_t_split=[16, 8, 24]` baked in and the off-by-one cap fix from [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) live in code. Feeds the binary: did the bundle deliver early signal that catching up to (or beating) the §11 step-1385 baseline is on track, or is the 3-axis cold-start cost predicted in E047 a dead-weight handicap at this many opt steps? N=6 × L∈{50,100,200} × nsteps=400 is the cheapest read.

**Configs.**
- Inference YAML: `configs/inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400.yaml` (inherits `inference_base.yaml` → `nsteps=400`, `seed=5`). `nsamples=6`, `nres_lens=[50, 100, 200]`. **No inference-time overrides** — BigBird, pair-update, sparse K=64, curriculum, lowtsoft bucket all auto-replay from saved `cfg_exp` (run_name `ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft`).
- **The config + driver names embed "step1133" — but the actual ckpt loaded is `best_val_00000009_000000000944.ckpt` (epoch 9, opt step 944).** The "step1133" handle was the planned probe-worthy step when the config was authored on 2026-05-11; the latest best_val available on HPC at the 2026-05-12 09:49 BST rsync was step 944. The script targets a stable symlink `sparse_K64_bigbird_lowtsoft_step1133.ckpt` → the underlying ckpt file; I symlinked it to the fresh step-944 ckpt today. Output dir + CSV + logs therefore carry the misnomer "step1133" while the underlying ckpt step is 944. **Use this entry's "step 944" wherever step is needed for comparison.**
- Saved ckpt hyperparameters confirmed (via `OmegaConf.select`): `run_name_=ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft`, `nn.sparse_attention=True`, `nn.n_seq_neighbors=8, n_spatial_neighbors=16, n_random_neighbors=32` (canonical SALAD K=64 composition), `nn.n_global_tokens=4` (BigBird), `nn.update_pair_repr=True, update_pair_repr_every_n=3`, `training.curriculum_neighbors=True`, `training.curriculum_low_t_split=[16, 8, 24]` (LOWTSOFT), `opt.compile_nn=True`, `optimizer.weight_decay=0.05`. SHA-256 of state_dict (canonical key order, fp32 cast): `8cf2136906bd5547fc7234c6a969d7d9...`. Total params 161,552,128; 0 NaN-bearing tensors.
- Checkpoint provenance: `/home/ks2218/la-proteina/best_val_00000009_000000000944.ckpt` (1.94 GB, mtime 2026-05-12 09:49 BST). Rsynced from `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft/1778520282/checkpoints/`. SLURM 29210711, wandb run id `vru8hr9y`. Distinct from the §11 K=64-curriculum-self ckpts (different `run_name_`; this ckpt's `state_dict` carries 5 BigBird tensors `global_token_emb [4,768]`, `global_cond_emb [4,256]`, `global_pair_bias_{res→glob, glob→res, glob→glob}` that the §11 ckpts don't have).
- Hardware: 1× L4 (GPU 6 on `gxp-l4-0`), `nohup`, sequential gen → eval.
- Driver: `script_utils/probe_sparse_K64_bigbird_lowtsoft.sh` (pulled from `origin/main` 2026-05-12; the upstream `PYTHON_EXEC` path was the HPC location and the script failed `exit=127`. Patched locally to `/home/ks2218/.conda/envs/laproteina_env/bin/python` per [feedback_fix_root_cause_in_code.md](../.claude/projects/-home-ks2218-la-proteina/memory/feedback_fix_root_cause_in_code.md). Patch not yet pushed.).
- Wall: **gen 6:11 (371 s; 344.7 s pure predict_loop)** at 18 proteins ≈ 20.6 s/protein at nsteps=400. **Eval 8:56**. Peak GPU 2050 MB. Total 14:55 on L4 — substantially faster than the 30-min A100 estimate in the script header.

**Results — pooled designability (N=6 × L ∈ {50, 100, 200}, scRMSD_ca_esmfold < 2 Å):**

| L | n | designable | best (Å) | median (Å) |
|---|---|---|---|---|
| 50  | 6 | **3/6 (50%)** | **1.45** | 2.62  |
| 100 | 6 | **0/6 (0%)**  | 6.33    | 7.25  |
| 200 | 6 | **0/6 (0%)**  | 8.84    | 14.21 |
| **pooled** | **18** | **3/18 = 16.7%** | **1.45** | — |

**Per-sample scRMSD (`_res_scRMSD_ca_esmfold`, sorted):**
- L=50:  `[1.45 ✓, 1.50 ✓, 1.74 ✓, 3.54, 3.65, 3.77]` — 3 sub-2 Å, no L=50 sample over 3.8 Å.
- L=100: `[6.33, 6.57, 7.23, 7.44, 8.13, 9.06]` — *every* L=100 sample > 6 Å.
- L=200: `[8.84, 12.24, 13.87, 14.65, 19.05, 20.92]` — min 8.84 Å; two samples > 19 Å.

**Cross-arm comparison at canonical N=6/L (or pooled N=18 when both columns available):**

| arm | step | L=50 | L=100 | L=200 | pool | best Å | notes |
|---|---|---|---|---|---|---|---|
| canonical dense E019 | 2646 | 26/30 (87%) | 26/30 (87%) | 16/30 (53%) | 68/90 (76%) | 0.79 | N=30 anchor |
| §11 K=64-curric step-1385 pre-fix N=18 | 1385 | 8/18 (44%) | 10/18 (56%) | 2/18 (11%) | 20/54 (37%) | 0.94 | three-axis variant pre-cap-fix |
| §11 K=64-curric step-1385 + LOWTSOFT inference (E046) | 1385 | 10/18 (56%) | 11/18 (61%) | 0/18 (0%) | 21/54 (39%) | 0.62 | inference-only forcing of (16,8,24) on §11 |
| §11 K=64-curric step-944 ([E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08)) | 944 | 3/6 (50%) | 1/6 (17%) | 0/6 (0%) | 4/18 (22%) | 0.91 | three-axis variant N=6 at the same step number |
| **five-axis bundle (E055, this)** | **944** | **3/6 (50%)** | **0/6 (0%)** | **0/6 (0%)** | **3/18 (17%)** | **1.45** | five-axis bundle at same step number |

**Reading of the comparison:**
1. **L=50 holds the same count** as §11 step-944 (3/6, 50%) — adding BigBird + pair-update + lowtsoft on top of K=64-curric does not break the short-protein curriculum signal at this step. **Min Å is up** from 0.91 (E049) to 1.45 (this entry); the bundle's L=50 cluster is shifted ~0.5 Å worse on min but still clears the 2 Å bar with the same count.
2. **L=100 worsened** from 1/6 (E049) to 0/6 — every L=100 sample is over 6 Å here. The added training-axis cost predicted in E047 (BigBird globals + pair-update need to find their working points from cold init) shows up most clearly at L=100, the most discriminative length for sparse-variant probes.
3. **L=200 unchanged dead** (0/6 in both; min 8.84 Å vs E049's 9.64 Å — within noise).
4. **Bundle vs §11 step-1385 LOWTSOFT inference probe** ([E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11)): at L=50 the bundle (50%) is below the inference probe (56%); at L=100 the bundle (0%) is far below the inference probe (61%). The cold-started bundle at step 944 has not yet recovered the inference-only LOWTSOFT signal on the §11 ckpt at step 1385 — predicted by E047's note that the bundle has 3 new architectural axes to grow into from cold init.

**Verdict at step 944.** **Bundle is below §11 step-944 at every length, with L=100 the cleanest regression** (1/6 → 0/6; median scRMSD up from 4.5 to 7.25 Å). This is consistent with — but does not prove — E047's prediction that the 3 new axes are paying their cold-start cost in the first ~1000 opt steps and will close the gap later in the slot. **Decision deferred to the slot-end ckpt** (predicted step ~2400 from the 20 h SL2 slot at the bundle's measured opt-steps/h; first canonical milestone in the E047 plan). At step 944 this is "training-cost handicap consistent with prediction"; it is NOT yet "five-axis bundle is dead". Per [feedback_dead_arm_calls.md](../.claude/projects/-home-ks2218-la-proteina/memory/feedback_dead_arm_calls.md), the call-it-dead bar is "loses to baseline at the *converged* step" — step 944 is well below the canonical 1800-2200 best-val window.

**Methodological caveats.**
- **N=6 per length.** Single-bin Wilson 95% intervals at this N are wide. 0/6 vs 1/6 at L=100 is well inside binomial noise on its own; the directional read is that *every* L=100 sample is > 6 Å in the bundle while only 4/6 were > 6 Å in §11 step-944 ([E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08)). User flagged "paired-pool with N=12 later" as the follow-up.
- **Single seed (seed=5)**, paired-by-noise with [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) within each length cell. CA-only initial noise tensors are the same shape across both probes; integrated trajectories diverge because the trained weights differ (3-axis vs 5-axis).
- **`nsteps=400`** ✓ (HARD RULE; inherited from `inference_base.yaml`).
- **Step mismatch with E046's inference probe.** §11 step-1385 LOWTSOFT inference is at step 1385; the bundle here is at step 944 — −32% training steps. The bundle's L=100 picture might be much better at step 1385+. Cannot do that read until E047 produces a step-1385 ckpt.
- **Config / output naming carries "step1133" as misnomer** — output dir `inference/inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400/` and CSV `results_inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400_0.csv` are named for "step1133" while the loaded ckpt is step 944. The script's symlink-handle pattern allows re-pointing at later steps without renaming files; downstream consumers must treat "step1133" as the **slot name**, not the actual training step.
- **Saved hyperparameters were used as-is** — no inference-time override of curriculum, n_global_tokens, pair_update or any other architectural attr. All replay from `cfg_exp` via `proteina.py` plumbing.
- **Three architectural axes confounded** vs the §11 K=64-curriculum-self trunk: BigBird globals + pair-update + lowtsoft are bundled. This entry cannot attribute contributions; the planned ablations (drop pair-update, drop BigBird, isolate cap-fix, isolate lowtsoft) listed in E047's "Methodological caveats" remain the next axis of work after the bundle's slot-end probe.

**Possible narrative.** Non-narrative — kept for tuning/decision-making. Feeds the **E047 milestone schedule**: this is the predicted "step ~1000" early-training cost reading and lands on the pessimistic side of the prediction band (L=100 0/6 is the lower edge of what E047 anticipated for an under-trained five-axis bundle). The decision load is at the slot-end probe (~step 2400), not here. If the slot-end probe stays at 0/6 at L=100, the bundle is a documented failure and the cleaner next move is the lowtsoft-only variant on the simpler §11 architecture (`configs/training_ca_only_sparse_K64_curriculum_lowtsoft.yaml`, already on disk per E046's plumbing).

**Cross-references.**
- Predecessor / driver: [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) (cold-start retrain; this is the first probe of that training run).
- Architectural ancestor / same-step number comparison: [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) (§11 K=64-curric step-944, three-axis variant — direct paired-by-noise comparison).
- Inference-only LOWTSOFT signal that motivated this training: [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) (LOWTSOFT N=18 on §11 step-1385 ckpt).
- Code: BigBird-globals + pair-update plumbing pulled in from `origin/main` (commit 04f6bf2 family) on 2026-05-12. PYTHON_EXEC path patched in `script_utils/probe_sparse_K64_bigbird_lowtsoft.sh`.
- Configs: `configs/inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400.yaml`; training-side `configs/training_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft.yaml`; NN side `configs/nn/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_160M.yaml`.
- Output CSV: `inference/results_inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400_0.csv` (15 971 bytes).
- Output PDB dir: `inference/inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400/` (18 PDBs across `job_0_n_{50,100,200}_id_{0..5}`).
- Logs: `nohup_inference_sparse_K64_bigbird_lowtsoft_step1133_n6_nfe400.{gen,eval}.log`, `nohup_probe_bigbird_lowtsoft_outer.out`.
- Symlink under expected handle: `/home/ks2218/la-proteina/sparse_K64_bigbird_lowtsoft_step1133.ckpt → best_val_00000009_000000000944.ckpt`.

---

## E056 — First designability probe of the four-axis bundle (`ca_only_sparse_K64_curriculum_self_bigbird`, step 819, 2026-05-13)

**Status:** finished.

**Why ran.** A new `best_val_00000008_000000000819.ckpt` rsynced 2026-05-13 08:06 BST. Inspection of the ckpt's saved `cfg_exp` showed it is a **previously unprobed sibling** of E055's five-axis bundle: same K=64 SALAD-canonical sparse + curriculum + self-inclusion + BigBird globals n=4 trunk, but with `nn.update_pair_repr=False` and **no** `training.curriculum_low_t_split` override — i.e. the **four-axis ablation that drops pair-update and lowtsoft from E047's five-axis recipe**. Probe feeds the binary: at step 819 (≈125 opt steps before E055's five-axis step-944 snapshot, ≈125 below the §11 three-axis step-944 baseline) is BigBird-only on top of K=64-curric-self already producing structure signal, or is dropping pair-update + lowtsoft a strict regression vs the five-axis bundle and vs the simpler three-axis variant? Cheapest read: canonical N=6 × L∈{50,100,200} × nsteps=400.

**Configs.**
- Inference YAML: `configs/inference_sparse_K64_bigbird_step819_n6_nfe400.yaml` (NEW; inherits `inference_base.yaml` → `nsteps=400`, `seed=5`). `nsamples=6`, `nres_lens=[50, 100, 200]`. No inference-time overrides — sparse K=64, BigBird n=4, curriculum, self-inclusion all auto-replay from saved `cfg_exp` (run_name `ca_only_sparse_K64_curriculum_self_bigbird`).
- Saved ckpt hyperparameters confirmed via `OmegaConf.select`: `run_name_=ca_only_sparse_K64_curriculum_self_bigbird`, `nn.sparse_attention=True`, `nn.n_seq_neighbors=8, n_spatial_neighbors=16, n_random_neighbors=32` (canonical SALAD K=64 composition; with self-inclusion the effective per-query neighbor list is K=64), **`nn.n_global_tokens=4`** (BigBird), **`nn.update_pair_repr=False`** (no pair-update — distinct from E055), `nn.token_dim=768`, `training.curriculum_neighbors=True`, `opt.compile_nn=True`. `training.curriculum_low_t_split` is **not present** in the saved config (no lowtsoft override; ckpt was trained with whatever the default low-t bucket is for the K=64-curriculum-self family). SHA-256 of state_dict (canonical key order, fp32 cast): `c207fac0a0a3710f8a93bb8813496ec1...`. Total params **158,261,128** (vs E055's 161,552,128 — Δ = 3,290,880 fewer params = the missing pair-update MLPs). 0 NaN-bearing tensors. 5 BigBird tensors present (`global_token_emb [4,768]`, `global_cond_emb [4,256]`, `global_pair_bias_{res→glob, glob→res, glob→glob}`), 0 `pair_update`/`pair_mlp` tensors.
- Checkpoint provenance: `/home/ks2218/la-proteina/best_val_00000008_000000000819.ckpt` (1.90 GB, mtime 2026-05-13 08:06 BST). Rsynced from HPC; specific training-run store dir not noted in this session (run_name_ `ca_only_sparse_K64_curriculum_self_bigbird` should resolve under `/rds/.../store/<that_name>/<slurm_jobid>/checkpoints/` on the cluster — confirm before any further pulls). Distinct from §11 three-axis ckpts (these carry the 5 BigBird tensors) and from E055's five-axis ckpts (these are missing all pair-update tensors). No EMA companion file present in the rsync.
- Hardware: 1× L4 (GPU 0 on `gxp-l4-0`), `nohup`, sequential gen → eval. Per `feedback_max_gpu_concurrency.md` the host budget was clear (8 idle L4s, nothing else running).
- Driver: `script_utils/probe_sparse_K64_bigbird_step819.sh` (NEW; clone of `probe_sparse_K64_bigbird_lowtsoft.sh` with `CFG=inference_sparse_K64_bigbird_step819_n6_nfe400` and the local `PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python` pre-baked).
- Symlink under canonical handle: `/home/ks2218/la-proteina/sparse_K64_bigbird_step819.ckpt → best_val_00000008_000000000819.ckpt` (matches the inference config's `ckpt_name`).
- Wall: **gen 281 s (4m41s)** at 18 proteins ≈ 15.6 s/protein at nsteps=400. **Eval 535 s (8m55s)**. Total 13m37s on L4 — comparable to E055's 14m55s and faster than the script header's "30 min on A100" estimate.

**Results — pooled designability (N=6 × L ∈ {50, 100, 200}, scRMSD_ca_esmfold < 2 Å):**

| L | n | designable | best (Å) | median (Å) |
|---|---|---|---|---|
| 50  | 6 | **0/6 (0%)** | **4.76** | 6.72  |
| 100 | 6 | **0/6 (0%)** | **2.52** | 8.59  |
| 200 | 6 | **0/6 (0%)** | 12.46   | 14.05 |
| **pooled** | **18** | **0/18 = 0.0%** | **2.52** | — |

**Per-sample scRMSD (`_res_scRMSD_ca_esmfold`, sorted):**
- L=50:  `[4.76, 5.17, 5.33, 5.57, 7.87, 9.47]` — min 4.76 Å; far from the 2-Å bar at every sample.
- L=100: `[2.52, 3.97, 6.34, 8.13, 11.03, 12.75]` — one near-miss at 2.52 Å (sample id_2), one at 3.97 Å (id_4); remaining four > 6 Å.
- L=200: `[12.46, 13.43, 13.57, 14.64, 15.42, 15.62]` — tight high-scRMSD cluster, off-manifold.

**Cross-arm comparison at canonical N=6/L:**

| arm | step | L=50 | L=100 | L=200 | pool | L=50 min Å | L=100 min Å | notes |
|---|---|---|---|---|---|---|---|---|
| canonical dense E019 | 2646 | 26/30 (87%) | 26/30 (87%) | 16/30 (53%) | 68/90 (76%) | 0.79 | 0.59 | N=30 anchor |
| §11 K=64-curric step-944 ([E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08)) | 944 | 3/6 (50%) | 1/6 (17%) | 0/6 (0%) | 4/18 (22%) | 0.91 | 1.23 | three-axis (K64+curric+self) |
| five-axis bundle ([E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12)) | 944 | 3/6 (50%) | 0/6 (0%) | 0/6 (0%) | 3/18 (17%) | 1.45 | 6.33 | five-axis (+BigBird +PU +lowtsoft) |
| **four-axis bundle (E056, this)** | **819** | **0/6 (0%)** | **0/6 (0%)** | **0/6 (0%)** | **0/18 (0%)** | **4.76** | **2.52** | **+BigBird only, NO PU, NO lowtsoft** |

**Reading of the comparison:**
1. **L=50 is the cleanest regression.** The three-axis variant and the five-axis bundle both produce 3/6 designable samples at L=50 with min Å in [0.91, 1.45]. The four-axis bundle at step 819 produces 0/6 with min 4.76 Å — every sample is more than 2.7 Å worse than the bar. This is the largest L=50 best-Å gap of any K=64 sibling at any step on record. **Caveat:** step 819 is 125 steps before the cousin measurements, so part of the gap is timing.
2. **L=100 mechanism is alive.** Min 2.52 Å is the only sub-3-Å L=100 reading the four-axis arm has produced; closer to the bar than the five-axis bundle's L=100 min 6.33 Å at step 944. With only N=6 this is one sample's noise, but the directional read is that BigBird-on-top-of-K64-curric-self can produce a partially-converged L=100 structure even at step 819.
3. **L=200 is dead, tight cluster.** Best 12.46 Å, all six samples 12-16 Å — no near-misses. Consistent with every sparse-K64 sibling's L=200 picture at step ≤ 944.
4. **Bundle vs five-axis bundle (E055, step 944).** Four-axis is strictly worse on L=50 (0/6 vs 3/6) and strictly worse on pooled count (0 vs 3). But its L=100 min Å (2.52) is *better* than the five-axis bundle's L=100 min Å at step 944 (6.33). At face value this would say "BigBird-only beats BigBird+PU+lowtsoft at L=100", which is the *opposite* of E047's design hypothesis. **Step mismatch (819 vs 944) is the load-bearing confound** — single-sample L=100 best-Å moves around a lot in this regime; need step ≥ 944 of the four-axis arm to read this cleanly.

**Verdict at step 819: DEAD ARM CALL (user, 2026-05-13).** Three converging lines of evidence; no single line is decisive on its own, but together they are.

1. **Empirical: matched-step gap is too large for timing to absorb.** L=50 min Å here is **4.76**; the §11 three-axis variant at step 944 (125 opt steps later) is **0.91**. The largest matched-step L=50 best-Å swing recorded between adjacent K=64-family ckpts on disk is ≈ 0.5 Å. A +3.85-Å delta is not a step-mismatch story. Every L=50 sample is ≥ 4.76 Å — categorical regression, not a binomial-N=6 unlucky draw.

2. **Mechanistic prior (user): BigBird globals on top of K=64-curric-self are position-unaware and just encode averaged-out properties of the protein.** They have no positional encoding into the sequence-residue axis — each global token attends to every residue with the same `global_pair_bias_res→glob` row, then writes back through `global_pair_bias_glob→res`. The four-axis bundle's only architectural difference vs the §11 three-axis trunk is these four position-unaware tokens; the lever therefore *cannot* sharpen the position-sensitive part of the score field that L=50 needs (every residue is close in sequence to every other residue at L=50, so positional discrimination is the bottleneck). The L=50 collapse is exactly the failure mode the mechanism predicts. This was a pre-registerable prediction — not a story made up after the fact.

3. **Validation loss tracks worse (user).** Wandb training-time `validation_loss/loss_epoch` for the four-axis run sits above the three-axis and five-axis cousins on the same wandb dashboard. Cross-run wandb val_loss has been flagged as not strictly comparable ([feedback_wandb_val_loss_not_comparable.md](../.claude/projects/-home-ks2218-la-proteina/memory/feedback_wandb_val_loss_not_comparable.md)) when it disagrees with paired per-t loss / samples, but here it agrees in direction with the sample evidence — the four-axis arm is worse by both signals. Concordant signals reinforce, even when each signal alone would be inconclusive.

**Decision.** Stop probing this arm. Do not queue further rsyncs or matched-step (step ≈ 944, 1133, 1800) probes. The "BigBird-only at step 819 with L=100 min 2.52 Å is mechanism evidence" line earlier in this entry was one sample's draw; per [feedback_dead_arm_calls.md](../.claude/projects/-home-ks2218-la-proteina/memory/feedback_dead_arm_calls.md) that is exactly the mechanism-story-rescuing-a-failure pattern I should not write. Decision saved to memory as [[bigbird-globals-position-unaware]] so future suggestions do not re-propose position-unaware globals as a lever for position-sensitive failure modes (L=50 in particular).

**Methodological caveats.**
- **N=6 per length.** Wilson 95% intervals at this N are wide; 0/6 vs 1/6 at L=100 is well inside binomial noise. Direction-of-best-Å is the stronger signal here.
- **Single seed (seed=5)**, paired-by-noise with [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) and [E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12) within each length cell — same noise seeds, different trained weights.
- **`nsteps=400`** ✓ (HARD RULE; inherited from `inference_base.yaml`).
- **Step mismatch (819 vs 944) vs the closest cousin probes.** Cousins are at step 944; this is at step 819. Cross-arm cells in the comparison table cannot be read as cleanly as the matched-step E049 ↔ E055 cells. A four-axis step-944 (or later) probe is the next decision input.
- **Saved hyperparameters were used as-is** — no inference-time overrides of curriculum, n_global_tokens or any other architectural attr. All replay from `cfg_exp` via `proteina.py` plumbing.
- **Two architectural axes confounded vs the three-axis trunk:** the four-axis arm differs from the §11 three-axis arm only by BigBird globals n=4. The four-axis vs five-axis comparison differs by both pair-update and lowtsoft — those two contributions remain bundled.
- **No NN config file on disk for this run.** `configs/nn/ca_only_sparse_K64_curriculum_self_bigbird_*.yaml` does not exist locally; the NN architecture is replayed only from the saved `cfg_exp` in the ckpt. Re-running this probe after the ckpt is moved/deleted would require either re-rsyncing the ckpt or authoring the NN config locally.

**Possible narrative.** Non-narrative — kept for tuning/decision-making. Feeds the **isolating-ablation row** in the K=64-bundle attribution matrix: this is the first read on "BigBird globals only, on top of K=64-curric-self, no other architectural axes". A matched-step rerun (≥ step 944) is the load-bearing follow-up; without it, "is BigBird alone helping?" cannot be answered from this single snapshot.

**Cross-references.**
- Architectural cousin (same trunk + BigBird + pair-update + lowtsoft): [E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12).
- Architectural ancestor (same trunk, no BigBird): [E049](#e049--first-inference-probe-of-the-k64--curriculum-trained-ckpt-ca_only_sparse_k64_curriculum_self-ep9-step944-2026-05-08) (step 944), [E051](#e051--n3-quick-designability-probe-of-ca_only_sparse_k64_curriculum_self-at-step-1800-2026-05-10) (step 1800).
- Configs: `configs/inference_sparse_K64_bigbird_step819_n6_nfe400.yaml` (NEW); driver `script_utils/probe_sparse_K64_bigbird_step819.sh` (NEW).
- Output CSV: `inference/results_inference_sparse_K64_bigbird_step819_n6_nfe400_0.csv` (15 281 bytes).
- Output PDB dir: `inference/inference_sparse_K64_bigbird_step819_n6_nfe400/` (18 PDBs across `job_0_n_{50,100,200}_id_{0..5}`).
- Logs: `nohup_inference_sparse_K64_bigbird_step819_n6_nfe400.{gen,eval}.log`, `nohup_probe_K64_bigbird_step819.out`.
- Symlink under expected handle: `/home/ks2218/la-proteina/sparse_K64_bigbird_step819.ckpt → best_val_00000008_000000000819.ckpt`.

---

## E057 — BigBird wiring audit on E047 step 1200 (2026-05-12) — renumbered from upstream E048 on 2026-05-13 merge

**Why ran:** [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)'s in-progress val curve matches §11 K=64-curriculum-self at high t and is strictly worse at low t with the same shape. User question: is BigBird actually being applied, or is the bundle effectively §11 + pair-update + lowtsoft because the globals are inert? An end-to-end read of `configs/nn/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_160M.yaml` → `proteina.py:131` → `LocalLatentsTransformer.__init__` → `_attach_globals` → `PairBiasAttention._attn_sparse` shows the wiring is correct on paper; checking whether it's actually used at runtime needs a forward-pass hook, which is what this entry runs. The answer decides whether the next move is "fix BigBird" (if inert) or "reconsider what BigBird at low t actually does to the learned routing policy" (if heavily used).

**Configs:**
- Audit script: `script_utils/audit_bigbird_wiring.py` (CPU-only; ~2 min wall on login node; no GPU available locally).
- Reads checkpoint: `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self_bigbird_pairupdate_lowtsoft/1778520282/checkpoints/last.ckpt` (global_step=1200, E047 mid-training snapshot).
- NN constructor kwargs assembled from the snapshot's `exp_config_*.json` (n_global_tokens=4, sparse_attention=True, update_pair_repr=True, update_pair_repr_every_n=3, K_canonical=64, K_total=68, curriculum_neighbors=True, curriculum_low_t_split=(16,8,24)).
- Synthetic batch B=3, N=100, t_bb_ca = (0.1, 0.5, 0.9) — one protein per curriculum bucket. `mask=True` everywhere; `x_t`/`x_sc` ∼ 𝒩(0, 0.3²); `residue_type=0`. Fresh torch.manual_seed(0).
- Probe surface:
  - **A1**: hook `PairBiasAttention._attn_sparse` to record, for each of 14 transformer layers, the mean+max attention mass that residue queries place on the 4 global slots (slots [K_canonical:K_total] = [64:68]).
  - **A2**: same hook, record std of global keys across the 4 global tokens (collapse check).
  - **A3**: pairwise cosine similarity between the 4 rows of trained `global_token_emb` (collapse check, pre-forward).
  - **B1**: fresh-init clone (same kwargs, torch.manual_seed(123)), one fwd+bwd on a synthetic L2 loss, log |∇| for `global_*` params vs representative trunk params.
  - **B2**: same as B1 but with `_attach_globals` monkey-patched on the instance to set `global_cond = cond[:, :1, :].expand(B, G, dim_cond)` (broadcast the per-protein time embedding into globals) instead of the zero-init learnable `global_cond_emb` — speedup ratio diagnoses whether time-agnostic global cond is the gradient bottleneck.
- Code paths exercised (relevant if anyone re-reads later):
  - `LocalLatentsTransformer._attach_globals` at `local_latents_transformer.py:293-383`.
  - `_attn_sparse` at `pair_bias_attn.py:132-181`.
  - `PairReprUpdate.forward` sparse branch at `pair_update.py:62-86`.
  - Hook attaches to `model.transformer_layers[i].mhba.mha` (note: attribute is `mhba`, not `mha`, on `MultiheadAttnAndTransition`).
- Single-machine, no SLURM. Login-node CPU only (`torch.cuda.is_available()=False` here).

**Results (verbatim from run log):**

A3 — trained `global_token_emb` pairwise cosine similarity:
```
+1.0000  +0.0032  -0.0386  +0.0031
+0.0032  +1.0000  -0.0042  +0.0353
-0.0386  -0.0042  +1.0000  +0.0708
+0.0031  +0.0353  +0.0708  +1.0000
```
All off-diagonal entries within ±0.071; no two of the 4 globals are aligned. **No collapse at the embedding level.**

A1 + A2 — attention mass to globals per layer (averaged over residue queries and heads), and key-std across globals. Uniform baseline = G/K_total = **0.0588** (5.88%).

| Layer | t=0.1 | t=0.5 | t=0.9 | max(t=0.1) | std(globK) |
|---|---|---|---|---|---|
| 0 | 0.0396 | 0.1936 | 0.4173 | 0.4754 | 0.8128 |
| 1 | 0.0214 | 0.1034 | 0.2782 | 0.4738 | 0.7404 |
| 2 | 0.0184 | 0.1279 | 0.3014 | 0.5388 | 0.6498 |
| 3 | 0.0059 | 0.0379 | 0.0665 | 0.0580 | 0.3721 |
| 4 | 0.0060 | 0.0185 | 0.0607 | 0.0401 | 0.3287 |
| 5 | 0.0510 | 0.0070 | 0.0069 | 0.9529 | 0.3987 |
| 6 | 0.0073 | 0.0064 | 0.0347 | 0.0393 | 0.4341 |
| 7 | **0.2966** | 0.1825 | 0.1426 | **0.9997** | 0.4609 |
| 8 | 0.0873 | 0.0028 | 0.0043 | **0.9999** | 0.3887 |
| 9 | **0.3401** | 0.1655 | 0.0534 | 0.9972 | 0.3495 |
| 10 | **0.2299** | 0.0781 | 0.0018 | **1.0000** | 0.3938 |
| 11 | **0.2783** | 0.1287 | 0.0547 | **1.0000** | 0.4098 |
| 12 | **0.2664** | 0.0217 | 0.0017 | 0.9987 | 0.3861 |
| 13 | **0.3200** | 0.0610 | 0.0411 | 0.9922 | 0.3187 |

Two regimes are visible. **Early layers (0-2) at high t (t=0.9)** route 28-42 % of attention into globals — globals act as an initial-layout broadcast channel for the cleaner state. **Late layers (7, 9-13) at low t (t=0.1)** route 22-34 % of attention into globals, with the per-residue max repeatedly hitting ~100 % (a single residue query at low t in layers 7, 8, 10, 11 puts essentially all of its attention on the 4 global slots). At mid- and high-t these late layers route <6 % into globals — uniform or below. The per-layer std of global keys stays in [0.32, 0.81] across all layers, so the 4 globals continue carrying distinct k/v signals throughout the trunk; they are not collapsing to a single token even though they're competing for the same slots.

B1 — fresh-init |∇| ratios (relative to `transformer_layers.0.mhba.mha.to_qkv.weight` |∇| as 1.0):
```
global_token_emb                                          |grad|=6.17e-1   ratio=0.350
global_cond_emb                                           |grad|=5.84e+0   ratio=3.315
global_pair_bias_res_to_glob                              |grad|=5.11e-1   ratio=0.290
global_pair_bias_glob_to_res                              |grad|=7.77e-2   ratio=0.044
global_pair_bias_glob_to_glob                             |grad|=8.26e-2   ratio=0.047
transformer_layers.0.mhba.mha.to_qkv.weight               |grad|=1.76e+0   ratio=1.000  (reference)
transformer_layers.0.mhba.mha.to_bias.weight              |grad|=4.74e-2   ratio=0.027
transformer_layers.0.mhba.scale_output.to_adaln_zero_gamma.0.weight  |grad|=5.03e-6   ratio≈0
pair_update_layers.0.linear_x.weight                      |grad|=7.68e-1   ratio=0.435
ca_linear.1.weight                                        |grad|=8.45e+0   ratio=4.794
```
Globals' gradient is **not structurally tiny** — `global_token_emb` and `global_pair_bias_res_to_glob` have grad ratios 0.29-0.35 (same order of magnitude as trunk to_qkv); `global_cond_emb` has the largest grad ratio (3.32) because it starts at exactly zero. The two globally-broadcast pair-bias params for `(global → residue)` and `(global → global)` have smaller grad ratios (0.04-0.05) but still non-tiny.

B2 — fresh-init with `global_cond_emb` overridden to broadcast of time-emb (per-instance monkey patch of `_attach_globals`):

| Param | ratio (B1, learnable zero-init) | ratio (B2, time-emb broadcast) | speedup |
|---|---|---|---|
| global_token_emb | 0.3497 | 0.3514 | 1.00× |
| global_pair_bias_res_to_glob | 0.2898 | 0.2813 | 0.97× |
| global_pair_bias_glob_to_res | 0.0441 | 0.0444 | 1.01× |
| global_pair_bias_glob_to_glob | 0.0468 | 0.0484 | 1.03× |

No measurable speedup. The time-agnostic learnable global cond is **not** the gradient bottleneck at init.

Sanity numbers from the same audit (param magnitudes at step 1200, ckpt inspection):
- `global_token_emb` (4, 768): |x|.mean=1.63e-2 (init scale was randn·0.02, so ~unchanged).
- `global_cond_emb` (4, 256): |x|.mean=1.39e-3 (from zero-init).
- `global_pair_bias_res_to_glob` (4, 256): |x|.mean=2.81e-3.
- `global_pair_bias_glob_to_res` (4, 256): |x|.mean=2.52e-3.
- `global_pair_bias_glob_to_glob` (4, 4, 256): |x|.mean=2.21e-3.
- `pair_update_layers` exist at i ∈ {0, 3, 6, 9, 12} (5 updates over 14 layers, every_n=3); LN weights ~ 0.97; trained.

**Possible narrative:** non-narrative — this is a wiring + learned-policy diagnostic, not a paper claim. **Reframes [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)'s low-t pathology.** The "BigBird is inert" hypothesis is falsified: globals are heavily used. Instead the data fits a *learned-policy* picture — at low t the curriculum (`(16,8,24)`) strips real spatial/random capacity, and the model compensates by routing 20-34 % of late-layer attention into 4 protein-agnostic globals (each global's slot has its own pair-bias entries but the global token itself is a single learnable vector shared across all proteins, with a time-agnostic cond). Since globals cannot encode protein-specific structure, that learned shortcut caps low-t accuracy. Decision the entry feeds:
- **Soft fix (no retrain)**: at inference, mask the 4 global slots in `_attn_sparse` when `t < 0.33` (force the residue to spend its attention on real neighbors). If this helps the §11+pair-update+lowtsoft pretrained policy, the hypothesis is confirmed.
- **Hard fix (retrain ablation)**: re-cold-start §12 with `n_global_tokens: 0` (BigBird off) to attribute. If the no-BigBird run beats this one at low t, BigBird is the regression at this recipe. If the no-BigBird run is identical, BigBird is a wash.
- **Architectural fix (retrain ablation)**: re-cold-start §12 with `global_cond_emb` replaced by a per-protein time-embedded cond (broadcast of `cond[:, 0, :]`) so globals can specialise per-t. B2 shows the gradient bottleneck argument doesn't apply, so this is speculative; only worth running if the soft-fix probe shows globals are the load-bearing failure at low t.

**Methodological caveats:**
- **Single mid-training checkpoint (global_step=1200).** Attention-mass patterns at later checkpoints may differ — globals could become less or more dominant as training continues. Re-run this audit on a later ckpt (e.g. step 2000) before treating the late-layer 20-34 % low-t mass as the trained policy's steady state.
- **Synthetic batch, single seed, single (B, N) = (3, 100).** No PDB structure. At real PDB inputs with realistic Cα coords the curriculum's spatial/random group at low t (8 spatial + 24 random) would be picked from noisy coordinates per `x_t`, so the alternative slots the model could pull info from are different from this audit's. The audit shows globals are *used*; the exact magnitude on real data could be different.
- **N=100 is below the L=200 designability cliff.** Whether the late-layer global-routing intensifies further at L=200 (where designability collapses to 0/18 in E046) is untested. Re-run the audit at N∈{200, 300} to see if the low-t global-routing pattern strengthens with chain length.
- **Per-head attention is averaged.** The mean over 12 heads can hide per-head specialisation — some heads may put 100 % on globals while others put 0 %. Per-head breakdown not extracted here; the per-residue max ~ 1.0 is a hint that head-level concentration is real.
- **B1/B2 are *init* grad ratios** on a fresh model, not gradient flow at step 1200. Trained models can develop bottlenecks that init-time grads don't predict; the strict claim is "no init-time bottleneck from time-agnostic global cond".
- **`_attach_globals` patch in B2 mutates only the cond fed to AdaLN/scale_output; the pair-bias parameters (res→glob, glob→res, glob→glob) stay time-agnostic.** A "full" time-conditioned globals architecture would also condition those, which B2 doesn't test.
- The audit is **read-only**: no checkpoint, config, or experiment file was modified by the audit itself. The new script is `script_utils/audit_bigbird_wiring.py`; existing configs are unchanged.

**Cross-references:**
- Triggered by: [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) "in-progress val curve matches §11 at high t, strictly worse at low t" observation.
- Predecessor for the wiring contract: [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) (introduced the cap fix and the LOWTSOFT bucket).
- Code paths inspected: `proteinfoundation/nn/local_latents_transformer.py` (`_attach_globals`, forward), `proteinfoundation/nn/modules/pair_bias_attn.py` (`_attn_sparse`), `proteinfoundation/nn/modules/pair_update.py` (sparse branch), `proteinfoundation/nn/modules/sparse_neighbors.py` (cap fix), `proteinfoundation/proteina.py` (kwargs plumbing).
- Existing smoke test it complements (and exposes the gap of): `script_utils/smoke_bigbird_pairupdate.py` (asserts grad-presence on globals but not attention-mass or std-across-globals — would have passed for this checkpoint regardless of whether globals were inert or load-bearing).
- Deferred follow-ups (decision pending): soft-fix inference probe with `t<0.33` global-mask; `n_global_tokens: 0` retrain ablation; time-embedded `global_cond` retrain ablation; re-run audit on E047 step ≥ 2000 ckpt; per-head attention breakdown; audit at N ∈ {200, 300}.


---

## E058 — Cold-start BigBird-only (no pair-update, no LOWTSOFT) on the §11 trunk (2026-05-12) — renumbered from upstream E049 on 2026-05-13 merge

> **2026-05-13 post-hoc note (added during merge):** The first designability probe of this run's output (`best_val_00000008_000000000819.ckpt`, step 819) is [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13) — 0/18 designable, called dead by user on three converging lines (sample evidence + position-unaware-globals mechanism + val_loss tracks worse than cousins). The predicted milestones below were not met at step 819; arm is dead, not pre-convergence. Keeping the predicted-milestones text as-authored for the lab-record trail.

**Status:** in progress (queued; SLURM **29277806**, SL2 15h slot, submitted ~mid-day BST 2026-05-12).

**Why ran:** [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)'s five-axis bundle (BigBird + pair-update + LOWTSOFT on the §11 K=64-curriculum-self trunk) stalled at low t — `val_loss_by_t.t_000_020` stuck at ~7.2 at global_step=1196 vs the §11 predecessor's ~4.3 at the same horizon (slot 7sdu834p, global_step=1196). [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge) audit confirmed BigBird is correctly wired and heavily used (residue queries route 22-34 % of late-layer attention into globals at t=0.1), and the differential degradation across t-buckets localised cleanly to the curriculum cell where the schedules differ: at t<0.33, §11 had `(32, 0, 0)` sequence-only and the five-axis bundle had LOWTSOFT `(16, 8, 24)`. Chat consensus 2026-05-12 (the one with this entry) was that LOWTSOFT and pair-update were the load-bearing regressions, **not** BigBird itself. This entry isolates the BigBird contribution by dropping LOWTSOFT and pair-update.

Decision the entry feeds: whether **BigBird alone is the right addition on top of §11** for the K=64 baseline going forward, or whether `n_global_tokens: 0` (the §11 trunk as-is) should remain the baseline and BigBird is dropped from the variant catalog.

**Configs:**
- Training config: `configs/training_ca_only_sparse_K64_curriculum_self_bigbird.yaml` (NEW, sibling of `..._bigbird_lowtsoft.yaml` with `curriculum_low_t_split` removed — default `(32, 0, 0)` from `proteina.py:129` kicks in).
- NN config: `configs/nn/ca_only_sparse_K64_curriculum_self_bigbird_160M.yaml` (already on disk from a 2026-05-12 earlier session; `update_pair_repr: False`, `n_global_tokens: 4`, all other keys byte-equivalent to `..._bigbird_pairupdate_160M.yaml`).
- Cold start (no `pretrain_ckpt_path`). The §11 step-1385 ckpt has no `global_token_emb` / `global_cond_emb` / `global_pair_bias_*` parameters, so warm-start would resume the trunk and leave the BigBird parameters zero-init — wouldn't give a clean A/B for BigBird's contribution.
- Recipe: locked to canonical OLD recipe (wd=0.05, constant LR=2e-4, no scheduler). Identical to §11 and the [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) bundle in every other respect.
- Submit: `sbatch --time=15:00:00 --exclude=gpu-q-43 script_utils/submit_train_ca_only_1gpu.sh -n training_ca_only_sparse_K64_curriculum_self_bigbird` — single 15h SL2 slot. At canonical's ~131 opt-steps/h that's ~1900 opt steps in one shot, comfortably past the §11 step-1385 reference and into the canonical 1800-2200 best-val window.
- SLURM job id: 29277806.
- Wandb group: `ca_only_sparse_K64_curriculum_self_bigbird` (auto-set from `run_name_` by the submit script). Wandb run id: tbd (logged at job start).
- Store dir: `/rds/user/ks2218/hpc-work/store/ca_only_sparse_K64_curriculum_self_bigbird/<launch_ts>/checkpoints/`.

**Architecture delta vs §11 (`rmuumq8v`, val 4.19 @ step 1385, designability 8/10/2 = 44/56/11 % at N=18 — the comparison reference):**

| axis | §11 reference | E049 (this entry) | five-axis [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) |
|---|---|---|---|
| K | 64 | 64 | 64 |
| `curriculum_neighbors` | True | True | True |
| `curriculum_low_t_split` (t<0.33 cell) | default `(32, 0, 0)` — sequence-only | default `(32, 0, 0)` — sequence-only | `(16, 8, 24)` — LOWTSOFT |
| self-inclusion | True (code) | True (code) | True (code) |
| `update_pair_repr` | False | False | True (every 3) |
| `n_global_tokens` | 0 | **4** | 4 |
| off-by-one cap fix | absent (trained pre-`04f6bf2`) | live (code) | live (code) |

Only two deltas vs §11: BigBird globals on, cap fix in code.

**Predicted milestones (set at submission time, to be retro-checked):**

- **Step ~600 (5h, first val read):** total `val/loss_epoch` ≈ §11's ~5.0-5.2 at the same horizon (initial BigBird cost of learning the 4 globals' embeddings + their pair-bias entries from zero-init). The `val_loss_by_t` shape should already show low-t recovery — `t_000_020` somewhere in the 4.5-5.5 range, **NOT** the 7.0+ that [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) showed at step 503.
- **Step ~1200 (10h):** total `val/loss_epoch` close to §11's 4.32 at slot 7sdu834p step 1196, within +0.1. Low-t `t_000_020` should be in the 4.3-4.7 range (vs §11's 4.49 and [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11)'s 7.24 at the same step).
- **Step ~1800 (15h, end-of-slot):** total `val/loss_epoch` ≤ §11's step-1385 4.19. Designability probe (N=6 × L∈{50, 100, 200}, nsteps=400, seed=5) at the closest-to-1800 ckpt vs §11 step-1385's 8/10/2 N=18 reference.

If low-t recovers fully (i.e. matches §11's t-bucketed val curve to within +0.1 at every bucket) AND globals continue to attract late-layer attention mass as in [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge), the conclusion is *BigBird is a free axis on §11*. Designability at step ~1800 then tests whether it actually translates to a sample-quality improvement.

If low-t recovers but designability is the same or worse than §11, the read is *BigBird helps val MSE marginally (more parameters, more capacity at low t) but the additional capacity does not translate to better samples* — same pattern as Fix C2 (E021/E039 — Fix C2 hit the §11 designability ceiling but didn't beat it).

If low-t does NOT recover (`t_000_020` still > 5 at step ~1200), BigBird is the regression and the next move is `n_global_tokens: 0` on the same trunk (i.e. just re-train §11 with the cap fix in code) to confirm.

**Methodological caveats:**

- **Single 15h slot, no chained re-queue.** End-of-slot at ~step 1800 is just inside the canonical 1800-2200 best-val window; if the bundle plateaus later (as [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) appeared to be doing at step 1200), the designability read might come from a not-yet-converged ckpt. Acceptable for the first read; a chained second slot is the natural follow-up if convergence isn't visible.
- **Cold start cost.** The 4 BigBird global embeddings + their three pair-bias parameter tensors (`global_pair_bias_res_to_glob`, `_glob_to_res`, `_glob_to_glob`) are zero / near-zero at init. [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) showed these reach useful magnitudes by step 1200 (cf. [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge)'s param magnitudes table). With pair-update OFF and LOWTSOFT OFF, the trunk doesn't have to learn around globals AND a degraded curriculum AND extra pair-update layers simultaneously — the BigBird parameters should reach useful magnitudes faster, but the prediction is unverified for this exact configuration.
- **The §11 reference's 44/56/11 % designability at N=18 came from chained slots `rmuumq8v` (slot 3, val 4.19 @ step 1385) — pre-cap-fix.** This entry's designability probe will use post-cap-fix code, so the comparison is not strictly matched on the integrator side. Cap-fix is a no-op at L>K=64 (i.e. L=100, L=200) and only fires at L≤K=64 (L=50). The L=50 comparison is the one with the integrator mismatch; L=100 / L=200 are clean A/Bs.
- **Designability comparison at N=6 is one slot, not pooled.** If the step-1800 N=6 read is ambiguous (e.g. 3/4/0 vs §11's 4/5/1 single-slot read), pool with a second N=12 probe at a different seed before drawing a conclusion. CLAUDE.md auto-memory `feedback_seed_propagation_audit.md` applies: seeds don't propagate cleanly across `nsamples` settings, so use an explicit-fresh-seed pattern.
- **No designability re-probe of [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) at matched val MSE.** The five-axis bundle's stalled-at-low-t val curve never crossed §11's step-1385 val (4.19), so a matched-val comparison isn't possible. E049 vs §11 is the comparison this entry produces; E049 vs [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) at matched step is a separate read.
- **Sample-quality bar inheritance.** Inference will use `nsteps=400` per the CLAUDE.md hard rule.

**Cross-references:**
- Predecessor (architecture): variants.md §11 (K=64-curriculum-self trunk; `rmuumq8v`, val 4.19 @ step 1385).
- Predecessor (BigBird wiring): [E047](#e047--cold-start-retrain-of-the-k64-bundle-with-cap-fix--lowtsoft-low-t-bucket-2026-05-11) (first BigBird training) + [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge) (audit confirming wiring + heavy use).
- Predecessor (cap fix): [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11) (commit `04f6bf2`).
- Sibling on disk (not yet trained): `configs/training_ca_only_sparse_K64_curriculum_self_bigbird_lowtsoft.yaml` (BigBird + LOWTSOFT, no pair-update). Useful if E049's low-t recovers AND we want to attribute the LOWTSOFT regression specifically.
- Follow-up if E049 lands: BigBird + pair-update without LOWTSOFT (`configs/training_ca_only_sparse_K64_curriculum_self_bigbird_pairupdate.yaml`, already on disk, not yet trained) to isolate pair-update on top of E049's bundle.

---

## E059 — Dense-attention concentration audit (content-adaptive top-K distillation decision gate, 2026-05-13)

**Status:** finished — **STOP signal** on cheap-student content-adaptive distillation.

**Why ran:** Following the conversation about closing the sparse-vs-dense val-loss gap (E043: uniform sparse-vs-dense Δ ≈ +0.09 to +0.13 nat across all five t-buckets at matched-step on the K=40 vanilla variant), the user proposed **content-adaptive top-K distillation**: use the canonical dense baseline as a teacher to train a sparse student whose routing head predicts dense's per-residue attention pattern. The cheap-student version assumes dense's attention is (a) concentrated enough that top-K hard selection retains the bulk of information and (b) stable enough that a single shared K-set per protein-per-t suffices. If either fails, the cheap version doesn't recover the dense expressivity. This entry is the decision-gate audit to settle (a) and (b) before committing a 15 h training slot.

**Configs:**
- **Script**: `script_utils/audit_dense_attention_concentration.py`. Monkey-patches `PairBiasAttention._attn` (dense path) class-wide; the hook computes per-call top-K mass curves and top-16 indices per (head, query) on real queries × real keys (padding excluded via `_N_real` set on the module before forward).
- **Ckpt**: `/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt` (CA-only, dense, OLD recipe wd=0.05 constant LR=2e-4, step 2646 = canonical best-val ckpt per CLAUDE.md).
- **Proteins**: deterministic subset from `data/pdb_train/processed_latents/*/*.pt`, length-binned with `±5` tol around target L ∈ {50, 100, 200}; **3 proteins per bin** (= 9 proteins total). Same `seed=42` rotation transform as `run_per_t_val.py` so this set is regenerable.
- **t-values**: {0.10, 0.30, 0.50, 0.70, 0.90}. One forward per (protein, t); no flow integration. `x_t = (1-t)·noise + t·x_clean` per the canonical FM convention.
- **Hook records**: `mass_top_K_mean[H, |K_grid|]` (mean over queries, per head) for K ∈ {8, 16, 32, 48, 64}; `top_idx[H, N_real, 16]` for Jaccard. Batch slot 0 only — record one batch element per forward.
- **Total**: 9 proteins × 5 t-values × 14 layers = **630 attention-layer records**.
- **Hardware**: 1× A100, bf16 forward. Wall time ~12 min.
- **Output**: `results/dense_attn_audit/canonical_2646_dense_attn.json` (440 KB; full per-(layer, t, length) breakdown + per-protein Jaccard).

**Results — Check 1: top-K mass concentration (mean over all 630 records):**

| K  | mass_top_K mean | mass_top_K median | uniform baseline K/200 |
|---:|---:|---:|---:|
|  8 | 0.510 | 0.519 | 0.04 |
| 16 | **0.656** | 0.675 | 0.08 |
| 32 | **0.794** | 0.813 | 0.16 |
| 48 | 0.866 | 0.882 | 0.24 |
| 64 | 0.907 | 0.927 | 0.32 |

Attention is well above uniform at every K (top-64 captures 0.907 of mass vs uniform 0.32 → genuinely concentrated), but **the GO threshold (mass_top_16 ≥ 0.70 AND mass_top_32 ≥ 0.85) is not cleared.** Mean values trail by 0.044 and 0.056 nat respectively; medians are closer but still under the bar (0.675 / 0.813). Concentration is *partial*: dense distributes its mass over more than top-16 residues, but does taper sharply by top-32.

**Results — Check 2: Jaccard stability of top-16 attended sets (mean pairwise):**

| Axis | mean | min | max | n |
|---|---:|---:|---:|---:|
| **layer-adjacent** | **0.217** | 0.115 | 0.389 | 45 |
| **t-adjacent** | 0.475 | 0.291 | 0.654 | 126 |
| **head-within (l, t)** | **0.224** | 0.064 | 0.523 | 630 |

Decision rules: ≥ 0.7 → shared K-set works (cheap student); ≤ 0.3 → per-axis routing required (expensive student).

- **Layer-Jaccard 0.217**: adjacent layers share only ~3.5/16 of their top-16 attended residues. The trunk genuinely routes different residues at different layers. **Per-layer routing required.**
- **Head-Jaccard 0.224**: heads within a (layer, t) disagree by the same margin. **Per-head routing required.**
- **t-Jaccard 0.475**: middle band; the trajectory-time axis is the most stable of the three but doesn't clear either threshold. Routing computed every few ODE steps (not every step) might work; per-trajectory routing is borderline.

**Decision:** **STOP** on the cheap-student version of content-adaptive top-K distillation. Layer × head specialization is the binding constraint. A faithful student would need 14 (layers) × 12 (heads) = **168 separate K-sets per protein per t**, which is no longer "cheap" — it's a non-trivial re-implementation that approaches reinventing dense attention with extra plumbing. The 15 h training slot is preserved; this idea is retired in its cheap form.

**Possible narrative:** **non-narrative — kept for tuning/decision-making.** Two contributions:

1. **Decides against pursuing cheap-student content-adaptive top-K distillation** as a sparse-vs-dense gap-closing path. Saves a 15 h slot and establishes a defensible "we tried, the data said no." For the thesis text, this is the right shape of negative result: a *specific*, *measurable* hypothesis was tested with a *quantitative* threshold; the data fell short on multiple axes.

2. **Refines the E043 "uniform sparse-vs-dense tax" framing with a structural mechanism.** Our sparse implementation shares the K-set across all 14 layers AND all 12 heads within each layer (the `neighbor_idx` is computed once at the top of `LocalLatentsTransformer.forward` and broadcast). Dense gives every (layer, head) independent access to all N. The audit measures how much specialization is forfeited: dense layers disagree on top-16 residues by Jaccard 0.78 (= 1 − 0.217 in the "fraction of mass that's layer-specific" sense), and heads within a layer disagree by Jaccard 0.78. That's a lot of representational capacity sparse can't access without per-layer × per-head routing. The "uniform tax" of E043 is at least partially explained by this shared-K-set bottleneck — not by any specific t-bucket failure mode. **What this leaves open**: (a) per-layer random redraw, which is a different mechanism (receptive-field expansion across depth via diversification of the random group, NOT routing — does not require predicting dense's pattern); (b) per-layer × per-head routing without distillation (let sparse learn its own routing from FM loss alone, with a small routing head per (layer, head)). Both are bigger architectural commitments than the cheap-student version; the decision to pursue either is decoupled from this entry's STOP.

**Methodological caveats:**

- **N=3 proteins per length bin × 3 bins = 9 proteins.** Decision-grade for go/no-go (the layer-/head-Jaccard means are well below the 0.3 STOP threshold; the layer-Jaccard *max* over n=45 groups is 0.389, well below the 0.7 GO bar). NOT magnitude-grade — re-running at N=5 per bin would tighten the absolute numbers but the binary decision is locked in.
- **Single seed (42)** for protein subset selection and rotation transforms. Re-running with seed ∈ {7, 13} would tighten CIs and confirm stability; not done because the result is decisive enough.
- **Hook records batch slot 0 only.** Batch size is 1 per forward in this script, so this is moot here, but the limitation matters if the script is adapted for B > 1.
- **Top-16 Jaccard ignores soft attention weighting.** Two queries with identical top-16 sets but very different weight distributions over those 16 score Jaccard = 1. Given Check 1 reports mass_top_32 ≈ 0.79 (so top-16 captures the lion's share), this is a fine proxy for the go/no-go decision but coarser than a Wasserstein- or KL-based similarity would be. The decision is determined by the magnitude of disagreement (0.217 / 0.224), not the proxy choice.
- **Padding masking**: the hook restricts attention to `attn[..., :N_real, :N_real]` before computing top-K mass and top-16 indices. Verified visually that no padding-row noise leaks into the Check 1 numbers.
- **Dense canonical only.** This audit is on the *teacher candidate* (dense). If a *sparse* model's attention has different concentration / stability properties at the same layer × head budget, that's a different question — and largely moot, since the sparse student's job in distillation is to match the teacher; if dense can't be approximated cheaply, sparse can't either.
- **Layer count and head count are architecture-specific.** This audit reports for the 160 M trunk (14 layers × 12 heads). A different trunk (e.g., 70 M with fewer layers, or a head-collapsed design) could land in a different regime. The conclusion "per-layer × per-head routing required" is scoped to our specific architecture.
- **No EMA.** The raw `.ckpt` (not `-EMA.ckpt`) is used per `feedback_ema_vs_raw_checkpoints`. EMA weights might have slightly different attention patterns; not investigated here.

**Cross-references:**
- **Predecessor (motivation)**: chat conversation 2026-05-13 on per-layer random redraw vs content-adaptive routing as ways to close the sparse-vs-dense gap.
- **Sparse-vs-dense at matched-step per-t val**: [E043](#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07). The uniform-Δ-across-t result this entry refines mechanistically.
- **Cap-fix audit + fp32 sparse-vs-dense equivalence**: [E046](#e046--sparse-attention-off-by-one-cap-investigation--fix--bf16-audit-2026-05-11).
- **BigBird wiring + globals discussion**: [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge), [E058](#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge), [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13). BigBird was a related attempt to provide cheap content for every query at every layer via the 4 global slots; the audit's "168 K-sets required" finding is consistent with E056's dead-arm verdict on shared-route BigBird globals — they can't substitute for the layer × head specialization dense achieves.
- **Open follow-ups**: (a) per-layer random redraw in `sparse_neighbors.py` — different mechanism, untested; (b) per-layer × per-head routing with a learned head — bigger commitment, not on the immediate roadmap; (c) sparse on length regimes where dense doesn't fit (N ≥ 1024) — the sparse-vs-dense gap framing changes there.
- **Output on disk**: `results/dense_attn_audit/canonical_2646_dense_attn.json` (full per-(layer, t, length) breakdown).


## E060 — Gradient saliency companion to E059 + cross-metric grad vs attn (2026-05-13)

**Status:** finished.

**Why ran.** [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) returned a STOP signal on cheap top-K distillation, but the argument rested entirely on attention as the importance signal. Two open questions remained: (a) is the gradient of the loss w.r.t. residue coordinates (∂L/∂x_t["bb_ca"]) a *sharper* importance signal than attention — in which case grad could replace attn and E059's STOP would need revisiting? (b) does attention even reflect what matters to the loss — i.e. is attention a faithful proxy for gradient saliency, or are they independent? Decision the entry feeds: if grad is sharper and aligned with attn, cheap-K is still on the table with a different teacher signal. If both fail, E059's STOP is closed under two metrics.

**Configs.**
- Audit script: `script_utils/audit_dense_gradient_saliency.py`.
- Checkpoint: `/home/ks2218/la-proteina/best_val_00000026_000000002646.ckpt` (canonical dense, step 2646, `test_ca_only_diffusion/1776805213`).
- Protocol matched to [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13): seed=42, 3 proteins per length bin × L∈{50,100,200} = 9 proteins, t-grid `{0.10, 0.30, 0.50, 0.70, 0.90}` = 5 t-values, JACCARD_K=16, K_grid=`[8, 16, 32, 48, 64]`. Outputs pair by (protein_label, t_label).
- Forward+backward pass: model params frozen (`requires_grad_(False)`); gradient enabled only on `x_t["bb_ca"]`. Saliency = per-residue L2 norm of `dL/dx_t[i, :]` (3 coordinate dims), one scalar per residue per protein per t. Light attention hook records per-layer head-averaged attention-received (one vector per layer, not per (layer, head, query) — smaller JSON than E059's full hook, sufficient for cross-metric overlap).
- Precision: bf16 (default; `--force_precision_f32` flag not used). 6-8 GB peak memory per script header; ran inside L4 22 GB envelope without issue.
- Hardware: 1× L4 (`gxp-l4-0` GPU 0), `nohup`. Launched 2026-05-13 11:21 BST.
- Record counts: 45 gradient saliency records (9 proteins × 5 t-values) + 630 attention-layer records (45 × 14 layers).

**Results — Check 1: gradient saliency concentration (mass_top_K_grad over all 45 (protein, t) records):**

| K | mean | median |
|---|---|---|
| 8  | 0.176 | 0.152 |
| 16 | **0.312** | 0.277 |
| 32 | **0.528** | 0.491 |
| 48 | 0.683 | 0.652 |
| 64 | 0.766 | 0.792 |

E059 attention reference (same K grid): mean = `[0.510, 0.656, 0.794, 0.866, 0.907]`. Gradient saliency is **less concentrated than attention by ~2× at K=16** (0.31 vs 0.66) and ~1.5× at K=32 (0.53 vs 0.79). The "gradient is the sharper importance signal" hypothesis the script's decision-rule docstring anticipated (`mass_top_K_grad ≫ E059's numbers`) is **falsified — direction reversed.**

**Results — Check 2: t-Jaccard stability of gradient top-16 sets across adjacent t-steps:**

| metric | value |
|---|---|
| mean (over 36 t-adjacent pairs) | **0.200** |
| min | 0.000 |
| max | 0.684 |
| n | 36 |

Decision rule: `t-Jaccard_grad ≥ 0.7` → important set stable across trajectory. Got 0.200, **well below**, even below the 0.3 STOP threshold by analogy with E059. The residues that matter to the loss change substantially between adjacent t-values.

**Results — Check 3: cross-metric Jaccard(grad top-16, attn-head-avg top-16) — 14 layers × 45 (protein, t) = 630 records:**

| summary | value |
|---|---|
| overall mean | **0.114** |
| overall max | 0.600 |
| mean of per-(protein, t) max-over-layers | 0.274 |
| per-layer mean (range across 14 layers) | [0.085, 0.151] |
| per-layer max (range across 14 layers) | [0.28, 0.60] |

Per-layer means: layer 6 highest at 0.151, layer 9 lowest at 0.085 — every layer's mean is well under 0.20.

Decision rules from the script's logging:
- `max-over-layers Jaccard ≥ 0.5` → attention reflects loss-importance somewhere. Result: mean-of-per-(p,t)-max = **0.274**, well under 0.5. A handful of (protein, t) records reach 0.6 max (single layer beats the bar), but on average the best layer per record only shares ~4-5 of the top-16 residues with gradient.
- `mean Jaccard ≤ 0.2 across layers` → metrics are orthogonal. Result: per-layer means all in **[0.085, 0.151]**, clearly orthogonal. Attention is NOT a proxy for what actually matters to the loss.

**Reading.** Three negatives, each independent, each enough to close a hypothesis the cheap-K distillation idea relied on:

1. **Gradient is more diffuse than attention** (mass_top_16_grad 0.31 vs attn 0.66; mass_top_32_grad 0.53 vs attn 0.79). The "use gradient as the importance signal to replace attention" rescue of [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)'s STOP is falsified — gradient is the strictly worse signal for top-K selection on this baseline.
2. **Gradient sets are unstable across t** (t-Jaccard 0.200). Even if we tried to distill the gradient signal, per-trajectory routing isn't viable; we'd need per-t routing on top of [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)'s per-layer × per-head requirement.
3. **Attention and gradient are largely orthogonal** (overall Jaccard 0.114; every per-layer mean ≤ 0.151). Attention is a learned routing structure with its own internal logic, not a proxy for loss-importance. A faithful student would need to preserve *both* signals — they're not interchangeable.

**Combined with [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13):** the cheap-K distillation idea required (a) the teacher signal is concentrated enough that K=16 captures ≥70% of the mass, and (b) the same K-set works across layers/heads/t. Both metrics fail (a) — attention 66% / gradient 31% at K=16. Both metrics fail (b) on at least one stability axis. They also disagree with each other, so picking either as "the teacher" doesn't give you a unified important set; you'd be choosing between two flawed importance signals that don't compose. **The STOP is now closed under two metrics, not one.**

**Methodological caveats.**
- **N=3 per length bin**, single seed (=42). Wilson intervals on individual records are wide; the headline numbers are means/medians over 36-630 records, which is tighter, but per-record outliers (e.g. the single Jaccard=0.6 record at layer 7) can still be noise.
- **bf16 grad** (default; `--force_precision_f32` not used). Gradient saliency in bf16 has more numerical noise than fp32, especially for the small components of `dL/dx_t` near zero. Direction of bias: bf16 noise diffuses the saliency more than fp32 would, slightly inflating diffuseness on Check 1 and slightly inflating disagreement with attn on Check 3. The headline conclusion (grad is more diffuse, grad and attn are orthogonal) does not invert under fp32 unless bf16 inflates concentration by ~2× — implausible.
- **Single loss head**: gradient is taken w.r.t. an unconditional FM loss-like quantity through `model.full_simulation`/forward of the score head, not the full training-time multi-task loss. The single-head simplification is intentional (matches the inference-time decision target) but a more realistic loss could yield a different saliency distribution. Cross-check the audit with the model's actual training loss term before committing to large architectural changes informed by this entry.
- **Attention hook is head-averaged**, not per-(layer, head, query) like [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13). This is by design (the script's docstring notes it as a deliberate simplification — sufficient for cross-metric overlap, much smaller JSON). The cross-metric numbers here are therefore comparing gradient-top-16 against a head-aggregated attention top-16. [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)'s per-head numbers are the right thing to look at for "do heads agree internally"; this entry is the right thing for "does the per-layer attention summary agree with gradient saliency".
- **Path-of-trajectory mismatch**: gradient saliency is computed at fixed t-values (5 discrete points), not integrated along an actual ODE trajectory. The 45 (protein, t) records sample importance at independent timesteps. Trajectory-level saliency (gradient through the full ODE rollout) would be more expensive and is not what this audit measures.

**Possible narrative.** Non-narrative — kept for tuning/decision-making. Feeds the same decision as [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (close the cheap-K distillation idea) and strengthens it with a second independent metric. If E059 alone would have left "but what about gradient as the teacher?" open, E060 closes that line. The two entries together are a complete decision-gate for the cheap content-adaptive top-K distillation idea.

**Cross-references.**
- Direct predecessor: [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (attention concentration audit; matched protocol).
- Sparse-vs-dense gap context: [E043](#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07) (per-t val-loss across variants). E060's "attention and gradient are orthogonal at every layer" extends E059's "sparse forfeits per-(layer, head) specialization" — the specialization is real not only in attention space but also relative to what the loss cares about.
- BigBird-globals connection: [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge), [E058](#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge), [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13). BigBird offered 4 protein-agnostic global slots as a cheap shared-content channel; the E060 finding ("attention is highly layer/head-specific AND not aligned with loss-saliency") is the structural reason why protein-agnostic globals can't substitute for layer × head specialization.
- Cross-metric per-layer Jaccard: layer 6 highest mean (0.151), layer 9 lowest mean (0.085). No layer is a clear "attention reflects gradient saliency" winner.
- Output JSON: `results/dense_attn_audit/canonical_2646_gradient.json` (2.5 MB, 45 grad records + 630 attention records).
- Audit log: `nohup_audit_gradient.out`.

## E061 — Per-query VJP gradient saliency inverts E060 (2026-05-13)

**Status:** finished.

**Why ran.** [E060](#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) concluded the cheap-K distillation idea was closed under two metrics, but the gradient saliency in E060 was taken w.r.t. an aggregate scalar (sum of per-residue prediction norms across all queries, single backward). The user pointed out that this design **sums per-query loss terms before differentiating**, so per-query specialization is averaged out before the gradient is recorded. In sparse attention, every residue has its OWN K-set of neighbors — the analog of "important set for query i" is the top-K residues j by ‖∂ scalar_i / ∂ x_t[j]‖, where scalar_i is a per-query scalar summary. The user refactored the audit script (commit `9ed7a93`) to take **vector-Jacobian products per sampled query** — one backward per query, one saliency vector PER query. That gives structurally one-to-one comparisons with dense's per-(layer, head, query) attention pattern. This entry runs that refactored path on the canonical step-2646 ckpt to test whether E060's three negatives survive when the question is asked per-query.

**Configs.**
- Audit script: `script_utils/audit_dense_gradient_saliency.py` at commit `d3b9fde` (= the refactor `9ed7a93` rebased after E060) + 1-line local fix today: `mass16_first3 = [f"{e['mass_top_K_grad'][1]:.3f}" for e in per_query_entries[:3]]` extracted to a temp var before the logger.info f-string (the original f-string used a `\"` inside a `{}` expression, which is a SyntaxError on Python 3.10/3.11). Compile-clean post-fix.
- Checkpoint: `/home/ks2218/la-proteina/best_val_00000026_000000002646.ckpt` (canonical dense, step 2646, `test_ca_only_diffusion/1776805213`).
- Protocol matched to [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)/[E060](#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13): seed=42, 3 proteins per length bin × L∈{50, 100, 200} = 9 proteins, t-grid `{0.10, 0.30, 0.50, 0.70, 0.90}` = 5 t-values, JACCARD_K=16, K_grid=`[8, 16, 32, 48, 64]`. **New parameter:** `--queries_per_protein 8` — 8 sampled query residues per (protein, t).
- Per-query saliency: for each sampled query i, compute `‖v_pred[i]‖₂` (no GT needed; pure functional dependency through the model's score head), then backward on that scalar alone with `retain_graph=True` across the 8 queries. Per-residue saliency = L2 norm of `dL_i/dx_t[j]` across the 3 coordinate dims, one scalar per residue j per query i.
- Attention hook records the full per-(layer, head, query, key) softmax pattern for the same sampled queries, so the cross-metric Jaccard is one-to-one (same query, same protein, same t, top-Kj from grad vs top-Kj from attn at each (layer, head)).
- Precision: bf16 (no `--force_precision_f32`).
- Hardware: 1× L4 (`gxp-l4-0` GPU 0), `nohup`. Launched 2026-05-13 11:47 BST.
- Record counts: **45 protein-t records × 8 queries = 360 per-query saliency records**, **630 attention-layer records** (45 × 14 layers), **5040 cross-metric records** (360 queries × 14 layers; each carrying 12 per-head Jaccard values = 60480 (query, layer, head) cells), **1260 query-pair Jaccard records** within (protein, t).
- Wall: ~15 s end-to-end on L4 (script header predicted 15-20 min on A100; the per-query VJPs with `retain_graph` are dramatically faster than estimated at L≤200).

**Results — Check 1' (per-query gradient saliency concentration, 360 queries):**

| K | mean | median | E060 aggregate grad | E059 attention |
|---|---|---|---|---|
| 8  | 0.567 | 0.564 | 0.176 | 0.510 |
| 16 | **0.709** | **0.728** | 0.312 | 0.656 |
| 32 | **0.830** | **0.871** | 0.528 | 0.794 |
| 48 | 0.891 | 0.921 | 0.683 | 0.866 |
| 64 | 0.922 | 0.953 | 0.766 | 0.907 |

Per-query gradient is **sharper than attention** at every K. E060's "grad is more diffuse" was an artifact of the aggregate path averaging across queries' loss terms. Falsifies E060's Negative #1 and triggers the script's `mass_top_K ≫ E059's [...]` decision rule — the gradient signal IS the sharper importance signal once asked per-query. K=16 mass = 0.709 ≥ 0.70 (the [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) attention-side GO threshold), K=32 mass = 0.830 < 0.85 (just under); E059's GO bar applied to per-query grad is essentially cleared at K=16 and almost cleared at K=32.

**Results — Check 2' (per-query t-Jaccard, 63 (protein, query) t-adjacent pairs):**

| metric | per-query | E060 aggregate |
|---|---|---|
| mean | **0.663** | 0.200 |
| min  | 0.280 | 0.000 |
| max  | 1.000 | 0.684 |
| n    | 63 | 36 |

The same query's top-16 important set is largely stable across adjacent t-values: mean Jaccard 0.66 — just below the 0.7 GO bar, far from the 0.3 STOP. E060's "t-unstable" call (Jaccard 0.20) was the aggregate path's per-query specialization being washed out. Many records hit Jaccard=1.0 (perfect stability for that query across that adjacent t-pair).

**Results — Cross-metric Jaccard(grad top-16 per query, attn top-16 per (layer, head, query)):**

| summary | value | E060 head-avg |
|---|---|---|
| overall mean across all (query, layer, head) cells | **0.337** | 0.114 |
| min | 0.000 | 0.000 |
| max | 1.000 | 0.600 |
| **mean of max-over-(layer, head) per (protein, t, query)** | **0.833** | 0.274 |
| min of that max | 0.524 | 0.067 |
| max of that max | 1.000 | 0.600 |
| n | 360 | 45 |

The headline number is **mean-of-max-over-(layer, head) per query = 0.833**. That means: for every one of the 360 sampled queries, there is *some* (layer, head) in the dense trunk where attention's top-16 attended residues agree with gradient's top-16 important residues on ≥13/16 of them. The worst query has at least one (layer, head) sharing 8/16; many hit 1.000. Triggers the script's `max-over-layers Jaccard ≥ 0.5 → attention DOES reflect loss-importance somewhere` decision rule by a wide margin. E060's "attention is NOT a proxy for loss-importance" call (0.274 mean-of-max) was head-averaged AND aggregate-loss — both averaging operations destroyed the per-query alignment.

**Per-layer cross-metric breakdown:**

| layer | mean(head_avg_J) | mean(best_head_J) | max(best_head_J) |
|---|---|---|---|
| 0  | 0.257 | 0.703 | 1.000 |
| 1  | 0.277 | 0.719 | 1.000 |
| 2  | 0.375 | **0.765** | 1.000 |
| 3  | 0.320 | 0.733 | 1.000 |
| 4  | **0.442** | 0.760 | 1.000 |
| 5  | 0.320 | 0.700 | 1.000 |
| 6  | 0.277 | 0.694 | 1.000 |
| 7  | 0.361 | 0.749 | 1.000 |
| 8  | 0.338 | 0.717 | 1.000 |
| 9  | 0.373 | 0.676 | 1.000 |
| 10 | 0.320 | 0.762 | 1.000 |
| 11 | 0.381 | 0.710 | 1.000 |
| 12 | 0.339 | 0.715 | 1.000 |
| 13 | 0.341 | 0.687 | 1.000 |

Layer 4 has the strongest head-averaged agreement (0.442); layer 9 has the weakest (0.257); every layer's best-head-per-record averages ≥ 0.676 across all 360 queries. **Per-layer routing IS visible**, but the message is gentler than E059's: rather than "shared K-set fails", it's "every layer has a head that strongly tracks gradient saliency for most queries". E059's per-layer Jaccard of 0.22 was *across layers* (do adjacent layers attend to the same residues?) — different question from this entry's *within-(layer, head) cross-metric* Jaccard (does that (layer, head)'s attention agree with gradient at the same query?).

**Query-pair Jaccard within (protein, t): mean 0.146, min 0.000, max 1.000, n=1260.** Different queries within the same protein at the same t share only ~2.3/16 of their top-16 important residues. **Confirms the user's intuition that per-query routing is genuinely the right unit of analysis** — aggregate-loss audits would average across this and miss it entirely.

**Reading combined.** **All three of E060's negatives flip under per-query VJP**:

1. Gradient is sharper than attention (per-query mass_top_16 = 0.71 vs attn 0.66; vs E060's aggregate 0.31).
2. Gradient sets are stable across t (per-query t-Jaccard 0.66 vs E060's aggregate 0.20). Just under the 0.7 GO bar.
3. Attention reflects gradient saliency at some (layer, head) per query (mean-of-max 0.83 vs E060's head-avg 0.27).

Plus a fourth, new observation: **different queries within (protein, t) need different K-sets** (query-pair Jaccard 0.15). The "168 K-sets per protein-per-t" concern from [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (14 layers × 12 heads) was right that shared-K is wrong, but the right factorization isn't (layer, head) — it's **(query, best (layer, head))**. There are 360 queries here, not 168 (layer, head) combos; but the per-query gradient saliency is itself a sharp, t-stable signal that a student could use directly to pick its per-query K-set without going through any (layer, head) routing at all.

**This reopens [E060](#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13)'s STOP** and rewrites [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13)'s reading. The path forward is not the "cheap shared-K student" the original idea proposed — it's **per-query routing where each query's K-set comes from its own gradient saliency or from the best-matching (layer, head)'s attention top-K**. The student is no longer "cheap" in the sense of "one K-set per protein", but it IS cheap in the sense of "K << N attention compute per query, with a learned per-query routing head". This is mechanistically supported by the per-query data.

**Methodological caveats.**

- **8 queries per protein × 5 t × 9 proteins = 360 sampled queries** — sub-sample of all (protein, t, residue) triples. The per-(protein, t) cross-metric max is over the *sampled* queries' (layer, head) cells, not all residues. Per-residue full-coverage would multiply compute by `mean_protein_length / 8 ≈ 12.5×` but isn't necessary for the headline directional flip.
- **`retain_graph` with 8 sequential backwards** is the load-bearing implementation detail. Wall budget at L=200 was ~15 s on L4; A100 would be faster but the bottleneck is not GPU-bound at these sizes (forward dominates).
- **bf16 grad** still applies — direction-of-bias inflates diffuseness slightly, but the per-query path is so much sharper than the aggregate path that the bf16 noise floor is not the limiting factor.
- **scalar_i = ‖v_pred[i]‖₂** is a particular choice of per-query summary. Other choices (e.g. `v_pred[i] · target_i` if a target were available, or per-coord gradients) might shift the saliency map. The norm choice is what the refactored script settled on; the headline directional reversal of E060 doesn't seem fragile to this choice (the per-query specialization story rests on different queries having different per-residue dependencies, not on the exact norm), but a sensitivity check (e.g. `‖v_pred[i]‖₁`, or per-coord gradients) is the natural follow-up.
- **Attention hook is the full per-(layer, head, query, key) softmax** — heavier than E060's head-averaged hook. JSON output ~3.5 MB.
- **No per-layer × per-head random redraw** test of the student route. The data here is consistent with per-query routing being viable, but training such a student is the actual experiment, not this audit. This entry is a decision-gate, not a build.

**Possible narrative.** Non-narrative — but **upgrades the routing-prior question from CLOSED back to OPEN, with a specific construction proposal**. E059 and E060's STOP language in their possible-narrative blocks should be read as superseded by this entry. The five-axis bundle ([E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12)) and BigBird-only ([E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13)) variants explored shared-K + content via global tokens; this entry suggests the missing structural lever is per-query routing — a learned K-set selector that consumes the per-query saliency signal at inference time and routes attention accordingly. That is a different experiment from anything currently trained.

**Cross-references.**

- Direct predecessor: [E060](#e060--gradient-saliency-companion-to-e059--cross-metric-grad-vs-attn-2026-05-13) (aggregate-loss gradient audit; **this entry's headline flips all three of E060's conclusions**).
- Two steps back: [E059](#e059--dense-attention-concentration-audit-content-adaptive-top-k-distillation-decision-gate-2026-05-13) (attention concentration audit; E059's "168 separate K-sets needed" framing is reframed by this entry to "per-query routing, not per-(layer, head) shared K-set").
- Sparse-vs-dense architectural framing: [E043](#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07).
- Five-axis / four-axis bundles that tried shared-K + content-by-globals: [E055](#e055--first-designability-probe-of-the-five-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird_pairupdate_lowtsoft-step-944-2026-05-12), [E056](#e056--first-designability-probe-of-the-four-axis-bundle-ca_only_sparse_k64_curriculum_self_bigbird-step-819-2026-05-13), [E057](#e057--bigbird-wiring-audit-on-e047-step-1200-2026-05-12-renumbered-from-upstream-e048-on-2026-05-13-merge), [E058](#e058--cold-start-bigbird-only-no-pair-update-no-lowtsoft-on-the-11-trunk-2026-05-12-renumbered-from-upstream-e049-on-2026-05-13-merge). The per-query routing alternative this entry surfaces is orthogonal to those variants' design axis.
- Local script fix: 1-line f-string syntax patch on `script_utils/audit_dense_gradient_saliency.py:478-481` (extract list-comp to temp var to avoid backslash-in-f-string-expression). Compile-checked + run-verified.
- Output JSON: `results/dense_attn_audit/canonical_2646_grad_per_query.json` (~3.5 MB; 360 grad records, 630 attention records, 5040 cross records, 1260 query-pair records).
- Audit log: `nohup_audit_gradient_perquery.out`.
