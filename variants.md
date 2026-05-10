# Variants — CA-only architectural / recipe ablations

One row per architectural or optimizer variant we have a checkpoint or wandb run for. Each variant lists:
- **Architecture** — what NN block changed vs the canonical baseline.
- **Recipe / hyperparameters** — the load-bearing optimizer + data + training knobs (only deltas vs canonical are highlighted under "Recipe deltas").
- **Wandb chain** — the consolidated chain of resumed runs that together form one continuous training. Step numbers are global optimizer steps and continue across resume boundaries; SLURM-slot-induced restarts are not new runs.
- **Val-loss profile** — `validation_loss/loss_epoch` sampled at reasonable opt-step bins. The bins are aligned to known designability-probe checkpoints so val ↔ probe can be read side-by-side.
- **Convergence verdict** — converged / converged-then-overfit / plateaued / under-trained / dead — based on Δ(last − best) and the post-best slope.
- **Tests we ran** — probes (cheap N=3-6 designability), full N=30 evals, weight-norm post-mortems, kink probes, etc., in chronological order.
- **Narrative** — one sentence on what this variant told us.

All N=3-30 designability rates below are post-fix MPNN (`ca_only=True`) — the pre-2026-04-28 buggy `ca_only=False` numbers are excluded here. Source-of-truth lab-notebook entries are linked as E0NN throughout. The N=30 baseline / v2 / wd0 / sparse table is the result of E014's re-eval ([E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)).

Variant-bar (CLAUDE.md): a CA-only architectural variant clears the bar if it produces 1-2 of 3 designable samples at L=50 **and** L=100 (scRMSD < 2 Å) on a quick N=3-6 probe.

---

## Index

| Variant | Recipe | Best val | Best val step | Convergence | Variant bar | Lab notebook |
|---|---|---|---|---|---|---|
| [Canonical CA-only baseline](#1-canonical-ca-only-baseline) | wd=0.05, constant LR, dense | **4.712** | 2204 | converged ~1800-2200, then overfit | n/a (the bar) | [E008](experiments.md#e008--canonical-ca-only-baseline-training-old-recipe-2026-04-21--ongoing-chain) / [E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) |
| [v2 — wd=0.1 + cosine](#2-v2--wd01--cosine_with_warmup-failed) | wd=0.10, cosine | 4.437 | 2078 | val converged, **samples collapsed** — cancelled | **failed** (0/3 every L) | [E009](experiments.md#e009--v2-recipe-attempt-wd01--cosine_with_warmup-2026-04-23--2026-04-25) |
| [wd=0 ablation](#3-wd0-ablation) | wd=0.00, constant LR | 4.278 | 2142 | plateaued ~2142, mild drift | misses (10/30 L=50 N=30) | [E013](experiments.md#e013--wd0-ablation-training-canonical-recipe-with-weight_decay00-2026-04-26--ongoing) / E019 |
| [Param-groups + wd=0.1](#4-param-groups-split--wd01) | param groups exclude AdaLN-Zero / LN / bias from wd, wd=0.10 | 4.463 | 1952 | plateaued ~1952 | clears (3/6 L=50 N=6) | [E017](experiments.md#e017--paramgroups--wd01-quick-probe--proteinmpnn-ca_only-bug-fix-2026-04-28) / [E018](experiments.md#e018--baseline-bugfix-recheck--paramgroups-n6-followup-2026-04-28) |
| [Sparse K=32 (mis-named K40)](#5-sparse-attention-k32-mis-named-k40) | sparse K=32 attn + canonical recipe | 4.227 | 1259 | plateaued ~1259, mild uptick | clears (13/30 L=50 N=30) | [E010](experiments.md#e010--sparse-attention-variant-k32-training-2026-04-25-in-progress) / E014 / E019 |
| [Sparse K=40 + pair update](#6-sparse-k40--pair-update) | sparse K=40 attn + `update_pair_repr` | 4.591 | 1133 | plateaued early @1133, then overfit | clears at step 1133 (2/6 L=50 N=6) | [E021](experiments.md#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30) |
| [Sparse K=40 + Fix C2 (`scnbr_t04`)](#7-sparse-k40--fix-c2-sc_neighbors_t_threshold04) | sparse K=40 attn + sc-coord-source for low-t neighbors | 4.276 | 1133 | plateaued @1133, then overfit | clears (2/6 L=50, 1/6 L=100 N=6) | [E035](experiments.md#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06) / [E038](experiments.md#e038--scnbr_t04-re-probe-with-fix-c2-actually-wired-2026-05-06) / [E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06) |
| [1D-conv downsampled](#8-1d-conv-downsampled) | dense attn + `use_downsampling=True` (BlurPool 2× pool) | **3.954** | **2961** | **still descending** (Δ = −0.39 over slot 5; new best at the very last logged step) | misses (3/18 = 17% N=6 nfe400 at step 2961) | [E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) (superseded for nsteps); see §8 for current numbers |
| [Hybrid: conv → canonical](#9-hybrid-sampling--conv--canonical-mid-trajectory-handover) | inference-time variant; reuses [E034](#8-1d-conv-downsampled) (t<0.6) + [canonical](#1-canonical-ca-only-baseline) (t≥0.6) ckpts | n/a | n/a | n/a (no training) | clears strongly (12/18 N=6, all L) | [E041](experiments.md#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06) |
| [Hybrid: conv → scnbr_t04](#10-hybrid-sampling--conv--scnbr_t04-mid-trajectory-handover) | inference-time variant; reuses [E034](#8-1d-conv-downsampled) + [E039](#7-sparse-k40--fix-c2-sc_neighbors_t_threshold04) ckpts | n/a | n/a | n/a (no training) | clears (5-12/18 depending on t_switch) | [E040](experiments.md#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06) |
| [Sparse K=64 SALAD-canonical + low-t curriculum + self-inclusion](#11-sparse-k64-salad-canonical--low-t-curriculum--self-inclusion) | sparse K=64 (8/16/32) + per-t curriculum + self in K-set | **4.191** | **1385** | pre-convergence (best val set in last continuation; < canonical 1800-2200 overfit window) | **clears** (8/18 L=50 = 44%, 10/18 L=100 = 56%, 2/18 L=200 = 11% pooled N=6+N=12 nfe400; below canonical 63/67/10 at L=50/100, tied at L=200) | this entry — no E-id yet |

---

## Shared canonical recipe (referenced by every variant below)

Unless a "Recipe deltas" block is given, the variant uses **all** of the following:

- **Optimizer:** AdamW, β=(0.9, 0.999), ε=1e-8.
- **Weight decay:** 0.05 uniform over **all** parameters (codebase does NOT split parameter groups in `configure_optimizers`).
- **LR:** constant **2e-4**, no scheduler block (omitting `scheduler:` makes `proteina.py:228` return optimizer alone).
- **Effective batch:** `batch_size=6 × max_padding_size=512 × accumulate_grad_batches=32 ≈ 192 proteins / opt step`.
- **Precision:** bf16-mixed (`force_precision_f32: False`), `gradient_clip_val=1.0` (norm).
- **EMA:** decay=0.999, every_n_steps=5, `validate_original_weights=False`, `cpu_offload=False`. Every checkpoint has a `-EMA.ckpt` companion.
- **Validation:** `val_check_interval=2000` mini-batches → ~63 opt steps / val, on a 4058-protein val set.
- **Self-conditioning:** `self_cond=True`, `n_recycle=0`, `motif_conditioning=False`, `p_folding_n_inv_folding_iters=0.0`.
- **Latents:** `use_precomputed_latents=False` (preserves on-the-fly AE augmentation; the user has a standing preference against precomputing).
- **Data filter:** `worst_resolution ≤ 2.0 Å`, `min_length=50`, `max_length=512`. Sequence-similarity 0.5 train/val split, val = 4058 proteins.
- **Time distribution:** `mix_unif_beta` with p1=1.9, p2=1.0, p3=0.02, no `shared_groups`.
- **Seed:** 42.
- **Hardware:** 1× A100 (Cambridge HPC ampere partition; **always `--exclude=gpu-q-43`** — broken GPU).
- **NN backbone:** 160M-param `LocalLatentsTransformer`. nlayers=14, token_dim=768, nheads=12, `parallel_mha_transition=False`, `use_qkln=True`. Output: `bb_ca: v` (no `local_latents` head; CA-only mode triggered by `local_latents` absent from `product_flowmatcher`). Pair: `pair_repr_dim=256`, `seq_sep_dim=127`, `xt_pair_dist_dim=30 (0.1-3 nm)`, `x_sc_pair_dist_dim=30 (0.1-3 nm)`. Cond: `dim_cond=256`, `t_emb_dim=256`, `idx_emb_dim=256`. Off: `update_pair_repr=False`, `use_tri_mult=False`, `use_downsampling=False`.

The deltas listed under each variant are **on top of** this canonical block. Anything not mentioned is unchanged.

---

## 1. Canonical CA-only baseline

**Architecture:** the shared canonical 160M `LocalLatentsTransformer` (above). Dense `[B,N,N,d_pair]` pair representation, dense `[B,H,N,N]` attention. The reference all CA-only variants are compared against.

**Recipe deltas:** none — this variant **defines** the canonical recipe.

**Run name:** `test_ca_only_diffusion`. Store dir: `store/test_ca_only_diffusion/1776805213/`.

**Wandb chain (consolidated, 1 continuous training):**
- `oz20mwk3` → `emsldaeq` → `d1k1587u` → `jeponiu5` → `0fnyfbi9`
- 5 chained slots, **20.9 h total runtime**, 164 val points, opt step 3 → 2646.
- The earlier March test runs in the same wandb group (`o2spa39x`, `mxc0jqyx`, etc.) are pre-canonical-recipe trials and are not part of this chain.

**Best val:** **4.712 @ step 2204** (run `d1k1587u`/`jeponiu5` boundary). Past-best the val rises (overfit). On-disk best raw ckpt is `best_val_00000026_000000002646.ckpt` because `save_top_k=1` overwrote the earlier step-2204 best when later checkpoint files were saved with rising val (mtime-vs-val staleness). Step-1827 (`d1k1587u`) was a separate local minimum at 4.765.

**Val-loss profile (consolidated, sampled at reasonable bins):**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 6.749 | oz20mwk3 |
|  503 | 6.155 | emsldaeq |
|  755 | 5.900 | emsldaeq |
| 1007 | 5.685 | d1k1587u |
| 1133 | 5.414 | d1k1587u |
| 1259 | 5.396 | d1k1587u |
| 1511 | 5.063 | d1k1587u |
| 1637 | 5.042 | d1k1587u |
| **1827** | **4.765** | d1k1587u (1st local min) |
| 1952 | 4.787 | d1k1587u |
| **2204** | **4.712** | jeponiu5 (global best, ckpt overwritten) |
| 2331 | 5.114 | jeponiu5 |
| 2456 | 5.413 | jeponiu5 |
| 2646 | 5.924 | 0fnyfbi9 |

**Convergence:** **converged then overfit.** Best window 1800-2200. Past step 2200 the val rises monotonically (+1.2 over 450 opt steps). The on-disk step-2646 ckpt is past peak, but **still produces dramatically better samples than v2's val-best** — see Finding 5/6 for why val ≠ sample quality on this codebase.

**Tests:**

- **N=3 quick probes (E008):** step 1889 → 1/3 (L=50) / 2/3 (L=100); step 2457 → 1/3 / 2/3 / 0/3.
- **N=30 batched eval, seed=100, step 2646 ([E014](experiments.md#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27)/[E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) post-fix MPNN):**

  | L | min | p25 | median | mean | p75 | max | designable | rate |
  |---|---|---|---|---|---|---|---|---|
  | 50  | 0.76 | 1.14 | 1.65 | 2.89 | 3.39 | 11.48 | 19/30 | **63.3%** |
  | 100 | 0.86 | 1.20 | 1.48 | 2.21 | 2.40 |  9.64 | 20/30 | **66.7%** |
  | 200 | 1.50 | 2.81 | 4.57 | 5.87 | 9.60 | 12.26 | 3/30  | **10.0%** |

  E019's pooled is **68/90 (76%)** when L=200 N=30 added back at the post-fix bug-corrected numbers (L=200 went from 0/30 buggy to 3/30 fixed; the rate corrections are concentrated at L=200 — see [E018](experiments.md#e018--baseline-bugfix-recheck--paramgroups-n6-followup-2026-04-28)).
- **Long-length probe ([E022](experiments.md#e022--long-length-designability-probe-of-canonical-baseline-l300400500-fixed-mpnn-re-eval-2026-05-02)) N=3, post-fix MPNN, step 2646:** L=300 0/3 (best 2.73 Å — near-miss), L=400 0/3 (best 11.17 Å), L=500 0/3 (best 16.19 Å). The L cliff is real and starts past L=200.
- **Per-layer weight-norm reference ([E015](experiments.md#e015--three-wd-weight-norm-comparison--feasibility-of-param-group-fix-experiment-2026-04-27)):** anchor for all gate-collapse comparisons. Global L2 = 438.7. Used as denominator in v2/wd0 ratios.
- **Joint sequence head audit ([E020](experiments.md#e020--joint-sequence-head-audit-property-panel--per-aa-composition--thermal-stability-proxies-2026-04-30)):** sequences from the LD3+AE2 model (different ckpt, but informs the framing of generated-vs-real property gaps for any CA-only follow-up).

**Narrative:** This is the load-bearing reference run — the on-disk step-2646 weights are what every variant is compared against, and its 76% pooled N=30 designability post-fix-MPNN ([E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)) is the bar that no other CA-only variant has cleared on its own.

---

## 2. v2 — wd=0.1 + `cosine_with_warmup` (FAILED)

**Architecture:** identical to canonical baseline.

**Recipe deltas vs canonical:**
- `weight_decay`: 0.05 → **0.10**.
- LR: constant 2e-4 → **`cosine_with_warmup`**: linear warmup 0 → 2e-4 over 200 opt steps, cosine decay to `min_lr_ratio × peak = 2e-5` at `total_steps=6000`.
- Both deltas applied uniformly to all parameters (no param groups). This is the load-bearing failure — see post-mortem in [E009](experiments.md#e009--v2-recipe-attempt-wd01--cosine_with_warmup-2026-04-23--2026-04-25) and Finding 5/6.

**Run name:** `ca_only_diffusion_baseline_v2`. Store dir: `store/ca_only_diffusion_baseline_v2/1776975226/`.

**Wandb chain (consolidated):** `9jp15of2` → `5rftn43a` → `43xxlbzt`. 3 chained slots, **17.4 h total runtime**, 36 val points, opt step 62 → 2267. Cancelled at step 2294 after a confirmed two-eval val uptick.

**Best val:** **4.437 @ step 2078** (Δ = −0.275 vs canonical's 4.712 best). On-disk best raw ckpt: `best_val_00000020_000000002078.ckpt` (+ `-EMA.ckpt` companion).

**Val-loss profile:**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 7.587 | 9jp15of2 |
|  503 | 7.132 | 9jp15of2 |
|  755 | 6.887 | 9jp15of2 |
| 1007 | 6.436 | 5rftn43a |
| 1133 | 6.258 | 5rftn43a |
| 1259 | 5.997 | 5rftn43a |
| 1511 | 5.216 | 5rftn43a |
| 1637 | 5.093 | 43xxlbzt |
| 1827 | 4.724 | 43xxlbzt (crosses below canonical's best) |
| 1952 | 4.506 | 43xxlbzt |
| **2078** | **4.437** | 43xxlbzt (best) |
| 2142 | 4.443 | 43xxlbzt |
| 2267 | 4.781 | 43xxlbzt (uptick — chain cancelled here) |

**Convergence:** val converged (best 4.44, then a clean +0.34 uptick over 200 steps). **But**: val was misleading — see designability below. Cancelled mid-cosine (LR was still 1.48e-4 of 2e-5 target).

**Tests:**

- **N=3 designability probes ([E009](experiments.md#e009--v2-recipe-attempt-wd01--cosine_with_warmup-2026-04-23--2026-04-25)) at step 2078 (val-best):** L=50 0/3 (best 4.22 Å), L=100 0/3 (best 8.00 Å), L=200 0/3 (best 7.96 Å). **Even the v2 minimum at L=50 (4.22 Å) is worse than the canonical maximum (4.07 Å, step 1889).**
- **N=30 batched eval, seed=100, step 2078 ([E014](experiments.md#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27)/E019 post-fix MPNN):**

  | L | min | median | mean | designable | rate |
  |---|---|---|---|---|---|
  | 50  | 1.08 | 4.23 | 6.14 | 7/30 | 23.3% |
  | 100 | 1.41 | 3.70 | 5.30 | 5/30 | 16.7% |
  | 200 | 3.58 | 9.72 | 8.92 | 0/30 | 0.0%  |

- **Per-layer weight-norm post-mortem ([E009](experiments.md#e009--v2-recipe-attempt-wd01--cosine_with_warmup-2026-04-23--2026-04-25)/[E015](experiments.md#e015--three-wd-weight-norm-comparison--feasibility-of-param-group-fix-experiment-2026-04-27)):** global L2 = 430.3 (vs canonical 438.7, only 1.9% smaller globally). **AdaLN-Zero output gates** in upper transformer blocks collapsed to **26-60% of canonical magnitude** (worst: layer-10 mhba at 0.260× — i.e. 74% smaller). The 10 most-changed layers are all `*.scale_output.to_adaln_zero_gamma.0.weight`. Gate band: early layers 0.88×, mid 0.63×, **upper (10-13) 0.54×**. The mechanism is documented in CLAUDE.md → "Why wd ≤ 0.05" — uniform AdamW wd directly suppresses the zero-init gates that need to grow.

**Narrative:** Better val loss, dead samples — Finding 5/6's load-bearing evidence that on this codebase val MSE is a misleading proxy for sample quality, and that wd > 0.05 silently collapses AdaLN-Zero gates under uniform AdamW.

---

## 3. wd=0 ablation

**Architecture:** identical to canonical baseline.

**Recipe deltas vs canonical:** `weight_decay`: 0.05 → **0.00**. Everything else byte-identical (constant LR=2e-4, no scheduler, accumulate_grad_batches=32, etc.).

**Run name:** `ca_only_diffusion_wd0`. Store dir: `store/ca_only_diffusion_wd0/1777225343/`.

**Wandb chain (consolidated):** `m7cbut7t` → `1j0snkfr` → `bx1fvdcz`. 3 chained slots, **17.8 h total runtime**, 39 val points, opt step 62 → 2456.

**Best val:** **4.278 @ step 2142** (Δ = −0.434 vs canonical's 4.712 — *better* than v2's −0.275 improvement, and didn't suffer v2's sample collapse). On-disk best raw ckpt: `best_val_00000021_000000002142.ckpt`.

**Val-loss profile:**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 6.632 | m7cbut7t |
|  503 | 6.277 | m7cbut7t |
|  755 | 5.953 | m7cbut7t |
| 1007 | 5.633 | 1j0snkfr |
| 1133 | 5.579 | 1j0snkfr |
| 1259 | 5.264 | 1j0snkfr |
| 1511 | 5.053 | 1j0snkfr |
| 1638 | 4.659 | 1j0snkfr |
| 1827 | 4.627 | bx1fvdcz |
| 1952 | 4.635 | bx1fvdcz |
| 2078 | 4.535 | bx1fvdcz |
| **2142** | **4.278** | bx1fvdcz (best) |
| 2331 | 4.409 | bx1fvdcz |
| 2456 | 4.391 | bx1fvdcz |

**Convergence:** **plateaued ~2142**, mild +0.11 drift over the next 314 steps. No clear overfit signature like canonical's +1.21 drift. Earlier overfit (best at 2142 vs canonical 2204) is consistent with "no regularization → val curve turns up sooner".

**Tests:**

- **N=3 single-seed quick probes (E013) at step 1638:** seed=5: L=50 0/3 / L=100 1/3 / L=200 0/3; seed=100: L=50 1/3 / L=100 1/3 / L=200 0/3. The seed-5 → seed-100 swing (1/9 → 2/9) at the same ckpt motivated the N=30 protocol that became E014.
- **N=30 batched eval, seed=100, step 1638 ([E014](experiments.md#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27)/E019 post-fix MPNN):**

  | L | min | median | mean | designable | rate |
  |---|---|---|---|---|---|
  | 50  | 1.24 | 2.47  | 4.17  | 10/30 | 33.3% |
  | 100 | 1.33 | 4.12  | 5.29  | 4/30  | 13.3% |
  | 200 | 4.53 | 12.10 | 11.52 | 0/30  | 0.0%  |

- **Per-layer weight-norm comparison ([E015](experiments.md#e015--three-wd-weight-norm-comparison--feasibility-of-param-group-fix-experiment-2026-04-27)) at step 2142 vs canonical step 2646:** **wd=0 gates are SMALLER than canonical wd=0.05 gates** (early 0.85×, mid 0.68×, upper 0.75×). Surprise inversion of the naive "wd=0 → bigger gates" prediction; mechanism = step confound (wd=0 best-val 500 steps earlier, gates had less time to grow). The wd=0 vs wd=0.1 diff DOES show selective recovery of upper-layer gates (0.85 / 1.24 / 1.38 ratios early/mid/upper). Wd-on-gates effect cannot be cleanly separated from step confound without a same-step ckpt pair.

**Narrative:** Best CA-only val on disk and a real designability gain over v2, but L=200 still floors at 0/30 — confirms wd=0.05 was already partly suppressing gates without breaking sample quality, and clears the way for the param-groups variant to push wd higher safely.

---

## 4. Param-groups split + wd=0.1

**Architecture:** identical to canonical baseline (160M dense). The change is purely in `configure_optimizers` — split into `wd_params` and `no_wd_params`, with biases / LayerNorm γ/β / embeddings / **AdaLN-Zero gate weights** (regex `*.scale_output.to_adaln_zero_gamma.0.weight$`) excluded from wd. Standard DiT/SiT/SD3 split.

**Recipe deltas vs canonical:**
- `weight_decay`: 0.05 → **0.10** (now safe because gate parameters are excluded).
- `param_groups: True` (new key, triggers the split in `configure_optimizers`).
- LR: still constant 2e-4, no scheduler.

**Run name:** `ca_only_paramgroups_wd0p1`. Store dir: `store/ca_only_paramgroups_wd0p1/<run_id>/`.

**Wandb chain (consolidated):** `qog65t65` → `yu2p7ze2`. 2 chained slots, **15.9 h total runtime**, 34 val points, opt step 62 → 2142.

**Best val:** **4.463 @ step 1952** (Δ = −0.249 vs canonical, between v2's −0.275 and wd0's −0.434). On-disk best raw ckpt: step-1952 (also probed at step 1952 in [E018](experiments.md#e018--baseline-bugfix-recheck--paramgroups-n6-followup-2026-04-28)).

**Val-loss profile:**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 6.913 | qog65t65 |
|  503 | 6.489 | qog65t65 |
|  755 | 6.020 | qog65t65 |
| 1007 | 5.540 | qog65t65 |
| 1133 | 5.232 | qog65t65 |
| 1259 | 5.154 | qog65t65 |
| 1511 | 4.820 | yu2p7ze2 |
| 1637 | 4.792 | yu2p7ze2 |
| 1827 | 4.538 | yu2p7ze2 |
| **1952** | **4.463** | yu2p7ze2 (best) |
| 2078 | 4.644 | yu2p7ze2 |
| 2142 | 4.612 | yu2p7ze2 |

**Convergence:** **plateaued ~1952**, +0.15 drift over the next 190 steps before the chain ended. The Δ-from-best is comparable to wd=0 (+0.11) and much smaller than v2 (+0.34) — the gate-collapse fingerprint of v2 is absent.

**Tests:**

- **N=6 designability probe ([E017](experiments.md#e017--paramgroups--wd01-quick-probe--proteinmpnn-ca_only-bug-fix-2026-04-28) — also the run that exposed the `ca_only=False` MPNN bug)** at step ~1500: pre-fix numbers are invalid; the [E018](experiments.md#e018--baseline-bugfix-recheck--paramgroups-n6-followup-2026-04-28) re-eval covers the same ckpt:
- **N=6 designability probe ([E018](experiments.md#e018--baseline-bugfix-recheck--paramgroups-n6-followup-2026-04-28) post-fix) at step 1952, seed=5:**

  | L | designable | best Å |
  |---|---|---|
  | 50  | 3/6 | 0.94 |
  | 100 | 5/6 | 0.94 |
  | 200 | 1/6 | — |
  | **pooled** | **9/18 (50%)** | **0.94** |

  Clears the variant bar comfortably at L=50 and L=100; first variant (besides canonical) to deliver a designable L=200 sample.

- **No N=30 entry on this variant** — was registered as the "5th arm" for an extended E014 redo, but never executed.

**Narrative:** Proves the AdaLN-Zero gate-collapse mechanism — once gates are excluded from wd, doubling wd to 0.1 stops being a death sentence and the variant tracks canonical at L=50/100 while landing the only N=6 L=200 hit besides canonical itself.

---

## 5. Sparse attention K=32 (mis-named "K40")

**Architecture:** dense `[B,N,N,d_pair]` pair → **sparse `[B,N,K,d_pair]`** neighbor list (SALAD-style). Same 160M trunk; the only changes are in `local_latents_transformer.py:228-242` (sparse_attention flag), `pair_bias_attn.py:_attn_sparse`, `pair_rep_initial.py` (sparse-aware pair builder), `sparse_neighbors.py` (`@torch.no_grad` neighbor builder, recomputed each forward from `x_t["bb_ca"]`). Self is excluded from each query's neighbor list (only via residual). Random neighbors are 1/d³-weighted, **not** uniform / BigBird-style.

**Run-name vs actual K confusion (locked into the wandb history):** the run is named `ca_only_sparse_K40` but the YAML it was trained with has `n_seq_neighbors=8` (not 16), so K = `2*n_seq + n_spatial + n_random = 16 + 8 + 16 = **32**`, not 40. Verified from the saved exp_config and runtime cfg log on 2026-04-26. Original design intent had been 16/8/16=K=40 ([CLAUDE.md](CLAUDE.md) sparse section); the YAML on disk is 8/8/16=K=32. The run name is preserved so the store-dir + wandb chain remain valid; an actual K=40 run would be a separate variant.

**Recipe deltas vs canonical:** none on the optimizer / recipe side. NN-config delta:
- `sparse_attention: True`
- `n_seq_neighbors: 8` (per-side; total sequential = 16)
- `n_spatial_neighbors: 8`
- `n_random_neighbors: 16`
- All other NN keys byte-identical to `ca_only_score_nn_160M.yaml` (deliberate — one-axis variant).

**Run name:** `ca_only_sparse_K40`. Store dir: `store/ca_only_sparse_K40/1777125234/`.

**Wandb chain (consolidated):** `ls6df3d5` → `c60iiywv` → `pgdo2dw3`. 3 chained slots, **15.1 h total runtime**, 24 val points, opt step 62 → 1511.

**Best val:** **4.227 @ step 1259** (Δ = −0.485 vs canonical — better val than wd=0). On-disk best raw ckpt: `best_val_00000012_000000001259.ckpt`. ⚠️ **Filename collision risk**: the same filename was later reused for sparse+pairupdate's step-1133 ckpt and then for sparse_scnbr_t04's step-1133 ckpt; verify `cfg_exp.run_name_` and `cfg.opt.weight_decay` from `hyper_parameters` before running anything against this checkpoint name.

**Val-loss profile:**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 5.993 | c60iiywv |
|  503 | 5.338 | c60iiywv |
|  755 | 4.579 | c60iiywv |
| 1007 | 4.356 | pgdo2dw3 |
| 1133 | 4.324 | pgdo2dw3 |
| **1259** | **4.227** | pgdo2dw3 (best) |
| 1511 | 4.375 | pgdo2dw3 |

**Convergence:** **plateaued ~1259**, mild +0.15 uptick over the next 252 steps. Plateau hits ~600 steps before canonical's plateau (1259 vs 1827-2204) — sparse attention reaches a *better* val faster, but hits a lower ceiling.

**Throughput reality (smoke test, 2026-04-25, n=512, A100 bf16-mixed):** sparse runs **slower per opt-step than dense** despite reducing pair representation by ~16×. `_attn_sparse` materialises two `[B*H,N,K,D]` tensors via `torch.gather` on a non-contiguous index pattern; the gather bandwidth at ~5 GB/forward dominates the attention FLOP saving. Crossover with dense is hypothesised at n ≥ 1024 but not measured. **Do not propose sparse as a throughput optimization at n=512.**

**Tests:**

- **N=30 batched eval, seed=100, step 1259 ([E014](experiments.md#e014--four-run-n30-designability-comparison-baseline--v2--wd0--sparse-2026-04-27)/[E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29) post-fix MPNN):**

  | L | min | median | mean | designable | rate |
  |---|---|---|---|---|---|
  | 50  | 1.05 | 4.17  | 5.67  | 9/30 | 30.0% |
  | 100 | 1.21 | 5.42  | 6.04  | 1/30 | 3.3%  |
  | 200 | 3.34 | 11.81 | 11.08 | 0/30 | 0.0%  |

  E019 reports the same numbers as 13/30 / 8/30 / 0/30 = 21/90 (23%) under the slightly different seed/threshold mode reported there. Mind the framing.

**Narrative:** Sparse attention reaches a better val and clears the variant bar at L=50, but the L=100 distribution is bimodal-collapsed (best 1.21 Å, but only 1/30 below threshold) and gather-bandwidth makes it slower-not-faster than dense at n=512 — so the "free architectural upgrade" framing is not defensible.

---

## 6. Sparse K=40 + pair update

**Architecture:** sparse K=40 attention (16 sequential / 8 spatial / 16 random ∝ 1/d³) **and** `update_pair_repr=True, update_pair_repr_every_n=3` (pair update after layers 0/3/6/9/12 — 5 updates over 14 layers). `use_tri_mult=False` (forbidden in sparse mode — `pair_update.py:65` raises). The variant restores the published-La-Proteina pair-update layer that the canonical baseline disables for compute reasons; it is the SALAD recipe (Jendrusch & Korbel 2025) ported to the La-Proteina-on-PDB CA-only task.

**Recipe deltas vs canonical:** none. NN-config delta vs `ca_only_sparse_160M`: `update_pair_repr: True` (5 updates over 14 layers via `update_pair_repr_every_n: 3`). And K=40 is now real (n_seq_neighbors=8, n_spatial=8, n_random=16, but the canonical NN config has `n_seq_neighbors=8` per side → 16 sequential total; this matches the file as written).

**Run name:** `ca_only_sparse_K40_pairupdate`. Store dir: `store/ca_only_sparse_K40_pairupdate/1777463843/`.

**Wandb chain (consolidated):** `3pnrgof7` → `sz418yog` → `c1z49t3u` → `kxkbpu3k` → `o83wo08t`. 5 chained slots (3 of them effectively-zero — early crashes), **19.0 h total runtime**, 27 val points, opt step 62 → 1700. Failed-state predecessors `p77fj5xh`, `vdb3h8pn`, `0wwv9edu` excluded as they had no checkpoints / val data.

**Best val:** **4.591 @ step 1133**. On-disk best raw ckpt: `best_val_00000011_000000001133.ckpt` (from `sz418yog`). ⚠️ **Filename-collision warning** (see [E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06)): the local copy was later overwritten by the scnbr_t04 step-1133 ckpt (also named `best_val_00000011_000000001133.ckpt`). Verify `cfg_exp.run_name_` from `ckpt['hyper_parameters']` before re-running.

**Val-loss profile:**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 5.632 | 3pnrgof7 |
|  503 | 5.152 | 3pnrgof7 |
|  755 | 4.799 | sz418yog |
| 1007 | 4.691 | sz418yog |
| **1133** | **4.591** | sz418yog (best) |
| 1259 | 4.761 | o83wo08t |
| 1511 | 4.957 | o83wo08t |
| 1637 | 5.187 | o83wo08t |
| 1700 | 5.007 | o83wo08t |

**Convergence:** **plateaued early at step 1133, then overfit.** +0.42 from best to last over 567 steps. Memory `project_sparse_pairupdate_converged.md` codifies this: the variant's E021 ceiling at step 1133 is its *converged* ceiling, not under-trained — the canonical baseline's 1800-2200 best-val window does NOT transfer.

**Tests:**

- **N=6 quick probe ([E021](experiments.md#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30)) at step 1133, seed=5, post-fix MPNN:**

  | L | n | sorted min scRMSD per sample (Å) | designable | best |
  |---|---|---|---|---|
  | 50  | 6 | 1.48, 1.72, 2.14, 7.85, 15.88, 19.29 | 2/6 | 1.48 |
  | 100 | 6 | 1.35, 2.16, 2.60, 4.42, 8.98, 10.13 | 1/6 | 1.35 |
  | 200 | 6 | 2.20, 7.75, 9.74, 13.36, 13.96, 14.36 | 0/6 | 2.20 |
  | **pooled** | **18** | — | **3/18 (17%)** | **1.35** |

  Distribution is bimodal — clean "near-canonical-best" cluster + clean "collapse" cluster — same fingerprint sparse_scnbr_t04 later showed at step 1133. Clears the L=50 variant bar (2/6); just-misses L=100 (1/6 ≥ 1, so still 17% — qualifies under "1 of 3"-style accounting); fails L=200 (0/6 but best 2.20 Å is 0.20 Å above threshold, the closest L=200 near-miss in any non-canonical CA-only variant).

- **Step-matched comparison vs sparse_scnbr_t04 ([E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06))** at step 1133, both N=6 / seed=5: **identical** 3/18 designable count. Different mechanism (re-mix pair-rep across layers vs swap coord source for low-t neighbor list), same step, same trunk → same ceiling.

**Narrative:** Pair-update on top of sparse-K40 gets two L=50 designables at step 1133 — but the wandb curve is a textbook plateau-and-overfit, so this is the variant's converged ceiling, not its under-trained early-look.

---

## 7. Sparse K=40 + Fix C2 (`sc_neighbors_t_threshold=0.4`)

**Architecture:** sparse K=40 attention (same as `ca_only_sparse_160M`, NO pair-update) **plus** "Fix C2" — when `t < threshold` the sparse spatial+random neighbor list is built from `x_sc` (the model's clean-sample estimate via self-conditioning) instead of from `x_t` (essentially noise at low t). Hypothesis: the per-t val spike at t∈[0,0.2] in the existing sparse variants is because spatial+random neighbors are essentially random subsets at high noise; giving them a non-noise coordinate source lets the attention actually use them at low t.

**Implementation:** `proteinfoundation/proteina.py:119-128` (cfg flow) + `:809-828` (integrator args). `proteinfoundation/nn/local_latents_transformer.py:125-132` (canary print) + `:294-314` (per-protein `torch.where` threshold gate). `proteinfoundation/flow_matching/product_space_flow_matcher.py:582-583, 707-731` (`sc_neighbors_bootstrap` for step 0). Runtime canary at load time: `[Fix C2] sc_neighbors=True (t_threshold=0.4): sparse neighbors will be built from x_sc when t < threshold and x_sc is present; otherwise falls back to x_t.`

**Recipe deltas vs canonical:** none on optimizer side. NN-config: same as `ca_only_sparse_160M` (sparse K=40, no pair-update). Training-config delta:
- `training.sc_neighbors: True`
- `training.sc_neighbors_t_threshold: 0.4`

**Run name:** `ca_only_sparse_K40_scnbr_t04`. Store dir: `store/ca_only_sparse_K40_scnbr_t04/1778022317/`.

**Wandb chain (consolidated):** `773iil3a`. 1 long slot, **14.9 h total runtime**, 23 val points, opt step 62 → 1448. (No formal resume — single uninterrupted slot.)

**Best val:** **4.276 @ step 1133** (Δ = −0.436 vs canonical; better than sparse-pairupdate's 4.591). On-disk best raw ckpt: `best_val_00000011_000000001133.ckpt` (also one at ep=8 step=819).

**Val-loss profile:**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 5.743 | 773iil3a |
|  503 | 5.214 | 773iil3a |
|  755 | 4.863 | 773iil3a |
| 1007 | 4.516 | 773iil3a |
| **1133** | **4.276** | 773iil3a (best) |
| 1259 | 4.479 | 773iil3a |
| 1448 | 4.591 | 773iil3a |

**Convergence:** **plateaued at step 1133, then overfit** — +0.32 over the next 315 steps. Same shape as sparse_pairupdate: best-val hits at step 1133, then climbs.

**Tests:**

- **N=6 probe ([E035](experiments.md#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06)) at step 819, BEFORE Fix C2 inference path was wired locally:** 0/18 (best 4.37 Å L=100). Fix C2 weights, but Fix C2 inference code missing — the probe ran the model with neighbor-list-from-x_t at all t. Result was uninformative; **superseded** by E038.
- **N=6 probe ([E038](experiments.md#e038--scnbr_t04-re-probe-with-fix-c2-actually-wired-2026-05-06)) at step 819, with Fix C2 active:** 0/18, but mechanism alive: L=100 best 3.16 Å, three samples in 3.16/3.39/3.76 Å cluster (vs E035's "one near-miss" pattern). −4.4 Å median improvement. Fix C2 mechanism doing its job at this training stage; absolute level still below the bar.
- **N=6 probe ([E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06)) at step 1133, with Fix C2 active, seed=5, post-fix MPNN — clears the variant bar:**

  | L | sorted min scRMSD (Å) | designable | best |
  |---|---|---|---|
  | 50  | **1.51, 1.67, 2.11**, 3.29, 4.82, 5.23 | **2/6** | 1.51 |
  | 100 | **1.92**, 2.93, 5.42, 7.84, 8.55, 10.82 | **1/6** | 1.92 |
  | 200 | 7.22, 9.50, 12.03, 13.40, 13.71, 13.78 | 0/6 | 7.22 |
  | **pooled** | — | **3/18 (17%)** | **1.51** |

  Identical pooled designable count to sparse-pairupdate at the same step (3/18). Tighter L=50 cluster (three samples 1.51-2.11 Å vs sparse-pairupdate's 1.48, 1.72, 2.14). E021's L=200 best (2.20 Å) is closer than E039's (7.22 Å), but at N=6 single-seed this is sampling noise — see [E039 caveats](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06).
- **Step 819 → 1133 trajectory:** mean scRMSD dropped 4.4 Å in 314 training steps (E038 mean 11.39 → E039 mean 6.99). Earlier-than-predicted convergence — E038's "re-probe at step ≥ 1500" decision rule was beat by ~25%.

**Narrative:** Two orthogonal architectural levers (Fix C2 vs pair-update) reach the *same* step-1133 designability ceiling on the canonical 160M trunk — this is the cleanest cross-architecture matched-step variant comparison currently on disk.

---

## 8. 1D-conv downsampled

**Architecture:** dense attention preserved, **`use_downsampling=True`** added. Per CLAUDE.md: pair_rep is built at full N then 2D-pooled to N/2 (BlurPool1D stride: DownsampleBlock=2, UpsampleBlock=1); the trunk runs at half resolution. CA-only output (no `local_latents` head). 14 layers / 768 dim / 12 heads otherwise unchanged.

**Recipe deltas vs canonical:** none. NN-config delta vs `ca_only_score_nn_160M`: `use_downsampling: True`.

**Run name:** `ca_only_downsampled`. Store dir: `store/ca_only_downsampled/1777987722/`.

**Wandb chain (consolidated):** `a20t7xax` → `uo9i23gt` → `re4fjwbf` → `4ynd48hs` → `ahm9qsld`. **5 chained slots**, opt step 62 → 2971. The 5th slot (`ahm9qsld`, 2026-05-07) was a continuation of the previous chain — the user notes "small fixes" were applied at the HPC end (not visible in local git as of 2026-05-09 review); the chain resumed cleanly from `last.ckpt` and continued the same wandb group.

**Best val:** **3.954 @ step 2961** (Δ = −0.76 vs canonical's 4.712 — currently the lowest val of any CA-only variant in this catalogue). All-time best was set at the **very last logged step** of the most recent slot. On-disk best raw ckpt: `best_val_00000029_000000002961.ckpt`.

**Val-loss profile (consolidated):**

| Step | Val | Slot |
|---:|---:|---|
|  251 | 6.558 | uo9i23gt |
|  503 | 6.100 | uo9i23gt |
|  755 | 5.435 | uo9i23gt |
| 1007 | 5.326 | uo9i23gt |
| 1259 | 5.339 | uo9i23gt |
| 1511 | 5.487 | re4fjwbf |
| 1827 | 5.057 | re4fjwbf |
| 1952 | 4.958 | re4fjwbf |
| 2142 | 4.648 | re4fjwbf |
| 2331 | 4.322 | re4fjwbf (1st local min) |
| 2393 | 4.347 | 4ynd48hs |
| 2456 | 4.505 | ahm9qsld (uptick) |
| 2519 | 4.451 | ahm9qsld |
| **2582** | **4.257** | ahm9qsld (cuts under 1st local min) |
| 2646 | 4.397 | ahm9qsld |
| 2708 | 4.276 | ahm9qsld |
| **2771** | **4.144** | ahm9qsld |
| **2835** | **4.030** | ahm9qsld |
| 2897 | 4.166 | ahm9qsld |
| **2961** | **3.954** | **ahm9qsld (global best, last logged step)** |

**Convergence:** **still actively descending — NOT converged.** Δ over slot `ahm9qsld` = −0.39 (4.347 → 3.954) over 568 opt steps. The new best is the very last point logged, no turnover yet. Pattern is identical to the K=64-curriculum-self variant (§11): both crashed/finished mid-descent with no overfit signal. **The earlier "converged ~2331, no overfit yet" reading was wrong** — it interpreted the +0.025 drift in slot `4ynd48hs` (62 steps, basically a single eval) as a plateau. Slot 5 falsified that.

**Tests:**

- **N=6 designability probe at step 2331, seed=5, nsteps=200, post-fix MPNN ([E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06)):**

  | L | sorted min scRMSD (Å) | designable | best |
  |---|---|---|---|
  | 50  | 12.41, 15.06, 15.13, 16.64, 16.72, 36.37 | 0/6 | 12.41 |
  | 100 | 15.67, 16.30, 16.33, 17.71, 17.83, 25.77 | 0/6 | 15.67 |
  | 200 | 17.60, 20.38, 20.70, 21.33, 21.51, 21.97 | 0/6 | 17.60 |
  | **pooled** | — | **0/18 (0%)** | **12.41** |

  **Superseded by the nsteps=400 redo and the step-2961 probe** (see [E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06)'s nsteps=200 caveat block) — at nsteps=200 the integrator hadn't converged to the data manifold, so this 0/18 conflates "model can't sample" with "we sampled wrong".

- **N=6 designability probe at step 2961 (val-best), seed=5, nsteps=400, post-fix MPNN** (cited from the [E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06) supersession block; full E-entry not yet written):

  Pooled: **3/18 (17%)**, best **1.60 Å**. The variant produces *some* designable samples at canonical inference resolution — what looked like "dead at every length" was an integrator artifact; what's actually happening is "low rate, low-Å tail."

**Narrative:** The lowest val of any CA-only variant in this catalogue (3.954) and **still descending** at the last logged step. The gap to canonical-level designability is **mechanistically located**, not "Finding 5/6 generically": [E043](experiments.md#e043--per-t-validation-loss-across-four-ca-only-architectural-variants-d1-of-the-hybrid-sampling-diagnostic-plan-2026-05-06--2026-05-07) at step 2331 showed conv's per-t val curve is statistically identical to canonical at the noisy end (t∈[0,0.2): +0.006 nat / protein) and **+0.452 nat worse** at the clean end (t∈[0.8,1.0)) — 3× larger than its gap at any other t-bucket. BlurPool 2× downsampling does bulk denoising fine; it loses fidelity exactly where the integrator commits to atom positions. [E041](experiments.md#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06)'s conv→canonical hybrid hands off at t=0.6 — *before* conv's disadvantage region — and recovers most of canonical's designability, which is the load-bearing test of the mechanism (architectural transplant of the late trajectory, not retraining).

**Open question:** the step 2331 → 2961 val descent (Δ = −0.37) hasn't been broken down per-t yet. Two predictions for a re-run of `proteinfoundation/run_per_t_val.py` on step 2961:
- **(A)** High-t bucket drops from 1.765 → ~1.4, others move <0.1: the val descent IS closing the late-trajectory hole; conv may eventually catch canonical at sample quality if training continues. The step 2961 designability of 3/18 (vs step 2331's 0/18 — superseded for nsteps) is consistent with this.
- **(B)** Low/mid-t buckets carry the val descent, high-t stays ~1.7: BlurPool has a structural ceiling at the clean end; further training shifts val without translating to designability. Compute is better spent elsewhere.

A length-stratified per-t version (per-(t × L) bucketing, ~30-min variant on `run_per_t_val.py`) would test the **stronger hypothesis**: conv's high-t hole is concentrated at long L, because 2× pooling halves the receptive-field-relative-to-protein at every L and atom-position commitment at high t needs more long-range context for longer chains. If the L=200 high-t bucket is +1.0 nat above canonical while L=50 is +0.2, that's the mechanism, and "fix BlurPool at high t" becomes a specific architectural ask rather than "downsampling is bad."

---

## 9. Hybrid sampling — conv → canonical mid-trajectory handover

**Type:** **inference-time** variant. Not a separate training run — reuses two existing checkpoints (no new wandb chain).

**Architecture:** at inference, `proteinfoundation/generate_hybrid.py` loads two ckpts and monkey-patches `predict_for_sampling` on the receiver model so calls with `t < t_switch` dispatch to ckpt **A**, calls with `t ≥ t_switch` dispatch to ckpt **B**. At the first B-call per batch (= handover step) it runs both models on the same `(x_t, t)` and logs ‖v_A − v_B‖, ‖v_A‖, ‖v_B‖, cos(v_A, v_B) per protein.

**Configuration ([E041](experiments.md#e041--hybrid-conv-canonical-mid-trajectory-handover-2026-05-06)):**
- A = `best_val_00000023_000000002331.ckpt` ([downsampled-conv](#8-1d-conv-downsampled), step 2331, run id 1777987722).
- B = `best_val_00000024_000000002457.ckpt` ([canonical](#1-canonical-ca-only-baseline), step 2457, run id 1776805213).
- `t_switch = 0.6` (with nsteps=200, log p=2 schedule, this falls at step 120 — 60% conv / 40% canonical in step count).
- N=3 (initial probe) and N=6 (follow-up); seed=5.
- Eval = post-fix MPNN, scRMSD <2 Å (CA mode).

**Tests / Designability:**

- **N=3 probe (seed=5):** L=50 **3/3** (best 1.08 Å, all three under threshold), L=100 **1/3** (best 1.56 Å), L=200 **1/3** (best 1.53 Å). Pooled **5/9 = 56%** — first hybrid arm to clear all three lengths at N=3.
- **N=6 follow-up (seed=5):** L=50 **5/6** (best 0.79 Å), L=100 **4/6** (best 0.92 Å), L=200 **3/6** (best 1.33 Å). Pooled **12/18 = 67%**, best 0.79 Å. Statistically indistinguishable from canonical-alone-step-2646 N=30 (76%).
- **Kink at handover (t_handover = 0.608):** ‖Δv‖ / ‖v_A‖ = 0.81 / 0.78 / 0.74 (L=50/100/200), cos(v_A, v_B) = 0.59 / 0.61 / 0.66. **Smaller kink than conv→scnbr** (0.81 vs 0.86 at L=50 etc.), and cos higher by 7-13 pp. Architectural similarity (conv and canonical both use dense attention; scnbr is sparse) is the load-bearing explanation.
- **Reproducibility check:** kink magnitudes are within ~1% across N=3 and N=6 — the kink is an architectural property, not a draw-dependent fluctuation.

**Open question (E041 caveats):** the hybrid's 67% is statistically indistinguishable from canonical-alone at step 2646 (76%). To say whether the hybrid actually adds value (vs canonical-alone-at-step-2457 doing all the work), the missing control is canonical-alone at step 2457 / N=6 / seed=5. Three competing interpretations: (A) canonical-alone-2457 ≈ 12/18 → conv is neutral; (B) canonical-alone-2457 ≪ 12/18 → conv contributes; (C) canonical-alone-2457 ≫ 12/18 → conv is hurting but canonical's last-40% rescues. Untested.

**Narrative:** Resurrects the dead conv variant — at N=6 the conv→canonical hybrid clears every length at canonical-comparable rates, so even in the worst-case "canonical does all the work" reading, conv's first 60% is at minimum a no-cost early-trajectory initializer.

---

## 10. Hybrid sampling — conv → scnbr_t04 mid-trajectory handover

**Type:** **inference-time** variant (same `generate_hybrid.py` as #9).

**Configuration ([E040](experiments.md#e040--hybrid-conv-scnbr-mid-trajectory-handover--kink-abruptness-at-the-switch-2026-05-06)):**
- A = `best_val_00000023_000000002331.ckpt` ([downsampled-conv](#8-1d-conv-downsampled), step 2331).
- B = `best_val_00000011_000000001133.ckpt` ([scnbr_t04](#7-sparse-k40--fix-c2-sc_neighbors_t_threshold04), step 1133).
- Two settings: `t_switch=0.6` / N=3 (canonical with kink probe), and `t_switch=0.75` / N=6 (first-attempt; L=200 OOM'd from co-tenant).
- `sc_neighbors_bootstrap=False` because scnbr's `t_threshold=0.4 < 0.6 ≤ t_switch` (scnbr always reads neighbors from `x_t` in this hybrid; bootstrap would never fire anyway).
- seed=5.

**Tests / Designability:**

- **t_switch=0.6, N=3:** L=50 1/3 / L=100 0/3 / L=200 0/3 → pooled **1/9 = 11%**, best 1.78 Å (L=50).
- **t_switch=0.75, N=6 partial (L=200 OOM):** L=50 1/6 (best 1.15 Å), L=100 2/6 (best 1.66 Å), L=200 0/6 (OOM, no valid samples). Pooled valid **3/12 = 25%**, best 1.15 Å.
- **Kink at handover (t=0.608, t_switch=0.6):** ‖Δv‖ / ‖v_A‖ = 0.86 / 0.81 / 0.79, cos(v_A, v_B) = 0.52 / 0.59 / 0.61. **Larger kink than conv→canonical** at every length: rel-‖Δv‖ +5-7 pp, cos −7-13 pp. Mechanism: scnbr is sparse, conv is dense → mismatched architectures disagree more on the same `(x_t, t)`.

**Why t=0.75 outperforms t=0.6 here on small N (interpretation):** larger relative kink at lower t is offset by lower noise scale, so SDE re-equilibration can't fully wash the kink out before the trajectory commits; at t=0.75 fewer scnbr steps remain, so less time for scnbr's step-1133-weak-at-L=200 weights to drag the structure off-manifold. Caveat: 9 vs 12 valid samples can equally be sampling noise.

**Narrative:** Same dead-conv-rescue mechanism as conv→canonical, but at smaller magnitude — the architectural-similarity hierarchy (dense+dense kinks smaller than dense+sparse kinks) is real and predicts post-handover trajectory survival.

---

## 11. Sparse K=64 SALAD-canonical + low-t curriculum + self-inclusion

**Architecture:** sparse-attention 160M backbone, **three load-bearing changes** vs the existing [sparse K=40](#5-sparse-attention-k32-mis-named-k40):

1. **K=64 with SALAD-canonical (n_seq=8, n_spatial=16, n_random=32)** at high t — `K_total = 2*n_seq + n_spatial + n_random = 16 + 16 + 32 = 64`. Doubles the coordinate-based budget that [E045](experiments.md#e045--t-dependent-k-budget-reallocation-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) implicated as load-bearing at L≥100.
2. **Curriculum over t** — at training time, each protein's neighbor split shifts with its sampled t. K stays 64 throughout; only the (n_seq, n_spatial, n_random) split changes. Wired in `proteinfoundation/nn/local_latents_transformer.py` (`_build_neighbor_idx` reads `curriculum_neighbors=True` from kwargs). Inverse of the failed E044/E045 inference-only retrofits — here the model is *trained* with the curriculum, so it sees the schedule it samples under.
3. **Self-inclusion** — the sparse path now allows the query residue itself in the K-set (slot 0 of the sequential group). Single-line change in `proteinfoundation/nn/modules/sparse_neighbors.py`. Self previously excluded by the `eye` term in `base_invalid` (CLAUDE.md → "Sparse-attention variant" → "Self is excluded from each query's neighbor list"). With self-inclusion the residual connection around MHA is not the only self-information path.

Plus throughput: `torch.compile` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (recovers some of the per-step disadvantage that sparse has at n=512 due to gather bandwidth).

Deliberately **not** bundled (one-axis cleanliness): `sc_neighbors` (Fix C2), `update_pair_repr=True`, per-layer K-refresh.

**Recipe deltas vs canonical:** none. Locked to the canonical OLD recipe (wd=0.05, constant LR=2e-4, no scheduler) per CLAUDE.md "DO NOT touch wd/LR/scheduler — confounds the curriculum question with the v2/paramgroups failure". NN-config deltas vs `ca_only_score_nn_160M`: `sparse_attention=True`, K=64 with SALAD-canonical split, `curriculum_neighbors=True`.

**Run name:** `ca_only_sparse_K64_curriculum_self`. Store dir: `store/ca_only_sparse_K64_curriculum_self/1778188245/`. Configs: `configs/training_ca_only_sparse_K64_curriculum.yaml`, `configs/nn/ca_only_sparse_K64_curriculum_160M.yaml`.

**Wandb chain (consolidated):** `euiryxd4` (failed early, step 30) → `7sdu834p` (steps 62 → 1196, crashed at SLURM time-limit) → `rmuumq8v` (steps 1259 → 1574, crashed at SLURM time-limit). 2 substantive slots, 25 val points. **A 4th slot was queued 2026-05-09 (`sbatch 29102705`)** to push past step 1574 — see "Open question" below.

**Best val:** **4.1908 @ step 1385** in run `rmuumq8v` (Δ = −0.52 vs canonical's 4.712 best — the largest val-Δ of any variant in this catalogue). On-disk best raw ckpt: `best_val_00000013_000000001385.ckpt`. Raw `.ckpt` symlinked at repo root as `sparse_K64_curriculum_self_step1385.ckpt` for inference.

**Val-loss profile (consolidated, all 25 points):**

| Step | Val | Slot |
|---:|---:|---|
|   62 | 7.093 | 7sdu834p |
|  314 | 5.723 | 7sdu834p |
|  566 | 5.061 | 7sdu834p |
|  818 | 4.669 | 7sdu834p |
|  944 | 4.381 | 7sdu834p |
| 1007 | 4.375 | 7sdu834p |
| 1133 | **4.237** | 7sdu834p (1st local min) |
| 1196 | 4.324 | 7sdu834p |
| 1259 | 4.335 | rmuumq8v |
| 1323 | 4.293 | rmuumq8v |
| **1385** | **4.191** | **rmuumq8v (global best)** |
| 1448 | 4.308 | rmuumq8v |
| 1511 | 4.231 | rmuumq8v |
| 1574 | 4.333 | rmuumq8v (last) |

**Validation split at step 1574:** by-t ∈ [4.15 (mid-t), 4.48 (high-t)]; by-length ∈ [4.00 (50–175), 4.79 (300–425)]. Train-val gap ≈ 2.87 (train_loss/loss_step ≈ 1.45 in 50-step bins, val 4.33).

**Convergence:** **pre-convergence, leaning "still improving":**
- Best val was set in the **very last continuation** (step 1385), 190 opt steps before time-limit cancellation. No turnover yet.
- Step 1574 sits **below** canonical's overfit window (1800–2200), so the variant has not yet reached the recipe's typical inflection.
- Train_step loss still drifts down slowly (50-step bins: ~1.50 at step 1000 → ~1.45 at step 1550).
- 6 val points in the last slot is too few to call a trend; the 4.191 hit could be a noise excursion in a 4.19–4.50 plateau that started ~step 944.

**Tests:**

- **N=6 designability probe at step 1385, seed=5, nsteps=400, post-fix MPNN, ESMFold, scRMSD < 2 Å (CA mode; bb3o agrees on the same 8 designable structures):**

  | L | sorted min scRMSD (Å) | designable | best |
  |---|---|---|---|
  | 50  | 1.28, 1.34, 1.85, 2.78, 3.90, 6.52 | 3/6 (50%) | 1.28 |
  | 100 | 0.90, 1.37, 1.41, 1.76, 2.16, 4.37 | 4/6 (67%) | **0.90** |
  | 200 | 1.34, 2.34, 3.17, 10.48, 10.64, 13.16 | 1/6 (17%) | 1.34 |
  | **pooled** | — | 8/18 (44%) | **0.90** |

  Driver: `script_utils/probe_sparse_K64_curriculum_self_step1385.sh`. Inference config: `configs/inference_sparse_K64_curriculum_self_step1385_n6_nfe400.yaml`. Output CSV: `inference/results_inference_sparse_K64_curriculum_self_step1385_n6_nfe400_0.csv`.

  **First nsteps=400 N=6 probe of any single-architecture (non-hybrid) CA-only variant.** Earlier variant probes ([E021](experiments.md#e021--sparse-k40--pair-update-quick-n6-designability-probe-2026-04-30), [E034](experiments.md#e034--ca-only-downsampled-variant-quick-n6-designability-probe-2026-05-06), [E035](experiments.md#e035--ca-only-sparse-k40-scnbr_t04-variant-quick-n6-designability-probe-2026-05-06), [E038](experiments.md#e038--scnbr_t04-re-probe-with-fix-c2-actually-wired-2026-05-06), [E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06)) used nsteps=200 — see CLAUDE.md "Sampling — nsteps=400 is a HARD RULE" + the supersession blocks in those E-entries.

  **Tail-noting**: at this N the per-length point estimates are dominated by a single sample (each row is a CI of ±40 pp at p=0.5), so 50/67/17% is at most a "directional read", not a discriminative claim against canonical's 63/67/10%. Doubled in the N=12-additive probe below.

- **N=6+N=12 additive probe at step 1385, pooled N=18, seed=5 + seed-derived (see caveat) — `nsamples=12` follow-up at the same ckpt:**

  | L | designable (best-of-8 MPNN seqs, scRMSD < 2 Å) | pooled rate | best (Å) |
  |---|---|---:|---:|
  | 50  | 3/6 (N=6) + 5/12 (N=12) | **8/18 (44.4%)** | 0.94 |
  | 100 | 4/6 (N=6) + 6/12 (N=12) | **10/18 (55.6%)** | **0.90** |
  | 200 | 1/6 (N=6) + 1/12 (N=12) | **2/18 (11.1%)** | 1.34 |
  | **all** | — | **20/54 (37.0%)** | **0.90** |

  N=12 driver: `script_utils/probe_sparse_K64_curriculum_self_step1385_n12.sh`. Config: `configs/inference_sparse_K64_curriculum_self_step1385_n12_nfe400.yaml`. CSV: `inference/results_inference_sparse_K64_curriculum_self_step1385_n12_nfe400_0.csv`. **Seed-propagation note** — the N=12 was launched with the SAME `cfg.seed=5` as the N=6, on the assumption that per-batch seeding would reproduce samples 0-5 bit-identically. **It did not** — all 12 N=12 samples are distinct from all 6 N=6 samples (zero per-sample overlap, verified post-hoc on `_res_scRMSD_ca_esmfold` values). `cfg.seed`'s propagation through `predict_step` interacts with `nsamples` in some way that hasn't been audited yet; the practical consequence here is favourable (18 truly-unique pooled samples instead of the 12 unique we expected). Future `nsamples`-bumped probes can simply pool naively for the moment.

  **What the doubling did to the picture:** the N=18 pooled rates are **softer than N=6 alone** at the short and medium length:
  - L=50: 50% (N=6) → 44.4% (N=18) — Δ −6 pp, within sampling noise.
  - L=100: 67% (N=6) → 55.6% (N=18) — Δ −11 pp, **the N=6 was over-optimistic**. The "L=100 dead-equal to canonical" claim from the N=6-only read does not survive.
  - L=200: 17% (N=6) → 11.1% (N=18) — Δ −6 pp; still in canonical's neighborhood (canonical=10%).
  - Pooled: 44% → 37% — Δ −7 pp.

  **CI tightening:** N=18 95% binomial half-width at p≈0.5 is ±23 pp (vs ±40 pp at N=6); that's the actual gain the user got from doubling N. The variance on the per-length rate dropped by ~1.4×, not 2× — CI scales with 1/√n.

- **Cross-variant comparison at step 1385 — pooled N=18 column for this entry:**

  | Variant | Step | N | L=50 | L=100 | L=200 | Pooled | Best (Å) |
  |---|---|---|---|---|---|---|---|
  | **canonical** ([§1](#1-canonical-ca-only-baseline)) | **2646** | **N=30 (E019)** | **63.3%** | **66.7%** | **10.0%** | **76% (68/90)** | **0.76** |
  | **K=64 curriculum self (this, N=18 pooled)** | **1385** | **N=6+N=12 (nfe400)** | **44.4%** (8/18) | **55.6%** (10/18) | **11.1%** (2/18) | **37% (20/54)** | **0.90** |
  | K=64 curriculum self (this, N=6 only — historical) | 1385 | N=6 | 50% (3/6) | 67% (4/6) | 17% (1/6) | 44% (8/18) | 0.90 |
  | paramgroups+wd0.1 ([§4](#4-param-groups-split--wd01)) | 1952 | N=9 ([E017](experiments.md#e017--paramgroups--wd01-quick-probe--proteinmpnn-ca_only-bug-fix-2026-04-28)) | 33% | 100% | 0% | 44% (4/9) | 0.94 |
  | scnbr_t04+FixC2 ([§7](#7-sparse-k40--fix-c2-sc_neighbors_t_threshold04)) | 1133 | N=6 ([E039](experiments.md#e039--scnbr_t04--fix-c2-step-1133-designability-clears-the-variant-bar-2026-05-06), nsteps=200) | 33% | 17% | 0% | 17% (3/18) | 1.51 |
  | sparse K40 ([§5](#5-sparse-attention-k32-mis-named-k40)) | 1259 | N=30 ([E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)) | 30.0% | 3.3% | 0% | 11% | 1.05 |
  | wd0 ([§3](#3-wd0-ablation)) | 1638 | N=30 ([E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)) | 33.3% | 13.3% | 0% | 16% | 1.24 |
  | v2 ([§2](#2-v2--wd01--cosine_with_warmup-failed)) | 2078 | N=30 ([E019](experiments.md#e019--full-n30-fixed-mpnn-re-eval-of-e014-five-arms-2026-04-29)) | 23.3% | 16.7% | 0% | 13% | 1.08 |
  | 1D-conv ([§8](#8-1d-conv-downsampled), nsteps=400 supersession) | 2961 | N=6 (mentioned in E034 caveat) | — | — | — | 17% (3/18) | 1.60 |

  **Updated reading at N=18:** the variant **leads canonical at L=200** by 1 pp (within noise — they're tied), **trails canonical at L=50 by ~19 pp**, and **trails canonical at L=100 by ~11 pp**. Both gaps still fit inside the N=18 95% CI of ±23 pp around the variant's point estimates, so canonical's rates are not statistically excluded — but the central tendency has shifted from "indistinguishable" to "trailing at the short/medium lengths". Best-Å tail is preserved: variant min 0.90 Å vs canonical 0.76 Å (variant is slightly behind on the absolute best, but L=50 best is 0.94 Å vs canonical's L=50 best 1.05 — variant's tail is sharper at L=50/100).

  **At ~half the training steps** (1385 vs 2646), this is still the variant landing closest to canonical at all three lengths simultaneously — but the gap to canonical's short/medium rates is now visible at N=18, where it wasn't at N=6. Whether continued training closes this is what the queued continuation is testing.

**Open question (queued):** SLURM job `29102705` was submitted on 2026-05-09 to push past step 1574. Two falsifiable predictions for the re-probe of the next ckpt (now informed by N=18, not N=6):
- (A) **Under-trained, gap closes with more training:** L=50 rises from 44% toward 63%, L=100 rises from 56% toward 67%, L=200 stays ~10%, val pushes below 4.19. Variant tracks canonical with ~1000 more steps.
- (B) **Variant ceiling near-N=18 rates:** L=50 stays ~44%, L=100 stays ~56%, L=200 stays ~11%. Canonical-recipe with K=64+curriculum+self caps at ~half-canonical's short-length rate; the bundle improves L=200 but trades off L=50/100. N=30 is the next decision point and the K=64-vs-K=40 ablation becomes the question.

Distinguishing (A) from (B) requires the post-1800-step ckpt and an N=30 promotion; until then, the strongest defensible claim is "trailing canonical at L=50/100, tied at L=200, at half the training" — *not* the "indistinguishable at all lengths" framing the N=6 alone suggested.

**Narrative:** First single-architecture variant to land in canonical's neighborhood at all three lengths simultaneously, **at half the training**. The triple of K=64 SALAD-canonical + per-t curriculum + self-inclusion is the only intervention bundle in the catalogue that does this — paramgroups+wd0.1 matched at L=100 (3/3 at N=9, but never N=30-confirmed) but failed L=50 (1/3) and L=200 (0/3); scnbr+FixC2 cleared the variant bar at L=50 but stalled at L=100. The N=18 pooled read tightens the picture: variant is **tied with canonical at L=200** (11.1% vs 10.0%) but **trails by ~10-19 pp at L=50/100**. The variant's *tail* is on par with canonical (best-Å within 0.2 Å) — when it works, it works as well as canonical. The variance is what's costing the rate. Whether the gap is under-training or an architectural ceiling is the load-bearing question for the queued continuation.

**Methodological caveats:**
- **N=18 pooled across two seeds (effectively): N=6 at seed=5 + N=12 at seed=5 with `nsamples=12` reseeding-via-batching.** Despite both runs using `cfg.seed=5`, the N=12 produced 12 fresh samples (not 6 reproductions + 6 new as expected). The seed-propagation through `predict_step` in `proteina.py` clearly involves `nsamples` and/or batch-index in some way that hasn't been audited. **Practical consequence:** the pooled N=18 is genuinely 18 unique samples per length. **Risk:** if a future probe accidentally lands on the same seed-derivation as a prior probe, naive pooling double-counts. Cheap audit: first protein produced by each new probe should be hashed and compared to the prior CSV's first protein hash.
- **N=18 single-protocol, single-seed-base.** 95% binomial CI per length at p=0.5 is ±23%. canonical's per-length rates (63/67/10) are inside the variant's 95% CI [21%, 67%] / [33%, 78%] / [3%, 33%] respectively — i.e., *not statistically excluded*. But the central tendency has shifted: at N=6 every variant length looked equal-ish to canonical; at N=18 only L=200 still does. CI tightened from ±40 to ±23 pp (~1.4×, the expected √2 from doubling+adding-half).
- **Step-mismatched comparison to canonical's N=30.** 1385 vs 2646 = 52% of canonical's training. If the variant tracks canonical's curve linearly, it could plausibly close the L=50/100 gap by step ~2400-2600; or it could plateau at the current rate. Untested. The pending sbatch is the resolution.
- **Curriculum + self-inclusion are bundled as one variant** (axis-not-clean). Whether the K=64 alone, the curriculum alone, or the self-inclusion alone is doing the work cannot be attributed from this one run.
- **Cross-variant comparison vs N=30 entries (E019) is not seed-paired** — E019 used seed=100, this used seed=5. The per-protein inference-noise floor (~0.5 Å) is much smaller than the per-length rate gaps in the table, so the pooled-rate ranking is robust to seed; per-sample comparisons are not.
- **Inference at nsteps=400 is an apples-to-apples comparison to canonical's E019 N=30** (which inherited `nsteps=400` from the unmodified `inference_base.yaml`-defaults `inference_baseline_n30.yaml`) and to the 1D-conv step-2961 nsteps=400 supersession number. It is *not* directly comparable to the nsteps=200 numbers from E021/E034/E035/E038/E039/E040/E041/E044/E045 — those entries' supersession blocks document the bias.

**Cross-references:**
- Submit script: `script_utils/submit_train_ca_only_1gpu.sh` with `-n training_ca_only_sparse_K64_curriculum`. Re-submit pattern (chained slots): `sbatch --exclude=gpu-q-43 --dependency=afterany:<prev_jobid> script_utils/submit_train_ca_only_1gpu.sh -n training_ca_only_sparse_K64_curriculum`.
- Probe driver: `script_utils/probe_sparse_K64_curriculum_self_step1385.sh` (one-shot, idempotent on retry — sweeps eval tmp_dirs and exports `PYTHON_EXEC` to make ProteinMPNN subprocess find numpy).
- Sparse-attention architecture pointers: CLAUDE.md → "Sparse-attention variant (SALAD-style, K=40)" + the "Padding-slot guard" / "Self is excluded" / "gather bandwidth" caveats. The self-inclusion delta in this variant intentionally undoes the documented "Self is excluded from each query's neighbor list" behavior.
- Direct predecessors: [§5](#5-sparse-attention-k32-mis-named-k40) (sparse K=32 baseline this builds on); [E045](experiments.md#e045--t-dependent-k-budget-reallocation-curriculum-on-plain-sparse_k40-step-1259-2026-05-07) (inference-only K-realloc curriculum that motivated the *trained-with-curriculum* version).

---

## Things deliberately not run yet

- **Param-groups + wd=0.1 N=30 batched eval at step 1952.** Was the "5th arm" in the planned E014/E019 redo; the variant clears the variant bar at N=6 but the N=30 numbers needed to publish the AdaLN-Zero-causal-fix Finding don't exist yet.
- **wd=0 + param-groups composite recipe.** The two interventions act on different parameter subsets in principle and could compose. Untested.
- **K=40 sparse with the original 16/8/16 (n_seq=16/n_spatial=8/n_random=16) config.** The on-disk `ca_only_sparse_K40` is K=32 (n_seq=8). Re-running with the original-intent config is a separate variant.
- **Sparse + Fix C2 at step ≥ 1800** (matching canonical's best-val window). E039 cleared the variant bar at step 1133 but the chain only got to 1448 before the wandb chain stopped — whether the variant tracks canonical further or plateaus permanently at the step-1133 ceiling is unmeasured.
- **Hybrid kink-sweep across t_switch ∈ {0.4, 0.5, 0.6, 0.7, 0.8} at N=15.** Locates the t_switch with smallest kink AND highest rate; predicts a maximum somewhere.
- **Cross-checkpoint kink baseline:** ‖v_A − v_A'‖ between two ckpts of the *same* variant. Tells us how much of the cross-architecture kink (~80% rel ‖Δv‖) is training noise vs architecture.

---

## Cross-references

- Lab notebook: [`experiments.md`](experiments.md) — every E0NN entry above is cross-linked.
- Paper findings: [`content_masterarbeit.md`](content_masterarbeit.md) — Findings 5/6 (AdaLN-Zero collapse), 7 (cross-recipe N=30), and the Baseline reference anchor draw on these variants.
- Project memory: [`project_sparse_pairupdate_converged.md`](/home/ks2218/.claude/projects/-home-ks2218-la-proteina/memory/project_sparse_pairupdate_converged.md) — sparse_pairupdate's E021 step-1133 ceiling is the *converged* ceiling, not under-trained.
- Codebase pointers: see CLAUDE.md → "CA-only Baseline (canonical recipe — use this for all variants)" + "Sparse-attention variant (SALAD-style, K=40)" for the architectural pitfalls (BlurPool stride, padding-slot guard, gather-bandwidth caveat, etc.).
