# CLAUDE.md

Guidance for Claude Code working in this repo.

## Useful Info

**Pull a checkpoint from Cambridge HPC** (run from repo root):
```bash
rsync -avhP ks2218@login.hpc.cam.ac.uk:/rds/user/ks2218/hpc-work/store/test_ca_only_diffusion/<RUN_ID>/checkpoints/<CKPT>.ckpt ./
```
The trailing `./` is mandatory — without it rsync only lists the file (`sent 20 bytes received 89 bytes`).

## Experiment Log (`experiments.md`) — auto-update, no matter how small

After ANY experiment, automatically and **without asking**, append an entry to `experiments.md` at the repo root. The bar for entry is "did we run code that produced a number" — training, eval, probe, smoke test, diagnostic, ablation, sweep, even a single-protein toy run. Do not wait for the user to confirm; do not ask whether the result "is worth logging"; do not batch-defer the write to "later" — write it the same turn you observed the result.

Each entry must contain:
- **ID + date** — assignable handle (next free `Eddd`).
- **Status** — finished / in progress / cancelled / failed.
- **Why ran** — the question the experiment answers and the decision/claim it feeds.
- **Configs** — exact setup: config files, recipe, hardware, run dir, wandb run IDs, checkpoint paths. Re-run-able from this entry alone.
- **Results** — every quantitative output. Tables, per-fold/per-length numbers, weight-norm diffs, designability counts. Not just the headline.
- **Possible narrative** — does this become a Finding? If yes, link to the corresponding section of `content_masterarbeit.md`. If no, write "non-narrative — kept for tuning/decision-making" and explain what decision it informs.
- **Methodological caveats** — what the data does *not* support.

Keep the index table at the top of `experiments.md` in sync (add a row pointing at the new section). Append-only: never delete an entry when it is later superseded; instead add a back-link from the old to the new. Log entries are the lab record; they outlive any one finding.

## Paper Findings Log (`content_masterarbeit.md`) — curated narrative subset

For experiments whose result is defensible enough to write into the paper as a claim, negative result, or methodological observation, also add a **Finding** entry to `content_masterarbeit.md`. The Finding format:
- **Experiment** (full setup — short, point at the corresponding `experiments.md` E-ID for the lab-notebook detail).
- **Numbers** (the headline numbers; per-fold if CV).
- **Narrow claim** (strictest defensible statement; avoid overclaiming).
- **Implikation** (cautiously-phrased broader significance, explicitly separated from the narrow claim).
- **Methodische Einschränkungen** (what the data does *not* support).

Maintain a `Future Experiment Ideas` section at the bottom. Write in English. Keep code/config names verbatim.

**Relationship between the two files:**
- `experiments.md` = comprehensive log of EVERY run (auto-updated, append-only).
- `content_masterarbeit.md` = curated narrative subset that becomes paper findings.
- Every Finding in `content_masterarbeit.md` should have a matching E-entry in `experiments.md` (the experiment record); the Finding is the paper claim, the experiment record is the lab-notebook detail.
- A `experiments.md` entry without a matching Finding is normal and expected — most tuning runs and diagnostics never become findings. Do not pressure them to.

## Project Overview

**La Proteina**: deep-learning codebase for atomistic protein generation via partially latent flow matching (backbone + side chains + sequence).

## Environment

**Canonical env on Cambridge HPC: `/home/ks2218/conda_envs/laproteina_env`** (NOT on RDS/Lustre — Python imports hang for minutes-to-hours when an OST is evicted, because `.pyc` files are sharded across OSTs).

Build: `script_utils/build_env.sh` (auto-installs micromamba, builds in /home).
- Pin **python=3.10**, not 3.11: pyg cp311 wheels for `torch 2.7.0+cu118` need GLIBC 2.32; HPC nodes have GLIBC 2.28.
- **torch 2.5.1**, not 2.7.0: same GLIBC reason for pyg's `torch_scatter`/`sparse`/`cluster` wheels.

`.env` needs `DATA_PATH=...` and `WANDB_API_KEY=...`.

## Common Commands

```bash
sbatch script_utils/submit_train.sh -n <config_name>          # train (cluster)
bash   script_utils/full_training_test.sh -n <config_name>    # train (local)
sbatch script_utils/compute_latents.sh                        # precompute latents
python proteinfoundation/generate.py --config_name <cfg>      # sample
python proteinfoundation/evaluate.py --config_name <cfg>      # evaluate
sbatch script_utils/run_developability.sh                     # property panel
```

**User preference: do NOT use precomputed latents for training.** Precomputing removes the per-epoch augmentation diversity from the on-the-fly AE encoder. Default `use_precomputed_latents: False`. Don't suggest precomputing as an optimization without explicit confirmation.

No formal test suite. Tracking via Weights & Biases.

## Sampling — `nsteps=400` is a HARD RULE

**Every inference run that produces structures evaluated with scRMSD or any structure-derived property MUST use `nsteps=400`.** This is non-negotiable, the user has restated it multiple times, and violating it has invalidated entries repeatedly. `inference_base.yaml` ships with `nsteps: 400` — *do not override it downward.*

The cliff is real: identical seed, identical model (LD3+AE2), identical L=300 — single-protein scRMSD goes from **22.5 Å at nsteps=200 → 0.80 Å at nsteps=400**. Below 400, the integrator hasn't converged to the data manifold; structures are off-manifold garbage even when the model is fine.

**Every nsteps<400 run produces a number that is below the integrator-convergence bar and therefore not comparable to canonical (E019).** This has poisoned multiple comparisons:
- E020 / E026 (1000-protein stratified property/AA-composition panels) — ran nsteps=200; superseded by E025 / E028 nsteps=400 regen, originals retained as historical record only.
- E021 / E034 / E035 / E038 / E039 (variant probes) and E040 / E041 (hybrid probes) — all ran nsteps=200 against canonical-at-nsteps=400. Variant numbers are handicapped; "variant vs canonical" claims are not defensible without an nsteps=400 re-probe.
- The original net_charge steering sweep (`run_steering_camsol_tango.sh`, NSTEPS=100) had a clean predictor-side dose-response with 22 Å scRMSD outputs.

**Hard rules for new configs and runs:**
- `configs/inference_*.yaml` should inherit `nsteps` from `inference_base.yaml`. Do not add a `generation.args.nsteps` override unless the goal is *not* a structure-quality eval (e.g. latent-trajectory diagnostics, predictor-side dose-response, per-t validation loss). "I want it to run faster" is not a reason.
- `script_utils/run_*.sh` driver scripts must pin `NSTEPS=400`, not 100/200.
- Wall-cost: `nsteps=400` is roughly 2× `nsteps=200` — ~3-4 h on 1× A100 for an N=6 designability probe at L∈{50, 100, 200}. Not catastrophic.
- The `*_quick.yaml` and `*_smoke.yaml` historical templates that overrode to 200 have been flipped to 400 (2026-05-07). Don't restore them. New `_quick`/`_smoke` configs must also stay at 400.
- An entry that uses `nsteps < 400` for designability evaluation must call it out as a methodological caveat in the entry's caveats list and explain why.

## Architecture Pointers

- `proteinfoundation/proteina.py` — central `LightningModule`. `_ca_only_mode` is True when `local_latents` is absent from `product_flowmatcher` (skips AE; `latent_dim=None`).
- `proteinfoundation/flow_matching/product_space_flow_matcher.py` — flow matching. `data_modes` is derived from config keys; `sample_t` reads `cfg_exp.loss.t_distribution.shared_groups`.
- `proteinfoundation/partial_autoencoder/` — VAE compressing full-atom → per-residue latent. Encoder is ~15-20% of GPU compute.
- `proteinfoundation/nn/LocalLatentsTransformer` — used for both full-latent and CA-only (different config). DiT-style **AdaLN-Zero** conditioning.
- `proteinfoundation/datasets/pdb_data.py` — `in_memory=True` requires `torch.multiprocessing.set_sharing_strategy('file_system')` (set in `train.py`). Symlink: `data/pdb_train/processed_latents → /rds/user/ks2218/hpc-work/processed_latents`.
- `precompute_latents.py` — atomic writes (`torch.save(tmp); os.rename`), `weights_only=False` (PyG Data objects), `keys_to_keep = ['mean','log_scale','coords_nm','coord_mask','id','residue_type','bb_ca']`. AE latents are NOT rotation-invariant — keep `GlobalRotationTransform` during training.
- `proteinfoundation/analysis/compute_developability.py` — 20-column property CSV (SWI, TANGO, CANYA, charge/pI, IUPred3, Shannon, Rg, hydrophobic patches, SAP, SCM). Resume-safe (skips processed `pdb_id`s). `--skip-tango` flag. Always-NaN: `camsol_intrinsic`. Single FreeSASA call per protein.

### Config Pitfalls
- `shared_groups` indents under `loss.t_distribution`, not `loss`. 4 spaces.
- `OmegaConf.get("k", default)` returns `None` when key is `null` — check explicitly.
- `autoencoder_ckpt_path` must be absent/null in CA-only (`proteina.py:66` asserts).
- `p_folding_n_inv_folding_iters: 0.0` in CA-only.
- `BlurPool1D` stride: `DownsampleBlock` = 2, `UpsampleBlock` = 1.

### Performance
- DDP sync (every `accumulate_grad_batches` steps) dominates, not the AE encoder.
- `ema.every_n_steps: 5` (every-step is too slow).
- 70M model is ~2× faster than 160M for the same architecture.

## CA-only Baseline (canonical recipe — use this for all variants)

Use the **OLD recipe** for any architectural variant (sparse attention, conv downsampling, etc.) so comparisons stay clean. The 2026-04-24 v2 recipe (wd=0.1 + cosine LR) is a documented failure — see below.

- **Config:** `configs/training_ca_only.yaml` with `weight_decay: 0.05` and the `scheduler:` block removed (constant LR). The on-disk YAML may carry v2 values — restore before training.
- **NN:** `configs/nn/ca_only_score_nn_160M.yaml`. Source-of-truth for the NN block: `store/test_ca_only_diffusion/1776805213/checkpoints/exp_config_test_ca_only_diffusion.json`. Diff variant configs against it.
- **Submit:** `script_utils/submit_train_ca_only_1gpu.sh` (1× A100 ampere). Chain via `--dependency=afterany:$prev`. **Always `--exclude=gpu-q-43`** — broken GPU; `afterany` re-routes the chain back to it.
- **Optimizer:** AdamW, β=(0.9, 0.999), eps=1e-8, **wd=0.05 (uniform; codebase doesn't split parameter groups)**.
- **LR:** constant 2e-4 (sqrt-scaled from the 4-GPU 4.15e-4 baseline). No warmup, no decay.
- **Effective batch:** `batch_size=6 × max_padding_size=512 × accumulate_grad_batches=32 ≈ 192 proteins/step`.
- **EMA:** decay=0.999, every_n_steps=5. Companion `-EMA.ckpt` per checkpoint.
- **Validation:** `val_check_interval=2000` mini-batches (~63 opt steps). Val set 4058 proteins. **`validation_loss/loss_epoch` is a misleading proxy for sample quality** under uniform-wd AdamW (see Finding 5/6) — always confirm with a designability probe (N=3-10 at L ∈ {50, 100, 200}) before trusting val numbers.
- **Convergence:** best val ≈ 4.71-4.77 around opt step 1800-2200, then overfits. Best ckpt on disk: `store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt`.
- **Wall-clock:** ~131 opt-steps/h with current pre-clip param-norm/grad-norm logging in `proteina.py:on_before_optimizer_step` (the two full-parameter L2 traversals are the bottleneck). Best is ~16h of chained slots.
- **Sample-quality bar (variants must clear):** 1-2/3 designable at L=50 and L=100 (scRMSD < 2 Å), not all lengths simultaneously.

**Variant checklist:** new NN config under `configs/nn/`; new training config with own `run_name_:`; lock recipe values verbatim; submit with `--exclude=gpu-q-43`. After ~1000 opt steps, run a designability probe — if 0/9 designable, debug instead of chasing val loss further.

### Why wd ≤ 0.05 (AdaLN-Zero × AdamW)

DiT-style AdaLN-Zero gates (`*.scale_output.to_adaln_zero_gamma`) are zero-initialized and must *grow* during training. `configure_optimizers` applies a single wd to all parameters — including these gates. At wd=0.05 gates win; at wd=0.1 they collapse, especially in deeper layers. Result: val MSE looks great, but conditioning is suppressed and integrated trajectories drift off the data manifold. To safely use wd > 0.05, first restructure `configure_optimizers` to split parameter groups (~15 lines, standard DiT/SiT pattern: exclude biases, LayerNorm γ/β, embeddings, AdaLN-Zero gates).

### v2 Failed Experiment (2026-04-24, kept for reference)

`weight_decay=0.1` + cosine warmup (peak 2e-4 @ step 200, decay to 2e-5 @ 6000). Best val 4.765 → 4.437 (Δ -0.33), but **0/3 designable at every L tested**, scRMSD 9-11 Å vs old recipe's 2-8 Å. Cause: AdaLN-Zero gates in layers 7-13 collapsed to 26-60% of old-recipe magnitudes (Finding 5/6).

- Store dir: `store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/`
- Wandb chain: `9jp15of2 → 5rftn43a → 43xxlbzt`
- **Don't use the v2 ckpt for anything.**

### Operational notes (any chained training)

- **`fetch_last_ckpt`** picks file by mtime. If the most recent file is a `best_val_*.ckpt` saved after `last.ckpt`, the chain resumes from that — costs up to `val_check_interval / accumulate_grad_batches` steps.
- **gpu-q-43**: broken GPU. `afterany` keeps re-routing to it. Always `--exclude=gpu-q-43`.
- **Wandb run-ID** isn't carried across slots — each chained slot creates a new run with the same name.

### Sparse-attention variant (SALAD-style, K=40)

Architectural variant of the CA-only baseline. Replaces dense `[B,N,N,d]` pair representation + dense attention with a per-residue neighbor list of **K=40 (16 sequential + 8 spatial + 16 random ∝ 1/d³)**. Note the per-side semantics of `n_seq_neighbors`: with `n_seq_neighbors=8`, `n_spatial_neighbors=8`, `n_random_neighbors=16` in the YAML, the resulting neighbor count is `2*n_seq + n_spatial + n_random = 16 + 8 + 16 = 40` (see `sparse_neighbors.py:14` docstring and `k_seq = min(2 * n_seq, N - 1)` at line 56).

Implementation files (worth knowing if you touch any of them):
- `proteinfoundation/nn/modules/sparse_neighbors.py` — neighbor list builder (`@torch.no_grad`, recomputed each forward from `x_t["bb_ca"]`).
- `proteinfoundation/nn/modules/pair_bias_attn.py` — `_attn_sparse` is the actual sparse attention. Dense vs sparse paths are switched by the presence of `neighbor_idx` argument.
- `proteinfoundation/nn/modules/pair_update.py` — sparse pair update; **raises if `use_tri_mult=True`** (line 65) because triangular multiplication needs the full N×N grid.
- `proteinfoundation/nn/modules/pair_rep_initial.py` — sparse-aware pair builder.
- `proteinfoundation/nn/feature_factory.py` — has `_gather_sparse_pairs` fallback (line 130) for any pair feature without `supports_sparse=True`. **Every pair feature in the current configs has the fast path; if you add a new pair feature, set `supports_sparse=True` on its class and implement the `neighbor_idx` branch, otherwise the fallback materialises the full `[B,N,N,d]` dense tensor and gathers from it — defeating the point.**
- `proteinfoundation/nn/local_latents_transformer.py:228-242` — wires it up; `self.sparse_attention` flag from kwargs.

Config files (created 2026-04-25):
- `configs/nn/ca_only_sparse_160M.yaml` — byte-equivalent to `ca_only_score_nn_160M.yaml` except the four `sparse_*` keys at the bottom. Don't add other deltas.
- `configs/training_ca_only_sparse.yaml` — locked to the canonical old recipe (wd=0.05, constant LR=2e-4, no scheduler). `run_name_: ca_only_sparse_K40`.

Implementation gotchas to remember when reading the code:
- **Self is excluded from each query's neighbor list** (`eye` is added to `base_invalid` in `sparse_neighbors.py:44`). Self-information propagates only via the residual connection around MHA. Diverges from standard transformer attention.
- **"Random" neighbors are 1/d³-weighted**, not uniform — NOT BigBird-style global tokens. Long-range information transport relies on multi-layer composition.
- **The neighbor list is rebuilt every forward from `x_t`** (the noisy CA at the current diffusion timestep). At t≈0 spatial+random groups are essentially random subsets and only sequential neighbors carry useful info.
- **Padding-slot guard**: short proteins (<K=40 residues) get padded with index 0; `slot_valid` mask in `sparse_neighbors.py:121-127` distinguishes "padded slot pointing at residue 0" from "real neighbor 0". Don't remove this — without it short proteins double-count residue 0 in attention.

**Performance reality at n=512 (1× A100, 160M model):**
- Sparse is **slower per opt step than dense**, not faster. `_attn_sparse` materialises two `[B*H, N, K, D]` tensors per layer via `gather` on a non-contiguous index pattern; at B=6, H=12, N=512, K=40, D=64 that's ≈ 5 GB of memory-bound traffic per forward — the FLOP savings get eaten by gather bandwidth.
- Crossover with dense is hypothesised at n ≥ 1024 but not measured. For maxl=800 where dense doesn't fit, sparse is "infinity faster" because it's the only thing that runs.
- **Do NOT propose sparse attention as a throughput optimisation at n=512.** The architectural value (memory headroom for longer sequences) is real; per-step throughput at n=512 is not.
- Allocator tweak that occasionally helps gather-heavy bf16: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Things deliberately NOT combined with sparse on the first variant run (so the comparison stays clean): `update_pair_repr=True`, `use_downsampling=True`, non-default K values. Sweep these only after K=40 produces a working result.

## Cluster Notes (Cambridge HPC)

- **Data:** `/rds/user/ks2218/hpc-work/processed` (536K raw `.pt`), `/rds/.../processed_latents` (latents). 355K original latents archived as `.tar` shards in `/rds/.../latent_shards/`.
- **Restore latents:** `mkdir -p .../processed_latents && ls .../latent_shards/*.tar | xargs -P 32 -I{} tar -xf {} -C .../processed_latents`. Re-symlink: `ln -sfn .../processed_latents data/pdb_train/processed_latents`.
- **RDS inode quota:** 1,048,576 file limit. ~92% full as of Apr 2026; space (1.1TB) is not the bottleneck.
- **Lustre + ProcessPoolExecutor:** never eagerly submit tens of thousands of futures when workers do Lustre I/O. Use rolling submission (`SUBMIT_CHUNK = workers * 4`).
- **Tarring on Lustre:** slow on login nodes. Use parallel-shard approach on compute nodes; skip `-z` (metadata, not disk space, is the bottleneck).
- **Ampere partition:** 1 GPU = 32 CPUs + ~250G RAM (full quarter-node), regardless of `--cpus-per-task`.
- **SLURM kills jobs at time limit without warning** — use atomic writes for outputs.
- **Do NOT use `set -e` in SLURM scripts on this cluster.** TaskProlog `mkdir /var/spool/slurm/slurmd/logs` permission error kills the script before it executes. Use `set -uo pipefail`.
- **TANGO binary:** extract from `tango2_3_1.linux64.zip` in repo root; `chmod +x`; on PATH or via `TANGO_EXE=...`.
- **IUPred3:** at `~/iupred3/` (auto-discovered via `IUPRED3_DIR` or `~/iupred3` default).

## Steerability Pipeline (`laproteina_steerability/`)

Diagnostic pipeline for the latent space (Part 1: geometry; Part 2: property probes). Each `.pt` cached latent file has `mean[L,8]`, `log_scale[L,8]`, `coords_nm[L,37,3]`, `coord_mask[L,37]`, `id`, `residue_type[L]` (CA = atom index 1 in OpenFold order).

```bash
python -m src.part1_latent_geometry.run --config config/default.yaml
python -m src.part2_property_probes.run --config config/default.yaml
# add --synthetic for smoke test
```

Config knobs: `data.length_range`, `data.subsample`, `part2.property_file`, `part2.property_granularity`, `part2.decisions.*`. Property file (Part 2): the developability CSV needs `pdb_id → protein_id` rename + all-protein granularity.

**Apr 2026 results (56K proteins, 300-800 residues):** all 8 latent dims used (PR=7.69/8, no collapse). Disentangled (max Pearson 0.10, max MI 0.28 nats). Dims 3,5,6,7 multimodal. Weak length sensitivity (r=-0.04). Within-protein variance ≈ between-protein → protein-level steering with averaged latents is viable. Part 2 not yet run on real data.

t-convention: code uses **t=0 noise, t=1 clean** (`z_t = (1-t)*noise + t*z_clean`). The steering predictor's t-convention is NOT verified — flagged in code.

## Gradient-Based Steering (`steering/`)

Guides sampling at inference via gradients from a trained property predictor — no retraining. At each ODE step: estimate `z_1_est = z_t + (1-t)*v`, predict properties, backprop through the predictor only (`v` is detached), unit-normalize, scale by schedule `w(t)`, add to `nn_out["local_latents"]["v"]`.

**Integration:** `steering_guide=None` parameter on `ProductSpaceFlowMatcher.full_simulation()`. When `None`, zero new code runs. When set, a small if-block adds the guidance gradient (always `.detach()`'d — safe inside `torch.no_grad()`).

**Design:**
- Reconstruction guidance only (predictor consumes the clean estimate, not noisy `z_t`).
- Gradient flows only through `z_t → x1_est → predictor`, never through flow weights.
- Z-score space for objectives → multi-objective weighting is scale-invariant.
- Per-protein unit normalization → `w_max` directly controls step size.
- Latent channel only (`local_latents`); `bb_ca` is never modified.

**Predictor checkpoint** (from `laproteina_steerability/src/multitask_predictor/train.py`): `model_state_dict` (PropertyTransformer, 128-dim, 3 layers, 4 heads), `stats_mean[13]`, `stats_std[13]`. Property → head index in `steering/registry.py`.

**Diagnostics** (when `log_diagnostics: true`): per-step `t`, `w`, `grad_norm_raw/final`, `predicted_properties`. Returned in `additional_info["steering_diagnostics"]`. **Logs batch element 0 only.**

**Limitations:** predictor architecture hardcoded (must match training); `direction: target` uses L2 in z-score space (poor for skewed properties).

## Known Issues / Fixes (recurring)

- **`Too many open files`** with `in_memory=True` + workers → `torch.multiprocessing.set_sharing_strategy('file_system')` in `train.py`.
- **`EOFError: Ran out of Input`** = SLURM kill mid-write. Always atomic-write `.pt`/CSV outputs.
- **CA-only not running CA-only**: shell scripts had a hardcoded `nn=local_latents_score_nn_160M` override. Now use `-n <config_name>`.
- **Stale CSV/FASTA → MMseqs2 "empty FASTA"**: delete `df_pdb_..._latents.csv` and `seq_df_pdb_..._latents.fasta` if `processed_latents/` was recreated.
