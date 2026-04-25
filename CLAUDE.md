# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper-Relevant Findings Log

Every time we run an experiment (new or re-run) and find something that could plausibly be a finding for the Masterarbeit paper, evaluate:
1. Is this reportable as a paper claim, a negative result, or a methodological observation?
2. If yes, append it to `content_masterarbeit.md` at the repo root.

Each finding entry must have:
- **Experiment:** Exact setup (config, data, architecture, hyperparams, run directory). Enough that someone could re-run it.
- **Numbers:** All quantitative results, per-fold if CV was used.
- **Narrow claim:** The strictest, fully defensible statement the data supports. Avoid overclaiming.
- **Implikation:** Cautiously-phrased broader significance, explicitly separated from the narrow claim.
- **Methodische Einschränkungen:** What the data does *not* support.

Also maintain a `Future Experiment Ideas` section at the bottom of that file — as we discuss experiments not yet run, add them there.

Write in English (the Masterarbeit/paper is in English). Keep code/config names as they appear in the codebase.

## Project Overview

**La Proteina** is a deep learning research codebase for atomistic protein generation via partially latent flow matching. It trains generative models to design protein structures (backbone + side chains) and sequences.

## Environment Setup

**Canonical env on Cambridge HPC: `/home/ks2218/conda_envs/laproteina_env`** (NOT `/rds/.../conda_root/...`). Reason: putting the env on RDS/Lustre makes Python `import` hang for ~minutes-to-hours when even one OST is evicted/disconn (Python's stdlib `.pyc` files get sharded across OSTs; one stuck `fstat` blocks every job's startup). /home is on a different (more reliable) filesystem.

To rebuild:
```bash
script_utils/build_env.sh   # auto-installs micromamba, mvs old env aside, builds in /home
```
The script pins **python=3.10** (NOT 3.11 as `environment.yaml` originally suggested) because pyg's cp311 wheels for `torch 2.7.0+cu118` require GLIBC 2.32, but Cambridge HPC nodes are on GLIBC 2.28. cp310 wheels work.

For first-time/manual install (matches what `build_env.sh` does):
```bash
mamba env create -f environment.yaml
mamba activate laproteina_env
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install graphein==1.7.7 --no-deps
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```
**Why torch 2.5.1 and not 2.7.0**: pyg-team's `torch_scatter`/`sparse`/`cluster` wheels for `pt27+cu118` were rebuilt against GLIBC 2.32, but Cambridge HPC nodes are GLIBC 2.28. The `pt25+cu118` wheels link against max GLIBC 2.14 and load fine. The actually-working /rds env uses torch 2.5.1.

Requires a `.env` file with:
```
DATA_PATH=/directory/where/you/want/dataset
WANDB_API_KEY=<key>
```

Checkpoints go in `./checkpoints_laproteina/` (local) or `/rds/user/ks2218/hpc-work/checkpoints_laproteina/` (cluster).

## Common Commands

**Training:**
```bash
# Submit training job (supports -n <config_name>, -d <data_path>, -c <checkpoint_dir>)
sbatch script_utils/submit_train.sh -n training_ca_only
sbatch script_utils/submit_train.sh -n training_local_latents

# Run locally (same flags)
bash script_utils/full_training_test.sh -n training_local_latents
```

**Precompute latents** (run before training with `use_precomputed_latents: True`):
```bash
sbatch script_utils/compute_latents.sh
```
**User preference: do NOT use precomputed latents for training.** Past experience indicated that switching to precomputed latents removed the per-epoch augmentation diversity ("multiple sides of the same protein") that the on-the-fly AE encoder produces. Default to `use_precomputed_latents: False` and run the AE encoder live each step, even though the throughput is slower. Don't suggest precomputing latents as an optimization without explicit confirmation.

**Sampling:**
```bash
python proteinfoundation/generate.py --config_name inference_ucond_notri
python proteinfoundation/generate.py --config_name inference_ucond_notri_long
python proteinfoundation/generate.py --config_name inference_motif_idx_aa
```

**Evaluation:**
```bash
bash script_utils/download_pmpnn_weights.sh  # one-time setup
python proteinfoundation/evaluate.py --config_name <config_name>
bash script_utils/gen_n_eval.sh
```

**Developability analysis:**
```bash
sbatch script_utils/run_developability.sh                    # full corpus, icelake CPU partition
bash script_utils/run_developability.sh --limit 100          # local test run
```

No formal test suite. Experiment tracking via Weights & Biases.

## Architecture

### Core Components

**`proteinfoundation/proteina.py`** — Central `LightningModule` (`Proteina` class). Orchestrates flow matching, neural networks, and autoencoders. Handles both training and inference.

Key flags:
- `self._ca_only_mode`: True when `local_latents` absent from `product_flowmatcher`. Skips AE entirely.
- `self.use_precomputed_latents`: Loads latents from disk (precomputed by `precompute_latents.py`) instead of running AE encoder each step.
- `latent_dim=None` in CA-only mode — the NN handles this gracefully (no `local_latents_linear` head).

**`proteinfoundation/flow_matching/product_space_flow_matcher.py`** — Flow matching logic:
- `ProductSpaceFlowMatcher`: Combines bb_ca and local_latents into a joint distribution.
- `self.data_modes` is derived from `product_flowmatcher` config keys — only iterates over present modalities.
- `sample_t` accesses `cfg_exp.loss.t_distribution.shared_groups` — must be under `t_distribution`, not `loss`.

**`proteinfoundation/partial_autoencoder/`** — VAE that compresses full-atom structures to per-residue latent vectors. AE encoder is only ~15-20% of total GPU compute — removing it gives modest speedup.

**`proteinfoundation/nn/`** — Neural network architectures:
- `LocalLatentsTransformer`: Used for both full latent model and CA-only (same class, different config).
- `ca_only_score_nn_160M.yaml`: CA-only config — `output_parameterization: {bb_ca: v}`, no `local_latents` entry.
- `DownsampleBlock` / `UpsampleBlock` in `modules/downsampling.py`: BlurPool1D stride must be 2 in DownsampleBlock (to halve sequence), 1 in UpsampleBlock (smoothing only, interpolate already restores length).

**`proteinfoundation/datasets/pdb_data.py`** — Data pipeline:
- `in_memory=True`: loads all data to RAM. Requires `torch.multiprocessing.set_sharing_strategy('file_system')` (set in `train.py`) to avoid "Too many open files".
- In-memory loading uses `ThreadPoolExecutor` for parallel loading — serial loading of 350K files took 40+ min.
- Precomputed latents path: symlink `/home/ks2218/la-proteina/data/pdb_train/processed_latents` → `/rds/user/ks2218/hpc-work/processed_latents`.

**`prepare_data_800.py`** — Downloads and processes PDB structures up to 800 residues (vs the default 200). Uses `configs/dataset/pdb/pdb_train_ucond_800.yaml`. The data pipeline in `pdb_data.py` (lines 569-591) scans `processed/` for existing `.pt` files and **only downloads CIF files for PDBs not already processed** — so with 536K files already in `processed/`, the delta for maxl800 is modest (~25-65K new proteins). Run via `sbatch script_utils/prepare_data_800.sh`. After completion, run `precompute_latents.py` with the new CSV.

**`precompute_latents.py`** — Precomputes AE latents to disk. Key details:
- Use atomic writes: `torch.save(data, tmp_path); os.rename(tmp_path, out_path)` — prevents corrupted files if job killed mid-write (SLURM kills jobs without warning at time limit).
- `weights_only=False` required for PyG `Data` objects — `weights_only=True` rejects them all and causes mass deletion if used in a validation loop.
- `FILTER_CSV` filters to only the proteins used in training (avoids processing unused proteins).
- `keys_to_keep = ['mean', 'log_scale', 'coords_nm', 'coord_mask', 'id', 'residue_type', 'bb_ca']`
- AE latents are NOT rotation-invariant (`GlobalRotationTransform` must still be applied during training, not removable from baked_in_names).
- Cambridge HPC Ampere partition: requesting 1 GPU gives full 1/4-node allocation (32 CPUs, ~250G RAM) regardless of `--cpus-per-task`. Set `NUM_WORKERS=32` to use all available CPUs.

**`proteinfoundation/analysis/compute_developability.py`** — Biophysical property panel:
- Computes 20-column CSV of per-protein properties from `.pt` files in `processed/`.
- Reads PyG Data objects (`coords`, `coord_mask`, `residue_type`, `residues`, `id`), applies `PDB_TO_OPENFOLD_INDEX_TENSOR` reindex (same as `PDBDataset.__getitem__`), then extracts sequence and structure for property computation.
- **Sequence-based**: SWI (mean per-residue solubility propensity), TANGO (aggregation via external binary), CANYA (nucleation via TF model, soft dep), net charge/pI (Biopython), IUPred3 (disorder, soft dep), Shannon entropy.
- **Structure-based**: radius of gyration, hydrophobic patches (union-find on exposed hydrophobic residues), SAP (Chennamsetty 2009), SCM (spatial charge map). All share a single FreeSASA call per protein.
- **Always NaN**: `camsol_intrinsic` (placeholder, no public binary), `canya_max_nucleation` (if not installed).
- Resume-safe: appends to CSV, skips already-processed `pdb_id`s on restart.
- Uses `spawn` multiprocessing context (safe with TensorFlow/PyTorch). TANGO creates temp dirs per call.
- `--skip-tango` flag skips TANGO computation entirely (columns become NaN).
- **Parallel submission**: uses batched/rolling submission (`SUBMIT_CHUNK = workers * 4`) to avoid flooding the executor. Eagerly submitting all 60K+ futures at once deadlocks on Lustre — the metadata storm from concurrent `torch.load` + temp file creation across all workers stalls the filesystem indefinitely. This is a general pattern to watch for on this cluster: never submit more than a few hundred futures at once with `ProcessPoolExecutor` + `spawn` when workers do Lustre I/O.
- `.pt` files are in graphein/PDB atom order on disk; the reindex to OpenFold order happens at load time (both here and in `PDBDataset`).
- Data: 536K `.pt` files sharded as `processed/<2-char-prefix>/<pdb_chain>.pt`. Estimated ~4h for 63K proteins (300-800 residues) with 31 workers including TANGO.
- `data.id` matches `Path(filename).stem` — resume logic relies on this.

### Configuration System

Hydra-based in `configs/`. Key top-level configs:
- `training_local_latents.yaml` — Full latent diffusion model
- `training_ca_only.yaml` — CA-only baseline (no latents, no AE)
- `training_ae.yaml` — Autoencoder training
- `inference_ucond_notri*.yaml` — Unconditional sampling
- `inference_motif_*.yaml` — Motif scaffolding

Sub-configs: `dataset/`, `nn/`, `nn_ae/`, `generation/`. Model sizes: 70M and 160M.

**Passing config name to submit script:**
```bash
sbatch script_utils/submit_train.sh -n training_ca_only
```
`full_training_test.sh` uses a manual `while [[ $# -gt 0 ]]` loop (not `getopts`) to parse `-n`, `-d`, `-c` flags. The `--config-name` arg is passed to Hydra; remaining args become extra Hydra overrides.

### Key Config Pitfalls

- `shared_groups` must be indented under `loss.t_distribution`, not under `loss`. Wrong indentation causes `ConfigAttributeError`.
- `OmegaConf` returns `None` for `null` keys even with a default in `.get("key", default)` — check explicitly for `None`.
- `generation: validation_local_latents` is safe to use with CA-only — it has `local_latents` ODE keys but they are never accessed during training (only used at inference via `configure_inference`).
- `autoencoder_ckpt_path` must be absent or null in CA-only config — the guard in `proteina.py:66` asserts this.
- `p_folding_n_inv_folding_iters` in CA-only config should be `0.0` unless you specifically want folding/inv-folding conditioning.

### Performance Notes

- **DDP sync** (every `accumulate_grad_batches` steps) is the dominant bottleneck, not the AE encoder.
- **EMA every step** adds significant overhead — use `every_n_steps: 5` not 1.
- Recommended settings for both full and CA-only: `accumulate_grad_batches: 8`, `ema.every_n_steps: 5`, `batch_size: 26`.
- CA-only vs full model: similar wall-clock time per step (same transformer backbone). CA-only advantage is faster validation loss convergence per epoch (simpler task, all gradient signal on bb_ca).
- For real speedup: use 70M model instead of 160M (~2x faster per step).

### CA-only baseline recipe — current canonical (use this)

Canonical training recipe for the 160M CA-only diffusion baseline. **This is the OLD recipe (wd=0.05, constant LR=2e-4). Use it for any architectural variant (sparse attention, conv downsampling) so the comparison stays clean.** The newer "v2" recipe (wd=0.1 + cosine LR) was attempted on 2026-04-24 and is documented in the next subsection as a failed experiment — do not use it.

- **Config:** `configs/training_ca_only.yaml` — but with `weight_decay: 0.05` (not 0.1) and the `scheduler:` block removed (constant LR). The current YAML on disk after the v2 attempt may still have the v2 values in it; restore them before training a new variant.
- **Submit script:** `script_utils/submit_train_ca_only_1gpu.sh` (1× A100 ampere; chain via `--dependency=afterany:$prev` for multi-slot training; **always pass `--exclude=gpu-q-43`** — that node has a broken GPU and `afterany` re-routes the chain back to it).
- **Optimizer:** `torch.optim.AdamW`. β1=0.9, β2=0.999, ε=1e-8 (PyTorch defaults). **Weight decay = 0.05** (uniform across all parameters — the codebase's `configure_optimizers` does not split into wd/no-wd parameter groups; see "Why wd ≤ 0.05" below).
- **LR schedule:** constant LR = 2e-4 (sqrt-scaled from the 4-GPU 4.15e-4 baseline). No warmup, no decay.
- **Effective batch:** `batch_size=6 × max_padding_size=512 × accumulate_grad_batches=32 = ~192 proteins/optimizer step`. Equivalent to the 4-GPU baseline.
- **EMA:** decay 0.999, every 5 steps (every-step EMA is too slow). Companion `-EMA.ckpt` written for every checkpoint event by `EmaModelCheckpoint`.
- **Training-step logging** (added in `proteina.py:on_before_optimizer_step` and `training_step` during the v2 attempt — kept because they're useful regardless of recipe): `train/grad_norm` (pre-clip), `train/param_norm` (global L2 over trainable params), `train/lr_pg0`, `validation_loss_by_len/len_<lo>_<hi>` for bins 50-175, 175-300, 300-425, 425-513.
- **Validation cadence:** `val_check_interval=2000` mini-batches → ~63 optimizer steps between val evals at `accumulate_grad_batches=32`. Val set has 4058 proteins. `validation_loss/loss_epoch` is the headline scalar, **but it is a misleading proxy for sample quality on this codebase under uniform-wd AdamW** — see Finding 5 and the v2 post-mortem subsection below. Always confirm a candidate "good" checkpoint with a cheap designability check (N=3-10 samples per length at 2-3 lengths) before believing the val number.
- **Convergence:** old runs reached best val ≈ 4.71-4.77 around opt step 1800-2200, then overfit (val rises to 5+ within 200-700 more steps). Best old raw checkpoint currently on disk: `store/test_ca_only_diffusion/1776805213/checkpoints/best_val_00000026_000000002646.ckpt` (the originally-cited step-2204 best from `jeponiu5` was overwritten by later `best_val_*` saves under `save_top_k=1`). If you need the exact-step-2204 ckpt back, you'd have to retrain.
- **Wall-clock:** ~131 opt-steps/hour on a single A100 with the current logging (the two full-parameter L2 traversals per step in `on_before_optimizer_step` are the bottleneck; the old runs that hit ~300/hr did not have this logging). Best is reachable in ~16 h of chained slots.

#### Canonical baseline run (the reference for variants)

The "old recipe" baseline that all architectural variants must match has its
saved exp-config on disk at
`store/test_ca_only_diffusion/1776805213/checkpoints/exp_config_test_ca_only_diffusion.json`.
Treat that JSON as the source of truth — when starting a new variant, diff
its NN block against your variant's resolved Hydra config and assert that
nothing other than the variant-specific keys differs.

The exact NN architecture (matches `configs/nn/ca_only_score_nn_160M.yaml`):
- `name: local_latents_transformer`, `output_parameterization: {bb_ca: v}` (CA-only — no `local_latents` head; `latent_dim=None` triggers the no-AE path in `proteina.py`)
- `nlayers: 14`, `token_dim: 768`, `nheads: 12`, `parallel_mha_transition: False`, `use_qkln: True`
- `feats_seq: [xt_bb_ca, x_sc_bb_ca, optional_ca_coors_nm_seq_feat, optional_res_type_seq_feat]`
- `feats_cond_seq: [time_emb_bb_ca]`
- `feats_pair_repr: [rel_seq_sep, xt_bb_ca_pair_dists, x_sc_bb_ca_pair_dists, optional_ca_pair_dist]`
- `feats_pair_cond: [time_emb_bb_ca]`
- `dim_cond=256`, `idx_emb_dim=256`, `t_emb_dim=256`, `pair_repr_dim=256`, `seq_sep_dim=127`
- `xt_pair_dist_dim=30, min=0.1, max=3` (nm); same for `x_sc`
- `update_pair_repr: False` (so `update_pair_repr_every_n` is dead code in the baseline)
- `use_tri_mult: False`, `use_downsampling: False`
- `strict_feats: False`

What the baseline run did NOT use (deliberate negatives — keep them off in variants unless the variant *is* one of these):
- No triangular multiplicative updates (would also be sparse-incompatible — `pair_update.py:65` raises if `use_tri_mult=True` with sparse attention).
- No pair-rep update layer (`update_pair_repr=False`).
- No conv downsample/upsample (`use_downsampling=False`).
- No LoRA (`lora.r: null`).
- No motif conditioning, no folding/inv-folding iters (`p_folding_n_inv_folding_iters: 0.0`), no recycling (`n_recycle: 0`).
- Self-conditioning IS on (`self_cond: True`).
- No precomputed latents (CA-only mode is AE-free anyway, but worth noting that the user preference applies: never enable `use_precomputed_latents` for any related run).

Recipe values (the canonical settings, copy verbatim into any variant's training config):
- `opt.lr: 0.0002` (constant — no scheduler block, or set `scheduler.name` to a no-op).
- `opt.weight_decay: 0.05`.
- `opt.accumulate_grad_batches: 32` (with `dataset.datamodule.batch_size: 6` and `max_padding_size: 512` → effective batch ≈ 192).
- `opt.dist_strategy: auto` for 1-GPU runs.
- `opt.val_check_interval: 2000`.
- `ema: { decay: 0.999, every_n_steps: 5, validate_original_weights: False, cpu_offload: False }`.
- `seed: 42`.
- `force_precision_f32: False` (bf16-mixed).
- `dataset.datamodule.dataselector.worst_resolution: 2.0`, `min_length: 50`, `max_length: 512`.
- `hardware.ngpus_per_node_: 1`, `hardware.nnodes_: 1`.

Reference results:
- Best val loss: ≈ 4.71-4.77 around opt step 1800-2200, then overfits.
- Sample quality: 1-2/3 designable at L=50 and L=100 (scRMSD < 2 Å threshold) at the best checkpoints; not all lengths are designable simultaneously. This is the bar variants must clear.
- Wandb runs in the chain: `d1k1587u` (best val 4.765 at step 1827), `jeponiu5` (best 4.712 at step 2204, ckpt overwritten), `0fnyfbi9` (latest, contains step 2646 ckpt currently on disk).

Checklist for training an architectural variant (sparse attention, conv downsampling, etc.):
1. Make a NEW NN config file under `configs/nn/` (e.g. `ca_only_sparse_160M.yaml`). Copy `ca_only_score_nn_160M.yaml` verbatim, change ONLY the keys the variant needs. Do not silently retune unrelated keys.
2. Make a NEW training config under `configs/` (e.g. `training_ca_only_sparse.yaml`) with `defaults: - nn: <your_new_nn_config>` and a fresh `run_name_:` so the new run goes to its own store dir.
3. In the training config, lock the recipe to the canonical values above (wd=0.05, constant LR=2e-4, accumulate_grad_batches=32, ema every_n_steps=5).
4. Submit with `script_utils/submit_train_ca_only_1gpu.sh -n <your_training_config>` and `--exclude=gpu-q-43`.
5. After ~1000 opt steps, run a designability probe (N=3 per length at L ∈ {50, 100, 200}). If 0/9 designable, the variant has the v2-style collapse and chasing val loss further is wasted compute — debug instead.

#### Why wd ≤ 0.05 — the AdaLN-Zero × AdamW interaction (mechanism)

The `LocalLatentsTransformer` uses DiT-style **AdaLN-Zero** conditioning blocks (`*.scale_output.to_adaln_zero_gamma` in `state_dict()`). These output gates are zero-initialised and need to *grow* during training to let conditioning influence each block's residual contribution. `configure_optimizers` in `proteinfoundation/proteina.py` applies a single `weight_decay` value to *all* parameters, including the AdaLN-Zero gates. Weight decay continuously pulls these gates back toward zero, which is in direct tension with the gradient signal trying to grow them. At wd=0.05 the gates win; at wd=0.1 the gates lose, especially in deeper transformer layers where gradient signal is weaker. The result is a model where conditioning is suppressed, validation MSE looks great (smoother, lower-variance velocity prediction), but generated samples collapse (no time-conditioning to push integrated trajectories onto the data manifold). See Finding 5 for the per-layer evidence (upper-layer gates at 26-60% of old-recipe magnitude in v2).

**To safely raise wd above 0.05 in this codebase you must first restructure `configure_optimizers` to split parameters into wd / no-wd groups** (the standard DiT/SiT/SD3 pattern: exclude biases, LayerNorm γ/β, embeddings, and AdaLN-Zero gate weights from wd). Roughly 15 lines. Until that change is in, treat wd=0.05 as a hard upper bound.

### v2 recipe attempt — failed experiment, kept for reference (2026-04-24)

The v2 attempt at improving the baseline used `weight_decay=0.1` + cosine_with_warmup LR schedule (peak 2e-4 at step 200, decay to 2e-5 at step 6000). It reduced best val loss from 4.765 → 4.437 (Δ -0.328), but produced **0/3 designable samples at every tested length** (L=50, 100, 200), with mean scRMSD 9-11 Å vs the old recipe's 2-8 Å. Per-layer weight diff confirmed the cause: the AdaLN-Zero output gates in transformer layers 7-13 had collapsed to 26-60% of their old-recipe magnitudes (see the mechanism subsection above and the full numbers in Finding 5).

- v2 store dir (preserved): `store/ca_only_diffusion_baseline_v2/1776975226/checkpoints/`
- v2 wandb runs: `9jp15of2` → `5rftn43a` → `43xxlbzt`
- **Do not use the v2 checkpoint for sampling, evaluation, or as a starting point for any architectural variant.**

Lesson encoded in this section: any future weight-decay tuning in this codebase requires restructuring `configure_optimizers` first.

#### Operational notes from the v2 run (kept because they apply to any chained training)

- **Resume gotcha:** `fetch_last_ckpt` (in `proteinfoundation/utils/fetch_last_ckpt.py`) picks the file with the most recent mtime. If the most recent file is a `best_val_*.ckpt` (saved on a val-improvement event after `last.ckpt`), the chain resumes from that, not from `last.ckpt`. Cost: up to ~`val_check_interval / accumulate_grad_batches` lost steps. Not a correctness bug but worth knowing — the v2 chain lost 63 steps this way at the slot-2 → slot-3 handoff.
- **Broken node:** `gpu-q-43` has a broken GPU. The v2 chain hit it three slots in a row because Slurm `--dependency=afterany` re-routes the next slot to the same node. Always include `--exclude=gpu-q-43` on `sbatch` for chained training.
- **Wandb run-ID across slots:** `RESUME=1` triggers `WandbLogger(resume="allow")` but the run ID isn't carried across slots, so each chained slot creates a *new* wandb run with the same run name. To trace v2 you have to follow the chain `9jp15of2 → 5rftn43a → 43xxlbzt` in the dashboard.

### Cluster (Cambridge HPC) Notes

- Data: `/rds/user/ks2218/hpc-work/processed` (raw PDB .pt files, 536K files), `/rds/user/ks2218/hpc-work/processed_latents` (precomputed latents, 63K files for 300-800 residue subset; 355K original latents archived, see below)
- Checkpoints: `/rds/user/ks2218/hpc-work/checkpoints_laproteina/`
- **RDS Inode Quota**: limit is 1,048,576 files. As of Apr 2026, ~962K used (~92%). Space quota is 1.1 TB with ~107 GB used — space is not the bottleneck, file count is.
- **Archived precomputed latents**: `processed_latents/` was archived to free ~355K inodes for the maxl800 dataset preparation. Archives are at `/rds/user/ks2218/hpc-work/latent_shards/` (~265 `.tar` files, one per 2-char shard directory). To restore:
  ```bash
  mkdir -p /rds/user/ks2218/hpc-work/processed_latents && \
  ls /rds/user/ks2218/hpc-work/latent_shards/*.tar | xargs -P 32 -I{} tar -xf {} -C /rds/user/ks2218/hpc-work/processed_latents
  ```
  After restoring, re-symlink if needed: `ln -sfn /rds/user/ks2218/hpc-work/processed_latents /home/ks2218/la-proteina/data/pdb_train/processed_latents`
- **Lustre + ProcessPoolExecutor**: Never eagerly submit tens of thousands of futures when workers do Lustre I/O (torch.load, temp files). The concurrent metadata operations deadlock the filesystem. Use rolling/batched submission (submit ~`workers*4` at a time, replenish as they complete).
- **Tarring on Lustre is slow**: 355K small files takes hours on login nodes. Use compute nodes with parallel shard approach: `ls processed_latents/ | xargs -P 32 -I{} tar -cf latent_shards/{}.tar -C processed_latents {}`. Skip `-z` (gzip) — saves CPU time and the bottleneck is metadata, not disk space.
- Ampere partition: 1 GPU = 32 CPUs + ~250G RAM (full 1/4-node), regardless of `--cpus-per-task` SBATCH header.
- SLURM kills jobs at time limit without warning — use atomic writes for any file outputs.
- Stale CSV/FASTA files cause MMseqs2 "empty FASTA" error — delete `df_pdb_..._latents.csv` and `seq_df_pdb_..._latents.fasta` if processed_latents directory was recreated.
- SLURM TaskProlog `mkdir /var/spool/slurm/slurmd/logs` permission error: the prolog runs a `mkdir` that fails. If the job script uses `set -e` (exit on error), this kills the script before it executes. **Do NOT use `set -e` in SLURM scripts on this cluster.** Use `set -uo pipefail` instead.
- TANGO binary: extract from `tango2_3_1.linux64.zip` in repo root, `chmod +x`, place on PATH or set `TANGO_EXE=/absolute/path`. The `run_developability.sh` pre-flight check will abort if missing.
- IUPred3: installed at `~/iupred3/` (extracted from `iupred3.tar.gz` in repo root). Script auto-discovers via `IUPRED3_DIR` env var or `~/iupred3` default.

## Steerability Analysis Pipeline (`laproteina_steerability/`)

Diagnostic pipeline to characterize the latent space and probe how well protein properties are encoded, before committing to steering field training. Lives in `laproteina_steerability/` with its own config, data layer, and CLI entry points.

### Running

All commands run from `laproteina_steerability/`:
```bash
# Part 1 — latent geometry (no properties needed, runs on cached encoder outputs alone)
python -m src.part1_latent_geometry.run --config config/default.yaml

# Part 2 — property probes (needs a property file)
python -m src.part2_property_probes.run --config config/default.yaml

# Smoke test on synthetic data
python -m src.part1_latent_geometry.run --config config/default.yaml --synthetic
python -m src.part2_property_probes.run --config config/default.yaml --synthetic
```

### Data Sources

**Cached latents** (`/rds/user/ks2218/hpc-work/processed_latents`): 355K `.pt` files, sharded as `<2-char-prefix>/<pdb_chain>.pt`. Each file contains:
- `mean` `[L, 8]` float32 — VAE encoder mean (the latent representation)
- `log_scale` `[L, 8]` float32 — VAE encoder log-variance
- `coords_nm` `[L, 37, 3]` float32 — full-atom coords in nm (CA is atom index 1 in OpenFold order)
- `coord_mask` `[L, 37]` — atom mask
- `id` string — protein identifier (e.g. `101m_A`)
- `residue_type` `[L]` — residue type indices

The field name mapping in `config/default.yaml` translates these to the loader's semantic names: `latents→mean`, `ca_coords→coords_nm` with `ca_atom_index: 1`, `protein_id→id`.

**Property file** (Part 2 only): Not yet created. Needs to be a parquet or CSV at the path set in `part2.property_file` with columns: `protein_id` (matching the `id` field in `.pt` files), `residue_index` (nullable for protein-level props), and one column per property. The developability CSV from `compute_developability.py` has most properties but uses `pdb_id` as the column name and is protein-level only — needs column rename and granularity config update before use.

### Config (`config/default.yaml`)

Key settings to adjust before running:
- `data.length_range`: `[50, 300]` by default. Set to `[300, 800]` for long proteins, or `null` for all lengths.
- `data.subsample`: `null` loads all proteins. Set to e.g. `1000` for a quick test. The loader pre-subsamples files before loading when this is set (avoids scanning 355K files).
- `part2.property_file`: Path to property table. Must be set before running Part 2.
- `part2.property_granularity`: Maps each property to `"protein"` or `"residue"`. If the property file only has protein-level values, set all to `"protein"`.
- `part2.decisions.*`: All steering decision thresholds (R² cutoffs, kNN gap) are config-driven for sweeping.

### What Each Part Produces

**Part 1** (latent geometry — outputs to `outputs/`):
- `figures/latent_marginals` — per-dim histograms + KDE
- `figures/latent_correlations` — Pearson + Spearman heatmaps
- `figures/latent_mutual_information` — MI heatmap (nonlinear dependence)
- `figures/pca_analysis` — scree plot + cumulative variance + participation ratio
- `figures/dim_utilization` — within-protein vs between-protein variance decomposition + ratio
- `figures/length_sensitivity` — latent norm and per-dim means vs protein length with Pearson r
- `tables/` — CSV of every numerical result (marginal stats, correlation matrices, PCA eigenvalues, utilization, length stats)
- `part1_summary.md` — key numbers: participation ratio, effective rank, max correlation, collapsed dim count

**Part 2** (property probes — outputs to `outputs/`):
- Probes: Ridge, MLP, kNN at noise levels t∈{0.3, 0.5, 0.8, 1.0} (code convention: t=0 noise, t=1 clean) × input variants (latent_only, latent_plus_backbone)
- `tables/probe_results.csv` — full results, one row per (property, t, variant, probe_type)
- `tables/steering_decisions.csv` — per-property go/no-go: steerable / nonlinear_encoded / goodhart_control / drop
- `figures/property_correlation_clustered` — clustered property-property heatmap
- `figures/umap_property_grid` — 3×3 UMAP grid colored by property (requires `umap-learn`)
- `part2_summary.md` — decision counts, top probe R² values, Goodhart pairs

### Architecture Notes

- The loader (`src/data/loader.py`) handles both `.pt` (PyG Data objects or dicts) and `.npz` files, with configurable field name mapping and automatic CA extraction from full-atom coordinate arrays via `ca_atom_index`.
- SE(3) normalization for `latent_plus_backbone` input variant: PCA-axis rotation with deterministic sign fix (largest |coord| per axis is positive). Known to be discontinuous for near-degenerate CA clouds — acceptable for a first pass.
- t-convention: all code uses t=0 = pure noise, t=1 = clean data (`z_t = (1-t)*noise + t*z_clean`), matching the La-Proteina codebase. **The steering predictor's t-convention has NOT been verified** — flagged in code comments and all output summaries.
- Grouped CV: folds are grouped by protein_id so the same protein never appears in both train and test.
- Missing optional deps: `umap-learn` (UMAP step skipped gracefully), `pyarrow` (falls back to CSV).

### Steerability Results (Apr 2026, 56K proteins, 300-800 residues)

**Part 1 — Latent geometry**: All 8 latent dims are well-utilized (participation ratio 7.69/8, no posterior collapse). Dims are well-disentangled (max off-diagonal Pearson 0.10, max MI 0.28 nats). Dims 3, 5, 6, 7 are multimodal (discrete structural clusters). Weak length sensitivity (r = -0.04). Within-protein variance dominates between-protein variance (~1.04 ratio), so protein-level steering works with averaged latents.

**Part 2 — Property probes**: Not yet run on real data. Requires the developability CSV as input (rename `pdb_id` → `protein_id`, set `part2.property_granularity` to all `"protein"`).

## Gradient-Based Steering (`steering/`)

Modular guidance system that steers the generative model toward desired biophysical properties during sampling, without retraining. Works by nudging the latent velocity field at each ODE step using gradients from a trained property predictor.

### How It Works (Intuition)

La-Proteina generates proteins by simulating a flow ODE from noise (t=0) to data (t=1). At each step, the flow model predicts a velocity `v` and updates the latent state: `z_{t+dt} = z_t + dt * v`. The velocity points toward "some protein" — but not necessarily one with the properties you want.

Steering adds a correction to that velocity. At each step:

1. **Estimate the final protein**: From the current state `z_t` and velocity `v`, compute where the trajectory would end up if it went straight: `z_1_est = z_t + (1-t) * v`. This is the flow's "clean estimate" of the final latent.

2. **Ask the predictor**: Feed `z_1_est` to a trained property predictor (a small transformer, ~350K params) that maps latent vectors to 13 biophysical properties (net charge, SAP, Rg, etc.). This predictor was trained on real protein latents paired with computed properties.

3. **Compute a gradient**: Backpropagate through the predictor to get `d(property) / d(z_t)` — the direction in latent space that increases (or decreases) the target property. Crucially, this gradient only flows through the predictor, never through the flow model's weights (v is detached).

4. **Add to velocity**: Scale the gradient by a schedule weight `w(t)` and add it to `v`. The modified velocity now points toward a protein with slightly more of the desired property.

The schedule `w(t)` is zero early in sampling (when the estimate is noisy and gradients are unreliable) and ramps up as the trajectory approaches clean data. This prevents the guidance from fighting the flow model during the critical early structure-forming phase.

Because the predictor operates in z-score space and the gradient is unit-normalised per protein, the guidance magnitude is controlled entirely by `w_max` — making it easy to sweep without worrying about property-specific scales.

### Architecture

```
steering/
├── __init__.py
├── config/
│   ├── default.yaml              # master switch: enabled: false
│   └── examples/
│       ├── net_charge_up.yaml
│       ├── sap_down.yaml
│       └── multi_objective.yaml
├── guide.py                      # SteeringGuide — core class
├── predictor.py                  # SteeringPredictor — loads checkpoint + z-score stats
├── schedules.py                  # w(t) functions: linear_ramp, cosine_ramp, constant
├── registry.py                   # property name <-> head index mapping (13 properties)
└── test_steering.py              # standalone tests (runs without flow model)
```

**Integration point**: A single `steering_guide=None` parameter on `ProductSpaceFlowMatcher.full_simulation()`. When `None`, zero new code executes inside the loop. When set, a 10-line if-block between `nn_out` computation and `simulation_step` calls `steering_guide.guide()` and adds the returned gradient to `nn_out["local_latents"]["v"]` (or `"v_guided"` if CFG is active). The guidance tensor is always `.detach()`'d before addition — safe inside the `torch.no_grad()` simulation context.

### Key Design Decisions

- **Reconstruction guidance only**: Feeds the flow's clean estimate `z_1_est` to the predictor at `t=1`. Simpler and more stable than noise-aware guidance (which would need the predictor to handle arbitrary noise levels).
- **Gradient through z_t, not through the flow model**: `v_theta` is detached in the guidance computation. Only the path `z_t -> x1_est -> predictor` carries gradients. This means guidance never interferes with the flow model's internal representations.
- **Z-score space for objectives**: The predictor outputs z-scored predictions. Gradients are computed in z-score space so that multi-objective weighting is scale-invariant across properties (net charge in [-20, 20] vs SAP in [0, 500]).
- **Unit normalisation**: After clipping, the gradient is normalised to unit norm per protein. `w_max` then directly controls the step size in latent space, independent of how sharp or flat the predictor's landscape is.
- **Latent channel only**: Guidance only modifies `local_latents`, never `bb_ca`. Backbone structure is determined by the flow model; side-chain/sequence properties are steered through the latent channel.

### Property-to-Head-Index Mapping

| Index | Property | Index | Property |
|-------|----------|-------|----------|
| 0 | swi | 7 | hydrophobic_patch_total_area |
| 1 | tango | 8 | hydrophobic_patch_n_large |
| 2 | net_charge | 9 | sap |
| 3 | pI | 10 | scm_positive |
| 4 | iupred3 | 11 | scm_negative |
| 5 | iupred3_fraction_disordered | 12 | rg |
| 6 | shannon_entropy | | |

### Running

```bash
# Standalone test (creates dummy checkpoint if none provided)
python -m steering.test_steering
python -m steering.test_steering --checkpoint path/to/fold_0_best.pt

# Integration: pass steering_guide to full_simulation()
from steering import SteeringGuide
guide = SteeringGuide({"enabled": True, "checkpoint": "...", "objectives": [...]})
x, info = flow_matcher.full_simulation(..., steering_guide=guide)
diagnostics = info["steering_diagnostics"]  # list of per-step dicts
```

### Diagnostics

When `log_diagnostics: true` (default), each step appends a dict to `steering_guide.diagnostics`:
- `t`: local_latents schedule time
- `w`: guidance weight after schedule
- `grad_norm_raw`: gradient norm before normalisation
- `grad_norm_final`: gradient norm after normalisation and scaling
- `predicted_properties`: dict of all 13 de-normalised property predictions on the clean estimate

Returned in `additional_info["steering_diagnostics"]` from `full_simulation()`. Diagnostics are reset at the start of each `full_simulation()` call.

### Predictor Checkpoint Format

The steering predictor loads from the same checkpoint format produced by `laproteina_steerability/src/multitask_predictor/train.py`. Contains:
- `model_state_dict`: PropertyTransformer weights (128-dim, 3 layers, 4 heads)
- `stats_mean`, `stats_std`: numpy arrays `[13]` for z-score de/normalisation

No separate normalisation stats file — everything is in the checkpoint.

### Known Limitations

- Diagnostics only log batch element 0 (sufficient for debugging, not for per-sample analysis in large batches).
- Predictor architecture is hardcoded (128-dim, 3 layers) — must match the training config. If the architecture changes, `steering/predictor.py` needs updating.
- `direction: target` uses L2 loss in z-score space, which may not behave well for properties with skewed distributions.

### Known Issues / Fixes Applied

- **`NameError: free variable 'os'`** in `pdb_data.py`: caused by `import os` inside a function body. Removed inner import.
- **`RuntimeError: Too many open files`**: `in_memory=True` + multiprocessing workers. Fixed by `torch.multiprocessing.set_sharing_strategy('file_system')` in `train.py`.
- **`EOFError: Ran out of Input`**: SLURM job killed mid-write → corrupted `.pt` file. Fix: atomic write pattern in `precompute_latents.py`.
- **CA-only not running CA-only**: shell script had hardcoded `nn=local_latents_score_nn_160M` override. Removed; now uses `-n <config_name>` flag.
- **`getopts` illegal option**: replaced `getopts` in `full_training_test.sh` with manual `while [[ $# -gt 0 ]]` loop.
- **`BlurPool1D` stride bug**: `DownsampleBlock` needs `stride=2` (to halve length); `UpsampleBlock` blur needs `stride=1` (smoothing only).
- **`shared_groups` indentation**: must be 4 spaces (under `t_distribution`), not 2 (under `loss`).
