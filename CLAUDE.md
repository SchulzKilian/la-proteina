# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**La Proteina** is a deep learning research codebase for atomistic protein generation via partially latent flow matching. It trains generative models to design protein structures (backbone + side chains) and sequences.

## Environment Setup

```bash
mamba env create -f environment.yaml
mamba activate laproteina_env
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install graphein==1.7.7 --no-deps
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

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
- `.pt` files are in graphein/PDB atom order on disk; the reindex to OpenFold order happens at load time (both here and in `PDBDataset`).
- Data: 536K `.pt` files sharded as `processed/<2-char-prefix>/<pdb_chain>.pt`. Estimated ~8-10h with 32 workers (TANGO subprocess is the bottleneck).
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

### Cluster (Cambridge HPC) Notes

- Data: `/rds/user/ks2218/hpc-work/processed` (raw PDB .pt files, 536K files), `/rds/user/ks2218/hpc-work/processed_latents` (precomputed latents, 355K files — currently archived, see below)
- Checkpoints: `/rds/user/ks2218/hpc-work/checkpoints_laproteina/`
- **RDS Inode Quota**: limit is 1,048,576 files. As of Apr 2026, ~962K used (~92%). Space quota is 1.1 TB with ~107 GB used — space is not the bottleneck, file count is.
- **Archived precomputed latents**: `processed_latents/` was archived to free ~355K inodes for the maxl800 dataset preparation. Archives are at `/rds/user/ks2218/hpc-work/latent_shards/` (~265 `.tar` files, one per 2-char shard directory). To restore:
  ```bash
  mkdir -p /rds/user/ks2218/hpc-work/processed_latents && \
  ls /rds/user/ks2218/hpc-work/latent_shards/*.tar | xargs -P 32 -I{} tar -xf {} -C /rds/user/ks2218/hpc-work/processed_latents
  ```
  After restoring, re-symlink if needed: `ln -sfn /rds/user/ks2218/hpc-work/processed_latents /home/ks2218/la-proteina/data/pdb_train/processed_latents`
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

### Known Issues / Fixes Applied

- **`NameError: free variable 'os'`** in `pdb_data.py`: caused by `import os` inside a function body. Removed inner import.
- **`RuntimeError: Too many open files`**: `in_memory=True` + multiprocessing workers. Fixed by `torch.multiprocessing.set_sharing_strategy('file_system')` in `train.py`.
- **`EOFError: Ran out of Input`**: SLURM job killed mid-write → corrupted `.pt` file. Fix: atomic write pattern in `precompute_latents.py`.
- **CA-only not running CA-only**: shell script had hardcoded `nn=local_latents_score_nn_160M` override. Removed; now uses `-n <config_name>` flag.
- **`getopts` illegal option**: replaced `getopts` in `full_training_test.sh` with manual `while [[ $# -gt 0 ]]` loop.
- **`BlurPool1D` stride bug**: `DownsampleBlock` needs `stride=2` (to halve length); `UpsampleBlock` blur needs `stride=1` (smoothing only).
- **`shared_groups` indentation**: must be 4 spaces (under `t_distribution`), not 2 (under `loss`).
