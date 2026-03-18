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

Checkpoints go in `./checkpoints_laproteina/`.

## Common Commands

**Training:**
```bash
# Autoencoder (VAE)
python proteinfoundation/partial_autoencoder/train.py

# Diffusion/flow matching model
python proteinfoundation/train.py
```

**Sampling:**
```bash
# Unconditional generation (≤500 residues)
python proteinfoundation/generate.py --config_name inference_ucond_notri

# Unconditional generation (300–800 residues)
python proteinfoundation/generate.py --config_name inference_ucond_notri_long

# Motif scaffolding variants
python proteinfoundation/generate.py --config_name inference_motif_idx_aa
```

**Evaluation:**
```bash
bash script_utils/download_pmpnn_weights.sh  # one-time setup
python proteinfoundation/evaluate.py --config_name <config_name>
bash script_utils/gen_n_eval.sh              # full generate + eval pipeline
```

No formal test suite exists. Experiment tracking is via Weights & Biases.

## Architecture

### Core Components

**`proteinfoundation/proteina.py`** — Central `LightningModule` (`Proteina` class). Orchestrates flow matching, neural networks, and autoencoders. Handles both training and inference; supports self-conditioning and guidance.

**`proteinfoundation/flow_matching/`** — Flow matching logic:
- `ProductSpaceFlowMatcher`: Combines backbone CA atoms and per-residue latent variables into a joint distribution.
- Supports both unconditional and motif-conditioned generation.

**`proteinfoundation/partial_autoencoder/`** — Two-stage encoding:
- Encoder compresses full atomistic structures to fixed-size per-residue latent vectors (bypasses variable side-chain atom counts).
- Decoder reconstructs full-atom coordinates from latents.
- Has its own training entrypoint (`train.py`) and config (`configs/training_ae.yaml`).

**`proteinfoundation/nn/`** — Neural network architectures:
- `LocalLatentsTransformer`: Indexed motif scaffolding (explicit conditioning on known residue positions).
- `LocalLatentsTransformerMotifUidx`: Unindexed motif scaffolding (implicit/soft conditioning).
- Feature factory module (~78K lines) for data featurization.

**`proteinfoundation/datasets/`** — Data pipeline:
- `BaseLightningDataModule` wraps PDB data loading.
- `GenDataset`: Training dataset with sequence clustering (via MMSeqs2).
- Transforms handle preprocessing and augmentation.

**`proteinfoundation/metrics/`** — Evaluation metrics:
- Designability via ProteinMPNN (inverse folding + ESMFold re-folding).
- Structural metrics: secondary structure content, CA–CA distance distributions.

**`proteinfoundation/utils/`** — Shared utilities: PDB I/O, coordinate transforms (Å ↔ nm), alignment, clustering, motif extraction.

### Configuration System

Hydra-based configuration in `configs/`. Key top-level configs:
- `training_local_latents.yaml` — Main diffusion model training
- `training_ae.yaml` — Autoencoder training
- `inference_ucond_notri*.yaml` — Unconditional sampling
- `inference_motif_*.yaml` — Motif scaffolding (idx/uidx × aa/tip variants)

Sub-configs are organized by `dataset/`, `nn/`, `nn_ae/`, `generation/`. Model sizes: 70M and 160M parameter variants.

### Model Variants

Seven latent diffusion checkpoints (LD1–LD7) for different tasks, three autoencoder checkpoints for different sequence length ranges.

### Key Conventions

- Coordinates are in nanometers internally; PDB files use Ångströms — conversions happen in utils.
- Graphein's CATH URL is monkey-patched at import time (compatibility fix).
- `ProteinMPNN/` and `openfold/` are vendored external tools, not installed packages.
- Distributed training uses PyTorch Lightning + DeepSpeed or DDP (configured via Hydra).
