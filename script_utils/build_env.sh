#!/bin/bash
# Build laproteina_env from scratch using micromamba into a /home prefix
# (avoids RDS/Lustre dependence at Python startup, which has caused
# fstat hangs when an OST is evicted/disconn). All caches go to /tmp so
# they don't eat the /home quota.
#
# Idempotent rebuild: delete $ENV_PATH first if you want a clean reinstall.
#
# Override via env vars (defaults shown):
#   ENV_PATH=/home/$USER/conda_envs/laproteina_env
#   MM=$HOME/bin/micromamba
#
# To install micromamba (single static binary, ~7 MB):
#   mkdir -p $HOME/bin && curl -fsSL \
#     https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/mm.tar.bz2 \
#     && tar -xjf /tmp/mm.tar.bz2 -C /tmp bin/micromamba \
#     && mv /tmp/bin/micromamba $HOME/bin/ && chmod +x $HOME/bin/micromamba
#
# Run in background so it survives ssh disconnects:
#   nohup script_utils/build_env.sh > /home/$USER/conda_envs/build_env.log 2>&1 &
set -uo pipefail

ENV_PATH=${ENV_PATH:-/home/$USER/conda_envs/laproteina_env}
MM=${MM:-$HOME/bin/micromamba}

export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/tmp/micromamba_root}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/tmp/conda_pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/tmp/pip_cache}
mkdir -p "$MAMBA_ROOT_PREFIX" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

ts() { date '+%H:%M:%S'; }

# Auto-install micromamba if missing.
if [ ! -x "$MM" ]; then
    echo "[$(ts)] micromamba not found at $MM — fetching..."
    mkdir -p "$(dirname "$MM")"
    TMP_TAR=/tmp/micromamba_$$.tar.bz2
    curl -fsSL --max-time 60 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$TMP_TAR" \
        || { echo "[$(ts)] FAIL: micromamba download"; exit 1; }
    tar -xjf "$TMP_TAR" -C /tmp bin/micromamba \
        || { echo "[$(ts)] FAIL: micromamba extract"; exit 1; }
    mv /tmp/bin/micromamba "$MM" && chmod +x "$MM"
    rm -f "$TMP_TAR"
    echo "[$(ts)] micromamba installed: $($MM --version)"
fi

# If a previous env exists, mv-aside (instant on same FS) and rm in background.
# Avoids blocking the build behind a slow `rm -rf` of tens of thousands of small files.
if [ -e "$ENV_PATH" ]; then
    SIDE="${ENV_PATH}_old_$(date +%s)"
    echo "[$(ts)] existing env at $ENV_PATH — moving to $SIDE and deleting in background"
    mv "$ENV_PATH" "$SIDE"
    nohup rm -rf "$SIDE" > /dev/null 2>&1 &
    disown
fi

echo "[$(ts)] === phase 1: micromamba create with conda packages ==="
$MM create -p "$ENV_PATH" -y \
    -c nvidia -c pytorch -c conda-forge -c pyg -c bioconda \
    python=3.10 pip \
    'transformers=4.48.3' \
    'mmseqs2=17.b804f' \
    || { echo "[$(ts)] FAIL: micromamba create"; exit 1; }
# Note: forcing python=3.10 even though environment.yaml says 3.11.
# Reason: pyg cp311 wheels for torch 2.7.0+cu118 require GLIBC 2.32, but
# Cambridge HPC nodes are on GLIBC 2.28. The cp310 wheels were built
# against older glibc and load fine.

# All pip versions below are pinned to match the working /rds env snapshot.
# torch 2.5.1 + pt25 pyg wheels are required for GLIBC compat (cluster has 2.28,
# pt27 wheels need 2.32). Other pins captured 2026-04-19 to keep the build
# reproducible without relying on environment.yaml drift.
echo "[$(ts)] === phase 2: torch + torchvision + torchaudio ==="
"$ENV_PATH/bin/pip" install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118 \
    || { echo "[$(ts)] FAIL: torch install"; exit 1; }

echo "[$(ts)] === phase 3: main pip deps (pinned to /rds versions) ==="
"$ENV_PATH/bin/pip" install \
    'python-dotenv==1.2.1' \
    'einops==0.8.2' \
    'dm-tree==0.1.9' \
    'lightning==2.6.1' \
    'pytorch-lightning==2.6.1' \
    'loguru==0.7.3' \
    'hydra-core==1.3.2' \
    'numpy==1.26.4' \
    'pandas==2.3.3' \
    'biopandas==0.5.1' \
    'wandb==0.24.1' \
    'wget==3.2' \
    'tqdm==4.67.1' \
    'ml-collections==1.1.0' \
    'jaxtyping==0.3.6' \
    'loralib==0.1.2' \
    'biopython==1.86' \
    'biotite==0.41.0' \
    'cpdb-protein==0.2.0' \
    'joblib==1.5.3' \
    'rich==14.3.1' \
    'deepdiff==8.6.1' \
    'multipledispatch==1.0.0' \
    'plotly==6.5.2' \
    'pydantic==2.12.5' \
    'rich-click==1.9.6' \
    'scikit-learn==1.7.2' \
    'seaborn==0.13.2' \
    'xarray==2025.6.1' \
    'torchmetrics==1.8.2' \
    || { echo "[$(ts)] FAIL: main pip deps"; exit 1; }

echo "[$(ts)] === phase 4: graphein no-deps and pyg extras ==="
"$ENV_PATH/bin/pip" install --no-deps 'graphein==1.7.8' \
    || { echo "[$(ts)] FAIL: graphein"; exit 1; }

"$ENV_PATH/bin/pip" install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu118.html \
    || { echo "[$(ts)] FAIL: pyg extras"; exit 1; }

echo "[$(ts)] === phase 5: smoke test ==="
"$ENV_PATH/bin/python" -c "
import torch, lightning, pandas, numpy, hydra
import torch_geometric, torch_scatter, torch_sparse, torch_cluster
import torchvision, torchaudio
import wandb, transformers, graphein
print('python', __import__('sys').version)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
print('lightning', lightning.__version__)
print('SMOKE OK')
" || { echo "[$(ts)] FAIL: smoke test"; exit 1; }

echo "[$(ts)] === DONE ==="
du -sh "$ENV_PATH"
