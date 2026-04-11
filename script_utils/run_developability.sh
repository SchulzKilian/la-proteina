#!/usr/bin/env bash
#SBATCH --job-name=developability
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/developability_%j.log
#SBATCH --mail-type=END,FAIL
#
# run_developability.sh
#
# Wrapper for compute_developability.py.
# Activates the laproteina_env conda environment, checks required tools,
# and runs the property panel computation.
#
# Usage:
#   bash script_utils/run_developability.sh [--output PATH] [--limit N] [--workers N]
#   sbatch script_utils/run_developability.sh

set -euo pipefail

# ── resolve repo root (script lives in script_utils/) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── defaults ─────────────────────────────────────────────────────────────────
OUTPUT_CSV="${DATA_PATH:-/rds/user/ks2218/hpc-work}/developability_panel.csv"
LIMIT=""
WORKERS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
WORKERS="$(( WORKERS > 1 ? WORKERS - 1 : 1 ))"
DATA_DIR="${DATA_PATH:-/rds/user/ks2218/hpc-work}"
FILTER_CSV=""   # set to path of df_pdb_*.csv to restrict to training corpus
TANGO_EXE="${TANGO_EXE:-tango_x86_64_release}"   # override with: export TANGO_EXE=/path/to/binary

# ── parse CLI args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)   OUTPUT_CSV="$2";  shift 2 ;;
        --limit)    LIMIT="$2";       shift 2 ;;
        --workers)  WORKERS="$2";     shift 2 ;;
        --data-dir) DATA_DIR="$2";    shift 2 ;;
        --filter-csv) FILTER_CSV="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── log file ──────────────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/developability_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  Developability panel  —  $(date)"
echo "  Output:   $OUTPUT_CSV"
echo "  Data dir: $DATA_DIR"
echo "  Workers:  $WORKERS"
echo "============================================================"

# ── activate conda environment ────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate laproteina_env

# ── check required tools ──────────────────────────────────────────────────────
MISSING=()

python -c "import torch" 2>/dev/null            || MISSING+=("torch (conda env issue?)")
python -c "import biopython" 2>/dev/null \
  || python -c "from Bio.SeqUtils.ProtParam import ProteinAnalysis" 2>/dev/null \
  || MISSING+=("biopython  →  pip install biopython")
python -c "import freesasa" 2>/dev/null          || MISSING+=("freesasa  →  pip install freesasa")
python -c "import tqdm" 2>/dev/null              || MISSING+=("tqdm  →  pip install tqdm")
python -c "import pandas" 2>/dev/null            || MISSING+=("pandas  →  pip install pandas")

# TANGO binary
if ! command -v "$TANGO_EXE" &>/dev/null; then
    MISSING+=("tango binary  →  scp tango2_3_1.linux64 to cluster, chmod +x, place on PATH or set TANGO_EXE=/path/to/tango")
fi

# CANYA (soft requirement — NaN if absent)
if ! python -c "from canya.canya import CANYA" 2>/dev/null && \
   ! command -v canya &>/dev/null; then
    echo "WARNING: CANYA not found — canya_max_nucleation will be NaN."
    echo "         Install: pip install git+https://github.com/lehner-lab/canya.git"
fi

# IUPred3 (soft requirement — NaN if absent)
if ! python -c "from iupred3 import iupred3_lib" 2>/dev/null && \
   ! python -c "import iupred3" 2>/dev/null; then
    echo "WARNING: IUPred3 not found — iupred3_mean/fraction will be NaN."
    echo "         Download from https://iupred.elte.hu/download_new  then:  pip install iupred3"
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: The following required tools are missing:"
    for item in "${MISSING[@]}"; do
        echo "  ✗ $item"
    done
    echo ""
    echo "Fix the above, then re-run."
    exit 1
fi

echo "All required tools present. Starting computation ..."
echo ""

# ── build python command ──────────────────────────────────────────────────────
PYTHON_CMD=(
    python "${REPO_ROOT}/proteinfoundation/analysis/compute_developability.py"
    "$OUTPUT_CSV"
    --data-dir "$DATA_DIR"
    --workers  "$WORKERS"
)

[[ -n "$LIMIT" ]]      && PYTHON_CMD+=(--limit      "$LIMIT")
[[ -n "$FILTER_CSV" ]] && PYTHON_CMD+=(--filter-csv "$FILTER_CSV")

export TANGO_EXE

# ── run ───────────────────────────────────────────────────────────────────────
echo "Command: ${PYTHON_CMD[*]}"
echo ""
"${PYTHON_CMD[@]}"
EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
exit $EXIT_CODE
