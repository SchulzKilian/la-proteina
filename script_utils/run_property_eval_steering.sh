#!/bin/bash
# Property eval pass for the steering sweep:
#   results/steering_camsol_tango_L500/{config}/guided/*.pt
# Per-config CSV at the same dir + a sequences.fasta so CamSol can be run
# externally (compute_camsol is NaN-only in this codebase; CamSol values are
# normally sourced from CamSolpH_results.txt produced off-tree).

set -uo pipefail
cd /home/ks2218/la-proteina

OUT_BASE=${OUT_BASE:-results/steering_camsol_tango_L500_nsteps400}
CFGS=(
    camsol_max_w1 camsol_max_w2 camsol_max_w4 camsol_max_w8 camsol_max_w16
    tango_min_w1 tango_min_w2 tango_min_w4 tango_min_w8 tango_min_w16
)

# Ensure TANGO is on PATH (compute_developability honors $TANGO_EXE)
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
# IUPred3 lives in ~/iupred3 — picked up via $IUPRED3_DIR or the default

START=$(date +%s)
echo "[$(date)] Property eval started."

i=0
N=${#CFGS[@]}
for cfg in "${CFGS[@]}"; do
    i=$((i + 1))
    GUIDED_DIR="$OUT_BASE/$cfg/guided"
    OUT_CSV="$OUT_BASE/$cfg/properties_guided.csv"
    FASTA="$OUT_BASE/$cfg/sequences_guided.fasta"

    echo ""
    echo "[$(date)] [${i}/${N}] $cfg"
    echo "  guided_dir = $GUIDED_DIR"

    # Property panel (everything except CamSol, which the codebase returns NaN for)
    python -m steering.evaluate_samples_dir \
        --samples_dir "$GUIDED_DIR" \
        --output_csv "$OUT_CSV"

    # Per-config FASTA so CamSol can be run externally on these sequences
    python << EOF
import torch, glob
from pathlib import Path
files = sorted(glob.glob("$GUIDED_DIR/*.pt"))
with open("$FASTA", "w") as fh:
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        pid = d["id"]
        seq = d["sequence"]
        fh.write(f">{pid} length={len(seq)}\n{seq}\n")
print(f"Wrote {len(files)} sequences to $FASTA")
EOF
done

END=$(date +%s)
echo ""
echo "[$(date)] Property eval complete. Total: $(( (END-START)/60 )) min."
