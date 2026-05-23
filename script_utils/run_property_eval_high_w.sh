#!/usr/bin/env bash
# Property panel pass for the high-w scout. CPU-bound (TANGO + IUPred +
# FreeSASA + Shannon + Rg + hydrophobic-patches), so safe to run in parallel
# with a GPU generation job on the same box.
#
# Resume-safe: skips cells that already have properties_guided.csv.
set -uo pipefail
cd /home/ks2218/la-proteina

export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

OUT_BASE=results/noise_aware_high_w_scout
CFGS=(
    camsol_max_w32 camsol_max_w64 camsol_max_w128
    tango_min_w32  tango_min_w64  tango_min_w128
)

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
START=$(date +%s)
echo "[$(date)] Property eval started."

for cfg in "${CFGS[@]}"; do
    GUIDED_DIR="$OUT_BASE/$cfg/guided"
    OUT_CSV="$OUT_BASE/$cfg/properties_guided.csv"
    if [[ -f "$OUT_CSV" ]]; then
        echo "[$(date)] [skip] $cfg — properties_guided.csv already exists"
        continue
    fi
    # Wait until the cell's guided/ has its expected 48 PDBs (16 seeds × 3 lengths).
    # Property eval handles partial sets fine, but we want a complete cell.
    if [[ ! -d "$GUIDED_DIR" ]]; then
        echo "[$(date)] [skip] $cfg — guided/ not yet created"
        continue
    fi
    n_pdb=$(ls "$GUIDED_DIR" 2>/dev/null | grep -c '\.pdb$')
    if [[ "$n_pdb" -lt 48 ]]; then
        echo "[$(date)] [skip] $cfg — only $n_pdb/48 PDBs (generation still in flight)"
        continue
    fi

    echo "[$(date)] $cfg : property panel on $n_pdb PDBs"
    "$PY" -m steering.evaluate_samples_dir \
        --samples_dir "$GUIDED_DIR" \
        --output_csv "$OUT_CSV"
done

END=$(date +%s)
echo "[$(date)] Property eval done in $(( (END-START)/60 )) min."
