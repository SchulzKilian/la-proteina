#!/bin/bash
# Canonical N=6 designability probe for last-v2.ckpt (rsynced 2026-05-10
# from /rds/.../store/test_ca_only_diffusion/1776805213/checkpoints/last-v2.ckpt
# — the canonical CA-only baseline run, late-training checkpoint).
#
# N=6 × L∈{50, 100, 200} × nsteps=400. ~30-40 min on 1× A100, longer on L4.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
CFG=inference_canonical_lastv2_n6_nfe400
GPU=${CUDA_VISIBLE_DEVICES:-0}
LOGBASE=/home/ks2218/la-proteina/nohup_${CFG}

echo "[$(date)] [GPU $GPU] === probe start: $CFG ==="
echo "[$(date)] [GPU $GPU] gen → ${LOGBASE}.gen.log"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON_EXEC proteinfoundation/generate.py \
    --config-name=$CFG > "${LOGBASE}.gen.log" 2>&1
GENRC=$?
echo "[$(date)] [GPU $GPU] gen exit=$GENRC"
if [ $GENRC -ne 0 ]; then
    echo "[$(date)] gen failed — last 20 lines of ${LOGBASE}.gen.log:"
    tail -20 "${LOGBASE}.gen.log"
    exit $GENRC
fi

# Sweep empty leftover tmp dirs (see CLAUDE.md / probe_sparse_*.sh notes).
GENDIR=/home/ks2218/la-proteina/inference/${CFG}
if [ -d "$GENDIR" ]; then
    for d in "$GENDIR"/job_0_n_*_id_*; do
        [ -d "$d/$(basename "$d")" ] && rmdir "$d/$(basename "$d")" 2>/dev/null
    done
fi

echo "[$(date)] [GPU $GPU] eval → ${LOGBASE}.eval.log"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON_EXEC proteinfoundation/evaluate.py \
    --config_name $CFG > "${LOGBASE}.eval.log" 2>&1
EVRC=$?
echo "[$(date)] [GPU $GPU] eval exit=$EVRC"
if [ $EVRC -ne 0 ]; then
    echo "[$(date)] eval failed — last 20 lines of ${LOGBASE}.eval.log:"
    tail -20 "${LOGBASE}.eval.log"
    exit $EVRC
fi

CSV=/home/ks2218/la-proteina/inference/results_${CFG}_0.csv
echo "[$(date)] === probe done ==="
if [ -f "$CSV" ]; then
    echo "Results CSV: $CSV"
    echo
    echo "scRMSD per sample (designable = scRMSD < 2 Å):"
    $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_csv('$CSV')
cols = [c for c in df.columns if 'scrmsd' in c.lower() or 'length' in c.lower() or 'len' == c.lower()]
print(df[cols].to_string())
"
fi
