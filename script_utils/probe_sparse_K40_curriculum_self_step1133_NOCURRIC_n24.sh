#!/bin/bash
# N=24 extension of E078 (curric-OFF probe) on the K=40+curric+self ckpt step 1133.
# Same ckpt, same seed=5 (from inference_base.yaml), nsteps=400, curriculum
# forced OFF at inference (static K=40 (8,8,16) at all t).
# Adds 18 new samples per L on top of E078's 6 (paired within seed=5 batch 0).
# Distribution-shift caveat carries over from E078.
# ~40-50 min on 1× L4.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
CFG=inference_sparse_K40_curriculum_self_step1133_NOCURRIC_n24_nfe400
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

GENDIR=/home/ks2218/la-proteina/inference/${CFG}
if [ -d "$GENDIR" ]; then
    for d in "$GENDIR"/job_0_n_*_id_*; do
        [ -d "$d/$(basename "$d")" ] && rm -rf "$d/$(basename "$d")"
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
    $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_csv('$CSV')
sc = [c for c in df.columns if 'scrmsd' in c.lower() and 'all' not in c.lower()]
df['_best'] = df[sc].min(axis=1)
print('Designability (best-scRMSD-per-sample < 2 Å):')
print(df.groupby('L').agg(
    n=('_best','size'),
    designable=('_best', lambda s: int((s < 2.0).sum())),
    median=('_best','median'),
    best=('_best','min'),
).to_string())
print()
print('Pooled: {}/{} = {:.1f}%'.format(int((df['_best']<2.0).sum()), len(df), 100*(df['_best']<2.0).mean()))
"
else
    echo "Expected CSV not found at $CSV — inspect ${LOGBASE}.eval.log."
fi
