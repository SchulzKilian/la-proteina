#!/bin/bash
# E084: scnbr_t04 step 1133 at nsteps=400 with sc_neighbors_bootstrap=False.
# Tests the bootstrap-degeneracy hypothesis for E083's 1/18 result.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
CFG=inference_scnbr_t04_step1133_nobootstrap_n6_nfe400
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

echo "[$(date)] [GPU $GPU] Fix C2 canary (should show sc_neighbors=True from ckpt hparams):"
grep -E "\[Fix C2\]|\[sc_neighbors override\]" "${LOGBASE}.gen.log" || \
    echo "  WARNING: no Fix C2 canary found"

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
    echo
    $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_csv('$CSV')
sc = [c for c in df.columns if 'scrmsd' in c.lower() and 'all' not in c.lower()]
df['_best'] = df[sc].min(axis=1)
df = df[df['_best'] > 0]   # drop -1 ESMFold sentinels
print('Designability summary (best-scRMSD-per-sample < 2 Å, sentinels dropped):')
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
