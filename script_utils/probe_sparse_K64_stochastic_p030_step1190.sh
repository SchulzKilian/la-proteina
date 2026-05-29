#!/bin/bash
# Canonical N=6 × L∈{50,100,200} × nsteps=400 designability probe on the
# `ca_only_sparse_K64_stochastic_p030` training run, step 1190.
# First probe of the stochastic_k=0.30 training variant.
# At inference: static SALAD-canonical K=64 (16/16/32) at all t.
#
# ~13-15 min per length on 1× L4. Wall-clock total: ~45 min.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
CFG=inference_sparse_K64_stochastic_p030_step1190_n6_nfe400
GPU=${CUDA_VISIBLE_DEVICES:-0}
LOGBASE=/home/ks2218/la-proteina/nohup_${CFG}

echo "[$(date)] [GPU $GPU] === probe start: $CFG ==="

# Pre-flight: generate.py early-exits if results CSV already exists.
# A prior crashed eval leaves an EMPTY CSV behind; self-heal by removing it.
CSV=/home/ks2218/la-proteina/inference/results_${CFG}_0.csv
if [ -f "$CSV" ]; then
    NROWS=$($PYTHON_EXEC -c "import pandas as pd; print(len(pd.read_csv('$CSV')))" 2>/dev/null || echo 0)
    if [ "$NROWS" -eq 0 ]; then
        echo "[$(date)] [GPU $GPU] Stale empty CSV at $CSV — removing so gen re-runs."
        rm -f "$CSV"
    else
        echo "[$(date)] [GPU $GPU] Found populated CSV at $CSV ($NROWS rows) — gen will early-exit per generate.py:448."
    fi
fi

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

# Sweep crash-left-over eval tmp_dirs (evaluate.py asserts non-existence).
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
    echo "scRMSD per sample (designable = scRMSD < 2 Å):"
    $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_csv('$CSV')
cols = [c for c in df.columns if 'scrmsd' in c.lower() or c == 'L' or 'length' in c.lower() or 'len' == c.lower()]
print(df[cols].to_string(index=False) if cols else df.to_string(index=False))
print()
scrmsd_cols = [c for c in df.columns if 'scrmsd' in c.lower()]
len_col = next((c for c in df.columns if c == 'L' or c.lower() in ('length','nres','seq_len')), None)
if scrmsd_cols and len_col:
    df['_best_scrmsd'] = df[scrmsd_cols].min(axis=1)
    summary = df.groupby(len_col).agg(
        n=('_best_scrmsd','size'),
        designable=('_best_scrmsd', lambda s: int((s < 2.0).sum())),
        median_scrmsd=('_best_scrmsd','median'),
        min_scrmsd=('_best_scrmsd','min'),
    )
    print('Designability summary (best-scRMSD-per-sample < 2 Å):')
    print(summary.to_string())
"
else
    echo "Expected CSV not found at $CSV — inspect ${LOGBASE}.eval.log."
fi
