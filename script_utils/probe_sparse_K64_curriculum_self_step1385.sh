#!/bin/bash
# One-shot designability probe for ca_only_sparse_K64_curriculum_self step 1385
# (best_val_00000013_000000001385.ckpt, val=4.1908 — best of full run).
#
# N=6 × L∈{50, 100, 200} × nsteps=400. ~30 min on 1× A100.
# Bar (CLAUDE.md "Sample-quality bar"): 1-2/3 designable at L=50 and L=100
# (scRMSD < 2 Å) — that's the convergence-relevant threshold.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/conda_envs/laproteina_env/bin/python
export PATH=/home/ks2218/conda_envs/laproteina_env/bin:$PATH
CFG=inference_sparse_K64_curriculum_self_step1385_n6_nfe400
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

# evaluate.py:207-209 creates tmp_dir = "<sample_dir>/<sample_basename>" before
# running MPNN. If MPNN crashes, tmp_dirs get left behind, and the next eval
# bombs every PDB on `assert not os.path.exists(tmp_dir)`. Sweep empty leftovers.
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
print(df[cols].to_string(index=False) if cols else df.to_string(index=False))
print()
# best scRMSD per sample (over folding models / modes if multi-col), grouped by length
scrmsd_cols = [c for c in df.columns if 'scrmsd' in c.lower()]
len_col = next((c for c in df.columns if c.lower() in ('length','nres','seq_len')), None)
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
