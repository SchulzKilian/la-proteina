#!/bin/bash
# Verification probe for the off-by-one cap fix in sparse_neighbors.py
# (replaces `min(2*n_seq, N-1)` with `min(2*n_seq, N)`).
#
# Same ckpt and same schedule as probe_sparse_K64_curriculum_self_step1385.sh,
# but runs against the FIXED build of sparse_neighbors.py. Writes to a
# distinct output prefix so prior (pre-fix) results stay intact for A/B.
#
# Pre-fix N=18 baseline at L∈{50, 100, 200} = 44.4% / 55.6% / 11.1%
# (variants.md §12). Post-fix expectations:
#   - L=50:  should change (bug fires here — 49/50 → 50/50 coverage per query)
#   - L=100: ~unchanged (bug doesn't apply; k_seq saturates at 2*n_seq=64 < N-1=99)
#   - L=200: ~unchanged (same reason)
#
# N=6 × L∈{50, 100, 200} × nsteps=400. ~30 min on 1× A100.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/conda_envs/laproteina_env/bin/python
export PATH=/home/ks2218/conda_envs/laproteina_env/bin:$PATH
CFG=inference_sparse_K64_step1385_FIXEDCAP_n6_nfe400
GPU=${CUDA_VISIBLE_DEVICES:-0}
LOGBASE=/home/ks2218/la-proteina/nohup_${CFG}

echo "[$(date)] [GPU $GPU] === probe start: $CFG ==="

# Pre-flight: generate.py:448 early-exits if results CSV already exists. A prior
# crashed eval leaves an EMPTY CSV behind; gen would then skip on every re-run.
# Self-heal: if the CSV is present but empty, remove it so gen re-runs.
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

# Sweep crash-left-over eval tmp_dirs (evaluate.py:207-209 asserts non-existence).
# Use `rm -rf` because crashed-MPNN leftovers contain partial files. The
# targeted path is the nested same-named DIRECTORY, never the sibling .pdb
# FILE — `[ -d … ]` guards against accidentally matching the file.
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
