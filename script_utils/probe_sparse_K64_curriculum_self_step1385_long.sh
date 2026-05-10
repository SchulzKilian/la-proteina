#!/bin/bash
# Long-length (L=300/400/500) N=6 probe of ca_only_sparse_K64_curriculum_self
# step 1385. Companion to the L∈{50,100,200} pooled-N=18 probe.
#
# L set matches canonical's E022 (N=3 per L; canonical was 0/3 at every L,
# best 2.73/11.17/16.19 Å at 300/400/500). N=6 here gives ~1.4× more
# resolution. Wall: ~2-3 h on 1× A100 (ESMFold dominates at L=500).

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/conda_envs/laproteina_env/bin/python
export PATH=/home/ks2218/conda_envs/laproteina_env/bin:$PATH
CFG=inference_sparse_K64_curriculum_self_step1385_long_n6_nfe400
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

# Sweep leftover empty eval tmp_dirs from any prior crashed eval (idempotent
# retry). evaluate.py:207-209 hard-asserts `not os.path.exists(tmp_dir)`.
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
    $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_csv('$CSV')
df = df[['pdb_path','L','_res_scRMSD_ca_esmfold','_res_scRMSD_bb3o_esmfold']].copy()
df['id'] = df['pdb_path'].str.extract(r'id_(\d+)').astype(int)
df = df.sort_values(['L','id']).reset_index(drop=True)
df['des_ca']   = (df['_res_scRMSD_ca_esmfold']   < 2.0)
df['des_bb3o'] = (df['_res_scRMSD_bb3o_esmfold'] < 2.0)
print('Per-sample (best-of-8 MPNN seqs, scRMSD in Å):')
print(df[['L','id','_res_scRMSD_ca_esmfold','_res_scRMSD_bb3o_esmfold','des_ca','des_bb3o']].to_string(index=False))
print()
g = df.groupby('L').agg(
    n=('id','size'),
    des_ca=('des_ca','sum'),
    des_bb3o=('des_bb3o','sum'),
    median_ca=('_res_scRMSD_ca_esmfold','median'),
    min_ca=('_res_scRMSD_ca_esmfold','min'),
    max_ca=('_res_scRMSD_ca_esmfold','max'),
)
print('Designability summary (best-scRMSD-per-sample < 2 Å):')
print(g.to_string())
print()
print(f'Total: ca-mode designable = {df[\"des_ca\"].sum()}/{len(df)}, bb3o-mode designable = {df[\"des_bb3o\"].sum()}/{len(df)}')
print()
print('Canonical reference (E022 step 2646, N=3, nsteps=400):')
print('  L=300: 0/3, best 2.73 Å')
print('  L=400: 0/3, best 11.17 Å')
print('  L=500: 0/3, best 16.19 Å')
"
else
    echo "Expected CSV not found at $CSV — inspect ${LOGBASE}.eval.log."
fi
