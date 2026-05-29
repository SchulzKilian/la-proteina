#!/bin/bash
# Canonical N=6 × L∈{50,100,200} × nsteps=400 designability probe on the
# `ca_only_sparse_K40_scnbr_t04` training run, step 1133.
# This is E083: the nsteps=400 redo of E039 (which ran nsteps=200 and got 3/18).
# Probes the question: does trained-with-Fix-C2 beat inference-only-Fix-C2 (E082:
# 5/18 t<0.4 on canonical sparse-K40) once both are at canonical inference resolution?
#
# sc_neighbors=True is stored in the ckpt hparams and fires automatically
# at the trained t_threshold=0.4. No sc_neighbors_override is added.
#
# ~13-15 min on 1× L4.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
CFG=inference_scnbr_t04_step1133_n6_nfe400
GPU=${CUDA_VISIBLE_DEVICES:-1}
LOGBASE=/home/ks2218/la-proteina/nohup_probe_scnbr_t04_step1133_nfe400

echo "[$(date)] [GPU $GPU] === E083 probe start: $CFG ==="
echo "[$(date)] [GPU $GPU] ckpt: /home/ks2218/la-proteina/best_val_00000011_000000001133.ckpt"
echo "[$(date)] [GPU $GPU] scnbr_t04 arm: sc_neighbors=True, t_threshold=0.4 (from ckpt hparams)"

# Pre-flight: remove old E039 CSV (nsteps=200, 18 rows) so generate.py does
# not early-exit. Also remove old PDB output directory for a fresh run.
CSV=/home/ks2218/la-proteina/inference/results_${CFG}_0.csv
GENDIR=/home/ks2218/la-proteina/inference/${CFG}

if [ -f "$CSV" ]; then
    NROWS=$($PYTHON_EXEC -c "import pandas as pd; print(len(pd.read_csv('$CSV')))" 2>/dev/null || echo 0)
    echo "[$(date)] [GPU $GPU] Found existing CSV at $CSV ($NROWS rows) — this is the E039 nsteps=200 result."
    echo "[$(date)] [GPU $GPU] Removing it so E083 nsteps=400 run generates fresh structures."
    rm -f "$CSV"
fi

if [ -d "$GENDIR" ]; then
    echo "[$(date)] [GPU $GPU] Removing old PDB output dir $GENDIR (E039 structures)."
    rm -rf "$GENDIR"
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

# Print summary from the results CSV
echo "[$(date)] [GPU $GPU] === E083 probe complete ==="
if [ -f "$CSV" ]; then
    echo "--- Results summary ---"
    $PYTHON_EXEC - <<'PYEOF'
import pandas as pd
import sys
csv = "/home/ks2218/la-proteina/inference/results_inference_scnbr_t04_step1133_n6_nfe400_0.csv"
df = pd.read_csv(csv)
for L in sorted(df["n_res"].unique()):
    sub = df[df["n_res"] == L]
    des = (sub["scRMSD"] < 2.0).sum()
    best = sub["scRMSD"].min()
    med = sub["scRMSD"].median()
    print(f"L={L:3d}: {des}/6 designable, best={best:.2f} Å, median={med:.2f} Å")
total = (df["scRMSD"] < 2.0).sum()
print(f"Pooled: {total}/18 ({100*total/18:.0f}%), mean scRMSD={df['scRMSD'].mean():.2f} Å")
PYEOF
else
    echo "Expected CSV not found at $CSV — inspect ${LOGBASE}.eval.log."
fi
