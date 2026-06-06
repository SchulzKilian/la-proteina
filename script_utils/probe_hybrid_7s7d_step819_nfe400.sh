#!/bin/bash
# Canonical N=6 × L∈{50,100,200} × nsteps=400 designability probe on the
# layer-hybrid sparse/dense model (ca_only_hybrid_7s7d), global_step 819.
#
# Architecture: 14-layer CA-only; first 7 sparse (K=40), last 7 dense.
# layer_sparse_mask=[T,T,T,T,T,T,T,F,F,F,F,F,F,F], layer_K_splits.
# CA-only: no AE ckpt, latent_dim=None.
# Single ckpt → generate.py (standard path, NOT generate_hybrid.py).
# NN config restored automatically from ckpt via load_from_checkpoint.
# nsteps=400 inherited from inference_base.yaml.
#
# E126: designability probe — does ca_only_hybrid_7s7d clear the
# canonical sample-quality bar (1-2/3 designable at L=50 and L=100)?

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
CFG=inference_hybrid_7s7d_step819_n6_nfe400
GPU=${CUDA_VISIBLE_DEVICES:-4}
LOGBASE=/home/ks2218/la-proteina/nohup_probe_hybrid_7s7d_step819_nfe400

echo "[$(date)] [GPU $GPU] === E126 probe start: $CFG ==="
echo "[$(date)] [GPU $GPU] ckpt: /home/ks2218/la-proteina/ca_only_hybrid_7s7d_step819.ckpt -> best_val_00000008_000000000819.ckpt"
echo "[$(date)] [GPU $GPU] model: ca_only_hybrid_7s7d (7 sparse + 7 dense layers, K=40, step 819)"

# Pre-flight: remove stale CSV and PDB dir to ensure fresh run
CSV=/home/ks2218/la-proteina/inference/results_${CFG}_0.csv
GENDIR=/home/ks2218/la-proteina/inference/laproteina_hybrid_7s7d_step819_n6_nfe400

if [ -f "$CSV" ]; then
    NROWS=$($PYTHON_EXEC -c "import pandas as pd; print(len(pd.read_csv('$CSV')))" 2>/dev/null || echo 0)
    echo "[$(date)] [GPU $GPU] Found existing CSV at $CSV ($NROWS rows) — removing for fresh run."
    rm -f "$CSV"
fi

if [ -d "$GENDIR" ]; then
    echo "[$(date)] [GPU $GPU] Removing old PDB output dir $GENDIR."
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

# Sweep crash-leftover eval tmp_dirs (evaluate.py asserts non-existence).
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

# Print per-length summary from the results CSV
echo "[$(date)] [GPU $GPU] === E126 probe complete ==="
if [ -f "$CSV" ]; then
    echo "--- Results summary ---"
    $PYTHON_EXEC - <<'PYEOF'
import pandas as pd
csv = "/home/ks2218/la-proteina/inference/results_inference_hybrid_7s7d_step819_n6_nfe400_0.csv"
df = pd.read_csv(csv)
for L in sorted(df["n_res"].unique()):
    sub = df[df["n_res"] == L]
    des = (sub["scRMSD"] < 2.0).sum()
    best = sub["scRMSD"].min()
    med = sub["scRMSD"].median()
    print(f"L={L:3d}: {des}/6 designable, best={best:.2f} A, median={med:.2f} A")
total = (df["scRMSD"] < 2.0).sum()
print(f"Pooled: {total}/18 ({100*total/18:.0f}%), mean scRMSD={df['scRMSD'].mean():.2f} A")
PYEOF
else
    echo "Expected CSV not found at $CSV — inspect ${LOGBASE}.eval.log."
fi
