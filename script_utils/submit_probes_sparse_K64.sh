#!/bin/bash
#SBATCH -J K64probes
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=2:00:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_K64probes_%j.out

# Bundle the three K=64 step-1385 inference probes into one short SL3 job:
#   1. probe_sparse_K64_curriculum_self_step1385.sh   (canonical schedule + off-by-one fix in code)
#   2. probe_sparse_K64_step1385_NOCURR_salad.sh      (curriculum OFF → static (8,16,32) at all t)
#   3. probe_sparse_K64_step1385_LOWTSOFT.sh          (curriculum ON, low-t bucket → (16,8,24))
#
# Same step-1385 ckpt for all three; each writes to its own
# inference/<CFG>/ output dir, so no cross-contamination. Each probe takes
# ~30 min (gen 5 min + eval 10-15 min); total ~90 min serial. Time budget
# 2h gives buffer.

source $HOME/.bashrc

# Activate laproteina_env via PATH prepend — same pattern as submit_train_ca_only_1gpu.sh
# (avoids `conda activate` re-resolving through /rds).
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export PYTHON_EXEC=$LAPROTEINA_ENV/bin/python
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env

echo "[$(date)] Running on node: $(hostname)"
echo "[$(date)] Using Python: $(which python)"
echo "[$(date)] GPUs available: $CUDA_VISIBLE_DEVICES"

cd /home/ks2218/la-proteina

run_probe() {
    local name="$1"
    local script="$2"
    echo
    echo "================================================================"
    echo "[$(date)] === BEGIN $name ==="
    echo "================================================================"
    bash "$script"
    local rc=$?
    echo "[$(date)] === END $name (rc=$rc) ==="
    return $rc
}

# Run all three probes; continue past failures so one bad probe doesn't kill the others.
run_probe "canonical"  script_utils/probe_sparse_K64_curriculum_self_step1385.sh || echo "[!] canonical probe failed, continuing"
run_probe "NOCURR"     script_utils/probe_sparse_K64_step1385_NOCURR_salad.sh    || echo "[!] NOCURR probe failed, continuing"
run_probe "LOWTSOFT"   script_utils/probe_sparse_K64_step1385_LOWTSOFT.sh        || echo "[!] LOWTSOFT probe failed, continuing"

echo
echo "================================================================"
echo "[$(date)] === ALL PROBES DONE — summary table ==="
echo "================================================================"
for CFG in \
    inference_sparse_K64_curriculum_self_step1385_n6_nfe400 \
    inference_sparse_K64_step1385_NOCURR_salad_n6_nfe400 \
    inference_sparse_K64_step1385_LOWTSOFT_n6_nfe400; do
    CSV=/home/ks2218/la-proteina/inference/results_${CFG}_0.csv
    echo
    echo "--- $CFG ---"
    if [ -f "$CSV" ]; then
        $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_csv('$CSV')
if len(df) == 0:
    print('  CSV exists but empty — check eval log')
else:
    scrmsd_cols = [c for c in df.columns if 'scrmsd' in c.lower()]
    len_col = next((c for c in df.columns if c == 'L' or c.lower() in ('length','nres','seq_len')), None)
    if scrmsd_cols and len_col:
        df['_best_scrmsd'] = df[scrmsd_cols].min(axis=1)
        s = df.groupby(len_col).agg(
            n=('_best_scrmsd','size'),
            designable=('_best_scrmsd', lambda s: int((s < 2.0).sum())),
            median_scrmsd=('_best_scrmsd','median'),
            min_scrmsd=('_best_scrmsd','min'),
        )
        print(s.to_string())
    else:
        print('  ⚠️ unexpected columns:', list(df.columns)[:10], '…')
"
    else
        echo "  ✗ CSV not found"
    fi
done
echo
echo "[$(date)] === DONE ==="
