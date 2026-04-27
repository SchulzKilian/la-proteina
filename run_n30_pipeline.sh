#!/bin/bash
# Sequential N=30/length × {50,100,200} eval across 4 runs.
# Output: per-run CSV under inference/results_inference_<run>_n30_0.csv
# Log:    /tmp/n30_pipeline.log (tee'd from each step)

set -uo pipefail

export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ks2218/la-proteina

LOG=/tmp/n30_pipeline.log
SEED=100

# (config_name, ckpt_filename)
declare -A CKPTS
CKPTS[baseline]=baseline_wd0.05_step2646.ckpt
CKPTS[v2]=v2_wd0.1_step2078.ckpt
CKPTS[wd0]=wd0_step1638.ckpt
CKPTS[sparse]=sparse_K40_step1259.ckpt

ORDER=(baseline v2 wd0 sparse)

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] === N=30 pipeline starting (seed=$SEED) ===" | tee -a "$LOG"

for run in "${ORDER[@]}"; do
    ckpt=${CKPTS[$run]}
    cfg=inference_${run}_n30
    out_dir=./inference/inference_${run}_n30
    csv=./inference/results_inference_${run}_n30_0.csv

    echo "[$(ts)] === [$run] start (ckpt=$ckpt) ===" | tee -a "$LOG"

    # Clean any prior outputs for this run + the always-overwritten inference_base
    rm -rf "$out_dir" ./inference/inference_base "$csv"

    echo "[$(ts)] [$run] generating..." | tee -a "$LOG"
    python proteinfoundation/generate.py \
        --config-name="$cfg" \
        ckpt_path=. \
        ckpt_name="$ckpt" \
        'generation=uncond_ca_only_n30' \
        seed=$SEED \
        2>&1 | tee -a "$LOG"

    if [ ! -d ./inference/inference_base ]; then
        echo "[$(ts)] [$run] FAILED: no inference_base produced" | tee -a "$LOG"
        continue
    fi
    mv ./inference/inference_base "$out_dir"
    n_pdb=$(find "$out_dir" -name '*.pdb' | wc -l)
    echo "[$(ts)] [$run] generation done; $n_pdb PDBs in $out_dir" | tee -a "$LOG"

    echo "[$(ts)] [$run] evaluating..." | tee -a "$LOG"
    python proteinfoundation/evaluate.py --config_name "$cfg" 2>&1 | tee -a "$LOG"

    if [ -f "$csv" ]; then
        echo "[$(ts)] [$run] eval done; CSV at $csv" | tee -a "$LOG"
    else
        echo "[$(ts)] [$run] WARNING: expected CSV $csv not found" | tee -a "$LOG"
    fi
done

echo "[$(ts)] === all runs done; running aggregation ===" | tee -a "$LOG"

python - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd, os
runs = ['baseline', 'v2', 'wd0', 'sparse']
rows = []
for r in runs:
    p = f'./inference/results_inference_{r}_n30_0.csv'
    if not os.path.exists(p):
        print(f'[!] missing CSV for {r}: {p}')
        continue
    df = pd.read_csv(p)
    df = df[df['_res_scRMSD_ca_esmfold'] >= 0]
    for L, g in df.groupby('L'):
        ca = g['_res_scRMSD_ca_esmfold']
        bb = g['_res_scRMSD_bb3o_esmfold']
        rows.append({
            'run': r, 'L': L, 'N': len(g),
            'ca_min': ca.min(), 'ca_p25': ca.quantile(0.25),
            'ca_med': ca.median(), 'ca_mean': ca.mean(),
            'ca_p75': ca.quantile(0.75), 'ca_max': ca.max(),
            'des_ca<2': int((ca < 2).sum()),
            'des_rate': float((ca < 2).mean()),
            'bb3o_min': bb.min(), 'bb3o_med': bb.median(),
        })
out = pd.DataFrame(rows)
print('\n=== AGGREGATE (N=30/length, seed=100) ===')
print(out.to_string(index=False, float_format='%.2f'))
out.to_csv('./inference/n30_aggregate.csv', index=False)
print('\nWritten ./inference/n30_aggregate.csv')
PY

echo "[$(ts)] === pipeline done ===" | tee -a "$LOG"
