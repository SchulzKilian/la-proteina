#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"

mkdir -p results/ablation_2026_05_17/constant_matched
echo "[constant_matched gen] $(date -Iseconds)"
$PY -m steering.generate \
  --proteina_config inference_ucond_notri_long \
  --steering_config steering/config/ablation_2026_05_17/constant_matched.yaml \
  --lengths 300 --seeds $SEEDS \
  --output_dir results/ablation_2026_05_17/constant_matched \
  --device cuda:0 --skip_unguided

echo "[constant_matched codesign] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $SEEDS \
  --cfgs constant_matched

echo "[summary] $(date -Iseconds)"
$PY scripts/eval_ablation_2026_05_17.py | grep -E "constant_matched|baseline|Baseline" | head -10
$PY -c "
import csv, statistics
rmsds = [float(r['coScRMSD_ca']) for r in csv.DictReader(open('results/ablation_2026_05_17/constant_matched/codesign_guided.csv')) if r['coScRMSD_ca']!='inf']
print(f'constant_matched codesign: n={len(rmsds)}, mean={statistics.mean(rmsds):.2f}, median={statistics.median(rmsds):.2f}, <2A={sum(1 for x in rmsds if x<2)}/16, <3A={sum(1 for x in rmsds if x<3)}/16')
"

echo "ALL_DONE_MATCHED $(date -Iseconds)"
