#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"

echo "[ug gen] $(date -Iseconds)"
GROUP=ug /home/ks2218/la-proteina/scripts/run_ablation_2026_05_17.sh

echo "[ug codesign] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 --seeds $SEEDS \
  --cfgs ug_K2 ug_K3 ug_K5

echo "[ug summary] $(date -Iseconds)"
$PY scripts/eval_ablation_2026_05_17.py 2>&1 | grep -E "ug_K|baseline|Baseline"
$PY <<PYEOF
import csv, statistics
for cell in ["ug_K2", "ug_K3", "ug_K5"]:
    path = f"results/ablation_2026_05_17/{cell}/codesign_guided.csv"
    try:
        rmsds = [float(r["coScRMSD_ca"]) for r in csv.DictReader(open(path)) if r["coScRMSD_ca"]!="inf"]
        print(f"{cell} codesign: n={len(rmsds)}, mean={statistics.mean(rmsds):.2f}, median={statistics.median(rmsds):.2f}, <2A={sum(1 for x in rmsds if x<2)}/16, <3A={sum(1 for x in rmsds if x<3)}/16")
    except Exception as e:
        print(f"{cell}: {e}")
PYEOF

echo "ALL_DONE_UG $(date -Iseconds)"
