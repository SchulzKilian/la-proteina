#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"

echo "[clean gen] $(date -Iseconds)"
GROUP=clean /home/ks2218/la-proteina/scripts/run_ablation_2026_05_17.sh

echo "[clean codesign] $(date -Iseconds)"
OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
  --lengths 300 400 500 --seeds $SEEDS \
  --cfgs clean_ens_n48

echo "[clean summary] $(date -Iseconds)"
$PY scripts/eval_ablation_2026_05_17.py 2>&1 | grep -E "clean|fixt1|NA-v1|Paired|L=300|L=400|L=500" | head -30

$PY <<'PYEOF'
import csv, statistics
path = "results/ablation_2026_05_17/clean_ens_n48/codesign_guided.csv"
rows = list(csv.DictReader(open(path)))
print(f"\nclean predictor codesign n={len(rows)}")
for L in [300, 400, 500]:
    rmsds = [float(r["coScRMSD_ca"]) for r in rows if r["protein_id"].endswith(f"_n{L}") and r["coScRMSD_ca"]!="inf"]
    if rmsds:
        print(f"  L={L}: n={len(rmsds)}, mean={statistics.mean(rmsds):.2f}, median={statistics.median(rmsds):.2f}, <2A={sum(1 for x in rmsds if x<2)}/{len(rmsds)}, <3A={sum(1 for x in rmsds if x<3)}/{len(rmsds)}")
rmsds = [float(r["coScRMSD_ca"]) for r in rows if r["coScRMSD_ca"]!="inf"]
if rmsds:
    print(f"  overall: n={len(rmsds)}, mean={statistics.mean(rmsds):.2f}, <2A={sum(1 for x in rmsds if x<2)}/{len(rmsds)}, <3A={sum(1 for x in rmsds if x<3)}/{len(rmsds)}")
PYEOF

echo "ALL_DONE_CLEAN $(date -Iseconds)"
