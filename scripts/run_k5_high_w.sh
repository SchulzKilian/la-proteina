#!/usr/bin/env bash
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
ALL_SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73"

for w in 96 128; do
  echo "[gen K5_w${w}] $(date -Iseconds)"
  mkdir -p results/ablation_2026_05_17/K5_w${w}
  $PY -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config steering/config/ablation_2026_05_17/K5_w${w}.yaml \
    --lengths 300 --seeds $ALL_SEEDS \
    --output_dir results/ablation_2026_05_17/K5_w${w} \
    --device cuda:0 --skip_unguided
done

for w in 96 128; do
  echo "[codesign K5_w${w}] $(date -Iseconds)"
  OUT_BASE=results/ablation_2026_05_17 $PY scripts/run_codesignability_sweep.py \
    --lengths 300 --seeds $ALL_SEEDS \
    --cfgs K5_w${w}
done

echo "[summary] $(date -Iseconds)"
# Extend the 2x2 summary script to include w=96 and w=128
$PY -c "
import sys
sys.path.insert(0, '/home/ks2218/la-proteina')
from scripts.summary_kw_2x2 import CELL_DIRS, collect, codesign_at_L300
from pathlib import Path
import statistics

EXTRA = {
    'K5_w96':  Path('results/ablation_2026_05_17/K5_w96'),
    'K5_w128': Path('results/ablation_2026_05_17/K5_w128'),
}
CELLS = {**CELL_DIRS, **EXTRA}

print()
print('=' * 110)
print(f\"{'cell':12s}  {'n_prop':>6}  {'pred':>7}  {'real':>7}  {'gap':>7}    {'n_cs':>5}  {'cs_μ':>5}  {'cs_med':>6}  {'<2Å':>5}  {'<3Å':>5}  {'max':>5}\")
print('=' * 110)
for name, d in CELLS.items():
    rows = collect(d)
    cs = codesign_at_L300(d / 'codesign_guided.csv')
    if not rows:
        print(f'{name:12s}  NO DATA')
        continue
    pred = statistics.mean(r['pred'] for r in rows)
    real = statistics.mean(r['real'] for r in rows)
    gap = statistics.mean(r['gap'] for r in rows)
    if cs:
        print(f\"{name:12s}  {len(rows):6d}  {pred:7.1f}  {real:7.1f}  {gap:+7.1f}    {cs['n']:5d}  {cs['mean']:5.2f}  {cs['median']:6.2f}  {cs['lt_2A']:2d}/{cs['n']:<2d}  {cs['lt_3A']:2d}/{cs['n']:<2d}  {cs['max']:5.2f}\")
    else:
        print(f\"{name:12s}  {len(rows):6d}  {pred:7.1f}  {real:7.1f}  {gap:+7.1f}    NO CODESIGN\")
"

echo "ALL_DONE_K5HIGHW $(date -Iseconds)"
