#!/usr/bin/env bash
# Top-row n=64 experiment (2026-05-30): bump L400-only, unguided + net_charge_target w32
# from n=16 (seeds 42-57) to n=64 by generating 48 NEW seeds (58-105).
# One generate call produces BOTH arms (guided w32 + unguided) at the same seeds,
# so the new unguided is paired with the new guided by construction.
# Existing 16 (seeds 42-57): w32 in results/net_charge_target_sweep/, unguided in
# results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv.
# Combined -> n=64 paired at L400. NSTEPS=400 (HARD RULE). Single L4, cuda:0.
set -uo pipefail   # NOT set -e (cluster TaskProlog quirk; harmless locally, kept for parity)

PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
DEVICE=cuda:0
NSTEPS=400
LENGTH=400
NEWTREE=results/net_charge_target_L400_n64
CFG=steering/config/sweep_net_charge_target/net_charge_target_w32.yaml
SEEDS=$(seq 58 105 | tr '\n' ' ')   # 48 new seeds -> 64 total with 42-57

echo "[$(date -u +%FT%TZ)] === n=64 L400 top-row: generating 48 new seeds (58-105), both arms ==="

# Step 1: generate guided w32 + unguided at L400, seeds 58-105 (one pass = both arms)
"$PY" -m steering.generate \
    --proteina_config inference_ucond_notri_long \
    --steering_config "$CFG" \
    --lengths $LENGTH \
    --seeds $SEEDS \
    --nsteps $NSTEPS \
    --output_dir "$NEWTREE/w32" \
    --device $DEVICE
echo "[$(date -u +%FT%TZ)] === generation done; guided=$(ls $NEWTREE/w32/guided/*.pdb 2>/dev/null | wc -l) unguided=$(ls $NEWTREE/w32/unguided/*.pdb 2>/dev/null | wc -l) ==="

# Step 2: present the unguided pdbs as a 'guided/' view so run_codesignability_sweep can eval them
mkdir -p "$NEWTREE/unguided"
ln -sfn "$(cd "$NEWTREE/w32/unguided" && pwd)" "$NEWTREE/unguided/guided"

# Step 3: codesign both arms (use_pdb_seq=True / ESMFold, keep_outputs=True)
OUT_BASE="$NEWTREE" "$PY" scripts/run_codesignability_sweep.py \
    --lengths $LENGTH \
    --seeds $(seq 58 105) \
    --cfgs w32 unguided
echo "[$(date -u +%FT%TZ)] === codesign done ==="

# Step 4: paired analysis, n=64 at L400 (existing 16 + new 48)
"$PY" - <<'PYEOF'
import pandas as pd, numpy as np
from scipy import stats

def load_codesign(path, seed_filter=None):
    d = pd.read_csv(path)
    d['s'] = d['protein_id'].str.extract(r'(s\d+)')
    d['L'] = d['protein_id'].str.extract(r'_?n(\d+)').astype(int)
    d = d[d.L == 400]
    if seed_filter is not None:
        d = d[d.s.isin(seed_filter)]
    return d.set_index('s')['coScRMSD_ca']

old_seeds = ['s%d' % s for s in range(42, 58)]
# guided w32: old (sweep tree) + new (n64 tree)
g_old = load_codesign('results/net_charge_target_sweep/net_charge_target_w32/codesign_guided.csv', old_seeds)
g_new = load_codesign('results/net_charge_target_L400_n64/w32/codesign_guided.csv')
guided = pd.concat([g_old, g_new])
# unguided: old (matched-seed file) + new (n64 tree)
u_old_df = pd.read_csv('results/noise_aware_ensemble_sweep/codesign_unsteered_matched_seed.csv')
u_old_df['s'] = u_old_df['protein_id'].str.extract(r'(s\d+)')
u_old = u_old_df[u_old_df.length == 400].set_index('s')['coScRMSD_ca']
u_new = load_codesign('results/net_charge_target_L400_n64/unguided/codesign_guided.csv')
unguided = pd.concat([u_old, u_new])

idx = guided.index.intersection(unguided.index)
g = guided.loc[idx].values; u = unguided.loc[idx].values
imp = ((u >= 2) & (g < 2)).sum(); wor = ((u < 2) & (g >= 2)).sum()
nd = imp + wor
mcp = stats.binomtest(min(imp, wor), nd, 0.5).pvalue if nd > 0 else 1.0
wp = stats.wilcoxon(u - g).pvalue
print('\n==================  L400 paired n=%d (w32 vs unguided)  ==================' % len(idx))
print(' unguided des=%.0f%% (med %.2f)  ->  w32 des=%.0f%% (med %.2f)'
      % (100*(u<2).mean(), np.median(u), 100*(g<2).mean(), np.median(g)))
print(' McNemar improve=%d worsen=%d  exact p=%.4f | Wilcoxon p=%.4f' % (imp, wor, mcp, wp))
print('==========================================================================')
PYEOF

echo "[$(date -u +%FT%TZ)] === n=64 L400 top-row pipeline complete ==="
