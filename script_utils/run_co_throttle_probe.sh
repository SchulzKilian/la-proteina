#!/usr/bin/env bash
# Contact-order throttle head-to-head: no-throttle vs geometric(bond/clash) vs rama proxy.
# Tests whether the Cα pseudo-rama manifoldness proxy moves the codesign-CO frontier on a
# CONFORMATIONAL target (rama is blind to Rg's scale-only push; it fires on CO).
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
SEEDS="42 43 44 45 46 47"; LENGTHS="100 200 300"; NS=400
declare -A CFG=( [co_baseline]=contact_order_baseline [co_geom]=contact_order_lookahead_prop [co_rama]=contact_order_lookahead_rama )
for arm in co_baseline co_geom co_rama; do
  out=results/co_probe/$arm
  n=$(ls "$out/guided"/*.pdb 2>/dev/null|wc -l)
  if [ "$n" -ge 18 ]; then echo "[$(date -u +%FT%TZ)] $arm has $n; skip gen"; continue; fi
  echo "[$(date -u +%FT%TZ)] gen $arm (${CFG[$arm]})"
  "$PY" -m steering.generate --proteina_config inference_ucond_notri_long \
    --steering_config steering/config/geometric_lookahead/${CFG[$arm]}.yaml \
    --lengths $LENGTHS --seeds $SEEDS --nsteps $NS --skip_unguided \
    --output_dir "$out" --device cuda:0
done
echo "[$(date -u +%FT%TZ)] === CO probe generation complete ==="
