#!/usr/bin/env bash
# Per-residue look-ahead throttle FULL FRONTIER on contact_order (minimize).
# Two per-residue proxies: geometric (bond/clash) and Cα pseudo-rama. beta calibrated
# per proxy (Stage-1 baseline run -> per-residue dP_p90). Grid: w{4,8,16,32} x 16 seeds
# (42-57) x L{300,400}. 8-seq MPNN designability (scRMSD_ca_min) to match the existing
# baseline_w* (no-throttle) cells in the same tree. 2 GPUs (geom on one, rama on the other).
# Compared offline against baseline_w* + unsteered. Resumable. nohup-safe.
set -uo pipefail
cd /home/ks2218/la-proteina
PY=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}

GPU_GEOM=${GPU_GEOM:-6}
GPU_RAMA=${GPU_RAMA:-7}
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57"
LENGTHS="300 400"
WS="4 8 16 32"
NSTEPS=400
ROOT=results/geom_lookahead_sweep
PRIORS=steering/throttle_priors/ca_pseudo_rama.pt
CFGDIR=steering/config/geometric_lookahead/perres_frontier
mkdir -p "$CFGDIR" results/_perres_calib

write_cfg () {  # $1=proxy(geometric|rama) $2=w $3=beta $4=mode  -> path on stdout
  local proxy=$1 w=$2 beta=$3 mode=$4
  local f="$CFGDIR/${proxy}_w${w}_${mode}.yaml"
  {
    echo "steering:"
    echo "  method: geometric_lookahead"
    echo "  enabled: true"
    echo "  channel: bb_ca"
    echo "  mode: ${mode}"
    echo "  proxy_type: ${proxy}"
    echo "  per_residue: true"
    echo "  f_map: exp"
    echo "  beta: ${beta}"
    echo "  objectives: [{property: contact_order, direction: minimize, weight: 1.0}]"
    echo "  schedule: {type: linear_ramp, w_max: ${w}.0, t_start: 0.3, t_end: 0.8}"
    if [ "$proxy" = "rama" ]; then
      echo "  proxy: {rama_priors_path: ${PRIORS}}"
    else
      echo "  proxy: {bond_target_nm: 0.38, clash_radius_nm: 0.40, lambda_clash: 1.0}"
    fi
    echo "  gradient_norm: unit"
    echo "  gradient_clip: 0.0"
    echo "  log_diagnostics: true"
  } > "$f"
  echo "$f"
}

calibrate () {  # $1=proxy $2=gpu -> echoes beta (s=0.2 at per-residue dP p90); fallback default
  local proxy=$1 gpu=$2
  local cfg; cfg=$(write_cfg "$proxy" 16 0.0 baseline)
  local out="results/_perres_calib/${proxy}"
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m steering.generate \
      --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
      --lengths 300 --seeds 42 --nsteps $NSTEPS --skip_unguided \
      --output_dir "$out" --device cuda:0 >/dev/null 2>&1 || true
  "$PY" - "$out" "$proxy" <<'PYEOF'
import sys, json, glob, numpy as np
out, proxy = sys.argv[1], sys.argv[2]
default = {"geometric": 130.0, "rama": 0.25}.get(proxy, 1.0)
fs = glob.glob(f"{out}/diagnostics/*.json")
beta = default
try:
    d = json.load(open(fs[0]))
    p90 = [e["pr_dP_p90"] for e in d if e.get("lambda0", 0) > 0 and e.get("pr_dP_p90", 0) > 0]
    if p90:
        anc = float(np.median(p90))
        beta = -np.log(0.2) / anc if anc > 1e-9 else default
except Exception:
    pass
print(f"{beta:.4f}")
PYEOF
}

echo "[$(date -u +%FT%TZ)] === Stage-1 calibration ==="
BETA_GEOM=$(calibrate geometric "$GPU_GEOM")
echo "[$(date -u +%FT%TZ)] geometric per-residue beta = $BETA_GEOM"
BETA_RAMA=$(calibrate rama "$GPU_RAMA")
echo "[$(date -u +%FT%TZ)] rama per-residue beta = $BETA_RAMA"

run_cell () {  # $1=proxy $2=w $3=beta $4=gpu
  local proxy=$1 w=$2 beta=$3 gpu=$4
  local cell="contact_order_${proxy}res_w${w}"
  local n_pdb; n_pdb=$(ls "$ROOT/$cell/guided"/*.pdb 2>/dev/null | wc -l)
  if [ "$n_pdb" -lt 32 ]; then
    local cfg; cfg=$(write_cfg "$proxy" "$w" "$beta" lookahead_proportional)
    echo "[$(date -u +%FT%TZ)] [GPU$gpu] gen $cell (beta=$beta)"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m steering.generate \
        --proteina_config inference_ucond_notri_long --steering_config "$cfg" \
        --lengths $LENGTHS --seeds $SEEDS --nsteps $NSTEPS --skip_unguided \
        --output_dir "$ROOT/$cell" --device cuda:0
  else
    echo "[$(date -u +%FT%TZ)] $cell has $n_pdb pdbs; skip gen"
  fi
  echo "[$(date -u +%FT%TZ)] [GPU$gpu] scRMSD designability $cell"
  CUDA_VISIBLE_DEVICES=$gpu OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "$cell" --seeds $SEEDS --lengths $LENGTHS
}

worker () {  # $1=proxy $2=beta $3=gpu
  local proxy=$1 beta=$2 gpu=$3
  for w in $WS; do run_cell "$proxy" "$w" "$beta" "$gpu"; done
  echo "[$(date -u +%FT%TZ)] === ${proxy} worker DONE ==="
}

echo "[$(date -u +%FT%TZ)] === gen+eval: geom on GPU$GPU_GEOM, rama on GPU$GPU_RAMA ==="
worker geometric "$BETA_GEOM" "$GPU_GEOM" > nohup_perres_geom.log 2>&1 &
worker rama "$BETA_RAMA" "$GPU_RAMA" > nohup_perres_rama.log 2>&1 &
wait

# --- baseline top-up: bring no-throttle contact_order w4/w32 to 16 seeds for a clean reference ---
echo "[$(date -u +%FT%TZ)] === baseline top-up (w4,w32 -> 16 seeds) ==="
CUDA_VISIBLE_DEVICES=$GPU_GEOM "$PY" -m steering.run_geom_lookahead_sweep --device cuda:0 \
    --targets contact_order --modes baseline --lambdas 4 32 --seeds 48 49 50 51 52 53 54 55 56 57 >/dev/null 2>&1 || true
for w in 4 32; do
  CUDA_VISIBLE_DEVICES=$GPU_GEOM OUT_BASE=$ROOT "$PY" script_utils/run_scrmsd_steering.py \
      --cfgs "contact_order_baseline_w${w}" --seeds 48 49 50 51 52 53 54 55 56 57 --lengths $LENGTHS || true
done

echo "[$(date -u +%FT%TZ)] === PERRES_FRONTIER_DONE ==="
