#!/bin/bash
# E100 — SDE-jitter sequence-entropy probe.
#
# Question: is the low Shannon entropy of unsteered La-Proteina sequences
# (3.47 bits vs AFDB-natural 4.05, E099) a mode-collapse artifact of the
# LOW latent-channel sampling temperature? The production baseline
# (inference_ucond_notri_long) runs local_latents at sc_scale_noise=0.05
# (HALF the inference_base default of 0.10) while bb_ca runs at 0.15.
# The sequence head reads local_latents, so a low latent temperature is the
# prime suspect for collapsed / low-complexity sequences.
#
# Design: 3 conditions, sweeping ONLY local_latents sc_scale_noise
#   {0.05 (baseline reproduce), 0.15, 0.30}, bb_ca fixed at 0.15.
# Matched (seed, length): identical seed_base + stratified bins -> the
# length-draw RNG and seed counter are identical across conditions, so the
# three dirs are paired by (seed, length); only the latent jitter differs.
# Smaller N than the 1000-panel: n_per_bin=4 over 5 bins of width 100 in
# [300,800) = 20 proteins/condition = 60 generations total. nsteps=400.
#
# Single GPU (cuda:0, the UUID-pinned device). Run under nohup.

cd /home/ks2218/la-proteina
source /opt/conda/etc/profile.d/conda.sh
conda activate laproteina_env
# conda's MKL activate scripts reference unbound vars; enable -u only after activation
set -uo pipefail

export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

ROOT=results/sde_jitter_entropy_probe
mkdir -p "$ROOT"

SEED_BASE=2000
NPB=4
BINW=100
LRANGE="300 800"
NSTEPS=400

echo "[$(date)] E100 SDE-jitter entropy probe START"

for LOCAL in 0.05 0.15 0.30; do
    OUT="$ROOT/local${LOCAL}"
    echo ""
    echo "=========================================================="
    echo "[$(date)] Condition local_latents sc_scale_noise=$LOCAL (bb_ca=0.15)"
    echo "  output: $OUT"
    echo "=========================================================="
    python -m steering.generate_baseline \
        --proteina_config inference_ucond_notri_long \
        --length_mode stratified \
        --bin_width $BINW \
        --length_range $LRANGE \
        --n_per_bin $NPB \
        --seed_base $SEED_BASE \
        --sc_scale_noise_local $LOCAL \
        --sc_scale_noise_bb 0.15 \
        --output_dir "$OUT" \
        --device cuda:0 \
        --nsteps $NSTEPS
    echo "[$(date)] Condition $LOCAL done. .pt count: $(find $OUT/samples -name '*.pt' 2>/dev/null | wc -l)"
done

echo ""
echo "[$(date)] Generation complete. Running sequence-diversity analysis..."
python script_utils/analyze_sde_jitter_entropy.py --root "$ROOT" \
    | tee "$ROOT/entropy_summary.txt"

echo "[$(date)] E100 probe COMPLETE"
