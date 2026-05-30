#!/bin/bash
# E100 follow-on: full developability panel on the SDE-jitter probe samples.
# Waits for the generation job to finish (COMPLETE marker in the gen log),
# then runs the 20-column compute_developability panel per condition and a
# property-comparison analysis (per-condition means + length-matched gap vs
# AFDB-natural). Complements the cheap sequence-diversity analysis.
#
# Single GPU not required (panel is CPU: TANGO/FreeSASA/IUPred). Run under nohup.

cd /home/ks2218/la-proteina
source /opt/conda/etc/profile.d/conda.sh
conda activate laproteina_env
set -uo pipefail

export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release
export IUPRED3_DIR=/home/ks2218/iupred3

ROOT=results/sde_jitter_entropy_probe
GENLOG=results_sde_jitter_probe.log

# Wait for generation to finish so the panel sees all samples.
echo "[$(date)] Panel job: waiting for generation COMPLETE marker in $GENLOG ..."
for i in $(seq 1 480); do
    if grep -q "E100 probe COMPLETE" "$GENLOG" 2>/dev/null; then
        echo "[$(date)] Generation complete; starting panel."
        break
    fi
    sleep 30
done

for LOCAL in 0.05 0.15 0.30; do
    OUT="$ROOT/local${LOCAL}"
    echo ""
    echo "[$(date)] Developability panel — local${LOCAL}"
    python -m steering.evaluate_samples_dir \
        --samples_dir "$OUT/samples" \
        --output_csv  "$OUT/properties_generated.csv"
    echo "[$(date)] local${LOCAL} panel rows: $(($(wc -l < $OUT/properties_generated.csv 2>/dev/null || echo 1) - 1))"
done

echo ""
echo "[$(date)] Running property comparison across jitter levels ..."
python script_utils/analyze_sde_jitter_panel.py --root "$ROOT" \
    | tee "$ROOT/panel_summary.txt"

echo "[$(date)] E100 developability panel COMPLETE"
