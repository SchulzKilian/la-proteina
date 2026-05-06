#!/bin/bash
# Regenerate the 1000-protein stratified unguided control at the canonical
# nsteps=400 (the original 1000 was at nsteps=200, which was wrong for any
# downstream scRMSD / structure-aware analysis). Then immediately re-run every
# distribution experiment that originally used the 1000-sample stratified set.
#
# Phases:
#   1. Generation              — 1000 .pt + .pdb samples, ~18 h on cuda:0
#   2. Property panel eval     — full 16-col developability panel per sample
#   3. Build FASTA             — for AA-composition & thermal-stability scripts
#   4. E020-A vs PDB           — property-panel comparison
#   5. E020-A vs AFDB (E026)
#   6. E020-B vs PDB           — AA-composition
#   7. E020-B vs AFDB (E026)
#   8. E020-C vs PDB           — thermal-stability sequence proxies (Tier 1)
#   9. E020-C vs AFDB (E026)
#
# Single GPU on cuda:0. Phases 4-9 are CPU.
# Output root: results/generated_stratified_300_800_nsteps400/ (and sibling
# results/property_comparison_nsteps400/, results/aa_composition_nsteps400/,
# results/thermal_stability_nsteps400/).
#
# Resume-safe: steering.generate_baseline picks up where it left off if the
# samples/ dir already exists; per-experiment outputs overwrite.

set -o pipefail
cd /home/ks2218/la-proteina
source /opt/conda/etc/profile.d/conda.sh
conda activate laproteina_env
set -u

export TANGO_EXE=/home/ks2218/la-proteina/tango_x86_64_release

GEN_DIR=results/generated_stratified_300_800_nsteps400
PROPS_CSV=$GEN_DIR/properties_generated.csv
GEN_FASTA=$GEN_DIR/sequences.fasta

PDB_PROPS_CSV=laproteina_steerability/data/properties.csv
AFDB_PROPS_CSV=data/afdb_ref/properties_afdb_refschema.csv
PDB_SEQS_FASTA=pdb_cluster_all_seqs.fasta
AFDB_SEQS_FASTA=data/afdb_ref/sequences.fasta

START=$(date +%s)
echo "[$(date)] PIPELINE START — output root: $GEN_DIR"

# ----------------------------------------------------------------------
# Phase 1: Generation (1000 samples, nsteps=400, length-stratified)
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 1/9 — Generate 1000 samples at nsteps=400"
echo "=========================================================="
mkdir -p "$GEN_DIR"
python -m steering.generate_baseline \
    --proteina_config inference_ucond_notri_long \
    --length_mode stratified \
    --bin_width 50 \
    --length_range 300 800 \
    --n_per_bin 100 \
    --seed_base 1000 \
    --output_dir "$GEN_DIR" \
    --device cuda:0 \
    --nsteps 400

echo "[$(date)] Phase 1 done. .pt count: $(find $GEN_DIR/samples -name '*.pt' 2>/dev/null | wc -l)"

# ----------------------------------------------------------------------
# Phase 2: Property panel eval (full 16-col panel — TANGO, FreeSASA, etc.)
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 2/9 — Property panel eval (compute_developability)"
echo "=========================================================="
python -m steering.evaluate_samples_dir \
    --samples_dir "$GEN_DIR/samples" \
    --output_csv "$PROPS_CSV"
echo "[$(date)] Phase 2 done. CSV rows: $(($(wc -l < $PROPS_CSV) - 1))"

# ----------------------------------------------------------------------
# Phase 3: Build sequences.fasta from the .pt files
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 3/9 — Build FASTA from .pt files"
echo "=========================================================="
python << EOF
import torch, glob
from pathlib import Path
files = sorted(glob.glob("$GEN_DIR/samples/*.pt"))
with open("$GEN_FASTA", "w") as fh:
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        fh.write(f">{d['id']} length={len(d['sequence'])}\n{d['sequence']}\n")
print(f"Wrote {len(files)} sequences to $GEN_FASTA")
EOF
echo "[$(date)] Phase 3 done."

# ----------------------------------------------------------------------
# Phase 4: E020-A — property panel vs PDB
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 4/9 — E020-A property panel vs PDB"
echo "=========================================================="
python proteinfoundation/analysis/compare_properties.py \
    --ref "$PDB_PROPS_CSV" \
    --gen "$PROPS_CSV" \
    --out results/property_comparison_nsteps400/stratified_vs_pdb \
    --ref-label "train (PDB)" \
    --gen-label "generated nsteps=400" \
    --length-min 300 --length-max 800
echo "[$(date)] Phase 4 done."

# ----------------------------------------------------------------------
# Phase 5: E026 — property panel vs AFDB
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 5/9 — E026 property panel vs AFDB"
echo "=========================================================="
python proteinfoundation/analysis/compare_properties.py \
    --ref "$AFDB_PROPS_CSV" \
    --gen "$PROPS_CSV" \
    --out results/property_comparison_nsteps400/stratified_vs_afdb \
    --ref-label "AFDB" \
    --gen-label "generated nsteps=400" \
    --length-min 300 --length-max 800
echo "[$(date)] Phase 5 done."

# ----------------------------------------------------------------------
# Phase 6: E020-B — AA composition vs PDB
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 6/9 — E020-B AA composition vs PDB"
echo "=========================================================="
python proteinfoundation/analysis/aa_composition.py \
    --gen "$GEN_FASTA" \
    --ref "$PDB_SEQS_FASTA" \
    --length-min 300 --length-max 800 \
    --out results/aa_composition_nsteps400/stratified_vs_pdb
echo "[$(date)] Phase 6 done."

# ----------------------------------------------------------------------
# Phase 7: E026 — AA composition vs AFDB
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 7/9 — E026 AA composition vs AFDB"
echo "=========================================================="
python proteinfoundation/analysis/aa_composition.py \
    --gen "$GEN_FASTA" \
    --ref "$AFDB_SEQS_FASTA" \
    --length-min 300 --length-max 800 \
    --out results/aa_composition_nsteps400/stratified_vs_afdb
echo "[$(date)] Phase 7 done."

# ----------------------------------------------------------------------
# Phase 8: E020-C — thermal stability Tier-1 proxies vs PDB
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 8/9 — E020-C thermal stability vs PDB (Tier 1)"
echo "=========================================================="
python proteinfoundation/analysis/thermal_stability.py \
    --gen "$GEN_FASTA" \
    --ref "$PDB_SEQS_FASTA" \
    --gen-label "generated nsteps=400" \
    --ref-label "PDB (300-800)" \
    --length-min 300 --length-max 800 \
    --out results/thermal_stability_nsteps400/stratified_vs_pdb
echo "[$(date)] Phase 8 done."

# ----------------------------------------------------------------------
# Phase 9: E026 — thermal stability Tier-1 proxies vs AFDB
# ----------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[$(date)] PHASE 9/9 — E026 thermal stability vs AFDB (Tier 1)"
echo "=========================================================="
python proteinfoundation/analysis/thermal_stability.py \
    --gen "$GEN_FASTA" \
    --ref "$AFDB_SEQS_FASTA" \
    --gen-label "generated nsteps=400" \
    --ref-label "AFDB (300-800)" \
    --length-min 300 --length-max 800 \
    --out results/thermal_stability_nsteps400/stratified_vs_afdb
echo "[$(date)] Phase 9 done."

END=$(date +%s)
echo ""
echo "[$(date)] PIPELINE DONE. Total wall: $(( (END-START)/60 )) min."
echo "Outputs:"
echo "  $GEN_DIR/                                    — 1000 samples + properties_generated.csv + sequences.fasta + manifest.csv"
echo "  results/property_comparison_nsteps400/       — E020-A vs PDB / vs AFDB"
echo "  results/aa_composition_nsteps400/            — E020-B vs PDB / vs AFDB"
echo "  results/thermal_stability_nsteps400/         — E020-C vs PDB / vs AFDB (Tier 1 only; TemStaPro Tier 2 needs separate sbatch)"
