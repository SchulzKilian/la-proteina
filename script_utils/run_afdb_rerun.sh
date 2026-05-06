#!/usr/bin/env bash
# script_utils/run_afdb_rerun.sh
#
# AFDB-as-reference rerun: redoes the experiments whose conclusions
# depended on a PDB-derived reference but should have been compared
# against AFDB (since La Proteina was trained on AFDB).
#
# Necessary set:
#   E020-A  developability panel  (compute_developability + compare_properties)
#   E020-B  per-AA composition    (aa_composition.py)
#   E020-C  thermal Tier-1 proxies (thermal_stability.py, no TemStaPro)
#   E023    aromatic burial       (aromatic_burial.py)
#   E024    aromatic burial follow-ups (aromatic_burial_followups.py)
#
# Skipped (non-essential / GPU-heavy):
#   E020-D  TemStaPro Tier-2b
#   E005    cheap-diagnostics (was non-narrative anyway)
#
# Each step writes a stamp file under data/afdb_ref/stamps/ so you can
# detach the tmux session, re-attach, and re-run this script — completed
# steps are skipped automatically.
#
# Usage:
#   tmux new -s afdb_rerun \
#       'bash script_utils/run_afdb_rerun.sh 2>&1 | tee -a logs/afdb_rerun.log'
#
#   detach:    Ctrl-b d
#   reattach:  tmux attach -t afdb_rerun
#   resume:    re-run the same command — completed stamps are skipped

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

# ── Config ──────────────────────────────────────────────────────────────
PYBIN=${PYBIN:-/home/ks2218/.conda/envs/laproteina_env/bin/python}
LEN_MIN=${LEN_MIN:-300}
LEN_MAX=${LEN_MAX:-800}
N_FINAL=${N_FINAL:-5000}        # final per-bin × n_bins ≈ this; length-stratified
PER_BIN=${PER_BIN:-1000}         # initial reservoir size per 50-residue bin (oversample for download failures)
N_PAR=${N_PAR:-16}                # parallel curl downloads
SEED=${SEED:-42}

AFDB_DIR=$REPO/data/afdb_ref
RAW_DIR=$AFDB_DIR/raw_pdb
PT_DIR=$AFDB_DIR/processed
FINAL_PDB_DIR=$AFDB_DIR/structures_final   # symlinks to final 5000 PDBs (for aromatic_burial)
FASTA=$AFDB_DIR/sequences.fasta
PROPERTIES_GEN_SCHEMA_CSV=$AFDB_DIR/properties_afdb.csv
PROPERTIES_REF_SCHEMA_CSV=$AFDB_DIR/properties_afdb_refschema.csv
ACC_CSV=$AFDB_DIR/accession_ids.csv         # full AFDB accession + length list (~8.7 GB)
ID_LIST=$AFDB_DIR/accessions.txt
LOG_DIR=$REPO/logs
STAMP=$AFDB_DIR/stamps

GEN_FASTA=$REPO/results/generated_stratified_300_800/sequences.fasta
GEN_PROPS_SEQONLY=$REPO/results/property_comparison/stratified_vs_pdb/properties_generated_seqonly.csv
GEN_PDB_DIR=$REPO/inference/inference_ucond_notri

OUT_PROP_CMP=$REPO/results/property_comparison_afdb/stratified_vs_afdb
OUT_THERM=$REPO/results/thermal_stability_afdb/stratified_vs_afdb
OUT_AROM=$REPO/results/aromatic_burial_afdb

# TANGO / IUPred3 (E020-A column requirements)
export TANGO_EXE=${TANGO_EXE:-$REPO/tango_x86_64_release}
export IUPRED3_DIR=${IUPRED3_DIR:-/home/ks2218/iupred3}
export DATA_PATH=$AFDB_DIR
# Vendored openfold lives at repo root — add to PYTHONPATH for all helper calls.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$AFDB_DIR" "$RAW_DIR" "$PT_DIR" "$FINAL_PDB_DIR" "$LOG_DIR" "$STAMP" \
         "$OUT_PROP_CMP" "$OUT_THERM" "$OUT_AROM"

stamp() { touch "$STAMP/$1"; echo "[$(date +%F\ %T)] [stamp] $1"; }
have()  { [[ -f "$STAMP/$1" ]]; }
banner() { echo; echo "=== [$(date +%F\ %T)] $* ==="; }

banner "Config: AFDB rerun — N_FINAL=$N_FINAL  PER_BIN=$PER_BIN  range=[$LEN_MIN,$LEN_MAX]  pyenv=$PYBIN"

# Sanity: required gen-side artifacts must be present.
for f in "$GEN_FASTA" "$GEN_PROPS_SEQONLY"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required gen-side file: $f"
        exit 1
    fi
done
if [[ ! -d "$GEN_PDB_DIR" ]]; then
    echo "ERROR: missing $GEN_PDB_DIR"
    exit 1
fi
if [[ ! -x "$TANGO_EXE" ]]; then
    echo "WARN: TANGO_EXE=$TANGO_EXE not executable — TANGO column will be NaN"
fi

# ── 1. Download AFDB accession_ids.csv (full AFDB list, ~8.7 GB) ────
# Format per row: accession,first_residue,last_residue,model_id,version
# (~214M rows). One-time download to local file so step 2's reservoir
# sample is replayable cheaply if you re-run with a different range or N.
if ! have 01_acc; then
    banner "[1/8] Download AFDB accession_ids.csv (~8.7 GB)"
    curl -fL --retry 5 --max-time 7200 \
        -o "$ACC_CSV" \
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/accession_ids.csv"
    [[ -s "$ACC_CSV" ]] || { echo "FAIL: empty AFDB accession CSV"; exit 1; }
    sz=$(du -h "$ACC_CSV" | cut -f1)
    echo "  downloaded $sz to $ACC_CSV"
    stamp 01_acc
fi

# ── 2. Random length-stratified sample from full AFDB ───────────────
# Reservoir-sample PER_BIN ids per 50-residue bin in [LMIN, LMAX].
# Random within each bin → uniform-over-AFDB approximation, length-balanced
# to match the gen distribution.
if ! have 02_sample; then
    banner "[2/8] Random-sample $((PER_BIN * (LEN_MAX-LEN_MIN)/50)) AFDB accessions ($PER_BIN per 50-residue bin)"
    "$PYBIN" "$REPO/script_utils/afdb_rerun_helpers.py" sample-afdb \
        --src "$ACC_CSV" \
        --out-list "$ID_LIST" \
        --length-min "$LEN_MIN" --length-max "$LEN_MAX" \
        --bin-width 50 --per-bin "$PER_BIN" --seed "$SEED"
    stamp 02_sample
fi

# ── 3. Parallel-download AFDB v4 PDBs ────────────────────────────────
if ! have 03_download; then
    banner "[3/8] Downloading AFDB PDBs ($N_PAR-way parallel)"
    n_total=$(wc -l < "$ID_LIST")
    echo "  target: $n_total accessions"
    # AFDB current bulk version is v6 (May 2026 — same AF2-monomer model).
    # Try v6 first, fall back to v5 / v4 in case the entry is stuck on an
    # older release. Output filename always normalised to model_v4.pdb so
    # the rest of the pipeline is version-agnostic.
    cat "$ID_LIST" | xargs -n1 -P "$N_PAR" -I {} bash -c '
        acc="{}"
        out="'"$RAW_DIR"'/AF-${acc}-F1-model_v4.pdb"
        [[ -s "$out" ]] && exit 0
        for v in 6 5 4; do
            if curl -fsSL --retry 2 --retry-delay 1 --max-time 60 \
                "https://alphafold.ebi.ac.uk/files/AF-${acc}-F1-model_v${v}.pdb" \
                -o "$out" 2>/dev/null; then
                [[ -s "$out" ]] && exit 0
            fi
        done
        rm -f "$out"
    '
    n_dl=$(find "$RAW_DIR" -maxdepth 1 -type f -name "AF-*.pdb" | wc -l)
    echo "  downloaded: $n_dl PDBs in $RAW_DIR"
    stamp 03_download
fi

# ── 4. Convert to .pt + extract FASTA + length-stratify to N_FINAL ──
if ! have 04_convert; then
    banner "[4/8] Convert AFDB PDBs → .pt and length-stratify to N=$N_FINAL"
    "$PYBIN" "$REPO/script_utils/afdb_rerun_helpers.py" convert \
        --raw-dir "$RAW_DIR" \
        --id-list "$ID_LIST" \
        --out-pt-dir "$PT_DIR" \
        --out-fasta "$FASTA" \
        --workers "$(( $(nproc) - 2 ))" \
        --n-final "$N_FINAL" \
        --length-min "$LEN_MIN" --bin-width 50 \
        --seed "$SEED"
    stamp 04_convert
fi

# ── 5. Symlink final 5000 PDBs to a clean dir for aromatic_burial ───
if ! have 05_link; then
    banner "[5/8] Symlinking final $N_FINAL PDBs to $FINAL_PDB_DIR"
    rm -rf "$FINAL_PDB_DIR"
    mkdir -p "$FINAL_PDB_DIR"
    n_link=0
    while IFS= read -r line; do
        [[ "$line" == \>* ]] || continue
        acc=${line#>}
        src=$RAW_DIR/AF-${acc}-F1-model_v4.pdb
        if [[ -f "$src" ]]; then
            ln -sf "$src" "$FINAL_PDB_DIR/AF-${acc}-F1-model_v4.pdb"
            n_link=$((n_link+1))
        fi
    done < "$FASTA"
    echo "  symlinked $n_link PDBs to $FINAL_PDB_DIR"
    stamp 05_link
fi

# ── 6. Developability panel  (E020-A on AFDB ref) ────────────────────
# This is the heaviest step — TANGO + FreeSASA + IUPred3 across $N_FINAL.
if ! have 06_dev; then
    banner "[6/8] Developability panel on AFDB (TANGO + FreeSASA + IUPred3)"
    workers=$(( $(nproc) - 2 ))
    "$PYBIN" "$REPO/proteinfoundation/analysis/compute_developability.py" \
        "$PROPERTIES_GEN_SCHEMA_CSV" \
        --data-dir "$AFDB_DIR" \
        --workers "$workers" \
        --min-length "$LEN_MIN" --max-length "$LEN_MAX"
    [[ -s "$PROPERTIES_GEN_SCHEMA_CSV" ]] || { echo "FAIL: empty properties CSV"; exit 1; }
    "$PYBIN" "$REPO/script_utils/afdb_rerun_helpers.py" rename-csv \
        --in-csv "$PROPERTIES_GEN_SCHEMA_CSV" \
        --out-csv "$PROPERTIES_REF_SCHEMA_CSV"
    stamp 06_dev
fi

# ── 7. Property panel comparison + AA composition + Thermal Tier-1 ──
if ! have 07_seqcmp; then
    banner "[7/8] property_comparison + aa_composition + thermal Tier-1 (AFDB ref)"

    # Property panel: gen seqonly CSV vs AFDB full panel.
    "$PYBIN" "$REPO/proteinfoundation/analysis/compare_properties.py" \
        --ref "$PROPERTIES_REF_SCHEMA_CSV" \
        --gen "$GEN_PROPS_SEQONLY" \
        --out "$OUT_PROP_CMP" \
        --ref-label "AFDB (300-800)" --gen-label "generated"

    # Per-AA composition: gen FASTA vs AFDB FASTA.
    "$PYBIN" "$REPO/proteinfoundation/analysis/aa_composition.py" \
        --gen "$GEN_FASTA" --ref "$FASTA" \
        --length-min "$LEN_MIN" --length-max "$LEN_MAX" \
        --out "$OUT_PROP_CMP/aa_composition.csv"

    # Thermal Tier-1 proxies (no TemStaPro).
    "$PYBIN" "$REPO/proteinfoundation/analysis/thermal_stability.py" \
        --gen "$GEN_FASTA" --ref "$FASTA" \
        --out "$OUT_THERM" \
        --gen-label "generated" --ref-label "AFDB ($LEN_MIN-$LEN_MAX)" \
        --length-min "$LEN_MIN" --length-max "$LEN_MAX"

    stamp 07_seqcmp
fi

# ── 8. Aromatic burial + follow-ups (E023 + E024 with AFDB ref) ─────
if ! have 08_arom; then
    banner "[8/8] Aromatic burial + follow-ups (AFDB ref)"
    "$PYBIN" "$REPO/proteinfoundation/analysis/aromatic_burial.py" \
        --gen-dir "$GEN_PDB_DIR" \
        --ref-dir "$FINAL_PDB_DIR" \
        --out-dir "$OUT_AROM" \
        --n-ref-sample 1000 \
        --seed "$SEED"

    if [[ -f "$OUT_AROM/per_residue.parquet" ]]; then
        "$PYBIN" "$REPO/proteinfoundation/analysis/aromatic_burial_followups.py" \
            --in-file "$OUT_AROM/per_residue.parquet" \
            --out-dir "$OUT_AROM/followups" \
            --seed "$SEED"
    else
        echo "WARN: $OUT_AROM/per_residue.parquet not present — skipping followups"
    fi
    stamp 08_arom
fi

banner "DONE — AFDB rerun complete"
echo
echo "Outputs:"
echo "  AFDB FASTA:             $FASTA"
echo "  AFDB structures (link): $FINAL_PDB_DIR"
echo "  AFDB properties (gen):  $PROPERTIES_GEN_SCHEMA_CSV"
echo "  AFDB properties (ref):  $PROPERTIES_REF_SCHEMA_CSV"
echo "  E020-A panel cmp:       $OUT_PROP_CMP/summary.csv"
echo "  E020-B AA composition:  $OUT_PROP_CMP/aa_composition.csv"
echo "  E020-C thermal Tier-1:  $OUT_THERM/summary.csv"
echo "  E023 aromatic burial:   $OUT_AROM/aromatic_frequencies.csv"
echo "  E024 followups:         $OUT_AROM/followups/results.md"
echo
echo "Now fill in numerical results under E026 in experiments.md."
