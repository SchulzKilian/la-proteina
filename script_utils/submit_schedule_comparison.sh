#!/bin/bash
#SBATCH -J schedule_cmp
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm_schedule_cmp_%j.out

# submit_schedule_comparison.sh
#
# Runs generation + evaluation for each schedule in a configurable list,
# then prints a side-by-side designability comparison.
#
# Usage:
#   sbatch script_utils/submit_schedule_comparison.sh --config inference_ucond_ca_only_70M --ckpt /path/to/last.ckpt
#   bash   script_utils/submit_schedule_comparison.sh --config inference_ucond_ca_only_70M --ckpt /path/to/last.ckpt \
#              --schedules baseline,power_bump_e0.1
#
# Flags:
#   --config    <name>   Hydra config name (without .yaml). Default: inference_ucond_ca_only_70M
#   --ckpt      <path>   Full path to checkpoint file. Required.
#   --ae_ckpt   <path>   Full path to AE checkpoint (full-latent models only).
#   --nsamples  <n>      Override nsamples in config.
#   --schedules <list>   Comma-separated schedule keys to compare. Default: baseline,power_bump_e0.1
#
# Available schedule keys:
#   baseline          — bb_ca log(p=2.0)  | local_latents power(p=2.0)           [config defaults]
#   power_0.5         — bb_ca power(p=0.5)| local_latents power(p=0.5)           [separate experiment]
#   power_bump_e0.05  — bb_ca log(p=2.0)  | local_latents power_bump(p=2.0, eps=0.05, mu=0.489)
#   power_bump_e0.1   — bb_ca log(p=2.0)  | local_latents power_bump(p=2.0, eps=0.1,  mu=0.489)
#   power_bump_e0.14  — bb_ca log(p=2.0)  | local_latents power_bump(p=2.0, eps=0.14, mu=0.489) [max monotone]
#   power_bump_e0.0   — bb_ca log(p=2.0)  | local_latents power_bump(p=2.0, eps=0.0)  [= baseline, sanity check]

source "$HOME/.bashrc"
conda activate laproteina_env

set -euo pipefail

cd "$HOME/la-proteina"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG_NAME="inference_ucond_ca_only_70M"
CKPT_PATH=""
AE_CKPT_PATH=""
NSAMPLES=""
SCHEDULES="baseline,power_bump_e0.1"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)     CONFIG_NAME="$2";   shift 2 ;;
        --ckpt)       CKPT_PATH="$2";     shift 2 ;;
        --ae_ckpt)    AE_CKPT_PATH="$2";  shift 2 ;;
        --nsamples)   NSAMPLES="$2";      shift 2 ;;
        --schedules)  SCHEDULES="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: --ckpt <path> is required."
    exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

CKPT_DIR=$(dirname "$CKPT_PATH")
CKPT_NAME=$(basename "$CKPT_PATH")

AE_OVERRIDE=""
if [[ -n "$AE_CKPT_PATH" ]]; then
    AE_OVERRIDE="++autoencoder_ckpt_path=$AE_CKPT_PATH"
fi

NSAMPLES_OVERRIDE=""
if [[ -n "$NSAMPLES" ]]; then
    NSAMPLES_OVERRIDE="++generation.dataset.nsamples=$NSAMPLES"
fi

# ── Schedule registry ─────────────────────────────────────────────────────────
# get_sched_label <key>  → human-readable label
get_sched_label() {
    case "$1" in
        baseline)
            echo "BASELINE  [bb_ca log p=2.0 | local_latents power p=2.0]"
            ;;
        power_0.5)
            echo "POWER_0.5  [bb_ca power p=0.5 | local_latents power p=0.5]"
            ;;
        power_bump_e0.05)
            echo "POWER_BUMP_EPS0.05  [bb_ca log p=2.0 | local_latents power_bump p=2.0 eps=0.05 mu=0.489]"
            ;;
        power_bump_e0.1)
            echo "POWER_BUMP_EPS0.1  [bb_ca log p=2.0 | local_latents power_bump p=2.0 eps=0.1 mu=0.489]"
            ;;
        power_bump_e0.14)
            echo "POWER_BUMP_EPS0.14  [bb_ca log p=2.0 | local_latents power_bump p=2.0 eps=0.14 mu=0.489]"
            ;;
        power_bump_e0.0)
            echo "POWER_BUMP_EPS0.0  [bb_ca log p=2.0 | local_latents power_bump p=2.0 eps=0.0 = baseline]"
            ;;
        *)
            echo "UNKNOWN[$1]"
            ;;
    esac
}

# get_sched_overrides <key>  → one Hydra override per line (empty = use config defaults)
get_sched_overrides() {
    case "$1" in
        baseline)
            # No overrides — use whatever the config specifies
            ;;
        power_0.5)
            echo "++generation.model.bb_ca.schedule.mode=power"
            echo "++generation.model.bb_ca.schedule.p=0.5"
            echo "++generation.model.local_latents.schedule.mode=power"
            echo "++generation.model.local_latents.schedule.p=0.5"
            ;;
        power_bump_e0.05)
            echo "++generation.model.local_latents.schedule.mode=power_with_middle_bump"
            echo "++generation.model.local_latents.schedule.p=2.0"
            echo "++generation.model.local_latents.schedule.eps=0.05"
            echo "++generation.model.local_latents.schedule.mu=0.4890"
            echo "++generation.model.local_latents.schedule.sigma=0.08"
            ;;
        power_bump_e0.1)
            # Clean ablation: only the bump changes vs baseline; p=2.0 matches baseline
            echo "++generation.model.local_latents.schedule.mode=power_with_middle_bump"
            echo "++generation.model.local_latents.schedule.p=2.0"
            echo "++generation.model.local_latents.schedule.eps=0.1"
            echo "++generation.model.local_latents.schedule.mu=0.4890"
            echo "++generation.model.local_latents.schedule.sigma=0.08"
            ;;
        power_bump_e0.14)
            # Max monotone eps with sigma=0.08, mu=0.4890
            echo "++generation.model.local_latents.schedule.mode=power_with_middle_bump"
            echo "++generation.model.local_latents.schedule.p=2.0"
            echo "++generation.model.local_latents.schedule.eps=0.14"
            echo "++generation.model.local_latents.schedule.mu=0.4890"
            echo "++generation.model.local_latents.schedule.sigma=0.08"
            ;;
        power_bump_e0.0)
            # Sanity check: eps=0 + p=2.0 should reproduce baseline local_latents exactly
            echo "++generation.model.local_latents.schedule.mode=power_with_middle_bump"
            echo "++generation.model.local_latents.schedule.p=2.0"
            echo "++generation.model.local_latents.schedule.eps=0.0"
            echo "++generation.model.local_latents.schedule.mu=0.4890"
            echo "++generation.model.local_latents.schedule.sigma=0.08"
            ;;
        *)
            echo "ERROR: Unknown schedule key: $1" >&2
            exit 1
            ;;
    esac
}

# ── Helper ────────────────────────────────────────────────────────────────────
run_variant() {
    local label="$1"
    local save_csv="$2"
    local sched_key="$3"

    local infer_dir="inference/${CONFIG_NAME}"
    local csv_src="inference/results_${CONFIG_NAME}_0.csv"

    # Build overrides array from registry
    local overrides=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && overrides+=("$line")
    done < <(get_sched_overrides "$sched_key")

    echo ""
    echo "── $label ─────────────────────────────────────────────────────"

    rm -rf "$infer_dir"
    rm -f  "$csv_src"

    echo "Generating..."
    local gen_cmd=(
        python proteinfoundation/generate.py
        --config-name "$CONFIG_NAME"
        ++config_name="$CONFIG_NAME"
        ++ckpt_path="$CKPT_DIR"
        ++ckpt_name="$CKPT_NAME"
    )
    [[ -n "$AE_OVERRIDE" ]]       && gen_cmd+=("$AE_OVERRIDE")
    [[ -n "$NSAMPLES_OVERRIDE" ]] && gen_cmd+=("$NSAMPLES_OVERRIDE")
    [[ ${#overrides[@]} -gt 0 ]]  && gen_cmd+=("${overrides[@]}")
    "${gen_cmd[@]}"

    echo "Evaluating..."
    python proteinfoundation/evaluate.py \
        --config_name "$CONFIG_NAME"

    if [[ -f "$csv_src" ]]; then
        cp "$csv_src" "$save_csv"
        echo "Results saved to $save_csv"
    else
        echo "WARNING: expected CSV not found at $csv_src"
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo "Schedule comparison"
echo "Config     : $CONFIG_NAME"
echo "Checkpoint : $CKPT_PATH"
echo "Schedules  : $SCHEDULES"
echo "================================================================"

IFS=',' read -ra SCHED_LIST <<< "$SCHEDULES"

declare -a CSV_FILES=()
declare -a REPORT_ARGS=()

for sched_key in "${SCHED_LIST[@]}"; do
    label=$(get_sched_label "$sched_key")
    csv_path="inference/results_${CONFIG_NAME}_${sched_key}_0.csv"
    run_variant "$label" "$csv_path" "$sched_key"
    CSV_FILES+=("$csv_path")
    # Pass as "LABEL:path" so the report script can show meaningful names
    REPORT_ARGS+=("${label}:${csv_path}")
done

# ── Summary comparison ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "RESULTS COMPARISON"
echo "================================================================"
python script_utils/schedule_comparison_report.py "${REPORT_ARGS[@]}"

echo ""
echo "Raw CSVs:"
for f in "${CSV_FILES[@]}"; do
    echo "  $f"
done
echo "Done."
