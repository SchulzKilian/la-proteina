#!/bin/bash
# Self-healing launcher for the dynamic-K training chain (E103-E105 spin-hang).
#
# The dynamic-K branch has an INTERMITTENT GPU-side spin-hang: a slot may run
# clean for many minutes, or freeze within a few minutes of resume (100% gpu-util
# / ~4% mem-util / ~90W, no progress). Because the dataloader reshuffles on every
# resume, simply resubmitting almost always dodges the race and makes progress.
#
# This loop: submit -> watch for the spin fingerprint -> on wedge, scancel +
# resubmit; on a healthy wall-clock TIMEOUT, resubmit the next chain slot. Stops
# on MAX_SLOTS, on an unexpected job state, or if it wedges MAX_CONSEC_WEDGE
# times in a row WITHOUT a healthy slot in between (meaning the race has become
# deterministic again and a human should look).
set -uo pipefail

CONFIG=training_ca_only_sparse_dynamicK
MAX_SLOTS=${MAX_SLOTS:-10}
MAX_CONSEC_WEDGE=${MAX_CONSEC_WEDGE:-5}
GRACE=${GRACE:-150}; POLL=${POLL:-30}; WINDOW=${WINDOW:-6}
POWER_MAX=170; MEMUTIL_MAX=10; GPUUTIL_MIN=80
HEAL=/home/ks2218/la-proteina/dynk_autoheal.log

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$HEAL"; }

log "=== autoheal start: config=$CONFIG MAX_SLOTS=$MAX_SLOTS MAX_CONSEC_WEDGE=$MAX_CONSEC_WEDGE ==="
slot=0; consec_wedge=0
while [ "$slot" -lt "$MAX_SLOTS" ]; do
    slot=$((slot+1))
    JOB=$(sbatch --parsable --exclude=gpu-q-43 \
          script_utils/submit_train_ca_only_1gpu.sh -n "$CONFIG" 2>>"$HEAL")
    if ! [[ "$JOB" =~ ^[0-9]+$ ]]; then log "sbatch failed (got '$JOB'); stopping."; exit 1; fi
    log "slot $slot/$MAX_SLOTS: submitted job $JOB"

    # wait for RUNNING
    NODE=""
    while :; do
        st=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
        [ -z "$st" ] && { log "job $JOB vanished before RUNNING; stopping."; exit 1; }
        if [ "$st" = "RUNNING" ]; then NODE=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); log "job $JOB RUNNING on $NODE"; break; fi
        sleep 20
    done
    sleep "$GRACE"

    bad=0; outcome=""
    while :; do
        st=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
        if [ "$st" != "RUNNING" ]; then
            fin=$(sacct -j "$JOB" -n -o State 2>/dev/null | head -1 | tr -d ' ')
            outcome="ended:${fin:-gone}"; break
        fi
        n=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); [ -n "$n" ] && NODE="$n"
        reading=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits" 2>/dev/null | head -1)
        if [ -z "$reading" ]; then sleep "$POLL"; continue; fi
        g=$(echo "$reading"|awk -F, '{gsub(/ /,"",$1);print $1}')
        m=$(echo "$reading"|awk -F, '{gsub(/ /,"",$2);print $2}')
        p=$(echo "$reading"|awk -F, '{gsub(/ /,"",$3);print $3}')
        w=$(awk -v g="$g" -v m="$m" -v p="$p" -v gm="$GPUUTIL_MIN" -v mm="$MEMUTIL_MAX" -v pm="$POWER_MAX" \
            'BEGIN{print (g>=gm && m<=mm && p<pm)?1:0}')
        if [ "$w" = "1" ]; then bad=$((bad+1)); else bad=0; fi
        if [ "$bad" -ge "$WINDOW" ]; then
            log "slot $slot: WEDGE on $JOB (gpu=${g}% mem=${m}% pow=${p}W). scancel + heal."
            scancel "$JOB"; outcome="wedged"; sleep 10; break
        fi
        sleep "$POLL"
    done

    case "$outcome" in
        wedged)
            consec_wedge=$((consec_wedge+1))
            log "slot $slot: wedged (consecutive=$consec_wedge/$MAX_CONSEC_WEDGE). resubmitting."
            if [ "$consec_wedge" -ge "$MAX_CONSEC_WEDGE" ]; then
                log "STOP: $consec_wedge consecutive wedges with no healthy slot — race looks deterministic, needs a human."
                exit 2
            fi
            ;;
        ended:TIMEOUT)
            consec_wedge=0
            log "slot $slot: healthy wall-clock TIMEOUT — chain advanced. continuing."
            ;;
        ended:COMPLETED)
            log "slot $slot: job COMPLETED (training finished?). stopping autoheal."; exit 0;;
        *)
            log "slot $slot: unexpected end ($outcome); stopping autoheal for inspection."; exit 0;;
    esac
done
log "reached MAX_SLOTS=$MAX_SLOTS; stopping. Re-run to continue."
