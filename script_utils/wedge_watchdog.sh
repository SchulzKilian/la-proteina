#!/bin/bash
# Wedge watchdog for the dynamic-K training slot (E101/E103 spin-hang detector).
#
# Auto-cancels the job the moment it shows the E101/E103 spin signature so it
# does not waste a 6h SL3 slot spinning (last time it burned ~1h45m before a
# manual kill).
#
# Wedge fingerprint (decisive, low false-positive):
#   utilization.gpu >= GPUUTIL_MIN  (pinned, spinning)
#   AND power.draw   <  POWER_MAX   (no real compute; healthy sparse = 250-400W)
#   AND util.memory  <= MEMUTIL_MAX (no gather traffic; healthy = high)
# This trio CANNOT occur during healthy training (mem-bound, >250W) or during
# idle/checkpoint pauses (gpu-util drops). Sustained WINDOW samples -> cancel.
#
# Usage: wedge_watchdog.sh <JOBID>
set -uo pipefail

JOB="${1:?usage: wedge_watchdog.sh <JOBID>}"
GRACE=${GRACE:-150}   # s after RUNNING before judging (model load / first steps)
POLL=${POLL:-30}      # s between samples
WINDOW=${WINDOW:-6}   # consecutive wedge-like samples to trigger
POWER_MAX=170    # W   (wedge ~87-130; healthy >250)
MEMUTIL_MAX=10   # %   (wedge 2-4; healthy high)
GPUUTIL_MIN=80   # %   (wedge 100; idle <80)

MONLOG="/home/ks2218/la-proteina/wedge_monitor_${JOB}.log"
SLOG="/home/ks2218/la-proteina/slurm_ca_1gpu_${JOB}.out"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MONLOG"; }

log "watchdog start for job $JOB (POWER_MAX=$POWER_MAX MEMUTIL_MAX=$MEMUTIL_MAX GPUUTIL_MIN=$GPUUTIL_MIN WINDOW=$WINDOW POLL=${POLL}s)"

# 1. Wait until RUNNING; capture node.
NODE=""
while :; do
    state=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
    if [ -z "$state" ]; then
        log "job $JOB not in queue (finished/cancelled before start). exit."
        exit 0
    fi
    if [ "$state" = "RUNNING" ]; then
        NODE=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null)
        log "job RUNNING on node=$NODE; grace ${GRACE}s before judging."
        break
    fi
    log "state=$state; waiting for RUNNING."
    sleep 30
done

sleep "$GRACE"

bad=0
while :; do
    state=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
    if [ "$state" != "RUNNING" ]; then
        log "job no longer RUNNING (state='${state:-gone}'). exit."
        exit 0
    fi
    # re-fetch node in case of requeue
    n=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null)
    [ -n "$n" ] && NODE="$n"

    reading=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
        "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits" 2>/dev/null | head -1)

    if [ -z "$reading" ]; then
        log "ssh/nvidia-smi sample failed on $NODE (inconclusive; bad unchanged at $bad)."
        sleep "$POLL"; continue
    fi

    g=$(echo "$reading" | awk -F',' '{gsub(/ /,"",$1); print $1}')
    m=$(echo "$reading" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    p=$(echo "$reading" | awk -F',' '{gsub(/ /,"",$3); print $3}')

    if [ -f "$SLOG" ]; then age=$(( $(date +%s) - $(stat -c %Y "$SLOG") )); else age="n/a"; fi

    wedgey=$(awk -v g="$g" -v m="$m" -v p="$p" \
        -v gm="$GPUUTIL_MIN" -v mm="$MEMUTIL_MAX" -v pm="$POWER_MAX" \
        'BEGIN{print (g>=gm && m<=mm && p<pm) ? 1 : 0}')

    if [ "$wedgey" = "1" ]; then
        bad=$((bad+1))
        log "gpu=${g}% mem=${m}% pow=${p}W log_age=${age}s  -> WEDGE-LIKE ($bad/$WINDOW)"
    else
        [ "$bad" -gt 0 ] && log "gpu=${g}% mem=${m}% pow=${p}W log_age=${age}s  -> healthy (reset $bad->0)" \
                         || log "gpu=${g}% mem=${m}% pow=${p}W log_age=${age}s  -> healthy"
        bad=0
    fi

    if [ "$bad" -ge "$WINDOW" ]; then
        log "WEDGE CONFIRMED ($bad consecutive samples, ~$((WINDOW*POLL/60)) min). Cancelling $JOB."
        scancel "$JOB" && log "scancel $JOB issued." || log "scancel FAILED (rc=$?)."
        exit 0
    fi
    sleep "$POLL"
done
