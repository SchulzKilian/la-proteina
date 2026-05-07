#!/bin/bash
# Queue: re-probe every variant/hybrid arm at nsteps=400 (E021/E034/E038/E039/E040/E041
# were all at nsteps=200, below the integrator-convergence bar — see CLAUDE.md
# "Sampling — nsteps=400 is a HARD RULE" and feedback_use_nsteps_400_for_designability.md).
#
# 2 GPUs in parallel, 3 sequential rounds. Each probe = N=6 × L∈{50,100,200} × nsteps=400.
# Wall: ~30 min per probe, ~90 min total.

set -uo pipefail
cd /home/ks2218/la-proteina

export PYTHON_EXEC=/home/ks2218/.conda/envs/laproteina_env/bin/python
export PATH=/home/ks2218/.conda/envs/laproteina_env/bin:$PATH
PY=$PYTHON_EXEC

probe() {
    local gpu=$1
    local cfg=$2
    local logbase="/home/ks2218/la-proteina/nohup_${cfg}"
    echo "[$(date)] [GPU $gpu] START $cfg" >> /home/ks2218/la-proteina/nohup_nfe400_queue.out
    CUDA_VISIBLE_DEVICES=$gpu $PY proteinfoundation/generate.py --config-name=$cfg \
        > "${logbase}.gen.log" 2>&1
    local genrc=$?
    echo "[$(date)] [GPU $gpu] gen exit=$genrc for $cfg" >> /home/ks2218/la-proteina/nohup_nfe400_queue.out
    if [ $genrc -ne 0 ]; then
        echo "[$(date)] [GPU $gpu] SKIP eval for $cfg (gen failed)" >> /home/ks2218/la-proteina/nohup_nfe400_queue.out
        return $genrc
    fi
    CUDA_VISIBLE_DEVICES=$gpu $PY proteinfoundation/evaluate.py --config_name $cfg \
        > "${logbase}.eval.log" 2>&1
    local evrc=$?
    echo "[$(date)] [GPU $gpu] eval exit=$evrc for $cfg" >> /home/ks2218/la-proteina/nohup_nfe400_queue.out
    return $evrc
}

echo "[$(date)] queue start" > /home/ks2218/la-proteina/nohup_nfe400_queue.out

# Round 1
probe 4 inference_downsampled_step2331_n6_nfe400 &
P1=$!
probe 5 inference_scnbr_t04_step819_n6_nfe400 &
P2=$!
wait $P1 $P2

# Round 2
probe 4 inference_scnbr_t04_step1133_n6_nfe400 &
P1=$!
probe 5 inference_hybrid_conv_to_canonical_t06_n6_nfe400 &
P2=$!
wait $P1 $P2

# Round 3
probe 4 inference_hybrid_conv_to_scnbr_t06_n6_nfe400 &
P1=$!
probe 5 inference_hybrid_conv_to_scnbr_t075_n6_nfe400 &
P2=$!
wait $P1 $P2

echo "[$(date)] queue done" >> /home/ks2218/la-proteina/nohup_nfe400_queue.out
