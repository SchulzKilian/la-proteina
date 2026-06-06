#!/bin/bash
# One L500 cell: generate (resume) + fold codesign. CFG, OUT, SEEDS via env.
set -o pipefail; cd /home/ks2218/la-proteina
export PATH=$HOME/.conda/envs/laproteina_env/bin:$PATH; export MPNN_INPROCESS=1
PY=$HOME/.conda/envs/laproteina_env/bin/python; export CUDA_VISIBLE_DEVICES=${CUDA:?}
"$PY" -m steering.generate --proteina_config inference_ucond_notri_long \
    --steering_config "${CFG:?}" --lengths 500 --seeds ${SEEDS:?} --nsteps 400 \
    --skip_unguided --resume --output_dir "${OUT:?}" --device cuda:0
"$PY" script_utils/run_burial_l500_eval.py fold "$OUT"
echo "=== CELL_DONE $OUT ==="
