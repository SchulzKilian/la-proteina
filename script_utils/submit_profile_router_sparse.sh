#!/bin/bash
#SBATCH -J profile_router_sparse
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --exclude=gpu-q-43
#SBATCH --output=slurm_profile_router_sparse_%j.out

source $HOME/.bashrc
export LAPROTEINA_ENV=/home/ks2218/conda_envs/laproteina_env
export PATH=$LAPROTEINA_ENV/bin:$PATH
export CONDA_PREFIX=$LAPROTEINA_ENV
export CONDA_DEFAULT_ENV=laproteina_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $HOME/la-proteina

echo "[+] node: $(hostname)  python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python script_utils/profile_router_sparse_microbatch.py \
    --B 6 --N 512 --n_iters 20 --warmup 3 --accum 32
