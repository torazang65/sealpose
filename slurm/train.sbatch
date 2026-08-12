#!/bin/bash
#SBATCH --job-name=sealpose
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu
#SBATCH --output=./logs/%x_%A_out.log
#SBATCH --error=./logs/%x_%A_error.log

# Generic launcher: all arguments are forwarded to train_lifting.py.
#   sbatch --job-name=<tag> slurm/train.sbatch --dataset 3dhp ...
# A checkpoint dir is derived from the job name, so each run keeps its own.

set -euo pipefail

REPO=/data2/sukhosong/sealpose
PY=/data2/sukhosong/.conda/envs/pose/bin/python

cd "$REPO"
mkdir -p "checkpoints/${SLURM_JOB_NAME}"

echo "job      : ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "node     : $(hostname)"
echo "gpu      : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "args     : $*"
echo "started  : $(date '+%F %T')"
echo "----------------------------------------------------------------"

"$PY" train_lifting.py "$@" \
    --save_path "checkpoints/${SLURM_JOB_NAME}/final.pth"

echo "----------------------------------------------------------------"
echo "finished : $(date '+%F %T')"
