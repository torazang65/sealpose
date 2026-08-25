#!/bin/bash
#SBATCH --job-name=pyjob
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu
#SBATCH --output=./logs/%x_%A_out.log
#SBATCH --error=./logs/%x_%A_error.log

# Generic launcher for a one-off analysis script:
#   sbatch --job-name=<tag> slurm/pyjob.sbatch <script.py> [args...]
set -euo pipefail
REPO=/data2/sukhosong/sealpose
PY=/data2/sukhosong/.conda/envs/pose/bin/python
cd "$REPO"
echo "job     : ${SLURM_JOB_NAME} (${SLURM_JOB_ID})   node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "started : $(date '+%F %T')"
echo "----------------------------------------------------------------"
"$PY" "$@"
echo "----------------------------------------------------------------"
echo "finished: $(date '+%F %T')"
