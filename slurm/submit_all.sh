#!/bin/bash
# Submit the four run.sh configurations as separate Slurm jobs.
# Hyperparameters are copied verbatim from run.sh; only --save_path is added
# (via train.sbatch) so the trained models actually survive the run.
#
# QOS normal2 allows gres/gpu=2, so at most two of these run concurrently;
# the rest wait in the queue. That is expected.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

COMMON="--dataset 3dhp --batch_size 1024 --num_epoch 50 --no_logging"

sbatch --job-name=lin-base slurm/train.sbatch $COMMON \
    --lr 2e-4 --task_net linear-large \
    --type baseline

sbatch --job-name=lin-seal slurm/train.sbatch $COMMON \
    --lr 2e-4 --task_net linear-large \
    --type dynamic --em_loss_type margin \
    --energy_weight 1e-3 --lr_loss 5e-4 --em_loss mpjpe

sbatch --job-name=gcn-base slurm/train.sbatch $COMMON \
    --lr 1e-2 --task_net semgcn --task_dropout 0 \
    --type baseline

sbatch --job-name=gcn-seal slurm/train.sbatch $COMMON \
    --lr 1e-2 --task_net semgcn --task_dropout 0 \
    --type dynamic --em_loss_type margin --centering hip --num_samples 1 \
    --energy_weight 1e-4 --lr_loss 1e-3 --em_loss mpjpe

echo
squeue -u "$USER" -o "%.10i %.10P %.12j %.8T %.10M %R"
