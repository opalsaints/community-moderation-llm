#!/bin/bash
# Generic SLURM wrapper for v3 training. Called by slurm_pilot.sh.
# All finetune_v3.py args are passed through via $@.
#
# Usage:
#   sbatch --partition=gpu_a100 scripts/slurm_v3_train.sh [finetune_v3.py args...]

#SBATCH --job-name=v3_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/v3_train_%j.out
#SBATCH --error=logs/v3_train_%j.err

set -e
mkdir -p logs

echo "=== v3 Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Args: $@"
echo "Start: $(date)"

source ~/capstone_env_2025/bin/activate
cd ~/capstone

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/finetune_v3.py "$@"

echo ""
echo "=== Training finished: $(date) ==="
