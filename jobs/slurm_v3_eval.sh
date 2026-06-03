#!/bin/bash
# Generic SLURM wrapper for v3 evaluation. Called by slurm_pilot.sh.
# All finetune_v3.py args are passed through via $@ (must include --skip-train).
#
# Usage:
#   sbatch --partition=gpu_a100 --dependency=afterok:12345 \
#     scripts/slurm_v3_eval.sh [finetune_v3.py args...] --skip-train

#SBATCH --job-name=v3_eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v3_eval_%j.out
#SBATCH --error=logs/v3_eval_%j.err

set -e
mkdir -p logs

echo "=== v3 Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Args: $@"
echo "Start: $(date)"

source ~/capstone_env_2025/bin/activate
cd ~/capstone

python3 scripts/finetune_v3.py "$@"

echo ""
echo "=== Evaluation finished: $(date) ==="
