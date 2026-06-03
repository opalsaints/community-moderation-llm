#!/bin/bash
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=download_data
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --account=<your-slurm-account>

module purge
module load 2023
source $HOME/capstone_env_2025/bin/activate

export PYTHONUNBUFFERED=1

python -u src/data/download_subreddit.py \
    --all \
    --after 2024-01-01 \
    --before 2025-01-01 \
    --output-dir data/raw/
