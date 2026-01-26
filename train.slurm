#!/bin/bash
#SBATCH --job-name=mvrce-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=hgx
#SBATCH --time=48:00:00
#SBATCH -w hgx2
# Create logs directory
mkdir -p logs

# Environment Setup
echo "Syncing dependencies..."
uv sync

# Set distributed variables for torchrun
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Starting training on $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Run with torchrun
# Using 'uv run' to ensure we use the project environment
# Batch size 2048 per GPU (Total 8192)
uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_dist.py \
    --data_path dataset/data \
    --batch_size 2048 \
    --embed_dim 512 \
    --depth 24 \
    --num_heads 8 \
    --ff_dim 2048 \
    --save_dir checkpoints_a100 \
    --log_interval 50 \
    --save_interval 2000 \
    --max_steps 500000
