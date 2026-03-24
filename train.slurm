#!/bin/bash
#SBATCH --job-name=mvrd-train
#SBATCH --partition=hgx
#SBATCH --nodelist=hgx2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=128:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Activate virtual environment
cd /raid/ai164201/MVRD

# Weights & Biases API Key
export WANDB_API_KEY=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

# Environment Variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29000 + RANDOM % 1000))

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1



echo "Starting training on node $SLURM_JOB_NODELIST with $SLURM_GPUS_PER_NODE GPUs."

uv run torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_dist.py \
    --config configs/config_small_no_scratchpad.yaml
