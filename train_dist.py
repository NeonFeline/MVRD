import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from model import ChessTransformer
from dataset.chess_dataset import FastChessDataset, StreamingShuffleDataset
from loss import ChessLoss

def setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def train(args):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Seed
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Logging
    if rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        # Check for wandb API key or login
        if "WANDB_API_KEY" in os.environ:
             wandb.init(project=args.project, name=args.run_name, config=args)
        else:
             print("WANDB_API_KEY not found. WandB logging might fail or run in offline mode.")
             wandb.init(project=args.project, name=args.run_name, config=args, mode="disabled")

    # Data Finding
    data_path = args.data_path
    if os.path.isdir(data_path):
        candidates = glob.glob(os.path.join(data_path, "*.jsonl.zst"))
        if candidates:
            data_path = candidates[0]
            if rank == 0: print(f"Found dataset file: {data_path}")
        else:
            if rank == 0: print(f"Warning: No .jsonl.zst found in {data_path}. Ensure data exists.")

    # Dataset
    # Note: num_workers must be 0 for this sharding logic to work correctly 
    # unless we update FastChessDataset to handle worker_info.
    dataset = FastChessDataset(
        data_path, 
        rank=rank, 
        world_size=world_size
    )
    dataset = StreamingShuffleDataset(dataset, buffer_size=args.buffer_size)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=0, # Keep 0 to avoid sharding complexity for now
        pin_memory=True
    )

    # Model
    model = ChessTransformer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim
    ).to(device)

    if rank == 0:
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    criterion = ChessLoss()

    step = 0
    model.train()
    
    iterator = iter(dataloader)
    
    if rank == 0: print("Starting training...")
    
    while step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            if rank == 0: print("Dataset exhausted, restarting iterator...")
            iterator = iter(dataloader)
            batch = next(iterator)
            
        # Move to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss, losses = criterion(outputs, batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        step += 1
        
        if rank == 0:
            if step % args.log_interval == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": step
                }
                for k, v in losses.items():
                    log_dict[f"train/{k}"] = v.item()
                if wandb.run:
                    wandb.log(log_dict)
                print(f"Step {step}: Loss {loss.item():.4f}")
            
            if step % args.save_interval == 0:
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_{step}.pt")
                torch.save(model.module.state_dict(), ckpt_path)

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, "final.pt"))
        if wandb.run:
            wandb.finish()

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/data")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--project", type=str, default="mvrce-chess")
    parser.add_argument("--run_name", type=str, default="run_A100")
    
    # Training Params
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=50000)
    
    # Model Params (Default small-ish, adjust for A100)
    # A100 80GB can handle much larger
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=2048)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    
    args = parser.parse_args()
    
    train(args)
