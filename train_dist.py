import os
import argparse
import glob
import random
import math
import yaml
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

def get_lr_schedule(step, total_steps, base_lr, warmup_pct=0.05, decay_pct=0.15):
    warmup_steps = int(total_steps * warmup_pct)
    decay_steps = int(total_steps * decay_pct)
    decay_start = total_steps - decay_steps
    
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    if step < decay_start:
        return base_lr
    
    progress = (step - decay_start) / max(1, decay_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def calculate_topk_accuracy(logits, targets, k=1):
    target_indices = torch.argmax(targets, dim=1)
    _, topk_indices = torch.topk(logits, k, dim=1)
    correct = torch.eq(topk_indices, target_indices.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

def train(config):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    train_cfg = config['training']
    model_cfg = config['model']

    # Seed
    seed = train_cfg.get('seed', 42) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Logging
    if rank == 0:
        if not os.path.exists(train_cfg['save_dir']):
            os.makedirs(train_cfg['save_dir'], exist_ok=True)
        # Check for wandb API key or login
        if "WANDB_API_KEY" in os.environ:
             wandb.init(project=train_cfg['project'], name=train_cfg['run_name'], config=config)
        else:
             print("WANDB_API_KEY not found. WandB logging might fail or run in offline mode.")
             wandb.init(project=train_cfg['project'], name=train_cfg['run_name'], config=config, mode="disabled")

    # Data Finding
    data_path = train_cfg['data_path']
    if os.path.isdir(data_path):
        candidates = glob.glob(os.path.join(data_path, "*.jsonl.zst"))
        if candidates:
            data_path = candidates[0]
            if rank == 0: print(f"Found dataset file: {data_path}")
        else:
            if rank == 0: print(f"Warning: No .jsonl.zst found in {data_path}. Ensure data exists.")

    # Dataset
    # Split: First 50k for validation, rest for training
    val_samples_per_rank = 50000 // world_size
    
    # Validation Dataset (First 50k)
    val_dataset = FastChessDataset(
        data_path,
        rank=rank,
        world_size=world_size,
        limit=val_samples_per_rank,
        skip=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['local_batch_size'],
        num_workers=train_cfg['workers'],
        pin_memory=True
    )
    
    # Training Dataset (Rest)
    train_dataset = FastChessDataset(
        data_path, 
        rank=rank, 
        world_size=world_size,
        skip=val_samples_per_rank # Skip the validation chunk
    )
    train_dataset = StreamingShuffleDataset(train_dataset, buffer_size=train_cfg['buffer_size'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg['local_batch_size'], 
        num_workers=train_cfg['workers'], 
        pin_memory=True
    )
    
    # Helper for infinite validation stream
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    
    val_iterator = cycle(val_loader)

    # Model
    model = ChessTransformer(
        vocab_size=model_cfg['vocab_size'],
        hidden_size=model_cfg['hidden_size'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        ff_dim=model_cfg['ff_dim'],
        num_eval_bins=model_cfg['num_eval_bins'],
        num_scratchpad=model_cfg['num_scratchpad']
    ).to(device)

    if rank == 0:
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer Separation (Muon vs AdamW)
    muon_params = []
    adam_decay = []
    adam_no_decay = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 2 and "embed" not in name and "head" not in name:
            muon_params.append(p)
        elif p.ndim >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
            
    optimizer_muon = optim.Muon(muon_params, lr=train_cfg['muon_learning_rate'], weight_decay=train_cfg['muon_weight_decay'], momentum=0.95)
    optimizer_adam = optim.AdamW([
        {'params': adam_decay, 'weight_decay': train_cfg['weight_decay']},
        {'params': adam_no_decay, 'weight_decay': 0.0}
    ], lr=train_cfg['adam_learning_rate'])

    criterion = ChessLoss()

    step = 0
    model.train()
    
    # Calculate total steps for scheduler
    val_count = 50000
    train_count = train_cfg['dataset_size'] - val_count
    if train_count <= 0:
        raise ValueError(f"Dataset length {train_cfg['dataset_size']} is too small for validation set of {val_count}")

    global_batch_size = train_cfg['local_batch_size'] * world_size
    steps_per_epoch = train_count // global_batch_size
    total_steps = train_cfg['epochs'] * steps_per_epoch
    
    if rank == 0: 
        print(f"Starting training for {train_cfg['epochs']} epochs.")
        print(f"Global Batch Size: {global_batch_size} ({train_cfg['local_batch_size']} per GPU * {world_size} GPUs)")
        print(f"Training samples: {train_count} | Steps per epoch: {steps_per_epoch}")
        print(f"Scheduler set for {total_steps} total steps.")
    
    # Metrics accumulator
    metrics_acc = {'loss': 0.0, 'acc_1': 0.0, 'acc_3': 0.0, 'acc_5': 0.0}
    
    for epoch in range(train_cfg['epochs']):
        if rank == 0: print(f"--- Epoch {epoch+1}/{train_cfg['epochs']} ---")
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Calculate Learning Rates
            lr_m = get_lr_schedule(step, total_steps, train_cfg['muon_learning_rate'], train_cfg['warmup_percentage'], train_cfg['decay_percentage'])
            lr_a = get_lr_schedule(step, total_steps, train_cfg['adam_learning_rate'], train_cfg['warmup_percentage'], train_cfg['decay_percentage'])
            
            for pg in optimizer_muon.param_groups: pg['lr'] = lr_m
            for pg in optimizer_adam.param_groups: pg['lr'] = lr_a
            
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()
            
            # Mixed Precision Forward
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch)
                loss, losses = criterion(outputs, batch)
                
                # Accumulate Stats (Every Step)
                with torch.no_grad():
                    legal_mask = batch['legal_mask']
                    masked_logits = outputs['policy'] + (1.0 - legal_mask) * -1e9
                    acc1 = calculate_topk_accuracy(masked_logits, batch['move_target'], k=1)
                    acc3 = calculate_topk_accuracy(masked_logits, batch['move_target'], k=3)
                    acc5 = calculate_topk_accuracy(masked_logits, batch['move_target'], k=5)
                    
                    metrics_acc['loss'] += loss.item()
                    metrics_acc['acc_1'] += acc1
                    metrics_acc['acc_3'] += acc3
                    metrics_acc['acc_5'] += acc5
                    
                    for k, v in losses.items():
                        if k not in metrics_acc: metrics_acc[k] = 0.0
                        metrics_acc[k] += v.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer_muon.step()
            optimizer_adam.step()
            
            step += 1
            
            # --- Logging & Validation ---
            if step % train_cfg['log_interval'] == 0:
                # 1. Training Logs
                if rank == 0:
                    # Average over interval
                    div = train_cfg['log_interval']
                    log_dict = {
                        "train/loss": metrics_acc['loss'] / div,
                        "train/acc_top1": metrics_acc['acc_1'] / div,
                        "train/acc_top3": metrics_acc['acc_3'] / div,
                        "train/acc_top5": metrics_acc['acc_5'] / div,
                        "lr/muon": lr_m,
                        "lr/adam": lr_a,
                        "step": step,
                        "epoch": epoch + 1
                    }
                    
                    for k in losses.keys():
                        if k in metrics_acc:
                            log_dict[f"train/{k}"] = metrics_acc[k] / div
                        
                    print(f"Ep {epoch+1} | Step {step}: Loss {log_dict['train/loss']:.4f} | Acc1 {log_dict['train/acc_top1']:.3f}")
                    
                    if wandb.run:
                        wandb.log(log_dict)

                    # Reset
                    metrics_acc = {'loss': 0.0, 'acc_1': 0.0, 'acc_3': 0.0, 'acc_5': 0.0}

            # 2. Validation (Every 200 steps)
            if step % 200 == 0:
                model.eval()
                val_metrics = {'loss': 0.0, 'acc_1': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'mate_loss': 0.0}
                val_steps = 50  # Validate on 50 batches (~6400 samples with bs=128)
                
                with torch.no_grad():
                    for _ in range(val_steps):
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = cycle(val_loader)
                            val_batch = next(val_iterator)
                            
                        val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
                        
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            val_outputs = model(val_batch)
                            val_loss, val_losses = criterion(val_outputs, val_batch)
                            
                            legal_mask = val_batch['legal_mask']
                            masked_logits = val_outputs['policy'] + (1.0 - legal_mask) * -1e9
                            val_acc1 = calculate_topk_accuracy(masked_logits, val_batch['move_target'], k=1)
                        
                        val_metrics['loss'] += val_loss.item()
                        val_metrics['acc_1'] += val_acc1
                        val_metrics['policy_loss'] += val_losses['policy'].item()
                        val_metrics['value_loss'] += val_losses['value'].item()
                        val_metrics['mate_loss'] += val_losses['mate'].item()

                # Average metrics
                for k in val_metrics:
                    val_metrics[k] /= val_steps
                
                if rank == 0:
                    print(f"--- Val Step {step}: Loss {val_metrics['loss']:.4f} | Acc1 {val_metrics['acc_1']:.3f}")
                    if wandb.run:
                        wandb.log({
                            "val/loss": val_metrics['loss'],
                            "val/acc_top1": val_metrics['acc_1'],
                            "val/policy_loss": val_metrics['policy_loss'],
                            "val/value_loss": val_metrics['value_loss'],
                            "val/mate_loss": val_metrics['mate_loss'],
                            "step": step
                        })
                
                model.train()

            if rank == 0 and step % train_cfg['save_interval'] == 0:
                ckpt_path = os.path.join(train_cfg['save_dir'], f"checkpoint_{step}.pt")
                torch.save(model.module.state_dict(), ckpt_path)

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(train_cfg['save_dir'], "final.pt"))
        if wandb.run:
            wandb.finish()

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)