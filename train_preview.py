import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.chess_dataset import FastChessDataset
from model import ChessTransformer
from loss import ChessLoss
import time
import wandb
import matplotlib.pyplot as plt
import numpy as np
import resource
import math

# Fix for "Too many open files"
torch.multiprocessing.set_sharing_strategy('file_system')

# Increase file descriptor limit
try:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except ValueError:
    pass

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
    _, topk_indices = torch.topk(logits, k, dim=1) # [B, k]
    correct = torch.eq(topk_indices, target_indices.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

def train_preview():
    # --- SWAPPABLE CONFIG ---
    TRAIN_SAMPLES = 500
    VAL_SAMPLES = 100
    SHUFFLE_BUFFER = 500
    EPOCHS = 10
    BATCH_SIZE = 16
    VAL_INTERVAL_STEPS = 20
    
    CHECKPOINT_INTERVAL = 5000 
    CHECKPOINT_PATH = "checkpoint_train_preview.pt"
    RESUME = False # Set to false for preview run
    
    WARMUP_PCT = 0.05
    DECAY_PCT = 0.15
    # -------------------------

    # Path to data - searching for any .zst in dataset/data
    import glob
    candidates = glob.glob("dataset/data/*.jsonl.zst")
    if candidates:
        DATA_PATH = candidates[0]
    else:
        DATA_PATH = "dataset/data/lichess_db_eval.jsonl.zst"
        
    GRAD_ACCUM = 1
    MUON_LR = 0.02 
    ADAM_LR = 3e-4
    ADAM_WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training on {DEVICE}...")

    # --- WandB Init ---
    # Running in disabled mode if API key not found for quick preview
    wandb_mode = "online" if "WANDB_API_KEY" in os.environ else "disabled"
    wandb.init(project="mvrce-chess", name="train_preview_rel_pos", resume="allow", mode=wandb_mode, config={
        "batch_size": BATCH_SIZE,
        "muon_lr": MUON_LR,
        "train_samples": TRAIN_SAMPLES,
        "epochs": EPOCHS,
        "schedule": "warmup_steady_cosine",
        "model": "768_24_rel_pos"
    })
    
    steps_per_epoch = math.ceil(TRAIN_SAMPLES / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    
    # --- Model & Optim ---
    # Matching config.yaml defaults but depth=4 for faster preview
    model = ChessTransformer(
        hidden_size=768, 
        depth=4, 
        num_heads=12, 
        num_scratchpad=16
    ).to(DEVICE)
    
    muon_params = []
    adam_decay = []
    adam_no_decay = []
    for name, p in model.named_parameters():
        is_embedding = any(kw in name for kw in ["embed", "emb", "scratchpad", "token", "query"])
        if p.ndim < 2 or is_embedding:
            adam_no_decay.append(p)
        elif p.ndim == 2 and "head" not in name:
            muon_params.append(p)
        else:
            adam_decay.append(p)

    optimizer_muon = optim.Muon(muon_params, lr=MUON_LR, weight_decay=0.1, momentum=0.95, eps=1e-8)
    optimizer_adam = optim.AdamW([
        {'params': adam_decay, 'weight_decay': ADAM_WEIGHT_DECAY},
        {'params': adam_no_decay, 'weight_decay': 0.0}
    ], lr=ADAM_LR)
    
    criterion = ChessLoss()
    
    # --- Resume Logic ---
    start_epoch = 0
    step = 0
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])
        optimizer_adam.load_state_dict(checkpoint['optimizer_adam_state_dict'])
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        print(f"Resumed from Epoch {start_epoch+1}, Step {step}")

    # --- Training Loop ---
    model.train()
    
    # Metrics
    metrics_acc = {
        'loss': 0.0,
        'acc_1': 0.0,
        'acc_3': 0.0,
        'acc_5': 0.0,
        'value_scalar': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        
        steps_already_done_in_epoch = step % steps_per_epoch
        samples_to_skip_in_epoch = steps_already_done_in_epoch * BATCH_SIZE
        
        ds_train = FastChessDataset(DATA_PATH, 
                                    skip=VAL_SAMPLES + samples_to_skip_in_epoch, 
                                    limit=TRAIN_SAMPLES - samples_to_skip_in_epoch)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=0)
        
        for batch in train_loader:
            if step >= total_steps: break
            
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch)
                loss, losses = criterion(outputs, batch)
                
                with torch.no_grad():
                    legal_mask = batch['legal_mask']
                    masked_logits = outputs['policy'].masked_fill(legal_mask == 0.0, float('-inf'))

                    # Compute Top-k
                    acc1 = calculate_topk_accuracy(masked_logits, batch['move_target'], k=1)
                    acc3 = calculate_topk_accuracy(masked_logits, batch['move_target'], k=3)
                    acc5 = calculate_topk_accuracy(masked_logits, batch['move_target'], k=5)
            
            metrics_acc['loss'] += loss.item()
            metrics_acc['acc_1'] += acc1
            metrics_acc['acc_3'] += acc3
            metrics_acc['acc_5'] += acc5
            metrics_acc['value_scalar'] += losses['value_scalar'].item()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(adam_decay + adam_no_decay, max_norm=MAX_GRAD_NORM)
            
            lr_m = get_lr_schedule(step, total_steps, MUON_LR, WARMUP_PCT, DECAY_PCT)
            lr_a = get_lr_schedule(step, total_steps, ADAM_LR, WARMUP_PCT, DECAY_PCT)
            for pg in optimizer_muon.param_groups: pg['lr'] = lr_m
            for pg in optimizer_adam.param_groups: pg['lr'] = lr_a

            optimizer_muon.step()
            optimizer_adam.step()
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()
            
            # Logging
            if step % 10 == 0:
                count = 10 if step > 0 else 1
                log_data = {
                    "train/loss": metrics_acc['loss'] / count,
                    "train/acc_top1": metrics_acc['acc_1'] / count,
                    "train/acc_top3": metrics_acc['acc_3'] / count,
                    "train/acc_top5": metrics_acc['acc_5'] / count,
                    "train/value_scalar_loss": metrics_acc['value_scalar'] / count,
                    "lr/muon": lr_m,
                    "step": step,
                    "epoch": epoch + 1
                }
                wandb.log(log_data, step=step)                print(f"Ep {epoch+1} | St {step} | Loss: {log_data['train/loss']:.4f} | Acc1: {log_data['train/acc_top1']:.3f} | ValScal: {log_data['train/value_scalar_loss']:.4f}")
                
                # Reset metrics
                metrics_acc = {k: 0.0 for k in metrics_acc}
            
            # Validation
            if step % VAL_INTERVAL_STEPS == 0 and step > 0:
                model.eval()
                v_metrics = {'loss': 0.0, 'acc_1': 0.0, 'value_scalar': 0.0}
                v_count = 0
                val_loader = DataLoader(FastChessDataset(DATA_PATH, skip=0, limit=VAL_SAMPLES), batch_size=BATCH_SIZE)
                
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_batch = {k: v.to(DEVICE) for k, v in v_batch.items()}
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            v_out = model(v_batch)
                            l, v_losses = criterion(v_out, v_batch)
                            
                            legal_mask = v_batch['legal_mask']
                            masked_logits = v_out['policy'].masked_fill(legal_mask == 0.0, float('-inf'))
                            
                            v_metrics['acc_1'] += calculate_topk_accuracy(masked_logits, v_batch['move_target'], k=1)
                            v_metrics['loss'] += l.item()
                            v_metrics['value_scalar'] += v_losses['value_scalar'].item()
                            v_count += 1
                            
                if v_count > 0:
                    wandb.log({
                        "val/loss": v_metrics['loss']/v_count,
                        "val/acc_top1": v_metrics['acc_1']/v_count,
                        "val/value_scalar_loss": v_metrics['value_scalar']/v_count,
                        "step": step
                    }, step=step)                    print(f"--- Validation --- Loss: {v_metrics['loss']/v_count:.4f} | Top1: {v_metrics['acc_1']/v_count:.3f} | ValScal: {v_metrics['value_scalar']/v_count:.4f}")
                model.train()

            step += 1
            
            # Checkpoint
            if step % CHECKPOINT_INTERVAL == 0:
                print(f"Saving checkpoint at step {step}...")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                    'optimizer_adam_state_dict': optimizer_adam.state_dict(),
                }, CHECKPOINT_PATH)

        if step >= total_steps: break
        
    print("Training Complete.")
    torch.save(model.state_dict(), "chess_model_muon_final.pt")
    wandb.finish()

if __name__ == "__main__":
    train_preview()