import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FastChessDataset
from model import ChessTransformer
from loss import ChessLoss
import time
import wandb
import matplotlib.pyplot as plt

def train_small_subset():
    # --- Config ---
    DATA_PATH = "dataset/data/lichess_db_eval.jsonl.zst"
    BATCH_SIZE = 16 
    GRAD_ACCUM = 1  # Effective Batch Size = 16
    MUON_LR = 0.02  
    ADAM_LR = 3e-4
    MAX_GRAD_NORM = 1.0 # Standard threshold for clipping
    EPOCHS = 1
    MAX_STEPS = 500 
    WARMUP_STEPS = 40 
    CHECKPOINT_INTERVAL = 250 # Save checkpoint every 50 steps
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoint.pt"
    
    print(f"Training on {DEVICE}...")

    # --- WandB Init ---
    wandb.init(project="mvrce-chess", name="overfit_test_resumable", resume="allow", config={
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "muon_lr": MUON_LR,
        "adam_lr": ADAM_LR,
        "max_steps": MAX_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL
    })
    
    # --- Data ---
    dataset = FastChessDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # --- Model ---
    model = ChessTransformer(depth=24, embed_dim=512).to(DEVICE)
    
    # --- Optimizer Parameter Partitioning ---
    muon_params = []
    adam_decay = []
    adam_no_decay = []

    for name, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in name and "head" not in name:
            muon_params.append(p)
        elif p.ndim >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)

    # Initialize Official PyTorch Muon with decreased epsilon
    optimizer_muon = optim.Muon(
        muon_params,
        lr=MUON_LR,
        weight_decay=0.1,
        momentum=0.95,
        eps=1e-8  # Decreased epsilon for more sensitivity
    )

    # Initialize AdamW
    optimizer_adam = optim.AdamW([
        {'params': adam_decay, 'weight_decay': 0.01},
        {'params': adam_no_decay, 'weight_decay': 0.0}
    ], lr=ADAM_LR)
    
    # --- Loss ---
    criterion = ChessLoss()
    
    # --- Loop State Init ---
    step = 0
    start_micro_step = 0
    total_loss_acc = 0.0
    acc_policy = 0.0
    acc_value = 0.0
    acc_mate = 0.0
    acc_accuracy = 0.0
    
    history = {
        'loss': [],
        'policy_loss': [],
        'value_loss': [],
        'mate_loss': [],
        'accuracy': []
    }

    # --- Load Checkpoint ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])
            optimizer_adam.load_state_dict(checkpoint['optimizer_adam_state_dict'])
            step = checkpoint['step']
            start_micro_step = checkpoint['micro_step']
            history = checkpoint['history']
            print(f"Resumed from step {step} (micro_step {start_micro_step})")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    
    # Get single batch for overfit test
    print("Fetching single batch for overfit test...")
    batch = next(iter(loader))
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    
    start_time = time.time()
    
    print(f"Starting training loop from micro-step {start_micro_step} to {MAX_STEPS}...")
    
    model.train()
    
    for i in range(start_micro_step, MAX_STEPS):
        # Forward with Mixed Precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(batch)
            loss, metrics = criterion(outputs, batch)
            
            # Compute Accuracy
            with torch.no_grad():
                # Apply Legal Mask (Same as in Loss)
                legal_mask = batch['legal_mask']
                masked_logits = outputs['policy'] + (1.0 - legal_mask) * -1e9
                
                pred_moves = torch.argmax(masked_logits, dim=1)
                target_moves = torch.argmax(batch['move_target'], dim=1)
                accuracy = (pred_moves == target_moves).float().mean()
        
        # Track metrics (unscaled)
        acc_policy += metrics['policy'].item()
        acc_value += metrics['value'].item()
        acc_mate += metrics['mate'].item()
        acc_accuracy += accuracy.item()
        
        loss = loss / GRAD_ACCUM
        loss.backward()
        
        total_loss_acc += loss.item() * GRAD_ACCUM
        
        # Step
        if (i + 1) % GRAD_ACCUM == 0:
            # --- Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

            # Linear Warmup
            step += 1
            if step < WARMUP_STEPS:
                lr_scale = step / WARMUP_STEPS
                for param_group in optimizer_muon.param_groups:
                    param_group['lr'] = MUON_LR * lr_scale
                for param_group in optimizer_adam.param_groups:
                    param_group['lr'] = ADAM_LR * lr_scale
            else:
                for param_group in optimizer_muon.param_groups:
                    param_group['lr'] = MUON_LR
                for param_group in optimizer_adam.param_groups:
                    param_group['lr'] = ADAM_LR

            optimizer_muon.step()
            optimizer_adam.step()
            
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()
            
            # Log
            if step % 10 == 0:
                elapsed = time.time() - start_time
                steps_in_interval = 10 * GRAD_ACCUM
                
                avg_total = total_loss_acc / steps_in_interval
                avg_policy = acc_policy / steps_in_interval
                avg_value = acc_value / steps_in_interval
                avg_mate = acc_mate / steps_in_interval
                avg_accuracy = acc_accuracy / steps_in_interval
                
                history['loss'].append(avg_total)
                history['policy_loss'].append(avg_policy)
                history['value_loss'].append(avg_value)
                history['mate_loss'].append(avg_mate)
                history['accuracy'].append(avg_accuracy)
                
                # WandB Log
                wandb.log({
                    "train/loss": avg_total,
                    "train/policy_loss": avg_policy,
                    "train/value_loss": avg_value,
                    "train/mate_loss": avg_mate,
                    "train/accuracy": avg_accuracy,
                    "step": step
                })
                
                print(f"Step {step}/{MAX_STEPS} | Total: {avg_total:.4f} | "
                      f"Policy: {avg_policy:.4f} | "
                      f"Value: {avg_value:.4f} | "
                      f"Mate: {avg_mate:.4f} | "
                      f"Acc: {avg_accuracy:.4f} | "
                      f"Time: {elapsed:.2f}s")
                
                total_loss_acc = 0.0
                acc_policy = 0.0
                acc_value = 0.0
                acc_mate = 0.0
                acc_accuracy = 0.0

            # Checkpoint
            if step % CHECKPOINT_INTERVAL == 0:
                print(f"Saving checkpoint at step {step}...")
                torch.save({
                    'step': step,
                    'micro_step': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                    'optimizer_adam_state_dict': optimizer_adam.state_dict(),
                    'history': history
                }, CHECKPOINT_PATH)
                
    print("Training Complete.")
    
    # --- Final Plotting (Replicated from utils.py) ---
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['loss'], label='Total Loss', linewidth=2)
    plt.plot(epochs, history['policy_loss'], label='Policy Loss', linestyle='--')
    plt.plot(epochs, history['value_loss'], label='Value Loss', linestyle='--')
    plt.plot(epochs, history['mate_loss'], label='Mate Loss', linestyle='--')
    plt.title('Training Losses')
    plt.xlabel('Step (x10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['accuracy'], label='Top-1 Accuracy', color='green')
    plt.title('Policy Accuracy')
    plt.xlabel('Step (x10)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = "training_metrics.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    # Log Final Plot to WandB
    wandb.log({"training_plot": wandb.Image(plot_path)})
    wandb.finish()
    
    torch.save(model.state_dict(), "chess_model_muon.pt")
    print("Model saved to chess_model_muon.pt")

if __name__ == "__main__":
    train_small_subset()
