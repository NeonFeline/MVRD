import torch
from dataset import FastChessDataset
from model import ChessTransformer
from loss import ChessLoss
from torch.utils.data import DataLoader

def test_loss_function():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Setup
    DATA_PATH = "dataset/data/lichess_db_eval.jsonl.zst"
    BATCH_SIZE = 4
    
    dataset = FastChessDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model = ChessTransformer().to(device)
    criterion = ChessLoss()
    
    # 2. Get Batch
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # 3. Forward
    outputs = model(batch)
    
    # 4. Compute Loss
    total_loss, metrics = criterion(outputs, batch)
    
    print("\n--- Loss Verification ---")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Policy Loss: {metrics['policy'].item():.4f}")
    print(f"Value Loss: {metrics['value'].item():.4f}")
    print(f"Mate Loss: {metrics['mate'].item():.4f}")
    
    # Check gradients
    total_loss.backward()
    print("Backward pass successful.")

if __name__ == "__main__":
    test_loss_function()

