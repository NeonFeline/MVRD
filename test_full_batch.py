import torch
from torch.utils.data import DataLoader
from dataset import FastChessDataset
from model import ChessTransformer
import time

def test_full_batch():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Data
    DATA_PATH = "dataset/data/lichess_db_eval.jsonl.zst"
    BATCH_SIZE = 1024
    dataset = FastChessDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    # 3. Setup Model
    print("Initializing model...")
    model = ChessTransformer().to(device)
    model.eval() # Evaluation mode

    # 4. Get one batch
    print(f"Fetching batch of size {BATCH_SIZE}...")
    iterator = iter(loader)
    batch = next(iterator)

    # Move batch to device
    # Note: dataset returns a dict of tensors
    device_batch = {k: v.to(device) for k, v in batch.items()}

    # 5. Forward Pass
    print("Performing forward pass...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(device_batch)
    end_time = time.time()

    # 6. Verify Outputs
    print("\n--- Results ---")
    print(f"Forward pass took: {end_time - start_time:.4f} seconds")
    for k, v in outputs.items():
        print(f"Output '{k}' shape: {v.shape}")

    # Check for NaN (sanity check)
    has_nan = any(torch.isnan(v).any() for v in outputs.values())
    print(f"Contains NaNs: {has_nan}")

if __name__ == "__main__":
    test_full_batch()
