import time
import torch
import os
from torch.utils.data import DataLoader
from dataset.chess_dataset import FastChessDataset

def benchmark():
    DATA_PATH = "dataset/data/lichess_db_eval.jsonl.zst"
    BATCH_SIZE = 4096
    TOTAL_ITEMS = 100000
    
    if not os.path.exists(DATA_PATH):
        print(f"File {DATA_PATH} not found.")
        return

    print(f"Benchmarking REAL Dataset: {DATA_PATH}")
    print(f"Batch Size: {BATCH_SIZE}, Target Items: {TOTAL_ITEMS}")
    
    dataset = FastChessDataset(DATA_PATH)
    
    # Test with num_workers=0
    print("\nRunning Workers=0...")
    loader_0 = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    start = time.time()
    count = 0
    for batch in loader_0:
        count += len(batch['board'])
        if count >= TOTAL_ITEMS: break
    end = time.time()
    print(f"Workers=0: Processed {count} items in {end-start:.4f}s ({(count/(end-start)):.2f} items/s)")

    # Test with num_workers=8
    print("\nRunning Workers=8...")
    loader_8 = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)
    start = time.time()
    count = 0
    try:
        for batch in loader_8:
            count += len(batch['board'])
            if count >= TOTAL_ITEMS: break
        end = time.time()
        print(f"Workers=8: Processed {count} items in {end-start:.4f}s ({(count/(end-start)):.2f} items/s)")
    except Exception as e:
        print(f"Workers=8 failed: {e}")

    # Test with num_workers=16
    print("\nRunning Workers=16...")
    loader_16 = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16)
    start = time.time()
    count = 0
    try:
        for batch in loader_16:
            count += len(batch['board'])
            if count >= TOTAL_ITEMS: break
        end = time.time()
        print(f"Workers=16: Processed {count} items in {end-start:.4f}s ({(count/(end-start)):.2f} items/s)")
    except Exception as e:
        print(f"Workers=16 failed: {e}")

if __name__ == "__main__":
    benchmark()
