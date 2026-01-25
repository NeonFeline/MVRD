import torch
from dataset import FastChessDataset
from torch.utils.data import DataLoader

def check_target_legality():
    DATA_PATH = "dataset/data/lichess_db_eval.jsonl.zst"
    dataset = FastChessDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=1024, num_workers=4)
    
    print("Checking dataset for illegal targets...")
    
    count_illegal = 0
    total = 0
    
    for i, batch in enumerate(loader):
        # move_target is one-hot [B, Vocab]
        # legal_mask is binary [B, Vocab]
        
        # We want to check if (move_target == 1) AND (legal_mask == 0) anywhere
        
        # Element-wise: move_target * (1 - legal_mask)
        # If this is 1 anywhere, it means we have a target that is considered illegal.
        
        illegal_targets = batch['move_target'] * (1.0 - batch['legal_mask'])
        sum_illegal = illegal_targets.sum()
        
        if sum_illegal > 0:
            indices = torch.nonzero(illegal_targets, as_tuple=True)
            print(f"Batch {i}: Found {sum_illegal.item()} illegal targets!")
            count_illegal += sum_illegal.item()
            
            # Print first fail
            b_idx = indices[0][0].item()
            vocab_idx = indices[1][0].item()
            print(f"  Sample {b_idx}, Token {vocab_idx}")
            
            # Can we see the FEN? No easily available here without modifying dataset yield.
            
        total += batch['move_target'].shape[0]
        
        if i > 20: break
        
    print(f"Checked {total} samples. Found {count_illegal} mismatches.")

if __name__ == "__main__":
    check_target_legality()
