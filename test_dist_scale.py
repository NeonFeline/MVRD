import torch
from loss import ChessLoss
from model import ChessTransformer

def test():
    model = ChessTransformer(depth=4, num_scratchpad=2)
    criterion = ChessLoss()
    
    B = 4
    batch = {
        'board': torch.zeros((B, 8, 8), dtype=torch.long),
        'turn': torch.zeros((B, 1)),
        'castling': torch.zeros((B, 4)),
        'counters': torch.zeros((B, 2)),
        'en_passant': torch.zeros((B, 1), dtype=torch.long),
        'move_target': torch.zeros((B, 4544)),
        'legal_mask': torch.ones((B, 4544)),
        'eval_target': torch.softmax(torch.randn(B, 128), dim=1),
        'score_scalar': torch.randn(B, 1),
        'mate_target': torch.randn(B, 1)
    }
    batch['move_target'][:, 0] = 1.0  # Make target one-hot
    
    out = model(batch)
    loss, losses = criterion(out, batch)
    
    print("--- Losses ---")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.item():.4f}")

if __name__ == '__main__':
    test()
