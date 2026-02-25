
import torch
import torch.optim as optim
from model import ChessTransformer
from loss import ChessLoss
import sys

# Mock optim.Muon if it doesn't exist to simulate the error or lack thereof
if not hasattr(optim, 'Muon'):
    print("optim.Muon not found, mocking it for reproduction script if needed, or failing if that's the issue.")
    # In the real script, the user said "Muon is not the problem", implies it might exist or they added it.
    # Let's assume for this reproduction we want to see if the *rest* of the code works.
    # We will use SGD as a placeholder if Muon is missing, just to check the loop.
    class MockMuon(optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.0, eps=1e-8):
            defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eps=eps)
            super().__init__(params, defaults)
        def step(self, closure=None):
            pass
    optim.Muon = MockMuon

def test_training_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model Config
    model_cfg = {
        'vocab_size': 4544,
        'hidden_size': 128, # Reduced for speed
        'depth': 2,
        'num_heads': 4,
        'ff_dim': 512,
        'num_eval_bins': 128,
        'num_scratchpad': 8
    }
    
    model = ChessTransformer(**model_cfg).to(device)
    criterion = ChessLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy Batch
    B = 2
    batch = {
        'board': torch.randint(0, 13, (B, 8, 8), dtype=torch.long).to(device),
        'turn': torch.rand((B, 1)).to(device),
        'castling': torch.rand((B, 4)).to(device),
        'counters': torch.rand((B, 2)).to(device),
        'en_passant': torch.zeros((B, 1), dtype=torch.long).to(device),
        'move_target': torch.nn.functional.one_hot(torch.randint(0, 4544, (B,)), 4544).float().to(device),
        'legal_mask': torch.ones((B, 4544), dtype=torch.float).to(device),
        'eval_target': torch.softmax(torch.randn(B, 128), dim=1).to(device),
        'mate_target': torch.rand((B, 1)).to(device),
        'score_scalar': torch.rand((B, 1)).to(device)
    }
    
    model.train()
    
    print("Forward Pass...")
    outputs = model(batch)
    
    print("Loss Calculation...")
    loss, losses = criterion(outputs, batch)
    print(f"Loss: {loss.item()}")
    
    print("Backward Pass...")
    loss.backward()
    
    print("Optimizer Step...")
    optimizer.step()
    optimizer.zero_grad()
    
    print("Training step successful.")

if __name__ == "__main__":
    test_training_step()
