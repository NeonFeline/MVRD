import torch
from loss import ChessLoss

# create dummy
criterion = ChessLoss()
B = 2

# Test with sparse legal mask (most moves illegal) to trigger the -inf masking path
legal_mask = torch.zeros(B, 4544)
legal_mask[:, :20] = 1.0  # Only 20 legal moves out of 4544

move_target = torch.zeros(B, 4544)
move_target[:, 0] = 1.0

outputs = {
    'policy': torch.randn(B, 4544),
    'value': torch.randn(B, 128),
    'value_scalar': torch.randn(B, 1),
    'mate': torch.randn(B, 1),
    'intermediate_preds': {
        '25': {
            'policy': torch.randn(B, 4544),
            'value': torch.randn(B, 128),
            'value_scalar': torch.randn(B, 1),
            'mate': torch.randn(B, 1),
        },
        '50': {
            'policy': torch.randn(B, 4544),
            'value': torch.randn(B, 128),
            'value_scalar': torch.randn(B, 1),
            'mate': torch.randn(B, 1),
        },
        '75': {
            'policy': torch.randn(B, 4544),
            'value': torch.randn(B, 128),
            'value_scalar': torch.randn(B, 1),
            'mate': torch.randn(B, 1),
        },
    }
}
batch = {
    'move_target': move_target,
    'legal_mask': legal_mask,
    'eval_target': torch.softmax(torch.randn(B, 128), dim=1),
    'score_scalar': torch.randn(B, 1),
    'mate_target': torch.randn(B, 1)
}

loss, losses = criterion(outputs, batch)
print("=== Loss Results ===")
any_nan = False
for k, v in losses.items():
    val = v.item()
    nan_flag = " *** NaN! ***" if torch.isnan(v) else ""
    if torch.isnan(v):
        any_nan = True
    print(f"  {k}: {val}{nan_flag}")

print()
if any_nan:
    print("FAIL: NaN detected in losses!")
else:
    print("PASS: All losses are finite.")

