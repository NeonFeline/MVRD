import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from model import ChessTransformer
from loss import ChessLoss

@pytest.fixture
def dummy_batch():
    """Creates a dummy batch of data mimicking FastChessDataset output."""
    B = 4
    vocab_size = 4544
    return {
        'board': torch.randint(0, 13, (B, 8, 8), dtype=torch.long),
        'turn': torch.randint(0, 2, (B, 1), dtype=torch.float), # 0.0 or 1.0
        'castling': torch.randint(0, 2, (B, 4), dtype=torch.float),
        'en_passant': torch.randint(0, 65, (B, 1), dtype=torch.long),
        'counters': torch.rand((B, 2), dtype=torch.float),
        # One-hot target
        'move_target': torch.nn.functional.one_hot(torch.randint(0, vocab_size, (B,)), vocab_size).float(),
        'legal_mask': torch.ones((B, vocab_size), dtype=torch.float), # All legal for simplicity
        'eval_target': torch.softmax(torch.randn(B, 128), dim=1),
        'mate_target': torch.rand((B, 1)) * 2 - 1, # [-1, 1]
        'score_scalar': torch.rand((B, 1)) * 2 - 1 # [-1, 1]
    }

def test_model_forward(dummy_batch):
    """Verifies that the model accepts the batch and produces correct output shapes."""
    # num_heads=8 to be divisible by hidden_size=64
    model = ChessTransformer(depth=2, hidden_size=64, num_heads=8) 
    
    outputs = model(dummy_batch)
    
    B = dummy_batch['board'].shape[0]
    vocab_size = 4544
    
    assert 'policy' in outputs
    assert 'value' in outputs
    assert 'value_scalar' in outputs
    assert 'mate' in outputs
    
    assert outputs['policy'].shape == (B, vocab_size)
    assert outputs['value'].shape == (B, 128)
    assert outputs['value_scalar'].shape == (B, 1)
    assert outputs['mate'].shape == (B, 1)
    
    # Check for NaNs
    assert not torch.isnan(outputs['policy']).any()
    assert not torch.isnan(outputs['value']).any()
    assert not torch.isnan(outputs['value_scalar']).any()

def test_loss_function_computation(dummy_batch):
    """Verifies that the loss function calculates scalar losses correctly."""
    model = ChessTransformer(depth=2, hidden_size=64, num_heads=8)
    criterion = ChessLoss()
    
    outputs = model(dummy_batch)
    
    loss, metrics = criterion(outputs, dummy_batch)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # Scalar
    
    assert 'policy' in metrics
    assert 'value' in metrics
    assert 'value_scalar' in metrics
    assert 'mate' in metrics
    assert 'total' in metrics # Should match return
    
    assert metrics['policy'] > 0
    assert metrics['value'] > 0 
    assert metrics['value_scalar'] >= 0
    assert metrics['mate'] >= 0

def test_loss_masking_mechanism():
    """
    Verifies that the policy loss correctly ignores illegal moves.
    """
    B = 2
    vocab = 10
    
    # Fake batch
    batch = {
        'move_target': torch.zeros((B, vocab)),
        'legal_mask': torch.zeros((B, vocab)),
        'eval_target': torch.softmax(torch.randn(B, 128), dim=1), # Valid prob dist
        'mate_target': torch.tanh(torch.randn(B, 1)), # Valid range [-1, 1]
        'score_scalar': torch.tanh(torch.randn(B, 1))
    }
    
    # Set target to index 5
    batch['move_target'][:, 5] = 1.0
    
    # Set legal mask to ONLY index 0 (so target 5 is technically "illegal" in mask)
    batch['legal_mask'][:, 0] = 1.0
    
    # Model outputs
    outputs = {
        'policy': torch.randn(B, vocab),
        'value': torch.randn(B, 128),
        'value_scalar': torch.randn(B, 1),
        'mate': torch.randn(B, 1)
    }
    
    criterion = ChessLoss()
    
    loss, _ = criterion(outputs, batch)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_loss_gradients(dummy_batch):
    """Verifies that gradients flow back to the model."""
    model = ChessTransformer(depth=2, hidden_size=64, num_heads=8)
    criterion = ChessLoss()
    
    outputs = model(dummy_batch)
    loss, _ = criterion(outputs, dummy_batch)
    
    loss.backward()
    
    # Check if weights have gradients
    # Check Policy Head (Attention Pool)
    assert model.policy_head.weight.grad is not None
    assert model.policy_query.grad is not None
    
    # Check Value Scalar Head
    assert model.value_scalar_head.weight.grad is not None
    
    # Check Embeddings
    assert model.piece_embedding.weight.grad is not None