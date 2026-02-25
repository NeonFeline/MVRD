import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessLoss(nn.Module):
    def __init__(self, policy_weight=1.0, value_weight=1.0, value_scalar_weight=1.0, mate_weight=1.0):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.value_scalar_weight = value_scalar_weight
        self.mate_weight = mate_weight
        
        # Standard losses
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): {'policy': logits, 'value': logits, 'value_scalar': scalar, 'mate': scalar}
            batch (dict): Batch data from FastChessDataset
        """
        losses = {}
        
        # 1. Policy Loss (Cross Entropy with Legal Move Masking)
        policy_logits = outputs['policy'] # [B, Vocab]
        target_move_onehot = batch['move_target'] # [B, Vocab]
        legal_mask = batch['legal_mask'] # [B, Vocab]
        
        # Safety: Ensure target is always considered legal
        # Fixes dataset/tokenizer mismatches that cause infinite loss
        legal_mask = torch.max(legal_mask, target_move_onehot)
        
        # Apply Mask: Set illegal moves to -inf
        # legal_mask is 1.0 for legal, 0.0 for illegal.
        masked_logits = policy_logits.masked_fill(legal_mask == 0.0, float('-inf'))
        
        # Target is one-hot, so we use indices for CrossEntropy or direct softmax
        # Since target is one-hot, easiest is: - sum(target * log_softmax(logits))
        # But standard CrossEntropyLoss expects class indices.
        # Let's use indices.
        target_indices = torch.argmax(target_move_onehot, dim=1) # [B]
        
        policy_loss = F.cross_entropy(masked_logits, target_indices)
        losses['policy'] = policy_loss
        
        # 2. Value Loss (KL Divergence)
        # Model outputs logits for 128 bins. Target is probability dist.
        value_logits = outputs['value'] # [B, 128]
        value_target = batch['eval_target'] # [B, 128]
        
        # KLDiv expects log_probs as input
        value_log_probs = F.log_softmax(value_logits, dim=1)
        value_loss = self.kl_loss(value_log_probs, value_target)
        losses['value'] = value_loss
        
        # 3. Value Scalar Loss (MSE)
        val_scalar_pred = outputs['value_scalar'] # [B, 1]
        
        # Check if score_scalar is in batch (backward compatibility/safety)
        if 'score_scalar' in batch:
            val_scalar_target = batch['score_scalar'] # [B, 1]
            val_scalar_loss = self.mse_loss(val_scalar_pred, val_scalar_target)
        else:
            # Fallback if dataset not updated (shouldn't happen in this flow)
            val_scalar_loss = torch.tensor(0.0, device=val_scalar_pred.device)
            
        losses['value_scalar'] = val_scalar_loss
        
        # 4. Mate Loss (MSE)
        # Target is scalar [-1, 1]. Model output is scalar.
        mate_pred = outputs['mate'] # [B, 1]
        mate_target = batch['mate_target'] # [B, 1]
        
        mate_loss = self.mse_loss(mate_pred, mate_target)
        losses['mate'] = mate_loss
        
        # Total Loss
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss +
            self.value_scalar_weight * val_scalar_loss +
            self.mate_weight * mate_loss
        )
        losses['total'] = total_loss
        
        return total_loss, losses