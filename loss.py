import torch
import torch.nn as nn
import torch.nn.functional as F
class ChessLoss(nn.Module):
    def __init__(self, policy_weight=1.0, value_weight=1.0, value_scalar_weight=1.0, mate_weight=1.0, temperature=2.0):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.value_scalar_weight = value_scalar_weight
        self.mate_weight = mate_weight
        self.temperature = temperature

        # Standard losses
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, batch, aux_multiplier=1.0):
        """
        Args:
            outputs (dict): {'policy': logits, 'value': logits, 'value_scalar': scalar, 'mate': scalar}
            batch (dict): Batch data from FastChessDataset
            aux_multiplier (float): Multiplier for auxiliary distillation loss (for annealing)
        """
        losses = {}
        T = self.temperature
        
        # 1. Policy Loss (Cross Entropy with Legal Move Masking)
        policy_logits = outputs['policy']
        target_move_onehot = batch['move_target']
        legal_mask = batch['legal_mask']
        legal_mask = torch.max(legal_mask, target_move_onehot)
        masked_logits = policy_logits.masked_fill(legal_mask == 0.0, float('-inf'))
        target_indices = torch.argmax(target_move_onehot, dim=1)
        
        policy_loss = F.cross_entropy(masked_logits, target_indices)
        losses['policy'] = policy_loss
        
        # 2. Value Loss (KL Divergence)
        value_logits = outputs['value']
        value_target = batch['eval_target']
        value_log_probs = F.log_softmax(value_logits, dim=1)
        value_loss = self.kl_loss(value_log_probs, value_target)
        losses['value'] = value_loss
        
        # 3. Value Scalar Loss (MSE)
        val_scalar_pred = outputs['value_scalar']
        if 'score_scalar' in batch:
            val_scalar_target = batch['score_scalar']
            val_scalar_loss = self.mse_loss(val_scalar_pred, val_scalar_target)
        else:
            val_scalar_loss = torch.tensor(0.0, device=val_scalar_pred.device)
        losses['value_scalar'] = val_scalar_loss
        
        # 4. Mate Loss (MSE)
        mate_pred = outputs['mate']
        mate_target = batch['mate_target']
        mate_loss = self.mse_loss(mate_pred, mate_target)
        losses['mate'] = mate_loss
        
        # Total Primary Loss
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss +
            self.value_scalar_weight * val_scalar_loss +
            self.mate_weight * mate_loss
        )
        losses['total'] = total_loss
        
        # 5. Scratchpad Orthogonality Loss (Small coefficient)
        # Encourages scratchpad tokens to be distinct.
        if 'scratchpad_hidden' in outputs:
            # x_scratch: [B, N_scratch, Hidden]
            x_scratch = outputs['scratchpad_hidden']
            B, N, H = x_scratch.shape
            if N > 1:
                # Normalize hidden states
                x_scratch_norm = F.normalize(x_scratch, p=2, dim=2)
                # Compute pairwise cosine similarity [B, N, N]
                sim_matrix = torch.bmm(x_scratch_norm, x_scratch_norm.transpose(1, 2))
                # Create mask for off-diagonal elements
                mask = torch.eye(N, device=x_scratch.device).unsqueeze(0).bool()
                # Sum of absolute off-diagonal similarities
                ortho_loss = sim_matrix.masked_fill(mask, 0).abs().mean()
                
                # Very small weight (0.01)
                total_loss = total_loss + 0.01 * ortho_loss
                losses['ortho_loss'] = ortho_loss
        
        # Self-Distillation Loss
        if outputs.get('intermediate_preds') and aux_multiplier > 0:
            dist_base_weights = {'25': 0.1, '50': 0.25, '75': 0.4}

            # Use large finite negative instead of -inf for distillation masking.
            # -inf causes NaN in KL div: exp(-inf) * (-inf - (-inf)) = 0 * NaN = NaN
            dist_masked_logits = policy_logits.masked_fill(legal_mask == 0.0, -1e9)

            teacher_pol_log_probs = F.log_softmax(dist_masked_logits.detach() / T, dim=1)
            teacher_val_log_probs = F.log_softmax(outputs['value'].detach() / T, dim=1)
            teacher_val_scalar = outputs['value_scalar'].detach()
            teacher_mate = outputs['mate'].detach()

            for key, base_weight in dist_base_weights.items():
                if key in outputs['intermediate_preds']:
                    inter_preds = outputs['intermediate_preds'][key]
                    weight = base_weight * aux_multiplier

                    # Policy distillation (Student masking + T)
                    inter_pol_logits = inter_preds['policy'].masked_fill(legal_mask == 0.0, -1e9)
                    inter_pol_log_probs = F.log_softmax(inter_pol_logits / T, dim=1)
                    # KL(P_teacher || P_student) scaled by T^2 as per Distillation paper
                    pol_dist_loss = F.kl_div(inter_pol_log_probs, teacher_pol_log_probs, log_target=True, reduction='batchmean') * (T**2)

                    # Value distillation (T)
                    inter_val_log_probs = F.log_softmax(inter_preds['value'] / T, dim=1)
                    val_dist_loss = F.kl_div(inter_val_log_probs, teacher_val_log_probs, log_target=True, reduction='batchmean') * (T**2)
                    
                    # Scalars (No T needed for MSE)
                    val_scalar_dist_loss = self.mse_loss(inter_preds['value_scalar'], teacher_val_scalar)
                    mate_dist_loss = self.mse_loss(inter_preds['mate'], teacher_mate)
                    
                    # Combined Distillation Loss for this depth
                    dist_loss = (
                        self.policy_weight * pol_dist_loss +
                        self.value_weight * val_dist_loss +
                        self.value_scalar_weight * val_scalar_dist_loss +
                        self.mate_weight * mate_dist_loss
                    )
                    
                    total_loss = total_loss + weight * dist_loss
                    losses[f'distill_{key}'] = dist_loss
            
            losses['total'] = total_loss
        
        return total_loss, losses