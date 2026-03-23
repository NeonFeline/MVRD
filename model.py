import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.RMSNorm):
    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps=eps)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w = nn.Linear(dim, hidden_dim, bias=bias)
        self.v = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, seq_len, drop_path_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        self.drop_path_rate = drop_path_rate
        
        # Layer-specific Relative Positional Bias Parameters
        self.seq_bias_emb = nn.Parameter(torch.zeros(num_heads, seq_len, seq_len))
        self.rel_rank_embed = nn.Parameter(torch.randn(15, num_heads) * 0.02)
        self.rel_file_embed = nn.Parameter(torch.randn(15, num_heads) * 0.02)
        
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_dim)

    def forward(self, x, rank_diff, file_diff):
        # [H, 64, 64]
        r_bias = self.rel_rank_embed[rank_diff]
        f_bias = self.rel_file_embed[file_diff]
        board_bias = (r_bias + f_bias).permute(2, 0, 1)
        
        B, seq_len, _ = x.shape
        start_idx = seq_len - 64
        padded_board_bias = F.pad(board_bias, (start_idx, 0, start_idx, 0), "constant", 0.0)
        attn_bias = self.seq_bias_emb + padded_board_bias
        # Expand for Batch: [B*H, Seq, Seq] using expand (no copy) + reshape
        attn_bias = attn_bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.num_heads, seq_len, seq_len)
        
        # Attention with Stochastic Depth
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_bias)
        x = x + drop_path(attn_out, self.drop_path_rate, self.training)
        
        # FFN with Stochastic Depth
        x_norm = self.norm2(x)
        x = x + drop_path(self.ff(x_norm), self.drop_path_rate, self.training)
        return x

class ChessOutputHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads, num_eval_bins):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        
        # Policy Pooling (Global)
        self.policy_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.policy_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Value Pooling (Focused on "Extra" tokens)
        self.value_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.value_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.value_head = nn.Linear(hidden_size, num_eval_bins, bias=False)
        self.value_scalar_head = nn.Linear(hidden_size, 1, bias=False)
        self.mate_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, num_extra):
        """
        x: [B, Seq, Hidden]
        num_extra: int, number of non-board tokens at the start
        """
        B = x.shape[0]
        x = self.norm(x)
        
        # 1. Policy Head: Attention Pooling over ALL tokens (Board + Scratchpad)
        pol_query = self.policy_query.expand(B, -1, -1)
        pol_pooled, _ = self.policy_attn(pol_query, x, x)
        pol_pooled = pol_pooled.squeeze(1)
        policy_logits = self.policy_head(pol_pooled)
        
        # 2. Value Heads: Attention Pooling over only EXTRA tokens (0 to num_extra)
        # This forces the scratchpad/extra tokens to carry the evaluation state.
        x_extra = x[:, :num_extra, :]
        val_query = self.value_query.expand(B, -1, -1)
        val_pooled, _ = self.value_attn(val_query, x_extra, x_extra)
        val_pooled = val_pooled.squeeze(1)
        
        value_logits = self.value_head(val_pooled)
        value_scalar = self.value_scalar_head(val_pooled)
        mate_logits = self.mate_head(val_pooled)
        
        return {
            'policy': policy_logits,
            'value': value_logits,
            'value_scalar': value_scalar,
            'mate': mate_logits
        }

class ChessTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size=4544,
        hidden_size=768, 
        depth=24, 
        num_heads=12, 
        ff_dim=2048,
        num_eval_bins=128,
        num_scratchpad=8,
        aux_loss_only_extra_tokens=False,
        use_aux_loss=True,
        drop_path_rate=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_scratchpad = num_scratchpad
        self.num_extra = 9 + num_scratchpad
        self.num_heads = num_heads
        self.aux_loss_only_extra_tokens = aux_loss_only_extra_tokens
        self.use_aux_loss = use_aux_loss
        
        # --- Embeddings ---
        self.piece_embedding = nn.Embedding(13, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, hidden_size) * 0.02)
        
        coords = torch.arange(64)
        ranks = coords // 8
        files = coords % 8
        self.register_buffer('rank_diff', ranks[:, None] - ranks[None, :] + 7)
        self.register_buffer('file_diff', files[:, None] - files[None, :] + 7)
        
        # --- Aux Tokens ---
        self.turn_embedding = nn.Embedding(2, hidden_size)
        self.castling_embedding = nn.Embedding(2, hidden_size)
        self.castling_pos_emb = nn.Parameter(torch.randn(1, 4, hidden_size) * 0.02)
        self.ep_embedding = nn.Embedding(65, hidden_size)
        self.counter_proj = nn.Linear(1, hidden_size)
        self.counter_pos_emb = nn.Parameter(torch.randn(1, 2, hidden_size) * 0.02)
        self.scratchpad = nn.Parameter(torch.randn(1, num_scratchpad, hidden_size) * 0.02)
        self.output_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        seq_len = self.num_extra + 64
        
        # Stochastic Depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # --- Transformer Encoder ---
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_dim, seq_len=seq_len, drop_path_rate=dpr[i])
            for i in range(depth)
        ])
        
        # --- Heads ---
        self.final_head = ChessOutputHead(hidden_size, vocab_size, num_heads, num_eval_bins)
        
        self.aux_heads = nn.ModuleDict()
        if self.use_aux_loss:
            for depth_key in ['25', '50', '75']:
                self.aux_heads[depth_key] = ChessOutputHead(hidden_size, vocab_size, num_heads, num_eval_bins)
        
        self.apply(self._init_weights)
        self._special_initialization()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _special_initialization(self):
        scale_factor = (2 * self.depth) ** -0.5
        for block in self.layers:
            nn.init.xavier_uniform_(block.attn.out_proj.weight, gain=scale_factor)
            nn.init.normal_(block.ff.w2.weight, mean=0.0, std=0.02 * scale_factor)
        
        nn.init.xavier_uniform_(self.final_head.policy_attn.out_proj.weight, gain=scale_factor)
        nn.init.xavier_uniform_(self.final_head.value_attn.out_proj.weight, gain=scale_factor)
        for head in self.aux_heads.values():
            nn.init.xavier_uniform_(head.policy_attn.out_proj.weight, gain=scale_factor)
            nn.init.xavier_uniform_(head.value_attn.out_proj.weight, gain=scale_factor)

    def forward(self, batch):
        board = batch['board']
        turn = batch['turn']
        castling = batch['castling']
        counters = batch['counters']
        ep = batch['en_passant']
        B = board.shape[0]
        
        # 1. Embeddings
        x_board = self.piece_embedding(board.view(B, 64)) + self.pos_embedding
        x_turn = self.turn_embedding(turn.long().squeeze(1)).unsqueeze(1)
        x_cast = self.castling_embedding(castling.long()) + self.castling_pos_emb
        x_ep = self.ep_embedding(ep.squeeze(1)).unsqueeze(1)
        x_count = self.counter_proj((counters / 100.0).unsqueeze(-1)) + self.counter_pos_emb
        x_scratch = self.scratchpad.expand(B, -1, -1)
        x_out = self.output_token.expand(B, -1, -1)
        
        # 2. Concat
        x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
        
        # 3. Transformer Layers
        intermediate_preds = {}
        target_layers = {}
        if self.use_aux_loss:
            target_layers = {
                (self.depth // 4) - 1: '25', 
                (self.depth // 2) - 1: '50', 
                ((3 * self.depth) // 4) - 1: '75'
            }

        for i, layer in enumerate(self.layers):
            x = layer(x, self.rank_diff, self.file_diff)
            if i in target_layers:
                depth_key = target_layers[i]
                if self.aux_loss_only_extra_tokens:
                    # Truncate to just extra tokens (Bottleneck)
                    x_inter_used = x[:, :self.num_extra, :]
                else:
                    x_inter_used = x
                intermediate_preds[depth_key] = self.aux_heads[depth_key](x_inter_used, self.num_extra)
            
        # 4. Final Head
        out = self.final_head(x, self.num_extra)
        out['intermediate_preds'] = intermediate_preds
        
        # Return scratchpad hidden states for orthogonality loss
        # Scratchpad starts at index 9 (after 1 out, 1 turn, 4 cast, 1 ep, 2 count)
        out['scratchpad_hidden'] = x[:, 9 : 9 + self.num_scratchpad, :]
        
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Smoke Test with custom scratchpad
    num_scratch = 16
    model = ChessTransformer(num_scratchpad=num_scratch)
    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"Scratchpad shape: {model.scratchpad.shape}")
    assert model.scratchpad.shape[1] == num_scratch

    # Fake Batch
    B = 2
    dummy_batch = {
        'board': torch.zeros((B, 8, 8), dtype=torch.long),
        'turn': torch.rand((B, 1)),
        'castling': torch.rand((B, 4)),
        'counters': torch.rand((B, 2)),
        'en_passant': torch.zeros((B, 1), dtype=torch.long)
    }
    out = model(dummy_batch)
    print("Policy Shape:", out['policy'].shape)
    print("Value Shape:", out['value'].shape)
    print("Value Scalar Shape:", out['value_scalar'].shape)
    print("Mate Shape:", out['mate'].shape)