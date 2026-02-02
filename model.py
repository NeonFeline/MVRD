import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch, seq, dim]
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w = nn.Linear(dim, hidden_dim, bias=bias)
        self.v = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w2(self.w(x) * F.silu(self.v(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSNorm(dim)
        
        # Layer-specific Relative Positional Bias Parameters
        # Range: -7 to +7 (15 indices) per head
        self.rel_rank_embed = nn.Parameter(torch.randn(15, num_heads) * 0.02)
        self.rel_file_embed = nn.Parameter(torch.randn(15, num_heads) * 0.02)
        
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_dim)

    def forward(self, x, rank_diff, file_diff):
        # rank_diff, file_diff: [64, 64] passed from global context
        
        # Construct bias for this layer
        # [15, H] -> [64, 64, H]
        r_bias = self.rel_rank_embed[rank_diff]
        f_bias = self.rel_file_embed[file_diff]
        
        # [H, 64, 64]
        board_bias = (r_bias + f_bias).permute(2, 0, 1)
        
        # Construct Full Mask
        B, seq_len, _ = x.shape
        attn_bias = torch.zeros(self.num_heads, seq_len, seq_len, device=x.device, dtype=x.dtype)
        start_idx = seq_len - 64
        
        # Fill board bias
        attn_bias[:, start_idx:, start_idx:] = board_bias
        
        # Expand for Batch: [B*H, Seq, Seq]
        # This materialization might be costly if B is huge, but necessary for MHA API.
        attn_bias = attn_bias.repeat(B, 1, 1)
        
        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_bias)
        x = x + attn_out
        
        # FFN
        x_norm = self.norm2(x)
        x = x + self.ff(x_norm)
        return x

class ChessTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size=4544,
        hidden_size=768, 
        depth=24, 
        num_heads=12, 
        ff_dim=2048,
        num_eval_bins=128,
        num_scratchpad=8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_scratchpad = num_scratchpad
        self.num_heads = num_heads
        
        # --- Embeddings ---
        self.piece_embedding = nn.Embedding(13, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, hidden_size) * 0.02)
        
        # Precompute relative indices for 8x8 board
        coords = torch.arange(64)
        ranks = coords // 8
        files = coords % 8
        rank_diff = ranks[:, None] - ranks[None, :] + 7 # 0..14
        file_diff = files[:, None] - files[None, :] + 7 # 0..14
        self.register_buffer('rank_diff', rank_diff)
        self.register_buffer('file_diff', file_diff)
        
        # --- Aux Tokens ---
        self.turn_embedding = nn.Embedding(2, hidden_size)
        self.castling_embedding = nn.Embedding(2, hidden_size)
        self.castling_pos_emb = nn.Parameter(torch.randn(1, 4, hidden_size) * 0.02)
        self.ep_embedding = nn.Embedding(65, hidden_size)
        self.counter_proj = nn.Linear(1, hidden_size)
        self.counter_pos_emb = nn.Parameter(torch.randn(1, 2, hidden_size) * 0.02)
        self.scratchpad = nn.Parameter(torch.randn(1, num_scratchpad, hidden_size) * 0.02)
        self.output_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # --- Transformer Encoder ---
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_dim)
            for _ in range(depth)
        ])
        
        self.final_norm = RMSNorm(hidden_size)
        
        # --- Heads ---
        # 1. Policy Head: Attention Pooling over Board Tokens
        self.policy_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.policy_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 2. Value Heads
        self.value_head = nn.Linear(hidden_size, num_eval_bins, bias=False)
        self.value_scalar_head = nn.Linear(hidden_size, 1, bias=False)
        self.mate_head = nn.Linear(hidden_size, 1, bias=False)
        
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
        nn.init.xavier_uniform_(self.policy_attn.out_proj.weight, gain=scale_factor)

    def forward(self, batch):
        board = batch['board']
        turn = batch['turn']
        castling = batch['castling']
        counters = batch['counters']
        ep = batch['en_passant']
        
        B = board.shape[0]
        
        # 1. Embeddings
        board_flat = board.view(B, 64)
        x_board = self.piece_embedding(board_flat) + self.pos_embedding
        
        turn_long = turn.long().squeeze(1)
        x_turn = self.turn_embedding(turn_long).unsqueeze(1)
        
        cast_long = castling.long() 
        x_cast = self.castling_embedding(cast_long) + self.castling_pos_emb
        
        ep_sq = ep.squeeze(1)
        x_ep = self.ep_embedding(ep_sq).unsqueeze(1)
        
        x_count = self.counter_proj((counters / 100.0).unsqueeze(-1)) + self.counter_pos_emb
        
        x_scratch = self.scratchpad.expand(B, -1, -1)
        x_out = self.output_token.expand(B, -1, -1)
        
        # 2. Concat
        x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
        
        # 3. Transformer Layers
        # Pass rank_diff and file_diff to each layer
        for layer in self.layers:
            x = layer(x, self.rank_diff, self.file_diff)
            
        x = self.final_norm(x)
        
        # 4. Heads
        board_out = x[:, -64:, :]
        pol_query = self.policy_query.expand(B, -1, -1)
        pol_pooled, _ = self.policy_attn(pol_query, board_out, board_out)
        pol_pooled = pol_pooled.squeeze(1)
        policy_logits = self.policy_head(pol_pooled)
        
        cls_token = x[:, 0, :]
        value_logits = self.value_head(cls_token)
        value_scalar = self.value_scalar_head(cls_token)
        mate_logits = self.mate_head(cls_token)
        
        return {
            'policy': policy_logits,
            'value': value_logits,
            'value_scalar': value_scalar,
            'mate': mate_logits
        }

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