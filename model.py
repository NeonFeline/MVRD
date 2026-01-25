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
        # W (gate) and V (value) combined in one layer for efficiency? 
        # But user specified W and V init explicitly. 
        # Usually implemented as: (xW) * SiLU(xV)
        self.w = nn.Linear(dim, hidden_dim, bias=bias)
        self.v = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        # x: [B, S, D]
        # SwiGLU(x) = (xW + b) * SiLU(xV + c)
        return self.w2(self.w(x) * F.silu(self.v(x)))

class ChessTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size=4544, # Move vocabulary
        embed_dim=512, 
        depth=24, 
        num_heads=8, 
        ff_dim=2048,
        num_eval_bins=128,
        num_scratchpad=8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_scratchpad = num_scratchpad
        
        # --- Embeddings ---
        # 1. Piece Embedding: 0=Empty, 1-6=White, 7-12=Black (13 total)
        self.piece_embedding = nn.Embedding(13, embed_dim)
        
        # 2. Positional Embedding: 64 squares (Learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        
        # 3. Auxiliary Tokens
        # Turn: 0 or 1
        self.turn_embedding = nn.Embedding(2, embed_dim)
        
        # Castling: 0 or 1 (We process 4 rights as 4 tokens)
        # Shared embedding for all 4 positions
        self.castling_embedding = nn.Embedding(2, embed_dim)
        # Positional embedding for the 4 castling tokens so the model knows which is which
        self.castling_pos_emb = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        
        # En Passant: 0-64
        self.ep_embedding = nn.Embedding(65, embed_dim)
        
        # Counters: Scalar -> Vector
        self.counter_proj = nn.Linear(1, embed_dim)
        # Learned positional tags for [Halfmove, Fullmove]
        self.counter_pos_emb = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
        
        # 4. Scratchpad Tokens (Learned Constants)
        self.scratchpad = nn.Parameter(torch.randn(1, num_scratchpad, embed_dim) * 0.02)
        
        # 5. Output Token (Dedicated CLS token)
        self.output_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # --- Transformer Encoder ---
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(depth)
        ])
        
        self.final_norm = RMSNorm(embed_dim)
        
        # --- Heads ---
        # 1. Policy Head (Move Prediction)
        self.policy_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 2. Value Head (CP Distribution)
        self.value_head = nn.Linear(embed_dim, num_eval_bins, bias=False)
        
        # 3. Mate Head (Win probability / Closeness)
        self.mate_head = nn.Linear(embed_dim, 1, bias=False)
        
        # --- Initialization ---
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

    def forward(self, batch):
        # Extract inputs
        board = batch['board']          # [B, 8, 8]
        turn = batch['turn']            # [B, 1] (Float 0.0/1.0)
        castling = batch['castling']    # [B, 4] (Float 0.0/1.0)
        counters = batch['counters']    # [B, 2] (Float)
        ep = batch['en_passant']        # [B, 1] (Int64)
        
        B = board.shape[0]
        dev = board.device
        
        # --- 1. Tokenize Board ---
        board_flat = board.view(B, 64)
        x_board = self.piece_embedding(board_flat) # [B, 64, D]
        x_board = x_board + self.pos_embedding
        
        # --- 2. Tokenize Aux Features ---
        
        # Turn Token
        # Convert float 0.0/1.0 back to long for embedding
        turn_long = turn.long().squeeze(1) # [B]
        x_turn = self.turn_embedding(turn_long).unsqueeze(1) # [B, 1, D]
        
        # Castling Tokens
        # [B, 4] float -> long
        cast_long = castling.long() 
        x_cast = self.castling_embedding(cast_long) # [B, 4, D]
        x_cast = x_cast + self.castling_pos_emb
        
        # En Passant Token
        ep_sq = ep.squeeze(1) # [B]
        x_ep = self.ep_embedding(ep_sq).unsqueeze(1) # [B, 1, D]
        
        # Counter Tokens
        # [B, 2] -> [B, 2, 1] -> [B, 2, D]
        x_count = self.counter_proj(counters.unsqueeze(-1))
        x_count = x_count + self.counter_pos_emb
        
        # Scratchpad Tokens
        # Expand [1, N, D] to [B, N, D]
        x_scratch = self.scratchpad.expand(B, -1, -1)
        
        # Output Token
        # Expand [1, 1, D] to [B, 1, D]
        x_out = self.output_token.expand(B, -1, -1)
        
        # --- 3. Concatenate Sequence ---
        # Order: [Output(1), Turn(1), Cast(4), EP(1), Count(2), Scratch(N), Board(64)]
        # Total Len = 1 + 1 + 4 + 1 + 2 + 8 + 64 = 81 tokens
        x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
        
        # --- 4. Transformer ---
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        
        # --- 5. Heads ---
        # Use the dedicated Output Token (Index 0)
        cls_token = x[:, 0, :]
        
        policy_logits = self.policy_head(cls_token)
        value_logits = self.value_head(cls_token)
        mate_logits = self.mate_head(cls_token)
        
        return {
            'policy': policy_logits,
            'value': value_logits,
            'mate': mate_logits
        }

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_dim)

    def forward(self, x):
        # Pre-Norm
        # Attention
        x_norm = self.norm1(x)
        # T-Fixup style: x + attn(norm(x))
        # Note: nn.MultiheadAttention returns (output, weights)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm) 
        x = x + attn_out
        
        # FFN
        x_norm = self.norm2(x)
        x = x + self.ff(x_norm)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Smoke Test
    model = ChessTransformer()
    print(f"Model Parameters: {count_parameters(model):,}")
    
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
    print("Mate Shape:", out['mate'].shape)
