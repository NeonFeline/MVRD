import torch
import numpy as np
import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from model import ChessTransformer
from dataset.chess_dataset import ChessMoveTokenizer

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# Load Tokenizer
tokenizer = ChessMoveTokenizer()

# Load Model
model = ChessTransformer(
    depth=24,
    embed_dim=768,
    num_heads=12,
    ff_dim=2048,
    vocab_size=tokenizer.vocab_size
)

checkpoint_path = "checkpoint_60000.pt"
if not os.path.exists(checkpoint_path):
    print(f"Warning: {checkpoint_path} not found. Ensure it exists in the current directory.")
else:
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle DDP prefix if present (though train_dist.py saves without it, just in case)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

model.to(device)
model.eval()

app = FastAPI()

# Enable CORS for the GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FenRequest(BaseModel):
    fen: str

def fen_to_batch(fen: str):
    board = chess.Board(fen)
    
    # 1. Board Tensor
    board_tensor = np.zeros((8, 8), dtype=np.int64)
    for sq, pc in board.piece_map().items():
        # 1-6=White, 7-12=Black
        val = pc.piece_type + (6 if pc.color == chess.BLACK else 0)
        # Matrix: Row 0 = Rank 8 (index 7 in numpy if 0-based from top), Row 7 = Rank 1
        # square_idx 0 = a1, 63 = h8
        # Rank 8 is squares 56-63. Rank 1 is 0-7.
        # We want visual board representation? 
        # FastChessDataset does: board_tensor[7 - (sq // 8), sq % 8] = val
        # This maps Rank 8 (sq // 8 = 7) to Row 0. Rank 1 (sq // 8 = 0) to Row 7. Correct.
        board_tensor[7 - (sq // 8), sq % 8] = val

    # 2. Auxiliary Features
    castling = np.array([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK))
    ], dtype=np.float32)

    ep_val = 0
    if board.ep_square is not None:
        ep_val = board.ep_square + 1
    ep_square = np.array([ep_val], dtype=np.int64)

    counters = np.array([board.halfmove_clock, board.fullmove_number], dtype=np.float32)

    turn = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)

    # Convert to Tensor and add Batch Dimension
    return {
        'board': torch.from_numpy(board_tensor).unsqueeze(0).to(device),
        'castling': torch.from_numpy(castling).unsqueeze(0).to(device),
        'en_passant': torch.from_numpy(ep_square).unsqueeze(0).to(device),
        'counters': torch.from_numpy(counters).unsqueeze(0).to(device),
        'turn': torch.from_numpy(turn).unsqueeze(0).to(device)
    }

@app.post("/move/")
async def get_move(request: FenRequest):
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    if board.is_game_over():
         return {"best_move": request.fen}

    batch = fen_to_batch(request.fen)

    with torch.no_grad():
        # Ensure mixed precision if used during training (bfloat16)
#        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs = model(batch)
        policy_logits = outputs['policy'][0] # [VocabSize]

    # Mask Illegal Moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {"best_move": request.fen} # Stalemate or Checkmate handled by is_game_over check usually

    # Create a mask of -inf
    masked_logits = torch.full_like(policy_logits, -float('inf'))
    
    legal_indices = []
    for move in legal_moves:
        uci = move.uci()
        idx = tokenizer.encode(uci)
        if idx != -1:
            legal_indices.append(idx)
    
    if not legal_indices:
        # Fallback: Pick a random legal move if tokenization fails for all (unlikely)
        import random
        random_move = random.choice(legal_moves)
        board.push(random_move)
        return {"best_move": board.fen()}

    # Set legal moves to their actual logits
    masked_logits[legal_indices] = policy_logits[legal_indices]

    # Greedy decoding (Best Move)
    best_idx = torch.argmax(masked_logits).item()
    move_str = tokenizer.decode(best_idx)
    
    if not move_str:
        # Fallback
        random_move = random.choice(legal_moves)
        board.push(random_move)
        return {"best_move": board.fen()}

    # Apply move
    # move_str is UCI (e.g. "e2e4")
    move = chess.Move.from_uci(move_str)
    board.push(move)

    return {"best_move": board.fen()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9400)
