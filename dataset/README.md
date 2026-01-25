# Chess Dataset Module

This module provides a high-performance, streaming data loader for Chess Evaluation datasets (specifically Lichess-style JSONL files, including Zstandard compressed `.jsonl.zst`).

It is designed to feed PyTorch models with pre-processed tensors representing the board state, legal moves, and evaluation targets.

## Components

### 1. `FastChessDataset`

A PyTorch `IterableDataset` that streams chess positions from disk, tokenizes them, and returns ready-to-use tensors.

**Features:**
- **Streaming:** Reads line-by-line, supporting datasets larger than RAM.
- **On-the-fly Decompression:** Natively handles `.zst` files using `zstandard` with released GIL for parallelism.
- **Multiprocessing Friendly:** Designed to work with `DataLoader(num_workers=N)`.
- **Fast:** Uses `orjson` for JSON parsing and NumPy for vectorized tensor construction.

**Outputs (Dictionary):**
- `board` (Int64Tensor `[8, 8]`):
  - 8x8 Board representation.
  - Encoding: 0=Empty, 1-6=White(P,N,B,R,Q,K), 7-12=Black(P,N,B,R,Q,K).
  - Orientation: Row 0 is Rank 8 (Black side), Row 7 is Rank 1 (White side).
- `castling` (FloatTensor `[4]`):
  - Binary availability: `[WhiteKingside, WhiteQueenside, BlackKingside, BlackQueenside]`.
- `turn` (FloatTensor `[1]`):
  - Side to move.
  - `1.0`: White.
  - `0.0`: Black.
- `en_passant` (Int64Tensor `[1]`):
  - Square index for en passant availability.
  - `0`: No en passant available.
  - `1-64`: Square index + 1 (e.g., index 0 becomes 1).
  - Designed for direct use with `nn.Embedding(65, ...)`.
- `counters` (FloatTensor `[2]`):
  - `[halfmove_clock, fullmove_number]`.
- `legal_mask` (FloatTensor `[VocabSize]`):
  - Binary mask (1.0 or 0.0) indicating which moves are legal in the current position.
  - Used to mask logits during training.
- `move_target` (FloatTensor `[VocabSize]`):
  - One-hot encoded vector of the best move (Principal Variation).
- `eval_target` (FloatTensor `[128]`):
  - Gaussian probability distribution of the Centipawn (CP) evaluation.
  - Peaks at the evaluation value.
- `mate_target` (FloatTensor `[1]`):
  - Scalar value indicating forced mate status.
  - `1.0`: Winning Mate in 1.
  - `> 0`: Winning Mate (decays as distance increases: `1 / (1 + 0.1 * dist)`).
  - `-1.0`: No forced mate (CP eval) or Losing/Being Mated.

### 2. `ChessMoveTokenizer`

Handles the conversion between UCI move strings (e.g., "e2e4", "a7a8q") and integer IDs.

**Features:**
- **O(1) Encoding:** Uses a pre-computed NumPy lookup table `(64, 64, 5)` for instant mapping.
- **Vocabulary:** Automatically generates all possible pseudo-legal moves (~1900-4500 tokens).

## Usage

```python
from torch.utils.data import DataLoader
from dataset import FastChessDataset

# 1. Initialize Dataset
dataset = FastChessDataset("path/to/lichess_db_eval.jsonl.zst")

# 2. Create DataLoader
# num_workers > 0 enables parallel decompression and processing
loader = DataLoader(dataset, batch_size=256, num_workers=4)

# 3. Train
for batch in loader:
    board = batch['board']          # [B, 8, 8]
    legal_mask = batch['legal_mask'] # [B, Vocab]
    target_move = batch['move_target'] # [B, Vocab]
    
    # Forward pass...
```

## Requirements
- `torch`
- `numpy`
- `python-chess`
- `orjson`
- `zstandard`
