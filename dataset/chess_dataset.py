import orjson
import os
import zstandard as zstd
import io
import torch
import numpy as np
import chess
from torch.utils.data import IterableDataset, DataLoader
from typing import Generator, Dict, Any

class ChessMoveTokenizer:
    """O(1) move tokenizer using a precomputed lookup table."""
    def __init__(self):
        # Mapping: [from_sq][to_sq][promo_idx] 
        # promo_idx: 0=none, 1=q, 2=r, 3=b, 4=n
        self.lookup = np.full((64, 64, 5), -1, dtype=np.int32)
        self.id_to_move = {}
        self._build_vocabulary()
        self.vocab_size = len(self.id_to_move)

    def _build_vocabulary(self):
        idx = 0
        promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
        # 1. Standard moves
        for f in range(64):
            for t in range(64):
                if f == t: continue
                self.lookup[f, t, 0] = idx
                move_str = chess.SQUARE_NAMES[f] + chess.SQUARE_NAMES[t]
                self.id_to_move[idx] = move_str
                idx += 1
        # 2. Promotions
        for f in range(64):
            for t in range(64):
                if f == t: continue
                r1, r2 = f // 8, t // 8
                if (r1 == 6 and r2 == 7) or (r1 == 1 and r2 == 0):
                    for char, p_idx in promo_map.items():
                        self.lookup[f, t, p_idx] = idx
                        move_str = chess.SQUARE_NAMES[f] + chess.SQUARE_NAMES[t] + char
                        self.id_to_move[idx] = move_str
                        idx += 1

    def encode(self, move_str):
        if not move_str or len(move_str) < 4: return -1
        try:
            f = chess.SQUARE_NAMES.index(move_str[:2])
            t = chess.SQUARE_NAMES.index(move_str[2:4])
        except ValueError:
            return -1
            
        p = 0
        if len(move_str) > 4:
            p = {'q':1, 'r':2, 'b':3, 'n':4}.get(move_str[4], 0)
        return self.lookup[f, t, p]
        
    def decode(self, idx):
        return self.id_to_move.get(idx, None)

class FastChessDataset(IterableDataset):
    def __init__(self, file_path, num_bins=128, skip=0, limit=None, rank=0, world_size=1):
        self.file_path = file_path
        self.tokenizer = ChessMoveTokenizer()
        self.num_bins = num_bins
        self.skip = skip
        self.limit = limit
        self.rank = rank
        self.world_size = world_size
        # Pre-calculate Gaussian x-axis
        self.bin_x = np.arange(num_bins, dtype=np.float32)

    def _create_eval_dist(self, cp, mate, sigma=2.0):
        target = 0.0
        if mate is not None:
            target = 1500.0 if mate > 0 else -1500.0
        else:
            target = float(cp or 0)
        
        mu = ((np.clip(target, -1500, 1500) + 1500) / 3000.0) * (self.num_bins - 1)
        dist = np.exp(-0.5 * ((self.bin_x - mu) / sigma) ** 2)
        sum_val = dist.sum()
        return dist / sum_val if sum_val > 0 else dist

    def __iter__(self):
        if not os.path.exists(self.file_path):
            return

        # Handle Worker Sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process loading (or main process)
            global_rank = self.rank
            global_world_size = self.world_size
        else:
            # Multi-process loading via DataLoader num_workers
            # We treat each worker as a separate entity in the global pool
            global_rank = self.rank * worker_info.num_workers + worker_info.id
            global_world_size = self.world_size * worker_info.num_workers

        count = 0
        line_idx = 0
        skipped = 0
        
        dctx = zstd.ZstdDecompressor()
        with open(self.file_path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                # Wrap in BufferedReader for efficient readline
                buffered_reader = io.BufferedReader(reader)
                while True:
                    line_bytes = buffered_reader.readline()
                    if not line_bytes:
                        break

                    current_line = line_idx
                    line_idx += 1
                    
                    # Sharding
                    if current_line % global_world_size != global_rank:
                        continue

                    # Skip logic
                    if skipped < self.skip:
                        skipped += 1
                        continue
                        
                    # Limit logic
                    if self.limit is not None and count >= self.limit:
                        break

                    try:
                        line = line_bytes.decode('utf-8')
                        data = orjson.loads(line)
                    except (UnicodeDecodeError, orjson.JSONDecodeError):
                        continue
                        
                    fen = data.get('fen')
                    evals = data.get('evals', [])
                    if not fen or not evals: continue
                    
                    best_eval = max(evals, key=lambda x: x.get('depth', 0))
                    pvs = best_eval.get('pvs', [])
                    if not pvs: continue
                    pv = pvs[0]
                    
                    # 1. Fast Board Parse
                    board = chess.Board(fen)
                    # 0=Empty, 1-6=White, 7-12=Black
                    board_tensor = np.zeros((8, 8), dtype=np.int64)
                    
                    # python-chess piece_map returns {square_idx: Piece}
                    # square_idx 0 = a1, 63 = h8
                    # Matrix: Row 0 = Rank 8, Row 7 = Rank 1
                    for sq, pc in board.piece_map().items():
                        val = pc.piece_type + (6 if pc.color == chess.BLACK else 0)
                        board_tensor[7 - (sq // 8), sq % 8] = val

                    # Auxiliary Features
                    # Castling: [WK, WQ, BK, BQ]
                    castling = np.array([
                        float(board.has_kingside_castling_rights(chess.WHITE)),
                        float(board.has_queenside_castling_rights(chess.WHITE)),
                        float(board.has_kingside_castling_rights(chess.BLACK)),
                        float(board.has_queenside_castling_rights(chess.BLACK))
                    ], dtype=np.float32)
                    
                    # En Passant: Square index (0-63) shifted to (1-64). 0 means None.
                    # This makes it compatible with nn.Embedding(65, ...).
                    ep_val = 0
                    if board.ep_square is not None:
                        ep_val = board.ep_square + 1
                    ep_square = np.array([ep_val], dtype=np.int64)
                    
                    # Game Counters
                    # halfmove_clock: moves since last pawn push or capture (for 50-move rule)
                    # fullmove_number: number of full moves.
                    counters = np.array([board.halfmove_clock, board.fullmove_number], dtype=np.float32)

                    # Turn (Side to Move)
                    # 1.0 for White, 0.0 for Black
                    turn = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)

                    # 2. Legal Mask
                    mask = np.zeros(self.tokenizer.vocab_size, dtype=np.float32)
                    legal_moves_set = set()
                    for m in board.legal_moves:
                        uci = m.uci()
                        legal_moves_set.add(uci)
                        idx = self.tokenizer.encode(uci)
                        if idx != -1: mask[idx] = 1.0

                    # 3. Targets
                    move_target = np.zeros(self.tokenizer.vocab_size, dtype=np.float32)
                    move_line = pv.get('line', '')
                    move_idx = -1
                    if move_line:
                        raw_move = move_line.split()[0]
                        
                        # Fix for dataset using e1h1 for O-O etc.
                        # Standard UCI uses e1g1 for O-O.
                        if raw_move not in legal_moves_set:
                            # Map King-Rook capture notation to King-Target notation
                            # White
                            if raw_move == 'e1h1' and 'e1g1' in legal_moves_set: raw_move = 'e1g1'
                            elif raw_move == 'e1a1' and 'e1c1' in legal_moves_set: raw_move = 'e1c1'
                            # Black
                            elif raw_move == 'e8h8' and 'e8g8' in legal_moves_set: raw_move = 'e8g8'
                            elif raw_move == 'e8a8' and 'e8c8' in legal_moves_set: raw_move = 'e8c8'
                            
                        move_idx = self.tokenizer.encode(raw_move)
                    
                    if move_idx != -1: 
                        move_target[move_idx] = 1.0
                    else:
                        # Skip if best move is somehow unencodable
                        continue
                    
                    # Mate Target
                    mate_val = -1.0
                    mate_score = pv.get('mate')
                    if mate_score is not None and mate_score > 0:
                        mate_val = 1.0 / (1.0 + 0.1 * (mate_score - 1))
                    
                    yield {
                        'board': torch.from_numpy(board_tensor),
                        'castling': torch.from_numpy(castling),
                        'en_passant': torch.from_numpy(ep_square),
                        'counters': torch.from_numpy(counters),
                        'turn': torch.from_numpy(turn),
                        'legal_mask': torch.from_numpy(mask),
                        'move_target': torch.from_numpy(move_target),
                        'eval_target': torch.from_numpy(self._create_eval_dist(pv.get('cp'), mate_score)),
                        'mate_target': torch.tensor([mate_val], dtype=torch.float32)
                    }
                    
                    count += 1

import random

class StreamingShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=10000):
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            if len(buffer) < self.buffer_size:
                buffer.append(item)
            else:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = item
        
        # Yield remaining buffer
        random.shuffle(buffer)
        for item in buffer:
            yield item

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        # Test basic
        ds = FastChessDataset(path)
        print(f"Iterating {path}...")
        for i, item in enumerate(ds):
            if i == 0:
                print("First item keys:", item.keys())
                print("Board shape:", item['board'].shape)
            if i >= 5: break
            
        # Test shuffle
        print("Testing Shuffle Wrapper...")
        ds_shuffled = StreamingShuffleDataset(FastChessDataset(path, limit=20), buffer_size=5)
        for i, item in enumerate(ds_shuffled):
            pass # Just ensure it runs
        print("Shuffle test passed.")