import pytest
import torch
import numpy as np
import orjson
import zstandard as zstd
import os
import chess
from dataset import ChessMoveTokenizer, FastChessDataset

@pytest.fixture
def tokenizer():
    """
    Fixture to provide an instance of ChessMoveTokenizer.
    This tokenizer is stateless, so one instance is sufficient for all tests.
    """
    return ChessMoveTokenizer()

@pytest.fixture
def dummy_zst_file(tmp_path):
    """
    Creates a temporary ZST file with diverse sample chess data.
    Includes:
    1. A standard opening position (CP evaluation).
    2. A winning mate position (Mate in 1).
    3. A losing mate position (Mate in -5).
    4. An end-game position (CP evaluation).
    """
    file_path = tmp_path / "test_data.jsonl.zst"
    
    sample_data = [
        # Case 1: Standard Opening (Black to move)
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "evals": [{"depth": 20, "pvs": [{"cp": 45, "line": "c7c5 g1f3"}]}]
        },
        # Case 2: Winning Mate (White to move, Mate in 1)
        {
            "fen": "7k/8/8/8/8/8/6R1/7K w - - 0 1", 
            "evals": [{"depth": 99, "pvs": [{"mate": 1, "line": "g2h2"}]}]
        },
        # Case 3: Losing Mate (White to move, Mate in -5)
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 
            "evals": [{"depth": 20, "pvs": [{"mate": -5, "line": "e2e4"}]}] 
        },
         # Case 4: Advanced position (White to move)
        {
            "fen": "r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQ - 4 6",
            "evals": [{"depth": 30, "pvs": [{"cp": 15, "line": "c1g5 h7h6"}]}]
        }
    ]
    
    cctx = zstd.ZstdCompressor()
    with open(file_path, "wb") as f:
        with cctx.stream_writer(f) as compressor:
            for item in sample_data:
                line = orjson.dumps(item) + b"\n"
                compressor.write(line)
                
    return str(file_path)

def test_tokenizer_structure(tokenizer):
    """
    Verifies the internal structure of the Tokenizer.
    - Vocab size must be within a reasonable range (~1900-4500) covering all legal chess moves.
    - Lookup table must be of shape (64, 64, 5) for (from, to, promotion).
    """
    assert tokenizer.vocab_size > 1800
    assert tokenizer.vocab_size < 5000
    assert tokenizer.lookup.shape == (64, 64, 5)

def test_tokenizer_encoding(tokenizer):
    """
    Verifies correct encoding and decoding of moves, including edge cases.
    """
    # 1. Standard Move: e2 -> e4
    e2e4_id = tokenizer.encode("e2e4")
    assert e2e4_id != -1
    assert tokenizer.decode(e2e4_id) == "e2e4"
    
    # 2. White Promotion: a7 -> a8 (Queen)
    prom_white = tokenizer.encode("a7a8q")
    assert prom_white != -1
    assert tokenizer.decode(prom_white) == "a7a8q"

    # 3. Black Promotion: h2 -> h1 (Knight)
    prom_black = tokenizer.encode("h2h1n")
    assert prom_black != -1
    assert tokenizer.decode(prom_black) == "h2h1n"
    
    # 4. Invalid Moves (Safety Checks)
    assert tokenizer.encode("") == -1
    assert tokenizer.encode("invalid") == -1
    assert tokenizer.encode("e2e5") != -1 # Valid string format, technically valid move key, though illegal on board
    assert tokenizer.encode("9999") == -1 # Invalid squares

def test_dataset_loading(dummy_zst_file):
    """
    Verifies that the FastChessDataset can be iterated and yields dictionaries 
    with the correct tensor shapes and data types.
    """
    dataset = FastChessDataset(dummy_zst_file)
    items = list(dataset)
    assert len(items) == 4 # We added a 4th item
    
    first_item = items[0]
    
    # Check Data Types
    assert isinstance(first_item['board'], torch.Tensor)
    assert isinstance(first_item['legal_mask'], torch.Tensor)
    assert isinstance(first_item['move_target'], torch.Tensor)
    assert isinstance(first_item['eval_target'], torch.Tensor)
    assert isinstance(first_item['mate_target'], torch.Tensor)
    
    # Check Shapes
    assert first_item['board'].shape == (8, 8)
    assert first_item['castling'].shape == (4,)
    assert first_item['en_passant'].shape == (1,)
    assert first_item['counters'].shape == (2,)
    assert first_item['turn'].shape == (1,)
    assert first_item['legal_mask'].shape == (dataset.tokenizer.vocab_size,)
    assert first_item['move_target'].shape == (dataset.tokenizer.vocab_size,)
    assert first_item['eval_target'].shape == (128,)
    assert first_item['mate_target'].shape == (1,)

def test_dataset_content_cp(dummy_zst_file):
    """
    Deep dive into a standard position (Case 1: Black to move, CP Eval).
    Verifies piece placement on the board tensor and correct target move encoding.
    """
    dataset = FastChessDataset(dummy_zst_file)
    items = list(dataset)
    item = items[0] 
    
    # 1. Board Representation Check
    # FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
    # Rank 8 (Index 0): r n b q k b n r (Black pieces: 10, 8, 9, 11, 12, 9, 8, 10)
    # Note: Our tensor mapping: 
    # White: P=1, N=2, B=3, R=4, Q=5, K=6
    # Black: P=7, N=8, B=9, R=10, Q=11, K=12
    
    # Check Black Rook at a8 (0,0)
    assert item['board'][0, 0] == 10 
    # Check Black King at e8 (0,4)
    assert item['board'][0, 4] == 12
    # Check White Pawn at e4 (Row 4, Col 4) -> Rank 4
    # Board rows go 0=Rank8 ... 7=Rank1.
    # Rank 4 is Row index 4.
    assert item['board'][4, 4] == 1 
    
    # 2. Turn Check
    # Black to move -> 0.0
    assert item['turn'].item() == 0.0

    # 3. Target Move Check
    # Best move is "c7c5"
    move_id = dataset.tokenizer.encode("c7c5")
    assert item['move_target'][move_id] == 1.0
    
    # 3. Legal Mask Check
    # "c7c5" is legal
    assert item['legal_mask'][move_id] == 1.0
    # "e2e4" is illegal (it's white's move but turn is black)
    # Actually, in this position, it is Black to move. "e2e4" is a White move.
    illegal_id = dataset.tokenizer.encode("e2e4")
    assert item['legal_mask'][illegal_id] == 0.0
    
    # 4. Mate Target Check
    # CP eval provided, so mate_target should be -1.0
    assert item['mate_target'].item() == -1.0

def test_dataset_content_mate_win(dummy_zst_file):
    """
    Verifies a position where the active player has a forced winning mate (Case 2).
    """
    dataset = FastChessDataset(dummy_zst_file)
    items = list(dataset)
    item = items[1] # Mate in 1
    
    # 1. Mate Score Check
    # Winning Mate in 1 should be exactly 1.0
    assert torch.isclose(item['mate_target'], torch.tensor([1.0]), atol=1e-5)
    
    # Check Turn (White to move -> 1.0)
    assert item['turn'].item() == 1.0

    # 2. Eval Distribution Check
    # Distribution should be heavily skewed towards the max bin (winning).
    peak_idx = torch.argmax(item['eval_target']).item()
    assert peak_idx > 110 # Max bin is 127

def test_dataset_content_mate_loss(dummy_zst_file):
    """
    Verifies a position where the active player is being mated (Case 3).
    """
    dataset = FastChessDataset(dummy_zst_file)
    items = list(dataset)
    item = items[2] # Mate in -5
    
    # 1. Mate Score Check
    # Losing mate is treated as -1.0 (generic bad/loss)
    assert item['mate_target'].item() == -1.0
    
    # 2. Eval Distribution Check
    # Distribution should be skewed towards min bin (losing).
    peak_idx = torch.argmax(item['eval_target']).item()
    assert peak_idx < 15 # Min bin is 0

def test_dataset_robustness(dummy_zst_file):
    """
    Simulates iterating through the dataset to ensure no runtime errors
    occur during parsing or tensor creation.
    """
    dataset = FastChessDataset(dummy_zst_file)
    
    count = 0
    for item in dataset:
        assert item['board'].shape == (8, 8)
        count += 1
    
    assert count == 4
