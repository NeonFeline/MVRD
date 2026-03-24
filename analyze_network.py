#!/usr/bin/env python3
"""
MVRD Chess Transformer - Neural Network Interpretability Analysis

Comprehensive analysis of extra tokens (output, turn, castling, en_passant,
counters, scratchpad) to understand what the network learned to do with them.

Sections:
  1. Embedding Analysis     - Are scratchpad embeddings ~0? Norms, cosine sim, PCA
  2. Attention Patterns     - Where do extra tokens attend? Do they get attended to?
  3. Activation Analysis    - How do activations evolve across layers?
  4. Noise Injection        - What happens when you add noise to scratchpad tokens?
  5. Masking (Ablation)     - What happens when you zero out extra tokens?
  6. Gradient Attribution   - XAI: which tokens matter most for policy/value?

Usage:
    python analyze_network.py --config configs/config_128_scratchpads.yaml \
                              --checkpoint checkpoints/.../final.pt \
                              --output analysis_output/
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import chess
import os
import sys
from pathlib import Path
from collections import OrderedDict
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from model import ChessTransformer
from dataset.chess_dataset import ChessMoveTokenizer


# ============================================================================
# Constants
# ============================================================================

FIXED_EXTRA_NAMES = [
    'OUT', 'TURN', 'WK_castle', 'WQ_castle', 'BK_castle', 'BQ_castle',
    'EN_PASS', 'HALF_CLK', 'FULL_CLK'
]

TEST_POSITIONS = OrderedDict({
    'Starting':   'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'Sicilian':   'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
    'Italian':    'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
    'Middlegame': 'r1bq1rk1/pppnnppp/4p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R1BQ1RK1 w - - 0 9',
    'Endgame':    '8/5k2/8/8/8/8/3K4/4Q3 w - - 0 1',
    'Tactical':   'r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/1BNP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8',
})

PIECE_SYMBOLS = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']


# ============================================================================
# Helpers
# ============================================================================

def get_token_names(num_scratchpad):
    """Full list of token names for the entire sequence."""
    names = list(FIXED_EXTRA_NAMES)
    for i in range(num_scratchpad):
        names.append(f'S{i}')
    for rank_idx in range(8):
        for file_idx in range(8):
            sq = chess.SQUARE_NAMES[(7 - rank_idx) * 8 + file_idx]
            names.append(sq)
    return names


def fen_to_batch(fen, device='cpu'):
    """Convert a FEN string to a model-compatible batch dict."""
    board = chess.Board(fen)

    board_tensor = np.zeros((8, 8), dtype=np.int64)
    for sq, pc in board.piece_map().items():
        val = pc.piece_type + (6 if pc.color == chess.BLACK else 0)
        board_tensor[7 - (sq // 8), sq % 8] = val

    castling = np.array([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK))
    ], dtype=np.float32)

    ep_val = 0
    if board.ep_square is not None:
        ep_val = board.ep_square + 1

    counters = np.array([board.halfmove_clock, board.fullmove_number], dtype=np.float32)
    turn = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)

    tokenizer = ChessMoveTokenizer()
    mask = np.zeros(tokenizer.vocab_size, dtype=np.float32)
    for m in board.legal_moves:
        idx = tokenizer.encode(m.uci())
        if idx != -1:
            mask[idx] = 1.0

    batch = {
        'board':      torch.from_numpy(board_tensor).unsqueeze(0).to(device),
        'castling':   torch.from_numpy(castling).unsqueeze(0).to(device),
        'en_passant': torch.tensor([[ep_val]], dtype=torch.long, device=device),
        'counters':   torch.from_numpy(counters).unsqueeze(0).to(device),
        'turn':       torch.from_numpy(turn).unsqueeze(0).to(device),
        'legal_mask': torch.from_numpy(mask).unsqueeze(0).to(device),
    }
    return batch, board


def manual_embed(model, batch):
    """Reproduce the embedding step from model.forward()."""
    board    = batch['board']
    turn     = batch['turn']
    castling = batch['castling']
    counters = batch['counters']
    ep       = batch['en_passant']
    B = board.shape[0]

    x_board   = model.piece_embedding(board.view(B, 64)) + model.pos_embedding
    x_turn    = model.turn_embedding(turn.long().squeeze(1)).unsqueeze(1)
    x_cast    = model.castling_embedding(castling.long()) + model.castling_pos_emb
    x_ep      = model.ep_embedding(ep.squeeze(1)).unsqueeze(1)
    x_count   = model.counter_proj((counters / 100.0).unsqueeze(-1)) + model.counter_pos_emb
    x_scratch = model.scratchpad.expand(B, -1, -1)
    x_out     = model.output_token.expand(B, -1, -1)

    parts = dict(output=x_out, turn=x_turn, castling=x_cast,
                 en_passant=x_ep, counters=x_count, scratchpad=x_scratch, board=x_board)

    x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
    return x, parts


def manual_forward(model, x):
    """Run transformer layers + final head on a pre-embedded tensor."""
    for layer in model.layers:
        x = layer(x, model.rank_diff, model.file_diff)
    out = model.final_head(x, model.num_extra)
    return out, x


def get_policy_probs(out, legal_mask):
    logits = out['policy']
    masked = logits.masked_fill(legal_mask == 0, float('-inf'))
    return F.softmax(masked, dim=-1)


def get_value_probs(out):
    return F.softmax(out['value'], dim=-1)


# ============================================================================
# Section 1: Embedding Analysis
# ============================================================================

def analyze_embeddings(model, out_dir, device):
    print("\n" + "=" * 80)
    print("SECTION 1: EMBEDDING ANALYSIS")
    print("=" * 80)

    with torch.no_grad():
        scratchpad = model.scratchpad.squeeze(0)          # [N_scratch, H]
        output_tok = model.output_token.squeeze(0)        # [1, H]
        piece_emb  = model.piece_embedding.weight         # [13, H]

        N_scratch = scratchpad.shape[0]

        # ---- 1a  Norms ----
        output_norm   = output_tok.norm(dim=1).cpu().numpy()[0]
        piece_norms   = piece_emb.norm(dim=1).cpu().numpy()

        print(f"\n  --- Embedding Norms ---")
        print(f"  Output token norm:     {output_norm:.4f}")
        if N_scratch > 0:
            scratch_norms = scratchpad.norm(dim=1).cpu().numpy()
            print(f"  Scratchpad norms:      mean={scratch_norms.mean():.4f}  std={scratch_norms.std():.4f}  "
                  f"min={scratch_norms.min():.4f}  max={scratch_norms.max():.4f}")
            near_zero = scratch_norms.max() < 0.1
            print(f"\n  >> Are scratchpad embeddings near zero? "
                  f"{'YES' if near_zero else 'NO – they carry significant magnitude'}")
        else:
            scratch_norms = np.array([])
            print(f"  Scratchpad:            NONE (num_scratchpad=0)")
        print(f"  Piece embedding norms: mean={piece_norms.mean():.4f}  std={piece_norms.std():.4f}")
        print(f"  Empty square (idx 0):  {piece_norms[0]:.4f}")

        # ---- 1b  Cosine similarity of scratchpad tokens ----
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        if N_scratch > 1:
            scratch_norm_t = F.normalize(scratchpad, dim=1)
            scratch_cos = (scratch_norm_t @ scratch_norm_t.T).cpu().numpy()
            triu_vals = scratch_cos[np.triu_indices(N_scratch, k=1)]

            if N_scratch <= 32:
                im = axes[0].imshow(scratch_cos, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[0].set_title(f'Scratchpad Pairwise Cosine Similarity ({N_scratch} tokens)')
                axes[0].set_xlabel('Token'); axes[0].set_ylabel('Token')
                plt.colorbar(im, ax=axes[0])
            else:
                axes[0].hist(triu_vals, bins=60, edgecolor='black', alpha=0.7)
                axes[0].axvline(x=0, color='red', linestyle='--')
                axes[0].set_title(f'Scratchpad Cosine-Sim Distribution '
                                  f'(mean={triu_vals.mean():.3f}, std={triu_vals.std():.3f})')
                axes[0].set_xlabel('Cosine Similarity'); axes[0].set_ylabel('Count')
            print(f"  Mean off-diag cosine sim: {triu_vals.mean():.4f}")
        else:
            axes[0].text(0.5, 0.5, f'No scratchpad tokens (N={N_scratch})',
                        transform=axes[0].transAxes, ha='center', va='center', fontsize=14)
            axes[0].set_title('Scratchpad Cosine Similarity')

        # ---- 1c  Norm bar chart ----
        if N_scratch > 0:
            axes[1].bar(range(N_scratch), scratch_norms, color='steelblue', alpha=0.7)
        axes[1].axhline(y=output_norm, color='red', ls='--',
                        label=f'Output token ({output_norm:.3f})')
        axes[1].axhline(y=piece_norms.mean(), color='green', ls='--',
                        label=f'Avg piece emb ({piece_norms.mean():.3f})')
        axes[1].set_title('Scratchpad Token Embedding Norms')
        axes[1].set_xlabel('Index'); axes[1].set_ylabel('L2 Norm'); axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '1_embedding_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ---- 1d  PCA ----
        all_tokens = torch.cat([output_tok, scratchpad, piece_emb], dim=0).cpu().numpy()
        labels = (['OUT'] +
                  [f'S{i}' for i in range(N_scratch)] +
                  [PIECE_SYMBOLS[i] for i in range(13)])
        colors = (['red'] +
                  ['blue'] * N_scratch +
                  ['gray'] + ['green'] * 6 + ['orange'] * 6)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_tokens)

        fig, ax = plt.subplots(figsize=(14, 10))
        for i, (cx, cy) in enumerate(coords):
            sz = 40 if labels[i].startswith('S') else 100
            al = 0.5 if labels[i].startswith('S') else 1.0
            ax.scatter(cx, cy, c=colors[i], s=sz, alpha=al)
            if not labels[i].startswith('S') or i < 6 or i == N_scratch:
                ax.annotate(labels[i], (cx, cy), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)

        ax.set_title(f'PCA of Token Embeddings '
                     f'(explained var: {pca.explained_variance_ratio_.sum():.1%})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend(handles=[
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   ms=10, label='Output'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',  ms=10, label='Scratchpad'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', ms=10, label='White pieces'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',ms=10, label='Black pieces'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',  ms=10, label='Empty'),
        ], loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '1_embedding_pca.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Plots saved to {out_dir}/1_embedding_*.png")


# ============================================================================
# Section 2: Attention Pattern Analysis
# ============================================================================

def analyze_attention_patterns(model, out_dir, device):
    print("\n" + "=" * 80)
    print("SECTION 2: ATTENTION PATTERN ANALYSIS")
    print("=" * 80)

    num_extra  = model.num_extra
    num_scratch = model.num_scratchpad
    num_layers = len(model.layers)

    # --- Hook storage ---
    attn_storage = {}
    head_attn_storage = {}
    hooks = []

    for li, layer in enumerate(model.layers):
        key = f'layer_{li}'
        attn_storage[key] = None
        def make_hook(k):
            def fn(module, inp, out):
                attn_storage[k] = out[1].detach().cpu()  # [B, Seq, Seq]
            return fn
        hooks.append(layer.attn.register_forward_hook(make_hook(key)))

    def head_hook(name):
        def fn(module, inp, out):
            head_attn_storage[name] = out[1].detach().cpu()
        return fn
    hooks.append(model.final_head.policy_attn.register_forward_hook(head_hook('policy')))
    hooks.append(model.final_head.value_attn.register_forward_hook(head_hook('value')))

    # --- Forward on each position ---
    all_results = {}
    with torch.no_grad():
        for pos_name, fen in TEST_POSITIONS.items():
            batch, _ = fen_to_batch(fen, device)
            model(batch)
            all_results[pos_name] = {
                'layers': [attn_storage[f'layer_{i}'] for i in range(num_layers)],
                'policy_head': head_attn_storage.get('policy'),
                'value_head':  head_attn_storage.get('value'),
            }

    for h in hooks:
        h.remove()

    # --- Plot per position ---
    for pos_name in list(TEST_POSITIONS.keys())[:3]:
        res = all_results[pos_name]
        seq_len = res['layers'][0].shape[-1]

        # Compute group-level flows
        e2e = np.zeros(num_layers)
        e2b = np.zeros(num_layers)
        b2e = np.zeros(num_layers)
        b2b = np.zeros(num_layers)
        scratch_received_mean = np.zeros(num_layers)
        fixed_received_mean   = np.zeros(num_layers)
        board_received_mean   = np.zeros(num_layers)

        for li, w_t in enumerate(res['layers']):
            w = w_t[0].numpy()  # [Seq, Seq]  (query, key)
            e2e[li] = w[:num_extra, :num_extra].sum() / num_extra
            e2b[li] = w[:num_extra, num_extra:].sum() / num_extra
            b2e[li] = w[num_extra:, :num_extra].sum() / 64
            b2b[li] = w[num_extra:, num_extra:].sum() / 64

            scratch_received_mean[li] = w[:, 9:num_extra].mean() if num_scratch > 0 else 0
            fixed_received_mean[li]   = w[:, :9].mean()
            board_received_mean[li]   = w[:, num_extra:].mean()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attention Patterns – {pos_name}', fontsize=14)
        xl = range(num_layers)

        ax = axes[0, 0]
        ax.plot(xl, e2e, 'b-o', ms=3, label='Extra→Extra')
        ax.plot(xl, e2b, 'r-o', ms=3, label='Extra→Board')
        ax.plot(xl, b2e, 'g-o', ms=3, label='Board→Extra')
        ax.plot(xl, b2b, 'm-o', ms=3, label='Board→Board')
        ax.set_title('Group-Level Attention Flow'); ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Attn Weight'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(xl, scratch_received_mean, 'b-o', ms=3, label='Scratchpad')
        ax.plot(xl, fixed_received_mean,   'r-o', ms=3, label='Fixed Extra')
        ax.plot(xl, board_received_mean,   'g-o', ms=3, label='Board')
        ax.set_title('Mean Attention Received (as Key)')
        ax.set_xlabel('Layer'); ax.set_ylabel('Mean Weight'); ax.legend(); ax.grid(True, alpha=0.3)

        is_zero = scratch_received_mean.max() < 1e-3
        print(f"\n  [{pos_name}] Scratchpad attn received: "
              f"mean={scratch_received_mean.mean():.6f}, max={scratch_received_mean.max():.6f}  "
              f"→ {'NEAR ZERO' if is_zero else 'Non-trivial'}")

        # Per extra-token mean attention received (averaged over layers)
        ax = axes[1, 0]
        per_tok = np.zeros(num_extra)
        for w_t in res['layers']:
            per_tok += w_t[0, :, :num_extra].mean(dim=0).numpy()
        per_tok /= num_layers

        # Group scratchpads if too many
        if num_scratch <= 32:
            labels_e = FIXED_EXTRA_NAMES + [f'S{i}' for i in range(num_scratch)]
            vals_e = per_tok
        else:
            gsz = max(1, num_scratch // 16)
            labels_e = list(FIXED_EXTRA_NAMES)
            vals_e = list(per_tok[:9])
            for g in range(0, num_scratch, gsz):
                end = min(g + gsz, num_scratch)
                labels_e.append(f'S{g}:{end-1}')
                vals_e.append(per_tok[9 + g : 9 + end].mean())
            vals_e = np.array(vals_e)

        bar_colors = ['red'] + ['orange'] * 8 + ['steelblue'] * (len(vals_e) - 9)
        ax.bar(range(len(vals_e)), vals_e, color=bar_colors, alpha=0.7)
        ax.set_xticks(range(len(labels_e)))
        ax.set_xticklabels(labels_e, rotation=60, ha='right', fontsize=6)
        ax.set_title('Avg Attention Received per Extra Token'); ax.set_ylabel('Mean Weight')
        ax.grid(True, alpha=0.3, axis='y')

        # Board→Extra attention heatmap across layers
        ax = axes[1, 1]
        sel_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        b2e_map = np.zeros((len(sel_layers), num_extra))
        for si, li in enumerate(sel_layers):
            b2e_map[si] = res['layers'][li][0, num_extra:, :num_extra].mean(dim=0).numpy()
        im = ax.imshow(b2e_map, aspect='auto', cmap='viridis')
        ax.set_yticks(range(len(sel_layers)))
        ax.set_yticklabels([f'L{l}' for l in sel_layers])
        ax.set_title('Board→Extra Attention (mean over board queries)')
        ax.set_xlabel('Extra Token Index'); plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'2_attention_{pos_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # --- Output-head attention ---
    for pos_name in ['Starting', 'Tactical']:
        res = all_results[pos_name]
        seq_len = res['layers'][0].shape[-1]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Policy head
        if res['policy_head'] is not None:
            pw = res['policy_head'][0, 0].numpy()
            ax = axes[0]
            ax.bar(range(num_extra), pw[:num_extra], color='red', alpha=0.7, label='Extra')
            ax.bar(range(num_extra, seq_len), pw[num_extra:], color='steelblue', alpha=0.7, label='Board')
            ax.set_title(f'Policy Head Attention – {pos_name}')
            ax.set_xlabel('Token Index'); ax.set_ylabel('Weight'); ax.legend()

        # Value head
        if res['value_head'] is not None:
            vw = res['value_head'][0, 0].numpy()
            ax = axes[1]
            vc = ['red'] + ['orange'] * 8 + ['steelblue'] * num_scratch
            ax.bar(range(len(vw)), vw, color=vc[:len(vw)], alpha=0.7)
            ax.set_title(f'Value Head Attention – {pos_name} (extra tokens only)')
            ax.set_xlabel('Extra Token Index'); ax.set_ylabel('Weight')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'2_head_attn_{pos_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # --- Detailed heatmap: last layer, starting position ---
    res = all_results['Starting']
    w = res['layers'][-1][0].numpy()
    seq_len = w.shape[0]
    names_full = get_token_names(num_scratch)

    # Subsample for readability
    extra_idx = list(range(num_extra))
    board_idx = list(range(num_extra, seq_len, max(1, 64 // 16)))
    show_idx = extra_idx + board_idx
    w_sub = w[np.ix_(show_idx, show_idx)]
    names_sub = [names_full[i] for i in show_idx]

    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(w_sub, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(names_sub)))
    ax.set_xticklabels(names_sub, rotation=90, fontsize=5)
    ax.set_yticks(range(len(names_sub)))
    ax.set_yticklabels(names_sub, fontsize=5)
    ax.set_title('Last-Layer Attention (Starting Position)  [row=query, col=key]')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_attention_heatmap_last_layer.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to {out_dir}/2_attention_*.png")


# ============================================================================
# Section 3: Activation Analysis
# ============================================================================

def analyze_activations(model, out_dir, device):
    print("\n" + "=" * 80)
    print("SECTION 3: ACTIVATION ANALYSIS")
    print("=" * 80)

    num_extra  = model.num_extra
    num_scratch = model.num_scratchpad
    num_layers = len(model.layers)

    layer_out = {}
    hooks = []
    for li, layer in enumerate(model.layers):
        k = f'l{li}'
        def make_hook(name):
            def fn(mod, inp, out):
                layer_out[name] = out.detach().cpu()
            return fn
        hooks.append(layer.register_forward_hook(make_hook(k)))

    all_act = {}
    with torch.no_grad():
        for pos_name, fen in TEST_POSITIONS.items():
            batch, _ = fen_to_batch(fen, device)
            x_init, _ = manual_embed(model, batch)
            model(batch)
            acts = [x_init.cpu()] + [layer_out[f'l{i}'] for i in range(num_layers)]
            all_act[pos_name] = acts

    for h in hooks:
        h.remove()

    # --- Plots ---
    for pos_name in ['Starting', 'Tactical', 'Middlegame']:
        acts = all_act[pos_name]
        nl = num_layers

        out_norms     = [a[0, 0].norm().item() for a in acts]
        fixed_norms   = [a[0, 1:9].norm(dim=1).mean().item() for a in acts]
        scratch_norms = [a[0, 9:num_extra].norm(dim=1).mean().item() for a in acts] if num_scratch > 0 else []
        board_norms   = [a[0, num_extra:].norm(dim=1).mean().item() for a in acts]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Activation Analysis – {pos_name}', fontsize=14)
        xl = range(nl + 1)

        # 3a norm trajectories
        ax = axes[0, 0]
        ax.plot(xl, out_norms,   'r-o', ms=3, label='Output')
        ax.plot(xl, fixed_norms, color='orange', marker='o', ms=3, label='Fixed Extra')
        if scratch_norms:
            ax.plot(xl, scratch_norms, 'b-o', ms=3, label='Scratchpad')
        ax.plot(xl, board_norms, 'g-o', ms=3, label='Board')
        ax.set_title('Activation L2 Norm Across Layers')
        ax.set_xlabel('Layer (0=embedding)'); ax.set_ylabel('Mean L2 Norm')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3b per-scratchpad norms at final layer
        ax = axes[0, 1]
        if num_scratch > 0:
            final_s = acts[-1][0, 9:num_extra]
            sn = final_s.norm(dim=1).numpy()
            ax.bar(range(num_scratch), sn, color='steelblue', alpha=0.7)
            board_avg = acts[-1][0, num_extra:].norm(dim=1).mean().item()
            ax.axhline(y=board_avg, color='green', ls='--', label=f'Board mean ({board_avg:.2f})')
            ax.set_title('Scratchpad Norms (Final Layer)')
            ax.set_xlabel('Index'); ax.set_ylabel('L2 Norm'); ax.legend()
        else:
            ax.text(0.5, 0.5, 'No scratchpad tokens', transform=ax.transAxes, ha='center')

        # 3c pairwise cosine similarity across layers
        ax = axes[1, 0]
        if num_scratch > 1:
            cos_per_layer = []
            for a in acts:
                s = a[0, 9:num_extra]
                sn = F.normalize(s, dim=1)
                c = (sn @ sn.T).numpy()
                cos_per_layer.append(c[np.triu_indices(num_scratch, k=1)].mean())
            ax.plot(xl, cos_per_layer, 'b-o', ms=3)
            ax.axhline(y=0, color='red', ls='--', alpha=0.5)
            ax.set_title('Mean Scratchpad Pairwise Cosine Similarity')
            ax.set_xlabel('Layer'); ax.set_ylabel('Mean Cos-Sim'); ax.grid(True, alpha=0.3)
            print(f"  [{pos_name}] Scratchpad cos-sim: embed={cos_per_layer[0]:.4f} → final={cos_per_layer[-1]:.4f}")

        # 3d activation change per layer
        ax = axes[1, 1]
        if num_scratch > 0:
            s_deltas = []; b_deltas = []
            for i in range(1, len(acts)):
                sd = (acts[i][0, 9:num_extra] - acts[i-1][0, 9:num_extra]).norm(dim=1).mean().item()
                bd = (acts[i][0, num_extra:] - acts[i-1][0, num_extra:]).norm(dim=1).mean().item()
                s_deltas.append(sd); b_deltas.append(bd)
            ax.plot(range(1, nl+1), s_deltas, 'b-o', ms=3, label='Scratchpad')
            ax.plot(range(1, nl+1), b_deltas, 'g-o', ms=3, label='Board')
            ax.set_title('Per-Layer Activation Change')
            ax.set_xlabel('Layer'); ax.set_ylabel('Mean ‖Δ‖'); ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'3_activations_{pos_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Plots saved to {out_dir}/3_activations_*.png")


# ============================================================================
# Section 4: Noise Injection Study
# ============================================================================

def noise_injection_study(model, out_dir, device):
    print("\n" + "=" * 80)
    print("SECTION 4: NOISE INJECTION STUDY")
    print("=" * 80)

    num_extra  = model.num_extra
    num_scratch = model.num_scratchpad
    noise_scales = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # --- Scratchpad noise ---
    scratch_results = {}
    board_results   = {}

    with torch.no_grad():
        for pos_name, fen in TEST_POSITIONS.items():
            batch, _ = fen_to_batch(fen, device)
            x_base, _ = manual_embed(model, batch)
            base_out, _ = manual_forward(model, x_base)
            base_pol = get_policy_probs(base_out, batch['legal_mask'])
            base_val = get_value_probs(base_out)
            base_sc  = base_out['value_scalar']

            pol_kl = []; val_kl = []; sc_diff = []
            for scale in noise_scales:
                if scale == 0:
                    pol_kl.append(0.); val_kl.append(0.); sc_diff.append(0.)
                    continue
                x_n = x_base.clone()
                x_n[:, 9:num_extra, :] += torch.randn_like(x_n[:, 9:num_extra, :]) * scale
                n_out, _ = manual_forward(model, x_n)
                n_pol = get_policy_probs(n_out, batch['legal_mask'])
                n_val = get_value_probs(n_out)
                pol_kl.append(F.kl_div(n_pol.log().clamp(min=-30), base_pol, reduction='sum').item())
                val_kl.append(F.kl_div(n_val.log().clamp(min=-30), base_val, reduction='sum').item())
                sc_diff.append((n_out['value_scalar'] - base_sc).abs().item())

            scratch_results[pos_name] = dict(pol_kl=pol_kl, val_kl=val_kl, sc_diff=sc_diff)

        # Board noise for comparison
        for pos_name in list(TEST_POSITIONS.keys())[:3]:
            batch, _ = fen_to_batch(TEST_POSITIONS[pos_name], device)
            x_base, _ = manual_embed(model, batch)
            base_out, _ = manual_forward(model, x_base)
            base_pol = get_policy_probs(base_out, batch['legal_mask'])
            bpk = []
            for scale in noise_scales:
                if scale == 0:
                    bpk.append(0.); continue
                x_n = x_base.clone()
                x_n[:, num_extra:, :] += torch.randn_like(x_n[:, num_extra:, :]) * scale
                n_out, _ = manual_forward(model, x_n)
                n_pol = get_policy_probs(n_out, batch['legal_mask'])
                bpk.append(F.kl_div(n_pol.log().clamp(min=-30), base_pol, reduction='sum').item())
            board_results[pos_name] = bpk

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Noise Injection Study', fontsize=14)

    for pn, r in scratch_results.items():
        axes[0].plot(noise_scales, r['pol_kl'], '-o', ms=4, label=pn)
    axes[0].set_title('Policy KL vs Scratchpad Noise')
    axes[0].set_xlabel('σ'); axes[0].set_ylabel('KL Div')
    axes[0].set_xscale('symlog', linthresh=0.01); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    for pn, r in scratch_results.items():
        axes[1].plot(noise_scales, r['val_kl'], '-o', ms=4, label=pn)
    axes[1].set_title('Value KL vs Scratchpad Noise')
    axes[1].set_xlabel('σ'); axes[1].set_ylabel('KL Div')
    axes[1].set_xscale('symlog', linthresh=0.01); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    for pn in list(TEST_POSITIONS.keys())[:3]:
        axes[2].plot(noise_scales, scratch_results[pn]['pol_kl'], '-o', ms=4, label=f'{pn} (scratch)')
        if pn in board_results:
            axes[2].plot(noise_scales, board_results[pn], '--s', ms=4, label=f'{pn} (board)')
    axes[2].set_title('Scratchpad vs Board Noise')
    axes[2].set_xlabel('σ'); axes[2].set_ylabel('Policy KL')
    axes[2].set_xscale('symlog', linthresh=0.01); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_noise_injection.png'), dpi=150, bbox_inches='tight')
    plt.close()

    for pn, r in scratch_results.items():
        idx1 = noise_scales.index(1.0)
        print(f"  [{pn}] σ=1.0 → Policy KL={r['pol_kl'][idx1]:.4f}, Value KL={r['val_kl'][idx1]:.4f}")

    print(f"  Plots saved to {out_dir}/4_noise_injection.png")


# ============================================================================
# Section 5: Masking (Ablation) Study
# ============================================================================

def masking_study(model, out_dir, device):
    print("\n" + "=" * 80)
    print("SECTION 5: MASKING (ABLATION) STUDY")
    print("=" * 80)

    num_extra   = model.num_extra
    num_scratch = model.num_scratchpad
    tokenizer   = ChessMoveTokenizer()

    # Group-level masks: name → (start, end) or None for baseline
    group_masks = OrderedDict({
        'baseline':        None,
        'output_token':    (0, 1),
        'turn':            (1, 2),
        'castling':        (2, 6),
        'en_passant':      (6, 7),
        'counters':        (7, 9),
        'all_scratchpad':  (9, num_extra),
        'all_extra':       (0, num_extra),
        'all_board':       (num_extra, num_extra + 64),
    })

    # Scratchpad-group masks (groups of ~16)
    scratch_group_masks = OrderedDict()
    if num_scratch > 0:
        gsz = max(1, min(16, num_scratch))
        for g in range(0, num_scratch, gsz):
            end = min(g + gsz, num_scratch)
            scratch_group_masks[f'scratch_{g}:{end-1}'] = (9 + g, 9 + end)

    results = {}
    with torch.no_grad():
        for pos_name, fen in TEST_POSITIONS.items():
            batch, _ = fen_to_batch(fen, device)
            x_base, _ = manual_embed(model, batch)
            base_out, _ = manual_forward(model, x_base)
            base_pol = get_policy_probs(base_out, batch['legal_mask'])
            base_val = get_value_probs(base_out)
            base_sc  = base_out['value_scalar'].item()
            base_idx = base_pol.argmax().item()
            base_move = tokenizer.decode(base_idx) or '???'
            base_prob = base_pol[0, base_idx].item()

            pr = {}
            all_masks = {**group_masks, **scratch_group_masks}
            for mask_name, span in all_masks.items():
                if span is None:
                    pr[mask_name] = dict(pol_kl=0., val_kl=0., sc_diff=0.,
                                         move=base_move, prob=base_prob, changed=False)
                    continue
                s, e = span
                x_m = x_base.clone()
                x_m[:, s:e, :] = 0.0
                m_out, _ = manual_forward(model, x_m)
                m_pol = get_policy_probs(m_out, batch['legal_mask'])
                m_val = get_value_probs(m_out)
                m_idx = m_pol.argmax().item()
                pr[mask_name] = dict(
                    pol_kl=F.kl_div(m_pol.log().clamp(min=-30), base_pol, reduction='sum').item(),
                    val_kl=F.kl_div(m_val.log().clamp(min=-30), base_val, reduction='sum').item(),
                    sc_diff=abs(m_out['value_scalar'].item() - base_sc),
                    move=tokenizer.decode(m_idx) or '???',
                    prob=m_pol[0, m_idx].item(),
                    changed=(m_idx != base_idx),
                )
            results[pos_name] = dict(base_move=base_move, base_prob=base_prob,
                                     base_scalar=base_sc, masks=pr)

    # --- Plot group masks ---
    gkeys = [k for k in group_masks if k != 'baseline']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Token Masking (Ablation) Study', fontsize=14)

    x = np.arange(len(gkeys))
    w = 0.8 / len(TEST_POSITIONS)

    # Policy KL
    ax = axes[0, 0]
    for i, (pn, r) in enumerate(results.items()):
        vals = [r['masks'][k]['pol_kl'] for k in gkeys]
        ax.bar(x + i * w, vals, w, label=pn, alpha=0.7)
    ax.set_xticks(x + w * len(TEST_POSITIONS) / 2)
    ax.set_xticklabels(gkeys, rotation=45, ha='right', fontsize=8)
    ax.set_title('Policy KL When Masking Groups'); ax.set_ylabel('KL Div')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3, axis='y')

    # Value KL
    ax = axes[0, 1]
    for i, (pn, r) in enumerate(results.items()):
        vals = [r['masks'][k]['val_kl'] for k in gkeys]
        ax.bar(x + i * w, vals, w, label=pn, alpha=0.7)
    ax.set_xticks(x + w * len(TEST_POSITIONS) / 2)
    ax.set_xticklabels(gkeys, rotation=45, ha='right', fontsize=8)
    ax.set_title('Value KL When Masking Groups'); ax.set_ylabel('KL Div')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3, axis='y')

    # Move-changed matrix
    ax = axes[1, 0]
    mat = np.array([[1 if results[pn]['masks'][k]['changed'] else 0
                     for k in gkeys]
                    for pn in results])
    im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(gkeys)))
    ax.set_xticklabels(gkeys, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(list(results.keys()), fontsize=8)
    ax.set_title('Does Best Move Change? (red=yes)')
    for ri, pn in enumerate(results):
        for ci, k in enumerate(gkeys):
            if results[pn]['masks'][k]['changed']:
                bm = results[pn]['base_move']
                mm = results[pn]['masks'][k]['move']
                ax.text(ci, ri, f'{bm}→{mm}', ha='center', va='center', fontsize=5, color='white')

    # Scratchpad group ablation
    ax = axes[1, 1]
    if scratch_group_masks:
        sgk = list(scratch_group_masks.keys())
        x2 = np.arange(len(sgk))
        w2 = 0.8 / min(3, len(results))
        for i, pn in enumerate(list(results.keys())[:3]):
            vals = [results[pn]['masks'][k]['pol_kl'] for k in sgk]
            ax.bar(x2 + i * w2, vals, w2, label=pn, alpha=0.7)
        ax.set_xticks(x2 + w2)
        ax.set_xticklabels(sgk, rotation=45, ha='right', fontsize=7)
        ax.set_title('Policy KL – Scratchpad Group Ablation')
        ax.set_ylabel('KL Div'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No scratchpad tokens', transform=ax.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '5_masking_study.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Print summary ---
    print("\n  --- Masking Results ---")
    for pn, r in results.items():
        print(f"\n  [{pn}] base={r['base_move']} (p={r['base_prob']:.3f}), scalar={r['base_scalar']:.4f}")
        for k in gkeys:
            m = r['masks'][k]
            tag = " ** MOVE CHANGED **" if m['changed'] else ""
            print(f"    {k:18s}  pol_kl={m['pol_kl']:8.4f}  val_kl={m['val_kl']:8.4f}  "
                  f"move={m['move']} (p={m['prob']:.3f}){tag}")

    print(f"\n  Plots saved to {out_dir}/5_masking_study.png")


# ============================================================================
# Section 6: Gradient-Based Attribution (XAI)
# ============================================================================

def gradient_attribution(model, out_dir, device):
    print("\n" + "=" * 80)
    print("SECTION 6: GRADIENT-BASED ATTRIBUTION (XAI)")
    print("=" * 80)

    num_extra   = model.num_extra
    num_scratch = model.num_scratchpad
    tokenizer   = ChessMoveTokenizer()

    all_results = {}

    for pos_name, fen in TEST_POSITIONS.items():
        batch, _ = fen_to_batch(fen, device)
        legal_mask = batch['legal_mask']

        # ---- Policy attribution ----
        with torch.enable_grad():
            board    = batch['board'];  turn = batch['turn']
            castling = batch['castling']; counters = batch['counters']; ep = batch['en_passant']
            B = board.shape[0]

            x_board   = model.piece_embedding(board.view(B, 64)) + model.pos_embedding
            x_turn    = model.turn_embedding(turn.long().squeeze(1)).unsqueeze(1)
            x_cast    = model.castling_embedding(castling.long()) + model.castling_pos_emb
            x_ep      = model.ep_embedding(ep.squeeze(1)).unsqueeze(1)
            x_count   = model.counter_proj((counters / 100.0).unsqueeze(-1)) + model.counter_pos_emb
            x_scratch = model.scratchpad.expand(B, -1, -1)
            x_out     = model.output_token.expand(B, -1, -1)

            x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
            x = x.detach().requires_grad_(True)

            x_cur = x
            for layer in model.layers:
                x_cur = layer(x_cur, model.rank_diff, model.file_diff)
            out = model.final_head(x_cur, model.num_extra)

            masked_logits = out['policy'].masked_fill(legal_mask == 0, float('-inf'))
            top_idx = masked_logits.argmax(dim=1)
            top_logit = masked_logits[0, top_idx[0]]

            model.zero_grad()
            top_logit.backward(retain_graph=True)
            pol_grad = x.grad.detach()[0]                     # [Seq, H]
            pol_importance = pol_grad.norm(dim=1).cpu().numpy()
            pol_ixg = (x.detach()[0] * pol_grad).sum(dim=1).cpu().numpy()

        # ---- Value attribution ----
        with torch.enable_grad():
            x2 = x.detach().requires_grad_(True)
            x_cur2 = x2
            for layer in model.layers:
                x_cur2 = layer(x_cur2, model.rank_diff, model.file_diff)
            out2 = model.final_head(x_cur2, model.num_extra)

            val_scalar = out2['value_scalar'][0, 0]
            model.zero_grad()
            val_scalar.backward()
            val_grad = x2.grad.detach()[0]
            val_importance = val_grad.norm(dim=1).cpu().numpy()
            val_ixg = (x2.detach()[0] * val_grad).sum(dim=1).cpu().numpy()

        all_results[pos_name] = dict(
            pol_imp=pol_importance, val_imp=val_importance,
            pol_ixg=pol_ixg, val_ixg=val_ixg,
            top_move=tokenizer.decode(top_idx[0].item()),
            val_sc=out['value_scalar'][0, 0].item(),
        )

    # --- Plots ---
    for pos_name in ['Starting', 'Tactical', 'Middlegame']:
        r = all_results[pos_name]
        seq_len = len(r['pol_imp'])

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Gradient Attribution – {pos_name}  '
                     f'(move={r["top_move"]}, val={r["val_sc"]:.3f})', fontsize=14)

        colors = (['red'] + ['orange'] * 8 +
                  ['steelblue'] * num_scratch + ['green'] * 64)

        # policy gradient norm
        ax = axes[0, 0]
        ax.bar(range(seq_len), r['pol_imp'], color=colors, alpha=0.7)
        ax.axvline(x=num_extra - 0.5, color='black', ls='--', alpha=0.5)
        ax.set_title('Policy Attribution (‖∇‖)'); ax.set_xlabel('Token'); ax.set_ylabel('Grad Norm')
        ax.grid(True, alpha=0.3, axis='y')

        # value gradient norm
        ax = axes[0, 1]
        ax.bar(range(seq_len), r['val_imp'], color=colors, alpha=0.7)
        ax.axvline(x=num_extra - 0.5, color='black', ls='--', alpha=0.5)
        ax.set_title('Value Attribution (‖∇‖)'); ax.set_xlabel('Token'); ax.set_ylabel('Grad Norm')
        ax.grid(True, alpha=0.3, axis='y')

        # Aggregated by group
        ax = axes[1, 0]
        groups = ['Output', 'Turn', 'Castling', 'EP', 'Counters', 'Scratchpad', 'Board']
        slices = [(0,1), (1,2), (2,6), (6,7), (7,9), (9,num_extra), (num_extra, seq_len)]
        pol_g = [r['pol_imp'][s:e].sum() for s, e in slices]
        val_g = [r['val_imp'][s:e].sum() for s, e in slices]
        xg = np.arange(len(groups))
        ax.bar(xg - 0.2, pol_g, 0.4, label='Policy', color='steelblue', alpha=0.7)
        ax.bar(xg + 0.2, val_g, 0.4, label='Value',  color='orange',    alpha=0.7)
        ax.set_xticks(xg); ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_title('Total Gradient Importance by Group')
        ax.set_ylabel('Σ ‖∇‖'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        # Board heatmap
        ax = axes[1, 1]
        board_imp = r['pol_imp'][num_extra:].reshape(8, 8)
        im = ax.imshow(board_imp, cmap='YlOrRd', aspect='equal')
        for ri in range(8):
            for fi in range(8):
                sq = chess.SQUARE_NAMES[(7 - ri) * 8 + fi]
                c = 'white' if board_imp[ri, fi] > board_imp.max() * 0.5 else 'black'
                ax.text(fi, ri, sq, ha='center', va='center', fontsize=6, color=c)
        ax.set_title('Policy Attribution on Board'); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'6_grad_attribution_{pos_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- Summary ---
    print("\n  --- Attribution Summary ---")
    for pn, r in all_results.items():
        seq_len = len(r['pol_imp'])
        sp = r['pol_imp'][9:num_extra].sum()
        bp = r['pol_imp'][num_extra:].sum()
        sv = r['val_imp'][9:num_extra].sum()
        bv = r['val_imp'][num_extra:].sum()
        tot_p = r['pol_imp'].sum()
        tot_v = r['val_imp'].sum()
        print(f"  [{pn}] move={r['top_move']}, val={r['val_sc']:.3f}")
        print(f"    Policy: scratchpad={sp/tot_p:.1%}  board={bp/tot_p:.1%}")
        print(f"    Value:  scratchpad={sv/tot_v:.1%}  board={bv/tot_v:.1%}")

    print(f"\n  Plots saved to {out_dir}/6_grad_attribution_*.png")


# ============================================================================
# Section 7: Summary Report
# ============================================================================

def generate_report(model, out_dir):
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    lines = [
        "=" * 80,
        "MVRD Chess Transformer – Interpretability Report",
        "=" * 80,
        "",
        f"Model:  hidden={model.hidden_size}  depth={model.depth}  "
        f"heads={model.num_heads}  scratchpad={model.num_scratchpad}",
        f"Total extra tokens: {model.num_extra}  (9 fixed + {model.num_scratchpad} scratchpad)",
        f"Sequence length: {model.num_extra + 64}",
        "",
        "Token layout:",
        f"  [0]       Output token           (learnable)",
        f"  [1]       Turn token              (embedding of side to move)",
        f"  [2-5]     Castling tokens          (WK, WQ, BK, BQ)",
        f"  [6]       En-passant token         (embedding of EP square)",
        f"  [7-8]     Counter tokens           (halfmove clock, fullmove number)",
        f"  [9-{model.num_extra-1:>3}]   Scratchpad tokens       ({model.num_scratchpad}x learnable)",
        f"  [{model.num_extra}-{model.num_extra+63}]  Board tokens            (64 squares)",
        "",
        "Architecture notes:",
        "  - Policy head attends to ALL tokens (extra + board)",
        "  - Value head ONLY attends to extra tokens (bottleneck design)",
        "  - Relative position bias between board squares only",
        "",
        "See plots and console output for detailed findings.",
        "=" * 80,
    ]

    report = "\n".join(lines)
    path = os.path.join(out_dir, 'report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(report)
    print(f"\n  Report saved to {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MVRD Chess Transformer – Neural Network Interpretability Analysis')
    parser.add_argument('--config',     type=str, required=True,  help='Config YAML path')
    parser.add_argument('--checkpoint', type=str, required=True,  help='Checkpoint .pt path')
    parser.add_argument('--output',     type=str, default='analysis_output', help='Output dir')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--positions',  type=str, nargs='*',
                        help='Extra FEN strings to include')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']

    print(f"Config: {args.config}")
    print(f"  hidden={mc['hidden_size']}  depth={mc['depth']}  heads={mc['num_heads']}  "
          f"scratchpad={mc['num_scratchpad']}")

    # Build model
    model = ChessTransformer(
        vocab_size              = mc.get('vocab_size', 4544),
        hidden_size             = mc['hidden_size'],
        depth                   = mc['depth'],
        num_heads               = mc['num_heads'],
        ff_dim                  = mc['ff_dim'],
        num_eval_bins           = mc.get('num_eval_bins', 128),
        num_scratchpad          = mc['num_scratchpad'],
        aux_loss_only_extra_tokens = mc.get('aux_loss_only_extra_tokens', False),
        use_aux_loss            = mc.get('use_aux_loss', True),
        drop_path_rate          = mc.get('drop_path_rate', 0.1),
    )

    # Load checkpoint
    print(f"Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    # Remap old style state dict if needed
    if "policy_query" in sd and "final_head.policy_query" not in sd:
        print("Detected old-style state dict, remapping keys...")
        remap = {
            "policy_query": "final_head.policy_query",
            "policy_attn.in_proj_weight": "final_head.policy_attn.in_proj_weight",
            "policy_attn.in_proj_bias": "final_head.policy_attn.in_proj_bias",
            "policy_attn.out_proj.weight": "final_head.policy_attn.out_proj.weight",
            "policy_attn.out_proj.bias": "final_head.policy_attn.out_proj.bias",
            "policy_head.weight": "final_head.policy_head.weight",
            "value_head.weight": "final_head.value_head.weight",
            "value_scalar_head.weight": "final_head.value_scalar_head.weight",
            "mate_head.weight": "final_head.mate_head.weight",
            "final_norm.weight": "final_head.norm.weight",
        }
        for old, new in remap.items():
            if old in sd:
                sd[new] = sd.pop(old)

    # Check for aux heads in state dict
    has_aux = any(k.startswith("aux_heads") for k in sd.keys())
    if not has_aux and model.use_aux_loss:
        print("Checkpoint has no aux heads, but model is configured with use_aux_loss=True. Re-initializing model without aux heads...")
        model = ChessTransformer(
            vocab_size              = mc.get('vocab_size', 4544),
            hidden_size             = mc['hidden_size'],
            depth                   = mc['depth'],
            num_heads               = mc['num_heads'],
            ff_dim                  = mc['ff_dim'],
            num_eval_bins           = mc.get('num_eval_bins', 128),
            num_scratchpad          = mc['num_scratchpad'],
            aux_loss_only_extra_tokens = mc.get('aux_loss_only_extra_tokens', False),
            use_aux_loss            = False,
            drop_path_rate          = mc.get('drop_path_rate', 0.1),
        )

    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval()

    step  = ckpt.get('step', '?')
    epoch = ckpt.get('epoch', '?')
    print(f"Loaded  step={step}  epoch={epoch}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Extra positions
    if args.positions:
        for i, fen in enumerate(args.positions):
            TEST_POSITIONS[f'Custom_{i}'] = fen

    # ---- Run all analyses ----
    analyze_embeddings(model, args.output, args.device)
    analyze_attention_patterns(model, args.output, args.device)
    analyze_activations(model, args.output, args.device)
    noise_injection_study(model, args.output, args.device)
    masking_study(model, args.output, args.device)
    gradient_attribution(model, args.output, args.device)
    generate_report(model, args.output)

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE – all outputs in {args.output}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
