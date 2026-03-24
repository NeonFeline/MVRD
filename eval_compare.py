#!/usr/bin/env python3
"""
Compare two models on held-out data: policy accuracy, value error, and
scratchpad ablation impact.

Usage:
    python eval_compare.py \
        --config_a configs/config_small_aux_extra_only.yaml \
        --checkpoint_a checkpoints/test_8L_384H_aux_extra_only/final.pt \
        --label_a "aux_extra_only" \
        --config_b configs/config_small_no_aux.yaml \
        --checkpoint_b checkpoints/test_8L_384H_no_aux/final.pt \
        --label_b "no_aux" \
        --num_samples 5000
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from model import ChessTransformer
from dataset.chess_dataset import FastChessDataset, ChessMoveTokenizer
from torch.utils.data import DataLoader


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']

    model = ChessTransformer(
        vocab_size=mc.get('vocab_size', 4544),
        hidden_size=mc['hidden_size'],
        depth=mc['depth'],
        num_heads=mc['num_heads'],
        ff_dim=mc['ff_dim'],
        num_eval_bins=mc.get('num_eval_bins', 128),
        num_scratchpad=mc['num_scratchpad'],
        aux_loss_only_extra_tokens=mc.get('aux_loss_only_extra_tokens', False),
        use_aux_loss=mc.get('use_aux_loss', True),
        drop_path_rate=mc.get('drop_path_rate', 0.1),
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, cfg


def manual_embed(model, batch):
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

    x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
    return x


def manual_forward(model, x):
    for layer in model.layers:
        x = layer(x, model.rank_diff, model.file_diff)
    out = model.final_head(x, model.num_extra)
    return out


@torch.no_grad()
def evaluate_model(model, dataloader, num_samples, device, also_ablate=True):
    """Evaluate model and optionally run scratchpad-ablated version."""
    num_extra = model.num_extra

    stats = defaultdict(list)
    ablated_stats = defaultdict(list)
    count = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch['board'].shape[0]

        # --- Normal forward ---
        out = model(batch)
        legal_mask = batch['legal_mask']
        target_idx = batch['move_target'].argmax(dim=1)

        # Policy
        logits = out['policy']
        masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))
        pred_idx = masked_logits.argmax(dim=1)
        correct = (pred_idx == target_idx).float()
        stats['policy_acc'].append(correct.cpu())

        # Top-5 accuracy
        _, top5 = masked_logits.topk(5, dim=1)
        top5_correct = (top5 == target_idx.unsqueeze(1)).any(dim=1).float()
        stats['top5_acc'].append(top5_correct.cpu())

        # Policy cross-entropy (clamp to avoid inf from -inf logits)
        policy_ce = F.cross_entropy(masked_logits, target_idx, reduction='none')
        policy_ce = policy_ce.clamp(max=20.0)
        stats['policy_ce'].append(policy_ce.cpu())

        # Value KL
        value_logits = out['value']
        value_log_probs = F.log_softmax(value_logits, dim=1)
        value_target = batch['eval_target']
        value_kl = F.kl_div(value_log_probs, value_target, reduction='none').sum(dim=1)
        stats['value_kl'].append(value_kl.cpu())

        # Value scalar MSE
        val_scalar = out['value_scalar']
        val_target = batch['score_scalar']
        val_mse = (val_scalar - val_target).pow(2).squeeze(1)
        stats['value_mse'].append(val_mse.cpu())

        # Mate MSE
        mate_pred = out['mate']
        mate_target = batch['mate_target']
        mate_mse = (mate_pred - mate_target).pow(2).squeeze(1)
        stats['mate_mse'].append(mate_mse.cpu())

        # --- Ablated forward (zero out scratchpad) ---
        if also_ablate:
            x = manual_embed(model, batch)
            x[:, 9:num_extra, :] = 0.0
            abl_out = manual_forward(model, x)

            abl_logits = abl_out['policy'].masked_fill(legal_mask == 0, float('-inf'))
            abl_pred = abl_logits.argmax(dim=1)
            abl_correct = (abl_pred == target_idx).float()
            ablated_stats['policy_acc'].append(abl_correct.cpu())

            abl_top5 = abl_logits.topk(5, dim=1)[1]
            abl_top5_correct = (abl_top5 == target_idx.unsqueeze(1)).any(dim=1).float()
            ablated_stats['top5_acc'].append(abl_top5_correct.cpu())

            abl_ce = F.cross_entropy(abl_logits, target_idx, reduction='none').clamp(max=20.0)
            ablated_stats['policy_ce'].append(abl_ce.cpu())

            abl_val_log = F.log_softmax(abl_out['value'], dim=1)
            abl_val_kl = F.kl_div(abl_val_log, value_target, reduction='none').sum(dim=1)
            ablated_stats['value_kl'].append(abl_val_kl.cpu())

            abl_val_mse = (abl_out['value_scalar'] - val_target).pow(2).squeeze(1)
            ablated_stats['value_mse'].append(abl_val_mse.cpu())

            # Did the move change?
            move_changed = (abl_pred != pred_idx).float()
            ablated_stats['move_changed'].append(move_changed.cpu())

        count += B
        if count >= num_samples:
            break

    # Aggregate
    result = {}
    for k, v in stats.items():
        result[k] = torch.cat(v).numpy()
    abl_result = {}
    if also_ablate:
        for k, v in ablated_stats.items():
            abl_result[k] = torch.cat(v).numpy()

    return result, abl_result, count


def print_results(label, stats, abl_stats, count):
    print(f"\n  {'='*60}")
    print(f"  {label}  ({count} samples)")
    print(f"  {'='*60}")
    print(f"  Policy top-1 acc:   {stats['policy_acc'].mean():.4f}")
    print(f"  Policy top-5 acc:   {stats['top5_acc'].mean():.4f}")
    print(f"  Policy CE loss:     {stats['policy_ce'].mean():.4f}")
    print(f"  Value KL loss:      {stats['value_kl'].mean():.4f}")
    print(f"  Value scalar MSE:   {stats['value_mse'].mean():.4f}")
    print(f"  Mate MSE:           {stats['mate_mse'].mean():.4f}")

    if abl_stats:
        print(f"\n  --- With scratchpad zeroed out ---")
        print(f"  Policy top-1 acc:   {abl_stats['policy_acc'].mean():.4f}  "
              f"(delta: {abl_stats['policy_acc'].mean() - stats['policy_acc'].mean():+.4f})")
        print(f"  Policy top-5 acc:   {abl_stats['top5_acc'].mean():.4f}  "
              f"(delta: {abl_stats['top5_acc'].mean() - stats['top5_acc'].mean():+.4f})")
        print(f"  Policy CE loss:     {abl_stats['policy_ce'].mean():.4f}  "
              f"(delta: {abl_stats['policy_ce'].mean() - stats['policy_ce'].mean():+.4f})")
        print(f"  Value KL loss:      {abl_stats['value_kl'].mean():.4f}  "
              f"(delta: {abl_stats['value_kl'].mean() - stats['value_kl'].mean():+.4f})")
        print(f"  Value scalar MSE:   {abl_stats['value_mse'].mean():.4f}  "
              f"(delta: {abl_stats['value_mse'].mean() - stats['value_mse'].mean():+.4f})")
        print(f"  Move changed:       {abl_stats['move_changed'].mean():.4f}  "
              f"({abl_stats['move_changed'].sum():.0f}/{len(abl_stats['move_changed'])} samples)")


def main():
    parser = argparse.ArgumentParser(description='Compare two models')
    parser.add_argument('--config_a',     required=True)
    parser.add_argument('--checkpoint_a', required=True)
    parser.add_argument('--label_a',      default='Model A')
    parser.add_argument('--config_b',     required=True)
    parser.add_argument('--checkpoint_b', required=True)
    parser.add_argument('--label_b',      default='Model B')
    parser.add_argument('--data_path',    default='dataset/data/lichess_db_eval.jsonl.zst')
    parser.add_argument('--num_samples',  type=int, default=5000)
    parser.add_argument('--batch_size',   type=int, default=256)
    parser.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output',       default='eval_output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load models
    print(f"Loading model A: {args.label_a}")
    model_a, cfg_a = load_model(args.config_a, args.checkpoint_a, args.device)
    print(f"  params: {sum(p.numel() for p in model_a.parameters()):,}")

    print(f"Loading model B: {args.label_b}")
    model_b, cfg_b = load_model(args.config_b, args.checkpoint_b, args.device)
    print(f"  params: {sum(p.numel() for p in model_b.parameters()):,}")

    # Create dataloaders (same data for both, first N samples)
    print(f"\nLoading data (limit={args.num_samples})...")
    ds_a = FastChessDataset(args.data_path, skip=0, limit=args.num_samples)
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, num_workers=4)

    ds_b = FastChessDataset(args.data_path, skip=0, limit=args.num_samples)
    dl_b = DataLoader(ds_b, batch_size=args.batch_size, num_workers=4)

    # Evaluate
    print(f"\nEvaluating {args.label_a}...")
    stats_a, abl_a, count_a = evaluate_model(model_a, dl_a, args.num_samples, args.device)
    print_results(args.label_a, stats_a, abl_a, count_a)

    print(f"\nEvaluating {args.label_b}...")
    stats_b, abl_b, count_b = evaluate_model(model_b, dl_b, args.num_samples, args.device)
    print_results(args.label_b, stats_b, abl_b, count_b)

    # --- Comparison plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Model Comparison: {args.label_a} vs {args.label_b}', fontsize=14)

    metrics = [
        ('policy_acc', 'Policy Top-1 Accuracy', True),
        ('top5_acc',   'Policy Top-5 Accuracy', True),
        ('policy_ce',  'Policy Cross-Entropy', False),
        ('value_kl',   'Value KL Divergence', False),
        ('value_mse',  'Value Scalar MSE', False),
        ('mate_mse',   'Mate MSE', False),
    ]

    for idx, (key, title, higher_better) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        vals = [
            stats_a[key].mean(), stats_b[key].mean(),
            abl_a.get(key, stats_a[key]).mean(),
            abl_b.get(key, stats_b[key]).mean(),
        ]
        labels = [args.label_a, args.label_b,
                  f'{args.label_a}\n(no scratch)', f'{args.label_b}\n(no scratch)']
        colors = ['#2196F3', '#FF9800', '#90CAF9', '#FFE0B2']

        bars = ax.bar(range(4), vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {args.label_a:>15} {args.label_b:>15}  {'Winner':>10}")
    print(f"{'-'*70}")

    for key, title, higher_better in metrics:
        va = stats_a[key].mean()
        vb = stats_b[key].mean()
        if higher_better:
            winner = args.label_a if va > vb else args.label_b
        else:
            winner = args.label_a if va < vb else args.label_b
        print(f"{title:<25} {va:>15.4f} {vb:>15.4f}  {winner:>10}")

    print(f"\n{'Ablation Impact':<25} {args.label_a:>15} {args.label_b:>15}")
    print(f"{'-'*70}")
    for key in ['policy_acc', 'value_kl', 'value_mse']:
        delta_a = abl_a[key].mean() - stats_a[key].mean()
        delta_b = abl_b[key].mean() - stats_b[key].mean()
        print(f"  {key} delta:          {delta_a:>+15.4f} {delta_b:>+15.4f}")

    print(f"\n  Move changed rate:     {abl_a['move_changed'].mean():>15.4f} {abl_b['move_changed'].mean():>15.4f}")

    print(f"\nPlot saved to {args.output}/comparison.png")


if __name__ == '__main__':
    main()
