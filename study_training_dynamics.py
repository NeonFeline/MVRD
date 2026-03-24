#!/usr/bin/env python3
"""
Study WHY scratchpad tokens don't help.

1. Training curves: accuracy/loss at each checkpoint for all 3 models
2. Scratchpad utilization over training: does the model gradually learn to use them?
3. FLOP-normalized comparison: is it just that 128-scratch does more compute per step?
4. Information flow analysis: where does the value head get its info from?
"""

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from model import ChessTransformer
from dataset.chess_dataset import FastChessDataset, ChessMoveTokenizer
from torch.utils.data import DataLoader


def load_model_from_config(cfg_path, device='cpu'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']
    model = ChessTransformer(
        vocab_size=mc.get('vocab_size', 4544), hidden_size=mc['hidden_size'],
        depth=mc['depth'], num_heads=mc['num_heads'], ff_dim=mc['ff_dim'],
        num_eval_bins=mc.get('num_eval_bins', 128), num_scratchpad=mc['num_scratchpad'],
        aux_loss_only_extra_tokens=mc.get('aux_loss_only_extra_tokens', False),
        use_aux_loss=mc.get('use_aux_loss', True),
        drop_path_rate=mc.get('drop_path_rate', 0.1),
    )
    return model, cfg


def load_checkpoint_into(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return ckpt.get('step', None)


def get_checkpoint_steps(ckpt_dir):
    """Return sorted list of (step, path) tuples."""
    paths = glob.glob(os.path.join(ckpt_dir, 'checkpoint_*.pt'))
    results = []
    for p in paths:
        try:
            step = int(os.path.basename(p).replace('checkpoint_', '').replace('.pt', ''))
            results.append((step, p))
        except ValueError:
            pass
    # Also include final.pt
    final = os.path.join(ckpt_dir, 'final.pt')
    if os.path.exists(final):
        if results:
            results.append((results[-1][0] + 2000, final))  # Approximate
        else:
            results.append((0, final))
    return sorted(results)


def eval_batch(model, batch, device):
    """Quick eval on a single batch, return dict of metrics."""
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    lm = batch['legal_mask']
    tgt = batch['move_target'].argmax(dim=1)
    ml = out['policy'].masked_fill(lm == 0, float('-inf'))
    pred = ml.argmax(dim=1)

    return {
        'acc1': (pred == tgt).float().mean().item(),
        'ce': F.cross_entropy(ml, tgt).clamp(max=20).item(),
        'vkl': F.kl_div(F.log_softmax(out['value'], dim=1),
                         batch['eval_target'], reduction='batchmean').item(),
        'vmse': (out['value_scalar'] - batch['score_scalar']).pow(2).mean().item(),
    }


def measure_scratchpad_utilization(model, batch, device):
    """Measure how much the model relies on scratchpad tokens."""
    num_extra = model.num_extra
    num_scratch = model.num_scratchpad
    if num_scratch == 0:
        return {'cos_sim': 0, 'ablation_kl': 0, 'val_head_entropy': 0,
                'scratch_attn_frac': 0, 'scratch_norm_ratio': 0}

    batch = {k: v.to(device) for k, v in batch.items()}

    # --- 1. Cosine similarity of scratchpad in final layer ---
    # Hook final layer
    final_output = {}
    def hook_fn(mod, inp, out):
        final_output['x'] = out.detach()
    h = model.layers[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        out = model(batch)
    h.remove()

    x_final = final_output['x']  # [B, Seq, H]
    scratch = x_final[:, 9:num_extra, :]  # [B, N_scratch, H]
    # Mean over batch
    s = scratch.mean(dim=0)  # [N_scratch, H]
    sn = F.normalize(s, dim=1)
    cos = (sn @ sn.T).cpu().numpy()
    cos_sim = cos[np.triu_indices(num_scratch, k=1)].mean()

    # --- 2. Scratchpad norm relative to board ---
    scratch_norm = scratch.norm(dim=-1).mean().item()
    board_norm = x_final[:, num_extra:, :].norm(dim=-1).mean().item()
    norm_ratio = scratch_norm / max(board_norm, 1e-8)

    # --- 3. Policy KL when ablating scratchpad ---
    lm = batch['legal_mask']
    base_pol = F.softmax(out['policy'].masked_fill(lm == 0, float('-inf')), dim=-1)

    # Manual forward with zeroed scratchpad
    B = batch['board'].shape[0]
    x_board = model.piece_embedding(batch['board'].view(B, 64)) + model.pos_embedding
    x_turn = model.turn_embedding(batch['turn'].long().squeeze(1)).unsqueeze(1)
    x_cast = model.castling_embedding(batch['castling'].long()) + model.castling_pos_emb
    x_ep = model.ep_embedding(batch['en_passant'].squeeze(1)).unsqueeze(1)
    x_count = model.counter_proj((batch['counters'] / 100.0).unsqueeze(-1)) + model.counter_pos_emb
    x_scratch = model.scratchpad.expand(B, -1, -1)
    x_out = model.output_token.expand(B, -1, -1)

    x = torch.cat([x_out, x_turn, x_cast, x_ep, x_count, x_scratch, x_board], dim=1)
    x[:, 9:num_extra, :] = 0.0

    with torch.no_grad():
        for layer in model.layers:
            x = layer(x, model.rank_diff, model.file_diff)
        abl_out = model.final_head(x, model.num_extra)

    abl_pol = F.softmax(abl_out['policy'].masked_fill(lm == 0, float('-inf')), dim=-1)
    ablation_kl = F.kl_div(abl_pol.log().clamp(min=-30), base_pol, reduction='batchmean').item()

    # --- 4. Value head attention entropy ---
    val_attn_weights = {}
    def val_hook(mod, inp, out):
        val_attn_weights['w'] = out[1].detach()
    h2 = model.final_head.value_attn.register_forward_hook(val_hook)
    with torch.no_grad():
        _ = model(batch)
    h2.remove()

    vw = val_attn_weights['w']  # [B, 1, num_extra]
    vw = vw.squeeze(1).mean(dim=0)  # [num_extra]
    # Entropy
    entropy = -(vw * vw.log().clamp(min=-30)).sum().item()
    max_entropy = np.log(num_extra)
    normalized_entropy = entropy / max_entropy  # 1.0 = uniform, 0.0 = peaked

    # --- 5. Fraction of attention going to scratchpad ---
    scratch_frac = vw[9:num_extra].sum().item()

    return {
        'cos_sim': cos_sim,
        'ablation_kl': ablation_kl,
        'val_head_entropy': normalized_entropy,
        'scratch_attn_frac': scratch_frac,
        'scratch_norm_ratio': norm_ratio,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = 'study_output'
    os.makedirs(out_dir, exist_ok=True)

    # Model configs
    models = {
        '0_scratch': {
            'config': 'configs/config_small_no_scratchpad.yaml',
            'ckpt_dir': 'checkpoints/test_8L_384H_no_scratchpad',
        },
        '128_no_aux': {
            'config': 'configs/config_small_no_aux.yaml',
            'ckpt_dir': 'checkpoints/test_8L_384H_no_aux',
        },
        '128_aux': {
            'config': 'configs/config_small_aux_extra_only.yaml',
            'ckpt_dir': 'checkpoints/test_8L_384H_aux_extra_only',
        },
    }

    # Load a fixed eval batch (2048 samples, same for all)
    print("Loading eval data...")
    ds = FastChessDataset('dataset/data/lichess_db_eval.jsonl.zst', skip=0, limit=2048)
    dl = DataLoader(ds, batch_size=2048, num_workers=4)
    eval_batch_data = next(iter(dl))
    print(f"  Loaded {eval_batch_data['board'].shape[0]} samples")

    # Checkpoints to evaluate (sample every N to keep it fast)
    sample_every = 4  # Every 4th checkpoint = every 8000 steps

    all_curves = {}
    all_utilization = {}

    for label, info in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {label}")
        print(f"{'='*60}")

        model, cfg = load_model_from_config(info['config'], device)
        ckpts = get_checkpoint_steps(info['ckpt_dir'])

        # Sample checkpoints
        sampled = ckpts[::sample_every]
        # Always include first and last
        if ckpts[0] not in sampled:
            sampled = [ckpts[0]] + sampled
        if ckpts[-1] not in sampled:
            sampled.append(ckpts[-1])

        steps = []
        metrics_over_time = defaultdict(list)
        util_over_time = defaultdict(list)

        for step, path in sampled:
            load_checkpoint_into(model, path, device)
            steps.append(step)

            # Eval metrics
            m = eval_batch(model, eval_batch_data, device)
            for k, v in m.items():
                metrics_over_time[k].append(v)

            # Scratchpad utilization
            u = measure_scratchpad_utilization(model, eval_batch_data, device)
            for k, v in u.items():
                util_over_time[k].append(v)

            print(f"  step={step:>6}  acc={m['acc1']:.4f}  ce={m['ce']:.4f}  "
                  f"vkl={m['vkl']:.4f}  abl_kl={u['ablation_kl']:.4f}  "
                  f"cos_sim={u['cos_sim']:.4f}")

        all_curves[label] = {'steps': steps, 'metrics': dict(metrics_over_time)}
        all_utilization[label] = {'steps': steps, 'util': dict(util_over_time)}

        del model
        torch.cuda.empty_cache()

    # ========================================================================
    # PLOTS
    # ========================================================================

    colors = {'0_scratch': '#2196F3', '128_no_aux': '#FF9800', '128_aux': '#4CAF50'}

    # --- 1. Training curves ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves: 0 vs 128 Scratchpad Tokens', fontsize=14)

    for key, title, ax_idx in [
        ('acc1', 'Policy Top-1 Accuracy', (0, 0)),
        ('ce',   'Policy Cross-Entropy',  (0, 1)),
        ('vkl',  'Value KL Divergence',   (1, 0)),
        ('vmse', 'Value Scalar MSE',      (1, 1)),
    ]:
        ax = axes[ax_idx]
        for label in models:
            c = all_curves[label]
            ax.plot(c['steps'], c['metrics'][key], '-o', ms=3,
                    color=colors[label], label=label)
        ax.set_title(title); ax.set_xlabel('Step'); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 2. Scratchpad utilization over training ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Scratchpad Utilization Over Training', fontsize=14)

    util_metrics = [
        ('cos_sim',           'Scratchpad Cosine Similarity\n(lower = more diverse)',  (0, 0)),
        ('ablation_kl',       'Policy KL When Scratchpad Zeroed\n(higher = more dependent)', (0, 1)),
        ('scratch_norm_ratio', 'Scratchpad/Board Norm Ratio',          (0, 2)),
        ('val_head_entropy',  'Value Head Attention Entropy\n(1=uniform, 0=peaked)', (1, 0)),
        ('scratch_attn_frac', 'Value Attn Fraction on Scratchpad',     (1, 1)),
    ]

    for key, title, ax_idx in util_metrics:
        ax = axes[ax_idx]
        for label in ['128_no_aux', '128_aux']:
            c = all_utilization[label]
            if key in c['util']:
                ax.plot(c['steps'], c['util'][key], '-o', ms=3,
                        color=colors[label], label=label)
        ax.set_title(title); ax.set_xlabel('Step')
        ax.legend(); ax.grid(True, alpha=0.3)

    # Last subplot: FLOP-normalized comparison
    ax = axes[1, 2]
    # 0-scratch: seq_len=73, 128-scratch: seq_len=201
    # Attention FLOPs per layer ~ seq_len^2 * hidden
    # Total FLOPs per step ~ depth * seq_len^2 * hidden * batch_size (roughly)
    flop_ratio = (201**2) / (73**2)  # ~7.58x
    ax.text(0.5, 0.7, f'FLOP ratio per step:\n128-scratch / 0-scratch ≈ {flop_ratio:.1f}x',
            transform=ax.transAxes, ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.4, f'At step 82000:\n'
            f'0-scratch has used ~82k "step-units"\n'
            f'128-scratch has used ~82k × {flop_ratio:.1f} = ~{82000*flop_ratio/1000:.0f}k "FLOP-units"\n\n'
            f'128-scratch uses {flop_ratio:.1f}x more compute\nbut performs WORSE',
            transform=ax.transAxes, ha='center', va='center', fontsize=10)
    ax.set_title('FLOP Analysis')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_scratchpad_utilization.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3. Convergence analysis: is 128-scratch still catching up? ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Is 128-scratch Still Catching Up?', fontsize=14)

    # Accuracy gap over time
    ax = axes[0]
    s0 = all_curves['0_scratch']
    for label in ['128_no_aux', '128_aux']:
        c = all_curves[label]
        # Align on steps
        common_steps = sorted(set(s0['steps']) & set(c['steps']))
        if common_steps:
            gap = []
            for s in common_steps:
                i0 = s0['steps'].index(s)
                ic = c['steps'].index(s)
                gap.append(c['metrics']['acc1'][ic] - s0['metrics']['acc1'][i0])
            ax.plot(common_steps, gap, '-o', ms=3, color=colors[label], label=f'{label} - 0_scratch')
    ax.axhline(y=0, color='black', ls='--', alpha=0.5)
    ax.set_title('Accuracy Gap vs 0-scratch Over Training')
    ax.set_xlabel('Step'); ax.set_ylabel('Δ Accuracy (positive = 128 is better)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Value KL gap
    ax = axes[1]
    for label in ['128_no_aux', '128_aux']:
        c = all_curves[label]
        common_steps = sorted(set(s0['steps']) & set(c['steps']))
        if common_steps:
            gap = []
            for s in common_steps:
                i0 = s0['steps'].index(s)
                ic = c['steps'].index(s)
                gap.append(c['metrics']['vkl'][ic] - s0['metrics']['vkl'][i0])
            ax.plot(common_steps, gap, '-o', ms=3, color=colors[label], label=f'{label} - 0_scratch')
    ax.axhline(y=0, color='black', ls='--', alpha=0.5)
    ax.set_title('Value KL Gap vs 0-scratch Over Training')
    ax.set_xlabel('Step'); ax.set_ylabel('Δ Value KL (negative = 128 is better)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_convergence_gap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    main()
