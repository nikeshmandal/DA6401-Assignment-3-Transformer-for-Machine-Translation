import os
import math
import torch
import wandb
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Transformer
from dataset import load_data_and_vocab
from lr_scheduler import NoamScheduler
from scheduler import LabelSmoothingLoss
from utils import train_epoch_v2, evaluate_epoch, compute_bleu, get_gradient_norms, greedy_decode


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_prediction_confidence(model, loader, src_vocab, tgt_vocab, device, max_batches=5):
    model.eval()
    src_pad_idx = src_vocab.pad_idx
    tgt_pad_idx = tgt_vocab.pad_idx
    confidences = []

    with torch.no_grad():
        for i, (src, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
            probs = torch.softmax(output, dim=-1)
            correct_probs = probs.gather(2, tgt_output.unsqueeze(-1).clamp(0, probs.size(-1)-1)).squeeze(-1)
            mask = (tgt_output != tgt_pad_idx)
            confidences.extend(correct_probs[mask].cpu().tolist())

    return np.mean(confidences) if confidences else 0.0


def log_attention_heatmaps(model, loader, src_vocab, tgt_vocab, device, run_name=""):
    model.eval()
    src_pad_idx = src_vocab.pad_idx

    for src, tgt in loader:
        src = src.to(device)
        src_sample = src[:1]
        break

    with torch.no_grad():
        enc_attn_weights = model.get_encoder_attention(src_sample, src_pad_idx)

    last_layer_attn = enc_attn_weights[-1][0].cpu().numpy()
    num_heads = last_layer_attn.shape[0]

    src_tokens = []
    for tok_id in src_sample[0].tolist():
        if tok_id == src_vocab.pad_idx:
            break
        src_tokens.append(src_vocab.itos[tok_id])

    seq_len = len(src_tokens)

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(3 * num_heads // 2, 6))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        attn = last_layer_attn[head_idx, :seq_len, :seq_len]
        ax = axes[head_idx]
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(src_tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(src_tokens, fontsize=7)
        ax.set_title(f"Head {head_idx + 1}", fontsize=9)

    plt.suptitle(f"Encoder Last Layer Attention Heads {run_name}", fontsize=10)
    plt.tight_layout()

    wandb.log({"attention_heatmaps": wandb.Image(fig)})
    plt.close(fig)


def train(config):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, src_tok, tgt_tok = load_data_and_vocab(
        batch_size=config['batch_size']
    )

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        use_scale=config['use_scale'],
        use_learned_pe=config['use_learned_pe']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    smoothing = config['label_smoothing']
    criterion = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=tgt_vocab.pad_idx,
        smoothing=smoothing
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    if config['use_noam']:
        scheduler = NoamScheduler(optimizer, config['d_model'], config['warmup_steps'])
    else:
        for pg in optimizer.param_groups:
            pg['lr'] = config['fixed_lr']
        scheduler = None

    best_val_loss = float('inf')
    best_bleu = 0.0
    global_step = 0

    wandb.watch(model, log='gradients', log_freq=100)

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input, src_vocab.pad_idx, tgt_vocab.pad_idx)
            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()

            total_norm, qk_norm = get_gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                lr = scheduler.step()
            else:
                lr = config['fixed_lr']

            global_step += 1

            num_tokens = (tgt_output != tgt_vocab.pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if global_step <= 1000:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/learning_rate": lr,
                    "train/grad_norm_total": total_norm,
                    "train/grad_norm_qk": qk_norm,
                    "global_step": global_step
                })
            elif global_step % 50 == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/learning_rate": lr,
                    "train/grad_norm_total": total_norm,
                    "global_step": global_step
                })

        train_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        val_loss = evaluate_epoch(model, val_loader, criterion, device, src_vocab.pad_idx, tgt_vocab.pad_idx)

        val_bleu = compute_bleu(model, val_loader, src_vocab, tgt_vocab, device, max_samples=500, use_beam=True, beam_size=4)
        avg_confidence = get_prediction_confidence(model, val_loader, src_vocab, tgt_vocab, device)

        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
            "val/bleu": val_bleu,
            "val/prediction_confidence": avg_confidence,
            "global_step": global_step
        })

        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }, f"best_model_{config['run_name']}.pt")

        if val_bleu > best_bleu:
            best_bleu = val_bleu

    log_attention_heatmaps(model, val_loader, src_vocab, tgt_vocab, device, config['run_name'])

    test_loss = evaluate_epoch(model, test_loader, criterion, device, src_vocab.pad_idx, tgt_vocab.pad_idx)
    test_bleu = compute_bleu(model, test_loader, src_vocab, tgt_vocab, device, max_samples=1000, use_beam=True, beam_size=4)

    wandb.log({
        "test/loss": test_loss,
        "test/bleu": test_bleu
    })

    print(f"Test Loss: {test_loss:.4f} | Test BLEU: {test_bleu:.4f}")

    wandb.summary['best_val_loss'] = best_val_loss
    wandb.summary['best_val_bleu'] = best_bleu
    wandb.summary['test_bleu'] = test_bleu

    return model, best_bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=['baseline', 'noam_vs_fixed', 'scale_ablation', 'label_smoothing', 'learned_pe'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, default='da6401-assignment3')
    args = parser.parse_args()

    base_config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'num_layers': args.num_layers,
        'max_len': 256,
        'dropout': args.dropout,
        'warmup_steps': args.warmup_steps,
        'use_noam': True,
        'fixed_lr': 1e-4,
        'label_smoothing': 0.1,
        'use_scale': True,
        'use_learned_pe': False,
    }

    if args.experiment == 'baseline':
        config = {**base_config, 'run_name': 'baseline'}
        wandb.init(project=args.wandb_project, name='baseline', config=config)
        train(config)

    elif args.experiment == 'noam_vs_fixed':
        ablation_config = {**base_config, 'num_epochs': min(base_config['num_epochs'], 20)}
        config_noam = {**ablation_config, 'use_noam': True, 'run_name': 'noam_scheduler'}
        wandb.init(project=args.wandb_project, name='noam_scheduler', config=config_noam)
        train(config_noam)
        wandb.finish()

        config_fixed = {**ablation_config, 'use_noam': False, 'run_name': 'fixed_lr'}
        wandb.init(project=args.wandb_project, name='fixed_lr_1e-4', config=config_fixed)
        train(config_fixed)

    elif args.experiment == 'scale_ablation':
        ablation_config = {**base_config, 'num_epochs': min(base_config['num_epochs'], 20)}
        config_with = {**ablation_config, 'use_scale': True, 'run_name': 'with_scale'}
        wandb.init(project=args.wandb_project, name='with_sqrt_dk_scaling', config=config_with)
        train(config_with)
        wandb.finish()

        config_without = {**ablation_config, 'use_scale': False, 'run_name': 'no_scale'}
        wandb.init(project=args.wandb_project, name='without_sqrt_dk_scaling', config=config_without)
        train(config_without)

    elif args.experiment == 'label_smoothing':
        ablation_config = {**base_config, 'num_epochs': min(base_config['num_epochs'], 20)}
        config_smooth = {**ablation_config, 'label_smoothing': 0.1, 'run_name': 'label_smooth_0.1'}
        wandb.init(project=args.wandb_project, name='label_smoothing_0.1', config=config_smooth)
        train(config_smooth)
        wandb.finish()

        config_no_smooth = {**ablation_config, 'label_smoothing': 0.0, 'run_name': 'no_label_smooth'}
        wandb.init(project=args.wandb_project, name='label_smoothing_0.0', config=config_no_smooth)
        train(config_no_smooth)

    elif args.experiment == 'learned_pe':
        ablation_config = {**base_config, 'num_epochs': min(base_config['num_epochs'], 20)}
        config_sin = {**ablation_config, 'use_learned_pe': False, 'run_name': 'sinusoidal_pe'}
        wandb.init(project=args.wandb_project, name='sinusoidal_pe', config=config_sin)
        train(config_sin)
        wandb.finish()

        config_learned = {**ablation_config, 'use_learned_pe': True, 'run_name': 'learned_pe'}
        wandb.init(project=args.wandb_project, name='learned_pe', config=config_learned)
        train(config_learned)

    wandb.finish()


if __name__ == '__main__':
    main()
