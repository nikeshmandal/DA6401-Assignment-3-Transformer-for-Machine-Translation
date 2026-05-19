import torch
import argparse
from evaluate import load as load_metric
from model import Transformer
from dataset import load_data_and_vocab
from utils import beam_search_decode, greedy_decode


def evaluate_bleu(checkpoint_path, batch_size=32, beam_size=4):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']

    _, _, test_loader, _, _, _, _ = load_data_and_vocab(batch_size=batch_size)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        use_scale=config.get('use_scale', True),
        use_learned_pe=config.get('use_learned_pe', False)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    bleu_metric = load_metric("bleu")
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            pred_sequences = beam_search_decode(
                model, src,
                src_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx,
                device, beam_size=beam_size
            )

            for i in range(src.size(0)):
                pred_tokens = []
                for tok_id in pred_sequences[i][1:]:
                    if tok_id == tgt_vocab.eos_idx:
                        break
                    if tok_id not in (tgt_vocab.pad_idx, tgt_vocab.sos_idx):
                        pred_tokens.append(tgt_vocab.itos[tok_id])

                ref_tokens = []
                for tok_id in tgt[i, 1:].tolist():
                    if tok_id == tgt_vocab.eos_idx:
                        break
                    if tok_id not in (tgt_vocab.pad_idx, tgt_vocab.sos_idx):
                        ref_tokens.append(tgt_vocab.itos[tok_id])

                if pred_tokens and ref_tokens:
                    all_preds.append(pred_tokens)
                    all_refs.append([ref_tokens])

    result = bleu_metric.compute(predictions=all_preds, references=all_refs)
    bleu = result['bleu']
    print(f"Test BLEU: {bleu:.4f} ({bleu*100:.2f})")
    return bleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beam_size', type=int, default=4)
    args = parser.parse_args()
    evaluate_bleu(args.checkpoint, args.batch_size, args.beam_size)
