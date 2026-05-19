import torch
import argparse
from model import Transformer
from utils import beam_search_decode


def translate(sentence, model, src_vocab, tgt_vocab, src_tokenizer, device, max_len=100, beam_size=4):
    model.eval()

    tokens = [tok.text.lower() for tok in src_tokenizer(sentence)]
    src_ids = [src_vocab.sos_idx] + [src_vocab.stoi.get(tok, src_vocab.unk_idx) for tok in tokens] + [src_vocab.eos_idx]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)

    pred_sequences = beam_search_decode(
        model, src_tensor,
        src_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx,
        device, max_len=max_len, beam_size=beam_size
    )

    pred_tokens = []
    for tok_id in pred_sequences[0][1:]:
        if tok_id == tgt_vocab.eos_idx:
            break
        if tok_id not in (tgt_vocab.pad_idx, tgt_vocab.sos_idx):
            pred_tokens.append(tgt_vocab.itos[tok_id])

    return ' '.join(pred_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sentence', type=str, default=None)
    parser.add_argument('--beam_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']

    import spacy
    src_tokenizer = spacy.load("de_core_news_sm")

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
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (Val BLEU: {checkpoint['val_bleu']:.4f})")

    if args.sentence:
        translation = translate(args.sentence, model, src_vocab, tgt_vocab, src_tokenizer, device, beam_size=args.beam_size)
        print(f"Source:      {args.sentence}")
        print(f"Translation: {translation}")
    else:
        test_sentences = [
            "Ein Mann schaut sich ein Gemälde in einem Museum an.",
            "Zwei Hunde spielen auf einer Wiese.",
            "Ein kleines Mädchen läuft auf der Straße.",
        ]
        for sent in test_sentences:
            translation = translate(sent, model, src_vocab, tgt_vocab, src_tokenizer, device, beam_size=args.beam_size)
            print(f"DE: {sent}")
            print(f"EN: {translation}")
            print()


if __name__ == '__main__':
    main()
