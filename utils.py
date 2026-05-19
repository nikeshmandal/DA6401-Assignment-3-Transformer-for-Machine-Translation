import torch
import torch.nn.functional as F
from evaluate import load as load_metric


def beam_search_decode(model, src, src_pad_idx, tgt_sos_idx, tgt_eos_idx, tgt_pad_idx, device, max_len=100, beam_size=4, length_penalty=0.6):
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        batch_size = src.size(0)
        results = []

        for b in range(batch_size):
            src_single = src[b:b+1]
            src_mask = model.make_src_mask(src_single, src_pad_idx)
            enc_output = model.encoder(src_single, src_mask)

            beams = [(0.0, [tgt_sos_idx], False)]

            for _ in range(max_len - 1):
                new_beams = []
                all_done = all(done for _, _, done in beams)
                if all_done:
                    break

                for score, tokens, done in beams:
                    if done:
                        new_beams.append((score, tokens, True))
                        continue

                    tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                    tgt_mask = model.make_tgt_mask(tgt_tensor, tgt_pad_idx)
                    dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
                    logits = model.fc_out(dec_output[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)[0]

                    topk_log_probs, topk_ids = log_probs.topk(beam_size)

                    for log_prob, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                        new_score = score + log_prob
                        new_tokens = tokens + [tok_id]
                        is_done = (tok_id == tgt_eos_idx)
                        new_beams.append((new_score, new_tokens, is_done))

                new_beams.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
                beams = new_beams[:beam_size]

            best_tokens = beams[0][1]
            results.append(best_tokens)

    return results



def train_epoch_v2(model, loader, optimizer, criterion, scheduler, device, src_pad_idx, tgt_pad_idx, clip=1.0):
    model.train()
    total_loss = 0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        num_tokens = (tgt_output != tgt_pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def evaluate_epoch(model, loader, criterion, device, src_pad_idx, tgt_pad_idx):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
            loss = criterion(output, tgt_output)

            num_tokens = (tgt_output != tgt_pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def greedy_decode(model, src, src_pad_idx, tgt_sos_idx, tgt_eos_idx, tgt_pad_idx, device, max_len=100):
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = model.make_src_mask(src, src_pad_idx)
        enc_output = model.encoder(src, src_mask)

        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), tgt_sos_idx, dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_mask = model.make_tgt_mask(tgt, tgt_pad_idx)
            dec_output = model.decoder(tgt, enc_output, src_mask, tgt_mask)
            logits = model.fc_out(dec_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            next_token[finished] = tgt_eos_idx
            tgt = torch.cat([tgt, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == tgt_eos_idx)
            if finished.all():
                break

    return tgt


def compute_bleu(model, loader, src_vocab, tgt_vocab, device, max_samples=500, use_beam=True, beam_size=4):
    bleu_metric = load_metric("bleu")
    model.eval()

    all_preds = []
    all_refs = []
    count = 0

    src_pad_idx = src_vocab.pad_idx
    tgt_pad_idx = tgt_vocab.pad_idx
    tgt_sos_idx = tgt_vocab.sos_idx
    tgt_eos_idx = tgt_vocab.eos_idx

    with torch.no_grad():
        for src, tgt in loader:
            if count >= max_samples:
                break
            src = src.to(device)
            tgt = tgt.to(device)

            if use_beam:
                pred_sequences = beam_search_decode(
                    model, src, src_pad_idx, tgt_sos_idx, tgt_eos_idx, tgt_pad_idx, device, beam_size=beam_size
                )
                for i in range(src.size(0)):
                    tokens = pred_sequences[i]
                    pred_tokens = []
                    for tok_id in tokens[1:]:
                        if tok_id == tgt_eos_idx:
                            break
                        if tok_id not in (tgt_pad_idx, tgt_sos_idx):
                            pred_tokens.append(tgt_vocab.itos[tok_id])

                    ref_tokens = []
                    for tok_id in tgt[i, 1:].tolist():
                        if tok_id == tgt_eos_idx:
                            break
                        if tok_id not in (tgt_pad_idx, tgt_sos_idx):
                            ref_tokens.append(tgt_vocab.itos[tok_id])

                    if pred_tokens and ref_tokens:
                        all_preds.append(pred_tokens)
                        all_refs.append([ref_tokens])
            else:
                pred_ids = greedy_decode(model, src, src_pad_idx, tgt_sos_idx, tgt_eos_idx, tgt_pad_idx, device)

                for i in range(src.size(0)):
                    pred_tokens = []
                    for tok_id in pred_ids[i, 1:].tolist():
                        if tok_id == tgt_eos_idx:
                            break
                        if tok_id not in (tgt_pad_idx, tgt_sos_idx):
                            pred_tokens.append(tgt_vocab.itos[tok_id])

                    ref_tokens = []
                    for tok_id in tgt[i, 1:].tolist():
                        if tok_id == tgt_eos_idx:
                            break
                        if tok_id not in (tgt_pad_idx, tgt_sos_idx):
                            ref_tokens.append(tgt_vocab.itos[tok_id])

                    if pred_tokens and ref_tokens:
                        all_preds.append(pred_tokens)
                        all_refs.append([ref_tokens])

            count += src.size(0)

    if not all_preds:
        return 0.0

    # evaluate BLEU expects strings not token lists
    str_preds = [' '.join(p) for p in all_preds]
    str_refs  = [[' '.join(r) for r in ref] for ref in all_refs]
    result = bleu_metric.compute(predictions=str_preds, references=str_refs)
    return result['bleu']


def get_gradient_norms(model):
    total_norm = 0.0
    qk_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            if 'W_q' in name or 'W_k' in name:
                qk_norm += param_norm.item() ** 2
    return total_norm ** 0.5, qk_norm ** 0.5
