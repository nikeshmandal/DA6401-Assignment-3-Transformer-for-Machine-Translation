import math
import torch
import torch.nn as nn


class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_lr(self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

    def get_lr(self):
        return self._get_lr()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, output, target):
        output = output.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)

        log_probs = torch.nn.functional.log_softmax(output, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0

        mask = (target != self.pad_idx)
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        loss = loss.masked_select(mask).mean()
        return loss
