### Name: Nikesh Kumar Mandal
### Roll: ID25M805
### DL ASSIGNMENT 3

# Wandb Link:
https://api.wandb.ai/links/id25m805-iitmaana/y8p456bb

# Github repo link:
https://github.com/nikeshmandal/DA6401-Assignment-3-Transformer-for-Machine-Translation.git

# DA6401 Assignment 3 — Transformer for Machine Translation (German to English)


Implementation of "Attention Is All You Need" (Vaswani et al., 2017) from scratch using PyTorch, trained on the Multi30k dataset.

---

## Project Structure

```
da6401_assignment_3/
├── model.py           # Transformer architecture (MHA, Encoder, Decoder, PE)
├── dataset.py         # Multi30k loading, vocabulary, tokenization
├── scheduler.py       # Noam LR scheduler and label smoothing loss
├── utils.py           # Training loop, evaluation, BLEU, greedy decode
├── train.py           # Main training script with all W&B experiments
├── translate.py       # Inference script for translating new sentences
├── evaluate_bleu.py   # Standalone BLEU evaluation on test set
├── requirements.txt
└── README.md
```

---

## Environment Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy language models

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 3. Log in to Weights & Biases

```bash
wandb login
```

Paste your API key from https://wandb.ai/authorize

---

## Training

All experiments are run via `train.py` with the `--experiment` flag.

### Baseline (best model — use this for submission)

```bash
python train.py --experiment baseline --epochs 15
```

This trains with:
- Noam scheduler (warmup 4000 steps)
- Label smoothing 0.1
- Sinusoidal positional encoding
- Scaled dot-product attention

The best checkpoint is saved as `best_model_baseline.pt`.

### W&B Experiments (required for report)

Run each experiment separately. Each produces two W&B runs that can be overlaid in the report.

**Experiment 1: Noam vs Fixed LR (Section 2.1)**
```bash
python train.py --experiment noam_vs_fixed --epochs 15
```

**Experiment 2: Scaling Factor Ablation (Section 2.2)**
```bash
python train.py --experiment scale_ablation --epochs 15
```

**Experiment 3: Label Smoothing (Section 2.5)**
```bash
python train.py --experiment label_smoothing --epochs 15
```

**Experiment 4: Learned vs Sinusoidal PE (Section 2.4)**
```bash
python train.py --experiment learned_pe --epochs 15
```

Attention heatmaps (Section 2.3) are automatically logged at the end of every training run.

---

## Inference

Translate a custom German sentence using a saved checkpoint:

```bash
python translate.py --checkpoint best_model_baseline.pt --sentence "Ein Mann läuft durch den Park."
```

Run default test sentences (no `--sentence` flag):

```bash
python translate.py --checkpoint best_model_baseline.pt
```

---

## Test BLEU Evaluation

```bash
python evaluate_bleu.py --checkpoint best_model_baseline.pt
```

---

## Hyperparameters (Default)

| Parameter | Value |
|---|---|
| d_model | 256 |
| num_heads | 8 |
| d_ff | 512 |
| num_layers | 3 |
| dropout | 0.1 |
| warmup_steps | 4000 |
| batch_size | 128 |
| label_smoothing | 0.1 |
| max_seq_len | 256 |
| optimizer | Adam (β1=0.9, β2=0.98, ε=1e-9) |

These follow the small-model configuration from the original paper, scaled down for resource-constrained training.

---

## Expected Training Time (Mac M4 Air, 16GB RAM, MPS)

| Configuration | Time per Epoch | Total (15 epochs) |
|---|---|---|
| Baseline | ~4-5 min | ~60-75 min |
| Each ablation (2 runs) | ~4-5 min each | ~8-10 min |
| All 4 experiments (8 runs) | — | ~6-7 hours |

**Tip:** Run experiments sequentially. MPS does not support all operations at full efficiency, so some ops fall back to CPU. The dataset is small (29k training pairs) so training is manageable.

---

## Expected BLEU Scores

| Configuration | Val BLEU (approx.) |
|---|---|
| Baseline (Noam + LS 0.1 + sinusoidal) | 28-33 |
| Fixed LR | 20-26 |
| Without sqrt(d_k) scaling | 15-22 |
| No label smoothing | 26-31 |
| Learned PE | 27-32 |

---

## Device Selection

The code auto-selects the best available device: MPS (Apple Silicon) > CUDA > CPU.

---


## Architecture Notes

**Why Post-LayerNorm?**
This implementation uses Post-LayerNorm (Sublayer output is normed after the residual addition) following the original paper exactly. Post-LN is harder to train but generally achieves slightly better final performance on small datasets. The Noam warmup is especially important with Post-LN to prevent early instability.

**MPS Notes**
Apple MPS does not support `float64`; all tensors default to `float32`. If you encounter MPS errors, add `PYTORCH_ENABLE_MPS_FALLBACK=1` before your training command:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --experiment baseline
```
