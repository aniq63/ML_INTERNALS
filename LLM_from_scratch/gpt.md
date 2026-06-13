# GPT From Scratch

A minimal, fully-commented implementation of a GPT (decoder-only Transformer) trained on the [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset — built entirely from scratch using PyTorch.

Inspired by [Andrej Karpathy's](https://github.com/karpathy) nanoGPT series.

---

## What It Does

The model learns to predict the next character in a sequence of Shakespeare text. After training it can generate new text that *sounds* like Shakespeare — just from raw character probabilities, with no pre-trained weights.

**Sample output after training:**

```
ROMEO:
The sun ariseth from the golden east,
And with his light doth chase the darksome night...
```

---

## How It Works — The Big Picture

```
Raw text
  ↓  Character tokenisation
Token IDs  (1-D sequence of integers)
  ↓  Random batch sampling
(batch_size, block_size) tensor
  ↓  Token Embedding + Positional Embedding
(batch_size, block_size, d_model)
  ↓  × N TransformerBlocks
      ├── LayerNorm
      ├── Multi-Head Causal Self-Attention
      ├── Residual connection
      ├── LayerNorm
      ├── Feed-Forward Network (FFN)
      └── Residual connection
  ↓  Final LayerNorm
  ↓  Linear head  (d_model → vocab_size)
Logits  →  Cross-entropy loss  →  AdamW update
```

---

## Concepts Explained

### 1. Character-Level Tokenisation
Each unique character in the dataset becomes one token. The vocabulary is just 65 characters (letters, punctuation, spaces, newlines). This is simpler than word-level or BPE tokenisation used in production LLMs, but the architecture is identical.

### 2. Token + Positional Embeddings
A Transformer has no built-in sense of word order. We fix this by adding two learned embedding tables:
- **Token embedding** — maps each character ID to a `d_model`-dimensional vector.
- **Positional embedding** — maps each *position* (0, 1, 2, …) to another `d_model`-dimensional vector.

We add them together. Now every token's representation carries both *what it is* and *where it is*.

### 3. Causal Self-Attention
The core of the Transformer. For each token, attention computes:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

- **Q (Query)** — "what am I looking for?"
- **K (Key)** — "what do I contain?"
- **V (Value)** — "what do I communicate?"

The **causal mask** ensures position `t` can only attend to positions `≤ t` — the model cannot "cheat" by looking at future tokens.

### 4. Multi-Head Attention
Instead of one big attention pass, we run `n_heads` smaller attention heads in parallel — each focusing on different aspects of the input (syntax, semantics, proximity, etc.). Their outputs are concatenated and projected back to `d_model`.

### 5. Feed-Forward Network (FFN)
After attention (which mixes information *between* tokens), the FFN processes each position *independently*. It expands the dimension to `4 × d_model`, applies GELU, then compresses back. This is where most of the model's "thinking" capacity lives.

### 6. LayerNorm & Residual Connections
- **Residual connections** (`x = x + sublayer(x)`) let gradients flow unchanged during backprop, preventing the vanishing-gradient problem in deep networks.
- **LayerNorm** normalises each token's feature vector to zero mean and unit variance, stabilising training. We use *Pre-Norm* (applied before each sub-layer), which trains more stably than the original Post-Norm.

### 7. Language-Model Head & Loss
A final linear layer maps `d_model → vocab_size`, giving one logit per character. Cross-entropy loss measures how surprised the model was by the true next character. Lower loss = better predictions.

### 8. Perplexity
`PPL = exp(loss)`. A perplexity of 5 means the model is roughly as uncertain as if it had to choose uniformly from 5 characters — far better than a random model (PPL ≈ 65).

### 9. Text Generation
At inference time, we auto-regressively sample one token at a time:
1. Feed the current context through the model.
2. Take the logits at the *last* position.
3. Divide by **temperature** to control randomness.
4. Optionally apply **top-k** filtering (keep only the k most likely tokens).
5. Sample from the resulting distribution, append the token, repeat.

---

## Project Structure

```
gpt_from_scratch.py   ← single-file implementation (all you need)
README.md
```

---

## Quick Start

```bash
# 1. Install dependency
pip install torch

# 2. Run  (downloads dataset automatically)
python gpt_from_scratch.py
```

No other dependencies. The dataset (~1 MB) is downloaded from GitHub on first run.

### Run in Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `gpt_from_scratch.py`, then run:
```python
!python gpt_from_scratch.py
```
A free T4 GPU will train ~5× faster than CPU.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BLOCK_SIZE` | 128 | Context window (tokens the model sees at once) |
| `BATCH_SIZE` | 32 | Sequences per gradient step |
| `D_MODEL` | 256 | Embedding / hidden dimension |
| `N_HEADS` | 8 | Number of attention heads |
| `N_LAYERS` | 6 | Number of stacked Transformer blocks |
| `DROPOUT` | 0.1 | Dropout rate for regularisation |
| `MAX_ITERS` | 5000 | Training steps |
| `LR` | 3e-4 | AdamW learning rate |

To experiment, change these constants at the top of the file. Bigger `D_MODEL` / `N_LAYERS` → better output, slower training.

---

## Expected Results

| Metric | Value |
|--------|-------|
| Validation loss | ~1.5 |
| Perplexity | ~4–5 |
| Training time (CPU) | ~20 min |
| Training time (T4 GPU) | ~4 min |

---

## Key Design Choices

| Choice | Why |
|--------|-----|
| Character-level tokenisation | Simplest vocabulary; no tokeniser library needed |
| Pre-LayerNorm | More stable training than original Post-LN |
| GELU activation | Smooth; used in GPT-2/3 |
| Weight tying (embedding ↔ LM head) | Fewer parameters, often better PPL |
| Gradient clipping (`max_norm=1.0`) | Prevents exploding gradients |
| AdamW optimiser | Better weight decay than plain Adam |

---

## Learning Resources

- [Andrej Karpathy — "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (YouTube)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (original Transformer paper)
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)

---

## License

MIT — do whatever you want with it.