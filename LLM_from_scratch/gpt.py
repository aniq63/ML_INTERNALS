"""
gpt_from_scratch.py
====================
A minimal but complete GPT (decoder-only Transformer) trained on the Tiny
Shakespeare dataset — built entirely from scratch with PyTorch.

Inspired by Andrej Karpathy's "makemore" / nanoGPT series.

Pipeline
--------
1.  Download & tokenise the dataset (character-level)
2.  Build a DataLoader (random batch sampler)
3.  Define the model:
        Token + Positional Embeddings
        → N × TransformerBlock  (Multi-Head Self-Attention + FFN + LayerNorm)
        → Language-Model Head   (linear projection → vocab logits)
4.  Train with AdamW
5.  Evaluate perplexity on the validation split
6.  Generate new text with temperature / top-k sampling

Run:
    pip install torch
    python gpt_from_scratch.py
"""

import math
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0.  REPRODUCIBILITY & DEVICE
# ============================================================

SEED = 1337
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")
print(f"PyTorch      : {torch.__version__}\n")


# ============================================================
# 1.  DATASET  — Tiny Shakespeare (character-level)
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
urllib.request.urlretrieve(DATA_URL, "input.txt")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset size : {len(text):,} characters")
print("First 200 chars:\n", text[:200], "\n")

# --- Vocabulary (all unique characters) ---
chars     = sorted(set(text))
vocab_size = len(chars)

# Character ↔ integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}   # char → int
itos = {i: ch for i, ch in enumerate(chars)}   # int  → char

def encode(s: str) -> list[int]:
    """Convert a string to a list of token IDs."""
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    """Convert a list of token IDs back to a string."""
    return "".join(itos[i] for i in ids)

print(f"Vocabulary size : {vocab_size}")
print(f"All characters  : {''.join(chars)}\n")

# --- Encode the full text into a 1-D LongTensor ---
data = torch.tensor(encode(text), dtype=torch.long)

# 90 / 10 train / validation split
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Train tokens : {train_data.shape[0]:,}")
print(f"Val tokens   : {val_data.shape[0]:,}\n")


# ============================================================
# 2.  HYPERPARAMETERS
# ============================================================

# Data
BLOCK_SIZE  = 128   # context window (tokens the model sees at once)
BATCH_SIZE  = 32    # sequences per gradient step

# Model
D_MODEL     = 256   # embedding / hidden dimension
N_HEADS     = 8     # number of attention heads (D_MODEL must be divisible)
N_LAYERS    = 6     # number of stacked Transformer blocks
DROPOUT     = 0.1   # dropout rate for regularisation

# Training
MAX_ITERS   = 5_000    # total gradient update steps
EVAL_ITERS  = 200      # batches to average for validation loss
EVAL_EVERY  = 500      # evaluate every N steps
LR          = 3e-4     # learning rate for AdamW


# ============================================================
# 3.  DATA LOADER
# ============================================================

def get_batch(split: str):
    """
    Sample a random mini-batch of (inputs, targets).

    Each sequence has length BLOCK_SIZE.
    The targets are the inputs shifted right by one token — the classic
    next-token prediction (language-model) objective.

    Args:
        split: "train" or "val"

    Returns:
        x: LongTensor (BATCH_SIZE, BLOCK_SIZE)  — input token IDs
        y: LongTensor (BATCH_SIZE, BLOCK_SIZE)  — target token IDs
    """
    src = train_data if split == "train" else val_data
    # Pick BATCH_SIZE random starting positions
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i      : i + BLOCK_SIZE    ] for i in ix])
    y   = torch.stack([src[i + 1  : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 4.  MODEL COMPONENTS
# ============================================================

class SingleHeadAttention(nn.Module):
    """
    One head of causal (masked) self-attention.

    Steps
    -----
    1. Project every token into Q, K, V vectors.
    2. Compute scaled dot-product attention scores.
    3. Apply a causal mask so position t can only attend to positions ≤ t.
    4. Softmax → attention weights.
    5. Weighted sum of V vectors → output.
    """

    def __init__(self, d_model: int, head_size: int):
        super().__init__()
        # Linear projections (no bias, matching GPT-2 convention)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.drop  = nn.Dropout(DROPOUT)

        # Lower-triangular mask stored as a non-learnable buffer
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, time-steps, channels

        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Scale dot-product scores to prevent vanishing gradients in softmax
        scale  = k.shape[-1] ** -0.5
        scores = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Mask future positions with −∞ (they become 0 after softmax)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.drop(weights)

        out = weights @ v   # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Run N_HEADS attention heads in parallel, then project back to D_MODEL.

    Splitting D_MODEL across heads means each head can specialise in
    different types of token relationships simultaneously.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_size = d_model // n_heads

        self.heads = nn.ModuleList(
            [SingleHeadAttention(d_model, head_size) for _ in range(n_heads)]
        )
        # Final linear to blend the concatenated head outputs
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each head produces (B, T, head_size); concatenate along last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, D_MODEL)
        out = self.drop(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Applied independently to every token position after attention.
    Expands to 4× the hidden dimension (per the original Transformer paper),
    applies GELU, then projects back.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer decoder block:

        x → LayerNorm → MultiHeadAttention → residual add
          → LayerNorm → FeedForward        → residual add

    Residual connections let gradients flow directly through the network
    during backpropagation (solve vanishing-gradient problem).
    Pre-LayerNorm (applied before each sub-layer) is used here for more
    stable training than the original Post-LayerNorm formulation.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn  = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection
        x = x + self.attn(self.ln1(x))
        # Feed-forward sub-layer with residual connection
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# 5.  GPT MODEL
# ============================================================

class GPT(nn.Module):
    """
    Decoder-only GPT architecture.

    Forward pass
    ------------
    Token IDs
      → Token Embedding  (learned vector per token)
    + Positional Embedding (learned vector per position)
      → N × TransformerBlock
      → Final LayerNorm
      → Linear head  (D_MODEL → vocab_size)
      → logits

    The loss is cross-entropy between logits and the next-token targets.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__()
        self.block_size = block_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(block_size, d_model)

        # Stacked Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        # Final normalisation before the language-model head
        self.ln_f = nn.LayerNorm(d_model)

        # Language-model head: maps hidden states → vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and LM-head weights
        # (reduces parameters and often improves performance)
        self.lm_head.weight = self.token_emb.weight

        # Initialise weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """Standard GPT weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,          # (B, T) token IDs
        targets: torch.Tensor = None,   # (B, T) token IDs — optional
    ):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        # Token + positional embeddings
        tok_emb = self.token_emb(idx)                              # (B, T, D)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))# (T, D)
        x       = tok_emb + pos_emb                                # (B, T, D)

        # Transformer blocks
        x = self.blocks(x)    # (B, T, D)
        x = self.ln_f(x)      # (B, T, D)

        # Project to vocabulary logits
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # Compute cross-entropy loss when targets are provided
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 6.  INSTANTIATE & INSPECT
# ============================================================

model = GPT(
    vocab_size  = vocab_size,
    d_model     = D_MODEL,
    block_size  = BLOCK_SIZE,
    n_layers    = N_LAYERS,
    n_heads     = N_HEADS,
).to(device)

print(f"Model parameters : {model.count_parameters():,}\n")


# ============================================================
# 7.  TRAINING LOOP
# ============================================================

@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """
    Average the loss over EVAL_ITERS batches for train and val splits.
    Returns a dict like {"train": 1.23, "val": 1.45}.
    Runs in no_grad mode to save memory.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(EVAL_ITERS):
            xb, yb      = get_batch(split)
            _, loss     = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("=" * 60)
print("TRAINING")
print("=" * 60)

for step in range(MAX_ITERS):

    # --- Evaluate periodically ---
    if step % EVAL_EVERY == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss()
        print(
            f"Step {step:5d}/{MAX_ITERS} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    # --- Forward + backward ---
    xb, yb     = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Gradient clipping prevents exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


# ============================================================
# 8.  EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model():
    """
    Detailed evaluation:
      - Perplexity (exp of validation loss)
      - Top-5 token predictions for the first few positions of a sample
    """
    model.eval()

    # --- Perplexity ---
    losses = [
        model(*get_batch("val"))[1].item()
        for _ in range(EVAL_ITERS)
    ]
    avg_loss   = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Validation loss : {avg_loss:.4f}")
    print(f"Perplexity      : {perplexity:.2f}")
    print(
        f"  (A random model would have PPL ≈ {vocab_size}; "
        f"we've narrowed it to ~{perplexity:.0f} choices per step)"
    )

    # --- Top-5 predictions for a sample sequence ---
    print("\n--- Top-5 predictions at first 5 positions ---")
    xb, yb = get_batch("val")
    logits, _ = model(xb[:1], yb[:1])  # single sequence

    print(f"Input  : '{decode(xb[0].tolist())[:30]}...'")
    print(f"Target : '{decode(yb[0].tolist())[:30]}...'")
    print()

    for pos in range(min(5, logits.shape[1])):
        probs               = F.softmax(logits[0, pos], dim=0)
        top5_probs, top5_idx = probs.topk(5)
        true_token          = yb[0, pos].item()
        true_char           = itos[true_token]
        hit                 = "✓" if true_token in top5_idx.tolist() else "✗"

        preds = ", ".join(
            f"'{itos[idx.item()]}' ({p:.1%})"
            for idx, p in zip(top5_idx, top5_probs)
        )
        print(f"  pos {pos}: true='{true_char}' | {preds} [{hit}]")

    model.train()
    return avg_loss, perplexity


avg_loss, perplexity = evaluate_model()


# ============================================================
# 9.  TEXT GENERATION
# ============================================================

@torch.no_grad()
def generate_text(
    model: GPT,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """
    Auto-regressively sample tokens from the model.

    Args:
        prompt         : String to condition generation on.
        max_new_tokens : Number of new characters to produce.
        temperature    : Values < 1 make output more focused / repetitive;
                         values > 1 make it more random / creative.
        top_k          : If set, restrict sampling to the top-k most-likely
                         tokens at each step (nucleus-style filtering).

    Returns:
        The full string (prompt + generated continuation).
    """
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to at most block_size tokens
        idx_cond = idx[:, -model.block_size:]

        # Forward pass → logits for the last position only
        logits, _    = model(idx_cond)
        logits_last  = logits[:, -1, :] / temperature  # (1, vocab_size)

        # Top-k filtering: zero out everything below the k-th logit
        if top_k is not None:
            top_k_logits, _ = logits_last.topk(top_k)
            min_val          = top_k_logits[:, -1].unsqueeze(-1)
            logits_last      = logits_last.masked_fill(
                logits_last < min_val, float("-inf")
            )

        # Sample from the probability distribution
        probs    = F.softmax(logits_last, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, idx_next], dim=1)

    model.train()
    return decode(idx[0].tolist())


print("\n" + "=" * 60)
print("TEXT GENERATION  (prompt = 'ROMEO:')")
print("=" * 60)

configs = [
    ("Temperature = 0.5 (focused)",   dict(temperature=0.5)),
    ("Temperature = 1.0 (standard)",  dict(temperature=1.0)),
    ("Temperature = 1.5 (creative)",  dict(temperature=1.5)),
    ("Top-k = 5   (restricted)",      dict(temperature=1.0, top_k=5)),
]

for title, kwargs in configs:
    print(f"\n--- {title} ---")
    print(generate_text(model, prompt="ROMEO:", max_new_tokens=300, **kwargs))





# =========================== OUTPUT ============================
# Using device : cuda
# PyTorch      : 2.11.0+cu128

# Dataset size : 1,115,394 characters
# First 200 chars:
#  First Citizen:
# Before we proceed any further, hear me speak.

# All:
# Speak, speak.

# First Citizen:
# You are all resolved rather to die than to famish?

# All:
# Resolved. resolved.

# First Citizen:
# First, you 

# Vocabulary size : 65
# All characters  : 
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

# Train tokens : 1,003,854
# Val tokens   : 111,540

# Model parameters : 4,783,872

# ============================================================
# TRAINING
# ============================================================
# Step     0/5000 | train loss 4.2072 | val loss 4.2063
# Step   500/5000 | train loss 2.0389 | val loss 2.1038
# Step  1000/5000 | train loss 1.6838 | val loss 1.8368
# Step  1500/5000 | train loss 1.5231 | val loss 1.7033
# Step  2000/5000 | train loss 1.4241 | val loss 1.6317
# Step  2500/5000 | train loss 1.3612 | val loss 1.5814
# Step  3000/5000 | train loss 1.3173 | val loss 1.5495
# Step  3500/5000 | train loss 1.2764 | val loss 1.5287
# Step  4000/5000 | train loss 1.2509 | val loss 1.5102
# Step  4500/5000 | train loss 1.2269 | val loss 1.5125
# Step  4999/5000 | train loss 1.2028 | val loss 1.5087

# ============================================================
# EVALUATION
# ============================================================
# Validation loss : 1.5032
# Perplexity      : 4.50
#   (A random model would have PPL ≈ 65; we've narrowed it to ~4 choices per step)

# --- Top-5 predictions at first 5 positions ---
# Input  : 'eir tutors: bid them use them ...'
# Target : 'ir tutors: bid them use them w...'

#   pos 0: true='i' | ' ' (22.1%), 'n' (13.8%), 'r' (12.4%), 's' (8.3%), 'a' (7.0%) [✗]
#   pos 1: true='r' | 'n' (32.6%), 'r' (22.6%), 'g' (17.6%), 't' (13.9%), 'v' (3.3%) [✓]
#   pos 2: true=' ' | ' ' (75.0%), 'e' (5.9%), 's' (4.8%), ',' (4.0%), '.' (2.7%) [✓]
#   pos 3: true='t' | 's' (10.9%), 't' (9.8%), 'o' (7.3%), 'w' (6.9%), 'a' (6.3%) [✓]
#   pos 4: true='u' | 'o' (45.7%), 'h' (31.6%), 'r' (10.0%), 'e' (4.1%), 'i' (3.4%) [✗]

# ============================================================
# TEXT GENERATION  (prompt = 'ROMEO:')
# ============================================================

# --- Temperature = 0.5 (focused) ---
# ROMEO:
# I do begun, sir, and a man too the love
# And be sign of a word to thee abuse the rest,
# Where the senate are to the wanton and thou shalt speak.

# PARIS:
# And so, what we will not?

# MENIUS:
# There's no more to the book of thine eyes;
# But that we will temper with the tribunes:
# Yet their gentleman seems m

# --- Temperature = 1.0 (standard) ---
# ROMEO:
# Why, hail he do lour, or else so.

# LADY GREY:
# I rather then, my good lord; and what they do
# And getten glad straight and hang ere they summeral
# For some lives to thee party to exquart theughtard
# with bows. And in my by such braggage as fast?
# If, I be noble in letter he may see
# had forward full abou

# --- Temperature = 1.5 (creative) ---
# ROMEO:
# And, come Ta'bey, sister.

# Provost:
# Well, guards he,
# Even restumble abovoc upon't.
# Pry-cipation were 'twas dare so, a bosseman;
# Not were a Emiman afford turn! Defit
# And, so: going'd him, and, make Puts Englands: his own now
# highes hook: the unfaults favot. Stalple
# how
# Is that commun made aharns
# uck

# --- Top-k = 5   (restricted) ---
# ROMEO:
# Now the gods to be made, and the gods wear of mine.

# KING RICHARD III:
# To him truly state, as if I will not say.

# KING RICHARD II:
# Stay.

# KING EDWARD III:
# Ay, thank thy things; as I will answer havour.
# Why dost thou have talk me with my fortune law to thee a few.

# KING HENRY VI:
# How now, who, who t
