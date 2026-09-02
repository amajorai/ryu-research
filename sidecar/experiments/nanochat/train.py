# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Minimal single-file GPT (nanoGPT-style) — the mutable half of the program.

Trains a small character-level transformer and prints `val_bpb` (validation
bits-per-byte; **lower is better**), the scalar the autoresearch sidecar parses.
The researcher agent edits the HYPERPARAMETERS block (and may restructure the
model/optimizer) to lower `val_bpb`, keeping a change only when it improves.

Faithful to Karpathy's nanoGPT: a compact GPT with causal self-attention, run on
a single GPU (CUDA or Apple MPS), data produced by `prepare.py`.
"""

import math
import os
import pickle
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── HYPERPARAMETERS (the agent edits these) ──────────────────────────────────
BLOCK_SIZE = 128
BATCH_SIZE = 32
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.1
EVAL_ITERS = 100
SEED = 1337
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_data() -> int:
    """Run prepare.py if the token arrays are missing; return vocab_size."""
    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "prepare.py")], check=True)
    # Local-only: meta.pkl is written by prepare.py in this same workspace dir,
    # never fetched from an untrusted source (standard nanoGPT convention).
    with open(meta_path, "rb") as fh:
        return int(pickle.load(fh)["vocab_size"])


def get_batch(split: str, dev: str):
    path = os.path.join(DATA_DIR, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i : i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(dev), y.to(dev)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, dropout=DROPOUT, batch_first=True)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, N_EMBD)
        self.pos = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def estimate_val_loss(model: GPT, dev: str) -> float:
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        x, y = get_batch("val", dev)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return float(losses.mean())


def main() -> None:
    torch.manual_seed(SEED)
    dev = device()
    vocab_size = ensure_data()

    model = GPT(vocab_size).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_at(it: int) -> float:
        if it < WARMUP_ITERS:
            return LEARNING_RATE * (it + 1) / WARMUP_ITERS
        progress = (it - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
        return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    for it in range(MAX_ITERS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(it)
        x, y = get_batch("train", dev)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    val_loss = estimate_val_loss(model, dev)
    # Character corpus ≈ 1 byte/token, so bits-per-byte = nats-per-token / ln(2).
    val_bpb = val_loss / math.log(2)
    print(f"val_loss = {val_loss:.4f}")
    print(f"val_bpb = {val_bpb:.6f}")


if __name__ == "__main__":
    main()
