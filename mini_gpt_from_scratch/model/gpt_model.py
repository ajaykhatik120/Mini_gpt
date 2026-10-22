from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 32, n_embd: int = 32, n_head: int = 4) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}")

        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.proj(attn_output)
        return x
