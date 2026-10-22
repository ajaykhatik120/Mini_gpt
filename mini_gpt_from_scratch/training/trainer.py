from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


def train_simple_model(model: nn.Module, tokens: list[int], epochs: int = 5, lr: float = 1e-3) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    if len(tokens) < 2:
        raise ValueError("At least two tokens are required for training")

    block_size = getattr(model, "block_size", len(tokens) - 1)

    for _ in range(epochs):
        total_loss = 0.0
        n_chunks = 0
        for i in range(0, len(tokens) - 1, block_size):
            chunk = tokens[i : i + block_size + 1]
            if len(chunk) < 2:
                continue
            xb = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0)
            yb = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_chunks += 1

    return float(total_loss / max(n_chunks, 1))

