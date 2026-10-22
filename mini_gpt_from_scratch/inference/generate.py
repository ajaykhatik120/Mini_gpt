from __future__ import annotations

import torch
import torch.nn as nn


def generate_text(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    model.eval()
    generated = prompt
    current = tokenizer.encode(prompt)
    input_ids = torch.tensor(current, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        block_size = getattr(model, "block_size", None)
        for _ in range(max_new_tokens):
            idx_cond = input_ids if block_size is None else input_ids[:, -block_size:]
            logits = model(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            next_token_item = int(next_token.item())
            generated += tokenizer.itos[next_token_item]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

    return generated
