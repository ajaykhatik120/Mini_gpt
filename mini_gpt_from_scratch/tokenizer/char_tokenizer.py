from __future__ import annotations

from collections import Counter
from typing import List


class CharTokenizer:
    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def fit(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for idx, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            self.fit(text)
        return [self.stoi[ch] for ch in text]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
