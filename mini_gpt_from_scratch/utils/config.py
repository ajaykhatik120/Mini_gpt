from dataclasses import dataclass


@dataclass
class Config:
    block_size: int = 32
    epochs: int = 50
    max_new_tokens: int = 30
