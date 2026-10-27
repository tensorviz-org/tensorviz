"""A small encoder-only transformer for the copy task."""
# tensorviz: input=torch.randint(1, 16, (1, 10))
import math

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 16, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 12, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.pos, std=1 / math.sqrt(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        x = self.encoder(x)
        return self.head(self.norm(x))

