"""One transformer block: pre-norm attention, pre-norm MLP, both residual."""
import torch.nn as nn

from models.attention import FeedForward, MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
