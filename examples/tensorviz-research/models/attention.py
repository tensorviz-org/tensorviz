"""Scaled dot-product attention, written out rather than calling F.sdpa.

Keeping the arithmetic visible is the point of this repository: the shapes are
what we are usually debugging.
"""
import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must divide evenly across heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        attended = self.drop(scores.softmax(dim=-1)) @ value
        merged = attended.transpose(1, 2).reshape(batch, seq, dim)
        return self.proj(merged)


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 128, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_model * expansion)
        self.act = nn.GELU()
        self.down = nn.Linear(d_model * expansion, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(self.act(self.up(x))))
