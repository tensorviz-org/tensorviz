"""A 101.94B decoder for shape-only capture regressions; never allocate real weights."""
import math

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class GroupedAttention(nn.Module):
    def __init__(self, width, heads, kv_heads):
        super().__init__()
        self.heads, self.kv_heads, self.head_width = heads, kv_heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.value = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, causal_mask):
        batch, length, width = x.shape
        q = self.query(x).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = self.key(x).reshape(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        v = self.value(x).reshape(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        frequencies = 10000 ** (-torch.arange(0, self.head_width, 2, dtype=torch.float32) / self.head_width)
        angles = torch.arange(length, dtype=torch.float32)[:, None] * frequencies[None, :]
        cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
        def rotary(value):
            even, odd = value[..., ::2], value[..., 1::2]
            return torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)
        q, k = rotary(q), rotary(k)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        weights = torch.softmax(scores.masked_fill(causal_mask, float('-inf')), dim=-1)
        mixed = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        return self.output(mixed)


class GatedMLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.down(self.activation(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, width, hidden, heads, kv_heads):
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = GroupedAttention(width, heads, kv_heads)
        self.ffn_norm = RMSNorm(width)
        self.ffn = GatedMLP(width, hidden)

    def forward(self, x, causal_mask):
        x = x + self.attention(self.attention_norm(x), causal_mask)
        return x + self.ffn(self.ffn_norm(x))


class Decoder100B(nn.Module):
    def __init__(self):
        super().__init__()
        if not isinstance(torch.empty(0), FakeTensor):
            raise RuntimeError('Decoder100B requires caller-owned FakeTensorMode')
        width, hidden, depth, heads, kv_heads = 14336, 38912, 48, 112, 8
        self.embedding = nn.Embedding(32768, width)
        self.blocks = nn.ModuleList([DecoderBlock(width, hidden, heads, kv_heads) for _ in range(depth)])
        self.final_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, 32768, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        length = input_ids.shape[1]
        causal_mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.lm_head(self.final_norm(x))
