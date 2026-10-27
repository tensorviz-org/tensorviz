"""A compact encoder, with honest static boundaries for attention internals."""
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.expand = nn.Linear(64, 256)
        self.gelu = nn.GELU()
        self.project = nn.Linear(256, 64)

    def forward(self, x):
        return self.project(self.gelu(self.expand(self.norm(x)))) + x


# tensorviz: input=2,16,64
class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.feed_forward = FeedForward()
        self.output_norm = nn.LayerNorm(64)

    def forward(self, x):
        attention, _ = self.attention(x, x, x, need_weights=False)
        x = attention + x
        return self.output_norm(self.feed_forward(x))
