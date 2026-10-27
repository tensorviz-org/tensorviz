"""The model under study: a small causal language model over characters.

Three levels of composition, which is deliberate — `TinyLM` holds
`TransformerBlock`s, each of which holds a `MultiHeadAttention` and a
`FeedForward`. Anything that claims to draw this project has to follow both hops.
"""
import torch
import torch.nn as nn

from models.blocks import TransformerBlock

# tensorviz: input=2,64


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 96,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        context: int = 64,
    ):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model=d_model, heads=heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        _, seq = idx.shape
        positions = torch.arange(seq, device=idx.device)
        x = self.token(idx) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, steps: int = 32, temperature: float = 1.0):
        for _ in range(steps):
            window = idx[:, -self.context :]
            logits = self(window)[:, -1, :] / temperature
            nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
