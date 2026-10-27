import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img=32, patch=4, dim=64):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, dim=64, depth=6, heads=4, classes=10):
        super().__init__()
        self.embed = PatchEmbed(dim=dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 65, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.embed(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


if __name__ == "__main__":
    print(TinyViT()(torch.randn(2, 3, 32, 32)).shape)
