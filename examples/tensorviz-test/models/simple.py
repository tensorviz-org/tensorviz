"""A plain five-layer model — the per-class layer graph, alongside the experiment canvas."""
import torch
import torch.nn as nn


# tensorviz: input=1,3,32,32
class Simple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv(x)))
        return self.head(x.flatten(1))
