"""Small executable architecture for exploring hierarchy and verified skip paths."""
import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)

    def forward(self, x):
        branch = self.activation(self.norm1(self.conv1(x)))
        branch = self.norm2(self.conv2(branch))
        return branch + x


# tensorviz: input=1,3,32,32
class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.stage1 = ResidualUnit()
        self.stage2 = ResidualUnit()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        return self.head(torch.flatten(x, 1))
