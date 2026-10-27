from torch import nn


class Head(nn.Module):
    def __init__(self, width=16, classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(width, classes)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(x)))
