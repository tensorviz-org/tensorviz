from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.norm = nn.BatchNorm2d(width)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.first = ConvBlock(width)
        self.second = ConvBlock(width)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = self.second(self.first(x))
        return x + self.drop(residual)
