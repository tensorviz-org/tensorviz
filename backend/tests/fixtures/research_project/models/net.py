import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))
