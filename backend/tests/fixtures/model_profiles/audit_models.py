import torch
from torch import nn

# tensorviz: input=2,4
class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
