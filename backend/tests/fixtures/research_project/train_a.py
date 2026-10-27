"""Discovered by its __main__ block, not by a declaration."""
from common import evaluate, load_config
from data.toy import make_loader
from models.net import Net


def train(epochs: int = 1):
    model = Net()
    loader = make_loader()
    optimiser = _optimiser(model)
    for _ in range(epochs):
        for batch in loader:
            loss = model(batch).sum()
            loss.backward()
            optimiser.step()
    return model


def _optimiser(model):
    import torch

    return torch.optim.SGD(model.parameters(), lr=0.1)


if __name__ == "__main__":
    cfg = load_config()
    trained = train()
    evaluate(trained, make_loader())
