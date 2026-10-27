import json


def load_config(path: str = "config.json") -> dict:
    with open(path) as handle:
        return json.load(handle)


def evaluate(model, loader):
    import torch

    model.eval()
    with torch.no_grad():
        return sum(1 for _ in loader)
