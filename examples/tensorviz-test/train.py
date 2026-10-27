"""Experiment 1: teach a tiny transformer to copy a sequence.

Run from this directory:  python train.py
The whole thing trains on CPU in well under a minute.
"""
import json

import tensorviz
import torch
import torch.nn.functional as F

from data.copy_task import make_loader
from models.tiny_transformer import TinyTransformer

tensorviz.experiment("copy-task-tiny-transformer")


def load_config(path: str = "config.json") -> dict:
    with open(path) as fh:
        return json.load(fh)


def train(model, loader, epochs: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / len(loader):.4f}")
    torch.save(model.state_dict(), "tiny_transformer.pt")
    return model


def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    cfg = load_config()
    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], max_len=cfg["seq_len"],
    )
    train_loader = make_loader(cfg["batch_size"], "train", seq_len=cfg["seq_len"], vocab_size=cfg["vocab_size"])
    val_loader = make_loader(cfg["batch_size"], "val", seq_len=cfg["seq_len"], vocab_size=cfg["vocab_size"])
    train(model, train_loader, cfg["epochs"], cfg["lr"])
    print(f"token accuracy: {evaluate(model, val_loader):.3f}")


if __name__ == "__main__":
    main()
