"""Baseline: train TinyLM on the embedded character corpus.

    python train.py

Trains on CPU in well under a minute. The declaration below names this file as an
experiment so the graph does not have to guess from the filename.
"""
import json
from pathlib import Path

import tensorviz

from data.loaders import make_splits
from evaluation.evaluate import evaluate
from models.tiny_lm import TinyLM
from training.trainer import LanguageModelTrainer

tensorviz.experiment("tinylm-baseline")
tensorviz.node("artifact", "runs/tiny_lm.pt")
tensorviz.edge("tinylm-trainer", "runs/tiny_lm.pt", "produces")


def load_config(path: str = "configs/base.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_model(config: dict, vocab_size: int) -> TinyLM:
    return TinyLM(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        depth=config["depth"],
        heads=config["heads"],
        context=config["context"],
    )


def main():
    config = load_config()
    train_loader, valid_loader = make_splits(
        batch_size=config["batch_size"], context=config["context"]
    )
    vocab_size = len(train_loader.dataset.dataset.vocabulary)

    model = build_model(config, vocab_size)
    trainer = LanguageModelTrainer(model, train_loader, lr=config["lr"])
    trainer.train(epochs=config["epochs"])

    report = evaluate(model, valid_loader)
    print(f"validation {report}")
    print(f"saved {trainer.save()}")


if __name__ == "__main__":
    main()
