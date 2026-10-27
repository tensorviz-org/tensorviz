"""Depth sweep, run as a plain script.

    python sweep.py

Deliberately has no `if __name__ == "__main__":` guard — plenty of research code
is written this way, executed top to bottom.
"""
from data.loaders import make_splits
from evaluation.evaluate import evaluate
from models.tiny_lm import TinyLM
from training.trainer import LanguageModelTrainer

DEPTHS = (2, 4)
results = {}

train_loader, valid_loader = make_splits(batch_size=16, context=64)
vocabulary = train_loader.dataset.dataset.vocabulary

for depth in DEPTHS:
    candidate = TinyLM(vocab_size=len(vocabulary), depth=depth, context=64)
    sweep_trainer = LanguageModelTrainer(candidate, train_loader, lr=3e-4)
    sweep_trainer.train(epochs=1)
    results[depth] = evaluate(candidate, valid_loader)
    print(f"depth {depth}: {results[depth]}")

best = min(results, key=lambda key: results[key]["loss"])
print(f"best depth {best}")
