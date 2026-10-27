"""The training loop, as a class so a sweep can drive it without a subprocess."""
import torch
import torch.nn.functional as F

from training.checkpoint import save_checkpoint
from training.metrics import summarise, token_accuracy


class LanguageModelTrainer:
    def __init__(self, model, loader, lr: float = 3e-4, log_every: int = 20):
        self.model = model
        self.loader = loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.log_every = log_every
        self.history = []

    def train(self, epochs: int = 1):
        self.model.train()
        for epoch in range(epochs):
            losses = []
            for step, (inputs, targets) in enumerate(self.loader):
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                losses.append(loss.item())
                if step % self.log_every == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            self.history.append(summarise(losses))
        return self.history[-1] if self.history else {}

    def save(self, path: str = "runs/tiny_lm.pt"):
        return save_checkpoint(self.model, path, meta={"history": self.history})
