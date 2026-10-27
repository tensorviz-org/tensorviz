"""Import forms a static projection deliberately refuses to guess.

`load_block_class` picks a module at runtime (computed string) and reads the class
off it with a computed getattr; `DynamicStage` then stores whatever came back. A
resolver that claimed to know which class `.inner` holds would be lying, so the
node stays opaque with a diagnostic instead.
"""
import importlib

import torch.nn as nn

_BACKENDS = {"fast": "models.blocks", "slow": "models.heads"}


def load_block_class(kind: str = "fast"):
    module = importlib.import_module(_BACKENDS[kind])
    return getattr(module, "Block")


class DynamicStage(nn.Module):
    def __init__(self, width: int = 8, kind: str = "fast"):
        super().__init__()
        block_cls = load_block_class(kind)
        self.inner = block_cls(width)
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return self.fc(self.inner(x))
