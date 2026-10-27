"""Relative-from import (`from .blocks import Block`) into a plain owner module."""
from .blocks import Block


class Mixer(Block):
    def __init__(self, width: int = 8):
        super().__init__(width)
        self.inner = Block(width)

    def forward(self, x):
        return self.inner(super().forward(x))
