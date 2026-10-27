"""Character-level corpus handling.

The corpus is embedded rather than downloaded so the repository trains on a
laptop with no network and no dataset cache.
"""
import torch
from torch.utils.data import Dataset

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
) * 64


def build_vocabulary(text: str = CORPUS):
    """Stable character vocabulary: sorted, so two runs agree on the ids."""
    symbols = sorted(set(text))
    return {symbol: index for index, symbol in enumerate(symbols)}


def encode(text: str, vocabulary):
    return torch.tensor([vocabulary[ch] for ch in text if ch in vocabulary], dtype=torch.long)


class CharCorpus(Dataset):
    """Fixed-length windows over one long encoded string."""

    def __init__(self, text: str = CORPUS, context: int = 64):
        self.vocabulary = build_vocabulary(text)
        self.encoded = encode(text, self.vocabulary)
        self.context = context

    def __len__(self):
        return max(0, len(self.encoded) - self.context - 1)

    def __getitem__(self, index):
        window = self.encoded[index : index + self.context]
        target = self.encoded[index + 1 : index + self.context + 1]
        return window, target
