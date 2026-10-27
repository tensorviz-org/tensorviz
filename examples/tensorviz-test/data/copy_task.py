"""Copy task: the target is the input sequence. Trivial, fast, and enough to see a
transformer learn attention on CPU in seconds."""
import torch
from torch.utils.data import DataLoader, Dataset


class CopyDataset(Dataset):
    def __init__(self, size: int = 2048, seq_len: int = 10, vocab_size: int = 16, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        # Token 0 is reserved as padding; sequences use 1..vocab_size-1.
        self.data = torch.randint(1, vocab_size, (size, seq_len), generator=g)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        x = self.data[i]
        return x, x.clone()


def make_loader(batch_size: int = 64, split: str = "train", **kw) -> DataLoader:
    seed = 0 if split == "train" else 1
    size = kw.pop("size", 2048 if split == "train" else 256)
    return DataLoader(CopyDataset(size=size, seed=seed, **kw), batch_size=batch_size, shuffle=split == "train")
