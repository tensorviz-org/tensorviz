"""Batching. Kept apart from the corpus so a sweep can vary it alone."""
from torch.utils.data import DataLoader, random_split

from data.char_dataset import CharCorpus


def make_loader(batch_size: int = 16, context: int = 64, shuffle: bool = True):
    corpus = CharCorpus(context=context)
    return DataLoader(corpus, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def make_splits(batch_size: int = 16, context: int = 64, holdout: float = 0.1):
    """Train and validation loaders over one corpus, split by index."""
    corpus = CharCorpus(context=context)
    validation = int(len(corpus) * holdout)
    train_part, valid_part = random_split(corpus, [len(corpus) - validation, validation])
    return (
        DataLoader(train_part, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(valid_part, batch_size=batch_size, drop_last=True),
    )
