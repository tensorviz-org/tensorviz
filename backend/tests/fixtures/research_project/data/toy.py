from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index


def make_loader(batch_size: int = 2):
    return DataLoader(ToyDataset(), batch_size=batch_size)
