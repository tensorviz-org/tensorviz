"""No nn.Module here either; M8/M9 trace it as a data pipeline.

`make_loader` is what `--kind data` traces (task.md M9). Every line of
`ToyDataset.__getitem__` that touches a tensor becomes a node, so the body is
deliberately one operation per line — a `__getitem__` written as a single
expression would be one node, and one node is not a pipeline.

Two samples of eight, like every other fixture here, and `num_workers=0` so the
`__getitem__` frames are in THIS process: a worker process runs them where no
`TorchDispatchMode` of ours is installed and the trace sees only the collate.
"""
import torch


def batches(count=2, width=8):
    for _ in range(count):
        yield torch.zeros(1, width)


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, count=4, width=8):
        self.count = count
        # Built here, outside anything a trace records: the stored samples are the
        # pipeline's INPUT, and a `__getitem__` that allocated them would be
        # measuring the fixture rather than the pipeline.
        self.samples = torch.ones(count, width)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # One tensor operation per line, and every line reads the previous one's
        # result: `--kind data` draws an edge per tensor identity, so an operation
        # with no traced input (a bare `torch.ones(...)`) is a node with no edge.
        raw = self.samples[index]
        scaled = raw * 0.5
        shifted = scaled + float(index)
        return shifted.view(2, self.width_of())

    def width_of(self):
        return self.samples.shape[1] // 2


def make_loader(batch_size=2):
    return torch.utils.data.DataLoader(ToyDataset(), batch_size=batch_size)
