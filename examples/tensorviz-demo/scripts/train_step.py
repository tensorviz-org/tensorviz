"""No nn.Module lives here on purpose: /api/ide/workspace/classes must return
exactly the four in models/. M9 turns this into a `--kind train_step` trace.

It is also a runnable script, which is what M8's debugger test needs: a
breakpoint set from a node lands in `models/resid.py`, and something has to call
that forward for the debugger to stop in it. Two samples of eight — the same
batch every fixture here uses, and the size the runtime capture is written for.
"""
import torch


def train_step(model, batch, optimizer):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(batch), batch)
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # The repository root, so `models.resid` imports the way the IDE's own trace
    # worker imports it (services/ingest/service.py::package_root).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.resid import Net

    net = Net()
    print(float(train_step(net, torch.zeros(2, 8), torch.optim.SGD(net.parameters(), lr=0.1))))
