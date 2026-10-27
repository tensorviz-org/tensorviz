"""Generate tiny local tensor fixtures; uses existing Torch and installs nothing."""
from pathlib import Path

import torch
from demo_models import TinyNet


def main():
    destination = Path(__file__).resolve().parent / 'fixtures'
    destination.mkdir(exist_ok=True)
    model = TinyNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    torch.save(model.state_dict(), destination / 'baseline.pt')
    with torch.no_grad():
        model.head.weight.add_(0.5)
    torch.save(model.state_dict(), destination / 'changed.pt')
    torch.save({'kwargs': {'x': torch.tensor([[1., 2., 3.], [3., 2., 1.]])},
                'batch': {'targets': torch.ones(2, 2)}}, destination / 'batch.pt')
    print(f'Created batch.pt, baseline.pt and changed.pt in {destination}')


if __name__ == '__main__':
    main()
