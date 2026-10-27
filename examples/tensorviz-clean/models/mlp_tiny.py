import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MlpTiny(nn.Module):
    def __init__(self):
        super().__init__()
        # Model layers
        self.mlp_tiny__n0 = nn.Linear(in_features=16, out_features=32)
        self.mlp_tiny__n1 = nn.ReLU(inplace=True)
        self.mlp_tiny__n2 = nn.Linear(in_features=32, out_features=4)

    def forward(self, x):
        out_mlp_tiny__n0 = self.mlp_tiny__n0(x)
        out_mlp_tiny__n1 = self.mlp_tiny__n1(out_mlp_tiny__n0)
        out_mlp_tiny__n2 = self.mlp_tiny__n2(out_mlp_tiny__n1)
        return out_mlp_tiny__n2

# Example usage:
if __name__ == "__main__":
    model = MlpTiny()
    model.eval()  # inference mode: BatchNorm/Dropout use running stats
    x = torch.randn([1, 16])
    output = model(x)
    out_shape = tuple(output.shape) if torch.is_tensor(output) else type(output)
    print("Output shape:", out_shape)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")