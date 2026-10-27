import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MlpReluStack(nn.Module):
    def __init__(self):
        super().__init__()
        # Model layers
        self.mlp_relu_stack__n0 = nn.Linear(in_features=32, out_features=64)
        self.mlp_relu_stack__n1 = nn.ReLU(inplace=True)
        self.mlp_relu_stack__n2 = nn.Linear(in_features=64, out_features=64)
        self.mlp_relu_stack__n3 = nn.ReLU(inplace=True)
        self.mlp_relu_stack__n4 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        out_mlp_relu_stack__n0 = self.mlp_relu_stack__n0(x)
        out_mlp_relu_stack__n1 = self.mlp_relu_stack__n1(out_mlp_relu_stack__n0)
        out_mlp_relu_stack__n2 = self.mlp_relu_stack__n2(out_mlp_relu_stack__n1)
        out_mlp_relu_stack__n3 = self.mlp_relu_stack__n3(out_mlp_relu_stack__n2)
        out_mlp_relu_stack__n4 = self.mlp_relu_stack__n4(out_mlp_relu_stack__n3)
        return out_mlp_relu_stack__n4

# Example usage:
if __name__ == "__main__":
    model = MlpReluStack()
    model.eval()  # inference mode: BatchNorm/Dropout use running stats
    x = torch.randn([1, 32])
    output = model(x)
    out_shape = tuple(output.shape) if torch.is_tensor(output) else type(output)
    print("Output shape:", out_shape)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")