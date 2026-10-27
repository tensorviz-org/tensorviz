import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MlpSoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Model layers
        self.mlp_softmax_classifier__n0 = nn.Linear(in_features=20, out_features=40)
        self.mlp_softmax_classifier__n1 = nn.ReLU(inplace=True)
        self.mlp_softmax_classifier__n2 = nn.Linear(in_features=40, out_features=5)
        self.mlp_softmax_classifier__n3 = nn.Softmax(dim=-1)

    def forward(self, x):
        out_mlp_softmax_classifier__n0 = self.mlp_softmax_classifier__n0(x)
        out_mlp_softmax_classifier__n1 = self.mlp_softmax_classifier__n1(out_mlp_softmax_classifier__n0)
        out_mlp_softmax_classifier__n2 = self.mlp_softmax_classifier__n2(out_mlp_softmax_classifier__n1)
        out_mlp_softmax_classifier__n3 = self.mlp_softmax_classifier__n3(out_mlp_softmax_classifier__n2)
        return out_mlp_softmax_classifier__n3

# Example usage:
if __name__ == "__main__":
    model = MlpSoftmaxClassifier()
    model.eval()  # inference mode: BatchNorm/Dropout use running stats
    x = torch.randn([1, 20])
    output = model(x)
    out_shape = tuple(output.shape) if torch.is_tensor(output) else type(output)
    print("Output shape:", out_shape)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")