import torch
import torch.nn as nn
import torch.nn.functional as F


class FiveLayerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FiveLayerCNN, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # Conv Layer 4
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        # Conv Layer 5
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling removes fixed input-size dependency
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Layer 5
        x = F.relu(self.bn5(self.conv5(x)))

        # Global pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x


# Example usage
model = FiveLayerCNN(num_classes=10)

x = torch.randn(8, 3, 224, 224)  # batch_size=8
output = model(x)

print(output.shape)  # [8, 10]
