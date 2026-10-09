import torch
import torchvision.models as models
import torch.nn as nn


class ResNet18_MultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(pretrained=True)
        # Modify first conv layer for 1-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize weights: use mean of pretrained weights
        with torch.no_grad():
            self.conv1.weight[:] = net.conv1.weight.mean(dim=1, keepdim=True)
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3
