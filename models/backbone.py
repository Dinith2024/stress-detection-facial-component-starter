"""
models/backbone.py
ResNet-18 feature extraction backbone for facial dynamics feature map extraction.

Outputs feature map F in R^(B x 512 x 7 x 7) for an input of (B x 3 x 224 x 224).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = False):
        """
        Initialize ResNet-18 backbone.
        Args:
            pretrained: Use ImageNet-pretrained weights (default True).
            freeze_early_layers: Freeze conv1 and layer1/layer2 to stabilize early fine-tuning.
        """
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Residual stages
        self.layer1 = resnet.layer1  # 64 channels,  56x56
        self.layer2 = resnet.layer2  # 128 channels, 28x28
        self.layer3 = resnet.layer3  # 256 channels, 14x14
        self.layer4 = resnet.layer4  # 512 channels, 7x7
        
        if freeze_early_layers:
            for param in [
                *self.conv1.parameters(),
                *self.bn1.parameters(),
                *self.layer1.parameters(),
                *self.layer2.parameters()
            ]:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)
        Returns:
            Feature map F of shape (B, 512, 7, 7)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    backbone = ResNet18Backbone(pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    feat = backbone(dummy_input)
    print("ResNet-18 Backbone Output Shape:", feat.shape)
    assert feat.shape == (4, 512, 7, 7), "Unexpected feature map shape."
