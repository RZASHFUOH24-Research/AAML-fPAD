"""
Backbone model: EnhancedResNet feature extractor.
Maps an input face image to a d-dimensional (un-normalised) embedding:
    phi = phi_F o phi_I
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class EnhancedResNet(nn.Module):
    """
    phi_I : ResNet-18 backbone (ImageNet-1K pre-trained), 512-d output.
    phi_F : projection head 512 -> 256 -> feature_dim
            (FC, BatchNorm, ReLU, Dropout(0.5), FC).
    """

    def __init__(self, feature_dim=128, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # drop the final FC layer, keep conv layers + global average pooling
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.flatten(1)          # [B, 512]
        return self.head(h)       # [B, feature_dim]
