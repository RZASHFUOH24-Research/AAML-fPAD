"""
Backbone model: EnhancedResNet50 feature extractor.
Maps input face images to a d-dimensional embedding (see Eq. 2 in the paper).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class EnhancedResNet50(nn.Module):
    """
    ResNet-50 backbone (ImageNet-pretrained) + projection head.
    Projection head: 2048 -> 512 -> feature_dim, with BatchNorm, ReLU, Dropout.
    """

    def __init__(self, feature_dim=128, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.resnet = nn.Sequential(*list(backbone.children())[:-1])

        self.fc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        x = self.fc1(features)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
