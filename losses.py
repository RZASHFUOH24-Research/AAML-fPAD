"""
Angular loss module: learnable class centres (Spoof / Live) on a normalised
hypersphere, with cosine-similarity based classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularLoss(nn.Module):
    """
    Two learnable centres (index 0 = Spoof, index 1 = Live).
    Features and centres are L2-normalised, then compared via cosine similarity.
    The loss pulls each sample's embedding toward its ground-truth class centre.
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(2, feature_dim))  # [Spoof, Live]

    def forward(self, features, labels):
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(self.centers, p=2, dim=1)

        cosine = torch.matmul(features_norm, centers_norm.t())  # [B, 2]
        true_class_cos = cosine[torch.arange(len(labels)), labels]

        angular_loss = torch.mean(1 - true_class_cos)
        return angular_loss, cosine
