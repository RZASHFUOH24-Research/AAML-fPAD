"""
Asymmetric Angular Margin Loss (AAML).

Two learnable class centres (index 0 = Spoof, index 1 = Live) live on the unit
hypersphere. Logits are alpha * cos(theta); a margin m is subtracted from the
cosine of the LIVE class for LIVE samples only. Spoof samples get no margin.
Softmax cross-entropy is applied to the resulting logits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricAngularMarginLoss(nn.Module):
    def __init__(self, feature_dim=128, scale=30.0, margin=0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin
        centers = torch.randn(2, feature_dim) / math.sqrt(feature_dim)  # std = 1/sqrt(d)
        self.centers = nn.Parameter(F.normalize(centers, dim=1))        # [Spoof, Live]

    def cosine(self, features):
        """Cosine similarity of each embedding to both class centres -> [B, 2]."""
        f = F.normalize(features, p=2, dim=1)
        c = F.normalize(self.centers, p=2, dim=1)
        return f @ c.t()

    def forward(self, features, labels):
        cosine = self.cosine(features)                              # [B, 2], no margin

        margin_term = torch.zeros_like(cosine)
        margin_term[:, 1] = self.margin * (labels == 1).to(cosine.dtype)

        logits = self.scale * (cosine - margin_term)
        loss = F.cross_entropy(logits, labels)
        return loss, cosine                                         # raw cosine for evaluation

    @torch.no_grad()
    def renormalize_centers(self):
        """Project the centres back onto the unit hypersphere (after each optimiser step)."""
        self.centers.copy_(F.normalize(self.centers, p=2, dim=1))
