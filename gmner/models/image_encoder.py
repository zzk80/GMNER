"""Vision encoder wrappers."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        output_dim: int = 768,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif self.backbone_name in {"vit_b_16", "vit-base-patch16-224"}:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            in_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f"Unsupported image backbone: {backbone_name}")

        self.projector = nn.Linear(in_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def freeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.flatten(start_dim=1)

        global_feature = self.norm(self.projector(features))

        # Default single-node image graph. Replace this with detector regions for full grounding.
        image_nodes = global_feature.unsqueeze(1)
        return global_feature, image_nodes
