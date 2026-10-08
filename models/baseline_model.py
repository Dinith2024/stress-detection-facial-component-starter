"""
models/baseline_model.py
Baseline Model: Unconditioned Hierarchical-Attention CNN.

As defined in Section 3 of Milestone 2:
- Same ResNet-18 backbone
- Same spatial attention module (with attention_dropout_p = 0.0)
- NO FiLM demographic conditioning (gamma=None, beta=None)
- Same classification head Linear(512, 3)

This provides a strict, non-strawman comparison baseline isolating the exact contribution
of demographic conditioning and attention-dropout regularization.
"""

import torch
import torch.nn as nn
from typing import Dict

from models.backbone import ResNet18Backbone
from models.spatial_attention import SpatialAttentionModule


class BaselineAttentionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained_backbone: bool = True,
        freeze_early_backbone: bool = False,
        head_dropout_p: float = 0.30
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Same CNN Backbone
        self.backbone = ResNet18Backbone(
            pretrained=pretrained_backbone,
            freeze_early_layers=freeze_early_backbone
        )
        
        # 2. Same Spatial Attention (NO attention-dropout)
        self.spatial_attention = SpatialAttentionModule(
            in_channels=512,
            reduction_channels=128,
            attention_dropout_p=0.0  # Switched off for baseline
        )
        
        # 3. Same Classification Head
        self.head_dropout = nn.Dropout(p=head_dropout_p)
        self.classifier = nn.Linear(512, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        age_bucket: torch.Tensor = None,  # Ignored in baseline
        gender: torch.Tensor = None,       # Ignored in baseline
        return_attention_map: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass without demographic conditioning.
        """
        feat_map = self.backbone(images)
        
        # Unconditioned spatial attention (gamma=None, beta=None)
        pooled_feat, attn_map = self.spatial_attention(feat_map, gamma=None, beta=None)
        
        dropped_feat = self.head_dropout(pooled_feat)
        logits = self.classifier(dropped_feat)
        probs = torch.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "attention_map": attn_map,
            "pooled_features": pooled_feat
        }


if __name__ == "__main__":
    model = BaselineAttentionModel(pretrained_backbone=False)
    dummy_imgs = torch.randn(2, 3, 224, 224)
    out = model(dummy_imgs)
    print("=== Baseline Model Forward Pass Verification ===")
    print("Logits Shape:        ", out["logits"].shape)
    print("Probabilities Shape: ", out["probabilities"].shape)
    print("Attention Map Shape: ", out["attention_map"].shape)
    print("Total Parameters:    ", sum(p.numel() for p in model.parameters() if p.requires_grad))
