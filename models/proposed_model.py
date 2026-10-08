"""
models/proposed_model.py
Proposed Model: Facial-Dynamics Stress Classifier with Age/Gender-Conditioned Hierarchical Attention.

Architecture:
1. Input: Facial image (B, 3, 224, 224) + Demographic priors (Age bucket, Gender)
2. Backbone: ResNet-18 feature extractor -> Feature map F in R^(B x 512 x 7 x 7)
3. Demographic Conditioning: FiLM generator -> scale (gamma) & shift (beta)
4. Attention: Hierarchical Spatial Attention modulated by (gamma, beta)
5. Regularization: Attention-dropout during training
6. Pooling: Attention-weighted Global Average Pooling -> Feature vector z in R^(B x 512)
7. Classification Head: Linear(512, 3) -> 3-class logits [low, moderate, high]
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from models.backbone import ResNet18Backbone
from models.film_module import FiLMGenerator
from models.spatial_attention import SpatialAttentionModule


class ProposedStressModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained_backbone: bool = True,
        freeze_early_backbone: bool = False,
        attention_dropout_p: float = 0.20,
        film_embed_dim: int = 16,
        film_hidden_dim: int = 32,
        head_dropout_p: float = 0.30
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. CNN Backbone
        self.backbone = ResNet18Backbone(
            pretrained=pretrained_backbone,
            freeze_early_layers=freeze_early_backbone
        )
        
        # 2. Demographic FiLM Conditioning Module
        self.film_generator = FiLMGenerator(
            num_age_buckets=3,
            num_genders=3,
            embed_dim=film_embed_dim,
            hidden_dim=film_hidden_dim,
            output_dim=1
        )
        
        # 3. Spatial Attention with Attention Dropout
        self.spatial_attention = SpatialAttentionModule(
            in_channels=512,
            reduction_channels=128,
            attention_dropout_p=attention_dropout_p
        )
        
        # 4. Classification Head
        self.head_dropout = nn.Dropout(p=head_dropout_p)
        self.classifier = nn.Linear(512, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        age_bucket: torch.Tensor,
        gender: torch.Tensor,
        return_attention_map: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, 224, 224)
            age_bucket: (B,) LongTensor [0, 1, 2]
            gender: (B,) LongTensor [0, 1, 2]
            return_attention_map: bool
        Returns:
            Dict containing:
                "logits": (B, 3) raw class logits
                "probabilities": (B, 3) softmax probabilities
                "attention_map": (B, 1, 7, 7) spatial attention weights
                "film_gamma": (B, 1, 1, 1) scale factor
                "film_beta": (B, 1, 1, 1) shift factor
        """
        # Extract visual feature map: (B, 512, 7, 7)
        feat_map = self.backbone(images)
        
        # Compute FiLM modulation parameters from demographic inputs
        gamma, beta = self.film_generator(age_bucket, gender)
        
        # Hierarchical spatial attention with FiLM modulation & attention-dropout
        pooled_feat, attn_map = self.spatial_attention(feat_map, gamma=gamma, beta=beta)
        
        # Classification
        dropped_feat = self.head_dropout(pooled_feat)
        logits = self.classifier(dropped_feat)
        probs = torch.softmax(logits, dim=-1)
        
        output = {
            "logits": logits,
            "probabilities": probs,
            "attention_map": attn_map,
            "film_gamma": gamma,
            "film_beta": beta,
            "pooled_features": pooled_feat
        }
        return output


if __name__ == "__main__":
    model = ProposedStressModel(pretrained_backbone=False)
    dummy_imgs = torch.randn(2, 3, 224, 224)
    dummy_age = torch.tensor([0, 1], dtype=torch.long)
    dummy_gen = torch.tensor([1, 0], dtype=torch.long)
    
    out = model(dummy_imgs, dummy_age, dummy_gen)
    print("=== Proposed Model Forward Pass Verification ===")
    print("Logits Shape:        ", out["logits"].shape)
    print("Probabilities Shape: ", out["probabilities"].shape)
    print("Attention Map Shape: ", out["attention_map"].shape)
    print("FiLM Gamma Shape:    ", out["film_gamma"].shape)
    print("Total Parameters:    ", sum(p.numel() for p in model.parameters() if p.requires_grad))
