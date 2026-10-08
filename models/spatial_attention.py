"""
models/spatial_attention.py
Hierarchical spatial attention module with optional FiLM modulation and attention-dropout regularization.

Computes a learned 2D attention map over the feature map regions (7x7),
enabling the network to dynamically focus on informative facial regions (brows, eyes, mouth).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialAttentionModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        reduction_channels: int = 128,
        attention_dropout_p: float = 0.20
    ):
        """
        Args:
            in_channels: Channel depth of backbone feature map (512 for ResNet-18)
            reduction_channels: Channel dimension in intermediate projection
            attention_dropout_p: Dropout probability for attention-dropout regularization
        """
        super().__init__()
        self.in_channels = in_channels
        self.attention_dropout_p = attention_dropout_p
        
        # 2-layer convolutional attention scoring head
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, 1, kernel_size=1, bias=True)
        )

    def forward(
        self,
        features: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, C, H, W) e.g., (B, 512, 7, 7)
            gamma: Optional (B, 1, 1, 1) scale factor from FiLM
            beta:  Optional (B, 1, 1, 1) shift factor from FiLM
        Returns:
            pooled_features: (B, C) attention-weighted global representation
            attention_weights: (B, 1, H, W) normalized spatial attention map
        """
        b, c, h, w = features.shape
        
        # Compute raw spatial attention logits: (B, 1, H, W)
        attn_logits = self.attention_net(features)
        
        # Apply FiLM demographic modulation if provided
        if gamma is not None and beta is not None:
            attn_logits = gamma * attn_logits + beta
            
        # Reshape to (B, H*W) for spatial softmax
        flat_logits = attn_logits.view(b, -1)
        
        # Attention-dropout regularization during training (Liu et al. [15])
        if self.training and self.attention_dropout_p > 0:
            # Generate dropout mask for spatial locations
            keep_p = 1.0 - self.attention_dropout_p
            drop_mask = (torch.rand_like(flat_logits) < keep_p)
            
            # If all regions in a batch item were accidentally masked out, keep all
            all_masked = (drop_mask.sum(dim=-1, keepdim=True) == 0)
            drop_mask = torch.where(all_masked, torch.ones_like(drop_mask, dtype=torch.bool), drop_mask)
            
            # Mask out dropped positions before softmax with large negative value
            flat_logits = flat_logits.masked_fill(~drop_mask, -1e9)
            
        # Softmax normalisation over spatial locations (H*W)
        attn_weights_flat = F.softmax(flat_logits, dim=-1)  # (B, H*W)
        attn_weights = attn_weights_flat.view(b, 1, h, w)   # (B, 1, H, W)
        
        # Weighted Global Pooling: sum over spatial dimensions (H, W)
        # features: (B, C, H, W), attn_weights: (B, 1, H, W)
        weighted_features = features * attn_weights
        pooled_features = weighted_features.sum(dim=(2, 3))  # (B, C)
        
        return pooled_features, attn_weights


if __name__ == "__main__":
    attn = SpatialAttentionModule(in_channels=512, attention_dropout_p=0.2)
    feat = torch.randn(4, 512, 7, 7)
    gamma = torch.ones(4, 1, 1, 1) * 1.1
    beta = torch.zeros(4, 1, 1, 1) + 0.1
    
    attn.train()
    pooled, weights = attn(feat, gamma, beta)
    print("Training Mode:")
    print("Pooled Feature Shape:", pooled.shape)
    print("Attention Weights Shape:", weights.shape)
    print("Attention Weight Sum per Sample:", weights.sum(dim=(1, 2, 3)).tolist())
    
    attn.eval()
    pooled_eval, weights_eval = attn(feat, gamma, beta)
    print("\nEvaluation Mode:")
    print("Attention Weight Sum per Sample:", weights_eval.sum(dim=(1, 2, 3)).tolist())
