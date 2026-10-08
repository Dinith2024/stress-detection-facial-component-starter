"""
training/loss.py
Loss functions and class-weighting utilities for class-imbalanced stress classification.

As per Section 1.2 of Milestone 2:
- Natural skew: ~45% low, 35% moderate, 20% high.
- Loss: Class-weighted cross-entropy loss with weights set to inverse class frequency,
  recomputed per fold from the training partition only.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Optional


def compute_inverse_class_weights(
    labels: Union[np.ndarray, List[int], torch.Tensor],
    num_classes: int = 3,
    smoothing: float = 0.0
) -> torch.Tensor:
    """
    Compute inverse class frequency weights from training labels:
    w_c = Total_Samples / (num_classes * count_c)
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
        
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    total_samples = len(labels_np)
    
    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    
    if smoothing > 0:
        # Optional soft balancing
        counts = counts ** (1.0 - smoothing)
        
    weights = total_samples / (num_classes * counts)
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    
    return torch.from_numpy(weights).float()


class ClassWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        """
        Args:
            weights: (num_classes,) tensor of inverse class frequency weights
            label_smoothing: optional label smoothing regularization
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


if __name__ == "__main__":
    # Test with simulated 45% / 35% / 20% distribution
    sim_labels = np.array([0]*450 + [1]*350 + [2]*200)
    weights = compute_inverse_class_weights(sim_labels, num_classes=3)
    print("Simulated Class Counts: [450 low, 350 moderate, 200 high]")
    print("Computed Inverse Class Weights:", weights.tolist())
    
    criterion = ClassWeightedCrossEntropyLoss(weights=weights)
    dummy_logits = torch.randn(6, 3)
    dummy_targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    loss = criterion(dummy_logits, dummy_targets)
    print("Sample Loss:", loss.item())
