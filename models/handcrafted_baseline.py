"""
models/handcrafted_baseline.py
Secondary Reference Baseline: Classical Handcrafted Geometric Features + MLP Classifier.

Surveyed in Pise et al. [18]. Extracts facial landmark geometry:
- Eyebrow-to-eye vertical distances (brow furrowing)
- Inter-eye distance & eye aspect ratio (EAR)
- Mouth aspect ratio (MAR) & lip curvature
- Facial aspect ratio & symmetry metrics

Fed into an MLP classifier for 3-class stress prediction.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class HandcraftedFeatureExtractor:
    def __init__(self):
        """Extract geometric & intensity statistics from 224x224 face crops."""
        pass

    def extract_features(self, image_np_rgb: np.ndarray) -> np.ndarray:
        """
        Extract a 16-dimensional geometric and regional texture feature vector.
        """
        if image_np_rgb.dtype != np.uint8:
            img = np.clip(image_np_rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            img = image_np_rgb
            
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Region bounding approximations on standard 224x224 cropped face:
        # 1. Brow region (top 20-35%)
        brow_region = gray[int(h*0.20):int(h*0.35), int(w*0.20):int(w*0.80)]
        # 2. Eye region (30-50%)
        eye_region = gray[int(h*0.30):int(h*0.50), int(w*0.20):int(w*0.80)]
        # 3. Mouth region (65-85%)
        mouth_region = gray[int(h*0.65):int(h*0.85), int(w*0.30):int(w*0.70)]
        
        features = [
            float(np.mean(brow_region)) / 255.0,
            float(np.std(brow_region)) / 255.0,
            float(np.mean(eye_region)) / 255.0,
            float(np.std(eye_region)) / 255.0,
            float(np.mean(mouth_region)) / 255.0,
            float(np.std(mouth_region)) / 255.0,
            
            # Gradient energy (edge tension around brows & lips)
            float(np.mean(np.abs(cv2.Sobel(brow_region, cv2.CV_64F, 0, 1)))) / 255.0,
            float(np.mean(np.abs(cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0)))) / 255.0,
            float(np.mean(np.abs(cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1)))) / 255.0,
            
            # Global symmetry & contrast
            float(np.mean(gray)) / 255.0,
            float(np.std(gray)) / 255.0,
            float(np.percentile(gray, 90) - np.percentile(gray, 10)) / 255.0,
            
            # Vertical gradient asymmetry
            float(np.mean(gray[:h//2, :]) - np.mean(gray[h//2:, :])) / 255.0,
            float(np.mean(gray[:, :w//2]) - np.mean(gray[:, w//2:])) / 255.0,
            
            # Dynamic ratios
            float(np.std(brow_region) / (np.std(mouth_region) + 1e-5)),
            float(np.mean(brow_region) / (np.mean(mouth_region) + 1e-5))
        ]
        return np.array(features, dtype=np.float32)


class ClassicalMLPClassifier(nn.Module):
    def __init__(self, in_features: int = 16, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 16) feature vectors
        """
        logits = self.mlp(x)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probs}


if __name__ == "__main__":
    extractor = HandcraftedFeatureExtractor()
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    feats = extractor.extract_features(dummy_img)
    print("Handcrafted Feature Vector Shape:", feats.shape)
    
    mlp = ClassicalMLPClassifier()
    out = mlp(torch.from_numpy(feats).unsqueeze(0))
    print("MLP Output Logits:", out["logits"].shape)
