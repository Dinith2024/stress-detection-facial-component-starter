"""
tests/test_models.py
Unit tests for Proposed Model, Baseline Model, FiLM, Spatial Attention, and Handcrafted Baseline.
"""

import unittest
import torch
import numpy as np

from models.backbone import ResNet18Backbone
from models.film_module import FiLMGenerator
from models.spatial_attention import SpatialAttentionModule
from models.proposed_model import ProposedStressModel
from models.baseline_model import BaselineAttentionModel
from models.handcrafted_baseline import ClassicalMLPClassifier, HandcraftedFeatureExtractor


class TestModels(unittest.TestCase):
    def test_backbone_shape(self):
        backbone = ResNet18Backbone(pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        feat = backbone(dummy)
        self.assertEqual(feat.shape, (2, 512, 7, 7))

    def test_film_generator(self):
        film = FiLMGenerator(embed_dim=16, hidden_dim=32)
        age = torch.tensor([0, 1, 2], dtype=torch.long)
        gen = torch.tensor([1, 0, 2], dtype=torch.long)
        gamma, beta = film(age, gen)
        
        self.assertEqual(gamma.shape, (3, 1, 1, 1))
        self.assertEqual(beta.shape, (3, 1, 1, 1))

    def test_spatial_attention(self):
        attn = SpatialAttentionModule(in_channels=512, attention_dropout_p=0.2)
        feat = torch.randn(2, 512, 7, 7)
        gamma = torch.ones(2, 1, 1, 1)
        beta = torch.zeros(2, 1, 1, 1)
        
        # Test eval mode
        attn.eval()
        pooled, weights = attn(feat, gamma, beta)
        self.assertEqual(pooled.shape, (2, 512))
        self.assertEqual(weights.shape, (2, 1, 7, 7))
        
        # Softmax sum check
        weight_sums = weights.sum(dim=(1, 2, 3))
        for s in weight_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_proposed_model_forward(self):
        model = ProposedStressModel(pretrained_backbone=False)
        imgs = torch.randn(4, 3, 224, 224)
        age = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        gen = torch.tensor([1, 0, 1, 2], dtype=torch.long)
        
        out = model(imgs, age, gen)
        self.assertEqual(out["logits"].shape, (4, 3))
        self.assertEqual(out["probabilities"].shape, (4, 3))
        self.assertEqual(out["attention_map"].shape, (4, 1, 7, 7))

    def test_baseline_model_forward(self):
        model = BaselineAttentionModel(pretrained_backbone=False)
        imgs = torch.randn(3, 3, 224, 224)
        out = model(imgs)
        self.assertEqual(out["logits"].shape, (3, 3))
        self.assertEqual(out["probabilities"].shape, (3, 3))
        self.assertEqual(out["attention_map"].shape, (3, 1, 7, 7))

    def test_handcrafted_baseline(self):
        extractor = HandcraftedFeatureExtractor()
        dummy_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        feats = extractor.extract_features(dummy_rgb)
        self.assertEqual(feats.shape, (16,))
        
        mlp = ClassicalMLPClassifier(in_features=16)
        out = mlp(torch.from_numpy(feats).unsqueeze(0))
        self.assertEqual(out["logits"].shape, (1, 3))
        self.assertEqual(out["probabilities"].shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
