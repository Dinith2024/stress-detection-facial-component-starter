"""
models/film_module.py
Feature-wise Linear Modulation (FiLM) generator for demographic conditioning.

Takes demographic priors:
- Age bucket: {0: 18-29, 1: 30-44, 2: 45+}
- Gender: {0: female, 1: male, 2: other}

Produces:
- gamma (scale factor)
- beta (shift factor)
Applied to attention logits: S_mod = gamma * S + beta
"""

import torch
import torch.nn as nn
from typing import Tuple


class FiLMGenerator(nn.Module):
    def __init__(
        self,
        num_age_buckets: int = 3,
        num_genders: int = 3,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 1
    ):
        """
        Args:
            num_age_buckets: 3 classes (18-29, 30-44, 45+)
            num_genders: 3 classes (female, male, other)
            embed_dim: Embedding dimension per demographic attribute
            hidden_dim: MLP hidden layer dimension
            output_dim: Output dimension (1 for spatial-scalar attention modulation)
        """
        super().__init__()
        self.age_embed = nn.Embedding(num_age_buckets, embed_dim)
        self.gender_embed = nn.Embedding(num_genders, embed_dim)
        
        input_dim = embed_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Gamma (scale) head initialized near 1.0 (gain = 0.01 + 1.0)
        self.gamma_head = nn.Linear(hidden_dim, output_dim)
        # Beta (shift) head initialized near 0.0
        self.beta_head = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()

    def _init_weights(self):
        # Identity initialization for FiLM: gamma = 1.0, beta = 0.0
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, age_bucket: torch.Tensor, gender: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            age_bucket: (B,) LongTensor of age bucket indices (0, 1, 2)
            gender: (B,) LongTensor of gender indices (0, 1, 2)
        Returns:
            gamma: (B, output_dim, 1, 1) scale tensor
            beta:  (B, output_dim, 1, 1) shift tensor
        """
        e_age = self.age_embed(age_bucket)
        e_gen = self.gender_embed(gender)
        
        demographics = torch.cat([e_age, e_gen], dim=-1)  # (B, 2 * embed_dim)
        features = self.mlp(demographics)
        
        gamma = self.gamma_head(features)  # (B, output_dim)
        beta = self.beta_head(features)    # (B, output_dim)
        
        # Reshape to (B, output_dim, 1, 1) for broadcasting over spatial (H, W)
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1)
        
        return gamma, beta


if __name__ == "__main__":
    film = FiLMGenerator()
    age = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    gen = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    gamma, beta = film(age, gen)
    print("FiLM Gamma Shape:", gamma.shape, "Initial Values:\n", gamma.squeeze())
    print("FiLM Beta Shape: ", beta.shape, "Initial Values:\n", beta.squeeze())
