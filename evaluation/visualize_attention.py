"""
evaluation/visualize_attention.py
Spatial attention heatmap overlay and demographic comparison visualizer.

Visualizes where the network focuses (brows, eyes, mouth) and how FiLM demographic
conditioning modulates the spatial focus for different age buckets and genders.
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union

from scripts.normalize import denormalize_image_tensor


def generate_attention_heatmap(
    attention_map_7x7: Union[torch.Tensor, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Interpolate (7, 7) spatial attention weights to (224, 224) RGB heatmap.
    """
    if isinstance(attention_map_7x7, torch.Tensor):
        attn_np = attention_map_7x7.squeeze().detach().cpu().numpy()
    else:
        attn_np = np.asarray(attention_map_7x7).squeeze()
        
    # Min-max normalization for heatmap display
    attn_min = attn_np.min()
    attn_max = attn_np.max()
    if attn_max - attn_min > 1e-6:
        attn_norm = (attn_np - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn_np)
        
    attn_uint8 = (attn_norm * 255.0).astype(np.uint8)
    # Upsample to target size using bicubic interpolation
    resized_attn = cv2.resize(attn_uint8, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(resized_attn, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_attention_on_image(
    image_rgb_or_tensor: Union[np.ndarray, torch.Tensor],
    attention_map_7x7: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.55
) -> np.ndarray:
    """
    Overlay spatial attention heatmap onto original face image.
    Returns: RGB blended uint8 image.
    """
    if isinstance(image_rgb_or_tensor, torch.Tensor):
        img_rgb = denormalize_image_tensor(image_rgb_or_tensor)
    else:
        img_rgb = image_rgb_or_tensor.copy()
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
            
    h, w = img_rgb.shape[:2]
    heatmap_rgb = generate_attention_heatmap(attention_map_7x7, target_size=(w, h))
    
    # Blend: alpha * heatmap + (1 - alpha) * original
    blended = cv2.addWeighted(heatmap_rgb, alpha, img_rgb, 1.0 - alpha, 0)
    return blended


def save_attention_comparison_figure(
    image_rgb: np.ndarray,
    proposed_attn: np.ndarray,
    baseline_attn: np.ndarray,
    demographic_str: str,
    pred_class_str: str,
    save_path: str
):
    """
    Save 3-panel figure comparing:
    1. Input Face Crop
    2. Unconditioned Baseline Attention
    3. FiLM-Conditioned Proposed Attention
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Face Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    base_overlay = overlay_attention_on_image(image_rgb, baseline_attn)
    axes[1].imshow(base_overlay)
    axes[1].set_title("Unconditioned Baseline Attention", fontsize=11)
    axes[1].axis("off")
    
    prop_overlay = overlay_attention_on_image(image_rgb, proposed_attn)
    axes[2].imshow(prop_overlay)
    axes[2].set_title(f"FiLM-Conditioned ({demographic_str})\nPred: {pred_class_str}", fontsize=11, fontweight="bold", color="darkgreen")
    axes[2].axis("off")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dummy_img = np.full((224, 224, 3), 180, dtype=np.uint8)
    cv2.circle(dummy_img, (112, 112), 60, (220, 160, 140), -1)
    dummy_attn = np.zeros((7, 7), dtype=np.float32)
    dummy_attn[2, 3] = 1.0  # Brow region highlight
    dummy_attn[4, 3] = 0.5  # Mouth region highlight
    
    overlay = overlay_attention_on_image(dummy_img, dummy_attn)
    print("Overlay Generated Shape:", overlay.shape)
