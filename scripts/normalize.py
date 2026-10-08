"""
scripts/normalize.py
Image resizing, photometric normalisation (YCrCb luminance histogram equalisation),
and ImageNet statistical standardization.

Pipeline:
1. Resize cropped face to 224x224 pixels.
2. Convert BGR -> YCrCb color space.
3. Apply CLAHE or Global Histogram Equalisation to Y (luminance) channel.
4. Convert back to RGB format.
5. Standardize using ImageNet statistics:
   mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
"""

import os
import cv2
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Union, Optional

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def equalize_luminance_ycrcb(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance (Y)
    channel in YCrCb color space to mitigate varying illumination conditions.
    Returns RGB image.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)
    
    # Adaptive histogram equalization on Y channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    y_eq = clahe.apply(y_chan)
    
    ycrcb_eq = cv2.merge((y_eq, cr_chan, cb_chan))
    rgb_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    return rgb_eq


def preprocess_and_normalize_frame(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_hist_eq: bool = True,
    to_torch_tensor: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Complete single-frame preprocessing pipeline:
    1. Resize to target_size (224, 224).
    2. Luminance equalisation in YCrCb space.
    3. Scale to [0, 1].
    4. ImageNet mean & std normalisation: (x - mean) / std.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image input for normalization.")
        
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    
    if apply_hist_eq:
        rgb = equalize_luminance_ycrcb(resized)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
    img_float = rgb.astype(np.float32) / 255.0
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    
    if to_torch_tensor:
        # Convert HWC -> CHW tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
        return tensor
        
    return normalized


def denormalize_image_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.
    Accepts CHW or BCHW tensor, returns HWC uint8 RGB numpy array.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
        
    np_img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    unnormalized = (np_img * IMAGENET_STD) + IMAGENET_MEAN
    unnormalized = np.clip(unnormalized * 255.0, 0, 255).astype(np.uint8)
    return unnormalized


def batch_normalize_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (224, 224)
) -> int:
    """
    Batch preprocess images from input_dir and save normalized PyTorch tensors (.pt).
    """
    in_p = Path(input_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    
    image_files = [f for f in in_p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    count = 0
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        tensor = preprocess_and_normalize_frame(img, target_size=target_size, to_torch_tensor=True)
        rel_path = img_path.relative_to(in_p).with_suffix(".pt")
        save_path = out_p / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, str(save_path))
        count += 1
        
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image resizing and photometric normalization")
    parser.add_argument("--test-synthetic", action="store_true", help="Test normalization on synthetic image")
    args = parser.parse_args()
    
    # Test normalization
    dummy_bgr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    out_tensor = preprocess_and_normalize_frame(dummy_bgr)
    print("=== Normalization Pipeline Test ===")
    print(f"Input Shape: (300, 300, 3) BGR")
    print(f"Output Tensor Shape: {out_tensor.shape} (CHW, float32)")
    print(f"Tensor Min: {out_tensor.min().item():.3f}, Max: {out_tensor.max().item():.3f}")
    print(f"ImageNet mean applied: {IMAGENET_MEAN}")
    print(f"ImageNet std applied:  {IMAGENET_STD}")
