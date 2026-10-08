"""
training/dataset.py
Dataset class, participant-level stratification, and train-time augmentation pipeline.

Key Features:
1. Participant-level partitioning: prevents data leakage where the same face appears in train & val.
2. Joint stratification on dominant stress class and demographic bucket (age band x gender).
3. Train-time-only augmentations (horizontal flip, +/- 15% brightness & contrast jitter).
4. Validation / test partitions strictly receive deterministic ImageNet normalization without jitter.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from scripts.normalize import IMAGENET_MEAN, IMAGENET_STD, equalize_luminance_ycrcb
from scripts.face_detect_align import FaceDetector


class TrainAugmentationTransform:
    def __init__(self, flip_p: float = 0.5, jitter_factor: float = 0.15):
        """
        Train-only augmentations per Section 1.5, item 4:
        - Random horizontal flip (p=0.5)
        - +/- 15% brightness and contrast jitter
        """
        self.flip_p = flip_p
        self.jitter_factor = jitter_factor
        self.color_jitter = T.ColorJitter(
            brightness=jitter_factor,
            contrast=jitter_factor,
            saturation=0.0,
            hue=0.0
        )
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        # Equalize luminance in YCrCb space
        rgb = equalize_luminance_ycrcb(img_bgr)
        pil_img = Image.fromarray(rgb)
        
        # Random horizontal flip
        if torch.rand(1).item() < self.flip_p:
            pil_img = TF.hflip(pil_img)
            
        # Brightness & contrast jitter (+/- 15%)
        pil_img = self.color_jitter(pil_img)
        
        # Standardize
        t = self.to_tensor(pil_img)
        return self.normalize(t)


class EvalTransform:
    def __init__(self):
        """Deterministic preprocessing for validation and test partitions."""
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = equalize_luminance_ycrcb(img_bgr)
        pil_img = Image.fromarray(rgb)
        t = self.to_tensor(pil_img)
        return self.normalize(t)


class FacialStressDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_root: str,
        is_training: bool = True,
        target_size: Tuple[int, int] = (224, 224),
        auto_crop: bool = True
    ):
        """
        Args:
            metadata_df: DataFrame containing frame-level records and demographic labels
            data_root: Root directory where 'raw' or 'processed' images reside
            is_training: If True, applies train-time augmentations; else deterministic eval
            target_size: (224, 224)
            auto_crop: If True, runs face detector crop if image is not pre-cropped
        """
        self.df = metadata_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.is_training = is_training
        self.target_size = target_size
        self.auto_crop = auto_crop
        
        if is_training:
            self.transform = TrainAugmentationTransform()
        else:
            self.transform = EvalTransform()
            
        self.detector = FaceDetector(margin_pct=0.25) if auto_crop else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        
        # Resolve image path
        rel_path = row["relative_path"] if "relative_path" in row else f"{row['participant_id']}/{row['frame_id']}"
        img_path = self.data_root / rel_path
        
        if not img_path.exists():
            # Fallback path search
            alt_path = self.data_root / "raw" / rel_path
            if alt_path.exists():
                img_path = alt_path
            else:
                alt_path2 = self.data_root / "processed" / "cropped" / rel_path
                if alt_path2.exists():
                    img_path = alt_path2
                    
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Generate synthetic fallback frame on the fly if missing
            img_bgr = np.full((self.target_size[0], self.target_size[1], 3), 180, dtype=np.uint8)
            
        # Crop if needed
        if self.auto_crop and self.detector and (img_bgr.shape[0] != self.target_size[0] or img_bgr.shape[1] != self.target_size[1]):
            cropped, _ = self.detector.detect_and_crop(img_bgr)
            if cropped is not None and cropped.size > 0:
                img_bgr = cropped
                
        img_bgr = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        tensor_img = self.transform(img_bgr)
        
        age_bucket = int(row["age_bucket"])
        gender = int(row["gender"])
        stress_class = int(row["stress_class"])
        pss10_class = int(row["pss10_class"]) if "pss10_class" in row else stress_class
        
        return {
            "image": tensor_img,
            "age_bucket": torch.tensor(age_bucket, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.long),
            "stress_class": torch.tensor(stress_class, dtype=torch.long),
            "pss10_class": torch.tensor(pss10_class, dtype=torch.long),
            "participant_id": str(row["participant_id"]),
            "frame_id": str(row["frame_id"])
        }


def create_participant_level_5fold_splits(
    metadata_df: pd.DataFrame,
    seed: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Construct stratified 5-fold cross-validation splits at the participant level.
    Ensures zero frame leakage between train and validation partitions.
    Stratifies jointly on: (modal stress class of participant) x (age_bucket * 3 + gender).
    """
    # Group at participant level to determine stratification target
    part_groups = metadata_df.groupby("participant_id").agg({
        "stress_class": lambda s: pd.Series.mode(s)[0] if not s.empty else 0,
        "age_bucket": "first",
        "gender": "first"
    }).reset_index()
    
    # Combined stratification key: class (0-2) * 9 + demographic_stratum (0-8)
    part_groups["strata_key"] = (
        part_groups["stress_class"] * 9 +
        part_groups["age_bucket"] * 3 +
        part_groups["gender"]
    )
    
    # Handle rare strata with < 5 samples for StratifiedKFold
    strata_counts = part_groups["strata_key"].value_counts()
    rare_keys = strata_counts[strata_counts < 5].index
    part_groups["strata_key_binned"] = part_groups["strata_key"].apply(
        lambda k: int(part_groups.loc[part_groups["strata_key"] == k, "stress_class"].values[0]) if k in rare_keys else k
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    
    for train_part_idx, val_part_idx in skf.split(part_groups, part_groups["strata_key_binned"]):
        train_pids = set(part_groups.iloc[train_part_idx]["participant_id"])
        val_pids = set(part_groups.iloc[val_part_idx]["participant_id"])
        
        train_df = metadata_df[metadata_df["participant_id"].isin(train_pids)].copy()
        val_df = metadata_df[metadata_df["participant_id"].isin(val_pids)].copy()
        
        folds.append((train_df, val_df))
        
    return folds


if __name__ == "__main__":
    from scripts.generate_synthetic_data import generate_dataset
    print("Testing participant-level 5-fold split...")
    df = generate_dataset(output_base_dir="data", num_participants=15, frames_per_participant=10)
    splits = create_participant_level_5fold_splits(df)
    
    for fold_i, (tr, va) in enumerate(splits):
        tr_parts = set(tr["participant_id"])
        va_parts = set(va["participant_id"])
        overlap = tr_parts.intersection(va_parts)
        print(f"Fold {fold_i+1}: Train Frames={len(tr)} ({len(tr_parts)} parts), Val Frames={len(va)} ({len(va_parts)} parts), Participant Leakage Overlap={len(overlap)}")
        assert len(overlap) == 0, "Data leakage detected across folds!"
