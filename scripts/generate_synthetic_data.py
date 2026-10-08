"""
scripts/generate_synthetic_data.py
Generates a realistic synthetic / pilot dataset simulating the 60-participant IT-worker cohort.

Features:
1. Simulates 60 pseudonymised participants (target cohort size in Milestone 2).
2. Demographic priors: Age buckets (18-29, 30-44, 45+) and Gender distribution.
3. 3-phase experimental protocol:
   - Phase 1: 10-min calm baseline (frames 1-10) -> mostly low stress (Likert 1-3)
   - Phase 2: 25-30 min workload & interruption stressor (frames 11-40) -> moderate (4-6) & high (7-10) stress
   - Phase 3: 10-min cooldown (frames 41-50) -> recovery to low/moderate
4. Generates facial frame images with distinct stress-correlated cues (eyebrow brow tension, mouth compression)
   and varied lighting/skin tones.
5. Produces session-level PSS-10 questionnaire scores and frame-level labels CSV.
"""

import os
import cv2
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.label_mapping import (
    map_likert_to_3class,
    map_age_to_bucket,
    map_pss10_to_class,
    STRESS_CLASSES,
    AGE_BUCKETS,
    GENDER_MAP
)


def generate_face_frame(
    stress_class: int,
    skin_tone_rgb: Tuple[int, int, int],
    lighting_factor: float = 1.0,
    frame_size: Tuple[int, int] = (480, 640),
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a facial frame image with visual cues correlated with stress_class (0: low, 1: mod, 2: high).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h, w = frame_size
    img = np.full((h, w, 3), int(210 * lighting_factor), dtype=np.uint8)
    
    # Add subtle office background details (monitor border, wall gradient)
    for row in range(h):
        grad = int(20 * (row / h))
        img[row, :] = np.clip(img[row, :] - grad, 0, 255)
        
    center_x = w // 2 + rng.integers(-15, 16)
    center_y = h // 2 + rng.integers(-10, 11)
    
    # Skin tone BGR
    r_c, g_c, b_c = skin_tone_rgb
    bgr_skin = (
        int(b_c * lighting_factor),
        int(g_c * lighting_factor),
        int(r_c * lighting_factor)
    )
    
    # Face ellipse
    face_rx = 90 + rng.integers(-5, 6)
    face_ry = 120 + rng.integers(-5, 6)
    cv2.ellipse(img, (center_x, center_y), (face_rx, face_ry), 0, 0, 360, bgr_skin, -1)
    
    # Eyebrows (stress dynamics: furrowed / lowered in high stress)
    brow_y_offset = 0 if stress_class == 0 else (-4 if stress_class == 1 else -8)
    brow_angle = 0 if stress_class == 0 else (6 if stress_class == 1 else 14)
    
    left_brow_start = (center_x - 65, center_y - 45 + brow_y_offset)
    left_brow_end = (center_x - 20, center_y - 40 + brow_y_offset + brow_angle)
    right_brow_start = (center_x + 20, center_y - 40 + brow_y_offset + brow_angle)
    right_brow_end = (center_x + 65, center_y - 45 + brow_y_offset)
    
    cv2.line(img, left_brow_start, left_brow_end, (30, 20, 10), 4)
    cv2.line(img, right_brow_start, right_brow_end, (30, 20, 10), 4)
    
    # Eyes (stress dynamics: squinted / narrowed in high stress)
    eye_h = 14 if stress_class == 0 else (10 if stress_class == 1 else 6)
    cv2.ellipse(img, (center_x - 45, center_y - 25), (16, eye_h), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (center_x + 45, center_y - 25), (16, eye_h), 0, 0, 360, (255, 255, 255), -1)
    
    # Pupils
    cv2.circle(img, (center_x - 45, center_y - 25), 6, (40, 25, 10), -1)
    cv2.circle(img, (center_x + 45, center_y - 25), 6, (40, 25, 10), -1)
    
    # Nose
    cv2.line(img, (center_x, center_y - 15), (center_x - 5, center_y + 20), (int(bgr_skin[0]*0.8), int(bgr_skin[1]*0.8), int(bgr_skin[2]*0.8)), 3)
    cv2.line(img, (center_x - 5, center_y + 20), (center_x + 6, center_y + 20), (int(bgr_skin[0]*0.8), int(bgr_skin[1]*0.8), int(bgr_skin[2]*0.8)), 3)
    
    # Mouth (stress dynamics: neutral/smile in low stress, straight/compressed in mod/high)
    if stress_class == 0:
        cv2.ellipse(img, (center_x, center_y + 55), (28, 12), 0, 0, 180, (40, 30, 160), 3)
    elif stress_class == 1:
        cv2.line(img, (center_x - 25, center_y + 58), (center_x + 25, center_y + 58), (40, 30, 160), 3)
    else:
        # Tense slightly downward lip corners
        cv2.line(img, (center_x - 26, center_y + 62), (center_x, center_y + 56), (35, 25, 150), 3)
        cv2.line(img, (center_x, center_y + 56), (center_x + 26, center_y + 62), (35, 25, 150), 3)
        
    # Add subtle camera noise
    noise = rng.normal(0, 4, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def generate_dataset(
    output_base_dir: str = "data",
    num_participants: int = 60,
    frames_per_participant: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic raw frames, labels CSV, and PSS-10 metadata for 60 participants.
    """
    rng = np.random.default_rng(seed)
    base_path = Path(output_base_dir)
    raw_dir = base_path / "raw"
    labels_dir = base_path / "labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Demographic distribution
    # Age bands: 0: 18-29 (50%), 1: 30-44 (35%), 2: 45+ (15%)
    age_choices = [rng.integers(21, 30) if i < 30 else (rng.integers(30, 45) if i < 51 else rng.integers(45, 60)) for i in range(num_participants)]
    gender_choices = [0 if i % 3 == 0 else (1 if i % 3 == 1 else 2 if i == 59 else 1) for i in range(num_participants)]
    
    # Representative South Asian skin tone variations (RGB)
    skin_tones = [
        (225, 185, 150),
        (205, 160, 125),
        (185, 140, 105),
        (165, 120, 85),
        (145, 100, 70)
    ]
    
    records = []
    
    for p_idx in range(num_participants):
        raw_pid = f"participant_real_name_{p_idx + 1:03d}"
        # Pseudonymised hash per Section 1.3 / Sri Lanka PDPA
        p_hash = hashlib.sha256(raw_pid.encode("utf-8")).hexdigest()[:12]
        pseudonym_id = f"SUBJ_{p_hash}"
        
        age = int(age_choices[p_idx])
        age_bucket = map_age_to_bucket(age)
        gender = int(gender_choices[p_idx])
        skin_tone = skin_tones[p_idx % len(skin_tones)]
        
        # PSS-10 baseline score (0-40)
        # Moderate baseline bias with normal spread
        pss_score = int(np.clip(rng.normal(loc=19.5, scale=6.0), 4, 38))
        pss_class = map_pss10_to_class(pss_score)
        
        p_dir = raw_dir / pseudonym_id
        p_dir.mkdir(parents=True, exist_ok=True)
        
        for f_idx in range(1, frames_per_participant + 1):
            frame_id = f"{pseudonym_id}_frame_{f_idx:03d}.jpg"
            frame_path = p_dir / frame_id
            
            # Phase 1: Baseline (1-10) -> mostly low stress (1-3)
            if f_idx <= 10:
                phase = "calm_baseline"
                likert = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.40, 0.12, 0.03]))
            # Phase 2: Workload & interruptions (11-40) -> moderate & high stress
            elif f_idx <= 40:
                phase = "simulated_workload"
                # Mid-phase spikes at interruptions
                if f_idx in [15, 20, 25, 30, 35]:
                    likert = int(rng.choice([6, 7, 8, 9, 10], p=[0.15, 0.30, 0.30, 0.15, 0.10]))
                else:
                    likert = int(rng.choice([3, 4, 5, 6, 7, 8], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05]))
            # Phase 3: Cooldown (41-50) -> recovery
            else:
                phase = "cooldown"
                likert = int(rng.choice([1, 2, 3, 4, 5], p=[0.30, 0.40, 0.20, 0.08, 0.02]))
                
            stress_class = map_likert_to_3class(likert)
            lighting = float(rng.uniform(0.85, 1.15))
            
            # Generate and save frame
            img = generate_face_frame(
                stress_class=stress_class,
                skin_tone_rgb=skin_tone,
                lighting_factor=lighting,
                rng=rng
            )
            cv2.imwrite(str(frame_path), img)
            
            # Observer rating for 20% subsample (f_idx % 5 == 0)
            is_subsample = (f_idx % 5 == 0)
            if is_subsample:
                # Moderate agreement with slight noise
                obs_c = stress_class if rng.random() < 0.72 else int(np.clip(stress_class + rng.choice([-1, 1]), 0, 2))
            else:
                obs_c = -1  # Not in 20% observer subsample
                
            records.append({
                "frame_id": frame_id,
                "participant_id": pseudonym_id,
                "frame_index": f_idx,
                "phase": phase,
                "relative_path": str(Path(pseudonym_id) / frame_id).replace("\\", "/"),
                "age": age,
                "age_bucket": age_bucket,
                "age_bucket_label": AGE_BUCKETS[age_bucket],
                "gender": gender,
                "gender_label": GENDER_MAP[gender],
                "pss10_score": pss_score,
                "pss10_class": pss_class,
                "pss10_class_label": STRESS_CLASSES[pss_class],
                "likert_raw": likert,
                "stress_class": stress_class,
                "stress_class_label": STRESS_CLASSES[stress_class],
                "is_observer_annotated": is_subsample,
                "observer_class": obs_c
            })
            
    df = pd.DataFrame(records)
    csv_path = labels_dir / "session_metadata.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Generated {len(df)} frames across {num_participants} participants.")
    print(f"Labels CSV saved to: {csv_path}")
    print("\nClass Distribution:")
    print(df["stress_class_label"].value_counts(normalize=True).round(3))
    print("\nAge Bucket Distribution:")
    print(df["age_bucket_label"].value_counts(normalize=True).round(3))
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic dataset generator for facial stress project")
    parser.add_argument("--participants", type=int, default=60, help="Number of participants (default 60)")
    parser.add_argument("--frames", type=int, default=50, help="Frames per participant (default 50)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    generate_dataset(
        output_base_dir=args.output_dir,
        num_participants=args.participants,
        frames_per_participant=args.frames
    )
