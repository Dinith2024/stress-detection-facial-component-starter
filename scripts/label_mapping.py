"""
scripts/label_mapping.py
Mapping and validation utilities for stress ratings.

Features:
1. Converts raw 10-point Likert stress self-reports (1-10) into 3-class stress labels:
   - Low (0): [1, 3]
   - Moderate (1): [4, 6]
   - High (2): [7, 10]
2. Supports configurable threshold schemes.
3. Computes inter-annotator agreement (Cohen's Kappa with 95% CI) between
   participant self-reports and independent observer ratings.
4. Maps PSS-10 raw survey totals (0-40) into construct validity percentile bands.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import cohen_kappa_score


# Label mapping constants
STRESS_CLASSES = {0: "low", 1: "moderate", 2: "high"}
INV_STRESS_CLASSES = {"low": 0, "moderate": 1, "high": 2}

# Demographic buckets
AGE_BUCKETS = {0: "18-29", 1: "30-44", 2: "45+"}
INV_AGE_BUCKETS = {"18-29": 0, "30-44": 1, "45+": 2}

GENDER_MAP = {0: "female", 1: "male", 2: "other"}
INV_GENDER_MAP = {"female": 0, "male": 1, "other": 2}


def map_likert_to_3class(
    rating: Union[int, float, np.ndarray, pd.Series],
    low_upper: float = 3.0,
    mod_upper: float = 6.0
) -> Union[int, np.ndarray, pd.Series]:
    """
    Map 10-point Likert scale (1-10) to 3 discrete classes.
    - Low (0): rating <= low_upper
    - Moderate (1): low_upper < rating <= mod_upper
    - High (2): rating > mod_upper
    """
    if isinstance(rating, (int, float)):
        if rating <= low_upper:
            return 0
        elif rating <= mod_upper:
            return 1
        else:
            return 2
    
    r = np.asarray(rating, dtype=float)
    classes = np.zeros_like(r, dtype=int)
    classes[(r > low_upper) & (r <= mod_upper)] = 1
    classes[r > mod_upper] = 2
    
    if isinstance(rating, pd.Series):
        return pd.Series(classes, index=rating.index, name="stress_class")
    return classes


def map_age_to_bucket(age: int) -> int:
    """Map numerical age to age bucket ID."""
    if age < 30:
        return 0  # 18-29
    elif age < 45:
        return 1  # 30-44
    else:
        return 2  # 45+


def map_pss10_to_class(pss_score: int) -> int:
    """
    Map Perceived Stress Scale (PSS-10, score range 0-40) into 3 stress categories.
    - 0-13: Low stress (0)
    - 14-26: Moderate stress (1)
    - 27-40: High perceived stress (2)
    """
    if pss_score <= 13:
        return 0
    elif pss_score <= 26:
        return 1
    else:
        return 2


def compute_cohens_kappa_with_ci(
    rater1: np.ndarray,
    rater2: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute Cohen's Kappa between two raters along with a 95% bootstrap confidence interval.
    """
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    assert len(r1) == len(r2), "Rater arrays must be of equal length."
    
    kappa_point = cohen_kappa_score(r1, r2)
    
    rng = np.random.default_rng(seed)
    n = len(r1)
    bootstrap_kappas = []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        try:
            k = cohen_kappa_score(r1[idx], r2[idx])
            bootstrap_kappas.append(k)
        except Exception:
            continue
            
    ci_lower = float(np.percentile(bootstrap_kappas, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(bootstrap_kappas, 100 * (1 - alpha / 2)))
    
    return {
        "kappa": float(kappa_point),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "sample_size": n
    }


def generate_synthetic_ratings_table(n_samples: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic rating table simulating the 20% observer-rated subsample.
    """
    rng = np.random.default_rng(seed)
    
    # 45% low (1-3), 35% mod (4-6), 20% high (7-10)
    base_probs = [0.45, 0.35, 0.20]
    true_classes = rng.choice([0, 1, 2], size=n_samples, p=base_probs)
    
    likert_ratings = []
    for c in true_classes:
        if c == 0:
            likert_ratings.append(rng.integers(1, 4))
        elif c == 1:
            likert_ratings.append(rng.integers(4, 7))
        else:
            likert_ratings.append(rng.integers(7, 11))
            
    likert_ratings = np.array(likert_ratings)
    self_report_class = map_likert_to_3class(likert_ratings)
    
    # Observer rating with moderate agreement (simulated noise ~ 30% flip)
    observer_class = []
    for sc in self_report_class:
        if rng.random() < 0.70:
            observer_class.append(sc)
        else:
            # Shift by +/- 1 class
            shift = rng.choice([-1, 1])
            new_c = np.clip(sc + shift, 0, 2)
            observer_class.append(int(new_c))
            
    df = pd.DataFrame({
        "frame_id": [f"synth_frame_{i:04d}.jpg" for i in range(n_samples)],
        "participant_id": [f"part_{(i // 10):03d}" for i in range(n_samples)],
        "likert_raw": likert_ratings,
        "self_report_class": self_report_class,
        "observer_class": observer_class
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label mapping and agreement tester")
    parser.add_argument("--test-synthetic", action="store_true", help="Run end-to-end test on synthetic ratings")
    args = parser.parse_args()
    
    if args.test_synthetic:
        df = generate_synthetic_ratings_table(n_samples=600)
        res = compute_cohens_kappa_with_ci(df["self_report_class"].values, df["observer_class"].values)
        print("=== Label Mapping & Inter-Annotator Agreement Test ===")
        print(f"Sample Size: {res['sample_size']} frames")
        print(f"Cohen's Kappa: {res['kappa']:.4f} (95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}])")
        print("\nClass Distribution:")
        print(df["self_report_class"].map(STRESS_CLASSES).value_counts(normalize=True))
