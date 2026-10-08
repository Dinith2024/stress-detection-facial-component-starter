"""
evaluation/statistical_tests.py
Statistical significance testing and effect-size estimation.

Implements Section 4.3 of Milestone 2:
1. Paired Wilcoxon Signed-Rank Test on 5 fold-level Macro-F1 differences (alpha = 0.05).
2. Paired Bootstrap 95% Confidence Interval on frame-level pooled predictions
   (10,000 resamples with replacement).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from sklearn.metrics import f1_score


def paired_wilcoxon_test(
    proposed_fold_f1s: List[float],
    baseline_fold_f1s: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform paired Wilcoxon signed-rank test on 5 fold Macro-F1 differences.
    """
    p_arr = np.asarray(proposed_fold_f1s, dtype=float)
    b_arr = np.asarray(baseline_fold_f1s, dtype=float)
    diffs = p_arr - b_arr
    
    # Check for identical values
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "is_significant": False,
            "alpha": alpha,
            "test_name": "Wilcoxon Signed-Rank Test"
        }
        
    try:
        # Wilcoxon test with zero_method='pratt' or 'wilcox'
        stat_res = stats.wilcoxon(p_arr, b_arr, alternative="greater", zero_method="wilcox")
        stat_val = float(stat_res.statistic)
        p_val = float(stat_res.pvalue)
    except Exception as e:
        stat_val = 0.0
        p_val = 1.0
        
    return {
        "statistic": stat_val,
        "p_value": round(p_val, 5),
        "mean_diff": round(float(np.mean(diffs)), 5),
        "std_diff": round(float(np.std(diffs, ddof=1)), 5),
        "proposed_mean": round(float(np.mean(p_arr)), 5),
        "baseline_mean": round(float(np.mean(b_arr)), 5),
        "is_significant": p_val < alpha,
        "alpha": alpha,
        "test_name": "Paired Wilcoxon Signed-Rank Test (one-sided: Proposed > Baseline)"
    }


def paired_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred_proposed: np.ndarray,
    y_pred_baseline: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute paired bootstrap 95% confidence interval on pooled frame-level predictions.
    Resamples frame indices with replacement 10,000 times, recomputing:
    Delta Macro-F1 = Macro-F1(Proposed) - Macro-F1(Baseline).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prop = np.asarray(y_pred_proposed, dtype=int)
    y_base = np.asarray(y_pred_baseline, dtype=int)
    n_samples = len(y_true)
    
    # Point estimate on entire dataset
    f1_prop_pt = f1_score(y_true, y_prop, average="macro", zero_division=0)
    f1_base_pt = f1_score(y_true, y_base, average="macro", zero_division=0)
    delta_pt = float(f1_prop_pt - f1_base_pt)
    
    rng = np.random.default_rng(seed)
    deltas = []
    
    for _ in range(n_resamples):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        f1_p = f1_score(y_true[idx], y_prop[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], y_base[idx], average="macro", zero_division=0)
        deltas.append(f1_p - f1_b)
        
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(deltas, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(deltas, 100 * (1.0 - alpha / 2.0)))
    bootstrap_mean = float(np.mean(deltas))
    bootstrap_std = float(np.std(deltas))
    
    # Empirical p-value that delta <= 0
    p_emp = float(np.mean(np.array(deltas) <= 0))
    
    return {
        "point_estimate_delta_macro_f1": round(delta_pt, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "bootstrap_mean": round(bootstrap_mean, 5),
        "bootstrap_std": round(bootstrap_std, 5),
        "empirical_p_value": round(p_emp, 5),
        "is_significant": ci_lower > 0.0
    }


def generate_statistical_report(
    wilcoxon_res: Dict[str, Any],
    bootstrap_res: Dict[str, Any]
) -> str:
    """Format full statistical findings into publication-ready markdown."""
    sig_status = "Statistically Significant" if (wilcoxon_res["is_significant"] or bootstrap_res["is_significant"]) else "Null Result (Not Significant)"
    
    report = [
        "## Statistical Significance and Hypothesis Testing Report",
        f"**Hypothesis Test Status**: `{sig_status}`",
        "",
        "### 1. Paired Wilcoxon Signed-Rank Test (5-Fold Level)",
        f"- **Proposed Model Mean Macro-F1**: `{wilcoxon_res.get('proposed_mean', 'N/A')}`",
        f"- **Baseline Model Mean Macro-F1**: `{wilcoxon_res.get('baseline_mean', 'N/A')}`",
        f"- **Mean Fold-to-Fold Delta**: `{wilcoxon_res.get('mean_diff', 'N/A')} +/- {wilcoxon_res.get('std_diff', 'N/A')}`",
        f"- **Wilcoxon Test Statistic (W)**: `{wilcoxon_res['statistic']}`",
        f"- **p-value**: `{wilcoxon_res['p_value']}` (alpha = {wilcoxon_res['alpha']})",
        f"- **Wilcoxon Decision**: {'Reject Null Hypothesis (H0)' if wilcoxon_res['is_significant'] else 'Fail to Reject Null (H0)'}",
        "",
        "### 2. Paired Bootstrap Confidence Interval (10,000 Resamples)",
        f"- **Point Estimate Delta Macro-F1**: `{bootstrap_res['point_estimate_delta_macro_f1']:.5f}`",
        f"- **95% Bootstrap Confidence Interval**: `[{bootstrap_res['ci_lower']:.5f}, {bootstrap_res['ci_upper']:.5f}]`",
        f"- **Bootstrap Std Error**: `{bootstrap_res['bootstrap_std']:.5f}`",
        f"- **Empirical p-value (Delta <= 0)**: `{bootstrap_res['empirical_p_value']:.5f}`",
        f"- **Effect Size Conclusion**: {'95% CI strictly excludes zero -> Robust positive effect.' if bootstrap_res['is_significant'] else '95% CI spans zero -> Inconclusive effect size.'}"
    ]
    return "\n".join(report)


if __name__ == "__main__":
    # Test with synthetic folds
    p_f1s = [0.82, 0.85, 0.81, 0.84, 0.86]
    b_f1s = [0.77, 0.80, 0.76, 0.79, 0.81]
    
    w_res = paired_wilcoxon_test(p_f1s, b_f1s)
    
    y_true = np.random.choice([0, 1, 2], size=1000)
    y_p = y_true.copy()
    y_b = y_true.copy()
    # add noise
    y_p[np.random.choice(1000, 150)] = np.random.choice([0, 1, 2], 150)
    y_b[np.random.choice(1000, 250)] = np.random.choice([0, 1, 2], 250)
    
    b_res = paired_bootstrap_confidence_interval(y_true, y_p, y_b, n_resamples=1000)
    print(generate_statistical_report(w_res, b_res))
