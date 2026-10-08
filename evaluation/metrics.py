"""
evaluation/metrics.py
Evaluation metrics for 3-class facial dynamics stress classification.

Metrics implemented per Section 4.1 of Milestone 2:
1. Macro-F1 (Primary Metric): unweighted average of F1 across low, moderate, and high stress.
2. Per-class Precision and Recall (detects minority high-stress weaknesses).
3. Macro-averaged One-vs-Rest AUC-ROC (threshold-independent separability).
4. Cohen's Kappa against PSS-10 percentile bands (construct-validity check).
5. Multi-class Confusion Matrix and overall accuracy.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score
)
from scripts.label_mapping import STRESS_CLASSES


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    y_pss10: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics suite for stress classification.
    Args:
        y_true: (N,) ground-truth class indices [0, 1, 2]
        y_pred: (N,) predicted class indices [0, 1, 2]
        y_probs: Optional (N, 3) predicted class probability distribution
        y_pss10: Optional (N,) PSS-10 derived stress class for construct validity check
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # 1. Primary metric: Macro-F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    
    # 2. Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    
    # 3. Macro-averaged One-vs-Rest AUC-ROC
    auc_roc = None
    if y_probs is not None and y_probs.shape[1] == 3:
        try:
            # Multi-class OvR AUC-ROC
            auc_roc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
        except Exception:
            auc_roc = None
            
    # 4. Cohen's Kappa against PSS-10
    cohen_kappa_pss10 = None
    if y_pss10 is not None:
        try:
            cohen_kappa_pss10 = float(cohen_kappa_score(y_pred, np.asarray(y_pss10, dtype=int)))
        except Exception:
            cohen_kappa_pss10 = None
            
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    results = {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "macro_auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
        "cohen_kappa_pss10": round(cohen_kappa_pss10, 4) if cohen_kappa_pss10 is not None else None,
        "per_class": {
            STRESS_CLASSES[i]: {
                "f1": round(float(per_class_f1[i]), 4),
                "precision": round(float(per_class_prec[i]), 4),
                "recall": round(float(per_class_rec[i]), 4)
            }
            for i in range(3)
        },
        "confusion_matrix": cm,
        "total_samples": len(y_true)
    }
    return results


def format_metrics_table(results: Dict[str, Any], title: str = "Evaluation Summary") -> str:
    """Format metrics dictionary into a clean markdown table."""
    lines = [
        f"### {title}",
        f"- **Primary Metric (Macro-F1)**: `{results['macro_f1']:.4f}`",
        f"- **Accuracy**: `{results['accuracy'] * 100:.2f}%`",
        f"- **Weighted-F1**: `{results['weighted_f1']:.4f}`",
        f"- **Macro AUC-ROC**: `{results['macro_auc_roc'] if results['macro_auc_roc'] is not None else 'N/A'}`",
        f"- **Cohen's Kappa (vs PSS-10)**: `{results['cohen_kappa_pss10'] if results['cohen_kappa_pss10'] is not None else 'N/A'}`",
        f"- **Total Evaluated Samples**: `{results['total_samples']}`",
        "",
        "| Stress Class | Precision | Recall | F1-Score |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for c_name, m in results["per_class"].items():
        lines.append(f"| **{c_name.capitalize()}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
        
    return "\n".join(lines)


if __name__ == "__main__":
    y_t = np.random.choice([0, 1, 2], size=300, p=[0.45, 0.35, 0.20])
    y_p = np.clip(y_t + np.random.choice([-1, 0, 1], size=300, p=[0.1, 0.8, 0.1]), 0, 2)
    y_pr = np.eye(3)[y_p] * 0.8 + 0.066
    
    res = compute_all_metrics(y_t, y_p, y_pr, y_pss10=y_t)
    print(format_metrics_table(res, "Synthetic Validation Run"))
