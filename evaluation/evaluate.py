"""
evaluation/evaluate.py
Inference, cross-fold benchmark evaluation, and publication-ready metrics reporting.

Loads trained checkpoints and evaluates:
1. Proposed Model (FiLM Demographics + Hierarchical Attention)
2. Baseline Model (Unconditioned Attention CNN)
3. Handcrafted Landmark Feature Baseline (Pise et al. [18])
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.proposed_model import ProposedStressModel
from models.baseline_model import BaselineAttentionModel
from models.handcrafted_baseline import ClassicalMLPClassifier, HandcraftedFeatureExtractor
from training.dataset import FacialStressDataset
from evaluation.metrics import compute_all_metrics, format_metrics_table
from evaluation.statistical_tests import (
    paired_wilcoxon_test,
    paired_bootstrap_confidence_interval,
    generate_statistical_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def evaluate_checkpoint(
    checkpoint_path: str,
    metadata_df: pd.DataFrame,
    data_root: str,
    model_type: str = "proposed",
    device_str: str = "auto",
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Evaluate single model checkpoint on metadata_df.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    if model_type == "proposed":
        model = ProposedStressModel(pretrained_backbone=False).to(device)
    else:
        model = BaselineAttentionModel(pretrained_backbone=False).to(device)
        
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from '{checkpoint_path}'")
    else:
        logger.warning(f"Checkpoint '{checkpoint_path}' not found, using initialized weights.")
        
    model.eval()
    dataset = FacialStressDataset(metadata_df, data_root=data_root, is_training=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_targets = []
    all_preds = []
    all_probs = []
    all_pss10 = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["stress_class"].to(device)
            age_bucket = batch["age_bucket"].to(device)
            gender = batch["gender"].to(device)
            pss10 = batch["pss10_class"].to(device)
            
            if model_type == "proposed":
                outputs = model(images, age_bucket, gender)
            else:
                outputs = model(images)
                
            probs = outputs["probabilities"].cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_pss10.extend(pss10.cpu().numpy())
            
    metrics = compute_all_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_probs=np.array(all_probs),
        y_pss10=np.array(all_pss10)
    )
    return {
        "metrics": metrics,
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs)
    }


def generate_benchmark_comparison_table(
    prop_metrics: Dict[str, Any],
    base_metrics: Dict[str, Any],
    save_markdown_path: Optional[str] = None
) -> str:
    """
    Format comparison table between Proposed Model and Baseline.
    """
    lines = [
        "# Model Benchmark & Comparative Evaluation Summary",
        "",
        "| Evaluation Metric | Baseline Model (Unconditioned Attention) | Proposed Model (FiLM Conditioned + Attn Dropout) | Delta Gain |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Macro-F1 (Primary)** | `{base_metrics['macro_f1']:.4f}` | `**{prop_metrics['macro_f1']:.4f}**` | `+{prop_metrics['macro_f1'] - base_metrics['macro_f1']:+.4f}` |",
        f"| **Overall Accuracy** | `{base_metrics['accuracy']*100:.2f}%` | `**{prop_metrics['accuracy']*100:.2f}%**` | `+{prop_metrics['accuracy']*100 - base_metrics['accuracy']*100:+.2f}%` |",
        f"| **Weighted-F1** | `{base_metrics['weighted_f1']:.4f}` | `{prop_metrics['weighted_f1']:.4f}` | `+{prop_metrics['weighted_f1'] - base_metrics['weighted_f1']:+.4f}` |",
        f"| **Macro AUC-ROC** | `{base_metrics['macro_auc_roc']}` | `{prop_metrics['macro_auc_roc']}` | `+{float(prop_metrics['macro_auc_roc'] or 0) - float(base_metrics['macro_auc_roc'] or 0):+.4f}` |",
        f"| **Cohen's Kappa (vs PSS-10)** | `{base_metrics['cohen_kappa_pss10']}` | `{prop_metrics['cohen_kappa_pss10']}` | `+{float(prop_metrics['cohen_kappa_pss10'] or 0) - float(base_metrics['cohen_kappa_pss10'] or 0):+.4f}` |",
        "",
        "### Class-Specific F1-Score Breakdown",
        "| Stress Class | Baseline F1 | Proposed F1 | Delta F1 |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for c in ["low", "moderate", "high"]:
        b_f1 = base_metrics["per_class"][c]["f1"]
        p_f1 = prop_metrics["per_class"][c]["f1"]
        lines.append(f"| **{c.capitalize()} Stress** | `{b_f1:.4f}` | `**{p_f1:.4f}**` | `{p_f1 - b_f1:+.4f}` |")
        
    table_str = "\n".join(lines)
    
    if save_markdown_path:
        Path(save_markdown_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_markdown_path, "w") as f:
            f.write(table_str)
            
    return table_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints")
    parser.add_argument("--proposed_ckpt", type=str, default="checkpoints/proposed_model_fold1.pt")
    parser.add_argument("--baseline_ckpt", type=str, default="checkpoints/baseline_model_fold1.pt")
    parser.add_argument("--metadata", type=str, default="data/labels/session_metadata.csv")
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()
    
    if os.path.exists(args.metadata):
        df = pd.read_csv(args.metadata)
        logger.info(f"Evaluating checkpoints on {len(df)} samples...")
        prop_res = evaluate_checkpoint(args.proposed_ckpt, df, args.data_root, model_type="proposed")
        base_res = evaluate_checkpoint(args.baseline_ckpt, df, args.data_root, model_type="baseline")
        
        table = generate_benchmark_comparison_table(prop_res["metrics"], base_res["metrics"])
        print("\n" + table)
