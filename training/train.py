"""
training/train.py
Stratified 5-Fold Cross-Validation Training Engine for Facial Dynamics Stress Classification.

Trains:
1. Proposed Model: FiLM-Conditioned Hierarchical Attention CNN + Attention Dropout
2. Baseline Model: Unconditioned Hierarchical Attention CNN (No FiLM, No Attention Dropout)

Training configuration per Section 2.2 and 3:
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: Cosine Annealing Learning Rate
- Loss: Class-Weighted Cross-Entropy (recalculated per training fold split)
- Stratification: Participant-level stratified 5-fold CV (zero frame leakage)
"""

import os
import copy
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.proposed_model import ProposedStressModel
from models.baseline_model import BaselineAttentionModel
from training.dataset import FacialStressDataset, create_participant_level_5fold_splits
from training.loss import compute_inverse_class_weights, ClassWeightedCrossEntropyLoss
from evaluation.metrics import compute_all_metrics, format_metrics_table
from evaluation.statistical_tests import (
    paired_wilcoxon_test,
    paired_bootstrap_confidence_interval,
    generate_statistical_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_proposed_model: bool = True
) -> float:
    """Train for single epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["stress_class"].to(device)
        age_bucket = batch["age_bucket"].to(device)
        gender = batch["gender"].to(device)
        
        optimizer.zero_grad()
        
        if is_proposed_model:
            outputs = model(images, age_bucket, gender)
        else:
            outputs = model(images)
            
        loss = criterion(outputs["logits"], targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
    return running_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_proposed_model: bool = True
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test dataloader."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    all_targets = []
    all_preds = []
    all_probs = []
    all_pss10 = []
    
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["stress_class"].to(device)
        age_bucket = batch["age_bucket"].to(device)
        gender = batch["gender"].to(device)
        pss10 = batch["pss10_class"].to(device)
        
        if is_proposed_model:
            outputs = model(images, age_bucket, gender)
        else:
            outputs = model(images)
            
        loss = criterion(outputs["logits"], targets)
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        probs = outputs["probabilities"].detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_pss10.extend(pss10.cpu().numpy())
        
    avg_loss = running_loss / max(1, total_samples)
    metrics = compute_all_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_probs=np.array(all_probs),
        y_pss10=np.array(all_pss10)
    )
    return avg_loss, metrics, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def run_5fold_cross_validation(
    metadata_path: str = "data/labels/session_metadata.csv",
    data_root: str = "data",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    output_dir: str = "checkpoints",
    device_str: str = "auto",
    quick_run: bool = False
) -> Dict[str, Any]:
    """
    Execute full 5-fold cross-validation benchmark comparing Proposed Model vs Baseline Model.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Resolve device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using compute device: {device}")
    
    # Load dataset metadata
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} records for {df['participant_id'].nunique()} participants from '{metadata_path}'")
    
    if quick_run:
        epochs = min(epochs, 3)
        
    folds = create_participant_level_5fold_splits(df)
    
    proposed_fold_results = []
    baseline_fold_results = []
    
    proposed_pooled_targets = []
    proposed_pooled_preds = []
    
    baseline_pooled_targets = []
    baseline_pooled_preds = []
    
    for fold_idx, (train_df, val_df) in enumerate(folds):
        fold_num = fold_idx + 1
        logger.info(f"\n{'='*30} FOLD {fold_num}/5 {'='*30}")
        logger.info(f"Train samples: {len(train_df)} ({train_df['participant_id'].nunique()} participants) | "
                    f"Val samples: {len(val_df)} ({val_df['participant_id'].nunique()} participants)")
        
        # Calculate class weights for this fold's training partition
        train_labels = train_df["stress_class"].values
        class_weights = compute_inverse_class_weights(train_labels, num_classes=3).to(device)
        criterion = ClassWeightedCrossEntropyLoss(weights=class_weights)
        
        train_ds = FacialStressDataset(train_df, data_root=data_root, is_training=True)
        val_ds = FacialStressDataset(val_df, data_root=data_root, is_training=False)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # -------------------------------------------------------------
        # 1. Train Proposed Model (FiLM Conditioned + Attention Dropout)
        # -------------------------------------------------------------
        logger.info(f"--- Training Proposed Model (FiLM Attention + Attention-Dropout) [Fold {fold_num}] ---")
        proposed_model = ProposedStressModel(pretrained_backbone=False).to(device)
        opt_prop = torch.optim.AdamW(proposed_model.parameters(), lr=lr, weight_decay=weight_decay)
        sched_prop = torch.optim.lr_scheduler.CosineAnnealingLR(opt_prop, T_max=epochs)
        
        best_prop_f1 = -1.0
        best_prop_metrics = None
        best_prop_weights = None
        best_prop_targets, best_prop_preds = None, None
        
        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch(proposed_model, train_loader, opt_prop, criterion, device, is_proposed_model=True)
            sched_prop.step()
            val_loss, val_metrics, y_t, y_p, _ = evaluate_model(proposed_model, val_loader, criterion, device, is_proposed_model=True)
            
            if val_metrics["macro_f1"] > best_prop_f1:
                best_prop_f1 = val_metrics["macro_f1"]
                best_prop_metrics = val_metrics
                best_prop_weights = copy.deepcopy(proposed_model.state_dict())
                best_prop_targets = y_t
                best_prop_preds = y_p
                
            if ep % 5 == 0 or ep == epochs:
                logger.info(f"[Proposed Ep {ep}/{epochs}] TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | "
                            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']*100:.1f}%")
                
        # Save best fold model
        prop_ckpt_path = out_path / f"proposed_model_fold{fold_num}.pt"
        torch.save(best_prop_weights, str(prop_ckpt_path))
        proposed_fold_results.append(best_prop_metrics)
        proposed_pooled_targets.extend(best_prop_targets)
        proposed_pooled_preds.extend(best_prop_preds)
        
        # -------------------------------------------------------------
        # 2. Train Baseline Model (Unconditioned, No Attention Dropout)
        # -------------------------------------------------------------
        logger.info(f"--- Training Baseline Model (Unconditioned Attention CNN) [Fold {fold_num}] ---")
        baseline_model = BaselineAttentionModel(pretrained_backbone=False).to(device)
        opt_base = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=weight_decay)
        sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=epochs)
        
        best_base_f1 = -1.0
        best_base_metrics = None
        best_base_weights = None
        best_base_targets, best_base_preds = None, None
        
        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch(baseline_model, train_loader, opt_base, criterion, device, is_proposed_model=False)
            sched_base.step()
            val_loss, val_metrics, y_t, y_p, _ = evaluate_model(baseline_model, val_loader, criterion, device, is_proposed_model=False)
            
            if val_metrics["macro_f1"] > best_base_f1:
                best_base_f1 = val_metrics["macro_f1"]
                best_base_metrics = val_metrics
                best_base_weights = copy.deepcopy(baseline_model.state_dict())
                best_base_targets = y_t
                best_base_preds = y_p
                
            if ep % 5 == 0 or ep == epochs:
                logger.info(f"[Baseline Ep {ep}/{epochs}] TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | "
                            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']*100:.1f}%")
                
        base_ckpt_path = out_path / f"baseline_model_fold{fold_num}.pt"
        torch.save(best_base_weights, str(base_ckpt_path))
        baseline_fold_results.append(best_base_metrics)
        baseline_pooled_targets.extend(best_base_targets)
        baseline_pooled_preds.extend(best_base_preds)
        
    # Aggregate statistics
    prop_f1s = [m["macro_f1"] for m in proposed_fold_results]
    base_f1s = [m["macro_f1"] for m in baseline_fold_results]
    
    wilcoxon_res = paired_wilcoxon_test(prop_f1s, base_f1s)
    bootstrap_res = paired_bootstrap_confidence_interval(
        y_true=np.array(proposed_pooled_targets),
        y_pred_proposed=np.array(proposed_pooled_preds),
        y_pred_baseline=np.array(baseline_pooled_preds),
        n_resamples=10000
    )
    
    summary_report = {
        "num_folds": 5,
        "proposed_model": {
            "mean_macro_f1": round(float(np.mean(prop_f1s)), 4),
            "std_macro_f1": round(float(np.std(prop_f1s, ddof=1)), 4),
            "fold_macro_f1s": prop_f1s,
            "mean_accuracy": round(float(np.mean([m['accuracy'] for m in proposed_fold_results])), 4)
        },
        "baseline_model": {
            "mean_macro_f1": round(float(np.mean(base_f1s)), 4),
            "std_macro_f1": round(float(np.std(base_f1s, ddof=1)), 4),
            "fold_macro_f1s": base_f1s,
            "mean_accuracy": round(float(np.mean([m['accuracy'] for m in baseline_fold_results])), 4)
        },
        "wilcoxon_test": wilcoxon_res,
        "bootstrap_ci": bootstrap_res
    }
    
    summary_json_path = out_path / "cross_validation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
        
    stat_md = generate_statistical_report(wilcoxon_res, bootstrap_res)
    stat_report_path = out_path / "statistical_significance_report.md"
    with open(stat_report_path, "w") as f:
        f.write(stat_md)
        
    logger.info("\n" + "="*70)
    logger.info("5-FOLD CROSS-VALIDATION COMPLETE")
    logger.info(f"Proposed Model Macro-F1: {summary_report['proposed_model']['mean_macro_f1']:.4f} +/- {summary_report['proposed_model']['std_macro_f1']:.4f}")
    logger.info(f"Baseline Model Macro-F1: {summary_report['baseline_model']['mean_macro_f1']:.4f} +/- {summary_report['baseline_model']['std_macro_f1']:.4f}")
    logger.info(f"Wilcoxon p-value:        {wilcoxon_res['p_value']} (Significant: {wilcoxon_res['is_significant']})")
    logger.info(f"95% Bootstrap CI:        [{bootstrap_res['ci_lower']:.5f}, {bootstrap_res['ci_upper']:.5f}]")
    logger.info("="*70)
    
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation Training Engine")
    parser.add_argument("--metadata", type=str, default="data/labels/session_metadata.csv")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true", help="Quick run with 2 epochs per fold")
    args = parser.parse_args()
    
    run_5fold_cross_validation(
        metadata_path=args.metadata,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quick_run=args.quick
    )
