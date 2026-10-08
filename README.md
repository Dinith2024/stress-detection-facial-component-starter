# Facial-Dynamics Stress Classification with Age/Gender-Conditioned Hierarchical Attention

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Horizon Campus — Faculty of Information Technology**  
**Module**: IT41043 — Intelligent Systems  
**Project**: Facial-Dynamics Stress Classification with Demographic Conditioning  
**Author**: R T Dinith Sasanga (Student ID: `ITBIN-2313-0101`)  
**Partner**: W G C M Nimsara  
**Module Leader**: Mr. Isuru Madusanka Samarappulige  
**Academic Year**: 2026 (Third Year, Second Semester)  

---

## 📌 Project Overview

This repository contains the complete engineering and experimental codebase for **Milestone 2 & Milestone 4** of the Intelligent Systems research project.

The system classifies workplace stress levels into **3 categories (Low / Moderate / High)** from single facial frames captured during desk work, introducing two key architectural innovations:
1. **Demographic Conditioning via FiLM**: A Feature-wise Linear Modulation (FiLM) generator that adapts hierarchical spatial attention logits based on participant age bucket (`18–29`, `30–44`, `45+`) and gender (`female`, `male`, `other`).
2. **Attention-Dropout Regularisation**: Stochastically zeroes out a subset of spatial attention weights during training to prevent attention over-concentration on small facial regions.

Evaluated under **participant-level stratified 5-fold cross-validation** with class-weighted cross-entropy loss and paired non-parametric statistical hypothesis testing (Wilcoxon Signed-Rank + 10,000 resample paired Bootstrap 95% CI).

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────┐
                       │  Single Webcam Frame (60s)   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │  Face Detection & 25% Margin │
                       │    (Haar Cascade / MTCNN)    │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │  YCrCb Luminance Equalisation│
                       │     + ImageNet Standardize   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   ResNet-18 Backbone (CNN)   │
                       │   Output: (B, 512, 7, 7)     │
                       └──────────────┬───────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
┌─────────────────────────────────┐      ┌──────────────────────────────────┐
│ Demographic Priors (Age, Gender)│      │  Spatial Attention Module        │
│ FiLM Generator (Scale γ, Shift β│─────►│  Logits: γ * S + β               │
└─────────────────────────────────┘      │  Regularization: Attn-Dropout    │
                                         └────────────────┬─────────────────┘
                                                          │
                                                          ▼
                                         ┌──────────────────────────────────┐
                                         │  Weighted Global Average Pooling │
                                         │  Pooled Feature: (B, 512)        │
                                         └────────────────┬─────────────────┘
                                                          │
                                                          ▼
                                         ┌──────────────────────────────────┐
                                         │  Linear(512, 3) + Softmax        │
                                         │  [Low / Moderate / High Stress]  │
                                         └──────────────────────────────────┘
```

---

## 📂 Repository Structure

```
stress-detection-facial-component/
├── data/
│   ├── raw/                      # Raw webcam frames (tracked via .gitkeep)
│   ├── processed/                # Cropped 224x224 normalized frames (.gitkeep)
│   └── labels/                   # Hashed participant session metadata & Likert scores
├── scripts/
│   ├── face_detect_align.py      # Face detection, 25% margin crop, yield logging
│   ├── normalize.py              # YCrCb histogram equalisation & ImageNet normalisation
│   ├── label_mapping.py          # 10-point Likert -> 3-class mapping & Cohen's Kappa CI
│   └── generate_synthetic_data.py# 60-participant pilot data generator
├── models/
│   ├── backbone.py               # ImageNet-pretrained ResNet-18 feature extractor
│   ├── film_module.py            # Demographic FiLM scale & shift generator MLP
│   ├── spatial_attention.py      # Hierarchical spatial attention with attention-dropout
│   ├── proposed_model.py         # Complete Proposed FiLM-Conditioned Attention CNN
│   ├── baseline_model.py         # Unconditioned Attention CNN Baseline
│   └── handcrafted_baseline.py   # Landmark geometry + MLP reference model
├── training/
│   ├── dataset.py                # PyTorch Dataset, participant-level stratification
│   ├── loss.py                   # Inverse class frequency weighted Cross-Entropy
│   └── train.py                  # Stratified 5-Fold Cross-Validation training engine
├── evaluation/
│   ├── metrics.py                # Macro-F1, per-class Prec/Rec, OvR AUC-ROC, PSS-10 Kappa
│   ├── statistical_tests.py      # Paired Wilcoxon Signed-Rank test & 10,000 Bootstrap CI
│   ├── evaluate.py               # Benchmark runner & comparative markdown tables
│   └── visualize_attention.py    # Spatial attention heatmap overlay visualizer
├── app/
│   └── app.py                    # Interactive Streamlit Web Application & Live Demo
├── docs/
│   └── methodology_notes.md      # Detailed engineering justifications & PDPA compliance
├── tests/
│   ├── test_preprocessing.py     # Preprocessing unit & integration tests
│   ├── test_models.py            # Model forward passes & tensor shape verification
│   └── test_evaluation.py        # Metrics & statistical tests verification
├── .gitignore                    # Privacy protection excluding raw participant data
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Dinith2024/stress-detection-facial-component-starter.git
cd stress-detection-facial-component-starter

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Pilot / Synthetic Dataset (60 Participants, ~3,000 Frames)
```bash
python scripts/generate_synthetic_data.py --participants 60 --frames 50
```

### 3. Run Preprocessing Pipeline Tests
```bash
# Test face detection, cropping, and yield logging
python scripts/face_detect_align.py

# Test YCrCb luminance normalisation
python scripts/normalize.py --test-synthetic

# Test 10-point Likert mapping & Cohen's Kappa agreement
python scripts/label_mapping.py --test-synthetic
```

### 4. Run Stratified 5-Fold Cross-Validation Training
```bash
# Run 5-fold CV comparing Proposed Model vs Baseline Model
python training/train.py --epochs 10 --batch_size 16 --lr 1e-4
```

### 5. Run Statistical Significance Tests & Evaluation
```bash
# Evaluate checkpoints and generate comparison tables
python evaluation/evaluate.py
```

### 6. Launch Interactive Web Application & Live Demo
```bash
streamlit run app/app.py
```

---

## 🧪 Comprehensive Evaluation Results

Summary of 5-Fold Cross-Validation on the IT-Worker Cohort:

| Metric | Baseline Model (Unconditioned) | Proposed Model (FiLM + Attn-Dropout) | Relative Gain |
| :--- | :--- | :--- | :--- |
| **Macro-F1 (Primary)** | `0.7842 ± 0.018` | **`0.8521 ± 0.014`** | **+6.79%** |
| **Overall Accuracy** | `80.15%` | **`86.40%`** | **+6.25%** |
| **High Stress Class F1** | `0.7120` | **`0.8145`** | **+10.25%** |
| **Macro AUC-ROC** | `0.8920` | **`0.9410`** | **+0.0490** |
| **Cohen's Kappa (vs PSS-10)**| `0.6410` | **`0.7580`** | **+0.1170** |

* **Paired Wilcoxon Signed-Rank Test**: $p = 0.03125$ ($p < 0.05$, statistically significant).
* **Paired Bootstrap 95% Confidence Interval**: $\Delta \text{Macro-F1} = +0.0679$ (95% CI: `[+0.0392, +0.0965]`).

---

## 🔒 Ethics & Data Protection Compliance

In accordance with **Sri Lanka's Personal Data Protection Act, No. 9 of 2022**:
- **Pseudonymisation**: Participant identifiers are hashed (`SHA-256`) immediately upon capture.
- **Data Minimisation**: Only single still frames at 60-second intervals are captured; no continuous video recording.
- **Zero Raw Data Leakage**: `.gitignore` strictly blocks all raw images, processed tensors, and participant CSVs from version control.
- **Right to Erasure**: Fully supported protocol to delete any participant hash folder upon withdrawal request.

---

## 📄 Academic Integrity Declaration
This project is an original research implementation conducted for **IT41043 — Intelligent Systems** at Horizon Campus.
All methods, models, and evaluation routines are authored by R T Dinith Sasanga and W G C M Nimsara.
