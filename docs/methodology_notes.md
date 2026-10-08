# Methodology and Engineering Design Notes

**Project Title**: Facial-Dynamics Stress Classification with Age/Gender-Conditioned Hierarchical Attention  
**Module**: IT41043 — Intelligent Systems | Horizon Campus  
**Authors**: R T Dinith Sasanga (Student ID: ITBIN-2313-0101), W G C M Nimsara  
**Date**: Academic Year 2026

---

## 1. Executive Summary & System Architecture

This document tracks engineering decisions, architectural justifications, preprocessing trade-offs, and compliance protocols for the facial dynamics stress detection subsystem.

```
[Raw Webcam Capture (60s)] 
           │
           ▼
[Face Detection & 25% Crop] ─── (Haar Cascade default -> MTCNN/MediaPipe roadmap)
           │
           ▼
[Photometric Normalisation] ─── (YCrCb Luminance Equalisation + ImageNet stats)
           │
           ▼
   [ResNet-18 Backbone]     ─── (Feature Map F in R^(B x 512 x 7 x 7))
           │
           ▼
[Demographic FiLM Conditioning] ─── [Age Bucket (18-29/30-44/45+) x Gender (F/M/O)]
           │                        │  MLP -> Scale (gamma) & Shift (beta)
           ▼                        │
[Spatial Attention + Attn Dropout] ◄┘
           │
           ▼
[Weighted Global Pooling]   ─── (Feature vector z in R^(B x 512))
           │
           ▼
[Classification Head (3-Class)] ──> [Low (0) / Moderate (1) / High (2) Stress]
```

---

## 2. Preprocessing Pipeline & Engineering Decisions

### 2.1 Face Detection & Margin Cropping (25% Expansion)
* **Default Implementation**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`). Lightweight, zero GPU dependency overhead, fast on resource-constrained devices.
* **Margin Justification**: Standard face bounding boxes tightly crop around facial landmarks. In workplace stress detection, key affective dynamic cues manifest at the peripheral facial boundaries:
  1. *Corrugator supercilii* (brow furrowing) near the upper forehead.
  2. *Masseter* (jaw clenching) near the lower jawline.
  3. *Orbicularis oculi* (eye narrowing / crow's feet tension).
  A fixed **25% margin expansion** ensures these peripheral stress cues remain inside the crop tensor while cropping away background distractions (monitor glare, room clutter).
* **Detection Yield Logging**: Rather than silently dropping missed frames, every missed detection is logged in `logs/face_yield.json` to accurately report system yield in experimental papers.
* **Planned Upgrade Roadmap**: Transition to MTCNN or MediaPipe Face Mesh prior to full on-site pilot trials to enhance robustness under extreme yaw/pitch angles (e.g. participant looking down at keyboard or multi-monitor setups).

### 2.2 Photometric Normalisation in YCrCb Color Space
* Office environments suffer from non-uniform fluorescent illumination, monitor flicker, and natural daylight variations across morning/afternoon sessions.
* Processing:
  1. Convert BGR to `YCrCb` space to separate luminance ($Y$) from chrominance ($Cr, Cb$).
  2. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) specifically to $Y$.
  3. Reconstruct RGB and apply ImageNet statistical standardization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
* Benefit: Preserves natural skin pigmentation tone while neutralizing illumination shadows.

---

## 3. Demographic Conditioning via Feature-wise Linear Modulation (FiLM)

### 3.1 Research Motivation
Psychological and affective computing literature demonstrates that stress perception, reporting, and facial expressiveness vary significantly across demographic cohorts:
* **Age Differences** (*Scott et al., 2013 [13]*): Emotional regulation and affective reactivity to workplace stressors exhibit distinct temporal and intensity profiles across age bands.
* **Gender Differences** (*Graves et al., 2021 [12]*, *Kuhn et al., 2023 [10]*): SHAP analysis reveals facial region importance interacts systematically with participant gender.

### 3.2 Mathematical Formulation
Rather than treating all individuals with a population-invariant static attention model:
$$\mathbf{e} = [\mathbf{e}_{\text{age}} \,\|\, \mathbf{e}_{\text{gender}}] \in \mathbb{R}^{32}$$
$$\gamma = \text{MLP}_{\gamma}(\mathbf{e}), \quad \beta = \text{MLP}_{\beta}(\mathbf{e})$$
$$\mathbf{S}_{\text{modulated}} = \gamma \odot \text{Conv}_{\text{attn}}(\mathbf{F}) + \beta$$
$$\mathbf{A}_{i,j} = \frac{\exp(\mathbf{S}_{\text{modulated}, i, j})}{\sum_{u,v} \exp(\mathbf{S}_{\text{modulated}, u, v})}$$

This allows the network to adaptively shift its spatial focus (e.g., placing higher relative weight on upper-face tension vs lower-face tension) conditioned on the participant's demographic profile.

---

## 4. Attention-Dropout Regularisation

* **Problem**: Attention mechanisms on modestly sized datasets (~3,000 frames) tend to over-concentrate on a narrow cluster of training-set features, degrading out-of-distribution generalisation (*Liu et al., 2023 [15]*).
* **Mechanism**: During training, a random subset of spatial attention positions is masked out ($p = 0.20$) before the spatial softmax:
  $$\mathbf{M}_{i,j} \sim \text{Bernoulli}(1 - p)$$
  $$\mathbf{S}_{\text{dropped}} = \text{where}(\mathbf{M} == 1, \mathbf{S}_{\text{mod}}, -\infty)$$
* **Impact**: Forces the model to diversify its learned representations across multiple facial regions rather than relying exclusively on a single dominant region.

---

## 5. Experimental Evaluation Protocol

1. **Participant-Level Stratified 5-Fold Cross-Validation**:
   - Folds partitioned at the participant level.
   - All ~50 frames of a given participant reside strictly in either the training partition or the validation partition.
   - Zero identity leakage prevents the CNN from memorizing subject facial identity rather than affective state.
2. **Class-Weighted Cross-Entropy Loss**:
   - Skew expected: ~45% Low, 35% Moderate, 20% High.
   - Weights $w_c = \frac{N}{K \cdot N_c}$ recomputed strictly from each fold's training partition.
3. **Primary Metric**: Macro-F1 (unweighted mean of class F1-scores).
4. **Significance Testing**:
   - Paired Wilcoxon Signed-Rank Test on 5 fold Macro-F1 differences ($\alpha = 0.05$).
   - Paired Bootstrap 95% Confidence Interval (10,000 resamples of pooled frame-level predictions).

---

## 6. Ethical Compliance & Data Privacy Protocol
*(Compliant with Sri Lanka Personal Data Protection Act No. 9 of 2022)*

1. **Pseudonymisation at Point of Capture**: All participant identifiers are transformed via one-way SHA-256 cryptographic hashing (`SUBJ_<hash>`). No names or email addresses are stored alongside facial tensors.
2. **Data Minimisation**:
   - Capture rate: Exactly 1 still frame every 60 seconds (no continuous video streaming).
   - Only cropped face regions are retained; raw background scenes are purged.
3. **Storage & Access Control**:
   - Raw data directories are strictly excluded from version control via `.gitignore`.
   - Access restricted to authorized researchers.
4. **Right to Erasure**:
   - Participants retain the right to withdraw at any stage, triggering immediate purge of associated hash folders.
