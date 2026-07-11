# Stress Detection and Management Application for IT Professionals — Facial Dynamics Component

IT41043 Intelligent Systems — Group Research Assignment, Horizon Campus, Academic Year 2026.

This repository holds the facial-dynamics stress-classification component of a larger
multimodal stress-detection system for IT professionals. It covers the branch of the
project that takes webcam-captured facial imagery and predicts a 3-class stress level
(low / moderate / high), conditioned on participant age and gender, using a
hierarchical-attention convolutional neural network. The keyboard/mouse-dynamics and
heart-rate-sensor components are maintained by the project partner in a companion
repository (linked below once available).

## Current status (Milestone 2)

- [x] Repository structure and dependency scaffolding
- [x] Face detection + crop script (`scripts/face_detect_align.py`) — working, Haar-cascade default
- [x] Photometric normalisation script (`scripts/normalize.py`) — working
- [x] Likert-to-class label mapping script (`scripts/label_mapping.py`) — working
- [ ] Pilot data collection (target: Week 8, pending Horizon Campus ethics approval)
- [ ] MTCNN swap-in for landmark-based alignment (see `docs/methodology_notes.md`)
- [ ] Model implementation (ResNet-18 backbone + attention module) — Milestone 3/4
- [ ] Training loop, cross-validation harness, baseline comparison — Milestone 4

This repository is **pre-data**: no participant data has been collected yet, so
`data/raw/`, `data/processed/`, and `data/labels/` are empty (tracked via `.gitkeep`
only). The scripts below are tested against synthetic and placeholder inputs and are
ready to run against real session data as soon as collection begins.

## Project structure

```
.
├── data/
│   ├── raw/              # untouched captured frames (never committed — see .gitignore)
│   ├── processed/        # face crops and normalised images (never committed)
│   └── labels/           # CSV label files (never committed — contains self-report data)
├── scripts/
│   ├── face_detect_align.py   # Stage 1: face detection + crop (Haar cascade)
│   ├── normalize.py           # Stage 2: resize, histogram-equalise, normalisation stats
│   └── label_mapping.py       # Likert rating → 3-class stress label
├── notebooks/            # exploratory analysis (empty pending pilot data)
├── models/                # model definitions and checkpoints (Milestone 3+)
├── docs/
│   └── methodology_notes.md   # working decisions log, kept separate from the report
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the preprocessing pipeline

```bash
# 1. Detect and crop faces from raw session frames
python scripts/face_detect_align.py \
    --input_dir data/raw \
    --output_dir data/processed/faces \
    --margin 0.25

# 2. Resize + normalise the crops
python scripts/normalize.py \
    --input_dir data/processed/faces \
    --output_dir data/processed/normalized \
    --size 224

# 3. Map raw Likert ratings to 3-class stress labels
python scripts/label_mapping.py \
    --input_csv data/labels/raw_ratings.csv \
    --output_csv data/labels/class_labels.csv
```

Each script can be run independently and logs its own progress (frames processed,
detection yield, class distribution) so pipeline issues surface early rather than
silently at training time.

## Data and ethics

No participant data is stored in this repository. Data collection follows the
protocol described in the Milestone 2 report (Section 1: Dataset Description),
including informed consent, Horizon Campus Faculty of IT ethics review, and
compliance with Sri Lanka's Personal Data Protection Act, No. 9 of 2022. Raw and
processed data directories are excluded from version control via `.gitignore`;
only code, documentation, and non-identifying summary statistics are tracked here.

## Team

- R.T.D. Sasanga (ITBIN-2313-0101) - facial-dynamics component (this repository)
- W.G.C.M. Nimsara - keyboard/mouse-dynamics and heart-rate-sensor components

Supervised by Mr. Isuru Madusanka Samarappulige. Module: IT41043 Intelligent Systems, Horizon Campus.
