"""
label_mapping.py

Converts raw self-report stress ratings (10-point Likert, collected
concurrently with each captured frame) plus baseline PSS-10 percentile
band into a single 3-class stress label: 0 = low, 1 = moderate, 2 = high.

Thresholds are defined in LIKERT_THRESHOLDS below and are deliberately
kept as named constants (not magic numbers inline) so they can be revised
after the pilot session without touching the rest of the pipeline. See
docs/methodology_notes.md for the justification of these cut points.

Expected input CSV columns:
    participant_id, session_id, frame_filename, likert_rating (1-10),
    pss10_percentile (0-100), annotator_id

Usage:
    python scripts/label_mapping.py \
        --input_csv data/labels/raw_ratings.csv \
        --output_csv data/labels/class_labels.csv
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIKERT_THRESHOLDS = {
    "low_max": 3,       # likert_rating <= 3  -> low
    "moderate_max": 6,  # 4 <= likert_rating <= 6 -> moderate
    # likert_rating >= 7 -> high
}

CLASS_NAMES = {0: "low", 1: "moderate", 2: "high"}


def likert_to_class(rating: float) -> int:
    if rating <= LIKERT_THRESHOLDS["low_max"]:
        return 0
    elif rating <= LIKERT_THRESHOLDS["moderate_max"]:
        return 1
    else:
        return 2


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stress_class"] = df["likert_rating"].apply(likert_to_class)
    df["stress_class_name"] = df["stress_class"].map(CLASS_NAMES)
    return df


def report_class_distribution(df: pd.DataFrame):
    counts = df["stress_class_name"].value_counts()
    total = len(df)
    logger.info("Class distribution (n=%d):", total)
    for name, count in counts.items():
        logger.info("  %-10s %5d  (%.1f%%)", name, count, 100 * count / total)


def main():
    parser = argparse.ArgumentParser(description="Map Likert stress ratings to 3-class labels.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required_cols = {"participant_id", "session_id", "frame_filename", "likert_rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    labelled = map_labels(df)
    report_class_distribution(labelled)
    labelled.to_csv(args.output_csv, index=False)
    logger.info("Wrote labelled dataset to %s", args.output_csv)


if __name__ == "__main__":
    main()
