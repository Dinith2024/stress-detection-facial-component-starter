"""
normalize.py

Stage 2 of the preprocessing pipeline (System Architecture, block 3).

Takes the face crops produced by face_detect_align.py and:
  1. resizes to a fixed 224x224 input size (matching the ResNet-18 backbone),
  2. applies histogram equalisation on the luminance channel for lighting
     robustness across session times/rooms,
  3. writes ImageNet-normalisation statistics (mean/std) to a JSON sidecar
     rather than baking them into the image, so the same crop can be reused
     for augmented training views without re-deriving normalisation.

Usage:
    python scripts/normalize.py \
        --input_dir data/processed/faces \
        --output_dir data/processed/normalized \
        --size 224
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def equalize_luminance(img_bgr: np.ndarray) -> np.ndarray:
    """Histogram-equalise the Y channel in YCrCb space, preserving colour."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def process_image(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read {path}")
    img = equalize_luminance(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def process_directory(input_dir: Path, output_dir: Path, size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    logger.info("Normalising %d images to %dx%d", len(files), size, size)

    for path in files:
        try:
            out_img = process_image(path, size)
        except ValueError as e:
            logger.warning(str(e))
            continue
        cv2.imwrite(str(output_dir / path.name), out_img)

    # Write normalisation stats once per run, not per image
    stats_path = output_dir / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "input_size": size}, f, indent=2)
    logger.info("Wrote normalisation stats to %s", stats_path)


def main():
    parser = argparse.ArgumentParser(description="Resize and photometrically normalise face crops.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output_dir), args.size)


if __name__ == "__main__":
    main()
