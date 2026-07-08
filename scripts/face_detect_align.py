"""
face_detect_align.py

Stage 1 of the preprocessing pipeline (System Architecture, block 2).

Detects a face in a raw session frame using OpenCV's Haar cascade detector,
crops to the detected bounding box with a fixed margin, and writes the crop
to disk. This is intentionally implemented against OpenCV's bundled Haar
cascade (rather than MTCNN) as a dependency-light default so the script runs
out of the box on any machine with `opencv-python` installed; swap in
`detect_face_mtcnn()` once `mtcnn`/`facenet-pytorch` is added to
requirements.txt if landmark-based alignment is required for the final
model (see docs/methodology_notes.md).

Usage:
    python scripts/face_detect_align.py \
        --input_dir data/raw \
        --output_dir data/processed/faces \
        --margin 0.25
"""

import argparse
import logging
import os
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_face_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's pretrained frontal-face Haar cascade."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    return cascade


def detect_and_crop(image_path: Path, cascade: cv2.CascadeClassifier, margin: float):
    """
    Detect the largest face in an image and return a margin-padded crop.

    Returns None if no face is detected (the caller is responsible for
    logging / counting these as exclusions in the dataset report).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None

    # Largest detected face by area (handles rare cases of a bystander in frame)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)

    return img[y0:y1, x0:x1]


def process_directory(input_dir: Path, output_dir: Path, margin: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    cascade = get_face_cascade()

    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    logger.info("Found %d candidate images in %s", len(files), input_dir)

    n_ok, n_fail = 0, 0
    for path in files:
        crop = detect_and_crop(path, cascade, margin)
        if crop is None:
            n_fail += 1
            logger.warning("No face detected, excluding: %s", path.name)
            continue
        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), crop)
        n_ok += 1

    logger.info("Done. %d faces extracted, %d excluded (no detection).", n_ok, n_fail)
    if n_ok + n_fail > 0:
        logger.info("Detection yield: %.1f%%", 100 * n_ok / (n_ok + n_fail))


def main():
    parser = argparse.ArgumentParser(description="Detect and crop faces from raw session frames.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of raw frames.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write face crops.")
    parser.add_argument("--margin", type=float, default=0.25, help="Fractional margin added around the detected box.")
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output_dir), args.margin)


if __name__ == "__main__":
    main()
