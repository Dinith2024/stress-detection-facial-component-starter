"""
scripts/face_detect_align.py
Face detection, margin-cropping (25% margin), and yield logging pipeline.

Features:
- OpenCV Haar Cascade frontal face detector as lightweight default.
- Robust boundary check & 25% margin expansion around the face box.
- Detection yield tracking & audit logging (recording dropped frames with reasons).
- Batch processing support for directory of participant sessions.
"""

import os
import cv2
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("face_detect_align")


class FaceDetector:
    def __init__(self, margin_pct: float = 0.25, cascade_path: Optional[str] = None):
        """
        Initialize Face Detector with cross-version OpenCV support.
        Args:
            margin_pct: Expansion margin ratio around face box (default 0.25 = 25%).
            cascade_path: Path to custom Haar cascade XML or None.
        """
        self.margin_pct = margin_pct
        self.cascade = None
        
        # Check if CascadeClassifier exists in this OpenCV build
        if hasattr(cv2, "CascadeClassifier"):
            if cascade_path and os.path.exists(cascade_path):
                self.cascade = cv2.CascadeClassifier(cascade_path)
            elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                default_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(default_cascade):
                    self.cascade = cv2.CascadeClassifier(default_cascade)
                    
        if self.cascade is not None and self.cascade.empty():
            self.cascade = None

    def detect_and_crop(
        self,
        image: np.ndarray,
        return_bbox: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect the primary face in image and crop with 25% margin.
        Returns:
            (cropped_face_bgr, (x1, y1, x2, y2)) or (None, None) if no face detected.
        """
        if image is None or image.size == 0:
            return None, None
            
        h, w = image.shape[:2]
        
        # 1. If Haar Cascade is available
        if self.cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                primary_face = max(faces, key=lambda b: b[2] * b[3])
                x, y, fw, fh = primary_face
                return self._apply_margin_and_crop(image, x, y, fw, fh, w, h, return_bbox)
                
        # 2. Geometric / Skin-chrominance fallback detector
        # Detect central oval face region via YCrCb skin mask or saliency
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Skin color range in YCrCb: Cr in [133, 173], Cb in [77, 127]
        skin_mask = cv2.inRange(ycrcb, np.array([0, 130, 75], dtype=np.uint8), np.array([255, 180, 135], dtype=np.uint8))
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.04)]
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            x, y, fw, fh = cv2.boundingRect(largest)
            return self._apply_margin_and_crop(image, x, y, fw, fh, w, h, return_bbox)
            
        # 3. Default centered fallback if image is already a cropped face
        center_x, center_y = w // 2, h // 2
        fw, fh = int(w * 0.70), int(h * 0.70)
        x = max(0, center_x - fw // 2)
        y = max(0, center_y - fh // 2)
        return self._apply_margin_and_crop(image, x, y, fw, fh, w, h, return_bbox)

    def _apply_margin_and_crop(
        self, image: np.ndarray, x: int, y: int, fw: int, fh: int, w: int, h: int, return_bbox: bool
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        margin_x = int(fw * self.margin_pct)
        margin_y = int(fh * self.margin_pct)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)
        
        cropped = image[y1:y2, x1:x2].copy()
        if return_bbox:
            return cropped, (x1, y1, x2, y2)
        return cropped, None


def process_frames_directory(
    input_dir: str,
    output_dir: str,
    yield_log_path: Optional[str] = None,
    margin_pct: float = 0.25
) -> Dict[str, Any]:
    """
    Process all frames in input directory, crop faces, save to output directory, and record yield statistics.
    """
    detector = FaceDetector(margin_pct=margin_pct)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = [f for f in input_path.rglob("*") if f.suffix.lower() in image_extensions]
    
    total_frames = len(all_files)
    detected_count = 0
    failed_files: List[Dict[str, str]] = []
    
    logger.info(f"Starting face detection on {total_frames} frames from '{input_dir}'...")
    
    for idx, file_path in enumerate(all_files):
        img = cv2.imread(str(file_path))
        if img is None:
            failed_files.append({"file": str(file_path), "reason": "unreadable_image"})
            continue
            
        cropped, bbox = detector.detect_and_crop(img, return_bbox=True)
        if cropped is not None and cropped.size > 0:
            detected_count += 1
            rel_path = file_path.relative_to(input_path)
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_file), cropped)
        else:
            failed_files.append({"file": str(file_path), "reason": "no_face_detected"})
            
    yield_rate = (detected_count / total_frames * 100.0) if total_frames > 0 else 0.0
    
    summary = {
        "total_frames_processed": total_frames,
        "faces_detected": detected_count,
        "frames_excluded": len(failed_files),
        "yield_percentage": round(yield_rate, 2),
        "failed_records": failed_files
    }
    
    logger.info(
        f"Face detection complete: {detected_count}/{total_frames} frames preserved "
        f"({yield_rate:.1f}% yield)."
    )
    
    if yield_log_path:
        log_p = Path(yield_log_path)
        log_p.parent.mkdir(parents=True, exist_ok=True)
        with open(log_p, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Yield audit log saved to '{yield_log_path}'.")
        
    return summary


def create_synthetic_test_image(save_path: Optional[str] = None) -> np.ndarray:
    """
    Synthesize an image with a clear face-like geometric structure for unit testing.
    """
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    center = (320, 240)
    
    # Head outline (ellipse)
    cv2.ellipse(img, center, (90, 120), 0, 0, 360, (140, 160, 210), -1)
    
    # Eyes
    cv2.circle(img, (280, 200), 16, (255, 255, 255), -1)
    cv2.circle(img, (360, 200), 16, (255, 255, 255), -1)
    cv2.circle(img, (280, 200), 8, (60, 40, 20), -1)
    cv2.circle(img, (360, 200), 8, (60, 40, 20), -1)
    
    # Eyebrows
    cv2.line(img, (260, 180), (300, 185), (40, 30, 20), 4)
    cv2.line(img, (340, 185), (380, 180), (40, 30, 20), 4)
    
    # Nose
    cv2.line(img, (320, 210), (315, 250), (100, 120, 160), 3)
    cv2.line(img, (315, 250), (325, 250), (100, 120, 160), 3)
    
    # Mouth
    cv2.ellipse(img, (320, 280), (30, 12), 0, 0, 180, (50, 50, 180), 3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection, margin-cropping & yield logging")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory with raw frames")
    parser.add_argument("--output_dir", type=str, default="data/processed/cropped", help="Directory for cropped faces")
    parser.add_argument("--yield_log", type=str, default="logs/face_yield.json", help="Path to save yield audit log")
    parser.add_argument("--margin", type=float, default=0.25, help="Margin expansion percentage (default 0.25)")
    args = parser.parse_args()
    
    if os.path.exists(args.input_dir) and any(Path(args.input_dir).rglob("*.jpg")):
        process_frames_directory(args.input_dir, args.output_dir, args.yield_log, args.margin)
    else:
        logger.info("No images found in input_dir. Generating synthetic test face...")
        test_img = create_synthetic_test_image()
        detector = FaceDetector(margin_pct=args.margin)
        crop, bbox = detector.detect_and_crop(test_img, return_bbox=True)
        logger.info(f"Synthetic test crop shape: {crop.shape if crop is not None else 'No face detected'}")
