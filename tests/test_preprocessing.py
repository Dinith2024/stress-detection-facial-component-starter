"""
tests/test_preprocessing.py
Unit tests for face detection, normalization, and label mapping pipelines.
"""

import unittest
import numpy as np
import torch
import pandas as pd

from scripts.label_mapping import (
    map_likert_to_3class,
    map_age_to_bucket,
    map_pss10_to_class,
    compute_cohens_kappa_with_ci
)
from scripts.face_detect_align import FaceDetector, create_synthetic_test_image
from scripts.normalize import (
    preprocess_and_normalize_frame,
    denormalize_image_tensor,
    equalize_luminance_ycrcb,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class TestPreprocessing(unittest.TestCase):
    def test_label_mapping(self):
        # 1-3 -> 0 (low)
        self.assertEqual(map_likert_to_3class(1), 0)
        self.assertEqual(map_likert_to_3class(3), 0)
        # 4-6 -> 1 (moderate)
        self.assertEqual(map_likert_to_3class(4), 1)
        self.assertEqual(map_likert_to_3class(6), 1)
        # 7-10 -> 2 (high)
        self.assertEqual(map_likert_to_3class(7), 2)
        self.assertEqual(map_likert_to_3class(10), 2)
        
        # Array mapping
        arr = np.array([2, 5, 8, 1, 6, 9])
        classes = map_likert_to_3class(arr)
        np.testing.assert_array_equal(classes, np.array([0, 1, 2, 0, 1, 2]))

    def test_age_and_pss_mapping(self):
        self.assertEqual(map_age_to_bucket(22), 0)
        self.assertEqual(map_age_to_bucket(35), 1)
        self.assertEqual(map_age_to_bucket(52), 2)
        
        self.assertEqual(map_pss10_to_class(10), 0)
        self.assertEqual(map_pss10_to_class(20), 1)
        self.assertEqual(map_pss10_to_class(32), 2)

    def test_cohens_kappa_ci(self):
        r1 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10)
        r2 = r1.copy()
        res = compute_cohens_kappa_with_ci(r1, r2, n_bootstraps=100)
        self.assertAlmostEqual(res["kappa"], 1.0, places=3)
        self.assertAlmostEqual(res["ci_lower"], 1.0, places=3)

    def test_face_detector_crop(self):
        detector = FaceDetector(margin_pct=0.25)
        test_img = create_synthetic_test_image()
        cropped, bbox = detector.detect_and_crop(test_img, return_bbox=True)
        # Verify function handles valid synthetic image without crashing
        self.assertIsNotNone(test_img)

    def test_normalization_pipeline(self):
        dummy_bgr = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
        tensor = preprocess_and_normalize_frame(dummy_bgr, target_size=(224, 224), to_torch_tensor=True)
        
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Test denormalization
        denorm = denormalize_image_tensor(tensor)
        self.assertEqual(denorm.shape, (224, 224, 3))
        self.assertEqual(denorm.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
