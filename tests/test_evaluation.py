"""
tests/test_evaluation.py
Unit tests for evaluation metrics and statistical hypothesis testing.
"""

import unittest
import numpy as np

from evaluation.metrics import compute_all_metrics
from evaluation.statistical_tests import paired_wilcoxon_test, paired_bootstrap_confidence_interval


class TestEvaluation(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_probs = np.eye(3)[y_true]
        
        metrics = compute_all_metrics(y_true, y_pred, y_probs, y_pss10=y_true)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_auc_roc"], 1.0)
        self.assertEqual(metrics["cohen_kappa_pss10"], 1.0)

    def test_wilcoxon_test(self):
        prop_f1s = [0.85, 0.88, 0.84, 0.87, 0.89]
        base_f1s = [0.78, 0.80, 0.77, 0.81, 0.82]
        
        res = paired_wilcoxon_test(prop_f1s, base_f1s, alpha=0.05)
        self.assertTrue(res["is_significant"])
        self.assertGreater(res["mean_diff"], 0.0)

    def test_bootstrap_ci(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] * 20)
        y_prop = y_true.copy()
        y_base = y_true.copy()
        # Add slight errors to baseline
        y_base[:30] = np.random.choice([0, 1, 2], size=30)
        
        ci_res = paired_bootstrap_confidence_interval(
            y_true, y_prop, y_base, n_resamples=500, confidence_level=0.95
        )
        self.assertGreaterEqual(ci_res["point_estimate_delta_macro_f1"], 0.0)
        self.assertIn("ci_lower", ci_res)
        self.assertIn("ci_upper", ci_res)


if __name__ == "__main__":
    unittest.main()
