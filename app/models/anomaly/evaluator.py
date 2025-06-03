"""
Evaluation metrics for anomaly detection models.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve

from app.models.base import BaseEvaluator

class AnomalyEvaluator(BaseEvaluator):
    """Evaluator for anomaly detection models."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detection performance."""
        # Convert to binary (1 for anomaly, 0 for normal)
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)

        # Basic metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'specificity': float(specificity),
            'true_positive_rate': float(recall),
            'false_positive_rate': float(1 - specificity),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }

    def evaluate_with_scores(
            self, y_true: np.ndarray, scores: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate using anomaly scores for threshold-independent metrics."""
        # Convert to binary
        y_true_binary = (y_true == -1).astype(int)

        # Invert scores so higher means more anomalous
        anomaly_scores = -scores

        # Calculate AUC-ROC if we have both classes
        if len(np.unique(y_true_binary)) > 1:
            auc_roc = roc_auc_score(y_true_binary, anomaly_scores)
        else:
            auc_roc = 0.5

        # Find optimal threshold
        optimal_threshold, optimal_f1 = self._find_optimal_threshold(
            y_true_binary, anomaly_scores
        )

        return {
            'auc_roc': float(auc_roc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_f1': float(optimal_f1)
        }

    @staticmethod
    def _find_optimal_threshold(
            y_true: np.ndarray, scores: np.ndarray
    ) -> Tuple[float, float]:
        """Find optimal threshold based on F1 score."""
        # Try different percentiles as thresholds
        percentiles = np.percentile(scores, np.arange(1, 20))

        best_f1 = 0
        best_threshold = percentiles[0]

        for threshold in percentiles:
            y_pred = (scores > threshold).astype(int)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate anomaly detection report."""
        report = "=" * 50 + "\n"
        report += "ANOMALY DETECTION EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        report += "Detection Performance:\n"
        report += f"  Precision: {metrics.get('precision', 0):.3f}\n"
        report += f"  Recall: {metrics.get('recall', 0):.3f}\n"
        report += f"  F1 Score: {metrics.get('f1_score', 0):.3f}\n\n"

        report += "Classification Metrics:\n"
        report += f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n"
        report += f"  Specificity: {metrics.get('specificity', 0):.3f}\n"

        if 'auc_roc' in metrics:
            report += f"  AUC-ROC: {metrics.get('auc_roc', 0):.3f}\n"

        report += "\nConfusion Matrix:\n"
        report += f"  True Positives: {metrics.get('true_positives', 0)}\n"
        report += f"  False Positives: {metrics.get('false_positives', 0)}\n"
        report += f"  False Negatives: {metrics.get('false_negatives', 0)}\n"
        report += f"  True Negatives: {metrics.get('true_negatives', 0)}\n"

        return report