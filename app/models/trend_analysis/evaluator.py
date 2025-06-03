"""
Evaluation metrics for trend analysis models.
"""

import numpy as np
from typing import Dict, List
import pandas as pd

from app.models.base import BaseEvaluator

class TrendEvaluator(BaseEvaluator):
    """Evaluator for trend analysis models."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate trend predictions."""
        # Trend accuracy metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Trend direction accuracy
        if len(y_true) > 1:
            true_changes = np.diff(y_true)
            pred_changes = np.diff(y_pred)
            direction_accuracy = np.mean(
                (true_changes > 0) == (pred_changes > 0)
            )
        else:
            direction_accuracy = 0

        # Trend correlation
        if len(y_true) > 2:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            correlation = 0

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_accuracy),
            'correlation': float(correlation),
            'mean_error': float(np.mean(y_pred - y_true))
        }

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate trend analysis report."""
        report = "=" * 50 + "\n"
        report += "TREND ANALYSIS EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        report += "Prediction Accuracy:\n"
        report += f"  MAE: {metrics.get('mae', 0):.2f}\n"
        report += f"  RMSE: {metrics.get('rmse', 0):.2f}\n"
        report += f"  Mean Error: {metrics.get('mean_error', 0):.2f}\n\n"

        report += "Trend Metrics:\n"
        report += f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.1%}\n"
        report += f"  Correlation: {metrics.get('correlation', 0):.3f}\n"

        return report