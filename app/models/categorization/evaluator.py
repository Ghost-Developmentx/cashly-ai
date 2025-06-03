"""
Evaluation metrics for categorization models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from app.models.base import BaseEvaluator

class CategorizationEvaluator(BaseEvaluator):
    """Evaluator for transaction categorization models."""

    def __init__(self):
        self.metrics_history = []
        self.category_performance = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate categorization metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Macro averages for balanced view
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_macro': float(macro_prec),
            'recall_macro': float(macro_rec),
            'f1_macro': float(macro_f1),
            'total_samples': len(y_true)
        }

        # Store for history
        self.metrics_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics
        })

        return metrics

    def evaluate_per_category(
            self, y_true: np.ndarray, y_pred: np.ndarray,
            categories: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance per category."""
        report = classification_report(
            y_true, y_pred,
            target_names=categories,
            output_dict=True,
            zero_division=0
        )

        # Extract per-category metrics
        category_metrics = {}
        for category in categories:
            if category in report:
                category_metrics[category] = {
                    'precision': report[category]['precision'],
                    'recall': report[category]['recall'],
                    'f1_score': report[category]['f1-score'],
                    'support': report[category]['support']
                }

        self.category_performance = category_metrics
        return category_metrics

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate evaluation report."""
        report = "=" * 50 + "\n"
        report += "CATEGORIZATION EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        report += "Overall Performance:\n"
        report += f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n"
        report += f"  F1 Score (weighted): {metrics.get('f1_weighted', 0):.3f}\n"
        report += f"  F1 Score (macro): {metrics.get('f1_macro', 0):.3f}\n\n"

        report += "Detailed Metrics:\n"
        report += f"  Precision (weighted): {metrics.get('precision_weighted', 0):.3f}\n"
        report += f"  Recall (weighted): {metrics.get('recall_weighted', 0):.3f}\n"
        report += f"  Total Samples: {metrics.get('total_samples', 0)}\n"

        if self.category_performance:
            report += "\nPer-Category Performance:\n"
            for cat, perf in self.category_performance.items():
                report += f"\n  {cat}:\n"
                report += f"    Precision: {perf['precision']:.3f}\n"
                report += f"    Recall: {perf['recall']:.3f}\n"
                report += f"    F1 Score: {perf['f1_score']:.3f}\n"

        return report

    @staticmethod
    def plot_confusion_matrix(
            y_true: np.ndarray, y_pred: np.ndarray,
            categories: List[str]
    ) -> Dict[str, Any]:
        """Generate confusion matrix data for plotting."""
        cm = confusion_matrix(y_true, y_pred)

        return {
            'matrix': cm.tolist(),
            'categories': categories,
            'normalized': (cm / cm.sum(axis=1)[:, np.newaxis]).tolist()
        }