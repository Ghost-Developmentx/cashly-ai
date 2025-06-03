"""
Evaluation metrics and reporting for forecast models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from numpy import floating

from app.models.base import BaseEvaluator

class ForecastEvaluator(BaseEvaluator):
    """Evaluator for time series forecasting models."""

    def __init__(self):
        self.metrics_history = []

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive forecast metrics."""
        # Basic metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Percentage errors
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            da = np.mean(true_direction == pred_direction) * 100
        else:
            da = 0

        # Symmetric MAPE
        smape = np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
        ) * 100

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'directional_accuracy': da,
            'r2': self._calculate_r2(y_true, y_pred)
        }

        # Store for tracking
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        return metrics

    def evaluate_by_horizon(self, y_true: np.ndarray, y_pred: np.ndarray,
                            horizons: List[int] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate metrics at different forecast horizons."""
        if horizons is None:
            horizons = [1, 7, 14, 30]

        horizon_metrics = {}

        for h in horizons:
            if h <= len(y_true):
                metrics = self.evaluate(y_true[:h], y_pred[:h])
                horizon_metrics[f'horizon_{h}'] = metrics

        return horizon_metrics

    def evaluate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze forecast residuals."""
        residuals = y_true - y_pred

        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': self._calculate_skewness(residuals),
            'kurtosis': self._calculate_kurtosis(residuals),
            'autocorrelation': self._calculate_autocorrelation(residuals),
            'ljung_box_p': self._ljung_box_test(residuals)
        }

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate a text report of evaluation metrics."""
        report = "=" * 50 + "\n"
        report += "FORECAST EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        report += "Error Metrics:\n"
        report += f"  MAE:  {metrics.get('mae', 0):.2f}\n"
        report += f"  RMSE: {metrics.get('rmse', 0):.2f}\n"
        report += f"  MAPE: {metrics.get('mape', 0):.1f}%\n"
        report += f"  SMAPE: {metrics.get('smape', 0):.1f}%\n\n"

        report += "Performance Metrics:\n"
        report += f"  R²: {metrics.get('r2', 0):.3f}\n"
        report += f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.1f}%\n"

        return report

    def plot_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                        dates: Optional[pd.DatetimeIndex] = None) -> plt.Figure:
        """Create evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time series plot
        ax = axes[0, 0]
        x = dates if dates is not None else range(len(y_true))
        ax.plot(x, y_true, label='Actual', alpha=0.7)
        ax.plot(x, y_pred, label='Predicted', alpha=0.7)
        ax.set_title('Actual vs Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

        # Scatter plot
        ax = axes[0, 1]
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_title('Prediction Scatter Plot')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        # Residuals plot
        ax = axes[1, 0]
        residuals = y_true - y_pred
        x = dates if dates is not None else range(len(residuals))
        ax.plot(x, residuals, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Residuals Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual')

        # Residuals histogram
        ax = axes[1, 1]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title('Residuals Distribution')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> int | floating[Any]:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    @staticmethod
    def _calculate_autocorrelation(data: np.ndarray, lag: int = 1) -> int | np.ndarray[tuple[int, ...], np.dtype[floating]]:
        """Calculate autocorrelation at given lag."""
        if len(data) <= lag:
            return 0
        return np.corrcoef(data[:-lag], data[lag:])[0, 1]

    @staticmethod
    def _ljung_box_test(residuals: np.ndarray, lags: int = 10) -> float:
        """Simplified Ljung-Box test (returns p-value)."""
        # This is a simplified version
        # In production, use statsmodels.stats.diagnostic.acorr_ljungbox
        n = len(residuals)
        if n <= lags:
            return 1.0

        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, lags + 1):
            if lag < n:
                ac = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrs.append(ac)

        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum(
            [(ac**2) / (n - k) for k, ac in enumerate(autocorrs, 1)]
        )

        # Approximate p-value (chi-square distribution)
        # In practice, use scipy.stats.chi2
        return 1.0 if lb_stat < lags else 0.0