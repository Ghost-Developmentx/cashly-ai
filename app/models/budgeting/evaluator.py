"""
Evaluation metrics for budget recommendation models.
"""

import numpy as np
from typing import Dict

from app.models.base import BaseEvaluator

class BudgetEvaluator(BaseEvaluator):
    """Evaluator for budget recommendation models."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate budget recommendations."""
        # For budget recommendations, we evaluate allocation accuracy
        # y_true and y_pred should be allocation percentages

        # Mean Absolute Error for allocations
        mae = np.mean(np.abs(y_true - y_pred))

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Allocation balance (how close to 100%)
        allocation_sum = np.sum(y_pred)
        balance_error = abs(1.0 - allocation_sum)

        return {
            'allocation_mae': float(mae),
            'allocation_rmse': float(rmse),
            'balance_error': float(balance_error),
            'total_allocation': float(allocation_sum)
        }

    @staticmethod
    def evaluate_budget_health(
            actual_spending: Dict[str, float],
            recommended_budget: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate financial health based on budget adherence."""
        metrics = {}

        # Overall adherence
        total_actual = sum(actual_spending.values())
        total_budget = sum(recommended_budget.values())

        if total_budget > 0:
            metrics['overall_adherence'] = min(total_actual / total_budget, 2.0)
        else:
            metrics['overall_adherence'] = 0

        # Category-wise adherence
        category_adherence = []
        over_budget_categories = 0

        for category, budget in recommended_budget.items():
            actual = actual_spending.get(category, 0)
            if budget > 0:
                adherence = actual / budget
                category_adherence.append(adherence)
                if adherence > 1.0:
                    over_budget_categories += 1

        metrics['avg_category_adherence'] = (
            np.mean(category_adherence) if category_adherence else 0
        )
        metrics['over_budget_categories'] = over_budget_categories
        metrics['categories_on_track'] = len(category_adherence) - over_budget_categories

        # Savings rate achievement
        if 'savings' in recommended_budget:
            recommended_savings_rate = recommended_budget['savings'] / total_budget
            actual_savings = total_budget - total_actual
            actual_savings_rate = actual_savings / total_budget if total_budget > 0 else 0

            metrics['savings_rate_achievement'] = (
                actual_savings_rate / recommended_savings_rate
                if recommended_savings_rate > 0 else 0
            )

        return metrics

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate budget evaluation report."""
        report = "=" * 50 + "\n"
        report += "BUDGET RECOMMENDATION EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        if 'allocation_mae' in metrics:
            report += "Allocation Accuracy:\n"
            report += f"  MAE: {metrics.get('allocation_mae', 0):.3f}\n"
            report += f"  RMSE: {metrics.get('allocation_rmse', 0):.3f}\n"
            report += f"  Balance Error: {metrics.get('balance_error', 0):.3f}\n\n"

        if 'overall_adherence' in metrics:
            report += "Budget Adherence:\n"
            report += f"  Overall: {metrics.get('overall_adherence', 0):.1%}\n"
            report += f"  Category Average: {metrics.get('avg_category_adherence', 0):.1%}\n"
            report += f"  Categories on Track: {metrics.get('categories_on_track', 0)}\n"
            report += f"  Over Budget: {metrics.get('over_budget_categories', 0)}\n"

        if 'savings_rate_achievement' in metrics:
            report += f"\nSavings Achievement: {metrics.get('savings_rate_achievement', 0):.1%}\n"

        return report