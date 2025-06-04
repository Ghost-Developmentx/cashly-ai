"""
Budget recommendation model using clustering and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from app.models.base import BaseModel
from app.models.budgeting.feature_engineering import (
    SpendingFeatureExtractor, IncomeFeatureExtractor
)

logger = logging.getLogger(__name__)

class BudgetRecommender(BaseModel):
    """Model for generating personalized budget recommendations."""

    def __init__(self, allocation_method: str = "50-30-20"):
        super().__init__(
            model_name="budget_recommender",
            model_type="custom"
        )
        self.allocation_method = allocation_method
        self.spending_extractor = SpendingFeatureExtractor()
        self.income_extractor = IncomeFeatureExtractor()
        self.scaler = StandardScaler()
        self.category_clusters = {}
        self.optimal_allocations = {}

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess transaction data for budgeting."""
        required_cols = ['date', 'amount', 'category']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Separate income and expenses
        self.income_df = df[df['amount'] > 0].copy()
        self.expense_df = df[df['amount'] < 0].copy()
        self.expense_df['amount'] = self.expense_df['amount'].abs()

        return df

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract budgeting features."""
        # Extract spending features
        if len(self.expense_df) > 0:
            expense_features = self.spending_extractor.transform(self.expense_df)
        else:
            expense_features = pd.DataFrame()

        # Extract income features
        if len(self.income_df) > 0:
            income_features = self.income_extractor.transform(self.income_df)
        else:
            income_features = pd.DataFrame()

        # Combine features
        features = {
            'total_income': self.income_df['amount'].sum() if len(self.income_df) > 0 else 0,
            'total_expenses': self.expense_df['amount'].sum() if len(self.expense_df) > 0 else 0,
            'num_categories': self.expense_df['category'].nunique() if len(self.expense_df) > 0 else 0,
            'expense_volatility': self.expense_df['amount'].std() if len(self.expense_df) > 1 else 0
        }

        return pd.DataFrame([features])

    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BudgetRecommender":
        """Train the budget recommendation model."""
        # Cluster spending categories
        if len(self.expense_df) > 0:
            self._cluster_categories()

        # Calculate optimal allocations
        self._calculate_optimal_allocations()

        # Store model components
        self.model = {
            'category_clusters': self.category_clusters,
            'optimal_allocations': self.optimal_allocations,
            'spending_extractor': self.spending_extractor,
            'income_extractor': self.income_extractor,
            'allocation_method': self.allocation_method
        }

        # Calculate metrics
        self.metrics = self._calculate_budget_metrics()

        self.is_fitted = True
        return self

    def _cluster_categories(self):
        """Cluster spending categories based on patterns."""
        # Aggregate by category
        category_stats = self.expense_df.groupby('category').agg({
            'amount': ['sum', 'mean', 'std', 'count']
        }).fillna(0)

        category_stats.columns = ['total', 'mean', 'std', 'count']

        if len(category_stats) >= 3:
            # Normalize features
            features_scaled = self.scaler.fit_transform(category_stats)

            # Cluster categories
            n_clusters = min(3, len(category_stats))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)

            # Assign clusters
            for i, category in enumerate(category_stats.index):
                self.category_clusters[category] = {
                    'cluster': int(clusters[i]),
                    'total_spending': float(category_stats.loc[category, 'total']),
                    'avg_transaction': float(category_stats.loc[category, 'mean'])
                }

    def _calculate_optimal_allocations(self):
        """Calculate optimal budget allocations."""
        total_income = self.income_df['amount'].sum() if len(self.income_df) > 0 else 0
        total_expenses = self.expense_df['amount'].sum() if len(self.expense_df) > 0 else 0

        if total_income == 0:
            return

        # Apply allocation method
        if self.allocation_method == "50-30-20":
            self.optimal_allocations = {
                'needs': 0.50 * total_income,
                'wants': 0.30 * total_income,
                'savings': 0.20 * total_income
            }
        elif self.allocation_method == "zero-based":
            # Optimize allocations based on current spending
            self._optimize_allocations(total_income, total_expenses)

        # Map categories to allocation types
        self._map_categories_to_types()

    def _optimize_allocations(self, income: float, expenses: float):
        """Optimize budget allocations using linear programming."""
        # Simple optimization: minimize variance while meeting constraints
        def objective(allocations):
            needs, wants, savings = allocations
            target_savings = 0.20 * income
            variance = (savings - target_savings) ** 2
            return variance

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: sum(x) - income},  # Total = income
            {'type': 'ineq', 'fun': lambda x: x[0] - 0.3 * income},  # Min needs
            {'type': 'ineq', 'fun': lambda x: x[2] - 0.1 * income},  # Min savings
        ]

        # Bounds
        bounds = [(0, income) for _ in range(3)]

        # Initial guess
        x0 = [0.5 * income, 0.3 * income, 0.2 * income]

        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            self.optimal_allocations = {
                'needs': result.x[0],
                'wants': result.x[1],
                'savings': result.x[2]
            }

    def _map_categories_to_types(self):
        """Map spending categories to budget types."""
        # Define essential categories
        essential_categories = [
            'housing', 'utilities', 'groceries', 'transportation',
            'healthcare', 'insurance', 'debt'
        ]

        for category, info in self.category_clusters.items():
            if any(essential in category.lower() for essential in essential_categories):
                info['budget_type'] = 'needs'
            elif info['total_spending'] < self.optimal_allocations.get('wants', 0) * 0.1:
                info['budget_type'] = 'wants'
            else:
                info['budget_type'] = 'needs'

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate budget recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Return optimal allocation percentages
        total = sum(self.optimal_allocations.values())
        if total > 0:
            return np.array([
                self.optimal_allocations.get('needs', 0) / total,
                self.optimal_allocations.get('wants', 0) / total,
                self.optimal_allocations.get('savings', 0) / total
            ])
        return np.array([0.5, 0.3, 0.2])

    def generate_recommendations(self, monthly_income: float) -> Dict[str, Any]:
        """Generate detailed budget recommendations."""
        recommendations = {
            'monthly_income': monthly_income,
            'allocations': {},
            'category_budgets': {},
            'savings_goal': monthly_income * 0.2,
            'adjustments': []
        }

        # Calculate allocations
        for budget_type, percentage in zip(['needs', 'wants', 'savings'], self.predict(None)):
            recommendations['allocations'][budget_type] = {
                'amount': monthly_income * percentage,
                'percentage': percentage * 100
            }

        # Category-specific budgets
        for category, info in self.category_clusters.items():
            budget_type = info.get('budget_type', 'wants')
            type_budget = recommendations['allocations'][budget_type]['amount']

            # Allocate proportionally within type
            category_weight = info['total_spending'] / sum(
                c['total_spending'] for c in self.category_clusters.values()
                if c.get('budget_type') == budget_type
            )

            recommendations['category_budgets'][category] = {
                'amount': type_budget * category_weight,
                'current_spending': info['total_spending'],
                'adjustment_needed': (type_budget * category_weight) - info['total_spending']
            }

        return recommendations

    def _calculate_budget_metrics(self) -> Dict[str, float]:
        """Calculate budget performance metrics."""
        total_income = self.income_df['amount'].sum() if len(self.income_df) > 0 else 0
        total_expenses = self.expense_df['amount'].sum() if len(self.expense_df) > 0 else 0

        if total_income > 0:
            savings_rate = (total_income - total_expenses) / total_income
            expense_ratio = total_expenses / total_income
        else:
            savings_rate = 0
            expense_ratio = 0

        # Calculate number of categories
        num_categories = self.expense_df['category'].nunique() if len(self.expense_df) > 0 else 0

        # Calculate total monthly spend (assuming data covers a period, normalize to monthly)
        if len(self.expense_df) > 0:
            # Get date range to calculate monthly average
            date_range = (self.expense_df['date'].max() - self.expense_df['date'].min()).days
            months = max(1, date_range / 30.44)  # Average days per month
            total_monthly_spend = total_expenses / months
        else:
            total_monthly_spend = 0

        return {
            'savings_rate': float(savings_rate),
            'expense_ratio': float(expense_ratio),
            'budget_variance': float(
                np.std([c['total_spending'] for c in self.category_clusters.values()])
                if self.category_clusters else 0
            ),
            'num_categories': int(num_categories),
            'total_monthly_spend': float(total_monthly_spend)
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate budget recommendations."""
        # This is a recommender system, so traditional evaluation doesn't apply
        # Instead, we measure budget adherence and financial health
        return self.metrics