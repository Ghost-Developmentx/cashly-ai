"""
Feature engineering for budget recommendation.
"""

import pandas as pd
import numpy as np
from typing import List

from app.models.base import BaseTransformer

class SpendingFeatureExtractor(BaseTransformer):
    """Extracts spending pattern features."""

    def fit(self, X: pd.DataFrame) -> "SpendingFeatureExtractor":
        """Fit to spending data."""
        if 'category' in X.columns:
            self.categories = X['category'].unique().tolist()
        else:
            self.categories = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract spending features."""
        df = X.copy()

        if len(df) == 0:
            return df

        # Monthly aggregation
        df['year_month'] = df['date'].dt.to_period('M')

        # Category spending patterns
        category_features = df.groupby(['category', 'year_month'])['amount'].agg([
            'sum', 'mean', 'count', 'std'
        ]).reset_index()

        # Overall spending features
        features = pd.DataFrame()

        # Spending volatility by category
        for category in df['category'].unique():
            cat_data = category_features[category_features['category'] == category]
            if len(cat_data) > 1:
                features[f'{category}_volatility'] = [cat_data['sum'].std()]
                features[f'{category}_trend'] = [
                    np.polyfit(range(len(cat_data)), cat_data['sum'], 1)[0]
                    if len(cat_data) > 2 else 0
                ]

        # Spending concentration
        total_by_category = df.groupby('category')['amount'].sum()
        total_spending = total_by_category.sum()

        if total_spending > 0:
            concentration = sum(
                (cat_total / total_spending) ** 2
                for cat_total in total_by_category
            )
            features['spending_concentration'] = [concentration]

        # Temporal patterns
        features['weekend_spending_ratio'] = [
            df[df['date'].dt.weekday >= 5]['amount'].sum() /
            (df['amount'].sum() + 1e-8)
        ]

        return features

class IncomeFeatureExtractor(BaseTransformer):
    """Extracts income pattern features."""

    def fit(self, X: pd.DataFrame) -> "IncomeFeatureExtractor":
        """No fitting needed."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract income features."""
        df = X.copy()

        if len(df) == 0:
            return pd.DataFrame()

        features = pd.DataFrame()

        # Income stability
        monthly_income = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()

        if len(monthly_income) > 1:
            features['income_stability'] = [
                1 - (monthly_income.std() / (monthly_income.mean() + 1e-8))
            ]
            features['income_trend'] = [
                np.polyfit(range(len(monthly_income)), monthly_income.values, 1)[0]
                if len(monthly_income) > 2 else 0
            ]
        else:
            features['income_stability'] = [1.0]
            features['income_trend'] = [0.0]

        # Income frequency
        days_between = df['date'].diff().dt.days.dropna()
        if len(days_between) > 0:
            features['income_frequency'] = [days_between.mode()[0] if len(days_between) > 0 else 30]
        else:
            features['income_frequency'] = [30]

        # Income sources (simplified)
        features['num_income_sources'] = [df['category'].nunique() if 'category' in df.columns else 1]

        return features