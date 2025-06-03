"""
Feature engineering for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import List

from app.models.base import BaseTransformer

class TransactionFeatureExtractor(BaseTransformer):
    """Extracts transaction-level features for anomaly detection."""

    def __init__(self):
        self.amount_stats = {}
        self.category_stats = {}

    def fit(self, X: pd.DataFrame) -> "TransactionFeatureExtractor":
        """Calculate statistics for normalization."""
        if 'amount' in X.columns:
            # Calculate robust statistics
            self.amount_stats = {
                'median': X['amount'].median(),
                'mad': np.median(np.abs(X['amount'] - X['amount'].median())),
                'q1': X['amount'].quantile(0.25),
                'q3': X['amount'].quantile(0.75)
            }

        if 'category' in X.columns:
            # Category statistics
            for category in X['category'].unique():
                cat_data = X[X['category'] == category]['amount']
                self.category_stats[category] = {
                    'median': cat_data.median(),
                    'mad': np.median(np.abs(cat_data - cat_data.median()))
                }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract transaction features."""
        df = X.copy()

        # Amount features
        if 'amount' in df.columns:
            # Z-score using robust statistics
            if self.amount_stats.get('mad', 0) > 0:
                df['amount_zscore'] = (
                        (df['amount'] - self.amount_stats['median']) /
                        self.amount_stats['mad']
                )
            else:
                df['amount_zscore'] = 0

            # IQR-based features
            iqr = self.amount_stats['q3'] - self.amount_stats['q1']
            df['amount_iqr_score'] = np.where(
                df['amount'] < self.amount_stats['q1'] - 1.5 * iqr, -1,
                np.where(df['amount'] > self.amount_stats['q3'] + 1.5 * iqr, 1, 0)
            )

            # Log transform for skewed data
            df['amount_log'] = np.log1p(np.abs(df['amount']))

        # Time features
        if 'date' in df.columns:
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)

        # Category features
        if 'category' in df.columns and self.category_stats:
            df['category_deviation'] = df.apply(
                lambda row: self._calculate_category_deviation(row),
                axis=1
            )

        return df

    def _calculate_category_deviation(self, row):
        """Calculate deviation from category norm."""
        category = row.get('category')
        amount = row.get('amount', 0)

        if category in self.category_stats:
            stats = self.category_stats[category]
            if stats['mad'] > 0:
                return abs(amount - stats['median']) / stats['mad']

        return 0

class BehaviorFeatureExtractor(BaseTransformer):
    """Extracts behavioral pattern features."""

    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [1, 7, 30]

    def fit(self, X: pd.DataFrame) -> "BehaviorFeatureExtractor":
        """No fitting needed."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features."""
        df = X.copy()

        if 'date' not in df.columns:
            return df

        # Sort by date
        df = df.sort_values('date')

        # Transaction frequency features
        df['days_since_last'] = df['date'].diff().dt.days.fillna(0)

        # Daily transaction count
        daily_counts = df.groupby(df['date'].dt.date).size()
        df['daily_transaction_count'] = df['date'].dt.date.map(daily_counts)

        # Rolling window features
        for window in self.window_sizes:
            if len(df) >= window:
                # Transaction count in window
                df[f'trans_count_{window}d'] = df['date'].dt.date.rolling(
                    window, min_periods=1
                ).count()

                # Amount statistics in window
                if 'amount' in df.columns:
                    df[f'amount_mean_{window}d'] = df['amount'].rolling(
                        window, min_periods=1
                    ).mean()
                    df[f'amount_std_{window}d'] = df['amount'].rolling(
                        window, min_periods=1
                    ).std().fillna(0)

        # Velocity features (rate of change)
        if 'amount' in df.columns:
            df['amount_velocity'] = df['amount'].diff() / (df['days_since_last'] + 1)
            df['amount_acceleration'] = df['amount_velocity'].diff()

        # Pattern breaking features
        if len(df) > 7:
            # Check if transaction breaks weekly pattern
            df['breaks_weekly_pattern'] = self._detect_pattern_break(
                df, period=7
            )

        return df

    @staticmethod
    def _detect_pattern_break(df: pd.DataFrame, period: int) -> pd.Series:
        """Detect if transaction breaks established pattern."""
        pattern_breaks = pd.Series(0, index=df.index)

        if 'amount' not in df.columns or len(df) < period * 2:
            return pattern_breaks

        # Look for transactions that deviate significantly from pattern
        for i in range(period, len(df)):
            # Get same day of week from previous periods
            current_dow = df.iloc[i]['day_of_week']

            # Find previous transactions on same day of week
            mask = (df.index < i) & (df['day_of_week'] == current_dow)
            similar_transactions = df.loc[mask, 'amount']

            if len(similar_transactions) >= 2:
                expected = similar_transactions.mean()
                std = similar_transactions.std()

                if std > 0:
                    z_score = abs(df.iloc[i]['amount'] - expected) / std
                    if z_score > 2:
                        pattern_breaks.iloc[i] = 1

        return pattern_breaks
