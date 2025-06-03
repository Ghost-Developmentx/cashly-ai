"""
Feature engineering for cash flow forecasting.
Extracts time-series features from transaction data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from app.models.base import BaseTransformer

class TimeSeriesFeatureExtractor(BaseTransformer):
    """Extracts time-series features for forecasting."""

    def __init__(self, lag_days: List[int] = None, window_sizes: List[int] = None):
        self.lag_days = lag_days or [1, 2, 3, 7, 14, 30]
        self.window_sizes = window_sizes or [3, 7, 14, 30]
        self.feature_names = []

    def fit(self, X: pd.DataFrame) -> "TimeSeriesFeatureExtractor":
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract time-series features."""
        df = X.copy()

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Calendar features
        df = self._add_calendar_features(df)

        # Lag features
        df = self._add_lag_features(df)

        # Rolling window features
        df = self._add_rolling_features(df)

        # Trend features
        df = self._add_trend_features(df)

        # Store feature names
        self.feature_names = [col for col in df.columns
                              if col not in ['date', 'amount', 'category']]

        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        if 'date' not in df.columns:
            return df

        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['days_in_month'] = df['date'].dt.days_in_month

        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features."""
        if 'amount' not in df.columns:
            return df

        for lag in self.lag_days:
            df[f'lag_{lag}'] = df['amount'].shift(lag)

            # Lag differences
            if lag > 1:
                df[f'lag_diff_{lag}'] = df['amount'] - df['amount'].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        if 'amount' not in df.columns:
            return df

        for window in self.window_sizes:
            # Basic statistics
            df[f'rolling_mean_{window}'] = df['amount'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['amount'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['amount'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['amount'].rolling(window=window).max()

            # More advanced statistics
            df[f'rolling_median_{window}'] = df['amount'].rolling(window=window).median()
            df[f'rolling_skew_{window}'] = df['amount'].rolling(window=window).skew()

            # Normalized values
            rolling_mean = df[f'rolling_mean_{window}']
            df[f'amount_vs_rolling_{window}'] = (
                    (df['amount'] - rolling_mean) / (rolling_mean + 1e-8)
            )

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features."""
        if len(df) < 7:
            return df

        # Simple moving average crossover
        if 'rolling_mean_7' in df.columns and 'rolling_mean_30' in df.columns:
            df['sma_crossover'] = (
                    df['rolling_mean_7'] > df['rolling_mean_30']
            ).astype(int)

        # Trend strength (using linear regression coefficient)
        df['trend_7d'] = self._calculate_trend(df['amount'], 7)
        df['trend_30d'] = self._calculate_trend(df['amount'], 30)

        return df

    @staticmethod
    def _calculate_trend(series: pd.Series, window: int) -> pd.Series:
        """Calculate trend using rolling linear regression."""
        def _trend(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0]

        return series.rolling(window=window).apply(_trend, raw=True)

class CategoryFeatureExtractor(BaseTransformer):
    """Extracts category-based features for forecasting."""

    def __init__(self):
        self.category_stats = {}

    def fit(self, X: pd.DataFrame) -> "CategoryFeatureExtractor":
        """Fit by calculating category statistics."""
        if 'category' in X.columns and 'amount' in X.columns:
            self.category_stats = (
                X.groupby('category')['amount']
                .agg(['mean', 'std', 'count'])
                .to_dict('index')
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add category-based features."""
        df = X.copy()

        if 'category' not in df.columns:
            return df

        # Category frequency encoding
        category_counts = df['category'].value_counts().to_dict()
        df['category_frequency'] = df['category'].map(category_counts)

        # Category statistics
        if self.category_stats:
            df['category_mean'] = df['category'].map(
                lambda x: self.category_stats.get(x, {}).get('mean', 0)
            )
            df['category_std'] = df['category'].map(
                lambda x: self.category_stats.get(x, {}).get('std', 0)
            )

            # Deviation from category mean
            df['amount_vs_category_mean'] = (
                                                    df['amount'] - df['category_mean']
                                            ) / (df['category_std'] + 1e-8)

        return df
