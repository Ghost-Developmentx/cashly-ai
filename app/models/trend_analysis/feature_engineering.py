"""
Feature engineering for trend analysis.
"""

import pandas as pd
import numpy as np
from typing import List
from scipy import signal

from app.models.base import BaseTransformer

class TrendFeatureExtractor(BaseTransformer):
    """Extracts trend-related features."""

    def __init__(self, periods: List[int] = None):
        self.periods = periods or [7, 14, 30, 90]

    def fit(self, X: pd.DataFrame) -> "TrendFeatureExtractor":
        """No fitting needed."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract trend features."""
        df = X.copy()

        if 'daily_sum' not in df.columns:
            return df

        # Moving averages
        for period in self.periods:
            if len(df) >= period:
                df[f'ma_{period}'] = df['daily_sum'].rolling(period).mean()
                df[f'ma_{period}_ratio'] = df['daily_sum'] / df[f'ma_{period}']

        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'ema_{alpha}'] = df['daily_sum'].ewm(alpha=alpha).mean()

        # Trend strength indicators
        if len(df) >= 30:
            # Linear trend over different windows
            for window in [7, 14, 30]:
                if len(df) >= window:
                    df[f'trend_slope_{window}'] = self._rolling_slope(
                        df['daily_sum'], window
                    )

        # Momentum indicators
        for lag in [1, 7, 30]:
            if len(df) > lag:
                df[f'momentum_{lag}'] = df['daily_sum'] - df['daily_sum'].shift(lag)
                df[f'momentum_pct_{lag}'] = df['daily_sum'].pct_change(lag)

        return df

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear regression slope."""
        def slope(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            coef = np.polyfit(x, values, 1)
            return coef[0]

        return series.rolling(window).apply(slope, raw=True)

class SeasonalityExtractor(BaseTransformer):
    """Extracts seasonality features."""

    def fit(self, X: pd.DataFrame) -> "SeasonalityExtractor":
        """No fitting needed."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract seasonality features."""
        df = X.copy()

        if 'date' not in df.columns:
            return df

        # Calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_month'] = (df['date'].dt.day - 1) // 7 + 1
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Holiday indicators (simplified)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['day_of_month'] <= 3
        df['is_month_end'] = df['day_of_month'] >= 28

        # Seasonal decomposition if enough data
        if len(df) >= 60:
            try:
                # Simple detrending
                detrended = signal.detrend(df['daily_sum'].fillna(0))
                df['detrended'] = detrended
            except:
                df['detrended'] = 0

        return df