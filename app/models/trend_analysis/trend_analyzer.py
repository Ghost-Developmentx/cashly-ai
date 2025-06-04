"""
Trend analysis model using time series decomposition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sklearn.linear_model import LinearRegression
from scipy import stats

from app.models.base import BaseModel
from app.models.trend_analysis.feature_engineering import (
    TrendFeatureExtractor, SeasonalityExtractor
)

logger = logging.getLogger(__name__)

class TrendAnalyzer(BaseModel):
    """Model for analyzing financial trends and patterns."""

    def __init__(self, window_size: int = 30):
        super().__init__(
            model_name="trend_analyzer",
            model_type="custom"
        )
        self.window_size = window_size
        self.trend_extractor = TrendFeatureExtractor()
        self.seasonality_extractor = SeasonalityExtractor()
        self.trends = {}

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess transaction data for trend analysis."""
        required_cols = ['date', 'amount']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Aggregate daily
        daily = df.groupby('date').agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).fillna(0)

        daily.columns = ['daily_sum', 'transaction_count', 'avg_amount', 'std_amount']
        daily = daily.reset_index()

        return daily

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract trend features."""
        df = data.copy()

        # Extract trend features
        df = self.trend_extractor.transform(df)

        # Extract seasonality features
        df = self.seasonality_extractor.transform(df)

        # Store feature names
        self.feature_names = [
            col for col in df.columns
            if col not in ['date', 'daily_sum']
        ]

        return df

    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TrendAnalyzer":
        """Analyze trends in the data."""
        # For trend analysis, we don't train in traditional sense
        # Instead, we calculate and store trend components

        # Calculate overall trend
        self.trends['overall'] = self._calculate_trend(X['daily_sum'].values)

        # Calculate moving averages
        for window in [7, 30, 90]:
            self.trends[f'ma_{window}'] = X['daily_sum'].rolling(window).mean()

        # Detect change points
        self.trends['change_points'] = self._detect_change_points(X['daily_sum'])

        # Store model components
        self.model = {
            'trends': self.trends,
            'trend_extractor': self.trend_extractor,
            'seasonality_extractor': self.seasonality_extractor,
            'feature_names': self.feature_names,
            'date_range': (X['date'].min(), X['date'].max())
        }

        # Calculate trend strength as a metric
        self.metrics = {
            'trend_strength': abs(self.trends['overall']['slope']),
            'r_squared': self.trends['overall']['r2'],
            'volatility': X['daily_sum'].std() / X['daily_sum'].mean()
        }
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Project trend forward."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Simple linear projection
        last_value = X['daily_sum'].iloc[-1]
        slope = self.trends['overall']['slope']

        # Project forward
        n_days = len(X)
        projections = []

        for i in range(n_days):
            projection = last_value + (slope * (i + 1))
            projections.append(projection)

        return np.array(projections)

    @staticmethod
    def _calculate_trend(values: np.ndarray) -> Dict[str, float]:
        """Calculate linear trend."""
        x = np.arange(len(values))

        # Fit linear regression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), values)

        # Calculate metrics
        predictions = model.predict(x.reshape(-1, 1))
        r2 = model.score(x.reshape(-1, 1), values)

        return {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r2': float(r2),
            'direction': 'increasing' if model.coef_[0] > 0 else 'decreasing'
        }

    @staticmethod
    def _detect_change_points(series: pd.Series) -> List[int]:
        """Detect significant changes in trend."""
        # Simple change point detection using rolling statistics
        window = min(30, len(series) // 4)
        if window < 7:
            return []

        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()

        # Z-score based detection
        z_scores = abs((series - rolling_mean) / rolling_std)
        change_points = z_scores[z_scores > 2.5].index.tolist()

        return change_points

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate trend prediction accuracy."""
        predictions = self.predict(X)

        # Use only the overlapping period
        min_len = min(len(y), len(predictions))
        y_true = y[:min_len]
        y_pred = predictions[:min_len]

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)

        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            da = np.mean(true_direction == pred_direction)
        else:
            da = 0

        return {
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'directional_accuracy': float(da)
        }