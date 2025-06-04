"""
Cash flow forecasting model using ensemble methods.
Integrates with MLflow for model management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import timedelta
import logging

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.models.base import BaseModel
from app.models.forecasting.feature_engineering import (
    TimeSeriesFeatureExtractor,
    CategoryFeatureExtractor
)
from app.utils.data_processing import prepare_timeseries_data


logger = logging.getLogger(__name__)

class CashFlowForecaster(BaseModel):
    """Ensemble model for cash flow forecasting."""

    def __init__(self, forecast_horizon: int = 30, method: str = "ensemble"):
        super().__init__(
            model_name=f"cash_flow_forecaster_{method}",
            model_type="sklearn"
        )
        self.forecast_horizon = forecast_horizon
        self.method = method
        self.scaler = StandardScaler()
        self.time_extractor = TimeSeriesFeatureExtractor()
        self.category_extractor = CategoryFeatureExtractor()
        self.models = {}
        self.ensemble_weights = {}

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess transaction data for forecasting."""
        required_cols = ['date', 'amount']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Group by date and sum amounts - ensure 'daily_sum' exists
        daily = df.groupby('date')['amount'].sum().reset_index()
        daily.columns = ['date', 'daily_sum']  # Explicitly name columns

        # Add other aggregations
        counts = df.groupby('date').size().reset_index(name='transaction_count')
        daily = daily.merge(counts, on='date', how='left')

        # Fill missing values
        daily = daily.fillna(0)

        return daily

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for forecasting."""
        # Time series features
        df = self.time_extractor.transform(data)

        # Category features if available
        if 'category' in df.columns:
            df = self.category_extractor.fit_transform(df)

        # Drop rows with NaN (from lag/rolling features)
        df = df.dropna()

        # Store feature names
        self.feature_names = [
            col for col in df.columns
            if col not in ['date', 'amount', 'category']
        ]

        return df

    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CashFlowForecaster":
        """Train the forecasting model."""
        # Extract target
        if y is None:
            y = X['amount']
            # Remove the date column before training
            feature_cols = [col for col in self.feature_names if col != 'date']
            X = X[feature_cols]

        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            X = X.drop(columns=datetime_cols)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train based on a method
        if self.method == "ensemble":
            self._train_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)
        else:
            self._train_single_model(X_train_scaled, y_train, X_test_scaled, y_test)

        # Store the composite model for MLflow
        self.model = {
            'models': self.models,
            'weights': self.ensemble_weights,
            'scaler': self.scaler,
            'time_extractor': self.time_extractor,
            'category_extractor': self.category_extractor,
            'feature_names': self.feature_names
        }

        return self

    def _train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble of models."""
        # Define models
        model_configs = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        # Train each model
        model_scores = {}
        for name, model in model_configs.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            model_scores[name] = 1 / (mae + 1e-8)  # Inverse MAE as score

            self.models[name] = model

            # Log individual model metrics
            self.metrics[f'{name}_mae'] = mae
            self.metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            self.metrics[f'{name}_r2'] = r2_score(y_test, y_pred)

        # Calculate ensemble weights (normalized scores)
        total_score = sum(model_scores.values())
        self.ensemble_weights = {
            name: score / total_score
            for name, score in model_scores.items()
        }

        # Evaluate ensemble
        ensemble_pred = self._ensemble_predict(X_test)
        self.metrics['ensemble_mae'] = mean_absolute_error(y_test, ensemble_pred)
        self.metrics['ensemble_rmse'] = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        self.metrics['ensemble_r2'] = r2_score(y_test, ensemble_pred)

    def _train_single_model(self, X_train, y_train, X_test, y_test):
        """Train a single model based on method."""
        model_map = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        model = model_map.get(self.method, LinearRegression())
        model.fit(X_train, y_train)

        self.models[self.method] = model
        self.ensemble_weights = {self.method: 1.0}

        # Evaluate
        y_pred = model.predict(X_test)
        self.metrics['mae'] = mean_absolute_error(y_test, y_pred)
        self.metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        self.metrics['r2'] = r2_score(y_test, y_pred)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Weighted average
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 0)
            ensemble_pred += weight * pred

        return ensemble_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Extract features if raw data provided
        if 'amount' in X.columns:
            X_features = X[self.feature_names]
        else:
            X_features = X

        # Scale features
        X_scaled = self.scaler.transform(X_features)

        # Make predictions
        if self.method == "ensemble":
            return self._ensemble_predict(X_scaled)
        else:
            return self.models[self.method].predict(X_scaled)

    def forecast(self, historical_data: pd.DataFrame, horizon: Optional[int] = None) -> pd.DataFrame:
        """Generate future predictions."""
        horizon = horizon or self.forecast_horizon

        # Preprocess and extract features
        processed_data = self.preprocess(historical_data)
        feature_data = self.extract_features(processed_data)

        # Get last date
        last_date = processed_data['date'].max()

        # Generate forecast
        forecast_data = self._generate_forecast(feature_data, last_date, horizon)

        return forecast_data

    def _generate_forecast(self, historical_features: pd.DataFrame,
                           last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Generate iterative forecast."""
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        forecasts = []
        recent_data = historical_features.copy()

        for date in forecast_dates:
            # Create feature row for prediction
            feature_row = self._create_forecast_features(recent_data, date)

            # Make prediction
            X = pd.DataFrame([feature_row])[self.feature_names]
            X_scaled = self.scaler.transform(X)

            if self.method == "ensemble":
                pred = self._ensemble_predict(X_scaled)[0]
            else:
                pred = self.models[self.method].predict(X_scaled)[0]

            # Store forecast
            forecasts.append({
                'date': date,
                'predicted_amount': pred,
                'lower_bound': pred * 0.8,  # Simple confidence interval
                'upper_bound': pred * 1.2
            })

            # Update recent data for next prediction
            new_row = feature_row.copy()
            new_row['amount'] = pred
            new_row['date'] = date
            recent_data = pd.concat([recent_data, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(forecasts)

    def _create_forecast_features(self, recent_data: pd.DataFrame,
                                  target_date: pd.Timestamp) -> Dict[str, Any]:
        """Create features for a future date."""
        features = {'day_of_week': target_date.dayofweek, 'day_of_month': target_date.day, 'month': target_date.month,
                    'quarter': target_date.quarter, 'year': target_date.year,
                    'is_weekend': int(target_date.dayofweek >= 5), 'is_month_start': int(target_date.is_month_start),
                    'is_month_end': int(target_date.is_month_end), 'days_in_month': target_date.days_in_month,
                    'day_of_week_sin': np.sin(2 * np.pi * target_date.dayofweek / 7),
                    'day_of_week_cos': np.cos(2 * np.pi * target_date.dayofweek / 7),
                    'month_sin': np.sin(2 * np.pi * target_date.month / 12),
                    'month_cos': np.cos(2 * np.pi * target_date.month / 12)}

        # Lag features from recent data
        amounts = recent_data['amount'].values
        for lag in self.time_extractor.lag_days:
            if len(amounts) >= lag:
                features[f'lag_{lag}'] = amounts[-lag]
            else:
                features[f'lag_{lag}'] = amounts[-1] if len(amounts) > 0 else 0

        # Rolling features
        for window in self.time_extractor.window_sizes:
            if len(amounts) >= window:
                window_data = amounts[-window:]
                features[f'rolling_mean_{window}'] = np.mean(window_data)
                features[f'rolling_std_{window}'] = np.std(window_data)
                features[f'rolling_min_{window}'] = np.min(window_data)
                features[f'rolling_max_{window}'] = np.max(window_data)
                features[f'rolling_median_{window}'] = np.median(window_data)
                features[f'rolling_skew_{window}'] = 0  # Simplified
            else:
                # Use available data
                features[f'rolling_mean_{window}'] = np.mean(amounts)
                features[f'rolling_std_{window}'] = np.std(amounts)
                features[f'rolling_min_{window}'] = np.min(amounts) if len(amounts) > 0 else 0
                features[f'rolling_max_{window}'] = np.max(amounts) if len(amounts) > 0 else 0
                features[f'rolling_median_{window}'] = np.median(amounts) if len(amounts) > 0 else 0
                features[f'rolling_skew_{window}'] = 0

        # Fill any missing features
        for feat in self.feature_names:
            if feat not in features:
                features[feat] = 0

        return features

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)

        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error
        }