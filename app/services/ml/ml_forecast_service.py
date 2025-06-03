"""
ML-powered forecasting service using new MLflow models.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta

from app.services.ml.model_manager import model_manager

logger = logging.getLogger(__name__)

class MLForecastService:
    """Service for ML-based cash flow forecasting."""

    def __init__(self):
        self.min_training_samples = 30

    async def forecast_with_ml(
            self,
            transactions: List[Dict[str, Any]],
            forecast_days: int = 30,
            method: str = "ensemble"
    ) -> Dict[str, Any]:
        """Generate ML-based forecast."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(transactions)

            if len(df) < self.min_training_samples:
                logger.warning(f"Insufficient data for ML forecast: {len(df)} samples")
                return self._fallback_forecast(df, forecast_days)

            # Get or train model
            forecaster = await self._get_or_train_forecaster(df, method)

            # Generate forecast
            forecast_df = forecaster.forecast(df, horizon=forecast_days)

            # Format response
            return self._format_forecast_response(forecast_df, forecaster)

        except Exception as e:
            logger.error(f"ML forecast failed: {e}")
            return self._error_response(str(e))

    async def _get_or_train_forecaster(
            self,
            df: pd.DataFrame,
            method: str
    ) -> Any:
        """Get existing model or train new one."""
        # Try to get existing model
        forecaster = await model_manager.get_model('forecaster')

        # Check if model needs retraining
        if self._should_retrain(forecaster, df):
            logger.info("Retraining forecaster model")
            result = await model_manager.train_model(
                'forecaster',
                df,
                method=method
            )

            if result['success']:
                forecaster = await model_manager.get_model('forecaster', force_reload=True)
            else:
                raise ValueError(f"Failed to train model: {result.get('error')}")

        return forecaster

    @staticmethod
    def _should_retrain(model: Any, new_data: pd.DataFrame) -> bool:
        """Determine if model should be retrained."""
        # Retrain if no model exists
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            return True

        # Retrain if significant new data
        if hasattr(model, 'model') and 'date_range' in model.model:
            last_date = model.model['date_range'][1]
            new_dates = pd.to_datetime(new_data['date'])

            # If we have 30+ days of new data, retrain
            days_of_new_data = (new_dates.max() - last_date).days
            if days_of_new_data > 30:
                return True

        return False

    @staticmethod
    def _format_forecast_response(
            forecast_df: pd.DataFrame,
            model: Any
    ) -> Dict[str, Any]:
        """Format forecast dataframe to response."""
        daily_predictions = []

        for _, row in forecast_df.iterrows():
            daily_predictions.append({
                'date': row['date'].isoformat(),
                'predicted_income': float(row.get('predicted_amount', 0)),
                'predicted_expenses': abs(float(row.get('predicted_amount', 0))),
                'net_change': float(row.get('predicted_amount', 0)),
                'confidence': float(row.get('upper_bound', 0) - row.get('lower_bound', 0)),
                'lower_bound': float(row.get('lower_bound', 0)),
                'upper_bound': float(row.get('upper_bound', 0))
            })

        # Calculate summary
        total_income = sum(d['predicted_income'] for d in daily_predictions if d['predicted_income'] > 0)
        total_expenses = sum(d['predicted_expenses'] for d in daily_predictions if d['predicted_income'] < 0)

        # Get model confidence
        model_confidence = 0.7  # Default
        if hasattr(model, 'metrics') and 'r2' in model.metrics:
            model_confidence = max(0.5, min(0.95, model.metrics['r2']))

        return {
            'daily_predictions': daily_predictions,
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_change': total_income - total_expenses,
            'ending_balance': total_income - total_expenses,
            'confidence': model_confidence,
            'method': 'ml_ensemble',
            'model_version': getattr(model, 'model_version', 'latest')
        }

    @staticmethod
    def _fallback_forecast(
            df: pd.DataFrame,
            forecast_days: int
    ) -> Dict[str, Any]:
        """Simple fallback forecast when ML is not available."""
        # Calculate daily average
        daily_avg = df['amount'].mean() if len(df) > 0 else 0
        daily_std = df['amount'].std() if len(df) > 1 else 0

        daily_predictions = []
        current_date = datetime.now().date()

        for i in range(forecast_days):
            date = current_date + timedelta(days=i+1)

            daily_predictions.append({
                'date': date.isoformat(),
                'predicted_income': max(0, daily_avg),
                'predicted_expenses': abs(min(0, daily_avg)),
                'net_change': daily_avg,
                'confidence': 0.5,
                'lower_bound': daily_avg - daily_std,
                'upper_bound': daily_avg + daily_std
            })

        return {
            'daily_predictions': daily_predictions,
            'total_income': sum(d['predicted_income'] for d in daily_predictions),
            'total_expenses': sum(d['predicted_expenses'] for d in daily_predictions),
            'net_change': daily_avg * forecast_days,
            'ending_balance': daily_avg * forecast_days,
            'confidence': 0.5,
            'method': 'statistical_fallback'
        }

    @staticmethod
    def _error_response(error: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            'daily_predictions': [],
            'total_income': 0,
            'total_expenses': 0,
            'net_change': 0,
            'ending_balance': 0,
            'confidence': 0,
            'method': 'error',
            'error': error
        }