"""
Centralized model management for all ML models.
Handles loading, training, and caching of MLflow models.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

from app.models.forecasting.cash_flow_forecaster import CashFlowForecaster
from app.models.categorization.transaction_categorizer import TransactionCategorizer
from app.models.trend_analysis.trend_analyzer import TrendAnalyzer
from app.models.budgeting.budget_recommender import BudgetRecommender
from app.models.anomaly.anomaly_detector import AnomalyDetector
from app.utils.mlflow_model_registry import mlflow_registry

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model lifecycle and caching."""

    def __init__(self):
        self._models = {}
        self._model_timestamps = {}
        self._cache_duration = timedelta(hours=1)
        self._lock = asyncio.Lock()

    async def get_model(self, model_type: str, force_reload: bool = False) -> Any:
        """Get a model instance, loading from MLflow if needed."""
        async with self._lock:
            # Check cache
            if not force_reload and self._is_cached(model_type):
                return self._models[model_type]

            # Load model
            model = await self._load_model(model_type)

            # Cache it
            self._models[model_type] = model
            self._model_timestamps[model_type] = datetime.now()

            return model

    def _is_cached(self, model_type: str) -> bool:
        """Check if model is cached and still valid."""
        if model_type not in self._models:
            return False

        timestamp = self._model_timestamps.get(model_type)
        if not timestamp:
            return False

        return datetime.now() - timestamp < self._cache_duration

    @staticmethod
    async def _load_model(model_type: str) -> Any:
        """Load model from MLflow or create new instance."""
        model_classes = {
            'forecaster': CashFlowForecaster,
            'categorizer': TransactionCategorizer,
            'trend_analyzer': TrendAnalyzer,
            'budget_recommender': BudgetRecommender,
            'anomaly_detector': AnomalyDetector
        }

        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = model_classes[model_type]
        model_instance = model_class()

        try:
            # Try to load from MLflow
            model_instance.load_latest(stage="Production")
            logger.info(f"Loaded {model_type} from MLflow")
        except Exception as e:
            logger.warning(f"Could not load {model_type} from MLflow: {e}")
            logger.info(f"Using new {model_type} instance")

        return model_instance

    async def train_model(
            self,
            model_type: str,
            training_data: Any,
            **kwargs
    ) -> Dict[str, Any]:
        """Train a model and save to MLflow."""
        model_instance = await self.get_model(model_type, force_reload=True)

        try:
            # Train model
            model_instance.fit(training_data, **kwargs)

            # Invalidate cache
            if model_type in self._models:
                del self._models[model_type]
                del self._model_timestamps[model_type]

            return {
                'success': True,
                'model_type': model_type,
                'metrics': model_instance.metrics,
                'model_uri': getattr(model_instance, 'model_uri', None)
            }

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            return {
                'success': False,
                'model_type': model_type,
                'error': str(e)
            }

    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()
        self._model_timestamps.clear()


# Global instance
model_manager = ModelManager()