"""
Machine-learning-based categorization using MLflow models.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd

from app.services.ml.model_manager import model_manager

logger = logging.getLogger(__name__)


class MLCategorizer:
    """Handles transaction categorization using ML models."""

    def __init__(self):
        self.min_confidence_threshold = 0.7

    async def categorize(
            self,
            description: str,
            amount: float,
            additional_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Categorize a single transaction."""
        try:
            # Get model
            categorizer = await model_manager.get_model('categorizer')

            if not categorizer.is_fitted:
                return self._no_model_response()

            # Prepare transaction data
            transaction_data = {
                'description': description,
                'amount': amount,
                'date': pd.Timestamp.now()
            }

            if additional_features:
                transaction_data.update(additional_features)

            df = pd.DataFrame([transaction_data])

            # Get predictions with confidence
            results = categorizer.predict_with_confidence(df)

            if results:
                result = results[0]
                return {
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'method': 'ml_model',
                    'alternatives': result.get('alternatives', []),
                    'is_confident': result.get('is_confident', False)
                }

            return self._no_prediction_response()

        except Exception as e:
            logger.error(f"ML categorization failed: {e}")
            return self._error_response(str(e))

    async def categorize_batch(
            self,
            transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Categorize multiple transactions."""
        try:
            # Get model
            categorizer = await model_manager.get_model('categorizer')

            if not categorizer.is_fitted:
                return [self._no_model_response() for _ in transactions]

            # Prepare data
            df = pd.DataFrame(transactions)

            # Ensure required columns
            if 'date' not in df.columns:
                df['date'] = pd.Timestamp.now()

            # Get predictions
            results = categorizer.predict_with_confidence(df)

            return results

        except Exception as e:
            logger.error(f"Batch categorization failed: {e}")
            return [self._error_response(str(e)) for _ in transactions]

    @staticmethod
    async def update_training(
            description: str,
            amount: float,
            category: str
    ) -> bool:
        """Update training data with user feedback."""
        try:
            # This would typically save to a training dataset
            # For now, log it
            logger.info(
                f"Training feedback: '{description}' -> '{category}' "
                f"(amount: {amount})"
            )

            # In production, you might:
            # 1. Save to a feedback table
            # 2. Periodically retrain with feedback data
            # 3. Use online learning to update immediately

            return True

        except Exception as e:
            logger.error(f"Failed to update training: {e}")
            return False

    @staticmethod
    def _no_model_response() -> Dict[str, Any]:
        """Response when no model is available."""
        return {
            'category': None,
            'confidence': 0.0,
            'method': 'ml_unavailable',
            'error': 'No trained model available'
        }

    @staticmethod
    def _no_prediction_response() -> Dict[str, Any]:
        """Response when prediction fails."""
        return {
            'category': 'Other',
            'confidence': 0.0,
            'method': 'ml_failed'
        }

    @staticmethod
    def _error_response(error: str) -> Dict[str, Any]:
        """Response for errors."""
        return {
            'category': None,
            'confidence': 0.0,
            'method': 'ml_error',
            'error': error
        }