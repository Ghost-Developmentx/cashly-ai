"""
Machine-learning-based categorization.
"""

import logging
from typing import Dict, Any, Optional
import joblib
import os

logger = logging.getLogger(__name__)


class MLCategorizer:
    """
    Handles transaction categorization using a machine learning model.

    This class is responsible for categorizing transactions by utilizing
    a pre-trained machine learning model. If the model is unavailable,
    categorization may fall back to alternative implementations. It also
    allows for additional updates to training data for future improvements.

    Attributes
    ----------
    model : Any or None
        Pre-trained machine learning model used for categorization.
    vectorizer : Any or None
        Feature vectorizer for preparing input data for the model.
    model_path : str
        Path to the saved machine learning model file.
    vectorizer_path : str
        Path to the saved feature vectorizer file.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_path = "models/categorization_model.pkl"
        self.vectorizer_path = "models/categorization_vectorizer.pkl"
        self._load_model()

    def _load_model(self):
        """Load pre-trained model if available."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info("Loaded ML categorization model")
            else:
                logger.info("No ML model found, using rule-based only")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")

    async def categorize(self, description: str, amount: float) -> Dict[str, Any]:
        """
        Categorize using ML model.

        Args:
            description: Transaction description
            amount: Transaction amount

        Returns:
            Category prediction with confidence
        """
        if not self.model or not self.vectorizer:
            return {"category": None, "confidence": 0.0, "method": "ml_unavailable"}

        try:
            # Prepare features
            features = self._prepare_features(description, amount)

            # Get prediction
            prediction = self.model.predict([features])[0]

            # Get confidence (probability)
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities)

            return {
                "category": prediction,
                "confidence": round(confidence, 2),
                "method": "ml_model",
            }

        except Exception as e:
            logger.error(f"ML categorization failed: {e}")
            return {"category": None, "confidence": 0.0, "method": "ml_error"}

    def _prepare_features(self, description: str, amount: float) -> str:
        """Prepare features for ML model."""
        # Simple feature: combine description with amount range
        amount_range = self._get_amount_range(amount)
        return f"{description} {amount_range}"

    @staticmethod
    def _get_amount_range(amount: float) -> str:
        """Get amount range category."""
        amount = abs(amount)
        if amount < 10:
            return "small"
        elif amount < 50:
            return "medium"
        elif amount < 200:
            return "large"
        else:
            return "xlarge"

    @staticmethod
    async def update_training(description: str, amount: float, category: str) -> bool:
        """Update training data (for future retraining)."""
        # In production, this would save to a training dataset
        logger.info(f"Recording training data: {description} -> {category}")
        return True
