"""
Intent classification services using context-aware embeddings.
"""

from .intent_service import IntentService
from .intent_learner import IntentLearner
from .fallback_classifier import FallbackClassifier

__all__ = ["IntentService", "IntentLearner", "FallbackClassifier"]
