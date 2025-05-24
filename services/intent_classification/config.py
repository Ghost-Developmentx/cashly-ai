import os
from typing import Dict, Any


class IntentConfig:
    """Configuration for intent classification service."""

    # Model paths
    MODEL_DIR = os.getenv("INTENT_MODEL_DIR", "models/intent_classifier")
    TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", "data/training")

    # Intent categories and thresholds
    CONFIDENCE_THRESHOLDS = {
        "high_confidence": float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8")),
        "medium_confidence": float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "0.6")),
        "low_confidence": float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.4")),
    }

    # Assistant routing
    ASSISTANT_ROUTING = {
        "transactions": "transaction_assistant",
        "accounts": "account_assistant",
        "invoices": "invoice_assistant",
        "forecasting": "forecasting_assistant",
        "budgets": "budget_assistant",
        "insights": "insights_assistant",
        "general": "general_assistant",
    }

    # Training configuration
    TRAINING_CONFIG = {
        "min_samples_per_intent": int(os.getenv("MIN_SAMPLES_PER_INTENT", "10")),
        "max_features": int(os.getenv("TFIDF_MAX_FEATURES", "5000")),
        "ngram_range": (1, 3),
        "test_size": float(os.getenv("TEST_SIZE", "0.2")),
        "random_state": int(os.getenv("RANDOM_STATE", "42")),
    }

    # Model selection
    USE_HUGGINGFACE = os.getenv("USE_HUGGINGFACE", "false").lower() == "true"
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium")

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete configuration as a dictionary."""
        return {
            "model_dir": cls.MODEL_DIR,
            "training_data_dir": cls.TRAINING_DATA_DIR,
            "confidence_thresholds": cls.CONFIDENCE_THRESHOLDS,
            "assistant_routing": cls.ASSISTANT_ROUTING,
            "training_config": cls.TRAINING_CONFIG,
            "use_huggingface": cls.USE_HUGGINGFACE,
            "huggingface_model": cls.HUGGINGFACE_MODEL,
        }
