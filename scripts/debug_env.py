# scripts/debug_env.py
"""
Debug script to check how environment variables are being loaded.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Load .env file
load_dotenv()

def check_assistant_ids():
    """Check how assistant IDs are being loaded."""
    print("=== Checking Assistant ID Environment Variables ===\n")

    assistant_types = [
        "TRANSACTION", "ACCOUNT", "INVOICE", "BANK_CONNECTION",
        "PAYMENT_PROCESSING", "FORECASTING", "BUDGET", "INSIGHTS"
    ]

    for assistant_type in assistant_types:
        env_key = f"{assistant_type}_ASSISTANT_ID"
        raw_value = os.getenv(env_key)

        if raw_value:
            print(f"{env_key}:")
            print(f"  Raw value: '{raw_value}'")
            print(f"  Length: {len(raw_value)}")
            print(f"{env_key}: NOT FOUND\n")


def test_ml_classification():
    """Test if ML classification service can be initialized."""
    print("\n=== Testing ML Classification Service ===\n")

    try:
        from app.services.intent_classification.async_intent_service import AsyncIntentService
        service = AsyncIntentService()
        print("✅ AsyncIntentService initialized successfully")
        print(f"  Min confidence threshold: {service.min_confidence_threshold}")

        # Test the embeddings
        from app.services.embeddings.async_embeddings import AsyncEmbeddingStorage
        storage = AsyncEmbeddingStorage()
        print("✅ Embedding storage initialized")

    except Exception as e:
        print(f"❌ Failed to initialize ML services: {e}")
        import traceback
        traceback.print_exc()


def test_unified_manager():
    """Test UnifiedAssistantManager initialization."""
    print("\n=== Testing UnifiedAssistantManager ===\n")

    try:
        from app.core.assistants import UnifiedAssistantManager
        manager = UnifiedAssistantManager()

        print("Configured assistants:")
        for assistant_type, config in manager.assistant_configs.items():
            status = "✅" if config.assistant_id else "❌"
            print(f"  {status} {assistant_type.value}: {config.assistant_id or 'NOT CONFIGURED'}")

    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_assistant_ids()
    test_ml_classification()
    test_unified_manager()