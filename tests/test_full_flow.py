"""Test the complete flow with detailed logging."""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def test_full_flow():
    """Test the full intent classification flow with logging."""
    print("🔍 Testing Full Intent Classification Flow")
    print("=" * 60)

    from services.intent_classification.intent_service import IntentService

    # Create service
    service = IntentService()

    # Test query
    query = "Show me all my invoices"

    # Test with no context (new conversation)
    print("\n1️⃣ Testing with no conversation history...")
    result = service.classify_and_route(
        query=query, user_context=None, conversation_history=None
    )

    print(f"\nResult:")
    print(f"  Intent: {result['classification']['intent']}")
    print(f"  Confidence: {result['classification']['confidence']:.1%}")
    print(f"  Method: {result['classification']['method']}")
    print(f"  Should route: {result['should_route']}")

    # Test with minimal context
    print("\n2️⃣ Testing with minimal user context...")
    result2 = service.classify_and_route(
        query=query, user_context={"user_id": "test_user_123"}, conversation_history=[]
    )

    print(f"\nResult:")
    print(f"  Intent: {result2['classification']['intent']}")
    print(f"  Confidence: {result2['classification']['confidence']:.1%}")
    print(f"  Method: {result2['classification']['method']}")


if __name__ == "__main__":
    test_full_flow()
