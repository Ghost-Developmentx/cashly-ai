"""Debug why IntentResolver is returning 'general' instead of 'invoices'."""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def debug_intent_resolver():
    """Debug the IntentResolver to see why it's failing."""
    print("🔍 Debugging IntentResolver")
    print("=" * 60)

    from services.intent_determination.intent_resolver import IntentResolver
    from services.embeddings.openai_client import OpenAIEmbeddingClient

    # Test query
    query = "Show me all my invoices"
    user_id = "test_user"
    conversation_id = "test_conv_1"

    # Create resolver
    resolver = IntentResolver()

    # Test 1: Direct embedding from query
    print("\n1️⃣ Testing with direct query embedding...")
    client = OpenAIEmbeddingClient()
    embedding = client.create_embedding(query)

    # Search directly
    search_results = resolver.search_service.search_similar(
        embedding=embedding, limit=10, similarity_threshold=0.5
    )
    print(f"Direct search found: {len(search_results)} results")
    if search_results:
        print(
            f"Top result: {search_results[0].intent} ({search_results[0].similarity_score:.3f})"
        )

    # Test 2: Through resolver
    print("\n2️⃣ Testing through resolver...")
    resolution = resolver.resolve_intent(
        query=query,
        conversation_history=[],  # Empty history
        user_id=user_id,
        conversation_id=conversation_id,
        user_context=None,
    )

    print(f"Resolution: {resolution['intent']} ({resolution['confidence']:.3f})")
    print(f"Method: {resolution['method']}")

    # Test 3: Check what embedding text is being used
    print("\n3️⃣ Checking embedding text...")
    try:
        # Try to see what the aggregator is doing
        processed = resolver.aggregator_service.process_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_history=[],
            user_context=None,
        )
        print(f"Embedding text: '{processed['embedding_text']}'")
        print(f"Original query: '{query}'")
    except Exception as e:
        print(f"Error processing conversation: {e}")


if __name__ == "__main__":
    debug_intent_resolver()
