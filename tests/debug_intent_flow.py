"""Debug the intent classification flow to see where it's failing."""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def debug_intent_flow():
    """Debug the full intent resolution flow."""
    print("🔍 Debugging Intent Classification Flow")
    print("=" * 60)

    from services.intent_classification.intent_service import IntentService
    from services.intent_determination.intent_resolver import IntentResolver
    from services.embeddings.openai_client import OpenAIEmbeddingClient
    from services.search.vector_search import VectorSearchService

    # Test query
    query = "Show me all my invoices"
    user_id = "test_user"

    # Step 1: Generate embedding
    print("\n1️⃣ Generating embedding...")
    client = OpenAIEmbeddingClient()
    embedding = client.create_embedding(query)
    print(f"✅ Embedding generated: {len(embedding)} dimensions")

    # Step 2: Search for similar conversations
    print("\n2️⃣ Searching for similar conversations...")
    search = VectorSearchService()
    search_results = search.search_similar(
        embedding=embedding, similarity_threshold=0.5, limit=10  # Lower threshold
    )
    print(f"✅ Found {len(search_results)} similar conversations")
    if search_results:
        print(f"   Top similarity: {search_results[0].similarity_score:.3f}")
        print(f"   Top intent: {search_results[0].intent}")

    # Step 3: Try the IntentResolver directly
    print("\n3️⃣ Testing IntentResolver directly...")
    resolver = IntentResolver()
    resolution = resolver.resolve_intent(
        query=query,
        conversation_history=[],
        user_id=user_id,
        conversation_id="test_conv_1",
        user_context={"test": True},
    )
    print(f"✅ Resolution:")
    print(f"   Intent: {resolution['intent']}")
    print(f"   Confidence: {resolution['confidence']:.3f}")
    print(f"   Method: {resolution['method']}")

    # Step 4: Test the full service
    print("\n4️⃣ Testing full IntentService...")
    service = IntentService()
    result = service.classify_and_route(query)
    print(f"✅ Final result:")
    print(f"   Intent: {result['classification']['intent']}")
    print(f"   Confidence: {result['classification']['confidence']:.1%}")
    print(f"   Method: {result['classification']['method']}")

    return result


if __name__ == "__main__":
    debug_intent_flow()
