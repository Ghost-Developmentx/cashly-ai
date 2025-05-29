"""Test vector search with adjusted confidence thresholds."""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def test_with_adjusted_thresholds():
    """Test with the new thresholds."""
    print("🔍 Testing Vector Search with Adjusted Thresholds")
    print("=" * 50)

    from services.search.vector_search import VectorSearchService
    from services.embeddings.openai_client import OpenAIEmbeddingClient

    # Create instances
    search = VectorSearchService()
    client = OpenAIEmbeddingClient()

    # Test query
    query = "Show me all my invoices"
    embedding = client.create_embedding(query)

    # Search with new threshold (0.6)
    results = search.search_similar(
        embedding=embedding, similarity_threshold=0.6, limit=10  # New threshold
    )

    print(f"Query: '{query}'")
    print(f"Results found: {len(results)}")

    if results:
        print("\nTop 5 results:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. {result.intent}: {result.similarity_score:.3f}")

    return len(results) > 0


def test_intent_classification():
    """Test if intent classification now uses vector search."""
    print("\n🎯 Testing Intent Classification")
    print("=" * 50)

    from services.intent_classification.intent_service_v2 import IntentService

    service = IntentService()

    # Test the problematic query
    query = "Show me all my invoices"
    result = service.classify_and_route(query)

    print(f"Query: '{query}'")
    print(f"Intent: {result['classification']['intent']}")
    print(f"Confidence: {result['classification']['confidence']:.1%}")
    print(f"Method: {result['classification']['method']}")

    if result["classification"]["method"] == "vector_search":
        print("🎉 SUCCESS: Using vector search!")
        return True
    else:
        print("⚠️ Still using fallback")
        return False


def test_intent_classification_with_vector():
    """Test intent classification to see if it now uses vector search."""
    print("\n🎯 Testing Intent Classification with Vector Search")
    print("=" * 55)

    try:
        from services.intent_classification.intent_service_v2 import IntentService

        intent_service = IntentService()

        test_queries = [
            "Show me all my invoices",
            "Can you show me my invoice list?",
            "What's my account balance?",
            "List recent transactions",
            "Create a budget",
        ]

        vector_successes = 0

        for query in test_queries:
            print(f"\n  Testing: '{query}'")

            # IMPORTANT: Provide proper context
            result = intent_service.classify_and_route(
                query=query,
                user_context={"user_id": "test_user"},  # Add this!
                conversation_history=[],  # Empty but not None
            )

            intent = result["classification"]["intent"]
            confidence = result["classification"]["confidence"]
            method = result["classification"]["method"]

            print(f"    -> {intent} ({confidence:.1%}) via {method}")

            # Check if we're now using vector search
            if method in ["vector_search", "context_aware_similarity"]:
                print("    🎉 Using vector search!")
                vector_successes += 1
            elif confidence >= 0.8:
                print("    ✅ High confidence fallback")
            else:
                print("    ⚠️ Fallback classification")
    finally:
        print(
            f"\n🎯 Test completed. Vector search success rate: {vector_successes / len(test_queries):.1%}"
        )


if __name__ == "__main__":
    test_with_adjusted_thresholds()
    test_intent_classification()
