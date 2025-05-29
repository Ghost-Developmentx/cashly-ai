"""Debug why IntentDeterminer is returning 'general' with low confidence."""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def debug_intent_determiner():
    """Debug the IntentDeterminer directly."""
    print("🔍 Debugging IntentDeterminer")
    print("=" * 60)

    from services.embeddings.openai_client import OpenAIEmbeddingClient
    from services.search.vector_search import VectorSearchService
    from services.intent_determination.intent_determiner import IntentDeterminer

    # Get search results
    query = "Show me all my invoices"
    client = OpenAIEmbeddingClient()
    embedding = client.create_embedding(query)

    search = VectorSearchService()
    search_results = search.search_similar(
        embedding=embedding, similarity_threshold=0.5, limit=10
    )

    print(f"Search results: {len(search_results)} found")

    # Group by intent
    intent_counts = {}
    for result in search_results:
        intent = result.intent
        if intent not in intent_counts:
            intent_counts[intent] = 0
        intent_counts[intent] += 1

    print(f"Intents found: {intent_counts}")

    # Test IntentDeterminer
    determiner = IntentDeterminer()
    intent, confidence, analysis = determiner.determine_intent(search_results)

    print(f"\nDetermined intent: {intent}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Analysis: {analysis}")

    # Check intent scores
    intent_scores = determiner._calculate_intent_scores(search_results)
    print(f"\nIntent scores:")
    for intent, score in intent_scores.items():
        print(
            f"  {intent}: score={score.score:.3f}, confidence={score.confidence:.3f}, count={score.evidence_count}"
        )


if __name__ == "__main__":
    debug_intent_determiner()
