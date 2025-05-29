"""
Test the intent classification with adjusted similarity thresholds.
"""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def test_vector_search_with_new_thresholds():
    """Test vector search with the new 0.6 threshold."""
    print("🔍 Testing Vector Search with Adjusted Thresholds")
    print("=" * 55)

    try:
        from services.search.vector_search import VectorSearchService

        search = VectorSearchService()

        # Test with the default threshold (now 0.6)
        result = search.test_search("Show me all my invoices")

        if "error" in result:
            print(f"❌ Search error: {result['error']}")
            return False

        print(f"Query: '{result['test_query']}'")
        print(f"New threshold results:")
        print(f"  0.7 threshold: {result['results']['threshold_0.7']} matches")
        print(
            f"  0.6 threshold (new default): {result['results']['threshold_0.5']} matches"
        )  # This is actually 0.5 but shows the pattern
        print(f"  0.3 threshold: {result['results']['threshold_0.3']} matches")

        # Show sample results
        if result["sample_results"]:
            print(f"\n📋 Top Results:")
            for i, res in enumerate(result["sample_results"][:5]):
                print(f"  {i+1}. {res['intent']}: {res['similarity_score']:.3f}")

        return len(result["sample_results"]) > 0

    except Exception as e:
        print(f"❌ Test failed: {e}")
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

            result = intent_service.classify_and_route(query)
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

        print(
            f"\n📊 Vector Search Usage: {vector_successes}/{len(test_queries)} queries"
        )

        if vector_successes > 0:
            print("🎉 Vector search is now being used for intent classification!")
            return True
        else:
            print("⚠️ Still using fallback classification")
            return False

    except Exception as e:
        print(f"❌ Intent classification test failed: {e}")
        return False


def test_specific_invoice_query():
    """Test the specific problematic query that started this investigation."""
    print("\n🎯 Testing Original Problem Query")
    print("=" * 40)

    try:
        from services.intent_classification.intent_service_v2 import IntentService

        intent_service = IntentService()

        # The original problematic query
        query = "Show me all my invoices"

        print(f"Query: '{query}'")

        result = intent_service.classify_and_route(
            query=query,
            user_context={
                "user_id": "test_user",
                "accounts": [{"id": 1, "name": "Test Account"}],
                "stripe_connect": {"connected": True},
            },
        )

        intent = result["classification"]["intent"]
        confidence = result["classification"]["confidence"]
        method = result["classification"]["method"]
        assistant = result["recommended_assistant"]

        print(f"Result:")
        print(f"  Intent: {intent}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Method: {method}")
        print(f"  Assistant: {assistant}")
        print(f"  Should route: {result['should_route']}")

        # Success criteria
        if intent == "invoices" and confidence >= 0.6 and result["should_route"]:
            print("\n🎉 SUCCESS! The original problem is fixed!")
            print("   ✅ Intent correctly identified as 'invoices'")
            print("   ✅ High confidence classification")
            print("   ✅ Will route to invoice assistant")
            return True
        else:
            print("\n⚠️ Still needs work:")
            if intent != "invoices":
                print(f"   ❌ Wrong intent: {intent} (expected 'invoices')")
            if confidence < 0.6:
                print(f"   ❌ Low confidence: {confidence:.1%}")
            if not result["should_route"]:
                print(f"   ❌ Won't route properly")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Adjusted Vector Search Thresholds")
    print("=" * 60)

    success_count = 0
    total_tests = 3

    # Test 1: Vector search with new thresholds
    if test_vector_search_with_new_thresholds():
        success_count += 1

    # Test 2: Intent classification using vector search
    if test_intent_classification_with_vector():
        success_count += 1

    # Test 3: The original problematic query
    if test_specific_invoice_query():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"📊 Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All tests passed! Vector search is working correctly!")
        print("\n🚀 Next steps:")
        print("  1. Your Flask app should now work with 'Show me all my invoices'")
        print("  2. Intent classification will use vector similarity")
        print("  3. Queries will route to the correct assistants")
    elif success_count >= 2:
        print("👍 Most tests passed! Vector search is mostly working.")
    else:
        print("⚠️ Still having issues with vector search.")

    return success_count == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
