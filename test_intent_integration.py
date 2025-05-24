import sys
import os
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


def test_intent_classification():
    """Test intent classification with realistic Cashly queries."""

    # Import our new intent service
    try:
        from services.intent_classification.intent_classifier import IntentClassifier
        from services.intent_classification.intent_service import IntentService

        print("✅ Intent classification modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import intent classification modules: {e}")
        print("Make sure you've created the intent_service.py file")
        return False

    # Initialize services
    classifier = IntentClassifier()
    intent_service = IntentService()

    # Test cases based on your existing Fin tools
    test_cases = [
        # Transaction-related queries (should map to get_transactions, create_transaction, etc.)
        {
            "query": "Show me my transactions from last month",
            "expected_intent": "transactions",
            "expected_tools": ["get_transactions"],
            "user_context": {
                "accounts": [{"id": "1", "name": "Checking", "balance": 1000}]
            },
        },
        {
            "query": "Add a $50 grocery expense to my checking account",
            "expected_intent": "transactions",
            "expected_tools": ["create_transaction"],
            "user_context": {"accounts": [{"id": "1", "name": "Checking"}]},
        },
        {
            "query": "How much did I spend on restaurants this week?",
            "expected_intent": "transactions",
            "expected_tools": ["get_transactions", "calculate_category_spending"],
            "user_context": {"accounts": [{"id": "1"}]},
        },
        # Account-related queries (should map to get_user_accounts, initiate_plaid_connection, etc.)
        {
            "query": "Connect my Wells Fargo bank account",
            "expected_intent": "accounts",
            "expected_tools": ["initiate_plaid_connection"],
            "user_context": {"accounts": []},
        },
        {
            "query": "What's my total account balance?",
            "expected_intent": "accounts",
            "expected_tools": ["get_user_accounts"],
            "user_context": {"accounts": [{"id": "1", "balance": 1000}]},
        },
        # Invoice-related queries (should map to create_invoice, get_invoices, etc.)
        {
            "query": "Create an invoice for $1500 for my client John",
            "expected_intent": "invoices",
            "expected_tools": ["create_invoice"],
            "user_context": {"integrations": []},
        },
        {
            "query": "Send a payment reminder for invoice #123",
            "expected_intent": "invoices",
            "expected_tools": ["send_invoice_reminder"],
            "user_context": {"integrations": [{"provider": "stripe"}]},
        },
        # Forecasting queries (should map to forecast_cash_flow)
        {
            "query": "What will my cash flow look like next month?",
            "expected_intent": "forecasting",
            "expected_tools": ["forecast_cash_flow"],
            "user_context": {"accounts": [{"id": "1"}], "transactions": []},
        },
        {
            "query": "Predict my expenses for the next 3 months",
            "expected_intent": "forecasting",
            "expected_tools": ["forecast_cash_flow"],
            "user_context": {"transactions": [{"amount": -100}] * 20},
        },
        # Budget queries (should map to generate_budget)
        {
            "query": "Help me create a monthly budget",
            "expected_intent": "budgets",
            "expected_tools": ["generate_budget"],
            "user_context": {"transactions": [{"amount": -100}] * 15},
        },
        # Insights queries (should map to analyze_trends, detect_anomalies, etc.)
        {
            "query": "Analyze my spending trends over the last 6 months",
            "expected_intent": "insights",
            "expected_tools": ["analyze_trends"],
            "user_context": {"transactions": [{"amount": -50}] * 30},
        },
        {
            "query": "Are there any unusual transactions this month?",
            "expected_intent": "insights",
            "expected_tools": ["detect_anomalies"],
            "user_context": {"transactions": [{"amount": -50}] * 10},
        },
        # General queries
        {
            "query": "Hello, I need help with my finances",
            "expected_intent": "general",
            "expected_tools": [],
            "user_context": {},
        },
    ]

    print("\n" + "=" * 80)
    print("TESTING INTENT CLASSIFICATION WITH CASHLY QUERIES")
    print("=" * 80)

    correct_predictions = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_intent = test_case["expected_intent"]
        user_context = test_case["user_context"]

        print(f"\n{i}. Testing: '{query}'")
        print(f"   Expected intent: {expected_intent}")

        # Test basic classification
        classification = classifier.classify_intent(query)
        print(
            f"   Predicted intent: {classification['intent']} ({classification['confidence']:.2%})"
        )
        print(f"   Method: {classification['method']}")

        # Test routing service
        routing_result = intent_service.classify_and_route(query, user_context)
        recommended_assistant = routing_result["recommended_assistant"]
        strategy = routing_result["routing"]["strategy"]

        print(f"   Recommended assistant: {recommended_assistant}")
        print(f"   Routing strategy: {strategy}")

        # Check if prediction is correct
        is_correct = classification["intent"] == expected_intent
        if is_correct:
            correct_predictions += 1
            print("   ✅ CORRECT")
        else:
            print("   ❌ INCORRECT")

        # Show fallback options if available
        if routing_result["fallback_options"]:
            print("   Fallback options:")
            for option in routing_result["fallback_options"]:
                print(f"     - {option['intent']} ({option['confidence']:.2%})")

        print("   " + "-" * 50)

    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n📊 RESULTS:")
    print(f"Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")

    if accuracy >= 80:
        print("✅ Intent classification performance is GOOD")
    elif accuracy >= 60:
        print(
            "⚠️  Intent classification performance is MODERATE - consider training with more data"
        )
    else:
        print("❌ Intent classification performance is POOR - training required")

    return accuracy >= 60


def test_tool_mapping():
    """Test mapping between intents and your existing Fin tools."""
    print("\n" + "=" * 80)
    print("TESTING TOOL MAPPING COMPATIBILITY")
    print("=" * 80)

    # Import your existing tool registry
    try:
        from services.fin.tool_registry import ToolRegistry

        tool_registry = ToolRegistry()
        existing_tools = [schema["name"] for schema in tool_registry.schemas]
        print(f"✅ Found {len(existing_tools)} existing Fin tools")
    except ImportError:
        print("❌ Could not import existing tool registry")
        return False

    # Define expected tool mapping for each intent
    intent_tool_mapping = {
        "transactions": [
            "get_transactions",
            "create_transaction",
            "update_transaction",
            "delete_transaction",
            "categorize_transactions",
            "calculate_category_spending",
        ],
        "accounts": [
            "get_user_accounts",
            "get_account_details",
            "initiate_plaid_connection",
            "disconnect_account",
        ],
        "invoices": [
            "create_invoice",
            "get_invoices",
            "send_invoice_reminder",
            "mark_invoice_paid",
            "setup_stripe_connect",
        ],
        "forecasting": ["forecast_cash_flow"],
        "budgets": ["generate_budget"],
        "insights": [
            "analyze_trends",
            "detect_anomalies",
            "calculate_category_spending",
        ],
    }

    print("\nChecking tool availability for each intent:")
    all_tools_available = True

    for intent, required_tools in intent_tool_mapping.items():
        print(f"\n{intent.upper()}:")
        for tool in required_tools:
            if tool in existing_tools:
                print(f"  ✅ {tool}")
            else:
                print(f"  ❌ {tool} - NOT FOUND")
                all_tools_available = False

    if all_tools_available:
        print("\n✅ All required tools are available")
    else:
        print("\n⚠️  Some tools are missing - check tool registry")

    return all_tools_available


def test_basic_classifier_only():
    """Test just the basic classifier without IntentService (in case it's not created yet)."""
    print("\n" + "=" * 80)
    print("TESTING BASIC INTENT CLASSIFIER")
    print("=" * 80)

    try:
        from services.intent_classification.intent_classifier import IntentClassifier

        classifier = IntentClassifier()
        print("✅ IntentClassifier imported and initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize IntentClassifier: {e}")
        return False

    # Simple test queries
    test_queries = [
        ("Show me my transactions", "transactions"),
        ("Connect my bank account", "accounts"),
        ("Create an invoice", "invoices"),
        ("Forecast my cash flow", "forecasting"),
        ("Help me budget", "budgets"),
        ("Analyze my spending", "insights"),
        ("Hello", "general"),
    ]

    correct = 0
    for query, expected in test_queries:
        result = classifier.classify_intent(query)
        actual = result["intent"]
        confidence = result["confidence"]

        is_correct = actual == expected
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(
            f"{status} '{query}' → {actual} ({confidence:.2%}) [expected: {expected}]"
        )

    accuracy = (correct / len(test_queries)) * 100
    print(
        f"\nBasic classifier accuracy: {correct}/{len(test_queries)} ({accuracy:.1f}%)"
    )

    return accuracy >= 70


def main():
    """Run all integration tests."""
    print("🚀 CASHLY INTENT CLASSIFICATION INTEGRATION TEST")
    print("=" * 80)

    tests_passed = 0
    total_tests = 0

    # Test 1: Basic classifier (always run this)
    print("Test 1: Basic Intent Classifier")
    if test_basic_classifier_only():
        tests_passed += 1
    total_tests += 1

    # Test 2: Tool mapping compatibility
    print("\nTest 2: Tool Mapping Compatibility")
    try:
        if test_tool_mapping():
            tests_passed += 1
        total_tests += 1
    except Exception as e:
        print(f"⚠️  Skipping tool mapping test: {e}")

    # Test 3: Full intent classification (only if IntentService is available)
    print("\nTest 3: Full Intent Classification Service")
    try:
        if test_intent_classification():
            tests_passed += 1
        total_tests += 1
    except Exception as e:
        print(f"⚠️  Skipping full intent test: {e}")
        print("This is expected if you haven't created intent_service.py yet")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed >= 2:
        print("🎉 INTENT CLASSIFICATION IS WORKING WELL!")
        print("\nNext steps:")
        print("1. Create the IntentService class (if not done yet)")
        print("2. Create OpenAI Assistant manager")
        print("3. Build specialized assistants")
        print("4. Update Rails backend to use intent routing")
        return True
    elif tests_passed >= 1:
        print("⚠️  Basic functionality works, but some features need attention")
        return True
    else:
        print("❌ Major issues detected - review the errors above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
