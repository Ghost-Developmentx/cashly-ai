import os
import sys
import asyncio
from dotenv import load_dotenv

# Add a project root to a path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


async def test_assistant_creation():
    """Test that assistants can be created."""
    print("1. Testing Assistant Creation")
    print("-" * 40)

    try:
        from services.openai_assistants.assistant_factory import AssistantFactory

        factory = AssistantFactory()

        # List existing assistants
        assistants = factory.list_all_assistants()
        print(f"✅ Found {len(assistants)} existing assistants")

        # Check if we have Cashly assistants
        cashly_assistants = [a for a in assistants if "Cashly" in a.get("name", "")]
        print(f"✅ Found {len(cashly_assistants)} Cashly assistants")

        for assistant in cashly_assistants:
            print(f"   - {assistant['name']} ({assistant['id']})")

        return len(cashly_assistants) > 0

    except Exception as e:
        print(f"❌ Error testing assistant creation: {e}")
        return False


async def test_assistant_manager():
    """Test the assistant manager."""
    print("\n2. Testing Assistant Manager")
    print("-" * 40)

    try:
        from services.openai_assistants.assistant_manager import AssistantManager

        manager = AssistantManager()

        # Health check
        health = manager.health_check()
        print(f"✅ Assistant Manager Status: {health['status']}")

        active_assistants = len(
            [a for a in health["assistants"].values() if a.get("status") == "active"]
        )
        print(f"✅ Active assistants: {active_assistants}")

        if health["missing_assistants"]:
            print(f"⚠️  Missing assistants: {', '.join(health['missing_assistants'])}")

        return health["status"] in ["healthy", "degraded"]

    except Exception as e:
        print(f"❌ Error testing assistant manager: {e}")
        return False


async def test_intent_integration():
    """Test intent classification integration."""
    print("\n3. Testing Intent Integration")
    print("-" * 40)

    try:
        from services.openai_assistants.openai_integration_service import (
            OpenAIIntegrationService,
        )

        service = OpenAIIntegrationService()

        # Health check
        health = service.health_check()
        print(f"✅ Integration Service Status: {health['status']}")

        # Test intent classification
        from services.intent_classification.intent_service import IntentService

        intent_service = IntentService()

        test_query = "Show me my transactions"
        routing = intent_service.classify_and_route(test_query)

        print(
            f"✅ Intent classification working: {routing['classification']['intent']}"
        )

        return health["status"] != "unhealthy"

    except Exception as e:
        print(f"❌ Error testing intent integration: {e}")
        return False


async def test_end_to_end():
    """Test end-to-end query processing."""
    print("\n4. Testing End-to-End Query Processing")
    print("-" * 40)

    try:
        from services.openai_assistants.openai_integration_service import (
            OpenAIIntegrationService,
        )

        service = OpenAIIntegrationService()

        # Test queries with mock data
        test_cases = [
            {
                "name": "Transaction Query",
                "query": "Show me my recent transactions",
                "user_context": {
                    "accounts": [{"id": "1", "name": "Checking", "balance": 1000}],
                    "transactions": [
                        {
                            "date": "2024-01-01",
                            "amount": -50,
                            "description": "Coffee",
                            "category": "Food",
                        }
                    ],
                },
            },
            {
                "name": "Account Query",
                "query": "What's my account balance?",
                "user_context": {
                    "accounts": [{"id": "1", "name": "Checking", "balance": 1000}]
                },
            },
            {
                "name": "Connection Query",
                "query": "I want to connect my bank account",
                "user_context": {"accounts": []},
            },
        ]

        success_count = 0

        for test_case in test_cases:
            print(f"\n   Testing: {test_case['name']}")
            print(f"   Query: '{test_case['query']}'")

            try:
                result = await service.process_financial_query(
                    query=test_case["query"],
                    user_id="test_user_123",
                    user_context=test_case["user_context"],
                )

                if result["success"]:
                    print(
                        f"   ✅ Success: {result['classification']['intent']} -> {result['classification']['assistant_used']}"
                    )
                    print(f"   ✅ Message: {result['message'][:100]}...")
                    print(f"   ✅ Actions: {len(result['actions'])}")
                    success_count += 1
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"   ❌ Exception: {e}")

        print(
            f"\n✅ End-to-end test results: {success_count}/{len(test_cases)} successful"
        )
        return success_count > 0

    except Exception as e:
        print(f"❌ Error in end-to-end testing: {e}")
        return False


async def test_function_calling():
    """Test function calling integration."""
    print("\n5. Testing Function Calling")
    print("-" * 40)

    try:
        # Test that the tool registry is available
        from services.fin.tool_registry import ToolRegistry

        tool_registry = ToolRegistry()
        tools = [schema["name"] for schema in tool_registry.schemas]

        print(f"✅ Tool registry loaded: {len(tools)} tools available")
        print(f"   Sample tools: {', '.join(tools[:5])}...")

        # Test tool execution with the EXACT signature from tool_registry.py
        test_result = tool_registry.execute(
            tool_name="get_user_accounts",
            tool_args={},
            user_id="test_user",
            transactions=[],  # This is correct - the execute method expects 'transactions'
            user_context={"accounts": []},
        )

        print(f"✅ Tool execution test successful")
        print(f"   Result type: {type(test_result)}")
        print(
            f"   Result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'Not a dict'}"
        )

        return True

    except Exception as e:
        print(f"❌ Error testing function calling: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_environment():
    """Check environment setup."""
    print("0. Checking Environment")
    print("-" * 40)

    load_dotenv()

    # Check the OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    else:
        print(f"✅ OpenAI API key found: {api_key[:10]}...")

    # Check assistant IDs
    assistant_vars = [
        "TRANSACTION_ASSISTANT_ID",
        "ACCOUNT_ASSISTANT_ID",
        "CONNECTION_ASSISTANT_ID",
        "INVOICE_ASSISTANT_ID",
        "FORECASTING_ASSISTANT_ID",
        "BUDGET_ASSISTANT_ID",
        "INSIGHTS_ASSISTANT_ID",
    ]

    found_assistants = 0
    for var in assistant_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
            found_assistants += 1
        else:
            print(f"⚠️  {var}: Not set")

    print(f"✅ Found {found_assistants}/{len(assistant_vars)} assistant IDs")

    return found_assistants > 0


async def main():
    """Run all tests."""
    print("🧪 TESTING OPENAI ASSISTANTS INTEGRATION")
    print("=" * 60)

    tests_passed = 0
    total_tests = 6

    # Test 0: Environment check
    if check_environment():
        tests_passed += 1

    # Test 1: Assistant creation
    if await test_assistant_creation():
        tests_passed += 1

    # Test 2: Assistant manager
    if await test_assistant_manager():
        tests_passed += 1

    # Test 3: Intent integration
    if await test_intent_integration():
        tests_passed += 1

    # Test 4: Function calling
    if await test_function_calling():
        tests_passed += 1

    # Test 5: End-to-end
    if await test_end_to_end():
        tests_passed += 1

    # Results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour OpenAI Assistants are ready for integration!")
        print("\nNext steps:")
        print("1. Update your Rails backend to use the new service")
        print("2. Test with real user queries")
        print("3. Monitor assistant performance and costs")

    elif tests_passed >= 4:
        print("✅ MOST TESTS PASSED!")
        print("\nYour system is mostly working. Address any failures above.")

    else:
        print("❌ SEVERAL TESTS FAILED!")
        print("\nPlease address the issues above before proceeding.")

    return tests_passed >= 4


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
