import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


async def debug_tool_registry():
    """Debug the tool registry parameter issue."""
    print("🔍 DEBUGGING TOOL REGISTRY")
    print("=" * 40)

    try:
        from services.fin.tool_registry import ToolRegistry

        tool_registry = ToolRegistry()
        print("✅ Tool registry imported")

        # Check the execute method signature
        import inspect

        execute_sig = inspect.signature(tool_registry.execute)
        print(f"✅ Execute method signature: {execute_sig}")

        # The correct way to call it based on your signature:
        # def execute(self, tool_name: str, tool_args: Dict[str, Any], *,
        #            user_id: str, transactions: List[Dict[str, Any]], user_context: Dict[str, Any])

        print(f"\n   Testing: Correct keyword arguments")
        try:
            result = tool_registry.execute(
                tool_name="get_user_accounts",
                tool_args={},
                user_id="test_user",
                transactions=[],
                user_context={"accounts": []},
            )
            print(f"   ✅ Success: {type(result)}")
            print(f"   ✅ Result: {result}")
            return "keyword_args"
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback

            traceback.print_exc()

        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def debug_intent_service():
    """Debug the intent service."""
    print("\n🔍 DEBUGGING INTENT SERVICE")
    print("=" * 40)

    try:
        from services.intent_classification.intent_service import IntentService

        intent_service = IntentService()
        print("✅ Intent service imported")

        # Test classification
        test_query = "Show me my transactions"
        result = intent_service.classify_and_route(test_query)

        print(f"✅ Classification result:")
        print(f"   Intent: {result['classification']['intent']}")
        print(f"   Confidence: {result['classification']['confidence']:.2%}")
        print(f"   Assistant: {result['recommended_assistant']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def debug_assistant_manager():
    """Debug the assistant manager."""
    print("\n🔍 DEBUGGING ASSISTANT MANAGER")
    print("=" * 40)

    try:
        from services.openai_assistants.assistant_manager import (
            AssistantManager,
            AssistantType,
        )

        manager = AssistantManager()
        print("✅ Assistant manager imported")

        # Check health
        health = manager.health_check()
        print(f"✅ Health status: {health['status']}")

        # Test with simple query (without tool calls)
        print("\n   Testing simple query...")
        try:
            response = await manager.process_query(
                query="Hello, I'm testing the assistant",
                assistant_type=AssistantType.TRANSACTION,
                user_id="debug_user",
            )
            print(f"   ✅ Simple query successful: {response.success}")
            print(f"   ✅ Response: {response.content[:100]}...")
            return True
        except Exception as e:
            print(f"   ❌ Simple query failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def debug_integration_service():
    """Debug the integration service."""
    print("\n🔍 DEBUGGING INTEGRATION SERVICE")
    print("=" * 40)

    try:
        from services.openai_assistants.openai_integration_service import (
            OpenAIIntegrationService,
        )

        service = OpenAIIntegrationService()
        print("✅ Integration service imported")

        # Test simple query first
        print("\n   Testing simple query...")
        try:
            result = await service.process_financial_query(
                query="Hello, I need help with my finances",
                user_id="debug_user",
                user_context={"accounts": []},
            )

            print(f"   ✅ Query processed: {result['success']}")
            print(f"   ✅ Intent: {result['classification']['intent']}")
            print(f"   ✅ Assistant: {result['classification']['assistant_used']}")
            print(f"   ✅ Message: {result['message'][:100]}...")

            return True

        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run debug tests."""
    print("🐛 DEBUGGING OPENAI ASSISTANTS INTEGRATION")
    print("=" * 60)

    load_dotenv()

    # Debug each component
    working_tool_config = await debug_tool_registry()
    intent_working = await debug_intent_service()
    assistant_working = await debug_assistant_manager()
    integration_working = await debug_integration_service()

    print("\n" + "=" * 60)
    print("DEBUGGING SUMMARY")
    print("=" * 60)

    if working_tool_config:
        print(f"✅ Tool registry works with: {working_tool_config}")
    else:
        print("❌ Tool registry has parameter issues")

    if intent_working:
        print("✅ Intent service working")
    else:
        print("❌ Intent service has issues")

    if assistant_working:
        print("✅ Assistant manager working")
    else:
        print("❌ Assistant manager has issues")

    if integration_working:
        print("✅ Integration service working")
    else:
        print("❌ Integration service has issues")

    return working_tool_config and intent_working and assistant_working


if __name__ == "__main__":
    asyncio.run(main())
