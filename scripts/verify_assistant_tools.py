import os
from app.services.openai_assistants import AssistantFactory
from app.services.openai_assistants.assistant_manager import AssistantType
from openai import OpenAI


def verify_assistant_tools():
    """Compare current assistant tools with expected tools."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    factory = AssistantFactory()

    assistant_ids = {
        AssistantType.TRANSACTION: os.getenv("TRANSACTION_ASSISTANT_ID"),
        AssistantType.ACCOUNT: os.getenv("ACCOUNT_ASSISTANT_ID"),
        AssistantType.CONNECTION: os.getenv("CONNECTION_ASSISTANT_ID"),
        AssistantType.INVOICE: os.getenv("INVOICE_ASSISTANT_ID"),
        AssistantType.FORECASTING: os.getenv("FORECASTING_ASSISTANT_ID"),
        AssistantType.BUDGET: os.getenv("BUDGET_ASSISTANT_ID"),
        AssistantType.INSIGHTS: os.getenv("INSIGHTS_ASSISTANT_ID"),
    }

    # Get expected configurations
    expected_configs = {
        AssistantType.TRANSACTION: factory._get_transaction_assistant_config(),
        AssistantType.ACCOUNT: factory._get_account_assistant_config(),
        AssistantType.CONNECTION: factory._get_connection_assistant_config(),
        AssistantType.INVOICE: factory._get_invoice_assistant_config(),
        AssistantType.FORECASTING: factory._get_forecasting_assistant_config(),
        AssistantType.BUDGET: factory._get_budget_assistant_config(),
        AssistantType.INSIGHTS: factory._get_insights_assistant_config(),
    }

    print("🔍 Verifying assistant tools...")
    print("=" * 60)

    for assistant_type, assistant_id in assistant_ids.items():
        if not assistant_id:
            print(f"⚠️  {assistant_type.value}: No assistant ID found")
            continue

        try:
            # Get current assistant from OpenAI
            current_assistant = client.beta.assistants.retrieve(assistant_id)
            current_tools = current_assistant.tools

            # Get expected tools
            expected_config = expected_configs[assistant_type]
            expected_tools = expected_config.get("tools", [])

            print(f"\n🤖 {assistant_type.value} ({assistant_id})")
            print(f"   Current tools: {len(current_tools)}")
            print(f"   Expected tools: {len(expected_tools)}")

            # Compare tool names
            current_tool_names = set()
            for tool in current_tools:
                if tool.type == "function":
                    current_tool_names.add(tool.function.name)
                else:
                    current_tool_names.add(tool.type)

            expected_tool_names = set()
            for tool in expected_tools:
                if tool["type"] == "function":
                    expected_tool_names.add(tool["function"]["name"])
                else:
                    expected_tool_names.add(tool["type"])

            # Check for differences
            missing_tools = expected_tool_names - current_tool_names
            extra_tools = current_tool_names - expected_tool_names

            if missing_tools:
                print(f"   ❌ Missing tools: {', '.join(missing_tools)}")

            if extra_tools:
                print(f"   ⚠️  Extra tools: {', '.join(extra_tools)}")

            if not missing_tools and not extra_tools:
                print(f"   ✅ Tools match expected configuration")
            else:
                print(f"   🔄 Needs update")

        except Exception as e:
            print(f"   ❌ Error checking {assistant_type.value}: {str(e)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        exit(1)

    verify_assistant_tools()
