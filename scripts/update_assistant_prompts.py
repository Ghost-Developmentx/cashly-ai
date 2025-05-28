import os
from services.openai_assistants.assistant_factory import AssistantFactory
from services.openai_assistants.assistant_manager import AssistantType


def update_assistant_prompts():
    """Update all assistant prompts with the latest configurations."""

    factory = AssistantFactory()

    # Get current assistant IDs from environment
    assistant_ids = {
        AssistantType.TRANSACTION: os.getenv("TRANSACTION_ASSISTANT_ID"),
        AssistantType.ACCOUNT: os.getenv("ACCOUNT_ASSISTANT_ID"),
        AssistantType.BANK_CONNECTION: os.getenv("BANK_CONNECTION_ASSISTANT_ID"),
        AssistantType.PAYMENT_PROCESSING: os.getenv("PAYMENT_PROCESSING_ASSISTANT_ID"),
        AssistantType.INVOICE: os.getenv("INVOICE_ASSISTANT_ID"),
        AssistantType.FORECASTING: os.getenv("FORECASTING_ASSISTANT_ID"),
        AssistantType.BUDGET: os.getenv("BUDGET_ASSISTANT_ID"),
        AssistantType.INSIGHTS: os.getenv("INSIGHTS_ASSISTANT_ID"),
    }

    # Get the latest configurations
    assistant_configs = {
        AssistantType.TRANSACTION: factory._get_transaction_assistant_config(),
        AssistantType.ACCOUNT: factory._get_account_assistant_config(),
        AssistantType.BANK_CONNECTION: factory._get_bank_connection_assistant_config(),
        AssistantType.PAYMENT_PROCESSING: factory._get_payment_processing_assistant_config(),
        AssistantType.INVOICE: factory._get_invoice_assistant_config(),
        AssistantType.FORECASTING: factory._get_forecasting_assistant_config(),
        AssistantType.BUDGET: factory._get_budget_assistant_config(),
        AssistantType.INSIGHTS: factory._get_insights_assistant_config(),
    }

    print("🤖 Updating OpenAI Assistant prompts...")
    print("=" * 50)

    updated_count = 0

    for assistant_type, assistant_id in assistant_ids.items():
        if not assistant_id:
            print(f"⚠️  {assistant_type.value}: No assistant ID found, skipping")
            continue

        config = assistant_configs[assistant_type]

        # Include tools in the update - OpenAI will replace all existing tools
        update_config = config

        print(f"🔄 Updating {assistant_type.value} assistant ({assistant_id})...")
        print(f"   📝 Instructions: {len(config['instructions'])} characters")
        print(f"   🛠️  Tools: {len(config.get('tools', []))} functions")

        try:
            success = factory.update_assistant(assistant_id, update_config)

            if success:
                print(
                    f"✅ {assistant_type.value}: Updated successfully (including tools)"
                )
                updated_count += 1
            else:
                print(f"❌ {assistant_type.value}: Update failed")

        except Exception as e:
            print(f"❌ {assistant_type.value}: Error - {str(e)}")

    print("=" * 50)
    print(
        f"✨ Updated {updated_count}/{len([aid for aid in assistant_ids.values() if aid])} assistants"
    )

    if updated_count > 0:
        print("\n🎉 All assistant prompts have been updated!")
        print("Your assistants now have the latest instructions and capabilities.")
    else:
        print(
            "\n⚠️  No assistants were updated. Check your environment variables and OpenAI API key."
        )


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        exit(1)

    update_assistant_prompts()
