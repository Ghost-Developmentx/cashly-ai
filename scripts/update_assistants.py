import os
import sys
from dotenv import load_dotenv, set_key
from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_bank_connection_assistant_config():
    """Configuration for Bank Connection Assistant (Plaid focused)."""
    return {
        "name": "Cashly Bank Connection Assistant",
        "instructions": """You are the Bank Connection Assistant for Cashly, specializing exclusively in connecting and managing bank accounts through Plaid.
        
        🚨 MANDATORY: When users want to connect/link/add bank accounts, IMMEDIATELY call initiate_plaid_connection()

    Your ONLY responsibilities:
    - Connect bank accounts via Plaid (call initiate_plaid_connection immediately)
    - Help users troubleshoot bank connection issues
    - Disconnect bank accounts when requested
    
    CRITICAL BEHAVIOR:
    - "connect bank account" → CALL initiate_plaid_connection() NOW
    - "link my bank" → CALL initiate_plaid_connection() NOW  
    - "add another account" → CALL initiate_plaid_connection() NOW
    
    DO NOT:
    ❌ Handle Stripe Connect (that's for Payment Processing Assistant)
    ❌ Handle invoice payments 
    ❌ Ask questions before acting - START THE CONNECTION IMMEDIATELY

    Key Guidelines:
    - Be encouraging and supportive during setup processes
    - Explain that bank connections are secure and encrypted
    - Address security concerns with reassurance about Plaid's security
    - Provide step-by-step guidance for connection processes
    - Mention that connections can be disconnected at any time
    - Focus ONLY on bank accounts - do NOT handle Stripe Connect or payment processing

    Available Tools:
    - initiate_plaid_connection: Start the bank account connection process
    - get_user_accounts: Show connected bank accounts and balances
    - get_account_details: Get detailed information for a specific account
    - disconnect_account: Remove connected bank accounts

    Security Notes:
    - All connections use bank-level encryption
    - Cashly never stores banking passwords
    - Users maintain full control over their connected accounts
    - Data is only used to provide financial insights

Important: You handle ONLY bank account connections via Plaid. For payment processing, invoices, or Stripe Connect, direct users to ask about "payment processing" or "accepting payments".""",
        "model": "gpt-4o",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "initiate_plaid_connection",
                    "description": "Start the bank account connection process via Plaid",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "institution_preference": {
                                "type": "string",
                                "description": "Optional preference for bank type (e.g., 'major_bank', 'credit_union', 'any')",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_user_accounts",
                    "description": "Get user's connected bank accounts",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_account_details",
                    "description": "Get detailed information for a specific account",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": "The account ID to get details for",
                            }
                        },
                        "required": ["account_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "disconnect_account",
                    "description": "Disconnect a bank account from the user's profile",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": "The account ID to disconnect",
                            }
                        },
                        "required": ["account_id"],
                    },
                },
            },
        ],
    }


def get_payment_processing_assistant_config():
    """Configuration for Payment Processing Assistant (Stripe Connect focused)."""
    return {
        "name": "Cashly Payment Processing Assistant",
        "instructions": """You are the Payment Processing Assistant for Cashly, specializing exclusively in Stripe Connect setup and payment processing for invoices.

🚨 MANDATORY: When users want payment processing/Stripe setup, IMMEDIATELY call the appropriate Stripe tools.

    Your ONLY responsibilities:
    - Set up Stripe Connect for payment processing
    - Help with incomplete/rejected Stripe accounts  
    - Open Stripe dashboards
    - Troubleshoot Stripe Connect issues
    
    CRITICAL BEHAVIOR:
    - "setup stripe" → CALL setup_stripe_connect() NOW
    - "stripe dashboard" → CALL create_stripe_connect_dashboard_link() NOW
    - "restart stripe" → CALL restart_stripe_connect_setup() NOW
    - "stripe requirements" → CALL get_stripe_connect_requirements() NOW
    
    DO NOT:
    ❌ Handle bank account connections (that's for Bank Connection Assistant)
    ❌ Handle transaction viewing
    ❌ Ask questions before acting - USE TOOLS IMMEDIATELY

    Key Guidelines:
    - Be encouraging about accepting payments and growing business
    - Explain Stripe Connect benefits clearly (professional invoices, fraud protection, etc.)
    - Address concerns about fees with value explanation
    - Provide step-by-step guidance for Stripe setup
    - Help with account recovery if setup gets stuck
    - Focus ONLY on payment processing - do NOT handle bank account connections

Available Tools:
- setup_stripe_connect: Set up Stripe Connect account for payment processing
- check_stripe_connect_status: Check current Stripe Connect status
- create_stripe_connect_dashboard_link: Create link to Stripe Express dashboard
- get_stripe_connect_earnings: Get earnings and fee information
- disconnect_stripe_connect: Disconnect Stripe Connect account

Platform Fee Information:
- 2.9% + Stripe's processing fees
- Covers payment processing, fraud protection, and platform maintenance
- Competitive with other payment processors
- Users keep majority of payment (97.1% after Stripe fees)

Recovery Scenarios:
- If account is rejected: Guide through creating fresh account
- If onboarding incomplete: Provide continue setup link
- If account inactive: Help troubleshoot and reactivate

Important: You handle ONLY payment processing via Stripe Connect. For bank account connections, direct users to ask about "connecting bank accounts" or "linking accounts".""",
        "model": "gpt-4o",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "setup_stripe_connect",
                    "description": "Set up Stripe Connect account for payment processing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "description": "Country code for the business (default: US)",
                                "default": "US",
                            },
                            "business_type": {
                                "type": "string",
                                "enum": [
                                    "individual",
                                    "company",
                                    "non_profit",
                                    "government_entity",
                                ],
                                "description": "Type of business entity",
                                "default": "individual",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_stripe_connect_status",
                    "description": "Check the current status of the user's Stripe Connect account",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_stripe_connect_dashboard_link",
                    "description": "Create a link to the Stripe Express dashboard for payment management",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stripe_connect_earnings",
                    "description": "Get earnings and platform fee information from Stripe Connect",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "period": {
                                "type": "string",
                                "enum": ["week", "month", "quarter", "year"],
                                "description": "Time period for earnings report",
                                "default": "month",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "disconnect_stripe_connect",
                    "description": "Disconnect the user's Stripe Connect account",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }


def main():
    """Main update function."""
    print("🔄 UPDATING CASHLY ASSISTANTS")
    print("=" * 50)

    # Load environment
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False

    client = OpenAI(api_key=api_key)

    try:
        # Step 1: Delete old Connection Assistant
        old_connection_id = os.getenv("CONNECTION_ASSISTANT_ID")
        if old_connection_id:
            print(f"\n1. Deleting old Connection Assistant: {old_connection_id}")
            try:
                client.beta.assistants.delete(assistant_id=old_connection_id)
                print("✅ Old Connection Assistant deleted")
            except Exception as e:
                print(f"⚠️  Could not delete old assistant: {e}")
        else:
            print("\n1. No old Connection Assistant found to delete")

        # Step 2: Create Bank Connection Assistant
        print("\n2. Creating Bank Connection Assistant...")
        bank_config = get_bank_connection_assistant_config()
        bank_assistant = client.beta.assistants.create(**bank_config)
        print(f"✅ Created Bank Connection Assistant: {bank_assistant.id}")

        # Step 3: Create Payment Processing Assistant
        print("\n3. Creating Payment Processing Assistant...")
        payment_config = get_payment_processing_assistant_config()
        payment_assistant = client.beta.assistants.create(**payment_config)
        print(f"✅ Created Payment Processing Assistant: {payment_assistant.id}")

        # Step 4: Update environment variables
        print("\n4. Updating environment variables...")
        env_file = "../.env"

        # Remove old CONNECTION_ASSISTANT_ID
        if old_connection_id:
            # Read .env file and remove the old line
            with open(env_file, "r") as f:
                lines = f.readlines()

            with open(env_file, "w") as f:
                for line in lines:
                    if not line.startswith("CONNECTION_ASSISTANT_ID="):
                        f.write(line)

        # Add new assistant IDs
        set_key(env_file, "BANK_CONNECTION_ASSISTANT_ID", bank_assistant.id)
        set_key(env_file, "PAYMENT_PROCESSING_ASSISTANT_ID", payment_assistant.id)

        print("✅ Environment variables updated")

        # Step 5: Display summary
        print("\n" + "=" * 60)
        print("🎉 ASSISTANT UPDATE COMPLETE!")
        print("=" * 60)

        print("\nRemoved:")
        print("❌ Connection Assistant (mixed Plaid + Stripe)")

        print("\nCreated:")
        print(f"✅ Bank Connection Assistant: {bank_assistant.id}")
        print(f"✅ Payment Processing Assistant: {payment_assistant.id}")

        print("\nNext Steps:")
        print("1. Update your assistant_manager.py to include the new assistant types")
        print("2. Update intent classification to route correctly:")
        print("   - 'connect bank account' → Bank Connection Assistant")
        print("   - 'setup stripe' → Payment Processing Assistant")
        print("3. Test the new routing with sample queries")
        print("4. Update your Rails backend to handle the new assistant types")

        print("\n💡 Routing Guidelines:")
        print(
            "Bank Connection: 'connect bank', 'link account', 'plaid', 'bank balance'"
        )
        print(
            "Payment Processing: 'stripe', 'accept payments', 'invoice setup', 'payment processing'"
        )

        return True

    except Exception as e:
        print(f"❌ Error updating assistants: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Update failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Update completed successfully!")
        sys.exit(0)
