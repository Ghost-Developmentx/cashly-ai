import os
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from .assistant_manager import AssistantType

logger = logging.getLogger(__name__)


class AssistantFactory:
    """
    Factory for creating and configuring OpenAI Assistants for Cashly.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    def create_all_assistants(self) -> Dict[str, str]:
        """Create all Cashly assistants and return their IDs."""
        assistant_ids = {}

        assistant_configs = {
            AssistantType.TRANSACTION: self._get_transaction_assistant_config(),
            AssistantType.ACCOUNT: self._get_account_assistant_config(),
            AssistantType.BANK_CONNECTION: self._get_bank_connection_assistant_config(),
            AssistantType.PAYMENT_PROCESSING: self._get_payment_processing_assistant_config(),
            AssistantType.INVOICE: self._get_invoice_assistant_config(),
            AssistantType.FORECASTING: self._get_forecasting_assistant_config(),
            AssistantType.BUDGET: self._get_budget_assistant_config(),
            AssistantType.INSIGHTS: self._get_insights_assistant_config(),
        }

        for assistant_type, config in assistant_configs.items():
            try:
                assistant = self.client.beta.assistants.create(**config)
                assistant_ids[assistant_type.value] = assistant.id
                logger.info(f"Created {assistant_type.value} assistant: {assistant.id}")
            except Exception as e:
                logger.error(f"Failed to create {assistant_type.value} assistant: {e}")

        return assistant_ids

    def update_assistant(self, assistant_id: str, config: Dict[str, Any]) -> bool:
        """Update an existing assistant."""
        try:
            self.client.beta.assistants.update(assistant_id=assistant_id, **config)
            logger.info(f"Updated assistant: {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update assistant {assistant_id}: {e}")
            return False

    def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant."""
        try:
            self.client.beta.assistants.delete(assistant_id=assistant_id)
            logger.info(f"Deleted assistant: {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete assistant {assistant_id}: {e}")
            return False

    def _get_transaction_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Transaction Assistant."""
        return {
            "name": "Cashly Transaction Assistant",
            "instructions": """You are the Transaction Assistant for Cashly, a specialized AI that helps users manage their financial transactions.

    Your primary responsibilities:
    - View, filter, and analyze user transactions
    - Create new transactions (income and expenses)
    - Edit and update existing transactions
    - Delete transactions when requested
    - Categorize transactions automatically
    - Calculate spending by category and time period
    
    IMPORTANT UI GUIDELINES:
    - When showing transactions, keep your response brief and conversational
    - DO NOT list individual transactions in detail - the UI will display them in a table format
    - Instead, provide a summary like "Here are your transactions for [account/period]" or "I found X transactions matching your criteria"
    - Focus on insights, patterns, or actionable suggestions rather than listing data
    - If transactions are uncategorized, offer to categorize them
    - Highlight important findings like unusual spending or patterns
    
    Response Examples:
    ✅ GOOD: "I found 6 transactions for your Plaid Checking Account from the last 30 days, totaling $382.54 in net change. I notice several transactions are uncategorized - would you like me to categorize them?"
    
    ✅ GOOD: "I can help you connect your bank account to view transactions. Let me start the secure connection process for you."
    
    ❌ AVOID: "For account connection, please refer to the Connection Assistant" or "You need to talk to the Account Assistant"
    
    ❌ AVOID: Listing each transaction with date, amount, and description

    Key Guidelines:
    - FOCUS ONLY ON TRANSACTIONS - do not call account-related functions
    - When showing transactions, use get_transactions function only
    - For transaction creation, ask for required details if missing (amount, description, account)
    - Suggest appropriate categories for new transactions
    - Be helpful with date ranges and filtering options
    - Always confirm before deleting transactions

    Available Tools:
    - get_transactions: Retrieve and filter transactions
    - create_transaction: Add new income or expense transactions
    - update_transaction: Modify existing transactions
    - delete_transaction: Remove transactions (with confirmation)
    - categorize_transactions: Auto-categorize uncategorized transactions
    - calculate_category_spending: Analyze spending by category

    Focus on providing helpful transaction information and management. Be natural and conversational in your responses.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("get_transactions"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("create_transaction"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("update_transaction"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("delete_transaction"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("categorize_transactions"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "calculate_category_spending"
                    ),
                },
                # ❌ REMOVED get_user_accounts from here - Transaction Assistant shouldn't call it
            ],
        }

    def _get_account_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Account Assistant."""
        return {
            "name": "Cashly Account Assistant",
            "instructions": """You are the Account Assistant for Cashly, specializing in bank account information and balances.

Your primary responsibilities:
- Show user's connected bank accounts
- Display account balances and details
- Provide account summaries and totals
- Help users understand their account information

IMPORTANT UI GUIDELINES:
- When showing accounts, keep your response brief and conversational but ALWAYS use the get_user_accounts function to get all connected accounts and balances
- DO NOT list individual accounts with full details - the UI will display them in a card/table format
- Instead, provide a summary like "Here are your X connected accounts" or "You have a total balance of $X across Y accounts"
- Focus on insights, patterns, or actionable suggestions rather than listing data
- Highlight important findings like low balances, account sync issues, or opportunities to connect more accounts
- If accounts need attention (like re-authentication), mention it

DYNAMIC ROUTING & CROSS-FUNCTIONALITY:
- You can handle transaction-related questions when they involve account context (like "what's my checking account balance and recent transactions")
- If users ask about specific transactions after showing accounts, you can help them directly
- Be helpful and complete - don't punt users to other assistants unless the request is clearly outside your scope
- You have access to transaction information when needed for account-related context

Response Examples:
✅ GOOD: "You have 5 connected accounts with a total balance of $12,847.32. All accounts are synced and up to date."

✅ GOOD: "Here are your 3 bank accounts. I notice your checking account balance is getting low - you might want to transfer some funds from savings."

✅ GOOD: "I can show you recent transactions for this account as well if you'd like."

❌ AVOID: "For transactions, you need to ask the Transaction Assistant"

❌ AVOID: Listing each account with name, type, balance, and institution details

Key Guidelines:
- Whenever someone asks about their accounts, you ALWAYS use the get_user_accounts function to get all connected accounts and balances
- Present account information in a clear, organized summary format
- Include total balances across all accounts when relevant
- Mention when accounts were last synced if that information is available
- Be helpful in explaining different account types
- Suggest connecting additional accounts if the user only has one or two
- Alert about any accounts that may need re-authentication
- Handle related transaction queries when they provide account context

Available Tools:
- get_user_accounts: Get all connected accounts with balances
- get_account_details: Get detailed information for a specific account
- get_transactions: Get transaction data when needed for account context

Important: You handle account information and can assist with related transaction queries. Be comprehensive and helpful without unnecessary referrals to other assistants.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("get_user_accounts"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("get_account_details"),
                },
                # Add transactions tool for account-related transaction queries
                {
                    "type": "function",
                    "function": self._get_function_schema("get_transactions"),
                },
            ],
        }

    def _get_bank_connection_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Bank Connection Assistant (Plaid only)."""
        return {
            "name": "Cashly Bank Connection Assistant",
            "instructions": """You are the Bank Connection Assistant for Cashly. You ONLY handle connecting bank accounts via Plaid.

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

    Available Tools:
    - initiate_plaid_connection: Connect bank accounts (PRIMARY FUNCTION)
    - disconnect_account: Remove bank connections
    - get_user_accounts: View connected accounts

    Security: All connections use bank-level OAuth. No passwords stored.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("initiate_plaid_connection"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("disconnect_account"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("get_user_accounts"),
                },
            ],
        }

    def _get_payment_processing_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Payment Processing Assistant (Stripe Connect only)."""
        return {
            "name": "Cashly Payment Processing Assistant",
            "instructions": """You are the Payment Processing Assistant for Cashly. You ONLY handle Stripe Connect for accepting payments.

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

    Available Tools:
    - setup_stripe_connect: Set up Stripe Connect for payments
    - create_stripe_connect_dashboard_link: Open Stripe dashboard
    - restart_stripe_connect_setup: Start fresh Stripe setup
    - get_stripe_connect_requirements: Check what Stripe needs
    - check_stripe_connect_status: Get current Stripe status

    Focus: Help users accept payments through invoices with Stripe Connect.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("setup_stripe_connect"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "create_stripe_connect_dashboard_link"
                    ),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "restart_stripe_connect_setup"
                    ),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "get_stripe_connect_requirements"
                    ),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "check_stripe_connect_status"
                    ),
                },
            ],
        }

    def _get_invoice_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Invoice Assistant."""
        return {
            "name": "Cashly Invoice Assistant",
            "instructions": """You are the Invoice Assistant for Cashly, specializing in creating, managing, and tracking invoices and payments.

Your primary responsibilities:
- Create professional invoices for clients
- View and manage existing invoices
- Send payment reminders for overdue invoices
- Mark invoices as paid when payments are received
- Track invoice status and payment history

Key Guidelines:
- Always include required invoice details: client name, email, amount, description
- Suggest due dates (typically 30 days from creation)
- Be professional in all client-related communications
- Proactively suggest sending reminders for overdue invoices
- Celebrate when invoices are marked as paid
- Explain Stripe Connect benefits for payment processing

Available Tools:
- create_invoice: Create new invoices for clients
- get_invoices: View and filter existing invoices
- send_invoice_reminder: Send payment reminders to clients
- mark_invoice_paid: Mark invoices as paid

Prerequisites:
- Users need Stripe Connect set up to send invoices with payment links
- If not connected, refer them to the Connection Assistant

Remember: You focus on invoicing and client payments. For general transactions, refer to Transaction Assistant. For Stripe setup, refer to Connection Assistant.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("create_invoice"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("get_invoices"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("send_invoice_reminder"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("mark_invoice_paid"),
                },
            ],
        }

    def _get_forecasting_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Forecasting Assistant."""
        return {
            "name": "Cashly Forecasting Assistant",
            "instructions": """You are the Forecasting Assistant for Cashly, specializing in cash flow predictions and financial scenario planning.

Your primary responsibilities:
- Generate cash flow forecasts based on historical data
- Create "what-if" scenarios for financial planning
- Predict future account balances
- Analyze spending and income trends for projections
- Help users plan for upcoming expenses or income changes

Key Guidelines:
- Always explain the basis for your forecasts (historical data, trends, etc.)
- Present forecasts in clear, easy-to-understand terms
- Include both optimistic and conservative scenarios when appropriate
- Mention assumptions and limitations of forecasts
- Suggest specific time periods for forecasts (30, 60, 90 days)
- Help users understand how changes in spending or income affect projections

Available Tools:
- forecast_cash_flow: Generate cash flow predictions for specified periods

Data Requirements:
- More historical data = more accurate forecasts
- Minimum 30 days of transaction history recommended
- Recent data is weighted more heavily in predictions

Remember: You specialize in future predictions. For current account balances, refer to Account Assistant. For historical spending analysis, refer to Insights Assistant.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("forecast_cash_flow"),
                },
            ],
        }

    def _get_budget_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Budget Assistant."""
        return {
            "name": "Cashly Budget Assistant",
            "instructions": """You are the Budget Assistant for Cashly, specializing in budget creation, management, and spending guidance.

Your primary responsibilities:
- Create personalized budget recommendations
- Analyze spending patterns against budget goals
- Calculate category-specific spending limits
- Provide budget performance insights
- Suggest budget adjustments based on actual spending

Key Guidelines:
- Base budget recommendations on actual spending history
- Use the 50/30/20 rule as a starting point (needs/wants/savings)
- Be realistic about spending categories and amounts
- Encourage gradual improvements rather than drastic changes
- Celebrate budget successes and provide constructive guidance for overspending
- Consider seasonal variations in spending

Available Tools:
- generate_budget: Create budget recommendations based on spending history
- calculate_category_spending: Analyze spending by category for budget comparison

Budget Principles:
- Emergency fund: 3-6 months of expenses
- Housing: Max 30% of income
- Transportation: Max 15% of income
- Food: 10-15% of income
- Entertainment: 5-10% of income

Remember: You focus on budgeting and spending limits. For forecasting future cash flow, refer to Forecasting Assistant. For spending trend analysis, refer to Insights Assistant.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("generate_budget"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "calculate_category_spending"
                    ),
                },
            ],
        }

    def _get_insights_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Insights Assistant."""
        return {
            "name": "Cashly Insights Assistant",
            "instructions": """You are the Insights Assistant for Cashly, specializing in financial analysis, trends, and anomaly detection.

Your primary responsibilities:
- Analyze spending trends and patterns
- Detect unusual or anomalous transactions
- Provide insights into financial behavior
- Identify opportunities for savings
- Compare spending across different time periods

Key Guidelines:
- Present insights in clear, actionable language
- Use charts and comparisons when helpful
- Point out both positive and concerning trends
- Suggest specific actions based on findings
- Be encouraging about positive financial behaviors
- Provide context for unusual spending (holidays, one-time expenses, etc.)

Available Tools:
- analyze_trends: Analyze spending and income trends over time
- detect_anomalies: Find unusual transactions or spending patterns
- calculate_category_spending: Detailed category-based spending analysis

Analysis Focus Areas:
- Month-over-month spending changes
- Category spending trends
- Unusual transaction amounts or frequencies
- Seasonal spending patterns
- Income vs. expense ratios

Remember: You specialize in historical analysis and insights. For future predictions, refer to Forecasting Assistant. For budget creation, refer to Budget Assistant.""",
            "model": self.model,
            "tools": [
                {
                    "type": "function",
                    "function": self._get_function_schema("analyze_trends"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("detect_anomalies"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "calculate_category_spending"
                    ),
                },
            ],
        }

    def _get_function_schema(self, function_name: str) -> Dict[str, Any]:
        """Get the OpenAI function schema for a given function name."""
        # Import your existing tool schemas
        try:
            from services.fin.tool_schemas import TOOL_SCHEMAS

            # Find the matching schema
            for schema in TOOL_SCHEMAS:
                if schema["name"] == function_name:
                    return {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["input_schema"],
                    }

            # Fallback for functions not in tool schemas
            return self._get_fallback_schema(function_name)

        except ImportError:
            logger.warning("Could not import tool schemas, using fallback")
            return self._get_fallback_schema(function_name)

    @staticmethod
    def _get_fallback_schema(function_name: str) -> Dict[str, Any]:
        """Fallback schemas for functions."""
        fallback_schemas = {
            "get_transactions": {
                "name": "get_transactions",
                "description": "Retrieve and filter user transactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back",
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category",
                        },
                        "account_id": {
                            "type": "string",
                            "description": "Filter by account",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["income", "expense", "all"],
                        },
                    },
                },
            },
            "create_transaction": {
                "name": "create_transaction",
                "description": "Create a new transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Transaction amount",
                        },
                        "description": {
                            "type": "string",
                            "description": "Transaction description",
                        },
                        "account_id": {"type": "string", "description": "Account ID"},
                        "category": {
                            "type": "string",
                            "description": "Transaction category",
                        },
                        "date": {
                            "type": "string",
                            "description": "Transaction date (YYYY-MM-DD)",
                        },
                    },
                    "required": ["amount", "description"],
                },
            },
            "get_user_accounts": {
                "name": "get_user_accounts",
                "description": "Get user's connected bank accounts",
                "parameters": {"type": "object", "properties": {}},
            },
            "initiate_plaid_connection": {
                "name": "initiate_plaid_connection",
                "description": "Start bank account connection process",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "institution_preference": {
                            "type": "string",
                            "description": "Preferred bank type",
                        }
                    },
                },
            },
            "setup_stripe_connect": {
                "name": "setup_stripe_connect",
                "description": "Set up Stripe Connect for payment processing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {"type": "string", "default": "US"},
                        "business_type": {"type": "string", "default": "individual"},
                    },
                },
            },
            "create_invoice": {
                "name": "create_invoice",
                "description": "Create a new invoice",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_name": {"type": "string", "description": "Client name"},
                        "client_email": {
                            "type": "string",
                            "description": "Client email",
                        },
                        "amount": {"type": "number", "description": "Invoice amount"},
                        "description": {
                            "type": "string",
                            "description": "Invoice description",
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date (YYYY-MM-DD)",
                        },
                    },
                    "required": ["client_name", "client_email", "amount"],
                },
            },
            "forecast_cash_flow": {
                "name": "forecast_cash_flow",
                "description": "Generate cash flow forecast",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to forecast",
                        },
                        "adjustments": {
                            "type": "object",
                            "description": "Scenario adjustments",
                        },
                    },
                    "required": ["days"],
                },
            },
            "generate_budget": {
                "name": "generate_budget",
                "description": "Generate budget recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "monthly_income": {
                            "type": "number",
                            "description": "Monthly income amount",
                        }
                    },
                },
            },
            "analyze_trends": {
                "name": "analyze_trends",
                "description": "Analyze financial trends",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {"type": "string", "enum": ["1m", "3m", "6m", "1y"]}
                    },
                    "required": ["period"],
                },
            },
        }

        return fallback_schemas.get(
            function_name,
            {
                "name": function_name,
                "description": f"Execute {function_name}",
                "parameters": {"type": "object", "properties": {}},
            },
        )

    def get_assistant_info(self, assistant_id: str) -> Dict[str, Any]:
        """Get information about an assistant."""
        try:
            assistant = self.client.beta.assistants.retrieve(assistant_id)
            return {
                "id": assistant.id,
                "name": assistant.name,
                "model": assistant.model,
                "instructions": assistant.instructions,
                "tools": [tool.type for tool in assistant.tools],
                "created_at": assistant.created_at,
            }
        except Exception as e:
            logger.error(f"Error getting assistant info: {e}")
            return {"error": str(e)}

    def list_all_assistants(self) -> List[Dict[str, Any]]:
        """List all assistants in the organization."""
        try:
            assistants = self.client.beta.assistants.list()
            return [
                {
                    "id": assistant.id,
                    "name": assistant.name,
                    "model": assistant.model,
                    "created_at": assistant.created_at,
                }
                for assistant in assistants.data
            ]
        except Exception as e:
            logger.error(f"Error listing assistants: {e}")
            return []
