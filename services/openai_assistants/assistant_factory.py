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
            AssistantType.CONNECTION: self._get_connection_assistant_config(),
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

Key Guidelines:
- Always be specific about transaction amounts, dates, and categories
- When showing transactions, include relevant details like date, amount, description, and category
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

Remember: You work specifically with transactions. For account balances, refer to the Account Assistant. For invoices, refer to the Invoice Assistant.""",
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

Key Guidelines:
- Present account information in a clear, organized format
- Include account names, types, balances, and institutions
- Calculate and show total balances across all accounts
- Mention when accounts were last synced if that information is available
- Be helpful in explaining different account types

Available Tools:
- get_user_accounts: Get all connected accounts with balances
- get_account_details: Get detailed information for a specific account

Important: You handle account information only. For connecting NEW accounts or Plaid/Stripe setup, refer users to the Connection Assistant. For transactions, refer to the Transaction Assistant.""",
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
            ],
        }

    def _get_connection_assistant_config(self) -> Dict[str, Any]:
        """Configuration for Connection Assistant."""
        return {
            "name": "Cashly Connection Assistant",
            "instructions": """You are the Connection Assistant for Cashly, specializing in setting up and managing integrations with banks, payment processors, and other financial services.

Your primary responsibilities:
- Help users connect bank accounts via Plaid
- Set up Stripe Connect for payment processing and invoicing
- Manage and disconnect existing integrations
- Troubleshoot connection issues
- Guide users through integration setup processes

Key Guidelines:
- Be encouraging and supportive during setup processes
- Explain the benefits of each integration clearly
- Address security concerns with reassurance about encryption and security
- Provide step-by-step guidance for connection processes
- Mention that connections can be disconnected at any time
- For Stripe Connect, explain the platform fee structure

Available Tools:
- initiate_plaid_connection: Start the bank account connection process
- disconnect_account: Remove connected bank accounts
- setup_stripe_connect: Set up Stripe Connect for payment processing
- check_stripe_connect_status: Check current Stripe Connect status
- connect_stripe: Connect basic Stripe account (for API keys)

Security Notes:
- All connections use bank-level encryption
- Cashly never stores banking passwords
- Users maintain full control over their connected accounts
- Data is only used to provide financial insights and services

Remember: You are the integration specialist. For account balances, refer to Account Assistant. For transactions, refer to Transaction Assistant.""",
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
                    "function": self._get_function_schema("setup_stripe_connect"),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema(
                        "check_stripe_connect_status"
                    ),
                },
                {
                    "type": "function",
                    "function": self._get_function_schema("connect_stripe"),
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
