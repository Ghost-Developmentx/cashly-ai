import os
import logging
import anthropic
import datetime
from typing import Dict, List, Any

import pandas as pd

from services.forecast_service import ForecastService
from services.categorize_service import CategorizationService
from services.budget_service import BudgetService
from services.insight_service import InsightService
from services.anomaly_service import AnomalyService

logger = logging.getLogger(__name__)


class FinService:
    """
    Financial AI Assistant service that uses Claude to interpret user queries and
    call appropriate financial services to answer questions.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.forecast_service = ForecastService()
        self.categorization_service = CategorizationService()
        self.budget_service = BudgetService()
        self.insight_service = InsightService()
        self.anomaly_service = AnomalyService()

    def process_query(
        self,
        user_id: str,
        query: str,
        transactions: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        user_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query using Claude and call appropriate financial services.

        Args:
            user_id: The user's ID
            query: The user's question
            transactions: List of user's transactions
            conversation_history: Previous messages in the conversation
            user_context: Additional context about the user (accounts, budget goals, etc.)

        Returns:
            Dictionary with the assistant's response and any actions to take
        """
        # Build prompt with context, conversation history, and tools/functions
        system_prompt = self._build_system_prompt(user_id, transactions, user_context)

        # Build messages from conversation history and current query
        messages = self._build_messages(query, conversation_history)

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=messages,
                extra_body={"tools": self._get_tools()},
            )

            # Process the response to extract text and any tool calls
            processed_response = self._process_response(
                response, user_id, transactions, user_context
            )
            return processed_response

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            return {
                "response_text": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "error": str(e),
            }

    def _build_system_prompt(
        self,
        user_id: str,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
    ) -> str:
        """Build the system prompt with all necessary context and instructions"""

        # Extract key financial metrics for prompt context
        financial_context = self._extract_financial_context(transactions, user_context)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Basic system prompt with Fin's personality and capabilities
        system_prompt = f"""
        You are Fin, an AI-powered financial assistant for the Cashly app. Today is {current_date}.

        Your role is to help users understand their finances, answer questions about their spending,
        income, budgets, and provide forecasts and financial advice.

        Important financial information about this user:
        - Total accounts: {financial_context.get('account_count', 'unknown')}
        - Current balance across all accounts: {financial_context.get('total_balance', 'unknown')}
        - Monthly income (avg. last 3 months): {financial_context.get('monthly_income', 'unknown')}
        - Monthly expenses (avg. last 3 months): {financial_context.get('monthly_expenses', 'unknown')}
        - Top spending categories: {', '.join(financial_context.get('top_categories', ['unknown']))}
        - Recurring expenses detected: {financial_context.get('recurring_expenses', 'unknown')}

        When answering:
        1. Be concise, friendly, and helpful
        2. For complex questions, use the appropriate tool to calculate the answer
        3. If the user asks a question that requires creating a forecast or scenario, use the forecast_cash_flow tool
        4. For category-based questions, analyze the transactions data
        5. For budget-related questions, use the budget tools
        6. When asked about trends or patterns, use the analyze_trends tool
        7. When the user asks about unusual spending or transactions, use the detect_anomalies tool

        IMPORTANT: Do not make up information. If you don't have enough data or the right tool to answer a question,
        tell the user you can't answer that question with the available data.

        Use your tools to generate accurate responses based on the user's financial data.
        """

        return system_prompt

    @staticmethod
    def _build_messages(
        query: str, conversation_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Build message list from conversation history and current query"""
        messages = []

        # Add conversation history if available
        if conversation_history:
            for message in conversation_history:
                messages.append(message)

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    @staticmethod
    def _extract_financial_context(
        transactions: List[Dict[str, Any]], user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key financial metrics from user data for context"""
        context = {}

        # Extract account information
        if user_context and "accounts" in user_context:
            context["account_count"] = len(user_context["accounts"])
            context["total_balance"] = sum(
                account.get("balance", 0) for account in user_context["accounts"]
            )
        else:
            context["account_count"] = "unknown"
            context["total_balance"] = "unknown"

        # Calculate income and expenses if there are transactions
        if transactions:
            # Get transactions from the last 3 months
            try:
                today = datetime.datetime.now().date()
                three_months_ago = today - datetime.timedelta(days=90)

                # Ensure dates are properly parsed
                recent_transactions = []
                for t in transactions:
                    try:
                        transaction_date = t["date"]
                        # Handle potential datetime objects
                        if isinstance(
                            transaction_date, (datetime.datetime, pd.Timestamp)
                        ):
                            transaction_date = transaction_date.strftime("%Y-%m-%d")

                        # Parse the date string
                        date_obj = datetime.datetime.strptime(
                            transaction_date, "%Y-%m-%d"
                        ).date()
                        if date_obj >= three_months_ago:
                            recent_transactions.append(t)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing transaction date: {e}")
                        continue

                # Calculate monthly income and expenses
                if recent_transactions:
                    monthly_income = (
                        sum(t["amount"] for t in recent_transactions if t["amount"] > 0)
                        / 3
                    )
                    monthly_expenses = (
                        sum(
                            abs(t["amount"])
                            for t in recent_transactions
                            if t["amount"] < 0
                        )
                        / 3
                    )

                    context["monthly_income"] = f"${monthly_income:.2f}"
                    context["monthly_expenses"] = f"${monthly_expenses:.2f}"

                    # Get top spending categories
                    category_spend = {}
                    for t in recent_transactions:
                        if t["amount"] < 0:
                            category = t.get("category", "Uncategorized")
                            category_spend[category] = category_spend.get(
                                category, 0
                            ) + abs(t["amount"])

                    top_categories = sorted(
                        category_spend.items(), key=lambda x: x[1], reverse=True
                    )
                    context["top_categories"] = (
                        [c[0] for c in top_categories[:3]]
                        if top_categories
                        else ["unknown"]
                    )

                    # Detect recurring expenses (simplified)
                    recurring_count = sum(
                        1 for t in recent_transactions if t.get("recurring", False)
                    )
                    context["recurring_expenses"] = (
                        f"${recurring_count} recurring items detected"
                    )
                else:
                    context["monthly_income"] = "unknown"
                    context["monthly_expenses"] = "unknown"
                    context["top_categories"] = ["unknown"]
                    context["recurring_expenses"] = "unknown"
            except Exception as e:
                logger.error(f"Error extracting financial context: {str(e)}")
                context["monthly_income"] = "unknown"
                context["monthly_expenses"] = "unknown"
                context["top_categories"] = ["unknown"]
                context["recurring_expenses"] = "unknown"

        return context

    @staticmethod
    def _get_tools() -> List[Dict[str, Any]]:
        """Define tools that Claude can use to answer financial questions"""
        tools = [
            {
                "name": "forecast_cash_flow",
                "description": "Create a cash flow forecast based on historical transactions and optional adjustments",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to forecast",
                        },
                        "adjustments": {
                            "type": "object",
                            "description": "Optional adjustments to the forecast",
                            "properties": {
                                "income_adjustment": {
                                    "type": "number",
                                    "description": "Amount to adjust monthly income by (positive or negative)",
                                },
                                "expense_adjustment": {
                                    "type": "number",
                                    "description": "Amount to adjust monthly expenses by (positive or negative)",
                                },
                                "category_adjustments": {
                                    "type": "object",
                                    "description": "Category-specific adjustments",
                                    "additionalProperties": {"type": "number"},
                                },
                            },
                        },
                    },
                    "required": ["days"],
                },
            },
            {
                "name": "analyze_trends",
                "description": "Analyze financial trends and patterns in transactions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "description": "Time period for analysis",
                            "enum": ["1m", "3m", "6m", "1y"],
                        }
                    },
                    "required": ["period"],
                },
            },
            {
                "name": "detect_anomalies",
                "description": "Detect unusual or anomalous transactions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "description": "Anomaly score threshold",
                        }
                    },
                },
            },
            {
                "name": "generate_budget",
                "description": "Generate budget recommendations based on spending patterns",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "monthly_income": {
                            "type": "number",
                            "description": "Monthly income amount (if different from calculated)",
                        }
                    },
                },
            },
            {
                "name": "calculate_category_spending",
                "description": "Calculate spending in specific categories over a time period",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of categories to analyze",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in ISO format (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in ISO format (YYYY-MM-DD)",
                        },
                    },
                    "required": ["categories"],
                },
            },
        ]

        return tools

    def _process_response(self, response, user_id, transactions, user_context):
        """Process Claude's response, handling any tool calls and formatting the final response"""
        # Get the text content
        content = response.content[0].text if hasattr(response, "content") else ""

        # Remove thinking tags if present
        if "<thinking>" in content and "</thinking>" in content:
            thinking_start = content.find("<thinking>")
            thinking_end = content.find("</thinking>") + len("</thinking>")
            thinking_content = content[thinking_start:thinking_end]
            content = content.replace(thinking_content, "").strip()

        result = {"response_text": content, "actions": []}

        # Check if there are any tool calls
        if (
            hasattr(response, "content")
            and hasattr(response.content[0], "tool_calls")
            and response.content[0].tool_calls
        ):
            tool_results = []

            for tool_call in response.content[0].tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.input

                # Execute the appropriate tool call
                tool_result = self._execute_tool_call(
                    tool_name, tool_args, user_id, transactions, user_context
                )

                tool_results.append(
                    {"tool": tool_name, "result": tool_result, "parameters": tool_args}
                )

                # Add any UI actions needed
                if tool_name == "forecast_cash_flow":
                    result["actions"].append(
                        {"type": "show_forecast", "data": tool_result}
                    )
                elif tool_name == "analyze_trends":
                    result["actions"].append(
                        {"type": "show_trends", "data": tool_result}
                    )
                elif tool_name == "generate_budget":
                    result["actions"].append(
                        {"type": "show_budget", "data": tool_result}
                    )
                elif tool_name == "calculate_category_spending":
                    result["actions"].append(
                        {"type": "show_categories", "data": tool_result}
                    )

            # Add tool results to the response
            result["tool_results"] = tool_results

        return result

    def _execute_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_id: str,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a specific tool call with the provided arguments"""
        try:
            # Normalize all dates in transactions
            normalized_transactions = self._normalize_transaction_dates(transactions)

            if tool_name == "forecast_cash_flow":
                days = tool_args.get("days", 30)
                adjustments = tool_args.get("adjustments", {})

                # If there are adjustments, use the scenario forecast method
                if adjustments:
                    return self.forecast_service.forecast_cash_flow_scenario(
                        user_id=user_id,
                        transactions=normalized_transactions,
                        forecast_days=days,
                        adjustments=adjustments,
                    )
                else:
                    return self.forecast_service.forecast_cash_flow(
                        user_id=user_id,
                        transactions=normalized_transactions,
                        forecast_days=days,
                    )

            elif tool_name == "analyze_trends":
                period = tool_args.get("period", "3m")
                return self.insight_service.analyze_trends(
                    user_id=user_id, transactions=normalized_transactions, period=period
                )

            elif tool_name == "detect_anomalies":
                threshold = tool_args.get("threshold")
                return self.anomaly_service.detect_anomalies(
                    user_id=user_id,
                    transactions=normalized_transactions,
                    threshold=threshold,
                )

            elif tool_name == "generate_budget":
                # Calculate monthly income if not provided
                monthly_income = tool_args.get("monthly_income")
                if not monthly_income:
                    # Simple calculation of monthly income from recent transactions
                    today = datetime.datetime.now().date()
                    one_month_ago = today - datetime.timedelta(days=30)

                    recent_transactions = []
                    for t in normalized_transactions:
                        try:
                            date_obj = datetime.datetime.strptime(
                                t["date"], "%Y-%m-%d"
                            ).date()
                            if date_obj >= one_month_ago:
                                recent_transactions.append(t)
                        except (ValueError, TypeError):
                            continue

                    monthly_income = sum(
                        t["amount"] for t in recent_transactions if t["amount"] > 0
                    )

                return self.budget_service.generate_budget(
                    user_id=user_id,
                    transactions=normalized_transactions,
                    monthly_income=monthly_income,
                )

            elif tool_name == "calculate_category_spending":
                categories = tool_args.get("categories", [])
                start_date_str = tool_args.get("start_date")
                end_date_str = tool_args.get("end_date")

                # Parse dates or use defaults
                today = datetime.datetime.now().date()

                if start_date_str:
                    start_date = datetime.datetime.strptime(
                        start_date_str, "%Y-%m-%d"
                    ).date()
                else:
                    start_date = today - datetime.timedelta(days=30)

                if end_date_str:
                    end_date = datetime.datetime.strptime(
                        end_date_str, "%Y-%m-%d"
                    ).date()
                else:
                    end_date = today

                # Filter transactions by date and category
                filtered_txns = []
                for t in normalized_transactions:
                    try:
                        date_obj = datetime.datetime.strptime(
                            t["date"], "%Y-%m-%d"
                        ).date()
                        if start_date <= date_obj <= end_date:
                            filtered_txns.append(t)
                    except (ValueError, TypeError):
                        continue

                if categories:
                    # Case-insensitive category matching
                    filtered_txns = [
                        t
                        for t in filtered_txns
                        if any(
                            c.lower() in str(t.get("category", "")).lower()
                            for c in categories
                        )
                    ]

                # Calculate totals by category
                category_totals = {}
                for t in filtered_txns:
                    if t["amount"] < 0:  # Only include expenses
                        category = t.get("category", "Uncategorized")
                        category_totals[category] = category_totals.get(
                            category, 0
                        ) + abs(t["amount"])

                # Format the results
                result = {
                    "time_period": f"{start_date} to {end_date}",
                    "total_spending": sum(category_totals.values()),
                    "category_breakdown": category_totals,
                    "transaction_count": len(filtered_txns),
                }

                return result

            else:
                logger.warning(f"Unknown tool call: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def _normalize_transaction_dates(transactions):
        """Ensure all transaction dates are in YYYY-MM-DD format"""
        normalized = []

        for t in transactions:
            # Create a copy to avoid modifying the original
            t_copy = dict(t)

            try:
                if "date" not in t_copy:
                    continue

                # Convert to string if it's not already
                date_str = str(t_copy["date"])

                # CASE 1: Handle pandas Timestamp objects
                if isinstance(t_copy["date"], pd.Timestamp):
                    date_str = t_copy["date"].strftime("%Y-%m-%d")

                # CASE 2: Handle datetime objects
                elif isinstance(t_copy["date"], datetime.datetime):
                    date_str = t_copy["date"].strftime("%Y-%m-%d")

                # CASE 3: Handle string with UTC timezone
                elif " UTC" in date_str:
                    date_str = date_str.split(" UTC")[0]

                # CASE 4: Handle string with time component
                if " 00:00:00" in date_str:
                    date_str = date_str.split(" 00:00:00")[0]

                # CASE 5: Further clean any remaining time information
                if " " in date_str:
                    date_str = date_str.split(" ")[0]

                # Validate the resulting date format
                datetime.datetime.strptime(date_str, "%Y-%m-%d")

                # Update the transaction with cleaned date
                t_copy["date"] = date_str
                normalized.append(t_copy)
            except (KeyError, ValueError, TypeError) as e:
                # Skip transactions with invalid dates
                logger.warning(
                    f"Skipping transaction with invalid date: {t.get('date', 'No date')} - Error: {str(e)}"
                )
                continue

        return normalized
