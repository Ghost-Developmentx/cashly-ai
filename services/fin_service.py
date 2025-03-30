import json
import os
import logging
import re
import traceback

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
            logger.info(f"Calling Claude API with system: {system_prompt[:100]}...")
            logger.info(f"Messages: {messages}")

            # Initial API call
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=messages,
                extra_body={"tools": self._get_tools()},
            )

            # Process the response to extract text and any tool calls
            processed_response = self._process_response(
                response, user_id, transactions, user_context, system_prompt, messages
            )

            logger.info(f"Sending response: {processed_response}")
            return processed_response

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            logger.error(traceback.format_exc())
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
        
        IMPORTANT: When calculating time periods or date ranges, ALWAYS use {current_date} as the current date. 
        For example, "last 30 days" means from {(datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")} 
        to {current_date}.

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

        IMPORTANT INSTRUCTIONS FOR TOOL USAGE:
        When you use tools to answer questions, you MUST:
        - ALWAYS provide a natural language response after receiving tool results
        - Format your response in a friendly, conversational way
        - Include specific numbers and insights from the tool output
        - NEVER return an empty response after using a tool

        For specific tools:
        - For forecast_cash_flow: Mention both the current balance and the projected future balance, and explain the trend
        - For calculate_category_spending: List the top categories with their specific amounts
        - For analyze_trends: Highlight the key patterns and changes in spending behavior
        - For detect_anomalies: Explain unusual transactions in plain language
        - For generate_budget: Summarize the budget recommendations and how they relate to current spending

        EXAMPLE RESPONSE AFTER USING forecast_cash_flow:
        "Based on your transaction history, your current balance is $5,234.56. In 30 days, your projected balance will be $6,789.12. This positive trend indicates you're saving more than you're spending, which is great for your financial health!"

        EXAMPLE RESPONSE AFTER USING calculate_category_spending:
        "Looking at your spending this month, your top categories are:
        1. Housing: $1,200.00
        2. Dining: $450.75 
        3. Transportation: $325.50

        Dining expenses make up about 22% of your total spending this month."

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

    def _process_response(
        self, response, user_id, transactions, user_context, system_prompt, messages
    ):
        """Process Claude's response, handling any tool calls and formatting the final response"""
        # Log the raw response for debugging
        logger.info(f"Raw Claude response: {response}")

        # Initialize response structures
        result = {"response_text": "", "actions": []}
        tool_results = []

        # Process each content block
        has_tool_use = False

        if hasattr(response, "content"):
            for content_block in response.content:
                # Handle text blocks
                if hasattr(content_block, "type") and content_block.type == "text":
                    text = content_block.text or ""

                    # Remove thinking tags
                    if "<thinking>" in text and "</thinking>" in text:
                        thinking_start = text.find("<thinking>")
                        thinking_end = text.find("</thinking>") + len("</thinking>")
                        thinking_content = text[thinking_start:thinking_end]
                        text = text.replace(thinking_content, "").strip()

                    # Add text to response
                    if text:
                        result["response_text"] += text

                # Handle tool use blocks
                elif (
                    hasattr(content_block, "type") and content_block.type == "tool_use"
                ):
                    has_tool_use = True
                    tool_name = content_block.name
                    tool_input = content_block.input
                    tool_id = content_block.id

                    # Log the tool call
                    logger.info(
                        f"Claude is calling tool: {tool_name} with input: {tool_input}"
                    )

                    # Execute the tool call
                    tool_result = self._execute_tool_call(
                        tool_name, tool_input, user_id, transactions, user_context
                    )

                    # Store the result for later use
                    tool_results.append(
                        {
                            "tool": tool_name,
                            "result": tool_result,
                            "parameters": tool_input,
                            "id": tool_id,
                        }
                    )

                    # Add UI actions based on tool type
                    if tool_name == "calculate_category_spending":
                        result["actions"].append(
                            {"type": "show_categories", "data": tool_result}
                        )
                    elif tool_name == "forecast_cash_flow":
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
                    elif tool_name == "detect_anomalies":
                        result["actions"].append(
                            {"type": "show_anomalies", "data": tool_result}
                        )

        # If Claude used tools, make a follow-up API call with the tool results
        if has_tool_use:
            # Create follow-up messages
            tool_messages = messages.copy()

            # Add the assistant's message with tool use blocks
            tool_messages.append(
                {"role": "assistant", "content": [block for block in response.content]}
            )

            tool_result_blocks = []
            for tool_call in tool_results:
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps(tool_call["result"]),
                    }
                )

            # Add user message with tool results
            tool_messages.append({"role": "user", "content": tool_result_blocks})

            try:
                # Make the follow-up call to Claude with the tool results included
                logger.info(f"Making follow-up call to Claude with tool results")
                follow_up_response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    system=system_prompt,
                    messages=tool_messages,
                )

                logger.info(f"Raw follow-up response: {follow_up_response}")

                # Extract the final text response
                final_text = ""
                if hasattr(follow_up_response, "content"):
                    for content_block in follow_up_response.content:
                        if (
                            hasattr(content_block, "type")
                            and content_block.type == "text"
                        ):
                            text = content_block.text or ""

                            # Remove thinking tags
                            if "<thinking>" in text and "</thinking>" in text:
                                thinking_start = text.find("<thinking>")
                                thinking_end = text.find("</thinking>") + len(
                                    "</thinking>"
                                )
                                thinking_content = text[thinking_start:thinking_end]
                                text = text.replace(thinking_content, "").strip()

                            # Remove search quality reflection tags
                            if (
                                "<search_quality_reflection>" in text
                                and "</search_quality_reflection>" in text
                            ):
                                reflection_start = text.find(
                                    "<search_quality_reflection>"
                                )
                                reflection_end = text.find(
                                    "</search_quality_reflection>"
                                ) + len("</search_quality_reflection>")
                                reflection_content = text[
                                    reflection_start:reflection_end
                                ]
                                text = text.replace(reflection_content, "").strip()

                            # Remove search quality score tags
                            if (
                                "<search_quality_score>" in text
                                and "</search_quality_score>" in text
                            ):
                                score_start = text.find("<search_quality_score>")
                                score_end = text.find("</search_quality_score>") + len(
                                    "</search_quality_score>"
                                )
                                score_content = text[score_start:score_end]
                                text = text.replace(score_content, "").strip()

                            # Remove result tags
                            if "<result>" in text and "</result>" in text:
                                # Extract only the content inside the result tags, as this is the actual response
                                result_start = text.find("<result>") + len("<result>")
                                result_end = text.find("</result>")
                                text = text[result_start:result_end].strip()

                            # Remove any other XML-like tags we might encounter
                            text = re.sub(r"<[^>]+>", "", text)

                            final_text += text

                # Use the follow-up response text
                if final_text:
                    result["response_text"] = final_text
            except Exception as e:
                logger.error(f"Error making follow-up call to Claude: {str(e)}")
                logger.error(traceback.format_exc())
                # In case of error, we'll fall back to our generated response

        # If we still don't have a response text, generate fallbacks based on tool results
        if has_tool_use and not result["response_text"]:
            # For category spending results, build a nice response
            if (
                tool_results
                and tool_results[0]["tool"] == "calculate_category_spending"
            ):
                try:
                    category_data = tool_results[0]["result"]
                    if "category_breakdown" in category_data:
                        # Sort categories by spending amount (descending)
                        sorted_categories = sorted(
                            category_data["category_breakdown"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )

                        # Build response
                        response_text = "Based on your transactions this month, your top spending categories are:\n\n"

                        # Add top 5 categories
                        for i, (category, amount) in enumerate(
                            sorted_categories[:5], 1
                        ):
                            response_text += (
                                f"{i}. {category.title()}: ${abs(amount):.2f}\n"
                            )

                        result["response_text"] = response_text
                except Exception as e:
                    logger.error(f"Error building category response: {str(e)}")
                    result["response_text"] = (
                        "I analyzed your spending categories, but encountered an error formatting the results."
                    )

            # For forecast cash flow results
            elif tool_results and tool_results[0]["tool"] == "forecast_cash_flow":
                try:
                    forecast_data = tool_results[0]["result"]
                    if "insights" in forecast_data:
                        insights = forecast_data["insights"]
                        current_balance = float(insights.get("current_balance", 0))
                        projected_balance = float(insights.get("projected_balance", 0))
                        days = tool_results[0]["parameters"].get("days", 30)
                        trend = insights.get("cash_flow_trend", "")

                        response_text = f"Based on your transaction history, your current balance is ${current_balance:,.2f}. "
                        response_text += f"In {days} days, your projected balance will be ${projected_balance:,.2f}. "

                        if trend == "positive":
                            response_text += "This positive trend indicates you're saving more than you're spending, which is great for your financial health!"
                        elif trend == "negative":
                            response_text += "This negative trend indicates you're spending more than you're earning. You might want to review your budget."
                        else:
                            response_text += "I'll continue monitoring your cash flow to help you stay on track with your financial goals."

                        result["response_text"] = response_text
                except Exception as e:
                    logger.error(f"Error building forecast response: {str(e)}")
                    result["response_text"] = (
                        "I've calculated your future balance, but encountered an error formatting the results."
                    )

            # Add fallbacks for other tools here as needed
            elif tool_results:
                tool_name = tool_results[0]["tool"]
                result["response_text"] = (
                    f"I've analyzed your financial data using the {tool_name} tool. Please check the visualizations for detailed information."
                )

        # Add tool results to the response
        if tool_results:
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
            t_copy = dict(t)
            try:
                if "date" not in t_copy:
                    continue

                # Simply extract the first 10 characters (YYYY-MM-DD)
                # from any date string or use strftime for datetime objects
                date_val = t_copy["date"]
                if isinstance(date_val, str):
                    # Extract YYYY-MM-DD format (first 10 characters)
                    t_copy["date"] = date_val[:10]
                elif hasattr(date_val, "strftime"):
                    t_copy["date"] = date_val.strftime("%Y-%m-%d")

                normalized.append(t_copy)
            except Exception as e:
                logger.warning(f"Skipping transaction with invalid date: {str(e)}")
                continue

        return normalized
