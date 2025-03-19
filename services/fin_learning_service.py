import os
import json
import logging
import anthropic
import datetime
import re
import numpy as np
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class FinLearningService:
    """
    Service to analyze feedback and improve Fin's capabilities over time.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.system_prompt_template = self._load_system_prompt_template()
        self.example_exchanges = self._load_example_exchanges()

    def process_learning_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a dataset of successful conversations to improve Fin.

        Args:
            dataset: List of conversation data with messages and metadata

        Returns:
            Processing results and status
        """
        if not dataset:
            return {"status": "error", "message": "Empty dataset provided"}

        logger.info(f"Processing learning dataset with {len(dataset)} conversations")

        # Extract useful patterns
        patterns = self._extract_patterns(dataset)

        # Update the system prompt with new patterns
        self._update_system_prompt(patterns)

        # Extract example exchanges for few-shot learning
        new_examples = self._extract_example_exchanges(dataset)
        self._update_example_exchanges(new_examples)

        # In a full implementation, we would also:
        # 1. Create a fine-tuning dataset for Claude
        # 2. Update tool definitions based on usage patterns
        # 3. Update function signatures and parameters

        return {
            "status": "success",
            "processed_conversations": len(dataset),
            "extracted_patterns": len(patterns),
            "new_examples": len(new_examples),
        }

    @staticmethod
    def _load_system_prompt_template() -> str:
        """Load the system prompt template from file or use default"""
        try:
            with open("prompts/fin_system_prompt.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Default system prompt if file doesn't exist
            return """
            You are Fin, an AI-powered financial assistant for the Cashly app.

            Your role is to help users understand their finances, answer questions about their spending,
            income, budgets, and provide forecasts and financial advice.

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

    @staticmethod
    def _load_example_exchanges() -> List[Dict[str, Any]]:
        """Load example exchanges from file or use defaults"""
        try:
            with open("prompts/fin_examples.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default examples if file doesn't exist or is invalid
            return []

    def _update_system_prompt(self, patterns: List[Dict[str, Any]]) -> None:
        """Update the system prompt based on extracted patterns"""
        # Extract query patterns
        query_patterns = [p for p in patterns if p["type"] == "query_pattern"]

        # Group by query type
        query_types = {}
        for pattern in query_patterns:
            query_type = pattern.get("query_type", "")
            if query_type:
                if query_type not in query_types:
                    query_types[query_type] = []
                query_types[query_type].append(pattern.get("example", ""))

        # Create new sections for the system prompt
        new_sections = []

        if query_types:
            new_sections.append(
                "\n\nThese are common financial questions you should be prepared to answer:"
            )

            for query_type, examples in query_types.items():
                example = examples[0] if examples else ""
                description = self._get_query_type_description(query_type)

                new_sections.append(f"- {description} (e.g., '{example}')")

        # Add tool success patterns
        tool_patterns = [p for p in patterns if p["type"] == "tool_success"]
        if tool_patterns:
            tool_mapping = {}
            for pattern in tool_patterns:
                tool = pattern.get("tool", "")
                if tool:
                    if tool not in tool_mapping:
                        tool_mapping[tool] = []
                    tool_mapping[tool].append(pattern.get("query_context", ""))

            new_sections.append(
                "\n\nFor these types of questions, use these specific tools:"
            )

            for tool, contexts in tool_mapping.items():
                context = contexts[0] if contexts else ""
                new_sections.append(
                    f"- Use '{tool}' when questions are like: '{context}'"
                )

        # Create updated system prompt
        updated_prompt = self.system_prompt_template

        # Add learning-derived sections
        if new_sections:
            updated_section = "\n".join(new_sections)

            # Check if we already have a LEARNING INSIGHTS section
            if "LEARNING INSIGHTS" in updated_prompt:
                # Replace existing section
                updated_prompt = re.sub(
                    r"(# LEARNING INSIGHTS\n)(.+?)(\n# |\Z)",
                    r"\1" + updated_section + r"\3",
                    updated_prompt,
                    flags=re.DOTALL,
                )
            else:
                # Add new section at the end
                updated_prompt += "\n\n# LEARNING INSIGHTS\n" + updated_section

        # Save updated prompt
        try:
            os.makedirs("prompts", exist_ok=True)
            with open("prompts/fin_system_prompt.txt", "w") as f:
                f.write(updated_prompt)

            self.system_prompt_template = updated_prompt
            logger.info("Updated system prompt template with new patterns")
        except Exception as e:
            logger.error(f"Failed to save updated system prompt: {str(e)}")

    def _extract_patterns(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract useful patterns from the dataset"""
        patterns = []

        for conversation in dataset:
            messages = conversation.get("messages", [])
            tools_used = conversation.get("tools_used", [])
            led_to_action = conversation.get("led_to_action", False)

            # Skip conversations that are too short
            if len(messages) < 2:
                continue

            # Group messages into exchanges (user question + assistant response)
            exchanges = []
            for i in range(0, len(messages) - 1, 2):
                if (
                    i + 1 < len(messages)
                    and messages[i]["role"] == "user"
                    and messages[i + 1]["role"] == "assistant"
                ):
                    exchanges.append(
                        {
                            "user": messages[i]["content"],
                            "assistant": messages[i + 1]["content"],
                            "tools": [
                                t
                                for t in tools_used
                                if t.get("timestamp", "")
                                > messages[i].get("timestamp", "")
                            ],
                        }
                    )

            # Analyze each exchange for patterns
            for exchange in exchanges:
                user_query = exchange["user"]
                assistant_response = exchange["assistant"]
                tools = exchange["tools"]

                # Look for financial query patterns
                financial_patterns = self._identify_financial_patterns(
                    user_query, assistant_response, tools
                )

                # If this exchange led to user action, it's especially valuable
                if led_to_action:
                    financial_patterns.append(
                        {
                            "type": "action_trigger",
                            "query": user_query,
                            "response_elements": self._extract_response_elements(
                                assistant_response
                            ),
                        }
                    )

                patterns.extend(financial_patterns)

        # Deduplicate patterns
        unique_patterns = []
        seen_patterns = set()

        for pattern in patterns:
            pattern_key = f"{pattern['type']}:{pattern.get('query_type', '')}"
            if pattern_key not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(pattern_key)

        return unique_patterns

    @staticmethod
    def _identify_financial_patterns(query, response, tools):
        """Identify patterns in financial queries and responses"""
        patterns = []

        # Check for specific financial query types
        query_lower = query.lower()

        # Spending questions
        if re.search(
            r"(spend|spent|spending|cost|costs|paid|pay)", query_lower
        ) and re.search(r"(last|this) (month|week|year)", query_lower):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "historical_spending",
                    "example": query,
                }
            )

        # Budget questions
        if re.search(r"budget", query_lower) and re.search(
            r"(create|make|suggest|recommend)", query_lower
        ):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "budget_creation",
                    "example": query,
                }
            )

        # Forecast questions
        if re.search(r"(forecast|project|predict|outlook|future)", query_lower):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "financial_forecast",
                    "example": query,
                }
            )

        # What-if scenario questions
        if re.search(r"what (if|would happen|would it look like)", query_lower):
            patterns.append(
                {
                    "type": "query_pattern",
                    "query_type": "what_if_scenario",
                    "example": query,
                }
            )

        # Tool effectiveness patterns
        for tool in tools:
            tool_name = tool.get("name", "")
            success = tool.get("success", False)

            if success:
                patterns.append(
                    {"type": "tool_success", "tool": tool_name, "query_context": query}
                )

        return patterns

    @staticmethod
    def _extract_response_elements(response):
        """Extract key elements from a successful response"""
        elements = []

        # Check for data points
        if re.search(r"\$[\d,.]+", response):
            elements.append("specific_amounts")

        # Check for time references
        if re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December|last month|this month|next month)",
            response,
        ):
            elements.append("time_references")

        # Check for comparisons
        if re.search(
            r"(more than|less than|compared to|versus|vs|increased|decreased)", response
        ):
            elements.append("comparisons")

        # Check for actionable advice
        if re.search(r"(recommend|suggest|advise|could|should|consider)", response):
            elements.append("actionable_advice")

        # Check for visualizations references
        if re.search(r"(chart|graph|visualization|diagram)", response):
            elements.append("visualization_references")

        return elements

    @staticmethod
    def _extract_example_exchanges(
        dataset: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract high-quality example exchanges from the dataset"""
        examples = []

        # Look for conversations with high ratings and that led to action
        for conversation in dataset:
            messages = conversation.get("messages", [])
            led_to_action = conversation.get("led_to_action", False)

            # Skip conversations that are too short
            if len(messages) < 2:
                continue

            # Only consider high-value conversations
            if not led_to_action:
                continue

            # Extract exchanges
            for i in range(0, len(messages) - 1, 2):
                if (
                    i + 1 < len(messages)
                    and messages[i]["role"] == "user"
                    and messages[i + 1]["role"] == "assistant"
                ):
                    examples.append(
                        {
                            "user": messages[i]["content"],
                            "assistant": messages[i + 1]["content"],
                        }
                    )

        # Take the top 10 most diverse examples
        if examples:
            # Very basic diversity check - just take different length questions
            examples.sort(key=lambda x: len(x["user"]))
            step = max(1, len(examples) // 10)
            diverse_examples = examples[::step][:10]
            return diverse_examples

        return []

    def _update_example_exchanges(self, new_examples: List[Dict[str, Any]]) -> None:
        """Update the example exchanges file with new examples"""
        if not new_examples:
            return

        # Combine with existing examples
        combined_examples = self.example_exchanges

        # Add new unique examples
        existing_user_queries = [ex["user"] for ex in combined_examples]
        for example in new_examples:
            if example["user"] not in existing_user_queries:
                combined_examples.append(example)
                existing_user_queries.append(example["user"])

        # Limit to a reasonable number
        if len(combined_examples) > 20:
            combined_examples = combined_examples[-20:]

        # Save to file
        try:
            os.makedirs("prompts", exist_ok=True)
            with open("prompts/fin_examples.json", "w") as f:
                json.dump(combined_examples, f, indent=2)

            self.example_exchanges = combined_examples
            logger.info(
                f"Updated example exchanges with {len(new_examples)} new examples"
            )
        except Exception as e:
            logger.error(f"Failed to save updated examples: {str(e)}")

    @staticmethod
    def _get_query_type_description(query_type: str) -> str:
        """Get a human-readable description for a query type"""
        descriptions = {
            "historical_spending": "Questions about past spending in specific categories or time periods",
            "budget_creation": "Requests to create or recommend budget allocations",
            "financial_forecast": "Questions about future financial projections",
            "what_if_scenario": "Hypothetical scenarios about financial changes",
            "savings_goals": "Questions about saving for specific goals",
            "investment_advice": "Questions about investment strategies",
            "expense_reduction": "Questions about reducing specific expenses",
        }

        return descriptions.get(query_type, query_type.replace("_", " ").title())
