import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationDataProcessor:
    """
    Process existing conversation data to create training datasets for intent classification.
    """

    def __init__(self):
        self.intent_mapping = {
            # Transaction-related patterns
            "transactions": [
                r"show.*transaction",
                r"view.*transaction",
                r"list.*transaction",
                r"add.*transaction",
                r"create.*transaction",
                r"new.*transaction",
                r"edit.*transaction",
                r"update.*transaction",
                r"change.*transaction",
                r"delete.*transaction",
                r"remove.*transaction",
                r"spent.*on",
                r"spending.*on",
                r"expenses.*for",
                r"categorize.*transaction",
                r"organize.*transaction",
                r"transaction.*history",
                r"recent.*transaction",
            ],
            # Account-related patterns
            "accounts": [
                r"connect.*account",
                r"link.*account",
                r"add.*bank",
                r"bank.*account",
                r"account.*balance",
                r"balance.*of",
                r"disconnect.*account",
                r"remove.*account",
                r"plaid.*connection",
                r"bank.*connection",
                r"my.*accounts",
                r"show.*accounts",
            ],
            # Invoice-related patterns
            "invoices": [
                r"create.*invoice",
                r"send.*invoice",
                r"new.*invoice",
                r"invoice.*for",
                r"bill.*client",
                r"payment.*request",
                r"send.*reminder",
                r"payment.*reminder",
                r"mark.*paid",
                r"invoice.*paid",
                r"stripe.*connect",
                r"payment.*processing",
            ],
            # Forecasting patterns
            "forecasting": [
                r"cash.*flow.*forecast",
                r"predict.*cash",
                r"future.*cash",
                r"forecast.*expenses",
                r"predict.*spending",
                r"what.*if",
                r"scenario.*planning",
                r"projection",
                r"future.*balance",
                r"projected.*balance",
            ],
            # Budget patterns
            "budgets": [
                r"create.*budget",
                r"budget.*plan",
                r"budget.*recommendations",
                r"spending.*limit",
                r"budget.*limit",
                r"expense.*limit",
                r"budget.*analysis",
                r"budget.*review",
            ],
            # Insights patterns
            "insights": [
                r"spending.*trend",
                r"financial.*trend",
                r"expense.*trend",
                r"analyze.*spending",
                r"financial.*analysis",
                r"unusual.*transaction",
                r"anomaly.*detection",
                r"category.*analysis",
                r"spending.*by.*category",
            ],
            # General patterns
            "general": [
                r"^hi$",
                r"^hello",
                r"^help",
                r"^thanks",
                r"financial.*advice",
                r"how.*to",
                r"what.*is",
                r"explain.*",
                r"can.*you.*help",
            ],
        }

    def extract_from_fin_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """
        Extract labeled training data from Fin conversation logs.

        Args:
            conversations: List of conversation dictionaries

        Returns:
            List of labeled examples
        """
        labeled_data = []

        for conv in conversations:
            conv_id = conv.get("id", "unknown")
            messages = conv.get("messages", [])

            for i, message in enumerate(messages):
                if message.get("role") == "user":
                    user_query = message.get("content", "").strip()
                    if not user_query:
                        continue

                    # Look at the assistant's response to infer intent
                    assistant_response = None
                    if (
                        i + 1 < len(messages)
                        and messages[i + 1].get("role") == "assistant"
                    ):
                        assistant_response = messages[i + 1].get("content", "")

                    # Classify based on user query and assistant response
                    intent = self._infer_intent_from_conversation(
                        user_query, assistant_response
                    )

                    labeled_data.append(
                        {
                            "conversation_id": conv_id,
                            "user_query": user_query,
                            "intent": intent,
                            "assistant_response": assistant_response,
                            "confidence": self._calculate_confidence(
                                user_query, intent
                            ),
                            "timestamp": message.get(
                                "timestamp", datetime.now().isoformat()
                            ),
                        }
                    )

        return labeled_data

    def _infer_intent_from_conversation(
        self, user_query: str, assistant_response: Optional[str] = None
    ) -> str:
        """
        Infer intent from user query and assistant response.
        """
        query_lower = user_query.lower()

        # Check direct pattern matches
        for intent, patterns in self.intent_mapping.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        # Check assistant response for action indicators
        if assistant_response:
            response_lower = assistant_response.lower()

            # Look for action indicators in response
            action_indicators = {
                "transactions": ["transaction", "spending", "expense", "income"],
                "accounts": ["account", "balance", "plaid", "bank"],
                "invoices": ["invoice", "payment", "stripe", "client"],
                "forecasting": ["forecast", "predict", "future", "cash flow"],
                "budgets": ["budget", "limit", "plan"],
                "insights": ["analysis", "trend", "insight", "pattern"],
            }

            for intent, indicators in action_indicators.items():
                if any(indicator in response_lower for indicator in indicators):
                    return intent

        # Default to general
        return "general"

    def _calculate_confidence(self, query: str, intent: str) -> float:
        """Calculate confidence score for the intent classification."""
        query_lower = query.lower()

        # Check how many patterns match for this intent
        if intent in self.intent_mapping:
            matches = sum(
                1
                for pattern in self.intent_mapping[intent]
                if re.search(pattern, query_lower)
            )
            if matches > 0:
                return min(0.6 + (matches * 0.15), 0.95)

        return 0.5  # Default confidence

    def process_tool_usage_data(self, tool_usage_stats: Dict) -> List[Dict]:
        """
        Process tool usage statistics to create training examples.

        Args:
            tool_usage_stats: Dictionary with tool usage data

        Returns:
            List of training examples
        """
        training_data = []

        # Map tools to intents
        tool_intent_mapping = {
            "get_transactions": "transactions",
            "create_transaction": "transactions",
            "update_transaction": "transactions",
            "delete_transaction": "transactions",
            "categorize_transactions": "transactions",
            "get_user_accounts": "accounts",
            "get_account_details": "accounts",
            "initiate_plaid_connection": "accounts",
            "disconnect_account": "accounts",
            "create_invoice": "invoices",
            "get_invoices": "invoices",
            "send_invoice_reminder": "invoices",
            "mark_invoice_paid": "invoices",
            "connect_stripe": "invoices",
            "forecast_cash_flow": "forecasting",
            "analyze_trends": "insights",
            "generate_budget": "budgets",
            "detect_anomalies": "insights",
            "calculate_category_spending": "insights",
        }

        for tool_name, stats in tool_usage_stats.items():
            intent = tool_intent_mapping.get(tool_name, "general")

            # Get context examples if available
            contexts = stats.get("contexts", [])
            parameters = stats.get("parameters", [])

            for i, context in enumerate(contexts):
                training_example = {
                    "user_query": context,
                    "intent": intent,
                    "tool_used": tool_name,
                    "confidence": 0.8,  # High confidence since we know the tool was used
                    "parameters": parameters[i] if i < len(parameters) else None,
                }
                training_data.append(training_example)

        return training_data

    def create_training_dataset(
        self,
        conversations: Optional[List[Dict]] = None,
        tool_usage: Optional[Dict] = None,
        output_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a comprehensive training dataset.

        Args:
            conversations: List of conversation data
            tool_usage: Tool usage statistics
            output_file: Optional file to save the dataset

        Returns:
            DataFrame with training data
        """
        all_data = []

        # Process conversations
        if conversations:
            conv_data = self.extract_from_fin_conversations(conversations)
            all_data.extend(conv_data)
            logger.info(f"Extracted {len(conv_data)} examples from conversations")

        # Process tool usage
        if tool_usage:
            tool_data = self.process_tool_usage_data(tool_usage)
            all_data.extend(tool_data)
            logger.info(f"Extracted {len(tool_data)} examples from tool usage")

        # Create DataFrame
        df = pd.DataFrame(all_data)

        if len(df) > 0:
            # Add some data quality metrics
            df["query_length"] = df["user_query"].str.len()
            df["word_count"] = df["user_query"].str.split().str.len()

            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=["user_query"], keep="first")
            logger.info(f"Removed {initial_count - len(df)} duplicate queries")

            # Filter out very short or very long queries
            df = df[(df["query_length"] >= 5) & (df["query_length"] <= 500)]

            # Save if requested
            if output_file:
                df.to_csv(output_file, index=False)
                logger.info(f"Saved training dataset to {output_file}")

        return df

    def analyze_intent_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of intents in the dataset."""
        if "intent" not in df.columns:
            return {}

        intent_counts = df["intent"].value_counts().to_dict()
        total_samples = len(df)

        analysis = {
            "total_samples": total_samples,
            "unique_intents": len(intent_counts),
            "intent_distribution": intent_counts,
            "intent_percentages": {
                intent: (count / total_samples) * 100
                for intent, count in intent_counts.items()
            },
            "min_samples_per_intent": (
                min(intent_counts.values()) if intent_counts else 0
            ),
            "max_samples_per_intent": (
                max(intent_counts.values()) if intent_counts else 0
            ),
        }

        return analysis

    def augment_training_data(
        self, df: pd.DataFrame, target_samples_per_intent: int = 50
    ) -> pd.DataFrame:
        """
        Augment training data to balance intent distribution.

        Args:
            df: Original training DataFrame
            target_samples_per_intent: Target number of samples per intent

        Returns:
            Augmented DataFrame
        """
        if "intent" not in df.columns:
            return df

        augmented_data = []

        for intent in df["intent"].unique():
            intent_data = df[df["intent"] == intent]
            current_count = len(intent_data)

            if current_count < target_samples_per_intent:
                # Need to augment this intent
                needed = target_samples_per_intent - current_count

                # Simple augmentation: rephrase existing queries
                for _ in range(needed):
                    original = intent_data.sample(1).iloc[0]
                    augmented_query = self._augment_query(original["user_query"])

                    augmented_example = original.copy()
                    augmented_example["user_query"] = augmented_query
                    augmented_example["confidence"] = max(
                        0.6, original.get("confidence", 0.7) - 0.1
                    )
                    augmented_example["augmented"] = True

                    augmented_data.append(augmented_example)

        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(
                f"Augmented dataset with {len(augmented_data)} additional samples"
            )

        return df

    def _augment_query(self, original_query: str) -> str:
        """Simple query augmentation by rephrasing."""
        query = original_query.strip()

        # Simple rephrasing patterns
        rephrase_patterns = [
            (r"^show me", "can you show me"),
            (r"^view", "I want to view"),
            (r"^create", "I need to create"),
            (r"^add", "please add"),
            (r"^help", "can you help me"),
            (r"my transactions", "the transactions"),
            (r"last month", "previous month"),
            (r"spending on", "expenses for"),
        ]

        # Apply one random rephrase pattern
        import random

        for pattern, replacement in rephrase_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return re.sub(pattern, replacement, query, flags=re.IGNORECASE)

        # If no pattern matches, add a prefix
        prefixes = ["Can you ", "Please ", "I want to ", "Help me "]
        return random.choice(prefixes) + query.lower()
