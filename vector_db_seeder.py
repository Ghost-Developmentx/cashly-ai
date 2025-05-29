"""
Seed the vector database with sample conversations for intent classification.
Run this script to populate initial training data.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from services.embeddings.openai_client import OpenAIEmbeddingClient
from services.embeddings.storage import EmbeddingStorage
from services.embeddings.context_builder import ConversationContextBuilder

logger = logging.getLogger(__name__)


class VectorDatabaseSeeder:
    """Seeds the vector database with sample conversation data."""

    def __init__(self):
        self.embedding_client = OpenAIEmbeddingClient()
        self.storage = EmbeddingStorage()
        self.context_builder = ConversationContextBuilder()

    def seed_database(self, user_id: str = "seed_user") -> bool:
        """
        Seed the database with sample conversations.

        Args:
            user_id: User ID to associate with seed data

        Returns:
            True if seeding was successful
        """
        logger.info("Starting vector database seeding...")

        # Define sample conversations for each intent
        sample_conversations = self._get_sample_conversations()

        seeded_count = 0
        for intent, conversations in sample_conversations.items():
            logger.info(
                f"Seeding {len(conversations)} conversations for intent: {intent}"
            )

            for i, conversation in enumerate(conversations):
                conversation_id = f"seed_{intent}_{i}_{int(datetime.now().timestamp())}"

                if self._seed_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    conversation=conversation,
                    intent=intent,
                ):
                    seeded_count += 1

        logger.info(f"Successfully seeded {seeded_count} conversations")
        return seeded_count > 0

    def _seed_conversation(
        self,
        conversation_id: str,
        user_id: str,
        conversation: Dict[str, Any],
        intent: str,
    ) -> bool:
        """Seed a single conversation."""
        try:
            # Build context from conversation
            context_text = self._build_context_from_conversation(conversation)

            # Generate embedding
            embedding = self.embedding_client.create_embedding(context_text)
            if not embedding:
                logger.error(f"Failed to generate embedding for {conversation_id}")
                return False

            # Store embedding
            embedding_id = self.storage.store_embedding(
                conversation_id=conversation_id,
                user_id=user_id,
                embedding=embedding,
                intent=intent,
                assistant_type=f"{intent}_assistant",
                metadata={
                    "message_count": len(conversation["messages"]),
                    "seeded": True,
                    "seed_date": datetime.now().isoformat(),
                    "topics": conversation.get("topics", []),
                },
                success_indicator=conversation.get("success", True),
            )

            return embedding_id is not None

        except Exception as e:
            logger.error(f"Error seeding conversation {conversation_id}: {e}")
            return False

    @staticmethod
    def _build_context_from_conversation(conversation: Dict[str, Any]) -> str:
        """Build context text from conversation data."""
        messages = conversation["messages"]
        context_parts = []

        # Add user profile if available
        if conversation.get("user_context"):
            context_parts.append(f"User Profile: {conversation['user_context']}")

        # Add conversation flow
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            context_parts.append(f"{role.title()}: {content}")

        # Add topics
        if conversation.get("topics"):
            context_parts.append(f"Topics: {', '.join(conversation['topics'])}")

        return "\n\n".join(context_parts)

    @staticmethod
    def _get_sample_conversations() -> Dict[str, List[Dict[str, Any]]]:
        """Get sample conversations for each intent."""
        return {
            "transactions": [
                {
                    "messages": [
                        {"role": "user", "content": "Show me my recent transactions"},
                        {
                            "role": "assistant",
                            "content": "Here are your recent transactions from the past 30 days...",
                        },
                    ],
                    "topics": ["transactions", "recent", "history"],
                    "success": True,
                    "user_context": "User with 3 connected accounts",
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What did I spend on groceries last month?",
                        },
                        {
                            "role": "assistant",
                            "content": "You spent $342.50 on groceries last month...",
                        },
                    ],
                    "topics": ["spending", "groceries", "category"],
                    "success": True,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Can you add a transaction for $50 coffee expense today?",
                        },
                        {
                            "role": "assistant",
                            "content": "I've added a $50 coffee expense for today...",
                        },
                    ],
                    "topics": ["create", "expense", "manual"],
                    "success": True,
                },
            ],
            "accounts": [
                {
                    "messages": [
                        {"role": "user", "content": "What's my account balance?"},
                        {
                            "role": "assistant",
                            "content": "Your total balance across all accounts is $2,450.32...",
                        },
                    ],
                    "topics": ["balance", "accounts", "total"],
                    "success": True,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "How many bank accounts do I have connected?",
                        },
                        {
                            "role": "assistant",
                            "content": "You have 3 bank accounts connected...",
                        },
                    ],
                    "topics": ["accounts", "connected", "count"],
                    "success": True,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "I want to connect my Chase account",
                        },
                        {
                            "role": "assistant",
                            "content": "I'll help you connect your Chase account through Plaid...",
                        },
                    ],
                    "topics": ["connect", "plaid", "bank"],
                    "success": True,
                },
            ],
            "invoices": [
                {
                    "messages": [
                        {"role": "user", "content": "Show me all my invoices"},
                        {
                            "role": "assistant",
                            "content": "Here are all your invoices...",
                        },
                    ],
                    "topics": ["invoices", "list", "all"],
                    "success": True,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Create an invoice for John Doe for $500",
                        },
                        {
                            "role": "assistant",
                            "content": "I've created an invoice for John Doe for $500...",
                        },
                    ],
                    "topics": ["create", "invoice", "client"],
                    "success": True,
                },
                {
                    "messages": [
                        {"role": "user", "content": "Send the invoice to my client"},
                        {
                            "role": "assistant",
                            "content": "I've sent the invoice to your client...",
                        },
                    ],
                    "topics": ["send", "invoice", "client"],
                    "success": True,
                },
            ],
            "forecasting": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What's my cash flow forecast for next month?",
                        },
                        {
                            "role": "assistant",
                            "content": "Based on your transaction history, here's your cash flow forecast...",
                        },
                    ],
                    "topics": ["forecast", "cash flow", "prediction"],
                    "success": True,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Predict my expenses for the next 90 days",
                        },
                        {
                            "role": "assistant",
                            "content": "Based on your spending patterns, here's your 90-day expense forecast...",
                        },
                    ],
                    "topics": ["predict", "expenses", "future"],
                    "success": True,
                },
            ],
            "budgets": [
                {
                    "messages": [
                        {"role": "user", "content": "Help me create a budget"},
                        {
                            "role": "assistant",
                            "content": "I'll help you create a budget based on your spending patterns...",
                        },
                    ],
                    "topics": ["budget", "create", "planning"],
                    "success": True,
                },
                {
                    "messages": [
                        {"role": "user", "content": "Am I staying within my budget?"},
                        {
                            "role": "assistant",
                            "content": "Let me check your budget performance...",
                        },
                    ],
                    "topics": ["budget", "tracking", "performance"],
                    "success": True,
                },
            ],
            "insights": [
                {
                    "messages": [
                        {"role": "user", "content": "Show me my spending trends"},
                        {
                            "role": "assistant",
                            "content": "Here are your spending trends over the past few months...",
                        },
                    ],
                    "topics": ["trends", "analysis", "spending"],
                    "success": True,
                },
                {
                    "messages": [
                        {"role": "user", "content": "Analyze my financial patterns"},
                        {
                            "role": "assistant",
                            "content": "Based on your transaction data, here are your financial patterns...",
                        },
                    ],
                    "topics": ["analysis", "patterns", "insights"],
                    "success": True,
                },
            ],
        }


def main():
    """Main seeding function."""
    seeder = VectorDatabaseSeeder()

    # Seed for multiple users to create variety
    users = ["seed_user_1", "seed_user_2", "seed_user_3"]

    total_seeded = 0
    for user_id in users:
        if seeder.seed_database(user_id):
            total_seeded += 1

    print(f"✅ Successfully seeded database for {total_seeded}/{len(users)} users")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
