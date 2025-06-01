"""
Enhanced vector database seeder with comprehensive conversation data.
Uses sample conversations and variations to create robust training data.
"""

import logging
import random
from typing import List, Dict, Any
from datetime import datetime

# Import our conversation data
from seed_data.sample_conversations import get_all_sample_conversations
from seed_data.conversation_variations import ConversationVariations

from app.services import OpenAIEmbeddingClient
from app.services import EmbeddingStorage
from app.services.embeddings.context_builder import ConversationContextBuilder

logger = logging.getLogger(__name__)


class EnhancedVectorSeeder:
    """Enhanced seeder with comprehensive conversation data and variations."""

    def __init__(self):
        self.embedding_client = OpenAIEmbeddingClient()
        self.storage = EmbeddingStorage()
        self.context_builder = ConversationContextBuilder()
        self.variations_generator = ConversationVariations()

        # Batch processing settings
        self.batch_size = 10
        self.max_conversations_per_intent = 100

    def seed_comprehensive_database(self, user_ids: List[str] = None) -> Dict[str, int]:
        """
        Seed database with comprehensive conversation data.

        Args:
            user_ids: List of user IDs to seed for (defaults to generated IDs)

        Returns:
            Dictionary with seeding statistics
        """
        if not user_ids:
            user_ids = [f"seed_user_{i}" for i in range(1, 6)]  # 5 users

        logger.info(f"Starting comprehensive seeding for {len(user_ids)} users...")

        # Get base conversations
        base_conversations = get_all_sample_conversations()

        # Generate variations
        logger.info("Generating conversation variations...")
        all_conversations = self.variations_generator.generate_all_variations(
            base_conversations
        )

        # Seed statistics
        stats = {"total_seeded": 0, "by_intent": {}, "by_user": {}, "failed": 0}

        # Seed for each user
        for user_id in user_ids:
            logger.info(f"Seeding conversations for {user_id}...")
            user_stats = self._seed_user_conversations(user_id, all_conversations)

            stats["by_user"][user_id] = user_stats["total"]
            stats["total_seeded"] += user_stats["total"]
            stats["failed"] += user_stats["failed"]

            # Aggregate intent stats
            for intent, count in user_stats["by_intent"].items():
                stats["by_intent"][intent] = stats["by_intent"].get(intent, 0) + count

        logger.info("Comprehensive seeding completed!")
        self._log_final_stats(stats)

        return stats

    def _seed_user_conversations(
        self, user_id: str, all_conversations: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Seed conversations for a single user."""
        user_stats = {"total": 0, "by_intent": {}, "failed": 0}

        for intent, conversations in all_conversations.items():
            logger.info(f"  Seeding {intent} conversations...")

            # Limit conversations per intent to avoid overwhelming the system
            limited_conversations = self._limit_conversations(conversations, intent)

            # Add some randomization to make users unique
            randomized_conversations = self._randomize_for_user(
                limited_conversations, user_id
            )

            # Seed in batches
            intent_count = 0
            failed_count = 0

            for batch in self._create_batches(randomized_conversations):
                batch_success, batch_failed = self._seed_conversation_batch(
                    user_id, intent, batch
                )
                intent_count += batch_success
                failed_count += batch_failed

            user_stats["by_intent"][intent] = intent_count
            user_stats["total"] += intent_count
            user_stats["failed"] += failed_count

            logger.info(f"    Seeded {intent_count} {intent} conversations")

        return user_stats

    @staticmethod
    def _limit_conversations(conversations: List[Dict], intent: str) -> List[Dict]:
        """Limit conversations per intent to avoid too much data."""
        # More conversations for key intents
        limits = {
            "invoices": 80,
            "transactions": 80,
            "accounts": 60,
            "forecasting": 40,
            "budgets": 40,
            "insights": 40,
            "bank_connection": 30,
            "payment_processing": 30,
        }

        limit = limits.get(intent, 50)

        if len(conversations) > limit:
            # Randomly sample to maintain diversity
            return random.sample(conversations, limit)

        return conversations

    def _randomize_for_user(
        self, conversations: List[Dict], user_id: str
    ) -> List[Dict]:
        """Add user-specific randomization to conversations."""
        randomized = []

        # Set seed for consistency per user
        random.seed(hash(user_id) % 1000)

        for conv in conversations:
            randomized_conv = conv.copy()

            # Randomly adjust success indicators (90% success rate)
            randomized_conv["success"] = random.random() > 0.1

            # Add user-specific context
            if "user_context" not in randomized_conv:
                randomized_conv["user_context"] = self._generate_user_context(user_id)

            # Add random metadata
            randomized_conv["metadata"] = {
                "user_variation": user_id,
                "randomized": True,
                "seed_timestamp": datetime.now().isoformat(),
            }

            randomized.append(randomized_conv)

        # Reset random seed
        random.seed()

        return randomized

    @staticmethod
    def _generate_user_context(user_id: str) -> str:
        """Generate user-specific context."""
        user_types = [
            "Freelance designer with 3 accounts",
            "Small business owner with Stripe Connect",
            "Consultant with multiple clients",
            "Startup founder tracking expenses",
            "Agency owner managing cash flow",
        ]

        # Consistent context per user
        user_hash = hash(user_id) % len(user_types)
        return user_types[user_hash]

    def _create_batches(self, conversations: List[Dict]) -> List[List[Dict]]:
        """Create batches for processing."""
        batches = []
        for i in range(0, len(conversations), self.batch_size):
            batch = conversations[i : i + self.batch_size]
            batches.append(batch)
        return batches

    def _seed_conversation_batch(
        self, user_id: str, intent: str, batch: List[Dict]
    ) -> tuple:
        """Seed a batch of conversations."""
        success_count = 0
        failed_count = 0

        for i, conversation in enumerate(batch):
            conversation_id = (
                f"seed_{intent}_{user_id}_{int(datetime.now().timestamp())}_{i}"
            )

            try:
                if self._seed_single_conversation(
                    conversation_id, user_id, conversation, intent
                ):
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error seeding conversation {conversation_id}: {e}")
                failed_count += 1

        return success_count, failed_count

    def _seed_single_conversation(
        self, conversation_id: str, user_id: str, conversation: Dict, intent: str
    ) -> bool:
        """Seed a single conversation."""
        try:
            # Build context from conversation
            context_text = self._build_context_from_conversation(conversation)

            # Generate embedding
            embedding = self.embedding_client.create_embedding(context_text)
            if not embedding:
                return False

            # Prepare metadata
            metadata = {
                "message_count": len(conversation["messages"]),
                "seeded": True,
                "seed_date": datetime.now().isoformat(),
                "topics": conversation.get("topics", []),
                "user_context_type": conversation.get("user_context", ""),
                **conversation.get("metadata", {}),
            }

            # Store embedding
            embedding_id = self.storage.store_embedding(
                conversation_id=conversation_id,
                user_id=user_id,
                embedding=embedding,
                intent=intent,
                assistant_type=f"{intent}_assistant",
                metadata=metadata,
                success_indicator=conversation.get("success", True),
            )

            return embedding_id is not None

        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return False

    @staticmethod
    def _build_context_from_conversation(conversation: Dict) -> str:
        """Build context text from conversation data."""
        context_parts = []

        # Add user context if available
        if conversation.get("user_context"):
            context_parts.append(f"User Profile: {conversation['user_context']}")

        # Add conversation messages
        for msg in conversation["messages"]:
            role = msg["role"].title()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        # Add topics
        if conversation.get("topics"):
            context_parts.append(f"Topics: {', '.join(conversation['topics'])}")

        return "\n\n".join(context_parts)

    @staticmethod
    def _log_final_stats(stats: Dict[str, Any]):
        """Log final seeding statistics."""
        logger.info("=== Seeding Complete ===")
        logger.info(f"Total conversations seeded: {stats['total_seeded']}")
        logger.info(f"Failed conversations: {stats['failed']}")
        logger.info("\nBy Intent:")
        for intent, count in stats["by_intent"].items():
            logger.info(f"  {intent}: {count} conversations")
        logger.info("\nBy User:")
        for user_id, count in stats["by_user"].items():
            logger.info(f"  {user_id}: {count} conversations")

    def verify_seeding(self) -> Dict[str, Any]:
        """Verify the seeding was successful by testing some searches."""
        logger.info("Verifying seeding results...")

        test_queries = [
            ("Show me all my invoices", "invoices"),
            ("What's my account balance?", "accounts"),
            ("List recent transactions", "transactions"),
            ("Create a budget", "budgets"),
            ("Forecast cash flow", "forecasting"),
        ]

        results = {}

        for query, expected_intent in test_queries:
            try:
                # Generate embedding for test query
                embedding = self.embedding_client.create_embedding(query)
                if embedding:
                    # Search for similar conversations
                    similar = self.storage.find_similar_conversations(
                        embedding=embedding, limit=5, similarity_threshold=0.7
                    )

                    results[query] = {
                        "expected_intent": expected_intent,
                        "found_similar": len(similar),
                        "top_intent": similar[0]["intent"] if similar else None,
                        "top_similarity": similar[0]["similarity"] if similar else 0,
                    }

                    logger.info(
                        f"'{query}' -> Found {len(similar)} similar, top: {similar[0]['intent'] if similar else 'None'}"
                    )

            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                results[query] = {"error": str(e)}

        return results


def main():
    """Main function for comprehensive seeding."""
    logging.basicConfig(level=logging.INFO)

    seeder = EnhancedVectorSeeder()

    # Seed with comprehensive data
    stats = seeder.seed_comprehensive_database()

    # Verify results
    verification = seeder.verify_seeding()

    print("\n🎉 Enhanced Vector Database Seeding Complete!")
    print(f"📊 Total conversations seeded: {stats['total_seeded']}")
    print(f"❌ Failed: {stats['failed']}")
    print("\n📈 Verification Results:")
    for query, result in verification.items():
        if "error" not in result:
            print(f"  '{query}' -> {result['found_similar']} similar found")


if __name__ == "__main__":
    main()
