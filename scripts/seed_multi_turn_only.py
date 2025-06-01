# seed_data/seed_multi_turn_only.py

"""
Seeder specifically for multi-turn conversations.
This will NOT re-seed existing single-turn conversations.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from seed_data.multi_turn_conversations import MultiTurnConversations
from app.services import OpenAIEmbeddingClient
from app.services import EmbeddingStorage
from app.services.embeddings.context_builder import ConversationContextBuilder

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MultiTurnSeeder:
    """Seeds ONLY multi-turn conversations to avoid duplicating existing data."""

    def __init__(self):
        self.embedding_client = OpenAIEmbeddingClient()
        self.storage = EmbeddingStorage()
        self.context_builder = ConversationContextBuilder()
        self.batch_size = 5

    def seed_multi_turn_only(self, user_ids: List[str] = None) -> Dict[str, int]:
        """
        Seed ONLY multi-turn conversations.

        Args:
            user_ids: List of user IDs (defaults to same as original seeding)

        Returns:
            Dictionary with seeding statistics
        """
        if not user_ids:
            user_ids = [f"seed_user_{i}" for i in range(1, 6)]  # Same 5 users

        logger.info(f"Starting multi-turn seeding for {len(user_ids)} users...")

        # Get ONLY multi-turn conversations
        multi_turn_conversations = (
            MultiTurnConversations.get_all_multi_turn_conversations()
        )

        # Stats tracking
        stats = {
            "total_seeded": 0,
            "by_category": {},
            "by_user": {},
            "failed": 0,
            "embeddings_created": 0,
        }

        # Process each category
        for category, conversations in multi_turn_conversations.items():
            logger.info(f"Processing {category} multi-turn conversations...")
            category_count = 0

            for user_id in user_ids:
                user_count = self._seed_user_multi_turn(
                    user_id, category, conversations
                )

                category_count += user_count["conversations"]
                stats["embeddings_created"] += user_count["embeddings"]
                stats["failed"] += user_count["failed"]

                # Update user stats
                if user_id not in stats["by_user"]:
                    stats["by_user"][user_id] = 0
                stats["by_user"][user_id] += user_count["conversations"]

            stats["by_category"][category] = category_count
            stats["total_seeded"] += category_count

        logger.info("Multi-turn seeding completed!")
        self._log_stats(stats)

        return stats

    def _seed_user_multi_turn(
        self, user_id: str, category: str, conversations: List[Dict]
    ) -> Dict[str, int]:
        """Seed multi-turn conversations for a single user."""
        user_stats = {"conversations": 0, "embeddings": 0, "failed": 0}

        for i, conversation in enumerate(conversations):
            # Generate unique conversation ID
            conversation_id = (
                f"multi_turn_{category}_{user_id}_{int(datetime.now().timestamp())}_{i}"
            )

            try:
                # Determine the primary intent
                primary_intent = self._determine_primary_intent(category, conversation)

                # Create embeddings at different stages
                embeddings_created = self._create_staged_embeddings(
                    conversation_id, user_id, conversation, primary_intent
                )

                if embeddings_created > 0:
                    user_stats["conversations"] += 1
                    user_stats["embeddings"] += embeddings_created
                else:
                    user_stats["failed"] += 1

            except Exception as e:
                logger.error(f"Error seeding conversation {conversation_id}: {e}")
                user_stats["failed"] += 1

        return user_stats

    @staticmethod
    def _determine_primary_intent(category: str, conversation: Dict) -> str:
        """Determine the primary intent for a conversation."""
        # Check metadata first
        if "metadata" in conversation and "spans_intents" in conversation["metadata"]:
            return conversation["metadata"]["spans_intents"][0]

        # Map categories to intents
        category_to_intent = {
            "invoices": "invoices",
            "transactions": "transactions",
            "cross_domain": "general",
            "error_handling": "general",
        }

        return category_to_intent.get(category, "general")

    def _create_staged_embeddings(
        self, conversation_id: str, user_id: str, conversation: Dict, intent: str
    ) -> int:
        """Create embeddings at different stages of the conversation."""
        messages = conversation["messages"]
        embeddings_created = 0

        # Define stages to create embeddings
        stages = [(len(messages), "full")]

        # Always create embedding for full conversation

        # For longer conversations, create intermediate embeddings
        if len(messages) >= 4:
            stages.append((2, "initial"))
        if len(messages) >= 6:
            stages.append((4, "mid"))

        for stage_length, stage_name in stages:
            if len(messages) >= stage_length:
                # Create partial conversation
                partial_messages = messages[:stage_length]

                # Build context
                context_text = self._build_multi_turn_context(
                    partial_messages, conversation.get("metadata", {}), stage_name
                )

                # Generate embedding
                embedding = self.embedding_client.create_embedding(context_text)

                if embedding:
                    # Store embedding
                    metadata = {
                        "conversation_type": "multi_turn",
                        "stage": stage_name,
                        "message_count": stage_length,
                        "total_messages": len(messages),
                        "category": conversation.get("metadata", {}).get(
                            "conversation_type", "multi_turn"
                        ),
                        "tools_sequence": conversation.get("metadata", {}).get(
                            "tools_sequence", []
                        ),
                        "seeded_at": datetime.now().isoformat(),
                        **conversation.get("metadata", {}),
                    }

                    embedding_id = self.storage.store_embedding(
                        conversation_id=f"{conversation_id}_{stage_name}",
                        user_id=user_id,
                        embedding=embedding,
                        intent=intent,
                        assistant_type=f"{intent}_assistant",
                        metadata=metadata,
                        success_indicator=conversation.get("success", True),
                    )

                    if embedding_id:
                        embeddings_created += 1
                        logger.debug(
                            f"Created {stage_name} embedding for {conversation_id}"
                        )

        return embeddings_created

    def _build_multi_turn_context(
        self, messages: List[Dict], metadata: Dict, stage: str
    ) -> str:
        """Build context specifically for multi-turn conversations."""
        context_parts = []

        # Add conversation flow with tools
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                # Summarize assistant response
                summary = self._summarize_response(content)
                context_parts.append(f"Assistant: {summary}")

                # Add tools if present
                tools = msg.get("tools_used", [])
                if tools:
                    tool_names = [t["name"] for t in tools]
                    context_parts.append(f"Tools used: {', '.join(tool_names)}")

        # Add metadata context
        if metadata.get("conversation_type"):
            context_parts.append(f"Conversation type: {metadata['conversation_type']}")

        if metadata.get("tools_sequence"):
            context_parts.append(
                f"Tool sequence: {' -> '.join(metadata['tools_sequence'])}"
            )

        # Add stage info
        context_parts.append(f"Conversation stage: {stage}")

        return "\n".join(context_parts)

    @staticmethod
    def _summarize_response(content: str) -> str:
        """Summarize assistant response for embedding."""
        # For multi-turn, we want to capture the essence without full text
        if len(content) < 100:
            return content

        # Extract key phrases
        if "created" in content.lower():
            return "Created resource successfully"
        elif "sent" in content.lower():
            return "Sent/delivered successfully"
        elif "updated" in content.lower():
            return "Updated resource"
        elif "found" in content.lower() or "here" in content.lower():
            return "Retrieved and displayed data"
        else:
            # Take first sentence
            return content.split(".")[0] + "."

    @staticmethod
    def _log_stats(stats: Dict[str, Any]):
        """Log seeding statistics."""
        logger.info("=== Multi-Turn Seeding Complete ===")
        logger.info(f"Total conversations seeded: {stats['total_seeded']}")
        logger.info(f"Total embeddings created: {stats['embeddings_created']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("\nBy Category:")
        for category, count in stats["by_category"].items():
            logger.info(f"  {category}: {count} conversations")
        logger.info("\nBy User:")
        for user_id, count in stats["by_user"].items():
            logger.info(f"  {user_id}: {count} conversations")


def main():
    """Run the multi-turn seeder."""
    logging.basicConfig(level=logging.INFO)

    seeder = MultiTurnSeeder()

    print("🚀 Starting multi-turn conversation seeding...")
    print("📝 This will ONLY add new multi-turn conversations")
    print("✅ Existing single-turn conversations will NOT be duplicated\n")

    stats = seeder.seed_multi_turn_only()

    print(f"\n✨ Successfully seeded {stats['total_seeded']} multi-turn conversations!")
    print(f"📊 Created {stats['embeddings_created']} total embeddings")


if __name__ == "__main__":
    main()
