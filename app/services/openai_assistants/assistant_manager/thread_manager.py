"""
Thread management for OpenAI Assistant conversations.
Handles thread lifecycle and message operations.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base import BaseManager

logger = logging.getLogger(__name__)

class ThreadManager(BaseManager):
    """
    Manages user-specific threads and related operations.

    This class serves as a manager for handling user-specific threads, supporting
    operations like creating new threads, retrieving messages, adding messages, and
    clearing threads. The class maintains in-memory storage for active threads and their
    metadata, enabling efficient access and management of thread-related data. This
    storage can be extended or replaced with external persistent solutions, like Redis,
    for scalability.

    Attributes
    ----------
    _active_threads : Dict[str, str]
        Dictionary mapping user identifiers to their active thread IDs.
    _thread_metadata : Dict[str, Dict[str, Any]]
        Dictionary containing metadata for threads, such as user ID, creation time,
        message count, and other thread-specific details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In-memory thread storage (consider Redis for production)
        self._active_threads: Dict[str, str] = {}
        self._thread_metadata: Dict[str, Dict[str, Any]] = {}

    async def get_or_create_thread(self, user_id: str) -> str:
        """
        Get existing thread or create new one for user.

        Args:
            user_id: User identifier

        Returns:
            Thread ID
        """
        # Check for existing thread
        if user_id in self._active_threads:
            thread_id = self._active_threads[user_id]
            logger.debug(f"Using existing thread {thread_id} for user {user_id}")
            return thread_id

        # Create new thread
        try:
            thread = await self.client.beta.threads.create()
            thread_id = thread.id

            # Store thread mapping
            self._active_threads[user_id] = thread_id
            self._thread_metadata[thread_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "message_count": 0
            }

            logger.info(f"Created new thread {thread_id} for user {user_id}")
            return thread_id

        except Exception as e:
            logger.error(f"Failed to create thread for user {user_id}: {e}")
            raise

    async def add_message(
            self,
            thread_id: str,
            content: str,
            role: str = "user",
            metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Add a message to a thread.

        Args:
            thread_id: Thread identifier
            content: Message content
            role: Message role (user/assistant)
            metadata: Optional message metadata

        Returns:
            Created message object
        """
        try:
            message = await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content,
                metadata=metadata or {}
            )

            # Update thread metadata
            if thread_id in self._thread_metadata:
                self._thread_metadata[thread_id]["message_count"] += 1
                self._thread_metadata[thread_id]["last_message_at"] = (
                    datetime.now().isoformat()
                )

            logger.debug(f"Added {role} message to thread {thread_id}")
            return message

        except Exception as e:
            logger.error(f"Failed to add message to thread {thread_id}: {e}")
            raise

    async def get_thread_messages(
            self,
            thread_id: str,
            limit: int = 20,
            order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum messages to retrieve
            order: Message order (asc/desc)

        Returns:
            List of formatted messages
        """
        try:
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=limit,
                order=order
            )

            formatted_messages = []
            for message in messages.data:
                formatted_messages.append(
                    self._format_message(message)
                )

            return formatted_messages

        except Exception as e:
            logger.error(f"Failed to get messages from thread {thread_id}: {e}")
            return []

    def clear_thread(self, user_id: str) -> bool:
        """
        Clear thread for a user.

        Args:
            user_id: User identifier

        Returns:
            True if thread was cleared
        """
        if user_id in self._active_threads:
            thread_id = self._active_threads[user_id]
            del self._active_threads[user_id]

            if thread_id in self._thread_metadata:
                del self._thread_metadata[thread_id]

            logger.info(f"Cleared thread for user {user_id}")
            return True

        return False

    def get_active_thread_count(self) -> int:
        """Get count of active threads."""
        return len(self._active_threads)

    def get_thread_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get thread information for a user."""
        if user_id not in self._active_threads:
            return None

        thread_id = self._active_threads[user_id]
        return self._thread_metadata.get(thread_id)

    @staticmethod
    def _format_message(message: Any) -> Dict[str, Any]:
        """Format a message object."""
        content = ""
        for content_block in message.content:
            if content_block.type == "text":
                content += content_block.text.value

        return {
            "id": message.id,
            "role": message.role,
            "content": content,
            "created_at": message.created_at,
            "metadata": message.metadata or {}
        }