"""
Migration helper for transitioning from sync to async.
Provides a compatibility layer during migration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from .manager import AsyncAssistantManager
from .types import AssistantType, AssistantResponse

logger = logging.getLogger(__name__)


class SyncCompatibilityWrapper:
    """
    Provides a synchronous interface to async manager.
    Use only during a migration period.
    """

    def __init__(self, async_manager: AsyncAssistantManager):
        self.async_manager = async_manager
        logger.warning(
            "Using sync compatibility wrapper. "
            "This should only be used during migration."
        )

    def process_query(
            self,
            query: str,
            assistant_type: AssistantType,
            user_id: str,
            user_context: Optional[Dict] = None,
            conversation_history: Optional[List[Dict]] = None
    ) -> AssistantResponse:
        """Sync wrapper for process_query."""
        return self._run_async(
            self.async_manager.process_query(
                query,
                assistant_type,
                user_id,
                user_context,
                conversation_history
            )
        )

    def get_conversation_history(
            self,
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Sync wrapper for get_conversation_history."""
        return self._run_async(
            self.async_manager.get_conversation_history(user_id, limit)
        )

    def health_check(self) -> Dict[str, Any]:
        """Sync wrapper for health_check."""
        return self._run_async(self.async_manager.health_check())

    def set_tool_executor(self, executor):
        """Set the tool executor (sync method)."""
        self.async_manager.set_tool_executor(executor)

    def clear_thread(self, user_id: str) -> bool:
        """Clear thread (sync method)."""
        return self.async_manager.clear_thread(user_id)

    @staticmethod
    def _run_async(coro):
        """Run async coroutine in the sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If the loop is already running, create a new task
                task = asyncio.create_task(coro)
                return asyncio.run_coroutine_threadsafe(
                    coro,
                    loop
                ).result()
            else:
                # If no loop is running, run normally
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)


def create_migration_manager(api_key: Optional[str] = None) -> Any:
    """
    Create a manager with migration compatibility.

    Returns sync-compatible wrapper around async manager.
    """
    async_manager = AsyncAssistantManager()
    return SyncCompatibilityWrapper(async_manager)