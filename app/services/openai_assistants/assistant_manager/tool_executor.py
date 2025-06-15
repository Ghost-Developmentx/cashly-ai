"""
Wrapper for compatibility with existing assistant manager.
IMPORTANT: This file should be named tool_executor.py (not tool_executer.py)
"""
from app.core.tools.executor import ToolExecutor as CoreToolExecutor

class ToolExecutor:
    """Compatibility wrapper for assistant manager."""

    def __init__(self, *args, **kwargs):
        self.core_executor = CoreToolExecutor()
        self._custom_executor = None

    def set_tool_executor(self, executor):
        """Store custom executor for backward compatibility."""
        self._custom_executor = executor

    async def execute_tool_calls(self, tool_calls, user_id, user_context=None):
        """
        Execute tool calls using core executor.

        Args:
            tool_calls: List of OpenAI tool calls
            user_id: User identifier
            user_context: Optional context dict

        Returns:
            Tuple of (tool_outputs, function_calls)
        """
        return await self.core_executor.execute_openai_tool_calls(
            tool_calls,
            user_id,
            user_context,
            transactions=user_context.get('transactions', []) if user_context else []
        )