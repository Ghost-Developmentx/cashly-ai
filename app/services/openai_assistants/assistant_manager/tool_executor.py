"""
Wrapper for compatibility with existing assistant manager.
"""
from app.core.tools.executor import ToolExecutor as CoreToolExecutor

class ToolExecutor:
    """Compatibility wrapper for assistant manager."""

    def __init__(self, *args, **kwargs):
        self.executor = CoreToolExecutor()

    def set_tool_executor(self, executor):
        # For compatibility - ignored since we use our own
        pass

    async def execute_tool_calls(self, tool_calls, user_id, user_context=None):
        # Delegate to core executor
        return await self.executor.execute_openai_tool_calls(
            tool_calls, user_id, user_context
        )