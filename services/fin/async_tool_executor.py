"""
Async tool execution with error handling and validation.
"""

import logging
from typing import Any, Dict, Callable, Awaitable
from datetime import datetime

from .utils import normalize_transaction_dates

logger = logging.getLogger(__name__)


class AsyncToolExecutor:
    """
    AsyncToolExecutor is a class used for executing tool handlers asynchronously.

    This class provides functionality to execute given async handlers with a specified
    execution context and handles any necessary preparatory or post-processing steps.
    It ensures that execution details are logged, results are validated, and any
    unexpected errors during execution are captured.

    Attributes
    ----------
    None
    """

    async def execute_tool(
        self,
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool handler asynchronously.

        Args:
            handler: Async tool handler function
            context: Execution context

        Returns:
            Tool execution result
        """
        try:
            # Normalize transaction dates if present
            if "transactions" in context:
                context["transactions"] = normalize_transaction_dates(
                    context["transactions"]
                )

            # Log execution start
            start_time = datetime.now()
            tool_name = handler.__name__
            logger.info(f"🔧 Executing tool: {tool_name}")

            # Execute the handler
            result = await handler(context)

            # Log execution complete
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Tool {tool_name} completed in {execution_time:.2f}s")

            # Validate result
            if self._is_valid_result(result):
                return result
            else:
                logger.warning(f"Invalid result from {tool_name}")
                return {"error": "Invalid tool result", "tool": tool_name}

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {
                "error": str(e),
                "tool": handler.__name__ if hasattr(handler, "__name__") else "unknown",
            }

    @staticmethod
    def _is_valid_result(result: Any) -> bool:
        """Validate tool result."""
        if not isinstance(result, dict):
            return False

        # Check for error responses
        if "error" in result and not result.get("success", True):
            return True  # Valid error response

        # Check for action responses
        if "action" in result:
            return True

        # Check for data responses
        if any(key in result for key in ["data", "result", "message", "transactions"]):
            return True

        return False
