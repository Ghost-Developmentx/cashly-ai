"""
Async tool execution for OpenAI Assistants.
Handles function calls with proper async/sync compatibility.
"""

import json
import logging
import asyncio
import inspect
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor
from .base import BaseManager

logger = logging.getLogger(__name__)

class ToolExecutor(BaseManager):
    """Handles async tool execution for assistants."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_executor: Optional[Callable] = None
        self._thread_pool = ThreadPoolExecutor(max_workers=5)

    def set_tool_executor(self, executor: Callable):
        """
        Set the tool executor function.

        Args:
            executor: Function to execute tools
        """
        self._tool_executor = executor
        logger.info(
            f"Tool executor configured: "
            f"{'async' if inspect.iscoroutinefunction(executor) else 'sync'}"
        )

    async def execute_tool_calls(
            self,
            tool_calls: List[Any],
            user_id: str,
            user_context: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Execute tool calls asynchronously.

        Args:
            tool_calls: List of tool calls from OpenAI
            user_id: User identifier
            user_context: User context data

        Returns:
            Tuple of (tool_outputs, function_calls)
        """
        if not self._tool_executor:
            logger.error("Tool executor not configured")
            return self._create_error_outputs(
                tool_calls,
                "Tool executor not configured"
            )

        tool_outputs = []
        function_calls = []

        # Execute tools concurrently
        tasks = []
        for tool_call in tool_calls:
            task = self._execute_single_tool(
                tool_call,
                user_id,
                user_context
            )
            tasks.append(task)

        # Wait for all tools to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            tool_call = tool_calls[i]

            if isinstance(result, Exception):
                logger.error(f"Tool execution failed: {result}")
                output, call = self._create_error_result(
                    tool_call,
                    str(result)
                )
            else:
                output, call = result

            tool_outputs.append(output)
            function_calls.append(call)

        return tool_outputs, function_calls

    async def _execute_single_tool(
            self,
            tool_call: Any,
            user_id: str,
            user_context: Optional[Dict]
    ) -> Tuple[Dict, Dict]:
        """Execute a single tool call."""
        function_name = tool_call.function.name

        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid function arguments: {e}")
            return self._create_error_result(
                tool_call,
                "Invalid function arguments"
            )

        logger.info(f"Executing tool: {function_name}")
        logger.debug(f"Tool args: {function_args}")

        try:
            # Execute the tool
            result = await self._execute_tool_async(
                function_name,
                function_args,
                user_id,
                user_context
            )

            # Create success output
            output = {
                "tool_call_id": tool_call.id,
                "output": json.dumps(result)
            }

            call = {
                "function": function_name,
                "arguments": function_args,
                "result": result
            }

            return output, call

        except Exception as e:
            logger.error(f"Tool {function_name} failed: {e}")
            return self._create_error_result(tool_call, str(e))

    async def _execute_tool_async(
            self,
            function_name: str,
            function_args: Dict,
            user_id: str,
            user_context: Optional[Dict]
    ) -> Dict:
        """Execute tool with async/sync compatibility."""
        if not self._tool_executor:
            raise ValueError("Tool executor not configured")

        # Prepare execution context
        execution_context = {
            "user_id": user_id,
            "transactions": (
                user_context.get("transactions", [])
                if user_context else []
            ),
            "user_context": user_context or {}
        }

        # Check if executor is async
        if inspect.iscoroutinefunction(self._tool_executor):
            # Execute async function directly
            return await self._tool_executor(
                function_name,
                function_args,
                **execution_context
            )
        else:
            # Execute sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._thread_pool,
                self._tool_executor,
                function_name,
                function_args,
                execution_context["user_id"],
                execution_context["transactions"],
                execution_context["user_context"]
            )

    @staticmethod
    def _create_error_result(
            tool_call: Any,
            error_message: str
    ) -> Tuple[Dict, Dict]:
        """Create error result for failed tool execution."""
        error_result = {"error": error_message}

        output = {
            "tool_call_id": tool_call.id,
            "output": json.dumps(error_result)
        }

        call = {
            "function": tool_call.function.name,
            "arguments": {},
            "result": error_result
        }

        return output, call

    @staticmethod
    def _create_error_outputs(
            tool_calls: List[Any],
            error_message: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Create error outputs for all tool calls."""
        tool_outputs = []
        function_calls = []

        for tool_call in tool_calls:
            output, call = ToolExecutor._create_error_result(
                tool_call,
                error_message
            )
            tool_outputs.append(output)
            function_calls.append(call)

        return tool_outputs, function_calls

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)