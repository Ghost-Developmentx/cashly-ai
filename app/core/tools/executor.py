import asyncio
import inspect
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from .registry import tool_registry

logger = logging.getLogger(__name__)

class ToolExecutor:
    """
    Executes tools with proper error handling and context management.
    Handles both async and sync functions seamlessly.
    """

    def __init__(self, rails_client=None):
        self.rails_client = rails_client
        self._execution_context = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
        logger.info("Initialized ToolExecutor")

    @asynccontextmanager
    async def context(self, **kwargs):
        """
        Context manager to set execution context for tools.

        Usage:
            async with executor.context(user_id="123", user_context={...}):
                a result = await executor.execute("tool_name", {...})
        """
        old_context = self._execution_context.copy()
        self._execution_context.update(kwargs)
        try:
            yield self
        finally:
            self._execution_context = old_context

    async def execute(
            self,
            tool_name: str,
            tool_args: Dict[str, Any],
            user_id: Optional[str] = None,
            transactions: Optional[List[Dict[str, Any]]] = None,
            user_context: Optional[Dict[str, Any]] = None,
            **extra_context
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with the given arguments.
        """
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        # Validate arguments
        try:
            validated_args = tool_registry.validate_tool_args(tool_name, tool_args)
        except ValueError as e:
            return {"error": f"Invalid arguments: {str(e)}"}

        # Prepare full context
        full_context = {
            **self._execution_context,
            **extra_context,
            "tool_args": validated_args,
            "user_id": user_id,
            "transactions": transactions or [],
            "user_context": user_context or {},
            "rails_client": self.rails_client
        }

        try:
            handler = tool.handler

            # Log execution
            logger.info(f"Executing tool: {tool_name}")
            logger.debug(f"Tool args: {validated_args}")

            # Check if handler is async
            if inspect.iscoroutinefunction(handler):
                result = await handler(full_context)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool,
                    handler,
                    full_context
                )

            # Log success
            logger.info(f"Tool {tool_name} executed successfully")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}", exc_info=True)
            return {"error": f"Tool execution failed: {str(e)}"}

    async def execute_openai_tool_calls(
            self,
            tool_calls: List[Any],
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None,
            transactions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Execute OpenAI Assistant tool calls.

        This replaces the duplicate logic in:
        - app/services/openai_assistants/assistant_manager/tool_executor.py

        Returns:
            Tuple of (tool_outputs, function_calls) for OpenAI and our system
        """
        tool_outputs = []
        function_calls = []

        # Execute all tools
        for tool_call in tool_calls:
            try:
                # Extract tool info
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Processing OpenAI tool call: {function_name}")

                # Execute the tool
                result = await self.execute(
                    tool_name=function_name,
                    tool_args=function_args,
                    user_id=user_id,
                    user_context=user_context,
                    transactions=transactions
                )

                # Format for OpenAI
                tool_output = {
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(result)
                }
                tool_outputs.append(tool_output)

                # Format for our system
                function_call = {
                    "function": function_name,
                    "arguments": function_args,
                    "result": result
                }
                function_calls.append(function_call)

            except Exception as e:
                logger.error(f"Failed to execute tool call {tool_call.id}: {e}")

                # Error output for OpenAI
                error_result = {"error": str(e)}
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(error_result)
                })

                # Error for our system
                function_calls.append({
                    "function": getattr(tool_call.function, 'name', 'unknown'),
                    "arguments": {},
                    "result": error_result
                })

        return tool_outputs, function_calls

    async def execute_batch(
            self,
            tool_requests: List[Dict[str, Any]],
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tools concurrently.

        Args:
            tool_requests: List of dicts with 'name' and 'args' keys
            user_id: User identifier
            user_context: Shared context for all tools

        Returns:
            List of results in the same order as requests
        """
        tasks = []

        for request in tool_requests:
            tool_name = request.get('name')
            tool_args = request.get('args', {})

            task = self.execute(
                tool_name=tool_name,
                tool_args=tool_args,
                user_id=user_id,
                user_context=user_context
            )
            tasks.append(task)

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "error": f"Execution failed: {str(result)}",
                    "tool": tool_requests[i].get('name', 'unknown')
                })
            else:
                final_results.append(result)

        return final_results

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
