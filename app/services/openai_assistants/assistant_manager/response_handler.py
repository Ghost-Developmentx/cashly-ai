"""
Response handling for async assistant runs.
Manages run completion and result extraction.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base import BaseManager
from .types import AssistantType, AssistantResponse

logger = logging.getLogger(__name__)

class ResponseHandler(BaseManager):
    """
    Manages responses by handling the completion of asynchronous operations.

    This class provides utilities for monitoring and managing the states of
    asynchronous operations, interacting with tools, handling completion or errors,
    and formatting responses effectively for clients. It ensures robust handling of
    multiple operational states like completed, failed, or requiring additional
    actions.

    Attributes
    ----------
    config : Any
        Configuration settings for managing response timeouts, execution limits, and other configurations.
    client : Any
        Client instance used for communication with external APIs or services.
    """

    async def wait_for_run_completion(
            self,
            thread_id: str,
            run_id: str,
            assistant_type: AssistantType,
            tool_executor: Optional[Any] = None,
            user_id: Optional[str] = None,
            user_context: Optional[Dict] = None
    ) -> AssistantResponse:
        """
        Wait for assistant run to complete.

        Args:
            thread_id: Thread identifier
            run_id: Run identifier
            assistant_type: Type of assistant
            tool_executor: Tool executor instance
            user_id: User identifier
            user_context: User context data

        Returns:
            AssistantResponse with results
        """
        start_time = asyncio.get_event_loop().time()
        function_calls = []
        poll_interval = 0.5

        while asyncio.get_event_loop().time() - start_time < self.config.timeout:
            try:
                # Get run status
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )

                logger.debug(f"Run {run_id} status: {run.status}")

                # Handle different run states
                if run.status == "completed":
                    return await self._handle_completed_run(
                        thread_id,
                        run_id,
                        assistant_type,
                        function_calls
                    )

                elif run.status == "requires_action":
                    # Execute required tools
                    function_calls = await self._handle_required_action(
                        thread_id,
                        run_id,
                        run,
                        tool_executor,
                        user_id,
                        user_context,
                        function_calls
                    )

                elif run.status in ["failed", "cancelled", "expired"]:
                    return self._create_error_response(
                        assistant_type,
                        function_calls,
                        f"Run {run.status}",
                        {
                            "run_id": run_id,
                            "thread_id": thread_id,
                            "status": run.status,
                            "last_error": getattr(run, "last_error", None)
                        }
                    )

                elif run.status == "cancelling":
                    logger.warning(f"Run {run_id} is being cancelled")

                # Wait before next poll
                await asyncio.sleep(poll_interval)

                # Increase poll interval slightly for efficiency
                poll_interval = min(poll_interval * 1.1, 2.0)

            except Exception as e:
                logger.error(f"Error polling run status: {e}")
                return self._create_error_response(
                    assistant_type,
                    function_calls,
                    "Failed to check run status",
                    {"error": str(e)}
                )

        # Timeout reached
        return self._create_error_response(
            assistant_type,
            function_calls,
            "Request timeout",
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "timeout": self.config.timeout
            }
        )

    async def _handle_required_action(
            self,
            thread_id: str,
            run_id: str,
            run: Any,
            tool_executor: Optional[Any],
            user_id: Optional[str],
            user_context: Optional[Dict],
            existing_calls: List[Dict]
    ) -> List[Dict]:
        """Handle required action (tool calls)."""
        if not tool_executor:
            logger.error("Tool executor not available for required action")
            return existing_calls

        try:
            # Get required tool calls
            tool_calls = run.required_action.submit_tool_outputs.tool_calls

            # Execute tools
            tool_outputs, new_calls = await tool_executor.execute_tool_calls(
                tool_calls,
                user_id or "unknown",
                user_context
            )

            # Submit tool outputs
            await self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=tool_outputs
            )

            # Combine function calls
            return existing_calls + new_calls

        except Exception as e:
            logger.error(f"Failed to handle required action: {e}")
            return existing_calls

    async def _handle_completed_run(
            self,
            thread_id: str,
            run_id: str,
            assistant_type: AssistantType,
            function_calls: List[Dict]
    ) -> AssistantResponse:
        """Handle completed run and extract response."""
        try:
            # Get the latest message
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=1
            )

            if not messages.data:
                return self._create_error_response(
                    assistant_type,
                    function_calls,
                    "No response generated",
                    {"run_id": run_id}
                )

            # Extract content
            content = self._extract_message_content(messages.data[0])

            return AssistantResponse(
                content=content,
                assistant_type=assistant_type,
                function_calls=function_calls,
                metadata={
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "completed_at": datetime.now().isoformat()
                },
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to handle completed run: {e}")
            return self._create_error_response(
                assistant_type,
                function_calls,
                "Failed to extract response",
                {"error": str(e)}
            )

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        """Extract text content from message."""
        content_parts = []

        for content_block in message.content:
            if content_block.type == "text":
                content_parts.append(content_block.text.value)

        return "\n".join(content_parts)

    @staticmethod
    def _create_error_response(
            assistant_type: AssistantType,
            function_calls: List[Dict],
            error_message: str,
            metadata: Dict[str, Any]
    ) -> AssistantResponse:
        """Create an error response."""
        return AssistantResponse(
            content=(
                "I apologize, but I encountered an error processing your request. "
                "Please try again or contact support if the issue persists."
            ),
            assistant_type=assistant_type,
            function_calls=function_calls,
            metadata={
                **metadata,
                "error_at": datetime.now().isoformat()
            },
            success=False,
            error=error_message
        )