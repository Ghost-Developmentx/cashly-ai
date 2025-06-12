"""
Query executor - executes queries with the selected assistant.
Single responsibility: execution only.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.core.assistants import UnifiedAssistantManager
from ...schemas.assistant import AssistantType, AssistantResponse

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of query execution."""
    success: bool
    assistant_response: AssistantResponse
    execution_time: float
    tool_calls_count: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "response": self.assistant_response.to_dict() if hasattr(self.assistant_response, 'to_dict') else {
                "content": self.assistant_response.content,
                "assistant_type": self.assistant_response.assistant_type.value,
                "function_calls": self.assistant_response.function_calls,
                "metadata": self.assistant_response.metadata,
                "success": self.assistant_response.success,
                "error": self.assistant_response.error
            },
            "execution_time": self.execution_time,
            "tool_calls_count": self.tool_calls_count,
            "error": self.error
        }

class QueryExecutor:
    """
    Executes queries with selected assistants.
    Handles the actual interaction with the assistant manager.
    """

    def __init__(self, assistant_manager: Optional[UnifiedAssistantManager] = None):
        """
        Initialize with an assistant manager.

        Args:
            assistant_manager: UnifiedAssistantManager instance
        """
        self.assistant_manager = assistant_manager or UnifiedAssistantManager()
        self.max_retries = 2
        self.timeout = 30.0

    async def execute(
            self,
            query: str,
            assistant_type: AssistantType,
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute query with the specified assistant.

        Args:
            query: User's query
            assistant_type: Selected assistant type
            user_id: User identifier
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            ExecutionResult with response and metadata
        """
        import time
        start_time = time.time()

        try:
            # Execute query with assistant
            response = await self.assistant_manager.query_assistant(
                assistant_type=assistant_type,
                query=query,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Count tool calls
            tool_calls_count = len(response.function_calls)

            # Create execution result
            result = ExecutionResult(
                success=response.success,
                assistant_response=response,
                execution_time=execution_time,
                tool_calls_count=tool_calls_count,
                error=response.error
            )

            logger.info(
                f"Query executed successfully with {assistant_type.value} "
                f"in {execution_time:.2f}s ({tool_calls_count} tool calls)"
            )

            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)

            # Create error response
            error_response = AssistantResponse(
                content="I encountered an error processing your request. Please try again.",
                assistant_type=assistant_type,
                function_calls=[],
                metadata={"error": str(e)},
                success=False,
                error=str(e)
            )

            return ExecutionResult(
                success=False,
                assistant_response=error_response,
                execution_time=time.time() - start_time,
                tool_calls_count=0,
                error=str(e)
            )

    async def execute_with_retry(
            self,
            query: str,
            assistant_type: AssistantType,
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute a query with retry logic.

        Args:
            Same as execute()

        Returns:
            ExecutionResult after retries if needed
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for query execution")

            result = await self.execute(
                query=query,
                assistant_type=assistant_type,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history
            )

            if result.success:
                return result

            last_error = result.error

            # Don't retry certain errors
            if last_error and any(msg in last_error.lower() for msg in [
                "not configured", "invalid", "unauthorized"
            ]):
                break

        logger.warning(f"Query execution failed after {self.max_retries} retries")
        return result

    async def execute_with_fallback(
            self,
            query: str,
            primary_assistant: AssistantType,
            fallback_assistant: AssistantType,
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute query with fallback to alternative assistant.

        Args:
            query: User's query
            primary_assistant: Primary assistant to try
            fallback_assistant: Fallback assistant if primary fails
            user_id: User identifier
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            ExecutionResult from successful execution
        """
        # Try primary assistant
        result = await self.execute(
            query=query,
            assistant_type=primary_assistant,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history
        )

        if result.success:
            return result

        # Try fallback assistant
        logger.info(f"Falling back from {primary_assistant.value} to {fallback_assistant.value}")

        fallback_result = await self.execute(
            query=query,
            assistant_type=fallback_assistant,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history
        )

        # Add metadata about fallback
        if fallback_result.assistant_response.metadata is None:
            fallback_result.assistant_response.metadata = {}

        fallback_result.assistant_response.metadata.update({
            "used_fallback": True,
            "original_assistant": primary_assistant.value,
            "fallback_assistant": fallback_assistant.value,
            "fallback_reason": result.error
        })

        return fallback_result
