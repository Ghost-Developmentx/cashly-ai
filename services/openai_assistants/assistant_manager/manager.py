"""
Main async assistant manager.
Orchestrates all components for processing queries.
"""

import logging
from typing import Dict, List, Optional, Any
from .config import AssistantConfig
from .context_enhancer import ContextEnhancer
from .thread_manager import ThreadManager
from .tool_executor import ToolExecutor
from .response_handler import ResponseHandler
from .types import AssistantType, AssistantResponse

logger = logging.getLogger(__name__)

class AsyncAssistantManager:
    """
    Unified async manager for OpenAI Assistants.
    Combines all components for complete query processing.
    """

    def __init__(self, config: Optional[AssistantConfig] = None):
        """Initialize with shared configuration."""
        self.config = config or AssistantConfig()

        # Initialize components
        self.thread_manager = ThreadManager(self.config)
        self.tool_executor = ToolExecutor(self.config)
        self.response_handler = ResponseHandler(self.config)
        self.context_enhancer = ContextEnhancer()

        logger.info("✅ Async Assistant Manager initialized")

    async def process_query(
            self,
            query: str,
            assistant_type: AssistantType,
            user_id: str,
            user_context: Optional[Dict] = None,
            conversation_history: Optional[List[Dict]] = None
    ) -> AssistantResponse:
        """
        Process a query through an assistant.

        Args:
            query: User's query
            assistant_type: Which assistant to use
            user_id: User identifier
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            AssistantResponse with results
        """
        try:
            # Validate assistant availability
            if not self.config.is_assistant_configured(assistant_type):
                return self._create_not_configured_response(assistant_type)

            # Get assistant ID
            assistant_id = self.config.get_assistant_id(assistant_type)

            # Get or create thread
            thread_id = await self.thread_manager.get_or_create_thread(user_id)

            # Enhance query with context
            enhancement = self.context_enhancer.enhance_query(
                query,
                user_context
            )

            # Add message to thread
            await self.thread_manager.add_message(
                thread_id,
                enhancement.enhanced_query,
                metadata={
                    "original_query": query,
                    "context_added": enhancement.has_context
                }
            )

            # Create and run assistant
            run = await self._create_assistant_run(
                thread_id,
                assistant_id,
                enhancement.additional_instructions
            )

            # Wait for completion
            response = await self.response_handler.wait_for_run_completion(
                thread_id=thread_id,
                run_id=run.id,
                assistant_type=assistant_type,
                tool_executor=self.tool_executor,
                user_id=user_id,
                user_context=user_context
            )

            # Log completion
            self._log_completion(user_id, assistant_type, response.success)

            return response

        except Exception as e:
            logger.error(
                f"Error processing query for {assistant_type.value}: {e}",
                exc_info=True
            )
            return self._create_error_response(assistant_type, e)

    async def _create_assistant_run(
            self,
            thread_id: str,
            assistant_id: str,
            additional_instructions: str
    ) -> Any:
        """Create an assistant run."""
        return await self.response_handler.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions
        )

    def set_tool_executor(self, executor):
        """
        Set the tool executor.

        Args:
            executor: Callable for executing tools
        """
        self.tool_executor.set_tool_executor(executor)

    async def get_conversation_history(
            self,
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.

        Args:
            user_id: User identifier
            limit: Maximum messages to retrieve

        Returns:
            List of conversation messages
        """
        thread_info = self.thread_manager.get_thread_info(user_id)
        if not thread_info:
            return []

        thread_id = self.thread_manager._active_threads.get(user_id)
        if not thread_id:
            return []

        return await self.thread_manager.get_thread_messages(
            thread_id,
            limit=limit
        )

    def clear_thread(self, user_id: str) -> bool:
        """Clear conversation thread for a user."""
        return self.thread_manager.clear_thread(user_id)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.

        Returns:
            Health status information
        """
        health = {
            "status": "healthy",
            "components": {},
            "statistics": {}
        }

        # Check configuration
        config_validation = self.config.validate()
        health["components"]["configuration"] = {
            "status": "healthy" if config_validation["valid"] else "degraded",
            "configured_assistants": len(config_validation["configured_assistants"]),
            "missing_assistants": len(config_validation["missing_assistants"])
        }

        # Check thread manager
        health["statistics"]["active_threads"] = (
            self.thread_manager.get_active_thread_count()
        )

        # Check each assistant
        health["assistants"] = {}
        for assistant_type in AssistantType:
            if self.config.is_assistant_configured(assistant_type):
                assistant_id = self.config.get_assistant_id(assistant_type)
                try:
                    assistant = await self.response_handler.client.beta.assistants.retrieve(
                        assistant_id
                    )
                    health["assistants"][assistant_type.value] = {
                        "status": "healthy",
                        "id": assistant_id,
                        "name": assistant.name
                    }
                except Exception as e:
                    health["assistants"][assistant_type.value] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health["status"] = "degraded"

        return health

    @staticmethod
    def _create_not_configured_response(
            assistant_type: AssistantType
    ) -> AssistantResponse:
        """Create response for unconfigured assistant."""
        return AssistantResponse(
            content=f"The {assistant_type.value} assistant is not configured. "
                    f"Please contact support to enable this feature.",
            assistant_type=assistant_type,
            function_calls=[],
            metadata={"reason": "not_configured"},
            success=False,
            error="Assistant not configured"
        )

    @staticmethod
    def _create_error_response(
            assistant_type: AssistantType,
            error: Exception
    ) -> AssistantResponse:
        """Create error response."""
        return AssistantResponse(
            content="I encountered an error processing your request. "
                    "Please try again or contact support if the issue persists.",
            assistant_type=assistant_type,
            function_calls=[],
            metadata={"error_type": type(error).__name__},
            success=False,
            error=str(error)
        )

    @staticmethod
    def _log_completion(
            user_id: str,
            assistant_type: AssistantType,
            success: bool
    ):
        """Log query completion."""
        status = "✅ Success" if success else "❌ Failed"
        logger.info(
            f"{status} - User: {user_id}, Assistant: {assistant_type.value}"
        )