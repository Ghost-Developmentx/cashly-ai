"""
Configuration for OpenAI Integration Service.
Manages service-level settings and initialization.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from ..assistant_manager import AssistantConfig

logger = logging.getLogger(__name__)


class IntegrationConfig(BaseModel):
    """
    Configuration for integration components, now using Pydantic BaseModel.
    """

    # Define fields for all the components we'll initialize
    assistant_config: Optional[AssistantConfig] = Field(default=None)
    assistant_manager: Optional[Any] = Field(default=None)
    intent_service: Optional[Any] = Field(default=None)
    router: Optional[Any] = Field(default=None)
    intent_mapper: Optional[Any] = Field(default=None)
    function_processor: Optional[Any] = Field(default=None)

    model_config = {
        "arbitrary_types_allowed": True  # Allow non-Pydantic types
    }

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize components after Pydantic initialization
        self._setup_components()

    def _setup_components(self):
        """Initialize all integration components."""
        try:
            # Import here to avoid circular imports
            from app.services.openai_assistants.assistant_manager import AsyncAssistantManager
            from app.services.intent_classification.async_intent_service import AsyncIntentService
            from app.services.openai_assistants.core.router import AssistantRouter
            from app.services.openai_assistants.core.intent_mapper import IntentMapper
            from app.services.openai_assistants.processors.function_processor import FunctionProcessor
            from app.services.openai_assistants.utils.constants import CROSS_ROUTING_PATTERNS

            # Now we can set these since they're defined as fields
            self.assistant_config = AssistantConfig()
            self.assistant_manager = AsyncAssistantManager(self.assistant_config)
            self.intent_service = AsyncIntentService()
            self.router = AssistantRouter(CROSS_ROUTING_PATTERNS)
            self.intent_mapper = IntentMapper()
            self.function_processor = FunctionProcessor()

            self._setup_tool_executor_sync()

        except ImportError as e:
            # Handle missing imports gracefully
            logger.warning(f"Could not initialize some components: {e}")

    def _setup_tool_executor_sync(self):
        """Configure tool executor from existing Fin service."""
        try:
            from app.services.fin.async_tool_registry import AsyncToolRegistry

            tool_registry = AsyncToolRegistry()

            async def async_tool_executor(
                    tool_name: str, tool_args: Dict[str, Any], **kwargs
            ) -> Dict[str, Any]:
                """Async wrapper for tool registry execution."""
                try:
                    return await tool_registry.execute(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        user_id=kwargs.get("user_id", "unknown"),
                        transactions=kwargs.get("transactions", []),
                        user_context=kwargs.get("user_context", {}),
                    )
                except Exception as e:
                    return {"error": f"Tool execution failed: {str(e)}"}

            self.assistant_manager.set_tool_executor(async_tool_executor)

        except Exception as e:
            logger.warning(f"Failed to setup tool executor: {e}")

    async def validate(self) -> Dict[str, Any]:
        """Validate all components are properly configured."""
        validation_results = {"components": {}, "is_valid": True}

        # Check assistant configuration
        if self.assistant_config:
            assistant_validation = self.assistant_config.validate()
            validation_results["components"]["assistants"] = assistant_validation
            if not assistant_validation["valid"]:
                validation_results["is_valid"] = False

        # Check if the tool executor is set
        has_tool_executor = (
            self.assistant_manager is not None
            and hasattr(self.assistant_manager, 'tool_executor')
        )
        validation_results["components"]["tool_executor"] = {
            "configured": has_tool_executor
        }

        # Check intent service
        try:
            if self.intent_service:
                test_result = await self.intent_service.classify_and_route("test")
                validation_results["components"]["intent_service"] = {
                    "status": "healthy",
                    "test_intent": test_result["classification"]["intent"],
                }
        except Exception as e:
            validation_results["components"]["intent_service"] = {
                "status": "error",
                "error": str(e),
            }
            validation_results["is_valid"] = False

        return validation_results
