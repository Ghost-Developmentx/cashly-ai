"""
Configuration for OpenAI Integration Service.
Manages service-level settings and initialization.
"""

import logging
from typing import Optional, Dict, Any
from ..assistant_manager import AsyncAssistantManager, AssistantConfig
from ...intent_classification.async_intent_service import AsyncIntentService
from ..core.router import AssistantRouter
from ..core.intent_mapper import IntentMapper
from ..processors.function_processor import FunctionProcessor
from ..utils.constants import CROSS_ROUTING_PATTERNS

logger = logging.getLogger(__name__)


class IntegrationConfig:
    """
    Configuration for integration components, including assistant management,
    intent handling, and tool execution.

    This class initializes and configures components required for handling
    assistant functionalities and intent routing. It ensures coordination
    between different elements of the integration and provides methods for
    component validation and tool execution setup.

    Attributes
    ----------
    assistant_config : AssistantConfig
        Represents the configuration for the assistant, including details
        necessary for initializing the assistant manager.
    assistant_manager : AsyncAssistantManager
        Handles interaction with the assistant system, including execution
        of tools and operations.
    intent_service : AsyncIntentService
        Manages intent-related services, including classification and routing
        of intents.
    router : AssistantRouter
        Handles routing of assistant requests based on defined routing patterns.
    intent_mapper : IntentMapper
        Maps intents to corresponding actions or services.
    function_processor : FunctionProcessor
        Processes functions associated with the assistant system, ensuring
        their proper execution and integration.

    Methods
    -------
    __init__()
        Initialize all integration components.
    _setup_tool_executor()
        Configure the tool executor to handle asynchronous tool executions
        by creating and setting a tool registry-based executor.
    validate()
        Validate proper configuration of all integration components to
        ensure they are functioning correctly and are in a healthy state.
    """

    def __init__(self):
        """Initialize all components."""
        # Initialize core managers
        self.assistant_config = AssistantConfig()
        self.assistant_manager = AsyncAssistantManager(self.assistant_config)
        self.intent_service = AsyncIntentService()
        # Initialize components
        self.router = AssistantRouter(CROSS_ROUTING_PATTERNS)
        self.intent_mapper = IntentMapper()
        self.function_processor = FunctionProcessor()
        # Setup tool executor
        self._setup_tool_executor()
        logger.info("✅ Integration components initialized")

    async def _setup_tool_executor(self):
        """Configure tool executor from existing Fin service."""
        try:
            from services.fin.async_tool_registry import AsyncToolRegistry

            # Create a tool registry instance
            tool_registry = AsyncToolRegistry()

            # Now we can directly use the async execute method
            async def async_tool_executor(
                tool_name: str, tool_args: Dict[str, Any], **kwargs
            ) -> Dict[str, Any]:
                """Async wrapper for tool registry execution."""
                try:
                    # Call the async execute method directly
                    return await tool_registry.execute(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        user_id=kwargs.get("user_id", "unknown"),
                        transactions=kwargs.get("transactions", []),
                        user_context=kwargs.get("user_context", {}),
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    return {"error": f"Tool execution failed: {str(e)}"}

            self.assistant_manager.set_tool_executor(async_tool_executor)
            logger.info("✅ Async tool executor configured")

        except Exception as e:
            logger.error(f"❌ Failed to setup tool executor: {e}")

    async def validate(self) -> Dict[str, Any]:
        """Validate all components are properly configured."""
        validation_results = {"components": {}, "is_valid": True}
        # Check assistant configuration
        assistant_validation = self.assistant_config.validate()
        validation_results["components"]["assistants"] = assistant_validation
        if not assistant_validation["valid"]:
            validation_results["is_valid"] = False
        # Check if the tool executor is set
        has_tool_executor = (
            self.assistant_manager.tool_executor._tool_executor is not None
        )
        validation_results["components"]["tool_executor"] = {
            "configured": has_tool_executor
        }
        # Check intent service
        try:
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
