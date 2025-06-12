"""
Configuration for OpenAI Integration Service.
Manages service-level settings and initialization.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IntegrationConfig(BaseModel):
    """
    Configuration for integration components, now using Pydantic BaseModel.
    """

    # Define fields for all the components we'll initialize
    assistant_manager: Optional[Any] = Field(default=None)
    intent_service: Optional[Any] = Field(default=None)
    router: Optional[Any] = Field(default=None)
    intent_mapper: Optional[Any] = Field(default=None)
    function_processor: Optional[Any] = Field(default=None)
    tool_executor: Optional[Any] = Field(default=None)

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
            from app.services.intent_classification.async_intent_service import AsyncIntentService
            from app.services.openai_assistants.core.router import AssistantRouter
            from app.services.openai_assistants.core.intent_mapper import IntentMapper
            from app.services.openai_assistants.processors.function_processor import FunctionProcessor
            from app.services.openai_assistants.utils.constants import CROSS_ROUTING_PATTERNS

            # Use new unified assistant manager
            from app.core.assistants import UnifiedAssistantManager

            # Initialize components
            self.assistant_manager = UnifiedAssistantManager()
            self.intent_service = AsyncIntentService()
            self.router = AssistantRouter(CROSS_ROUTING_PATTERNS)
            self.intent_mapper = IntentMapper()
            self.function_processor = FunctionProcessor()

            # Setup unified tool executor
            self._setup_tool_executor()

        except ImportError as e:
            # Handle missing imports gracefully
            logger.warning(f"Could not initialize some components: {e}")

    def _setup_tool_executor(self):
        """Configure unified tool executor."""
        try:
            # Import the unified tool system
            from app.core.tools import ToolExecutor

            # Import handlers to ensure they're registered
            import app.core.tools.handlers.transactions
            import app.core.tools.handlers.accounts
            import app.core.tools.handlers.invoices
            import app.core.tools.handlers.stripe
            import app.core.tools.handlers.analytics

            # Initialize Rails client if available
            rails_client = None
            try:
                from app.services.fin.async_rails_client import AsyncRailsClient
                rails_client = AsyncRailsClient()
            except ImportError:
                logger.warning("Rails client not available")

            # Create tool executor with Rails client
            self.tool_executor = ToolExecutor(rails_client=rails_client)

            # Set the tool executor on the assistant manager
            async def tool_executor_wrapper(
                    tool_name: str,
                    tool_args: Dict[str, Any],
                    **kwargs
            ) -> Dict[str, Any]:
                """Wrapper to match expected signature."""
                return await self.tool_executor.execute(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    user_id=kwargs.get("user_id", "unknown"),
                    transactions=kwargs.get("transactions", []),
                    user_context=kwargs.get("user_context", {})
                )

            self.assistant_manager.set_tool_executor(tool_executor_wrapper)

            logger.info("✅ Unified tool executor configured successfully")

        except Exception as e:
            logger.warning(f"Failed to setup unified tool executor: {e}")

    async def validate(self) -> Dict[str, Any]:
        """Validate all components are properly configured."""
        validation_results = {"components": {}, "is_valid": True}

        # Check assistant manager
        if self.assistant_manager:
            try:
                assistant_validation = await self.assistant_manager.validate_all_assistants()
                validation_results["components"]["assistants"] = assistant_validation
                if not assistant_validation.get("valid", True):
                    validation_results["is_valid"] = False
            except Exception as e:
                validation_results["components"]["assistants"] = {
                    "status": "error",
                    "error": str(e)
                }
                validation_results["is_valid"] = False

        # Check if the tool executor is set
        validation_results["components"]["tool_executor"] = {
            "configured": self.tool_executor is not None,
            "using_unified_system": True
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

        # Validate tool registry
        try:
            from app.core.tools import tool_registry
            registered_tools = tool_registry.list_tools()
            validation_results["components"]["tool_registry"] = {
                "status": "healthy",
                "registered_tools_count": len(registered_tools),
                "tools": registered_tools[:10]  # Show first 10
            }
        except Exception as e:
            validation_results["components"]["tool_registry"] = {
                "status": "error",
                "error": str(e)
            }

        return validation_results
