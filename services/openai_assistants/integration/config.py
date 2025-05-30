"""
Configuration for OpenAI Integration Service.
Manages service-level settings and initialization.
"""

import logging
from typing import Optional, Dict, Any
from ..assistant_manager import AsyncAssistantManager, AssistantConfig
from ...intent_classification.intent_service import IntentService
from ..core.router import AssistantRouter
from ..core.intent_mapper import IntentMapper
from ..processors.function_processor import FunctionProcessor
from ..utils.constants import CROSS_ROUTING_PATTERNS

logger = logging.getLogger(__name__)


class IntegrationConfig:
    """Configuration and component initialization for integration service."""

    def __init__(self):
        """Initialize all components."""
        # Initialize core managers
        self.assistant_config = AssistantConfig()
        self.assistant_manager = AsyncAssistantManager(self.assistant_config)
        self.intent_service = IntentService()

        # Initialize components
        self.router = AssistantRouter(CROSS_ROUTING_PATTERNS)
        self.intent_mapper = IntentMapper()
        self.function_processor = FunctionProcessor()

        # Setup tool executor
        self._setup_tool_executor()

        logger.info("✅ Integration components initialized")

    def _setup_tool_executor(self):
        """Configure tool executor from existing Fin service."""
        try:
            from services.fin.tool_registry import ToolRegistry

            tool_registry = ToolRegistry()

            # Create an async wrapper for tool executor
            async def async_tool_executor(
                tool_name: str, tool_args: Dict[str, Any], **kwargs
            ) -> Dict[str, Any]:
                """Async wrapper for tool registry execution."""
                try:
                    import asyncio

                    loop = asyncio.get_event_loop()

                    return await loop.run_in_executor(
                        None,
                        tool_registry.execute,
                        tool_name,
                        tool_args,
                        kwargs.get("user_id", "unknown"),
                        kwargs.get("transactions", []),
                        kwargs.get("user_context", {}),
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    return {"error": f"Tool execution failed: {str(e)}"}

            self.assistant_manager.set_tool_executor(async_tool_executor)
            logger.info("✅ Async tool executor configured")

        except Exception as e:
            logger.error(f"❌ Failed to setup tool executor: {e}")

    def validate(self) -> Dict[str, Any]:
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
            test_result = self.intent_service.classify_and_route("test")
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
