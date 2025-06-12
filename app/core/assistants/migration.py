"""
Migration adapter to support existing code during transition.
Provides backward compatibility while we update the codebase.
"""

import logging
from typing import Dict, Any, Optional, List

from .manager import UnifiedAssistantManager, AssistantType, AssistantResponse
from .factory import AssistantFactory

logger = logging.getLogger(__name__)


class AsyncAssistantManager:
    """
    Backward compatibility wrapper for existing AsyncAssistantManager.
    Maps old interface to new UnifiedAssistantManager.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize with compatibility layer."""
        self.manager = UnifiedAssistantManager()
        self.config = config
        logger.info("Using migration adapter for AsyncAssistantManager")

    async def process_query(
            self,
            query: str,
            assistant_type: AssistantType,
            user_id: str,
            user_context: Optional[Dict] = None,
            conversation_history: Optional[List[Dict]] = None
    ) -> AssistantResponse:
        """Process query using new manager."""
        return await self.manager.query_assistant(
            assistant_type=assistant_type,
            query=query,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history
        )

    def set_tool_executor(self, executor):
        """Set tool executor."""
        self.manager.set_tool_executor(executor)

    @staticmethod
    async def get_conversation_history(
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history (simplified for now)."""
        # In the new system, we'd need to implement thread message retrieval
        logger.warning("get_conversation_history not fully implemented in migration")
        return []

    def clear_thread(self, user_id: str) -> bool:
        """Clear user thread."""
        return self.manager.clear_user_thread(user_id)

    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return await self.manager.health_check()


class AssistantFactoryManager:
    """
    Backward compatibility wrapper for existing AssistantFactoryManager.
    Maps old factory pattern to new configuration-based approach.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with new factory."""
        self.factory = AssistantFactory()
        self.manager = self.factory.manager

        # For compatibility
        self.factories = {}
        self.assistant_registry = {}

        logger.info("Using migration adapter for AssistantFactoryManager")

    async def create_all_assistants(self) -> Dict[str, str]:
        """Create all assistants using new factory."""
        result = await self.factory.create_all_assistants()

        # Update registry for compatibility
        for assistant_type, assistant_id in result.items():
            self.assistant_registry[AssistantType(assistant_type)] = assistant_id

        return result

    def create_assistant(self, assistant_type: AssistantType) -> str:
        """Create single assistant (made async in new system)."""
        import asyncio
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(
            self.factory.create_assistant(assistant_type)
        )

    def get_assistant_info(
            self,
            assistant_type: AssistantType,
            assistant_id: str
    ) -> Dict[str, Any]:
        """Get assistant info."""
        return self.manager.get_assistant_info(assistant_type)

    def validate_all_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Validate configurations."""
        import asyncio
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.factory.validate_assistants())

        # Convert to old format
        old_format = {}
        for assistant_type, info in result.get("assistants", {}).items():
            old_format[assistant_type] = {
                "valid": info.get("tools_valid", False) and info.get("has_id", False),
                "issues": [],
                "warnings": []
            }

            if not info.get("tools_valid"):
                for tool in info.get("tools", []):
                    if not tool["found"]:
                        old_format[assistant_type]["issues"].append(
                            f"Tool not found: {tool['name']}"
                        )

        return old_format

    def health_check(self) -> Dict[str, Any]:
        """Health check."""
        import asyncio
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.manager.health_check())

    def list_available_assistants(self) -> List[Dict[str, str]]:
        """List available assistants."""
        assistants = []

        for assistant_type, config in self.manager.assistant_configs.items():
            assistants.append({
                "type": assistant_type.value,
                "name": config.name,
                "factory": "UnifiedFactory",  # For compatibility
                "created": bool(config.assistant_id)
            })

        return assistants


# Factory functions for migration
def create_assistant_manager(config: Optional[Any] = None) -> AsyncAssistantManager:
    """Create assistant manager with migration adapter."""
    return AsyncAssistantManager(config)


def create_factory_manager(api_key: Optional[str] = None) -> AssistantFactoryManager:
    """Create factory manager with migration adapter."""
    return AssistantFactoryManager(api_key)