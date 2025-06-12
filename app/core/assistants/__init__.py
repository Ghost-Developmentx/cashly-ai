"""
Unified assistant management system.
Replaces the complex factory pattern with configuration-driven approach.
"""

from .manager import (
    UnifiedAssistantManager,
    AssistantType,
    AssistantConfig,
    AssistantResponse
)
from .factory import AssistantFactory

# For backward compatibility during migration
from .migration import (
    AsyncAssistantManager,
    AssistantFactoryManager,
    create_assistant_manager,
    create_factory_manager
)

__all__ = [
    # New unified system
    "UnifiedAssistantManager",
    "AssistantFactory",
    "AssistantType",
    "AssistantConfig",
    "AssistantResponse",

    # Migration compatibility
    "AsyncAssistantManager",
    "AssistantFactoryManager",
    "create_assistant_manager",
    "create_factory_manager"
]