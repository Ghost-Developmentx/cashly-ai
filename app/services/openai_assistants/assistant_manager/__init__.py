"""
Async OpenAI Assistant Manager package.
Provides a unified interface for assistant operations.
"""

from .types import AssistantType, AssistantResponse
from .config import AssistantConfig
from .manager import AsyncAssistantManager

# For backward compatibility during migration
from .types import AssistantType as AssistantTypeEnum

__all__ = [
    "AsyncAssistantManager",
    "AssistantType",
    "AssistantTypeEnum",
    "AssistantResponse",
    "AssistantConfig"
]

# Version info
__version__ = "2.0.0"