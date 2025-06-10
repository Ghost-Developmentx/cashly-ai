"""
Unified tool system.
Single source of truth for all tools in the application.
"""

from .registry import tool_registry, get_tool_schemas, get_openai_tools
from .executor import ToolExecutor

from .handlers import (
    transactions,
    accounts,
    invoices,
    stripe,
    analytics
)

__all__ = [
    "tool_registry",
    "ToolExecutor",
    "get_tool_schemas",
    "get_openai_tools"
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"Tool system initialized with {len(tool_registry.list_tools())} tools")