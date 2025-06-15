# app/core/tools/__init__.py
"""
Unified tool system.
Single source of truth for all tools in the application.
"""

from .registry import tool_registry, get_tool_schemas, get_openai_tools
from .executor import ToolExecutor

# Import all handlers to ensure registration
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

# Log tool registration status
registered_tools = tool_registry.list_tools()
logger.info(f"✅ Tool system initialized with {len(registered_tools)} tools")
logger.debug(f"Registered tools: {registered_tools}")