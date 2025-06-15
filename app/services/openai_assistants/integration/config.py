# app/services/openai_assistants/integration/config.py
"""
Configuration for OpenAI Integration Service.
Simplified to only handle tool registration since pipeline handles everything else.
"""

import logging

logger = logging.getLogger(__name__)


def ensure_tools_registered():
    """
    Ensure all tool handlers are imported and registered.
    This is critical for the tools to be available to the assistants.
    """
    try:
        # Import all tool handlers to trigger registration
        import app.core.tools.handlers.transactions
        import app.core.tools.handlers.accounts
        import app.core.tools.handlers.invoices
        import app.core.tools.handlers.stripe
        import app.core.tools.handlers.analytics

        # Verify registration
        from app.core.tools import tool_registry
        registered_count = len(tool_registry.list_tools())

        logger.info(f"✅ Tool handlers imported successfully")
        logger.info(f"✅ Registered {registered_count} tools: {tool_registry.list_tools()[:5]}...")

        return True

    except ImportError as e:
        logger.error(f"Failed to import tool handlers: {e}")
        raise


# Run registration on module import
ensure_tools_registered()
