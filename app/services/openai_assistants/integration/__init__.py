"""
OpenAI Integration Service package.
Provides a thin wrapper around QueryPipeline for API compatibility.
"""

# Import config first to ensure tools are registered
from . import config
from .service import OpenAIIntegrationService

__all__ = ["OpenAIIntegrationService"]

# Version
__version__ = "3.0.0"  # Major version bump for pipeline integration