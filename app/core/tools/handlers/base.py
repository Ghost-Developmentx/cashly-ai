"""
Base handler class for tool implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseToolHandler(ABC):
    """
    Base class for tool handlers.
    Not strictly required but provides a template for complex handlers.
    """

    def __init__(self, rails_client=None):
        self.rails_client = rails_client

    @abstractmethod
    async def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments before execution."""
        pass

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given context."""
        pass
