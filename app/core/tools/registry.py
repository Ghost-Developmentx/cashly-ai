from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import inspect
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolDefinition:
    """Definition of a tool with all its metadata."""
    name: str
    description: str
    handler: Callable
    schema: Dict[str, Any]
    requires_confirmation: bool = False
    category: str = "general"

class ToolRegistry:
    """
    Single source of truth for all tools in the system.
    Replaces duplicate tool registries and provides a unified interface.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
        logger.info("Initializing unified ToolRegistry")

    def register(
            self,
            name: str,
            description: str,
            schema: Dict[str, Any],
            category: str = "general",
            requires_confirmation: bool = False
    ) -> Callable:
        """
        Decorator to register tools with the registry.

        Usage:
            @tool_registry.register(
                name="get_transactions",
                description="Get user transactions",
                schema={...})
            async def get_transactions(context):
                ...
        """
        def decorator(func: Callable) -> Callable:
            # Validate the handler is callable
            if not callable(func):
                raise ValueError(f"Tool handler for {name} must be callable")

            tool_def = ToolDefinition(
                name=name,
                description=description,
                handler=func,
                schema=schema,
                requires_confirmation=requires_confirmation,
                category=category
            )

            self._tools[name] = tool_def

            # Track by category
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)

            logger.info(f"Registered tool: {name} in category: {category}")
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get just the handler function for a tool."""
        tool = self._tools.get(name)
        return tool.handler if tool else None

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_tools_by_category(self, category: str) -> List[str]:
        """List tools in a specific category."""
        return self._categories.get(category, [])

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas in the format expected by our system.
        Compatible with the existing TOOL_SCHEMAS format.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
                "category": tool.category,
                "requires_confirmation": tool.requires_confirmation
            }
            for tool in self._tools.values()
        ]

    def get_openai_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get tool configurations in OpenAI Assistant format.
        Replaces the duplicate logic in assistant factories.

        Args:
            tool_names: Specific tools to include. If None, includes all tools.

        Returns:
            List of OpenAI-compatible tool configurations
        """
        tools_to_include = tool_names or list(self._tools.keys())

        openai_tools = []
        for tool_name in tools_to_include:
            tool = self._tools.get(tool_name)
            if tool:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema
                    }
                })
            else:
                logger.warning(f"Tool {tool_name} not found in registry")

        return openai_tools

    def validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic validation of tool arguments against schema.
        Returns validated args or raises ValueError.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Here you could add JSON schema validation
        # For now, just return the args
        return args

    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get metadata about a tool for logging/debugging."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Tool {tool_name} not found"}

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "requires_confirmation": tool.requires_confirmation,
            "is_async": inspect.iscoroutinefunction(tool.handler),
            "parameters": list(tool.schema.get("properties", {}).keys())
        }

tool_registry = ToolRegistry()

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Get all tool schemas - backwards compatible function."""
    return tool_registry.get_schemas()


def get_openai_tools(tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get OpenAI-formatted tools - backwards compatible function."""
    return tool_registry.get_openai_tools(tool_names)