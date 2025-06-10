"""
Base Assistant Factory for OpenAI Assistants.
Provides common functionality for all assistant factories.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseAssistantFactory(ABC):
    """
    Abstract base class for OpenAI Assistant factories.
    Provides common functionality and defines the interface for specialized factories.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    @abstractmethod
    def get_assistant_config(self) -> Dict[str, Any]:
        """
        Get the configuration for this assistant type.
        Must be implemented by each specialized factory.

        Returns:
            Dictionary containing assistant configuration
        """
        pass

    @abstractmethod
    def get_assistant_name(self) -> str:
        """
        Get the name for this assistant type.
        Must be implemented by each specialized factory.

        Returns:
            Human-readable assistant name
        """
        pass

    def create_assistant(self) -> str:
        """
        Create an OpenAI assistant with the configuration from get_assistant_config().

        Returns:
            Assistant ID of the created assistant

        Raises:
            Exception: If assistant creation fails
        """
        try:
            config = self.get_assistant_config()
            assistant = self.client.beta.assistants.create(**config)

            assistant_name = self.get_assistant_name()
            logger.info(f"Created {assistant_name}: {assistant.id}")

            return assistant.id

        except Exception as e:
            assistant_name = self.get_assistant_name()
            logger.error(f"Failed to create {assistant_name}: {e}")
            raise

    def update_assistant(self, assistant_id: str, **updates) -> bool:
        """
        Update an existing assistant with new configuration.

        Args:
            assistant_id: ID of the assistant to update
            **updates: Configuration updates to apply

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Get current config and merge with updates
            current_config = self.get_assistant_config()
            updated_config = {**current_config, **updates}

            self.client.beta.assistants.update(
                assistant_id=assistant_id, **updated_config
            )

            assistant_name = self.get_assistant_name()
            logger.info(f"Updated {assistant_name}: {assistant_id}")

            return True

        except Exception as e:
            assistant_name = self.get_assistant_name()
            logger.error(f"Failed to update {assistant_name} {assistant_id}: {e}")
            return False

    def delete_assistant(self, assistant_id: str) -> bool:
        """
        Delete an assistant.

        Args:
            assistant_id: ID of the assistant to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.client.beta.assistants.delete(assistant_id=assistant_id)

            assistant_name = self.get_assistant_name()
            logger.info(f"Deleted {assistant_name}: {assistant_id}")

            return True

        except Exception as e:
            assistant_name = self.get_assistant_name()
            logger.error(f"Failed to delete {assistant_name} {assistant_id}: {e}")
            return False

    def get_assistant_info(self, assistant_id: str) -> Dict[str, Any]:
        """
        Get information about an assistant.

        Args:
            assistant_id: ID of the assistant

        Returns:
            Dictionary containing assistant information
        """
        try:
            assistant = self.client.beta.assistants.retrieve(assistant_id)

            return {
                "id": assistant.id,
                "name": assistant.name,
                "model": assistant.model,
                "instructions": assistant.instructions,
                "tools": [tool.type for tool in assistant.tools],
                "created_at": assistant.created_at,
                "factory_type": self.__class__.__name__,
            }

        except Exception as e:
            logger.error(f"Error getting assistant info for {assistant_id}: {e}")
            return {"error": str(e)}

    def _get_function_schema(self, function_name: str) -> Dict[str, Any]:
        from app.core.tools.registry import tool_registry

        tool = tool_registry.get_tool(function_name)
        if tool:
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema
            }

        return self._get_fallback_schema(function_name)

    @staticmethod
    def _get_fallback_schema(function_name: str) -> Dict[str, Any]:
        """
        Get a fallback schema for functions not found in tool schemas.

        Args:
            function_name: Name of the function

        Returns:
            Fallback function schema
        """
        # Common fallback schemas for frequently used functions
        fallback_schemas = {
            "get_transactions": {
                "name": "get_transactions",
                "description": "Retrieve and filter user transactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back",
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category",
                        },
                        "account_id": {
                            "type": "string",
                            "description": "Filter by account",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["income", "expense", "all"],
                        },
                    },
                },
            },
            "create_transaction": {
                "name": "create_transaction",
                "description": "Create a new transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Transaction amount",
                        },
                        "description": {
                            "type": "string",
                            "description": "Transaction description",
                        },
                        "account_id": {"type": "string", "description": "Account ID"},
                        "category": {
                            "type": "string",
                            "description": "Transaction category",
                        },
                        "date": {
                            "type": "string",
                            "description": "Transaction date (YYYY-MM-DD)",
                        },
                    },
                    "required": ["amount", "description"],
                },
            },
            "get_user_accounts": {
                "name": "get_user_accounts",
                "description": "Get user's connected bank accounts",
                "parameters": {"type": "object", "properties": {}},
            },
            "initiate_plaid_connection": {
                "name": "initiate_plaid_connection",
                "description": "Start bank account connection process",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "institution_preference": {
                            "type": "string",
                            "description": "Preferred bank type",
                        }
                    },
                },
            },
            "setup_stripe_connect": {
                "name": "setup_stripe_connect",
                "description": "Set up Stripe Connect for payment processing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {"type": "string", "default": "US"},
                        "business_type": {"type": "string", "default": "individual"},
                    },
                },
            },
            "create_invoice": {
                "name": "create_invoice",
                "description": "Create a new invoice",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_name": {"type": "string", "description": "Client name"},
                        "client_email": {
                            "type": "string",
                            "description": "Client email",
                        },
                        "amount": {"type": "number", "description": "Invoice amount"},
                        "description": {
                            "type": "string",
                            "description": "Invoice description",
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date (YYYY-MM-DD)",
                        },
                    },
                    "required": ["client_name", "client_email", "amount"],
                },
            },
            "forecast_cash_flow": {
                "name": "forecast_cash_flow",
                "description": "Generate cash flow forecast",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to forecast",
                        },
                        "adjustments": {
                            "type": "object",
                            "description": "Scenario adjustments",
                        },
                    },
                    "required": ["days"],
                },
            },
            "generate_budget": {
                "name": "generate_budget",
                "description": "Generate budget recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "monthly_income": {
                            "type": "number",
                            "description": "Monthly income amount",
                        }
                    },
                },
            },
            "analyze_trends": {
                "name": "analyze_trends",
                "description": "Analyze financial trends",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {"type": "string", "enum": ["1m", "3m", "6m", "1y"]}
                    },
                    "required": ["period"],
                },
            },
        }

        return fallback_schemas.get(
            function_name,
            {
                "name": function_name,
                "description": f"Execute {function_name}",
                "parameters": {"type": "object", "properties": {}},
            },
        )

    @staticmethod
    def _build_tools_list(function_names: List[str]) -> List[Dict[str, Any]]:
        from app.core.tools.registry import tool_registry

        return tool_registry.get_openai_tools(function_names)

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the assistant configuration before creation.

        Returns:
            Validation results with any issues found
        """
        config = self.get_assistant_config()
        issues = []
        warnings = []

        # Check required fields
        required_fields = ["name", "instructions", "model"]
        for field in required_fields:
            if not config.get(field):
                issues.append(f"Missing required field: {field}")

        # Check instruction length (OpenAI has limits)
        instructions = config.get("instructions", "")
        if len(instructions) > 32000:  # OpenAI limit is around 32k characters
            issues.append(
                f"Instructions too long: {len(instructions)} characters (max ~32000)"
            )

        # Check tool configurations
        tools = config.get("tools", [])
        if len(tools) > 128:  # OpenAI limit
            issues.append(f"Too many tools: {len(tools)} (max 128)")

        # Check for duplicate function names
        function_names = []
        for tool in tools:
            if tool.get("type") == "function":
                func_name = tool.get("function", {}).get("name")
                if func_name:
                    if func_name in function_names:
                        warnings.append(f"Duplicate function: {func_name}")
                    else:
                        function_names.append(func_name)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config_summary": {
                "name": config.get("name"),
                "instruction_length": len(instructions),
                "tool_count": len(tools),
                "function_count": len(function_names),
            },
        }
