"""
Unified Assistant Manager - Single source of truth for all assistants.
Replaces the complex factory pattern with configuration-driven approach.
"""

import logging
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from openai import AsyncOpenAI
from ...schemas.assistant import AssistantConfig, AssistantResponse, AssistantType
from .helpers.assistant_helpers import enhance_query_with_context
from app.core.tools import tool_registry
from app.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)


class UnifiedAssistantManager:
    """
    Unified manager for all OpenAI assistants.
    Replaces AssistantFactoryManager and multiple factory classes.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with configuration file.

        Args:
            config_path: Path to assistants.yaml (defaults to app/config/assistants.yaml)
        """
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "assistants.yaml"
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(
            f"Initialized UnifiedAssistantManager with API key {settings.OPENAI_API_KEY[:4]}...{settings.OPENAI_API_KEY[-4:]}"
        )

        # Load configuration
        self.assistant_configs: Dict[AssistantType, AssistantConfig] = {}
        self.routing_patterns: Dict[str, Any] = {}
        self._load_configuration()

        # Track threads per user
        self._user_threads: Dict[str, str] = {}

        # Tool executor will be set by integration
        self._tool_executor = None

        logger.info(f"Initialized UnifiedAssistantManager with {len(self.assistant_configs)} assistants")

    def _load_configuration(self):
        """Load assistant configurations from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Process environment variables in the config
            config_str = yaml.dump(config)
            config_str = os.path.expandvars(config_str)
            config = yaml.safe_load(config_str)

            # Load assistant configurations
            for assistant_key, assistant_config in config.get('assistants', {}).items():
                try:
                    assistant_type = AssistantType(assistant_key)

                    # Get assistant ID from the environment
                    env_key = f"{assistant_key.upper()}_ASSISTANT_ID"
                    assistant_id = os.getenv(env_key)

                    self.assistant_configs[assistant_type] = AssistantConfig(
                        name=assistant_config['name'],
                        model=assistant_config['model'],
                        tools=assistant_config['tools'],
                        instructions=assistant_config['instructions'],
                        assistant_id=assistant_id
                    )

                except ValueError:
                    logger.warning(f"Unknown assistant type: {assistant_key}")

            # Load routing patterns
            self.routing_patterns = config.get('routing_patterns', {})

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def set_tool_executor(self, executor):
        """Set the tool executor for function calls."""
        self._tool_executor = executor
        logger.info("Tool executor configured")

    async def query_assistant(
            self,
            assistant_type: AssistantType,
            query: str,
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AssistantResponse:
        """
        Query an assistant with user input.

        Args:
            assistant_type: Which assistant to query
            query: User's query
            user_id: User identifier
            user_context: Optional context (accounts, transactions, etc.)
            conversation_history: Optional previous messages

        Returns:
            AssistantResponse with results
        """
        # Get assistant configuration
        config = self.assistant_configs.get(assistant_type)
        if not config:
            return AssistantResponse(
                content="Assistant type not found",
                assistant_type=assistant_type,
                function_calls=[],
                metadata={"error": "not_configured"},
                success=False,
                error=f"Assistant {assistant_type.value} not configured"
            )

        if not config.assistant_id:
            return AssistantResponse(
                content=f"The {assistant_type.value} assistant is not available",
                assistant_type=assistant_type,
                function_calls=[],
                metadata={"error": "no_assistant_id"},
                success=False,
                error=f"No assistant ID configured for {assistant_type.value}"
            )

        try:
            # Get or create thread for user
            thread_id = await self._get_or_create_thread(user_id)

            # Add context to query if provided
            enhanced_query = enhance_query_with_context(query, user_context)

            # Add message to thread
            await self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=enhanced_query
            )

            # Run assistant
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=config.assistant_id
            )

            # Wait for completion and handle tool calls
            result = await self._wait_for_completion(
                thread_id,
                run.id,
                user_id,
                user_context
            )

            return AssistantResponse(
                content=result['content'],
                assistant_type=assistant_type,
                function_calls=result.get('function_calls', []),
                metadata={
                    'thread_id': thread_id,
                    'run_id': run.id,
                    'tools_used': len(result.get('function_calls', []))
                },
                success=True,
                thread_id=thread_id
            )

        except Exception as e:
            logger.error(f"Error querying assistant {assistant_type}: {e}")
            return AssistantResponse(
                content="I encountered an error processing your request",
                assistant_type=assistant_type,
                function_calls=[],
                metadata={"error": str(e)},
                success=False,
                error=str(e)
            )

    async def _get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread or create new one for user."""
        if user_id not in self._user_threads:
            thread = await self.client.beta.threads.create()
            self._user_threads[user_id] = thread.id
        return self._user_threads[user_id]

    async def _wait_for_completion(
            self,
            thread_id: str,
            run_id: str,
            user_id: str,
            user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Wait for assistant run to complete and handle tool calls."""
        import asyncio
        import json

        function_calls = []

        while True:
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )

            if run.status == 'completed':
                break
            elif run.status == 'requires_action':
                # Handle tool calls
                if self._tool_executor:
                    tool_outputs = []

                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        # Execute tool
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        result = await self._tool_executor(
                            tool_name,
                            tool_args,
                            user_id=user_id,
                            transactions=user_context.get('transactions', []) if user_context else [],
                            user_context=user_context or {}
                        )

                        tool_outputs.append({
                            'tool_call_id': tool_call.id,
                            'output': json.dumps(result)
                        })

                        function_calls.append({
                            'function': tool_name,
                            'arguments': tool_args,
                            'result': result
                        })

                    # Submit tool outputs
                    await self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run_id,
                        tool_outputs=tool_outputs
                    )
                else:
                    logger.error("Tool executor not configured")
                    break
            elif run.status in ['failed', 'cancelled', 'expired']:
                raise Exception(f"Run failed with status: {run.status}")

            await asyncio.sleep(0.5)

        # Get the latest message
        messages = await self.client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=1
        )

        content = ""
        if messages.data:
            for content_block in messages.data[0].content:
                if content_block.type == 'text':
                    content += content_block.text.value

        return {
            'content': content,
            'function_calls': function_calls
        }

    async def create_or_update_assistant(
            self,
            assistant_type: AssistantType,
            force_update: bool = False
    ) -> str:
        """
        Create or update an assistant based on configuration.

        Args:
            assistant_type: Type of assistant to create/update
            force_update: Force update even if assistant exists

        Returns:
            Assistant ID
        """
        config = self.assistant_configs.get(assistant_type)
        if not config:
            raise ValueError(f"No configuration for assistant type: {assistant_type}")

        # Get tool configurations from registry
        tools = []
        for tool_name in config.tools:
            tool = tool_registry.get_tool(tool_name)
            if tool:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema
                    }
                })
            else:
                logger.warning(f"Tool {tool_name} not found in registry")

        assistant_config = {
            "name": config.name,
            "instructions": config.instructions,
            "model": config.model,
            "tools": tools
        }

        if config.assistant_id and not force_update:
            # Update existing assistant
            try:
                await self.client.beta.assistants.update(
                    config.assistant_id,
                    **assistant_config
                )
                logger.info(f"Updated assistant {assistant_type.value}: {config.assistant_id}")
                return config.assistant_id
            except Exception as e:
                logger.error(f"Failed to update assistant: {e}")
                # Fall through to create new one

        # Create new assistant
        assistant = await self.client.beta.assistants.create(**assistant_config)

        # Update config with new ID
        config.assistant_id = assistant.id

        # Save to environment variable
        env_key = f"{assistant_type.value.upper()}_ASSISTANT_ID"
        os.environ[env_key] = assistant.id

        logger.info(f"Created assistant {assistant_type.value}: {assistant.id}")
        logger.info(f"Set environment variable {env_key}={assistant.id}")

        return assistant.id

    async def validate_all_assistants(self) -> Dict[str, Any]:
        """Validate all assistant configurations."""
        results = {
            "valid": True,
            "assistants": {},
            "errors": []
        }

        for assistant_type, config in self.assistant_configs.items():
            assistant_result = {
                "configured": True,
                "has_id": bool(config.assistant_id),
                "tools_valid": True,
                "tools": []
            }

            # Check tools
            for tool_name in config.tools:
                tool = tool_registry.get_tool(tool_name)
                if tool:
                    assistant_result["tools"].append({
                        "name": tool_name,
                        "found": True
                    })
                else:
                    assistant_result["tools"].append({
                        "name": tool_name,
                        "found": False
                    })
                    assistant_result["tools_valid"] = False
                    results["valid"] = False
                    results["errors"].append(f"Tool {tool_name} not found for {assistant_type.value}")

            # Check if assistant exists
            if config.assistant_id:
                try:
                    assistant = await self.client.beta.assistants.retrieve(config.assistant_id)
                    assistant_result["exists"] = True
                    assistant_result["name"] = assistant.name
                except Exception:
                    assistant_result["exists"] = False
                    results["errors"].append(f"Assistant {config.assistant_id} not found for {assistant_type.value}")

            results["assistants"][assistant_type.value] = assistant_result

        return results

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all assistants."""
        health = {
            "status": "healthy",
            "assistants": {},
            "summary": {
                "total": len(self.assistant_configs),
                "configured": 0,
                "healthy": 0
            }
        }

        for assistant_type, config in self.assistant_configs.items():
            assistant_health = {
                "configured": bool(config.assistant_id),
                "status": "not_configured"
            }

            if config.assistant_id:
                health["summary"]["configured"] += 1
                try:
                    # Try to retrieve assistant
                    assistant = await self.client.beta.assistants.retrieve(config.assistant_id)
                    assistant_health["status"] = "healthy"
                    assistant_health["name"] = assistant.name
                    assistant_health["model"] = assistant.model
                    health["summary"]["healthy"] += 1
                except Exception as e:
                    assistant_health["status"] = "error"
                    assistant_health["error"] = str(e)
                    health["status"] = "degraded"

            health["assistants"][assistant_type.value] = assistant_health

        return health

    def clear_user_thread(self, user_id: str) -> bool:
        """Clear thread for a user."""
        if user_id in self._user_threads:
            del self._user_threads[user_id]
            return True
        return False

    def get_assistant_info(self, assistant_type: AssistantType) -> Dict[str, Any]:
        """Get information about an assistant."""
        config = self.assistant_configs.get(assistant_type)
        if not config:
            return {"error": f"Assistant {assistant_type.value} not configured"}

        return {
            "type": assistant_type.value,
            "name": config.name,
            "model": config.model,
            "tools": config.tools,
            "has_id": bool(config.assistant_id),
            "assistant_id": config.assistant_id
        }
