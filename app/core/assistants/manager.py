"""
Unified Assistant Manager - Single source of truth for all assistants.
Replaces the complex factory pattern with configuration-driven approach.
"""

import logging
import yaml
import os
import asyncio
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from openai import AsyncOpenAI
from ...schemas.assistant import AssistantConfig, AssistantResponse, AssistantType
from .helpers.assistant_helpers import enhance_query_with_context
from app.core.tools import tool_registry
from app.core.config import get_settings
from dotenv import load_dotenv

load_dotenv()

settings = get_settings()
logger = logging.getLogger(__name__)


class UnifiedAssistantManager:
    """
    Unified manager for all OpenAI assistants.
    Replaces AssistantFactoryManager and multiple factory classes.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with a configuration file."""
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "assistants.yaml"
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

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

            # Process environment variables
            config_str = yaml.dump(config)
            config_str = os.path.expandvars(config_str)
            config = yaml.safe_load(config_str)

            # Load assistant configurations
            for assistant_key, assistant_config in config.get('assistants', {}).items():
                try:
                    assistant_type = AssistantType(assistant_key)
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
        """Query an assistant with user input."""
        config = self.assistant_configs.get(assistant_type)
        if not config or not config.assistant_id:
            raise ValueError(f"Assistant {assistant_type} not configured")

        # Get or create thread
        thread_id = await self._get_or_create_thread(user_id)

        # Enhance query with context
        enhanced_query = await enhance_query_with_context(
            query, user_context, conversation_history
        )

        # Create message
        await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=enhanced_query
        )

        # Create and poll run
        run = await self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=config.assistant_id
        )

        # Wait for completion and handle tool calls
        result = await self._wait_for_run_completion(
            thread_id, run.id, user_id, user_context
        )

        return AssistantResponse(
            content=result['content'],
            assistant_type=assistant_type,
            function_calls=result['function_calls'],
            metadata={
                'thread_id': thread_id,
                'run_id': run.id,
                'model': config.model
            },
            success=True,
            thread_id=thread_id
        )

    async def create_or_update_assistant(
            self,
            assistant_type: AssistantType,
            force_update: bool = False
    ) -> str:
        """Create or update an assistant based on configuration."""
        config = self.assistant_configs.get(assistant_type)
        if not config:
            raise ValueError(f"No configuration for assistant type: {assistant_type}")

        # Get OpenAI-formatted tools from registry
        tools = tool_registry.get_openai_tools(config.tools)

        assistant_config = {
            "name": config.name,
            "instructions": config.instructions,
            "model": config.model,
            "tools": tools  # Use properly formatted tools from registry
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

        # Create a new assistant
        assistant = await self.client.beta.assistants.create(**assistant_config)

        # Update config with new ID
        config.assistant_id = assistant.id

        # Save to environment variable
        env_key = f"{assistant_type.value.upper()}_ASSISTANT_ID"
        os.environ[env_key] = assistant.id

        logger.info(f"Created assistant {assistant_type.value}: {assistant.id}")
        logger.info(f"Set environment variable {env_key}={assistant.id}")
        logger.info(f"Registered tools: {[tool['function']['name'] for tool in tools]}")

        return assistant.id

    async def validate_all_assistants(self) -> Dict[str, Any]:
        """Validate all assistant configurations."""
        results = {
            "valid": True,
            "assistants": {},
            "summary": {
                "total": len(self.assistant_configs),
                "configured": 0,
                "missing": 0,
                "tool_issues": 0
            }
        }

        for assistant_type, config in self.assistant_configs.items():
            assistant_result = {
                "has_id": bool(config.assistant_id),
                "tools_requested": config.tools,
                "tools_found": [],
                "tools_missing": [],
                "tools_valid": True
            }

            # Check tools
            for tool_name in config.tools:
                if tool_registry.get_tool(tool_name):
                    assistant_result["tools_found"].append(tool_name)
                else:
                    assistant_result["tools_missing"].append(tool_name)
                    assistant_result["tools_valid"] = False

            # Update summary
            if config.assistant_id:
                results["summary"]["configured"] += 1
            else:
                results["summary"]["missing"] += 1

            if not assistant_result["tools_valid"]:
                results["summary"]["tool_issues"] += 1
                results["valid"] = False

            results["assistants"][assistant_type.value] = assistant_result

        return results

    async def _get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread or create new one for user."""
        if user_id in self._user_threads:
            return self._user_threads[user_id]

        thread = await self.client.beta.threads.create()
        self._user_threads[user_id] = thread.id
        return thread.id

    async def _wait_for_run_completion(
            self,
            thread_id: str,
            run_id: str,
            user_id: str,
            user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Wait for run completion and handle tool calls."""
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
                if self._tool_executor and run.required_action:
                    tool_outputs = []
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    for tool_call in tool_calls:
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
                    logger.error("Tool executor not configured or no action required")
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

    def clear_user_thread(self, user_id: str) -> bool:
        """Clear thread for a user."""
        if user_id in self._user_threads:
            del self._user_threads[user_id]
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all assistants."""
        validation = await self.validate_all_assistants()

        return {
            "status": "healthy" if validation["valid"] else "degraded",
            "assistants": validation["assistants"],
            "summary": validation["summary"],
            "tool_registry": {
                "total_tools": len(tool_registry.list_tools()),
                "tools": tool_registry.list_tools()
            }
        }

    def get_assistant_info(self, assistant_type: AssistantType) -> Dict[str, Any]:
        """Get information about a specific assistant."""
        config = self.assistant_configs.get(assistant_type)
        if not config:
            return {"error": f"No configuration for {assistant_type}"}

        return {
            "type": assistant_type.value,
            "name": config.name,
            "model": config.model,
            "tools": config.tools,
            "has_assistant_id": bool(config.assistant_id),
            "configured": bool(config.assistant_id)
        }
