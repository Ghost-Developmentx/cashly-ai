import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AssistantType(Enum):
    TRANSACTION = "transaction"
    ACCOUNT = "account"
    CONNECTION = "connection"
    INVOICE = "invoice"
    FORECASTING = "forecasting"
    BUDGET = "budget"
    INSIGHTS = "insights"
    GENERAL = "general"


@dataclass
class AssistantResponse:
    """Response from an OpenAI Assistant."""

    content: str
    assistant_type: AssistantType
    function_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class AssistantManager:
    """
    Manages OpenAI Assistants for Cashly financial queries.
    Handles routing, function calling, and response processing.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Assistant ID mapping
        self.assistant_ids = {
            AssistantType.TRANSACTION: os.getenv("TRANSACTION_ASSISTANT_ID"),
            AssistantType.ACCOUNT: os.getenv("ACCOUNT_ASSISTANT_ID"),
            AssistantType.CONNECTION: os.getenv("CONNECTION_ASSISTANT_ID"),
            AssistantType.INVOICE: os.getenv("INVOICE_ASSISTANT_ID"),
            AssistantType.FORECASTING: os.getenv("FORECASTING_ASSISTANT_ID"),
            AssistantType.BUDGET: os.getenv("BUDGET_ASSISTANT_ID"),
            AssistantType.INSIGHTS: os.getenv("INSIGHTS_ASSISTANT_ID"),
        }

        # Configuration
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.timeout = int(os.getenv("ASSISTANT_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))

        # Active threads (for conversation continuity)
        self.active_threads: Dict[str, str] = {}  # user_id -> thread_id

        # Tool executor (we'll inject this from your existing Fin service)
        self.tool_executor = None

    def set_tool_executor(self, executor):
        """Set the tool executor from your existing Fin service."""
        self.tool_executor = executor

    async def process_query(
        self,
        query: str,
        assistant_type: AssistantType,
        user_id: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> AssistantResponse:
        """
        Process a query with the specified assistant.

        Args:
            query: User's question
            assistant_type: Which assistant to use
            user_id: User identifier
            user_context: User context (accounts, integrations, etc.)
            conversation_history: Previous messages

        Returns:
            AssistantResponse with the result
        """
        try:
            assistant_id = self.assistant_ids.get(assistant_type)
            if not assistant_id:
                return AssistantResponse(
                    content=f"Assistant {assistant_type.value} is not configured",
                    assistant_type=assistant_type,
                    function_calls=[],
                    metadata={},
                    success=False,
                    error="Assistant not configured",
                )

            # Get or create a thread for this user
            thread_id = await self._get_or_create_thread(user_id)

            # Add context to the message if provided
            enhanced_query = self._enhance_query_with_context(query, user_context)

            # Create a message in a thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=enhanced_query
            )

            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=self._get_additional_instructions(user_context),
            )

            # Wait for completion and handle function calls
            response = await self._wait_for_completion(
                thread_id, run.id, user_id, user_context
            )

            return response

        except Exception as e:
            logger.error(
                f"Error processing query with {assistant_type.value} assistant: {e}"
            )
            return AssistantResponse(
                content="I apologize, but I encountered an error processing your request. Please try again.",
                assistant_type=assistant_type,
                function_calls=[],
                metadata={},
                success=False,
                error=str(e),
            )

    async def _get_or_create_thread(self, user_id: str) -> str:
        """Get existing thread or create new one for user."""
        if user_id in self.active_threads:
            return self.active_threads[user_id]

        # Create a new thread
        thread = self.client.beta.threads.create()
        self.active_threads[user_id] = thread.id

        return thread.id

    @staticmethod
    def _enhance_query_with_context(query: str, user_context: Optional[Dict]) -> str:
        """Add relevant context to the user's query."""
        if not user_context:
            return query

        context_parts = []

        # Add account context
        accounts = user_context.get("accounts", [])
        if accounts:
            account_summary = f"Connected accounts: {len(accounts)} accounts with total balance of ${sum(acc.get('balance', 0) for acc in accounts):,.2f}"
            context_parts.append(account_summary)

        # Add integration context
        integrations = user_context.get("integrations", [])
        if integrations:
            integration_list = [
                integration.get("provider") for integration in integrations
            ]
            context_parts.append(f"Active integrations: {', '.join(integration_list)}")

        # Add transaction context
        transactions = user_context.get("transactions", [])
        if transactions:
            context_parts.append(
                f"Available transaction data: {len(transactions)} transactions"
            )

        if context_parts:
            context_info = "User context: " + "; ".join(context_parts)
            return f"{context_info}\n\nUser query: {query}"

        return query

    @staticmethod
    def _get_additional_instructions(user_context: Optional[Dict]) -> str:
        """Generate additional instructions based on user context."""
        if not user_context:
            return ""

        instructions = []

        # Account-specific instructions
        accounts = user_context.get("accounts", [])
        if not accounts:
            instructions.append(
                "Note: User has no connected bank accounts. Suggest connecting accounts for better insights."
            )
        elif len(accounts) == 1:
            instructions.append(
                "Note: User has one connected account. Consider suggesting additional accounts for comprehensive tracking."
            )

        # Integration-specific instructions
        integrations = user_context.get("integrations", [])
        has_stripe = any(
            integration.get("provider") == "stripe" for integration in integrations
        )
        if not has_stripe:
            instructions.append(
                "Note: User doesn't have Stripe Connect. For invoice-related queries, suggest setting up Stripe Connect."
            )

        return " ".join(instructions)

    async def _wait_for_completion(
        self, thread_id: str, run_id: str, user_id: str, user_context: Optional[Dict]
    ) -> AssistantResponse:
        """Wait for run completion and handle function calls."""
        start_time = time.time()
        function_calls = []

        while time.time() - start_time < self.timeout:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run_id
            )

            if run.status == "completed":
                # Get the assistant's response
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id, limit=1
                )

                if messages.data:
                    content = ""
                    for content_block in messages.data[0].content:
                        if content_block.type == "text":
                            content += content_block.text.value

                    return AssistantResponse(
                        content=content,
                        assistant_type=self._get_assistant_type_from_run(run),
                        function_calls=function_calls,
                        metadata={"run_id": run_id, "thread_id": thread_id},
                        success=True,
                    )

            elif run.status == "requires_action":
                # Handle function calls
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    logger.info(
                        f"Executing function: {function_name} with args: {function_args}"
                    )

                    # Execute the function using your existing tool executor
                    if self.tool_executor:
                        try:
                            result = self.tool_executor(
                                function_name,
                                function_args,
                                user_id=user_id,
                                transactions=(
                                    user_context.get("transactions", [])
                                    if user_context
                                    else []
                                ),
                                user_context=user_context or {},
                            )

                            function_calls.append(
                                {
                                    "function": function_name,
                                    "arguments": function_args,
                                    "result": result,
                                }
                            )

                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "output": json.dumps(result),
                                }
                            )

                        except Exception as e:
                            logger.error(
                                f"Error executing function {function_name}: {e}"
                            )
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "output": json.dumps({"error": str(e)}),
                                }
                            )
                    else:
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(
                                    {"error": "Tool executor not configured"}
                                ),
                            }
                        )

                # Submit tool outputs
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs
                )

            elif run.status in ["failed", "cancelled", "expired"]:
                return AssistantResponse(
                    content="I encountered an error processing your request. Please try again.",
                    assistant_type=self._get_assistant_type_from_run(run),
                    function_calls=function_calls,
                    metadata={
                        "run_id": run_id,
                        "thread_id": thread_id,
                        "status": run.status,
                    },
                    success=False,
                    error=f"Run {run.status}",
                )

            # Wait before checking again
            await asyncio.sleep(0.5)

        # Timeout
        return AssistantResponse(
            content="Your request is taking longer than expected. Please try again.",
            assistant_type=AssistantType.GENERAL,
            function_calls=function_calls,
            metadata={"run_id": run_id, "thread_id": thread_id},
            success=False,
            error="Timeout",
        )

    def _get_assistant_type_from_run(self, run) -> AssistantType:
        """Determine the assistant type from run."""
        assistant_id = run.assistant_id

        for assistant_type, aid in self.assistant_ids.items():
            if aid == assistant_id:
                return assistant_type

        return AssistantType.GENERAL

    def clear_thread(self, user_id: str):
        """Clear the thread for a user (start fresh conversation)."""
        if user_id in self.active_threads:
            del self.active_threads[user_id]

    def get_conversation_history(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        if user_id not in self.active_threads:
            return []

        thread_id = self.active_threads[user_id]

        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id, limit=limit
            )

            history = []
            for message in reversed(messages.data):
                content = ""
                for content_block in message.content:
                    if content_block.type == "text":
                        content += content_block.text.value

                history.append(
                    {
                        "role": message.role,
                        "content": content,
                        "timestamp": message.created_at,
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check the health of all assistants."""
        health = {"status": "healthy", "assistants": {}, "missing_assistants": []}

        for assistant_type, assistant_id in self.assistant_ids.items():
            if assistant_id:
                try:
                    assistant = self.client.beta.assistants.retrieve(assistant_id)
                    health["assistants"][assistant_type.value] = {
                        "id": assistant_id,
                        "name": assistant.name,
                        "model": assistant.model,
                        "status": "active",
                    }
                except Exception as e:
                    health["assistants"][assistant_type.value] = {
                        "id": assistant_id,
                        "status": "error",
                        "error": str(e),
                    }
                    health["status"] = "degraded"
            else:
                health["missing_assistants"].append(assistant_type.value)
                health["status"] = "degraded"

        return health
