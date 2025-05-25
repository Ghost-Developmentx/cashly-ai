import logging
from typing import Dict, List, Any, Optional
from .assistant_manager import AssistantManager, AssistantType, AssistantResponse
from ..intent_classification.intent_service import IntentService

logger = logging.getLogger(__name__)


class OpenAIIntegrationService:
    """Enhanced integration service with seamless assistant routing."""

    def __init__(self):
        self.assistant_manager = AssistantManager()
        self.intent_service = IntentService()

        self.intent_to_assistant = {
            "transactions": AssistantType.TRANSACTION,
            "accounts": AssistantType.ACCOUNT,
            "invoices": AssistantType.INVOICE,
            "forecasting": AssistantType.FORECASTING,
            "budgets": AssistantType.BUDGET,
            "insights": AssistantType.INSIGHTS,
            "general": AssistantType.TRANSACTION,  # Default fallback
            "connection": AssistantType.CONNECTION,
        }

        # Define cross-assistant routing patterns
        self.routing_patterns = {
            AssistantType.CONNECTION: {
                "account_balance": AssistantType.ACCOUNT,
                "total_balance": AssistantType.ACCOUNT,
                "account_details": AssistantType.ACCOUNT,
                "how_much_money": AssistantType.ACCOUNT,
            },
            AssistantType.ACCOUNT: {
                "transactions": AssistantType.TRANSACTION,
                "spending": AssistantType.TRANSACTION,
                "expenses": AssistantType.TRANSACTION,
                "recent_activity": AssistantType.TRANSACTION,
            },
            AssistantType.TRANSACTION: {
                "forecast": AssistantType.FORECASTING,
                "predict": AssistantType.FORECASTING,
                "cash_flow": AssistantType.FORECASTING,
                "future": AssistantType.FORECASTING,
            },
        }

        self._setup_tool_executor()

    def _setup_tool_executor(self):
        """Set up the tool executor from your existing Fin service."""
        try:
            from services.fin.tool_registry import ToolRegistry

            tool_registry = ToolRegistry()

            def tool_executor_wrapper(tool_name, tool_args, **kwargs):
                """Wrapper that calls your tool registry with the correct signature."""
                user_id = kwargs.get("user_id", "unknown")
                user_context = kwargs.get("user_context", {})
                transactions = kwargs.get("transactions", [])

                try:
                    # Call with the EXACT signature: keyword-only arguments after *
                    return tool_registry.execute(
                        tool_name,  # positional
                        tool_args,  # positional
                        user_id=user_id,  # keyword-only (after *)
                        transactions=transactions,  # keyword-only (after *)
                        user_context=user_context,  # keyword-only (after *)
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    return {"error": f"Tool execution failed: {str(e)}"}

            self.assistant_manager.set_tool_executor(tool_executor_wrapper)
            logger.info("✅ Tool executor connected successfully")

        except Exception as e:
            logger.error(f"❌ Failed to connect tool executor: {e}")

    async def process_financial_query(
        self,
        query: str,
        user_id: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced query processing with seamless re-routing.
        """
        try:
            # Step 1: Initial intent classification
            routing_result = self.intent_service.classify_and_route(
                query, user_context, conversation_history
            )

            classification = routing_result["classification"]
            initial_intent = classification["intent"]
            confidence = classification["confidence"]

            logger.info(f"🎯 Intent: {initial_intent} (confidence: {confidence:.2%})")

            # Step 2: Determine which assistant to use
            initial_assistant = self._get_assistant_for_intent(
                initial_intent, routing_result
            )

            logger.info(f"🤖 Initial assistant: {initial_assistant.value}")

            # Step 3: Process with the initial assistant
            assistant_response = await self.assistant_manager.process_query(
                query=query,
                assistant_type=initial_assistant,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history,
            )

            logger.info(f"🔧 Assistant response success: {assistant_response.success}")
            logger.info(
                f"🔧 Function calls count: {len(assistant_response.function_calls)}"
            )
            logger.info(
                f"🔧 Response content length: {len(assistant_response.content)}"
            )

            final_assistant = initial_assistant

            # Step 4: Check if we need to re-route
            if self._should_reroute(assistant_response, initial_assistant, query):
                logger.info("🔄 Attempting seamless re-routing")

                correct_assistant = self._determine_correct_assistant(
                    assistant_response, initial_assistant, query
                )

                if correct_assistant and correct_assistant != initial_assistant:
                    logger.info(
                        f"🔄 Re-routing: {initial_assistant.value} -> {correct_assistant.value}"
                    )

                    rerouted_response = await self.assistant_manager.process_query(
                        query=query,
                        assistant_type=correct_assistant,
                        user_id=user_id,
                        user_context=user_context,
                        conversation_history=conversation_history,
                    )

                    if (
                        rerouted_response.success
                        and len(rerouted_response.function_calls) > 0
                    ):
                        assistant_response = rerouted_response
                        final_assistant = correct_assistant
                        logger.info("✅ Re-routing successful")
                    else:
                        logger.info(
                            "⚠️ Re-routing didn't improve response, using original"
                        )

            # Step 5: Process function calls and create actions
            actions = self._process_function_calls_to_actions(
                assistant_response.function_calls
            )
            logger.info(f"🔧 Generated {len(actions)} actions from function calls")

            # Step 6: Create tool_results for Rails compatibility
            tool_results = []
            for func_call in assistant_response.function_calls:
                tool_results.append(
                    {
                        "tool": func_call.get("function"),
                        "parameters": func_call.get("arguments", {}),
                        "result": func_call.get("result", {}),
                    }
                )

            logger.info(f"🔧 Generated {len(tool_results)} tool_results for Rails")

            # Step 7: Format response in the expected format
            response = {
                "message": assistant_response.content,
                "response_text": assistant_response.content,  # Rails expects this key
                "actions": actions,
                "tool_results": tool_results,  # Rails expects this key
                "classification": {
                    "intent": initial_intent,
                    "confidence": confidence,
                    "assistant_used": final_assistant.value,
                    "method": classification.get("method", "unknown"),
                    "rerouted": final_assistant != initial_assistant,
                    "original_assistant": (
                        initial_assistant.value
                        if final_assistant != initial_assistant
                        else None
                    ),
                },
                "routing": {
                    "strategy": routing_result["routing"]["strategy"],
                    "fallback_options": routing_result.get("fallback_options", []),
                },
                "success": assistant_response.success,
                "metadata": {
                    "user_id": user_id,
                    "query_length": len(query),
                    "function_calls_count": len(assistant_response.function_calls),
                    **assistant_response.metadata,
                },
            }

            logger.info(f"📤 Final response keys: {list(response.keys())}")
            logger.info(
                f"📤 Response has message: {bool(response.get('response_text'))}"
            )
            logger.info(
                f"📤 Tool results count: {len(response.get('tool_results', []))}"
            )

            return response

        except Exception as e:
            logger.error(f"❌ Error processing financial query: {e}")
            import traceback

            traceback.print_exc()

            return {
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "response_text": "I apologize, but I encountered an error processing your request. Please try again.",
                "actions": [],
                "tool_results": [],
                "classification": {"intent": "general", "confidence": 0.0},
                "routing": {"strategy": "error"},
                "success": False,
                "error": str(e),
            }

    def _should_reroute(
        self, response: AssistantResponse, current_assistant: AssistantType, query: str
    ) -> bool:
        """Determine if we should re-route to a different assistant."""

        # If we got function calls and a good response, don't re-route
        if len(response.function_calls) > 0 and response.success:
            return False

        # Check if the response contains routing suggestions
        routing_phrases = [
            "refer to",
            "ask the",
            "contact the",
            "check with the",
            "please use",
            "try the",
            "for this, you'll need",
        ]

        response_lower = response.content.lower()

        for phrase in routing_phrases:
            if phrase in response_lower:
                return True

        # Check if the response is too generic for a specific question
        if len(response.function_calls) == 0 and len(response.content) < 100:
            return True

        # Check specific patterns based on the current assistant
        if current_assistant in self.routing_patterns:
            patterns = self.routing_patterns[current_assistant]
            query_lower = query.lower()

            for pattern_key in patterns.keys():
                if pattern_key in query_lower:
                    return True

        return False

    def _determine_correct_assistant(
        self, response: AssistantResponse, current_assistant: AssistantType, query: str
    ) -> Optional[AssistantType]:
        """Determine which assistant should handle the query."""

        # Check response content for assistant mentions
        response_lower = response.content.lower()

        assistant_mentions = {
            "transaction": AssistantType.TRANSACTION,
            "account": AssistantType.ACCOUNT,
            "connection": AssistantType.CONNECTION,
            "invoice": AssistantType.INVOICE,
            "forecasting": AssistantType.FORECASTING,
            "budget": AssistantType.BUDGET,
            "insights": AssistantType.INSIGHTS,
        }

        for mention, assistant_type in assistant_mentions.items():
            if f"{mention} assistant" in response_lower:
                return assistant_type

        # Check routing patterns
        if current_assistant in self.routing_patterns:
            patterns = self.routing_patterns[current_assistant]
            query_lower = query.lower()

            for pattern_key, target_assistant in patterns.items():
                if pattern_key in query_lower:
                    return target_assistant

        # Fallback: re-classify the query
        routing_result = self.intent_service.classify_and_route(query)
        new_intent = routing_result["classification"]["intent"]
        return self.intent_to_assistant.get(new_intent, AssistantType.TRANSACTION)

    def _get_assistant_for_intent(
        self, intent: str, routing_result: Dict
    ) -> AssistantType:
        """Determine which assistant to use based on intent and routing strategy."""
        strategy = routing_result["routing"]["strategy"]

        # For high confidence, use the mapped assistant
        if strategy == "direct_route":
            return self.intent_to_assistant.get(intent, AssistantType.TRANSACTION)

        # For medium confidence with fallback, still use primary but with caution
        elif strategy == "route_with_fallback":
            return self.intent_to_assistant.get(intent, AssistantType.TRANSACTION)

        # For low confidence, use a more general approach
        elif strategy in ["general_with_context", "general_fallback"]:
            return AssistantType.TRANSACTION

        # Default fallback
        return AssistantType.TRANSACTION

    def _process_function_calls_to_actions(
        self, function_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Convert function calls to frontend actions."""
        actions = []

        for func_call in function_calls:
            function_name = func_call.get("function")
            result = func_call.get("result", {})

            logger.info(f"🔧 Processing function call: {function_name}")
            logger.info(
                f"🔧 Result keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}"
            )

            # Map function names to action types
            action_mapping = {
                "get_transactions": "show_transactions",
                "get_user_accounts": "show_accounts",
                "create_invoice": "invoice_created",
                "get_invoices": "show_invoices",
                "initiate_plaid_connection": "initiate_plaid_connection",
                "setup_stripe_connect": "setup_stripe_connect",
                "forecast_cash_flow": "show_forecast",
                "generate_budget": "show_budget",
                "analyze_trends": "show_trends",
                "detect_anomalies": "show_anomalies",
                "create_transaction": "transaction_created",
                "update_transaction": "transaction_updated",
                "delete_transaction": "transaction_deleted",
            }

            action_type = action_mapping.get(function_name, f"show_{function_name}")

            # Handle special cases
            if result.get("action"):
                # This is an action result (like from Plaid or Stripe)
                actions.append(
                    {
                        "type": result["action"],
                        "data": result,
                        "function_called": function_name,
                    }
                )
            elif result.get("error"):
                # Error result
                actions.append(
                    {
                        "type": "error",
                        "data": result,
                        "function_called": function_name,
                    }
                )
            else:
                # Regular data result
                actions.append(
                    {
                        "type": action_type,
                        "data": result,
                        "function_called": function_name,
                    }
                )

        logger.info(f"🔧 Generated actions: {[a['type'] for a in actions]}")
        return actions

    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user."""
        self.assistant_manager.clear_thread(user_id)

    def get_conversation_history(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        return self.assistant_manager.get_conversation_history(user_id, limit)

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        # Check assistant manager
        assistant_health = self.assistant_manager.health_check()

        # Check intent service
        try:
            test_result = self.intent_service.classify_and_route("test query")
            intent_healthy = True
        except Exception as e:
            intent_healthy = False
            intent_error = str(e)

        # Overall health
        overall_status = "healthy"
        if assistant_health["status"] != "healthy":
            overall_status = "degraded"
        if not intent_healthy:
            overall_status = "unhealthy"

        health_report = {
            "status": overall_status,
            "components": {
                "assistant_manager": assistant_health,
                "intent_service": {
                    "status": "healthy" if intent_healthy else "unhealthy",
                    "error": intent_error if not intent_healthy else None,
                },
            },
            "available_assistants": len(
                [aid for aid in self.assistant_manager.assistant_ids.values() if aid]
            ),
            "missing_assistants": [
                atype.value
                for atype, aid in self.assistant_manager.assistant_ids.items()
                if not aid
            ],
        }

        return health_report

    def get_analytics(self, user_id: str, recent_queries: List[str]) -> Dict[str, Any]:
        """Get analytics for recent queries."""
        try:
            intent_analytics = self.intent_service.get_routing_analytics(recent_queries)

            # Add assistant usage analytics
            assistant_usage = {}
            for query in recent_queries:
                routing = self.intent_service.classify_and_route(query)
                intent = routing["classification"]["intent"]
                assistant = self.intent_to_assistant.get(
                    intent, AssistantType.TRANSACTION
                )
                assistant_usage[assistant.value] = (
                    assistant_usage.get(assistant.value, 0) + 1
                )

            return {
                "intent_analytics": intent_analytics,
                "assistant_usage": assistant_usage,
                "total_queries": len(recent_queries),
            }

        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}
