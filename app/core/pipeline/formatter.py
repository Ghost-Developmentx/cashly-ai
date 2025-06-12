"""
Response formatter - formats responses consistently for the API.
Single responsibility: formatting only.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ...schemas.classification import ClassificationResult
from .router import RoutingDecision
from .executor import ExecutionResult

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Format pipeline results into consistent API responses.
    Ensures all responses follow the same structure.
    """

    @staticmethod
    def format_success_response(
            execution_result: ExecutionResult,
            classification: ClassificationResult,
            routing: RoutingDecision,
            query: str,
            user_id: str,
            processing_time: float
    ) -> Dict[str, Any]:
        """
        Format a successful query response.

        Args:
            execution_result: Result from query execution
            classification: Classification result
            routing: Routing decision
            query: Original query
            user_id: User identifier
            processing_time: Total processing time

        Returns:
            Formatted response dictionary
        """
        assistant_response = execution_result.assistant_response

        # Extract actions from function calls
        actions = ResponseFormatter._format_actions(assistant_response.function_calls)

        # Build response
        response = {
            "success": True,
            "message": assistant_response.content,
            "response_text": assistant_response.content,  # For Rails compatibility
            "actions": actions,
            "tool_results": ResponseFormatter._format_tool_results(assistant_response.function_calls),
            "classification": {
                "intent": classification.intent.value,
                "confidence": classification.confidence,
                "assistant_used": assistant_response.assistant_type.value,
                "method": classification.method,
                "rerouted": routing.should_reroute,
                "original_assistant": routing.alternative_assistant.value if routing.alternative_assistant else None
            },
            "routing": {
                "decision": routing.assistant.value,
                "confidence": routing.confidence,
                "reason": routing.reason
            },
            "metadata": {
                "query": query,
                "user_id": user_id,
                "processing_time": round(processing_time, 3),
                "execution_time": round(execution_result.execution_time, 3),
                "tool_calls_count": execution_result.tool_calls_count,
                "timestamp": datetime.now().isoformat(),
                "thread_id": assistant_response.thread_id,
                **assistant_response.metadata
            }
        }

        # Add any error info if present
        if assistant_response.error:
            response["warnings"] = [assistant_response.error]

        return response

    @staticmethod
    def format_error_response(
            error: Exception,
            query: str,
            user_id: str,
            classification: Optional[ClassificationResult] = None,
            routing: Optional[RoutingDecision] = None
    ) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error: The exception that occurred
            query: Original query
            user_id: User identifier
            classification: Optional classification result
            routing: Optional routing decision

        Returns:
            Formatted error response
        """
        error_message = str(error)

        # Determine user-friendly error message
        if "not configured" in error_message.lower():
            user_message = "This feature is not currently available. Please contact support."
        elif "timeout" in error_message.lower():
            user_message = "The request took too long to process. Please try again."
        elif "rate limit" in error_message.lower():
            user_message = "Too many requests. Please wait a moment and try again."
        else:
            user_message = "I encountered an error processing your request. Please try again or contact support if the issue persists."

        response = {
            "success": False,
            "message": user_message,
            "response_text": user_message,
            "actions": [],
            "tool_results": [],
            "error": error_message,
            "metadata": {
                "query": query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__
            }
        }

        # Add classification info if available
        if classification:
            response["classification"] = {
                "intent": classification.intent.value,
                "confidence": classification.confidence,
                "assistant_used": None,
                "method": classification.method,
                "rerouted": False
            }

        # Add routing info if available
        if routing:
            response["routing"] = {
                "decision": routing.assistant.value,
                "confidence": routing.confidence,
                "reason": routing.reason
            }

        return response

    @staticmethod
    def _format_actions(function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format function calls into frontend actions.

        Args:
            function_calls: List of function call results

        Returns:
            List of formatted actions
        """
        actions = []

        # Map function names to action types
        action_mapping = {
            "create_transaction": "create_transaction",
            "update_transaction": "update_transaction",
            "delete_transaction": "delete_transaction",
            "initiate_plaid_connection": "connect_bank",
            "disconnect_account": "disconnect_account",
            "create_invoice": "create_invoice",
            "send_invoice": "send_invoice",
            "setup_stripe_connect": "setup_stripe",
            "forecast_cash_flow": "show_forecast",
            "generate_budget": "show_budget",
            "detect_anomalies": "show_anomalies"
        }

        for call in function_calls:
            function_name = call.get("function", "")
            result = call.get("result", {})

            # Skip if there's an error in the result
            if result.get("error"):
                continue

            # Map to action type
            action_type = action_mapping.get(function_name, f"show_{function_name}")

            # Create action
            action = {
                "type": action_type,
                "data": result,
                "function_called": function_name
            }

            actions.append(action)

        return actions

    @staticmethod
    def _format_tool_results(function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format function calls into tool results.

        Args:
            function_calls: List of function call results

        Returns:
            List of formatted tool results
        """
        tool_results = []

        for call in function_calls:
            tool_result = {
                "tool": call.get("function", "unknown"),
                "parameters": call.get("arguments", {}),
                "result": call.get("result", {}),
                "success": not bool(call.get("result", {}).get("error"))
            }

            tool_results.append(tool_result)

        return tool_results

    @staticmethod
    def format_streaming_response(
            content_chunk: str,
            is_final: bool = False,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format a streaming response chunk.

        Args:
            content_chunk: Chunk of content to stream
            is_final: Whether this is the final chunk
            metadata: Optional metadata

        Returns:
            Formatted streaming response
        """
        response = {
            "type": "content",
            "content": content_chunk,
            "is_final": is_final
        }

        if metadata:
            response["metadata"] = metadata

        return response
