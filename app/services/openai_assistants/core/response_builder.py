"""
Builds and formats responses for Rails backend compatibility.
"""

import logging
from typing import Dict, List, Any
from ..assistant_manager import AssistantResponse, AssistantType

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """
    Utility class for building response objects and managing response structure.

    This class provides static methods to construct and format responses, both
    successful and error cases, for a conversational assistant system. Additionally,
    it contains methods to process tool results, classify metadata, and manage
    response logging. The class aims to produce outputs compliant with specific
    integrations and detailed enough for system debugging.

    Methods include functionality to handle both user-intended responses and
    exceptions in execution, while ensuring proper metadata alignment for subsequent
    processing layers.
    """

    @staticmethod
    def build_response(
        assistant_response: AssistantResponse,
        actions: List[Dict[str, Any]],
        classification: Dict[str, Any],
        routing_result: Dict[str, Any],
        query: str,
        user_id: str,
        final_assistant: AssistantType,
        initial_assistant: AssistantType,
    ) -> Dict[str, Any]:
        """
        Build the final response in the expected format.

        Args:
            assistant_response: Response from assistant
            actions: Processed actions
            classification: Intent classification results
            routing_result: Routing decision results
            query: Original user query
            user_id: User identifier
            final_assistant: Final assistant that handled query
            initial_assistant: Initial assistant attempted

        Returns:
            Formatted response dictionary
        """
        # Create tool_results for Rails compatibility
        tool_results = ResponseBuilder._create_tool_results(
            assistant_response.function_calls
        )

        # Build classification metadata
        classification_meta = ResponseBuilder._build_classification_metadata(
            classification, final_assistant, initial_assistant
        )

        # Build the complete response
        response = {
            "message": assistant_response.content,
            "response_text": assistant_response.content,  # Rails expects this key
            "actions": actions,
            "tool_results": tool_results,  # Rails expects this key
            "classification": classification_meta,
            "routing": {
                "strategy": routing_result["routing"]["strategy"],
                "fallback_options": routing_result.get("fallback_options", []),
            },
            "success": assistant_response.success,
            "metadata": ResponseBuilder._build_metadata(
                user_id, query, assistant_response
            ),
        }

        ResponseBuilder._log_response_summary(response)
        return response

    @staticmethod
    def build_error_response(
        error: Exception, query: str = "", user_id: str = ""
    ) -> Dict[str, Any]:
        """
        Build an error response.

        Args:
            error: Exception that occurred
            query: Original query (if available)
            user_id: User identifier (if available)

        Returns:
            Error response dictionary
        """
        return {
            "message": "I apologize, but I encountered an error processing your request. Please try again.",
            "response_text": "I apologize, but I encountered an error processing your request. Please try again.",
            "actions": [],
            "tool_results": [],
            "classification": {
                "intent": "general",
                "confidence": 0.0,
                "assistant_used": "general",
                "method": "error",
            },
            "routing": {"strategy": "error"},
            "success": False,
            "error": str(error),
            "metadata": {
                "user_id": user_id,
                "query_length": len(query),
                "error_type": type(error).__name__,
            },
        }

    @staticmethod
    def _create_tool_results(function_calls: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create tool_results for Rails compatibility.

        Args:
            function_calls: List of function call results

        Returns:
            List of tool result dictionaries
        """
        tool_results = []

        for func_call in function_calls:
            tool_results.append(
                {
                    "tool": func_call.get("function"),
                    "parameters": func_call.get("arguments", {}),
                    "result": func_call.get("result", {}),
                }
            )

        logger.info(f"🔧 Generated {len(tool_results)} tool_results for Rails")
        return tool_results

    @staticmethod
    def _build_classification_metadata(
        classification: Dict[str, Any],
        final_assistant: AssistantType,
        initial_assistant: AssistantType,
    ) -> Dict[str, Any]:
        """
        Build classification metadata.

        Args:
            classification: Initial classification results
            final_assistant: Final assistant used
            initial_assistant: Initial assistant attempted

        Returns:
            Classification metadata dictionary
        """
        return {
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "assistant_used": final_assistant.value,
            "method": classification.get("method", "unknown"),
            "rerouted": final_assistant != initial_assistant,
            "original_assistant": (
                initial_assistant.value
                if final_assistant != initial_assistant
                else None
            ),
        }

    @staticmethod
    def _build_metadata(
        user_id: str, query: str, assistant_response: AssistantResponse
    ) -> Dict[str, Any]:
        """
        Build response metadata.

        Args:
            user_id: User identifier
            query: Original query
            assistant_response: Response from assistant

        Returns:
            Metadata dictionary
        """
        metadata = {
            "user_id": user_id,
            "query_length": len(query),
            "function_calls_count": len(assistant_response.function_calls),
        }

        # Merge with assistant response metadata
        if assistant_response.metadata:
            metadata.update(assistant_response.metadata)

        return metadata

    @staticmethod
    def _log_response_summary(response: Dict[str, Any]):
        """
        Log a summary of the response.

        Args:
            response: Complete response dictionary
        """
        logger.info(f"📤 Final response keys: {list(response.keys())}")
        logger.info(f"📤 Response has message: {bool(response.get('response_text'))}")
        logger.info(f"📤 Tool results count: {len(response.get('tool_results', []))}")
        logger.info(f"📤 Actions count: {len(response.get('actions', []))}")
