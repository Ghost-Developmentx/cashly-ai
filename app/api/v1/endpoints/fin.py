"""
Fin conversational AI endpoints using OpenAI Assistants.
Replaces Flask FinController.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from logging import getLogger
from datetime import datetime

from app.core.dependencies import get_openai_service, get_current_user_id
from app.core.exceptions import ValidationError, ExternalServiceError
from app.api.v1.schemas.fin import (
    QueryRequest,
    QueryResponse,
    ConversationMessage,
    UserContext,
    AssistantHealth,
    AnalyticsRequest,
    AnalyticsResponse,
    ConversationHistory,
    Action,
    ToolResult,
    Classification,
)
from app.api.v1.schemas.base import SuccessResponse
from app.services.openai_assistants import OpenAIIntegrationService

logger = getLogger(__name__)
router = APIRouter()


@router.post(
    "/conversations/query",
    response_model=QueryResponse,
    summary="Process natural language query",
    description="Process financial queries using OpenAI Assistants",
)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    service: OpenAIIntegrationService = Depends(get_openai_service),
) -> QueryResponse:
    """Process a natural language query using OpenAI Assistants."""
    try:
        # Prepare user context
        user_context = request.user_context.dict() if request.user_context else {}

        # Ensure user_id is in context
        if "user_id" not in user_context:
            user_context["user_id"] = request.user_id

        # Add conversation_id if not present
        if not user_context.get("conversation_id"):
            user_context["conversation_id"] = (
                f"conv_{request.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

        # Convert conversation history
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata,
                }
                for msg in request.conversation_history
            ]

        logger.info(
            f"📥 Processing query for user {request.user_id}: {request.query[:50]}..."
        )

        # Process query
        result = await service.process_financial_query(
            query=request.query,
            user_id=request.user_id,
            user_context=user_context,
            conversation_history=conversation_history,
        )

        if not result.get("success", False):
            raise ExternalServiceError(
                "OpenAI", result.get("error", "Processing failed")
            )

        # Format response
        response = QueryResponse(
            message=result["message"],
            response_text=result["response_text"],
            actions=[Action(**action) for action in result.get("actions", [])],
            tool_results=[
                ToolResult(**tool) for tool in result.get("tool_results", [])
            ],
            classification=Classification(**result["classification"]),
            routing=result["routing"],
            success=result["success"],
            metadata=result.get("metadata", {}),
            conversation_id=user_context.get("conversation_id"),
        )

        # Log analytics in background
        if response.success:
            background_tasks.add_task(
                _log_query_analytics,
                user_id=request.user_id,
                query=request.query,
                intent=response.classification.intent,
                assistant=response.classification.assistant_used,
                success=True,
            )

        logger.info(
            f"📤 Processed query successfully. "
            f"Intent: {response.classification.intent}, "
            f"Actions: {len(response.actions)}"
        )

        return response

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ExternalServiceError as e:
        logger.error(f"OpenAI service error: {e}")
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}"
        )


@router.get(
    "/health",
    response_model=AssistantHealth,
    summary="Check OpenAI Assistant health",
    description="Get health status of OpenAI Assistant system",
)
async def health_check(
    service: OpenAIIntegrationService = Depends(get_openai_service),
) -> AssistantHealth:
    """Health check for the OpenAI Assistants system."""
    try:
        health = await service.health_check()

        return AssistantHealth(
            status=health["status"],
            components=health["components"],
            available_assistants=health["summary"]["available_assistants"],
            missing_assistants=health["summary"]["missing_assistants"],
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Return degraded status instead of failing
        return AssistantHealth(
            status="unhealthy",
            components={"error": {"status": "error", "message": str(e)}},
            available_assistants=[],
            missing_assistants=[],
        )


@router.post(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Get query analytics",
    description="Get analytics for user queries and assistant usage",
)
async def get_analytics(
    request: AnalyticsRequest,
    service: OpenAIIntegrationService = Depends(get_openai_service),
) -> AnalyticsResponse:
    """Get analytics for recent queries and assistant usage."""
    try:
        analytics = service.get_analytics(
            user_id=request.user_id, recent_queries=request.recent_queries
        )

        return AnalyticsResponse(
            user_id=request.user_id,
            intent_analytics=analytics["intent_analytics"],
            assistant_usage=analytics["assistant_usage"],
            performance_metrics=analytics["performance_metrics"],
            total_queries=analytics["total_queries"],
        )

    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate analytics: {str(e)}"
        )


@router.delete(
    "/conversations/{user_id}",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Clear conversation",
    description="Clear conversation history for a user",
)
async def clear_conversation(
    user_id: str, service: OpenAIIntegrationService = Depends(get_openai_service)
) -> SuccessResponse[Dict[str, Any]]:
    """Clear conversation history for a user."""
    try:
        service.clear_conversation(user_id)

        return SuccessResponse(
            data={"user_id": user_id, "cleared_at": datetime.now().isoformat()},
            message="Conversation cleared successfully",
        )

    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear conversation: {str(e)}"
        )


@router.get(
    "/conversations/{user_id}/history",
    response_model=ConversationHistory,
    summary="Get conversation history",
    description="Get recent conversation history for a user",
)
async def get_conversation_history(
    user_id: str,
    limit: int = 10,
    service: OpenAIIntegrationService = Depends(get_openai_service),
) -> ConversationHistory:
    """Get conversation history for a user."""
    try:
        history = await service.get_conversation_history(user_id, limit)

        # Format messages
        messages = [
            ConversationMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                timestamp=(
                    datetime.fromisoformat(msg["timestamp"])
                    if "timestamp" in msg
                    else None
                ),
                metadata=msg.get("metadata"),
            )
            for msg in history
        ]

        return ConversationHistory(
            user_id=user_id,
            conversation_id=f"conv_{user_id}",
            messages=messages,
            created_at=messages[0].timestamp if messages else datetime.now(),
            last_message_at=messages[-1].timestamp if messages else datetime.now(),
            message_count=len(messages),
        )

    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get conversation history: {str(e)}"
        )


async def _log_query_analytics(
    user_id: str, query: str, intent: str, assistant: str, success: bool
):
    """Background task to log query analytics."""
    try:
        logger.info(
            f"Analytics: user={user_id}, intent={intent}, "
            f"assistant={assistant}, success={success}"
        )
        # In production, this would write to analytics database
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")
