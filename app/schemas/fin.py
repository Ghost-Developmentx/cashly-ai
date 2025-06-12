"""
Fin conversational AI schemas for OpenAI Assistant integration.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Conversation message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Individual conversation message."""

    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class UserContext(BaseModel):
    """User context for AI processing."""

    user_id: str
    accounts: Optional[List[Dict[str, Any]]] = None
    transactions: Optional[List[Dict[str, Any]]] = None
    stripe_connect: Optional[Dict[str, Any]] = None
    integrations: Optional[List[Dict[str, Any]]] = None
    preferences: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None


class QueryRequest(BaseModel):
    """Natural language query request."""

    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None, max_length=50, description="Recent conversation messages"
    )
    user_context: Optional[UserContext] = None

    @field_validator("query")
    def validate_query_content(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool = True


class Action(BaseModel):
    """Frontend action to perform."""

    type: str = Field(..., description="Action type for frontend")
    data: Dict[str, Any] = Field(..., description="Action data")
    function_called: str = Field(..., description="Backend function that was called")


class Classification(BaseModel):
    """Query classification results."""

    intent: str
    confidence: float = Field(..., ge=0, le=1)
    assistant_used: str
    method: str
    rerouted: bool = False
    original_assistant: Optional[str] = None


class QueryResponse(BaseModel):
    """Complete AI query response."""

    message: str = Field(..., description="AI response message")
    response_text: str = Field(
        ..., description="Same as message for Rails compatibility"
    )
    actions: List[Action] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    classification: Classification
    routing: Dict[str, Any]
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    conversation_id: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConversationHistory(BaseModel):
    """User's conversation history."""

    user_id: str
    conversation_id: str
    messages: List[ConversationMessage]
    created_at: datetime
    last_message_at: datetime
    message_count: int


class AssistantHealth(BaseModel):
    """OpenAI Assistant health status."""

    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, Dict[str, Any]]
    available_assistants: List[str]
    missing_assistants: List[str]


class AnalyticsRequest(BaseModel):
    """Analytics query request."""

    user_id: str = Field(..., min_length=1)
    recent_queries: List[str] = Field(default_factory=list, max_length=100)
    time_period: Optional[str] = Field(None, pattern="^(1d|7d|30d|90d)$")


class AnalyticsResponse(BaseModel):
    """Query analytics response."""

    user_id: str
    intent_analytics: Dict[str, Any]
    assistant_usage: Dict[str, int]
    performance_metrics: Dict[str, Any]
    total_queries: int
    generated_at: datetime = Field(default_factory=datetime.now)
