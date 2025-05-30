"""
Type definitions for query handler.
Defines data structures used throughout query processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from ...assistant_manager import AssistantType, AssistantResponse


class ProcessingPhase(Enum):
    """Query processing phases."""
    CLASSIFICATION = "classification"
    ROUTING = "routing"
    EXECUTION = "execution"
    REROUTING = "rerouting"
    RESPONSE_BUILDING = "response_building"


@dataclass
class QueryContext:
    """Context for query processing."""
    query: str
    user_id: str
    user_context: Optional[Dict[str, Any]]
    conversation_history: Optional[List[Dict[str, Any]]]

    @property
    def has_context(self) -> bool:
        """Check if user context is available."""
        return self.user_context is not None and len(self.user_context) > 0

    @property
    def has_history(self) -> bool:
        """Check if the conversation history exists."""
        return self.conversation_history is not None and len(self.conversation_history) > 0


@dataclass
class ProcessingResult:
    """Result of query processing."""
    assistant_response: AssistantResponse
    actions: List[Dict[str, Any]]
    classification: Dict[str, Any]
    routing_result: Dict[str, Any]
    initial_assistant: AssistantType
    final_assistant: AssistantType
    processing_time: float
    rerouted: bool = False

    @property
    def was_rerouted(self) -> bool:
        """Check if a query was rerouted."""
        return self.initial_assistant != self.final_assistant


@dataclass
class RoutingDecision:
    """Routing decision information."""
    should_reroute: bool
    target_assistant: Optional[AssistantType]
    confidence: float
    reason: str

    @property
    def is_confident(self) -> bool:
        """Check if the routing decision is confident."""
        return self.confidence >= 0.7