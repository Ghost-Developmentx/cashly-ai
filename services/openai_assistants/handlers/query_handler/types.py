"""
Type definitions for query handler.
Defines data structures used throughout query processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from ...assistant_manager import AssistantType, AssistantResponse


class ProcessingPhase(Enum):
    """
    ProcessingPhase enumeration class.

    Defines the various phases involved in the processing pipeline. This enumeration
    is utilized to classify and identify the distinct phases during processing tasks
    to ensure clear communication and systematic control flow.

    Attributes
    ----------
    CLASSIFICATION : str
        Represents the classification phase in the processing lifecycle.
    ROUTING : str
        Represents the routing phase in the processing lifecycle.
    EXECUTION : str
        Represents the execution phase in the processing lifecycle.
    REROUTING : str
        Represents the rerouting phase in the processing lifecycle.
    RESPONSE_BUILDING : str
        Represents the response building phase in the processing lifecycle.
    """
    CLASSIFICATION = "classification"
    ROUTING = "routing"
    EXECUTION = "execution"
    REROUTING = "rerouting"
    RESPONSE_BUILDING = "response_building"


@dataclass
class QueryContext:
    """
    Represents the context of a user's query in a conversational system.

    This class holds metadata for a user's query, such as the query text,
    user identification details, associated user-specific context, and
    conversational history. It facilitates access to critical information
    about the user’s interaction by encapsulating all necessary fields and
    provides utility properties to evaluate the availability of user context
    and conversation history.

    Attributes
    ----------
    query : str
        The text of the query initiated by the user.
    user_id : str
        The unique identifier for the user making the query.
    user_context : Optional[Dict[str, Any]]
        The contextual information specific to the user.
    conversation_history : Optional[List[Dict[str, Any]]]
        The previous interaction or messages in the current conversation flow.
    """
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
    """
    Data class to encapsulate the results of a processing pipeline.

    This class stores responses from an assistant, actions, classification data, routing results,
    and other related metadata about processing. It is used to centralize and organize the outcome
    of a query and its associated attributes for easier consumption across various application layers.

    Attributes
    ----------
    assistant_response : AssistantResponse
        The response object from the assistant containing the processed query details.
    actions : List[Dict[str, Any]]
        A list of actions determined as part of the processing result.
    classification : Dict[str, Any]
        The classification result of the query, structured as a dictionary.
    routing_result : Dict[str, Any]
        Information related to the query's routing determination.
    initial_assistant : AssistantType
        The type or ID of the assistant initially handling the query.
    final_assistant : AssistantType
        The type or ID of the assistant eventually handling the query.
    processing_time : float
        The total time taken for the processing, measured in seconds.
    rerouted : bool, optional
        Indicates whether the query was rerouted during processing. Defaults to False.
    """
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
    """
    Represents a decision regarding the routing of a task or query.

    This class encapsulates the details of a routing decision, including whether
    rerouting should occur, the target destination for the rerouting, the level
    of confidence in the decision, and the reason for the decision. It is intended
    to be used in systems where routing logic is dynamically determined based on
    various factors.

    Attributes
    ----------
    should_reroute : bool
        Indicates whether rerouting should occur.
    target_assistant : Optional[AssistantType]
        The target assistant to which rerouting is directed if applicable.
    confidence : float
        The confidence level of the routing decision, represented as a number
        between 0.0 and 1.0.
    reason : str
        The justification or explanation for the routing decision.
    """
    should_reroute: bool
    target_assistant: Optional[AssistantType]
    confidence: float
    reason: str

    @property
    def is_confident(self) -> bool:
        """Check if the routing decision is confident."""
        return self.confidence >= 0.7