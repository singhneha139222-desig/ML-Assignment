from enum import Enum
from typing import Dict, List, TypedDict


class IntentLabel(str, Enum):
    GREETING = "GREETING"
    QUERY = "QUERY"
    HIGH_INTENT = "HIGH_INTENT"


class AgentPhase(str, Enum):
    START = "START"
    QUERY_HANDLING = "QUERY_HANDLING"
    LEAD_DETECTION = "LEAD_DETECTION"
    INFO_COLLECTION = "INFO_COLLECTION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    COMPLETE = "COMPLETE"


class DecisionAction(str, Enum):
    ANSWER_RAG = "ANSWER_RAG"
    MOVE_TO_LEAD = "MOVE_TO_LEAD"
    ASK_FOLLOW_UP = "ASK_FOLLOW_UP"
    EXECUTE_TOOL = "EXECUTE_TOOL"
    COMPLETE = "COMPLETE"


class AgentState(TypedDict, total=False):
    current_input: str
    response: str
    phase: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    next_action: str
    lead_active: bool
    entities: Dict[str, str]
    missing_fields: List[str]
    retrieved_chunks: List[str]
    conversation_history: List[Dict[str, str]]
    intent_progression: List[str]
    tool_called: bool
    lead_capture_result: str
    error: str


def initialize_state() -> AgentState:
    return {
        "current_input": "",
        "response": "",
        "phase": AgentPhase.START.value,
        "intent": IntentLabel.QUERY.value,
        "intent_confidence": 0.0,
        "intent_reasoning": "",
        "next_action": DecisionAction.COMPLETE.value,
        "lead_active": False,
        "entities": {},
        "missing_fields": ["name", "email", "platform"],
        "retrieved_chunks": [],
        "conversation_history": [],
        "intent_progression": [],
        "tool_called": False,
        "lead_capture_result": "",
        "error": "",
    }
