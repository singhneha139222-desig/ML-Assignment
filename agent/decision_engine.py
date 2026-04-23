from typing import Dict, List, Tuple

from agent.state import AgentPhase, AgentState, DecisionAction, IntentLabel
from utils.validators import validate_email, validate_name, validate_platform


class DecisionEngine:
    REQUIRED_FIELDS = ["name", "email", "platform"]

    def get_missing_fields(self, entities: Dict[str, str]) -> List[str]:
        missing: List[str] = []

        name = entities.get("name", "")
        email = entities.get("email", "")
        platform = entities.get("platform", "")

        if not validate_name(name):
            missing.append("name")
        if not validate_email(email):
            missing.append("email")
        if not validate_platform(platform):
            missing.append("platform")

        return missing

    def decide(self, state: AgentState) -> Tuple[DecisionAction, AgentPhase, List[str]]:
        intent = state.get("intent", IntentLabel.QUERY.value)
        entities = state.get("entities", {})
        lead_active = state.get("lead_active", False)

        missing_fields = self.get_missing_fields(entities)

        if lead_active and intent == IntentLabel.QUERY.value:
            return DecisionAction.ANSWER_RAG, AgentPhase.QUERY_HANDLING, missing_fields

        if intent == IntentLabel.GREETING.value and not lead_active:
            return DecisionAction.COMPLETE, AgentPhase.COMPLETE, missing_fields

        if intent == IntentLabel.QUERY.value and not lead_active:
            return DecisionAction.ANSWER_RAG, AgentPhase.QUERY_HANDLING, missing_fields

        if intent == IntentLabel.HIGH_INTENT.value and not lead_active:
            return DecisionAction.MOVE_TO_LEAD, AgentPhase.LEAD_DETECTION, missing_fields

        if lead_active:
            if not missing_fields:
                return DecisionAction.EXECUTE_TOOL, AgentPhase.TOOL_EXECUTION, missing_fields
            return DecisionAction.ASK_FOLLOW_UP, AgentPhase.INFO_COLLECTION, missing_fields

        return DecisionAction.COMPLETE, AgentPhase.COMPLETE, missing_fields
