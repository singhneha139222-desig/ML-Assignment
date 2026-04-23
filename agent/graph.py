from typing import Dict

from langgraph.graph import END, StateGraph

from agent.decision_engine import DecisionEngine
from agent.intent_engine import IntentEngine
from agent.rag_pipeline import RAGPipeline
from agent.state import AgentPhase, AgentState, DecisionAction, initialize_state
from agent.tools import execute_lead_capture
from utils.logger import get_logger
from utils.validators import extract_entities_from_text


class SocialToLeadAgent:
    def __init__(self, knowledge_path: str = "data/knowledge.json") -> None:
        self.logger = get_logger("SocialToLeadAgent")
        self.intent_engine = IntentEngine()
        self.rag = RAGPipeline(knowledge_path=knowledge_path)
        self.decision_engine = DecisionEngine()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("decide", self._decision_node)
        workflow.add_node("query_handling", self._query_handling_node)
        workflow.add_node("lead_detection", self._lead_detection_node)
        workflow.add_node("info_collection", self._info_collection_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("complete", self._complete_node)

        workflow.set_entry_point("classify_intent")

        workflow.add_edge("classify_intent", "decide")

        workflow.add_conditional_edges(
            "decide",
            self._route_from_decision,
            {
                DecisionAction.ANSWER_RAG.value: "query_handling",
                DecisionAction.MOVE_TO_LEAD.value: "lead_detection",
                DecisionAction.ASK_FOLLOW_UP.value: "info_collection",
                DecisionAction.EXECUTE_TOOL.value: "tool_execution",
                DecisionAction.COMPLETE.value: "complete",
            },
        )

        workflow.add_conditional_edges(
            "lead_detection",
            self._route_after_lead_detection,
            {
                DecisionAction.ASK_FOLLOW_UP.value: "info_collection",
                DecisionAction.EXECUTE_TOOL.value: "tool_execution",
            },
        )

        workflow.add_edge("query_handling", "complete")
        workflow.add_edge("info_collection", "complete")
        workflow.add_edge("tool_execution", "complete")
        workflow.add_edge("complete", END)

        return workflow.compile()

    def _classify_intent_node(self, state: AgentState) -> AgentState:
        user_input = state.get("current_input", "")
        history = [
            f"{entry.get('role', 'user')}: {entry.get('content', '')}"
            for entry in state.get("conversation_history", [])
        ]

        classification = self.intent_engine.classify(user_input, history)
        entities = dict(state.get("entities", {}))
        extracted = extract_entities_from_text(user_input)
        entities.update(extracted)

        intent_progression = list(state.get("intent_progression", []))
        intent_progression.append(classification.intent)

        return {
            **state,
            "phase": AgentPhase.START.value,
            "intent": classification.intent,
            "intent_confidence": classification.confidence,
            "intent_reasoning": classification.reasoning,
            "entities": entities,
            "intent_progression": intent_progression,
            "error": "",
        }

    def _decision_node(self, state: AgentState) -> AgentState:
        action, phase, missing_fields = self.decision_engine.decide(state)
        return {
            **state,
            "next_action": action.value,
            "phase": phase.value,
            "missing_fields": missing_fields,
        }

    def _query_handling_node(self, state: AgentState) -> AgentState:
        answer, chunks = self.rag.answer_with_context(state.get("current_input", ""), k=3)
        response = answer

        if state.get("lead_active") and state.get("missing_fields"):
            response += "\n\nIf you still want to continue signup, please share your " + state["missing_fields"][0] + "."

        return {
            **state,
            "phase": AgentPhase.QUERY_HANDLING.value,
            "response": response,
            "retrieved_chunks": chunks,
        }

    def _lead_detection_node(self, state: AgentState) -> AgentState:
        missing = self.decision_engine.get_missing_fields(state.get("entities", {}))

        if missing:
            return {
                **state,
                "phase": AgentPhase.INFO_COLLECTION.value,
                "lead_active": True,
                "missing_fields": missing,
                "next_action": DecisionAction.ASK_FOLLOW_UP.value,
            }

        return {
            **state,
            "phase": AgentPhase.TOOL_EXECUTION.value,
            "lead_active": True,
            "missing_fields": [],
            "next_action": DecisionAction.EXECUTE_TOOL.value,
        }

    def _info_collection_node(self, state: AgentState) -> AgentState:
        missing = self.decision_engine.get_missing_fields(state.get("entities", {}))
        current_input = state.get("current_input", "")

        prompts: Dict[str, str] = {
            "name": "To proceed, please share your full name.",
            "email": "Please provide a valid email address so we can contact you.",
            "platform": "Which platform are you publishing on (for example YouTube, Instagram, TikTok)?",
        }

        if not missing:
            return {
                **state,
                "phase": AgentPhase.INFO_COLLECTION.value,
                "response": "Thanks. I have everything required and will proceed with lead capture now.",
                "next_action": DecisionAction.EXECUTE_TOOL.value,
            }

        first_missing = missing[0]
        retry_text = ""
        if first_missing == "email" and "@" in current_input and not state.get("entities", {}).get("email"):
            retry_text = "The email format looks invalid. "

        return {
            **state,
            "phase": AgentPhase.INFO_COLLECTION.value,
            "missing_fields": missing,
            "response": retry_text + prompts[first_missing],
        }

    def _tool_execution_node(self, state: AgentState) -> AgentState:
        entities = state.get("entities", {})

        success, result = execute_lead_capture(
            name=entities.get("name", ""),
            email=entities.get("email", ""),
            platform=entities.get("platform", ""),
        )

        if success:
            response = "Perfect, your lead is captured. A specialist will reach out shortly."
        else:
            response = result + " Please provide valid details to continue."

        return {
            **state,
            "phase": AgentPhase.TOOL_EXECUTION.value,
            "lead_capture_result": result,
            "tool_called": success,
            "lead_active": not success,
            "response": response,
        }

    def _complete_node(self, state: AgentState) -> AgentState:
        response = state.get("response", "").strip()

        if not response:
            if state.get("intent") == "GREETING":
                response = "Hello. I can help with AutoStream questions or onboarding when you are ready."
            else:
                response = "Done."

        return {
            **state,
            "phase": AgentPhase.COMPLETE.value,
            "response": response,
        }

    def _route_from_decision(self, state: AgentState) -> str:
        return state.get("next_action", DecisionAction.COMPLETE.value)

    def _route_after_lead_detection(self, state: AgentState) -> str:
        return state.get("next_action", DecisionAction.ASK_FOLLOW_UP.value)

    def process_message(self, user_message: str, state: AgentState = None):
        active_state = initialize_state() if state is None else dict(state)
        active_state["current_input"] = user_message
        active_state["response"] = ""
        active_state["phase"] = AgentPhase.START.value

        history = list(active_state.get("conversation_history", []))
        history.append({"role": "user", "content": user_message})
        active_state["conversation_history"] = history

        updated_state = self.graph.invoke(active_state)

        assistant_reply = updated_state.get("response", "")
        updated_history = list(updated_state.get("conversation_history", []))
        updated_history.append({"role": "assistant", "content": assistant_reply})
        updated_state["conversation_history"] = updated_history

        return assistant_reply, updated_state
