import os
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from utils.logger import get_logger


class IntentClassification(BaseModel):
    intent: str = Field(description="One of GREETING, QUERY, HIGH_INTENT")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class IntentEngine:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.logger = get_logger("IntentEngine")
        self.parser = PydanticOutputParser(pydantic_object=IntentClassification)

        self.prompt = PromptTemplate.from_template(
            """
You are an intent classifier for AutoStream (SaaS video editing platform).
Classify the user message into exactly one class:
- GREETING: social/opening messages with no product ask and no buying intent
- QUERY: informational/product questions (pricing, policies, features)
- HIGH_INTENT: clear buying/trial/demo/signup/contact intent

Return JSON only.
{format_instructions}

Conversation snippet:
{history}

User message:
{message}
""".strip()
        )

        self.llm = None
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            self.logger.warning(
                "OPENAI_API_KEY not found. Intent classification will use deterministic fallback."
            )

    def classify(self, message: str, history: List[str]) -> IntentClassification:
        history_block = "\n".join(history[-6:]) if history else ""

        if self.llm is None:
            return self._fallback_classification(message)

        try:
            payload = self.prompt.format(
                format_instructions=self.parser.get_format_instructions(),
                history=history_block,
                message=message,
            )
            raw = self.llm.invoke(payload).content
            parsed = self.parser.parse(raw)
            normalized = parsed.intent.strip().upper()
            if normalized not in {"GREETING", "QUERY", "HIGH_INTENT"}:
                raise ValueError(f"Unsupported intent label: {normalized}")
            parsed.intent = normalized
            return parsed
        except Exception as exc:
            self.logger.exception("LLM classification failed; using fallback. Error: %s", exc)
            fallback = self._fallback_classification(message)
            fallback.reasoning += " Fallback triggered due to classifier failure."
            return fallback

    def _fallback_classification(self, message: str) -> IntentClassification:
        text = message.lower().strip()

        from utils.validators import extract_entities_from_text
        if extract_entities_from_text(message):
            return IntentClassification(
                intent="HIGH_INTENT",
                confidence=0.85,
                reasoning="Message contains entity information, continuing lead flow.",
            )

        high_intent_markers = [
            "buy",
            "purchase",
            "subscribe",
            "trial",
            "demo",
            "book",
            "sales",
            "contact",
            "sign up",
            "get started",
            "interested",
        ]
        greeting_markers = ["hi", "hello", "hey", "good morning", "good evening"]

        if any(marker in text for marker in high_intent_markers):
            return IntentClassification(
                intent="HIGH_INTENT",
                confidence=0.82,
                reasoning="Message includes transactional intent markers.",
            )

        if any(text == marker or text.startswith(marker + " ") for marker in greeting_markers):
            return IntentClassification(
                intent="GREETING",
                confidence=0.91,
                reasoning="Message is primarily a greeting with no product objective.",
            )

        question_words = ["what", "how", "why", "when", "where", "is", "can", "do", "does", "?"]
        is_question = any(qw in text.split() for qw in question_words) or "?" in text

        if not is_question and (len(text.split()) <= 3 or "@" in text):
            return IntentClassification(
                intent="HIGH_INTENT",
                confidence=0.7,
                reasoning="Message is short or contains special characters, likely a lead flow response rather than a query.",
            )

        return IntentClassification(
            intent="QUERY",
            confidence=0.76,
            reasoning="Message appears to request information.",
        )
