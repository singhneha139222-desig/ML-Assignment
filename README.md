# Social-to-Lead Agentic Workflow for AutoStream

Production-grade conversational agent system for converting social conversations into qualified leads. This project uses an explicit state machine, retrieval-augmented generation (RAG), deterministic transition logic, memory, and strict tool-gating controls to avoid premature lead capture.

## 1. Setup

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies with: pip install -r requirements.txt
3. Optional but recommended: set OPENAI_API_KEY for LLM-based intent classification and contextual answer generation.

PowerShell example:
$env:OPENAI_API_KEY="your_key"

If no API key is provided, the system still runs using deterministic fallback for classification and non-hallucinating retrieval responses.

## 2. Run

Run with:
python main.py

Optional:
python main.py --knowledge data/knowledge.json --log-level DEBUG

CLI commands:
- exit or quit: terminate session
- /reset: clear memory (history, entities, intent progression)

## 3. Architecture (System Design)

The architecture is composed of six layers. Input layer receives user utterances from CLI and appends them to conversation history. Intent classification layer uses structured LLM output (Pydantic JSON schema) with deterministic fallback and emits intent, confidence, and reasoning for every turn. Retrieval layer implements a real RAG stack: knowledge loading from JSON, chunking through RecursiveCharacterTextSplitter, embedding generation via sentence-transformers, indexing in Chroma vector DB, top-k similarity retrieval with a confidence threshold, and context injection into the answer LLM. Decision layer enforces deterministic policy by mapping intent plus lead-state plus entity completeness to explicit actions (ANSWER_RAG, MOVE_TO_LEAD, ASK_FOLLOW_UP, EXECUTE_TOOL, COMPLETE). Tool execution layer exposes a guarded lead-capture function and only executes when the conversation is inside high-intent lead flow and all required fields are valid. State management layer is implemented through LangGraph state and includes history, entities, intent progression, retrieved evidence, phase, and execution results.

LangGraph is preferred over basic LangChain chains here because the workflow is not linear. It has explicit state transitions, conditional routing, deterministic branching, and inspectable graph semantics, all of which are required for governance and production safety. LangChain alone is useful for retrieval and model abstraction, but LangGraph is a better fit for multi-step agent control with strict transition constraints.

The system avoids premature tool execution by separating state progression from action execution: intent classification does not call tools, lead detection does not call tools, and information collection only prompts for missing fields. Tool node is reachable only when transition logic confirms high-intent flow and complete validated entities (name, email, platform).

## 4. State Management and Transitions

States:
- START
- QUERY_HANDLING
- LEAD_DETECTION
- INFO_COLLECTION
- TOOL_EXECUTION
- COMPLETE

Deterministic transition logic:
1. START to intent classification
2. Classification to decision
3. Decision routes to one action node
4. Action node to COMPLETE

Memory persisted across turns:
- conversation_history
- entities
- intent_progression
- lead_active

## 5. RAG Behavior and Hallucination Control

The agent never answers from model prior alone. It retrieves top-k chunks from the vector DB and injects only those chunks into the answer prompt with strict instruction to answer from context only. If retrieval is empty or weak (below threshold), response is exactly: I don't have that information.

## 6. WhatsApp Webhook Integration (Inbound to Agent to Outbound)

Typical integration pattern:
1. WhatsApp provider (Meta Cloud API or Twilio) posts inbound message to your webhook endpoint.
2. Webhook handler extracts sender ID and text.
3. Load per-user agent state from Redis or Postgres.
4. Call agent.process_message(user_text, state).
5. Persist updated state.
6. Send response back using WhatsApp send-message API.

Example flow pseudo-sequence:
- inbound webhook payload to parse from and message
- state_store.get(from)
- reply, state = agent.process_message(message, state)
- state_store.set(from, state)
- whatsapp_client.send_text(from, reply)

## 7. Demo Script (Exact)

Use these exact user turns:
1. User: What are your pricing plans?
2. User: I want to get started with AutoStream for my channel.
3. User: My name is Rahul Verma
4. User: c
5. User: rahul.verma@example.com
6. User: YouTube

Expected path:
- Turn 1: RAG answer from pricing chunks
- Turn 2: transition to high-intent lead flow
- Turn 3-6: stepwise info collection with invalid email retry
- Final turn: lead capture tool executes successfully

## 8. Sample Conversations and Edge Cases

Case A: User asks query during lead collection
- User: I want a demo
- Agent: asks for missing fields
- User: Before that, what is your refund policy?
- Agent: answers from RAG and reminds missing lead field

Case B: User skips email
- User provides name and platform only
- Agent continues requesting email until valid

Case C: Invalid email format
- User: my email is test@bad
- Agent retries for valid email

Case D: Intent shift from greeting to high intent
- User: Hi
- User: I want to subscribe now
- Agent escalates into lead flow

## 9. Project Layout

project/
├── main.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── intent_engine.py
│   ├── rag_pipeline.py
│   ├── decision_engine.py
│   ├── tools.py
├── data/
│   └── knowledge.json
├── utils/
│   ├── validators.py
│   ├── logger.py
├── requirements.txt
├── README.md
