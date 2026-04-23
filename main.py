import argparse
import logging
from pathlib import Path

from agent.graph import SocialToLeadAgent
from agent.state import initialize_state
from utils.logger import set_all_log_levels


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoStream Social-to-Lead Agentic Workflow CLI"
    )
    parser.add_argument(
        "--knowledge",
        default="data/knowledge.json",
        help="Path to knowledge base JSON file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_log_levels(getattr(logging, args.log_level))

    knowledge_path = Path(args.knowledge)
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")

    agent = SocialToLeadAgent(knowledge_path=str(knowledge_path))
    state = initialize_state()

    print("AutoStream Agent is running. Type 'exit' to stop, '/reset' to clear memory.")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Agent: Please enter a message.")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Session closed.")
            break

        if user_input.lower() == "/reset":
            state = initialize_state()
            print("Agent: Memory reset complete.")
            continue

        try:
            response, state = agent.process_message(user_input, state=state)
            print(f"Agent: {response}")
        except Exception as exc:
            print("Agent: Internal error occurred. Please retry.")
            print(f"[debug] {exc}")


if __name__ == "__main__":
    main()
