import os
from agent.graph import SocialToLeadAgent
from agent.state import initialize_state
from utils.logger import set_all_log_levels
import logging

def run_demo():
    set_all_log_levels(logging.WARNING) # Reduce noise
    agent = SocialToLeadAgent()
    state = initialize_state()

    inputs = [
        "What are your pricing plans?",
        "I want to get started with AutoStream for my channel.",
        "My name is Rahul Verma",
        "rahul-verma@invalid",
        "rahul.verma@example.com",
        "YouTube"
    ]

    print("--- Starting AutoStream Agent Demo ---")
    for i in inputs:
        print(f"\nYou: {i}")
        response, state = agent.process_message(i, state)
        print(f"Agent: {response}")
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    run_demo()
