#!/usr/bin/env python3
"""
Example script demonstrating how to use the Autonomous Teaching Assistant.
"""

from src.autonomous_ta.agent import TextbookAgent
from src.autonomous_ta.load_data import parse_book


def main():
    print("=" * 60)
    print("Autonomous Teaching Assistant - Example Usage")
    print("=" * 60)
    
    # Step 1: Parse the textbook (only needed once)
    print("\n[Step 1] Parsing textbook...")
    print("Note: Skip this step if you've already parsed your PDF")
    # Uncomment the next line to parse a new PDF:
    # parse_book()
    
    # Step 2: Initialize the agent
    print("\n[Step 2] Initializing agent...")
    agent = TextbookAgent(model="gpt-4o-mini")
    
    # Step 3: Ask questions
    print("\n[Step 3] Asking questions...\n")
    
    questions = [
        "What is regression and why is it useful?",
        "Explain the central limit theorem.",
        "What are the assumptions of linear regression?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}: {question}")
        print("=" * 60)
        
        answer, chunks = agent.answer_question(question, top_k=5)
        
        if answer:
            print(f"\nAnswer:\n{answer}")
            print(f"\nRetrieved {len(chunks)} chunks from the textbook.")
        else:
            print("Unable to generate an answer.")
        
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
