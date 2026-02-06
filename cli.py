#!/usr/bin/env python3
"""
Command-line interface for the Autonomous Teaching Assistant.
"""

import argparse
import sys
from pathlib import Path

from src.autonomous_ta.agent import TextbookAgent


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Teaching Assistant - Answer questions from textbook content"
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question to answer",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output including retrieved chunks",
    )

    args = parser.parse_args()

    try:
        print("Initializing agent...")
        agent = TextbookAgent(model=args.model)
        print("\nAnswering question...\n")
        
        answer, chunks = agent.answer_question(args.question, top_k=args.top_k, model=args.model)
        
        if answer:
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(answer)
            
            if args.verbose and chunks:
                print("\n" + "=" * 60)
                print("RETRIEVED CHUNKS")
                print("=" * 60)
                for i, chunk in enumerate(chunks, 1):
                    print(f"\n[{i}] Chapter: {chunk['chapter']}")
                    print(f"    Page: {chunk['page']}")
                    print(f"    Book: {chunk['book']}")
                    print(f"    Distance: {chunk.get('distance', 'N/A'):.4f}")
                    if args.verbose:
                        print(f"    Preview: {chunk['chunk_text'][:200]}...")
        else:
            print("Unable to generate an answer for this question.")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure you have:")
        print("1. Processed your PDF using: python -m src.autonomous_ta.load_data")
        print("2. Placed the PDF in data/raw/ directory", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
