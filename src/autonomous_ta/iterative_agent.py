from dotenv import load_dotenv
from openai import OpenAI

from src.autonomous_ta.vector_db import VectorDB

import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class IterativeTextbookAgent:
    def __init__(self, max_steps=3):
        self.db = VectorDB()
        self.db.build_index()
        self.max_steps = max_steps

    def choose_chapters(self, question, available_chapters):
        prompt = f"""
        You are planning how to answer a textbook question.

        Question:
        {question}

        Available chapters:
        {available_chapters}

        Return ONLY a JSON array of chapter titles to consult.
        Do not include any explanation or extra text.

        Example:
        ["1.2 Data, Sampling, and Variation in Data and Sampling"]
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(  # fmt: off
                "Failed to parse chapter list. Falling back to all chapters."
            )
            return available_chapters

    def retrieve_chunks(self, question, chapters, top_k=5):
        results = self.db.query(  # fmt: off
            question, top_k=top_k, chapter_keywords=chapters
        )
        return results

    def synthesize_answer(self, question, chunks):
        context = ""
        for chunk in chunks:
            context += f"[{chunk['chapter']} | Page {chunk['page']}]\n"
            context += f"{chunk['chunk_text']}\n\n"
        prompt = f"""
        Use the following textbook excerpts to answer the question.

        Textbook content:
        {context}

        Question:
        {question}

        Answer clearly and concisely.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    def evaluate_answer(self, question, answer):
        prompt = f"""
        Evaluate the following answer to the question.

        Question:
        {question}

        Answer:
        {answer}

        Is this answer COMPLETE and WELL-SUPPORTED by textbook content?
        Respond with ONLY one word: YES or NO.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def answer_question(self, question):
        available_chapters = self.db.list_chapters()
        consulted_chapters = set()
        all_chunks = []
        for step in range(self.max_steps):
            print(f"\nStep {step + 1}")
            chapters = self.choose_chapters(question, available_chapters)
            new_chapters = [
                chapter
                for chapter in chapters
                if chapter not in consulted_chapters  # fmt: off
            ]
            if not new_chapters:
                print("No new chapters to consult.")
                break
            consulted_chapters.update(new_chapters)
            print(f"Consulting chapters: {new_chapters}")
            chunks = self.retrieve_chunks(question, new_chapters)
            all_chunks.extend(chunks)
            answer = self.synthesize_answer(question, all_chunks)
            verdict = self.evaluate_answer(question, answer)
            print(f"Self-evaluation: {verdict}")
            print("\n=== FINAL ANSWER ===")
            print(answer)
            if verdict == "YES":
                return answer, all_chunks
        return answer, all_chunks
