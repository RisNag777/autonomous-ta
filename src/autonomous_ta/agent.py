from dotenv import load_dotenv
from openai import OpenAI

from src.autonomous_ta.vector_db import VectorDB

import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class TextbookAgent:
    def __init__(self):
        self.db = VectorDB()
        self.db.build_index()

    def extract_chapter_keywords(self, question):
        keywords = [word.lower() for word in question.split() if len(word) > 4]
        return keywords

    def answer_question(self, question, top_k=5, model="gpt-4o-mini"):
        chapter_keywords = self.extract_chapter_keywords(question)
        results = self.db.query(
            question, top_k=top_k, chapter_keywords=chapter_keywords
        )
        context = ""
        for result in results:
            context += f"[Chapter: {result['chapter']}, "
            context += f"Page: {result['page']}]\n{result['chunk_text']}\n\n"
        prompt = f"""
        You are a helpful teaching assistant. Use the following textbook content to answer the question below.
        Answer only based on the textbook content, but do your best even if the answer is not explicitly stated.

        Textbook content:
        {context}

        Question: {question}
        Answer:
        """  # noqa: E501
        completion = client.chat.completions.create(
            model=model,  # fmt: off
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message.content
        print("=== ANSWER ===")
        print(answer)
        print("\n=== CHUNKS RETRIEVED ===")
        for result in results:
            print(f"{result['chapter']} (Page {result['page']}): ")
            print(f"{result['chunk_text'][:200]}...\n")
        return answer, results
