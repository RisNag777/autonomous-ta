from dotenv import load_dotenv
from openai import OpenAI

from autonomous_ta.vector_db import VectorDB

import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class TextbookAgent:
    def __init__(self):
        self.db = VectorDB()
        self.db.build_index()

    def answer_question(self, question, top_k=5, model="gpt-4o-mini"):
        results = self.db.query(question, top_k=top_k)
        context = ""
        for result in results:
            context += f"[Chapter: {result['chapter']}, "
            +f"Page: {result['page']}]\n{result['chunk_text']}\n\n"
        prompt = f"""
        You are a helpful teaching assistant. Use the following textbook content to answer the question below.
        If the answer is not contained in the text, say "I cannot answer that from the textbook."

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
        return answer, results
