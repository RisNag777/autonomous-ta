from dotenv import load_dotenv
from openai import OpenAI

from src.autonomous_ta.vector_db import VectorDB

import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class TextbookAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.db = VectorDB()
        self.db.build_index()
        self.model = model

    def choose_chapters(self, question, available_chapters):
        prompt = f"""
        You are figuring out how to answer a question from a given list of
        chapters from a textbook.

        Question:
        {question}

        Available Chapters:
        {available_chapters}

        Select the MOST relevant chapters to consult.
        Return ONLY a JSON array of chapter titles to consult.
        Do not include any explanation or extra text.

        Example:
        ["1.2 Data, Sampling, and Variation in Data and Sampling"]
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return eval(response.choices[0].message.content)

    def synthesize_answer(self, question, chunks):
        context = ""
        for chunk in chunks:
            context += f"[Chapter: {chunk['chapter']}, "
            context += f"Page: {chunk['page']}]\n{chunk['chunk_text']}\n\n"
        prompt = f"""
        You are a helpful teaching assistant. Use the following textbook
        content to answer the question below.
        Answer only based on the textbook content, but do your best even if
        the answer is not explicitly stated.

        Textbook content:
        {context}

        Question:
        {question}

        Answer:
        """  # noqa: E501
        completion = client.chat.completions.create(
            model=self.model,  # fmt: off
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message.content
        return answer

    def find_chapters(self, question, available_chapters, consulted_chapters):
        chapters = self.choose_chapters(question, available_chapters)
        new_chapters = [
            chapter
            for chapter in chapters  # fmt: off
            if chapter not in consulted_chapters
        ]
        if not new_chapters:
            print("No new chapters to consult.")
            return [], consulted_chapters
        consulted_chapters.update(new_chapters)
        print(f"Consulting Chapters - {new_chapters}")
        return new_chapters, consulted_chapters

    def evaluate_answer(self, question, answer):
        prompt = f"""
        Evaluate the following answer to the question

        Question:
        {question}

        Answer:
        {answer}

        Is this answer COMPLETE and WELL-SUPPORTED by the content in the
        textbook?
        Respond with ONLY ONE WORD: YES or NO.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def answer_question(self, question, top_k=5, model="gpt-4o-mini"):
        available_chapters = self.db.list_chapters()
        consulted_chapters = set()
        all_chunks = []

        chapters, consulted_chapters = self.find_chapters(
            question, available_chapters, consulted_chapters
        )
        if not chapters:
            print(f"Unable to find content related to '{question}'")
            return "", ""
        results = self.db.query(
            question, top_k=top_k, chapter_keywords=chapters
        )
        all_chunks.extend(results)
        answer = self.synthesize_answer(question, all_chunks)
        verdict = self.evaluate_answer(question, answer)
        print(f"Self-evaluation: {verdict}")
        if verdict == "YES":
            print("=== ANSWER ===")
            print(answer)
            print("\n=== CHUNKS RETRIEVED ===")
            for result in results:
                print(f"{result['chapter']} (Page {result['page']}): ")
            return answer, all_chunks
        else:
            print(f"Unable to find content related to '{question}'")
            return "", ""
