import os
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

base_url_dict = {"openstax": "https://openstax.org"}


def get_openai_client():
    load_dotenv()
    api_key=os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def get_table_of_contents(base, book):
    toc_url = base + "/books/" + book + "/pages/preface"
    client = get_openai_client()
    prompt = """
    You are Indexy, a bot that takes a link to a book, goes to the website
    and returns the table of contents of the book as a list.
    """
    messages = [{"role": "system", "content": prompt}]
    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0
    )
    ai_response_content = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_response_content})
    print(ai_response_content)


def download_book(base, book):
    output_dir = Path("data/raw/openstax_statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = base_url_dict[base.lower()]
    book = book.lower().replace(" ", "-")
    toc_links = get_table_of_contents(base_url, book)

    for toc in toc_links:
        url = f"{base_url}/books/{book}/pages/{toc}"
        out_file = output_dir / f"{toc}.html"
        print(f"Downloading {url} → {out_file}")
        r = requests.get(url)
        if r.status_code == 200:
            out_file.write_text(r.text, encoding="utf-8")
        else:
            print(f"Failed: {r.status_code}")

    print(f"Downloaded {len(toc_links)} pages")
    