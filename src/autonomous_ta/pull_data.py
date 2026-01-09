from pathlib import Path

import fitz
import json


DATA_DIR = Path("data/raw/openstax_statistics")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages_text.append({"page_num": page_num, "text": text})
    return pages_text, doc


def get_toc(doc):
    toc = doc.get_toc()
    chapters = []
    for entry in toc:
        level, title, page_num = entry
        chapters.append({"level": level, "title": title, "page_num": page_num})
    return chapters


def chunk_text(pages, max_tokens=300):
    chunks = []
    current = " "
    for page in pages:
        paragraphs = page["text"].split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) == 0:
                continue
            if len(current) + len(para) < max_tokens * 4:
                current += " " + para
            else:
                chunks.append(current.strip())
                current = para
    if current:
        chunks.append(current.strip())
    return chunks


def parse_book(book_pdf_path):
    print(f"Loading PDF from {book_pdf_path}")
    pages, doc = load_pdf(book_pdf_path)
    toc = get_toc(doc)

    print("Extracted Table of Contents:")
    for c in toc:
        print(f"{c['title']} (page {c['page_num']})")

    print("Chunking pages...")
    chunks = chunk_text(pages)

    output_file = DATA_DIR / "intro_stats_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_file}")
