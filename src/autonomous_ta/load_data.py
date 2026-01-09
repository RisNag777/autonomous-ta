from pathlib import Path

import fitz
import json
import os


DATA_DIR = Path("data/raw/")
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


def chunk_text(pages, toc, max_tokens=300):
    chunks = []
    current_chunk = " "
    current_chapter = "Unknown"
    chapter_index = 0
    for page in pages:
        while (
            chapter_index + 1 < len(toc)
            and page["page_num"] >= toc[chapter_index + 1]["page_num"]
        ):
            chapter_index += 1
        current_chapter = toc[chapter_index]["title"]

        paragraphs = page["text"].split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) == 0:
                continue
            if len(current_chunk) + len(para) < max_tokens * 4:
                current_chunk += " " + para
            else:
                chunks.append(
                    {
                        "chapter_title": current_chapter,
                        "page_num": page["page_num"],
                        "chunk_text": current_chunk.strip(),
                    }
                )
                current_chunk = para
    if current_chunk:
        chunks.append(
            {
                "chapter_title": current_chapter,
                "page_num": page["page_num"],
                "chunk_text": current_chunk.strip(),
            }
        )
    return chunks


def parse_book():
    for file in os.listdir(DATA_DIR):
        if file.split(".")[-1] == "pdf":
            print(f"Loading PDF from {file}")
            pages, doc = load_pdf(DATA_DIR / file)
            toc = get_toc(doc)

            print("Extracted Table of Contents:")
            for c in toc:
                print(f"----{c['title']} (page {c['page_num']})")

            print("Chunking pages...")
            chunks = chunk_text(pages, toc)

            output_file = DATA_DIR / f"{file}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(chunks)} chunks to {output_file}")