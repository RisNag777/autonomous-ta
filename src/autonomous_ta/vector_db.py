from pathlib import Path
from sentence_transformers import SentenceTransformer

import faiss
import json
import numpy as np
import os


DATA_DIR = Path("data/raw/")


class VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.metadata = []

    def build_index(self):
        for file in os.listdir(DATA_DIR):
            if file.split(".")[-1] == "json":
                CHUNKS_FILE = DATA_DIR / file
                with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                self.texts = [chunk["chunk_text"] for chunk in chunks]
                self.metadata = [
                    {  # fmt: off
                        "chapter": chunk["chapter_title"],
                        "page": chunk["page_num"],
                        "book": file,
                    }
                    for chunk in chunks
                ]

                print("Computing embeddings...")
                embeddings = self.model.encode(  # fmt: off
                    self.texts, show_progress_bar=True
                )
                embeddings = np.array(embeddings).astype("float32")

                # Build FAISS index
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(embeddings)
                print(f"Index for {file} built with {len(self.texts)} chunks")

    def query(self, question, top_k=5, chapter_keywords=None):
        filtered_indices = list(range(len(self.texts)))
        if chapter_keywords:
            chapter_keywords_lower = [  # fmt: off
                keyword.lower() for keyword in chapter_keywords
            ]
            filtered_indices = [
                pos
                for pos, meta in enumerate(self.metadata)
                if any(
                    keyword in meta["chapter"].lower()
                    for keyword in chapter_keywords_lower
                )
            ]
            if not filtered_indices:
                filtered_indices = list(range(len(self.texts)))
        filtered_embeddings = np.array(
            [self.model.encode(self.texts[pos]) for pos in filtered_indices]
        ).astype("float32")

        q_vec = self.model.encode([question]).astype("float32")
        index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        index.add(filtered_embeddings)
        distances, indices = index.search(  # fmt: off
            q_vec, min(top_k, len(filtered_indices))
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            real_idx = filtered_indices[idx]
            results.append(
                {
                    "chunk_text": self.texts[real_idx],
                    "chapter": self.metadata[real_idx]["chapter"],
                    "page": self.metadata[real_idx]["page"],
                    "book": self.metadata[real_idx]["book"],
                    "distance": float(dist),
                }
            )
        return results

    def list_chapters(self):
        return sorted(set(meta["chapter"] for meta in self.metadata))
