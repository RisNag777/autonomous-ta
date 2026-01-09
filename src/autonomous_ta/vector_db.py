from pathlib import Path
from sentence_transformers import SentenceTransformer

import faiss
import json
import numpy as np
import os


DATA_DIR = Path("data/raw/")
# EMBEDDINGS_FILE = DATA_DIR / "intro_stats_faiss.index"


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

    def query(self, question, top_k=5):
        # Embed the query
        q_vec = self.model.encode([question]).astype("float32")
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append(
                {
                    "chunk_text": self.texts[idx],
                    "chapter": self.metadata[idx]["chapter"],
                    "page": self.metadata[idx]["page"],
                    "distance": float(dist),
                }
            )
        return results
