# rag_engine.py

import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pdf_handler import load_pdf_text, chunk_text

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_DIR = "storage/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

class RAGEngine:
    def __init__(self):
        self.index = None
        self.chunk_to_text = []
        self.doc_sources = []  # For tracking which doc each chunk came from

    def add_document(self, file_path, persistent=False):
        text = load_pdf_text(file_path)
        chunks = chunk_text(text)
        embeddings = EMBEDDING_MODEL.encode(chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        self.chunk_to_text.extend(chunks)
        self.doc_sources.extend([file_path] * len(chunks))

        if persistent:
            self.save_index()

    def retrieve_context(self, query, top_k=3):
        if self.index is None:
            return ""
        query_embedding = EMBEDDING_MODEL.encode([query])
        D, I = self.index.search(np.array(query_embedding), top_k)
        return "\n".join([self.chunk_to_text[i] for i in I[0]])

    def save_index(self):
        faiss.write_index(self.index, os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
            pickle.dump({
                "chunk_to_text": self.chunk_to_text,
                "doc_sources": self.doc_sources
            }, f)

    def load_index(self):
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        meta_path = os.path.join(INDEX_DIR, "meta.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.chunk_to_text = meta["chunk_to_text"]
                self.doc_sources = meta["doc_sources"]
