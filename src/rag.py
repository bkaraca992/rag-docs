"""
RAG helper: load FAISS index + metadata and answer queries.

- If OPENAI_API_KEY is set and use_llm=True: attempts to generate an answer with OpenAI.
- Otherwise: returns concatenated retrieved chunks as a naive "answer".
"""
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any
import openai

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

class RAG:
    def __init__(self, index_path="data/faiss_index", meta_path="data/metadata.json", embed_model_name="all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embed_model_name = embed_model_name
        self.model = SentenceTransformer(self.embed_model_name)
        self.index = None
        self.texts = []
        self.meta = []
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.texts = data.get("texts", [])
            self.meta = data.get("meta", [])

    def _embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True)
        return v

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        qv = self._embed_query(query)
        D, I = self.index.search(qv, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({"score": float(score), "text": self.texts[idx], "source": self.meta[idx].get("source")})
        return results

    def _call_openai(self, prompt: str) -> str:
        if not OPENAI_KEY:
            return ""
        openai.api_key = OPENAI_KEY
        try:
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=256,
                temperature=0.0
            )
            return resp.choices[0].text.strip()
        except Exception as e:
            return f"OpenAI call failed: {e}"

    def answer(self, question: str, top_k: int = 4, use_llm: bool = True) -> Dict[str, Any]:
        retrieved = self.retrieve(question, top_k=top_k)
        combined_context = "\n\n---\n\n".join([r["text"] for r in retrieved])
        answer = None
        if use_llm and OPENAI_KEY:
            prompt = f"Use the context below to answer the question. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{combined_context}\n\nQuestion: {question}\n\nAnswer:"
            answer = self._call_openai(prompt)
        else:
            # Fallback: naive extractive-like answer by returning top chunk(s)
            answer = combined_context[:2000] or "No context retrieved."
        return {"question": question, "answer": answer, "sources": retrieved}
