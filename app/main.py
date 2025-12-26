from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
from src.rag import RAG

app = FastAPI(title="rag-docs API", version="0.1")

# paths used by the demo
INDEX_PATH = os.environ.get("RAG_INDEX_PATH", "data/faiss_index")
META_PATH = os.environ.get("RAG_META_PATH", "data/metadata.json")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

rag = RAG(index_path=INDEX_PATH, meta_path=META_PATH, embed_model_name=EMBED_MODEL)

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    use_llm: Optional[bool] = True

@app.post("/qa")
def answer(req: QARequest):
    """
    Query the RAG system. If OpenAI key is available and use_llm=True, generation will be attempted.
    """
    result = rag.answer(req.question, top_k=req.top_k, use_llm=req.use_llm)
    return result

@app.get("/")
def root():
    return {"message": "rag-docs: visit /docs for API"}
