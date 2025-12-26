# rag-docs — Retrieval-Augmented QA over PDFs

[![Status: Prototype](https://img.shields.io/badge/status-prototype-orange.svg)](./README.md) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]

One-line: A small end-to-end RAG demo that ingests documents (PDFs), builds a FAISS vector index with SentenceTransformers embeddings, and answers user queries by retrieving context and generating answers with an LLM (optional — OpenAI or fallback).

Live demo: none (local demo included via Gradio).  
Project status: Prototype — working ingestion, retrieval, and a simple answer-generation glue. Not production-hardened (no auth, limited error handling, simple chunking). See "Project status" below.

Table of contents
- About
- Quickstart (10 mins)
- Architecture
- Files & folders
- Example usage
- Project status & ethics
- License

About
-----
This repo demonstrates a common RAG pattern:
1. Ingest PDFs -> extract text -> chunk
2. Embed chunks using SentenceTransformers
3. Store embeddings in FAISS
4. Query: retrieve top-k chunks -> generate answer with an LLM (OpenAI) or a local fallback strategy

The goal is an honest, reproducible demo that you can extend to support more documents, other vectorstores, or a production deployment.

Quickstart (local)
------------------
1. Clone:
   ```bash
   git clone https://github.com/<your-user>/rag-docs.git
   cd rag-docs
   ```

2. Create env and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ingest sample documents (place PDFs in `data/pdfs/`):
   ```bash
   mkdir -p data/pdfs
   # copy a few PDFs into data/pdfs
   python src/ingest.py --pdf-dir data/pdfs --index-path data/faiss_index --meta-path data/metadata.json
   ```

4. Run the API:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - Open http://localhost:8000/docs for OpenAPI swagger
   - For a simple UI: run `python app/ui_gradio.py` (opens a Gradio web app)

Architecture
------------
- Ingest -> chunk -> embed -> FAISS
- Query -> retrieve -> (optional) LLM for answer synthesis
- Components are intentionally modular (swap embedding model or vector DB)

Files & folders
---------------
- app/ — FastAPI server and simple Gradio UI
- src/ — ingestion, embedding, retrieval glue
- data/ — target for index & metadata (not committed)
- notebooks/ — short notes & experiment ideas
- requirements.txt, Dockerfile, .github/workflows/ci.yml, LICENSE

Example usage
-------------
- Ingest PDFs as above
- Query locally:
  - curl example:
    ```bash
    curl -X POST "http://localhost:8000/qa" -H "Content-Type: application/json" -d '{"question":"What is the main contribution?"}'
    ```

Project status & ethics
-----------------------
Prototype: ingestion, FAISS index creation, retrieval, and a simple LLM-based answer step (OpenAI). This repo is intended as a learning/demo project. It is not production ready — limitations:
- No authentication or rate limiting.
- Simple chunking (fixed-size windows).
- No long-document streaming or advanced reranking.
- If using any documents with private data, remove them before sharing.

License
-------
MIT © 2025 <Your Name>
