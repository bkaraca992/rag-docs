"""
Ingest PDFs: extract, chunk, embed, and save FAISS index + metadata.

Example:
python src/ingest.py --pdf-dir data/pdfs --index-path data/faiss_index --meta-path data/metadata.json
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def extract_text_from_pdf(pdf_path):
    text = []
    try:
        reader = PdfReader(str(pdf_path))
        for p in reader.pages:
            text.append(p.extract_text() or "")
    except Exception as e:
        print(f"Failed to read {pdf_path}: {e}")
    return "\n".join(text)

def chunk_text(text, chunk_size=500, overlap=50):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_index(text_chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.index_factory(dim, "IDMap,Flat")
    index.add(embeddings)
    return index, embeddings

def save_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def main(args):
    pdf_dir = Path(args.pdf_dir)
    all_chunks = []
    metadata = []
    for pdf in pdf_dir.glob("*.pdf"):
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadata.append({"source": str(pdf.name), "chunk_index": i})
    if not all_chunks:
        print("No text found in PDFs.")
        return
    print(f"Embedding {len(all_chunks)} chunks with model {args.model}")
    index, embeddings = build_index(all_chunks, model_name=args.model)
    save_index(index, args.index_path)
    # Save metadata (texts + metadata)
    meta_out = {"texts": all_chunks, "meta": metadata}
    Path(args.meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print("Index and metadata saved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", required=True)
    p.add_argument("--index-path", default="data/faiss_index")
    p.add_argument("--meta-path", default="data/metadata.json")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--chunk-size", type=int, default=150)
    p.add_argument("--overlap", type=int, default=30)
    args = p.parse_args()
    main(args)
