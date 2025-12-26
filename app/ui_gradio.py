"""
Simple Gradio UI to demo RAG locally.
Run: python app/ui_gradio.py
"""
import gradio as gr
from src.rag import RAG

rag = RAG(index_path="data/faiss_index", meta_path="data/metadata.json", embed_model_name="all-MiniLM-L6-v2")

def query_fn(q):
    res = rag.answer(q, top_k=4, use_llm=False)  # default to non-LLM mode for local demo
    # Format a readable response
    answer = res.get("answer")
    sources = res.get("sources", [])
    return answer or "No answer", "\n\n".join([f"- {s['source']} (score: {s['score']:.3f})\n{s['text'][:300]}..." for s in sources])

iface = gr.Interface(
    fn=query_fn,
    inputs=gr.Textbox(lines=2, label="Question"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Top retrieved chunks")],
    title="rag-docs — Local Demo",
    description="Retrieval Augmented QA demo (prototype)."
)

if __name__ == "__main__":
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
