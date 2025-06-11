import gradio as gr
from main import (
    kg_retriever,
    dense_retriever,
    ans_gen,
    hybrid_retrieve
)

def ask_question(question):
    if not question.strip():
        return "", "", ""

    # Retrieve Top-5
    kg_hits = kg_retriever.retrieve(question, k=5)
    kg_texts = [f"{s} {r} {o}" for (s, r, o) in kg_hits]
    kg_items = [{'text': txt, 'source': 'KG'} for txt in kg_texts]
    ans_kg = ans_gen.generate_answer(question, kg_items)

    dense_hits = dense_retriever.retrieve(question, k=5)
    dense_texts = [d['text'] for d in dense_hits]
    dense_items = [{'text': txt, 'source': 'Dense'} for txt in dense_texts]
    ans_dense = ans_gen.generate_answer(question, dense_items)

    hybrid_hits = hybrid_retrieve(question, kg_retriever, dense_retriever)
    ans_hybrid = ans_gen.generate_answer(question, hybrid_hits)

    return ans_kg, ans_dense, ans_hybrid

# Gradio UI setup
demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=[
        gr.Textbox(label="KG-based Answer"),
        gr.Textbox(label="Dense-based Answer"),
        gr.Textbox(label="Hybrid Answer"),
    ],
    title="LightRAG QA Demo",
    description="Ask any fact-based question. This demo shows answers generated from KG-only, Dense-only, and Hybrid retrieval methods using Flan-T5."
)

if __name__ == "__main__":
    demo.launch()
