import json
import argparse
import logging
from typing import List, Optional

import gradio as gr
from ollama import Client

from knowledge_graph import KnowledgeGraph1
from constructKG import extract_triples_llm
from retriever_pipeline import run_retriever_pipeline, KGRetriever, RealDenseRetriever
from metrics_utils import f1_score, recall_at_k, hallucination_rate
from dataset import load_nq_data
from stats_tests import print_all_tests
from answer_generators import AnswerGenerator, OllamaAnswerGenerator
from explainer import (
    generate_detailed_explanation, init_explainer,
    retrieve_relevant_triples,
)
from feedback_ui import process_interaction

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Globals set by init_pipeline ───────────────────────────────────────────────
kg_retriever: Optional[KGRetriever] = None
dense_retriever: Optional[RealDenseRetriever] = None
ans_gen: Optional[AnswerGenerator] = None
ui_gen: Optional[OllamaAnswerGenerator] = None


def hybrid_retrieve(query, kg_retriever, dense_retriever, k=5) -> List[dict]:
    kg_items = [{'text': f"{s} {r} {o}", 'source': 'KG'} for (s, r, o) in kg_retriever.retrieve(query, k)]
    dense_items = [{'text': d['text'], 'source': 'Dense'} for d in dense_retriever.retrieve(query, k)]
    seen, combined = set(), []
    for item in kg_items + dense_items:
        if item['text'] not in seen:
            combined.append(item)
            seen.add(item['text'])
    return combined


def init_pipeline(use_llm=False, sample_data=None):
    global kg_retriever, dense_retriever, ans_gen, ui_gen

    # 1) Load data
    if sample_data is None:
        sample_data = load_nq_data(limit=5)

    # 2) Build KG
    if use_llm:
        logger.info("Using Ollama for triple extraction...")
        llm_client = Client(host="http://localhost:11434")
        triples = []
        for q, _, c in sample_data:
            triples.extend(extract_triples_llm(c, question=q, llm_client=llm_client, model="llama3.2:3b"))
        kg_graph = KnowledgeGraph1()
        for t in triples:
            kg_graph.add_triple(*t)
        total_triples = len(kg_graph.find_edges())
        if total_triples == 0:
            logger.warning(
                "0 triples extracted — the model may be too small to follow the format. "
                "Try: ollama pull mistral  or  ollama pull llama3.2:3b"
            )
        logger.info("Total triples in KG (LLM): %d", total_triples)
    else:
        logger.info("Using spaCy for triple extraction...")
        kg_graph = run_retriever_pipeline(sample_data, use_llm=False, verbose=True)

    # 3) Initialize explainer over the full KG
    init_explainer(
        triples=kg_graph.find_edges(),
        embed_model='all-MiniLM-L6-v2',
        rerank_model='cross-encoder/ms-marco-TinyBERT-L-2-v2',
    )

    # 4) Instantiate retrievers & generators
    kg_retriever    = KGRetriever(kg_graph.find_edges())
    dense_retriever = RealDenseRetriever([ctx for (_, _, ctx) in sample_data])
    ans_gen         = AnswerGenerator()
    ui_gen          = OllamaAnswerGenerator(model_name="llama3.2:3b")

    return sample_data


def launch_gradio_ui():
    def ask_with_explanation(question: str):
        if not question.strip():
            return "", "", "", "", ""

        kg_hits  = kg_retriever.retrieve(question, k=5)
        kg_texts = [f"{s} {r} {o}" for (s, r, o) in kg_hits]
        ans_kg   = ui_gen.generate_answer(question, [{'text': t, 'source': 'KG'} for t in kg_texts])
        kg_path  = "\n".join(f"{s} —[{r}]→ {o}" for (s, r, o) in kg_hits)

        dense_hits = dense_retriever.retrieve(question, k=5)
        ans_dense  = ui_gen.generate_answer(
            question, [{'text': d['text'], 'source': 'Dense'} for d in dense_hits]
        )

        hybrid_hits = hybrid_retrieve(question, kg_retriever, dense_retriever)
        ans_hybrid  = ui_gen.generate_answer(question, hybrid_hits)

        triples            = retrieve_relevant_triples(question, top_k=5)
        nl_why_explanation = generate_detailed_explanation(
            question, ans_hybrid, triples, ui_gen, max_tokens=128, temperature=0.3
        )

        logger.info("Answered question: %.80s", question)
        return ans_kg, kg_path, ans_dense, ans_hybrid, nl_why_explanation

    with gr.Blocks(css="""
        .final-answer { display: none; }
        .gr-button { margin-top: 8px; margin-bottom: 16px; }
    """) as demo:
        gr.Markdown("## Explainable QA Chatbot")

        inp     = gr.Textbox(label="Enter your question", placeholder="Type your question here…", lines=1)
        ask_btn = gr.Button("Answer")

        out_kg     = gr.Textbox(label="KG Answer")
        out_path   = gr.Textbox(label="KG Triple Path", lines=4)
        out_dense  = gr.Textbox(label="Dense Answer")
        out_hybrid = gr.Textbox(label="Hybrid Answer")
        out_exp    = gr.Markdown(label="Why Explanation")

        clarity = gr.Slider(1, 7, step=1, label="Clarity")
        helpful = gr.Slider(1, 7, step=1, label="Helpfulness")
        fb_btn  = gr.Button("Submit Feedback")

        ask_btn.click(
            fn=ask_with_explanation,
            inputs=[inp],
            outputs=[out_kg, out_path, out_dense, out_hybrid, out_exp],
            show_progress=True,
        )
        fb_btn.click(
            fn=process_interaction,
            inputs=[inp, out_hybrid, out_exp, clarity, helpful],
            outputs=[],
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--version", action="version", version="LightRAG-X 1.0")
    args = parser.parse_args()

    data  = init_pipeline(use_llm=args.use_llm)
    total = len(data)

    f1s           = {"KG": [], "Dense": [], "Hybrid": []}
    ems           = {"KG": [], "Dense": [], "Hybrid": []}
    recalls       = {"KG": [], "Dense": [], "Hybrid": []}
    hallucinations = {"KG": [], "Dense": [], "Hybrid": []}

    for question, gold, _ in data:
        kg_hits      = kg_retriever.retrieve(question)
        kg_texts     = [f"{s} {r} {o}" for (s, r, o) in kg_hits]
        dense_hits   = dense_retriever.retrieve(question)
        dense_texts  = [d['text'] for d in dense_hits]
        hybrid_hits  = hybrid_retrieve(question, kg_retriever, dense_retriever)
        hybrid_texts = [d['text'] for d in hybrid_hits]

        relevant = lambda texts: {t for t in texts if gold.lower() in t.lower()}
        recalls["KG"].append(recall_at_k(kg_texts, relevant(kg_texts), 5))
        recalls["Dense"].append(recall_at_k(dense_texts, relevant(dense_texts), 5))
        recalls["Hybrid"].append(recall_at_k(hybrid_texts, relevant(hybrid_texts), 5))

        for mode, texts in [("KG", kg_texts), ("Dense", dense_texts), ("Hybrid", hybrid_texts)]:
            answer = ans_gen.generate_answer(question, [{'text': t, 'source': mode} for t in texts])
            f1s[mode].append(f1_score(answer, gold))
            ems[mode].append(int(answer.strip().lower() == gold.strip().lower()))
            hallucinations[mode].append(hallucination_rate(answer, texts))

    logger.info("=== Retrieval Recall@5 ===")
    for mode in recalls:
        logger.info("%s Recall@5: %.2f%%", mode, 100 * sum(recalls[mode]) / total)

    logger.info("=== Answer EM / F1 ===")
    for mode in ems:
        logger.info("%s EM: %.2f%%  F1: %.2f%%", mode,
                    100 * sum(ems[mode]) / total, 100 * sum(f1s[mode]) / total)

    logger.info("=== Hallucination Rate (>50%% tokens) ===")
    for mode in hallucinations:
        logger.info("%s: %.2f%%", mode, 100 * sum(hallucinations[mode]) / total)

    print_all_tests(f1s, ems, hallucinations)

    # Export metrics to JSON
    summary = {
        "total_examples": total,
        "recall_at_5":        {mode: round(sum(recalls[mode]) / total, 4)       for mode in recalls},
        "exact_match":        {mode: round(sum(ems[mode]) / total, 4)            for mode in ems},
        "f1_score":           {mode: round(sum(f1s[mode]) / total, 4)            for mode in f1s},
        "hallucination_rate": {mode: round(sum(hallucinations[mode]) / total, 4) for mode in hallucinations},
    }
    tag      = "spacy" if not args.use_llm else ui_gen.model
    filename = f"evaluation_metrics_{tag}.json"
    with open(filename, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    logger.info("Metrics exported to %s", filename)

    launch_gradio_ui()
