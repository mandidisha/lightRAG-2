import os
import argparse
from typing import List, Tuple, Optional
import multiprocessing as mp

from knowledge_graph import KnowledgeGraph1
from constructKG import hybrid_triple_extraction, extract_triples_llm
from retriever_pipeline import run_retriever_pipeline, KGRetriever, RealDenseRetriever
from metrics_utils import f1_score, recall_at_k, hallucination_rate
from dataset import load_nq_data
from stats_tests import print_all_tests

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ollama import Client
import gradio as gr
from answer_generators import AnswerGenerator, OllamaAnswerGenerator
from explainer import generate_detailed_explanation, init_explainer, generate_explanation
# ← explainer & feedback
from explainer import init_explainer, generate_explanation
from feedback_ui import process_interaction
import subprocess, json, argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--use_llm", action="store_true")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--version", action="version", version="LightRAG-X 1.0")
args = parser.parse_args()

# # Capture Git commit
# commit = (
#     subprocess.check_output(["git", "rev-parse", "HEAD"])
#     .strip()
#     .decode("ascii")
# )
# metadata = {
#     "git_commit": commit,
#     "utc_timestamp": datetime.utcnow().isoformat() + "Z",
#     "flags": vars(args)
# }
# print(f"[Run metadata] {json.dumps(metadata)}")
# with open("run_metadata.json", "w", encoding="utf-8") as f:
#     json.dump(metadata, f, indent=2)

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
        print("Using Ollama for triple extraction...")
        llm_client = Client(host="http://localhost:11434")
        triples = []
        for q, _, c in sample_data:
            triples.extend(extract_triples_llm(c, question=q, llm_client=llm_client))
        kg_graph = KnowledgeGraph1()
        for t in triples:
            kg_graph.add_triple(*t)
        all_triples = kg_graph.find_edges()
        print(f"[RetrieverPipeline] Total triples in KG (LLM): {len(all_triples)}")

    else:
        print("Using spaCy for triple extraction...")
        kg_graph = run_retriever_pipeline(sample_data, use_llm=False, verbose=True)

    # 3) Initialize explainer over the full KG
    init_explainer(
        triples=kg_graph.find_edges(),
        embed_model='all-MiniLM-L6-v2',
        rerank_model='cross-encoder/ms-marco-TinyBERT-L-2-v2'
    )

    # 4) Instantiate retrievers & generators
    kg_retriever    = KGRetriever(kg_graph.find_edges())
    dense_retriever = RealDenseRetriever([ctx for (_, _, ctx) in sample_data])
    ans_gen         = AnswerGenerator()       # Flan-T5 for evaluation
    ui_gen          = OllamaAnswerGenerator()  # Mistral for UI

    return sample_data


import gradio as gr
from explainer import retrieve_relevant_triples, generate_explanation, verbalize_explanation

def launch_gradio_ui():
    def ask_with_explanation(question: str):
        if not question.strip():
            return "", "", "", "", ""

        # — your existing retrieval + answer logic —
        kg_hits    = kg_retriever.retrieve(question, k=5)
        kg_texts   = [f"{s} {r} {o}" for (s, r, o) in kg_hits]
        ans_kg     = ui_gen.generate_answer(question, [{'text': t, 'source': 'KG'} for t in kg_texts])
        kg_path    = "\n".join(f"{s} —[{r}]→ {o}" for (s, r, o) in kg_hits)

        dense_hits = dense_retriever.retrieve(question, k=5)
        ans_dense  = ui_gen.generate_answer(
            question,
            [{'text': d['text'], 'source': 'Dense'} for d in dense_hits]
        )

        hybrid_hits = hybrid_retrieve(question, kg_retriever, dense_retriever)
        ans_hybrid  = ui_gen.generate_answer(question, hybrid_hits)

        # — new explainer steps —
        # 1) get the raw top-K triples
        triples             = retrieve_relevant_triples(question, top_k=5)

        # 2) your existing highlight-in-HTML provenance
        provenance_html     = generate_explanation(question, ans_hybrid)

        # 3) a natural-language “why” paragraph
        nl_why_explanation  = generate_detailed_explanation(question,           # question
        ans_hybrid,         # answer string
        triples,            # list of (s, r, o) triples
        ui_gen,             # your OllamaAnswerGenerator instance
        max_tokens=128,     # optional cap
        temperature=0.3     # optional sampling temperature)
        )
        # return all five outputs, using nl_why_explanation in out_exp
        return ans_kg, kg_path, ans_dense, ans_hybrid, nl_why_explanation

    with gr.Blocks(css="""
        .final-answer { display: none; }
        .gr-button { margin-top: 8px; margin-bottom: 16px; }
    """) as demo:
        gr.Markdown("## Explainable QA Chatbot")

        inp      = gr.Textbox(label="Enter your question", placeholder="Type your question here…", lines=1)
        ask_btn  = gr.Button("Answer")

        out_kg     = gr.Textbox(label="KG Answer")
        out_path   = gr.Textbox(label="KG Triple Path", lines=4)
        out_dense  = gr.Textbox(label="Dense Answer")
        out_hybrid = gr.Textbox(label="Hybrid Answer")
        out_exp    = gr.Markdown(label="Why Explanation")  # now shows the NL paragraph

        clarity = gr.Slider(1, 7, step=1, label="Clarity")
        helpful = gr.Slider(1, 7, step=1, label="Helpfulness")
        fb_btn  = gr.Button("Submit Feedback")

        ask_btn.click(
            fn=ask_with_explanation,
            inputs=[inp],
            outputs=[out_kg, out_path, out_dense, out_hybrid, out_exp],
            show_progress=True
        )

        fb_btn.click(
            fn=process_interaction,
            inputs=[inp, out_hybrid, out_exp, clarity, helpful],
            outputs=[]
        )

    demo.launch()



if __name__ == "__main__":
    # Use fork on macOS to avoid semaphore KeyErrors
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_llm", action="store_true")
    args = parser.parse_args()

    # Initialize pipeline
    data = init_pipeline(use_llm=args.use_llm)
    total = len(data)

    # Prepare metrics storage
    f1s = {"KG": [], "Dense": [], "Hybrid": []}
    ems = {"KG": [], "Dense": [], "Hybrid": []}
    recalls = {"KG": [], "Dense": [], "Hybrid": []}
    hallucinations = {"KG": [], "Dense": [], "Hybrid": []}

    # Evaluation loop
    for question, gold, _ in data:
        kg_hits     = kg_retriever.retrieve(question)
        kg_texts    = [f"{s} {r} {o}" for (s, r, o) in kg_hits]
        dense_hits  = dense_retriever.retrieve(question)
        dense_texts = [d['text'] for d in dense_hits]
        hybrid_hits = hybrid_retrieve(question, kg_retriever, dense_retriever)
        hybrid_texts= [d['text'] for d in hybrid_hits]

        # Recall@5
        relevant = lambda texts: {t for t in texts if gold.lower() in t.lower()}
        recalls["KG"].append(recall_at_k(kg_texts, relevant(kg_texts), 5))
        recalls["Dense"].append(recall_at_k(dense_texts, relevant(dense_texts), 5))
        recalls["Hybrid"].append(recall_at_k(hybrid_texts, relevant(hybrid_texts), 5))

        # EM / F1 / Hallucination
        for mode, texts in [("KG", kg_texts), ("Dense", dense_texts), ("Hybrid", hybrid_texts)]:
            answer = ans_gen.generate_answer(
                question,
                [{'text': t, 'source': mode} for t in texts]
            )
            f1s[mode].append(f1_score(answer, gold))
            ems[mode].append(int(answer.strip().lower() == gold.strip().lower()))
            hallucinations[mode].append(hallucination_rate(answer, texts))

    # Print aggregated metrics
    print("\n=== Retrieval Recall@5 ===")
    for mode in recalls:
        print(f"{mode} Recall@5: {sum(recalls[mode])/total:.2%}")
    print("\n=== Answer EM / F1 ===")
    for mode in ems:
        print(f"{mode} EM: {sum(ems[mode])/total:.2%}  F1: {sum(f1s[mode])/total:.2%}")
    print("\n=== Hallucination Rate (>50% tokens) ===")
    for mode in hallucinations:
        rate = sum(hallucinations[mode]) / total
        print(f"{mode}: {rate:.2%}")

    # Statistical tests
    print_all_tests(f1s, ems, hallucinations)


    # 2) Export everything to JSON
import json

summary = {
    "total_examples": total,
    "recall_at_5": {
        mode: round(sum(recalls[mode]) / total, 4)
        for mode in recalls
    },
    "exact_match": {
        mode: round(sum(ems[mode]) / total, 4)
        for mode in ems
    },
    "f1_score": {
        mode: round(sum(f1s[mode]) / total, 4)
        for mode in f1s
    },
    "hallucination_rate": {
        mode: round(sum(hallucinations[mode]) / total, 4)
        for mode in hallucinations
    }
}

  # use "spacy" when use_llm=False, else the actual model name ("mistral", "llama3", etc.)
tag = "spacy" if not args.use_llm else ui_gen.model
filename = f"evaluation_metrics_{tag}.json"

with open(filename, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
print(f"[Metrics exported] {filename}")


    # Launch the combined QA + explanation + feedback UI
launch_gradio_ui()
