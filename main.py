import os
import re
import argparse
from typing import List, Tuple, Optional
import multiprocessing as mp

from knowledge_graph import KnowledgeGraph1
from constructKG import hybrid_triple_extraction, extract_triples_llm
from retriever_pipeline import run_retriever_pipeline
from metrics_utils import f1_score, recall_at_k, hallucination_rate
from dataset import load_nq_data

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
from ollama import Client
from stats_tests import print_all_tests

class AnswerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device if device else ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        self.model = self.model.to(self.device)

    def generate_answer(self, question: str, retrieved: List[dict], max_length: int = 64) -> str:
        prompt_lines = [f"Question: {question}\n", "Relevant facts or passages:"]
        for idx, item in enumerate(retrieved, start=1):
            prompt_lines.append(f"{idx}. ({item['source']}) {item['text']}")
        prompt_lines.append("\nAnswer:")
        prompt = "\n".join(prompt_lines)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

class OllamaAnswerGenerator:
    def __init__(self, model_name="mistral", host="http://localhost:11434"):
        self.client = Client(host=host)
        self.model = model_name

    def generate_answer(self, question: str, retrieved: List[dict]) -> str:
        context_passages = "\n".join(f"- {item['text']}" for item in retrieved)
        prompt = (
            f"Answer the question using only the facts below. Be concise and informative.\n\n"
            f"Question: {question}\n\n"
            f"Facts:\n{context_passages}\n\n"
            f"Answer:"
        )
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4}
        )
        return response['message']['content'].strip()

class KGRetriever:
    def __init__(self, triples: List[Tuple[str, str, str]], embed_model='all-MiniLM-L6-v2', rerank_model='cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        self.triples = triples
        self.triple_texts = [f"{s} {r} {o}" for s, r, o in triples]
        self.embedder = SentenceTransformer(embed_model)
        embeddings = self.embedder.encode(self.triple_texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.reranker = CrossEncoder(rerank_model)

    def retrieve(self, query: str, k=5, rerank_topk=20) -> List[Tuple[str, str, str]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, I = self.index.search(q_emb, rerank_topk)
        candidates = [(self.triples[i], self.triple_texts[i]) for i in I[0]]
        scores = self.reranker.predict([[query, txt] for _, txt in candidates])
        candidates = [t for _, (t, _) in sorted(zip(scores, candidates), key=lambda x: -x[0])]
        return candidates[:k]

class RealDenseRetriever:
    def __init__(self, corpus: List[str], embed_model='all-MiniLM-L6-v2'):
        self.corpus = corpus
        self.embedder = SentenceTransformer(embed_model)
        self.embeddings = self.embedder.encode(corpus, convert_to_numpy=True)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k=5) -> List[dict]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, I = self.index.search(q_emb, k)
        return [{'text': self.corpus[i]} for i in I[0]]

def hybrid_retrieve(query, kg_retriever, dense_retriever, k=5) -> List[dict]:
    kg = [{'text': f"{s} {r} {o}", 'source': 'KG'} for (s, r, o) in kg_retriever.retrieve(query, k)]
    dense = [{'text': d['text'], 'source': 'Dense'} for d in dense_retriever.retrieve(query, k)]
    seen, combined = set(), []
    for item in kg + dense:
        if item['text'] not in seen:
            combined.append(item)
            seen.add(item['text'])
    return combined

def init_pipeline(use_llm=False, sample_data=None):
    global kg_retriever, dense_retriever, ans_gen, ui_gen

    if sample_data is None:
        sample_data = load_nq_data(limit=5000)

    if use_llm:
        print(" Using Ollama for triple extraction...")
        llm_client = Client(host="http://localhost:11434")
        triples = []
        for q, _, c in sample_data:
            triples.extend(extract_triples_llm(c, question=q, llm_client=llm_client))
        kg_graph = KnowledgeGraph1()
        for t in triples:
            kg_graph.add_triple(*t)
    else:
        print(" Using spaCy for triple extraction...")
        kg_graph = run_retriever_pipeline(sample_data, use_llm=False, verbose=True)

    kg_retriever = KGRetriever(kg_graph.find_edges())
    dense_retriever = RealDenseRetriever([ctx for (_, _, ctx) in sample_data])
    ans_gen = AnswerGenerator()
    ui_gen = OllamaAnswerGenerator()
    return sample_data

if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_llm", action="store_true")
    args = parser.parse_args()

    data = init_pipeline(use_llm=args.use_llm)
    total = len(data)
    f1s, ems = {"KG": [], "Dense": [], "Hybrid": []}, {"KG": [], "Dense": [], "Hybrid": []}
    recalls = {"KG": [], "Dense": [], "Hybrid": []}
    hallucinations = {"KG": [], "Dense": [], "Hybrid": []}

    for question, gold, context in data:
        kg_hits = kg_retriever.retrieve(question)
        kg_texts = [f"{s} {r} {o}" for (s, r, o) in kg_hits]
        dense_hits = dense_retriever.retrieve(question)
        dense_texts = [d['text'] for d in dense_hits]
        hybrid_hits = hybrid_retrieve(question, kg_retriever, dense_retriever)
        hybrid_texts = [d['text'] for d in hybrid_hits]

        relevant = lambda texts: {t for t in texts if gold.lower() in t.lower()}
        recalls["KG"].append(recall_at_k(kg_texts, relevant(kg_texts), 5))
        recalls["Dense"].append(recall_at_k(dense_texts, relevant(dense_texts), 5))
        recalls["Hybrid"].append(recall_at_k(hybrid_texts, relevant(hybrid_texts), 5))

        for mode, texts in [("KG", kg_texts), ("Dense", dense_texts), ("Hybrid", hybrid_texts)]:
            answer = ans_gen.generate_answer(question, [{'text': t, 'source': mode} for t in texts])
            f1s[mode].append(f1_score(answer, gold))
            ems[mode].append(int(answer.lower().strip() == gold.lower().strip()))
            hallucinations[mode].append(hallucination_rate(answer, texts))

    print("\n=== Retrieval Recall@5 ===")
    for mode in recalls:
        print(f"{mode} Recall@5:  {sum(recalls[mode])/total:.2%}")
    print("\n=== Answer EM / F1 ===")
    for mode in ems:
        print(f"{mode} EM: {sum(ems[mode])/total:.2%}  F1: {sum(f1s[mode])/total:.2%}")
    print("\n=== Hallucination Rate (>50% tokens hallucinated) ===")
    for mode in hallucinations:
        rate = sum(hallucinations[mode]) / total
        print(f"{mode}: {rate:.2%}")

    print_all_tests(f1s, ems, hallucinations)

    def ask_ui(q):
        kg = [{'text': f"{s} {r} {o}", 'source': 'KG'} for (s, r, o) in kg_retriever.retrieve(q)]
        dense = [{'text': d['text'], 'source': 'Dense'} for d in dense_retriever.retrieve(q)]
        hybrid = hybrid_retrieve(q, kg_retriever, dense_retriever)
        return (
            ui_gen.generate_answer(q, kg),
            ui_gen.generate_answer(q, dense),
            ui_gen.generate_answer(q, hybrid),
        )

    gr.Interface(
        fn=ask_ui,
        inputs=gr.Textbox(label="Enter your question"),
        outputs=[
            gr.Textbox(label="KG Answer (Mistral)"),
            gr.Textbox(label="Dense Answer (Mistral)"),
            gr.Textbox(label="Hybrid Answer (Mistral)")
        ],
        title="LightRAG QA UI",
        description="Flan-T5 for evaluation | Mistral via Ollama for UI display"
    ).launch()
