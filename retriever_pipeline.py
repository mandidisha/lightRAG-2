import re
from typing import List, Tuple, Optional

from knowledge_graph import KnowledgeGraph1
from constructKG import (
    nlp,
    hybrid_triple_extraction,
    extract_triples_llm,
)
from ollama import Client
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Default toggle for this module (can be overridden by passing use_llm argument)
UseLlm = False


class KGRetriever:
    """
    A FAISS + CrossEncoder retriever over a list of KG triples.
    """
    def __init__(
        self,
        triples: List[Tuple[str, str, str]],
        embed_model: str = 'all-MiniLM-L6-v2',
        rerank_model: str = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
    ):
        self.triples = triples
        # flatten each triple into a single string for embedding
        self.triple_texts = [f"{s} {r} {o}" for s, r, o in triples]

        # build embedding index
        self.embedder = SentenceTransformer(embed_model)
        embeddings = self.embedder.encode(self.triple_texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # cross‐encoder for reranking
        self.reranker = CrossEncoder(rerank_model)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        rerank_topk: int = 20
    ) -> List[Tuple[str, str, str]]:
        """
        1) Embed the query and search the FAISS index to get rerank_topk candidates.
        2) Rerank those with the CrossEncoder.
        3) Return the top-k triples.
        """
        # embed & normalize
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # initial retrieval
        _, I = self.index.search(q_emb, rerank_topk)
        candidates = [(self.triples[i], self.triple_texts[i]) for i in I[0]]

        # rerank
        scores = self.reranker.predict([[query, txt] for _, txt in candidates])
        # sort descending by score
        sorted_triples = [t for _, (t, _) in sorted(zip(scores, candidates), key=lambda x: -x[0])]

        return sorted_triples[:k]


class RealDenseRetriever:
    """
    A simple FAISS-based dense retriever over a corpus of passages.
    """
    def __init__(self, corpus: List[str], embed_model: str = 'all-MiniLM-L6-v2'):
        self.corpus = corpus
        self.embedder = SentenceTransformer(embed_model)
        self.embeddings = self.embedder.encode(corpus, convert_to_numpy=True)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """
        Embed the query, search the FAISS index, and return the top-k passages.
        """
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, I = self.index.search(q_emb, k)
        return [{'text': self.corpus[i]} for i in I[0]]


def run_retriever_pipeline(
    qa_pairs: List[Tuple[str, str, str]],
    use_llm: Optional[bool] = None,
    verbose: bool = True
) -> KnowledgeGraph1:
    """
    Build a KnowledgeGraph1 from (question, answer, context) triples.
    - If use_llm is False, uses spaCy + regex extraction.
    - If use_llm is True, uses Ollama LLaMA 3 extraction.
    - If use_llm is None, falls back to the module‐level UseLlm flag.
    """
    # determine extraction mode
    mode_llm = use_llm if (use_llm is not None) else UseLlm

    llm_client = Client(host="http://localhost:11434") if mode_llm else None
    kg = KnowledgeGraph1()
    num_with_gold = 0

    for question, answer, context in qa_pairs:
        if not (context and answer):
            continue

        found = False
        doc = nlp(context)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text or len(sent_text) > 400:
                continue

            if mode_llm:
                triples = extract_triples_llm(sent_text, question, llm_client=llm_client)
            else:
                triples = hybrid_triple_extraction(sent_text)

            for subj, rel, obj in triples:
                kg.add_triple(subj, rel, obj)
                # mark if we've captured a gold‐answer triple
                if (answer.lower() in subj.lower() 
                    or answer.lower() in rel.lower() 
                    or answer.lower() in obj.lower()):
                    found = True

        if found:
            num_with_gold += 1

    if verbose:
        all_edges = kg.find_edges()
        print(f"[RetrieverPipeline] Total triples in KG: {len(all_edges)}")
        print(f"[RetrieverPipeline] Contexts with ≥1 gold‐answer triple: {num_with_gold}")
        print("[RetrieverPipeline] First 10 triples:")
        for edge in all_edges[:10]:
            print(edge)

    return kg
