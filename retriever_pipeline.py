import re
from typing import List, Tuple, Optional

from knowledge_graph import KnowledgeGraph1
from constructKG import (
    nlp,
    hybrid_triple_extraction,
    extract_triples_llm,
)

# ✅ Use Ollama instead of llama_cpp
from ollama import Client

# Default toggle for this module (can be overridden by passing use_llm argument)
UseLlm = False

def run_retriever_pipeline(
    qa_pairs: List[Tuple[str, str, str]],
    use_llm: Optional[bool] = None,
    verbose: bool = True
) -> KnowledgeGraph1:
    """
    Build a KnowledgeGraph1 from a list of (question, answer, context) triples.
    If use_llm is False, apply hybrid_triple_extraction (spaCy + regex).
    If use_llm is True, apply extract_triples_llm (via Ollama LLaMA 3).
    If use_llm is None, fall back to this module's UseLlm.
    Returns the built KnowledgeGraph1.
    """

    # Decide mode
    mode_llm = use_llm if (use_llm is not None) else UseLlm

    # ✅ Initialize Ollama client only if using LLM extraction
    llm_client = None
    if mode_llm:
        llm_client = Client(host="http://localhost:11434")

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
                # ✅ Use LLM-based extraction with Ollama
                triples = extract_triples_llm(sent_text, question, llm_client=llm_client)
            else:
                # Use spaCy + regex extraction
                triples = hybrid_triple_extraction(sent_text)

            for subj, rel, obj in triples:
                kg.add_triple(subj, rel, obj)
                if (answer.lower() in subj.lower() 
                    or answer.lower() in rel.lower() 
                    or answer.lower() in obj.lower()):
                    found = True

        if found:
            num_with_gold += 1

    if verbose:
        print(f"[RetrieverPipeline] Total triples in KG: {len(kg.find_edges())}")
        print(f"[RetrieverPipeline] Contexts with ≥1 gold‐answer triple: {num_with_gold}")
        print("[RetrieverPipeline] First 10 triples:")
        for edge in kg.find_edges()[:10]:
            print(edge)

    return kg
