# explainer.py

from typing import List, Tuple, Optional, Callable
import html
from retriever_pipeline import KGRetriever

# Optional: if you need the LLM wrapper class here
from answer_generators import OllamaAnswerGenerator  

# Global singleton for fast retrieval
_explainer: Optional[KGRetriever] = None

def init_explainer(
    triples: List[Tuple[str, str, str]],
    embed_model: str = 'all-MiniLM-L6-v2',
    rerank_model: Optional[str] = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
):
    """
    Build a FAISS + CrossEncoder index over your KG triples.
    Call once at startup (e.g. in init_pipeline).
    """
    global _explainer
    _explainer = KGRetriever(
        triples,
        embed_model=embed_model,
        rerank_model=rerank_model
    )

def retrieve_relevant_triples(
    question: str,
    top_k: int = 5,
    rerank_topk: Optional[int] = None
) -> List[Tuple[str, str, str]]:
    """
    Return the top_k (subject, relation, object) triples most relevant to question.
    Requires init_explainer() first.
    """
    if _explainer is None:
        raise RuntimeError("Explainer not initialized. Call init_explainer(...) first.")
    return _explainer.retrieve(
        question,
        k=top_k,
        rerank_topk=rerank_topk or top_k * 4
    )

def generate_explanation(
    question: str,
    answer: str,
    top_k: int = 5
) -> str:
    """
    Highlight any verbatim triple mentions in the answer.
    If none are found, append an HTML provenance list.
    Returns HTML-safe markup for Gradio’s HTML component.
    """
    triples = retrieve_relevant_triples(question, top_k=top_k)
    explanation = html.escape(answer)

    # highlight any exact matches
    for subj, rel, obj in triples:
        triple_text = f"{subj} —[{rel}]→ {obj}"
        escaped = html.escape(triple_text)
        explanation = explanation.replace(
            escaped,
            f"<mark>{escaped}</mark>"
        )

    # if nothing matched, append provenance list
    if all(f"{s} —[{r}]→ {o}" not in answer for s, r, o in triples):
        explanation += "<hr><p><em>Provenance:</em><br>"
        explanation += "<br>".join(
            html.escape(f"{s} —[{r}]→ {o}") for s, r, o in triples
        )
        explanation += "</p>"

    return explanation

def verbalize_explanation(
    question: str,
    triples: List[Tuple[str, str, str]],
    llm_generate_fn: Callable[[str, List[dict]], str],
    style: str = "concise"
) -> str:
    """
    Turn triples into a single paragraph explaining *why* they support the answer.
    `llm_generate_fn` should be your ui_gen.generate_answer signature.
    """
    # build bullet list
    bullet_list = "\n".join(f"- {s} —[{r}]→ {o}" for s, r, o in triples)
    prompt = (
        f"Question: {question}\n\n"
        f"I have these supporting facts from a knowledge graph:\n"
        f"{bullet_list}\n\n"
        f"Explain *why* these facts support the answer in a {style} paragraph."
    )
    return llm_generate_fn(prompt, [])

def generate_detailed_explanation(
    question: str,
    answer: str,
    triples: List[Tuple[str, str, str]],
    llm_client: OllamaAnswerGenerator,
    max_tokens: int = 128,
    temperature: float = 0.3
) -> str:
    """
    For each triple, output a numbered, one-sentence explanation
    of its relevance to the answer.
    """
    # numbered list of triples
    numbered = "\n".join(
        f"{i}. {s} —[{r}]→ {o}"
        for i, (s, r, o) in enumerate(triples, 1)
    )
    prompt = (
        f"You have answered:\n\"{answer}\"\n\n"
        f"And these KG triples supported that answer:\n{numbered}\n\n"
        f"For *each* triple above, write a one-sentence explanation of why it is relevant."
    )
    resp = llm_client.client.chat(
        model=llm_client.model,
        messages=[{"role":"user","content":prompt}],
        options={"temperature": temperature, "max_tokens": max_tokens}
    )
    return resp["message"]["content"].strip()
