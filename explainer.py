from retriever_pipeline import KGRetriever
from typing import List, Tuple, Optional

# Global singleton retriever instance
_explainer: KGRetriever = None

def init_explainer(
    triples: List[Tuple[str, str, str]],
    embed_model: str = 'all-MiniLM-L6-v2',
    rerank_model: Optional[str] = None
):
    """
    Build the FAISS + optional CrossEncoder index over your existing KG triples.
    Call this once after you construct your KG.
    """
    global _explainer
    _explainer = KGRetriever(
        triples,
        embed_model=embed_model,
        rerank_model=rerank_model
    )

def retrieve_relevant_triples(
    question: str,
    top_k: int = 5
) -> List[Tuple[str, str, str]]:
    """
    Return the top_k (subject, relation, object) triples most
    relevant to `question`. Requires init_explainer(...) first.
    """
    if _explainer is None:
        raise RuntimeError(
            "Explainer not initialized. Call init_explainer(triples, ...) first."
        )
    return _explainer.retrieve(question, k=top_k)
