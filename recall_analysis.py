# recall_analysis.py

from metrics_utils import recall_at_k

def evaluate_recall_at_k(retriever, gold_facts_map, questions, ks=[1,3,5,10]):
    """
    gold_facts_map: dict mapping each question -> set of gold (s,r,o) triples
    questions: list of questions
    ks: list of K values to test
    """
    results = {k: [] for k in ks}
    for q in questions:
        gold = gold_facts_map.get(q, set())
        retrieved = retriever.retrieve(q, k=max(ks))
        for k in ks:
            results[k].append(recall_at_k(retrieved, gold, k))
    # Compute average recall
    return {k: sum(scores)/len(scores) for k, scores in results.items()}

# Example usage in run_exp1.py after building retriever:
# gold_map = {...}  # build this from your gold.json by mapping question->set of triples
# recalls = evaluate_recall_at_k(retriever, gold_map, [e["question"] for e in gold_data])
# print("Recall@K:", recalls)
