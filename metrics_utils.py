import re
from collections import Counter
from typing import List

def normalize_text(s):
    """Lowercase, remove articles/punctuation/extra whitespace as done in SQuAD evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(prediction: str, ground_truth: str) -> int:
    """Return 1 if prediction exactly matches the ground truth after normalization, else 0."""
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Compute Recall@k: fraction of relevant items retrieved in top-k.
    `retrieved` is an ordered list of items (e.g., retrieved facts or answers),
    `relevant` is the set of all relevant items for the query.
    """
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    retrieved_relevant = sum(1 for item in retrieved_k if item in relevant)
    return retrieved_relevant / len(relevant)

def hallucination_rate(answer: str, retrieved_texts: List[str], threshold: float = 0.5) -> bool:
    """
    Return True if >threshold of answer tokens are hallucinated (not found in retrieved text).
    """
    retrieved_corpus = " ".join(retrieved_texts).lower()
    answer_tokens = answer.lower().split()
    hallucinated_tokens = [t for t in answer_tokens if t not in retrieved_corpus]
    return len(hallucinated_tokens) / len(answer_tokens) > threshold if answer_tokens else False


def evaluate(predictions, references):
    """
    Compute average EM/F1 over a list of predictions and a list of lists of ground truths.
    `predictions`: List[str]
    `references`: List[List[str]]
    Returns (avg_em, avg_f1).
    """
    em_scores = []
    f1_scores = []
    for pred, truths in zip(predictions, references):
        em = max(exact_match(pred, t) for t in truths)
        f1 = max(f1_score(pred, t) for t in truths)
        em_scores.append(em)
        f1_scores.append(f1)
    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    return avg_em, avg_f1
