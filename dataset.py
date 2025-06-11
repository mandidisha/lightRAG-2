# dataset.py
from datasets import load_dataset
import re
import random

def load_nq_data(split='train', limit=None):
    """
    Load and preprocess Natural Questions data for a specific split.
    Returns a list of (question, answer, context) tuples.
    If 'limit' is provided, only the first N samples are returned.
    """
    dataset = load_dataset('cjlovering/natural-questions-short', 'default')
    qa_data = []
    for idx, item in enumerate(dataset[split]):
        # Extract question text
        if item.get('questions'):
            question = item['questions'][0].get('input_text', "")
        else:
            question = ""
        # Extract short answer text (span_text if available, otherwise input_text)
        answer = ""
        if item.get('answers'):
            ans_entry = item['answers'][0]
            if 'span_text' in ans_entry and ans_entry['span_text']:
                answer = ans_entry['span_text']
            elif 'input_text' in ans_entry:
                answer = ans_entry['input_text']
        # Extract context (concatenate list of contexts if multiple)
        context = ""
        if item.get('contexts'):
            if isinstance(item['contexts'], list):
                context = " ".join(item['contexts'])
            else:
                context = str(item['contexts'])
        # Clean context: remove citation markers like [1], [2], ... for clarity
        context = re.sub(r'\[\d+\]', '', context)
        # Add to list
        qa_data.append((question.strip(), answer.strip(), context.strip()))
        if limit and idx >= limit - 1:
            break
    return qa_data

def build_retrieval_contexts(num_gold=None, num_distractors=None, seed=42, open_domain=True):
    import random
    random.seed(seed)

    if open_domain:
        # Open-domain: use ONLY train contexts
        train_data = load_nq_data(split='train')
        train_contexts = [ctx for _, _, ctx in train_data]
        if num_distractors is not None:
            retrieval_contexts = random.sample(train_contexts, min(num_distractors, len(train_contexts)))
        else:
            retrieval_contexts = train_contexts
        return retrieval_contexts
    else:
        # Closed-book: gold contexts + distractors (old method)
        val_data = load_nq_data(split='validation', limit=num_gold)
        gold_contexts = [ctx for _, _, ctx in val_data]
        train_data = load_nq_data(split='train')
        train_contexts = [ctx for _, _, ctx in train_data]
        if num_distractors is None:
            distractor_contexts = train_contexts
        else:
            distractor_contexts = random.sample(train_contexts, min(num_distractors, len(train_contexts)))
        all_contexts = gold_contexts + distractor_contexts
        return all_contexts



# Example usages (uncomment to test/run):
if __name__ == "__main__":
    # 1. Load training data (for training an encoder, etc.)
    train = load_nq_data(split='train', limit=None)
    print("Train example:")
    for q, a, ctx in train:
        print(f"Q: {q}\nA: {a}\nContext: {ctx[:80]}...\n")

    # 2. Load gold validation data (for evaluation)
    val = load_nq_data(split='validation', limit=5)
    print("Validation (gold) example:")
    for q, a, ctx in val:
        print(f"Q: {q}\nA: {a}\nContext: {ctx[:80]}...\n")

    # 3. Build retrieval context pool (gold + distractors)
    retrieval_contexts = build_retrieval_contexts(num_gold=120, num_distractors=5000)
    print(f"\nTotal retrieval contexts: {len(retrieval_contexts)}")
    print("Sample context:", retrieval_contexts[0][:120])
