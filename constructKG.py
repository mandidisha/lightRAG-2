import os
import re
import spacy
import textacy
from fuzzywuzzy import fuzz
from knowledge_graph import KnowledgeGraph1
import warnings
import atexit
import multiprocessing as mp
from ollama import Client  # Replaces llama_cpp

# --- Environment Settings ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Resource Tracker Fix for macOS ---
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")
try:
    mp.resource_tracker.unregister('/semaphore_name', 'semaphore')
except (KeyError, AttributeError):
    pass

# --- Load spaCy ---
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    raise e

# --- Heuristic Filter ---
BAD_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
}

def is_informative_triple(triple):
    subj, rel, obj = triple
    if len(subj) < 2 or len(obj) < 2:
        return False
    if subj.lower() in BAD_WORDS or obj.lower() in BAD_WORDS:
        return False
    if subj.isdigit() or obj.isdigit():
        return False
    if re.fullmatch(r"\d{4}", subj) or re.fullmatch(r"\d{4}", obj):
        return False
    if subj.lower() == obj.lower():
        return False
    if all(w.lower() in BAD_WORDS for w in obj.split()):
        return False
    return True

# --- spaCy + textacy SVO Triples ---
def extract_svo_triples(text):
    doc = nlp(text)
    triples = []
    for triple in textacy.extract.subject_verb_object_triples(doc):
        def get_txt(x):
            if hasattr(x, "text"):
                return x.text
            elif isinstance(x, list):
                return " ".join([t.text if hasattr(t, "text") else str(t) for t in x])
            else:
                return str(x)
        if len(triple) == 3:
            subj, verb, obj = triple
            triples.append((get_txt(subj), get_txt(verb), get_txt(obj)))
    return triples

# --- Regex + SVO Hybrid Extraction ---
def hybrid_triple_extraction(text):
    triples = []
    for match in re.finditer(r'([\w\s]+?)\s+(is|was|are|were)\s+(?:an?|the)?\s*([\w\s\-]+)', text, re.IGNORECASE):
        subj, rel, obj = match.groups()
        triples.append((subj.strip(), rel.strip(), obj.strip()))

    for match in re.finditer(r'([\w\s]+),\s+([\w\s]+)', text):
        subj, obj = match.groups()
        triples.append((subj.strip(), "is", obj.strip()))

    for match in re.finditer(r'([\w\s]+)\s+(won|awarded|received|earned)\s+([\w\s]+)', text, re.IGNORECASE):
        subj, rel, obj = match.groups()
        triples.append((subj.strip(), rel.strip(), obj.strip()))

    for match in re.finditer(r'([\w\s]+)\s*\(([\w\s\.\-\d]+)\)', text):
        subj, obj = match.groups()
        triples.append((subj.strip(), "has_note", obj.strip()))

    triples += extract_svo_triples(text)
    seen = set()
    return [t for t in triples if t not in seen and not seen.add(t) and is_informative_triple(t)]

# --- Ollama Chat Call ---
def llm_ollama_generate(prompt, llm_client, max_tokens=128):
    response = llm_client.chat(
        model="mixtral",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens}
    )
    return response["message"]["content"]

# --- LLM Triple Extraction ---
def extract_triples_llm(text, question=None, llm_client=None, max_tokens=128):
    if llm_client is None:
        raise ValueError("extract_triples_llm requires an llm_client when use_llm=True")

    prompt = "You are a factual information extractor.\n\n" \
             "Your task is to extract only clear and meaningful (subject, relation, object) triples from the passage below.\n" \
             "Each triple must express a distinct factual relation, without vague entities or generic connections.\n\n" \
             "Use this format:\n" \
             "Albert Einstein | born in | Germany\n" \
             "Marie Curie | won | Nobel Prize\n" \
             "Cleveland Browns | drafted | Baker Mayfield\n\n" \
             "Guidelines:\n" \
             "- Use concise entity names (avoid pronouns and generic terms).\n" \
             "- Avoid dates and years as subjects or objects.\n" \
             "- Skip redundant or vague triples.\n" \
             "- Output only properly formatted triples, one per line.\n\n"

    if question:
        prompt += f"Question: {question}\n"
    prompt += f"Passage:\n{text}\n\nTriples:\n"

    output = llm_ollama_generate(prompt, llm_client=llm_client, max_tokens=max_tokens)

    triples = []
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            triples.append(tuple(parts))

    return [t for t in set(triples) if is_informative_triple(t)]

# --- Test Block ---
if __name__ == "__main__":
    from dataset import load_nq_data
    sample_data = load_nq_data(limit=3)
    llm_client = Client(host="http://localhost:11434")
    for question, answer, context in sample_data:
        print("\n====CONTEXT====\n" + context[:300])
        triples = extract_triples_llm(context, question, llm_client=llm_client)
        print("====TRIPLES====")
        for t in triples:
            print(t)
