
# 🧠 LightRAG: Explainable QA with Hybrid Retrieval and Knowledge Graphs

**LightRAG** is a research-grade Retrieval-Augmented Generation (RAG) system that enhances factual QA using:

- 🔄 **Hybrid retrieval** (dense + knowledge graph)
- 📖 **Explainable answer generation**
- 📊 **Quantitative + statistical evaluation**

This implementation supports a Master’s thesis focused on **trust and explainability in generative AI systems**.

---

## 🔍 Motivation

As LLMs grow in popularity, the need for **transparency** and **grounded QA** becomes essential.

LightRAG aims to:
- Improve factual accuracy with **structured triples**
- Reduce hallucination via **hybrid evidence grounding**
- Evaluate retrieval and generation using **standard metrics** + **statistical tests**

---

## 📦 Project Structure

```
.
├── main.py                   # Pipeline + evaluation + UI launcher
├── constructKG.py           # Hybrid triple extraction (spaCy + LLM)
├── knowledge_graph.py       # Lightweight triple-based KG class
├── retriever_pipeline.py    # End-to-end KG building pipeline
├── dataset.py               # Loads Natural Questions subset
├── metrics_utils.py         # EM, F1, Recall@K, hallucination, tests
├── README.md                # You're here!
└── requirements.txt         # Dependencies
```

---

## 🔧 System Components

### 1️⃣ Triple Extraction

| Method | Description |
|--------|-------------|
| `extract_triples_spacy` | Uses spaCy NER and rules |
| `extract_triples_llm`   | Uses Mistral (Ollama) for semantic parsing |
| `hybrid_triple_extraction` | Switches between both |

---

### 2️⃣ Retrieval Modules

- `KGRetriever` – FAISS + CrossEncoder reranking on triple graph
- `DenseRetriever` – MiniLM dense passage retriever
- `HybridRetriever` – Combines both, filters duplicates

---

### 3️⃣ Answer Generation

| Model        | Role                 |
|--------------|----------------------|
| **Flan-T5**  | Evaluation only (EM/F1 scoring) |
| **Mistral**  | UI answer display via Ollama  |

---

### 4️⃣ Metrics & Testing

- **Recall@5** – Does gold answer appear in retrieved context?
- **Exact Match (EM)** – String match (normalized)
- **F1 Score** – Token-level match
- **Hallucination Rate** – 50%+ hallucinated answer = fail
- **Significance Tests**:
  - Paired t-tests (F1/EM)
  - Wilcoxon tests (non-parametric)
  - Chi-squared (hallucination frequency)

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> ⚠️ You must have Ollama installed and running locally with `mistral` pulled:
> https://ollama.com/library/mistral

---

### 2. Run with spaCy extraction (fast)

```bash
python main.py
```

### 3. Run with LLM extraction (more accurate)

```bash
python main.py --use_llm
```

---

## 🧪 Evaluation Example

```
=== Retrieval Recall@5 ===
KG Recall@5:  39.50%
Dense Recall@5:  99.00%
Hybrid Recall@5:  20.88%

=== Answer EM / F1 ===
KG EM: 26.00%  F1: 41.00%
Dense EM: 35.50%  F1: 57.08%
Hybrid EM: 34.00%  F1: 53.10%

=== Hallucination Rate (>50%) ===
KG: 10.00%
Dense: 0.50%
Hybrid: 0.50%

=== Statistical Significance Tests ===
KG vs Dense → F1 p=0.0002 ✅
KG vs Hybrid → F1 p=0.0009 ✅
Dense vs Hybrid → F1 p=0.6709 ❌
```

---

## 🧪 Research Questions Supported

- **RQ1**: Can hybrid retrieval improve factual accuracy over dense-only methods?
- **RQ2**: Does KG-based grounding reduce hallucination?
- **RQ3**: Are model differences statistically significant?

---

## 📚 Dataset

- Uses: [Natural Questions - Short Form](https://huggingface.co/datasets/cjlovering/natural-questions-short)
- Limit: Subset of 100–300 samples for efficient QA evaluation

---

## 🎛 Gradio UI (Real-time Demo)

- Enter your own questions
- See answers from:
  - KG-based retrieval (Mistral)
  - Dense retrieval (Mistral)
  - Hybrid combined (Mistral)
- Compare retrieved facts per model

---

## 📘 Academic References

- Lewis et al., 2020. *Retrieval-Augmented Generation*. NeurIPS.
- Petroni et al., 2019. *Language Models as Knowledge Bases?*. EMNLP.
- Guu et al., 2020. *REALM: Retrieval-Augmented Language Model Pre-Training*.
- Rasooli et al., 2021. *Natural Questions: Benchmarking QA*. Google Research.

---

## 📄 License

MIT License — free for academic and research use.
