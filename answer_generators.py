import os
from typing import List, Optional
from ollama import Client
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
    def __init__(self, model_name="llama3", host="http://localhost:11434"):
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
            options={"temperature": 0.4,
                     "max_tokens": 64
                     }
        )
        
        return response['message']['content'].strip()
