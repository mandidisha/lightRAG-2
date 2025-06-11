from datasets import load_dataset
import json

# Load validation split (not train!)
dataset = load_dataset('cjlovering/natural-questions-short', 'default')
val_set = dataset['validation']

gold = []
for item in val_set.select(range(800)):  # or random.sample if you want random subset
    question = item['questions'][0].get('input_text', "")
    answer = ""
    if item.get('answers'):
        ans_entry = item['answers'][0]
        if 'span_text' in ans_entry and ans_entry['span_text']:
            answer = ans_entry['span_text']
        elif 'input_text' in ans_entry:
            answer = ans_entry['input_text']
    gold.append({"question": question, "answer": answer})

with open("gold.json", "w") as f:
    json.dump(gold, f, indent=2)
