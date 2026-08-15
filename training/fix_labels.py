import json
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

with open('../datasets/inspiration_coach_data.json') as f:
    data = json.load(f)

texts = []
for item in data:
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Response:
{item['response']}"""
    texts.append(text)

dataset = Dataset.from_dict({"text": texts})

def tokenize_fn(examples):
    tokens = tokenizer(
        examples['text'],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
tokenized.save_to_disk('../datasets/tokenized_coach')
print("Done")