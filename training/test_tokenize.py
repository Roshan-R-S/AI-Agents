import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

with open('../datasets/inspiration_coach_data.json') as f:
    data = json.load(f)

text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data[0]['instruction']}

### Response:
{data[0]['response']}"""

tokens = tokenizer(text, truncation=True, max_length=256, padding='max_length')
print(f'Tokens length: {len(tokens["input_ids"])}')
print(f'Non-pad tokens: {sum(1 for t in tokens["input_ids"] if t != tokenizer.pad_token_id)}')
print(f'Sample tokens: {tokens["input_ids"][:20]}')