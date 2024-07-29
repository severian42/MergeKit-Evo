import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
model_id = "evol_merge"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

question = "Janet's duck lays 16 eggs a day. Janet consumes 3 as part of her breakfast every morning and uses 4 daily to bake muffins for her friends. She sells the rest at the market for $2 each. How much does she make at the market each day?"
prompt = f"Please solve the following math problem. Describe the reasoning process of the solution before giving the final answer. In the answer, include only the integer answer without any additional text.\nProblem:{question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=2048,
  temperature=0.75,
  top_p=0.95,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
