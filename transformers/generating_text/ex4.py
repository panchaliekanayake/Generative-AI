import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt
prompt = "It was a dark and stormy"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Tokenize input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate 20 new tokens
with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=20)

# Decode full generated sequence
decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print results
print("Input IDs:", input_ids[0])
print("Output IDs:", output_ids[0])
print("Generated text:")
print(decoded_text)