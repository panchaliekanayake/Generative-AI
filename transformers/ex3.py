import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt to test
prompt = "It was a dark and stormy"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Tokenize input as PyTorch tensors
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Run the model
with torch.no_grad():
    outputs = model(input_ids)

# Logits shape: [batch_size, sequence_length, vocab_size]
print("Logits shape:", outputs.logits.shape)

# Take logits for the final token position
final_logits = outputs.logits[0, -1]

# Most likely next token id
best_token_id = final_logits.argmax()
print("Best token id:", best_token_id.item())

# Decode most likely next token
best_token = tokenizer.decode(best_token_id)
print("Most likely next token:", repr(best_token))

# Top 10 candidate next tokens by raw logits
print("\nTop 10 next tokens by logit:")
top10_logits = torch.topk(final_logits, 10)
for idx in top10_logits.indices:
    print(repr(tokenizer.decode(idx)))

# Convert logits to probabilities with softmax
print("\nTop 10 next tokens by probability:")
top10_probs = torch.topk(final_logits.softmax(dim=0), 10)
for value, idx in zip(top10_probs.values, top10_probs.indices):
    print(f"{tokenizer.decode(idx)!r:<15} {value.item():.2%}")