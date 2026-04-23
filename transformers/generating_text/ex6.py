import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose model
MODEL_ID = "Qwen/Qwen2-0.5B"

# Input text
prompt = "It was a dark and stormy"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# -----------------------------
# 1) Tokenize the prompt
# -----------------------------
input_ids = tokenizer(prompt).input_ids
print("Token IDs:")
print(input_ids)
print()

print("Decoded tokens:")
for t in input_ids:
    print(f"{t}\t: {tokenizer.decode([t])}")
print()

# -----------------------------
# 2) Prepare tensor for model
# -----------------------------
inputs = tokenizer(prompt, return_tensors="pt")

# -----------------------------
# 3) Run the model
# -----------------------------
with torch.no_grad():
    outputs = model(**inputs)

print("Logits shape:")
print(outputs.logits.shape)   # [batch_size, sequence_length, vocab_size]
print()

# -----------------------------
# 4) Get logits for the next token
#    after the full prompt
# -----------------------------
final_logits = outputs.logits[0, -1]

# Most likely next token
best_token_id = final_logits.argmax().item()
best_token = tokenizer.decode([best_token_id])

print("Most likely next token ID:")
print(best_token_id)
print()

print("Most likely next token:")
print(repr(best_token))
print()

# -----------------------------
# 5) Top 10 candidate next tokens
#    using raw logits
# -----------------------------
top10_logits = torch.topk(final_logits, 10)

print("Top 10 next-token candidates:")
for index in top10_logits.indices:
    token_id = index.item()
    print(f"{token_id}\t: {repr(tokenizer.decode([token_id]))}")
print()

# -----------------------------
# 6) Convert logits to probabilities
# -----------------------------
probs = torch.softmax(final_logits, dim=0)
top10_probs = torch.topk(probs, 10)

print("Top 10 next-token probabilities:")
for value, index in zip(top10_probs.values, top10_probs.indices):
    token_id = index.item()
    token_text = tokenizer.decode([token_id])
    print(f"{repr(token_text):<15} {value.item():.2%}")