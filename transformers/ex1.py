from transformers import AutoTokenizer
# Use the id of the model you want to use
# GPT-2 "openai-community/gpt2"
# Qwen "Qwen/Qwen2-0.5B"
# SmolLM "HuggingFaceTB/SmolLM-135M"
prompt = "It was a dark and stormy"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
input_ids = tokenizer(prompt).input_ids
input_ids
[2132, 572, 264, 6319, 323, 13458, 88]
for t in input_ids:
    print(t, "\t:", tokenizer.decode(t))


