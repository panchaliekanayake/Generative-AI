from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

prompt = "It was a dark and stormy"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model(input_ids)

print(outputs.logits.shape)
