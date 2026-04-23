import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prompt
prompt = "It was a dark and stormy"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)
model.eval()

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Beam search generation
with torch.no_grad():
    beam_output = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=30,
        early_stopping=True
    )

# Decode and print
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))