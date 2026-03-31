from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    device="cuda:0"
)

result = generator("Once upon a time", max_length=30)
print(result)
