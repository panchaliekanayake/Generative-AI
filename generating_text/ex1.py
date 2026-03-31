from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device="cuda:0",
)

print(classifier("This movie is disgustingly good!"))

