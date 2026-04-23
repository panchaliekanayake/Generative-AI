from transformers import pipeline
pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device="cuda")
data = pipe("electric rock solo, very intense")
print(data)