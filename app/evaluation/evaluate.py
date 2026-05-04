from transformers import pipeline

def evaluate(model_path):

    pipe = pipeline("text-generation", model=model_path)

    return pipe("Explain AI simply", max_new_tokens=50)