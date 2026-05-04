from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

generator = pipeline("text-generation", model="./model")

@app.get("/generate")
def generate(prompt: str):
    return {"output": generator(prompt, max_new_tokens=100)}