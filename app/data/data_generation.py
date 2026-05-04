from transformers import pipeline
from app.config import Config
from app.utils.logger import get_logger
from typing import Dict,List,Any

logger = get_logger("data_gen")

generator = pipeline(
    "text-generation",
    model=Config.TEACHER_MODEL,
    device_map="auto"
)

def generate_data(prompts : List)->Dict[str,Any]:
    dataset = []

    for p in prompts:
        out = generator(p, max_new_tokens=200)[0]["generated_text"]

        dataset.append({
            "instruction": p,
            "input": "",
            "output": out
        })

        logger.info(f"Generated: {p}")

    return dataset