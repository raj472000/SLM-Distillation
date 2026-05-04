from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from app.config import Config

def load_teacher():

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16"
    )

    return AutoModelForCausalLM.from_pretrained(
        Config.TEACHER_MODEL,
        quantization_config=bnb,
        device_map="auto"
    )